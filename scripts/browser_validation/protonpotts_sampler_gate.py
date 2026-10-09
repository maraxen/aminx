"""X8b gate: the JS ProtonPotts sampler (decoder_sampler.mjs) from PDB text vs the UNPADDED eager decoder, under Node.

task_id 261009_protonpotts-onnx, backlog #5816 (B2). Sidecar: protonpotts_sampler_gate.bth.toml (pre-registered before this script
existed). The reference runs `make_encode` / `make_decode` on the unpadded structure; the JS side pads to the bucket, aligns the
uniform stream (caller step k -> padded step pad+k) and trims. Injected streams make the two directly comparable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

from scripts.browser_validation.layer_a_common import emit  # noqa: E402
from scripts.browser_validation.potts_browser_parity import git_state  # noqa: E402
from scripts.browser_validation.protonpotts_decode_export_gate import (  # noqa: E402
  CONFIGS,
  FIXED_EVERY,
  FIXED_OFFSET,
  FRAGILE,
  SEED,
  V,
  _margin,
  _structures,
)
from scripts.browser_validation.protonpotts_export_gate import _bucket_for, _load_model, _sha256  # noqa: E402

logger = logging.getLogger("protonpotts_sampler_gate")
LOG_BAR = 1.0e-4
SEEDS = (11, 12)


def _rel(ref: np.ndarray, got: np.ndarray) -> float:
  ref = np.asarray(ref, dtype=np.float64)
  return float(np.max(np.abs(ref - np.asarray(got, dtype=np.float64))) / max(1.0, float(np.max(np.abs(ref)))))


def build_reference(potts_root: Path, state_npz: Path):  # noqa: ANN201
  import equinox as eqx  # noqa: PLC0415
  import jax  # noqa: PLC0415
  import jax.numpy as jnp  # noqa: PLC0415

  from aminx.families.protonpotts_mpnn.decode import ProtonPottsARDecode  # noqa: PLC0415
  from aminx.families.protonpotts_mpnn.decode_export import make_decode, make_encode, upstream_cdf_order  # noqa: PLC0415

  model = _load_model(state_npz)
  perturbed_dec = jax.tree_util.tree_map(
    lambda leaf: leaf + 1e-2 * jax.random.normal(jax.random.PRNGKey(1), leaf.shape, leaf.dtype)
    if (eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating)) else leaf, model.mpnn.decoder)
  bad_model = eqx.tree_at(lambda m: m.mpnn.decoder, model, perturbed_dec)
  encode, decode, bad_decode = jax.jit(make_encode(model)), jax.jit(make_decode(model)), jax.jit(make_decode(bad_model))
  cdf_order = upstream_cdf_order()
  rng = np.random.default_rng(SEED)
  cases = []
  for st in _structures(potts_root):
    length, raw = st["length"], st["raw"]
    h_v, h_e, e_idx = (np.asarray(x) for x in encode(*[jnp.asarray(raw[k]) for k in ("coords", "present", "residue_idx", "chain_index")]))
    for config in CONFIGS:
      temperature = np.full(length, 0.1, np.float32)
      bias = np.zeros((length, V), np.float32)
      designed = np.ones(length, bool)
      if config == "bias_temp":
        temperature = rng.uniform(0.3, 1.0, size=length).astype(np.float32)
        bias = (0.5 * rng.standard_normal((length, V))).astype(np.float32)
      elif config == "fixed":
        designed[FIXED_OFFSET::FIXED_EVERY] = False
      uniforms = rng.uniform(size=length).astype(np.float32)
      noise = rng.standard_normal(length).astype(np.float32)
      ins = [h_v, h_e, e_idx, raw["present"], raw["pad_valid"], st["native"], designed, temperature, bias, uniforms, noise]
      jin = [jnp.asarray(a) for a in ins]
      ref = [np.asarray(x) for x in decode(*jin)]
      bad = np.asarray(bad_decode(*jin)[2])
      eager = ProtonPottsARDecode(layers=model.mpnn.decoder.layers, w_s_embed=model.mpnn.w_s_embed, w_out=model.mpnn.w_out)(
        *jin[:10], noise=jin[10], cdf_order=jnp.asarray(cdf_order))
      margin = _margin(np.asarray(eager.probs_sample), ref[1], uniforms, cdf_order, length)
      cases.append({"id": f"{st['name']}__{config}", "structure": st["name"], "config": config, "length": length,
                    "bucket": _bucket_for(length), "designed": designed, "temperature": temperature, "bias": bias,
                    "uniforms": uniforms, "noise": noise, "ref": ref, "bad_log_probs": bad, "margin": margin,
                    "fragile": margin < FRAGILE})
    logger.info("reference %s done", st["name"])
  return cases


def node_case(c: dict[str, Any], pdb: Path, *, roll: bool = False) -> dict[str, Any]:
  flat_t = c["temperature"].tolist() if c["config"] == "bias_temp" else float(np.float32(0.1))
  u = np.roll(c["uniforms"], 1) if roll else c["uniforms"]
  return {"id": c["id"] + ("__rolled" if roll else ""), "bucket": c["bucket"], "pdb_file": str(pdb),
          "designed": c["designed"].astype(int).tolist(), "temperature": flat_t, "bias": c["bias"].reshape(-1).tolist(),
          "uniforms": u.tolist(), "noise": c["noise"].tolist()}


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0915
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--models", type=Path, required=True, help="decoder artifact dir (X8a manifest)")
  parser.add_argument("--manifest-sha256", required=True)
  parser.add_argument("--potts-root", type=Path,
                      default=Path(os.environ.get("AMINX_POTTS_ROOT", "~/repos/PottsMPNN")).expanduser())
  parser.add_argument("--state-npz", type=Path,
                      default=Path(os.environ.get("AMINX_PROTONPOTTS_STATE", "~/scratch/v6_state.npz")).expanduser())
  parser.add_argument("--node-bin", default=os.environ.get("NODE_BIN"))
  parser.add_argument("--ort-web-dir", default=os.environ.get("ORT_WEB_DIR"))
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  git_hash, git_clean = git_state(REPO)
  result: dict[str, Any] = {
    "jax_arm_ok": False, "artifacts_verified": False, "node_ok": False, "num_threads": 0, "jax_version": "",
    "onnxruntime_web_version": "", "node_version": "", "git_hash": git_hash, "git_clean": git_clean,
    "manifest_sha256": args.manifest_sha256, "cases": [], "n_cases": 0, "n_pass": 0, "n_fragile_cases": 0, "all_exact": False,
    "seeded_repeatable": False, "seeded_distinct": False, "control_tokens_detected": False, "control_float_detected": False,
    "error": "",
  }
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    import jax  # noqa: PLC0415

    result["jax_version"] = str(jax.__version__)
    if _sha256(args.models / "MANIFEST.json") != args.manifest_sha256:
      msg = "decoder MANIFEST.json sha256 does not match --manifest-sha256"
      raise RuntimeError(msg)  # noqa: TRY301
    manifest = json.loads((args.models / "MANIFEST.json").read_text())
    for b in manifest["buckets"]:
      for g in b["graphs"].values():
        if _sha256(args.models / g["file"]) != g["sha256"]:
          msg = f"{g['file']} sha256 does not match the manifest"
          raise RuntimeError(msg)  # noqa: TRY301
    result["artifacts_verified"] = True
    cases = build_reference(args.potts_root, args.state_npz)
    result["jax_arm_ok"] = True
    result["n_fragile_cases"] = sum(c["fragile"] for c in cases)
    logger.info("reference arm: %d cases in %.1fs", len(cases), time.perf_counter() - t0)

    pdbs = {}
    from scripts.browser_validation.protonpotts_export_gate import STRUCTURES  # noqa: PLC0415

    for name, (rel, _sha) in STRUCTURES.items():
      pdbs[name] = args.potts_root / "energy_benchmark_datasets" / rel
    node_cases = [node_case(c, pdbs[c["structure"]]) for c in cases]
    node_cases += [node_case(c, pdbs[c["structure"]], roll=True) for c in cases if c["config"] == "bias_temp"]
    for name in pdbs:
      bucket = next(c["bucket"] for c in cases if c["structure"] == name)
      for seed in (SEEDS[0], SEEDS[0], SEEDS[1]):
        node_cases.append({"id": f"{name}__seed{seed}__{sum(1 for n in node_cases if n['id'].startswith(f'{name}__seed{seed}'))}",
                           "bucket": bucket, "pdb_file": str(pdbs[name]), "designed": None, "temperature": float(np.float32(0.1)),
                           "bias": None, "uniforms": None, "noise": None, "seed": seed})
    cases_path = args.work_dir / "cases.json"
    cases_path.write_text(json.dumps({"cases": node_cases}))
    out_path = args.work_dir / "node_result.json"
    runner = REPO / "browser" / "protonpotts-scorer" / "run_sampler_node.mjs"
    proc = subprocess.run(  # noqa: S603 -- fixed argv we built
      [args.node_bin, str(runner), "--ort-dir", args.ort_web_dir, "--models", str(args.models),
       "--cases", str(cases_path), "--out", str(out_path)], capture_output=True, text=True, check=False, timeout=3600)
    if proc.returncode != 0:
      msg = f"node runner rc={proc.returncode}: {proc.stderr[-3000:]}"
      raise RuntimeError(msg)  # noqa: TRY301
    node = json.loads(out_path.read_text())
    result["node_ok"] = True
    result["num_threads"] = int(node.get("num_threads") or 0)
    got = {r["id"]: r for r in node["results"]}

    rows = []
    for c in cases:
      g = got.get(c["id"])
      if not g or not g["ok"]:
        rows.append({"id": c["id"], "ran": False, "passed": False, "error": ((g or {}).get("error") or "missing")[:1500]})
        continue
      length = c["length"]
      ref_seq, ref_order, ref_lp = c["ref"]
      seq_eq = bool(np.array_equal(np.asarray(g["sequence"]), ref_seq[:length]))
      order_eq = bool(np.array_equal(np.asarray(g["decoding_order"]), ref_order[:length]))
      lp = _rel(ref_lp[:length], np.asarray(g["log_probs"]).reshape(length, V))
      rows.append({"id": c["id"], "ran": True, "fragile": bool(c["fragile"]), "margin": c["margin"], "sequence_exact": seq_eq,
                   "order_exact": order_eq, "log_probs_rel": lp,
                   "passed": bool((seq_eq or c["fragile"]) and order_eq and lp <= LOG_BAR)})
    result["cases"] = rows
    result["n_cases"] = len(rows)
    result["n_pass"] = sum(bool(r["passed"]) for r in rows)
    result["all_exact"] = bool(rows) and result["n_pass"] == len(rows)

    # controls: rolled uniforms (bias_temp, T >= 0.3 so the draws matter) must change the sequence; perturbed weights the log-probs
    rolled = [(c, got.get(c["id"] + "__rolled")) for c in cases if c["config"] == "bias_temp"]
    result["control_tokens_detected"] = bool(rolled) and all(
      g and g["ok"] and not np.array_equal(np.asarray(g["sequence"]), c["ref"][0][: c["length"]]) for c, g in rolled)
    result["control_float_detected"] = all(
      (g := got.get(c["id"])) and g["ok"] and _rel(c["bad_log_probs"][: c["length"]], np.asarray(g["log_probs"]).reshape(c["length"], V)) > LOG_BAR
      for c in cases if c["config"] == "bias_temp")

    repeat, distinct = [], []
    for name in pdbs:
      a, b, d = (next((r for i, r in got.items() if i.startswith(f"{name}__seed{s}__") and i.endswith(f"__{k}")), None)
                 for s, k in ((SEEDS[0], 0), (SEEDS[0], 1), (SEEDS[1], 0)))
      ok = all(x and x["ok"] for x in (a, b, d))
      repeat.append(bool(ok and a["sequence"] == b["sequence"] and a["decoding_order"] == b["decoding_order"]
                         and a["log_probs"] == b["log_probs"]))
      distinct.append(bool(ok and a["sequence"] != d["sequence"]))
    result["seeded_repeatable"] = bool(repeat) and all(repeat)
    result["seeded_distinct"] = bool(distinct) and all(distinct)
    import shutil  # noqa: PLC0415

    node_bin = shutil.which(args.node_bin) or args.node_bin
    result["node_version"] = subprocess.run([node_bin, "--version"], capture_output=True, text=True, check=False).stdout.strip()  # noqa: S603
    pkg = Path(args.ort_web_dir) / "node_modules" / "onnxruntime-web" / "package.json"
    result["onnxruntime_web_version"] = json.loads(pkg.read_text()).get("version", "") if pkg.exists() else ""
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("sampler gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("exact=%s (%s/%s) fragile=%s seeded=%s/%s controls=%s/%s", result["all_exact"], result["n_pass"], result["n_cases"],
              result["n_fragile_cases"], result["seeded_repeatable"], result["seeded_distinct"], result["control_tokens_detected"],
              result["control_float_detected"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
