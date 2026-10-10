"""X8a gate: the 30-token decoder graphs (encode, decode) on real structures, ORT-CPU and ORT-Web, deterministic across runs.

task_id 261009_protonpotts-onnx, backlog #5816 (B2). Sidecar: protonpotts_decode_export_gate.bth.toml (pre-registered before this script
existed). ADR decisions 11-12; spec §53-54. Weights are baked in, every runtime value is an input. The JAX arm runs entirely before
jax2onnx is imported.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

from scripts.browser_validation.potts_export_spike import (  # noqa: E402
  DTYPE_TAG,
  _graph_facts,
  _read_web_outputs,
  _run_web,
  compare,
)
from scripts.browser_validation.protonpotts_export_gate import (  # noqa: E402
  BUCKETS,
  CHECKPOINT_SHA256,
  STRUCTURES,
  _bucket_for,
  _load_model,
  _pad_inputs,
  _sha256,
)

logger = logging.getLogger("protonpotts_decode_export_gate")

V = 30
SEED = 5
FIXED_EVERY, FIXED_OFFSET = 7, 3
N_REPEATS = 5
FRAGILE = 1.0e-5
ENCODE_NAMES = ["coords", "present", "residue_idx", "chain_index"]
DECODE_NAMES = ["h_v", "h_e", "e_idx", "present", "pad_valid", "s_true", "designed", "temperature", "bias", "uniforms", "noise"]
CONFIGS = ("default", "bias_temp", "fixed")


def _structures(root: Path) -> list[dict[str, Any]]:
  from aminx.families.protonpotts_mpnn.features import featurize_pdb  # noqa: PLC0415

  out = []
  for name, (rel, want) in STRUCTURES.items():
    path = root / "energy_benchmark_datasets" / rel
    if _sha256(path) != want:
      msg = f"{path} sha256 does not match the pin"
      raise RuntimeError(msg)
    feats = featurize_pdb(path, None)
    length = int(feats["S"].shape[1])
    raw = {
      "coords": np.asarray(feats["X"][0, :, :4, :], dtype=np.float32), "present": np.ones(length, dtype=np.float32),
      "residue_idx": np.asarray(feats["R_idx"][0], dtype=np.int32),
      "chain_index": np.asarray(feats["chain_labels"][0], dtype=np.int32), "pad_valid": np.ones(length, dtype=np.bool_),
    }
    out.append({"name": name, "length": length, "raw": raw, "native": np.asarray(feats["S"][0], dtype=np.int32)})
  return out


def _margin(probs_sample: np.ndarray, order: np.ndarray, uniforms: np.ndarray, cdf_order: np.ndarray, length: int) -> float:
  """Smallest relative distance of any real step's draw to a CDF boundary (accumulating in upstream's order)."""
  worst = np.inf
  for k, pos in enumerate(order):
    if int(pos) >= length:
      continue
    row = probs_sample[int(pos)][cdf_order]
    cdf = np.cumsum(row)
    target = np.float32(uniforms[k]) * cdf[-1]
    worst = min(worst, float(np.abs(cdf - target).min() / cdf[-1]))
  return float(worst)


def _cells(model: Any, structures: list[dict[str, Any]]) -> list[dict[str, Any]]:  # noqa: ANN401
  import jax  # noqa: PLC0415
  import jax.numpy as jnp  # noqa: PLC0415

  from aminx.families.protonpotts_mpnn.decode import ProtonPottsARDecode  # noqa: PLC0415
  from aminx.families.protonpotts_mpnn.decode_export import make_decode, make_encode, upstream_cdf_order  # noqa: PLC0415

  encode = jax.jit(make_encode(model))
  decode = jax.jit(make_decode(model))
  cdf_order = upstream_cdf_order()
  rng = np.random.default_rng(SEED)
  cells = []
  for st in structures:
    bucket = _bucket_for(st["length"])
    padded = _pad_inputs(st["raw"], bucket)
    enc_in = [np.ascontiguousarray(padded[k]) for k in ENCODE_NAMES]
    h_v, h_e, e_idx = (np.asarray(x) for x in encode(*[jnp.asarray(a) for a in enc_in]))
    s_true = np.zeros(bucket, dtype=np.int32)
    s_true[: st["length"]] = st["native"]
    base_designed = np.zeros(bucket, dtype=bool)
    base_designed[: st["length"]] = True
    present = padded["present"]
    pad_valid = padded["pad_valid"]
    decode_cases = []
    for config in CONFIGS:
      temperature = np.full(bucket, 0.1, np.float32)
      bias = np.zeros((bucket, V), np.float32)
      designed = base_designed.copy()
      if config == "bias_temp":
        temperature = rng.uniform(0.3, 1.0, size=bucket).astype(np.float32)
        bias = (0.5 * rng.standard_normal((bucket, V))).astype(np.float32)
      elif config == "fixed":
        designed[FIXED_OFFSET : st["length"] : FIXED_EVERY] = False
      uniforms = rng.uniform(size=bucket).astype(np.float32)
      noise = rng.standard_normal(bucket).astype(np.float32)
      ins = [h_v, h_e, e_idx, present, pad_valid, s_true, designed, temperature, bias, uniforms, noise]
      ref = [np.asarray(x) for x in decode(*[jnp.asarray(a) for a in ins])]
      eager = ProtonPottsARDecode(layers=model.mpnn.decoder.layers, w_s_embed=model.mpnn.w_s_embed, w_out=model.mpnn.w_out)(
        *[jnp.asarray(a) for a in (h_v, h_e, e_idx, present, pad_valid, s_true, designed, temperature, bias, uniforms)],
        noise=jnp.asarray(noise), cdf_order=jnp.asarray(cdf_order))
      margin = _margin(np.asarray(eager.probs_sample), ref[1], uniforms, cdf_order, st["length"])
      decode_cases.append({"config": config, "inputs": ins, "reference": ref, "margin": margin, "fragile": margin < FRAGILE})
    cells.append({"structure": st["name"], "bucket": bucket, "length": st["length"],
                  "encode": {"inputs": enc_in, "reference": [h_v, h_e, e_idx]}, "decode": decode_cases})
    logger.info("%s bucket=%d margins=%s", st["name"], bucket, [round(c["margin"], 8) for c in decode_cases])
  return cells


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--potts-root", type=Path,
                      default=Path(os.environ.get("AMINX_POTTS_ROOT", "~/repos/PottsMPNN")).expanduser())
  parser.add_argument("--state-npz", type=Path,
                      default=Path(os.environ.get("AMINX_PROTONPOTTS_STATE", "~/scratch/v6_state.npz")).expanduser())
  parser.add_argument("--node-bin", default=os.environ.get("NODE_BIN"))
  parser.add_argument("--ort-web-dir", default=os.environ.get("ORT_WEB_DIR"))
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  from scripts.browser_validation.layer_a_common import emit  # noqa: PLC0415

  result: dict[str, Any] = {
    "versions": {}, "jax_arm_ok": False, "jax_enable_x64": False, "checkpoint_sha256": CHECKPOINT_SHA256, "cells": [],
    "graphs": {}, "n_cases": 0, "n_pass_cpu": 0, "n_pass_web": 0, "n_fragile_cases": 0, "all_exported": False,
    "all_pass": False, "deterministic": False, "n_nondeterministic": 0, "control_float_detected": False,
    "control_int_detected": False, "ort_web_available": False, "manifest_sha256": "", "error": "",
  }
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    if "jax2onnx" in sys.modules:
      msg = "jax2onnx imported before the JAX arm"
      raise RuntimeError(msg)  # noqa: TRY301
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    result["jax_enable_x64"] = bool(jax.config.jax_enable_x64)
    if result["jax_enable_x64"]:
      msg = "jax_enable_x64 is set; refusing"
      raise RuntimeError(msg)  # noqa: TRY301
    from aminx.families.protonpotts_mpnn.decode_export import make_decode, make_encode, upstream_cdf_order  # noqa: PLC0415
    from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6  # noqa: PLC0415

    model = _load_model(args.state_npz)
    cells = _cells(model, _structures(args.potts_root))
    result["n_fragile_cases"] = sum(c["fragile"] for cell in cells for c in cell["decode"])

    # ---- controls before any conversion --------------------------------------------------------------
    first = cells[0]
    ref_case = first["decode"][0]
    noise_key = jax.random.PRNGKey(1)
    perturbed = jax.tree_util.tree_map(
      lambda leaf: leaf + 1e-2 * jax.random.normal(noise_key, leaf.shape, leaf.dtype)
      if (eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating)) else leaf, model.mpnn.decoder)
    p_out = [np.asarray(x) for x in jax.jit(make_decode(eqx.tree_at(lambda m: m.mpnn.decoder, model, perturbed)))(
      *[jnp.asarray(a) for a in ref_case["inputs"]])]
    result["control_float_detected"] = not compare([ref_case["reference"][2]], [p_out[2]])["ok"]
    other = list(ref_case["inputs"])
    other[-1] = np.ascontiguousarray(other[-1][::-1])
    o_out = [np.asarray(x) for x in jax.jit(make_decode(model))(*[jnp.asarray(a) for a in other])]
    result["control_int_detected"] = not compare(ref_case["reference"][:2], o_out[:2])["ok"]
    result["jax_arm_ok"] = True
    logger.info("JAX arm: %d cells in %.1fs", len(cells), time.perf_counter() - t0)

    # ---- conversion ---------------------------------------------------------------------------------------
    import onnxruntime as ort  # noqa: PLC0415
    import xtrax  # noqa: PLC0415
    from xtrax.export import ONNX, convert_to_onnx, run_onnx  # noqa: PLC0415

    from scripts.browser_validation.p07_knobs_gate import embed_external_data  # noqa: PLC0415

    fns = {"encode": make_encode(model), "decode": make_decode(model)}
    names = {"encode": ENCODE_NAMES, "decode": DECODE_NAMES}
    exported_ok = True
    entries: dict[int, dict[str, Any]] = {b: {} for b in BUCKETS}
    for bucket in BUCKETS:
      cell = next((c for c in cells if c["bucket"] == bucket), None)
      if cell is None:
        continue
      examples = {"encode": cell["encode"]["inputs"], "decode": cell["decode"][0]["inputs"]}
      for graph in ("encode", "decode"):
        path = args.work_dir / f"protonpotts_{graph}_L{bucket}.onnx"
        try:
          ins = examples[graph]
          convert_to_onnx(fns[graph], [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in ins], ONNX, out_path=path)
          embed_external_data(path)
          facts = _graph_facts(path)
          entries[bucket][graph] = {
            "file": path.name, "bytes": facts["bytes"], "sha256": _sha256(path), "input_names": names[graph],
            "input_dtypes": [str(a.dtype) for a in ins], "input_shapes": [list(a.shape) for a in ins],
          }
          result["graphs"][f"{graph}_L{bucket}"] = {"converted": True, **facts}
        except Exception as exc:  # noqa: BLE001
          exported_ok = False
          result["graphs"][f"{graph}_L{bucket}"] = {"converted": False, "error": f"{type(exc).__name__}: {exc}"[:3000]}
    result["all_exported"] = exported_ok
    manifest = {
      "family": "protonpottsmpnn_decoder", "checkpoint_sha256": CHECKPOINT_SHA256, "alphabet": PROTONPOTTS_V6.name,
      "vocabulary": list(PROTONPOTTS_V6.symbols), "n_tokens": PROTONPOTTS_V6.size,
      "cdf_order": [int(i) for i in upstream_cdf_order()], "fragile_margin": FRAGILE,
      "buckets": [{"bucket": b, "graphs": entries[b]} for b in BUCKETS if entries[b]],
    }
    manifest_path = args.work_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result["manifest_sha256"] = _sha256(manifest_path)

    # ---- ORT-CPU (+ determinism) and ORT-Web ---------------------------------------------------------------
    web_cases: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    def add_case(cell: dict[str, Any], graph: str, label: str, ins: list[np.ndarray], ref: list[np.ndarray], *,
                 repeat: bool, fragile: bool = False) -> None:
      path = args.work_dir / f"protonpotts_{graph}_L{cell['bucket']}.onnx"
      rec = {"structure": cell["structure"], "graph": graph, "label": label, "fragile": fragile, "cpu": "not_converted",
             "web": "not_converted", "deterministic": None}
      checks.append(rec)
      if not result["graphs"][f"{graph}_L{cell['bucket']}"]["converted"]:
        return
      cmp_ref = ref[1:2] if fragile else ref  # fragile: the decoding order only
      pick = (lambda outs: outs[1:2]) if fragile else (lambda outs: outs)  # noqa: E731
      try:
        outs = run_onnx(path, *ins)
        cmp = compare(cmp_ref, pick(outs))
        rec.update(cpu="pass" if cmp["ok"] else "mismatch", cpu_max_rel=cmp["max_rel"], cpu_why=cmp["why"])
      except Exception as exc:  # noqa: BLE001
        rec.update(cpu="not_runnable", error=f"{type(exc).__name__}: {exc}"[:1500])
      if repeat and rec["cpu"] in ("pass", "mismatch"):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feed = {i.name: np.asarray(a) for i, a in zip(sess.get_inputs(), ins, strict=True)}
        runs = [sess.run(None, feed) for _ in range(N_REPEATS)]
        rec["deterministic"] = all(all(np.array_equal(r0, rk) for r0, rk in zip(runs[0], run, strict=True)) for run in runs[1:])
      case_id = f"{cell['structure']}_{graph}_{label}"
      io_dir = args.work_dir / f"io_{case_id}"
      io_dir.mkdir(exist_ok=True)
      specs = []
      for i, arr in enumerate(ins):
        arr = np.ascontiguousarray(arr)
        tag = DTYPE_TAG[arr.dtype]
        f = io_dir / f"in_{i}.bin"
        (arr.astype(np.uint8) if tag == "bool" else arr).tofile(f)
        specs.append({"file": str(f), "dtype": tag, "dims": list(arr.shape)})
      web_cases.append({"id": case_id, "model": str(path), "out_dir": str(io_dir), "inputs": specs})
      rec["case_id"] = case_id
      rec["ref"] = cmp_ref
      rec["pick"] = fragile

    seen_bucket: set[int] = set()
    for cell in cells:
      first_in_bucket = cell["bucket"] not in seen_bucket
      seen_bucket.add(cell["bucket"])
      add_case(cell, "encode", "all", cell["encode"]["inputs"], cell["encode"]["reference"], repeat=first_in_bucket)
      for i, dc in enumerate(cell["decode"]):
        add_case(cell, "decode", dc["config"], dc["inputs"], dc["reference"], repeat=first_in_bucket and i == 0,
                 fragile=dc["fragile"])

    web = _run_web(web_cases, args.work_dir, args.node_bin, args.ort_web_dir)
    result["ort_web_available"] = web["available"]
    result["ort_web_note"] = web["why"]
    for rec in checks:
      if rec["cpu"] == "not_converted":
        continue
      got = web["results"].get(rec.get("case_id", ""))
      if not web["available"] or got is None:
        rec["web"] = "unavailable"
      elif not got["ok"]:
        rec.update(web="not_runnable", web_error=(got.get("error") or "")[:1500])
      else:
        outs = _read_web_outputs(got)
        cmp = compare(rec["ref"], outs[1:2] if rec["pick"] else outs)
        rec.update(web="pass" if cmp["ok"] else "mismatch", web_max_rel=cmp["max_rel"], web_why=cmp["why"],
                   web_num_threads=got.get("num_threads"))
    for rec in checks:
      rec.pop("ref", None)
      rec.pop("pick", None)

    import jax2onnx  # noqa: PLC0415

    result["versions"] = {"jax": jax.__version__, "xtrax": xtrax.__version__,
                          "jax2onnx": getattr(jax2onnx, "__version__", None),
                          "onnxruntime": ort.__version__, "onnxruntime_web": web.get("ort_web_version")}
    result["cells"] = checks
    result["n_cases"] = len(checks)
    result["n_pass_cpu"] = sum(r["cpu"] == "pass" for r in checks)
    result["n_pass_web"] = sum(r["web"] == "pass" for r in checks)
    result["n_nondeterministic"] = sum(r["deterministic"] is False for r in checks)
    result["deterministic"] = any(r["deterministic"] for r in checks) and result["n_nondeterministic"] == 0
    result["all_pass"] = bool(checks) and exported_ok and all(r["cpu"] == "pass" and r["web"] == "pass" for r in checks)
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("cpu %s/%s web %s fragile %s deterministic %s", result["n_pass_cpu"], result["n_cases"], result["n_pass_web"],
              result["n_fragile_cases"], result["deterministic"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
