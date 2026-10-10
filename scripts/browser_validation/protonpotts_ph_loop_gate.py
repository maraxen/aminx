"""X7b gate: the SHIPPING browser pH design loop (ph_design.mjs) vs aminx's design_structure, under Node/ORT-Web.

task_id 261009_protonpotts-onnx, backlog #5816 (B1). Sidecar: protonpotts_ph_loop_gate.bth.toml (pre-registered before this script
existed). The X6a scoring artifacts and the X7a per-block artifacts are verified by sha256 against their manifests, not re-exported.
The reference is ``design_structure`` -- the core of the real ``sample`` purpose -- run on the UNPADDED structure with the driver's own
``protonpotts_table``; the browser pads to a bucket, so agreement also covers padding.
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

from scripts.browser_validation.protonpotts_export_gate import STRUCTURES, _bucket_for, _load_model  # noqa: E402
from scripts.browser_validation.protonpotts_score_gate import _verify_artifacts  # noqa: E402

logger = logging.getLogger("protonpotts_ph_loop_gate")

ENERGY_BAR = 1.0e-4
SAMPLES = 2
BLOCK_ROUNDS = 10
MAX_MUTATIONS = 20
SEED = 31
CASES = (
  ("hpg_t0", ("HIS-P", "ASP-P", "GLU-P"), 0.0),
  ("hpg_t005", ("HIS-P", "ASP-P", "GLU-P"), 0.05),
  ("h_t005", ("HIS-P",), 0.05),
)


def _rel(ref: float, got: float) -> float:
  return abs(float(got) - float(ref)) / max(1.0, abs(float(ref)))


def _design(table, e_idx, native, binder, res_id, config, uniforms):  # noqa: ANN001, ANN202
  from aminx.families.protonpotts_mpnn.ph_design import design_structure  # noqa: PLC0415

  import jax.numpy as jnp  # noqa: PLC0415

  return design_structure(table, e_idx, jnp.asarray(native), binder, res_id, config,
                          uniforms=uniforms if config.temperature > 0 else None)


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--scoring-models", type=Path, required=True, help="X6a artifact dir")
  parser.add_argument("--scoring-manifest-sha256", required=True)
  parser.add_argument("--ph-models", type=Path, required=True, help="X7a artifact dir")
  parser.add_argument("--ph-manifest-sha256", required=True)
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
    "jax_arm_ok": False, "artifacts_verified": False, "node_ok": False, "num_threads": 0, "cases": [], "n_cases": 0,
    "n_pass": 0, "control_tokens_detected": False, "control_energy_detected": False,
    "scoring_manifest_sha256": args.scoring_manifest_sha256, "ph_manifest_sha256": args.ph_manifest_sha256, "error": "",
  }
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    if jax.config.jax_enable_x64:
      msg = "jax_enable_x64 is set; refusing"
      raise RuntimeError(msg)  # noqa: TRY301
    from aminx.families.protonpotts_mpnn.energy import protonpotts_energies, protonpotts_table  # noqa: PLC0415
    from aminx.families.protonpotts_mpnn.features import featurize_pdb, kept_residues  # noqa: PLC0415
    from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig  # noqa: PLC0415

    _verify_artifacts(args.scoring_models, args.scoring_manifest_sha256)
    _verify_artifacts(args.ph_models, args.ph_manifest_sha256)
    result["artifacts_verified"] = True
    model = _load_model(args.state_npz)

    rng = np.random.default_rng(SEED)
    refs: dict[str, Any] = {}
    node_cases: list[dict[str, Any]] = []
    rolled: dict[str, Any] = {}
    context: dict[str, Any] = {}
    for name, (rel, _sha) in STRUCTURES.items():
      pdb = args.potts_root / "energy_benchmark_datasets" / rel
      feats = featurize_pdb(pdb, None)
      kept = kept_residues(pdb)
      native = np.asarray(feats["S"][0], dtype=np.int32)
      length = native.shape[0]
      chains = [k[0] for k in kept]
      binder_chain = chains[-1]
      binder = np.asarray([c == binder_chain for c in chains])
      res_id = np.asarray([k[1] for k in kept], dtype=np.int32)
      coords = jnp.asarray(feats["X"][0, :, :4, :], dtype=jnp.float32)
      args_t = (coords, jnp.ones(length, jnp.float32), jnp.asarray(feats["R_idx"][0]),
                jnp.asarray(feats["chain_labels"][0]), jnp.ones(length, bool))
      table, e_idx = jax.jit(lambda *a, _m=model: protonpotts_table(_m, *a))(*args_t)
      context[name] = {"args_t": args_t, "native": native}
      for case, types, temperature in CASES:
        cid = f"{name}__{case}"
        config = PHDesignConfig(binder_chain=binder_chain, center_types=types, temperature=temperature,
                                samples_per_site=SAMPLES, max_mutations=MAX_MUTATIONS, block_max_rounds=BLOCK_ROUNDS)
        need = BLOCK_ROUNDS * MAX_MUTATIONS
        uniforms = [rng.uniform(size=need).astype(np.float32) for _ in range(SAMPLES)] if temperature > 0 else None
        refs[cid] = _design(table, e_idx, native, binder, res_id, config, uniforms)
        if temperature > 0:
          rolled[cid] = _design(table, e_idx, native, binder, res_id, config, [np.roll(u, 1) for u in uniforms])
        node_cases.append({
          "id": cid, "bucket": _bucket_for(length), "pdb_file": str(pdb), "labels": None,
          "config": {"binderChain": binder_chain, "centerTypes": list(types), "temperature": temperature,
                     "samplesPerSite": SAMPLES, "maxMutations": MAX_MUTATIONS, "blockMaxRounds": BLOCK_ROUNDS},
          "uniforms": None if uniforms is None else [u.tolist() for u in uniforms],
        })
        logger.info("reference %s: %d designs", cid, len(refs[cid]))
    result["jax_arm_ok"] = True
    logger.info("reference arm in %.1fs", time.perf_counter() - t0)

    cases_path = args.work_dir / "ph_cases.json"
    out_path = args.work_dir / "ph_result.json"
    cases_path.write_text(json.dumps({"cases": node_cases}))
    runner = REPO / "browser" / "protonpotts-scorer" / "run_ph_design_node.mjs"
    proc = subprocess.run(  # noqa: S603 -- fixed argv we built
      [args.node_bin, str(runner), "--ort-dir", args.ort_web_dir, "--scoring-models", str(args.scoring_models),
       "--ph-models", str(args.ph_models), "--cases", str(cases_path), "--out", str(out_path)],
      capture_output=True, text=True, timeout=7200, check=False,
    )
    if proc.returncode != 0 or not out_path.exists():
      msg = f"node runner rc={proc.returncode}: {proc.stderr[-3000:]}"
      raise RuntimeError(msg)  # noqa: TRY301
    node = json.loads(out_path.read_text())
    result["node_ok"] = True
    result["num_threads"] = int(node.get("num_threads") or 0)
    got = {r["id"]: r for r in node["results"]}

    def same(js_designs: list[dict[str, Any]], ref_designs: list[Any]) -> dict[str, Any]:
      row: dict[str, Any] = {"n_js": len(js_designs), "n_ref": len(ref_designs)}
      ok = len(js_designs) == len(ref_designs) > 0
      worst = 0.0
      sequences = designable = pins = draws = True
      for js, ref in zip(js_designs, ref_designs, strict=False):
        sequences &= bool(np.array_equal(np.asarray(js["sequence"]), ref.sequence))
        designable &= list(js["designable"]) == [int(d) for d in ref.designable]
        pins &= [(p["position"], p["protIdx"], p["resId"]) for p in js["pins"]] == [
          (p.position, p.prot_idx, p.res_id) for p in ref.pins] and js["label"] == ref.label
        draws &= int(js["n_draws"]) == int(ref.n_draws)
        worst = max(worst, _rel(ref.final_potts_energy, js["final_potts_energy"]),
                    _rel(ref.selective_energy, js["selective_energy"]))
      row.update(sequences_exact=bool(sequences), designable_equal=bool(designable), pins_equal=bool(pins),
                 draws_equal=bool(draws), worst_energy_rel=worst)
      row["passed"] = bool(ok and sequences and designable and pins and draws and worst <= ENERGY_BAR)
      return row

    tok_hits = tok_n = 0
    for cid, ref in refs.items():
      g = got.get(cid, {"ok": False, "error": "missing", "designs": None})
      if not g["ok"]:
        result["cases"].append({"id": cid, "ran": False, "passed": False, "error": (g.get("error") or "")[:2000]})
        continue
      row = {"id": cid, "ran": True, **same(g["designs"], ref)}
      result["cases"].append(row)
      if cid in rolled:
        tok_n += 1
        tok_hits += int(not all(np.array_equal(np.asarray(j["sequence"]), r.sequence)
                                for j, r in zip(g["designs"], rolled[cid], strict=False)))
    # energy control: the JS sequences scored under a perturbed Potts head must disagree with the JS energies
    noise_key = jax.random.PRNGKey(1)
    perturbed = jax.tree_util.tree_map(
      lambda leaf: leaf + 1e-2 * jax.random.normal(noise_key, leaf.shape, leaf.dtype)
      if (eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating)) else leaf, model.potts_head)
    bad_model = eqx.tree_at(lambda m: m.potts_head, model, perturbed)
    ctrl = next(cid for cid in refs if got.get(cid, {}).get("ok"))
    name = ctrl.split("__", 1)[0]
    designs = got[ctrl]["designs"]
    seqs = jnp.asarray(np.stack([np.asarray(d["sequence"]) for d in designs]).astype(np.int32))
    p_energy = np.asarray(protonpotts_energies(bad_model, *context[name]["args_t"], seqs))
    result["control_energy_detected"] = all(
      _rel(p, d["final_potts_energy"]) > ENERGY_BAR for p, d in zip(p_energy, designs, strict=True))
    result["control_tokens_detected"] = tok_n > 0 and tok_hits == tok_n
    result["n_cases"] = len(result["cases"])
    result["n_pass"] = sum(bool(c.get("passed")) for c in result["cases"])
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("ph loop gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("cases pass %s/%s", result["n_pass"], result["n_cases"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
