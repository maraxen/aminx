"""X4 converge gate: browser potts_converge (JS loop over the one-sweep graph) vs aminx's while_loop.

task_id 261007_potts-onnx-export, backlog #5814. Sidecar: potts_converge_gate.bth.toml (pre-registered).

aminx: PottsRefine("potts_converge", max_iters=1000) -- lax.while_loop(ener != 0 and iters < 1000)
over sweeps, sweep i reading uniforms row i (refine.py:316-331). Browser: potts_sampler.mjs
optimizationMode "potts_converge" calls the exported ONE-sweep refine graph repeatedly, placing
uniforms row i in the graph's row 0 and stopping on the graph's ener_delta output == 0 or 1000
sweeps. Full browser path (PDB -> inputs -> sample) under Node wasm on the X2 artifacts passed in.
Graded: decoded tokens, refined tokens and sweep count n_iters exact; final ener_delta equal within
the X0/X2 float bar (or both exactly 0).
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
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

from scripts.browser_validation.potts_export_gate import CHECKPOINT_SHA256, _cells, _sha256, _structures  # noqa: E402
from scripts.browser_validation.potts_knobs_gate import _spec  # noqa: E402
from scripts.browser_validation.potts_loop_gate import _verify_artifacts, _write  # noqa: E402

logger = logging.getLogger("potts_converge_gate")

FLOAT_REL_BAR = 1.0e-4
MAX_ITERS = 1000
SEEDS = (31, 32)
CELLS = (("3dkm.pdb", 128), ("4jox.pdb", 128))
DECODE_TEMPERATURE = 0.1


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0915
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--models", type=Path, required=True)
  parser.add_argument("--manifest-sha256", required=True)
  parser.add_argument("--potts-root", type=Path,
                      default=Path(os.environ.get("AMINX_POTTS_ROOT", "~/repos/PottsMPNN")).expanduser())
  parser.add_argument("--node-bin", default=os.environ.get("NODE_BIN"))
  parser.add_argument("--ort-web-dir", default=os.environ.get("ORT_WEB_DIR"))
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  from scripts.browser_validation.layer_a_common import emit  # noqa: PLC0415

  result: dict[str, Any] = {
    "jax_arm_ok": False, "artifacts_verified": False, "node_ok": False, "manifest_sha256": "",
    "samples": [], "n_samples": 0, "n_pass": 0, "all_exact": False, "control_tokens_detected": False,
    "n_iters_seen": [], "error": "",
  }
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    if jax.config.jax_enable_x64:
      msg = "jax_enable_x64 is set; refusing"
      raise RuntimeError(msg)
    from aminx.families.potts_mpnn.decode import floor_temperature  # noqa: PLC0415
    from aminx.families.potts_mpnn.driver import PottsMPNNDriver, _chain_sequences, _featurize_one  # noqa: PLC0415
    from aminx.families.potts_mpnn.etab import pad_etab_energy  # noqa: PLC0415
    from aminx.families.potts_mpnn.featurize import parse_pdb_upstream  # noqa: PLC0415
    from aminx.families.potts_mpnn.refine import PottsRefine  # noqa: PLC0415
    from aminx.families.potts_mpnn.sample_host import _dummy_binding, prepare_sample  # noqa: PLC0415
    from aminx.run.options import PottsMPNNOptions  # noqa: PLC0415

    _verify_artifacts(args.models, args.manifest_sha256)
    result["artifacts_verified"] = True
    result["manifest_sha256"] = args.manifest_sha256
    ckpt = args.potts_root / "vanilla_model_weights" / "pottsmpnn_20.pt"
    if _sha256(ckpt) != CHECKPOINT_SHA256:
      msg = f"{ckpt} does not match the pinned checkpoint"
      raise RuntimeError(msg)
    model = PottsMPNNDriver().load(SimpleNamespace(model_local_path=str(ckpt), num_samples=1, potts_mpnn=None,
                                                   run_spec=None))
    _unused, fns = _cells(model, _structures(args.potts_root / "inputs" / "example_pdbs"), seed=0)
    options = PottsMPNNOptions()
    opt_t = floor_temperature(float(options.optimization_temperature))
    refiner = PottsRefine(layers=model.mpnn.decoder.layers, w_s_embed=model.mpnn.w_s_embed, w_out=model.mpnn.w_out)
    pdb_dir = args.potts_root / "inputs" / "example_pdbs"

    refs: dict[str, dict[str, Any]] = {}
    node_cells = []
    for pdb_name, bucket in CELLS:
      parsed = parse_pdb_upstream(pdb_dir / pdb_name, skip_gaps=options.skip_gaps)[0]
      features = _featurize_one(parsed, options, str(parsed["name"]))
      chains = tuple((c.letter, c.sequence) for c in _chain_sequences(parsed, features))
      r = prepare_sample(features, chains, options, _spec({}), l_pad=bucket)
      h_v, h_e, e_idx, forward, _table = (np.asarray(x) for x in jax.jit(fns[bucket]["encode"])(
        r.coords, r.present, r.residue_idx, r.chain_index, r.pad_valid))
      etab_pad = pad_etab_energy(jnp.asarray(forward))
      tables = _dummy_binding(bucket, 22, jnp.float32)

      def converge(seq, order, uniforms, *, _r=r, _etab=etab_pad, _e=e_idx, _hv=h_v, _he=h_e, _tables=tables):  # noqa: ANN001, ANN202
        out = refiner(
          "potts_converge", seq, _etab, _e, _r.pad_valid, _r.present, _r.chain_mask, _r.chain_m_pos, order,
          uniforms, _r.omit, _r.bias, _r.bias_by_res, _r.pssm_coef, _r.pssm_bias, _r.pssm_log_odds_mask,
          _r.omit_aa_mask, _r.tie_groups, _r.tied_beta, _hv, _he, _tables,
          temperature=opt_t, pssm_multi=float(options.pssm_multi), pssm_bias_flag=bool(options.pssm_bias_flag),
          pssm_log_odds_flag=bool(options.pssm_log_odds_flag), binding="none", tied=False,
          tied_epistasis=bool(options.tied_epistasis), max_iters=MAX_ITERS,
        )
        return out.sequence, out.n_iters, out.ener_delta

      converge_jit = jax.jit(converge)
      for seed in SEEDS:
        rng = np.random.default_rng(seed)
        randn = rng.standard_normal(bucket).astype(np.float32)
        uniforms = rng.uniform(size=bucket).astype(np.float32)
        refine_uniforms = rng.uniform(size=(MAX_ITERS, bucket)).astype(np.float32)
        dec_in = [h_v, h_e, e_idx, r.present, r.pad_valid, r.s_true, r.chain_mask, r.chain_m_pos, r.tie_groups,
                  r.tied_beta, randn, uniforms, r.omit, r.bias, r.bias_by_res, r.pssm_coef, r.pssm_bias,
                  r.pssm_log_odds_mask, r.omit_aa_mask, np.asarray([DECODE_TEMPERATURE], np.float32)]
        seq, _rank, order, _hvs = (np.asarray(x) for x in jax.jit(fns[bucket]["decode"])(*dec_in))
        refined, n_iters, ener = (np.asarray(x) for x in converge_jit(
          jnp.asarray(seq, jnp.int32), jnp.asarray(order, jnp.int32), jnp.asarray(refine_uniforms)))
        sid = f"{pdb_name.removesuffix('.pdb')}_L{bucket}__s{seed}"
        refs[sid] = {"pdb": pdb_name, "bucket": bucket, "seed": seed, "sequence": seq, "refined": refined,
                     "n_iters": int(n_iters), "ener_delta": float(ener)}
        cdir = args.work_dir / sid
        cdir.mkdir(exist_ok=True)
        node_cells.append({
          "id": sid, "pdb": str(pdb_dir / pdb_name), "bucket": bucket, "controls": {},
          "options": {"refine": True, "temperature": DECODE_TEMPERATURE,
                      "optimizationTemperature": float(options.optimization_temperature),
                      "optimizationMode": "potts_converge"},
          "out_dir": str(cdir),
          "noise": {"randn": _write(randn, cdir / "randn.bin"), "uniforms": _write(uniforms, cdir / "uniforms.bin"),
                    "refineUniforms": _write(refine_uniforms, cdir / "refine_uniforms.bin")},
        })
    result["jax_arm_ok"] = True
    result["n_iters_seen"] = sorted({v["n_iters"] for v in refs.values()})
    logger.info("JAX arm: %d samples in %.1fs, n_iters %s", len(refs), time.perf_counter() - t0, result["n_iters_seen"])

    cells_path = args.work_dir / "converge_cells.json"
    out_path = args.work_dir / "converge_result.json"
    cells_path.write_text(json.dumps({"cells": node_cells}))
    runner = REPO / "browser" / "potts-sampler" / "run_potts_browser_node.mjs"
    proc = subprocess.run(  # noqa: S603 -- fixed argv we built
      [args.node_bin, str(runner), "--ort-dir", args.ort_web_dir, "--models", str(args.models),
       "--cells", str(cells_path), "--out", str(out_path)],
      capture_output=True, text=True, timeout=14400, check=False,
    )
    if proc.returncode != 0 or not out_path.exists():
      msg = f"node runner rc={proc.returncode}: {proc.stderr[-3000:]}"
      raise RuntimeError(msg)
    node = json.loads(out_path.read_text())
    result["node_ok"] = True
    got = {}
    for rec in node["results"]:
      outs = {}
      if rec["ok"]:
        for name, spec in rec["outputs"].items():
          outs[name] = spec["value"] if "value" in spec else np.fromfile(spec["file"], dtype=np.int32)
      got[rec["id"]] = {"ok": rec["ok"], "error": rec.get("error"), "outputs": outs}

    for sid, ref in refs.items():
      g = got.get(sid, {"ok": False, "error": "missing", "outputs": {}})
      row = {k: ref[k] for k in ("pdb", "bucket", "seed", "n_iters")} | {"id": sid, "ran": g["ok"]}
      if not g["ok"]:
        row.update(passed=False, error=(g["error"] or "")[:1500])
      else:
        o = g["outputs"]
        row["sequence_exact"] = bool(np.array_equal(o["sequence"], ref["sequence"]))
        row["refined_exact"] = bool(np.array_equal(o["refined_sequence"], ref["refined"]))
        row["js_n_iters"] = int(o.get("n_iters", -1))
        row["n_iters_equal"] = row["js_n_iters"] == ref["n_iters"]
        js_ener = float(o.get("ener_delta", float("nan")))
        both_zero = js_ener == 0.0 and ref["ener_delta"] == 0.0
        row["ener_rel"] = 0.0 if both_zero else abs(js_ener - ref["ener_delta"]) / max(abs(ref["ener_delta"]), 1e-30)
        row["passed"] = (row["sequence_exact"] and row["refined_exact"] and row["n_iters_equal"]
                         and row["ener_rel"] <= FLOAT_REL_BAR)
      result["samples"].append(row)

    hits, n = 0, 0
    for sid, ref in refs.items():
      other = sid.rsplit("__s", 1)[0] + f"__s{SEEDS[1] if ref['seed'] == SEEDS[0] else SEEDS[0]}"
      g = got.get(sid)
      if g and g["ok"]:
        n += 1
        hits += int(not np.array_equal(g["outputs"]["refined_sequence"], refs[other]["refined"]))
    result["control_tokens_detected"] = n > 0 and hits == n
    result["n_samples"] = len(result["samples"])
    result["n_pass"] = sum(bool(s.get("passed")) for s in result["samples"])
    result["all_exact"] = result["n_samples"] > 0 and result["n_pass"] == result["n_samples"]
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("converge gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("exact %s/%s", result["n_pass"], result["n_samples"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
