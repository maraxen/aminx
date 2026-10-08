"""X5 runtime-controls gate: the FULL browser path with design controls, token-exact vs JAX.

task_id 261007_potts-onnx-export, backlog #5815. Sidecar: potts_knobs_gate.bth.toml (pre-registered).

Browser arm (Node, onnxruntime-web wasm, 1 thread, no Python inputs):
  PDB text -> potts_inputs.buildPottsInputs -> potts_controls.applyPottsControls -> potts_sampler.sample
JAX arm: parse_pdb_upstream -> _featurize_one -> prepare_sample(spec with the same controls) -> the
export-gate graph functions (encode, decode, energy, refine) with the same noise and temperatures.

Artifacts: the temperature-input X2 set (verified by the manifest sha256 passed in). Per case x cell x
seed: decoded tokens, decoding order and refined tokens exact, sample energy max|d|/|ref| <= 1e-4
(the X0/X2 bars). Each control must also visibly ACT (knob effects, below), because an ignored
control passes exactness trivially when both arms ignore it the same way -- except the JAX arm is
aminx's own prepare_sample, so inertness there would be an aminx defect worth knowing either way.

The JAX arm (`jax_reference`), the per-sample grading (`grade_sample`) and the cross-seed comparator
control (`cross_seed_hits`) are importable so the browser-page parity gate (potts_browser_parity.py)
grades against exactly the same reference and comparator as this gate.
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

from scripts.browser_validation.potts_export_gate import (
  CHECKPOINT_SHA256,
  N_MUTANTS,
  _cells,
  _sha256,
  _structures,
)
from scripts.browser_validation.potts_loop_gate import _verify_artifacts, _write


def _require(ok: bool, msg: str) -> None:  # noqa: FBT001 -- a plain predicate, not a flag
  """Raise RuntimeError(msg) unless `ok`. Keeps raise statements out of try blocks (TRY301)."""
  if not ok:
    raise RuntimeError(msg)

logger = logging.getLogger("potts_knobs_gate")

FLOAT_REL_BAR = 1.0e-4
SEEDS = (21, 22)
CELLS = (("3gg7.pdb", 256), ("3dkm.pdb", 128))
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
DEFAULT = {"refine": True, "temperature": 0.1, "optimizationTemperature": 0.0}


def _cases(l_total: int) -> dict[str, dict[str, Any]]:
  w_bias = [0.0] * 21
  w_bias[ALPHABET.index("W")] = 3.0
  p_bias = [[0.0] * 21 for _ in range(l_total)]
  for row in range(min(30, l_total)):
    p_bias[row][ALPHABET.index("P")] = 3.0
  combo_bias = [0.0] * 21
  combo_bias[ALPHABET.index("W")] = 2.0
  fixed = sorted({*range(20), *range(0, l_total, 7)})
  return {
    "default": {"controls": {}, "options": dict(DEFAULT)},
    "omit": {"controls": {"omit_aa": "ACDW"}, "options": dict(DEFAULT)},
    "fixed": {"controls": {"fixed_positions": fixed}, "options": dict(DEFAULT)},
    "bias_global": {"controls": {"bias": w_bias}, "options": dict(DEFAULT)},
    "bias_per_res": {"controls": {"bias": p_bias}, "options": dict(DEFAULT)},
    "temperature": {"controls": {}, "options": {**DEFAULT, "temperature": 1.0}},
    "opt_temperature": {"controls": {}, "options": {**DEFAULT, "optimizationTemperature": 0.5}},
    "no_refine": {"controls": {}, "options": {**DEFAULT, "refine": False}},
    "combined": {"controls": {"omit_aa": "C", "fixed_positions": list(range(10)), "bias": combo_bias},
                 "options": {**DEFAULT, "temperature": 0.3}},
  }


def _spec(controls: dict[str, Any]) -> SimpleNamespace:
  bias = controls.get("bias")
  return SimpleNamespace(
    fixed_positions=None if "fixed_positions" not in controls else np.asarray(controls["fixed_positions"], np.int32),
    omit_aa=tuple(controls.get("omit_aa", "")),
    bias=None if bias is None else np.asarray(bias, np.float32),
    num_samples=1,
  )


def jax_reference(
  potts_root: Path,
  work_dir: Path,
  cells: tuple[tuple[str, int], ...] = CELLS,
  seeds: tuple[int, ...] = SEEDS,
  case_names: frozenset[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
  """JAX arm: aminx's own sample path plus the export-gate graph functions, per (case, cell, seed).

  Returns `(refs, node_cells)`. `refs` maps sample id -> reference outputs (tokens, order, refined
  tokens, energy, and the controls used). `node_cells` is the cell list the browser runner consumes
  (absolute PDB paths and float32 noise files written under `work_dir/<sample id>/`). `case_names`
  restricts the cases built (None = all nine). Imports JAX; only call from a gate allowed to.
  """
  import jax  # noqa: PLC0415
  import jax.numpy as jnp  # noqa: PLC0415

  if jax.config.jax_enable_x64:
    msg = "jax_enable_x64 is set; refusing"
    raise RuntimeError(msg)
  from aminx.families.potts_mpnn.driver import (  # noqa: PLC0415
    PottsMPNNDriver,
    _chain_sequences,
    _featurize_one,
  )
  from aminx.families.potts_mpnn.etab import model_to_etab, pad_etab_energy  # noqa: PLC0415
  from aminx.families.potts_mpnn.featurize import parse_pdb_upstream  # noqa: PLC0415
  from aminx.families.potts_mpnn.sample_host import prepare_sample  # noqa: PLC0415
  from aminx.run.options import PottsMPNNOptions  # noqa: PLC0415

  ckpt = potts_root / "vanilla_model_weights" / "pottsmpnn_20.pt"
  if _sha256(ckpt) != CHECKPOINT_SHA256:
    msg = f"{ckpt} does not match the pinned checkpoint"
    raise RuntimeError(msg)
  model = PottsMPNNDriver().load(SimpleNamespace(model_local_path=str(ckpt), num_samples=1, potts_mpnn=None,
                                                 run_spec=None))
  # Graph functions only; _cells' own sample data is not used here.
  _unused, fns = _cells(model, _structures(potts_root / "inputs" / "example_pdbs"), seed=0)
  options = PottsMPNNOptions()
  pdb_dir = potts_root / "inputs" / "example_pdbs"

  refs: dict[str, dict[str, Any]] = {}
  node_cells: list[dict[str, Any]] = []
  for pdb_name, bucket in cells:
    parsed = parse_pdb_upstream(pdb_dir / pdb_name, skip_gaps=options.skip_gaps)[0]
    features = _featurize_one(parsed, options, str(parsed["name"]))
    chains = tuple((c.letter, c.sequence) for c in _chain_sequences(parsed, features))
    l_total = int(features.L_total)
    base = prepare_sample(features, chains, options, _spec({}), l_pad=bucket)
    enc = [np.asarray(x) for x in jax.jit(fns[bucket]["encode"])(
      base.coords, base.present, base.residue_idx, base.chain_index, base.pad_valid)]
    h_v, h_e, e_idx, forward, table = enc
    etab_pad = np.asarray(pad_etab_energy(jnp.asarray(forward)))
    for case, cfg in _cases(l_total).items():
      if case_names is not None and case not in case_names:
        continue
      r = prepare_sample(features, chains, options, _spec(cfg["controls"]), l_pad=bucket)
      opt = cfg["options"]
      for seed in seeds:
        rng = np.random.default_rng(seed)
        randn = rng.standard_normal(bucket).astype(np.float32)
        uniforms = rng.uniform(size=bucket).astype(np.float32)
        refine_uniforms = rng.uniform(size=(8, bucket)).astype(np.float32)
        dec_in = [h_v, h_e, e_idx, r.present, r.pad_valid, r.s_true, r.chain_mask, r.chain_m_pos, r.tie_groups,
                  r.tied_beta, randn, uniforms, r.omit, r.bias, r.bias_by_res, r.pssm_coef, r.pssm_bias,
                  r.pssm_log_odds_mask, r.omit_aa_mask, np.asarray([opt["temperature"]], np.float32)]
        seq, _rank, order, _hvs = (np.asarray(x) for x in jax.jit(fns[bucket]["decode"])(*dec_in))
        etab_seqs = np.repeat(np.asarray(model_to_etab(jnp.asarray(seq)))[None], N_MUTANTS, 0).astype(np.int32)
        energy = float(np.asarray(jax.jit(fns[bucket]["energy"])(table, e_idx, r.pad_valid, etab_seqs)[0])[0])
        refined = None
        if opt["refine"]:
          ref_in = [seq.astype(np.int32), etab_pad, e_idx, r.pad_valid, r.present, r.chain_mask, r.chain_m_pos,
                    order.astype(np.int32), refine_uniforms, r.omit, r.bias, r.bias_by_res, r.pssm_coef,
                    r.pssm_bias, r.pssm_log_odds_mask, r.omit_aa_mask, r.tie_groups, r.tied_beta, h_v, h_e,
                    np.asarray([opt["optimizationTemperature"]], np.float32)]
          refined = np.asarray(jax.jit(fns[bucket]["refine"])(*ref_in)[0])
        sid = f"{case}__{pdb_name.removesuffix('.pdb')}_L{bucket}__s{seed}"
        refs[sid] = {"case": case, "pdb": pdb_name, "bucket": bucket, "seed": seed, "l_total": l_total,
                     "sequence": seq, "decoding_order": order, "refined_sequence": refined,
                     "sample_energy": energy, "s_true": np.asarray(r.s_true),
                     "designable": (np.asarray(r.chain_m_pos) * np.asarray(r.chain_mask) > 0)[:l_total],
                     "controls": cfg["controls"]}
        cdir = work_dir / sid
        cdir.mkdir(parents=True, exist_ok=True)
        node_cells.append({
          "id": sid, "pdb": str(pdb_dir / pdb_name), "bucket": bucket, "controls": cfg["controls"],
          "options": opt, "out_dir": str(cdir),
          "noise": {"randn": _write(randn, cdir / "randn.bin"), "uniforms": _write(uniforms, cdir / "uniforms.bin"),
                    "refineUniforms": _write(refine_uniforms, cdir / "refine_uniforms.bin")},
        })
  return refs, node_cells


def grade_sample(sid: str, ref: dict[str, Any], g: dict[str, Any]) -> dict[str, Any]:
  """Exactness row for one browser sample `g` ({ok, error, outputs}) against its JAX reference."""
  row = {k: ref[k] for k in ("case", "pdb", "bucket", "seed")} | {"id": sid, "ran": g["ok"]}
  if not g["ok"]:
    row.update(passed=False, error=(g["error"] or "")[:1500])
    return row
  o = g["outputs"]
  row["sequence_exact"] = bool(np.array_equal(o["sequence"], ref["sequence"]))
  row["order_exact"] = bool(np.array_equal(o["decoding_order"], ref["decoding_order"]))
  if ref["refined_sequence"] is None:
    row["refined_exact"] = "refined_sequence" not in o
  else:
    row["refined_exact"] = bool(np.array_equal(o.get("refined_sequence"), ref["refined_sequence"]))
  row["energy_rel"] = abs(o["sample_energy"] - ref["sample_energy"]) / max(abs(ref["sample_energy"]), 1e-30)
  row["passed"] = (row["sequence_exact"] and row["order_exact"] and row["refined_exact"]
                   and row["energy_rel"] <= FLOAT_REL_BAR)
  return row


def cross_seed_hits(refs: dict[str, dict[str, Any]], got: dict[str, dict[str, Any]]) -> tuple[int, int]:
  """Comparator control: each browser sample vs the JAX tokens of the OTHER seed, same case/cell.

  Returns `(hits, n)` where `hits` counts samples whose tokens differ from the other seed's (the
  comparator is live if every sample is flagged) and `n` counts samples that ran.
  """
  hits, n = 0, 0
  for sid, ref in refs.items():
    other = sid.rsplit("__s", 1)[0] + f"__s{SEEDS[1] if ref['seed'] == SEEDS[0] else SEEDS[0]}"
    g = got.get(sid)
    if g and g["ok"]:
      n += 1
      hits += int(not np.array_equal(g["outputs"]["sequence"], refs[other]["sequence"]))
  return hits, n


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
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
    "samples": [], "n_samples": 0, "n_pass": 0, "all_exact": False, "knob_effects": {},
    "knob_effects_ok": False, "control_tokens_detected": False, "error": "",
  }
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    _verify_artifacts(args.models, args.manifest_sha256)
    result["artifacts_verified"] = True
    result["manifest_sha256"] = args.manifest_sha256
    refs, node_cells = jax_reference(args.potts_root, args.work_dir)
    result["jax_arm_ok"] = True
    logger.info("JAX arm: %d samples in %.1fs", len(refs), time.perf_counter() - t0)

    cells_path = args.work_dir / "knob_cells.json"
    out_path = args.work_dir / "knob_result.json"
    cells_path.write_text(json.dumps({"cells": node_cells}))
    runner = REPO / "browser" / "potts-sampler" / "run_potts_browser_node.mjs"
    proc = subprocess.run(  # noqa: S603 -- fixed argv we built
      [args.node_bin, str(runner), "--ort-dir", args.ort_web_dir, "--models", str(args.models),
       "--cells", str(cells_path), "--out", str(out_path)],
      capture_output=True, text=True, timeout=7200, check=False,
    )
    _require(
      proc.returncode == 0 and out_path.exists(),
      f"node runner rc={proc.returncode}: {proc.stderr[-3000:]}",
    )
    node = json.loads(out_path.read_text())
    result["node_ok"] = True
    got = {}
    for rec in node["results"]:
      outs = {}
      if rec["ok"]:
        for name, spec in rec["outputs"].items():
          outs[name] = spec["value"] if "value" in spec else np.fromfile(spec["file"], dtype=np.int32)
      got[rec["id"]] = {"ok": rec["ok"], "error": rec.get("error"), "outputs": outs}

    # ---- exactness ----------------------------------------------------------------------------
    for sid, ref in refs.items():
      result["samples"].append(grade_sample(sid, ref, got.get(sid, {"ok": False, "error": "missing", "outputs": {}})))

    # ---- knob effects (on the JAX reference, which the browser arm must equal exactly) --------
    def by(case: str) -> list[dict[str, Any]]:
      return [refs[k] for k in sorted(refs) if refs[k]["case"] == case]

    def letter_count(rows: list[dict[str, Any]], letter: str, upto: int | None = None) -> int:
      idx = ALPHABET.index(letter)
      return int(sum(np.sum(r["sequence"][: (upto or r["l_total"])] == idx) for r in rows))

    eff: dict[str, bool] = {}
    omit_idx = [ALPHABET.index(c) for c in "ACDW"]
    eff["omit_absent"] = all(
      not np.isin(r["sequence"][: r["l_total"]][r["designable"]], omit_idx).any()
      and not np.isin(r["refined_sequence"][: r["l_total"]][r["designable"]], omit_idx).any()
      for r in by("omit"))
    eff["fixed_native"] = all(
      np.array_equal(r["sequence"][r["controls"]["fixed_positions"]], r["s_true"][r["controls"]["fixed_positions"]])
      and np.array_equal(r["refined_sequence"][r["controls"]["fixed_positions"]],
                         r["s_true"][r["controls"]["fixed_positions"]])
      for r in by("fixed"))
    eff["bias_global_raises_W"] = letter_count(by("bias_global"), "W") > letter_count(by("default"), "W")
    eff["bias_per_res_raises_P"] = letter_count(by("bias_per_res"), "P", 30) > letter_count(by("default"), "P", 30)
    eff["temperature_changes_draw"] = all(
      not np.array_equal(a["sequence"], b["sequence"]) for a, b in zip(by("temperature"), by("default"), strict=True))
    eff["opt_temperature_changes_refine"] = any(
      not np.array_equal(a["refined_sequence"], b["refined_sequence"])
      for a, b in zip(by("opt_temperature"), by("default"), strict=True))
    eff["no_refine_keeps_decode"] = all(
      np.array_equal(a["sequence"], b["sequence"]) for a, b in zip(by("no_refine"), by("default"), strict=True))
    result["knob_effects"] = eff
    result["knob_effects_ok"] = all(eff.values())

    # Comparator control: each browser sample vs the JAX tokens of the OTHER seed, same case/cell.
    hits, n = cross_seed_hits(refs, got)
    result["control_tokens_detected"] = n > 0 and hits == n
    result["n_samples"] = len(result["samples"])
    result["n_pass"] = sum(bool(r.get("passed")) for r in result["samples"])
    result["all_exact"] = result["n_samples"] > 0 and result["n_pass"] == result["n_samples"]
  except Exception as exc:
    logger.exception("knobs gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("exact %s/%s, knob effects %s", result["n_pass"], result["n_samples"], result["knob_effects"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
