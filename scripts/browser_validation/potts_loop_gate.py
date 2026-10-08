"""X3 loop gate: the SHIPPING browser/potts-sampler/potts_sampler.mjs vs the JAX sampler, token-exact.

task_id 261007_potts-onnx-export, backlog #5813. Sidecar: potts_loop_gate.bth.toml (pre-registered).

The X2 gate (run d6506199) proved each exported graph in isolation. This gate proves the JS that
chains them: encode -> decode -> energy -> one-sweep refine, run by the shipping .mjs under Node
with onnxruntime-web (wasm, 1 thread) on the X2 artifacts (verified by sha256 against the X2
MANIFEST, not re-exported). The JAX reference is the same chain built from the same graph
functions (potts_export_gate._cells), fed IDENTICAL noise: per (structure, bucket) cell, three
seeds of decode randn + uniforms and refine uniforms, generated here and written for both arms.
Structure inputs are the Python driver path's (prepare_sample); the JS input builder is graded
separately (X3a). Comparator and bars are X0/X2's: integers exact, float max|d|/max|ref| <= 1e-4.
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

from scripts.browser_validation.potts_export_gate import (  # noqa: E402
  CHECKPOINT_SHA256,
  N_MUTANTS,
  _cells,
  _sha256,
  _structures,
)

logger = logging.getLogger("potts_loop_gate")

SEEDS = (11, 12, 13)
FLOAT_REL_BAR = 1.0e-4
STRUCT_FROM_ENCODE = ("coords", "present", "residue_idx", "chain_index", "pad_valid")
STRUCT_FROM_DECODE = {5: "s_true", 6: "chain_mask", 7: "chain_m_pos", 8: "tie_groups", 9: "tied_beta",
                      12: "omit", 13: "bias", 14: "bias_by_res", 15: "pssm_coef", 16: "pssm_bias",
                      17: "pssm_log_odds_mask", 18: "omit_aa_mask"}
TAG = {np.dtype(np.float32): "float32", np.dtype(np.int32): "int32", np.dtype(np.bool_): "bool"}


def _write(arr: np.ndarray, path: Path) -> dict[str, Any]:
  arr = np.ascontiguousarray(arr)
  tag = TAG[arr.dtype]
  (arr.astype(np.uint8) if tag == "bool" else arr).tofile(path)
  return {"file": str(path), "dtype": tag, "dims": list(arr.shape)}


def _verify_artifacts(models: Path, want_sha256: str) -> dict[str, Any]:
  manifest_path = models / "MANIFEST.json"
  got = _sha256(manifest_path)
  if got != want_sha256:
    msg = f"MANIFEST sha256 {got} != expected X2 manifest {want_sha256}"
    raise RuntimeError(msg)
  manifest = json.loads(manifest_path.read_text())
  for bucket in manifest["buckets"]:
    for graph in bucket["graphs"].values():
      file_sha = _sha256(models / graph["file"])
      if file_sha != graph["sha256"]:
        msg = f"{graph['file']} sha256 {file_sha} != manifest {graph['sha256']}"
        raise RuntimeError(msg)
  return manifest


def _rel(ref: float, got: float) -> float:
  return abs(float(got) - float(ref)) / max(abs(float(ref)), 1e-30)


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0915
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--models", type=Path, required=True, help="X2 artifact dir (MANIFEST.json + .onnx)")
  parser.add_argument("--manifest-sha256", required=True,
                      help="sha256 of the X2 gate run's MANIFEST.json (its result field manifest_sha256)")
  parser.add_argument("--potts-root", type=Path,
                      default=Path(os.environ.get("AMINX_POTTS_ROOT", "~/repos/PottsMPNN")).expanduser())
  parser.add_argument("--node-bin", default=os.environ.get("NODE_BIN"))
  parser.add_argument("--ort-web-dir", default=os.environ.get("ORT_WEB_DIR"))
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  from scripts.browser_validation.layer_a_common import emit  # noqa: PLC0415

  result: dict[str, Any] = {
    "jax_arm_ok": False, "artifacts_verified": False, "samples": [], "n_samples": 0, "n_pass": 0,
    "all_pass": False, "control_tokens_detected": False, "control_energy_detected": False,
    "node_ok": False, "num_threads": 0, "manifest_sha256": "", "error": "",
  }
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    if jax.config.jax_enable_x64:
      msg = "jax_enable_x64 is set; refusing"
      raise RuntimeError(msg)
    from aminx.families.potts_mpnn.driver import PottsMPNNDriver  # noqa: PLC0415
    from aminx.families.potts_mpnn.etab import model_to_etab, pad_etab_energy  # noqa: PLC0415

    _verify_artifacts(args.models, args.manifest_sha256)
    result["manifest_sha256"] = args.manifest_sha256
    result["artifacts_verified"] = True
    ckpt = args.potts_root / "vanilla_model_weights" / "pottsmpnn_20.pt"
    if _sha256(ckpt) != CHECKPOINT_SHA256:
      msg = f"{ckpt} does not match the pinned checkpoint"
      raise RuntimeError(msg)
    model = PottsMPNNDriver().load(SimpleNamespace(model_local_path=str(ckpt), num_samples=1, potts_mpnn=None,
                                                   run_spec=None))
    structures = _structures(args.potts_root / "inputs" / "example_pdbs")
    cells, fns = _cells(model, structures, seed=0)
    # Energy control model: the X0/X2 planted difference (1e-2 noise on every float leaf of the
    # Potts head). Run 1 (b0e1e1b8) used a cross-seed energy control, which did not fire: two
    # low-temperature draws can differ by a few residues and land within 1e-4 relative energy.
    import equinox as eqx  # noqa: PLC0415

    noise_key = jax.random.PRNGKey(1)
    perturbed_head = jax.tree_util.tree_map(
      lambda leaf: leaf + 1e-2 * jax.random.normal(noise_key, leaf.shape, leaf.dtype)
      if (eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating)) else leaf,
      model.potts_head,
    )
    p_cells, _ = _cells(eqx.tree_at(lambda m: m.potts_head, model, perturbed_head), structures, seed=0)

    # ---- JAX reference chain per (cell, seed), identical noise for both arms ---------------
    node_cells = []
    refs = {}
    for ci, cell in enumerate(cells):
      bucket = cell["bucket"]
      enc_in = cell["inputs"]["encode"]
      dec_base = cell["inputs"]["decode"]
      ref_base = cell["inputs"]["refine"]
      h_v, h_e, e_idx, forward, table = cell["reference"]["encode"]
      struct = dict(zip(STRUCT_FROM_ENCODE, enc_in, strict=True))
      struct.update({name: dec_base[i] for i, name in STRUCT_FROM_DECODE.items()})
      cell_dir = args.work_dir / f"cell{ci}"
      cell_dir.mkdir(exist_ok=True)
      struct_specs = {name: _write(arr, cell_dir / f"{name}.bin") for name, arr in struct.items()}
      for seed in SEEDS:
        rng = np.random.default_rng(seed)
        randn = rng.standard_normal(bucket).astype(np.float32)
        uniforms = rng.uniform(size=bucket).astype(np.float32)
        refine_uniforms = rng.uniform(size=(8, bucket)).astype(np.float32)
        dec_in = list(dec_base)
        dec_in[10], dec_in[11] = randn, uniforms
        seq, rank_flat, order, _hvs = (np.asarray(x) for x in jax.jit(fns[bucket]["decode"])(*dec_in))
        etab_seqs = np.repeat(np.asarray(model_to_etab(jnp.asarray(seq)))[None], N_MUTANTS, axis=0).astype(np.int32)
        energy = float(np.asarray(jax.jit(fns[bucket]["energy"])(table, e_idx, struct["pad_valid"], etab_seqs)[0])[0])
        ref_in = list(ref_base)
        ref_in[0] = seq.astype(np.int32)
        ref_in[1] = np.asarray(pad_etab_energy(jnp.asarray(forward)))
        ref_in[7] = order.astype(np.int32)  # upstream_refine_order, one sample: the AR order
        ref_in[8] = refine_uniforms
        refined = np.asarray(jax.jit(fns[bucket]["refine"])(*ref_in)[0])
        p_table = p_cells[ci]["reference"]["encode"][4]
        p_energy = float(np.asarray(jax.jit(fns[bucket]["energy"])(p_table, e_idx, struct["pad_valid"],
                                                                    etab_seqs)[0])[0])
        sid = f"c{ci}_s{seed}"
        refs[sid] = {"sequence": seq, "decoding_order": order, "rank_flat": rank_flat,
                     "refined_sequence": refined, "sample_energy": energy, "control_energy": p_energy,
                     "structure": cell["structure"], "bucket": bucket, "l_total": cell["l_total"], "seed": seed}
        noise_specs = {
          "randn": _write(randn, cell_dir / f"randn_{seed}.bin"),
          "uniforms": _write(uniforms, cell_dir / f"uniforms_{seed}.bin"),
          "refineUniforms": _write(refine_uniforms, cell_dir / f"refine_uniforms_{seed}.bin"),
        }
        node_cells.append({"id": sid, "bucket": bucket, "out_dir": str(cell_dir / "out"), "refine": True,
                           "temperature": float(dec_base[19]), "optimization_temperature": float(ref_base[20]),
                           "inputs": struct_specs, "noise": noise_specs})
    result["jax_arm_ok"] = True
    logger.info("JAX arm: %d samples in %.1fs", len(refs), time.perf_counter() - t0)

    # ---- shipping JS loop under Node ----------------------------------------------------------
    cells_path = args.work_dir / "loop_cells.json"
    out_path = args.work_dir / "loop_result.json"
    cells_path.write_text(json.dumps({"cells": node_cells}))
    runner = REPO / "browser" / "potts-sampler" / "run_potts_loop_node.mjs"
    proc = subprocess.run(  # noqa: S603 -- fixed argv we built
      [args.node_bin, str(runner), "--ort-dir", args.ort_web_dir, "--models", str(args.models),
       "--cells", str(cells_path), "--out", str(out_path)],
      capture_output=True, text=True, timeout=3600, check=False,
    )
    if proc.returncode != 0 or not out_path.exists():
      msg = f"node runner rc={proc.returncode}: {proc.stderr[-3000:]}"
      raise RuntimeError(msg)
    node = json.loads(out_path.read_text())
    result["node_ok"] = True
    result["num_threads"] = int(node.get("num_threads") or 0)
    got: dict[str, dict[str, Any]] = {}
    for rec in node["results"]:
      outs = {}
      if rec["ok"]:
        for name, spec in rec["outputs"].items():
          outs[name] = spec["value"] if "value" in spec else np.fromfile(spec["file"], dtype=np.int32)
      got[rec["id"]] = {"ok": rec["ok"], "error": rec.get("error"), "outputs": outs}

    # ---- grade --------------------------------------------------------------------------------
    for sid, ref in refs.items():
      g = got.get(sid, {"ok": False, "error": "missing", "outputs": {}})
      row = {k: ref[k] for k in ("structure", "bucket", "l_total", "seed")} | {"id": sid, "ran": g["ok"]}
      if not g["ok"]:
        row.update(passed=False, error=(g["error"] or "")[:2000])
      else:
        o = g["outputs"]
        row["sequence_exact"] = bool(np.array_equal(o["sequence"], ref["sequence"]))
        row["order_exact"] = bool(np.array_equal(o["decoding_order"], ref["decoding_order"]))
        row["refined_exact"] = bool(np.array_equal(o["refined_sequence"], ref["refined_sequence"]))
        row["energy_rel"] = _rel(ref["sample_energy"], o["sample_energy"])
        row["passed"] = (row["sequence_exact"] and row["order_exact"] and row["refined_exact"]
                         and row["energy_rel"] <= FLOAT_REL_BAR)
      result["samples"].append(row)

    # Controls: tokens -- each JS sample vs the JAX tokens of the NEXT seed in the same cell;
    # energy -- each JS sample energy vs JAX's energy of the SAME sequence under the perturbed
    # Potts head. Both must fail the comparator for every sample.
    tok_hits, en_hits, n_ctrl = 0, 0, 0
    for sid, ref in refs.items():
      ci, seed = sid.split("_s")
      nxt = f"{ci}_s{SEEDS[(SEEDS.index(int(seed)) + 1) % len(SEEDS)]}"
      g = got.get(sid)
      if not g or not g["ok"]:
        continue
      n_ctrl += 1
      tok_hits += int(not np.array_equal(g["outputs"]["sequence"], refs[nxt]["sequence"]))
      en_hits += int(_rel(ref["control_energy"], g["outputs"]["sample_energy"]) > FLOAT_REL_BAR)
    result["control_tokens_detected"] = n_ctrl > 0 and tok_hits == n_ctrl
    result["control_energy_detected"] = n_ctrl > 0 and en_hits == n_ctrl
    result["n_samples"] = len(result["samples"])
    result["n_pass"] = sum(bool(r.get("passed")) for r in result["samples"])
    result["all_pass"] = result["n_samples"] > 0 and result["n_pass"] == result["n_samples"]
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("loop gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("samples pass %s/%s", result["n_pass"], result["n_samples"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
