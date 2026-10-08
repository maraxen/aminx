"""X2 gate: the shipped PottsMPNN graphs, real checkpoint, real structures, ORT-CPU and ORT-Web.

task_id 261007_potts-onnx-export, backlog #5812 (and the layer-b/c half of #5813, #5814).
Sidecar: potts_export_gate.bth.toml (pre-registered before any run). Spec §8 / §8.1 for the
spike this builds on (runs 1340fe84, 241749b2).

Four graphs per bucket (128, 256), weights baked in, every runtime value an input:

  E   encode    coords, present, residue_idx, chain_index, pad_valid
                -> h_v, h_e, e_idx, forward etab, padded energy table   (once per structure)
  En  energy    padded table, e_idx, pad_valid, sequences[N]  -> energies[N]  (score / ddG)
  D   decode    PottsARDecode (sample, untied): noise and uniforms are inputs
  R   refine    one PottsRefine "potts" sweep (the default optimization_mode)

Each graph is checked in ISOLATION on its own JAX-computed inputs (no ORT output feeds another
graph; chaining is X5's browser-loop gate). The JAX arm runs entirely before jax2onnx is
imported. Comparator and bars are X0's, unchanged: integers exact, floats
max|d|/max|ref| <= 1e-4. Weights: the sha256-pinned vanilla pottsmpnn_20.pt, loaded through
PottsMPNNDriver. Structures: the sha256-pinned upstream example PDBs, featurized by the driver's
own path (parse_pdb_upstream -> _featurize_one -> prepare_sample(l_pad=bucket)); every
structure that fits a bucket is run in it, so the L256 cells include a heavily padded one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
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

logger = logging.getLogger("potts_export_gate")

BUCKETS = (128, 256)
CHECKPOINT_SHA256 = "77e797fd30fb4da11151d0d6f0d55d13ea25c00c45048dcc4aa634dc3620aa5c"
PDB_SHA256 = {
  "3dkm.pdb": "3d847d573e648b631983885a02f41bc14d8a8a11ac893b1c8c76ec5c46781c86",
  "3gg7.pdb": "2203e4a69287a5b968a78df5fb80b29f10fbdb37f73f64363cefbee8f1899712",
  "4jox.pdb": "c16543717793ced9e9475df6213a52054d05b70dd02069d41fac82e164f09ee8",
}
N_MUTANTS = 8
DECODE_TEMPERATURE = 0.1
GRAPHS = ("encode", "energy", "decode", "refine")
INPUT_NAMES = {
  "encode": ["coords", "present", "residue_idx", "chain_index", "pad_valid"],
  "energy": ["table", "e_idx", "pad_valid", "sequences"],
  "decode": ["h_v", "h_e", "e_idx", "present", "pad_valid", "s_true", "chain_mask", "chain_m_pos",
             "tie_groups", "tied_beta", "randn", "uniforms", "omit", "bias", "bias_by_res",
             "pssm_coef", "pssm_bias", "pssm_log_odds_mask", "omit_aa_mask", "temperature"],
  "refine": ["sequence", "etab", "e_idx", "pad_valid", "present", "chain_mask", "chain_m_pos",
             "order", "uniforms", "omit", "bias", "bias_by_res", "pssm_coef", "pssm_bias",
             "pssm_log_odds_mask", "omit_aa_mask", "tie_groups", "tied_beta", "h_v", "h_e", "temperature"],
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _structures(pdb_dir: Path) -> list[dict[str, Any]]:
  """Parse + featurize each pinned PDB with the driver's own helpers (unpadded)."""
  from aminx.families.potts_mpnn.driver import _chain_sequences, _featurize_one  # noqa: PLC0415
  from aminx.families.potts_mpnn.featurize import parse_pdb_upstream  # noqa: PLC0415
  from aminx.run.options import PottsMPNNOptions  # noqa: PLC0415

  options = PottsMPNNOptions()
  out = []
  for name, want in PDB_SHA256.items():
    path = pdb_dir / name
    got = _sha256(path)
    if got != want:
      msg = f"{path} sha256 {got} != pinned {want}"
      raise RuntimeError(msg)
    parsed = parse_pdb_upstream(path, skip_gaps=options.skip_gaps)[0]
    features = _featurize_one(parsed, options, str(parsed["name"]))
    chains = tuple((c.letter, c.sequence) for c in _chain_sequences(parsed, features))
    out.append({"name": name, "features": features, "chains": chains, "l_total": int(features.L_total)})
  return out


def _cells(
  model: Any,  # noqa: ANN401
  structures: list[dict[str, Any]],
  seed: int,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
  """Per (bucket, structure): each graph's concrete inputs and its JAX reference outputs."""
  import jax  # noqa: PLC0415
  import jax.numpy as jnp  # noqa: PLC0415

  from aminx.families.potts_mpnn.decode import PottsARDecode, floor_temperature_array  # noqa: PLC0415
  from aminx.families.potts_mpnn.etab import model_to_etab, pad_etab_energy, potts_energy  # noqa: PLC0415
  from aminx.families.potts_mpnn.refine import PottsRefine  # noqa: PLC0415
  from aminx.families.potts_mpnn.sample_host import (  # noqa: PLC0415
    _dummy_binding,
    _encode,
    prepare_sample,
    upstream_refine_order,
  )
  from aminx.run.options import PottsMPNNOptions  # noqa: PLC0415

  options = PottsMPNNOptions()
  spec = SimpleNamespace(fixed_positions=None, omit_aa=(), bias=None, num_samples=1)
  mods = {"layers": model.mpnn.decoder.layers, "w_s_embed": model.mpnn.w_s_embed, "w_out": model.mpnn.w_out}
  decode = PottsARDecode(**mods)
  refiner = PottsRefine(**mods)
  opt_temperature = np.float32(options.optimization_temperature)  # raw option; floored in-graph
  rng = np.random.default_rng(seed)

  fns = {}
  cells = []
  for bucket in BUCKETS:
    tables = _dummy_binding(bucket, 22, jnp.float32)

    def encode(coords, present, residue_idx, chain_index, pad_valid):  # noqa: ANN001, ANN202
      return tuple(_encode(model, coords, present, residue_idx, chain_index, pad_valid))

    def energy(table, e_idx, pad_valid, sequences):  # noqa: ANN001, ANN202
      return (potts_energy(table, e_idx, pad_valid, sequences),)

    def dec(*a):  # noqa: ANN002, ANN202
      *arrays, temperature = a
      out = decode(*arrays, temperature=temperature, pssm_multi=float(options.pssm_multi),
                   pssm_bias_flag=bool(options.pssm_bias_flag), pssm_log_odds_flag=bool(options.pssm_log_odds_flag))
      return (out.sequence, out.rank_flat, out.decoding_order, out.h_v_stack)

    def ref(*a, _tables=tables):  # noqa: ANN002, ANN202
      *arrays, temperature = a
      out = refiner(
        "potts", *arrays, _tables,
        temperature=floor_temperature_array(temperature, arrays[1].dtype), pssm_multi=float(options.pssm_multi),
        pssm_bias_flag=bool(options.pssm_bias_flag), pssm_log_odds_flag=bool(options.pssm_log_odds_flag),
        binding="none", tied=False, tied_epistasis=bool(options.tied_epistasis), max_iters=1,
      )
      return (out.sequence,)

    fns[bucket] = {"encode": encode, "energy": energy, "decode": dec, "refine": ref}
    for st in structures:
      if st["l_total"] > bucket:
        continue
      r = prepare_sample(st["features"], st["chains"], options, spec, l_pad=bucket)
      enc_in = [r.coords, r.present, r.residue_idx, r.chain_index, r.pad_valid]
      enc_out = [np.asarray(x) for x in jax.jit(encode)(*enc_in)]
      h_v, h_e, e_idx, forward, table = enc_out
      native = np.asarray(model_to_etab(jnp.asarray(r.s_true)))
      seqs = np.repeat(native[None], N_MUTANTS, axis=0).astype(np.int32)
      real = int(r.l_total)
      for row in range(1, N_MUTANTS):
        pos = rng.integers(0, real)
        seqs[row, pos] = (seqs[row, pos] + rng.integers(1, 20)) % 20
      en_in = [table, e_idx, r.pad_valid, seqs]
      randn = rng.standard_normal(bucket).astype(np.float32)
      uniforms = rng.uniform(size=bucket).astype(np.float32)
      dec_in = [h_v, h_e, e_idx, r.present, r.pad_valid, r.s_true, r.chain_mask, r.chain_m_pos,
                r.tie_groups, r.tied_beta, randn, uniforms, r.omit, r.bias, r.bias_by_res,
                r.pssm_coef, r.pssm_bias, r.pssm_log_odds_mask, r.omit_aa_mask, np.float32(DECODE_TEMPERATURE)]
      dec_out = [np.asarray(x) for x in jax.jit(dec)(*dec_in)]
      order = upstream_refine_order(dec_out[2], r.chain_mask, rng.standard_normal(bucket).astype(np.float32),
                                    num_samples=1, chain_suffix="", stored_orders_present=True)
      ref_in = [dec_out[0].astype(np.int32), np.asarray(pad_etab_energy(jnp.asarray(forward))), e_idx,
                r.pad_valid, r.present, r.chain_mask, r.chain_m_pos, np.asarray(order, np.int32),
                rng.uniform(size=(8, bucket)).astype(np.float32), r.omit, r.bias, r.bias_by_res,
                r.pssm_coef, r.pssm_bias, r.pssm_log_odds_mask, r.omit_aa_mask, r.tie_groups,
                r.tied_beta, h_v, h_e, opt_temperature]
      cell = {"bucket": bucket, "structure": st["name"], "l_total": real, "inputs": {}, "reference": {}}
      for graph, ins in (("encode", enc_in), ("energy", en_in), ("decode", dec_in), ("refine", ref_in)):
        ins = [np.ascontiguousarray(x) for x in ins]
        cell["inputs"][graph] = ins
        cell["reference"][graph] = (
          enc_out if graph == "encode" else dec_out if graph == "decode"
          else [np.asarray(x) for x in jax.jit(fns[bucket][graph])(*ins)]
        )
      cells.append(cell)
  return cells, fns


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--potts-root", type=Path,
                      default=Path(os.environ.get("AMINX_POTTS_ROOT", "~/repos/PottsMPNN")).expanduser())
  parser.add_argument("--node-bin", default=os.environ.get("NODE_BIN"))
  parser.add_argument("--ort-web-dir", default=os.environ.get("ORT_WEB_DIR"))
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  from scripts.browser_validation.layer_a_common import emit  # noqa: PLC0415

  result: dict[str, Any] = {
    "versions": {}, "jax_arm_ok": False, "jax_enable_x64": False, "checkpoint_sha256": "",
    "cells": [], "graphs": {}, "n_cells": 0, "n_graph_checks": 0, "n_pass_cpu": 0, "n_pass_web": 0,
    "all_exported": False, "all_pass": False, "control_float_detected": False,
    "control_int_detected": False, "ort_web_available": False, "manifest_sha256": "", "error": "",
  }
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    if "jax2onnx" in sys.modules:
      msg = "jax2onnx imported before the JAX arm"
      raise RuntimeError(msg)
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415

    result["jax_enable_x64"] = bool(jax.config.jax_enable_x64)
    if result["jax_enable_x64"]:
      msg = "jax_enable_x64 is set; refusing"
      raise RuntimeError(msg)

    from aminx.families.potts_mpnn.driver import PottsMPNNDriver  # noqa: PLC0415
    from aminx.families.potts_mpnn.etab import ETAB_ALPHABET  # noqa: PLC0415
    from aminx.families.potts_mpnn.featurize import MODEL_ALPHABET  # noqa: PLC0415

    ckpt = args.potts_root / "vanilla_model_weights" / "pottsmpnn_20.pt"
    result["checkpoint_sha256"] = _sha256(ckpt)
    if result["checkpoint_sha256"] != CHECKPOINT_SHA256:
      msg = f"{ckpt} sha256 {result['checkpoint_sha256']} != pinned {CHECKPOINT_SHA256}"
      raise RuntimeError(msg)
    model = PottsMPNNDriver().load(SimpleNamespace(model_local_path=str(ckpt), num_samples=1, potts_mpnn=None,
                                                   run_spec=None))
    structures = _structures(args.potts_root / "inputs" / "example_pdbs")

    # ---- JAX arm: every reference and both controls before any conversion ------------
    cells, fns = _cells(model, structures, args.seed)
    noise_key = jax.random.PRNGKey(args.seed + 1)
    perturbed_head = jax.tree_util.tree_map(
      lambda leaf: leaf + 1e-2 * jax.random.normal(noise_key, leaf.shape, leaf.dtype)
      if (eqx.is_array(leaf) and jax.numpy.issubdtype(leaf.dtype, jax.numpy.floating)) else leaf,
      model.potts_head,
    )
    p_cells, _ = _cells(eqx.tree_at(lambda m: m.potts_head, model, perturbed_head), structures, args.seed)
    result["control_float_detected"] = not compare(cells[0]["reference"]["encode"],
                                                   p_cells[0]["reference"]["encode"])["ok"]
    alt = list(cells[0]["inputs"]["decode"])
    alt[11] = np.ascontiguousarray(alt[11][::-1])
    alt_out = [np.asarray(x) for x in jax.jit(fns[cells[0]["bucket"]]["decode"])(*alt)]
    result["control_int_detected"] = not compare(cells[0]["reference"]["decode"][:1], alt_out[:1])["ok"]
    result["jax_arm_ok"] = True
    logger.info("JAX arm: %d cells in %.1fs", len(cells), time.perf_counter() - t0)

    # ---- conversion (one artifact per graph per bucket) --------------------------------
    import xtrax  # noqa: PLC0415
    from xtrax.export import ONNX, convert_to_onnx, run_onnx  # noqa: PLC0415

    from scripts.browser_validation.p07_knobs_gate import embed_external_data  # noqa: PLC0415

    exported_ok = True
    manifest_buckets = []
    for bucket in BUCKETS:
      first = next(c for c in cells if c["bucket"] == bucket)
      entries = {}
      for graph in GRAPHS:
        ins = first["inputs"][graph]
        path = args.work_dir / f"pottsmpnn_{graph}_L{bucket}.onnx"
        key = f"{graph}_L{bucket}"
        try:
          convert_to_onnx(fns[bucket][graph], [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in ins], ONNX,
                          out_path=path)
          embed_external_data(path)
          facts = _graph_facts(path)
          entries[graph] = {
            "file": path.name, "bytes": facts["bytes"], "sha256": _sha256(path),
            "input_names": INPUT_NAMES[graph], "input_dtypes": [str(a.dtype) for a in ins],
            "input_shapes": [list(a.shape) for a in ins],
          }
          result["graphs"][key] = {"converted": True, **facts}
        except Exception as exc:  # noqa: BLE001
          exported_ok = False
          result["graphs"][key] = {"converted": False, "error": f"{type(exc).__name__}: {exc}"[:3000]}
      manifest_buckets.append({"bucket": bucket, "graphs": entries})
    result["all_exported"] = exported_ok
    manifest = {
      "family": "pottsmpnn", "checkpoint": "pottsmpnn_20.pt", "checkpoint_sha256": CHECKPOINT_SHA256,
      "alphabet": MODEL_ALPHABET, "n_tokens": len(MODEL_ALPHABET), "etab_alphabet": ETAB_ALPHABET,
      "decode_temperature_default": DECODE_TEMPERATURE, "buckets": manifest_buckets,
    }
    manifest_path = args.work_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result["manifest_sha256"] = _sha256(manifest_path)

    # ---- ORT-CPU + ORT-Web per cell -----------------------------------------------------
    web_cases = []
    for cell in cells:
      row = {"bucket": cell["bucket"], "structure": cell["structure"], "l_total": cell["l_total"], "graphs": {}}
      for graph in GRAPHS:
        path = args.work_dir / f"pottsmpnn_{graph}_L{cell['bucket']}.onnx"
        entry = {"cpu": "not_converted", "web": "not_converted", "cpu_max_rel": None, "web_max_rel": None}
        row["graphs"][graph] = entry
        if not result["graphs"][f"{graph}_L{cell['bucket']}"]["converted"]:
          continue
        ins = cell["inputs"][graph]
        try:
          cmp = compare(cell["reference"][graph], run_onnx(path, *ins))
          entry.update(cpu="pass" if cmp["ok"] else "mismatch", cpu_max_rel=cmp["max_rel"], cpu_why=cmp["why"])
        except Exception as exc:  # noqa: BLE001
          entry.update(cpu="not_runnable", error=f"{type(exc).__name__}: {exc}"[:2000])
        case_id = f"{graph}_L{cell['bucket']}_{cell['structure'].removesuffix('.pdb')}"
        io_dir = args.work_dir / f"io_{case_id}"
        io_dir.mkdir(exist_ok=True)
        specs_in = []
        for i, arr in enumerate(ins):
          tag = DTYPE_TAG[arr.dtype]
          f = io_dir / f"in_{i}.bin"
          (arr.astype(np.uint8) if tag == "bool" else arr).tofile(f)
          specs_in.append({"file": str(f), "dtype": tag, "dims": list(arr.shape)})
        web_cases.append({"id": case_id, "model": str(path), "out_dir": str(io_dir), "inputs": specs_in})
        entry["case_id"] = case_id
      result["cells"].append(row)

    web = _run_web(web_cases, args.work_dir, args.node_bin, args.ort_web_dir)
    result["ort_web_available"] = web["available"]
    result["ort_web_note"] = web["why"]
    for cell, row in zip(cells, result["cells"], strict=True):
      for graph, entry in row["graphs"].items():
        rec = web["results"].get(entry.get("case_id", ""))
        if entry["cpu"] == "not_converted":
          continue
        if not web["available"] or rec is None:
          entry["web"] = "unavailable"
        elif not rec["ok"]:
          entry.update(web="not_runnable", web_error=(rec.get("error") or "")[:2000])
        else:
          cmp = compare(cell["reference"][graph], _read_web_outputs(rec))
          entry.update(web="pass" if cmp["ok"] else "mismatch", web_max_rel=cmp["max_rel"], web_why=cmp["why"],
                       web_num_threads=rec.get("num_threads"))

    import jax2onnx  # noqa: PLC0415
    import onnxruntime  # noqa: PLC0415

    result["versions"] = {"jax": jax.__version__, "xtrax": xtrax.__version__,
                          "jax2onnx": getattr(jax2onnx, "__version__", None),
                          "onnxruntime": onnxruntime.__version__, "onnxruntime_web": web.get("ort_web_version")}
    checks = [e for row in result["cells"] for e in row["graphs"].values()]
    result["n_cells"] = len(result["cells"])
    result["n_graph_checks"] = len(checks)
    result["n_pass_cpu"] = sum(e["cpu"] == "pass" for e in checks)
    result["n_pass_web"] = sum(e["web"] == "pass" for e in checks)
    result["all_pass"] = bool(checks) and exported_ok and (
      result["n_pass_cpu"] == result["n_graph_checks"] == result["n_pass_web"]
    )
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  for row in result["cells"]:
    logger.info("%s L%d: %s", row["structure"], row["bucket"],
                {g: (e["cpu"], e["web"]) for g, e in row["graphs"].items()})
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
