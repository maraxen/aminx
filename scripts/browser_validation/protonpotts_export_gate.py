"""X6a gate: the shipped ProtonPottsMPNN scoring graphs, real converted checkpoint, real structures, ORT-CPU and ORT-Web.

task_id 261009_protonpotts-onnx, backlog #5816. Sidecar: protonpotts_export_gate.bth.toml (pre-registered before this
script existed). Builds on potts_export_gate.py (the V=21 PottsMPNN gate) and the scope spec
261007_potts-onnx-export-scope.md section 4.

Two graphs per bucket (256, 1024), weights baked in, every runtime value an input:

  table   coords, present, residue_idx, chain_index, pad_valid -> merged pair table [L,K,30,30], e_idx
          (MPNNEncode -> PottsHead -> ONE reciprocal merge; aminx token order)
  energy  table, e_idx, pad_valid, sequences[8, L] -> energies[8]            (score:energy / score:ddg)

Each graph is checked in ISOLATION on its own JAX-computed inputs. The JAX arm runs entirely before jax2onnx is imported.
Comparator and bars are the PottsMPNN export gate's: integers exact, floats max|d|/max|ref| <= 1e-4. Structures are
featurized UNLABELLED by featurize_pdb, which is what the driver does without a labels file, then padded with the
potts_mpnn.featurize.pad recipe (coords 0, present 0, chain 0, residue_idx -100, pad_valid false).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

logger = logging.getLogger("protonpotts_export_gate")

SMALL_BUCKET = 256
LARGE_BUCKET = 1024
BUCKETS = (SMALL_BUCKET, LARGE_BUCKET)
CHECKPOINT_SHA256 = "a39872250c0b8eb4c0b4cb6472a6edad0d6f6134a50c28b611386e83c592fe7b"
STRUCTURES = {
  "1EL1": ("fireprot_pdbs/1EL1.pdb", "9777be2e8bb883d5426f6477f0d8c2a53ce69660e985c632032f6686565764b4"),
  "1BVC": ("fireprot_pdbs/1BVC.pdb", "0fbc6cefc3acbc93855ca3d1efeb9775f7a6501649a963990ce45b887b433ce3"),
  "1OLR": ("fireprot_pdbs/1OLR.pdb", "4b5d40e2f27acd8f90e9d58186f8ebb22b893b1364529ac672fe09aaf667ec01"),
  "6m0j": ("covid_pdbs/6m0j.pdb", "bb80f2283ff059a3744debd3d1727f8d63c918b923390044d12ef45fabd86ce8"),
}
N_SEQS = 8
FLOAT_REL_BAR = 1.0e-4
GRAPHS = ("table", "energy")
INPUT_NAMES = {
  "table": ["coords", "present", "residue_idx", "chain_index", "pad_valid"],
  "energy": ["table", "e_idx", "pad_valid", "sequences"],
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _bucket_for(length: int) -> int:
  return SMALL_BUCKET if length <= SMALL_BUCKET else LARGE_BUCKET


def _pad_inputs(raw: dict[str, np.ndarray], bucket: int) -> dict[str, np.ndarray]:
  """The browser's padding, the potts_mpnn.featurize.pad recipe."""
  length = raw["coords"].shape[0]
  n_pad = bucket - length

  def tail(row: np.ndarray, fill: float) -> np.ndarray:
    return np.concatenate([row, np.full((n_pad, *row.shape[1:]), fill, dtype=row.dtype)]) if n_pad else row

  return {
    "coords": tail(raw["coords"], 0.0),
    "present": tail(raw["present"], 0.0),
    "residue_idx": tail(raw["residue_idx"], -100),
    "chain_index": tail(raw["chain_index"], 0),
    "pad_valid": np.arange(bucket) < length,
  }


def _structures(root: Path) -> list[dict[str, Any]]:
  from aminx.families.protonpotts_mpnn.features import featurize_pdb  # noqa: PLC0415

  out = []
  for name, (rel, want) in STRUCTURES.items():
    path = root / "energy_benchmark_datasets" / rel
    got = _sha256(path)
    if got != want:
      msg = f"{path} sha256 {got} != pinned {want}"
      raise RuntimeError(msg)
    feats = featurize_pdb(path, None)
    length = int(feats["S"].shape[1])
    raw = {
      "coords": np.asarray(feats["X"][0, :, :4, :], dtype=np.float32),
      "present": np.ones(length, dtype=np.float32),
      "residue_idx": np.asarray(feats["R_idx"][0], dtype=np.int32),
      "chain_index": np.asarray(feats["chain_labels"][0], dtype=np.int32),
      "pad_valid": np.ones(length, dtype=np.bool_),
    }
    out.append({"name": name, "length": length, "raw": raw, "native": np.asarray(feats["S"][0], dtype=np.int32)})
  return out


def _load_model(state_npz: Path) -> Any:  # noqa: ANN401
  path = Path(__file__).resolve().parents[1] / "parity" / "convert_protonpotts_checkpoint.py"
  spec = importlib.util.spec_from_file_location("convert_protonpotts_checkpoint", path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  meta = json.loads(state_npz.with_suffix(".json").read_text(encoding="utf-8"))
  if meta["checkpoint_sha256"] != CHECKPOINT_SHA256:
    msg = f"{state_npz}: checkpoint sha256 {meta['checkpoint_sha256']} != pinned {CHECKPOINT_SHA256}"
    raise RuntimeError(msg)
  with np.load(state_npz) as z:
    state = {key: z[key] for key in z.files}
  return module.convert_state(state)


def _table_fn(model: Any):  # noqa: ANN202, ANN401
  from aminx.families.protonpotts_mpnn.energy import protonpotts_table  # noqa: PLC0415

  def table(coords, present, residue_idx, chain_index, pad_valid):  # noqa: ANN001, ANN202
    merged, e_idx = protonpotts_table(model, coords, present, residue_idx, chain_index, pad_valid)
    return (merged, e_idx)

  return table


def _energy_fn():  # noqa: ANN202
  from aminx.families.potts_mpnn.etab import potts_energy  # noqa: PLC0415

  def energy(table, e_idx, pad_valid, sequences):  # noqa: ANN001, ANN202
    return (potts_energy(table, e_idx, pad_valid, sequences),)

  return energy


def _sequences(native: np.ndarray, bucket: int, rng: np.random.Generator, n_tokens: int) -> np.ndarray:
  """Native plus seven single mutants over ALL tokens (protonation tokens included), padded with token 0."""
  rows = np.repeat(native[None], N_SEQS, axis=0).astype(np.int32)
  for row in range(1, N_SEQS):
    pos = int(rng.integers(0, native.shape[0]))
    rows[row, pos] = (rows[row, pos] + int(rng.integers(1, n_tokens))) % n_tokens
  pad = np.zeros((N_SEQS, bucket - native.shape[0]), dtype=np.int32)
  return np.concatenate([rows, pad], axis=1)


def _cells(model: Any, structures: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:  # noqa: ANN401
  import jax  # noqa: PLC0415
  import jax.numpy as jnp  # noqa: PLC0415

  from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6  # noqa: PLC0415

  table_fn, energy_fn = _table_fn(model), _energy_fn()
  rng = np.random.default_rng(seed)
  cells = []
  for st in structures:
    bucket = _bucket_for(st["length"])
    padded = _pad_inputs(st["raw"], bucket)
    t_in = [np.ascontiguousarray(padded[k]) for k in INPUT_NAMES["table"]]
    t_out = [np.asarray(x) for x in jax.jit(table_fn)(*t_in)]
    seqs = _sequences(st["native"], bucket, rng, PROTONPOTTS_V6.size)
    e_in = [np.ascontiguousarray(a) for a in (t_out[0], t_out[1], padded["pad_valid"], seqs)]
    e_out = [np.asarray(x) for x in jax.jit(energy_fn)(*e_in)]
    # padding invariance: the same structure, unpadded, through the same functions
    raw = st["raw"]
    u_table = [np.asarray(x) for x in jax.jit(table_fn)(*[jnp.asarray(raw[k]) for k in INPUT_NAMES["table"]])]
    u_seqs = seqs[:, : st["length"]]
    u_energy = np.asarray(jax.jit(energy_fn)(u_table[0], u_table[1], jnp.asarray(raw["pad_valid"]), u_seqs)[0])
    pad_cmp = compare([e_out[0]], [u_energy])
    cells.append({
      "bucket": bucket, "structure": st["name"], "l_total": st["length"],
      "inputs": {"table": t_in, "energy": e_in}, "reference": {"table": t_out, "energy": e_out},
      "padding_ok": bool(pad_cmp["ok"]), "padding_max_rel": pad_cmp["max_rel"],
    })
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
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  from scripts.browser_validation.layer_a_common import emit  # noqa: PLC0415

  result: dict[str, Any] = {
    "versions": {}, "jax_arm_ok": False, "jax_enable_x64": False, "checkpoint_sha256": CHECKPOINT_SHA256,
    "cells": [], "graphs": {}, "n_cells": 0, "n_graph_checks": 0, "n_pass_cpu": 0, "n_pass_web": 0,
    "all_exported": False, "all_pass": False, "small_all_pass": False,
    "padding_invariant": False, "padding_invariant_small": False,
    "control_float_detected": False, "control_int_detected": False, "control_energy_detected": False,
    "ort_web_available": False, "manifest_sha256": "", "error": "",
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

    from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6  # noqa: PLC0415

    model = _load_model(args.state_npz)
    structures = _structures(args.potts_root)

    # ---- JAX arm: every reference, padding check and all three controls before any conversion ----
    cells = _cells(model, structures, args.seed)
    noise_key = jax.random.PRNGKey(args.seed + 1)
    perturbed_head = jax.tree_util.tree_map(
      lambda leaf: leaf + 1e-2 * jax.random.normal(noise_key, leaf.shape, leaf.dtype)
      if (eqx.is_array(leaf) and jax.numpy.issubdtype(leaf.dtype, jax.numpy.floating)) else leaf,
      model.potts_head,
    )
    first = next(c for c in cells if c["bucket"] == SMALL_BUCKET)
    p_table = [np.asarray(x) for x in jax.jit(_table_fn(eqx.tree_at(lambda m: m.potts_head, model, perturbed_head)))(
      *first["inputs"]["table"])]
    result["control_float_detected"] = not compare(first["reference"]["table"][:1], p_table[:1])["ok"]
    moved = list(first["inputs"]["table"])
    moved[0] = (moved[0] + np.random.default_rng(args.seed + 2).normal(0.0, 2.0, moved[0].shape)).astype(np.float32)
    m_table = [np.asarray(x) for x in jax.jit(_table_fn(model))(*moved)]
    result["control_int_detected"] = not compare(first["reference"]["table"][1:], m_table[1:])["ok"]
    swapped = list(first["inputs"]["energy"])
    swapped[3] = np.ascontiguousarray(swapped[3][::-1])
    s_energy = [np.asarray(x) for x in jax.jit(_energy_fn())(*swapped)]
    result["control_energy_detected"] = not compare(first["reference"]["energy"], s_energy)["ok"]
    result["padding_invariant"] = all(c["padding_ok"] for c in cells)
    result["padding_invariant_small"] = all(c["padding_ok"] for c in cells if c["bucket"] == SMALL_BUCKET)
    result["jax_arm_ok"] = True
    logger.info("JAX arm: %d cells in %.1fs", len(cells), time.perf_counter() - t0)

    # ---- conversion (one artifact per graph per bucket) ----------------------------------------
    import xtrax  # noqa: PLC0415
    from xtrax.export import ONNX, convert_to_onnx, run_onnx  # noqa: PLC0415

    from scripts.browser_validation.p07_knobs_gate import embed_external_data  # noqa: PLC0415

    fns = {"table": _table_fn(model), "energy": _energy_fn()}
    exported_ok = True
    manifest_buckets = []
    for bucket in BUCKETS:
      cell0 = next(c for c in cells if c["bucket"] == bucket)
      entries = {}
      for graph in GRAPHS:
        ins = cell0["inputs"][graph]
        path = args.work_dir / f"protonpottsmpnn_{graph}_L{bucket}.onnx"
        try:
          convert_to_onnx(fns[graph], [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in ins], ONNX, out_path=path)
          embed_external_data(path)
          facts = _graph_facts(path)
          entries[graph] = {
            "file": path.name, "bytes": facts["bytes"], "sha256": _sha256(path),
            "input_names": INPUT_NAMES[graph], "input_dtypes": [str(a.dtype) for a in ins],
            "input_shapes": [list(a.shape) for a in ins],
          }
          result["graphs"][f"{graph}_L{bucket}"] = {"converted": True, **facts}
        except Exception as exc:  # noqa: BLE001
          exported_ok = False
          result["graphs"][f"{graph}_L{bucket}"] = {"converted": False, "error": f"{type(exc).__name__}: {exc}"[:3000]}
      manifest_buckets.append({"bucket": bucket, "graphs": entries})
    result["all_exported"] = exported_ok
    manifest = {
      "family": "protonpottsmpnn", "checkpoint": "potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt",
      "checkpoint_sha256": CHECKPOINT_SHA256, "alphabet": PROTONPOTTS_V6.name,
      "vocabulary": list(PROTONPOTTS_V6.symbols), "n_tokens": PROTONPOTTS_V6.size,
      "n_sequences_per_energy_call": N_SEQS, "pad": {"coords": 0.0, "present": 0.0, "chain_index": 0,
                                                       "residue_idx": -100, "sequences": 0},
      "buckets": manifest_buckets,
    }
    manifest_path = args.work_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result["manifest_sha256"] = _sha256(manifest_path)

    # ---- ORT-CPU + ORT-Web per cell ---------------------------------------------------------------
    web_cases = []
    for cell in cells:
      row = {"bucket": cell["bucket"], "structure": cell["structure"], "l_total": cell["l_total"],
             "padding_ok": cell["padding_ok"], "padding_max_rel": cell["padding_max_rel"], "graphs": {}}
      for graph in GRAPHS:
        path = args.work_dir / f"protonpottsmpnn_{graph}_L{cell['bucket']}.onnx"
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
        case_id = f"{graph}_L{cell['bucket']}_{cell['structure']}"
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
    small = [e for row in result["cells"] if row["bucket"] == SMALL_BUCKET for e in row["graphs"].values()]
    result["n_cells"] = len(result["cells"])
    result["n_graph_checks"] = len(checks)
    result["n_pass_cpu"] = sum(e["cpu"] == "pass" for e in checks)
    result["n_pass_web"] = sum(e["web"] == "pass" for e in checks)
    result["all_pass"] = bool(checks) and exported_ok and all(e["cpu"] == "pass" and e["web"] == "pass" for e in checks)
    result["small_all_pass"] = bool(small) and all(e["cpu"] == "pass" and e["web"] == "pass" for e in small)
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
