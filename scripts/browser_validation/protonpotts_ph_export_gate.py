"""X7a gate: the per-block pH design graphs on real structures, ORT-CPU and ORT-Web, bitwise-repeatable across runs.

task_id 261009_protonpotts-onnx, backlog #5816 (B1). Sidecar: protonpotts_ph_export_gate.bth.toml (pre-registered before this
script existed). ADR decision 11; spec §54.

Graphs per bucket (256, 1024): ``visit`` and ``zblock`` for each pin count P in (1, 2, 3), ``pool`` and ``field_at`` once. Weights are
not baked in (the merged pair table is an INPUT), so the files are small and the same bytes serve every structure of a bucket.
Each graph is checked in isolation on its own JAX-computed inputs; the JAX arm runs entirely before jax2onnx is imported.
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
  _table_fn,
)

logger = logging.getLogger("protonpotts_ph_export_gate")

BLOCK = 3
DEPTH = 2
CHUNK = 64
PIN_VARIANTS = {1: ("HIS-P",), 2: ("ASP-P", "GLU-P"), 3: ("HIS-P", "ASP-P", "GLU-P")}
DEP_MAP = {"HIS-P": ("HIS-S",), "ASP-P": ("ASP-D",), "GLU-P": ("GLU-D",)}
LAMBDA = 0.3
TEMPERATURE = 0.05
UNIFORM = 0.37
N_REPEATS = 5
SENTINEL = 10**6
VISIT_NAMES = ["table", "e_idx", "seq", "block", "block_valid", "pin_positions", "pin_prot", "pin_dep", "pin_dep_valid",
               "valid_tokens", "rep_mask", "residue_number", "wh", "wsel", "uniform", "temperature"]
ZBLOCK_NAMES = ["table", "e_idx", "seq", "block", "block_valid", "pin_positions", "pin_prot", "pin_dep", "pin_dep_valid",
                "valid_tokens"]
POOL_NAMES = ["var_h", "has_h", "var_s", "has_s", "w_h", "w_s"]
FIELD_NAMES = ["table", "e_idx", "seq", "positions"]


def _racy_potentials():  # noqa: ANN202
  """The PRE-FIX ``block_stability_potentials`` (repeated-index scatter-adds), kept to prove the determinism check can see it."""
  import jax.numpy as jnp  # noqa: PLC0415

  from aminx.families.protonpotts_mpnn import ph_potentials as pp  # noqa: PLC0415

  def racy(table, e_idx, seq, block, block_valid):  # noqa: ANN001, ANN202
    length, k_slots = e_idx.shape
    n_block = block.shape[0]
    vocab = table.shape[-1]
    dtype = table.dtype
    block = block.astype(jnp.int32)
    member_slot = jnp.where(block_valid, block, length)
    pos_to_b = jnp.full((length,), -1, dtype=jnp.int32).at[member_slot].set(jnp.arange(n_block, dtype=jnp.int32), mode="drop")
    e_rows = e_idx[block]
    t_rows = table[block]
    src_ok = block_valid[:, None]
    is_self = e_rows == block[:, None]
    tgt_b = pos_to_b[e_rows]
    diagonal = jnp.diagonal(t_rows, axis1=-2, axis2=-1)
    nbr_tok = seq[e_rows]
    index = jnp.broadcast_to(nbr_tok[:, :, None, None], (*t_rows.shape[:-1], 1))
    gathered = jnp.take_along_axis(t_rows, index, axis=-1)[..., 0]
    out_vals = jnp.where(is_self[..., None], diagonal, gathered)
    unary = jnp.where((src_ok & (is_self | (tgt_b < 0)))[..., None], out_vals, 0).sum(axis=1).astype(dtype)
    pair_mask = src_ok & ~is_self & (tgt_b >= 0)
    b_idx = jnp.broadcast_to(jnp.arange(n_block, dtype=jnp.int32)[:, None], (n_block, k_slots))
    t_idx = jnp.maximum(tgt_b, 0)
    lo = jnp.where(pair_mask, jnp.minimum(b_idx, t_idx), 0).reshape(-1)
    hi = jnp.where(pair_mask, jnp.maximum(b_idx, t_idx), 0).reshape(-1)
    oriented = jnp.where((b_idx < t_idx)[..., None, None], t_rows, jnp.swapaxes(t_rows, -1, -2))
    oriented = jnp.where(pair_mask[..., None, None], oriented, 0).reshape(-1, vocab, vocab)
    pair = jnp.zeros((n_block, n_block, vocab, vocab), dtype=dtype).at[lo, hi].add(oriented.astype(dtype))
    src, slot, tgt, valid = pp.incoming_edges(e_idx)
    tgt_in = pos_to_b[tgt]
    in_ok = valid & (pos_to_b[src] < 0) & (tgt_in >= 0)
    in_vals = jnp.where(in_ok[:, None], pp._incoming_terms(table, seq, src, slot), 0)  # noqa: SLF001
    unary = unary.at[jnp.maximum(tgt_in, 0)].add(in_vals.astype(dtype))
    return unary, pair

  return racy


def _tag(arr: np.ndarray) -> str:
  return DTYPE_TAG[arr.dtype]


def _structures(root: Path) -> list[dict[str, Any]]:
  from aminx.families.protonpotts_mpnn.features import featurize_pdb, kept_residues  # noqa: PLC0415

  out = []
  for name, (rel, want) in STRUCTURES.items():
    path = root / "energy_benchmark_datasets" / rel
    if _sha256(path) != want:
      msg = f"{path} sha256 does not match the pin"
      raise RuntimeError(msg)
    feats = featurize_pdb(path, None)
    length = int(feats["S"].shape[1])
    kept = kept_residues(path)
    raw = {
      "coords": np.asarray(feats["X"][0, :, :4, :], dtype=np.float32), "present": np.ones(length, dtype=np.float32),
      "residue_idx": np.asarray(feats["R_idx"][0], dtype=np.int32),
      "chain_index": np.asarray(feats["chain_labels"][0], dtype=np.int32), "pad_valid": np.ones(length, dtype=np.bool_),
    }
    out.append({"name": name, "length": length, "raw": raw, "native": np.asarray(feats["S"][0], dtype=np.int32),
                "res_id": np.asarray([k[1] for k in kept], dtype=np.int32)})
  return out


def _pins_arrays(plan: Any, p: int) -> dict[str, np.ndarray]:  # noqa: ANN401
  pin_dep = np.zeros((p, DEPTH), dtype=np.int32)
  pin_dep_valid = np.zeros((p, DEPTH), dtype=np.bool_)
  for i, pin in enumerate(plan.pins):
    pin_dep[i, : len(pin.dep_idxs)] = pin.dep_idxs
    pin_dep_valid[i, : len(pin.dep_idxs)] = True
  return {"pin_positions": np.asarray([pin.position for pin in plan.pins], dtype=np.int32),
          "pin_prot": np.asarray([pin.prot_idx for pin in plan.pins], dtype=np.int32),
          "pin_dep": pin_dep, "pin_dep_valid": pin_dep_valid}


def _cells(model: Any, structures: list[dict[str, Any]]) -> list[dict[str, Any]]:  # noqa: ANN401, C901, PLR0915
  import jax  # noqa: PLC0415
  import jax.numpy as jnp  # noqa: PLC0415

  from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig  # noqa: PLC0415
  from aminx.families.protonpotts_mpnn.ph_descent import rep_class_mask  # noqa: PLC0415
  from aminx.families.protonpotts_mpnn.ph_export import field_at, make_visit, pool, zblock  # noqa: PLC0415
  from aminx.families.protonpotts_mpnn.ph_plan import block_table, plan_from_center_types, valid_token_mask  # noqa: PLC0415
  from aminx.families.protonpotts_mpnn.ph_potentials import candidate_energies  # noqa: PLC0415

  config = PHDesignConfig(binder_chain="A")
  visit = jax.jit(make_visit(config))
  zb = jax.jit(zblock)
  pool_j = jax.jit(pool)
  field_j = jax.jit(field_at)
  table_fn = jax.jit(_table_fn(model))
  valid_tokens = valid_token_mask(config.forbidden_tokens)
  rep_mask = rep_class_mask(config.repetitive_window_parents).astype(np.float32)
  cells = []
  for st in structures:
    bucket = _bucket_for(st["length"])
    padded = _pad_inputs(st["raw"], bucket)
    table, e_idx = (np.asarray(x) for x in table_fn(*[jnp.asarray(padded[k]) for k in
                                                       ("coords", "present", "residue_idx", "chain_index", "pad_valid")]))
    seq = np.zeros(bucket, dtype=np.int32)
    seq[: st["length"]] = st["native"]
    n_chain = int(st["raw"]["chain_index"].max()) + 1
    binder = np.zeros(bucket, dtype=bool)
    binder[: st["length"]] = st["raw"]["chain_index"] == n_chain - 1
    res_id = np.zeros(bucket, dtype=np.int32)
    res_id[: st["length"]] = st["res_id"]
    residue_number = np.where(binder, res_id, -SENTINEL - 100 * np.arange(bucket)).astype(np.int32)
    field = np.asarray(candidate_energies(jnp.asarray(table), jnp.asarray(e_idx), jnp.asarray(seq)))
    for p, types in PIN_VARIANTS.items():
      plan = plan_from_center_types(field, e_idx, binder, res_id, types, DEP_MAP, infill_scope=config.infill_scope,
                                    neighbour_k=config.neighbour_k, max_mutations=config.max_mutations)
      if plan is None or not plan.designable:
        logger.info("%s P=%d: no plan", st["name"], p)
        continue
      blocks, valid = block_table(e_idx, plan.designable, BLOCK)
      pins = _pins_arrays(plan, p)
      seq_p = seq.copy()
      seq_p[pins["pin_positions"]] = pins["pin_prot"]
      common = {"table": table, "e_idx": e_idx.astype(np.int32), "seq": seq_p}
      zin = lambda b: [common["table"], common["e_idx"], common["seq"], blocks[b].astype(np.int32), valid[b], pins["pin_positions"],
                       pins["pin_prot"], pins["pin_dep"], pins["pin_dep_valid"], valid_tokens]  # noqa: E731
      n = blocks.shape[0]
      zstats = [[np.asarray(x) for x in zb(*[jnp.asarray(a) for a in zin(b)])] for b in range(n)]
      var_h = np.zeros(bucket, np.float32)
      var_s = np.zeros(bucket, np.float32)
      has_h = np.zeros(bucket, np.int32)
      has_s = np.zeros(bucket, np.int32)
      for b, (vh, hh, vs, hs) in enumerate(zstats):
        var_h[b], has_h[b], var_s[b], has_s[b] = vh[0], hh[0], vs[0], hs[0]
      w_h = np.asarray([1.0 - LAMBDA], np.float32)
      w_s = np.asarray([LAMBDA], np.float32)
      pool_in = [var_h, has_h, var_s, has_s, w_h, w_s]
      zscales, wh, wsel = (np.asarray(x) for x in pool_j(*[jnp.asarray(a) for a in pool_in]))
      chosen = sorted({0, n - 1, *np.linspace(0, n - 1, 4).astype(int).tolist()})
      visit_cases = []
      for b in chosen:
        for temperature in (0.0, TEMPERATURE):
          ins = [common["table"], common["e_idx"], common["seq"], blocks[b].astype(np.int32), valid[b], pins["pin_positions"],
                 pins["pin_prot"], pins["pin_dep"], pins["pin_dep_valid"], valid_tokens, rep_mask, residue_number, wh, wsel,
                 np.asarray([UNIFORM], np.float32), np.asarray([temperature], np.float32)]
          visit_cases.append({"block": b, "temperature": temperature, "inputs": ins,
                              "reference": [np.asarray(x) for x in visit(*[jnp.asarray(a) for a in ins])]})
      positions = np.arange(CHUNK, dtype=np.int32)
      field_in = [common["table"], common["e_idx"], common["seq"], positions]
      cells.append({
        "structure": st["name"], "bucket": bucket, "p": p, "n_blocks": n, "common": common,
        "zblock_cases": [{"block": b, "inputs": zin(b), "reference": zstats[b]} for b in range(n)],
        "pool_case": {"inputs": pool_in, "reference": [zscales, wh, wsel]},
        "visit_cases": visit_cases,
        "field_case": {"inputs": field_in, "reference": [np.asarray(x) for x in field_j(*[jnp.asarray(a) for a in field_in])]},
        "example": {"zblock": zin(0), "visit": visit_cases[0]["inputs"], "pool": pool_in, "field": field_in},
        "head_inputs": {"coords": padded["coords"], "present": padded["present"], "residue_idx": padded["residue_idx"],
                        "chain_index": padded["chain_index"], "pad_valid": padded["pad_valid"]},
      })
      logger.info("%s P=%d bucket=%d blocks=%d", st["name"], p, bucket, n)
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
    "graphs": {}, "n_cases": 0, "n_pass_cpu": 0, "n_pass_web": 0, "all_exported": False, "all_pass": False,
    "deterministic": False, "n_nondeterministic": 0, "control_float_detected": False, "control_int_detected": False,
    "control_race_detected": False, "ort_web_available": False, "manifest_sha256": "", "error": "",
  }
  args.work_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  try:
    if "jax2onnx" in sys.modules:
      msg = "jax2onnx imported before the JAX arm"
      raise RuntimeError(msg)  # noqa: TRY301
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415

    result["jax_enable_x64"] = bool(jax.config.jax_enable_x64)
    if result["jax_enable_x64"]:
      msg = "jax_enable_x64 is set; refusing"
      raise RuntimeError(msg)  # noqa: TRY301
    import jax.numpy as jnp  # noqa: PLC0415

    from aminx.families.protonpotts_mpnn import ph_descent  # noqa: PLC0415
    from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig  # noqa: PLC0415
    from aminx.families.protonpotts_mpnn.ph_export import field_at, make_visit, pool, zblock  # noqa: PLC0415

    model = _load_model(args.state_npz)
    structures = _structures(args.potts_root)
    cells = _cells(model, structures)
    if not cells:
      msg = "no cell produced a plan"
      raise RuntimeError(msg)  # noqa: TRY301

    # ---- controls (JAX side) before any conversion ------------------------------------------------------
    first = next((c for c in cells if c["bucket"] == BUCKETS[0] and c["p"] == 3), cells[0])
    noise_key = jax.random.PRNGKey(1)
    perturbed = jax.tree_util.tree_map(
      lambda leaf: leaf + 1e-2 * jax.random.normal(noise_key, leaf.shape, leaf.dtype)
      if (eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating)) else leaf, model.potts_head)
    head = first["head_inputs"]
    p_table = np.asarray(jax.jit(_table_fn(eqx.tree_at(lambda m: m.potts_head, model, perturbed)))(
      *[jnp.asarray(head[k]) for k in ("coords", "present", "residue_idx", "chain_index", "pad_valid")])[0])
    p_inputs = list(first["zblock_cases"][0]["inputs"])
    p_inputs[0] = p_table
    p_stats = [np.asarray(x) for x in jax.jit(zblock)(*[jnp.asarray(a) for a in p_inputs])]
    result["control_float_detected"] = not compare(first["zblock_cases"][0]["reference"][:1], p_stats[:1])["ok"]
    visit_refs = [vc["reference"] for vc in first["visit_cases"] if vc["temperature"] == 0.0]
    result["control_int_detected"] = len(visit_refs) > 1 and not all(
      compare(visit_refs[0], other)["ok"] for other in visit_refs[1:])
    result["jax_arm_ok"] = True
    logger.info("JAX arm: %d cells in %.1fs", len(cells), time.perf_counter() - t0)

    # ---- conversion ------------------------------------------------------------------------------------
    import onnxruntime as ort  # noqa: PLC0415
    import xtrax  # noqa: PLC0415
    from xtrax.export import ONNX, convert_to_onnx, run_onnx  # noqa: PLC0415

    from scripts.browser_validation.p07_knobs_gate import embed_external_data  # noqa: PLC0415

    config = PHDesignConfig(binder_chain="A")
    fns = {"visit": make_visit(config), "zblock": zblock, "pool": pool, "field": field_at}
    names = {"visit": VISIT_NAMES, "zblock": ZBLOCK_NAMES, "pool": POOL_NAMES, "field": FIELD_NAMES}
    exported_ok = True
    entries: dict[int, dict[str, Any]] = {b: {} for b in BUCKETS}
    seen: set[str] = set()
    for cell in cells:
      for graph, key in (("visit", "visit"), ("zblock", "zblock"), ("pool", "pool"), ("field", "field")):
        gname = f"{graph}_L{cell['bucket']}" + (f"_P{cell['p']}" if graph in ("visit", "zblock") else "")
        if gname in seen:
          continue
        seen.add(gname)
        path = args.work_dir / f"protonpotts_ph_{gname}.onnx"
        ins = cell["example"][key]
        try:
          convert_to_onnx(fns[graph], [jax.ShapeDtypeStruct(np.asarray(a).shape, np.asarray(a).dtype) for a in ins], ONNX,
                          out_path=path)
          embed_external_data(path)
          facts = _graph_facts(path)
          entries[cell["bucket"]][gname] = {
            "file": path.name, "bytes": facts["bytes"], "sha256": _sha256(path), "input_names": names[graph],
            "input_dtypes": [str(np.asarray(a).dtype) for a in ins], "input_shapes": [list(np.asarray(a).shape) for a in ins],
          }
          result["graphs"][gname] = {"converted": True, **facts}
        except Exception as exc:  # noqa: BLE001
          exported_ok = False
          result["graphs"][gname] = {"converted": False, "error": f"{type(exc).__name__}: {exc}"[:3000]}
    result["all_exported"] = exported_ok

    # RACE control: the pre-fix objective, exported and run repeatedly at the default thread count.
    racy_path = args.work_dir / "racy_visit.onnx"
    original = ph_descent.block_stability_potentials
    ph_descent.block_stability_potentials = _racy_potentials()
    try:
      racy_ins = first["example"]["visit"]
      racy_ins = [*racy_ins[:-1], np.asarray([0.0], np.float32)]
      racy_fn = make_visit(config)
      convert_to_onnx(lambda *a: (racy_fn(*a)[0],), [jax.ShapeDtypeStruct(np.asarray(a).shape, np.asarray(a).dtype)
                                                       for a in racy_ins], ONNX, out_path=racy_path)
      embed_external_data(racy_path)
      sess = ort.InferenceSession(str(racy_path), providers=["CPUExecutionProvider"])
      feed = {i.name: np.asarray(a) for i, a in zip(sess.get_inputs(), racy_ins, strict=True)}
      outs = [np.asarray(sess.run(None, feed)[0]) for _ in range(N_REPEATS + 3)]
      result["control_race_detected"] = not all(np.array_equal(outs[0], o) for o in outs[1:])
    finally:
      ph_descent.block_stability_potentials = original
    manifest = {
      "family": "protonpottsmpnn_ph", "checkpoint_sha256": CHECKPOINT_SHA256, "block_size": BLOCK, "depth": DEPTH,
      "chunk": CHUNK, "pin_counts": sorted(PIN_VARIANTS), "combined_lambda": LAMBDA, "temperature_checked": TEMPERATURE,
      "repetitive_window": {"weight": config.repetitive_window_weight, "radius": config.repetitive_window_radius,
                            "parents": list(config.repetitive_window_parents)},
      "forbidden_tokens": list(config.forbidden_tokens), "buckets": [{"bucket": b, "graphs": entries[b]} for b in BUCKETS],
    }
    manifest_path = args.work_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result["manifest_sha256"] = _sha256(manifest_path)

    # ---- ORT-CPU (+ determinism) and ORT-Web ----------------------------------------------------------------
    web_cases: list[dict[str, Any]] = []
    table_files: dict[str, str] = {}
    checks: list[dict[str, Any]] = []

    def gpath(graph: str, cell: dict[str, Any]) -> Path:
      suffix = f"_P{cell['p']}" if graph in ("visit", "zblock") else ""
      return args.work_dir / f"protonpotts_ph_{graph}_L{cell['bucket']}{suffix}.onnx"

    def add_case(cell: dict[str, Any], graph: str, label: str, ins: list[np.ndarray], ref: list[np.ndarray], *,
                 web: bool, repeat: bool) -> None:
      path = gpath(graph, cell)
      rec = {"structure": cell["structure"], "p": cell["p"], "graph": graph, "label": label, "cpu": "not_converted",
             "web": "not_converted", "deterministic": None}
      checks.append(rec)
      if not result["graphs"][path.stem.removeprefix("protonpotts_ph_")]["converted"]:
        return
      try:
        outs = run_onnx(path, *ins)
        cmp = compare(ref, outs)
        rec.update(cpu="pass" if cmp["ok"] else "mismatch", cpu_max_rel=cmp["max_rel"], cpu_why=cmp["why"])
      except Exception as exc:  # noqa: BLE001
        rec.update(cpu="not_runnable", error=f"{type(exc).__name__}: {exc}"[:1500])
      if repeat and rec["cpu"] in ("pass", "mismatch"):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feed = {i.name: np.asarray(a) for i, a in zip(sess.get_inputs(), ins, strict=True)}
        runs = [sess.run(None, feed) for _ in range(N_REPEATS)]
        rec["deterministic"] = all(all(np.array_equal(r0, rk) for r0, rk in zip(runs[0], run, strict=True)) for run in runs[1:])
      if web:
        case_id = f"{cell['structure']}_P{cell['p']}_{graph}_{label}"
        io_dir = args.work_dir / f"io_{case_id}"
        io_dir.mkdir(exist_ok=True)
        specs = []
        for i, arr in enumerate(ins):
          arr = np.ascontiguousarray(arr)
          tag = _tag(arr)
          if arr.nbytes > 1_000_000:  # the pair table is shared by every case of its cell
            key = f"{cell['structure']}_{arr.shape}_{i}_{arr.dtype}"
            if key not in table_files:
              f = args.work_dir / f"shared_{len(table_files)}.bin"
              arr.tofile(f)
              table_files[key] = str(f)
            specs.append({"file": table_files[key], "dtype": tag, "dims": list(arr.shape)})
            continue
          f = io_dir / f"in_{i}.bin"
          (arr.astype(np.uint8) if tag == "bool" else arr).tofile(f)
          specs.append({"file": str(f), "dtype": tag, "dims": list(arr.shape)})
        web_cases.append({"id": case_id, "model": str(path), "out_dir": str(io_dir), "inputs": specs})
        rec["case_id"] = case_id
        rec["ref"] = ref

    for cell in cells:
      for zc in cell["zblock_cases"]:
        add_case(cell, "zblock", f"b{zc['block']}", zc["inputs"], zc["reference"], web=zc["block"] in (0, cell["n_blocks"] - 1),
                 repeat=zc["block"] == 0)
      add_case(cell, "pool", "pool", cell["pool_case"]["inputs"], cell["pool_case"]["reference"], web=True, repeat=False)
      for i, vc in enumerate(cell["visit_cases"]):
        add_case(cell, "visit", f"b{vc['block']}_t{vc['temperature']}", vc["inputs"], vc["reference"], web=i < 4, repeat=i < 2)
      add_case(cell, "field", "chunk0", cell["field_case"]["inputs"], cell["field_case"]["reference"], web=True, repeat=False)

    web = _run_web(web_cases, args.work_dir, args.node_bin, args.ort_web_dir)
    result["ort_web_available"] = web["available"]
    result["ort_web_note"] = web["why"]
    for rec in checks:
      if rec["cpu"] == "not_converted":
        continue
      got = web["results"].get(rec.get("case_id", ""))
      if rec.get("case_id") is None:
        rec["web"] = "not_run"
      elif not web["available"] or got is None:
        rec["web"] = "unavailable"
      elif not got["ok"]:
        rec.update(web="not_runnable", web_error=(got.get("error") or "")[:1500])
      else:
        cmp = compare(rec["ref"], _read_web_outputs(got))
        rec.update(web="pass" if cmp["ok"] else "mismatch", web_max_rel=cmp["max_rel"], web_why=cmp["why"],
                   web_num_threads=got.get("num_threads"))
    for rec in checks:
      rec.pop("ref", None)

    import jax2onnx  # noqa: PLC0415

    result["versions"] = {"jax": jax.__version__, "xtrax": xtrax.__version__,
                          "jax2onnx": getattr(jax2onnx, "__version__", None),
                          "onnxruntime": ort.__version__, "onnxruntime_web": web.get("ort_web_version")}
    result["cells"] = checks
    result["n_cases"] = len(checks)
    result["n_pass_cpu"] = sum(r["cpu"] == "pass" for r in checks)
    run_web = [r for r in checks if r.get("case_id")]
    result["n_pass_web"] = sum(r["web"] == "pass" for r in run_web)
    result["n_nondeterministic"] = sum(r["deterministic"] is False for r in checks)
    result["deterministic"] = any(r["deterministic"] for r in checks) and result["n_nondeterministic"] == 0
    result["all_pass"] = bool(checks) and exported_ok and all(r["cpu"] == "pass" for r in checks) and all(
      r["web"] == "pass" for r in run_web)
  except Exception as exc:  # noqa: BLE001 -- a graded run must still emit and exit 0
    logger.exception("gate failed")
    result["error"] = f"{type(exc).__name__}: {exc}"[:4000]
  result["elapsed_s"] = time.perf_counter() - t0
  emit(result, args.out)
  logger.info("cpu %s/%s web %s deterministic %s", result["n_pass_cpu"], result["n_cases"], result["n_pass_web"],
              result["deterministic"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
