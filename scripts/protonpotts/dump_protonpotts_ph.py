"""Dump upstream ProtonPottsMPNN pH-design runs (block descent, greedy) on the sealed tables, f32 and f64 (P4g).

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §45-§46. Run in the `aminx-oracles-protonpotts`
environment; imports upstream `mpnn`, torch and numpy only, never aminx::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/protonpotts/dump_protonpotts_ph.py --checkpoint <ckpt> --encoder-dir <P4e out dir> \\
        --features-dir ~/projects/aminx-oracles-protonpotts --out <dir> \\
        --cell pkad_unlabelled=<1BVC.pdb>,<chain> --cell multichain=<6m0j.pdb>,E ...

WHAT RUNS. The real ``PottsMPNNPHEngine.run_ph_redesign`` (serial), with ``run_forward_compat`` returning the SEALED
P4e merged table and ``E_idx`` instead of running the model. The optimiser consumes nothing else from the model
(spec §45.1), so what is compared later is the optimiser, on tables the encoder wave already graded. For float64 the
default dtype is double for the run, as in the P4e dump.

THE RANDOMNESS SEAM. Upstream draws each block with ``torch.multinomial(softmax(-(J - min J)/T), 1)`` (T > 0) or
``argmin`` (T <= 0). ``torch.multinomial`` is replaced by a seeded inverse-CDF shim: ``choice = first index with
cumsum(p) >= u * sum(p)`` for a uniform ``u`` from a numpy stream. Every ``(u, choice)`` is recorded per design, so the
aminx side can replay the SAME uniforms. T = 0 consumes none.

RECORDED, per design call: the initial and final sequence (upstream token order), pins, designable positions, the block
membership of every position, the pooled z-scales ``(sdH, sdSel, sdGlob)``, the uniforms and choices, the trajectory and
the scored ``PHDesignOutput``. Per cell: the native ``S``, masks, residue ids and names, the vocabulary.

Threads are pinned to 1 (spec §32).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
  os.environ[_var] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_protonpotts_features import _content_hash  # noqa: E402

logger = logging.getLogger("dump_protonpotts_ph")

SEED = 0
PRECISIONS = (("f32", torch.float32), ("f64", torch.float64))
FEATURES = {
  "pkad_unlabelled": "features_v6/protonpotts_v6_features",
  "multichain": "features_v6/protonpotts_v6_features",
  "gap": "features_v6_p4c/protonpotts_v6_features_p4c",
  "only_o_1olr": "features_v6_p4d/protonpotts_v6_features_p4c",
}
DEP_MAP = {"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]}
FORBIDDEN = ["HIS-A", "ASP-A", "GLU-A", "UNK"]
REP_PARENTS = ["ARG", "LYS", "HIS", "ASP", "GLU"]


def _criteria() -> dict[str, object]:
  """The four configurations, labelled. Production values from ``inference/design_ph.py``."""
  from mpnn.inference_engines.potts_mpnn_ph import PHDesignCriteria

  common = dict(
    backend="potts", seed_source="native", block_size=3, combined_lambda=0.3, dep_map=DEP_MAP,
    forbidden_tokens=FORBIDDEN, repetitive_window_parents=REP_PARENTS, repetitive_window_radius=2,
    repetitive_window_weight=1.0, record_trajectory=True,
  )  # fmt: skip
  placed = dict(
    method="block_descent", center_types=["HIS-P", "ASP-P", "GLU-P"], placement_by="scan_potts",
    placement_region=["all"], neighbour_k=16, max_mutations=20,
  )  # fmt: skip
  free = dict(method="greedy_energy_block", center_count=0, infill_scope="chain", cv_patience=1, cv_max=2)
  return {
    "block_T0": PHDesignCriteria(temperature=0.0, samples_per_site=1, **placed, **common),
    "block_T05": PHDesignCriteria(temperature=0.05, samples_per_site=2, **placed, **common),
    "greedy_T0": PHDesignCriteria(temperature=0.0, samples_per_site=1, **free, **common),
    "greedy_T05": PHDesignCriteria(temperature=0.05, samples_per_site=2, **free, **common),
  }


class _Stream:
  """The inverse-CDF replacement for ``torch.multinomial(probs, 1)``, with a record of every draw."""

  def __init__(self) -> None:
    self.rng = np.random.default_rng(SEED)
    self.log: list[tuple[float, int]] = []

  def reseed(self, seed: int) -> None:
    self.rng = np.random.default_rng(seed)

  def multinomial(self, probs, num_samples=1, replacement=False, *, generator=None):  # noqa: ANN001, ANN201
    del replacement
    if num_samples != 1 or probs.dim() != 1 or generator is not None:
      msg = f"unexpected multinomial call: num_samples={num_samples}, dim={probs.dim()}"
      raise RuntimeError(msg)
    u = float(self.rng.random())
    cumulative = np.cumsum(probs.detach().cpu().double().numpy())
    choice = min(int(np.searchsorted(cumulative, u * cumulative[-1], side="left")), len(cumulative) - 1)
    self.log.append((u, choice))
    return torch.tensor([choice], dtype=torch.long)


def _np(value):  # noqa: ANN001, ANN201
  return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _jsonable(value):  # noqa: ANN001, ANN201
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return _jsonable(dataclasses.asdict(value))
  if isinstance(value, dict):
    return {str(k): _jsonable(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(v) for v in value]
  if isinstance(value, (np.floating, np.integer)):
    return value.item()
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, torch.Tensor):
    return value.detach().cpu().tolist()
  return value


def _run_cell(engine, ph, atom_array, chain: str, tables: dict[str, np.ndarray], dtype, criteria: dict):  # noqa: ANN001, ANN201
  """One cell at one precision. Returns (arrays, meta)."""
  stream = _Stream()
  table = torch.from_numpy(tables["etab_postmerge"]).to(dtype)[None]
  e_idx = torch.from_numpy(tables["E_idx"])[None]
  calls: list[dict] = []
  designs: list[dict] = []
  ctx_info: dict = {}
  state: dict = {"label": None, "call": None}
  originals = {
    "forward": ph.run_forward_compat, "multinomial": torch.multinomial, "partners": ph._block_partners,
    "zscales": ph._block_zscales, "block": ph._block_descent, "greedy": ph._greedy_energy_block,
    "score": ph.PottsMPNNPHEngine._score_design, "design_one": ph.PottsMPNNPHEngine._design_one,
    "build": ph.PottsMPNNPHEngine._build_context,
  }  # fmt: skip
  by_id = {id(crit): name for name, crit in criteria.items()}

  def forward(model, network_input):  # noqa: ANN001, ANN202
    del model, network_input
    return {"etab_out": table, "E_idx": e_idx}

  def build(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
    ctx = originals["build"](self, *args, **kwargs)
    token_aa = ctx.token_aa
    ctx_info.update(
      S_native=_np(ctx.S_native).copy(), free_mask=_np(ctx.free_mask).copy(), chainA=_np(ctx.chainA_t).copy(),
      res_id=_np(token_aa.res_id).copy(), res_name=np.asarray(token_aa.res_name).astype(str),
      unknown=np.asarray(ctx.unknown_indices, dtype=np.int64), canonical_map=_np(ctx.canonical_map).copy(),
      eidx=np.asarray(ctx.eidx_np).copy(),
      idx_to_token=[str(ctx.encoding.idx_to_token[i]) for i in range(int(ctx.V))],
    )  # fmt: skip
    return ctx

  def partners(ctx, p, neigh_set, block_size):  # noqa: ANN001, ANN202
    block = originals["partners"](ctx, p, neigh_set, block_size)
    if state["call"] is not None:
      state["call"]["blocks"].setdefault(int(p), [int(x) for x in block])
    return block

  def zscales(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
    out = originals["zscales"](*args, **kwargs)
    if state["call"] is not None:
      state["call"]["zscales"] = [float(x) for x in out]
    return out

  def wrap_optimiser(name: str):  # noqa: ANN202
    def run(ctx, crit, valid_mask, pins, neigh, initial_sequence, rng, trajectory=None):  # noqa: ANN001, ANN202
      stream.reseed(SEED * 1_000_003 + len(calls))
      start = len(stream.log)
      record = {
        "index": len(calls), "method": name, "label": state["label"], "temperature": float(crit.temperature),
        "S_in": _np(initial_sequence).copy(), "neigh": [int(x) for x in neigh], "blocks": {},
        "valid": _np(valid_mask).copy(), "zscales": None,
        "pins": [
          {"position": int(p.position), "protonation_type": p.protonation_type, "prot_idx": int(p.prot_idx),
           "dep_idxs": [int(x) for x in p.dep_idxs], "res_id": int(p.res_id)} for p in pins
        ],
      }  # fmt: skip
      state["call"] = record
      try:
        out = originals[name](ctx, crit, valid_mask, pins, neigh, initial_sequence, rng, trajectory=trajectory)
      finally:
        state["call"] = None
      record["S_out"] = _np(out).copy()
      draws = stream.log[start:]
      record["uniforms"] = np.asarray([u for u, _ in draws], dtype=np.float64)
      record["choices"] = np.asarray([c for _, c in draws], dtype=np.int64)
      record["trajectory"] = _jsonable(trajectory)
      calls.append(record)
      return out

    return run

  def score(self, ctx, crit, seq, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
    out = originals["score"](self, ctx, crit, seq, **kwargs)
    designs.append({"label": state["label"], "call_index": len(calls) - 1, **_jsonable(out)})
    return out

  def design_one(self, ctx, crit, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
    state["label"] = by_id[id(crit)]
    return originals["design_one"](self, ctx, crit, *args, **kwargs)

  ph.run_forward_compat = forward
  torch.multinomial = stream.multinomial
  ph._block_partners = partners
  ph._block_zscales = zscales
  ph._block_descent = wrap_optimiser("block")
  ph._greedy_energy_block = wrap_optimiser("greedy")
  ph.PottsMPNNPHEngine._score_design = score
  ph.PottsMPNNPHEngine._design_one = design_one
  ph.PottsMPNNPHEngine._build_context = build
  previous_default = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    design_set = engine.run_ph_redesign(
      atom_array=atom_array, binder_chain=chain, criteria_list=list(criteria.values()), seed=SEED, n_jobs=1
    )
  finally:
    torch.set_default_dtype(previous_default)
    ph.run_forward_compat = originals["forward"]
    torch.multinomial = originals["multinomial"]
    ph._block_partners = originals["partners"]
    ph._block_zscales = originals["zscales"]
    ph._block_descent = originals["block"]
    ph._greedy_energy_block = originals["greedy"]
    ph.PottsMPNNPHEngine._score_design = originals["score"]
    ph.PottsMPNNPHEngine._design_one = originals["design_one"]
    ph.PottsMPNNPHEngine._build_context = originals["build"]

  arrays: dict[str, np.ndarray] = {}
  for key, value in ctx_info.items():
    if key != "idx_to_token":
      arrays[f"ctx_{key}"] = np.asarray(value)
  meta_calls = []
  for call in calls:
    i = call["index"]
    for key in ("S_in", "S_out", "valid", "uniforms", "choices"):
      arrays[f"c{i}_{key}"] = np.asarray(call[key])
    meta_calls.append({k: v for k, v in call.items() if k not in ("S_in", "S_out", "valid", "uniforms", "choices")})
  final = [
    {"label": d["label"], "final_potts_energy": d["final_potts_energy"], "selective_energy": d["selective_energy"]}
    for d in designs
  ]
  arrays["design_final_potts_energy"] = np.asarray([d["final_potts_energy"] for d in final], dtype=np.float64)
  arrays["design_selective_energy"] = np.asarray(
    [np.nan if d["selective_energy"] is None else d["selective_energy"] for d in final], dtype=np.float64
  )
  return arrays, {
    "calls": meta_calls, "designs": designs, "n_returned": len(design_set),
    "idx_to_token": ctx_info.get("idx_to_token"),
  }  # fmt: skip


def _load(path: Path) -> dict[str, np.ndarray]:
  with np.load(path) as z:
    return {key: z[key] for key in z.files}


def _dump(args: argparse.Namespace, out: Path) -> dict[str, dict]:
  import mpnn.inference_engines.potts_mpnn_ph as ph
  import mpnn.pipelines.potts_mpnn as pipelines
  import mpnn.potts_inference as inference
  from mpnn.utils.inference import MPNNInferenceInput

  pipelines.get_protonation_state_transforms = lambda **_: []
  inference._PIPELINE_CACHE.clear()  # noqa: SLF001 -- trap 1 of the P4 dumper (spec §23.4)
  torch.set_num_threads(1)
  out.mkdir(parents=True, exist_ok=True)
  engine = ph.PottsMPNNPHEngine(
    checkpoint_path=str(args.checkpoint), extended_vocab="v6", out_directory=None, write_fasta=False,
    write_structures=False,
  )  # fmt: skip
  criteria = _criteria()
  report: dict[str, dict] = {}
  for cell, (pdb, chain) in args.cells.items():
    # The SAME loader prepare_potts_input uses for a path (atomworks, occupancy and altlocs handled), so the
    # engine featurises the residues the sealed tables describe. A first graded attempt that read the file with
    # biotite directly gave the engine more residues than the sealed 6m0j table (IndexError in placement).
    atom_array = MPNNInferenceInput.from_atom_array_and_dict(input_dict={"structure_path": str(pdb)}).atom_array
    sealed = _load(args.features_dir / FEATURES[cell] / f"{cell}.npz")
    entry: dict = {"binder_chain": chain, "precisions": {}}
    for name, dtype in PRECISIONS:
      tables = _load(args.encoder_dir / "protonpotts_v6_encoder" / f"{cell}_{name}.npz")
      arrays, meta = _run_cell(engine, ph, atom_array, chain, tables, dtype, criteria)
      np.savez(out / f"{cell}_{name}.npz", **arrays)
      (out / f"{cell}_{name}.json").write_text(json.dumps(meta, sort_keys=True, default=str), encoding="utf-8")
      entry["precisions"][name] = {
        "content_sha256": _content_hash(arrays),
        "meta_sha256": hashlib.sha256(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest(),
        "n_calls": len(meta["calls"]),
        "dtypes": sorted({str(a.dtype) for k, a in arrays.items() if k.startswith("design_")}),
      }
    entry["sealed_L"] = int(sealed["S"].shape[-1])
    report[cell] = entry
  return report


def _flatten(report: dict[str, dict]) -> dict[str, str]:
  return {
    f"{c}/{p}": v["content_sha256"] + v["meta_sha256"] for c, e in report.items() for p, v in e["precisions"].items()
  }


def _child(args: argparse.Namespace) -> dict[str, str]:
  with tempfile.TemporaryDirectory() as tmp:
    result = Path(tmp) / "hashes.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", str(args.checkpoint),
               "--encoder-dir", str(args.encoder_dir), "--features-dir", str(args.features_dir),
               "--out", str(Path(tmp) / "child"), "--child-hashes-to", str(result)]  # fmt: skip
    for name, (pdb, chain) in args.cells.items():
      command += ["--cell", f"{name}={pdb},{chain}"]
    subprocess.run(command, check=True, capture_output=True)
    return json.loads(result.read_text(encoding="utf-8"))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--encoder-dir", type=Path, required=True)
  parser.add_argument("--features-dir", type=Path, required=True)
  parser.add_argument("--cell", action="append", default=[], metavar="NAME=PDB,CHAIN")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--child-hashes-to", type=Path, default=None)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  args.cells = {}
  for item in args.cell:
    name, _, rest = item.partition("=")
    pdb, _, chain = rest.rpartition(",")
    args.cells[name] = (Path(pdb), chain)

  directory = args.out / "protonpotts_v6_ph"
  report = _dump(args, directory)
  if args.child_hashes_to is not None:
    args.child_hashes_to.write_text(json.dumps(_flatten(report)), encoding="utf-8")
    return 0

  reproducible = _child(args) == _flatten(report)
  complete = shim_consistent = f64_ok = precisions_differ = engine_matches_sealed = True
  token_agreement: dict[str, float] = {}
  labels = set(_criteria())
  for cell in args.cells:
    metas = {p: json.loads((directory / f"{cell}_{p}.json").read_text(encoding="utf-8")) for p in ("f32", "f64")}
    arrs = {p: _load(directory / f"{cell}_{p}.npz") for p in ("f32", "f64")}
    for p in ("f32", "f64"):
      seen = {c["label"] for c in metas[p]["calls"]}
      complete &= seen == labels and all(
        c["zscales"] is not None for c in metas[p]["calls"] if c["method"] == "block"
      )
      for c in metas[p]["calls"]:
        used = arrs[p][f"c{c['index']}_uniforms"].shape[0]
        shim_consistent &= (used == 0) if c["temperature"] <= 0 else (used > 0)
    sealed = _load(args.features_dir / FEATURES[cell] / f"{cell}.npz")
    tables = _load(args.encoder_dir / "protonpotts_v6_encoder" / f"{cell}_f32.npz")
    for p in ("f32", "f64"):
      engine_matches_sealed &= bool(
        arrs[p]["ctx_S_native"].shape == sealed["S"][0].shape
        and np.array_equal(arrs[p]["ctx_S_native"], sealed["S"][0])
        and np.array_equal(arrs[p]["ctx_eidx"], tables["E_idx"])
      )
    f64_ok &= arrs["f64"]["design_final_potts_energy"].dtype == np.float64
    precisions_differ &= bool(
      np.abs(arrs["f64"]["design_final_potts_energy"] - arrs["f32"]["design_final_potts_energy"].astype(np.float64)).max()
      > 0.0
    )
    same = [
      bool(np.array_equal(arrs["f32"][f"c{i}_S_out"], arrs["f64"][f"c{i}_S_out"]))
      for i in range(min(len(metas["f32"]["calls"]), len(metas["f64"]["calls"])))
    ]
    token_agreement[cell] = float(np.mean(same)) if same else float("nan")
  flags = {
    "engine_matches_sealed": bool(engine_matches_sealed),
    "complete": bool(complete),
    "shim_consistent": bool(shim_consistent),
    "f64_is_float64": bool(f64_ok),
    "precisions_differ": bool(precisions_differ),
    "reproducible": bool(reproducible),
    "n_cells": len(report),
  }
  (args.out / "ph_manifest.json").write_text(
    json.dumps(
      {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
       "torch_version": torch.__version__, "seed": SEED, "cells": {n: [str(p), c] for n, (p, c) in args.cells.items()},
       "report": report, "f32_vs_f64_same_final_tokens": token_agreement, "flags": flags},
      indent=2, sort_keys=True),
    encoding="utf-8",
  )  # fmt: skip
  logger.info("flags %s agreement %s", flags, token_agreement)
  results = os.environ.get("BTH_RESULTS_PATH")
  if results:
    Path(results).write_text(json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8")
  else:
    logger.warning("BTH_RESULTS_PATH unset: this run will record outcome 'unknown'")
  return 0


if __name__ == "__main__":
  sys.exit(main())
