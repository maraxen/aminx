"""Dump upstream ProtonPottsMPNN pH-engine runs of the decoder-backed and sampling methods, f32 and f64 (P4j).

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §56; debts #2616, #2617. Sidecar: ``dump_protonpotts_methods.bth.toml``
(committed before this script). Run in the `aminx-oracles-protonpotts` environment; imports upstream `mpnn`, torch and numpy only,
never aminx::

    bth run --project-slug aminx -- <oracle-env python> scripts/protonpotts/dump_protonpotts_methods.py \\
        --checkpoint <ckpt> --encoder-dir <P4e out dir> --features-dir ~/projects/aminx-oracles-protonpotts --out <dir> \\
        --cell multichain=<6m0j.pdb>,E --cell pkad_unlabelled=<1BVC.pdb>,A

WHAT RUNS. The real ``PottsMPNNPHEngine.run_ph_redesign`` (serial, real checkpoint, real featurisation, real model forward) with
eleven criteria in one call per unit (cell x precision). The only changes are the randomness shims below and, for float64, the three
widening changes of P4h (``dump_protonpotts_decoder._f64_changes``) plus ``engine.model.double()``.

THE RANDOMNESS SEAM. ``torch.multinomial``, ``torch.randint`` and ``torch.randperm`` are replaced by draws from a seeded numpy stream
(inverse CDF for the multinomial, the cumsum taken in float64 as in the P4g dump). They act only inside an outermost optimiser call;
outside one they delegate to torch. A call signature other than the engine's raises. Every draw is recorded with what it saw.

UNITS. Each unit runs in its own process with its own timeout and writes ``<cell>_<precision>.npz/.json`` plus a completion stamp holding
the sha256 of the inputs and of both artifacts; a re-run reuses a unit only when all of them still match (the standing preemption rule).
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

logger = logging.getLogger("dump_protonpotts_methods")

SEED = 0
UNIT_TIMEOUT_S = 3 * 3600
PRECISIONS = (("f32", torch.float32), ("f64", torch.float64))
FEATURES = {
  "pkad_unlabelled": "features_v6/protonpotts_v6_features",
  "multichain": "features_v6/protonpotts_v6_features",
}
REPRODUCE_UNIT = ("pkad_unlabelled", "f32")
DEP_MAP = {"HIS-P": ["HIS-S"], "ASP-P": ["ASP-D"], "GLU-P": ["GLU-D"]}
FORBIDDEN = ["HIS-A", "ASP-A", "GLU-A", "UNK"]
REP_PARENTS = ["ARG", "LYS", "HIS", "ASP", "GLU"]
LABELS = (
  "ar_potts", "ar_decoder", "mcmc_potts", "mcmc_potts_T0", "mcmc_potts_nonsel", "mcmc_mpnn", "combined_potts",
  "two_phase_potts", "two_phase_mpnn", "place_mpnn", "gibbs",
)  # fmt: skip


def _criteria() -> dict[str, object]:
  """The eleven configurations, labelled. Production values from ``inference/design_ph.py``."""
  from mpnn.inference_engines.potts_mpnn_ph import PHDesignCriteria

  common = dict(
    seed_source="native", block_size=3, combined_lambda=0.3, dep_map=DEP_MAP, forbidden_tokens=FORBIDDEN,
    repetitive_window_parents=REP_PARENTS, repetitive_window_radius=2, repetitive_window_weight=1.0, record_trajectory=False,
    center_types=["HIS-P"], placement_by="scan_potts", placement_region=["all"], neighbour_k=8, max_mutations=8,
    temperature=0.1, samples_per_site=1, cv_patience=1, cv_max=2,
  )  # fmt: skip

  def make(**overrides):  # noqa: ANN003, ANN202
    return PHDesignCriteria(**{**common, **overrides})

  return {
    "ar_potts": make(method="autoregressive", backend="mpnn", selective_source="potts"),
    "ar_decoder": make(method="autoregressive", backend="mpnn", selective_source="decoder"),
    "mcmc_potts": make(method="converged_mcmc", backend="potts"),
    "mcmc_potts_T0": make(method="converged_mcmc", backend="potts", temperature=0.0),
    "mcmc_potts_nonsel": make(method="converged_mcmc", backend="potts", selective=False),
    "mcmc_mpnn": make(method="converged_mcmc", backend="mpnn", cv_max=1),
    "combined_potts": make(method="converged_mcmc_combined", backend="potts"),
    "two_phase_potts": make(method="two_phase", backend="potts", two_phase_frac=0.5),
    "two_phase_mpnn": make(method="two_phase", backend="mpnn", two_phase_frac=0.5, cv_max=1),
    "place_mpnn": make(method="converged_mcmc", backend="potts", placement_by="scan_mpnn", placement_seq_masked=True),
    "gibbs": make(method="gibbs", backend="potts", num_designs=2),
  }  # fmt: skip


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


class _Stream:
  """Seeded replacements for ``torch.multinomial`` / ``randint`` / ``randperm``, recording into the active call."""

  def __init__(self, state: dict) -> None:
    self.state = state
    self.rng = np.random.default_rng(SEED)

  def reseed(self, seed: int) -> None:
    self.rng = np.random.default_rng(seed)

  def multinomial(self, probs, num_samples=1, replacement=False, *, generator=None):  # noqa: ANN001, ANN201
    call = self.state["call"]
    if call is None:
      return self.state["originals"]["multinomial"](probs, num_samples, replacement, generator=generator)
    if probs.dim() == 2:  # noqa: PLR2004
      # The MODEL's own sampler (logits_to_sample) draws a token over all [L, V] rows even in teacher forcing, where the
      # result is unused. It is not an engine draw: delegate to torch and count it.
      state_counts = self.state["model_multinomial"]
      state_counts[0] += 1
      return self.state["originals"]["multinomial"](probs, num_samples, replacement, generator=generator)
    if num_samples != 1 or probs.dim() != 1 or generator is not None:
      msg = f"unexpected multinomial call: num_samples={num_samples}, dim={probs.dim()}"
      raise RuntimeError(msg)
    u = float(self.rng.random())
    cumulative = np.cumsum(probs.detach().cpu().double().numpy())
    total = cumulative[-1]
    choice = min(int(np.searchsorted(cumulative, u * total, side="left")), len(cumulative) - 1)
    call["mult"].append((_np(probs).copy(), u, choice, float(np.abs(cumulative / total - u).min())))
    return torch.tensor([choice], dtype=torch.long)

  def randint(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
    call = self.state["call"]
    if call is None:
      return self.state["originals"]["randint"](*args, **kwargs)
    if len(args) != 2 or kwargs or tuple(args[1]) != (1,):
      msg = f"unexpected randint call: args={args} kwargs={kwargs}"
      raise RuntimeError(msg)
    high = int(args[0])
    value = int(self.rng.integers(high))
    call["randint"].append((high, value))
    return torch.tensor([value], dtype=torch.long)

  def randperm(self, n, *args, device=None, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN201
    call = self.state["call"]
    if call is None:
      return self.state["originals"]["randperm"](n, *args, device=device, **kwargs)
    if args or kwargs:
      msg = f"unexpected randperm call: args={args} kwargs={kwargs}"
      raise RuntimeError(msg)
    perm = self.rng.permutation(int(n)).astype(np.int64)
    call["perm"].append(perm)
    return torch.from_numpy(perm).to(device) if device is not None else torch.from_numpy(perm)


def _empty_call(index: int, name: str, label: str | None) -> dict:
  return {
    "index": index, "method": name, "label": label, "pins": [], "neigh": [], "order": None, "zscales": None,
    "mult": [], "randint": [], "perm": [], "pick": [],
  }  # fmt: skip


def _run_unit(engine, ph, atom_array, chain: str, tables: dict[str, np.ndarray], dtype, criteria: dict):  # noqa: ANN001, ANN201, C901, PLR0915
  """One cell at one precision. Returns (arrays, meta)."""
  del tables  # the engine runs the real model; the sealed tables are only compared against afterwards
  state: dict = {"call": None, "label": None, "originals": {}, "model_multinomial": [0]}
  stream = _Stream(state)
  calls: list[dict] = []
  designs: list[dict] = []
  placements: list[dict] = []
  ctx_info: dict = {}
  by_id = {id(crit): name for name, crit in criteria.items()}
  gibbs_original = ph.PottsMPNN.potts_gibbs_optimize
  originals = {
    "multinomial": torch.multinomial, "randint": torch.randint, "randperm": torch.randperm, "pick": ph._pick,
    "zscales": ph._sampler_zscales, "infill": ph._masked_infill, "converge": ph._converge, "two_phase": ph._two_phase,
    "ranked": ph._ranked_candidates, "score": ph.PottsMPNNPHEngine._score_design,
    "build": ph.PottsMPNNPHEngine._build_context, "placement_one": ph.PottsMPNNPHEngine._run_placement_one_seed,
    "whole_chain": ph.PottsMPNNPHEngine._run_whole_chain,
  }  # fmt: skip
  state["originals"] = originals

  def build(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
    ctx = originals["build"](self, *args, **kwargs)
    token_aa = ctx.token_aa
    ctx_info.update(
      S_native=_np(ctx.S_native).copy(), free_mask=_np(ctx.free_mask).copy(), chainA=_np(ctx.chainA_t).copy(),
      res_id=_np(token_aa.res_id).copy(), res_name=np.asarray(token_aa.res_name).astype(str),
      unknown=np.asarray(ctx.unknown_indices, dtype=np.int64), canonical_map=_np(ctx.canonical_map).copy(),
      eidx=np.asarray(ctx.eidx_np).copy(), etab_dtype=str(ctx.scorer.etab_out.dtype),
      idx_to_token=[str(ctx.encoding.idx_to_token[i]) for i in range(int(ctx.V))],
    )  # fmt: skip
    return ctx

  def pick(score, temperature):  # noqa: ANN001, ANN202
    result = originals["pick"](score, temperature)
    call = state["call"]
    if call is not None:
      call["pick"].append((_np(score).copy(), float(temperature), int(result)))
    return result

  def zscales(ctx, S, order, pins, valid_idx):  # noqa: ANN001, ANN202
    out = originals["zscales"](ctx, S, order, pins, valid_idx)
    if state["call"] is not None:
      state["call"]["zscales"] = [float(x) for x in out]
    return out

  def pins_record(pins) -> list[dict]:  # noqa: ANN001
    return [
      {"position": int(p.position), "protonation_type": p.protonation_type, "prot_idx": int(p.prot_idx),
       "dep_idxs": [int(x) for x in p.dep_idxs], "res_id": int(p.res_id)} for p in pins
    ]  # fmt: skip

  def outermost(name: str, fn, make_record):  # noqa: ANN001, ANN202
    def run(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
      if state["call"] is not None:
        return fn(*args, **kwargs)  # nested (two_phase -> converge): the outer call owns the record
      record = _empty_call(len(calls), name, state["label"])
      make_record(record, args, kwargs)
      stream.reseed(SEED * 1_000_003 + len(calls))
      state["call"] = record
      try:
        out = fn(*args, **kwargs)
      finally:
        state["call"] = None
      record["S_out"] = _np(out[0] if isinstance(out, tuple) else out).copy()
      calls.append(record)
      return out

    return run

  def rec_infill(record, args, kwargs):  # noqa: ANN001, ANN202
    ctx, crit, valid_mask, pins, order, initial = args[:6]
    record.update(
      pins=pins_record(pins), order=[int(x) for x in order], neigh=sorted(int(x) for x in order),
      S_in=_np(initial).copy(), valid=_np(valid_mask).copy(), temperature=float(crit.temperature),
    )  # fmt: skip

  def rec_converge(record, args, kwargs):  # noqa: ANN001, ANN202
    ctx, crit, field_at, valid_mask, pins, neigh, initial = args[:7]
    start = kwargs.get("S0", args[7] if len(args) > 7 else None)
    record.update(
      pins=pins_record(pins), neigh=[int(x) for x in neigh], S_in=_np(initial if start is None else start).copy(),
      valid=_np(valid_mask).copy(), temperature=float(crit.temperature),
    )  # fmt: skip

  def rec_gibbs(record, args, kwargs):  # noqa: ANN001, ANN202
    del args
    seq_init, free_mask, valid = kwargs["seq_init"], kwargs["free_mask"], kwargs["valid_aa_mask"]
    record.update(
      S_in=_np(seq_init).copy(), free_mask=_np(free_mask).copy(), valid=_np(valid).copy(),
      temperature=float(kwargs["temperature"]), neigh=[int(x) for x in _np(free_mask).nonzero()[0]],
    )  # fmt: skip

  def ranked(ctx, crit, field_fn, prot_idx, dep_idxs, initial_sequence, region_idx, mask_seq):  # noqa: ANN001, ANN202
    out = originals["ranked"](ctx, crit, field_fn, prot_idx, dep_idxs, initial_sequence, region_idx, mask_seq)
    placements.append({
      "label": state["label"], "prot_idx": int(prot_idx), "mask_seq": bool(mask_seq),
      "ranked": [[int(p), float(s)] for p, s in out],
    })  # fmt: skip
    return out

  def placement_one(self, ctx, crit, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
    state["label"] = by_id[id(crit)]
    return originals["placement_one"](self, ctx, crit, *args, **kwargs)

  def whole_chain(self, ctx, crit, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
    state["label"] = by_id[id(crit)]
    return originals["whole_chain"](self, ctx, crit, *args, **kwargs)

  def score(self, ctx, crit, seq, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
    out = originals["score"](self, ctx, crit, seq, **kwargs)
    designs.append({"label": state["label"], "call_index": len(calls) - 1, **_jsonable(out)})
    return out

  wrapped_gibbs = outermost("gibbs", gibbs_original, rec_gibbs)

  def gibbs(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
    return wrapped_gibbs(*args, **kwargs)

  torch.multinomial, torch.randint, torch.randperm = stream.multinomial, stream.randint, stream.randperm
  ph._pick, ph._sampler_zscales, ph._ranked_candidates = pick, zscales, ranked
  ph._masked_infill = outermost("infill", originals["infill"], rec_infill)
  ph._converge = outermost("converge", originals["converge"], rec_converge)
  ph._two_phase = outermost("two_phase", originals["two_phase"], rec_converge)
  ph.PottsMPNN.potts_gibbs_optimize = staticmethod(gibbs)
  ph.PottsMPNNPHEngine._score_design = score
  ph.PottsMPNNPHEngine._build_context = build
  ph.PottsMPNNPHEngine._run_placement_one_seed = placement_one
  ph.PottsMPNNPHEngine._run_whole_chain = whole_chain
  try:
    design_set = engine.run_ph_redesign(
      atom_array=atom_array, binder_chain=chain, criteria_list=list(criteria.values()), seed=SEED, n_jobs=1
    )
  finally:
    torch.multinomial, torch.randint, torch.randperm = originals["multinomial"], originals["randint"], originals["randperm"]
    ph._pick, ph._sampler_zscales, ph._ranked_candidates = originals["pick"], originals["zscales"], originals["ranked"]
    ph._masked_infill, ph._converge, ph._two_phase = originals["infill"], originals["converge"], originals["two_phase"]
    ph.PottsMPNN.potts_gibbs_optimize = staticmethod(gibbs_original)
    ph.PottsMPNNPHEngine._score_design = originals["score"]
    ph.PottsMPNNPHEngine._build_context = originals["build"]
    ph.PottsMPNNPHEngine._run_placement_one_seed = originals["placement_one"]
    ph.PottsMPNNPHEngine._run_whole_chain = originals["whole_chain"]

  arrays: dict[str, np.ndarray] = {}
  for key, value in ctx_info.items():
    if key not in ("idx_to_token", "etab_dtype"):
      arrays[f"ctx_{key}"] = np.asarray(value)
  meta_calls = []
  for call in calls:
    i = call["index"]
    dtype_np = call["pick"][0][0].dtype if call["pick"] else (call["mult"][0][0].dtype if call["mult"] else None)
    arrays[f"c{i}_S_in"], arrays[f"c{i}_S_out"], arrays[f"c{i}_valid"] = call["S_in"], call["S_out"], call["valid"]
    if call["pick"]:
      arrays[f"c{i}_pick_score"] = np.stack([p[0] for p in call["pick"]])
      arrays[f"c{i}_pick_T"] = np.asarray([p[1] for p in call["pick"]], dtype=np.float64)
      arrays[f"c{i}_pick_result"] = np.asarray([p[2] for p in call["pick"]], dtype=np.int64)
    if call["mult"]:
      arrays[f"c{i}_mult_probs"] = np.stack([m[0] for m in call["mult"]])
      arrays[f"c{i}_mult_u"] = np.asarray([m[1] for m in call["mult"]], dtype=np.float64)
      arrays[f"c{i}_mult_choice"] = np.asarray([m[2] for m in call["mult"]], dtype=np.int64)
      arrays[f"c{i}_mult_margin"] = np.asarray([m[3] for m in call["mult"]], dtype=np.float64)
    if call["randint"]:
      arrays[f"c{i}_randint_n"] = np.asarray([r[0] for r in call["randint"]], dtype=np.int64)
      arrays[f"c{i}_randint_val"] = np.asarray([r[1] for r in call["randint"]], dtype=np.int64)
    if call["perm"]:
      arrays[f"c{i}_perm"] = np.concatenate(call["perm"])
      arrays[f"c{i}_perm_len"] = np.asarray([len(p) for p in call["perm"]], dtype=np.int64)
    if "free_mask" in call:
      arrays[f"c{i}_free_mask"] = call["free_mask"]
    meta_calls.append({
      k: v for k, v in call.items()
      if k not in ("mult", "randint", "perm", "pick", "S_in", "S_out", "valid", "free_mask")
    } | {"n_mult": len(call["mult"]), "n_randint": len(call["randint"]), "n_perm": len(call["perm"]),
         "n_pick": len(call["pick"]), "score_dtype": None if dtype_np is None else str(dtype_np)})  # fmt: skip
  arrays["design_final_potts_energy"] = np.asarray([d["final_potts_energy"] for d in designs], dtype=np.float64)
  arrays["design_selective_energy"] = np.asarray(
    [np.nan if d["selective_energy"] is None else d["selective_energy"] for d in designs], dtype=np.float64
  )
  return arrays, {
    "calls": meta_calls, "designs": designs, "placements": placements, "n_returned": len(design_set),
    "model_multinomial_calls": state["model_multinomial"][0],
    "idx_to_token": ctx_info.get("idx_to_token"), "etab_dtype": ctx_info.get("etab_dtype"),
  }  # fmt: skip


def _load(path: Path) -> dict[str, np.ndarray]:
  with np.load(path) as z:
    return {key: z[key] for key in z.files}


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit_inputs(args: argparse.Namespace, cell: str, precision: str) -> dict[str, str]:
  pdb, chain = args.cells[cell]
  return {
    "script_sha256": _sha(Path(__file__).resolve()), "checkpoint_sha256": args.checkpoint_sha, "pdb_sha256": _sha(pdb),
    "chain": chain, "criteria_sha256": hashlib.sha256(json.dumps(_jsonable(_criteria()), sort_keys=True).encode()).hexdigest(),
    "torch_version": torch.__version__, "cell": cell, "precision": precision,
  }  # fmt: skip


def _run_one_unit(args: argparse.Namespace, cell: str, precision: str, out: Path) -> None:
  """Run one unit in THIS process and write its artifacts and completion stamp into ``out``."""
  import contextlib

  import mpnn.inference_engines.potts_mpnn_ph as ph
  import mpnn.model.mpnn as mm
  import mpnn.pipelines.potts_mpnn as pipelines
  import mpnn.potts_inference as inference
  from dump_protonpotts_decoder import _f64_changes
  from mpnn.utils.inference import MPNNInferenceInput

  dtype = dict(PRECISIONS)[precision]
  pipelines.get_protonation_state_transforms = lambda **_: []
  inference._PIPELINE_CACHE.clear()  # noqa: SLF001 -- trap 1 of the P4 dumper (spec §23.4)
  torch.set_num_threads(1)
  out.mkdir(parents=True, exist_ok=True)
  pdb, chain = args.cells[cell]
  atom_array = MPNNInferenceInput.from_atom_array_and_dict(input_dict={"structure_path": str(pdb)}).atom_array
  # The default dtype must be the run's BEFORE the engine is built: upstream creates its RBF centres with torch.linspace at the default
  # dtype, and a float64 run built at float32 carries ~3e-6 of rounding in the edge features (P4e, dump_protonpotts_encoder.py:96-106).
  previous_default = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    engine = ph.PottsMPNNPHEngine(
      checkpoint_path=str(args.checkpoint), extended_vocab="v6", out_directory=None, write_fasta=False, write_structures=False,
    )  # fmt: skip
    built_in = str(torch.get_default_dtype())
    guard = contextlib.nullcontext()
    if dtype == torch.float64:
      engine.model = engine.model.double()
      guard = _f64_changes(mm, engine.model.graph_featurization_module.positional_embedding.embed_positional_features)
    # The engine prepares its own float32 input features. The P4e dump cast every floating feature to double for its float64 run
    # (dump_protonpotts_encoder.py:118-121); without the same cast, distances and RBF features are computed from float32 inputs.
    cast_keys: set[str] = set()
    original_prepare = ph.prepare_potts_input

    def prepare_in_dtype(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
      batch = original_prepare(*args, **kwargs)
      features = batch["network_input"]["input_features"]
      for key, value in list(features.items()):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
          features[key] = value.to(dtype)
          cast_keys.add(key)
      return batch

    if dtype == torch.float64:
      ph.prepare_potts_input = prepare_in_dtype
    try:
      with guard:
        arrays, meta = _run_ph(engine, ph, atom_array, chain, dtype)
    finally:
      ph.prepare_potts_input = original_prepare
    meta["default_dtype_at_build"] = built_in
    meta["features_cast_to_double"] = sorted(cast_keys)
  finally:
    torch.set_default_dtype(previous_default)
  npz, js = out / f"{cell}_{precision}.npz", out / f"{cell}_{precision}.json"
  np.savez(npz, **arrays)
  js.write_text(json.dumps(meta, sort_keys=True, default=str), encoding="utf-8")
  stamp = {
    "inputs": _unit_inputs(args, cell, precision), "npz_sha256": _sha(npz), "json_sha256": _sha(js),
    "content_sha256": _content_hash(arrays), "meta_sha256": hashlib.sha256(js.read_bytes()).hexdigest(),
  }  # fmt: skip
  (out / f"{cell}_{precision}.stamp.json").write_text(json.dumps(stamp, sort_keys=True), encoding="utf-8")


def _run_ph(engine, ph, atom_array, chain: str, dtype):  # noqa: ANN001, ANN202
  return _run_unit(engine, ph, atom_array, chain, {}, dtype, _criteria())


def _stamp_ok(args: argparse.Namespace, directory: Path, cell: str, precision: str) -> dict | None:
  stamp_path = directory / f"{cell}_{precision}.stamp.json"
  npz, js = directory / f"{cell}_{precision}.npz", directory / f"{cell}_{precision}.json"
  if not (stamp_path.exists() and npz.exists() and js.exists()):
    return None
  stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
  if stamp["inputs"] != _unit_inputs(args, cell, precision) or stamp["npz_sha256"] != _sha(npz) or stamp["json_sha256"] != _sha(js):
    return None
  return stamp


def _child_command(args: argparse.Namespace, cell: str, precision: str, out: Path) -> list[str]:
  command = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", str(args.checkpoint), "--encoder-dir", str(args.encoder_dir),
             "--features-dir", str(args.features_dir), "--out", str(out), "--unit", f"{cell}:{precision}"]  # fmt: skip
  for name, (pdb, chain) in args.cells.items():
    command += ["--cell", f"{name}={pdb},{chain}"]
  return command


def _dump(args: argparse.Namespace, directory: Path) -> tuple[dict[str, dict], list[dict]]:
  """Every unit, reusing verified-complete ones. Returns (stamps by unit, reuse record)."""
  directory.mkdir(parents=True, exist_ok=True)
  stamps: dict[str, dict] = {}
  reuse: list[dict] = []
  for cell in args.cells:
    for precision, _ in PRECISIONS:
      key = f"{cell}/{precision}"
      stamp = _stamp_ok(args, directory, cell, precision)
      if stamp is not None:
        reuse.append({"unit": key, "reused": True, "content_sha256": stamp["content_sha256"]})
        logger.info("unit %s reused (stamp matches)", key)
      else:
        logger.info("unit %s running", key)
        subprocess.run(_child_command(args, cell, precision, directory), check=True, timeout=UNIT_TIMEOUT_S)  # noqa: S603
        stamp = _stamp_ok(args, directory, cell, precision)
        if stamp is None:
          msg = f"unit {key} finished without a valid stamp"
          raise RuntimeError(msg)
        reuse.append({"unit": key, "reused": False, "content_sha256": stamp["content_sha256"]})
      stamps[key] = stamp
  return stamps, reuse


def _reproduce(args: argparse.Namespace, stamps: dict[str, dict]) -> bool:
  cell, precision = REPRODUCE_UNIT
  if cell not in args.cells:
    return False
  with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(_child_command(args, cell, precision, Path(tmp)), check=True, timeout=UNIT_TIMEOUT_S)  # noqa: S603
    again = json.loads((Path(tmp) / f"{cell}_{precision}.stamp.json").read_text(encoding="utf-8"))
  first = stamps[f"{cell}/{precision}"]
  return again["content_sha256"] == first["content_sha256"] and again["meta_sha256"] == first["meta_sha256"]


def _multinomial_choice(probs: np.ndarray, u: float) -> int:
  cumulative = np.cumsum(probs.astype(np.float64))
  return min(int(np.searchsorted(cumulative, u * cumulative[-1], side="left")), len(cumulative) - 1)


def _grade_unit(arrs: dict[str, np.ndarray], meta: dict, precision: str) -> dict[str, bool]:  # noqa: C901, PLR0912
  """Claims 2-5 for one unit."""
  want = np.float64 if precision == "f64" else np.float32
  by_label: dict[str, list[dict]] = {}
  for call in meta["calls"]:
    by_label.setdefault(call["label"], []).append(call)
  designs_by_label = {d["label"] for d in meta["designs"]}
  complete = set(by_label) == set(LABELS) and designs_by_label == set(LABELS)
  double = True
  shim = True
  cover = True
  for call in meta["calls"]:
    i = call["index"]
    label, temperature = call["label"], call["temperature"]
    for key in (f"c{i}_pick_score", f"c{i}_mult_probs"):
      if key in arrs:
        double &= arrs[key].dtype == want
    n_mult, n_randint, n_pick, n_perm = call["n_mult"], call["n_randint"], call["n_pick"], call["n_perm"]
    if n_mult:
      probs, us, choices = arrs[f"c{i}_mult_probs"], arrs[f"c{i}_mult_u"], arrs[f"c{i}_mult_choice"]
      shim &= all(_multinomial_choice(probs[k], us[k]) == int(choices[k]) for k in range(n_mult))
    if label.startswith("ar_"):
      cover &= call["method"] == "infill" and n_pick == len(call["order"]) == n_mult and n_mult >= 1
    elif label == "gibbs":
      cover &= call["method"] == "gibbs" and n_perm >= 1 and n_mult >= 1
    elif label.startswith("two_phase"):
      cover &= call["method"] == "two_phase" and n_mult >= len(call["neigh"]) and n_pick >= len(call["neigh"])
    else:
      cover &= call["method"] == "converge" and n_randint >= 1
      cover &= (n_mult == 0) if temperature <= 0 else (n_mult == n_randint)
    if n_pick:
      scores, temps, results = arrs[f"c{i}_pick_score"], arrs[f"c{i}_pick_T"], arrs[f"c{i}_pick_result"]
      stochastic = temps > 0
      cover &= int(stochastic.sum()) == n_mult
      if n_mult:
        cover &= bool(np.array_equal(results[stochastic], arrs[f"c{i}_mult_choice"]))
        shifted = -scores[stochastic].astype(np.float64) / temps[stochastic][:, None]
        soft = np.exp(shifted - shifted.max(axis=1, keepdims=True))
        soft /= soft.sum(axis=1, keepdims=True)
        cover &= bool(np.abs(soft - arrs[f"c{i}_mult_probs"].astype(np.float64)).max() < 1e-4)
      if (~stochastic).any():
        cover &= bool(np.array_equal(results[~stochastic], scores[~stochastic].argmin(axis=1)))
  return {"complete": bool(complete), "double": bool(double), "shim": bool(shim), "cover": bool(cover)}


def main() -> int:  # noqa: C901, PLR0915
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--encoder-dir", type=Path, required=True)
  parser.add_argument("--features-dir", type=Path, required=True)
  parser.add_argument("--cell", action="append", default=[], metavar="NAME=PDB,CHAIN")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--unit", default=None, metavar="CELL:PRECISION")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  args.cells = {}
  for item in args.cell:
    name, _, rest = item.partition("=")
    pdb, _, chain = rest.rpartition(",")
    args.cells[name] = (Path(pdb), chain)
  args.checkpoint_sha = _sha(args.checkpoint)

  if args.unit is not None:
    cell, _, precision = args.unit.partition(":")
    _run_one_unit(args, cell, precision, args.out)
    return 0

  directory = args.out / "protonpotts_v6_methods"
  stamps, reuse = _dump(args, directory)
  reproducible = _reproduce(args, stamps)
  engine_matches_sealed = complete = f64_ok = shim_ok = cover_ok = precisions_differ = built_ok = True
  agreement: dict[str, float] = {}
  margins: dict[str, float] = {}
  for cell in args.cells:
    sealed = _load(args.features_dir / FEATURES[cell] / f"{cell}.npz")
    tables = _load(args.encoder_dir / "protonpotts_v6_encoder" / f"{cell}_f32.npz")
    units = {}
    for precision, _ in PRECISIONS:
      arrs = _load(directory / f"{cell}_{precision}.npz")
      meta = json.loads((directory / f"{cell}_{precision}.json").read_text(encoding="utf-8"))
      units[precision] = (arrs, meta)
      engine_matches_sealed &= bool(
        arrs["ctx_S_native"].shape == sealed["S"][0].shape and np.array_equal(arrs["ctx_S_native"], sealed["S"][0])
        and np.array_equal(arrs["ctx_eidx"], tables["E_idx"])
      )
      built_ok &= meta.get("default_dtype_at_build") == ("torch.float64" if precision == "f64" else "torch.float32")
      built_ok &= bool(meta.get("features_cast_to_double")) == (precision == "f64")
      flags = _grade_unit(arrs, meta, precision)
      complete &= flags["complete"]
      f64_ok &= flags["double"]
      shim_ok &= flags["shim"]
      cover_ok &= flags["cover"]
      margin = [float(arrs[f"c{c['index']}_mult_margin"].min()) for c in meta["calls"] if c["n_mult"]]
      margins[f"{cell}/{precision}"] = min(margin) if margin else float("nan")
    e32, e64 = units["f32"][0]["design_final_potts_energy"], units["f64"][0]["design_final_potts_energy"]
    precisions_differ &= bool(e32.shape != e64.shape or np.abs(e32 - e64).max() > 0.0)
    n_calls = min(len(units["f32"][1]["calls"]), len(units["f64"][1]["calls"]))
    same = [bool(np.array_equal(units["f32"][0][f"c{i}_S_out"], units["f64"][0][f"c{i}_S_out"])) for i in range(n_calls)]
    agreement[cell] = float(np.mean(same)) if same else float("nan")
  flags = {
    "engine_matches_sealed": bool(engine_matches_sealed), "complete": bool(complete), "f64_is_float64": bool(f64_ok), "engine_built_in_dtype": bool(built_ok),
    "precisions_differ": bool(precisions_differ), "shim_is_the_draw": bool(shim_ok), "draws_cover_methods": bool(cover_ok),
    "reproducible": bool(reproducible), "n_cells": len(args.cells), "n_units": len(stamps),
  }  # fmt: skip
  (args.out / "methods_manifest.json").write_text(
    json.dumps(
      {"checkpoint_sha256": args.checkpoint_sha, "torch_version": torch.__version__, "seed": SEED,
       "cells": {n: [str(p), c] for n, (p, c) in args.cells.items()}, "units": stamps, "reuse": reuse,
       "f32_vs_f64_same_final_tokens": agreement, "min_draw_margin": margins, "flags": flags, "labels": list(LABELS)},
      indent=2, sort_keys=True),
    encoding="utf-8",
  )  # fmt: skip
  logger.info("flags %s agreement %s", flags, agreement)
  results = os.environ.get("BTH_RESULTS_PATH")
  if results:
    Path(results).write_text(json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8")
  else:
    logger.warning("BTH_RESULTS_PATH unset: this run will record outcome 'unknown'")
  return 0


if __name__ == "__main__":
  sys.exit(main())
