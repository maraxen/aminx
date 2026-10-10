"""The ``protonpotts_ph_methods`` wave: aminx's host-loop pH methods against the real engine, from the P4j dump, per step.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §56; debts #2616, #2617. Sidecar: ``protonpotts_ph_methods_parity.bth.toml``
(committed before this script). Run where aminx is importable::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/parity/protonpotts_ph_methods_parity.py \\
        --state-npz <v6_state.npz> --methods-dir <dir holding protonpotts_v6_methods/ and methods_manifest.json> \\
        --features-dir ~/projects/aminx-oracles-protonpotts

WHAT IS COMPARED, per cell (2), precision (f64, f32) and call (11 labels): the placement (pins, designable set, every binder position's score);
a FORCED replay (the dump's choices, randint values, permutations and visiting order fed in place of its draws): every multinomial event's
probabilities, the final sequence (exact), the draw counts (exact), the z-scales, the final energies; a FREE replay (the dump's uniforms):
every non-fragile choice and the final sequence; and that non-designed positions hold. Tolerances, the fragile-draw rule and the twenty
controls are fixed in the sidecar.

The input check ``valid_mask_matches`` (aminx's ``valid_token_mask`` against the mask the engine used) and ``eidx_matches`` (aminx's neighbour
index against the engine's) are folded into ``placement_exact``: they are inputs of the plan. This is stricter than the sidecar's text, not looser.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import logging
import os
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from aminx.families.potts_mpnn.etab import potts_energy  # noqa: E402
from aminx.families.potts_mpnn.model import cast_floating  # noqa: E402
from aminx.families.protonpotts_mpnn import convert, ph_methods, ph_plan  # noqa: E402
from aminx.families.protonpotts_mpnn.energy import protonpotts_table  # noqa: E402
from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig  # noqa: E402
from aminx.families.protonpotts_mpnn.ph_methods import (  # noqa: E402
  DecoderField,
  Draws,
  MethodContext,
  ReplayStop,
  converge,
  gibbs,
  masked_infill,
  plan_for_methods,
  selective_energy_sum,
  two_phase,
)
from aminx.families.protonpotts_mpnn.ph_plan import Pin, valid_token_mask  # noqa: E402
from aminx.families.protonpotts_mpnn.vocab import token_index  # noqa: E402

logger = logging.getLogger("protonpotts_ph_methods_parity")

CELLS = ("multichain", "pkad_unlabelled")
FEATURES = "features_v6/protonpotts_v6_features"
CONTROL_CELL = "pkad_unlabelled"
F64_TOL = 1e-9
F32_FACTOR = 10.0
F32_MIN_BAND = 1e-6
FRAGILE_MARGIN = 1e-5
V = 30
TAKE = convert.token_permutation()  # TAKE[k] = the upstream index of the token at aminx index k
INV = np.argsort(TAKE)  # INV[u] = the aminx index of upstream token u
HIS_P = token_index("HIS-P")
LABELS = (
  "ar_potts", "ar_decoder", "mcmc_potts", "mcmc_potts_T0", "mcmc_potts_nonsel", "mcmc_mpnn", "combined_potts",
  "two_phase_potts", "two_phase_mpnn", "place_mpnn", "gibbs",
)  # fmt: skip
COMMON = {
  "center_types": ("HIS-P",), "neighbour_k": 8, "max_mutations": 8, "temperature": 0.1, "samples_per_site": 1, "cv_patience": 1,
  "cv_max": 2, "combined_lambda": 0.3, "block_size": 3, "record_trajectory": False,
}  # fmt: skip
OVERRIDES: dict[str, dict] = {
  "ar_potts": {"method": "autoregressive", "backend": "mpnn", "selective_source": "potts"},
  "ar_decoder": {"method": "autoregressive", "backend": "mpnn", "selective_source": "decoder"},
  "mcmc_potts": {"method": "converged_mcmc", "backend": "potts"},
  "mcmc_potts_T0": {"method": "converged_mcmc", "backend": "potts", "temperature": 0.0},
  "mcmc_potts_nonsel": {"method": "converged_mcmc", "backend": "potts", "selective": False},
  "mcmc_mpnn": {"method": "converged_mcmc", "backend": "mpnn", "cv_max": 1},
  "combined_potts": {"method": "converged_mcmc_combined", "backend": "potts"},
  "two_phase_potts": {"method": "two_phase", "backend": "potts", "two_phase_frac": 0.5},
  "two_phase_mpnn": {"method": "two_phase", "backend": "mpnn", "two_phase_frac": 0.5, "cv_max": 1},
  "place_mpnn": {"method": "converged_mcmc", "backend": "potts", "placement_by": "scan_mpnn"},
  "gibbs": {"method": "gibbs", "backend": "potts", "samples_per_site": 2},
}  # fmt: skip


def _load_converter() -> Callable:
  path = Path(__file__).resolve().parent / "convert_protonpotts_checkpoint.py"
  spec = importlib.util.spec_from_file_location("convert_protonpotts_checkpoint", path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.convert_state


def _npz(path: Path) -> dict[str, np.ndarray]:
  with np.load(path) as z:
    return {key: z[key] for key in z.files}


def label_config(label: str) -> PHDesignConfig:
  return PHDesignConfig(**{**COMMON, **OVERRIDES[label]})


def band(floor: float | None) -> float:
  return F64_TOL if floor is None else max(F32_FACTOR * floor, F32_MIN_BAND)


def rel(a: float, b: float) -> float:
  return abs(a - b) / max(abs(b), 1.0)


@dataclasses.dataclass
class Override:
  """A deliberate error fed to one comparison; every field's default is the faithful behaviour."""

  config: dict = dataclasses.field(default_factory=dict)
  cdf_none: bool = False
  uniforms_roll: int = 0
  randints_roll: int = 0
  perms_reverse: bool = False
  valid_all: bool = False
  pin_dep: int | None = None
  centre_in_designable: bool = False
  field_temperature: float | None = None
  nudge: float = 0.0


class Unit:
  """One cell in one precision: aminx's table and encoder states, and the dump's arrays and metadata for that precision."""

  def __init__(self, model, features: dict[str, np.ndarray], dtype, up: dict[str, np.ndarray], meta: dict) -> None:  # noqa: ANN001
    self.model, self.dtype, self.up, self.meta = model, dtype, up, meta
    coords = jnp.asarray(features["X"][0, :, :4, :], dtype=dtype)
    n = int(coords.shape[0])
    self.graph_args = (
      coords, jnp.ones(n, dtype=dtype), jnp.asarray(features["R_idx"][0]), jnp.asarray(features["chain_labels"][0]),
      jnp.ones(n, dtype=bool),
    )  # fmt: skip
    self.table, self.e_idx = protonpotts_table(cast_floating(model, dtype), *self.graph_args)
    self.e_idx_np = np.asarray(self.e_idx)
    self.native = INV[up["ctx_S_native"]].astype(np.int32)
    self.binder = up["ctx_chainA"].astype(bool)
    self.res_id = up["ctx_res_id"]
    self.valid = valid_token_mask(label_config("mcmc_potts").forbidden_tokens)
    self.eidx_matches = bool(np.array_equal(self.e_idx_np, up["ctx_eidx"]))
    self.ctx = self.context()
    self.calls = {c["label"]: c for c in meta["calls"]}

  def context(self, *, field_temperature: float | None = None, valid_all: bool = False) -> MethodContext:
    kwargs = {} if field_temperature is None else {"temperature": field_temperature}
    field = DecoderField.from_model(self.model, self.graph_args, **kwargs)
    valid = np.ones(V, dtype=bool) if valid_all else self.valid
    return MethodContext(self.table, self.e_idx, valid, INV, field)


def _draws(unit_up: dict, i: int, *, forced: bool, ov: Override, stop_after: int | None = None) -> Draws:
  u = unit_up.get(f"c{i}_mult_u", np.zeros(0))
  choices = INV[unit_up.get(f"c{i}_mult_choice", np.zeros(0, dtype=np.int64))]
  randints = unit_up.get(f"c{i}_randint_val", np.zeros(0, dtype=np.int64))
  perms: list[np.ndarray] = []
  if f"c{i}_perm" in unit_up:
    flat, lengths = unit_up[f"c{i}_perm"], unit_up[f"c{i}_perm_len"]
    perms = [flat[a:b] for a, b in zip(np.cumsum(lengths) - lengths, np.cumsum(lengths), strict=True)]
  if ov.uniforms_roll:
    u = np.roll(u, ov.uniforms_roll)
  if ov.randints_roll:
    randints = np.roll(randints, ov.randints_roll)
  if ov.perms_reverse:
    perms = [p[::-1].copy() for p in perms]
  return Draws(
    uniforms=u, randints=randints, perms=perms, choices=choices if forced else None, cdf_order=None if ov.cdf_none else INV,
    stop_after=stop_after,
  )  # fmt: skip


def _pins(call: dict, ov: Override) -> list[Pin]:
  pins = [
    Pin(p["position"], p["protonation_type"], int(INV[p["prot_idx"]]), tuple(int(INV[d]) for d in p["dep_idxs"]), p["res_id"])
    for p in call["pins"]
  ]  # fmt: skip
  if ov.pin_dep is not None:
    pins = [p._replace(dep_idxs=(ov.pin_dep,)) for p in pins]
  return pins


def run_method(unit: Unit, call: dict, up: dict, *, forced: bool, ov: Override, stop_after: int | None = None) -> dict:
  """One optimiser call on aminx: its sequences, events, counts and z-scales. ``error`` names a replay that ran out of draws."""
  i = call["index"]
  config = dataclasses.replace(label_config(call["label"]), **ov.config)
  ctx = unit.context(field_temperature=ov.field_temperature, valid_all=ov.valid_all)
  pins = _pins(call, ov)
  s_in = INV[up[f"c{i}_S_in"]].astype(np.int32)
  neigh = list(call["neigh"]) + ([p.position for p in pins] if ov.centre_in_designable else [])
  draws = _draws(up, i, forced=forced, ov=ov, stop_after=stop_after)
  zscales = None
  try:
    if call["method"] == "infill":
      result = masked_infill(ctx, config, pins, list(call["order"]), s_in, draws)
      seqs, zscales = [result.sequence], result.zscales
    elif call["method"] == "converge":
      seqs = [converge(ctx, config, pins, neigh, s_in, draws).sequence]
    elif call["method"] == "two_phase":
      seqs = [two_phase(ctx, config, pins, neigh, s_in, draws).sequence]
    else:
      seqs = gibbs(ctx, config, s_in, up[f"c{i}_free_mask"].astype(bool), draws)
  except ReplayStop:
    return {"error": None, "stopped": True, "seqs": None, "zscales": None, "choices": [e.choice for e in draws.events]}
  except (IndexError, ValueError) as exc:
    return {"error": str(exc), "events": [e.probs for e in draws.events], "choices": [e.choice for e in draws.events]}
  return {
    "error": None, "seqs": seqs, "zscales": zscales, "ctx": ctx, "pins": pins, "config": config,
    "probs": np.stack([e.probs for e in draws.events]).astype(np.float64) if draws.events else np.zeros((0, V)),
    "choices": [e.choice for e in draws.events],
    "counts": {"events": len(draws.events), "randint": draws._used["randint"], "perm": draws._used["perm"]},  # noqa: SLF001
  }  # fmt: skip


def placement(unit: Unit, call: dict, ov: Override) -> dict:
  """Plan (pins, designable) and every binder position's placement score against the dump's ranked list."""
  if call["method"] == "gibbs":
    return {"applies": False, "scores": None, "ok": True, "diff": 0.0}
  config = dataclasses.replace(label_config(call["label"]), **ov.config)
  ctx = unit.context(field_temperature=ov.field_temperature, valid_all=ov.valid_all)
  plan = plan_for_methods(ctx, unit.native, unit.binder, unit.res_id, unit.e_idx_np, config)
  want_pins = _pins(call, Override())
  pins_ok = plan is not None and [(p.position, p.protonation_type, p.prot_idx) for p in plan.pins] == [
    (p.position, p.protonation_type, p.prot_idx) for p in want_pins
  ]
  designable_ok = plan is not None and list(plan.designable) == sorted(call["neigh"])
  field, base = ph_methods.placement_field(ctx, unit.native, unit.binder, config.placement_by)
  pin = want_pins[0]
  scores = ph_plan.placement_scores(field, pin.prot_idx, pin.dep_idxs, unit.binder, selective=config.selective, base_sequence=base)
  entry = next(p for p in unit.meta["placements"] if p["label"] == call["label"])
  want = {int(pos): float(s) for pos, s in entry["ranked"]}
  positions_ok = set(np.flatnonzero(np.isfinite(scores)).tolist()) == set(want)
  diff = max((abs(float(scores[pos]) - s) for pos, s in want.items() if np.isfinite(scores[pos])), default=0.0)
  return {
    "applies": True, "pins_ok": bool(pins_ok), "designable_ok": bool(designable_ok), "positions_ok": bool(positions_ok),
    "scores": np.asarray([scores[pos] for pos in sorted(want)], dtype=np.float64), "diff": float(diff),
    "ok": bool(pins_ok and designable_ok and positions_ok),
  }  # fmt: skip


def measure(unit: Unit, call: dict, up: dict, ov: Override, *, free: bool = True) -> dict:
  """Everything graded for one call except the bands: the raw differences from the dump, and aminx's own outputs (for the f32 floors)."""
  i = call["index"]
  out: dict = {"label": call["label"], "index": i}
  place = placement(unit, call, ov)
  out["placement"] = place
  forced = run_method(unit, call, up, forced=True, ov=ov)
  want_probs = up.get(f"c{i}_mult_probs", np.zeros((0, V)))[..., TAKE].astype(np.float64)
  want_seq = INV[up[f"c{i}_S_out"]]
  if forced["error"] is not None:
    out.update(forced_error=forced["error"], probs_diff=float("inf"), counts_ok=False, forced_seq_ok=False)
    return out
  probs = forced["probs"]
  if ov.nudge and probs.size:
    probs = probs.copy()
    probs.flat[0] += ov.nudge
  out["probs"] = probs
  out["probs_diff"] = float(np.abs(probs - want_probs).max()) if probs.shape == want_probs.shape and probs.size else (
    0.0 if probs.shape == want_probs.shape else float("inf")
  )
  want_counts = {"events": call["n_mult"], "randint": call["n_randint"], "perm": call["n_perm"]}
  out["counts_ok"] = forced["counts"] == want_counts
  seqs = forced["seqs"]
  out["forced_seq_ok"] = bool(np.array_equal(np.asarray(seqs), want_seq if want_seq.ndim == 2 else want_seq[None]))
  out["zscales"] = forced["zscales"]
  if (forced["zscales"] is None) != (call["zscales"] is None):
    out["zscales_diff"] = float("inf")  # one side computed z-scales and the other did not
  elif call["zscales"] is None:
    out["zscales_diff"] = 0.0
  else:
    out["zscales_diff"] = max(rel(float(a), float(b)) for a, b in zip(forced["zscales"], call["zscales"], strict=True))
  ctx, pins = forced["ctx"], forced["pins"]
  valid = jnp.ones(unit.native.shape[0], dtype=bool)
  designs = [d for d in unit.meta["designs"] if d["call_index"] == i]
  energies = []
  for seq in seqs:
    e = float(potts_energy(unit.table, unit.e_idx, valid, jnp.asarray(seq, dtype=jnp.int32)))
    s = selective_energy_sum(ctx, seq, pins) if pins else 0.0
    energies.append((e, s))
  out["energies"] = np.asarray(energies, dtype=np.float64)
  out["energies_diff"] = (
    max(max(rel(e, float(d["final_potts_energy"])), rel(s, float(d["selective_energy"] or 0.0))) for (e, s), d in zip(energies, designs, strict=True))
    if len(designs) == len(energies) else float("inf")
  )  # fmt: skip
  s_in = INV[up[f"c{i}_S_in"]]
  if call["method"] == "gibbs":
    outside = ~up[f"c{i}_free_mask"].astype(bool)
  else:
    outside = np.ones(unit.native.shape[0], dtype=bool)
    outside[list(call["neigh"])] = False
    outside[[p.position for p in pins]] = False
  starts = np.atleast_2d(s_in)
  out["hold_ok"] = all(bool(np.array_equal(seq[outside], starts[min(r, len(starts) - 1)][outside])) for r, seq in enumerate(seqs))
  # FREE replay: the dump's uniforms. Only the prefix before the first fragile event is compared, so the replay stops there.
  margins = up.get(f"c{i}_mult_margin", np.zeros(0))
  fragile = margins < FRAGILE_MARGIN
  out["n_fragile"] = int(fragile.sum())
  out["exempt"] = bool(fragile.any())
  want_choices = INV[up.get(f"c{i}_mult_choice", np.zeros(0, dtype=np.int64))]
  if not free:
    return out
  stop = int(np.argmax(fragile)) if fragile.any() else None
  if stop == 0:
    out.update(choices_ok=True, free_seq_ok=True)
    return out
  free = run_method(unit, call, up, forced=False, ov=ov, stop_after=stop)
  if free["error"] is not None:
    out["choices_ok"] = bool(fragile.any())
    out["free_seq_ok"] = bool(fragile.any())
  else:
    got = np.asarray(free["choices"], dtype=np.int64)
    limit = int(np.argmax(fragile)) if fragile.any() else len(want_choices)
    out["choices_ok"] = bool(np.array_equal(got[:limit], want_choices[:limit])) and (fragile.any() or len(got) == len(want_choices))
    out["free_seq_ok"] = bool(fragile.any() or np.array_equal(np.asarray(free["seqs"]), want_seq if want_seq.ndim == 2 else want_seq[None]))
  return out


def verdict(m: dict, floors: dict | None) -> dict:
  """Apply the bands to a measurement. ``floors`` is None in float64, else the per-array f32 spreads of this call."""
  f = floors or {}
  probs_band = band(f.get("probs") if floors else None)
  scores_band = band(f.get("scores") if floors else None)
  rel_band = band(f.get("rel") if floors else None)
  place = m["placement"]
  ratios = [m["probs_diff"] / probs_band, place["diff"] / scores_band, m.get("zscales_diff", 0.0) / rel_band, m.get("energies_diff", 0.0) / rel_band]
  checks = {
    "placement": bool(place["ok"] and place["diff"] <= scores_band),
    "probs": bool(m["probs_diff"] <= probs_band),
    "counts": bool(m.get("counts_ok", False)),
    "forced_seq": bool(m.get("forced_seq_ok", False)),
    "zscales": bool(m.get("zscales_diff", 0.0) <= rel_band),
    "energies": bool(m.get("energies_diff", float("inf")) <= rel_band),
    "hold": bool(m.get("hold_ok", False)),
    "choices": bool(m.get("choices_ok", False)),
    "free_seq": bool(m.get("free_seq_ok", False)),
  }  # fmt: skip
  return {"checks": checks, "ok": all(checks.values()), "worst_over_band": float(max(r for r in ratios if np.isfinite(r)) if any(np.isfinite(r) for r in ratios) else 0.0), **{k: m.get(k) for k in ("n_fragile", "exempt")}}


def f32_floors(m32: dict, m64_on_32: dict) -> dict:
  """aminx float32 against aminx float64 on the SAME forced states (the float32 dump's choices): the instrument's float32 noise."""
  if "probs" not in m32 or "probs" not in m64_on_32:
    return {"probs": None, "scores": None, "rel": None}
  probs = float(np.abs(m32["probs"] - m64_on_32["probs"]).max()) if m32["probs"].size and m32["probs"].shape == m64_on_32["probs"].shape else 0.0
  scores = 0.0
  if m32["placement"]["scores"] is not None and m64_on_32["placement"]["scores"] is not None:
    scores = float(np.abs(m32["placement"]["scores"] - m64_on_32["placement"]["scores"]).max())
  rels = [rel(float(a), float(b)) for a, b in zip(m32["energies"].ravel(), m64_on_32["energies"].ravel(), strict=True)]
  if m32["zscales"] is not None and m64_on_32["zscales"] is not None:
    rels += [rel(float(a), float(b)) for a, b in zip(m32["zscales"], m64_on_32["zscales"], strict=True)]
  return {"probs": probs, "scores": scores, "rel": float(max(rels, default=0.0))}


# ---- the twenty deliberate errors --------------------------------------------------------------------------------------------------


def _outgoing_only(table, e_idx, seq, positions):  # noqa: ANN001, ANN202
  from aminx.families.protonpotts_mpnn.ph_potentials import _outgoing  # noqa: PLC0415

  return _outgoing(table, e_idx, seq, positions)


def controls() -> dict[str, tuple[tuple[str, ...], Override, Callable[[ExitStack], None] | None]]:
  """name -> (labels graded under the error, the override, an optional patcher entering its mocks on an ExitStack)."""
  original_std = np.std
  original_scores = ph_plan.placement_scores

  def ddof1(stack: ExitStack) -> None:
    stack.enter_context(mock.patch.object(np, "std", lambda a, *args, **kw: original_std(a, *args, ddof=1, **kw)))

  def selective_formula(stack: ExitStack) -> None:
    def forced_selective(field, prot_idx, dep_idxs, mask, *, selective=True, base_sequence=None):  # noqa: ANN001, ANN202, ARG001
      return original_scores(field, prot_idx, dep_idxs, mask, selective=True, base_sequence=base_sequence)

    stack.enter_context(mock.patch.object(ph_plan, "placement_scores", forced_selective))

  def unmasked(stack: ExitStack) -> None:
    def scan(ctx, native, binder, placement_by):  # noqa: ANN001, ANN202
      del binder
      return ctx.need_decoder()(native), np.asarray(native, dtype=np.int32)

    stack.enter_context(mock.patch.object(ph_methods, "placement_field", scan))

  def outgoing(stack: ExitStack) -> None:
    stack.enter_context(mock.patch.object(ph_methods, "_GIBBS_ROWS", _outgoing_only))

  return {
    "cdf_in_aminx_order": (("ar_potts", "mcmc_potts", "two_phase_potts"), Override(cdf_none=True), None),
    "uniforms_shifted_one": (("ar_potts", "mcmc_potts"), Override(uniforms_roll=1), None),
    "randints_shifted_one": (("mcmc_potts", "two_phase_potts"), Override(randints_roll=1), None),
    "permutations_reversed": (("gibbs",), Override(perms_reverse=True), None),
    "field_untempered": (("ar_potts", "mcmc_mpnn"), Override(field_temperature=1.0), None),
    "zscales_sample_std": (("ar_potts",), Override(), ddof1),
    "temperature_doubled": (("ar_potts", "mcmc_potts"), Override(config={"temperature": 0.2}), None),
    "combined_lambda_wrong": (("combined_potts",), Override(config={"combined_lambda": 0.6}), None),
    "two_phase_frac_wrong": (("two_phase_potts",), Override(config={"two_phase_frac": 0.9}), None),
    "selective_flipped": (("mcmc_potts", "mcmc_potts_nonsel"), Override(), None),  # per-label flip, see run_control
    "backend_swapped": (("mcmc_potts", "mcmc_mpnn"), Override(), None),  # per-label swap, see run_control
    "selective_source_swapped": (("ar_potts", "ar_decoder"), Override(), None),  # per-label swap, see run_control
    "valid_mask_all_true": (("ar_potts", "mcmc_potts"), Override(valid_all=True), None),
    "pin_contrast_wrong": (("ar_potts", "mcmc_potts"), Override(pin_dep=HIS_P), None),
    "centre_among_designable": (("mcmc_potts",), Override(centre_in_designable=True), None),
    "scan_unmasked": (("place_mpnn",), Override(), unmasked),
    "scan_selective_formula": (("mcmc_potts_nonsel",), Override(), selective_formula),
    "gibbs_outgoing_only": (("gibbs",), Override(), outgoing),
    "cv_cap_wrong": (("mcmc_potts",), Override(config={"cv_max": 1}), None),
    "nudged_4x_band": (("mcmc_potts",), Override(nudge=4.0 * F64_TOL), None),
  }  # fmt: skip


SWAPS = {
  "selective_flipped": lambda cfg: {"selective": not cfg.selective},
  "backend_swapped": lambda cfg: {"backend": "mpnn" if cfg.backend == "potts" else "potts"},
  "selective_source_swapped": lambda cfg: {"selective_source": "decoder" if cfg.selective_source == "potts" else "potts"},
}


def run_control(name: str, labels: tuple[str, ...], ov: Override, patcher: Callable | None, unit: Unit) -> bool:
  """True when the error is REJECTED: some graded call in its scope fails the (float64) comparison."""
  rejected = False
  for label in labels:
    call = unit.calls[label]
    override = ov
    if name in SWAPS:
      override = dataclasses.replace(ov, config=SWAPS[name](label_config(label)))
    with ExitStack() as stack:
      if patcher is not None:
        patcher(stack)
      m = measure(unit, call, unit.up, override)
    v = verdict(m, None)
    logger.info("control %-26s %-18s ok=%s %s", name, label, v["ok"], {k: x for k, x in v["checks"].items() if not x})
    rejected |= not v["ok"]
  return rejected


def main() -> int:  # noqa: C901, PLR0915
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--state-npz", type=Path, required=True)
  parser.add_argument("--methods-dir", type=Path, required=True)
  parser.add_argument("--features-dir", type=Path, required=True)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

  state_meta = json.loads(args.state_npz.with_suffix(".json").read_text(encoding="utf-8"))
  manifest = json.loads((args.methods_dir / "methods_manifest.json").read_text(encoding="utf-8"))
  checkpoint_matches = state_meta["checkpoint_sha256"] == manifest["checkpoint_sha256"] and list(manifest["labels"]) == list(LABELS)
  model = _load_converter()(_npz(args.state_npz))

  per_call: dict[str, dict] = {}
  flags = {k: True for k in ("f64_in_band", "f32_in_band", "placement_exact", "forced_sequences_exact", "draw_counts_exact", "choices_exact",
                             "sequences_exact", "nondesigned_hold", "energies_in_band", "zscales_in_band")}  # fmt: skip
  n_fragile = n_exempt = n_calls = 0
  worst = 0.0
  units: dict[tuple[str, str], Unit] = {}
  directory = args.methods_dir / "protonpotts_v6_methods"
  for cell in CELLS:
    features = _npz(args.features_dir / FEATURES / f"{cell}.npz")
    for precision, dtype in (("f64", jnp.float64), ("f32", jnp.float32)):
      up = _npz(directory / f"{cell}_{precision}.npz")
      meta = json.loads((directory / f"{cell}_{precision}.json").read_text(encoding="utf-8"))
      units[(cell, precision)] = Unit(model, features, dtype, up, meta)
    inputs_ok = all(units[(cell, p)].eidx_matches for p in ("f64", "f32"))
    flags["placement_exact"] &= inputs_ok
    for label in LABELS:
      u64, u32 = units[(cell, "f64")], units[(cell, "f32")]
      m64 = measure(u64, u64.calls[label], u64.up, Override())
      v64 = verdict(m64, None)
      m32 = measure(u32, u32.calls[label], u32.up, Override())
      m64_on_32 = measure(u64, u32.calls[label], u32.up, Override(), free=False)
      floors = f32_floors(m32, m64_on_32)
      v32 = verdict(m32, floors)
      for precision, v in (("f64", v64), ("f32", v32)):
        c = v["checks"]
        n_calls += 1
        n_fragile += int(v["n_fragile"] or 0)
        n_exempt += int(bool(v["exempt"]))
        worst = max(worst, v["worst_over_band"])
        flags["f64_in_band" if precision == "f64" else "f32_in_band"] &= c["probs"]
        flags["placement_exact"] &= c["placement"]
        flags["forced_sequences_exact"] &= c["forced_seq"]
        flags["draw_counts_exact"] &= c["counts"]
        flags["choices_exact"] &= c["choices"]
        flags["sequences_exact"] &= c["free_seq"]
        flags["nondesigned_hold"] &= c["hold"]
        flags["energies_in_band"] &= c["energies"]
        flags["zscales_in_band"] &= c["zscales"]
        per_call[f"{cell}/{precision}/{label}"] = {"ok": v["ok"], "checks": c, "worst_over_band": v["worst_over_band"],
                                                   "n_fragile": v["n_fragile"], "exempt": v["exempt"],
                                                   "probs_diff": m64["probs_diff"] if precision == "f64" else m32["probs_diff"],
                                                   "floors": floors if precision == "f32" else None}  # fmt: skip
        logger.info("%s/%s/%-18s ok=%s %s", cell, precision, label, v["ok"], {k: x for k, x in c.items() if not x})

  # ---- the twenty deliberate errors, on the control cell in float64 ---------------------------------------------------------
  control_unit = units[(CONTROL_CELL, "f64")]
  mutants: dict[str, str] = {}
  for name, (labels, ov, patcher) in controls().items():
    mutants[name] = "failed" if run_control(name, labels, ov, patcher, control_unit) else "passed"
    logger.info("mutant %-28s %s", name, "DETECTED" if mutants[name] == "failed" else "NOT DETECTED")

  clean_ok = all(flags.values()) and checkpoint_matches and n_calls == 44
  results = {
    "clean": "pass" if clean_ok else "fail", "n_cells": len(CELLS), "n_calls": n_calls, **{k: bool(v) for k, v in flags.items()},
    "checkpoint_matches": bool(checkpoint_matches), "n_fragile_events": int(n_fragile), "n_calls_exempt_from_free_run": int(n_exempt),
    "worst_over_band": float(worst), "n_listed": len(mutants), "n_failed": sum(v == "failed" for v in mutants.values()),
    "mutants": mutants, "undetected": sorted(k for k, v in mutants.items() if v == "passed"), "per_call": per_call,
  }  # fmt: skip
  path = os.environ.get("BTH_RESULTS_PATH")
  if not path:
    msg = "protonpotts_ph_methods_parity requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  Path(path).write_text(json.dumps(results, indent=2, default=float) + "\n", encoding="utf-8")
  logger.info("clean=%s", results["clean"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
