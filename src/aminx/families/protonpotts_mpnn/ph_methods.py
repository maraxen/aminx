"""The remaining pH-engine methods: ``autoregressive``, ``converged_mcmc`` (plain and combined), ``two_phase`` and ``gibbs`` (debts #2616, #2617).

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §56. Upstream is ``ProtonPottsMPNN/foundry/models/mpnn/src/mpnn/inference_engines/
potts_mpnn_ph.py`` (``UP:``) and ``model/pottsmpnn.py`` (``potts_gibbs_optimize``). Every function below is a host loop over primitives that
are already graded: the merged-table conditional energies (:func:`ph_potentials.candidate_energies_at`) and the teacher-forced decoder
(:func:`decode.teacher_forced`, pattern ``conditional_minus_self``). Each step is one jitted call, so none of this is under ``vmap``.

- ``_masked_infill`` (UP:1908-1938) -> :func:`masked_infill`. Locks every centre, sets the designable set to ``X``, then visits ``order`` and samples each
  position from ``softmax(-J/T)`` with ``T = max(temperature, 1e-3)``. ``selective_source='potts'``: ``J = (1-lam) nat/sdNat + lam sel/sdSel`` where
  ``nat`` is the decoder field ``-log p`` and ``sel`` the Potts centre gap, each scaled by the std of single-mutation deltas
  (:func:`sampler_zscales`). ``'decoder'``: ``J = -(p(a|centres on) - lam p(a|centres off))``, no Potts and no z-scaling.
- ``_converge`` (UP:1941-1957) -> :func:`converge`. A random neighbour per step (``randint``), one pick, stop after ``cv_patience*N`` unchanged steps or
  ``cv_max*N`` steps. ``_score_at`` (UP:1795) is :func:`score_at`.
- ``_two_phase`` (UP:1960-1982) -> :func:`two_phase`.
- ``PottsMPNN.potts_gibbs_optimize`` -> :func:`gibbs`.

Randomness. Every draw goes through a :class:`Draws`. Fresh (a seeded numpy stream) or replay (recorded uniforms, ``randint`` values and permutations,
in order). In FORCED mode the recorded CHOICES are used instead of the uniforms, so one near-tie cannot cascade and every step's probabilities are compared
at the recorded state (the wave's use). The inverse CDF accumulates, in float64, the probabilities in ``cdf_order`` (upstream's token order), as the
oracle's shim does. ``temperature <= 0`` picks the first minimum and draws nothing.

Not ported: ``sequence_decoded_prob_score`` and ``sequence_entropy`` of the scored design (aminx's :class:`PHDesign` has no such fields),
``placement_by='random'`` and multi-centre enumeration (``center_protonation_types``), ``seed_source='inverse'``, ``record_trajectory``.
"""

from __future__ import annotations

import dataclasses
import random
from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aminx.families.potts_mpnn.etab import potts_energy
from aminx.families.potts_mpnn.model import cast_floating
from aminx.families.protonpotts_mpnn.decode import teacher_forced
from aminx.families.protonpotts_mpnn.ph_design import PHDesign
from aminx.families.protonpotts_mpnn.ph_plan import (
  Pin,
  Plan,
  plan_from_center_types,
  plan_from_explicit_centers,
  valid_token_mask,
)
from aminx.families.protonpotts_mpnn.ph_potentials import candidate_energies, candidate_energies_at
from aminx.families.protonpotts_mpnn.ph_sample import (
  _JIT_STATES,
  UPSTREAM_SAMPLE_TEMPERATURE,
  _cdf_order,
)
from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6

if TYPE_CHECKING:
  from collections.abc import Callable, Sequence

  from aminx.families.potts_mpnn.model import PottsMPNN
  from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig

UNK_INDEX = PROTONPOTTS_V6.symbols.index("X")
MIN_TEMPERATURE = 1e-3  # UP:1927 (autoregressive) and potts_gibbs_optimize's caller (UP:1217)
STABILITY_EPS = 1e-6  # the z-scale floor, UP:1865-1866
MAX_GIBBS_SWEEPS = 1000  # UP:1217 max_iters

_CANDIDATES_AT = jax.jit(candidate_energies_at)
_CANDIDATES = jax.jit(candidate_energies)


class ReplayStop(Exception):
  """Raised by :class:`Draws` once ``stop_after`` draws were made: a replay that only needs a prefix of its draws."""


class PickEvent(NamedTuple):
  """One inverse-CDF draw: the probabilities it saw (aminx token order), the uniform used and the token chosen."""

  probs: np.ndarray
  uniform: float
  choice: int


class Draws:
  """The randomness of one optimiser call: fresh from a seed, replayed from records, or forced to recorded choices.

  ``uniforms`` / ``randints`` / ``perms`` are consumed in order, one array per kind. ``choices`` (aminx tokens), when given, replaces every
  multinomial draw by the next recorded choice (FORCED); ``uniforms`` is then only reported. ``events`` collects every multinomial draw.
  """

  def __init__(
    self,
    *,
    seed: int | None = None,
    uniforms: Sequence[float] | None = None,
    randints: Sequence[int] | None = None,
    perms: Sequence[Sequence[int]] | None = None,
    choices: Sequence[int] | None = None,
    cdf_order: np.ndarray | None = None,
    stop_after: int | None = None,
  ) -> None:
    if all(x is None for x in (seed, uniforms, choices, randints, perms)):
      msg = "Draws needs a seed (fresh) or recorded draws (uniforms, randints, perms or choices)"
      raise ValueError(msg)
    self.rng = np.random.default_rng(seed) if seed is not None else None
    self._uniforms = None if uniforms is None else [float(u) for u in uniforms]
    self._randints = None if randints is None else [int(r) for r in randints]
    self._perms = None if perms is None else [np.asarray(p, dtype=np.int64) for p in perms]
    self._choices = None if choices is None else [int(c) for c in choices]
    self._used = {"uniform": 0, "randint": 0, "perm": 0, "choice": 0}
    self.cdf_order = cdf_order
    self.stop_after = stop_after
    self.events: list[PickEvent] = []
    self.randint_log: list[int] = []
    self.perm_log: list[np.ndarray] = []

  def _next(self, kind: str, source: list | None, fresh: Callable[[], object]):
    if source is None:
      if self.rng is None:
        msg = f"no recorded {kind} and no seed"
        raise ValueError(msg)
      return fresh()
    index = self._used[kind]
    if index >= len(source):
      msg = f"replay ran out of {kind} draws after {index}"
      raise IndexError(msg)
    self._used[kind] = index + 1
    return source[index]

  def _check_stop(self) -> None:
    if self.stop_after is not None and len(self.events) >= self.stop_after:
      raise ReplayStop

  def randint(self, high: int) -> int:
    """Index into a list of ``high`` neighbours (upstream ``torch.randint(len(neigh), (1,))``)."""
    value = self._next("randint", self._randints, lambda: int(self.rng.integers(high)))
    if not 0 <= int(value) < high:
      msg = f"recorded randint {value} outside [0, {high})"
      raise ValueError(msg)
    self.randint_log.append(int(value))
    return int(value)

  def permutation(self, size: int) -> np.ndarray:
    """A permutation of ``range(size)`` (upstream ``torch.randperm``)."""
    perm = self._next("perm", self._perms, lambda: self.rng.permutation(size).astype(np.int64))
    if len(perm) != size:
      msg = f"recorded permutation has length {len(perm)}, expected {size}"
      raise ValueError(msg)
    self.perm_log.append(np.asarray(perm))
    return np.asarray(perm)

  def pick_probs(self, probs: np.ndarray) -> int:
    """One draw from ``probs`` (aminx token order): the inverse CDF in ``cdf_order``, or the next forced choice."""
    if self._choices is not None:
      choice = self._next("choice", self._choices, lambda: 0)
      u = (
        self._next("uniform", self._uniforms, lambda: 0.0)
        if self._uniforms is not None
        else float("nan")
      )
      self.events.append(PickEvent(np.array(probs, copy=True), float(u), int(choice)))
      self._check_stop()
      return int(choice)
    u = float(self._next("uniform", self._uniforms, lambda: float(self.rng.random())))
    order = np.arange(probs.shape[0]) if self.cdf_order is None else np.asarray(self.cdf_order)
    cumulative = np.cumsum(np.asarray(probs)[order].astype(np.float64))
    position = min(
      int(np.searchsorted(cumulative, u * cumulative[-1], side="left")), len(cumulative) - 1,
    )
    choice = int(order[position])
    self.events.append(PickEvent(np.array(probs, copy=True), u, choice))
    self._check_stop()
    return choice


def softmax_probs(score: np.ndarray, temperature: float) -> np.ndarray:
  """``softmax(-score / T)`` in the score's dtype (upstream ``F.softmax(-score / temperature)``); ``+inf`` scores get probability 0."""
  z = (-score) / score.dtype.type(temperature)
  z = z - z.max()
  p = np.exp(z)
  return p / p.sum()


def pick(score: np.ndarray, temperature: float, draws: Draws) -> int:
  """Upstream ``_pick`` (UP:1774): the first argmin at ``temperature <= 0``, else one multinomial draw from ``softmax(-score/T)``."""
  if temperature <= 0:
    return int(np.argmin(score))
  return draws.pick_probs(softmax_probs(score, temperature))


class DecoderField:
  """Upstream ``ctx.field_mpnn``: ``-log_probs`` of the teacher-forced decoder, ``conditional_minus_self`` (UP:1157-1165).

  The prepared input's temperature (float32 0.1, as for ``mpnn_sample``) and a zero bias, so the field is the TEMPERED log-probability.
  ``__call__(seq)`` returns the full ``(L, V)`` field of ``seq`` (aminx tokens). Tokens may include ``X``.
  """

  def __init__(
    self, scored_mpnn, h_v, h_e, e_idx, present, temperature: float = UPSTREAM_SAMPLE_TEMPERATURE,
  ) -> None:
    dtype = h_v.dtype
    self.dtype = np.dtype(dtype)
    length = h_v.shape[0]
    self._args = (
      scored_mpnn.decoder, scored_mpnn.w_s_embed, scored_mpnn.w_out, h_v, h_e, e_idx, present.astype(dtype),
    )  # fmt: skip
    self._temperature = jnp.full((length,), temperature, dtype=dtype)
    self._bias = jnp.zeros((length, PROTONPOTTS_V6.size), dtype=dtype)
    self._order = jnp.arange(length, dtype=jnp.int32)
    self.n_calls = 0

  @staticmethod
  def from_model(
    model: PottsMPNN, graph_args: Sequence[jax.Array], temperature: float = UPSTREAM_SAMPLE_TEMPERATURE,
  ) -> DecoderField:  # fmt: skip
    """Build from the model and the driver's ``(coords, present, residue_idx, chain_index, pad_valid)``."""
    coords, present, residue_idx, chain_index, _pad_valid = graph_args
    h_v, h_e, nbr = _JIT_STATES(model, coords, present, residue_idx, chain_index)
    scored = cast_floating(model, h_v.dtype)
    return DecoderField(scored.mpnn, h_v, h_e, nbr, present, temperature)

  def __call__(self, seq: np.ndarray) -> np.ndarray:
    self.n_calls += 1
    out = _field(
      *self._args, jnp.asarray(seq, dtype=jnp.int32), self._temperature, self._bias, self._order,
    )
    return np.asarray(out)


@eqx.filter_jit
def _field(decoder, w_s_embed, w_out, h_v, h_e, e_idx, present, seq, temperature, bias, order):
  _logits, log_probs = teacher_forced(
    decoder, w_s_embed, w_out, h_v, h_e, e_idx, present, seq, temperature, bias,
    pattern="conditional_minus_self", decoding_order=order,
  )  # fmt: skip
  return -log_probs


@dataclasses.dataclass
class MethodContext:
  """What the methods read: the merged Potts table, the neighbour index, the valid-token mask and the optional decoder field."""

  table: jax.Array
  e_idx: jax.Array
  valid: np.ndarray  # (V,) bool, upstream ``valid_aa_mask``
  cdf_order: np.ndarray  # (V,) aminx index of the j-th upstream token
  field_mpnn: DecoderField | None = None

  def rows(self, seq: np.ndarray, positions: Sequence[int]) -> np.ndarray:
    """Conditional Potts energies ``(N, V)`` at ``positions`` of ``seq`` (upstream ``scorer.cond_energy_at``)."""
    out = _CANDIDATES_AT(
      self.table,
      self.e_idx,
      jnp.asarray(seq, dtype=jnp.int32),
      jnp.asarray(list(positions), dtype=jnp.int32),
    )
    return np.asarray(out)

  def field_at(self, backend: str) -> Callable[[np.ndarray, int], np.ndarray]:
    """``field_at(S, i)``, the ``(V,)`` energy row of position ``i`` (UP:1220-1232)."""
    if backend == "potts":
      return lambda seq, i: self.rows(seq, [i])[0]
    if self.field_mpnn is None:
      msg = "backend='mpnn' needs the decoder field (model and graph arguments)"
      raise ValueError(msg)
    return lambda seq, i: self.field_mpnn(seq)[i]

  def need_decoder(self) -> DecoderField:
    if self.field_mpnn is None:
      msg = "this method needs the decoder field (model and graph arguments)"
      raise ValueError(msg)
    return self.field_mpnn


class MethodResult(NamedTuple):
  sequence: np.ndarray  # (L,) aminx tokens
  events: tuple[PickEvent, ...]
  zscales: (
    tuple[float, float] | None
  )  # (sdNat, sdSel) of autoregressive with selective_source='potts'


def _gap(row: np.ndarray, pin: Pin) -> float:
  """``e_P - mean_d e_D`` of one centre, in python floats as upstream accumulates it (``_selective_energy_of``)."""
  e_p = float(row[pin.prot_idx])
  ed = 0.0
  for d in pin.dep_idxs:
    ed += float(row[d])
  return e_p - ed / len(pin.dep_idxs)


def selective_energy_sum(ctx: MethodContext, seq: np.ndarray, pins: Sequence[Pin]) -> float:
  """Sum over centres of ``e_P - mean_d e_D``, every centre locked protonated (UP:2550 ``_selective_energy_sum``)."""
  s = np.array(seq, dtype=np.int32, copy=True)
  for p in pins:
    s[p.position] = p.prot_idx
  rows = ctx.rows(s, [p.position for p in pins])
  return float(sum(_gap(rows[k], p) for k, p in enumerate(pins)))


def sampler_zscales(
  ctx: MethodContext, seq: np.ndarray, order: Sequence[int], pins: Sequence[Pin], valid_idx: Sequence[int],
) -> tuple[float, float]:  # fmt: skip
  """Std of single-mutation deltas of naturalness and selectivity over ``order`` (UP:1847-1866). Centres locked."""
  s = np.array(seq, dtype=np.int32, copy=True)
  for p in pins:
    s[p.position] = p.prot_idx
  nat_full = ctx.need_decoder()(s)
  sel0 = selective_energy_sum(ctx, s, pins)
  nat_d: list[float] = []
  sel_d: list[float] = []
  for j in order:
    cur = int(s[j])
    base = float(nat_full[j, cur])
    for a in valid_idx:
      nat_d.append(float(nat_full[j, a]) - base)
      s[j] = a
      sel_d.append(selective_energy_sum(ctx, s, pins) - sel0)
    s[j] = cur
  sd_nat = max(float(np.std(nat_d)) if nat_d else 1.0, STABILITY_EPS)
  sd_sel = max(float(np.std(sel_d)) if sel_d else 1.0, STABILITY_EPS)
  return sd_nat, sd_sel


def selective_row(
  ctx: MethodContext, valid_idx: Sequence[int], seq: np.ndarray, j: int, pins: Sequence[Pin], dtype,
) -> np.ndarray:
  """``(V,)`` centre-gap energy of each candidate at ``j`` (UP:1869); invalid tokens stay ``+inf``. ``seq[j]`` is restored."""
  out = np.full(PROTONPOTTS_V6.size, np.inf, dtype=dtype)
  cur = int(seq[j])
  for a in valid_idx:
    seq[j] = a
    out[a] = selective_energy_sum(ctx, seq, pins)
  seq[j] = cur
  return out


def selective_reward_decoder(
  ctx: MethodContext, valid_idx: Sequence[int], seq: np.ndarray, j: int, pins: Sequence[Pin], lam: float, dtype,
) -> np.ndarray:  # fmt: skip
  """Pure-decoder reward ``p(a | centres target) - lam p(a | centres off)`` at ``j`` (UP:1883-1905), higher is better; invalid ``-inf``."""
  field = ctx.need_decoder()
  p_target = np.exp(-field(seq)[j])
  n_off = max(len(p.dep_idxs) for p in pins)
  p_off = np.zeros(PROTONPOTTS_V6.size, dtype=p_target.dtype)
  saved = [int(seq[p.position]) for p in pins]
  for t in range(n_off):
    for p in pins:
      seq[p.position] = p.dep_idxs[min(t, len(p.dep_idxs) - 1)]
    p_off += np.exp(-field(seq)[j])
  for p, token in zip(pins, saved, strict=True):
    seq[p.position] = token
  reward = p_target - lam * (p_off / n_off)
  out = np.full(PROTONPOTTS_V6.size, -np.inf, dtype=dtype)
  out[list(valid_idx)] = reward[list(valid_idx)]
  return out


def masked_infill(
  ctx: MethodContext, config: PHDesignConfig, pins: Sequence[Pin], order: Sequence[int], initial: np.ndarray, draws: Draws,
) -> MethodResult:  # fmt: skip
  """Upstream ``_masked_infill``: lock the centres, set ``order`` to ``X``, then decode ``order`` one position at a time."""
  lam = float(config.combined_lambda)
  valid_idx = [int(a) for a in np.flatnonzero(ctx.valid)]
  seq = np.array(initial, dtype=np.int32, copy=True)
  for p in pins:
    seq[p.position] = p.prot_idx
  decoder_selective = config.selective_source == "decoder"
  zscales = None
  if not decoder_selective:
    zscales = sampler_zscales(ctx, seq, order, pins, valid_idx)
  for i in order:
    seq[i] = UNK_INDEX
  temperature = max(float(config.temperature), MIN_TEMPERATURE)
  field = ctx.need_decoder()
  for i in order:
    if decoder_selective:
      score = -selective_reward_decoder(ctx, valid_idx, seq, i, pins, lam, field.dtype)
    else:
      nat = field(seq)[i]
      sel = selective_row(ctx, valid_idx, seq, i, pins, nat.dtype)
      sd_nat, sd_sel = zscales
      score = (1.0 - lam) * (nat / sd_nat) + lam * (sel / sd_sel)
    score = np.where(ctx.valid, score, score.dtype.type(np.inf))
    seq[i] = pick(score, temperature, draws)
  return MethodResult(seq, tuple(draws.events), zscales)


def score_at(
  ctx: MethodContext, field_at: Callable[[np.ndarray, int], np.ndarray], seq: np.ndarray, i: int, pins: Sequence[Pin],
  *, selective: bool, lam: float | None,
) -> np.ndarray:  # fmt: skip
  """Upstream ``_score_at`` (UP:1795-1820): the stability row, minus the mean deprotonated-centre row when selective (or blended by ``lam``)."""
  for p in pins:
    seq[p.position] = p.prot_idx
  e_stab = np.array(field_at(seq, i), copy=True)
  if not selective and lam is None:
    return np.where(ctx.valid, e_stab, e_stab.dtype.type(np.inf))
  ed = np.zeros_like(e_stab)
  for p in pins:
    c = np.zeros_like(e_stab)
    for d in p.dep_idxs:
      seq[p.position] = d
      c += field_at(seq, i)
    seq[p.position] = p.prot_idx
    ed += c / len(p.dep_idxs)
  ed = ed / len(pins)
  score = (1.0 + lam) * e_stab - lam * ed if lam is not None else e_stab - ed
  return np.where(ctx.valid, score, score.dtype.type(np.inf))


def converge(
  ctx: MethodContext, config: PHDesignConfig, pins: Sequence[Pin], neigh: Sequence[int], initial: np.ndarray, draws: Draws,
  *, start: np.ndarray | None = None, backend: str | None = None, method: str | None = None, selective: bool | None = None,
) -> MethodResult:  # fmt: skip
  """Upstream ``_converge``: random single-site moves until ``cv_patience*N`` unchanged steps or ``cv_max*N`` steps."""
  method = config.method if method is None else method
  backend = config.backend if backend is None else backend
  lam = float(config.combined_lambda) if method == "converged_mcmc_combined" else None
  selective = (config.selective if selective is None else selective) if lam is None else False
  field_at = ctx.field_at(backend)
  seq = np.array(initial if start is None else start, dtype=np.int32, copy=True)
  for p in pins:
    seq[p.position] = p.prot_idx
  neigh = [int(i) for i in neigh]
  patience, cap = config.cv_patience * len(neigh), config.cv_max * len(neigh)
  since = step = 0
  while neigh and since < patience and step < cap:
    i = neigh[draws.randint(len(neigh))]
    token = pick(
      score_at(ctx, field_at, seq, i, pins, selective=selective, lam=lam),
      float(config.temperature),
      draws,
    )
    since = 0 if token != int(seq[i]) else since + 1
    seq[i] = token
    step += 1
  return MethodResult(seq, tuple(draws.events), None)


def two_phase(
  ctx: MethodContext, config: PHDesignConfig, pins: Sequence[Pin], neigh: Sequence[int], initial: np.ndarray, draws: Draws,
) -> MethodResult:  # fmt: skip
  """Upstream ``_two_phase``: commit the least-disruptive selective picks, then converge the rest without selectivity."""
  field_at = ctx.field_at(config.backend)
  seq = np.array(initial, dtype=np.int32, copy=True)
  for p in pins:
    seq[p.position] = p.prot_idx
  neigh = [int(i) for i in neigh]
  sel_pick: dict[int, int] = {}
  disruption: dict[int, float] = {}
  for i in neigh:
    a = pick(
      score_at(ctx, field_at, seq, i, pins, selective=True, lam=None),
      float(config.temperature),
      draws,
    )
    e_stab = field_at(seq, i)
    sel_pick[i] = a
    disruption[i] = float(e_stab[a] - e_stab[int(seq[i])])
  ranked = sorted(neigh, key=lambda i: disruption[i])
  k = max(1, min(len(neigh) - 1, round(config.two_phase_frac * len(neigh))))
  for i in ranked[:k]:
    seq[i] = sel_pick[i]
  committed = set(ranked[:k])
  rest = [i for i in neigh if i not in committed]
  return converge(
    ctx, config, pins, rest, initial, draws, start=seq, method="converged_mcmc", selective=False,
  )  # fmt: skip


def gibbs(
  ctx: MethodContext, config: PHDesignConfig, seq_init: np.ndarray, free_mask: np.ndarray, draws: Draws,
) -> list[np.ndarray]:  # fmt: skip
  """Upstream ``potts_gibbs_optimize`` (convergence mode): sweeps over the free positions in a random order, each position resampled from its
  conditional Potts energy at ``T = max(temperature, 1e-3)``, until a sweep changes nothing or ``MAX_GIBBS_SWEEPS``. One sequence per row."""
  free = np.flatnonzero(np.asarray(free_mask, dtype=bool))
  temperature = max(float(config.temperature), MIN_TEMPERATURE)
  out: list[np.ndarray] = []
  for row in np.atleast_2d(seq_init):
    seq = np.array(row, dtype=np.int32, copy=True)
    for _sweep in range(MAX_GIBBS_SWEEPS):
      mutations = 0
      for position in free[draws.permutation(len(free))]:
        energies = ctx.rows(seq, [int(position)])[0]
        energies = np.where(ctx.valid, energies, energies.dtype.type(np.inf))
        new = draws.pick_probs(softmax_probs(energies, temperature))
        mutations += int(new != int(seq[position]))
        seq[position] = new
      if mutations == 0:
        break
    out.append(seq)
  return out


# ---- the structure-level driver ------------------------------------------------------------------------------------------------------


def placement_field(
  ctx: MethodContext, native: np.ndarray, binder_mask: np.ndarray, placement_by: str,
) -> tuple[np.ndarray, np.ndarray]:
  """``(field, scanned_sequence)`` the placement ranks by (UP:2563-2600 ``_ranked_candidates``).

  ``scan_potts``: the Potts conditional energies ``(L, V)`` of the native sequence. ``scan_mpnn``: the decoder field of the native sequence with the
  binder positions set to ``X`` (``_place_scan_mpnn`` always masks; ``placement_seq_masked`` is not what decides it).
  """
  if placement_by == "scan_mpnn":
    scan_seq = np.array(native, dtype=np.int32, copy=True)
    scan_seq[np.asarray(binder_mask, dtype=bool)] = UNK_INDEX
    return ctx.need_decoder()(scan_seq), scan_seq
  field = np.asarray(_CANDIDATES(ctx.table, ctx.e_idx, jnp.asarray(native, dtype=jnp.int32)))
  return field, np.asarray(native, dtype=np.int32)


def plan_for_methods(
  ctx: MethodContext, native: np.ndarray, binder_mask: np.ndarray, res_id: np.ndarray, e_idx_np: np.ndarray, config: PHDesignConfig,
) -> Plan | None:  # fmt: skip
  """Placement plan for the new methods: explicit centres, or ``center_types`` ranked by ``scan_potts`` or ``scan_mpnn``.

  ``selective=False`` ranks by ``e_P`` minus the scanned sequence's own token energy instead of the centre gap (``_ranked_candidates``, UP:2563).
  """
  dep_map = config.dep_map_dict()
  common = {
    "infill_scope": config.infill_scope,
    "neighbour_k": config.neighbour_k,
    "max_mutations": config.max_mutations,
  }
  if config.explicit_centers:
    return plan_from_explicit_centers(
      e_idx_np, binder_mask, res_id, config.explicit_centers, dep_map, **common,
    )
  field, base_seq = placement_field(ctx, native, binder_mask, config.placement_by)
  return plan_from_center_types(
    field, e_idx_np, binder_mask, res_id, config.center_types, dep_map, **common,
    selective=config.selective, base_sequence=base_seq,
  )  # fmt: skip


def design_methods(
  model: PottsMPNN | None,
  graph_args: Sequence[jax.Array] | None,
  table: jax.Array,
  e_idx: jax.Array,
  native: np.ndarray,
  binder_mask: np.ndarray,
  res_id: np.ndarray,
  config: PHDesignConfig,
  *,
  key: jax.Array | None = None,
  replay: Sequence[dict] | None = None,
) -> list[PHDesign]:
  """All designs of one structure for ``autoregressive``, ``converged_mcmc(_combined)``, ``two_phase`` or ``gibbs``.

  ``config.samples_per_site`` designs (``gibbs``: that many rows of one call), in sample order, each scored by its whole-sequence Potts energy and
  centre gap. Give ``key`` (a JAX key, folded per sample) or ``replay`` (one dict per sample with ``uniforms``, ``randints``, ``perms`` and, for
  ``autoregressive``, ``order``). Designs are returned in ``(final_potts_energy, order)`` order like :func:`ph_design.design_structure`.
  """
  binder_mask = np.asarray(binder_mask, dtype=bool)
  if not binder_mask.any():
    msg = "binder_mask has no True entries: no designable binder position"
    raise ValueError(msg)
  if key is None and replay is None:
    msg = f"{config.method} needs a jax PRNG key or replayed draws"
    raise ValueError(msg)
  native_np = np.asarray(native, dtype=np.int32)
  e_idx_np = np.asarray(e_idx, dtype=np.int32)
  needs_decoder = (
    config.backend == "mpnn"
    or config.method == "autoregressive"
    or config.placement_by == "scan_mpnn"
  )
  field = None
  if needs_decoder:
    if model is None or graph_args is None:
      msg = f"method={config.method!r} backend={config.backend!r} placement_by={config.placement_by!r} needs the model and graph arguments"
      raise ValueError(msg)
    field = DecoderField.from_model(model, graph_args)
  ctx = MethodContext(
    table, e_idx, valid_token_mask(config.forbidden_tokens), np.asarray(_cdf_order()), field,
  )
  length = native_np.shape[0]
  n = int(config.samples_per_site)
  if replay is not None and len(replay) != n:
    msg = f"replay has {len(replay)} entries, expected samples_per_site={n}"
    raise ValueError(msg)

  def draws_for(sample: int) -> Draws:
    if replay is not None:
      r = replay[sample]
      return Draws(uniforms=r.get("uniforms", []), randints=r.get("randints", []), perms=r.get("perms", []),
                   choices=r.get("choices"), cdf_order=ctx.cdf_order)  # fmt: skip
    seed = int(jax.random.randint(jax.random.fold_in(key, sample), (), 0, 2**31 - 1))
    return Draws(seed=seed, cdf_order=ctx.cdf_order)

  designs: list[PHDesign] = []
  valid = jnp.ones(length, dtype=bool)

  def finish(
    seq: np.ndarray,
    sample: int,
    pins: tuple[Pin, ...],
    designable: tuple[int, ...],
    label: str,
    n_draws: int,
  ) -> None:
    energy = float(potts_energy(table, e_idx, valid, jnp.asarray(seq, dtype=jnp.int32)))
    gaps = (
      tuple(
        _gap(row, p) for row, p in zip(ctx.rows(seq, [p.position for p in pins]), pins, strict=True)
      )
      if pins
      else ()
    )
    designs.append(PHDesign(np.asarray(seq, dtype=np.int32), config.method, sample, pins, designable, label, energy,
                            float(sum(gaps)), gaps, n_draws))  # fmt: skip

  if config.method == "gibbs":
    draws = draws_for(0)
    free = tuple(int(i) for i in np.flatnonzero(binder_mask))
    seqs = gibbs(ctx, config, np.broadcast_to(native_np, (n, length)), binder_mask, draws)
    for sample, seq in enumerate(seqs):
      finish(seq, sample, (), free, "", len(draws.events))
  else:
    plan = plan_for_methods(ctx, native_np, binder_mask, np.asarray(res_id), e_idx_np, config)
    if plan is None or not plan.designable:
      return []
    for sample in range(n):
      draws = draws_for(sample)
      if config.method == "autoregressive":
        if replay is not None:
          order = list(replay[sample]["order"])
        else:
          order = random.Random(int(jax.random.randint(jax.random.fold_in(key, 10_007 + sample), (), 0, 2**31 - 1))).sample(
            list(plan.designable), len(plan.designable))  # fmt: skip
        result = masked_infill(ctx, config, plan.pins, order, native_np, draws)
      elif config.method == "two_phase":
        result = two_phase(ctx, config, plan.pins, plan.designable, native_np, draws)
      else:
        result = converge(ctx, config, plan.pins, plan.designable, native_np, draws)
      finish(
        result.sequence,
        sample,
        tuple(plan.pins),
        tuple(plan.designable),
        plan.label,
        len(result.events),
      )
  designs.sort(key=lambda d: d.final_potts_energy)
  return designs
