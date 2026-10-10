"""Centre-free greedy energy block descent for ProtonPottsMPNN pH-design (pure JAX, Potts energy backend).

Port of upstream ``_greedy_energy_block`` (``ProtonPottsMPNN/foundry/models/mpnn/src/mpnn/inference_engines/
potts_mpnn_ph.py`` 2390-2532). There are no pinned centres and no selective term. Each step:

1. scores every designable position on the CURRENT sequence, ``delta_i = min_{a valid} e_i[a] - e_i[S_i]``,
   where ``e_i`` is the conditional energy row (upstream ``cond_energy_at`` 2456-2468, here
   ``candidate_energies_at``). Position i improves when ``delta_i < -1e-4``;
2. stops when no position improves (2470-2472);
3. takes the ``block_size`` most-improving positions, sorted by ``(delta, position)`` (2473-2474). The block is
   shorter than ``block_size`` when fewer positions improve, so its size varies per step. Missing members are padded
   as in ``ph_descent`` (``block_valid`` False, token 0 only, zero contribution);
4. commits the exact argmin (``temperature <= 0``) or one inverse-CDF draw (``temperature > 0``, one uniform) over
   all joint assignments of the block (2479-2516). The block objective is ``wH * stability + repetitive-window``;
5. scores the full Potts energy H of the new sequence and keeps the best one seen, with strict improvement by 1e-9
   (2527-2528). The loop ends after ``cv_patience * N`` non-improving steps or ``cv_max * N`` steps in total
   (2459). The BEST sequence is returned, not the last (2530-2531).

``wH = 1 / sdH``, with the single-mutation z-scale ``sdH`` (``_zscale`` 1995-2003, ``_block_mutation_tables``
2006-2032) taken over every designable position and every valid token, the current token included as a zero.
It is the population std, floored at 1e-6, and 1.0 when there are no entries. The block-level z-scale of
``ph_descent`` is NOT used, and ``zscale_mode`` is ignored.

Before the loop, each designable token is canonicalised to its bare parent residue when that residue is valid
(2444-2448). Forbidden protonation microstates such as ``HIS-A`` therefore become ``H``. Non-designable positions
are never touched.

Several upstream comments are wrong. The comment at 2403 and the docstring at 2440-2443 say greedy keeps the design
plain 20-AA. That is false under the production forbidden list (``HIS-A``, ``ASP-A``, ``GLU-A``, ``UNK``): the
protonated states ``HIS-P``, ``HIS-S``, ``ASP-P``, ``ASP-D``, ``GLU-P`` and ``GLU-D`` are drawable. This module
follows the code.

Contract of the inputs (same convention as ``ph_descent``, aminx v6 token order):

- ``table (L, K, V, V)``, ``e_idx (L, K)``: merged Potts table and neighbour index, slot 0 = self.
- ``designable (N,)`` int: the free positions, ascending. For the centre-free plan these are all free binder
  positions (``ph_plan.plan_centre_free``).
- ``valid_tokens (V,)`` bool: ``ph_plan.valid_token_mask``.
- ``rep_mask (V,)`` float: from :func:`ph_descent.rep_class_mask`. Upstream's ``repetitive_window_parents ==
  ["ALL"]`` mode (anti-repetition over every amino acid, ``rep_all`` at 2422) is NOT ported. It cannot be
  expressed as a class mask, and ``rep_class_mask(["ALL"])`` is all zeros, so the window term would silently vanish.
  Do not pass ``"ALL"``.
- ``residue_number (L,)`` int: as in ``ph_descent``.
- ``uniforms (M,)``: one consumed per block update when ``config.temperature > 0``. ``M`` must be at least
  ``max(1, cv_max * N)``, checked statically.

Knobs. ``combined_lambda`` and ``zscale_mode`` are ignored, since greedy has no selective term and always uses the
single-mutation z-scale. ``self_weight != 1`` is refused, because upstream applies it to the stability term through
``crit.stability_weights()`` (2411). ``global_weight`` and ``adjacent_repeat_weight`` are not read by upstream's
greedy function and are therefore not refused. ``sweep_order`` does not apply, since greedy has no sweep. Trajectory
recording is not ported.

Sampling (``temperature > 0``) uses the inverse CDF of ``softmax(-(J - min J) / T)``, as in ``ph_descent.select_joint``.
Upstream draws with ``torch.multinomial`` (2373), so the sampled stream is not upstream-identical. The result's
``n_draws`` counts block updates at temperature > 0 and is 0 at temperature 0.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, Bool, Float, Int

from aminx.families.protonpotts_mpnn.ph_descent import (
  _digits,
  _joint,
  _repetitive,
  select_joint,
)
from aminx.families.protonpotts_mpnn.ph_potentials import (
  block_stability_potentials,
  candidate_energies_at,
)
from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6, canonical_letter, token_index

if TYPE_CHECKING:
  from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig

_IMPROVE_TOL = 1e-4  # upstream ``tol`` (2410)
_BEST_TOL = 1e-9  # upstream strict-improvement margin (2527)


class GreedyResult(NamedTuple):
  seq: Int[Array, " L"]  # BEST sequence found (not the last)
  n_draws: Int[Array, ""]  # uniforms consumed: block updates at temperature > 0, else 0
  steps: Int[Array, ""]  # block updates executed
  zscale: Float[Array, ""]  # single-mutation z-scale sdH used for wH


def _canonical_index_table() -> np.ndarray:
  """``canonical[a]``: aminx index of the bare parent residue of token ``a`` (``HIS-A`` -> ``H``)."""
  return np.array(
    [token_index(canonical_letter(idx)) for idx in range(PROTONPOTTS_V6.size)],
    dtype=np.int32,
  )


def _check_supported(config: PHDesignConfig) -> None:
  if config.self_weight != 1.0:
    msg = (
      f"greedy_energy_block does not support self_weight={config.self_weight!r} "
      "(the self/pair reweighting of the stability term)"
    )
    raise NotImplementedError(msg)


def _single_zscale(
  energies: Float[Array, "N V"],
  seq: Int[Array, " L"],
  designable: Int[Array, " N"],
  valid_tokens: Bool[Array, " V"],
) -> Float[Array, ""]:
  """Population std of the finite valid single-mutation deltas, floored at 1e-6, else 1.0 (upstream 1995-2003)."""
  cur = seq[designable]
  deltas = energies - jnp.take_along_axis(energies, cur[:, None], axis=1)
  finite = valid_tokens[None, :] & jnp.isfinite(deltas)
  count = jnp.sum(finite)
  safe = jnp.maximum(count, 1).astype(energies.dtype)
  vals = jnp.where(finite, deltas, 0.0)
  mean = jnp.sum(vals) / safe
  var = jnp.sum(jnp.where(finite, (vals - mean) ** 2, 0.0)) / safe
  sd = jnp.maximum(jnp.sqrt(var), 1e-6)
  return jnp.where(count > 0, sd, 1.0).astype(energies.dtype)


def _improving_block(
  energies: Float[Array, "N V"],
  seq: Int[Array, " L"],
  designable: Int[Array, " N"],
  valid_tokens: Bool[Array, " V"],
  n_block: int,
) -> tuple[Int[Array, " B"], Bool[Array, " B"], Int[Array, ""]]:
  """Up to ``n_block`` most-improving designable positions, by ``(delta, position)`` ascending.

  Returns ``(block, block_valid, n_improving)``. Padded slots carry ``-1`` and ``block_valid`` False.
  """
  cur = seq[designable]
  masked = jnp.where(valid_tokens[None, :], energies, jnp.inf)
  cur_energy = jnp.take_along_axis(energies, cur[:, None], axis=1)[:, 0]
  delta = jnp.min(masked, axis=1) - cur_energy
  improving = delta < -_IMPROVE_TOL
  n_improving = jnp.sum(improving).astype(jnp.int32)
  key = jnp.where(improving, delta, jnp.inf)
  order = jnp.lexsort((designable, key))  # primary: key, secondary: position
  slots = jnp.arange(n_block)
  picked = designable[order[jnp.minimum(slots, designable.shape[0] - 1)]]
  block_valid = slots < n_improving
  block = jnp.where(block_valid, picked, -1)
  return block, block_valid, n_improving


def _block_objective(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
  block: Int[Array, " B"],
  block_valid: Bool[Array, " B"],
  valid_tokens: Bool[Array, " V"],
  rep_mask: Float[Array, " V"],
  residue_number: Int[Array, " L"],
  *,
  rw: float,
  rrad: int,
  wh: Float[Array, ""],
) -> Float[Array, " S"]:
  """Flat ``(V,)*B`` objective ``wH * stability + repetitive window`` over the block (upstream 2479-2514).

  Padded axes allow only token 0, so the finite entries are exactly the real-member enumeration in the same
  relative order. ``argmin`` and the inverse CDF therefore pick the same assignment as an unpadded block.
  """
  n_block = block.shape[0]
  vocab = table.shape[-1]
  block_safe = jnp.where(block_valid, block, 0)
  unary, pair = block_stability_potentials(table, e_idx, seq, block_safe, block_valid)
  rep_unary, rep_pair = _repetitive(seq, block, block_valid, residue_number, rep_mask, rw, rrad)
  member = block_valid[:, None]
  allowed = jnp.where(member, valid_tokens[None, :], (jnp.arange(vocab) == 0)[None, :])
  unary_tot = jnp.where(member, wh * unary + rep_unary, 0.0)
  pair_tot = wh * pair + rep_pair
  return _joint(unary_tot, pair_tot, allowed, n_block, vocab).reshape(-1)


def _potts_energy(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
) -> Array:
  """Full Potts energy ``sum_i sum_k table[i, k, s_i, s_{e_idx[i, k]}]`` over every directed edge, slot 0 included."""
  length, k_slots = e_idx.shape
  rows = jnp.arange(length)[:, None]
  slots = jnp.arange(k_slots)[None, :]
  return jnp.sum(table[rows, slots, seq[:, None], seq[e_idx]])


def _canonicalise(
  seq: Int[Array, " L"],
  designable: Int[Array, " N"],
  valid_tokens: Bool[Array, " V"],
) -> Int[Array, " L"]:
  """Replace each designable token by its bare parent residue when that residue is valid (upstream 2444-2448)."""
  is_designable = jnp.zeros((seq.shape[0],), dtype=bool).at[designable].set(True)
  canonical_seq = jnp.asarray(_canonical_index_table())[seq]
  return jnp.where(is_designable & valid_tokens[canonical_seq], canonical_seq, seq)


def _greedy_step(
  state: tuple[Array, ...],
  *,
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  designable: Int[Array, " N"],
  valid_tokens: Bool[Array, " V"],
  rep_mask: Float[Array, " V"],
  residue_number: Int[Array, " L"],
  uniforms: Float[Array, " M"],
  n_block: int,
  temperature: float,
  cdf_order: Array | None,
  rw: float,
  rrad: int,
  wh: Float[Array, ""],
) -> tuple[Array, ...]:
  """One greedy step on the loop state ``(seq, best_seq, best_h, since, step, draws, done)``.

  When no position improves, the state is returned unchanged with ``done`` set (upstream break, 2470-2472).
  """
  length, vocab = table.shape[0], table.shape[-1]
  n_uniform = uniforms.shape[0]
  seq_now, best_seq, best_h, since, step, draws, _ = state
  energies = candidate_energies_at(table, e_idx, seq_now, designable)
  block, block_valid, n_improving = _improving_block(
    energies,
    seq_now,
    designable,
    valid_tokens,
    n_block,
  )
  stop = n_improving == 0
  flat = _block_objective(
    table,
    e_idx,
    seq_now,
    block,
    block_valid,
    valid_tokens,
    rep_mask,
    residue_number,
    rw=rw,
    rrad=rrad,
    wh=wh,
  )
  if temperature > 0:
    uniform = uniforms[jnp.minimum(draws, n_uniform - 1)]
    choice = select_joint(flat, uniform, temperature, n_block=n_block, cdf_order=cdf_order)
  else:
    choice = jnp.argmin(flat).astype(jnp.int32)
  target = jnp.where(block_valid, block, length)
  seq_new = seq_now.at[target].set(
    _digits(choice, n_block, vocab).astype(seq_now.dtype),
    mode="drop",
  )
  seq_next = jnp.where(stop, seq_now, seq_new)
  h_next = _potts_energy(table, e_idx, seq_next)
  better = (~stop) & (h_next < best_h - _BEST_TOL)
  best_seq = jnp.where(better, seq_next, best_seq)
  best_h = jnp.where(better, h_next, best_h)
  since = jnp.where(stop, since, jnp.where(better, 0, since + 1))
  step = jnp.where(stop, step, step + 1)
  if temperature > 0:
    draws = jnp.where(stop, draws, draws + 1)
  return seq_next, best_seq, best_h, since, step, draws, stop


def greedy_energy_block(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
  designable: Int[Array, " N"],
  valid_tokens: Bool[Array, " V"],
  rep_mask: Float[Array, " V"],
  residue_number: Int[Array, " L"],
  *,
  config: PHDesignConfig,
  uniforms: Float[Array, " M"],
  cdf_order: Array | None = None,
) -> GreedyResult:
  """Centre-free greedy energy block descent (upstream ``_greedy_energy_block`` 2390-2532).

  Returns the best sequence seen, its uniform and step counts, and the z-scale. ``cdf_order`` is passed to
  ``select_joint`` and exists for parity with a recorded upstream sample. It is left unset in production.
  Unsupported knobs raise ``NotImplementedError`` before any work. A temperature above 0 needs at least
  ``max(1, cv_max * N)`` uniforms, else ``ValueError``.
  """
  _check_supported(config)
  table = jnp.asarray(table)
  dtype = table.dtype
  e_idx = jnp.asarray(e_idx, dtype=jnp.int32)
  seq = jnp.asarray(seq, dtype=jnp.int32)
  designable = jnp.asarray(designable, dtype=jnp.int32)
  valid_tokens = jnp.asarray(valid_tokens, dtype=bool)
  rep_mask = jnp.asarray(rep_mask, dtype=dtype)
  residue_number = jnp.asarray(residue_number, dtype=jnp.int32)
  uniforms = jnp.asarray(uniforms, dtype=dtype)

  n_designable = designable.shape[0]
  if n_designable == 0:  # upstream returns the untouched clone (2437-2438)
    return GreedyResult(
      seq=seq,
      n_draws=jnp.asarray(0, dtype=jnp.int32),
      steps=jnp.asarray(0, dtype=jnp.int32),
      zscale=jnp.ones((), dtype=dtype),
    )

  temperature = float(config.temperature)
  n_block = max(1, int(config.block_size))
  patience = int(config.cv_patience) * n_designable
  cap = int(config.cv_max) * n_designable
  n_uniform = uniforms.shape[0]
  if temperature > 0 and n_uniform < max(1, cap):
    msg = (
      f"uniforms has {n_uniform} entries; sampling needs up to cv_max * N = {cap} "
      "(one per block update)"
    )
    raise ValueError(msg)

  seq = _canonicalise(seq, designable, valid_tokens)
  zscale = _single_zscale(
    candidate_energies_at(table, e_idx, seq, designable),
    seq,
    designable,
    valid_tokens,
  )
  wh = (1.0 / zscale).astype(dtype)
  rw = float(config.repetitive_window_weight)
  rrad = max(1, int(config.repetitive_window_radius))
  step_fn = partial(
    _greedy_step,
    table=table,
    e_idx=e_idx,
    designable=designable,
    valid_tokens=valid_tokens,
    rep_mask=rep_mask,
    residue_number=residue_number,
    uniforms=uniforms,
    n_block=n_block,
    temperature=temperature,
    cdf_order=cdf_order,
    rw=rw,
    rrad=rrad,
    wh=wh,
  )

  def keep_going(state: tuple[Array, ...]) -> Array:
    _, _, _, since, step, _, done = state
    return (~done) & (since < patience) & (step < cap)

  zero = jnp.zeros((), dtype=jnp.int32)
  init = (seq, seq, _potts_energy(table, e_idx, seq), zero, zero, zero, jnp.zeros((), dtype=bool))
  _, best_seq, _, _, steps, draws, _ = lax.while_loop(keep_going, step_fn, init)
  return GreedyResult(seq=best_seq, n_draws=draws, steps=steps, zscale=zscale)
