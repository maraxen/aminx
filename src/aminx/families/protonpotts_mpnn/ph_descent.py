"""Block coordinate descent for ProtonPottsMPNN pH-design on the Potts energy backend (pure JAX).

Port of the sweep of upstream ``_block_descent`` (``ProtonPottsMPNN/foundry/models/mpnn/src/mpnn/
inference_engines/potts_mpnn_ph.py`` 2193-2387) with its z-scale freezing (``_block_zscales`` 2055-2123),
the selective gap (``_selective_energy_of`` / ``_selective_energy_sum`` 2539-2556) and the windowed
repetitive-density term (``_parent_vocab_mask`` 2129-2140, ``combined_unary`` 2262-2306, and the within-block
pairwise term 2352-2362). Several upstream docstrings disagree with the code; this module follows the code.
Block construction (``_block_partners`` 2041-2052) lives in ``ph_plan.block_table``.

The optimised objective, with the z-scales frozen once per run from the seed sequence, is::

    F(S) = wH * H(S) + wSel * sel(S) + rw * R(S)
    wH = (1 - lam) / sdH,   wSel = lam / sdSel,   rw = config.repetitive_window_weight

``H`` is the full Potts energy, ``sel(S) = sum_c (e_c(P_c) - mean_d e_c(d))`` over the pinned centres c
(every pin held at its protonated token), and ``R(S)`` is the sum over unordered position pairs with
``0 < |residue_number_p - residue_number_q| <= repetitive_window_radius`` of ``rep[S_p] * rep[S_q]``. Each
block visit minimises F exactly over the block, so at temperature 0 F is non-increasing across visits.

Contract of the inputs (all in aminx v6 token order, see ``vocab``):

- ``table (L, K, V, V)``, ``e_idx (L, K)``: merged Potts table and neighbour index, slot 0 = self.
- ``plan_designable (N,)``, ``blocks (N, B)``, ``block_valid (N, B)``: from ``ph_plan.Plan.designable`` and
  ``ph_plan.block_table``. Row ``n`` is the visited position ``plan_designable[n]`` followed by its partners.
  Padded entries (``block_valid`` False) carry ``-1`` and are never written.
- ``pin_positions (P,)``, ``pin_prot (P,)``, ``pin_dep (P, D)``, ``pin_dep_valid (P, D)``: the pinned centres.
- ``valid_tokens (V,)`` bool: ``ph_plan.valid_token_mask``.
- ``rep_mask (V,)`` float: 1.0 for tokens whose parent residue is in ``repetitive_window_parents``; build it
  with :func:`rep_class_mask`.
- ``residue_number (L,)`` int: residue ids. Upstream counts repetitive-window neighbours only among chain-A
  positions (``res_id_to_pos``, potts_mpnn_ph.py 1660). Positions outside the binder chain must therefore carry
  residue numbers that no binder residue lies within the radius of, or they will be counted here and not upstream.
- ``uniforms (M,)`` float in [0, 1): one consumed per block visit when ``config.temperature > 0``. ``M`` must be at
  least ``block_max_rounds * N`` (checked statically); the result's ``n_draws`` reports how many were used.

Z-scales (:func:`block_zscales`, upstream ``zscale_mode="block"``): for each block the population variance of
the finite entries of the joint stability tensor is taken (within-block pairwise variance included), and the
selective unary's variance is taken over blocks whose selectivity varies. Each pool is ``sqrt`` of the mean
per-block variance, floored at ``1e-6``, and ``1.0`` when no block contributes. The block's variance enumerates
only the real members, since padded axes are restricted to index 0 (a zero-probability alternative to changing
the axis size, which would also change the normalisation of the sampled distribution).

Not ported (refused with ``NotImplementedError`` at the top of :func:`block_descent` where a knob exists):
``global_weight``, ``adjacent_repeat_weight``, ``self_weight != 1``, ``zscale_mode != "block"``, and
``sweep_order != "position"``. ``PHDesignConfig`` has no ``repetitive_window_gate_types`` field, so the
upstream gate is absent and the window term is active whenever its weight is non-zero and a parent set is
given, which is upstream's behaviour when the gate is empty. Trajectory recording (``_record``) is not
ported; the result carries no trajectory.

Sampling (``temperature > 0``) uses the inverse CDF of ``softmax(-(J - min J) / T)``. The first index with
``cdf > u * total`` is taken, rather than ``cdf >= u * total``, so ``u == 0`` selects the first finite
index and never a zero-probability forbidden one. The result is clipped to the last index.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, Bool, Float, Int

from aminx.families.protonpotts_mpnn.ph_potentials import block_stability_potentials
from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6, THREE_TO_ONE, canonical_letter

if TYPE_CHECKING:
  from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig


class DescentResult(NamedTuple):
  seq: Int[Array, " L"]  # final sequence
  n_draws: Int[Array, ""]  # uniforms consumed (0 at temperature 0)
  rounds: Int[Array, ""]  # sweeps executed, including the final sweep that changed nothing
  zscales: Float[Array, " 3"]  # (sdH, sdSel, sdGlob) as upstream returns them; sdGlob is always 1.0


_LETTER_TO_THREE: dict[str, str] = {letter: name for name, letter in THREE_TO_ONE.items()}


def rep_class_mask(parents: Sequence[str]) -> np.ndarray:
  """(V,) float32 mask, 1.0 where a token's parent residue is in ``parents``.

  Port of upstream ``_parent_vocab_mask`` (potts_mpnn_ph.py 2129-2140). A protonation token shares its
  parent residue, so ``HIS-P``, ``HIS-S``, ``HIS-A`` and ``H`` all have parent ``"HIS"``. ``X`` has no
  three-letter parent and is never in the mask.
  """
  wanted = set(parents)
  mask = np.zeros(PROTONPOTTS_V6.size, dtype=np.float32)
  for idx in range(PROTONPOTTS_V6.size):
    name = _LETTER_TO_THREE.get(canonical_letter(idx))
    if name is not None and name in wanted:
      mask[idx] = 1.0
  return mask


def _check_supported(config: PHDesignConfig) -> None:
  """Refuse upstream knobs this port does not implement, naming each one."""
  checks = (
    (config.global_weight != 0.0, "global_weight != 0 (the upstream global protonation term)"),
    (
      config.adjacent_repeat_weight != 0.0,
      "adjacent_repeat_weight != 0 (the adjacent-repeat bias)",
    ),
    (config.self_weight != 1.0, "self_weight != 1.0 (self/pair reweighting of the stability term)"),
    (config.zscale_mode != "block", f"zscale_mode={config.zscale_mode!r} (only 'block' is ported)"),
    (
      config.sweep_order != "position",
      f"sweep_order={config.sweep_order!r} (only 'position' is ported)",
    ),
  )
  for unsupported, what in checks:
    if unsupported:
      raise NotImplementedError(f"block_descent does not support {what}")


def _set_pins(seq: Array, pin_positions: Array, pin_prot: Array) -> Array:
  return seq.at[pin_positions].set(pin_prot.astype(seq.dtype))


def _axis_shape(n_block: int, vocab: int, axes: tuple[int, ...]) -> tuple[int, ...]:
  shape = [1] * n_block
  for ax in axes:
    shape[ax] = vocab
  return tuple(shape)


def _joint(
  unary: Float[Array, "B V"],
  pair: Float[Array, "B B V V"],
  allowed: Bool[Array, "B V"],
  n_block: int,
  vocab: int,
) -> Float[Array, "..."]:
  """Joint tensor ``(V,) * B`` of the block objective: unary + upper-triangular pair + forbidden tokens.

  ``allowed[b, a]`` False adds ``+inf`` on axis ``b``. Padded axes are passed with only index 0 allowed, so
  their finite entries are exactly the real-member enumeration.
  """
  joint = jnp.zeros((vocab,) * n_block, dtype=unary.dtype)
  for b in range(n_block):
    joint = joint + unary[b].reshape(_axis_shape(n_block, vocab, (b,)))
    forbidden = jnp.where(allowed[b], 0.0, jnp.inf).astype(unary.dtype)
    joint = joint + forbidden.reshape(_axis_shape(n_block, vocab, (b,)))
  for bi in range(n_block):
    for bj in range(bi + 1, n_block):
      joint = joint + pair[bi, bj].reshape(_axis_shape(n_block, vocab, (bi, bj)))
  return joint


def _selective_unary(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  pin_positions: Int[Array, " P"],
  pin_prot: Int[Array, " P"],
  pin_dep: Int[Array, "P D"],
  pin_dep_valid: Bool[Array, "P D"],
  block: Int[Array, " B"],
  block_valid: Bool[Array, " B"],
) -> Float[Array, "B V"]:
  """Absolute selective contribution of each block member's token, ``G[b, x]``, from its pin edges.

  The selective term is ``sel = sum_c gap_c``, with ``gap_c = e_c(P_c) - mean_d e_c(d)`` and ``e_c(X)`` the
  candidate energy of token X at pin c with the rest held fixed. The token of a designable member p reaches
  ``gap_c`` only through the edges between c and p, and ``gap_c`` is linear in each of them, so ``sel`` is an
  exact sum of per-position unaries and needs no enumeration. Setting p to ``x`` contributes::

      G[p, x] = sum_{k>=1: e_idx[c,k]==p} (table[c,k,P_c,x] - mean_d table[c,k,d_c,x])     (edge c -> p)
              + sum_{k>=1: e_idx[p,k]==c} (table[p,k,x,P_c] - mean_d table[p,k,x,d_c])     (edge p -> c)

  and the block objective uses ``G[p, x] - G[p, current]``. Self-edges of a pin (``k == 0``) do not depend on p.
  Pins are fixed, so the result does not depend on the rest of the sequence.
  """
  n_pin = pin_positions.shape[0]
  n_block = block.shape[0]
  vocab = table.shape[-1]
  dtype = table.dtype
  if n_pin == 0:
    return jnp.zeros((n_block, vocab), dtype=dtype)

  block_safe = jnp.where(block_valid, block, 0)
  dep_safe = jnp.where(pin_dep_valid, pin_dep, 0)
  dep_w = pin_dep_valid.astype(dtype)  # (P, D)
  dep_count = jnp.maximum(dep_w.sum(axis=1), 1.0)  # (P,)
  k_ok = jnp.arange(e_idx.shape[1]) >= 1  # slot 0 is the self slot

  # Pin rows: the edge pin -> member, indexed table[c, k, token_at_c, token_at_member].
  t_pin = table[pin_positions]  # (P, K, V, V)
  pin_ar = jnp.arange(n_pin)
  prot_rows = t_pin[pin_ar, :, pin_prot]  # (P, K, V)
  dep_rows = t_pin[pin_ar[:, None], :, dep_safe]  # (P, D, K, V)
  dep_mean = jnp.einsum("pdkx,pd->pkx", dep_rows, dep_w) / dep_count[:, None, None]
  gap_out = prot_rows - dep_mean  # (P, K, V)
  rows_pin = e_idx[pin_positions]  # (P, K)
  out_hit = (
    (rows_pin[:, :, None] == block_safe[None, None, :])
    & block_valid[None, None, :]
    & k_ok[None, :, None]
  )  # (P, K, B)
  g_out = jnp.einsum("pkb,pkx->bx", out_hit.astype(dtype), gap_out)

  # Member rows: the edge member -> pin, indexed table[member, k, token_at_member, token_at_c].
  t_blk = table[block_safe]  # (B, K, V, V)
  prot_in = t_blk[:, :, :, pin_prot]  # (B, K, V, P)
  dep_in = t_blk[:, :, :, dep_safe]  # (B, K, V, P, D)
  dep_in_mean = jnp.einsum("bkxpd,pd->bkxp", dep_in, dep_w) / dep_count
  gap_in = prot_in - dep_in_mean  # (B, K, V, P)
  rows_blk = e_idx[block_safe]  # (B, K)
  in_hit = (
    (rows_blk[:, :, None] == pin_positions[None, None, :])
    & block_valid[:, None, None]
    & k_ok[None, :, None]
  )  # (B, K, P)
  g_in = jnp.einsum("bkp,bkxp->bx", in_hit.astype(dtype), gap_in)
  return g_out + g_in


def _repetitive(
  seq: Int[Array, " L"],
  block: Int[Array, " B"],
  block_valid: Bool[Array, " B"],
  residue_number: Int[Array, " L"],
  rep_mask: Float[Array, " V"],
  rw: float,
  rrad: int,
) -> tuple[Float[Array, "B V"], Float[Array, "B B V V"]]:
  """Repetitive-window unary and within-block pairwise terms (upstream 2262-2306 and 2352-2362).

  The unary of member p is ``rw * n_ctx(p) * rep_mask``, where ``n_ctx`` counts class residues at residue
  distance ``1..rrad`` from p among positions that are not block members, read off the working sequence. The
  pairwise term is ``rw * outer(rep_mask, rep_mask)`` for each member pair within ``rrad`` residues.
  """
  length = seq.shape[0]
  n_block = block.shape[0]
  dtype = rep_mask.dtype
  block_safe = jnp.where(block_valid, block, 0)
  res_block = residue_number[block_safe]  # (B,)

  in_block = (
    jnp.zeros((length,), dtype=bool)
    .at[jnp.where(block_valid, block, length)]
    .set(True, mode="drop")
  )
  dist = jnp.abs(residue_number[None, :] - res_block[:, None])  # (B, L)
  near = (dist >= 1) & (dist <= rrad)
  is_class = rep_mask[seq] > 0  # (L,)
  n_ctx = jnp.sum(near & is_class[None, :] & ~in_block[None, :], axis=1).astype(dtype)  # (B,)
  unary = rw * n_ctx[:, None] * rep_mask[None, :]  # (B, V)

  close = jnp.abs(res_block[:, None] - res_block[None, :]) <= rrad  # (B, B)
  upper = jnp.triu(jnp.ones((n_block, n_block), dtype=bool), k=1)
  member_pair = block_valid[:, None] & block_valid[None, :] & close & upper  # (B, B)
  outer = rep_mask[:, None] * rep_mask[None, :]  # (V, V)
  pair = (rw * member_pair.astype(dtype))[:, :, None, None] * outer[
    None,
    None,
    :,
    :,
  ]  # (B, B, V, V)
  return unary, pair


def block_objective(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
  block: Int[Array, " B"],
  block_valid: Bool[Array, " B"],
  pin_positions: Int[Array, " P"],
  pin_prot: Int[Array, " P"],
  pin_dep: Int[Array, "P D"],
  pin_dep_valid: Bool[Array, "P D"],
  valid_tokens: Bool[Array, " V"],
  rep_mask: Float[Array, " V"],
  residue_number: Int[Array, " L"],
  *,
  config: PHDesignConfig,
  wh: Float[Array, ""],
  wsel: Float[Array, ""],
) -> Float[Array, " S"]:
  """Flat block objective ``J`` over all ``V**B`` joint assignments of one block.

  ``J(x) - J(x0)`` equals ``F(x) - F(x0)`` for the objective in the module docstring, with every other position
  fixed at ``seq`` (pins must already be set in ``seq``). Row-major with block axis 0 slowest, so
  ``np.unravel_index(index, [V] * B)`` recovers the assignment. Invalid tokens are ``+inf`` on every axis, and
  padded members (``block_valid`` False) take only token 0, with zero contribution.
  """
  _check_supported(config)
  table = jnp.asarray(table)
  e_idx = jnp.asarray(e_idx, dtype=jnp.int32)
  seq = jnp.asarray(seq, dtype=jnp.int32)
  block = jnp.asarray(block, dtype=jnp.int32)
  block_valid = jnp.asarray(block_valid, dtype=bool)
  pin_positions = jnp.asarray(pin_positions, dtype=jnp.int32)
  pin_prot = jnp.asarray(pin_prot, dtype=jnp.int32)
  pin_dep = jnp.asarray(pin_dep, dtype=jnp.int32)
  pin_dep_valid = jnp.asarray(pin_dep_valid, dtype=bool)
  valid_tokens = jnp.asarray(valid_tokens, dtype=bool)
  rep_mask = jnp.asarray(rep_mask, dtype=table.dtype)
  residue_number = jnp.asarray(residue_number, dtype=jnp.int32)
  vocab = table.shape[-1]
  n_block = block.shape[0]
  block_safe = jnp.where(block_valid, block, 0)

  unary, pair = block_stability_potentials(table, e_idx, seq, block_safe, block_valid)
  sel_abs = _selective_unary(
    table,
    e_idx,
    pin_positions,
    pin_prot,
    pin_dep,
    pin_dep_valid,
    block,
    block_valid,
  )
  cur = seq[block_safe]
  sel_rel = sel_abs - jnp.take_along_axis(sel_abs, cur[:, None], axis=1)

  rep_unary, rep_pair = _repetitive(
    seq,
    block,
    block_valid,
    residue_number,
    rep_mask,
    float(config.repetitive_window_weight),
    max(1, int(config.repetitive_window_radius)),
  )

  member = block_valid[:, None]
  unary_tot = jnp.where(member, wh * unary + wsel * sel_rel + rep_unary, 0.0)
  pair_tot = wh * pair + rep_pair
  allowed = jnp.where(member, valid_tokens[None, :], (jnp.arange(vocab) == 0)[None, :])
  return _joint(unary_tot, pair_tot, allowed, n_block, vocab).reshape(-1)


def _finite_var(joint: Float[Array, "..."]) -> tuple[Array, Array]:
  """Population variance and count of the finite entries of a joint tensor (upstream ``.var(unbiased=False)``)."""
  finite = jnp.isfinite(joint)
  count = jnp.sum(finite)
  safe_count = jnp.maximum(count, 1).astype(joint.dtype)
  values = jnp.where(finite, joint, 0.0)
  mean = jnp.sum(values) / safe_count
  var = jnp.sum(jnp.where(finite, (values - mean) ** 2, 0.0)) / safe_count
  return var, count


def _pool(var: Array, has: Array, dtype: np.dtype) -> Array:
  """Upstream ``_pool``: ``max(sqrt(mean of the contributing variances), 1e-6)``, or 1.0 when none contribute."""
  count = jnp.sum(has)
  mean = jnp.sum(jnp.where(has, var, 0.0)) / jnp.maximum(count, 1).astype(dtype)
  return jnp.where(count > 0, jnp.maximum(jnp.sqrt(mean), 1e-6), 1.0).astype(dtype)


def block_zscales(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
  blocks: Int[Array, "N B"],
  block_valid: Bool[Array, "N B"],
  pin_positions: Int[Array, " P"],
  pin_prot: Int[Array, " P"],
  pin_dep: Int[Array, "P D"],
  pin_dep_valid: Bool[Array, "P D"],
  valid_tokens: Bool[Array, " V"],
) -> Float[Array, " 3"]:
  """Frozen ``(sdH, sdSel, sdGlob)`` of upstream ``_block_zscales`` (``zscale_mode="block"``), with ``sdGlob=1``.

  Pins are set in ``seq`` first, as upstream does, so the caller may pass the raw seed sequence.
  """
  seq = _set_pins(seq, pin_positions, pin_prot)
  vocab = table.shape[-1]
  dtype = table.dtype
  n_block = blocks.shape[1]
  block_safe_all = jnp.where(block_valid, blocks, 0)

  def per_block(_: None, xs: tuple[Array, Array]) -> tuple[None, tuple[Array, Array, Array, Array]]:
    block, valid = xs
    block_safe = jnp.where(valid, block, 0)
    member = valid[:, None]
    allowed = jnp.where(member, valid_tokens[None, :], (jnp.arange(vocab) == 0)[None, :])

    unary, pair = block_stability_potentials(table, e_idx, seq, block_safe, valid)
    var_h, count_h = _finite_var(_joint(unary, pair, allowed, n_block, vocab))

    sel_abs = _selective_unary(
      table,
      e_idx,
      pin_positions,
      pin_prot,
      pin_dep,
      pin_dep_valid,
      block,
      valid,
    )
    cur = seq[block_safe]
    sel_rel = sel_abs - jnp.take_along_axis(sel_abs, cur[:, None], axis=1)
    sel_rel = jnp.where(member & valid_tokens[None, :], sel_rel, 0.0)
    varies = jnp.sum(jnp.abs(sel_rel)) > 0
    joint_s = _joint(
      sel_rel,
      jnp.zeros((n_block, n_block, vocab, vocab), dtype=dtype),
      allowed,
      n_block,
      vocab,
    )
    var_s, count_s = _finite_var(joint_s)

    has_h = count_h > 1
    has_s = varies & (count_s > 1)
    return None, (var_h, has_h, var_s, has_s)

  _, (var_h, has_h, var_s, has_s) = lax.scan(per_block, None, (block_safe_all, block_valid))
  sd_h = _pool(var_h, has_h, dtype)
  sd_s = _pool(var_s, has_s, dtype)
  return jnp.stack([sd_h, sd_s, jnp.asarray(1.0, dtype=dtype)])


def select_joint(
  flat: Float[Array, " S"],
  uniform: Float[Array, ""],
  temperature: float,
  *,
  n_block: int | None = None,
  cdf_order: Array | None = None,
) -> Array:
  """Index of the chosen joint assignment in the flat objective ``flat`` (lower is better).

  ``temperature <= 0`` gives the first minimum (``jnp.argmin``, matching ``torch.argmin``). Otherwise the index
  is drawn by inverse CDF of ``softmax(-(flat - min) / T)``: the first index whose cumulative probability exceeds
  ``uniform * total``, clipped to the last index. ``uniform == 0`` therefore selects the first finite index.

  The order in which entries enter the CDF decides which entry a given uniform selects. By default that is the
  flat (aminx token) order. ``cdf_order`` (``(V,)``, entry ``j`` = the aminx index of the token that comes ``j``-th)
  accumulates the CDF in another token order instead, along every block axis, and maps the draw back. Any order is a
  valid sampler; upstream accumulates in ITS token order, so parity with an upstream dump that recorded
  ``(uniform, choice)`` pairs needs ``cdf_order`` set to the upstream-to-aminx map. ``n_block`` is required with it.
  """
  if temperature <= 0:
    return jnp.argmin(flat).astype(jnp.int32)
  shifted = flat - jnp.min(flat)
  probs = jnp.exp(-shifted / temperature)
  if cdf_order is None:
    cdf = jnp.cumsum(probs)
    target = uniform * cdf[-1]
    index = jnp.sum(cdf <= target).astype(jnp.int32)
    return jnp.minimum(index, flat.shape[0] - 1)
  if n_block is None:
    msg = "select_joint: n_block is required with cdf_order"
    raise ValueError(msg)
  order = jnp.asarray(cdf_order, dtype=jnp.int32)
  vocab = order.shape[0]
  grid = probs.reshape((vocab,) * n_block)
  for axis in range(n_block):
    grid = jnp.take(grid, order, axis=axis)
  cdf = jnp.cumsum(grid.reshape(-1))
  target = uniform * cdf[-1]
  position = jnp.minimum(jnp.sum(cdf <= target).astype(jnp.int32), cdf.shape[0] - 1)
  digits = order[_digits(position, n_block, vocab)]
  powers = vocab ** jnp.arange(n_block - 1, -1, -1)
  return jnp.sum(digits * powers).astype(jnp.int32)


def _digits(choice: Array, n_block: int, vocab: int) -> Array:
  """Per-axis tokens of a row-major flat index, axis 0 slowest (``np.unravel_index(choice, [V] * B)``)."""
  return jnp.stack([(choice // vocab ** (n_block - 1 - b)) % vocab for b in range(n_block)]).astype(
    jnp.int32,
  )


def block_descent(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
  plan_designable: Int[Array, " N"],
  blocks: Int[Array, "N B"],
  block_valid: Bool[Array, "N B"],
  pin_positions: Int[Array, " P"],
  pin_prot: Int[Array, " P"],
  pin_dep: Int[Array, "P D"],
  pin_dep_valid: Bool[Array, "P D"],
  valid_tokens: Bool[Array, " V"],
  rep_mask: Float[Array, " V"],
  residue_number: Int[Array, " L"],
  *,
  config: PHDesignConfig,
  uniforms: Float[Array, " M"],
  cdf_order: Array | None = None,
) -> DescentResult:
  """Gauss-Seidel block descent over the designable positions, as upstream ``_block_descent`` 2193-2387.

  Pins are set to their protonated token first. Z-scales are frozen once from that sequence. Each sweep visits
  the designable rows in order; each visit commits the exact argmin (or an inverse-CDF sample) of its block
  objective, and later visits see the updated sequence. Sweeps repeat while the previous one changed something,
  up to ``config.block_max_rounds``.

  ``cdf_order`` (default None) sets the token order of the inverse-CDF sampler; see ``select_joint``. It exists
  for parity with an upstream dump and is left unset in production.

  ``plan_designable`` only fixes the row count and order (row ``n`` visits ``plan_designable[n]``, which is
  ``blocks[n, 0]``). Unsupported upstream knobs raise ``NotImplementedError`` before any work is done.
  """
  _check_supported(config)
  if blocks.shape[0] != plan_designable.shape[0]:
    raise ValueError(
      f"blocks has {blocks.shape[0]} rows but plan_designable has {plan_designable.shape[0]}",
    )
  table = jnp.asarray(table)
  e_idx = jnp.asarray(e_idx, dtype=jnp.int32)
  seq = jnp.asarray(seq, dtype=jnp.int32)
  blocks = jnp.asarray(blocks, dtype=jnp.int32)
  block_valid = jnp.asarray(block_valid, dtype=bool)
  pin_positions = jnp.asarray(pin_positions, dtype=jnp.int32)
  pin_prot = jnp.asarray(pin_prot, dtype=jnp.int32)
  pin_dep = jnp.asarray(pin_dep, dtype=jnp.int32)
  pin_dep_valid = jnp.asarray(pin_dep_valid, dtype=bool)
  valid_tokens = jnp.asarray(valid_tokens, dtype=bool)
  rep_mask = jnp.asarray(rep_mask, dtype=table.dtype)
  residue_number = jnp.asarray(residue_number, dtype=jnp.int32)
  uniforms = jnp.asarray(uniforms, dtype=table.dtype)

  dtype = table.dtype
  seq = _set_pins(seq, pin_positions, pin_prot)
  n_designable = blocks.shape[0]
  if n_designable == 0:
    return DescentResult(
      seq=seq,
      n_draws=jnp.asarray(0, dtype=jnp.int32),
      rounds=jnp.asarray(0, dtype=jnp.int32),
      zscales=jnp.ones((3,), dtype=dtype),
    )

  temperature = float(config.temperature)
  max_rounds = int(config.block_max_rounds)
  n_uniform = uniforms.shape[0]
  if temperature > 0 and n_uniform < max(1, max_rounds * n_designable):
    raise ValueError(
      f"uniforms has {n_uniform} entries; sampling needs up to block_max_rounds * N = "
      f"{max_rounds * n_designable}",
    )
  zscales = block_zscales(
    table,
    e_idx,
    seq,
    blocks,
    block_valid,
    pin_positions,
    pin_prot,
    pin_dep,
    pin_dep_valid,
    valid_tokens,
  )
  lam = float(config.combined_lambda)
  wh = (1.0 - lam) / zscales[0]
  wsel = lam / zscales[1]
  seq_final, draws_final, rounds_final = _sweep_to_convergence(
    table,
    e_idx,
    seq,
    blocks,
    block_valid,
    pin_positions,
    pin_prot,
    pin_dep,
    pin_dep_valid,
    valid_tokens,
    rep_mask,
    residue_number,
    uniforms,
    config=config,
    wh=wh,
    wsel=wsel,
    cdf_order=cdf_order,
  )
  return DescentResult(
    seq=seq_final,
    n_draws=draws_final,
    rounds=rounds_final,
    zscales=zscales,
  )


def _sweep_to_convergence(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
  blocks: Int[Array, "N B"],
  block_valid: Bool[Array, "N B"],
  pin_positions: Int[Array, " P"],
  pin_prot: Int[Array, " P"],
  pin_dep: Int[Array, "P D"],
  pin_dep_valid: Bool[Array, "P D"],
  valid_tokens: Bool[Array, " V"],
  rep_mask: Float[Array, " V"],
  residue_number: Int[Array, " L"],
  uniforms: Float[Array, " M"],
  *,
  config: PHDesignConfig,
  wh: Float[Array, ""],
  wsel: Float[Array, ""],
  cdf_order: Array | None = None,
) -> tuple[Array, Array, Array]:
  """Repeat full sweeps until one changes nothing or ``block_max_rounds`` is reached.

  Returns ``(seq, n_draws, rounds)``. Upstream's loop is ``while changed and rounds < max``, and ``rounds``
  counts the final sweep that changed nothing.
  """
  temperature = float(config.temperature)
  n_block = blocks.shape[1]
  vocab = table.shape[-1]
  length = seq.shape[0]
  max_rounds = int(config.block_max_rounds)
  n_uniform = uniforms.shape[0]

  def visit(
    carry: tuple[Array, Array, Array],
    xs: tuple[Array, Array],
  ) -> tuple[tuple[Array, Array, Array], None]:
    seq_now, draws, changed = carry
    block, valid = xs
    flat = block_objective(
      table,
      e_idx,
      seq_now,
      block,
      valid,
      pin_positions,
      pin_prot,
      pin_dep,
      pin_dep_valid,
      valid_tokens,
      rep_mask,
      residue_number,
      config=config,
      wh=wh,
      wsel=wsel,
    )
    if temperature > 0:
      uniform = uniforms[jnp.minimum(draws, n_uniform - 1)]
      choice = select_joint(flat, uniform, temperature, n_block=n_block, cdf_order=cdf_order)
      draws = draws + 1
    else:
      choice = jnp.argmin(flat).astype(jnp.int32)
    digits = _digits(choice, n_block, vocab)
    current = seq_now[jnp.where(valid, block, 0)]
    moved = jnp.any(valid & (digits != current))
    target = jnp.where(valid, block, length)
    seq_next = seq_now.at[target].set(digits.astype(seq_now.dtype), mode="drop")
    return (seq_next, draws, changed | moved), None

  def sweep(state: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
    seq_now, draws, rounds, _ = state
    init_changed = jnp.zeros((), dtype=bool)
    (seq_next, draws_next, changed), _ = lax.scan(
      visit,
      (seq_now, draws, init_changed),
      (blocks, block_valid),
    )
    return seq_next, draws_next, rounds + 1, changed

  def keep_going(state: tuple[Array, Array, Array, Array]) -> Array:
    _, _, rounds, changed = state
    return changed & (rounds < max_rounds)

  zero = jnp.zeros((), dtype=jnp.int32)
  init = (seq, zero, zero, jnp.ones((), dtype=bool))
  seq_final, draws_final, rounds_final, _ = lax.while_loop(keep_going, sweep, init)
  return seq_final, draws_final, rounds_final
