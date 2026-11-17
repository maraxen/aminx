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

The three objective knobs of debt #2621 are ported (graded by the ``protonpotts_ph_objective_knobs`` wave; wave pending):

- ``global_weight`` (upstream ``combined_unary`` 2273-2284, ``_global_protonation_dH`` 2666-2696, ``_block_zscales`` 2114-2119):
  adds ``(global_weight / sdGlob) * (g(S with p=a) - g(S))`` to the unary of member ``p``, with
  ``g = H(binder His/Asp/Glu -> protonated) - H(binder Asp/Glu -> deprotonated, His -> HIS-S)`` of the WORKING sequence (the
  residue classes are the NATIVE sequence's, chain A only; see :func:`global_masks`). SIGN: the weight multiplies ``g`` itself, so a
  POSITIVE ``global_weight`` MINIMISES ``g`` and favours the protonated state. Upstream's docstrings say the opposite; this follows
  the code. ``sdGlob`` is a third frozen z-scale, the pooled variance of the same unary over blocks where it varies.
- ``adjacent_repeat_weight`` (upstream 2257-2259, 2337-2351): ``+w`` per sequence-adjacent (residue number differs by 1) pair on the
  same canonical amino acid, raw (not z-scaled), chain A only; within the block as a pairwise term, against a fixed neighbour as a unary.
- ``self_weight`` (upstream ``stability_weights`` 285-288, ``reweighted`` 1610-1627, ``stability_scorer`` 1671-1685): scales the
  slot-0 field of the table by ``self_weight`` for the STABILITY term only (pairs at 1), in both the frozen ``sdH`` and the per-visit
  stability potentials. Selective, global and reported energies stay on the true table; ``self_weight == 1`` is an exact no-op.

``zscale_mode="single_mutation"`` (:func:`single_mutation_zscales`, upstream ``_block_mutation_tables`` 2006-2038 and the branch at 2243-2255)
freezes the same three scales from single-position deltas instead: over the designable rows, the population std of the finite
valid-token deltas of the UNWEIGHTED stability energy (``self_weight`` is not applied, unlike block mode), of the selective unary
(rows where it varies; ``1e-6`` when no row varies, since the all-zero table has std 0) and of the global unary (rows where it varies;
``1.0`` when the term is off or no row varies). Not ported browser-side (``ph_design.mjs`` only has the block graphs).

Not ported (refused with ``NotImplementedError`` at the top of :func:`block_descent` where a knob exists): unknown ``zscale_mode``/``sweep_order`` values. The visit
order ``sweep_order`` (``"position"``, ``"knn"``, ``"energy"``) is applied by ``ph_design`` before this function
is called, so ``plan_designable`` arrives already in visit order.``PHDesignConfig`` has no ``repetitive_window_gate_types`` field, so the
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
from xtrax.tiling import AxisSpec

from aminx.families.potts_mpnn.etab import potts_energy
from aminx.families.protonpotts_mpnn.ph_potentials import (
  block_stability_potentials,
  candidate_energies_at,
)
from aminx.families.protonpotts_mpnn.vocab import (
  PROTONPOTTS_V6,
  THREE_TO_ONE,
  canonical_letter,
  token_index,
)
from aminx.tiling.dispatch import make_axis_dispatch
from aminx.tiling.planner import plan_axis_strategy

if TYPE_CHECKING:
  from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig


class DescentResult(NamedTuple):
  seq: Int[Array, " L"]  # final sequence
  n_draws: Int[Array, ""]  # uniforms consumed (0 at temperature 0)
  rounds: Int[Array, ""]  # sweeps executed, including the final sweep that changed nothing
  zscales: Float[Array, " 3"]  # (sdH, sdSel, sdGlob) as upstream returns them; sdGlob is 1.0 unless global_weight != 0


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


# Visit orders ported from upstream ``_block_descent``. The order itself is computed by ``ph_design`` before the
# block table is built; ``block_descent`` only checks that the knob is one of these.
SWEEP_ORDERS = ("position", "knn", "energy")
ZSCALE_MODES = ("block", "single_mutation")


def _check_supported(config: PHDesignConfig) -> None:
  """Refuse upstream knobs this port does not implement, naming each one."""
  checks = (
    (
      config.zscale_mode not in ZSCALE_MODES,
      f"zscale_mode={config.zscale_mode!r} (only {', '.join(map(repr, ZSCALE_MODES))} are ported)",
    ),
    (
      config.sweep_order not in SWEEP_ORDERS,
      f"sweep_order={config.sweep_order!r} (only {', '.join(map(repr, SWEEP_ORDERS))} are ported)",
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


def _allowed_tokens(member: Bool[Array, "B 1"], valid_tokens: Bool[Array, " V"], vocab: int) -> Bool[Array, "B V"]:
  """``allowed[b, a]``: a real member may take any valid token; a padded member only token 0.

  Boolean algebra rather than ``jnp.where`` on booleans: the same values, but it lowers to ``And``/``Or``/``Not``,
  which ONNX Runtime implements, where it has no ``Where`` kernel for a BOOL operand (the browser export of the
  per-block graphs; Potts spec 261007_potts-onnx-export-scope.md section 8 met the same).
  """
  first_only = (jnp.arange(vocab) == 0)[None, :]
  return (member & valid_tokens[None, :]) | (~member & first_only)


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


# --- the three objective knobs of debt #2621 -------------------------------------------------------------------------

_CANON = tuple(canonical_letter(i) for i in range(PROTONPOTTS_V6.size))
_HIS_P, _HIS_S = token_index("HIS-P"), token_index("HIS-S")
_ASP_P, _ASP_D = token_index("ASP-P"), token_index("ASP-D")
_GLU_P, _GLU_D = token_index("GLU-P"), token_index("GLU-D")


class GlobalMasks(NamedTuple):
  """Positions the global term protonates / deprotonates: binder (chain A) positions whose NATIVE residue is His / Asp / Glu."""

  his: Bool[Array, " L"]
  asp: Bool[Array, " L"]
  glu: Bool[Array, " L"]


def global_masks(native: Int[Array, " L"], binder_mask: Bool[Array, " L"]) -> GlobalMasks:
  """Residue-class masks of upstream ``_global_protonation_dH`` (2683-2686).

  Upstream reads ``ctx.token_aa.res_name`` (the STRUCTURE's residue names, fixed for the run) ANDed with ``ctx.chainA_t``. Here the
  native sequence's canonical letter plays ``res_name`` (``HIS-P`` and ``H`` are both His), so ``native`` must be the sequence BEFORE
  the pins are set. The masks do not follow the working sequence: a position whose native residue is Asp is protonated by the
  global term whatever the design put there.
  """
  native = jnp.asarray(native, dtype=jnp.int32)
  binder = jnp.asarray(binder_mask, dtype=bool)

  def klass(letter: str) -> Array:
    return jnp.asarray([c == letter for c in _CANON], dtype=bool)[native] & binder

  return GlobalMasks(his=klass("H"), asp=klass("D"), glu=klass("E"))


def _global_sequences(seq: Int[Array, " L"], masks: GlobalMasks) -> tuple[Array, Array]:
  """``(sp, sd)``: ``seq`` with the class residues protonated, and with Asp/Glu deprotonated and His at HIS-S (upstream 2688-2692)."""
  sp = jnp.where(masks.his, _HIS_P, jnp.where(masks.asp, _ASP_P, jnp.where(masks.glu, _GLU_P, seq)))
  sd = jnp.where(masks.his, _HIS_S, jnp.where(masks.asp, _ASP_D, jnp.where(masks.glu, _GLU_D, seq)))
  return sp.astype(seq.dtype), sd.astype(seq.dtype)


def global_protonation_dH(  # noqa: N802 -- upstream's name
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
  masks: GlobalMasks,
) -> Float[Array, ""]:
  """``g = H(all binder titratable -> protonated) - H(-> deprotonated)``, upstream ``_global_protonation_dH`` (v6 branch).

  Lower ``g`` prefers the protonated state. Neutral His is the single ``HIS-S`` token, so upstream's mean over the neutral tokens
  has one term. Full Potts energies on ``table``.
  """
  seq = jnp.asarray(seq, dtype=jnp.int32)
  sp, sd = _global_sequences(seq, masks)
  valid = jnp.ones(seq.shape, dtype=bool)
  return potts_energy(table, e_idx, valid, sp) - potts_energy(table, e_idx, valid, sd)


def _global_rel(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
  block: Int[Array, " B"],
  block_valid: Bool[Array, " B"],
  masks: GlobalMasks,
) -> Float[Array, "B V"]:
  """``rel[b, a] = g(seq with block[b]=a) - g(seq)`` for each member alone, the rest held at ``seq`` (upstream ``dGlob``).

  ``g`` depends on ``seq[p]`` only if ``p`` is outside the three residue-class masks (a class position is overwritten in both
  ``sp`` and ``sd``), and then only through the candidate-energy row of ``p`` in ``sp`` and in ``sd``: ``H(x with p=a) - H(x with
  p=b) == E[p, a] - E[p, b]``. Each member is taken as a single-member block, so edges to the other members count as fixed context.
  Class positions get exactly 0, as upstream's ``g - glob0`` of an unchanged ``g``.
  """
  block_safe = jnp.where(block_valid, block, 0)
  sp, sd = _global_sequences(seq, masks)
  rows = []
  for b in range(block.shape[0]):
    pos, val = block_safe[b : b + 1], block_valid[b : b + 1]
    u_p, _ = block_stability_potentials(table, e_idx, sp, pos, val)
    u_d, _ = block_stability_potentials(table, e_idx, sd, pos, val)
    rows.append(u_p[0] - u_d[0])
  absolute = jnp.stack(rows)  # (B, V)
  cur = seq[block_safe]
  rel = absolute - jnp.take_along_axis(absolute, cur[:, None], axis=1)
  in_class = (masks.his | masks.asp | masks.glu)[block_safe]
  return jnp.where(in_class[:, None], 0.0, rel)


def stability_table(table: Float[Array, "L K V V"], self_weight: float) -> Float[Array, "L K V V"]:
  """The table the STABILITY term reads: slot 0 (the self field) scaled by ``self_weight``, pairs at 1.

  Upstream ``_PottsScorer.reweighted`` (1610-1627) via ``ctx.stability_scorer`` (1671-1685), which returns the unweighted scorer when
  ``w_self == w_pair`` (here ``self_weight == 1``). Same dtype as ``table``.
  """
  weight = float(self_weight)
  if weight == 1.0:
    return table
  return jnp.asarray(table).at[:, 0].multiply(weight)


def _same_canonical(dtype: np.dtype) -> Float[Array, "V V"]:
  """``same[a, b] = 1`` when tokens ``a`` and ``b`` are the same canonical amino acid (``ASP-P == ASP-D``); upstream 2259."""
  return jnp.asarray([[float(a == b) for b in _CANON] for a in _CANON], dtype=dtype)


def _adjacent_repeat(
  seq: Int[Array, " L"],
  block: Int[Array, " B"],
  block_valid: Bool[Array, " B"],
  residue_number: Int[Array, " L"],
  arw: float,
  dtype: np.dtype,
) -> tuple[Float[Array, "B V"], Float[Array, "B B V V"]]:
  """Adjacent-repeat unary and within-block pairwise terms (upstream 2337-2351).

  ``+arw`` per sequence-adjacent pair (residue numbers differing by exactly 1) on the same canonical amino acid. Both members in the
  block: pairwise ``arw * same`` (a). One member in the block and the neighbour fixed (anywhere outside the block, at its CURRENT
  token): unary ``arw * same[:, seq[q]]`` (b). Raw, not z-scaled. Neighbours are found through ``residue_number``, so, as for the
  repetitive window, non-binder positions must carry residue numbers no binder residue is adjacent to (upstream: chain A only).
  """
  length = seq.shape[0]
  n_block = block.shape[0]
  same = _same_canonical(dtype)
  block_safe = jnp.where(block_valid, block, 0)
  res_block = residue_number[block_safe]  # (B,)
  in_block = (
    jnp.zeros((length,), dtype=bool).at[jnp.where(block_valid, block, length)].set(True, mode="drop")
  )
  adjacent = jnp.abs(residue_number[None, :] - res_block[:, None]) == 1  # (B, L)
  fixed_nbr = (adjacent & ~in_block[None, :]).astype(dtype)  # (B, L)
  unary = arw * jnp.einsum("bl,vl->bv", fixed_nbr, same[:, seq])  # (B, V)

  close = jnp.abs(res_block[:, None] - res_block[None, :]) == 1  # (B, B)
  upper = jnp.triu(jnp.ones((n_block, n_block), dtype=bool), k=1)
  member_pair = block_valid[:, None] & block_valid[None, :] & close & upper
  pair = (arw * member_pair.astype(dtype))[:, :, None, None] * same[None, None, :, :]
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
  wglob: Float[Array, ""] | float = 0.0,
  glob_masks: GlobalMasks | None = None,
  stab_table: Float[Array, "L K V V"] | None = None,
) -> Float[Array, " S"]:
  """Flat block objective ``J`` over all ``V**B`` joint assignments of one block.

  ``J(x) - J(x0)`` equals ``F(x) - F(x0)`` for the objective in the module docstring, with every other position
  fixed at ``seq`` (pins must already be set in ``seq``). Row-major with block axis 0 slowest, so
  ``np.unravel_index(index, [V] * B)`` recovers the assignment. Invalid tokens are ``+inf`` on every axis, and
  padded members (``block_valid`` False) take only token 0, with zero contribution.

  The objective knobs of debt #2621: ``wglob`` is ``global_weight / sdGlob`` and ``glob_masks`` the :func:`global_masks`
  (required when ``config.global_weight != 0``, else ``ValueError``); ``stab_table`` is :func:`stability_table` (default: built
  from ``config.self_weight``); ``config.adjacent_repeat_weight`` needs nothing more.
  """
  _check_supported(config)
  want_global = float(config.global_weight) != 0.0
  if want_global and glob_masks is None:
    msg = "global_weight != 0 needs glob_masks (see global_masks)"
    raise ValueError(msg)
  table = jnp.asarray(table)
  stab = stability_table(table, config.self_weight) if stab_table is None else jnp.asarray(stab_table)
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

  unary, pair = block_stability_potentials(stab, e_idx, seq, block_safe, block_valid)
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
  unary_tot = wh * unary + wsel * sel_rel
  if want_global:
    assert glob_masks is not None  # noqa: S101 -- narrowed above
    unary_tot = unary_tot + wglob * _global_rel(table, e_idx, seq, block, block_valid, glob_masks)
  unary_tot = unary_tot + rep_unary
  pair_tot = wh * pair + rep_pair
  arw = float(config.adjacent_repeat_weight)
  if arw != 0.0:
    adj_unary, adj_pair = _adjacent_repeat(seq, block, block_valid, residue_number, arw, table.dtype)
    unary_tot = unary_tot + adj_unary
    pair_tot = pair_tot + adj_pair
  unary_tot = jnp.where(member, unary_tot, 0.0)
  allowed = _allowed_tokens(member, valid_tokens, vocab)
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


# The independent blocks of one design. Declared here and not in ``aminx.tiling.axes``: that registry is imported by
# the host runner, so an edit there would stale every ledger row for one constant. Move it there with the next
# wave that touches shared files. ``default_batch_size=0`` lets the planner choose the tile.
_PH_BLOCKS = AxisSpec(
  name="n_ph_blocks",
  cardinality=64,
  default_batch_size=0,
  tile_granularity=1,
  heterogeneous=False,
)


def _block_activation_bytes(vocab: int, n_block: int, dtype: np.dtype) -> float:
  """Live bytes one block holds while its z-scales are computed (a few ``V**B`` joints plus the pair tensors)."""
  joints = 4 * vocab**n_block
  pairs = 2 * n_block * n_block * vocab * vocab
  return float((joints + pairs + 4 * n_block * vocab) * np.dtype(dtype).itemsize)


def block_zstats(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
  block: Int[Array, " B"],
  valid: Bool[Array, " B"],
  pin_positions: Int[Array, " P"],
  pin_prot: Int[Array, " P"],
  pin_dep: Int[Array, "P D"],
  pin_dep_valid: Bool[Array, "P D"],
  valid_tokens: Bool[Array, " V"],
  *,
  stab_table: Float[Array, "L K V V"] | None = None,
) -> tuple[Array, Array, Array, Array]:
  """z-scale statistics of ONE block: ``(var_h, has_h, var_s, has_s)``. ``seq`` must already carry the pins.

  The per-block part of :func:`block_zscales`, lifted out so the browser can export it as its own graph and pool
  the blocks itself. ``stab_table`` (default ``table``) is :func:`stability_table`, which only the stability variance reads.
  """
  stab = table if stab_table is None else stab_table
  vocab = table.shape[-1]
  dtype = table.dtype
  n_block = block.shape[0]
  block_safe = jnp.where(valid, block, 0)
  member = valid[:, None]
  allowed = _allowed_tokens(member, valid_tokens, vocab)

  unary, pair = block_stability_potentials(stab, e_idx, seq, block_safe, valid)
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
  return var_h, has_h, var_s, has_s


def block_zstats_global(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
  block: Int[Array, " B"],
  valid: Bool[Array, " B"],
  valid_tokens: Bool[Array, " V"],
  glob_masks: GlobalMasks,
) -> tuple[Array, Array]:
  """``(var_g, has_g)`` of ONE block: the variance of the unary global term, as the selective one (upstream 2114-2119).

  The block's joint is the broadcast sum of its members' ``g(S with p=a) - g(S)`` rows over the valid tokens; it contributes only if
  some row is non-zero and the joint has more than one finite entry. ``seq`` must already carry the pins.
  """
  vocab = table.shape[-1]
  n_block = block.shape[0]
  member = valid[:, None]
  allowed = _allowed_tokens(member, valid_tokens, vocab)
  rel = _global_rel(table, e_idx, seq, block, valid, glob_masks)
  rel = jnp.where(member & valid_tokens[None, :], rel, 0.0)
  varies = jnp.sum(jnp.abs(rel)) > 0
  joint = _joint(rel, jnp.zeros((n_block, n_block, vocab, vocab), dtype=table.dtype), allowed, n_block, vocab)
  var_g, count_g = _finite_var(joint)
  return var_g, varies & (count_g > 1)


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
  *,
  stab_table: Float[Array, "L K V V"] | None = None,
  glob_masks: GlobalMasks | None = None,
) -> Float[Array, " 3"]:
  """Frozen ``(sdH, sdSel, sdGlob)`` of upstream ``_block_zscales`` (``zscale_mode="block"``).

  Pins are set in ``seq`` first, as upstream does, so the caller may pass the raw seed sequence. ``sdGlob`` is pooled only when
  ``glob_masks`` is given (upstream ``want_global``), else 1.0. ``stab_table`` is :func:`stability_table` (default ``table``).
  """
  seq = _set_pins(seq, pin_positions, pin_prot)
  vocab = table.shape[-1]
  dtype = table.dtype
  n_block = blocks.shape[1]
  block_safe_all = jnp.where(block_valid, blocks, 0)

  def per_block(xs: tuple[Array, Array]) -> tuple[Array, ...]:
    block, valid = xs
    stats = block_zstats(
      table, e_idx, seq, block, valid, pin_positions, pin_prot, pin_dep, pin_dep_valid, valid_tokens,
      stab_table=stab_table,
    )
    if glob_masks is None:
      return stats
    return (*stats, *block_zstats_global(table, e_idx, seq, block, valid, valid_tokens, glob_masks))

  # The blocks are independent, so this is a map, not a scan. Which map (one vmap, or tiles of it) is xtrax's call:
  # each block holds a V**B joint, and the planner demotes Vmap to SafeMap when all N of them would not fit.
  n_blocks = int(blocks.shape[0])
  strategy = plan_axis_strategy(
    _PH_BLOCKS,
    n_blocks,
    None,
    activation_bytes_per_element=_block_activation_bytes(vocab, n_block, dtype),
  )
  iterator = make_axis_dispatch(strategy, axis=_PH_BLOCKS.name)
  outs = iterator(per_block, (block_safe_all, block_valid))
  sd_h = _pool(outs[0], outs[1], dtype)
  sd_s = _pool(outs[2], outs[3], dtype)
  sd_g = _pool(outs[4], outs[5], dtype) if glob_masks is not None else jnp.asarray(1.0, dtype=dtype)
  return jnp.stack([sd_h, sd_s, sd_g])


def _delta_std(deltas: Array, rows: Array, valid_tokens: Array) -> Array:
  """Upstream ``_zscale`` (1995-2003): population std of the finite entries of ``deltas[rows][:, valid_tokens]``.

  Floored at ``1e-6``; ``1.0`` when no finite entry is selected.
  """
  finite = rows[:, None] & valid_tokens[None, :] & jnp.isfinite(deltas)
  count = jnp.sum(finite)
  safe = jnp.maximum(count, 1).astype(deltas.dtype)
  vals = jnp.where(finite, deltas, 0.0)
  mean = jnp.sum(vals) / safe
  var = jnp.sum(jnp.where(finite, (vals - mean) ** 2, 0.0)) / safe
  sd = jnp.maximum(jnp.sqrt(var), 1e-6)
  return jnp.where(count > 0, sd, 1.0).astype(deltas.dtype)


def single_mutation_zscales(
  table: Float[Array, "L K V V"],
  e_idx: Int[Array, "L K"],
  seq: Int[Array, " L"],
  designable: Int[Array, " N"],
  pin_positions: Int[Array, " P"],
  pin_prot: Int[Array, " P"],
  pin_dep: Int[Array, "P D"],
  pin_dep_valid: Bool[Array, "P D"],
  valid_tokens: Bool[Array, " V"],
  *,
  glob_masks: GlobalMasks | None = None,
) -> Float[Array, " 3"]:
  """Frozen ``(sdH, sdSel, sdGlob)`` of upstream ``zscale_mode="single_mutation"`` (2006-2038, branch 2243-2255).

  ``dH[n, a] = E[i, a] - E[i, cur]`` of the UNWEIGHTED ``table`` (upstream ``ctx.scorer``; ``self_weight`` does not apply), ``dSel``
  and ``dGlob`` the change of the selective / global term when ``designable[n]`` alone is set to ``a``, all pins at their protonated
  token. The current-token entry is 0 and counts. ``sdSel`` and ``sdGlob`` use only the rows where the term varies; ``sdSel`` is
  ``1e-6`` when none does and ``sdGlob`` is ``1.0`` when ``glob_masks`` is None (upstream ``want_global`` False) or none does.
  Pins are set in ``seq`` first.
  """
  seq = _set_pins(seq, pin_positions, pin_prot)
  designable = designable.astype(jnp.int32)
  n_rows = designable.shape[0]
  dtype = table.dtype
  all_rows = jnp.ones((n_rows,), dtype=bool)
  cur = seq[designable]

  e_abs = candidate_energies_at(table, e_idx, seq, designable)
  d_h = e_abs - jnp.take_along_axis(e_abs, cur[:, None], axis=1)
  sd_h = _delta_std(d_h, all_rows, valid_tokens)

  sel_abs = _selective_unary(table, e_idx, pin_positions, pin_prot, pin_dep, pin_dep_valid, designable, all_rows)
  d_sel = jnp.where(valid_tokens[None, :], sel_abs - jnp.take_along_axis(sel_abs, cur[:, None], axis=1), 0.0)
  sel_rows = jnp.sum(jnp.abs(d_sel), axis=1) > 0
  # No varying row: upstream falls back to the whole (all-zero) table, whose std is 0 -> the 1e-6 floor, not 1.0.
  sd_s = _delta_std(d_sel, jnp.where(jnp.any(sel_rows), sel_rows, all_rows), valid_tokens)

  if glob_masks is None:
    sd_g = jnp.asarray(1.0, dtype=dtype)
  else:

    def one_row(pos: Array) -> Array:
      return _global_rel(table, e_idx, seq, pos[None], jnp.ones((1,), dtype=bool), glob_masks)[0]

    d_glob = jnp.where(valid_tokens[None, :], lax.map(one_row, designable), 0.0)
    glob_rows = jnp.sum(jnp.abs(d_glob), axis=1) > 0
    sd_g = jnp.where(jnp.any(glob_rows), _delta_std(d_glob, glob_rows, valid_tokens), 1.0).astype(dtype)
  return jnp.stack([sd_h, sd_s, sd_g]).astype(dtype)


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
  binder_mask: Bool[Array, " L"] | None = None,
) -> DescentResult:
  """Gauss-Seidel block descent over the designable positions, as upstream ``_block_descent`` 2193-2387.

  Pins are set to their protonated token first. Z-scales are frozen once from that sequence. Each sweep visits
  the designable rows in order; each visit commits the exact argmin (or an inverse-CDF sample) of its block
  objective, and later visits see the updated sequence. Sweeps repeat while the previous one changed something,
  up to ``config.block_max_rounds``.

  ``cdf_order`` (default None) sets the token order of the inverse-CDF sampler; see ``select_joint``. It exists
  for parity with an upstream dump and is left unset in production.

  ``binder_mask (L,)`` (the chain-A positions) is required when ``config.global_weight != 0``: with the NATIVE ``seq`` (before the pins
  are set) it fixes the His/Asp/Glu positions the global term titrates (:func:`global_masks`). Else it is unused.

  ``plan_designable`` only fixes the row count and order (row ``n`` visits ``plan_designable[n]``, which is
  ``blocks[n, 0]``). Unsupported upstream knobs raise ``NotImplementedError`` before any work is done.
  """
  _check_supported(config)
  gw = float(config.global_weight)
  if gw != 0.0 and binder_mask is None:
    msg = "global_weight != 0 needs binder_mask (chain-A positions) to fix the residues the global term titrates"
    raise ValueError(msg)
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
  glob_masks = global_masks(seq, binder_mask) if gw != 0.0 and binder_mask is not None else None  # native, pre-pin
  seq = _set_pins(seq, pin_positions, pin_prot)
  stab_table = stability_table(table, config.self_weight)
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
  if config.zscale_mode == "single_mutation":
    zscales = single_mutation_zscales(
      table,
      e_idx,
      seq,
      blocks[:, 0],
      pin_positions,
      pin_prot,
      pin_dep,
      pin_dep_valid,
      valid_tokens,
      glob_masks=glob_masks,
    )
  else:
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
      stab_table=stab_table,
      glob_masks=glob_masks,
    )
  lam = float(config.combined_lambda)
  wh = (1.0 - lam) / zscales[0]
  wsel = lam / zscales[1]
  # SIGN: upstream's ``wGlob = gw / sdGlob`` multiplies ``(g - g0)`` and the descent MINIMISES the objective, so a POSITIVE
  # global_weight minimises g = H(protonated) - H(deprotonated), i.e. favours the PROTONATED state (upstream's docstrings say the
  # reverse). Follows the code; ``wglob`` is 0 (term absent) when global_weight == 0.
  wglob = gw / zscales[2] if gw != 0.0 else 0.0
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
    wglob=wglob,
    glob_masks=glob_masks,
    stab_table=stab_table,
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
  wglob: Float[Array, ""] | float = 0.0,
  glob_masks: GlobalMasks | None = None,
  stab_table: Float[Array, "L K V V"] | None = None,
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
      wglob=wglob,
      glob_masks=glob_masks,
      stab_table=stab_table,
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
