"""Autoregressive PottsMPNN decode.

``PottsARDecode`` reuses the stock ``DecoderLayer`` stack. One structure is
stepped with ``lax.scan`` over tied groups. Untied proteins are the same path
with every group a singleton.

PSSMMix follows upstream ``decoder`` (``potts_mpnn_utils.py:1391-1404``) in
order: softmax of ``logits / T - omit·1e8 + bias / T + bias_by_res / T``, then
the PSSM bias mix when ``(pssm_bias_flag and coef is non-empty) or pssm_bias
is non-empty``, then optional log-odds renormalisation ``p·(mask + 0.001)``,
then ``omit_AA_mask`` renormalisation, then the inverse-CDF draw of §7.1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from aminx.families.potts_mpnn.alphabet import POTTS_MPNN
from aminx.utils.concatenate import concatenate_neighbor_nodes

if TYPE_CHECKING:
  from aminx.model.decoder import DecoderLayer

_BIG = jnp.iinfo(jnp.int32).max
_OMIT_SCALE = 1.0e8


class ARResult(NamedTuple):
  """One decoded sequence and the group schedule that produced it."""

  sequence: Int[Array, " L"]
  rank_flat: Int[Array, " L"]
  decoding_order: Int[Array, " L"]
  h_v_stack: Float[Array, "layers L H"]


def _as_dtype(value: float, dtype: jnp.dtype) -> Array:
  """Constant scalar as ``dtype``. Outside traced callables so L-DRV R4 stays clear."""
  return jnp.asarray(value, dtype=dtype)


def floor_temperature(temperature: float) -> float:
  """Upstream ``sample_seqs.py:38-39``: ``0`` becomes ``1e-6`` before the draw."""
  if temperature == 0.0:
    return 1.0e-6
  return float(temperature)


def floor_temperature_array(temperature: Float[Array, ""], dtype: jnp.dtype) -> Float[Array, ""]:
  """``floor_temperature`` for a traced scalar: ``0`` becomes ``1e-6``, as ``dtype``.

  Lets an exported graph take temperature as a runtime input; a Python float still goes
  through ``floor_temperature`` and gives the same value.
  """
  value = jnp.asarray(temperature, dtype=dtype)
  return jnp.where(value == 0, jnp.asarray(1.0e-6, dtype=dtype), value)


def categorical_draw(probs: Float[Array, " V"], uniform: Float[Array, ""]) -> Int[Array, ""]:
  """Inverse-CDF draw shared with the oracle shim.

  ``c = cumsum(p)``; ``i = searchsorted(c, u·c[-1], side='right')``;
  ``i = min(i, last index with p > 0)``. A zero-probability tail is never
  returned. ``u = 0`` lands on the first positive bin.
  """
  cdf = jnp.cumsum(probs, axis=-1)
  target = uniform.astype(cdf.dtype) * cdf[-1]
  index = jnp.sum(cdf <= target).astype(jnp.int32)
  positions = jnp.arange(probs.shape[-1], dtype=jnp.int32)
  last = jnp.max(jnp.where(probs > 0, positions, jnp.int32(0)))
  return jnp.minimum(index, last)


def mask_refine_x(
  logits: Float[Array, " V"],
  x_index: int = POTTS_MPNN.x_index,
) -> Float[Array, " V"]:
  """Make the ``X`` index structurally undrawable.

  ``x_index`` defaults to the shipped alphabet's X (index 20, ``POTTS_MPNN``).
  It must be a Python ``int`` (static), not a traced value.

  Upstream refine builds 20 candidates and then draws from 21 letters, raising
  ``IndexError`` when ``X`` is selected (debt #2260). The f64 oracles mask that
  logit (``refine_omit_index = 20``, ``constant[20] = 1``). Subtracting ``1e8``
  before the softmax keeps the draw inside ``0..19`` even when the unmasked
  ``X`` logit is the largest. This port does not reproduce the ``IndexError``.
  """
  x_slot = jnp.int32(x_index)
  return logits.at[x_slot].add(jnp.asarray(-_OMIT_SCALE, dtype=logits.dtype))


def pssm_mix(
  logits_over_t: Float[Array, " V"],
  temperature: Float[Array, ""],
  omit: Float[Array, " V"],
  bias: Float[Array, " V"],
  bias_by_res: Float[Array, " V"],
  pssm_coef: Float[Array, ""],
  pssm_bias: Float[Array, " V"],
  pssm_multi: Float[Array, ""],
  pssm_log_odds_mask: Float[Array, " V"],
  omit_aa_mask: Float[Array, " V"],
  *,
  apply_pssm_bias: Bool[Array, ""],
  apply_log_odds: Bool[Array, ""],
  apply_omit_aa: Bool[Array, ""],
) -> Float[Array, " V"]:
  """PSSMMix in upstream order. Gates are scalar bools, so the math stays pure."""
  raw = (
    logits_over_t
    - omit * jnp.asarray(_OMIT_SCALE, dtype=logits_over_t.dtype)
    + bias / temperature
    + bias_by_res / temperature
  )
  probs = jax.nn.softmax(raw)
  mixed = (1.0 - pssm_coef * pssm_multi) * probs + (pssm_coef * pssm_multi) * pssm_bias
  probs = jnp.where(apply_pssm_bias, mixed, probs)
  logged = probs * (pssm_log_odds_mask + jnp.asarray(0.001, dtype=probs.dtype))
  logged = logged / jnp.sum(logged, axis=-1, keepdims=True)
  probs = jnp.where(apply_log_odds, logged, probs)
  omitted = probs * (1.0 - omit_aa_mask)
  omitted = omitted / jnp.sum(omitted, axis=-1, keepdims=True)
  return jnp.where(apply_omit_aa, omitted, probs)


def neighbor_valid(e_idx: Int[Array, "L K"], pad_valid: Bool[Array, " L"]) -> Bool[Array, "L K"]:
  """True when the neighbour index is in range and ``pad_valid``."""
  length = pad_valid.shape[0]
  e_idx = e_idx.astype(jnp.int32)
  in_range = (e_idx >= 0) & (e_idx < length)
  safe = jnp.clip(e_idx, 0, length - 1)
  return in_range & pad_valid[safe]


def readout(w_out: eqx.nn.Linear, hidden: Float[Array, " H"]) -> Float[Array, " V"]:
  """``W_out`` as a matmul in ``hidden.dtype``. No ``vmap`` (L-DRV R1).

  Parameters stay in the module dtype (float64 when x64 built the layer). The
  logits carry is the activation dtype, so the matmul must not promote.
  """
  dtype = hidden.dtype
  logits = hidden @ w_out.weight.astype(dtype).T
  if w_out.bias is not None:
    logits = logits + w_out.bias.astype(dtype)
  return logits


def embed_at(w_s: eqx.nn.Embedding, token: Int[Array, ""]) -> Float[Array, " H"]:
  """One embedding row. Callers pass a token inside ``0 .. vocab-1``."""
  return w_s.weight[token.astype(jnp.int32)]


def schedule_groups(
  tie_groups: Int[Array, "G M"],
  pad_valid: Bool[Array, " L"],
  randn: Float[Array, " L"],
  chain_mask: Float[Array, " L"],
  chain_m_pos: Float[Array, " L"],
  present: Float[Array, " L"],
  decoding_order: Int[Array, " L"] | None,
) -> tuple[Int[Array, " G"], Int[Array, " G"], Int[Array, " L"], Int[Array, " L"]]:
  """Group visit order and ``rank_flat`` from spec §4.3.

  Injected ``decoding_order`` is a permutation of ``0 .. L_pad-1`` (real rows
  first, then ``arange(L_total, L_pad)``). Otherwise the order is
  ``argsort((chain_mask·chain_M_pos·present + 1e-4)·|randn|)`` with pad rows
  sent last.
  """
  length = pad_valid.shape[0]
  if decoding_order is None:
    key = (chain_mask * chain_m_pos * present + jnp.asarray(1.0e-4, dtype=randn.dtype)) * jnp.abs(
      randn,
    )
    key = jnp.where(pad_valid, key, jnp.asarray(jnp.inf, dtype=key.dtype))
    order = jnp.argsort(key).astype(jnp.int32)
  else:
    order = decoding_order.astype(jnp.int32)
  rank = jnp.argsort(order).astype(jnp.int32)
  valid_member = tie_groups >= 0
  safe_member = jnp.where(valid_member, tie_groups, jnp.int32(0))
  group_key = jnp.where(valid_member, rank[safe_member], _BIG).min(axis=1)
  group_order = jnp.argsort(group_key, stable=True).astype(jnp.int32)
  size = valid_member.sum(axis=1).astype(jnp.int32)
  size_sorted = size[group_order]
  starts = jnp.cumsum(size_sorted) - size_sorted
  offset = jnp.zeros_like(size).at[group_order].set(starts)
  member_index = jnp.cumsum(valid_member, axis=1).astype(jnp.int32) - 1
  scattered = jnp.where(valid_member, tie_groups, jnp.int32(length))
  rank_flat = jnp.full((length,), _BIG, dtype=jnp.int32)
  rank_flat = rank_flat.at[scattered].set(offset[:, None] + member_index, mode="drop")
  rank_flat = jnp.where(
    pad_valid,
    rank_flat,
    jnp.int32(length) + jnp.arange(length, dtype=jnp.int32),
  )
  return group_order, size, rank_flat, order


def forward_context(
  h_v: Float[Array, "L H"],
  h_e: Float[Array, "L K H"],
  e_idx: Int[Array, "L K"],
  rank_flat: Int[Array, " L"],
  present: Float[Array, " L"],
) -> tuple[Float[Array, "L K"], Float[Array, "L K D"]]:
  """AR ``mask_bw`` and ``h_EXV_fw``. ``m = present`` (not ``chain_M_pos``)."""
  e_idx = e_idx.astype(jnp.int32)
  attended = rank_flat[e_idx] < rank_flat[:, None]
  mask_attend = attended.astype(present.dtype)
  mask_bw = present[:, None] * mask_attend
  mask_fw = present[:, None] * (1.0 - mask_attend)
  zeros = jnp.zeros_like(h_v)
  h_ex = concatenate_neighbor_nodes(zeros, h_e, e_idx)
  h_exv = concatenate_neighbor_nodes(h_v, h_ex, e_idx)
  return mask_bw, mask_fw[..., None] * h_exv


def update_row(
  layers: tuple[DecoderLayer, ...],
  h_v_stack: Float[Array, "layers L H"],
  h_s: Float[Array, "L H"],
  h_e: Float[Array, "L K H"],
  e_idx: Int[Array, "L K"],
  mask_bw: Float[Array, "L K"],
  h_exv_fw: Float[Array, "L K D"],
  present: Float[Array, " L"],
  nbr_valid: Bool[Array, "L K"],
  position: Int[Array, ""],
) -> Float[Array, "layers L H"]:
  """One residue, every decoder layer, scattering row ``position``."""
  position = position.astype(jnp.int32)
  h_e_row = h_e[position][None]
  e_row = e_idx[position][None]
  h_es = concatenate_neighbor_nodes(h_s, h_e_row, e_row)
  bw_row = mask_bw[position][None]
  fw_row = h_exv_fw[position][None]
  valid_row = nbr_valid[position][None]
  present_row = present[position][None]
  for index, layer in enumerate(layers):
    gathered = concatenate_neighbor_nodes(h_v_stack[index], h_es, e_row)
    context = bw_row[..., None] * gathered + fw_row
    updated = layer(
      h_v_stack[index][position][None],
      context,
      present_row,
      attention_mask=valid_row,
      inference=True,
    )
    h_v_stack = h_v_stack.at[index + 1, position].set(updated[0])
  return h_v_stack


class PottsARDecode(eqx.Module):
  """Incremental per-position decode. Fields are the MPNN submodules themselves."""

  layers: tuple[DecoderLayer, ...]
  w_s_embed: eqx.nn.Embedding
  w_out: eqx.nn.Linear

  def __call__(
    self,
    h_v: Float[Array, "L H"],
    h_e: Float[Array, "L K H"],
    e_idx: Int[Array, "L K"],
    present: Float[Array, " L"],
    pad_valid: Bool[Array, " L"],
    s_true: Int[Array, " L"],
    chain_mask: Float[Array, " L"],
    chain_m_pos: Float[Array, " L"],
    tie_groups: Int[Array, "G M"],
    tied_beta: Float[Array, " L"],
    randn: Float[Array, " L"],
    uniforms: Float[Array, " draws"],
    omit: Float[Array, " V"],
    bias: Float[Array, " V"],
    bias_by_res: Float[Array, "L V"],
    pssm_coef: Float[Array, " L"],
    pssm_bias: Float[Array, "L V"],
    pssm_log_odds_mask: Float[Array, "L V"],
    omit_aa_mask: Float[Array, "L V"],
    *,
    temperature: float | Float[Array, ""],
    pssm_multi: float,
    pssm_bias_flag: bool,
    pssm_log_odds_flag: bool,
    decoding_order: Int[Array, " L"] | None = None,
  ) -> ARResult:
    """Decode one structure. ``uniforms[k]`` is the k-th multinomial call.

    ``temperature`` may be a Python float or a scalar array (a runtime input of an
    exported graph); both are floored the same way.
    """
    if isinstance(temperature, (int, float)):
      temperature_value = _as_dtype(floor_temperature(temperature), h_v.dtype)
    else:
      temperature_value = floor_temperature_array(temperature, h_v.dtype)
    group_order, size, rank_flat, _order = schedule_groups(
      tie_groups.astype(jnp.int32),
      pad_valid,
      randn.astype(h_v.dtype),
      chain_mask.astype(h_v.dtype),
      chain_m_pos.astype(h_v.dtype),
      present.astype(h_v.dtype),
      None if decoding_order is None else decoding_order.astype(jnp.int32),
    )
    mask_bw, h_exv_fw = forward_context(h_v, h_e, e_idx, rank_flat, present.astype(h_v.dtype))
    nbr = neighbor_valid(e_idx, pad_valid)
    n_layers = len(self.layers)
    h_v_stack = jnp.concatenate(
      [h_v[None], jnp.zeros((n_layers, *h_v.shape), dtype=h_v.dtype)],
      axis=0,
    )
    sequence = jnp.zeros(h_v.shape[:1], dtype=jnp.int32)
    h_s = jnp.zeros_like(h_v)
    apply_bias = jnp.bool_((pssm_bias_flag and pssm_coef.size > 0) or pssm_bias.size > 0)
    apply_log_odds = jnp.bool_(pssm_log_odds_flag)
    apply_omit = jnp.bool_(omit_aa_mask.size > 0)
    multi = _as_dtype(pssm_multi, h_v.dtype)

    def group_step(
      carry: tuple[Array, Array, Array, Array],
      g_step: Array,
    ) -> tuple[tuple[Array, Array, Array, Array], None]:
      sequence_c, h_s_c, h_v_c, cursor = carry
      group = group_order[g_step.astype(jnp.int32)]
      group_size = size[group]

      def skip(
        state: tuple[Array, Array, Array, Array],
      ) -> tuple[Array, Array, Array, Array]:
        return state

      def body(
        state: tuple[Array, Array, Array, Array],
      ) -> tuple[Array, Array, Array, Array]:
        return _decode_group(
          self,
          state,
          group=group,
          group_size=group_size,
          tie_groups=tie_groups.astype(jnp.int32),
          tied_beta=tied_beta.astype(h_v.dtype),
          present=present.astype(h_v.dtype),
          s_true=s_true.astype(jnp.int32),
          chain_mask=chain_mask.astype(h_v.dtype),
          chain_m_pos=chain_m_pos.astype(h_v.dtype),
          h_e=h_e,
          e_idx=e_idx.astype(jnp.int32),
          mask_bw=mask_bw,
          h_exv_fw=h_exv_fw,
          nbr=nbr,
          uniforms=uniforms.astype(h_v.dtype),
          temperature=temperature_value,
          omit=omit.astype(h_v.dtype),
          bias=bias.astype(h_v.dtype),
          bias_by_res=bias_by_res.astype(h_v.dtype),
          pssm_coef=pssm_coef.astype(h_v.dtype),
          pssm_bias=pssm_bias.astype(h_v.dtype),
          pssm_log_odds_mask=pssm_log_odds_mask.astype(h_v.dtype),
          omit_aa_mask=omit_aa_mask.astype(h_v.dtype),
          apply_pssm_bias=apply_bias,
          apply_log_odds=apply_log_odds,
          apply_omit_aa=apply_omit,
          pssm_multi=multi,
        )

      sequence_c, h_s_c, h_v_c, cursor = jax.lax.cond(
        group_size == 0,
        skip,
        body,
        (sequence_c, h_s_c, h_v_c, cursor),
      )
      return (sequence_c, h_s_c, h_v_c, cursor), None

    init = (sequence, h_s, h_v_stack, jnp.int32(0))
    steps = jnp.arange(group_order.shape[0], dtype=jnp.int32)
    (sequence, _h_s, h_v_stack, _cursor), _rest = jax.lax.scan(group_step, init, steps)
    flat_order = jnp.argsort(rank_flat).astype(jnp.int32)
    return ARResult(
      sequence=sequence,
      rank_flat=rank_flat,
      decoding_order=flat_order,
      h_v_stack=h_v_stack,
    )


def _decode_group(
  module: PottsARDecode,
  state: tuple[Array, Array, Array, Array],
  *,
  group: Int[Array, ""],
  group_size: Int[Array, ""],
  tie_groups: Int[Array, "G M"],
  tied_beta: Float[Array, " L"],
  present: Float[Array, " L"],
  s_true: Int[Array, " L"],
  chain_mask: Float[Array, " L"],
  chain_m_pos: Float[Array, " L"],
  h_e: Float[Array, "L K H"],
  e_idx: Int[Array, "L K"],
  mask_bw: Float[Array, "L K"],
  h_exv_fw: Float[Array, "L K D"],
  nbr: Bool[Array, "L K"],
  uniforms: Float[Array, " draws"],
  temperature: Float[Array, ""],
  omit: Float[Array, " V"],
  bias: Float[Array, " V"],
  bias_by_res: Float[Array, "L V"],
  pssm_coef: Float[Array, " L"],
  pssm_bias: Float[Array, "L V"],
  pssm_log_odds_mask: Float[Array, "L V"],
  omit_aa_mask: Float[Array, "L V"],
  apply_pssm_bias: Bool[Array, ""],
  apply_log_odds: Bool[Array, ""],
  apply_omit_aa: Bool[Array, ""],
  pssm_multi: Float[Array, ""],
) -> tuple[Array, Array, Array, Array]:
  """One tied group. A ``present==0`` member stops the layer updates."""
  sequence, h_s, h_v_stack, cursor = state
  members = tie_groups[group]
  vocab = omit.shape[0]
  logits0 = jnp.zeros((vocab,), dtype=temperature.dtype)
  init = (h_v_stack, logits0, jnp.zeros((), dtype=jnp.bool_), jnp.int32(0))

  def member_step(
    carry: tuple[Array, Array, Array, Array],
    member_slot: Array,
  ) -> tuple[tuple[Array, Array, Array, Array], None]:
    h_v_c, logits, stopped, s_star = carry
    slot = member_slot.astype(jnp.int32)
    position = members[slot]
    valid = position >= 0
    safe = jnp.clip(position, 0, present.shape[0] - 1)
    masked = valid & (present[safe] == 0)
    start = masked & ~stopped
    stopped_now = stopped | masked
    s_star = jnp.where(start, s_true[safe], s_star)
    run = valid & ~stopped_now

    def run_member(operand: tuple[Array, Array]) -> tuple[Array, Array]:
      h_v_run, logits_run = operand
      h_v_run = update_row(
        module.layers,
        h_v_run,
        h_s,
        h_e,
        e_idx,
        mask_bw,
        h_exv_fw,
        present,
        nbr,
        safe,
      )
      hidden = h_v_run[-1, safe]
      # Python ``/ temperature`` would keep ``logits`` dtype; ``temperature`` is
      # already that dtype. Divide by the integer group size so a strong-typed
      # float scalar cannot promote the logits carry.
      extra = tied_beta[safe] * (readout(module.w_out, hidden) / temperature) / group_size
      return h_v_run, logits_run + extra

    def hold(operand: tuple[Array, Array]) -> tuple[Array, Array]:
      return operand

    h_v_c, logits = jax.lax.cond(run, run_member, hold, (h_v_c, logits))
    return (h_v_c, logits, stopped_now, s_star), None

  slots = jnp.arange(members.shape[0], dtype=jnp.int32)
  (h_v_stack, logits, stopped, s_star), _unused = jax.lax.scan(member_step, init, slots)
  last = members[group_size - 1]
  last_safe = jnp.clip(last, 0, present.shape[0] - 1)

  def stopped_token(_logits: Array) -> tuple[Array, Array]:
    return s_star, cursor

  def drawn_token(logits_in: Array) -> tuple[Array, Array]:
    probs = pssm_mix(
      logits_in,
      temperature,
      omit,
      bias,
      bias_by_res[last_safe],
      pssm_coef[last_safe],
      pssm_bias[last_safe],
      pssm_multi,
      pssm_log_odds_mask[last_safe],
      omit_aa_mask[last_safe],
      apply_pssm_bias=apply_pssm_bias,
      apply_log_odds=apply_log_odds,
      apply_omit_aa=apply_omit_aa,
    )
    drawn = categorical_draw(probs, uniforms[cursor])
    designed = chain_mask[last_safe] * chain_m_pos[last_safe] * present[last_safe]
    token = jnp.where(designed == 0, s_true[last_safe], drawn).astype(jnp.int32)
    return token, cursor + jnp.int32(1)

  token, cursor = jax.lax.cond(stopped, stopped_token, drawn_token, logits)
  valid_member = members >= 0
  length = sequence.shape[0]
  positions = jnp.where(valid_member, members, jnp.int32(length))
  sequence = sequence.at[positions].set(token, mode="drop")
  embedded = embed_at(module.w_s_embed, token.astype(jnp.int32))
  embedded_rows = jnp.broadcast_to(embedded, (positions.shape[0], embedded.shape[0]))
  h_s = h_s.at[positions].set(embedded_rows, mode="drop")
  return sequence, h_s, h_v_stack, cursor
