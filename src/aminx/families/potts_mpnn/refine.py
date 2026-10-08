"""Potts sequence refinement.

``potts`` is one sweep. ``potts_converge`` is a ``lax.while_loop`` whose stop
test is ``(ener_delta != 0) & (iters < max_iters)`` with ``max_iters`` 1000,
matching ``run_utils.py``. ``nodes`` is one Gibbs sweep whose attention is the
self-slot mask, not the autoregressive order.

``X`` (model index 20) is masked before every refine draw. See
``mask_refine_x``. Upstream raises ``IndexError`` when that index is drawn;
this port does not.

Tied groups keep the upstream quirks: a group with any fixed or missing member
is overwritten with that member's current residue; ``tied_epistasis`` reads
energy, bias, PSSM, and omit at the last listed member; tied binding does not
subtract the current-identity reference; tied ``nodes`` skips only ``present==0``
members and copies the current token, not ``S_true``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from aminx.families.potts_mpnn.decode import (
  PottsARDecode,
  categorical_draw,
  embed_at,
  mask_refine_x,
  neighbor_valid,
  pssm_mix,
  readout,
  schedule_groups,
  update_row,
)
from aminx.families.potts_mpnn.etab import positional_potts_energy
from aminx.utils.concatenate import concatenate_neighbor_nodes

if TYPE_CHECKING:
  from aminx.model.decoder import DecoderLayer

_DEFAULT_ITERS = 1000


class BindingTables(NamedTuple):
  """One structure's partitions, padded to a common length."""

  etab: Float[Array, "P Lp K A A"]
  e_idx: Int[Array, "P Lp K"]
  pad_valid: Bool[Array, "P Lp"]
  complex_index: Int[Array, "P Lp"]
  partition_of: Int[Array, " L"]
  local_of: Int[Array, " L"]
  inter_mask: Bool[Array, " L"]


class RefineResult(NamedTuple):
  """Refined tokens, how many sweeps ran, and the last sweep's ``ener_delta``."""

  sequence: Int[Array, " L"]
  n_iters: Int[Array, ""]
  ener_delta: Float[Array, ""]


def nodes_attention(
  present: Float[Array, " L"],
  chain_m_pos: Float[Array, " L"],
  n_neighbors: int,
) -> tuple[Float[Array, "L K"], Float[Array, "L K"]]:
  """Gibbs masks. Slot 0 is the self edge: backward off, forward on.

  ``m = present·chain_M_pos``. The autoregressive ``h_EXV_fw`` is not reused.
  """
  slots = jnp.arange(n_neighbors)
  residue = present * chain_m_pos
  mask_bw = residue[:, None] * (slots[None, :] != 0).astype(present.dtype)
  mask_fw = residue[:, None] * (slots[None, :] == 0).astype(present.dtype)
  return mask_bw, mask_fw


def check_tied_only_groups(
  tie_groups: Int[Array, "G M"],
  inter_mask: Bool[Array, " L"],
  *,
  tied_epistasis: bool,
  binding: str,
) -> None:
  """Upstream ``predicted_E /= num_pos`` crashes when ``num_pos`` is 0.

  aminx raises ``ValueError`` naming the group instead of ``ZeroDivisionError``.
  """
  if tied_epistasis or binding != "only":
    return
  groups = jnp.asarray(tie_groups)
  for row in range(int(groups.shape[0])):
    members = [int(index) for index in groups[row] if int(index) >= 0]
    if not members:
      continue
    if all(not bool(inter_mask[index]) for index in members):
      msg = (
        f"tied group {members} has no interface residue under binding_energy_optimization='only'"
      )
      raise ValueError(msg)


class PottsRefine(eqx.Module):
  """One structure, one mode. ``layers`` / embeddings are the MPNN modules.

  ``X`` handling (debt #2260, spec §3 note 2): upstream builds 20 candidates
  but samples 21 letters and raises ``IndexError`` if ``X`` is drawn. The f64
  oracles were dumped with ``X`` masked (``refine_omit_index = 20``,
  ``constant[20] = 1.0``). ``mask_refine_x`` subtracts ``1e8`` from logit 20
  before the softmax, so ``X`` is structurally undrawable and this module does
  not reproduce that ``IndexError``.
  """

  layers: tuple[DecoderLayer, ...]
  w_s_embed: eqx.nn.Embedding
  w_out: eqx.nn.Linear

  def __call__(
    self,
    mode: str,
    sequence: Int[Array, " L"],
    etab: Float[Array, "L K A A"],
    e_idx: Int[Array, "L K"],
    pad_valid: Bool[Array, " L"],
    present: Float[Array, " L"],
    chain_mask: Float[Array, " L"],
    chain_m_pos: Float[Array, " L"],
    order: Int[Array, " L"],
    uniforms: Float[Array, "iters draws"],
    omit: Float[Array, " V"],
    bias: Float[Array, " V"],
    bias_by_res: Float[Array, "L V"],
    pssm_coef: Float[Array, " L"],
    pssm_bias: Float[Array, "L V"],
    pssm_log_odds_mask: Float[Array, "L V"],
    omit_aa_mask: Float[Array, "L V"],
    tie_groups: Int[Array, "G M"],
    tied_beta: Float[Array, " L"],
    h_v: Float[Array, "L H"],
    h_e: Float[Array, "L K H"],
    binding_tables: BindingTables,
    *,
    temperature: float,
    pssm_multi: float,
    pssm_bias_flag: bool,
    pssm_log_odds_flag: bool,
    binding: str,
    tied: bool,
    tied_epistasis: bool,
    max_iters: int = _DEFAULT_ITERS,
  ) -> RefineResult:
    """Refine ``sequence``. ``order`` visits real rows; pad rows stay put."""
    if mode == "nodes":
      return self._nodes(
        sequence,
        e_idx,
        pad_valid,
        present,
        chain_mask,
        chain_m_pos,
        order,
        uniforms,
        omit,
        bias,
        bias_by_res,
        pssm_coef,
        pssm_bias,
        pssm_log_odds_mask,
        omit_aa_mask,
        tie_groups,
        tied_beta,
        h_v,
        h_e,
        temperature=temperature,
        pssm_multi=pssm_multi,
        pssm_bias_flag=pssm_bias_flag,
        pssm_log_odds_flag=pssm_log_odds_flag,
        tied=tied,
        tied_epistasis=tied_epistasis,
      )
    if mode not in ("potts", "potts_converge"):
      msg = f"unknown optimization_mode {mode!r}"
      raise ValueError(msg)
    iters = 1 if mode == "potts" else max_iters
    return self._potts(
      sequence,
      etab,
      e_idx,
      pad_valid,
      present,
      chain_mask,
      chain_m_pos,
      order,
      uniforms,
      omit,
      bias,
      bias_by_res,
      pssm_coef,
      pssm_bias,
      pssm_log_odds_mask,
      omit_aa_mask,
      tie_groups,
      tied_beta,
      binding_tables,
      temperature=temperature,
      pssm_multi=pssm_multi,
      pssm_bias_flag=pssm_bias_flag,
      pssm_log_odds_flag=pssm_log_odds_flag,
      binding=binding,
      tied=tied,
      tied_epistasis=tied_epistasis,
      max_iters=iters,
    )

  def _potts(
    self,
    sequence: Int[Array, " L"],
    etab: Float[Array, "L K A A"],
    e_idx: Int[Array, "L K"],
    pad_valid: Bool[Array, " L"],
    present: Float[Array, " L"],
    chain_mask: Float[Array, " L"],
    chain_m_pos: Float[Array, " L"],
    order: Int[Array, " L"],
    uniforms: Float[Array, "iters draws"],
    omit: Float[Array, " V"],
    bias: Float[Array, " V"],
    bias_by_res: Float[Array, "L V"],
    pssm_coef: Float[Array, " L"],
    pssm_bias: Float[Array, "L V"],
    pssm_log_odds_mask: Float[Array, "L V"],
    omit_aa_mask: Float[Array, "L V"],
    tie_groups: Int[Array, "G M"],
    tied_beta: Float[Array, " L"],
    binding_tables: BindingTables,
    *,
    temperature: float,
    pssm_multi: float,
    pssm_bias_flag: bool,
    pssm_log_odds_flag: bool,
    binding: str,
    tied: bool,
    tied_epistasis: bool,
    max_iters: int,
  ) -> RefineResult:
    del tied_beta
    temperature_value = jnp.asarray(temperature, dtype=etab.dtype)
    dtype = etab.dtype
    apply_bias = jnp.bool_((pssm_bias_flag and pssm_coef.size > 0) or pssm_bias.size > 0)
    apply_log_odds = jnp.bool_(pssm_log_odds_flag)
    apply_omit = jnp.bool_(omit_aa_mask.size > 0)

    def sweep(seq: Array, iteration: Array) -> tuple[Array, Array]:
      if tied:
        return _tied_energy_sweep(
          seq,
          iteration,
          etab=etab,
          e_idx=e_idx,
          pad_valid=pad_valid,
          present=present.astype(dtype),
          chain_mask=chain_mask.astype(dtype),
          chain_m_pos=chain_m_pos.astype(dtype),
          order=order.astype(jnp.int32),
          tie_groups=tie_groups.astype(jnp.int32),
          uniforms=uniforms.astype(dtype),
          omit=omit.astype(dtype),
          bias=bias.astype(dtype),
          bias_by_res=bias_by_res.astype(dtype),
          pssm_coef=pssm_coef.astype(dtype),
          pssm_bias=pssm_bias.astype(dtype),
          pssm_log_odds_mask=pssm_log_odds_mask.astype(dtype),
          omit_aa_mask=omit_aa_mask.astype(dtype),
          tables=binding_tables,
          temperature=temperature_value,
          pssm_multi=jnp.asarray(pssm_multi, dtype=dtype),
          apply_pssm_bias=apply_bias,
          apply_log_odds=apply_log_odds,
          apply_omit_aa=apply_omit,
          binding=binding,
          tied_epistasis=tied_epistasis,
        )
      return _untied_energy_sweep(
        seq,
        iteration,
        etab=etab,
        e_idx=e_idx,
        pad_valid=pad_valid,
        present=present.astype(dtype),
        chain_mask=chain_mask.astype(dtype),
        chain_m_pos=chain_m_pos.astype(dtype),
        order=order.astype(jnp.int32),
        uniforms=uniforms.astype(dtype),
        omit=omit.astype(dtype),
        bias=bias.astype(dtype),
        bias_by_res=bias_by_res.astype(dtype),
        pssm_coef=pssm_coef.astype(dtype),
        pssm_bias=pssm_bias.astype(dtype),
        pssm_log_odds_mask=pssm_log_odds_mask.astype(dtype),
        omit_aa_mask=omit_aa_mask.astype(dtype),
        tables=binding_tables,
        temperature=temperature_value,
        pssm_multi=jnp.asarray(pssm_multi, dtype=dtype),
        apply_pssm_bias=apply_bias,
        apply_log_odds=apply_log_odds,
        apply_omit_aa=apply_omit,
        binding=binding,
      )

    def cond(state: tuple[Array, Array, Array]) -> Array:
      _seq, ener, iters = state
      return (ener != 0) & (iters < jnp.int32(max_iters))

    def body(state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
      seq, _ener, iters = state
      seq, ener = sweep(seq, iters)
      return seq, ener, iters + jnp.int32(1)

    start = (
      sequence.astype(jnp.int32),
      jnp.asarray(1, dtype=dtype),
      jnp.int32(0),
    )
    sequence_out, ener, n_iters = jax.lax.while_loop(cond, body, start)
    return RefineResult(sequence=sequence_out, n_iters=n_iters, ener_delta=ener)

  def _nodes(
    self,
    sequence: Int[Array, " L"],
    e_idx: Int[Array, "L K"],
    pad_valid: Bool[Array, " L"],
    present: Float[Array, " L"],
    chain_mask: Float[Array, " L"],
    chain_m_pos: Float[Array, " L"],
    order: Int[Array, " L"],
    uniforms: Float[Array, "iters draws"],
    omit: Float[Array, " V"],
    bias: Float[Array, " V"],
    bias_by_res: Float[Array, "L V"],
    pssm_coef: Float[Array, " L"],
    pssm_bias: Float[Array, "L V"],
    pssm_log_odds_mask: Float[Array, "L V"],
    omit_aa_mask: Float[Array, "L V"],
    tie_groups: Int[Array, "G M"],
    tied_beta: Float[Array, " L"],
    h_v: Float[Array, "L H"],
    h_e: Float[Array, "L K H"],
    *,
    temperature: float,
    pssm_multi: float,
    pssm_bias_flag: bool,
    pssm_log_odds_flag: bool,
    tied: bool,
    tied_epistasis: bool,
  ) -> RefineResult:
    decoder = PottsARDecode(
      layers=tuple(self.layers),
      w_s_embed=self.w_s_embed,
      w_out=self.w_out,
    )
    dtype = h_v.dtype
    temperature_value = jnp.asarray(temperature, dtype=dtype)
    sequence = sequence.astype(jnp.int32)
    safe_token = jnp.clip(sequence, 0, self.w_s_embed.weight.shape[0] - 1)
    h_s = self.w_s_embed.weight[safe_token]
    n_layers = len(self.layers)
    h_v_stack = jnp.concatenate(
      [h_v[None], jnp.zeros((n_layers, *h_v.shape), dtype=dtype)],
      axis=0,
    )
    mask_bw, mask_fw = nodes_attention(
      present.astype(dtype),
      chain_m_pos.astype(dtype),
      e_idx.shape[1],
    )
    if tied and tied_epistasis:
      mask_bw = _zero_same_group(mask_bw, tie_groups.astype(jnp.int32), e_idx.astype(jnp.int32))
    zeros = jnp.zeros_like(h_v)
    h_ex = concatenate_neighbor_nodes(zeros, h_e, e_idx)
    h_exv_fw = mask_fw[..., None] * concatenate_neighbor_nodes(h_v, h_ex, e_idx)
    nbr = neighbor_valid(e_idx, pad_valid)
    apply_bias = jnp.bool_((pssm_bias_flag and pssm_coef.size > 0) or pssm_bias.size > 0)
    apply_log_odds = jnp.bool_(pssm_log_odds_flag)
    apply_omit = jnp.bool_(omit_aa_mask.size > 0)
    row_uniforms = uniforms.astype(dtype)[0]

    if tied:
      sequence = _tied_nodes_sweep(
        decoder,
        sequence,
        h_s,
        h_v_stack,
        h_e,
        e_idx.astype(jnp.int32),
        mask_bw,
        h_exv_fw,
        nbr,
        present.astype(dtype),
        chain_mask.astype(dtype),
        tie_groups.astype(jnp.int32),
        tied_beta.astype(dtype),
        order.astype(jnp.int32),
        pad_valid,
        row_uniforms,
        omit.astype(dtype),
        bias.astype(dtype),
        bias_by_res.astype(dtype),
        pssm_coef.astype(dtype),
        pssm_bias.astype(dtype),
        pssm_log_odds_mask.astype(dtype),
        omit_aa_mask.astype(dtype),
        temperature=temperature_value,
        pssm_multi=jnp.asarray(pssm_multi, dtype=dtype),
        apply_pssm_bias=apply_bias,
        apply_log_odds=apply_log_odds,
        apply_omit_aa=apply_omit,
      )
    else:
      sequence = _untied_nodes_sweep(
        decoder,
        sequence,
        h_s,
        h_v_stack,
        h_e,
        e_idx.astype(jnp.int32),
        mask_bw,
        h_exv_fw,
        nbr,
        present.astype(dtype),
        chain_mask.astype(dtype),
        chain_m_pos.astype(dtype),
        order.astype(jnp.int32),
        row_uniforms,
        omit.astype(dtype),
        bias.astype(dtype),
        bias_by_res.astype(dtype),
        pssm_coef.astype(dtype),
        pssm_bias.astype(dtype),
        pssm_log_odds_mask.astype(dtype),
        omit_aa_mask.astype(dtype),
        temperature=temperature_value,
        pssm_multi=jnp.asarray(pssm_multi, dtype=dtype),
        apply_pssm_bias=apply_bias,
        apply_log_odds=apply_log_odds,
        apply_omit_aa=apply_omit,
      )
    return RefineResult(
      sequence=sequence,
      n_iters=jnp.int32(1),
      ener_delta=jnp.asarray(0, dtype=dtype),
    )


def _partition_sequence(seq: Array, tables: BindingTables, partition: Array) -> Array:
  index = tables.complex_index[partition]
  safe = jnp.clip(index, 0, seq.shape[0] - 1)
  gathered = seq[safe]
  return jnp.where(index >= 0, gathered, jnp.int32(0))


def _energy_vector(
  seq: Array,
  position: Array,
  etab: Array,
  e_idx: Array,
  pad_valid: Array,
  tables: BindingTables,
  *,
  binding: str,
  tied: bool,
) -> Array:
  base = positional_potts_energy(etab, e_idx, pad_valid, seq, position)
  partition = jnp.clip(tables.partition_of[position], 0, tables.etab.shape[0] - 1)
  local = jnp.clip(tables.local_of[position], 0, tables.etab.shape[1] - 1)
  part_seq = _partition_sequence(seq, tables, partition)
  unbound = positional_potts_energy(
    tables.etab[partition],
    tables.e_idx[partition],
    tables.pad_valid[partition],
    part_seq,
    local,
  )
  current = jnp.clip(seq[position], 0, base.shape[0] - 1)
  untied = (base - base[current]) - (unbound - unbound[current])
  adjusted = (base - unbound) if tied else untied
  if binding == "none":
    return base
  # Non-interface sites keep the absolute energy. ``only`` never samples them.
  return jnp.where(tables.inter_mask[position], adjusted, base)


def _sample_logits(
  energy: Array,
  position: Array,
  *,
  temperature: Array,
  omit: Array,
  bias: Array,
  bias_by_res: Array,
  pssm_coef: Array,
  pssm_bias: Array,
  pssm_log_odds_mask: Array,
  omit_aa_mask: Array,
  pssm_multi: Array,
  apply_pssm_bias: Array,
  apply_log_odds: Array,
  apply_omit_aa: Array,
  uniform: Array,
) -> tuple[Array, Array]:
  """21-way draw with ``X`` masked. Returns ``(token, energy_of_token)``."""
  vocab = omit.shape[0]
  logits = -energy[:vocab] / temperature
  logits = mask_refine_x(logits)
  probs = pssm_mix(
    logits,
    temperature,
    omit,
    bias,
    bias_by_res[position],
    pssm_coef[position],
    pssm_bias[position],
    pssm_multi,
    pssm_log_odds_mask[position],
    omit_aa_mask[position],
    apply_pssm_bias=apply_pssm_bias,
    apply_log_odds=apply_log_odds,
    apply_omit_aa=apply_omit_aa,
  )
  token = categorical_draw(probs, uniform)
  return token, energy[token]


def _untied_energy_sweep(
  seq: Array,
  iteration: Array,
  *,
  etab: Array,
  e_idx: Array,
  pad_valid: Array,
  present: Array,
  chain_mask: Array,
  chain_m_pos: Array,
  order: Array,
  uniforms: Array,
  omit: Array,
  bias: Array,
  bias_by_res: Array,
  pssm_coef: Array,
  pssm_bias: Array,
  pssm_log_odds_mask: Array,
  omit_aa_mask: Array,
  tables: BindingTables,
  temperature: Array,
  pssm_multi: Array,
  apply_pssm_bias: Array,
  apply_log_odds: Array,
  apply_omit_aa: Array,
  binding: str,
) -> tuple[Array, Array]:
  def step(
    carry: tuple[Array, Array, Array],
    position: Array,
  ) -> tuple[tuple[Array, Array, Array], None]:
    seq_c, ener, cursor = carry
    position = position.astype(jnp.int32)
    active = (present[position] != 0) & (chain_m_pos[position] != 0) & (chain_mask[position] != 0)
    if binding == "only":
      active = active & tables.inter_mask[position]
    energy = _energy_vector(
      seq_c,
      position,
      etab,
      e_idx,
      pad_valid,
      tables,
      binding=binding,
      tied=False,
    )
    token, chosen = _sample_logits(
      energy,
      position,
      temperature=temperature,
      omit=omit,
      bias=bias,
      bias_by_res=bias_by_res,
      pssm_coef=pssm_coef,
      pssm_bias=pssm_bias,
      pssm_log_odds_mask=pssm_log_odds_mask,
      omit_aa_mask=omit_aa_mask,
      pssm_multi=pssm_multi,
      apply_pssm_bias=apply_pssm_bias,
      apply_log_odds=apply_log_odds,
      apply_omit_aa=apply_omit_aa,
      uniform=uniforms[iteration, cursor],
    )
    seq_c = seq_c.at[position].set(jnp.where(active, token, seq_c[position]))
    ener = ener + jnp.where(active, chosen, jnp.zeros_like(chosen))
    cursor = cursor + jnp.where(active, jnp.int32(1), jnp.int32(0))
    return (seq_c, ener, cursor), None

  init = (seq, jnp.asarray(0, dtype=etab.dtype), jnp.int32(0))
  (seq, ener, _cursor), _rest = jax.lax.scan(step, init, order)
  return seq, ener


def _tied_energy_sweep(
  seq: Array,
  iteration: Array,
  *,
  etab: Array,
  e_idx: Array,
  pad_valid: Array,
  present: Array,
  chain_mask: Array,
  chain_m_pos: Array,
  order: Array,
  tie_groups: Array,
  uniforms: Array,
  omit: Array,
  bias: Array,
  bias_by_res: Array,
  pssm_coef: Array,
  pssm_bias: Array,
  pssm_log_odds_mask: Array,
  omit_aa_mask: Array,
  tables: BindingTables,
  temperature: Array,
  pssm_multi: Array,
  apply_pssm_bias: Array,
  apply_log_odds: Array,
  apply_omit_aa: Array,
  binding: str,
  tied_epistasis: bool,
) -> tuple[Array, Array]:
  """One tied sweep. ``order`` only builds the group schedule."""
  length = seq.shape[0]
  pad_rows = jnp.arange(length, dtype=jnp.int32) >= length
  full_order = jnp.concatenate([order, jnp.arange(order.shape[0], length, dtype=jnp.int32)])
  # ``schedule_groups`` expects a permutation of ``L``. ``order`` is that
  # permutation when the caller passes one. Pad rows are absent from ``order``
  # only when ``order`` is shorter; tests pass a full permutation.
  del pad_rows
  if full_order.shape[0] != length:
    full_order = order
  group_order, size, _rank, _raw = schedule_groups(
    tie_groups,
    jnp.ones((length,), dtype=jnp.bool_),
    jnp.zeros((length,), dtype=temperature.dtype),
    chain_mask,
    chain_m_pos,
    present,
    full_order.astype(jnp.int32),
  )

  def step(
    carry: tuple[Array, Array, Array],
    g_step: Array,
  ) -> tuple[tuple[Array, Array, Array], None]:
    seq_c, ener, cursor = carry
    group = group_order[g_step]
    group_size = size[group]
    members = tie_groups[group]

    def skip(state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
      return state

    def visit(state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
      return _visit_tied_group(
        state,
        members=members,
        group_size=group_size,
        iteration=iteration,
        etab=etab,
        e_idx=e_idx,
        pad_valid=pad_valid,
        present=present,
        chain_mask=chain_mask,
        chain_m_pos=chain_m_pos,
        uniforms=uniforms,
        omit=omit,
        bias=bias,
        bias_by_res=bias_by_res,
        pssm_coef=pssm_coef,
        pssm_bias=pssm_bias,
        pssm_log_odds_mask=pssm_log_odds_mask,
        omit_aa_mask=omit_aa_mask,
        tables=tables,
        temperature=temperature,
        pssm_multi=pssm_multi,
        apply_pssm_bias=apply_pssm_bias,
        apply_log_odds=apply_log_odds,
        apply_omit_aa=apply_omit_aa,
        binding=binding,
        tied_epistasis=tied_epistasis,
      )

    return jax.lax.cond(group_size == 0, skip, visit, (seq_c, ener, cursor)), None

  steps = jnp.arange(group_order.shape[0], dtype=jnp.int32)
  init = (seq, jnp.asarray(0, dtype=etab.dtype), jnp.int32(0))
  (seq, ener, _cursor), _rest = jax.lax.scan(step, init, steps)
  return seq, ener


def _visit_tied_group(
  state: tuple[Array, Array, Array],
  *,
  members: Array,
  group_size: Array,
  iteration: Array,
  etab: Array,
  e_idx: Array,
  pad_valid: Array,
  present: Array,
  chain_mask: Array,
  chain_m_pos: Array,
  uniforms: Array,
  omit: Array,
  bias: Array,
  bias_by_res: Array,
  pssm_coef: Array,
  pssm_bias: Array,
  pssm_log_odds_mask: Array,
  omit_aa_mask: Array,
  tables: BindingTables,
  temperature: Array,
  pssm_multi: Array,
  apply_pssm_bias: Array,
  apply_log_odds: Array,
  apply_omit_aa: Array,
  binding: str,
  tied_epistasis: bool,
) -> tuple[Array, Array, Array]:
  seq, ener, cursor = state
  length = seq.shape[0]
  found = jnp.zeros((), dtype=jnp.bool_)
  frozen_aa = jnp.int32(0)

  def freeze_step(carry: tuple[Array, Array], slot: Array) -> tuple[tuple[Array, Array], None]:
    found_c, amino = carry
    position = members[slot]
    valid = position >= 0
    safe = jnp.clip(position, 0, length - 1)
    bad = valid & ((present[safe] == 0) | (chain_mask[safe] == 0) | (chain_m_pos[safe] == 0))
    take = bad & ~found_c
    amino = jnp.where(take, seq[safe], amino)
    return (found_c | bad, amino), None

  slots = jnp.arange(members.shape[0], dtype=jnp.int32)
  (found, frozen_aa), _rest = jax.lax.scan(freeze_step, (found, frozen_aa), slots)
  last = members[group_size - 1]
  last_safe = jnp.clip(last, 0, length - 1)

  def frozen_write(current: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
    seq_c, ener_c, cursor_c = current
    valid = members >= 0
    positions = jnp.where(valid, members, jnp.int32(length))
    seq_c = seq_c.at[positions].set(frozen_aa, mode="drop")
    return seq_c, ener_c, cursor_c

  def designed(current: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
    seq_c, ener_c, cursor_c = current
    any_interface = jnp.zeros((), dtype=jnp.bool_)
    interface_slots = jnp.arange(members.shape[0], dtype=jnp.int32)

    def _any(flag: Array, slot: Array) -> tuple[Array, None]:
      position = members[slot]
      valid = position >= 0
      safe = jnp.clip(position, 0, length - 1)
      return flag | (valid & tables.inter_mask[safe]), None

    any_interface, _marks = jax.lax.scan(_any, any_interface, interface_slots)
    if tied_epistasis:
      energy = _energy_vector(
        seq_c,
        last_safe,
        etab,
        e_idx,
        pad_valid,
        tables,
        binding=binding,
        tied=True,
      )
      included = jnp.ones((), dtype=jnp.bool_) if binding != "only" else any_interface
    else:
      energy, included = _average_member_energy(
        seq_c,
        members,
        group_size,
        etab,
        e_idx,
        pad_valid,
        tables,
        binding=binding,
      )
    token, chosen = _sample_logits(
      energy,
      last_safe,
      temperature=temperature,
      omit=omit,
      bias=bias,
      bias_by_res=bias_by_res,
      pssm_coef=pssm_coef,
      pssm_bias=pssm_bias,
      pssm_log_odds_mask=pssm_log_odds_mask,
      omit_aa_mask=omit_aa_mask,
      pssm_multi=pssm_multi,
      apply_pssm_bias=apply_pssm_bias,
      apply_log_odds=apply_log_odds,
      apply_omit_aa=apply_omit_aa,
      uniform=uniforms[iteration, cursor_c],
    )
    # Non-epistasis writes the candidate's residue at the leaked last member.
    # If that member was not mutated in the averaged set, the original token stays.
    written = token if tied_epistasis else jnp.where(included, token, seq_c[last_safe])
    take = included
    valid = members >= 0
    positions = jnp.where(valid, members, jnp.int32(length))
    seq_c = jnp.where(take, seq_c.at[positions].set(written, mode="drop"), seq_c)
    ener_c = ener_c + jnp.where(take, chosen, jnp.asarray(0, dtype=ener_c.dtype))
    cursor_c = cursor_c + jnp.where(take, jnp.int32(1), jnp.int32(0))
    return seq_c, ener_c, cursor_c

  return jax.lax.cond(found, frozen_write, designed, (seq, ener, cursor))


def _average_member_energy(
  seq: Array,
  members: Array,
  group_size: Array,
  etab: Array,
  e_idx: Array,
  pad_valid: Array,
  tables: BindingTables,
  *,
  binding: str,
) -> tuple[Array, Array]:
  """Mean of per-member energies. The last member is ``included`` for the write."""
  total = jnp.zeros(etab.shape[-1], dtype=etab.dtype)
  count = jnp.asarray(0, dtype=etab.dtype)
  last_included = jnp.zeros((), dtype=jnp.bool_)

  def step(
    carry: tuple[Array, Array, Array],
    slot: Array,
  ) -> tuple[tuple[Array, Array, Array], None]:
    total_c, count_c, last_c = carry
    position = members[slot]
    valid = (slot < group_size) & (position >= 0)
    safe = jnp.clip(position, 0, seq.shape[0] - 1)
    energy = _energy_vector(
      seq,
      safe,
      etab,
      e_idx,
      pad_valid,
      tables,
      binding=binding,
      tied=True,
    )
    use = valid
    if binding == "only":
      use = use & tables.inter_mask[safe]
    total_c = total_c + jnp.where(use, energy, jnp.zeros_like(energy))
    count_c = count_c + jnp.where(use, 1, 0)
    last_c = jnp.where(slot == (group_size - 1), use, last_c)
    return (total_c, count_c, last_c), None

  slots = jnp.arange(members.shape[0], dtype=jnp.int32)
  (total, count, last_included), _rest = jax.lax.scan(
    step,
    (total, count, last_included),
    slots,
  )
  mean = total / jnp.maximum(count, jnp.asarray(1, dtype=etab.dtype))
  return mean, last_included


def _untied_nodes_sweep(
  decoder: PottsARDecode,
  sequence: Array,
  h_s: Array,
  h_v_stack: Array,
  h_e: Array,
  e_idx: Array,
  mask_bw: Array,
  h_exv_fw: Array,
  nbr: Array,
  present: Array,
  chain_mask: Array,
  chain_m_pos: Array,
  order: Array,
  uniforms: Array,
  omit: Array,
  bias: Array,
  bias_by_res: Array,
  pssm_coef: Array,
  pssm_bias: Array,
  pssm_log_odds_mask: Array,
  omit_aa_mask: Array,
  *,
  temperature: Array,
  pssm_multi: Array,
  apply_pssm_bias: Array,
  apply_log_odds: Array,
  apply_omit_aa: Array,
) -> Array:
  def step(
    carry: tuple[Array, Array, Array, Array],
    position: Array,
  ) -> tuple[tuple[Array, Array, Array, Array], None]:
    seq_c, h_s_c, h_v_c, cursor = carry
    position = position.astype(jnp.int32)
    active = (present[position] != 0) & (chain_m_pos[position] != 0) & (chain_mask[position] != 0)

    def run(state: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
      seq_r, h_s_r, h_v_r, cursor_r = state
      h_v_r = update_row(
        decoder.layers,
        h_v_r,
        h_s_r,
        h_e,
        e_idx,
        mask_bw,
        h_exv_fw,
        present,
        nbr,
        position,
      )
      logits = readout(decoder.w_out, h_v_r[-1, position]) / temperature
      logits = mask_refine_x(logits)
      probs = pssm_mix(
        logits,
        temperature,
        omit,
        bias,
        bias_by_res[position],
        pssm_coef[position],
        pssm_bias[position],
        pssm_multi,
        pssm_log_odds_mask[position],
        omit_aa_mask[position],
        apply_pssm_bias=apply_pssm_bias,
        apply_log_odds=apply_log_odds,
        apply_omit_aa=apply_omit_aa,
      )
      token = categorical_draw(probs, uniforms[cursor_r])
      seq_r = seq_r.at[position].set(token)
      h_s_r = h_s_r.at[position].set(embed_at(decoder.w_s_embed, token))
      return seq_r, h_s_r, h_v_r, cursor_r + jnp.int32(1)

    def hold(state: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
      return state

    return jax.lax.cond(active, run, hold, (seq_c, h_s_c, h_v_c, cursor)), None

  init = (sequence, h_s, h_v_stack, jnp.int32(0))
  (sequence, _h_s, _h_v, _cursor), _rest = jax.lax.scan(step, init, order)
  return sequence


def _tied_nodes_sweep(
  decoder: PottsARDecode,
  sequence: Array,
  h_s: Array,
  h_v_stack: Array,
  h_e: Array,
  e_idx: Array,
  mask_bw: Array,
  h_exv_fw: Array,
  nbr: Array,
  present: Array,
  chain_mask: Array,
  tie_groups: Array,
  tied_beta: Array,
  order: Array,
  pad_valid: Array,
  uniforms: Array,
  omit: Array,
  bias: Array,
  bias_by_res: Array,
  pssm_coef: Array,
  pssm_bias: Array,
  pssm_log_odds_mask: Array,
  omit_aa_mask: Array,
  *,
  temperature: Array,
  pssm_multi: Array,
  apply_pssm_bias: Array,
  apply_log_odds: Array,
  apply_omit_aa: Array,
) -> Array:
  del pad_valid
  group_order, size, _rank, _raw = schedule_groups(
    tie_groups,
    jnp.ones((sequence.shape[0],), dtype=jnp.bool_),
    jnp.zeros((sequence.shape[0],), dtype=temperature.dtype),
    jnp.ones_like(present),
    jnp.ones_like(present),
    present,
    order.astype(jnp.int32),
  )

  def step(
    carry: tuple[Array, Array, Array, Array],
    g_step: Array,
  ) -> tuple[tuple[Array, Array, Array, Array], None]:
    group = group_order[g_step]
    group_size = size[group]

    def skip(state: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
      return state

    def visit(state: tuple[Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array]:
      return _tied_nodes_group(
        decoder,
        state,
        members=tie_groups[group],
        group_size=group_size,
        tied_beta=tied_beta,
        h_e=h_e,
        e_idx=e_idx,
        mask_bw=mask_bw,
        h_exv_fw=h_exv_fw,
        nbr=nbr,
        present=present,
        chain_mask=chain_mask,
        uniforms=uniforms,
        omit=omit,
        bias=bias,
        bias_by_res=bias_by_res,
        pssm_coef=pssm_coef,
        pssm_bias=pssm_bias,
        pssm_log_odds_mask=pssm_log_odds_mask,
        omit_aa_mask=omit_aa_mask,
        temperature=temperature,
        pssm_multi=pssm_multi,
        apply_pssm_bias=apply_pssm_bias,
        apply_log_odds=apply_log_odds,
        apply_omit_aa=apply_omit_aa,
      )

    return jax.lax.cond(group_size == 0, skip, visit, carry), None

  steps = jnp.arange(group_order.shape[0], dtype=jnp.int32)
  (sequence, _h_s, _h_v, _cursor), _rest = jax.lax.scan(
    step,
    (sequence, h_s, h_v_stack, jnp.int32(0)),
    steps,
  )
  return sequence


def _tied_nodes_group(
  decoder: PottsARDecode,
  state: tuple[Array, Array, Array, Array],
  *,
  members: Array,
  group_size: Array,
  tied_beta: Array,
  h_e: Array,
  e_idx: Array,
  mask_bw: Array,
  h_exv_fw: Array,
  nbr: Array,
  present: Array,
  chain_mask: Array,
  uniforms: Array,
  omit: Array,
  bias: Array,
  bias_by_res: Array,
  pssm_coef: Array,
  pssm_bias: Array,
  pssm_log_odds_mask: Array,
  omit_aa_mask: Array,
  temperature: Array,
  pssm_multi: Array,
  apply_pssm_bias: Array,
  apply_log_odds: Array,
  apply_omit_aa: Array,
) -> tuple[Array, Array, Array, Array]:
  sequence, h_s, h_v_stack, cursor = state
  length = sequence.shape[0]
  logits0 = jnp.zeros((omit.shape[0],), dtype=temperature.dtype)
  init = (h_v_stack, logits0, jnp.zeros((), dtype=jnp.bool_), jnp.int32(0))

  def member_step(
    carry: tuple[Array, Array, Array, Array],
    slot: Array,
  ) -> tuple[tuple[Array, Array, Array, Array], None]:
    h_v_c, logits, stopped, copied = carry
    position = members[slot]
    valid = position >= 0
    safe = jnp.clip(position, 0, length - 1)
    masked = valid & (present[safe] == 0)
    start = masked & ~stopped
    stopped_now = stopped | masked
    copied = jnp.where(start, sequence[safe], copied)
    run = valid & ~stopped_now

    def run_member(operand: tuple[Array, Array]) -> tuple[Array, Array]:
      h_v_run, logits_run = operand
      h_v_run = update_row(
        decoder.layers,
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
      extra = tied_beta[safe] * (readout(decoder.w_out, h_v_run[-1, safe]) / temperature)
      extra = extra / group_size
      return h_v_run, logits_run + extra

    h_v_c, logits = jax.lax.cond(run, run_member, lambda operand: operand, (h_v_c, logits))
    return (h_v_c, logits, stopped_now, copied), None

  slots = jnp.arange(members.shape[0], dtype=jnp.int32)
  (h_v_stack, logits, stopped, copied), _unused = jax.lax.scan(member_step, init, slots)
  last = members[group_size - 1]
  last_safe = jnp.clip(last, 0, length - 1)

  def from_mask(_logits: Array) -> tuple[Array, Array]:
    return copied, cursor

  def from_draw(logits_in: Array) -> tuple[Array, Array]:
    logits_in = mask_refine_x(logits_in)
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
    token = jnp.where(chain_mask[last_safe] == 0, sequence[last_safe], drawn).astype(jnp.int32)
    return token, cursor + jnp.int32(1)

  token, cursor = jax.lax.cond(stopped, from_mask, from_draw, logits)
  valid = members >= 0
  positions = jnp.where(valid, members, jnp.int32(length))
  sequence = sequence.at[positions].set(token, mode="drop")
  embedded = embed_at(decoder.w_s_embed, token)
  embedded_rows = jnp.broadcast_to(embedded, (positions.shape[0], embedded.shape[0]))
  h_s = h_s.at[positions].set(embedded_rows, mode="drop")
  return sequence, h_s, h_v_stack, cursor


def _zero_same_group(
  mask_bw: Float[Array, "L K"],
  tie_groups: Int[Array, "G M"],
  e_idx: Int[Array, "L K"],
) -> Float[Array, "L K"]:
  """Drop neighbours that share a tied group (``tied_epistasis`` on ``nodes``)."""
  length = mask_bw.shape[0]
  group_of = jnp.full((length,), jnp.int32(-1))
  rows = jnp.arange(tie_groups.shape[0], dtype=jnp.int32)
  valid = tie_groups >= 0
  safe = jnp.where(valid, tie_groups, jnp.int32(0))
  group_of = group_of.at[safe].set(
    jnp.where(valid, rows[:, None], group_of[safe]),
    mode="drop",
  )
  neighbour = jnp.clip(e_idx, 0, length - 1)
  same = (group_of[neighbour] == group_of[:, None]) & (group_of[:, None] >= 0)
  return jnp.where(same, jnp.asarray(0, dtype=mask_bw.dtype), mask_bw)
