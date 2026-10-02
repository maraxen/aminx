"""Two-structure LASEr sampling. Ports ``LASErMPNN.tied_sample``.

Sequence probabilities are mixed in probability space,
``λ P1 + (1-λ) P2``, and one draw is written to both structures. χ is sampled
independently. The χ uniform stream is ``(sample, step, χ, structure)`` with
structure 1 consumed before structure 2 at each χ index. Getting that axis
order wrong consumes the injected uniforms in a different sequence and changes
the draws even when every logit is right.

Input χ is ``nan_to_num``'d and then kept wherever the residue is fixed, unless
``repack_all``. A NaN fixed χ therefore becomes 0 degrees. That is upstream
``tied_sample`` (``utils/model.py:491-492`` and ``:690-692``). It deliberately
differs from the non-tied sample path, which keeps input χ only where the
residue is fixed and the angle is non-NaN (``utils/model.py:907-914``).
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from aminx.model.laser.decoder import LaserDecoder, binned_degree_basis
from aminx.model.laser.encoders import LaserEncoder, encode_structure
from aminx.model.laser.graphs import GraphStructure, pack_edges, scatter_edge_values
from aminx.model.laser.joint_decode import (
  _CHI_ANGLES,
  _CHI_BINS,
  _CHI_MASK,
  _MASK_CLASS,
  _N_DECODER,
  LaserJointDecode,
  categorical_draw,
)
from aminx.model.laser.layers import apply_linear

_BIN_WIDTH = 5.0


@dataclass(frozen=True, slots=True)
class TiedDecodeResult:
  """Structure 1 is the primary record. Structure 2 carries its own χ."""

  sequence: np.ndarray
  sequence_logits_1: np.ndarray
  sequence_logits_2: np.ndarray
  chi_degrees_1: np.ndarray
  chi_degrees_2: np.ndarray
  chi_logits_1: np.ndarray
  chi_logits_2: np.ndarray
  chi_bins_1: np.ndarray
  chi_bins_2: np.ndarray


def mix_sequence_probabilities(
  logits_1: jax.Array,
  logits_2: jax.Array,
  lambda_: float,
  temperature: float,
  *,
  on_logits: bool,
) -> jax.Array:
  """λ mixes probabilities. Mixing the logits first is the negative control."""
  if on_logits:
    return jax.nn.softmax((lambda_ * logits_1 + (1.0 - lambda_) * logits_2) / temperature)
  return lambda_ * jax.nn.softmax(logits_1 / temperature) + (1.0 - lambda_) * jax.nn.softmax(
    logits_2 / temperature,
  )


def _tied_temperature(value: float | None) -> float:
  """``None`` and ``0`` both sample at 1e-6. Tied has no argmax sequence path."""
  if value is None or value == 0.0:
    return 1e-6
  return value


def _refuse_tied_knobs(
  *,
  seq_min_p: float,
  chi_min_p: float,
  fs_sequence_temp: float | None,
  disable_charged_fs: bool,
  ignore_chain_mask_zeros: bool,
) -> None:
  """These knobs are not ``tied_sample`` parameters. Setting one is an error."""
  flagged: list[str] = []
  if seq_min_p != 0.0:
    flagged.append("seq_min_p")
  if chi_min_p != 0.0:
    flagged.append("chi_min_p")
  if fs_sequence_temp is not None:
    flagged.append("fs_sequence_temp")
  if disable_charged_fs:
    flagged.append("disable_charged_fs")
  if ignore_chain_mask_zeros:
    flagged.append("ignore_chain_mask_zeros")
  if flagged:
    msg = "tied_second_input rejects " + ", ".join(flagged)
    raise ValueError(msg)


def tied_chi_index(step: int, chi: int, structure: int, *, n_chi: int = 4) -> int:
  """Flat index into ``(step, χ, structure)`` with structure varying fastest."""
  return (step * n_chi + chi) * 2 + structure


def _pack_chi_uniforms(chi_uniforms: np.ndarray, n_res: int) -> np.ndarray:
  """``(decode_step, χ, structure)``, structure varying fastest.

  Upstream ``tied_sample`` walks columns of ``decoding_order`` and, in each
  column, draws χ for structure 1 then structure 2. The injected stream's step
  axis is that column, not the residue number: under a reversed order the
  first eight χ draws belong to the last residue. A ``(length, 4, 2)`` buffer
  is already one row per column. Taking ``array[0]`` because the array is
  3-D keeps only the first residue and zero-fills every later step.
  """
  chi_u = np.asarray(chi_uniforms, dtype=np.float64)
  if chi_u.ndim == 4 and int(chi_u.shape[0]) == 1:
    chi_u = chi_u[0]
  if tuple(int(dim) for dim in chi_u.shape) == (n_res, 4, 2):
    return np.ascontiguousarray(chi_u)
  flat = np.zeros((n_res, 4, 2), dtype=np.float64)
  usable = min(int(flat.size), int(chi_u.size))
  flat.reshape(-1)[:usable] = np.reshape(chi_u, (-1,))[:usable]
  return flat


_LASER_ALPHABET = "ARNDCEQGHILKMFPSTWYVX"


def _pack_encoded(
  encoder: LaserEncoder,
  backbone: np.ndarray,
  ligand_coords: np.ndarray,
  ligand_atomic_numbers: np.ndarray,
  ligand_subbatch: np.ndarray,
  period_index: np.ndarray,
  group_index: np.ndarray,
  structure: GraphStructure,
) -> tuple[jax.Array, ...]:
  encoded = encode_structure(
    encoder,
    backbone,
    ligand_coords,
    ligand_atomic_numbers,
    ligand_subbatch,
    period_index,
    group_index,
    structure,
  )
  n_res = int(encoded.prot_scalars.shape[0])
  packed_pr = pack_edges(n_res, encoded.pr_pr_idx)
  packed_lp = pack_edges(n_res, encoded.lig_pr_idx)
  return (
    jnp.asarray(encoded.prot_scalars),
    jnp.asarray(encoded.prot_vectors),
    jnp.asarray(encoded.lig_scalars),
    jnp.asarray(scatter_edge_values(n_res, packed_pr, encoded.pr_pr_eattr)),
    jnp.asarray(packed_pr.neighbours),
    jnp.asarray(packed_pr.mask),
    jnp.asarray(scatter_edge_values(n_res, packed_lp, encoded.lig_pr_eattr)),
    jnp.asarray(packed_lp.neighbours),
    jnp.asarray(packed_lp.mask),
  )


def _write_layers(
  decoder: LaserDecoder,
  node_s: jax.Array,
  node_v: jax.Array,
  sequence: jax.Array,
  chi_degrees: jax.Array,
  pr_edges: jax.Array,
  pr_neighbours: jax.Array,
  pr_mask: jax.Array,
  lig_s: jax.Array,
  lp_edges: jax.Array,
  lp_neighbours: jax.Array,
  lp_mask: jax.Array,
  decoding_order: jax.Array,
  step: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  chi_flat = jnp.reshape(
    jnp.nan_to_num(binned_degree_basis(chi_degrees)),
    (chi_degrees.shape[0], _CHI_ANGLES * _CHI_BINS),
  )
  for index in range(3):
    features, row_mask, slot_neighbours = decoder._teacher_edges(  # noqa: SLF001
      node_s[index],
      node_s[0],
      pr_edges,
      pr_neighbours,
      pr_mask,
      sequence,
      chi_flat,
      decoding_order,
    )
    empty = jnp.zeros((*pr_neighbours.shape, 0), dtype=node_s.dtype)
    updated_s, updated_v, _edges = decoder.protein_decoder_layers[index].hetgat(
      node_s[index],
      node_v[index],
      (features, lig_s),
      (slot_neighbours, lp_neighbours),
      (empty, lp_edges),
      (row_mask, lp_mask),
      (False, False),
    )
    node_s = node_s.at[index + 1, step].set(updated_s[step])
    node_v = node_v.at[index + 1, step].set(updated_v[step])
  return node_s, node_v


def _one_chi(
  decoder: LaserDecoder,
  joint: LaserJointDecode,
  *,
  prot_scalars: jax.Array,
  prot_vectors: jax.Array,
  seq_emb: jax.Array,
  chi_prev: jax.Array,
  index: int,
  uniform: jax.Array | None,
  chi_temperature: float | None,
  chain_fixed: jax.Array,
  input_angle: jax.Array,
  repack_all: bool,
  defined: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  logits = decoder.chi_prediction_layers[index](
    jnp.concatenate((prot_scalars, seq_emb, chi_prev)),
  )
  if chi_temperature is None:
    stored = logits
    bin_index = jnp.argmax(logits).astype(jnp.int32)
  else:
    stored = logits
    bin_index = categorical_draw(
      jax.nn.softmax(stored / chi_temperature),
      jnp.asarray(uniform),
    )
  one_hot = jax.nn.one_hot(bin_index, _CHI_BINS, dtype=stored.dtype)
  offset = joint.chi_offset_prediction_layers[index](jnp.concatenate((stored, one_hot)))[0]
  centers = jnp.arange(-180.0, 180.0, _BIN_WIDTH, dtype=stored.dtype)
  angle = jnp.remainder(centers[bin_index] + offset + 180.0, 360.0) - 180.0
  # nan_to_num already ran on input_angle. Fixed rows keep that value, including
  # the 0 that replaced NaN, unless repack_all. Non-tied sampling would also
  # require the original angle to be finite, and would leave NaN in place.
  if not repack_all:
    angle = jnp.where(chain_fixed, input_angle, angle)
  enc = binned_degree_basis(jnp.reshape(angle, (1, 1)))[0, 0]
  chi_prev = jnp.concatenate((chi_prev, enc))
  if index < 3:
    updated_s, updated_v = decoder.chi_vector_layer_norms[index](
      *decoder.chi_vector_update_layers[index](
        jnp.concatenate((prot_scalars, chi_prev))[None, :],
        prot_vectors[None, :, :],
      ),
    )
    prot_scalars = updated_s[0]
    prot_vectors = updated_v[0]
  kept_logits = jnp.where(defined, stored, jnp.zeros_like(stored))
  kept_angle = jnp.where(defined, angle, jnp.asarray(jnp.nan, dtype=angle.dtype))
  kept_bin = jnp.where(defined, bin_index, jnp.int32(-1))
  return prot_scalars, prot_vectors, chi_prev, kept_logits, kept_angle, kept_bin


@eqx.filter_jit
def _scan_tied(  # noqa: PLR0915
  decoder: LaserDecoder,
  joint: LaserJointDecode,
  packed1: tuple[jax.Array, ...],
  packed2: tuple[jax.Array, ...],
  decoding_order: Int[Array, " l"],
  input_sequence: Int[Array, " l"],
  input_chi_1: Float[Array, "l 4"],
  input_chi_2: Float[Array, "l 4"],
  chain_mask: Bool[Array, " l"],
  disabled: Bool[Array, " 21"],
  budget_mask: Bool[Array, " l"],
  seq_uniforms: Float[Array, " l"],
  chi_uniforms: Float[Array, "l 4 2"],
  *,
  sequence_temperature: float,
  chi_temperature: float | None,
  lambda_: float,
  repack_all: bool,
  lambda_on_logits: bool,
  ala_budget: int,
  gly_budget: int,
) -> tuple[jax.Array, ...]:
  """Scan residues. Each step reads the embedding and χ the previous step wrote."""
  s1, v1, lig1, pr_e1, pr_n1, pr_m1, lp_e1, lp_n1, lp_m1 = packed1
  s2, v2, lig2, pr_e2, pr_n2, pr_m2, lp_e2, lp_n2, lp_m2 = packed2
  length = s1.shape[0]
  dtype = s1.dtype
  node_s1 = jnp.zeros((4, *s1.shape), dtype=dtype).at[0].set(s1)
  node_v1 = jnp.zeros((4, *v1.shape), dtype=dtype).at[0].set(v1)
  node_s2 = jnp.zeros((4, *s2.shape), dtype=dtype).at[0].set(s2)
  node_v2 = jnp.zeros((4, *v2.shape), dtype=dtype).at[0].set(v2)
  init = (
    jnp.full((length,), _MASK_CLASS, dtype=jnp.int32),
    jnp.full((length, _CHI_ANGLES), jnp.nan, dtype=dtype),
    jnp.full((length, _CHI_ANGLES), jnp.nan, dtype=dtype),
    node_s1,
    node_v1,
    node_s2,
    node_v2,
    jnp.zeros((length, 21), dtype=dtype),
    jnp.zeros((length, 21), dtype=dtype),
    jnp.zeros((length, _CHI_ANGLES, _CHI_BINS), dtype=dtype),
    jnp.zeros((length, _CHI_ANGLES, _CHI_BINS), dtype=dtype),
    jnp.full((length, _CHI_ANGLES), -1, dtype=jnp.int32),
    jnp.full((length, _CHI_ANGLES), -1, dtype=jnp.int32),
    jnp.int32(0),
    jnp.int32(0),
    jnp.int32(0),
  )

  def body(
    carry: tuple[jax.Array, ...],
    residue: Int[Array, ""],
  ) -> tuple[tuple[jax.Array, ...], None]:
    (
      sequence,
      chi1,
      chi2,
      ns1,
      nv1,
      ns2,
      nv2,
      logits1,
      logits2,
      chi_logits1,
      chi_logits2,
      bins1,
      bins2,
      ala_count,
      gly_count,
      step_i,
    ) = carry
    ns1, nv1 = _write_layers(
      decoder,
      ns1,
      nv1,
      sequence,
      chi1,
      pr_e1,
      pr_n1,
      pr_m1,
      lig1,
      lp_e1,
      lp_n1,
      lp_m1,
      decoding_order,
      residue,
    )
    ns2, nv2 = _write_layers(
      decoder,
      ns2,
      nv2,
      sequence,
      chi2,
      pr_e2,
      pr_n2,
      pr_m2,
      lig2,
      lp_e2,
      lp_n2,
      lp_m2,
      decoding_order,
      residue,
    )
    raw1 = apply_linear(decoder.sequence_output_layer, ns1[_N_DECODER, residue])
    raw2 = apply_linear(decoder.sequence_output_layer, ns2[_N_DECODER, residue])
    sampling = ~chain_mask[residue]
    floor = jnp.finfo(raw1.dtype).min
    raw1 = jnp.where(sampling & disabled, floor, raw1)
    raw2 = jnp.where(sampling & disabled, floor, raw2)
    # Budgets are tied_sample parameters. They floor both structures' logits.
    # Upstream applies the budget only on budget_residue_mask. The default mask is empty.
    ala_over = (ala_count >= ala_budget) & budget_mask[residue]
    gly_over = (gly_count >= gly_budget) & budget_mask[residue]
    raw1 = raw1.at[0].set(jnp.where(ala_over, floor, raw1[0]))
    raw2 = raw2.at[0].set(jnp.where(ala_over, floor, raw2[0]))
    raw1 = raw1.at[7].set(jnp.where(gly_over, floor, raw1[7]))
    raw2 = raw2.at[7].set(jnp.where(gly_over, floor, raw2[7]))
    mixed = mix_sequence_probabilities(
      raw1,
      raw2,
      lambda_,
      sequence_temperature,
      on_logits=lambda_on_logits,
    )
    choice = categorical_draw(mixed, seq_uniforms[step_i])
    chosen = jnp.where(chain_mask[residue], input_sequence[residue], choice).astype(jnp.int32)
    sequence = sequence.at[residue].set(chosen)
    aa = jnp.where(chosen == 20, 7, chosen)
    defined = _CHI_MASK[aa]
    seq_emb = decoder.sequence_label_embedding.weight[chosen]
    ps1 = ns1[_N_DECODER, residue]
    pv1 = nv1[_N_DECODER, residue]
    ps2 = ns2[_N_DECODER, residue]
    pv2 = nv2[_N_DECODER, residue]
    prev1 = jnp.zeros((0,), dtype=dtype)
    prev2 = jnp.zeros((0,), dtype=dtype)
    for index in range(4):
      uniform1 = None if chi_temperature is None else chi_uniforms[step_i, index, 0]
      uniform2 = None if chi_temperature is None else chi_uniforms[step_i, index, 1]
      ps1, pv1, prev1, logit1, angle1, bin1 = _one_chi(
        decoder,
        joint,
        prot_scalars=ps1,
        prot_vectors=pv1,
        seq_emb=seq_emb,
        chi_prev=prev1,
        index=index,
        uniform=uniform1,
        chi_temperature=chi_temperature,
        chain_fixed=chain_mask[residue],
        input_angle=input_chi_1[residue, index],
        repack_all=repack_all,
        defined=defined[index],
      )
      ps2, pv2, prev2, logit2, angle2, bin2 = _one_chi(
        decoder,
        joint,
        prot_scalars=ps2,
        prot_vectors=pv2,
        seq_emb=seq_emb,
        chi_prev=prev2,
        index=index,
        uniform=uniform2,
        chi_temperature=chi_temperature,
        chain_fixed=chain_mask[residue],
        input_angle=input_chi_2[residue, index],
        repack_all=repack_all,
        defined=defined[index],
      )
      chi_logits1 = chi_logits1.at[residue, index].set(logit1)
      chi_logits2 = chi_logits2.at[residue, index].set(logit2)
      bins1 = bins1.at[residue, index].set(bin1)
      bins2 = bins2.at[residue, index].set(bin2)
      chi1 = chi1.at[residue, index].set(angle1)
      chi2 = chi2.at[residue, index].set(angle2)
    ala_count = ala_count + (chosen == 0).astype(jnp.int32)
    gly_count = gly_count + (chosen == 7).astype(jnp.int32)
    updated = (
      sequence,
      chi1,
      chi2,
      ns1,
      nv1,
      ns2,
      nv2,
      logits1.at[residue].set(raw1),
      logits2.at[residue].set(raw2),
      chi_logits1,
      chi_logits2,
      bins1,
      bins2,
      ala_count,
      gly_count,
      step_i + 1,
    )
    return updated, None

  # Sequential: residue t reads the embedding and χ RBF residue t-1 wrote.
  final, _unused = jax.lax.scan(body, init, decoding_order)
  return final


def tied_decode(
  encoder: LaserEncoder,
  decoder: LaserDecoder,
  joint: LaserJointDecode,
  backbone_1: np.ndarray,
  backbone_2: np.ndarray,
  ligand_coords_1: np.ndarray,
  ligand_coords_2: np.ndarray,
  ligand_atomic_numbers_1: np.ndarray,
  ligand_atomic_numbers_2: np.ndarray,
  ligand_subbatch_1: np.ndarray,
  ligand_subbatch_2: np.ndarray,
  period_index: np.ndarray,
  group_index: np.ndarray,
  structure: GraphStructure,
  sequence_indices: np.ndarray,
  chi_angles_1: np.ndarray,
  chi_angles_2: np.ndarray,
  chain_mask: np.ndarray,
  decoding_order: np.ndarray,
  sequence_uniforms: np.ndarray,
  chi_uniforms: np.ndarray,
  *,
  sequence_temperature: float | None = None,
  chi_temperature: float | None = None,
  lambda_: float = 0.0,
  repack_all: bool = False,
  lambda_on_logits: bool = False,
  disabled_residues: tuple[str, ...] = ("X",),
  seq_min_p: float = 0.0,
  chi_min_p: float = 0.0,
  fs_sequence_temp: float | None = None,
  disable_charged_fs: bool = False,
  ignore_chain_mask_zeros: bool = False,
  ala_budget: int = 4,
  gly_budget: int = 0,
  budget_mask: np.ndarray | None = None,
) -> TiedDecodeResult:
  """Encode both structures and scan one shared order."""
  _refuse_tied_knobs(
    seq_min_p=seq_min_p,
    chi_min_p=chi_min_p,
    fs_sequence_temp=fs_sequence_temp,
    disable_charged_fs=disable_charged_fs,
    ignore_chain_mask_zeros=ignore_chain_mask_zeros,
  )
  if not 0.0 <= lambda_ <= 1.0:
    msg = "tied_interpolation_lambda must lie in [0, 1]"
    raise ValueError(msg)
  if int(np.asarray(backbone_1).shape[0]) != int(np.asarray(backbone_2).shape[0]):
    msg = "tied structures must have the same residue count"
    raise ValueError(msg)
  disabled = np.zeros((21,), dtype=bool)
  for letter in disabled_residues:
    disabled[_LASER_ALPHABET.index(letter)] = True
  # NaN fixed χ becomes 0 degrees. See the module docstring.
  chi1 = np.nan_to_num(np.asarray(chi_angles_1, dtype=np.float64))
  chi2 = np.nan_to_num(np.asarray(chi_angles_2, dtype=np.float64))
  packed1 = _pack_encoded(
    encoder,
    backbone_1,
    ligand_coords_1,
    ligand_atomic_numbers_1,
    ligand_subbatch_1,
    period_index,
    group_index,
    structure,
  )
  packed2 = _pack_encoded(
    encoder,
    backbone_2,
    ligand_coords_2,
    ligand_atomic_numbers_2,
    ligand_subbatch_2,
    period_index,
    group_index,
    structure,
  )
  n_res = int(np.asarray(decoding_order).shape[0])
  seq_u = np.asarray(sequence_uniforms, dtype=np.float64).reshape(-1)
  if seq_u.shape[0] < n_res:
    seq_u = np.pad(seq_u, (0, n_res - seq_u.shape[0]))
  chi_u = _pack_chi_uniforms(chi_uniforms, n_res)
  final = _scan_tied(
    decoder,
    joint,
    packed1,
    packed2,
    jnp.asarray(decoding_order, dtype=jnp.int32),
    jnp.asarray(sequence_indices, dtype=jnp.int32),
    jnp.asarray(chi1),
    jnp.asarray(chi2),
    jnp.asarray(chain_mask, dtype=bool),
    jnp.asarray(disabled),
    jnp.asarray(np.zeros((n_res,), dtype=bool) if budget_mask is None else budget_mask),
    jnp.asarray(seq_u[:n_res]),
    jnp.asarray(chi_u),
    sequence_temperature=_tied_temperature(sequence_temperature),
    chi_temperature=chi_temperature,
    lambda_=lambda_,
    repack_all=repack_all,
    lambda_on_logits=lambda_on_logits,
    ala_budget=ala_budget,
    gly_budget=gly_budget,
  )
  (
    sequence,
    chi_d1,
    chi_d2,
    _ns1,
    _nv1,
    _ns2,
    _nv2,
    logits1,
    logits2,
    clog1,
    clog2,
    bins1,
    bins2,
    *_rest,
  ) = final
  return TiedDecodeResult(
    sequence=np.asarray(sequence, dtype=np.int64),
    sequence_logits_1=np.asarray(logits1),
    sequence_logits_2=np.asarray(logits2),
    chi_degrees_1=np.asarray(chi_d1),
    chi_degrees_2=np.asarray(chi_d2),
    chi_logits_1=np.asarray(clog1),
    chi_logits_2=np.asarray(clog2),
    chi_bins_1=np.asarray(bins1, dtype=np.int64),
    chi_bins_2=np.asarray(bins2, dtype=np.int64),
  )
