"""One autoregressive LASEr step: sequence, then χ1..χ4.

Ports one iteration of ``LASErMPNN.sample`` (``utils/model.py:776-934``).
``decode_residue`` puts the residue first in ``decoding_order``, which is the
dump's one-step setup. ``decode_order`` scans that same step over the full
permutation. ``decode_order_batched`` encodes once and vmaps that scan over a
leading sample axis. A residue becomes visible to later ones only through the sequence
embedding and the binned-χ RBF the step already writes into the carry.

Masked edges are not rebuilt from scratch. ``LaserDecoder._teacher_edges`` is
the bank-it-once construction: the masked branch reads the encoder snapshot
(``node_scalars[0]``) and the mask-class embedding, while the unmasked branch
reads the incoming sequence and χ. Calling it per decoder layer refreshes only
the unmasked source scalars, which is what the upstream layer loop does.

Math:
    $$T_{eff} = T$$ when no first-shell temperature is set, otherwise
    $$T_{eff,t} = T_{fs}$$ on a first-shell row and $$T$$ (or $$10^{-6}$$ when
    $$T$$ is absent) elsewhere. $$T_{eff}$$ is None only when both are absent,
    and that case is argmax with no min-p. Otherwise
    $$\\ell' = \\mathrm{minp}(\\ell),\\quad
      p = \\mathrm{softmax}(\\ell' / T_{eff}),$$
    and the stored logits are $$\\ell'$$, not $$\\ell'/T_{eff}$$.

Pseudocode:
    edges = teacher_edges(node_stack[layer], encoder, seq, chi)
    node_stack[layer+1, t] = decoder_layer(edges)[t]
    logits = linear(node_stack[-1, t]) + bias[t]
    logits[disabled or over-budget] = finfo.min
    logits[charged first shell] = -inf
    stored = logits if T_eff is None else minp(logits)
    sample = argmax(logits) if T_eff is None else categorical(softmax(stored / T_eff))
    for k in 0..3:
        chi_logits = head_k(scalars, seq_emb, chi_prev)
        bin = argmax or minp -> /chi_T -> categorical
        angle = remainder(center[bin] + offset + 180, 360) - 180
        chi_prev = cat(chi_prev, RBF(angle))          # every k
        write chi_enc, chi_deg, chi_logits where mask  # chi_enc is what later residues see
        if k < 3: scalars, vectors = chi_gvp(...)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from numpy.typing import NDArray

from aminx.model.laser.decoder import LaserDecoder, binned_degree_basis
from aminx.model.laser.encoders import LaserEncoder, encode_structure
from aminx.model.laser.graphs import (
  GraphStructure,
  pack_edges,
  scatter_edge_values,
)
from aminx.model.laser.layers import DenseMLP, apply_linear

_NODE = 256
_CHI_BINS = 72
_CHI_ANGLES = 4
_BIN_WIDTH = 5.0
_N_DECODER = 3
_MASK_CLASS = 21
_X = 20
_GLY = 7
_ALA = 0
# LASEr index order. X is not a rotatable amino acid; upstream maps it onto GLY
# before the chi-mask lookup, and the GLY row is all False.
_CHI_MASK = jnp.asarray(
  [
    [0, 0, 0, 0],  # A
    [1, 1, 1, 1],  # R
    [1, 1, 0, 0],  # N
    [1, 1, 0, 0],  # D
    [1, 1, 0, 0],  # C
    [1, 1, 1, 0],  # E
    [1, 1, 1, 0],  # Q
    [0, 0, 0, 0],  # G
    [1, 1, 0, 0],  # H
    [1, 1, 0, 0],  # I
    [1, 1, 0, 0],  # L
    [1, 1, 1, 1],  # K
    [1, 1, 1, 0],  # M
    [1, 1, 0, 0],  # F
    [1, 1, 0, 0],  # P
    [1, 1, 0, 0],  # S
    [1, 1, 0, 0],  # T
    [1, 1, 0, 0],  # W
    [1, 1, 1, 0],  # Y
    [1, 0, 0, 0],  # V
    [0, 0, 0, 0],  # X, same row upstream builds by substituting G
  ],
  dtype=jnp.bool_,
)
# R, D, E, K. Charged suppression is a different fill from the disabled-residue fill.
_CHARGED = jnp.asarray(
  [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  dtype=jnp.bool_,
)
_LASER_ALPHABET = "ARNDCEQGHILKMFPSTWYVX"


class JointStepInputs(eqx.Module):
  """Arrays for one residue. Temperatures and flags stay Python, so they are static."""

  ligand_scalars: Float[Array, "a h"]
  pr_edges: Float[Array, "l kp e"]
  pr_neighbours: Int[Array, "l kp"]
  pr_mask: Bool[Array, "l kp"]
  lp_edges: Float[Array, "l kl e"]
  lp_neighbours: Int[Array, "l kl"]
  lp_mask: Bool[Array, "l kl"]
  decoding_order: Int[Array, " l"]
  sequence: Int[Array, " l"]
  chi_degrees: Float[Array, "l 4"]
  input_sequence: Int[Array, " l"]
  input_chi: Float[Array, "l 4"]
  chain_mask: Bool[Array, " l"]
  first_shell: Bool[Array, " l"]
  budget_mask: Bool[Array, " l"]
  charged_mask: Bool[Array, " l"]
  disabled: Bool[Array, " 21"]
  bias: Float[Array, "l 21"]
  uniforms: Float[Array, " draws"]
  step_index: Int[Array, ""]
  ala_count: Int[Array, ""]
  gly_count: Int[Array, ""]
  node_scalars: Float[Array, "4 l h"]
  node_vectors: Float[Array, "4 l v 3"]


def categorical_draw(probs: Float[Array, " v"], uniform: Float[Array, ""]) -> Int[Array, ""]:
  """Inverse-CDF index, matching the oracle shim's float64 ``searchsorted(..., right=True)``.

  The shim promotes the categorical probabilities before the CDF, including on
  an f32 model. Doing the comparison in float32 moves a uniform that lands on
  a bin edge.
  """
  probabilities = probs.astype(jnp.float64)
  cdf = jnp.cumsum(probabilities)
  target = uniform.astype(jnp.float64) * cdf[-1]
  index = jnp.sum(cdf <= target).astype(jnp.int32)
  positions = jnp.arange(probabilities.shape[-1], dtype=jnp.int32)
  last = jnp.max(jnp.where(probabilities > 0, positions, jnp.int32(0)))
  clamped = jnp.minimum(index, last)
  return jnp.clip(clamped, 0, probabilities.shape[-1] - 1)


def minp_warp_logits(logits: Float[Array, " v"], min_p: float) -> Float[Array, " v"]:
  """Relative min-p. ``min_p == 0`` returns ``logits`` unchanged, with no ``-inf`` fill.

  ``min_p`` is a static Python float. The threshold is ``min_p`` times the top
  probability, and the highest token is kept even when it falls under that line.
  """
  if min_p == 0.0:
    return logits
  probs = jax.nn.softmax(logits)
  top = jnp.max(probs)
  remove = probs < (min_p * top)
  order = jnp.argsort(-probs)
  sorted_remove = remove[order].at[0].set(False)
  remove = sorted_remove[jnp.argsort(order)]
  return jnp.where(remove, -jnp.inf, logits)


def _draw(
  probs: Float[Array, " v"],
  uniforms: Float[Array, " draws"],
  cursor: Int[Array, ""],
) -> tuple[Int[Array, ""], Int[Array, ""]]:
  token = categorical_draw(probs, uniforms[cursor])
  return token, cursor + jnp.int32(1)


def _dtype_scalar(value: float, like: Float[Array, " v"]) -> Float[Array, ""]:
  """A 0-d tensor in ``like``'s dtype. A bare Python float would widen f32 division to f64."""
  return jnp.zeros((), dtype=like.dtype) + value


def _sequence_choice_array(
  logits: Float[Array, " 21"],
  uniforms: Float[Array, " draws"],
  cursor: Int[Array, ""],
  first_shell: Bool[Array, ""],
  *,
  sequence_temperature: Float[Array, ""],
  fs_sequence_temp: float | None,
  seq_min_p: float,
) -> tuple[Float[Array, " 21"], Int[Array, ""], Int[Array, ""]]:
  """Sequence choice when ``sequence_temperature`` is a 0-d array.

  NaN is the batched stand-in for ``None``. With no first-shell temperature
  that row is argmax and consumes no sequence draw. Otherwise NaN uses the
  ``1e-6`` fallback the scalar ``None`` branch uses.
  """
  sampling_logits = minp_warp_logits(logits, seq_min_p)
  is_none = jnp.isnan(sequence_temperature)
  if fs_sequence_temp is None:
    safe = jnp.where(
      is_none,
      jnp.asarray(1.0, dtype=sampling_logits.dtype),
      sequence_temperature.astype(sampling_logits.dtype),
    )
    choice_s, cursor_s = _draw(jax.nn.softmax(sampling_logits / safe), uniforms, cursor)
    stored = jnp.where(is_none, logits, sampling_logits)
    choice = jnp.where(is_none, jnp.argmax(logits).astype(jnp.int32), choice_s)
    return stored, choice, jnp.where(is_none, cursor, cursor_s)
  fallback = jnp.where(
    is_none,
    _dtype_scalar(1e-6, sampling_logits),
    sequence_temperature.astype(sampling_logits.dtype),
  )
  t_eff = jnp.where(
    first_shell,
    _dtype_scalar(fs_sequence_temp, sampling_logits),
    fallback,
  )
  choice, cursor_out = _draw(jax.nn.softmax(sampling_logits / t_eff), uniforms, cursor)
  return sampling_logits, choice, cursor_out


class LaserJointDecode(eqx.Module):
  """χ offset heads plus one joint-decode step. The decoder and encoder stay outside."""

  chi_offset_prediction_layers: tuple[DenseMLP, DenseMLP, DenseMLP, DenseMLP]

  def __init__(self, *, key: PRNGKeyArray) -> None:
    keys = jax.random.split(key, _CHI_ANGLES)
    # Offset input is the χ logit concatenated with the sampled bin's one-hot.
    self.chi_offset_prediction_layers = (
      DenseMLP(2 * _CHI_BINS, _NODE, 1, key=keys[0]),
      DenseMLP(2 * _CHI_BINS, _NODE, 1, key=keys[1]),
      DenseMLP(2 * _CHI_BINS, _NODE, 1, key=keys[2]),
      DenseMLP(2 * _CHI_BINS, _NODE, 1, key=keys[3]),
    )

  def __call__(
    self,
    decoder: LaserDecoder,
    inputs: JointStepInputs,
    *,
    sequence_temperature: float | None,
    chi_temperature: float | None,
    fs_sequence_temp: float | None,
    seq_min_p: float,
    chi_min_p: float,
    ala_budget: int,
    gly_budget: int,
    ignore_chain_mask_zeros: bool,
    repack_all: bool,
  ) -> tuple[
    Float[Array, " 21"],
    Int[Array, ""],
    Float[Array, "4 72"],
    Float[Array, " 4"],
    Int[Array, ""],
  ]:
    """Decode ``inputs.step_index`` from the incoming carry. Five draws when both temperatures are set.

    Returns stored sequence logits, the chosen index, χ logits, χ degrees, and
    how many uniform draws this step consumed.
    """
    stored, chosen, chi_logits, chi_degrees, cursor, _node_s, _node_v = _run_joint_step(
      self,
      decoder,
      inputs,
      sequence_temperature=sequence_temperature,
      chi_temperature=chi_temperature,
      fs_sequence_temp=fs_sequence_temp,
      seq_min_p=seq_min_p,
      chi_min_p=chi_min_p,
      ala_budget=ala_budget,
      gly_budget=gly_budget,
      ignore_chain_mask_zeros=ignore_chain_mask_zeros,
      repack_all=repack_all,
    )
    return stored, chosen, chi_logits, chi_degrees, cursor


@eqx.filter_jit
def _run_joint_step(  # noqa: PLR0915 — χ loop stays in this traced body so L-DRV still sees it
  joint: LaserJointDecode,
  decoder: LaserDecoder,
  inputs: JointStepInputs,
  *,
  sequence_temperature: float | None,
  chi_temperature: float | None,
  fs_sequence_temp: float | None,
  seq_min_p: float,
  chi_min_p: float,
  ala_budget: int,
  gly_budget: int,
  ignore_chain_mask_zeros: bool,
  repack_all: bool,
) -> tuple[
  Float[Array, " 21"],
  Int[Array, ""],
  Float[Array, "4 72"],
  Float[Array, " 4"],
  Int[Array, ""],
  Float[Array, "4 l h"],
  Float[Array, "4 l v 3"],
]:
  """One residue. Returns the five step outputs plus the updated node stacks.

  The stacks are part of the carry: the next residue's unmasked edges read
  the decoder row this step wrote. ``__call__`` drops them so the one-step
  dump keeps its five-tuple.
  """
  step = inputs.step_index
  # NaN χ is the initial chi_enc. Edges read it as zeros; a written angle is
  # the RBF later residues will see.
  chi_flat = jnp.reshape(
    jnp.nan_to_num(binned_degree_basis(inputs.chi_degrees)),
    (inputs.chi_degrees.shape[0], _CHI_ANGLES * _CHI_BINS),
  )
  node_s = inputs.node_scalars
  node_v = inputs.node_vectors
  for index in range(3):
    # Masked branch keeps node_scalars[0], the encoder snapshot banked at init.
    features, row_mask, slot_neighbours = decoder._teacher_edges(  # noqa: SLF001
      node_s[index],
      node_s[0],
      inputs.pr_edges,
      inputs.pr_neighbours,
      inputs.pr_mask,
      inputs.sequence,
      chi_flat,
      inputs.decoding_order,
    )
    empty = jnp.zeros((*inputs.pr_neighbours.shape, 0), dtype=node_s.dtype)
    updated_s, updated_v, _edges = decoder.protein_decoder_layers[index].hetgat(
      node_s[index],
      node_v[index],
      (features, inputs.ligand_scalars),
      (slot_neighbours, inputs.lp_neighbours),
      (empty, inputs.lp_edges),
      (row_mask, inputs.lp_mask),
      (False, False),
    )
    # Only this residue is written. Other rows of the upper stacks stay zero
    # until their own step, which is what the next residue's unmasked edges read.
    node_s = node_s.at[index + 1, step].set(updated_s[step])
    node_v = node_v.at[index + 1, step].set(updated_v[step])

  logits = apply_linear(decoder.sequence_output_layer, node_s[_N_DECODER, step])
  logits = logits + inputs.bias[step]
  # Disabled residues and over-budget ALA/GLY use finfo.min. Charged
  # suppression uses -inf. The two fills are deliberately different.
  sampling = inputs.chain_mask[step] if ignore_chain_mask_zeros else ~inputs.chain_mask[step]
  floor = jnp.finfo(logits.dtype).min
  logits = jnp.where(sampling & inputs.disabled, floor, logits)
  ala_over = (inputs.ala_count >= ala_budget) & inputs.budget_mask[step]
  gly_over = (inputs.gly_count >= gly_budget) & inputs.budget_mask[step]
  logits = logits.at[_ALA].set(jnp.where(ala_over, floor, logits[_ALA]))
  logits = logits.at[_GLY].set(jnp.where(gly_over, floor, logits[_GLY]))
  charged_here = inputs.charged_mask[step]
  logits = jnp.where(charged_here & _CHARGED, -jnp.inf, logits)

  cursor = jnp.int32(0)
  # T_eff is None only when no temperature and no first-shell temperature are
  # set. That path is argmax and does not warp. Otherwise min-p runs on the
  # untempered logits, and the temperature divides the warped tensor.
  # A traced temperature (the batched path) is a 0-d array. NaN means this
  # sample's sequence temperature is None. Python floats stay on the branch
  # below so a single ``decode_order`` call is unchanged.
  if isinstance(sequence_temperature, jax.Array):
    stored, choice, cursor = _sequence_choice_array(
      logits,
      inputs.uniforms,
      cursor,
      inputs.first_shell[step],
      sequence_temperature=sequence_temperature,
      fs_sequence_temp=fs_sequence_temp,
      seq_min_p=seq_min_p,
    )
  elif sequence_temperature is None and fs_sequence_temp is None:
    stored = logits
    choice = jnp.argmax(logits).astype(jnp.int32)
  else:
    stored = minp_warp_logits(logits, seq_min_p)
    if fs_sequence_temp is None:
      if sequence_temperature is None:
        msg = "argmax is the only path with no sequence temperature"
        raise RuntimeError(msg)
      tempered = stored / sequence_temperature
    else:
      # First-shell rows use fs_sequence_temp. Every other row uses T, or
      # 1e-6 when T itself is absent, so T_eff is never None in this branch.
      fallback = 1e-6 if sequence_temperature is None else sequence_temperature
      t_eff = jnp.where(
        inputs.first_shell[step],
        _dtype_scalar(fs_sequence_temp, stored),
        _dtype_scalar(fallback, stored),
      )
      tempered = stored / t_eff
    choice, cursor = _draw(jax.nn.softmax(tempered), inputs.uniforms, cursor)
  # Stored sequence_logits are post-min-p and untempered (bias already in
  # ``logits``, and the oracle recomputes with a zero bias row). They are
  # not the pre-min-p tensor: removed tokens stay -inf and the draw
  # renormalises over the tokens that remain.
  if ignore_chain_mask_zeros:
    chosen = choice
  else:
    chosen = jnp.where(inputs.chain_mask[step], inputs.input_sequence[step], choice)

  aa_for_chi = jnp.where(chosen == _X, _GLY, chosen)
  chi_mask = _CHI_MASK[aa_for_chi]
  seq_emb = decoder.sequence_label_embedding.weight[chosen]
  prot_scalars = node_s[_N_DECODER, step]
  prot_vectors = node_v[_N_DECODER, step]
  chi_prev = jnp.zeros((0,), dtype=stored.dtype)
  chi_logits_row = jnp.zeros((_CHI_ANGLES, _CHI_BINS), dtype=stored.dtype)
  chi_degrees_row = jnp.full((_CHI_ANGLES,), jnp.nan, dtype=stored.dtype)
  centers = jnp.arange(-180.0, 180.0, _BIN_WIDTH, dtype=stored.dtype)
  for index in range(4):
    chi_logits_k = decoder.chi_prediction_layers[index](
      jnp.concatenate((prot_scalars, seq_emb, chi_prev)),
    )
    if chi_temperature is None:
      stored_chi = chi_logits_k
      bin_index = jnp.argmax(chi_logits_k).astype(jnp.int32)
    else:
      stored_chi = minp_warp_logits(chi_logits_k, chi_min_p)
      bin_index, cursor = _draw(
        jax.nn.softmax(stored_chi / chi_temperature),
        inputs.uniforms,
        cursor,
      )
    one_hot = jax.nn.one_hot(bin_index, _CHI_BINS, dtype=stored_chi.dtype)
    offset = joint.chi_offset_prediction_layers[index](
      jnp.concatenate((stored_chi, one_hot)),
    )[0]
    # Floor-mod, matching torch.remainder. fmod would keep the sign of a
    # negative numerator and land in a different bin.
    angle = jnp.remainder(centers[bin_index] + offset + 180.0, 360.0) - 180.0
    if not ignore_chain_mask_zeros and not repack_all:
      use_input = inputs.chain_mask[step] & ~jnp.isnan(inputs.input_chi[step, index])
      angle = jnp.where(use_input, inputs.input_chi[step, index], angle)
    enc = binned_degree_basis(jnp.reshape(angle, (1, 1)))[0, 0]
    # Always append. chi_enc is what later residues see, and it is written
    # only where the χ mask allows; the next head of this residue still
    # conditions on the angle just sampled, mask or not.
    chi_prev = jnp.concatenate((chi_prev, enc))
    keep = chi_mask[index]
    chi_logits_row = chi_logits_row.at[index].set(
      jnp.where(keep, stored_chi, chi_logits_row[index]),
    )
    chi_degrees_row = chi_degrees_row.at[index].set(
      jnp.where(keep, angle, chi_degrees_row[index]),
    )
    if index == 3:
      continue
    updated_scalars, updated_vectors = decoder.chi_vector_layer_norms[index](
      *decoder.chi_vector_update_layers[index](
        jnp.concatenate((prot_scalars, chi_prev))[None, :],
        prot_vectors[None, :, :],
      ),
    )
    prot_scalars = updated_scalars[0]
    prot_vectors = updated_vectors[0]
  return stored, chosen, chi_logits_row, chi_degrees_row, cursor, node_s, node_v


@eqx.filter_jit
def _decode_step_jit(
  joint: LaserJointDecode,
  decoder: LaserDecoder,
  inputs: JointStepInputs,
  *,
  sequence_temperature: float | None,
  chi_temperature: float | None,
  fs_sequence_temp: float | None,
  seq_min_p: float,
  chi_min_p: float,
  ala_budget: int,
  gly_budget: int,
  ignore_chain_mask_zeros: bool,
  repack_all: bool,
) -> tuple[
  Float[Array, " 21"],
  Int[Array, ""],
  Float[Array, "4 72"],
  Float[Array, " 4"],
  Int[Array, ""],
]:
  return joint(
    decoder,
    inputs,
    sequence_temperature=sequence_temperature,
    chi_temperature=chi_temperature,
    fs_sequence_temp=fs_sequence_temp,
    seq_min_p=seq_min_p,
    chi_min_p=chi_min_p,
    ala_budget=ala_budget,
    gly_budget=gly_budget,
    ignore_chain_mask_zeros=ignore_chain_mask_zeros,
    repack_all=repack_all,
  )


@dataclass(frozen=True, slots=True)
class DecodeForward:
  """One residue of ``sample``, plus the uniform-draw count for that residue."""

  step_index: NDArray[np.int64]
  sequence_logits: NDArray[np.floating]
  sequence_index: NDArray[np.int64]
  chi_logits: NDArray[np.floating]
  chi_degrees: NDArray[np.floating]
  uniforms_consumed: NDArray[np.int64]


def _disabled_mask(letters: tuple[str, ...]) -> NDArray[np.bool_]:
  mask = np.zeros((len(_LASER_ALPHABET),), dtype=bool)
  for letter in letters:
    mask[_LASER_ALPHABET.index(letter)] = True
  return mask


def _rank_zero_order(n_res: int, step_index: int) -> NDArray[np.int32]:
  """Permutation with ``step_index`` first.

  Rank 0 makes every protein edge into that residue masked, which is the
  dump's ``order[0]`` condition. The relative order of the other residues
  does not enter this step's edges.
  """
  order = np.empty((n_res,), dtype=np.int32)
  order[0] = step_index
  cursor = 1
  for index in range(n_res):
    if index == step_index:
      continue
    order[cursor] = index
    cursor += 1
  return order


def decode_residue(
  encoder: LaserEncoder,
  decoder: LaserDecoder,
  joint: LaserJointDecode,
  backbone: Float[NDArray[np.floating], "l 5 3"],
  ligand_coords: Float[NDArray[np.floating], "a 3"],
  ligand_atomic_numbers: Int[NDArray[np.integer], " a"],
  ligand_subbatch: Int[NDArray[np.integer], " a"],
  period_index: Int[NDArray[np.integer], " 118"],
  group_index: Int[NDArray[np.integer], " 118"],
  structure: GraphStructure,
  sequence_indices: Int[NDArray[np.integer], " l"],
  chi_angles: Float[NDArray[np.floating], "l 4"],
  first_shell: Bool[NDArray[np.bool_], " l"],
  step_index: int,
  uniforms: Float[NDArray[np.floating], " draws"],
  *,
  sequence_temperature: float | None = None,
  chi_temperature: float | None = None,
  fs_sequence_temp: float | None = None,
  seq_min_p: float = 0.0,
  chi_min_p: float = 0.0,
  ala_budget: int = 4,
  gly_budget: int = 0,
  disabled_residues: tuple[str, ...] = ("X",),
  ignore_chain_mask_zeros: bool = False,
  repack_all: bool = False,
  protein_loop: bool = True,
) -> DecodeForward:
  """Encode, initialise the sample carry, and decode the single residue ``step_index``.

  The carry starts as the spec requires: sequence 21 on every row, including
  fixed ones, χ degrees NaN, χ logits left at zero until written, and
  ``node_stack[0]`` equal to the encoder with the upper layers zero.
  """
  encoded = encode_structure(
    encoder,
    backbone,
    ligand_coords,
    ligand_atomic_numbers,
    ligand_subbatch,
    period_index,
    group_index,
    structure,
    protein_loop=protein_loop,
  )
  n_res = int(encoded.prot_scalars.shape[0])
  packed_pr = pack_edges(n_res, encoded.pr_pr_idx)
  packed_lp = pack_edges(n_res, encoded.lig_pr_idx)
  pr_edges = scatter_edge_values(n_res, packed_pr, encoded.pr_pr_eattr)
  lp_edges = scatter_edge_values(n_res, packed_lp, encoded.lig_pr_eattr)
  dtype = encoded.prot_scalars.dtype
  draws = np.asarray(uniforms, dtype=np.float64)
  if draws.ndim == 2:
    draws = draws[0]
  chain_mask = np.zeros((n_res,), dtype=bool)
  chain_mask[step_index] = True
  scalars = np.asarray(encoded.prot_scalars)
  vectors = np.asarray(encoded.prot_vectors)
  # stack[0] is the encoder. The upper layers stay zero until a step writes
  # its own residue; this call writes only ``step_index``.
  node_s = np.zeros((4, *scalars.shape), dtype=scalars.dtype)
  node_v = np.zeros((4, *vectors.shape), dtype=vectors.dtype)
  node_s[0] = scalars
  node_v[0] = vectors
  inputs = JointStepInputs(
    ligand_scalars=jnp.asarray(encoded.lig_scalars),
    pr_edges=jnp.asarray(pr_edges),
    pr_neighbours=jnp.asarray(packed_pr.neighbours),
    pr_mask=jnp.asarray(packed_pr.mask),
    lp_edges=jnp.asarray(lp_edges),
    lp_neighbours=jnp.asarray(packed_lp.neighbours),
    lp_mask=jnp.asarray(packed_lp.mask),
    decoding_order=jnp.asarray(_rank_zero_order(n_res, step_index)),
    sequence=jnp.full((n_res,), _MASK_CLASS, dtype=jnp.int32),
    chi_degrees=jnp.full((n_res, _CHI_ANGLES), jnp.nan, dtype=dtype),
    input_sequence=jnp.asarray(sequence_indices),
    input_chi=jnp.asarray(chi_angles, dtype=dtype),
    chain_mask=jnp.asarray(chain_mask),
    first_shell=jnp.asarray(first_shell),
    budget_mask=jnp.zeros((n_res,), dtype=jnp.bool_),
    charged_mask=jnp.zeros((n_res,), dtype=jnp.bool_),
    disabled=jnp.asarray(_disabled_mask(disabled_residues)),
    bias=jnp.zeros((n_res, 21), dtype=dtype),
    uniforms=jnp.asarray(draws),
    step_index=jnp.asarray(step_index, dtype=jnp.int32),
    ala_count=jnp.zeros((), dtype=jnp.int32),
    gly_count=jnp.zeros((), dtype=jnp.int32),
    node_scalars=jnp.asarray(node_s),
    node_vectors=jnp.asarray(node_v),
  )
  stored, chosen, chi_logits, chi_degrees, consumed = _decode_step_jit(
    joint,
    decoder,
    inputs,
    sequence_temperature=sequence_temperature,
    chi_temperature=chi_temperature,
    fs_sequence_temp=fs_sequence_temp,
    seq_min_p=seq_min_p,
    chi_min_p=chi_min_p,
    ala_budget=ala_budget,
    gly_budget=gly_budget,
    ignore_chain_mask_zeros=ignore_chain_mask_zeros,
    repack_all=repack_all,
  )
  return DecodeForward(
    step_index=np.asarray(step_index, dtype=np.int64),
    sequence_logits=np.asarray(stored),
    sequence_index=np.asarray(chosen, dtype=np.int64),
    chi_logits=np.asarray(chi_logits),
    chi_degrees=np.asarray(chi_degrees),
    uniforms_consumed=np.asarray(consumed, dtype=np.int64),
  )


# One sequence draw plus four χ draws. The step's cursor restarts at 0, so the
# scan hands it this many fresh uniforms rather than the absolute stream index.
_DRAWS_PER_STEP = 5


class _Carry(NamedTuple):
  sequence: Int[Array, " l"]
  chi_degrees: Float[Array, "l 4"]
  node_scalars: Float[Array, "4 l h"]
  node_vectors: Float[Array, "4 l v 3"]
  ala_count: Int[Array, ""]
  gly_count: Int[Array, ""]
  sequence_logits: Float[Array, "l 21"]
  chi_logits: Float[Array, "l 4 72"]
  cursor: Int[Array, ""]


def _sampling(
  sequence_temperature: float | None,
  chi_temperature: float | None,
  fs_sequence_temp: float | None,
) -> bool:
  return (
    sequence_temperature is not None or chi_temperature is not None or fs_sequence_temp is not None
  )


def _commit_residue(
  carry: _Carry,
  residue: Int[Array, ""],
  joint: LaserJointDecode,
  decoder: LaserDecoder,
  *,
  ligand_scalars: Float[Array, "a h"],
  pr_edges: Float[Array, "l kp e"],
  pr_neighbours: Int[Array, "l kp"],
  pr_mask: Bool[Array, "l kp"],
  lp_edges: Float[Array, "l kl e"],
  lp_neighbours: Int[Array, "l kl"],
  lp_mask: Bool[Array, "l kl"],
  decoding_order: Int[Array, " l"],
  input_sequence: Int[Array, " l"],
  input_chi: Float[Array, "l 4"],
  chain_mask: Bool[Array, " l"],
  first_shell: Bool[Array, " l"],
  budget_mask: Bool[Array, " l"],
  charged_mask: Bool[Array, " l"],
  disabled: Bool[Array, " 21"],
  bias: Float[Array, "l 21"],
  uniforms: Float[Array, " draws"],
  sequence_temperature: float | None,
  chi_temperature: float | None,
  fs_sequence_temp: float | None,
  seq_min_p: float,
  chi_min_p: float,
  ala_budget: int,
  gly_budget: int,
  ignore_chain_mask_zeros: bool,
  repack_all: bool,
) -> _Carry:
  """Write one residue into the carry. Later residues see it only through this write."""
  if _sampling(sequence_temperature, chi_temperature, fs_sequence_temp):
    step_uniforms = jax.lax.dynamic_slice(uniforms, (carry.cursor,), (_DRAWS_PER_STEP,))
  else:
    step_uniforms = uniforms
  inputs = JointStepInputs(
    ligand_scalars=ligand_scalars,
    pr_edges=pr_edges,
    pr_neighbours=pr_neighbours,
    pr_mask=pr_mask,
    lp_edges=lp_edges,
    lp_neighbours=lp_neighbours,
    lp_mask=lp_mask,
    decoding_order=decoding_order,
    sequence=carry.sequence,
    chi_degrees=carry.chi_degrees,
    input_sequence=input_sequence,
    input_chi=input_chi,
    chain_mask=chain_mask,
    first_shell=first_shell,
    budget_mask=budget_mask,
    charged_mask=charged_mask,
    disabled=disabled,
    bias=bias,
    uniforms=step_uniforms,
    step_index=residue,
    ala_count=carry.ala_count,
    gly_count=carry.gly_count,
    node_scalars=carry.node_scalars,
    node_vectors=carry.node_vectors,
  )
  stored, chosen, chi_logits, chi_degrees, consumed, node_s, node_v = _run_joint_step(
    joint,
    decoder,
    inputs,
    sequence_temperature=sequence_temperature,
    chi_temperature=chi_temperature,
    fs_sequence_temp=fs_sequence_temp,
    seq_min_p=seq_min_p,
    chi_min_p=chi_min_p,
    ala_budget=ala_budget,
    gly_budget=gly_budget,
    ignore_chain_mask_zeros=ignore_chain_mask_zeros,
    repack_all=repack_all,
  )
  chosen_i = chosen.astype(jnp.int32)
  return _Carry(
    sequence=carry.sequence.at[residue].set(chosen_i),
    chi_degrees=carry.chi_degrees.at[residue].set(chi_degrees),
    node_scalars=node_s,
    node_vectors=node_v,
    ala_count=carry.ala_count + (chosen_i == _ALA).astype(jnp.int32),
    gly_count=carry.gly_count + (chosen_i == _GLY).astype(jnp.int32),
    sequence_logits=carry.sequence_logits.at[residue].set(stored),
    chi_logits=carry.chi_logits.at[residue].set(chi_logits),
    cursor=carry.cursor + consumed.astype(jnp.int32),
  )


@eqx.filter_jit
def _scan_residues(
  joint: LaserJointDecode,
  decoder: LaserDecoder,
  ligand_scalars: Float[Array, "a h"],
  pr_edges: Float[Array, "l kp e"],
  pr_neighbours: Int[Array, "l kp"],
  pr_mask: Bool[Array, "l kp"],
  lp_edges: Float[Array, "l kl e"],
  lp_neighbours: Int[Array, "l kl"],
  lp_mask: Bool[Array, "l kl"],
  decoding_order: Int[Array, " l"],
  input_sequence: Int[Array, " l"],
  input_chi: Float[Array, "l 4"],
  chain_mask: Bool[Array, " l"],
  first_shell: Bool[Array, " l"],
  budget_mask: Bool[Array, " l"],
  charged_mask: Bool[Array, " l"],
  disabled: Bool[Array, " 21"],
  bias: Float[Array, "l 21"],
  uniforms: Float[Array, " draws"],
  node_scalars: Float[Array, "4 l h"],
  node_vectors: Float[Array, "4 l v 3"],
  *,
  sequence_temperature: float | None,
  chi_temperature: float | None,
  fs_sequence_temp: float | None,
  seq_min_p: float,
  chi_min_p: float,
  ala_budget: int,
  gly_budget: int,
  ignore_chain_mask_zeros: bool,
  repack_all: bool,
) -> tuple[
  Int[Array, " l"],
  Float[Array, "l 21"],
  Float[Array, "l 4"],
  Float[Array, "l 4 72"],
  Float[Array, "4 l h"],
]:
  """Scan the single step over ``decoding_order``.

  A ``chain_mask == 0`` row under ``ignore_chain_mask_zeros`` does not run the
  step, but it stays in ``decoding_order``, so later residues still treat it as
  a predecessor and read embed(21), zero χ and the untouched stack row.
  """
  dtype = node_scalars.dtype
  length = node_scalars.shape[1]
  init = _Carry(
    sequence=jnp.full((length,), _MASK_CLASS, dtype=jnp.int32),
    chi_degrees=jnp.full((length, _CHI_ANGLES), jnp.nan, dtype=dtype),
    node_scalars=node_scalars,
    node_vectors=node_vectors,
    ala_count=jnp.int32(0),
    gly_count=jnp.int32(0),
    sequence_logits=jnp.zeros((length, 21), dtype=dtype),
    chi_logits=jnp.zeros((length, _CHI_ANGLES, _CHI_BINS), dtype=dtype),
    cursor=jnp.int32(0),
  )

  def body(carry: _Carry, residue: Int[Array, ""]) -> tuple[_Carry, None]:
    def commit(current: _Carry) -> _Carry:
      return _commit_residue(
        current,
        residue,
        joint,
        decoder,
        ligand_scalars=ligand_scalars,
        pr_edges=pr_edges,
        pr_neighbours=pr_neighbours,
        pr_mask=pr_mask,
        lp_edges=lp_edges,
        lp_neighbours=lp_neighbours,
        lp_mask=lp_mask,
        decoding_order=decoding_order,
        input_sequence=input_sequence,
        input_chi=input_chi,
        chain_mask=chain_mask,
        first_shell=first_shell,
        budget_mask=budget_mask,
        charged_mask=charged_mask,
        disabled=disabled,
        bias=bias,
        uniforms=uniforms,
        sequence_temperature=sequence_temperature,
        chi_temperature=chi_temperature,
        fs_sequence_temp=fs_sequence_temp,
        seq_min_p=seq_min_p,
        chi_min_p=chi_min_p,
        ala_budget=ala_budget,
        gly_budget=gly_budget,
        ignore_chain_mask_zeros=ignore_chain_mask_zeros,
        repack_all=repack_all,
      )

    # Upstream ``continue``s. The row is still a predecessor via the order mask.
    if ignore_chain_mask_zeros:
      updated = jax.lax.cond(chain_mask[residue], commit, lambda current: current, carry)
    else:
      updated = commit(carry)
    return updated, None

  # Sequential: residue t reads the embedding and χ RBF residue t-1 wrote.
  final, _unused = jax.lax.scan(body, init, decoding_order)
  sequence = final.sequence
  if ignore_chain_mask_zeros:
    sequence = jnp.where(chain_mask, sequence, jnp.int32(_X))
  return sequence, final.sequence_logits, final.chi_degrees, final.chi_logits, final.node_scalars


@dataclass(frozen=True, slots=True)
class JointDecodeResult:
  """One full pass of ``sample``: sequence, χ, and the decoder rows the scan kept."""

  sequence: NDArray[np.int64]
  sequence_logits: NDArray[np.floating]
  chi_degrees: NDArray[np.floating]
  chi_logits: NDArray[np.floating]
  chi_bins: NDArray[np.int64]
  node_scalars: NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class JointDecodeBatch:
  """``decode_order_batched``: the same fields as ``JointDecodeResult``, with a leading sample axis."""

  sequence: NDArray[np.int64]
  sequence_logits: NDArray[np.floating]
  chi_degrees: NDArray[np.floating]
  chi_logits: NDArray[np.floating]
  chi_bins: NDArray[np.int64]
  node_scalars: NDArray[np.floating]


def chi_position_mask(sequence: NDArray[np.integer]) -> NDArray[np.bool_]:
  """χ slots defined for each letter. X uses the GLY row, which is all False."""
  letters = np.asarray(sequence)
  gly = np.where(letters == _X, _GLY, letters)
  return np.asarray(_CHI_MASK)[gly]


def circular_abs_delta_deg(
  left: NDArray[np.floating],
  right: NDArray[np.floating],
) -> NDArray[np.floating]:
  """Smaller arc between two degree values. ±180 is a short gap, not 360."""
  delta = np.mod(
    np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)),
    360.0,
  )
  return np.minimum(delta, 360.0 - delta)


def decode_order(
  encoder: LaserEncoder,
  decoder: LaserDecoder,
  joint: LaserJointDecode,
  backbone: Float[NDArray[np.floating], "l 5 3"],
  ligand_coords: Float[NDArray[np.floating], "a 3"],
  ligand_atomic_numbers: Int[NDArray[np.integer], " a"],
  ligand_subbatch: Int[NDArray[np.integer], " a"],
  period_index: Int[NDArray[np.integer], " 118"],
  group_index: Int[NDArray[np.integer], " 118"],
  structure: GraphStructure,
  sequence_indices: Int[NDArray[np.integer], " l"],
  chi_angles: Float[NDArray[np.floating], "l 4"],
  chain_mask: Bool[NDArray[np.bool_], " l"],
  first_shell: Bool[NDArray[np.bool_], " l"],
  decoding_order: Int[NDArray[np.integer], " l"],
  uniforms: Float[NDArray[np.floating], " draws"],
  *,
  sequence_temperature: float | None = None,
  chi_temperature: float | None = None,
  fs_sequence_temp: float | None = None,
  seq_min_p: float = 0.0,
  chi_min_p: float = 0.0,
  ala_budget: int = 4,
  gly_budget: int = 0,
  disabled_residues: tuple[str, ...] = ("X",),
  ignore_chain_mask_zeros: bool = False,
  repack_all: bool = False,
  repack_only: bool = False,
  budget_mask: Bool[NDArray[np.bool_], " l"] | None = None,
  charged_mask: Bool[NDArray[np.bool_], " l"] | None = None,
  bias: Float[NDArray[np.floating], "l 21"] | None = None,
  protein_loop: bool = True,
) -> JointDecodeResult:
  """Encode once, then scan ``decoding_order``. Fixed letters enter at their own step.

  ``repack_only`` fixes every sequence (``chain_mask`` all True) and repacks χ,
  matching ``run_inference.py`` before ``sample``. ``repack_all`` alone leaves
  ``chain_mask`` as given and only drops input-χ retention.
  """
  prepared = _encoded_scan_inputs(
    encoder,
    backbone,
    ligand_coords,
    ligand_atomic_numbers,
    ligand_subbatch,
    period_index,
    group_index,
    structure,
    sequence_indices,
    chi_angles,
    chain_mask,
    first_shell,
    budget_mask=budget_mask,
    charged_mask=charged_mask,
    bias=bias,
    disabled_residues=disabled_residues,
    repack_all=repack_all,
    repack_only=repack_only,
    protein_loop=protein_loop,
  )
  order = np.asarray(decoding_order, dtype=np.int32)
  if int(order.shape[0]) != prepared.n_res:
    msg = f"decoding_order length {order.shape[0]} != structure length {prepared.n_res}"
    raise ValueError(msg)
  draws = _pad_uniform_row(
    uniforms,
    prepared.n_res,
    sampling=_sampling(sequence_temperature, chi_temperature, fs_sequence_temp),
  )
  sequence, seq_logits, chi_degrees, chi_logits, nodes = _scan_residues(
    joint,
    decoder,
    prepared.ligand_scalars,
    prepared.pr_edges,
    prepared.pr_neighbours,
    prepared.pr_mask,
    prepared.lp_edges,
    prepared.lp_neighbours,
    prepared.lp_mask,
    jnp.asarray(order),
    prepared.input_sequence,
    prepared.input_chi,
    prepared.chain_mask,
    prepared.first_shell,
    prepared.budget_mask,
    prepared.charged_mask,
    prepared.disabled,
    prepared.bias,
    jnp.asarray(draws),
    prepared.node_scalars,
    prepared.node_vectors,
    sequence_temperature=sequence_temperature,
    chi_temperature=chi_temperature,
    fs_sequence_temp=fs_sequence_temp,
    seq_min_p=seq_min_p,
    chi_min_p=chi_min_p,
    ala_budget=ala_budget,
    gly_budget=gly_budget,
    ignore_chain_mask_zeros=ignore_chain_mask_zeros,
    repack_all=prepared.effective_repack,
  )
  chi_logits_np = np.asarray(chi_logits)
  return JointDecodeResult(
    sequence=np.asarray(sequence, dtype=np.int64),
    sequence_logits=np.asarray(seq_logits),
    chi_degrees=np.asarray(chi_degrees),
    chi_logits=chi_logits_np,
    chi_bins=np.argmax(chi_logits_np, axis=-1).astype(np.int64),
    node_scalars=np.asarray(nodes),
  )


class _ScanInputs(NamedTuple):
  """Encoder broadcast plus the host masks one structure shares across samples."""

  ligand_scalars: Array
  pr_edges: Array
  pr_neighbours: Array
  pr_mask: Array
  lp_edges: Array
  lp_neighbours: Array
  lp_mask: Array
  input_sequence: Array
  input_chi: Array
  chain_mask: Array
  first_shell: Array
  budget_mask: Array
  charged_mask: Array
  disabled: Array
  bias: Array
  node_scalars: Array
  node_vectors: Array
  n_res: int
  effective_repack: bool


def _encoded_scan_inputs(
  encoder: LaserEncoder,
  backbone: Float[NDArray[np.floating], "l 5 3"],
  ligand_coords: Float[NDArray[np.floating], "a 3"],
  ligand_atomic_numbers: Int[NDArray[np.integer], " a"],
  ligand_subbatch: Int[NDArray[np.integer], " a"],
  period_index: Int[NDArray[np.integer], " 118"],
  group_index: Int[NDArray[np.integer], " 118"],
  structure: GraphStructure,
  sequence_indices: Int[NDArray[np.integer], " l"],
  chi_angles: Float[NDArray[np.floating], "l 4"],
  chain_mask: Bool[NDArray[np.bool_], " l"],
  first_shell: Bool[NDArray[np.bool_], " l"],
  *,
  budget_mask: Bool[NDArray[np.bool_], " l"] | None,
  charged_mask: Bool[NDArray[np.bool_], " l"] | None,
  bias: Float[NDArray[np.floating], "l 21"] | None,
  disabled_residues: tuple[str, ...],
  repack_all: bool,
  repack_only: bool,
  protein_loop: bool,
) -> _ScanInputs:
  """Encode ``backbone`` once and pack the graph the residue scan reads."""
  encoded = encode_structure(
    encoder,
    backbone,
    ligand_coords,
    ligand_atomic_numbers,
    ligand_subbatch,
    period_index,
    group_index,
    structure,
    protein_loop=protein_loop,
  )
  n_res = int(encoded.prot_scalars.shape[0])
  mask = np.asarray(chain_mask, dtype=bool)
  # repack_only is a host rewrite of the batch, not a per-step branch.
  if repack_only:
    mask = np.ones((n_res,), dtype=bool)
  packed_pr = pack_edges(n_res, encoded.pr_pr_idx)
  packed_lp = pack_edges(n_res, encoded.lig_pr_idx)
  pr_edges = scatter_edge_values(n_res, packed_pr, encoded.pr_pr_eattr)
  lp_edges = scatter_edge_values(n_res, packed_lp, encoded.lig_pr_eattr)
  dtype = encoded.prot_scalars.dtype
  scalars = np.asarray(encoded.prot_scalars)
  vectors = np.asarray(encoded.prot_vectors)
  node_s = np.zeros((4, *scalars.shape), dtype=scalars.dtype)
  node_v = np.zeros((4, *vectors.shape), dtype=vectors.dtype)
  node_s[0] = scalars
  node_v[0] = vectors
  region = (
    np.zeros((n_res,), dtype=bool) if budget_mask is None else np.asarray(budget_mask, dtype=bool)
  )
  charged = (
    np.zeros((n_res,), dtype=bool) if charged_mask is None else np.asarray(charged_mask, dtype=bool)
  )
  logit_bias = np.zeros((n_res, 21), dtype=dtype) if bias is None else np.asarray(bias, dtype=dtype)
  return _ScanInputs(
    ligand_scalars=jnp.asarray(encoded.lig_scalars),
    pr_edges=jnp.asarray(pr_edges),
    pr_neighbours=jnp.asarray(packed_pr.neighbours),
    pr_mask=jnp.asarray(packed_pr.mask),
    lp_edges=jnp.asarray(lp_edges),
    lp_neighbours=jnp.asarray(packed_lp.neighbours),
    lp_mask=jnp.asarray(packed_lp.mask),
    input_sequence=jnp.asarray(sequence_indices, dtype=jnp.int32),
    input_chi=jnp.asarray(chi_angles, dtype=dtype),
    chain_mask=jnp.asarray(mask),
    first_shell=jnp.asarray(first_shell),
    budget_mask=jnp.asarray(region),
    charged_mask=jnp.asarray(charged),
    disabled=jnp.asarray(_disabled_mask(disabled_residues)),
    bias=jnp.asarray(logit_bias),
    node_scalars=jnp.asarray(node_s),
    node_vectors=jnp.asarray(node_v),
    n_res=n_res,
    effective_repack=repack_all or repack_only,
  )


def _pad_uniform_row(uniforms: NDArray[np.floating], n_res: int, *, sampling: bool) -> np.ndarray:
  """Match ``decode_order``'s uniform padding for one sample."""
  draws = np.asarray(uniforms, dtype=np.float64)
  if draws.ndim == 2:
    draws = draws[0]
  draws = np.reshape(draws, (-1,))
  if sampling:
    need = n_res * _DRAWS_PER_STEP
    if int(draws.shape[0]) < need:
      draws = np.pad(draws, (0, need - int(draws.shape[0])))
  elif int(draws.shape[0]) == 0:
    draws = np.zeros((1,), dtype=np.float64)
  return np.asarray(draws, dtype=np.float64)


def _is_temperature_scalar(value: object) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def _temperature_rows(
  sequence_temperature: float | None | Sequence[float | None] | NDArray[np.floating],
  batch: int,
) -> list[float | None]:
  """One temperature per sample. A scalar is broadcast. NaN in an array is ``None``."""
  if sequence_temperature is None or _is_temperature_scalar(sequence_temperature):
    value = None if sequence_temperature is None else float(sequence_temperature)
    return [value] * batch
  floating_vector = isinstance(sequence_temperature, np.ndarray) and np.issubdtype(
    sequence_temperature.dtype,
    np.floating,
  )
  if floating_vector:
    flat = np.asarray(sequence_temperature, dtype=np.float64).reshape(-1)
    rows: list[float | None] = [None if np.isnan(value) else float(value) for value in flat]
  else:
    rows = []
    for item in sequence_temperature:
      if item is None:
        rows.append(None)
      elif _is_temperature_scalar(item):
        rows.append(float(item))
      else:
        msg = "sequence_temperature entries must be floats or None"
        raise TypeError(msg)
  if len(rows) != batch:
    msg = f"sequence_temperature length {len(rows)} != batch {batch}"
    raise ValueError(msg)
  return rows


def _uniform_rows(
  uniforms: NDArray[np.floating] | Sequence[NDArray[np.floating]],
  batch: int,
) -> list[np.ndarray]:
  """Per-sample uniform streams. ``(batch, draws)`` or a sequence of rows."""
  if isinstance(uniforms, np.ndarray) and uniforms.ndim == 2 and int(uniforms.shape[0]) == batch:
    return [np.asarray(uniforms[index], dtype=np.float64) for index in range(batch)]
  if isinstance(uniforms, np.ndarray) and uniforms.ndim == 1 and batch == 1:
    return [np.asarray(uniforms, dtype=np.float64)]
  if isinstance(uniforms, (list, tuple)) and len(uniforms) == batch:
    return [np.asarray(row, dtype=np.float64) for row in uniforms]
  msg = f"uniforms must be ({batch}, draws) or a sequence of {batch} rows"
  raise ValueError(msg)


def _temperature_vector(rows: list[float | None], dtype: np.dtype) -> np.ndarray:
  """Model-dtype temperatures. NaN marks a ``None`` (argmax) row."""
  out = np.empty((len(rows),), dtype=dtype)
  for index, value in enumerate(rows):
    out[index] = np.nan if value is None else np.asarray(value, dtype=dtype)
  return out


@eqx.filter_jit
def _scan_residues_batched(
  joint: LaserJointDecode,
  decoder: LaserDecoder,
  ligand_scalars: Float[Array, "a h"],
  pr_edges: Float[Array, "l kp e"],
  pr_neighbours: Int[Array, "l kp"],
  pr_mask: Bool[Array, "l kp"],
  lp_edges: Float[Array, "l kl e"],
  lp_neighbours: Int[Array, "l kl"],
  lp_mask: Bool[Array, "l kl"],
  orders: Int[Array, "batch l"],
  input_sequence: Int[Array, " l"],
  input_chi: Float[Array, "l 4"],
  chain_mask: Bool[Array, " l"],
  first_shell: Bool[Array, " l"],
  budget_mask: Bool[Array, " l"],
  charged_mask: Bool[Array, " l"],
  disabled: Bool[Array, " 21"],
  bias: Float[Array, "l 21"],
  draws: Float[Array, "batch draws"],
  node_scalars: Float[Array, "4 l h"],
  node_vectors: Float[Array, "4 l v 3"],
  temperatures: Float[Array, " batch"],
  *,
  per_temperature: bool,
  sequence_temperature: float | None,
  chi_temperature: float | None,
  fs_sequence_temp: float | None,
  seq_min_p: float,
  chi_min_p: float,
  ala_budget: int,
  gly_budget: int,
  ignore_chain_mask_zeros: bool,
  repack_all: bool,
  max_rows: int | None,
) -> tuple[
  Int[Array, "batch l"],
  Float[Array, "batch l 21"],
  Float[Array, "batch l 4"],
  Float[Array, "batch l 4 72"],
  Float[Array, "batch 4 l h"],
]:
  """Map ``_scan_residues`` over samples, vectorized ``max_rows`` rows at a time.

  Encoder tensors are closed over, so every row shares one copy. ``max_rows=None``
  vectorizes the whole batch; otherwise ``lax.map`` bounds memory to ``max_rows``
  rows per step, whatever the batch size.
  """

  def one(
    order: Int[Array, " l"],
    row_draws: Float[Array, " draws"],
    temperature: float | None | Float[Array, ""],
  ) -> tuple[
    Int[Array, " l"],
    Float[Array, "l 21"],
    Float[Array, "l 4"],
    Float[Array, "l 4 72"],
    Float[Array, "4 l h"],
  ]:
    return _scan_residues(
      joint,
      decoder,
      ligand_scalars,
      pr_edges,
      pr_neighbours,
      pr_mask,
      lp_edges,
      lp_neighbours,
      lp_mask,
      order,
      input_sequence,
      input_chi,
      chain_mask,
      first_shell,
      budget_mask,
      charged_mask,
      disabled,
      bias,
      row_draws,
      node_scalars,
      node_vectors,
      sequence_temperature=temperature,
      chi_temperature=chi_temperature,
      fs_sequence_temp=fs_sequence_temp,
      seq_min_p=seq_min_p,
      chi_min_p=chi_min_p,
      ala_budget=ala_budget,
      gly_budget=gly_budget,
      ignore_chain_mask_zeros=ignore_chain_mask_zeros,
      repack_all=repack_all,
    )

  def row(
    xs: tuple[Int[Array, " l"], Float[Array, " draws"], Float[Array, ""]],
  ) -> tuple[Array, Array, Array, Array, Array]:
    order, row_draws, temperature = xs
    return one(order, row_draws, temperature if per_temperature else sequence_temperature)

  xs = (orders, draws, temperatures)
  batch = int(orders.shape[0])
  if max_rows is None or batch <= max_rows:
    return jax.vmap(row)(xs)
  return jax.lax.map(row, xs, batch_size=max_rows)


def decode_order_batched(
  encoder: LaserEncoder,
  decoder: LaserDecoder,
  joint: LaserJointDecode,
  backbone: Float[NDArray[np.floating], "l 5 3"],
  ligand_coords: Float[NDArray[np.floating], "a 3"],
  ligand_atomic_numbers: Int[NDArray[np.integer], " a"],
  ligand_subbatch: Int[NDArray[np.integer], " a"],
  period_index: Int[NDArray[np.integer], " 118"],
  group_index: Int[NDArray[np.integer], " 118"],
  structure: GraphStructure,
  sequence_indices: Int[NDArray[np.integer], " l"],
  chi_angles: Float[NDArray[np.floating], "l 4"],
  chain_mask: Bool[NDArray[np.bool_], " l"],
  first_shell: Bool[NDArray[np.bool_], " l"],
  decoding_order: Int[NDArray[np.integer], "batch l"],
  uniforms: Float[NDArray[np.floating], "batch draws"]
  | Sequence[Float[NDArray[np.floating], " draws"]],
  *,
  sequence_temperature: float | None | Sequence[float | None] | NDArray[np.floating] = None,
  chi_temperature: float | None = None,
  fs_sequence_temp: float | None = None,
  seq_min_p: float = 0.0,
  chi_min_p: float = 0.0,
  ala_budget: int = 4,
  gly_budget: int = 0,
  disabled_residues: tuple[str, ...] = ("X",),
  ignore_chain_mask_zeros: bool = False,
  repack_all: bool = False,
  repack_only: bool = False,
  budget_mask: Bool[NDArray[np.bool_], " l"] | None = None,
  charged_mask: Bool[NDArray[np.bool_], " l"] | None = None,
  bias: Float[NDArray[np.floating], "l 21"] | None = None,
  protein_loop: bool = True,
  max_rows: int | None = None,
) -> JointDecodeBatch:
  """Encode once, then map ``_scan_residues`` over the leading sample axis.

  ``decoding_order`` is ``(batch, L)`` and ``uniforms`` is ``(batch, draws)``
  or one row per sample. ``sequence_temperature`` is a scalar or one value per
  sample (``None`` is argmax). A scalar stays on the ``decode_order`` branch.
  Rows that disagree are passed as a model-dtype vector, with NaN for ``None``.
  ``max_rows`` caps how many rows one vectorized step holds (``None``: all of
  them), so peak memory does not grow with the batch.
  """
  if max_rows is not None and max_rows < 1:
    msg = f"max_rows must be positive or None, got {max_rows}"
    raise ValueError(msg)
  prepared = _encoded_scan_inputs(
    encoder,
    backbone,
    ligand_coords,
    ligand_atomic_numbers,
    ligand_subbatch,
    period_index,
    group_index,
    structure,
    sequence_indices,
    chi_angles,
    chain_mask,
    first_shell,
    budget_mask=budget_mask,
    charged_mask=charged_mask,
    bias=bias,
    disabled_residues=disabled_residues,
    repack_all=repack_all,
    repack_only=repack_only,
    protein_loop=protein_loop,
  )
  order = np.asarray(decoding_order, dtype=np.int32)
  if order.ndim != 2:
    msg = f"decoding_order must have shape (batch, length), got {order.shape}"
    raise ValueError(msg)
  batch = int(order.shape[0])
  if int(order.shape[1]) != prepared.n_res:
    msg = f"decoding_order length {order.shape[1]} != structure length {prepared.n_res}"
    raise ValueError(msg)
  rows = _temperature_rows(sequence_temperature, batch)
  per_temperature = not all(value == rows[0] for value in rows)
  scalar_temperature = rows[0]
  padded = [
    _pad_uniform_row(
      row,
      prepared.n_res,
      sampling=_sampling(temperature, chi_temperature, fs_sequence_temp),
    )
    for row, temperature in zip(_uniform_rows(uniforms, batch), rows, strict=True)
  ]
  width = max(int(row.shape[0]) for row in padded)
  stacked = np.zeros((batch, width), dtype=np.float64)
  for index, row in enumerate(padded):
    stacked[index, : int(row.shape[0])] = row
  scalar_dtype = np.dtype(prepared.node_scalars.dtype)
  temperature_vector = (
    _temperature_vector(rows, scalar_dtype)
    if per_temperature
    else np.zeros((batch,), dtype=scalar_dtype)
  )
  sequence, seq_logits, chi_degrees, chi_logits, nodes = _scan_residues_batched(
    joint,
    decoder,
    prepared.ligand_scalars,
    prepared.pr_edges,
    prepared.pr_neighbours,
    prepared.pr_mask,
    prepared.lp_edges,
    prepared.lp_neighbours,
    prepared.lp_mask,
    jnp.asarray(order),
    prepared.input_sequence,
    prepared.input_chi,
    prepared.chain_mask,
    prepared.first_shell,
    prepared.budget_mask,
    prepared.charged_mask,
    prepared.disabled,
    prepared.bias,
    jnp.asarray(stacked),
    prepared.node_scalars,
    prepared.node_vectors,
    jnp.asarray(temperature_vector),
    per_temperature=per_temperature,
    sequence_temperature=None if per_temperature else scalar_temperature,
    chi_temperature=chi_temperature,
    fs_sequence_temp=fs_sequence_temp,
    seq_min_p=seq_min_p,
    chi_min_p=chi_min_p,
    ala_budget=ala_budget,
    gly_budget=gly_budget,
    ignore_chain_mask_zeros=ignore_chain_mask_zeros,
    repack_all=prepared.effective_repack,
    max_rows=max_rows,
  )
  chi_logits_np = np.asarray(chi_logits)
  return JointDecodeBatch(
    sequence=np.asarray(sequence, dtype=np.int64),
    sequence_logits=np.asarray(seq_logits),
    chi_degrees=np.asarray(chi_degrees),
    chi_logits=chi_logits_np,
    chi_bins=np.argmax(chi_logits_np, axis=-1).astype(np.int64),
    node_scalars=np.asarray(nodes),
  )
