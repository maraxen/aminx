"""Teacher-forced LASEr scoring.

Ports ``LASErMPNN.get_logits_for_score`` (no ``decoding_order_generator``).
The encoder runs first; its protein scalars are banked before the decoder
loop because masked edges keep that snapshot while unmasked edges read the
scalars the decoder just wrote.

Math:
    An edge is unmasked when the source residue's argsort rank is strictly
    less than the sink's. Unmasked edges carry the ground-truth sequence
    embedding, the chi RBF, and the current source scalars. Masked edges
    carry the mask-class embedding, zeros in place of chi, and the encoder
    scalars. Both branches have the same width, so the dense neighbour axis
    selects them with one ``where``.

Pseudocode:
    lig, prot, edges = encoder(structure)
    bank encoder scalars and protein edges
    chi_rbf = nan_to_num(binned_degree_basis(chi))
    for layer in decoder_layers:
        features = where(unmasked, teacher_features, mask_features)
        prot = layer(masked_first(features), prot, lig, edges)
    sequence_logits = linear(prot.scalars)
    order_log_probs = ones(decoding_order)
    chi_logits = nan, written only where chi is finite
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from numpy.typing import NDArray

from aminx.model.laser.encoders import LaserEncoder, encode_structure
from aminx.model.laser.graphs import (
  GraphStructure,
  pack_edges,
  scatter_edge_values,
)
from aminx.model.laser.layers import (
  GVP,
  DenseMLP,
  EquivariantLayerNorm,
  HeteroGATv2,
  apply_linear,
)

_NODE = 256
_EDGE = 128
_V_PROT = 10
_HEADS = 1
_UPSCALE = 4
_DROPOUT = 0.1
_N_DECODER = 3
_CHI_BINS = 72
_CHI_ANGLES = 4
_CHI_FLAT = _CHI_ANGLES * _CHI_BINS
_BIN_WIDTH = 5.0
_CHI_STD = _BIN_WIDTH / 2.0
_MASK_CLASS = 21
_VOCAB = 22
_SEQ_OUT = 21
# (sequence or mask embedding) + chi RBF + protein edge + source scalars.
_EXPANDED = (2 * _NODE) + _EDGE + _CHI_FLAT


def binned_degree_basis(degrees: Float[Array, "l 4"]) -> Float[Array, "l 4 72"]:
  """Circular Gaussian RBF on ``[-180, 180)`` used by ``RotamerBuilder``.

  NaN angles stay NaN. Callers that build edge features must ``nan_to_num``
  so a missing chi becomes zeros rather than poisoning the edge.
  """
  flat = jnp.reshape(degrees, (-1, 1))
  centers = jnp.arange(-180.0, 180.0, _BIN_WIDTH, dtype=degrees.dtype)
  positive = jnp.remainder(centers[None, :] - flat, 360.0)
  negative = jnp.remainder(flat - centers[None, :], 360.0)
  distance = jnp.minimum(positive, negative)
  density = jnp.exp(-0.5 * jnp.square(distance / _CHI_STD))
  totals = jnp.sum(density, axis=1, keepdims=True)
  encoded = density / totals
  return jnp.reshape(encoded, (*degrees.shape, _CHI_BINS))


def _masked_first_order(
  unmasked: Bool[Array, "l k"],
  mask: Bool[Array, "l k"],
) -> Int[Array, "l k"]:
  """Permutation that lists masked real edges before unmasked ones.

  Upstream concatenates the masked edge list in front of the unmasked list
  before the joint softmax. The reduction sum follows that order, so the
  dense slots have to as well. Padding is pushed to the end, where the mask
  already drops it. The slot index breaks ties, so the argsort need not be stable.
  """
  width = unmasked.shape[1]
  slots = jnp.arange(width)
  group = jnp.where(mask, jnp.where(unmasked, 1, 0), 2)
  return jnp.argsort(group * (width + 1) + slots, axis=1)


class _DecoderLayer(eqx.Module):
  """One ``LASErMPNN_Decoder``: hetero GAT, no edge update.

  Protein sources are the expanded teacher-forced edge features (width 928)
  with an empty edge channel. Ligand sources stay the ligand scalars.
  """

  hetgat: HeteroGATv2

  def __init__(self, *, key: PRNGKeyArray) -> None:
    self.hetgat = HeteroGATv2(
      2,
      ((_EXPANDED, _NODE), (_NODE, _NODE)),
      (0, _EDGE),
      0,
      _HEADS,
      _DROPOUT,
      num_vectors=_V_PROT,
      use_mlp_node_update=True,
      use_residual_node_update=True,
      compute_edge_updates=False,
      output_node_embedding_dim=_NODE,
      atten_dimension_upscale_factor=_UPSCALE,
      key=key,
    )


class LaserDecoder(eqx.Module):
  """Teacher-forced decoder and chi head. Field names match the checkpoint."""

  protein_decoder_layers: tuple[_DecoderLayer, _DecoderLayer, _DecoderLayer]
  sequence_label_embedding: eqx.nn.Embedding
  sequence_output_layer: eqx.nn.Linear
  chi_prediction_layers: tuple[DenseMLP, DenseMLP, DenseMLP, DenseMLP]
  chi_vector_update_layers: tuple[GVP, GVP, GVP]
  chi_vector_layer_norms: tuple[
    EquivariantLayerNorm,
    EquivariantLayerNorm,
    EquivariantLayerNorm,
  ]

  def __init__(self, *, key: PRNGKeyArray) -> None:
    keys = jax.random.split(key, 10)
    built = tuple(_DecoderLayer(key=sub) for sub in jax.random.split(keys[0], _N_DECODER))
    self.protein_decoder_layers = (built[0], built[1], built[2])
    self.sequence_label_embedding = eqx.nn.Embedding(_VOCAB, _NODE, key=keys[1])
    self.sequence_output_layer = eqx.nn.Linear(_NODE, _SEQ_OUT, key=keys[2])
    # Input width is 2*256 + i*72: scalars, sequence embedding, then chi so far.
    self.chi_prediction_layers = (
      DenseMLP(2 * _NODE, _NODE, _CHI_BINS, key=keys[3]),
      DenseMLP(2 * _NODE + _CHI_BINS, _NODE, _CHI_BINS, key=keys[4]),
      DenseMLP(2 * _NODE + 2 * _CHI_BINS, _NODE, _CHI_BINS, key=keys[5]),
      DenseMLP(2 * _NODE + 3 * _CHI_BINS, _NODE, _CHI_BINS, key=keys[6]),
    )
    # Three updates: the last chi logit has no following angle to condition.
    self.chi_vector_update_layers = (
      GVP((_NODE + _CHI_BINS, _V_PROT), (_NODE, _V_PROT), vector_gate=True, key=keys[7]),
      GVP((_NODE + 2 * _CHI_BINS, _V_PROT), (_NODE, _V_PROT), vector_gate=True, key=keys[8]),
      GVP((_NODE + 3 * _CHI_BINS, _V_PROT), (_NODE, _V_PROT), vector_gate=True, key=keys[9]),
    )
    self.chi_vector_layer_norms = (
      EquivariantLayerNorm((_NODE, _V_PROT), vector_only=True),
      EquivariantLayerNorm((_NODE, _V_PROT), vector_only=True),
      EquivariantLayerNorm((_NODE, _V_PROT), vector_only=True),
    )

  def _teacher_edges(
    self,
    prot_s: Float[Array, "l h"],
    encoder_s: Float[Array, "l h"],
    pr_edges: Float[Array, "l k e"],
    pr_neighbours: Int[Array, "l k"],
    pr_mask: Bool[Array, "l k"],
    sequence_indices: Int[Array, " l"],
    chi_flat: Float[Array, "l chi"],
    decoding_order: Int[Array, " l"],
  ) -> tuple[Float[Array, "l k feat"], Bool[Array, "l k"]]:
    """Dense edge features. One ``where`` selects the unmasked branch.

    Masked edges read ``encoder_s``, the protein scalars captured once before
    the decoder loop. Unmasked edges read ``prot_s``, which each decoder layer
    replaces. Upstream builds the masked node features once and recomputes the
    unmasked node features on every iteration; using the current scalars for
    both branches scores a different model.
    """
    sort_idx = jnp.argsort(decoding_order)
    source = pr_neighbours
    n_res = prot_s.shape[0]
    sink = jnp.broadcast_to(jnp.arange(n_res)[:, None], source.shape)
    # Strict: a self-edge compares equal and stays on the mask-token branch.
    unmasked = sort_idx[source] < sort_idx[sink]
    table = self.sequence_label_embedding.weight
    sequence = table[sequence_indices]
    mask_ids = jnp.full(sequence_indices.shape, _MASK_CLASS)
    masked_sequence = table[mask_ids]
    seq_u = sequence[source]
    seq_m = masked_sequence[source]
    chi_u = chi_flat[source]
    chi_m = jnp.zeros_like(chi_u)
    node_u = prot_s[source]
    node_m = encoder_s[source]
    unmasked_feat = jnp.concatenate((seq_u, chi_u, pr_edges, node_u), axis=-1)
    masked_feat = jnp.concatenate((seq_m, chi_m, pr_edges, node_m), axis=-1)
    features = jnp.where(unmasked[..., None], unmasked_feat, masked_feat)
    order = _masked_first_order(unmasked, pr_mask)
    index = jnp.broadcast_to(order[:, :, None], features.shape)
    features = jnp.take_along_axis(features, index, axis=1)
    mask = jnp.take_along_axis(pr_mask, order, axis=1)
    return features, mask

  def __call__(
    self,
    prot_s: Float[Array, "l h"],
    prot_v: Float[Array, "l v 3"],
    lig_s: Float[Array, "a h"],
    pr_edges: Float[Array, "l kp e"],
    pr_neighbours: Int[Array, "l kp"],
    pr_mask: Bool[Array, "l kp"],
    lp_edges: Float[Array, "l kl e"],
    lp_neighbours: Int[Array, "l kl"],
    lp_mask: Bool[Array, "l kl"],
    decoding_order: Int[Array, " l"],
    sequence_indices: Int[Array, " l"],
    chi_angles: Float[Array, "l 4"],
  ) -> tuple[
    Float[Array, "l 21"],
    Float[Array, "l 4 72"],
    Float[Array, "l h"],
    Float[Array, " l"],
  ]:
    """Sequence logits, chi logits, banked encoder scalars, and order log-probs.

    Order log-probs are identically one. The scoring dump passes no decoding
    order generator, and that branch is ``ones_like(decoding_order)``.
    Chi logits start as NaN and are written only where ``chi_angles`` is finite.
    """
    # Banked before any decoder write. Masked edges keep this snapshot.
    encoder_s = prot_s
    chi_angle_encoding = jnp.nan_to_num(binned_degree_basis(chi_angles))
    n_res = prot_s.shape[0]
    chi_flat = jnp.reshape(chi_angle_encoding, (n_res, _CHI_FLAT))
    for layer in self.protein_decoder_layers:
      expanded, row_mask = self._teacher_edges(
        prot_s,
        encoder_s,
        pr_edges,
        pr_neighbours,
        pr_mask,
        sequence_indices,
        chi_flat,
        decoding_order,
      )
      empty = jnp.zeros((*pr_neighbours.shape, 0), dtype=prot_s.dtype)
      prot_s, prot_v, _edges = layer.hetgat(
        prot_s,
        prot_v,
        (expanded, lig_s),
        (pr_neighbours, lp_neighbours),
        (empty, lp_edges),
        (row_mask, lp_mask),
        (False, False),
      )
    sequence_logits = apply_linear(self.sequence_output_layer, prot_s)
    # No decoding-order generator was passed to the dump.
    order_log_probs = jnp.ones(decoding_order.shape, dtype=prot_s.dtype)
    sequence_nodes = self.sequence_label_embedding.weight[sequence_indices]
    output_chi = jnp.full((n_res, _CHI_ANGLES, _CHI_BINS), jnp.nan, dtype=prot_s.dtype)
    prev_chi = jnp.zeros((n_res, 0), dtype=prot_s.dtype)
    prot_scalars = prot_s
    for index, chi_layer in enumerate(self.chi_prediction_layers):
      logits = chi_layer(jnp.concatenate((prot_scalars, sequence_nodes, prev_chi), axis=-1))
      valid = ~jnp.isnan(chi_angles[:, index])
      output_chi = output_chi.at[:, index, :].set(
        jnp.where(valid[:, None], logits, output_chi[:, index, :]),
      )
      width = index + 1
      prev_chi = jnp.reshape(chi_angle_encoding[:, :width, :], (n_res, width * _CHI_BINS))
      if index == 3:
        continue
      updated_s, prot_v = self.chi_vector_layer_norms[index](
        *self.chi_vector_update_layers[index](
          jnp.concatenate((prot_scalars, prev_chi), axis=-1),
          prot_v,
        ),
      )
      prot_scalars = updated_s
    return sequence_logits, output_chi, encoder_s, order_log_probs


@dataclass(frozen=True, slots=True)
class ScoreForward:
  """``get_logits_for_score`` returns, plus the edge index the encoder built."""

  sequence_logits: NDArray[np.floating]
  chi_logits: NDArray[np.floating]
  encoder_scalars: NDArray[np.floating]
  encoder_edges: NDArray[np.floating]
  ligand_scalars: NDArray[np.floating]
  lig_prot_edges: NDArray[np.floating]
  decoding_order_log_probs: NDArray[np.floating]
  pr_pr_idx: NDArray[np.int64]
  lig_pr_idx: NDArray[np.int64]


@eqx.filter_jit
def _score_jit(
  model: LaserDecoder,
  prot_s: Float[Array, "l h"],
  prot_v: Float[Array, "l v 3"],
  lig_s: Float[Array, "a h"],
  pr_edges: Float[Array, "l kp e"],
  pr_neighbours: Int[Array, "l kp"],
  pr_mask: Bool[Array, "l kp"],
  lp_edges: Float[Array, "l kl e"],
  lp_neighbours: Int[Array, "l kl"],
  lp_mask: Bool[Array, "l kl"],
  decoding_order: Int[Array, " l"],
  sequence_indices: Int[Array, " l"],
  chi_angles: Float[Array, "l 4"],
) -> tuple[
  Float[Array, "l 21"],
  Float[Array, "l 4 72"],
  Float[Array, "l h"],
  Float[Array, " l"],
]:
  return model(
    prot_s,
    prot_v,
    lig_s,
    pr_edges,
    pr_neighbours,
    pr_mask,
    lp_edges,
    lp_neighbours,
    lp_mask,
    decoding_order,
    sequence_indices,
    chi_angles,
  )


def score_structure(
  encoder: LaserEncoder,
  decoder: LaserDecoder,
  backbone: Float[NDArray[np.floating], "l 5 3"],
  ligand_coords: Float[NDArray[np.floating], "a 3"],
  ligand_atomic_numbers: Int[NDArray[np.integer], " a"],
  ligand_subbatch: Int[NDArray[np.integer], " a"],
  period_index: Int[NDArray[np.integer], " 118"],
  group_index: Int[NDArray[np.integer], " 118"],
  structure: GraphStructure,
  decoding_order: Int[NDArray[np.integer], " l"],
  sequence_indices: Int[NDArray[np.integer], " l"],
  chi_angles: Float[NDArray[np.floating], "l 4"],
  *,
  protein_loop: bool = True,
) -> ScoreForward:
  """Encode, then teacher-force ``decoding_order``. The order is an input.

  Re-deriving it from the uniform stream would mix an order bug into the
  score comparison. ``decoding_order`` is the dump's own permutation.
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
  sequence_logits, chi_logits, encoder_s, order_lp = _score_jit(
    decoder,
    jnp.asarray(encoded.prot_scalars),
    jnp.asarray(encoded.prot_vectors),
    jnp.asarray(encoded.lig_scalars),
    jnp.asarray(pr_edges),
    jnp.asarray(packed_pr.neighbours),
    jnp.asarray(packed_pr.mask),
    jnp.asarray(lp_edges),
    jnp.asarray(packed_lp.neighbours),
    jnp.asarray(packed_lp.mask),
    jnp.asarray(decoding_order),
    jnp.asarray(sequence_indices),
    jnp.asarray(chi_angles, dtype=dtype),
  )
  return ScoreForward(
    sequence_logits=np.asarray(sequence_logits),
    chi_logits=np.asarray(chi_logits),
    encoder_scalars=np.asarray(encoder_s),
    encoder_edges=np.asarray(encoded.pr_pr_eattr),
    ligand_scalars=np.asarray(encoded.lig_scalars),
    lig_prot_edges=np.asarray(encoded.lig_pr_eattr),
    decoding_order_log_probs=np.asarray(order_lp),
    pr_pr_idx=encoded.pr_pr_idx,
    lig_pr_idx=encoded.lig_pr_idx,
  )
