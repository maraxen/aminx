"""PottsMPNN edge graph on top of stock ``ProteinFeatures`` weights.

Stock ``aminx.model.features.ProteinFeatures`` (the MPNN runner path) differs
from upstream Potts ``ProteinFeatures`` (``potts_mpnn_utils.py`` 1143-1220):

- kNN. Stock writes ``+inf`` on ``mask_i * mask_j == 0`` pairs, so every
  invalid neighbour ranks after every finite pair. Upstream uses
  ``D = mask_2D * sqrt(|dCa|^2 + 1e-6)``, ``D_max = rowmax(D)``,
  ``D_adjust = D + (1 - mask_2D) * D_max`` and takes the ``k`` smallest
  ``D_adjust``. Invalid pairs sit at ``D_max`` (tied with the farthest valid
  pair), not at infinity.
- Ca-Ca RBF. Stock uses the raw gathered Ca distance. Upstream uses
  ``D_neighbors`` (gathered ``D_adjust``), so a present->non-present edge is
  the RBF of ``D_max_i``.
- Other 24 atom-pair RBFs. Both use ``sqrt(|A_i - B_j|^2 + 1e-6)`` on raw
  coordinates, with no mask. A zeroed gap row has N/CA/C/O at the origin and
  Cb from that frame (also the origin), so N-N into the gap is the distance
  from ``N_i`` to the origin.
- Tie-break. ``top_k`` orders equal distances lower-index-first (§4.1a).
  Positional offsets and chain-equality embeddings are gathered for every
  selected edge, then ``edge_embedding`` (no bias), layer norm, and ``W_e``.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from aminx.model.features import ProteinFeatures, top_k
from aminx.utils.coordinates import compute_backbone_coordinates, compute_backbone_distance
from aminx.utils.graph import compute_neighbor_offsets
from aminx.utils.radial_basis import RADIAL_BASES, RBF_SIGMA, compute_radial_basis, rbf_centers


class PottsEdgeGraph(NamedTuple):
  """Embedded edges, neighbour indices, and the pre-linear RBF block."""

  edge_features: Float[Array, "L K H"]
  neighbor_indices: Int[Array, "L K"]
  rbf: Float[Array, "L K 400"]


def _rbf(distance: Float[Array, "L K"]) -> Float[Array, "L K 16"]:
  """Upstream ``_rbf``: 16 Gaussians on ``[2, 22]``."""
  scaled = (distance[..., None] - rbf_centers(distance.dtype)) / RBF_SIGMA
  return jnp.exp(-jnp.square(scaled))


def _linear(layer: eqx.nn.Linear, x: Float[Array, "... I"]) -> Float[Array, "... O"]:
  """``eqx.nn.Linear`` applied over the trailing axis of ``x``."""
  out = x @ layer.weight.T
  if layer.bias is not None:
    out = out + layer.bias
  return out


def _layer_norm(norm: eqx.nn.LayerNorm, x: Float[Array, "... H"]) -> Float[Array, "... H"]:
  """``eqx.nn.LayerNorm`` over the trailing axis, same ops as its ``__call__``."""
  mean = jnp.mean(x, axis=-1, keepdims=True)
  variance = jnp.var(x, axis=-1, keepdims=True)
  out = (x - mean) * jax.lax.rsqrt(variance + norm.eps)
  if norm.weight is not None:
    out = out * norm.weight
  if norm.bias is not None:
    out = out + norm.bias
  return out


def potts_edge_features(
  features: ProteinFeatures,
  coords: Float[Array, "L 4 3"],
  mask: Float[Array, " L"],
  residue_index: Int[Array, " L"],
  chain_index: Int[Array, " L"],
) -> PottsEdgeGraph:
  """Build the Potts edge graph, reusing ``features`` projections.

  Ca-Ca distances follow ``_dist``: non-present pairs contribute ``D_max`` to
  kNN and to the Ca-Ca RBF. The other 24 pair RBFs use raw coordinates.
  """
  backbone = compute_backbone_coordinates(coords)
  distances = compute_backbone_distance(backbone)
  mask_2d = mask[:, None] * mask[None, :]
  masked = mask_2d * distances
  distance_max = jnp.max(masked, axis=-1, keepdims=True)
  adjusted = masked + (1.0 - mask_2d) * distance_max

  k = min(features.k_neighbors, coords.shape[0])
  _values, neighbor_indices = top_k(-adjusted, k)
  neighbor_indices = jnp.asarray(neighbor_indices, dtype=jnp.int32)

  rbf = compute_radial_basis(backbone, neighbor_indices)
  ca_neighbors = jnp.take_along_axis(adjusted, neighbor_indices, axis=1)
  rbf = rbf.at[..., :RADIAL_BASES].set(_rbf(ca_neighbors).astype(rbf.dtype))

  neighbor_offsets = compute_neighbor_offsets(residue_index, neighbor_indices)
  edge_chains = (chain_index[:, None] == chain_index[None, :]).astype(jnp.int32)
  edge_chains_neighbors = jnp.take_along_axis(edge_chains, neighbor_indices, axis=1)
  max_relative = (features.w_pos.weight.shape[1] - 2) // 2
  neighbor_offset_factor = jnp.minimum(
    jnp.maximum(neighbor_offsets + max_relative, 0),
    2 * max_relative,
  )
  edge_chain_factor = (1 - edge_chains_neighbors) * (2 * max_relative + 1)
  encoded_offset = neighbor_offset_factor * edge_chains_neighbors + edge_chain_factor
  encoded_offset_one_hot = jax.nn.one_hot(encoded_offset, 2 * max_relative + 2)
  # Broadcast over (L, K) instead of vmap: L-DRV R1 bans vmap in families/.
  encoded_positions = _linear(features.w_pos, encoded_offset_one_hot.astype(rbf.dtype))
  edges_concat = jnp.concatenate([encoded_positions, rbf], axis=-1)
  embedded = _linear(features.w_e, edges_concat)
  normalized = _layer_norm(features.norm_edges, embedded)
  projected = _linear(features.w_e_proj, normalized)
  return PottsEdgeGraph(
    edge_features=projected,
    neighbor_indices=neighbor_indices,
    rbf=rbf,
  )
