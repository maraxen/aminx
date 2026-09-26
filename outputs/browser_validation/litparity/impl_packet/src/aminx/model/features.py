"""Feature extraction module for Aminx.

This module contains the ProteinFeatures class that extracts and projects
features from raw protein coordinates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from proxide.physics.constants import BOLTZMANN_KCAL

from aminx.utils.coordinates import (
  apply_noise_to_coordinates,
  compute_backbone_coordinates,
  compute_backbone_distance,
)
from aminx.utils.graph import compute_neighbor_offsets
from aminx.utils.radial_basis import compute_radial_basis

if TYPE_CHECKING:
  from aminx.types.arrays import (
    AlphaCarbonMask,
    BackboneNoise,
    ChainIndex,
    EdgeFeatures,
    NeighborIndices,
    NodeFeatures,
    ResidueIndex,
    StructureAtomicCoordinates,
  )


PRNGKeyArray = jax.Array

LayerNorm = eqx.nn.LayerNorm

# Feature extraction constants
MAXIMUM_RELATIVE_FEATURES = 32
POS_EMBED_DIM = 16


def top_k(x: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
  """Select the ``k`` largest entries along the last axis, descending.

  Deliberately NOT ``jax.lax.top_k``: JAX lowers that to a ``stablehlo.composite``
  wrapping ``chlo.top_k``, and IREE's StableHLO importer marks that composite
  explicitly illegal, so an otherwise-clean export dies at ``iree-compile`` with
  ``failed to legalize operation 'stablehlo.composite'``. Measured 260911 against
  ``iree-base-compiler 3.11.0rc20260316`` on every ``input_type`` IREE offers
  (``stablehlo``, ``stablehlo_xla``, ``auto``) -- a gap in the importer, not a
  flag we are missing.

  Every kNN selection under ``aminx.model`` routes here, which is what makes the
  model compilable; ``tests/export/test_top_k_export.py`` enforces that rather
  than asserting it, because it was briefly untrue. When this function was first
  introduced it was described as the package's only selection site while three
  more ``jax.lax.top_k`` calls were live in ``ligand_features`` and ``packer``,
  so the ligand and sidechain paths stayed uncompilable while the export tests
  reported green.

  Tie-breaking is stated here rather than inherited from the backend.
  ``jax.lax.top_k`` breaks ties toward the lower index, and the obvious
  replacement -- ``jnp.argsort(-x, stable=True)`` -- reproduces that only for as
  long as whoever runs the artifact honours stable-sort tie order. IREE does not.
  Measured 260911: an integer ``lax.sort_key_val`` over 64 slots holding 4
  distinct keys disagreed with XLA at 45 of its 64 positions. Nothing in the
  export pipeline catches that, because the divergence is carried entirely by the
  integer indices while the gathered values stay bit-identical, so a float parity
  check reports ``max_abs_diff 0.0`` and passes.

  So the index is folded into the sort key: sorting lexicographically on
  ``(-x, index)`` with ``num_keys=2`` is a strict total order -- no two entries
  ever compare equal, because no two indices are equal -- and every correct sort
  must then return the same permutation whether or not it is stable. That removes
  the dependency rather than restating it, which is why ``is_stable`` is passed
  ``False`` here: it is genuinely irrelevant, and asking for it would imply
  otherwise.

  Returns:
    ``(values, indices)``, matching ``jax.lax.top_k``'s contract on finite input.
    Non-finite input keeps the ordering ``jnp.argsort(-x)`` gave: NaNs sort last
    rather than first, and signed zeros compare equal. That differs from
    ``jax.lax.top_k`` and is unchanged from the previous implementation.
  """
  index = jax.lax.broadcasted_iota(jnp.int32, x.shape, x.ndim - 1)
  _, order = jax.lax.sort((-x, index), dimension=-1, is_stable=False, num_keys=2)
  order = order[..., :k]
  return jnp.take_along_axis(x, order, axis=-1), order


class ProteinEdgeStageTensors(NamedTuple):
  """Intermediate edge tensors for parity diagnosis (JAX protein feature path)."""

  neighbor_indices: jax.Array
  rbf: jax.Array
  encoded_positions: jax.Array
  edges_concat: jax.Array
  after_w_e: jax.Array
  after_norm: jax.Array
  final: jax.Array
  node_features_out: jax.Array | None
  prng_key: jax.Array


class ProteinFeatures(eqx.Module):
  """Extracts and projects features from raw protein coordinates.

  This module encapsulates k-NN, RBF, positional encodings, and edge projections.
  Note: W_e projection is NOT here - it's in the main model (matches ColabDesign).
  """

  w_pos: eqx.nn.Linear
  w_e: eqx.nn.Linear
  norm_edges: LayerNorm
  w_e_proj: eqx.nn.Linear  # Final edge projection
  k_neighbors: int = eqx.field(static=True)
  rbf_dim: int = eqx.field(static=True)
  pos_embed_dim: int = eqx.field(static=True)
  num_positional_embeddings: int = eqx.field(static=True)

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    k_neighbors: int,
    num_positional_embeddings: int = 16,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize feature extraction layers."""
    del node_features  # Unused, kept for API compatibility

    keys = jax.random.split(key, 3)

    self.k_neighbors = k_neighbors
    self.rbf_dim = 16
    self.pos_embed_dim = 16  # Fixed output dim
    self.num_positional_embeddings = num_positional_embeddings

    pos_one_hot_dim = 2 * num_positional_embeddings + 2
    edge_embed_in_dim = 16 + 16 * 25  # Matches POS_EMBED_DIM + RBF_DIM * 25

    # Positional encodings in reference models typically don't use bias
    self.w_pos = eqx.nn.Linear(pos_one_hot_dim, 16, use_bias=True, key=keys[0])
    self.w_e = eqx.nn.Linear(edge_embed_in_dim, edge_features, use_bias=False, key=keys[1])
    self.norm_edges = LayerNorm(edge_features)
    self.w_e_proj = eqx.nn.Linear(edge_features, edge_features, key=keys[2])

  def forward_edge_stages(
    self,
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    backbone_noise: BackboneNoise | None,
    backbone_noise_mode: str = "direct",
    structure_mapping: jnp.ndarray | None = None,
    initial_node_features: jnp.ndarray | None = None,
    rbf_features: jnp.ndarray | None = None,
    neighbor_indices: jnp.ndarray | None = None,
  ) -> ProteinEdgeStageTensors:
    """Run the edge pipeline and return intermediates for JAX vs PyTorch diagnosis.

    Stages mirror the reference ordering: concat embeddings → ``w_e`` → layer norm
    → ``w_e_proj`` (final edge features returned from ``__call__``).
    """
    node_features = None if initial_node_features is None else initial_node_features

    if backbone_noise is None:
      backbone_noise = jnp.array(0.0)

    # Use precomputed RBF if available (skips noise augmentation on graph)
    if rbf_features is not None:
      if neighbor_indices is None:
        # We still need neighbor indices to align sequence features
        # We compute them on the static coordinates (assuming they match proxide's implicit indices)
        backbone_atom_coordinates = compute_backbone_coordinates(structure_coordinates)
        distances = compute_backbone_distance(backbone_atom_coordinates)
      else:
        # If neighbors provided, we just need coordinates for other ops, no distance calc needed
        backbone_atom_coordinates = compute_backbone_coordinates(structure_coordinates)
        # We set distances to None or dummy as we won't use them for K-NN
        distances = None

      # Logic for masking structure_mapping/mask is shared
    else:
      # Resolve Sigma and apply noise
      if backbone_noise_mode == "thermal":
        # Apply 0.5 factor here as well for consistency
        thermal_energy = jnp.maximum(0.5 * BOLTZMANN_KCAL * backbone_noise, 0.0)
        final_sigma = jnp.sqrt(thermal_energy)
      else:
        final_sigma = backbone_noise

      noised_coordinates, prng_key = apply_noise_to_coordinates(
        prng_key,
        structure_coordinates,
        backbone_noise=final_sigma,
      )
      backbone_atom_coordinates = compute_backbone_coordinates(noised_coordinates)
      distances = compute_backbone_distance(backbone_atom_coordinates)

    if distances is not None:
      distances_masked = jnp.array(
        jnp.where(
          (mask[:, None] * mask[None, :]).astype(jnp.bool_),
          distances,
          jnp.inf,
        ),
      )

      if structure_mapping is not None:
        same_structure = structure_mapping[:, jnp.newaxis] == structure_mapping[jnp.newaxis, :]
        distances_masked = jnp.array(
          jnp.where(
            same_structure.astype(jnp.bool_),
            distances_masked,
            jnp.inf,
          ),
        ).squeeze()

      k = min(self.k_neighbors, structure_coordinates.shape[0])
      _, neighbor_indices = top_k(-distances_masked, k)
      neighbor_indices = jnp.array(neighbor_indices, dtype=jnp.int32)

    # At this point neighbor_indices must be populated (either passed in or computed)
    if neighbor_indices is None:
      raise ValueError(
        "neighbor_indices is None. This could happen if distances is None and no neighbor_indices were provided.",
      )
    neighbor_indices = jnp.array(neighbor_indices, dtype=jnp.int32)

    if rbf_features is not None:
      rbf = rbf_features
    else:
      rbf = compute_radial_basis(backbone_atom_coordinates, neighbor_indices)
    neighbor_offsets = compute_neighbor_offsets(residue_index, neighbor_indices)

    edge_chains = (chain_index[:, None] == chain_index[None, :]).astype(jnp.int32)
    if edge_chains is None:
      # This should be impossible if chain_index is not None
      pass
    edge_chains_neighbors = jnp.take_along_axis(
      edge_chains,
      neighbor_indices,
      axis=1,
    )

    MAX_REL = (self.w_pos.weight.shape[1] - 2) // 2
    neighbor_offset_factor = jnp.minimum(
      jnp.maximum(neighbor_offsets + MAX_REL, 0),
      2 * MAX_REL,
    )
    edge_chain_factor = (1 - edge_chains_neighbors) * (2 * MAX_REL + 1)
    encoded_offset = neighbor_offset_factor * edge_chains_neighbors + edge_chain_factor
    encoded_offset_one_hot = jax.nn.one_hot(
      encoded_offset,
      2 * MAX_REL + 2,
    )

    encoded_positions = jax.vmap(jax.vmap(self.w_pos))(encoded_offset_one_hot)

    edges_concat = jnp.concatenate([encoded_positions, rbf], axis=-1)

    after_w_e = jax.vmap(jax.vmap(self.w_e))(edges_concat)

    after_norm = jax.vmap(jax.vmap(self.norm_edges))(after_w_e)

    final = jax.vmap(jax.vmap(self.w_e_proj))(after_norm)

    return ProteinEdgeStageTensors(
      neighbor_indices=neighbor_indices,
      rbf=rbf,
      encoded_positions=encoded_positions,
      edges_concat=edges_concat,
      after_w_e=after_w_e,
      after_norm=after_norm,
      final=final,
      node_features_out=node_features,
      prng_key=prng_key,
    )

  def __call__(
    self,
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    backbone_noise: BackboneNoise | None,
    backbone_noise_mode: str = "direct",
    structure_mapping: jnp.ndarray | None = None,
    initial_node_features: jnp.ndarray | None = None,
    rbf_features: jnp.ndarray | None = None,
    neighbor_indices: jnp.ndarray | None = None,
  ) -> tuple[EdgeFeatures, NeighborIndices, NodeFeatures | None, PRNGKeyArray]:
    """Extract and project features from protein structure.

    Args:
      prng_key: PRNG key for coordinate noise.
      structure_coordinates: Atomic coordinates (N, CA, C, O).
      mask: Alpha carbon mask.
      residue_index: Residue indices.
      chain_index: Chain indices.
      backbone_noise: Noise to add to backbone coordinates.
      backbone_noise_mode: Mode for backbone noise ("direct" or "thermal").
      structure_mapping: Optional (N,) array mapping each residue to a structure ID.
                        When provided (multi-state mode), prevents cross-structure
                        neighbors to avoid information leakage between conformational states.
      initial_node_features: Optional initial node features.
      rbf_features: Optional precomputed RBF features from proxide (N, K, 400).
                    If provided, dynamic graph building/noise is skipped for edges.
      neighbor_indices: Optional precomputed neighbor indices from proxide (N, K).
                        Must be provided if rbf_features is provided to ensure consistency.

    Returns:
      Tuple of (edge_features, neighbor_indices, updated_prng_key).

    """
    stages = self.forward_edge_stages(
      prng_key,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      backbone_noise,
      backbone_noise_mode=backbone_noise_mode,
      structure_mapping=structure_mapping,
      initial_node_features=initial_node_features,
      rbf_features=rbf_features,
      neighbor_indices=neighbor_indices,
    )
    return stages.final, stages.neighbor_indices, stages.node_features_out, stages.prng_key
