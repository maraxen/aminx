"""Ligand-aware feature extraction module for Aminx.

This module contains the ProteinFeaturesLigand class that extracts features
from both protein coordinates and ligand/atomic context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from aminx.model.features import top_k
from aminx.model.ligand_tiling import map_chunks_axis0
from aminx.utils.coordinates import apply_noise_to_coordinates

if TYPE_CHECKING:
  from aminx.types.arrays import (
    AlphaCarbonMask,
    ChainIndex,
    ResidueIndex,
    StructureAtomicCoordinates,
  )

PRNGKeyArray = jax.Array
LayerNorm = eqx.nn.LayerNorm

# Static periodic table features: (3, 119)
# 0: Atomic Number (0-118)
# 1: Group (1..18, 0 for null)
# 2: Period (1..7, 0 for null)
_R1_G = [0, 1, 18]
_R23_G = [1, 2, 13, 14, 15, 16, 17, 18]
_R45_G = list(range(1, 19))
_R67_G = [1, 2] + [3] * 15 + list(range(4, 19))
_GROUP_FEATURES = jnp.array(_R1_G + _R23_G + _R23_G + _R45_G + _R45_G + _R67_G + _R67_G)

_R1_P = [0, 1, 1]
_R2_P = [2] * 8
_R3_P = [3] * 8
_R4_P = [4] * 18
_R5_P = [5] * 18
_R6_P = [6] * 32
_R7_P = [7] * 32
_PERIOD_FEATURES = jnp.array(_R1_P + _R2_P + _R3_P + _R4_P + _R5_P + _R6_P + _R7_P)

PERIODIC_TABLE_FEATURES = jnp.stack(
  [
    jnp.arange(119),
    _GROUP_FEATURES,
    _PERIOD_FEATURES,
  ],
)

SIDE_CHAIN_ATOM_TYPES = jnp.array(
  [
    6,
    6,
    6,
    8,
    8,
    16,
    6,
    6,
    6,
    7,
    7,
    8,
    8,
    16,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    8,
    8,
    6,
    7,
    7,
    8,
    6,
    6,
    6,
    7,
    8,
  ],
)


class PositionalEncodings(eqx.Module):
  """Positional encodings for residues and chains."""

  w_pos: eqx.nn.Linear
  num_embeddings: int = eqx.field(static=True)

  def __init__(self, num_embeddings: int, *, key: PRNGKeyArray) -> None:
    # num_embeddings here refers to the number of relative features (e.g. 16 or 32)
    # The output dimension is ALWAYS 16 in the reference models
    self.num_embeddings = num_embeddings
    # Input to linear is [offset_one_hot(2*num_pos + 1), chain_one_hot(1)]
    self.w_pos = eqx.nn.Linear(2 * num_embeddings + 2, 16, use_bias=False, key=key)

  def __call__(self, offset: jax.Array, same_chain: jax.Array) -> jax.Array:
    # offset: (N, K) relative residue indices
    # same_chain: (N, K) boolean for same-chain status

    # Clamp offset to [-num_embeddings, num_embeddings]
    d_offset = jnp.clip(offset + self.num_embeddings, 0, 2 * self.num_embeddings).astype(jnp.int32)
    offset_one_hot = jax.nn.one_hot(d_offset, 2 * self.num_embeddings + 1)

    # Combine with chain info
    # same_chain=1 for same chain

    chain_info = (1.0 - same_chain.astype(jnp.float32))[..., None]

    # Combined input: (N, K, 2*num_embeddings + 1 + 1)
    d_combined = jnp.concatenate([offset_one_hot, chain_info], axis=-1)

    return jax.vmap(jax.vmap(self.w_pos))(d_combined)


class ProteinFeaturesLigand(eqx.Module):
  """Extracts features from protein coordinates and ligand context."""

  embeddings: PositionalEncodings
  edge_embedding: eqx.nn.Linear
  norm_edges: LayerNorm

  node_project_down: eqx.nn.Linear
  norm_nodes: LayerNorm

  type_linear: eqx.nn.Linear

  y_nodes: eqx.nn.Linear
  y_edges: eqx.nn.Linear

  norm_y_edges: LayerNorm
  norm_y_nodes: LayerNorm

  w_e_proj: eqx.nn.Linear  # Added to match W_e in ProteinMPNN

  k_neighbors: int = eqx.field(static=True)
  atom_context_num: int = eqx.field(static=True)
  use_side_chains: bool = eqx.field(static=True)
  # Chunk axis-0 ligand pairwise edges / node projections for bounded GPU peak memory (>0 tiles).
  # <=0 restores legacy monolithic (L*M*M,) GEMMs (fine for small L / debugging parity).
  ligand_l_chunk: int = eqx.field(static=True)

  def _make_angle_features(
    self, atom_n: jax.Array, atom_ca: jax.Array, atom_c: jax.Array, ligand_atom_coords: jax.Array,
  ) -> jax.Array:
    """Port of _make_angle_features from PyTorch."""
    v1 = atom_n - atom_ca
    v2 = atom_c - atom_ca
    e1 = v1 / (jnp.linalg.norm(v1, axis=-1, keepdims=True) + 1e-8)
    e1_v2_dot = jnp.sum(e1 * v2, axis=-1, keepdims=True)
    u2 = v2 - e1 * e1_v2_dot
    e2 = u2 / (jnp.linalg.norm(u2, axis=-1, keepdims=True) + 1e-8)
    e3 = jnp.cross(e1, e2, axis=-1)

    # R_residue: (L, 3, 3) relative rotation matrix
    R_residue = jnp.stack([e1, e2, e3], axis=-1)

    # local_vectors: relative position of ligand_atom_coords in residue frame
    # (L, M, 3)
    diff = ligand_atom_coords - atom_ca[:, None, :]
    local_vectors = jnp.einsum("lqp, lyq -> lyp", R_residue, diff)

    rxy = jnp.sqrt(local_vectors[..., 0] ** 2 + local_vectors[..., 1] ** 2 + 1e-8)
    f1 = local_vectors[..., 0] / rxy
    f2 = local_vectors[..., 1] / rxy
    rxyz = jnp.linalg.norm(local_vectors, axis=-1) + 1e-8
    f3 = rxy / rxyz
    f4 = local_vectors[..., 2] / rxyz

    return jnp.stack([f1, f2, f3, f4], axis=-1)

  def _rbf(self, D: jax.Array) -> jax.Array:
    """Standard RBF bins [2.0, 22.0] for LigandMPNN."""
    D_min, D_max, D_count = 2.0, 22.0, 16
    D_mu = jnp.linspace(D_min, D_max, D_count)
    D_sigma = (D_max - D_min) / D_count
    return jnp.exp(-(((D[..., None] - D_mu) / D_sigma) ** 2))

  def _get_rbf(self, A: jax.Array, B: jax.Array, E_idx: jax.Array) -> jax.Array:
    D_A_B = jnp.sqrt(jnp.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1) + 1e-6)
    D_neighbors = jnp.take_along_axis(D_A_B, E_idx, axis=1)
    return self._rbf(D_neighbors)

  def _y_edges_coords_to_embed(self, Y: jax.Array) -> jax.Array:
    """Dense ligand-ligand pairwise RBF + y_edges projection for residue slab Y [* , M , 3]."""
    raw = jnp.sqrt(jnp.sum((Y[:, :, None, :] - Y[:, None, :, :]) ** 2, axis=-1) + 1e-6)
    y_edges = self._rbf(raw)
    l_, m_, m2_, f_in = y_edges.shape
    f_out = self.y_edges.weight.shape[0]
    flat = y_edges.reshape(l_ * m_ * m2_, f_in) @ self.y_edges.weight.T
    if self.y_edges.bias is not None:
      flat = flat + self.y_edges.bias
    return flat.reshape(l_, m_, m2_, f_out)

  def _y_nodes_proj(self, Y_t_1hot_: jax.Array) -> jax.Array:
    """Apply y_nodes to ligand atom one-hots [* , M , 147]."""
    l_, m_, f_in = Y_t_1hot_.shape
    f_out = self.y_nodes.weight.shape[0]
    flat = Y_t_1hot_.reshape(l_ * m_, f_in) @ self.y_nodes.weight.T
    if self.y_nodes.bias is not None:
      flat = flat + self.y_nodes.bias
    return flat.reshape(l_, m_, f_out)

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    k_neighbors: int,
    atom_context_num: int = 16,
    num_positional_embeddings: int = 16,
    ligand_l_chunk: int = 16,
    *,
    use_side_chains: bool = False,
    key: PRNGKeyArray,
  ) -> None:
    keys = jax.random.split(key, 8)

    self.k_neighbors = k_neighbors
    self.atom_context_num = atom_context_num
    self.use_side_chains = use_side_chains
    self.ligand_l_chunk = ligand_l_chunk

    self.embeddings = PositionalEncodings(num_positional_embeddings, key=keys[0])
    # edge_in = 16 + 16 * 25 = 416
    self.edge_embedding = eqx.nn.Linear(416, edge_features, use_bias=False, key=keys[1])
    self.norm_edges = LayerNorm(edge_features)

    # node input: 5 * 16 (rbf) + 64 (type) + 4 (angle) = 148
    self.node_project_down = eqx.nn.Linear(148, node_features, key=keys[2])
    self.norm_nodes = LayerNorm(node_features)

    self.type_linear = eqx.nn.Linear(147, 64, key=keys[3])
    self.y_nodes = eqx.nn.Linear(147, node_features, use_bias=False, key=keys[4])
    self.y_edges = eqx.nn.Linear(16, node_features, use_bias=False, key=keys[5])

    self.norm_y_edges = LayerNorm(node_features)
    self.norm_y_nodes = LayerNorm(node_features)

    self.w_e_proj = eqx.nn.Linear(edge_features, edge_features, key=keys[7])

  def __call__(
    self,
    _key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    ligand_coords: jnp.ndarray,  # Ligand coords [L, M, 3]
    ligand_atom_types: jnp.ndarray,  # Ligand types [L, M]
    ligand_mask: jnp.ndarray,  # Ligand mask [L, M]
    backbone_noise: float = 0.0,
    structure_mapping: jnp.ndarray | None = None,
    *,
    atom_37: jnp.ndarray | None = None,
    atom_37_mask: jnp.ndarray | None = None,
    chain_mask: jnp.ndarray | None = None,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # 0. Noise Application
    # TODO: allow caller to supply an arbitrary noise_fn(key, coords, backbone_noise) -> coords
    # so both Aminx and LigandMPNN can share a configurable noise strategy via StageSet.
    noise_key, _key = jax.random.split(_key)
    noised_coords, _ = apply_noise_to_coordinates(
      noise_key,
      structure_coordinates,
      backbone_noise=backbone_noise,
    )
    structure_coordinates = jnp.where(
      backbone_noise > 0,
      noised_coords,
      structure_coordinates,
    )

    # N, CA, C, O
    N = structure_coordinates[:, 0, :]
    Ca = structure_coordinates[:, 1, :]
    C = structure_coordinates[:, 2, :]
    O = structure_coordinates[:, 3, :]

    # Virtual Cb
    b = Ca - N
    c = C - Ca
    a = jnp.cross(b, c, axis=-1)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

    # 1. Edge Features
    dist_ca = jnp.sqrt(jnp.sum((Ca[:, None, :] - Ca[None, :, :]) ** 2, axis=-1) + 1e-6)
    mask_2d = mask[:, None] * mask[None, :]
    dist_ca = dist_ca * mask_2d + (1.0 - mask_2d) * 1e4
    if structure_mapping is not None:
      same_structure = structure_mapping[:, None] == structure_mapping[None, :]
      dist_ca = jnp.where(same_structure, dist_ca, 1e4)

    k = min(self.k_neighbors, Ca.shape[0])
    # aminx.model.features.top_k, not jax.lax.top_k: the latter lowers to a
    # stablehlo.composite IREE refuses to legalize, and its tie order would come
    # from the backend's sort stability. See that function's docstring.
    _, E_idx = top_k(-dist_ca, k)

    RBF_all = []
    # CA-CA RBF
    r_ca_ca = jnp.take_along_axis(dist_ca, E_idx, axis=1)
    RBF_all.append(self._rbf(r_ca_ca))

    # All other 24 pairs
    pairs = [
      (N, N),
      (C, C),
      (O, O),
      (Cb, Cb),
      (Ca, N),
      (Ca, C),
      (Ca, O),
      (Ca, Cb),
      (N, C),
      (N, O),
      (N, Cb),
      (Cb, C),
      (Cb, O),
      (O, C),
      (N, Ca),
      (C, Ca),
      (O, Ca),
      (Cb, Ca),
      (C, N),
      (O, N),
      (Cb, N),
      (C, Cb),
      (O, Cb),
      (C, O),
    ]
    for p_a, p_b in pairs:
      RBF_all.append(self._get_rbf(p_a, p_b, E_idx))

    RBF_all = jnp.concatenate(RBF_all, axis=-1)

    offset = residue_index[:, None] - residue_index[None, :]
    E_offset = jnp.take_along_axis(offset, E_idx, axis=1)
    E_chains = jnp.take_along_axis((chain_index[:, None] == chain_index[None, :]), E_idx, axis=1)

    E_pos = self.embeddings(E_offset, E_chains)
    E = jnp.concatenate([E_pos, RBF_all], axis=-1)
    E = jax.vmap(jax.vmap(self.edge_embedding))(E)
    E = jax.vmap(jax.vmap(self.norm_edges))(E)
    E = jax.vmap(jax.vmap(self.w_e_proj))(E)

    if self.use_side_chains:
      if atom_37 is None or atom_37_mask is None:
        msg = "atom_37 and atom_37_mask must be provided when use_side_chains=True"
        raise ValueError(msg)

      chain_mask_in = (
        jnp.zeros_like(mask, dtype=jnp.float32)
        if chain_mask is None
        else chain_mask.astype(jnp.float32)
      )

      e_idx_sub = E_idx[:, :16]
      atom_37_mask = atom_37_mask * (1.0 - chain_mask_in[:, None])
      r_m = atom_37_mask[:, 5:][e_idx_sub]
      r = atom_37[:, 5:, :][e_idx_sub]
      r_t = jnp.broadcast_to(
        SIDE_CHAIN_ATOM_TYPES[None, None, :],
        (mask.shape[0], e_idx_sub.shape[1], SIDE_CHAIN_ATOM_TYPES.shape[0]),
      )

      r = r.reshape(mask.shape[0], -1, 3)
      r_m = r_m.reshape(mask.shape[0], -1)
      r_t = r_t.reshape(mask.shape[0], -1)

      ligand_coords = jnp.concatenate([r, ligand_coords], axis=1)
      ligand_mask = jnp.concatenate([r_m.astype(ligand_mask.dtype), ligand_mask], axis=1)
      ligand_atom_types = jnp.concatenate(
        [r_t.astype(ligand_atom_types.dtype), ligand_atom_types], axis=1,
      )

    # Always select atom_context_num nearest atoms — must happen whether or not
    # use_side_chains is True.  Without this, A=155 flows into _y_edges_coords_to_embed
    # and pre-allocates (L, 155, 155, node_features) ≈ 20 GiB on flat-multistate inputs.
    # Clamp k to actual atom count: dummy ligands (with_ligand=False) have A=1 which is
    # smaller than atom_context_num=16, matching the pattern at line 303 for protein graph.
    cb_y_distances = jnp.sum((Cb[:, None, :] - ligand_coords) ** 2, axis=-1)
    mask_y = mask[:, None] * ligand_mask
    cb_y_distances_adjusted = cb_y_distances * mask_y + (1.0 - mask_y) * 10000.0
    k_y = min(self.atom_context_num, ligand_coords.shape[1])
    # See the note at the CA-CA selection above: aminx's top_k, not jax.lax's.
    _, e_idx_y = top_k(-cb_y_distances_adjusted, k_y)

    ligand_coords = jnp.take_along_axis(ligand_coords, e_idx_y[:, :, None], axis=1)
    ligand_atom_types = jnp.take_along_axis(ligand_atom_types, e_idx_y, axis=1)
    ligand_mask = jnp.take_along_axis(ligand_mask, e_idx_y, axis=1)

    # 2. Node/Ligand Features
    # type_1hot: (L, M, 147)
    Y_t_cast = ligand_atom_types.astype(jnp.int32)
    Y_t_g = PERIODIC_TABLE_FEATURES[1, Y_t_cast]
    Y_t_p = PERIODIC_TABLE_FEATURES[2, Y_t_cast]

    Y_t_1hot_ = jnp.concatenate(
      [
        jax.nn.one_hot(Y_t_cast, 120),
        jax.nn.one_hot(Y_t_g.astype(jnp.int32), 19),
        jax.nn.one_hot(Y_t_p.astype(jnp.int32), 8),
      ],
      axis=-1,
    )

    Y_t_1hot = jax.vmap(jax.vmap(self.type_linear))(Y_t_1hot_)

    r_n_y = self._rbf(jnp.sqrt(jnp.sum((N[:, None, :] - ligand_coords) ** 2, axis=-1) + 1e-6))
    r_ca_y = self._rbf(jnp.sqrt(jnp.sum((Ca[:, None, :] - ligand_coords) ** 2, axis=-1) + 1e-6))
    r_c_y = self._rbf(jnp.sqrt(jnp.sum((C[:, None, :] - ligand_coords) ** 2, axis=-1) + 1e-6))
    r_o_y = self._rbf(jnp.sqrt(jnp.sum((O[:, None, :] - ligand_coords) ** 2, axis=-1) + 1e-6))
    r_cb_y = self._rbf(jnp.sqrt(jnp.sum((Cb[:, None, :] - ligand_coords) ** 2, axis=-1) + 1e-6))

    f_angles = self._make_angle_features(N, Ca, C, ligand_coords)

    D_all = jnp.concatenate([r_n_y, r_ca_y, r_c_y, r_o_y, r_cb_y, Y_t_1hot, f_angles], axis=-1)
    V = jax.vmap(jax.vmap(self.node_project_down))(D_all)
    V = jax.vmap(jax.vmap(self.norm_nodes))(V)

    # Ligand–ligand edges: O(L·M²); chunk axis L when ligand_l_chunk > 0
    if self.ligand_l_chunk <= 0:
      Y_edges = self._y_edges_coords_to_embed(ligand_coords)
    else:
      Y_edges = map_chunks_axis0(
        ligand_coords,
        chunk_size=self.ligand_l_chunk,
        # Unbound callable avoids JAX 0.9.x WeakKeyDictionary cache keying BoundMethod(self=_)
        # when outer JIT traces bool from autoregressive submatrices (TracerBoolConversionError).
        fn=lambda slab, _self=self: ProteinFeaturesLigand._y_edges_coords_to_embed(_self, slab),
      )

    if self.ligand_l_chunk <= 0:
      Y_nodes = self._y_nodes_proj(Y_t_1hot_)
    else:
      Y_nodes = map_chunks_axis0(
        Y_t_1hot_,
        chunk_size=self.ligand_l_chunk,
        fn=lambda slab2, _self=self: ProteinFeaturesLigand._y_nodes_proj(_self, slab2),
      )

    # Apply layer normalization via vmap (still needed to preserve batch normalization semantics)
    Y_edges = jax.vmap(jax.vmap(jax.vmap(self.norm_y_edges)))(Y_edges)
    Y_nodes = jax.vmap(jax.vmap(self.norm_y_nodes))(Y_nodes)

    return V, E, E_idx, Y_nodes, Y_edges, ligand_mask
