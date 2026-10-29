"""Caliby 3D-coordinate token-level positional encodings and pairwise RBF graph features.

Ports ``CalibyPositionalEncodings`` and ``TokenFeatures`` (both ``model_version=0``
with 8481 raw edge channels and ``model_version=1`` with 1057 raw edge channels)
from ``ProteinDesignLab/caliby`` (``caliby/model/lr_potts_model/features.py``).
"""

from __future__ import annotations

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

# 5 backbone atom indices within Atom23 (N=0, CA=1, C=2, O=3, CB=4)
_BB5_IDXS: jax.Array = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
# 4 backbone atom indices within Atom23 (N=0, CA=1, C=2, O=3)
_BB4_IDXS: jax.Array = jnp.array([0, 1, 2, 3], dtype=jnp.int32)


def _linear(layer: eqx.nn.Linear, x: Float[Array, "... I"]) -> Float[Array, "... O"]:
  """Apply ``eqx.nn.Linear`` over the trailing axis of ``x`` without ``vmap``."""
  out = x @ layer.weight.T
  if layer.bias is not None:
    out = out + layer.bias
  return out


def _layer_norm(norm: eqx.nn.LayerNorm, x: Float[Array, "... H"]) -> Float[Array, "... H"]:
  """Apply ``eqx.nn.LayerNorm`` over the trailing axis of ``x`` without ``vmap``."""
  mean = jnp.mean(x, axis=-1, keepdims=True)
  variance = jnp.var(x, axis=-1, keepdims=True)
  out = (x - mean) * jax.lax.rsqrt(variance + norm.eps)
  if norm.weight is not None:
    out = out * norm.weight
  if norm.bias is not None:
    out = out + norm.bias
  return out


def make_virtual_cb(
  n_xyz: Float[Array, "L 3"],
  ca_xyz: Float[Array, "L 3"],
  c_xyz: Float[Array, "L 3"],
) -> Float[Array, "L 3"]:
  """Compute ideal tetrahedral virtual C-beta coordinates from backbone N, CA, C.

  Matches ``ProteinMPNN`` / ``Caliby`` virtual C-beta construction:
  ``b = CA - N``, ``c = C - CA``, ``a = cross(b, c)``,
  ``Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + CA``.

  Parameters
  ----------
  n_xyz : Float[Array, "L 3"]
    Backbone N coordinates.
  ca_xyz : Float[Array, "L 3"]
    Backbone CA coordinates.
  c_xyz : Float[Array, "L 3"]
    Backbone C coordinates.

  Returns
  -------
  Float[Array, "L 3"]
    Idealized virtual C-beta coordinates of shape ``(L, 3)``.
  """
  b = ca_xyz - n_xyz
  c = c_xyz - ca_xyz
  a = jnp.cross(b, c)
  return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_xyz


def compute_caliby_knn(
  ca_xyz: Float[Array, "L 3"],
  ca_mask: Float[Array, " L"],
  top_k: int,
  eps: float = 1e-6,
) -> tuple[Float[Array, "L K"], Int[Array, "L K"]]:
  """Compute masked pairwise CA Euclidean distances and top-K nearest neighbors.

  Parameters
  ----------
  ca_xyz : Float[Array, "L 3"]
    Representative token CA coordinates.
  ca_mask : Float[Array, " L"]
    Token validity mask (``1.0`` valid, ``0.0`` padded/missing).
  top_k : int
    Maximum number of nearest neighbors ``K``.
  eps : float, default=1e-6
    Small constant inside ``sqrt(sum(dX^2) + eps)``.

  Returns
  -------
  tuple[Float[Array, "L K"], Int[Array, "L K"]]
    Tuple ``(d_neighbors, e_idx)`` of neighbor distances and integer indices.
  """
  num_tokens = ca_xyz.shape[0]
  k_eff = min(top_k, num_tokens)
  mask_2d = ca_mask[:, None] * ca_mask[None, :]
  diff = ca_xyz[:, None, :] - ca_xyz[None, :, :]
  dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + eps) * mask_2d
  dist_adjusted = dist + (1.0 - mask_2d) * 10000.0
  neg_d_neighbors, e_idx = jax.lax.top_k(-dist_adjusted, k_eff)
  return -neg_d_neighbors, e_idx.astype(jnp.int32)


def compute_rbf_grid(
  dist: Float[Array, "..."],
  *,
  d_min: float = 2.0,
  d_max: float = 22.0,
  d_count: int = 16,
) -> Float[Array, "... R"]:
  """Expand pairwise distances into Gaussian radial basis functions.

  Parameters
  ----------
  dist : Float[Array, "..."]
    Pairwise Euclidean distance tensor.
  d_min : float, default=2.0
    Minimum RBF center in Angstroms.
  d_max : float, default=22.0
    Maximum RBF center in Angstroms.
  d_count : int, default=16
    Number of RBF centers.

  Returns
  -------
  Float[Array, "... R"]
    Gaussian RBF tensor with trailing dimension ``d_count``.
  """
  d_mu = jnp.linspace(d_min, d_max, d_count, dtype=dist.dtype)
  d_sigma = (d_max - d_min) / d_count
  diff = (dist[..., None] - d_mu) / d_sigma
  return jnp.exp(-(diff * diff))


_RESIDUE_INDEX_GAP: int = 100


class CalibyPositionalEncodings(eqx.Module):
  """Caliby relative residue positional encoding module matching ``atom_mpnn.py:319-332``.

  Attributes
  ----------
  linear : eqx.nn.Linear
    Linear projection from ``2 * max_relative_feature + 2 = 66``
    one-hot channels to ``num_positional_embeddings`` (default ``16``).
  max_relative_feature : int
    Maximum clipped relative offset (default ``32``).
  """

  linear: eqx.nn.Linear
  max_relative_feature: int = eqx.field(static=True, default=32)

  def __init__(
    self,
    num_embeddings: int = 16,
    max_relative_feature: int = 32,
    *,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``CalibyPositionalEncodings``.

    Parameters
    ----------
    num_embeddings : int, default=16
      Output embedding dimension.
    max_relative_feature : int, default=32
      Maximum relative sequence/token offset before clipping.
    key : jax.Array | None, optional
      PRNG key for weight initialization.

    Returns
    -------
    None
      Initializes instance state.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    in_dim = 2 * max_relative_feature + 2
    self.linear = eqx.nn.Linear(in_dim, num_embeddings, key=key)
    self.max_relative_feature = max_relative_feature

  def __call__(
    self,
    residue_idx: Int[Array, " L"],
    token_idx: Int[Array, " L"],
    asym_id: Int[Array, " L"],
    entity_id: Int[Array, " L"],
    sym_id: Int[Array, " L"],
    e_idx: Int[Array, "L K"],
  ) -> Float[Array, "L K P"]:
    """Compute relative positional embeddings for each ``(i, j in E_idx[i])`` edge.

    Parameters
    ----------
    residue_idx : Int[Array, " L"]
      Residue sequence indices.
    token_idx : Int[Array, " L"]
      Monotonic token indices.
    asym_id : Int[Array, " L"]
      Chain (asymmetric unit) IDs.
    entity_id : Int[Array, " L"]
      Entity IDs.
    sym_id : Int[Array, " L"]
      Symmetry copy IDs within an entity.
    e_idx : Int[Array, "L K"]
      Neighbor index table of shape ``(L, K)``.

    Returns
    -------
    Float[Array, "L K P"]
      Positional edge embeddings of shape ``(L, K, num_embeddings)``.
    """
    del token_idx, entity_id, sym_id
    dtype = self.linear.weight.dtype
    num_classes = 2 * self.max_relative_feature + 2
    na_bucket = 2 * self.max_relative_feature + 1

    same_chain = asym_id[:, None] == asym_id[e_idx]
    d = (
      residue_idx[:, None]
      - residue_idx[e_idx]
      + _RESIDUE_INDEX_GAP * (1 - same_chain.astype(jnp.int32))
    )
    d_clipped = jnp.clip(
      d + self.max_relative_feature,
      0,
      2 * self.max_relative_feature,
    )
    d_final = jnp.where(same_chain, d_clipped, na_bucket)
    oh = jax.nn.one_hot(d_final, num_classes, dtype=dtype)
    return _linear(self.linear, oh)


class CalibyTokenFeatures(eqx.Module):
  """Caliby token and pairwise RBF edge featurizer supporting ``model_version`` 0 and 1.

  Attributes
  ----------
  embeddings : CalibyPositionalEncodings
    Relative positional encoding module (16 channels).
  edge_embedding : eqx.nn.Linear
    Linear projection from raw edge channels (``8481`` for v0, ``1057`` for v1)
    to ``edge_features`` (``128``), without bias (``use_bias=False``).
  norm_edges : eqx.nn.LayerNorm
    LayerNorm over projected edge features.
  node_embedding : eqx.nn.Linear
    Linear projection from ``molecule_type_encoding`` (5 channels) to
    ``node_features`` (``128``), without bias (``use_bias=False``).
  norm_nodes : eqx.nn.LayerNorm
    LayerNorm over projected node features.
  top_k : int
    Maximum neighbors ``K`` (default ``48``).
  num_rbf : int
    Number of Gaussian RBF bins per atom pair (default ``16``).
  num_positional_embeddings : int
    Positional embedding dimension (default ``16``).
  model_version : int
    Feature layout version: ``0`` (full ``23x23`` all-atom RBF grid = 8481 channels)
    or ``1`` (compact ``5x5`` backbone + ``4x10`` ligand RBF grid = 1057 channels).
  """

  embeddings: CalibyPositionalEncodings
  edge_embedding: eqx.nn.Linear
  norm_edges: eqx.nn.LayerNorm
  node_embedding: eqx.nn.Linear
  norm_nodes: eqx.nn.LayerNorm
  top_k: int = eqx.field(static=True, default=48)
  num_rbf: int = eqx.field(static=True, default=16)
  num_positional_embeddings: int = eqx.field(static=True, default=16)
  model_version: int = eqx.field(static=True, default=0)

  def __init__(
    self,
    edge_features: int = 128,
    node_features: int = 128,
    num_positional_embeddings: int = 16,
    num_rbf: int = 16,
    top_k: int = 48,
    model_version: int = 0,
    *,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``CalibyTokenFeatures``.

    Parameters
    ----------
    edge_features : int, default=128
      Output edge feature dimension.
    node_features : int, default=128
      Output node feature dimension.
    num_positional_embeddings : int, default=16
      Positional encoding dimension.
    num_rbf : int, default=16
      Number of radial basis functions per atom pair.
    top_k : int, default=48
      Number of nearest neighbors ``K``.
    model_version : int, default=0
      Feature layout version (``0`` for 8481 edge_in, ``1`` for 1057 edge_in).
    key : jax.Array
      PRNG key for weight initialization.

    Returns
    -------
    None
      Initializes instance state.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 3)
    self.embeddings = CalibyPositionalEncodings(
      num_embeddings=num_positional_embeddings,
      key=keys[0],
    )
    if model_version == 0:
      edge_in = num_positional_embeddings + num_rbf * (23 * 23) + 1
    elif model_version == 1:
      edge_in = num_positional_embeddings + num_rbf * (5 * 5) + num_rbf * (4 * 10) + 1
    else:
      msg = f"Unsupported Caliby TokenFeatures model_version: {model_version}"
      raise ValueError(msg)

    self.edge_embedding = eqx.nn.Linear(edge_in, edge_features, use_bias=False, key=keys[1])
    self.norm_edges = eqx.nn.LayerNorm(edge_features)
    self.node_embedding = eqx.nn.Linear(5, node_features, use_bias=False, key=keys[2])
    self.norm_nodes = eqx.nn.LayerNorm(node_features)
    self.top_k = top_k
    self.num_rbf = num_rbf
    self.num_positional_embeddings = num_positional_embeddings
    self.model_version = model_version

  def __call__(
    self,
    atom23_xyz: Float[Array, "L 23 3"],
    atom23_mask: Float[Array, "L 23"],
    residue_idx: Int[Array, " L"],
    token_idx: Int[Array, " L"],
    asym_id: Int[Array, " L"],
    entity_id: Int[Array, " L"],
    sym_id: Int[Array, " L"],
    molecule_type_encoding: Float[Array, "L 5"] | None = None,
    is_ligand: Bool[Array, " L"] | None = None,
  ) -> tuple[Float[Array, "L H"], Float[Array, "L K E"], Int[Array, "L K"]]:
    """Compute initial node embeddings ``V``, edge embeddings ``E``, and neighbor indices ``E_idx``.

    Parameters
    ----------
    atom23_xyz : Float[Array, "L 23 3"]
      All-atom coordinates in AF3 23-atom representation (``N, CA, C, O, CB, ...``).
    atom23_mask : Float[Array, "L 23"]
      Atom presence mask of shape ``(L, 23)``.
    residue_idx : Int[Array, " L"]
      Residue sequence numbers.
    token_idx : Int[Array, " L"]
      Token sequence indices.
    asym_id : Int[Array, " L"]
      Chain IDs.
    entity_id : Int[Array, " L"]
      Entity IDs.
    sym_id : Int[Array, " L"]
      Symmetry IDs.
    molecule_type_encoding : Float[Array, "L 5"] | None, optional
      5-channel one-hot molecule type encoding (defaulting to protein ``[1, 0, 0, 0, 0]``).
    is_ligand : Bool[Array, " L"] | None, optional
      Boolean mask indicating non-protein/ligand tokens (default all ``False``).

    Returns
    -------
    tuple[Float[Array, "L H"], Float[Array, "L K E"], Int[Array, "L K"]]
      Tuple ``(V, E, E_idx)`` of node features, edge features, and neighbor indices.
    """
    dtype = self.edge_embedding.weight.dtype
    num_tokens = atom23_xyz.shape[0]
    xyz = jnp.nan_to_num(atom23_xyz.astype(dtype), nan=0.0)
    mask23 = atom23_mask.astype(dtype)

    if is_ligand is None:
      lig_bool = jnp.zeros((num_tokens,), dtype=jnp.bool_)
    else:
      lig_bool = is_ligand.astype(jnp.bool_)

    # Representative CA coordinate (index 1 for protein, index 0 for ligand)
    ca_xyz = jnp.where(lig_bool[:, None], xyz[:, 0, :], xyz[:, 1, :])
    ca_mask = jnp.where(lig_bool, mask23[:, 0], mask23[:, 1])

    _, e_idx = compute_caliby_knn(ca_xyz, ca_mask, self.top_k)
    num_k = e_idx.shape[1]

    # Replace masked/missing atoms with CA before virtual CB & RBF computation (atom_mpnn.py:424)
    xyz_ca_filled = jnp.where(mask23[..., None] > 0, xyz, xyz[:, 1:2, :])

    # Populate virtual CB at index 4 where CB is missing (e.g., Glycine or backbone-only)
    cb_virtual = make_virtual_cb(xyz_ca_filled[:, 0, :], xyz_ca_filled[:, 1, :], xyz_ca_filled[:, 2, :])
    has_bb = (mask23[:, 0] > 0) & (mask23[:, 1] > 0) & (mask23[:, 2] > 0) & (~lig_bool)
    use_virtual_cb = has_bb & (mask23[:, 4] <= 0)
    cb_coords = jnp.where(use_virtual_cb[:, None], cb_virtual, xyz_ca_filled[:, 4, :])

    xyz_with_cb = jnp.concatenate([xyz_ca_filled[:, :4, :], cb_coords[:, None, :], xyz_ca_filled[:, 5:, :]], axis=1)

    e_positional = self.embeddings(
      residue_idx=residue_idx,
      token_idx=token_idx,
      asym_id=asym_id,
      entity_id=entity_id,
      sym_id=sym_id,
      e_idx=e_idx,
    )

    # Edge ligand indicator: whether either endpoint i or j is a ligand token
    edge_lig = (lig_bool[:, None] | lig_bool[e_idx]).astype(dtype)[..., None]

    if self.model_version == 0:
      # Vectorized 5D broadcast over all 23 x 23 atom pairs: (L, K, 23, 23, 3)
      xyz_i = xyz_with_cb[:, None, :, None, :]
      xyz_j = xyz_with_cb[e_idx][:, :, None, :, :]
      diff_all = xyz_i - xyz_j
      dist_all = jnp.sqrt(jnp.sum(diff_all * diff_all, axis=-1) + 1e-6)
      rbf_all = compute_rbf_grid(dist_all, d_count=self.num_rbf)
      rbf_flat = rbf_all.reshape(num_tokens, num_k, 23 * 23 * self.num_rbf)
      raw_edges = jnp.concatenate([e_positional, rbf_flat, edge_lig], axis=-1)
    else:
      # model_version == 1:
      # 1. 5x5 backbone-to-backbone RBFs (N, CA, C, O, CB) -> 25 * 16 = 400 channels
      bb5_xyz = xyz_with_cb[:, _BB5_IDXS, :]
      bb_diff = bb5_xyz[:, None, :, None, :] - bb5_xyz[e_idx][:, :, None, :, :]
      bb_dist = jnp.sqrt(jnp.sum(bb_diff * bb_diff, axis=-1) + 1e-6)
      rbf_bb = compute_rbf_grid(bb_dist, d_count=self.num_rbf)
      rbf_bb_flat = rbf_bb.reshape(num_tokens, num_k, 5 * 5 * self.num_rbf)

      # 2. 4x10 protein-backbone (N, CA, C, O) to sidechain/ligand slots 4..13 RBFs -> 40 * 16 = 640 channels
      bb4_xyz = xyz_with_cb[:, :4, :]
      sc10_xyz = xyz_with_cb[:, 4:14, :]
      sc_diff = bb4_xyz[:, None, :, None, :] - sc10_xyz[e_idx][:, :, None, :, :]
      sc_dist = jnp.sqrt(jnp.sum(sc_diff * sc_diff, axis=-1) + 1e-6)
      rbf_sc = compute_rbf_grid(sc_dist, d_count=self.num_rbf)
      rbf_sc_flat = rbf_sc.reshape(num_tokens, num_k, 4 * 10 * self.num_rbf)
      raw_edges = jnp.concatenate([e_positional, rbf_bb_flat, rbf_sc_flat, edge_lig], axis=-1)

    edges = _layer_norm(self.norm_edges, _linear(self.edge_embedding, raw_edges))

    if molecule_type_encoding is None:
      nodes = jnp.zeros((num_tokens, self.node_embedding.weight.shape[0]), dtype=dtype)
    else:
      mol_enc = molecule_type_encoding.astype(dtype)
      nodes = _layer_norm(self.norm_nodes, _linear(self.node_embedding, mol_enc))

    return cast("Float[Array, 'L H']", nodes), cast("Float[Array, 'L K E']", edges), e_idx
