"""JAX/Equinox implementation of the LigandMPNN Side-Chain Packer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from aminx.model.decoder import DecoderLayer, DecoderLayerJ
from aminx.model.dropout import Dropout
from aminx.model.encoder import EncoderLayer
from aminx.model.features import top_k
from aminx.types.bundles import PackerBundle, PackerResult
from aminx.types.configs import InferenceConfig
from aminx.utils.concatenate import concatenate_neighbor_nodes
from aminx.utils.coordinates import apply_noise_to_coordinates

if TYPE_CHECKING:
  from aminx.types.arrays import (
    PRNGKeyArray,
  )

# Constants for periodic table features
# Group; 19 categories including 0
PERIODIC_TABLE_GROUP = [
  0,
  1,
  18,
  1,
  2,
  13,
  14,
  15,
  16,
  17,
  18,
  1,
  2,
  13,
  14,
  15,
  16,
  17,
  18,
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  12,
  13,
  14,
  15,
  16,
  17,
  18,
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  12,
  13,
  14,
  15,
  16,
  17,
  18,
  1,
  2,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  12,
  13,
  14,
  15,
  16,
  17,
  18,
  1,
  2,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  12,
  13,
  14,
  15,
  16,
  17,
  18,
]
# Period; 8 categories including 0
PERIODIC_TABLE_PERIOD = [
  0,
  1,
  1,
  2,
  2,
  2,
  2,
  2,
  2,
  2,
  2,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  3,
  4,
  4,
  4,
  4,
  4,
  4,
  4,
  4,
  4,
  4,
  4,
  4,
  4,
  4,
  4,
  4,
  4,
  4,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  6,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
]

PERIODIC_TABLE_FEATURES = jnp.stack(
  [
    jnp.arange(119),
    jnp.array(PERIODIC_TABLE_GROUP),
    jnp.array(PERIODIC_TABLE_PERIOD),
  ],
)


def compute_rbf(
  dist: jnp.ndarray,
  lower_bound: float = 0.0,
  upper_bound: float = 20.0,
  num_bins: int = 16,
) -> jnp.ndarray:
  """Compute radial basis functions."""
  mu = jnp.linspace(lower_bound, upper_bound, num_bins)
  sigma = (upper_bound - lower_bound) / num_bins
  return jnp.exp(-jnp.square((dist[..., None] - mu) / sigma))


def gather_nodes(nodes: jnp.ndarray, neighbor_idx: jnp.ndarray) -> jnp.ndarray:
  """Gather node features based on neighbor indices."""
  return nodes[neighbor_idx]


def gather_edges(edges: jnp.ndarray, neighbor_idx: jnp.ndarray) -> jnp.ndarray:
  """Gather edge features based on neighbor indices."""
  return jax.vmap(lambda e, idx: e[idx])(edges, neighbor_idx)


class PositionalEncodings(eqx.Module):
  """Binned positional encodings."""

  linear: eqx.nn.Linear
  max_relative_feature: int = eqx.field(static=True)

  def __init__(self, num_embeddings: int, max_relative_feature: int = 32, *, key: PRNGKeyArray):
    self.max_relative_feature = max_relative_feature
    self.linear = eqx.nn.Linear(2 * max_relative_feature + 2, num_embeddings, key=key)

  def __call__(self, offset: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    d = jnp.clip(
      offset + self.max_relative_feature,
      0,
      2 * self.max_relative_feature,
    ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
    d_onehot = jax.nn.one_hot(d.astype(jnp.int32), 2 * self.max_relative_feature + 2)
    return jax.vmap(jax.vmap(self.linear))(d_onehot)


class PackerProteinFeatures(eqx.Module):
  """Port of ProteinFeatures from sc_utils.py."""

  enc_edge_embedding: eqx.nn.Linear
  enc_norm_edges: eqx.nn.LayerNorm
  enc_node_embedding: eqx.nn.Linear
  enc_norm_nodes: eqx.nn.LayerNorm

  w_xy_project_down1: eqx.nn.Linear
  dec_edge_embedding1: eqx.nn.Linear
  dec_norm_edges1: eqx.nn.LayerNorm
  dec_node_embedding1: eqx.nn.Linear
  dec_norm_nodes1: eqx.nn.LayerNorm

  node_project_down: eqx.nn.Linear
  norm_nodes: eqx.nn.LayerNorm

  type_linear: eqx.nn.Linear
  y_nodes: eqx.nn.Linear
  y_edges: eqx.nn.Linear
  norm_y_edges: eqx.nn.LayerNorm
  norm_y_nodes: eqx.nn.LayerNorm

  positional_embeddings: PositionalEncodings

  # Static fields
  top_k: int = eqx.field(static=True)
  num_rbf: int = eqx.field(static=True)
  atom_context_num: int = eqx.field(static=True)
  lower_bound: float = eqx.field(static=True)
  upper_bound: float = eqx.field(static=True)
  ca_idx: int = eqx.field(static=True)
  n_idx: int = eqx.field(static=True)
  c_idx: int = eqx.field(static=True)
  o_idx: int = eqx.field(static=True)

  def __init__(
    self,
    edge_features: int = 128,
    node_features: int = 128,
    num_positional_embeddings: int = 16,
    num_rbf: int = 16,
    top_k: int = 30,
    atom_context_num: int = 16,
    lower_bound: float = 0.0,
    upper_bound: float = 20.0,
    *,
    atom37_order: bool = False,
    key: PRNGKeyArray,
  ):
    self.top_k = top_k
    self.num_rbf = num_rbf
    self.atom_context_num = atom_context_num
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound
    self.n_idx = 0
    self.ca_idx = 1
    self.c_idx = 2
    self.o_idx = 4 if atom37_order else 3

    keys = jax.random.split(key, 12)

    enc_node_in = 21
    enc_edge_in = num_positional_embeddings + num_rbf * 25
    self.enc_edge_embedding = eqx.nn.Linear(enc_edge_in, edge_features, use_bias=False, key=keys[0])
    self.enc_norm_edges = eqx.nn.LayerNorm(edge_features)
    self.enc_node_embedding = eqx.nn.Linear(enc_node_in, node_features, use_bias=False, key=keys[1])
    self.enc_norm_nodes = eqx.nn.LayerNorm(node_features)

    dec_node_in = 14 * atom_context_num * num_rbf
    dec_edge_in = num_rbf * 14 * 14 + 42
    self.w_xy_project_down1 = eqx.nn.Linear(num_rbf + 120, num_rbf, key=keys[2])
    self.dec_edge_embedding1 = eqx.nn.Linear(
      dec_edge_in, edge_features, use_bias=False, key=keys[3],
    )
    self.dec_norm_edges1 = eqx.nn.LayerNorm(edge_features)
    self.dec_node_embedding1 = eqx.nn.Linear(
      dec_node_in, node_features, use_bias=False, key=keys[4],
    )
    self.dec_norm_nodes1 = eqx.nn.LayerNorm(node_features)

    self.node_project_down = eqx.nn.Linear(5 * num_rbf + 64 + 4, node_features, key=keys[5])
    self.norm_nodes = eqx.nn.LayerNorm(node_features)

    self.type_linear = eqx.nn.Linear(147, 64, key=keys[6])
    self.y_nodes = eqx.nn.Linear(147, node_features, use_bias=False, key=keys[7])
    self.y_edges = eqx.nn.Linear(num_rbf, node_features, use_bias=False, key=keys[8])
    self.norm_y_edges = eqx.nn.LayerNorm(node_features)
    self.norm_y_nodes = eqx.nn.LayerNorm(node_features)

    self.positional_embeddings = PositionalEncodings(num_positional_embeddings, key=keys[9])

  def _dist(self, x: jnp.ndarray, mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    dist_sq = jnp.sum(jnp.square(x[:, None, :] - x[None, :, :]), axis=-1)
    mask_2d = mask[:, None] * mask[None, :]
    dist_sq = jnp.where(mask_2d > 0, dist_sq, 1e8)  # Use larger value for inf

    # Python min, not jnp.minimum: both operands are static (``top_k`` is a
    # static eqx field, ``x.shape[0]`` an int), and the slice width inside
    # ``features.top_k`` has to be a Python int rather than a staged array.
    k = min(self.top_k, x.shape[0])
    # features.top_k, not jax.lax.top_k -- the latter lowers to a
    # stablehlo.composite IREE marks illegal, and leaves tie order to the
    # backend's sort stability. See that function's docstring.
    dist, idx = top_k(-dist_sq, k)
    return -dist, idx

  def _make_angle_features(
    self,
    atom_n: jnp.ndarray,
    atom_ca: jnp.ndarray,
    atom_c: jnp.ndarray,
    ligand_atom_coords: jnp.ndarray,
  ) -> jnp.ndarray:
    """Compute angle features matching PyTorch sc_utils implementation.

    Uses Gram-Schmidt orthonormalization to create local coordinate frame,
    then computes cylindrical coordinate-like features.

    Returns: [L, M, 4] features (f1, f2, f3, f4)
    """

    def _get_angles(n_pos, ca_pos, c_pos, y_all):
      # Build local coordinate frame via Gram-Schmidt
      v1 = n_pos - ca_pos  # N-CA vector
      v2 = c_pos - ca_pos  # C-CA vector

      e1 = v1 / (jnp.linalg.norm(v1) + 1e-8)
      e1_v2_dot = jnp.sum(e1 * v2)
      u2 = v2 - e1 * e1_v2_dot
      e2 = u2 / (jnp.linalg.norm(u2) + 1e-8)
      e3 = jnp.cross(e1, e2)

      # Build rotation matrix [3, 3]
      R_residue = jnp.stack([e1, e2, e3], axis=-1)  # [3, 3]

      # Transform Y to local coords
      y_rel = y_all - ca_pos[None, :]  # [M, 3]
      local_vectors = jnp.einsum("qp,mq->mp", R_residue, y_rel)  # [M, 3]

      # Compute cylindrical features
      rxy = jnp.sqrt(local_vectors[:, 0] ** 2 + local_vectors[:, 1] ** 2 + 1e-8)
      f1 = local_vectors[:, 0] / rxy
      f2 = local_vectors[:, 1] / rxy
      rxyz = jnp.linalg.norm(local_vectors, axis=-1) + 1e-8
      f3 = rxy / rxyz
      f4 = local_vectors[:, 2] / rxyz

      return jnp.stack([f1, f2, f3, f4], axis=-1)  # [M, 4]

    return jax.vmap(_get_angles)(atom_n, atom_ca, atom_c, ligand_atom_coords)

  def features_encode(self, prng_key: PRNGKeyArray, bundle: PackerBundle) -> tuple:
    s = bundle.sequence
    x = bundle.backbone_coords
    y = bundle.ligand_coords
    y_m = bundle.ligand_mask
    y_t = bundle.ligand_atom_types
    mask = bundle.mask
    r_idx = bundle.residue_index
    chain_labels = bundle.chain_labels
    backbone_noise = bundle.backbone_noise

    if backbone_noise > 0:
      x, _ = apply_noise_to_coordinates(prng_key, x, backbone_noise=backbone_noise)

    ca = x[:, self.ca_idx, :]
    n = x[:, self.n_idx, :]
    c = x[:, self.c_idx, :]
    o = x[:, self.o_idx, :]

    b = ca - n
    c_vec = c - ca
    a = jnp.cross(b, c_vec)
    cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca

    _, E_idx = self._dist(ca, mask)

    backbone_coords = [n, ca, c, o, cb]
    rbf_all = []
    for atom1 in backbone_coords:
      for atom2 in backbone_coords:
        dist = jnp.sqrt(jnp.sum(jnp.square(atom1[:, None, :] - atom2[E_idx]), axis=-1) + 1e-6)
        rbf_all.append(compute_rbf(dist, self.lower_bound, self.upper_bound, self.num_rbf))

    rbf_all = jnp.concatenate(rbf_all, axis=-1)  # [L, K, 25*num_rbf]

    offset = r_idx[:, None] - r_idx[None, :]
    offset_gathered = gather_edges(offset[..., None], E_idx)[..., 0]

    d_chains = (chain_labels[:, None] == chain_labels[None, :]).astype(jnp.int32)
    e_chains = gather_edges(d_chains[..., None], E_idx)[..., 0]

    e_positional = self.positional_embeddings(offset_gathered, e_chains)
    e = jnp.concatenate([e_positional, rbf_all], axis=-1)
    e = jax.vmap(jax.vmap(self.enc_edge_embedding))(e)
    e_shape = e.shape
    e = jax.vmap(self.enc_norm_edges)(e.reshape(-1, e_shape[-1])).reshape(e_shape)

    y_t = y_t.astype(jnp.int32)
    y_t_g = PERIODIC_TABLE_FEATURES[1][y_t]
    y_t_p = PERIODIC_TABLE_FEATURES[2][y_t]

    y_t_g_1hot = jax.nn.one_hot(y_t_g, 19)
    y_t_p_1hot = jax.nn.one_hot(y_t_p, 8)
    y_t_1hot_base = jax.nn.one_hot(y_t, 120)

    y_t_1hot_all = jnp.concatenate([y_t_1hot_base, y_t_g_1hot, y_t_p_1hot], axis=-1)
    y_t_1hot = jax.vmap(jax.vmap(self.type_linear))(y_t_1hot_all)

    def _get_y_dist_rbf(coords, y_coords):
      d_y = jnp.sqrt(jnp.sum(jnp.square(coords[:, None, :] - y_coords), axis=-1) + 1e-6)
      return compute_rbf(d_y, self.lower_bound, self.upper_bound, self.num_rbf)

    d_n_y = _get_y_dist_rbf(n, y)
    d_ca_y = _get_y_dist_rbf(ca, y)
    d_c_y = _get_y_dist_rbf(c, y)
    d_o_y = _get_y_dist_rbf(o, y)
    d_cb_y = _get_y_dist_rbf(cb, y)

    f_angles = self._make_angle_features(n, ca, c, y)

    d_all = jnp.concatenate([d_n_y, d_ca_y, d_c_y, d_o_y, d_cb_y, y_t_1hot, f_angles], axis=-1)
    e_context = jax.vmap(jax.vmap(self.node_project_down))(d_all)
    e_shape = e_context.shape
    e_context = jax.vmap(self.norm_nodes)(e_context.reshape(-1, e_shape[-1])).reshape(e_shape)

    # y is [L, M, 3], y_diff should be [L, M, M, 3]
    y_diff = y[:, :, None, :] - y[:, None, :, :]  # [L, M, M, 3]
    y_dist = jnp.sqrt(jnp.sum(jnp.square(y_diff), axis=-1) + 1e-6)  # [L, M, M]
    y_edges_rbf = jax.vmap(
      lambda d: compute_rbf(d, self.lower_bound, self.upper_bound, self.num_rbf),
    )(y_dist)  # [L, M, M, num_rbf]
    y_edges = jax.vmap(jax.vmap(jax.vmap(self.y_edges)))(y_edges_rbf)  # [L, M, M, hidden_dim]
    y_nodes = jax.vmap(jax.vmap(self.y_nodes))(y_t_1hot_all)

    y_edges_shape = y_edges.shape
    y_edges = jax.vmap(self.norm_y_edges)(y_edges.reshape(-1, y_edges_shape[-1])).reshape(
      y_edges_shape,
    )
    y_nodes_shape = y_nodes.shape
    y_nodes = jax.vmap(self.norm_y_nodes)(y_nodes.reshape(-1, y_nodes_shape[-1])).reshape(
      y_nodes_shape,
    )

    v = jax.nn.one_hot(s, 21)
    v = jax.vmap(self.enc_node_embedding)(v)
    v_shape = v.shape
    v = jax.vmap(self.enc_norm_nodes)(v.reshape(-1, v_shape[-1])).reshape(v_shape)

    return v, e, E_idx, y_nodes, y_edges, e_context, y_m

  def features_decode(self, bundle: PackerBundle, E_idx: jax.Array) -> tuple:
    s = bundle.sequence
    x = bundle.backbone_coords
    x_m = bundle.backbone_mask
    mask = bundle.mask

    y = bundle.ligand_coords[:, : self.atom_context_num, :]
    y_m = bundle.ligand_mask[:, : self.atom_context_num]
    y_t = bundle.ligand_atom_types[:, : self.atom_context_num]

    x_m = x_m * mask[:, None]

    rbf_sidechain = []
    x_m_gathered = gather_nodes(x_m, E_idx)  # [L, K, 14]

    # Match PyTorch _get_rbf: compute full distance matrix then gather
    for i in range(14):
      for j in range(14):
        # Compute full L×L distance matrix for atoms i and j
        atom_i = x[:, i, :]  # [L, 3]
        atom_j = x[:, j, :]  # [L, 3]
        d_full = jnp.sqrt(
          jnp.sum(jnp.square(atom_i[:, None, :] - atom_j[None, :, :]), axis=-1) + 1e-6,
        )  # [L, L]

        # Gather neighbor distances using E_idx
        d_neighbors = gather_edges(d_full[..., None], E_idx)[..., 0]  # [L, K]

        rbf_features = compute_rbf(
          d_neighbors, self.lower_bound, self.upper_bound, self.num_rbf,
        )  # [L, K, num_rbf]
        rbf_features = rbf_features * x_m[:, i, None, None] * x_m_gathered[:, :, j, None]
        rbf_sidechain.append(rbf_features)

    rbf_sidechain = jnp.concatenate(rbf_sidechain, axis=-1)  # [L, K, 14*14*num_rbf]

    # x is [L, 14, 3], y is [L, M, 3]
    # Need d_xy of shape [L, 14, M]
    d_xy = jnp.sqrt(
      jnp.sum(jnp.square(x[:, :, None, :] - y[:, None, :, :]), axis=-1) + 1e-6,
    )  # [L, 14, M]
    xy_features = compute_rbf(
      d_xy, self.lower_bound, self.upper_bound, self.num_rbf,
    )  # [L, 14, M, num_rbf]
    xy_features = xy_features * x_m[:, :, None, None] * y_m[:, None, :, None]

    y_t_1hot = jax.nn.one_hot(y_t.astype(jnp.int32), 120)  # [L, M, 120]
    # xy_features: [L, 14, M, num_rbf]
    # y_t_1hot: [L, M, 120]
    # Need y_t_1hot_expanded: [L, 14, M, 120]
    y_t_1hot_expanded = jnp.broadcast_to(
      y_t_1hot[:, None, :, :],
      (xy_features.shape[0], xy_features.shape[1], xy_features.shape[2], 120),
    )
    xy_y_t = jnp.concatenate([xy_features, y_t_1hot_expanded], axis=-1)
    xy_y_t = jax.vmap(jax.vmap(jax.vmap(self.w_xy_project_down1)))(xy_y_t)
    xy_features_flat = xy_y_t.reshape((xy_y_t.shape[0], -1))

    v = jax.vmap(self.dec_node_embedding1)(xy_features_flat)
    v_shape = v.shape
    v = jax.vmap(self.dec_norm_nodes1)(v.reshape(-1, v_shape[-1])).reshape(v_shape)

    s_1h = jax.nn.one_hot(s, 21)
    s_1h_gathered = gather_nodes(s_1h, E_idx)
    s_features = jnp.concatenate(
      [jnp.broadcast_to(s_1h[:, None, :], (s_1h.shape[0], E_idx.shape[1], 21)), s_1h_gathered],
      axis=-1,
    )

    f = jnp.concatenate([rbf_sidechain, s_features], axis=-1)
    f = jax.vmap(jax.vmap(self.dec_edge_embedding1))(f)
    f_shape = f.shape
    f = jax.vmap(self.dec_norm_edges1)(f.reshape(-1, f_shape[-1])).reshape(f_shape)

    return v, f


class Packer(eqx.Module):
  """Side-chain torsion angle packer from LigandMPNN.

  Predicts von Mises–Fisher (VMF) mixture parameters for chi1–chi4 torsion angles
  given backbone and ligand context. Uses dual-encoder cross-attention architecture
  matching the reference LigandMPNN implementation.

  Parameters
  ----------
  features : PackerProteinFeatures
      Feature extraction module.
  w_e : eqx.nn.Linear
      Edge feature projection.
  w_v : eqx.nn.Linear
      Node feature projection.
  w_f : eqx.nn.Linear
      Sidechain edge feature projection.
  w_v_sc : eqx.nn.Linear
      Sidechain node feature projection.
  linear_down : eqx.nn.Linear
      Dimension reduction for sidechain context.
  w_torsions : eqx.nn.Linear
      Output projection to VMF mixture parameters.
  encoder_layers : tuple[EncoderLayer, ...]
      Backbone encoder layers.
  context_encoder_layers : tuple[DecoderLayer, ...]
      Protein cross-attention layers.
  y_context_encoder_layers : tuple[DecoderLayerJ, ...]
      Ligand context encoder layers.
  decoder_layers : tuple[DecoderLayer, ...]
      Sidechain decoder layers.
  w_c : eqx.nn.Linear
      Protein context projection.
  w_e_context : eqx.nn.Linear
      Edge context projection.
  w_nodes_y : eqx.nn.Linear
      Ligand node projection.
  w_edges_y : eqx.nn.Linear
      Ligand edge projection.
  v_c : eqx.nn.Linear
      Protein-ligand fusion layer.
  v_c_norm : eqx.nn.LayerNorm
      Layer normalization after fusion.
  h_v_c_dropout : Dropout
      Dropout layer.
  num_mix : int
      Number of VMF mixture components. Static (not a JAX array).
  hidden_dim : int
      Hidden layer dimension. Static (not a JAX array).

  References
  ----------
  .. [LigandMPNN] Dauparas, J., et al. "Atomic context-conditioned protein
     sequence design using LigandMPNN." *Nature Methods* 22(4):717-723 (2025).
     https://doi.org/10.1038/s41592-025-02626-1

  .. [LigandMPNN-code] Dauparas, J. LigandMPNN source code (commit 3870631).
     https://github.com/dauparas/LigandMPNN

  """

  features: PackerProteinFeatures
  w_e: eqx.nn.Linear
  w_v: eqx.nn.Linear
  w_f: eqx.nn.Linear
  w_v_sc: eqx.nn.Linear
  linear_down: eqx.nn.Linear
  w_torsions: eqx.nn.Linear

  encoder_layers: tuple[EncoderLayer, ...]
  context_encoder_layers: tuple[DecoderLayer, ...]
  y_context_encoder_layers: tuple[DecoderLayerJ, ...]
  decoder_layers: tuple[DecoderLayer, ...]

  w_c: eqx.nn.Linear
  w_e_context: eqx.nn.Linear
  w_nodes_y: eqx.nn.Linear
  w_edges_y: eqx.nn.Linear
  v_c: eqx.nn.Linear
  v_c_norm: eqx.nn.LayerNorm

  h_v_c_dropout: Dropout

  num_mix: int = eqx.field(static=True)
  hidden_dim: int = eqx.field(static=True)

  def __init__(
    self,
    edge_features: int = 128,
    node_features: int = 128,
    num_positional_embeddings: int = 16,
    num_rbf: int = 16,
    top_k: int = 30,
    atom_context_num: int = 16,
    hidden_dim: int = 128,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 3,
    dropout: float = 0.1,
    num_mix: int = 3,
    *,
    atom37_order: bool = False,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the Packer model.

    Parameters
    ----------
    edge_features : int
        Edge feature dimension. Default: 128.
    node_features : int
        Node feature dimension. Default: 128.
    num_positional_embeddings : int
        Positional embedding dimension. Default: 16.
    num_rbf : int
        Number of RBF basis functions. Default: 16.
    top_k : int
        Number of top-k neighbors. Default: 30.
    atom_context_num : int
        Number of ligand atom neighbors. Default: 16.
    hidden_dim : int
        Hidden layer dimension. Default: 128.
    num_encoder_layers : int
        Number of encoder layers. Default: 3.
    num_decoder_layers : int
        Number of decoder layers. Default: 3.
    dropout : float
        Dropout rate. Default: 0.1.
    num_mix : int
        Number of VMF mixture components. Default: 3.
    atom37_order : bool
        Whether to use ATOM37 atom ordering. Default: False.
    key : PRNGKeyArray
        PRNG key for weight initialization.

    """
    self.num_mix = num_mix
    self.hidden_dim = hidden_dim

    keys = jax.random.split(key, 20)
    self.features = PackerProteinFeatures(
      edge_features=edge_features,
      node_features=node_features,
      num_positional_embeddings=num_positional_embeddings,
      num_rbf=num_rbf,
      top_k=top_k,
      atom37_order=atom37_order,
      atom_context_num=atom_context_num,
      key=keys[0],
    )

    self.w_e = eqx.nn.Linear(edge_features, hidden_dim, key=keys[1])
    self.w_v = eqx.nn.Linear(node_features, hidden_dim, key=keys[2])
    self.w_f = eqx.nn.Linear(edge_features, hidden_dim, key=keys[3])
    self.w_v_sc = eqx.nn.Linear(node_features, hidden_dim, key=keys[4])
    self.linear_down = eqx.nn.Linear(2 * hidden_dim, hidden_dim, key=keys[5])
    self.w_torsions = eqx.nn.Linear(hidden_dim, 4 * 3 * num_mix, key=keys[6])

    self.encoder_layers = tuple(
      EncoderLayer(hidden_dim, hidden_dim, hidden_dim, dropout_rate=dropout, key=k)
      for k in jax.random.split(keys[7], num_encoder_layers)
    )

    self.w_c = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[8])
    self.w_e_context = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[9])
    self.w_nodes_y = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[10])
    self.w_edges_y = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[11])

    self.context_encoder_layers = tuple(
      DecoderLayer(hidden_dim, hidden_dim * 2, hidden_dim, dropout_rate=dropout, key=k)
      for k in jax.random.split(keys[12], 2)
    )

    self.v_c = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=keys[13])
    self.v_c_norm = eqx.nn.LayerNorm(hidden_dim)
    self.h_v_c_dropout = Dropout(dropout)

    self.y_context_encoder_layers = tuple(
      DecoderLayerJ(hidden_dim, hidden_dim, dropout=dropout, key=k)
      for k in jax.random.split(keys[14], 2)
    )

    self.decoder_layers = tuple(
      DecoderLayer(hidden_dim, hidden_dim * 3, hidden_dim, dropout_rate=dropout, key=k)
      for k in jax.random.split(keys[15], num_decoder_layers)
    )

  def encode(
    self, bundle: PackerBundle, *, key: PRNGKeyArray | None = None, inference: bool = False,
  ) -> tuple:
    """Encode protein structure with ligand context via message-passing layers.

    Applies featurization, then stacked encoder layers over protein graph with
    ligand atom context integration, producing encoded node and edge features.

    Parameters
    ----------
    bundle : PackerBundle
        Packed input bundle (sequence, coordinates, ligand, masks).
    key : PRNGKeyArray | None
        PRNG key for encoder stochasticity (optional).
    inference : bool
        If True, disable dropout. Default: False.

    Returns
    -------
    tuple
        Tuple of (node_features, edge_features, neighbor_indices):
        - node_features: Shape ``(L, D)`` — encoded residue features.
        - edge_features: Shape ``(L, K, D_edge)`` — encoded neighbor edge context.
        - neighbor_indices: Shape ``(L, K)`` — neighbor indices for gather.

    References
    ----------
    .. [LigandMPNN-code] Dauparas, J. LigandMPNN source code (commit 3870631).
       https://github.com/dauparas/LigandMPNN

    """
    mask = bundle.mask
    v, e, E_idx, y_nodes, y_edges, e_context, y_m = self.features.features_encode(key, bundle)

    if key is None:
      inference = True
    keys = jax.random.split(key, 10) if key is not None else [None] * 10

    h_e_context = jax.vmap(jax.vmap(self.w_e_context))(e_context)
    h_v = jax.vmap(self.w_v)(v)
    h_e = jax.vmap(jax.vmap(self.w_e))(e)

    mask_attend = gather_nodes(mask[:, None], E_idx)[..., 0]
    mask_attend = mask[:, None] * mask_attend

    for i, layer in enumerate(self.encoder_layers):
      h_v, h_e = layer(h_v, h_e, E_idx, mask, mask_attend, inference=inference, key=keys[i])

    h_v_c = jax.vmap(self.w_c)(h_v)
    y_m_edges = y_m[:, :, None] * y_m[:, None, :]
    y_nodes = jax.vmap(jax.vmap(self.w_nodes_y))(y_nodes)
    y_edges = jax.vmap(jax.vmap(jax.vmap(self.w_edges_y)))(y_edges)

    for i in range(2):
      y_nodes = self.y_context_encoder_layers[i](
        y_nodes, y_edges, y_m, y_m_edges, inference=inference, key=keys[5 + i],
      )
      h_e_context_cat = jnp.concatenate([h_e_context, y_nodes], axis=-1)
      h_v_c = self.context_encoder_layers[i](
        h_v_c, h_e_context_cat, mask, inference=inference, key=keys[7 + i],
      )

    h_v_c = jax.vmap(self.v_c)(h_v_c)
    h_v = h_v + jax.vmap(self.v_c_norm)(self.h_v_c_dropout(h_v_c, key=keys[9], inference=inference))

    return h_v, h_e, E_idx

  def decode(
    self,
    bundle: PackerBundle,
    h_v: jax.Array,
    h_e: jax.Array,
    E_idx: jax.Array,
    *,
    key: PRNGKeyArray | None = None,
    inference: bool = False,
  ) -> PackerResult:
    """Predict VMF mixture parameters for side-chain torsion angles.

    Parameters
    ----------
    bundle : PackerBundle
        Packed input bundle (sequence, coordinates, ligand, masks).
    h_v : jax.Array
        Encoded node features from encode(). Shape ``(L, hidden_dim)``.
    h_e : jax.Array
        Encoded edge features from encode(). Shape ``(L, K, hidden_dim)``.
    E_idx : jax.Array
        Neighbor indices from encode(). Shape ``(L, K)``.
    key : PRNGKeyArray | None
        PRNG key for dropout (optional).
    inference : bool
        If True, disable dropout. Default: False.

    Returns
    -------
    PackerResult
        VMF mixture parameters: ``mean``, ``concentration``, ``mix_logits``.
        - ``mean``: Shape ``(L, 4, num_mix)`` — VMF means for chi1–chi4.
        - ``concentration``: Shape ``(L, 4, num_mix)`` — VMF concentration parameters.
        - ``mix_logits``: Shape ``(L, 4, num_mix)`` — Mixture logits.

    """
    mask = bundle.mask
    v, f = self.features.features_decode(bundle, E_idx)

    h_f = jax.vmap(jax.vmap(self.w_f))(f)
    h_ef = jnp.concatenate([h_e, h_f], axis=-1)

    h_v_sc = jax.vmap(self.w_v_sc)(v)
    h_v_combined = jnp.concatenate([h_v, h_v_sc], axis=-1)
    h_v = jax.vmap(self.linear_down)(h_v_combined)

    if key is None:
      inference = True
    keys = (
      jax.random.split(key, len(self.decoder_layers))
      if key is not None
      else [None] * len(self.decoder_layers)
    )

    for i, layer in enumerate(self.decoder_layers):
      h_ev = concatenate_neighbor_nodes(h_v, h_ef, E_idx)
      h_v = layer(h_v, h_ev, mask, inference=inference, key=keys[i])

    torsions = jax.vmap(self.w_torsions)(h_v)
    torsions = torsions.reshape((h_v.shape[0], 4, self.num_mix, 3))

    mean = torsions[..., 0]
    concentration = 0.1 + jax.nn.softplus(torsions[..., 1])
    mix_logits = torsions[..., 2]

    return PackerResult(mean=mean, concentration=concentration, mix_logits=mix_logits)

  def __call__(
    self,
    prng_key: PRNGKeyArray,
    bundle: PackerBundle,
    config: InferenceConfig,
  ) -> PackerResult:
    """End-to-end packing: encode structure, then decode side-chain torsions.

    Parameters
    ----------
    prng_key : PRNGKeyArray
        PRNG key for encoder and decoder stochasticity.
    bundle : PackerBundle
        Packed input bundle (sequence, coordinates, ligand, masks).
    config : InferenceConfig
        Inference configuration (unused, accepted for API uniformity).

    Returns
    -------
    PackerResult
        VMF mixture parameters for side-chain torsion angles:
        - ``mean``: Shape ``(L, 4, num_mix)`` — VMF means for chi1–chi4.
        - ``concentration``: Shape ``(L, 4, num_mix)`` — VMF concentration.
        - ``mix_logits``: Shape ``(L, 4, num_mix)`` — Mixture logits.

    References
    ----------
    .. [LigandMPNN-code] Dauparas, J. LigandMPNN source code (commit 3870631).
       https://github.com/dauparas/LigandMPNN

    """
    keys = jax.random.split(prng_key, 2)
    h_v, h_e, E_idx = self.encode(bundle, key=keys[0])
    return self.decode(bundle, h_v, h_e, E_idx, key=keys[1])
