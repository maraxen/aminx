"""FAMPNN coordinate frame transforms, Atom37/Atom14 conversions, and pairwise RBF features.

Ports ``atom37_to_atom14``, ``atom14_to_atom37``, ``backbone_coords_to_frames``,
``get_rotation_frames``, ``transform_sidechain_frame``, ``ProteinFeatures``, and
``SidechainProteinFeatures`` from ``richardshuai/fampnn`` (``fampnn/model/features.py``).
"""

from __future__ import annotations

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from aminx.families.fampnn.constants import (
  RESTYPE_ATOM14_MASK,
  RESTYPE_ATOM14_TO_ATOM37,
  RESTYPE_ATOM37_MASK,
  RESTYPE_ATOM37_TO_ATOM14,
)

# 5 backbone atom indices in Atom37: N=0, CA=1, C=2, CB=3, O=4
_ATOM37_BB5_IDXS: jax.Array = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)


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


def make_virtual_cb_fampnn(
  n_xyz: Float[Array, "L 3"],
  ca_xyz: Float[Array, "L 3"],
  c_xyz: Float[Array, "L 3"],
) -> Float[Array, "L 3"]:
  """Compute tetrahedral virtual C-beta coordinates from backbone N, CA, C.

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
    Virtual C-beta coordinates ``(L, 3)``.
  """
  b = ca_xyz - n_xyz
  c = c_xyz - ca_xyz
  a = jnp.cross(b, c)
  return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_xyz


def atom37_to_atom14(
  atom37_xyz: Float[Array, "L 37 3"],
  atom37_mask: Float[Array, "L 37"],
  aatype: Int[Array, " L"],
) -> tuple[Float[Array, "L 14 3"], Float[Array, "L 14"]]:
  """Convert Atom37 coordinates and masks to compact Atom14 representation.

  Parameters
  ----------
  atom37_xyz : Float[Array, "L 37 3"]
    Atom37 coordinates ``(L, 37, 3)``.
  atom37_mask : Float[Array, "L 37"]
    Atom37 presence mask ``(L, 37)``.
  aatype : Int[Array, " L"]
    Residue type indices in FAMPNN alphabet ``0..20``.

  Returns
  -------
  tuple[Float[Array, "L 14 3"], Float[Array, "L 14"]]
    Tuple ``(atom14_xyz, atom14_mask)``.
  """
  aa_clamped = jnp.clip(aatype, 0, 20)
  gather_idx = RESTYPE_ATOM14_TO_ATOM37[aa_clamped]
  std_mask14 = RESTYPE_ATOM14_MASK[aa_clamped].astype(atom37_mask.dtype)

  atom14_xyz = jnp.take_along_axis(atom37_xyz, gather_idx[..., None], axis=1)
  atom14_mask = jnp.take_along_axis(atom37_mask, gather_idx, axis=1) * std_mask14
  atom14_xyz = atom14_xyz * atom14_mask[..., None]
  return atom14_xyz, atom14_mask


def atom14_to_atom37(
  atom14_xyz: Float[Array, "L 14 3"],
  atom14_mask: Float[Array, "L 14"],
  aatype: Int[Array, " L"],
) -> tuple[Float[Array, "L 37 3"], Float[Array, "L 37"]]:
  """Convert Atom14 coordinates and masks back to Atom37 representation.

  Parameters
  ----------
  atom14_xyz : Float[Array, "L 14 3"]
    Atom14 coordinates ``(L, 14, 3)``.
  atom14_mask : Float[Array, "L 14"]
    Atom14 presence mask ``(L, 14)``.
  aatype : Int[Array, " L"]
    Residue type indices in FAMPNN alphabet ``0..20``.

  Returns
  -------
  tuple[Float[Array, "L 37 3"], Float[Array, "L 37"]]
    Tuple ``(atom37_xyz, atom37_mask)``.
  """
  aa_clamped = jnp.clip(aatype, 0, 20)
  gather_idx = RESTYPE_ATOM37_TO_ATOM14[aa_clamped]
  std_mask37 = RESTYPE_ATOM37_MASK[aa_clamped].astype(atom14_mask.dtype)

  atom37_xyz = jnp.take_along_axis(atom14_xyz, gather_idx[..., None], axis=1)
  atom37_mask = jnp.take_along_axis(atom14_mask, gather_idx, axis=1) * std_mask37
  atom37_xyz = atom37_xyz * atom37_mask[..., None]
  return atom37_xyz, atom37_mask


def get_rotation_frames(
  n_xyz: Float[Array, "L 3"],
  ca_xyz: Float[Array, "L 3"],
  c_xyz: Float[Array, "L 3"],
  eps: float = 1e-8,
) -> Float[Array, "L 3 3"]:
  """Compute Gram-Schmidt local rotation frames ``R in SO(3)`` from N, CA, C.

  Matches upstream ``get_rotation_frames`` (``e1`` along ``C - CA``, ``e2``
  orthogonalized from ``N - CA``, ``e3 = e1 x e2``; columns stacked as ``[e1, e2, e3]``).

  Parameters
  ----------
  n_xyz : Float[Array, "L 3"]
    Backbone N coordinates ``(L, 3)``.
  ca_xyz : Float[Array, "L 3"]
    Backbone CA coordinates ``(L, 3)``.
  c_xyz : Float[Array, "L 3"]
    Backbone C coordinates ``(L, 3)``.
  eps : float, default=1e-8
    Small constant preventing division by zero.

  Returns
  -------
  Float[Array, "L 3 3"]
    Local rotation matrices ``(L, 3, 3)`` with columns ``[e1, e2, e3]``.
  """
  v1 = c_xyz - ca_xyz
  v2 = n_xyz - ca_xyz
  e1 = v1 / jnp.sqrt(jnp.sum(v1 * v1, axis=-1, keepdims=True) + eps)
  u2 = v2 - jnp.sum(e1 * v2, axis=-1, keepdims=True) * e1
  e2 = u2 / jnp.sqrt(jnp.sum(u2 * u2, axis=-1, keepdims=True) + eps)
  e3 = jnp.cross(e1, e2)
  return jnp.stack([e1, e2, e3], axis=-1)


def backbone_coords_to_frames(
  atom37_xyz: Float[Array, "L 37 3"],
) -> tuple[Float[Array, "L 3 3"], Float[Array, "L 3"]]:
  """Extract local rotation matrices ``R`` and CA translations ``t`` from Atom37 coordinates.

  Parameters
  ----------
  atom37_xyz : Float[Array, "L 37 3"]
    Atom37 coordinates ``(L, 37, 3)`` where ``0=N, 1=CA, 2=C``.

  Returns
  -------
  tuple[Float[Array, "L 3 3"], Float[Array, "L 3"]]
    Tuple ``(rotations, translations)`` of shapes ``(L, 3, 3)`` and ``(L, 3)``.
  """
  n_xyz = atom37_xyz[:, 0, :]
  ca_xyz = atom37_xyz[:, 1, :]
  c_xyz = atom37_xyz[:, 2, :]
  rot = get_rotation_frames(n_xyz, ca_xyz, c_xyz)
  return rot, ca_xyz


def global_to_local_atom37(
  atom37_xyz: Float[Array, "L 37 3"],
  atom37_mask: Float[Array, "L 37"],
  *,
  scale: float = 1.0,
) -> Float[Array, "L 37 3"]:
  """Transform global Atom37 coordinates into local CA-centered backbone frames.

  Parameters
  ----------
  atom37_xyz : Float[Array, "L 37 3"]
    Global Atom37 coordinates ``(L, 37, 3)``.
  atom37_mask : Float[Array, "L 37"]
    Atom37 validity mask ``(L, 37)``.
  scale : float, default=1.0
    Coordinate divisor (e.g., ``sigma_data = 1.9`` for sidechain diffusion).

  Returns
  -------
  Float[Array, "L 37 3"]
    Local frame coordinates ``((X - CA) @ R / scale) * mask``.
  """
  rot, ca_xyz = backbone_coords_to_frames(atom37_xyz)
  shifted = atom37_xyz - ca_xyz[:, None, :]
  local_xyz = jnp.einsum("lai,lij->laj", shifted, rot) / scale
  return local_xyz * atom37_mask[..., None]


def local_to_global_atom37(
  local_atom37_xyz: Float[Array, "L 37 3"],
  backbone_atom37_xyz: Float[Array, "L 37 3"],
  atom37_mask: Float[Array, "L 37"],
  *,
  scale: float = 1.0,
) -> Float[Array, "L 37 3"]:
  """Transform local CA-centered Atom37 coordinates back into global coordinates.

  Parameters
  ----------
  local_atom37_xyz : Float[Array, "L 37 3"]
    Local frame Atom37 coordinates ``(L, 37, 3)``.
  backbone_atom37_xyz : Float[Array, "L 37 3"]
    Reference Atom37 coordinates carrying global backbone N, CA, C, O.
  atom37_mask : Float[Array, "L 37"]
    Atom37 validity mask ``(L, 37)``.
  scale : float, default=1.0
    Coordinate multiplier (e.g., ``sigma_data = 1.9`` for sidechain diffusion).

  Returns
  -------
  Float[Array, "L 37 3"]
    Global Atom37 coordinates ``(local * scale @ R^T + CA) * mask``.
  """
  rot, ca_xyz = backbone_coords_to_frames(backbone_atom37_xyz)
  global_xyz = jnp.einsum("laj,lij->lai", local_atom37_xyz * scale, rot) + ca_xyz[:, None, :]
  return global_xyz * atom37_mask[..., None]


def compute_fampnn_knn(
  ca_xyz: Float[Array, "L 3"],
  mask: Float[Array, " L"],
  top_k: int = 48,
  eps: float = 1e-6,
) -> tuple[Float[Array, "L K"], Int[Array, "L K"]]:
  """Compute pairwise CA distances and top-K nearest neighbors.

  Parameters
  ----------
  ca_xyz : Float[Array, "L 3"]
    Backbone CA coordinates ``(L, 3)``.
  mask : Float[Array, " L"]
    Valid residue mask ``(L,)``.
  top_k : int, default=48
    Number of nearest neighbors ``K``.
  eps : float, default=1e-6
    Small constant inside ``sqrt``.

  Returns
  -------
  tuple[Float[Array, "L K"], Int[Array, "L K"]]
    Tuple ``(d_neighbors, e_idx)``.
  """
  num_l = ca_xyz.shape[0]
  k_eff = min(top_k, num_l)
  mask_2d = mask[:, None] * mask[None, :]
  diff = ca_xyz[:, None, :] - ca_xyz[None, :, :]
  dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + eps) * mask_2d
  dist_adj = dist + (1.0 - mask_2d) * 10000.0
  neg_d, e_idx = jax.lax.top_k(-dist_adj, k_eff)
  return -neg_d, e_idx.astype(jnp.int32)


def compute_fampnn_rbf(
  dist: Float[Array, "..."],
  num_rbf: int = 16,
  d_min: float = 2.0,
  d_max: float = 22.0,
) -> Float[Array, "... R"]:
  """Expand distances into ``num_rbf`` Gaussian radial basis functions.

  Parameters
  ----------
  dist : Float[Array, "..."]
    Pairwise Euclidean distances.
  num_rbf : int, default=16
    Number of RBF centers.
  d_min : float, default=2.0
    Minimum RBF center in Angstroms.
  d_max : float, default=22.0
    Maximum RBF center in Angstroms.

  Returns
  -------
  Float[Array, "... R"]
    RBF expansion with trailing dimension ``num_rbf``.
  """
  d_mu = jnp.linspace(d_min, d_max, num_rbf, dtype=dist.dtype)
  d_sigma = (d_max - d_min) / num_rbf
  diff = (dist[..., None] - d_mu) / d_sigma
  return jnp.exp(-(diff * diff))


class FAMPNNPositionalEncodings(eqx.Module):
  """ProteinMPNN-style relative residue positional encodings (``66 -> 16``).

  Attributes
  ----------
  linear : eqx.nn.Linear
    Linear projection ``66 -> num_embeddings``.
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
    """Initialize ``FAMPNNPositionalEncodings``.

    Parameters
    ----------
    num_embeddings : int, default=16
      Output positional embedding dimension.
    max_relative_feature : int, default=32
      Maximum relative offset before clipping.
    key : jax.Array | None, optional
      PRNG key for weight initialization.

    Returns
    -------
    None
      Initializes instance state.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    self.linear = eqx.nn.Linear(2 * max_relative_feature + 2, num_embeddings, key=key)
    self.max_relative_feature = max_relative_feature

  def __call__(
    self,
    offset: Int[Array, "L K"],
    same_chain: Int[Array, "L K"],
  ) -> Float[Array, "L K P"]:
    """Embed relative residue offsets and same-chain mask.

    Parameters
    ----------
    offset : Int[Array, "L K"]
      Pairwise residue index differences ``R_i - R_j``.
    same_chain : Int[Array, "L K"]
      Binary indicator ``1`` when ``i`` and ``j`` belong to the same chain, else ``0``.

    Returns
    -------
    Float[Array, "L K P"]
      Positional embeddings of shape ``(L, K, num_embeddings)``.
    """
    dtype = self.linear.weight.dtype
    num_classes = 2 * self.max_relative_feature + 2
    na_bucket = 2 * self.max_relative_feature + 1
    d = jnp.clip(
      offset + self.max_relative_feature,
      0,
      2 * self.max_relative_feature,
    )
    d_final = jnp.where(same_chain > 0, d, na_bucket)
    oh = jax.nn.one_hot(d_final, num_classes, dtype=dtype)
    return _linear(self.linear, oh)


class FAMPNNProteinFeatures(eqx.Module):
  """Backbone 5-atom (N, CA, C, O, virtual CB) pairwise RBF featurizer for ``FAEncoder``.

  Projects ``16 (pos) + 25 * 16 (5x5 RBFs) = 416`` channels to ``128`` edge channels.

  Attributes
  ----------
  embeddings : FAMPNNPositionalEncodings
    Relative positional encoding module (``66 -> 16``).
  edge_embedding : eqx.nn.Linear
    Linear projection ``416 -> edge_features`` (``use_bias=False``).
  norm_edges : eqx.nn.LayerNorm
    LayerNorm on projected edge features.
  top_k : int
    Maximum neighbors ``K`` (default ``48``).
  num_rbf : int
    Number of RBF centers per atom pair (default ``16``).
  """

  embeddings: FAMPNNPositionalEncodings
  edge_embedding: eqx.nn.Linear
  norm_edges: eqx.nn.LayerNorm
  top_k: int = eqx.field(static=True, default=48)
  num_rbf: int = eqx.field(static=True, default=16)

  def __init__(
    self,
    edge_features: int = 128,
    num_positional_embeddings: int = 16,
    num_rbf: int = 16,
    top_k: int = 48,
    *,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``FAMPNNProteinFeatures``.

    Parameters
    ----------
    edge_features : int, default=128
      Output edge feature dimension.
    num_positional_embeddings : int, default=16
      Positional embedding dimension.
    num_rbf : int, default=16
      Number of RBF bins per atom pair.
    top_k : int, default=48
      Number of nearest neighbors ``K``.
    key : jax.Array | None, optional
      PRNG key for weight initialization.

    Returns
    -------
    None
      Initializes instance state.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key, 2)
    self.embeddings = FAMPNNPositionalEncodings(num_embeddings=num_positional_embeddings, key=k1)
    edge_in = num_positional_embeddings + num_rbf * 25
    self.edge_embedding = eqx.nn.Linear(edge_in, edge_features, use_bias=False, key=k2)
    self.norm_edges = eqx.nn.LayerNorm(edge_features)
    self.top_k = top_k
    self.num_rbf = num_rbf

  def __call__(
    self,
    atom37_xyz: Float[Array, "L 37 3"],
    mask: Float[Array, " L"],
    residue_idx: Int[Array, " L"],
    chain_idx: Int[Array, " L"],
  ) -> tuple[Float[Array, "L K E"], Int[Array, "L K"]]:
    """Compute initial edge embeddings ``E`` and neighbor indices ``E_idx``.

    Parameters
    ----------
    atom37_xyz : Float[Array, "L 37 3"]
      Atom37 coordinates ``(L, 37, 3)`` where ``0=N, 1=CA, 2=C, 3=CB, 4=O``.
    mask : Float[Array, " L"]
      Valid residue mask ``(L,)``.
    residue_idx : Int[Array, " L"]
      Residue sequence numbers ``(L,)``.
    chain_idx : Int[Array, " L"]
      Chain IDs ``(L,)``.

    Returns
    -------
    tuple[Float[Array, "L K E"], Int[Array, "L K"]]
      Tuple ``(E, E_idx)`` of projected edge features and neighbor indices.
    """
    dtype = self.edge_embedding.weight.dtype
    xyz = jnp.nan_to_num(atom37_xyz.astype(dtype), nan=0.0)
    n_xyz = xyz[:, 0, :]
    ca_xyz = xyz[:, 1, :]
    c_xyz = xyz[:, 2, :]
    o_xyz = xyz[:, 4, :]
    cb_xyz = make_virtual_cb_fampnn(n_xyz, ca_xyz, c_xyz)

    _, e_idx = compute_fampnn_knn(ca_xyz, mask.astype(dtype), top_k=self.top_k)
    num_l, num_k = e_idx.shape

    # 5 backbone atoms in upstream order: (CA, N, C, O, CB)
    bb5 = jnp.stack([ca_xyz, n_xyz, c_xyz, o_xyz, cb_xyz], axis=1)
    diff = bb5[:, None, :, None, :] - bb5[e_idx][:, :, None, :, :]
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-6)
    rbf = compute_fampnn_rbf(dist, num_rbf=self.num_rbf).reshape(num_l, num_k, 25 * self.num_rbf)

    offset = residue_idx[:, None] - residue_idx[e_idx]
    same_chain = (chain_idx[:, None] == chain_idx[e_idx]).astype(jnp.int32)
    e_pos = self.embeddings(offset, same_chain)

    raw_e = jnp.concatenate([e_pos, rbf], axis=-1)
    edges = _layer_norm(self.norm_edges, _linear(self.edge_embedding, raw_e))
    return cast("Float[Array, 'L K E']", edges), e_idx


class SidechainProteinFeatures(eqx.Module):
  """Sidechain-aware pairwise RBF featurizer for FAMPNN's ``SidechainConfidenceModule`` (pSCE).

  Projects ``16 (pos) + (5*5 + 15) * 16 = 656`` channels to ``128`` edge channels.

  Attributes
  ----------
  embeddings : FAMPNNPositionalEncodings
    Relative positional encoding module (``66 -> 16``).
  edge_embedding : eqx.nn.Linear
    Linear projection ``656 -> edge_features`` (``use_bias=False``).
  norm_edges : eqx.nn.LayerNorm
    LayerNorm on projected edge features.
  top_k : int
    Maximum neighbors ``K`` (default ``48``).
  num_rbf : int
    Number of RBF centers (default ``16``).
  """

  embeddings: FAMPNNPositionalEncodings
  edge_embedding: eqx.nn.Linear
  norm_edges: eqx.nn.LayerNorm
  top_k: int = eqx.field(static=True, default=48)
  num_rbf: int = eqx.field(static=True, default=16)

  def __init__(
    self,
    edge_features: int = 128,
    num_positional_embeddings: int = 16,
    num_rbf: int = 16,
    top_k: int = 48,
    *,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``SidechainProteinFeatures``.

    Parameters
    ----------
    edge_features : int, default=128
      Output edge feature dimension.
    num_positional_embeddings : int, default=16
      Positional embedding dimension.
    num_rbf : int, default=16
      Number of RBF bins.
    top_k : int, default=48
      Number of nearest neighbors ``K``.
    key : jax.Array | None, optional
      PRNG key for weight initialization.

    Returns
    -------
    None
      Initializes instance state.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key, 2)
    self.embeddings = FAMPNNPositionalEncodings(num_embeddings=num_positional_embeddings, key=k1)
    edge_in = num_positional_embeddings + num_rbf * (25 + 15)
    self.edge_embedding = eqx.nn.Linear(edge_in, edge_features, use_bias=False, key=k2)
    self.norm_edges = eqx.nn.LayerNorm(edge_features)
    self.top_k = top_k
    self.num_rbf = num_rbf

  def __call__(
    self,
    atom14_xyz: Float[Array, "L 14 3"],
    atom14_mask: Float[Array, "L 14"],
    mask: Float[Array, " L"],
    residue_idx: Int[Array, " L"],
    chain_idx: Int[Array, " L"],
  ) -> tuple[Float[Array, "L K E"], Int[Array, "L K"]]:
    """Compute sidechain-aware edge features ``E`` and neighbor indices ``E_idx``.

    Parameters
    ----------
    atom14_xyz : Float[Array, "L 14 3"]
      Atom14 coordinates ``(L, 14, 3)``.
    atom14_mask : Float[Array, "L 14"]
      Atom14 validity mask ``(L, 14)``.
    mask : Float[Array, " L"]
      Valid residue mask ``(L,)``.
    residue_idx : Int[Array, " L"]
      Residue sequence numbers ``(L,)``.
    chain_idx : Int[Array, " L"]
      Chain IDs ``(L,)``.

    Returns
    -------
    tuple[Float[Array, "L K E"], Int[Array, "L K"]]
      Tuple ``(E, E_idx)``.
    """
    dtype = self.edge_embedding.weight.dtype
    xyz = jnp.nan_to_num(atom14_xyz.astype(dtype), nan=0.0)
    m14 = atom14_mask.astype(dtype)

    n_xyz = xyz[:, 0, :]
    ca_xyz = xyz[:, 1, :]
    c_xyz = xyz[:, 2, :]
    o_xyz = xyz[:, 3, :]
    cb_xyz = make_virtual_cb_fampnn(n_xyz, ca_xyz, c_xyz)

    _, e_idx = compute_fampnn_knn(ca_xyz, mask.astype(dtype), top_k=self.top_k)
    num_l, num_k = e_idx.shape

    # 1. 5x5 backbone RBFs -> 400 channels
    bb5 = jnp.stack([ca_xyz, n_xyz, c_xyz, o_xyz, cb_xyz], axis=1)
    diff_bb = bb5[:, None, :, None, :] - bb5[e_idx][:, :, None, :, :]
    dist_bb = jnp.sqrt(jnp.sum(diff_bb * diff_bb, axis=-1) + 1e-6)
    rbf_bb = compute_fampnn_rbf(dist_bb, num_rbf=self.num_rbf).reshape(num_l, num_k, 25 * self.num_rbf)

    # 2. 15 sidechain summary distance RBFs (CA_i to 10 sidechain atoms of j + 5 CB_i to sidechain summary) -> 240 channels
    sc10_j = xyz[e_idx][:, :, 4:14, :]
    sc10_mask_j = m14[e_idx][:, :, 4:14]
    diff_ca_sc = ca_xyz[:, None, None, :] - sc10_j
    dist_ca_sc = jnp.sqrt(jnp.sum(diff_ca_sc * diff_ca_sc, axis=-1) + 1e-6)
    rbf_ca_sc = compute_fampnn_rbf(dist_ca_sc, num_rbf=self.num_rbf) * sc10_mask_j[..., None]

    sc5_j = xyz[e_idx][:, :, 4:9, :]
    sc5_mask_j = m14[e_idx][:, :, 4:9]
    diff_cb_sc = cb_xyz[:, None, None, :] - sc5_j
    dist_cb_sc = jnp.sqrt(jnp.sum(diff_cb_sc * diff_cb_sc, axis=-1) + 1e-6)
    rbf_cb_sc = compute_fampnn_rbf(dist_cb_sc, num_rbf=self.num_rbf) * sc5_mask_j[..., None]

    rbf_sc = jnp.concatenate(
      [
        rbf_ca_sc.reshape(num_l, num_k, 10 * self.num_rbf),
        rbf_cb_sc.reshape(num_l, num_k, 5 * self.num_rbf),
      ],
      axis=-1,
    )

    offset = residue_idx[:, None] - residue_idx[e_idx]
    same_chain = (chain_idx[:, None] == chain_idx[e_idx]).astype(jnp.int32)
    e_pos = self.embeddings(offset, same_chain)

    raw_e = jnp.concatenate([e_pos, rbf_bb, rbf_sc], axis=-1)
    edges = _layer_norm(self.norm_edges, _linear(self.edge_embedding, raw_e))
    return cast("Float[Array, 'L K E']", edges), e_idx
