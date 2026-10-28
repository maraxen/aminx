"""Geometric Vector Perceptron (GVP) graph embedding and encoder layers for FAMPNN.

Ports ``GVP``, ``LayerNorm`` (``GVPLayerNorm``), ``GVPGraphEmbedding``, ``GVPConv``,
``GVPConvLayer``, and ``GVPEncoder`` from ``richardshuai/fampnn`` (``fampnn/model/gvp/``).
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from aminx.families.fampnn.features import (
  _layer_norm,
  _linear,
  compute_fampnn_knn,
  compute_fampnn_rbf,
  get_rotation_frames,
)


def _safe_norm(
  v: Float[Array, "... V 3"],
  *,
  axis: int = -1,
  keepdims: bool = False,
  eps: float = 1e-8,
) -> Float[Array, "..."]:
  """Compute Euclidean vector norm ``sqrt(sum(v^2) + eps)`` avoiding NaN gradients at zero."""
  return jnp.sqrt(jnp.sum(v * v, axis=axis, keepdims=keepdims) + eps)


def _normalize_zero_safe(
  v: Float[Array, "... 3"],
  *,
  axis: int = -1,
) -> Float[Array, "... 3"]:
  """Normalize 3D vectors to unit length, mapping exact zero vectors to zero."""
  sq = jnp.sum(v * v, axis=axis, keepdims=True)
  norm = jnp.sqrt(jnp.where(sq > 0, sq, 1.0))
  return jnp.where(sq > 0, v / norm, jnp.zeros_like(v))


def _compute_backbone_dihedrals(
  n_xyz: Float[Array, "L 3"],
  ca_xyz: Float[Array, "L 3"],
  c_xyz: Float[Array, "L 3"],
  *,
  eps: float = 1e-7,
) -> Float[Array, "L 6"]:
  """Compute 6-channel ``[cos(phi, psi, omega), sin(phi, psi, omega)]`` backbone dihedrals."""
  num_l = n_xyz.shape[0]
  dtype = n_xyz.dtype
  if num_l < 2:
    return jnp.concatenate([jnp.ones((num_l, 3), dtype=dtype), jnp.zeros((num_l, 3), dtype=dtype)], axis=-1)
  bb3 = jnp.stack([n_xyz, ca_xyz, c_xyz], axis=1).reshape(3 * num_l, 3)
  dx = bb3[1:] - bb3[:-1]
  u = _normalize_zero_safe(dx, axis=-1)
  u_2 = u[:-2]
  u_1 = u[1:-1]
  u_0 = u[2:]
  n_2 = _normalize_zero_safe(jnp.cross(u_2, u_1), axis=-1)
  n_1 = _normalize_zero_safe(jnp.cross(u_1, u_0), axis=-1)
  cos_d = jnp.clip(jnp.sum(n_2 * n_1, axis=-1), -1.0 + eps, 1.0 - eps)
  sign_d = jnp.sign(jnp.sum(u_2 * n_1, axis=-1))
  rad = sign_d * jnp.arccos(cos_d)
  rad_padded = jnp.pad(rad, (1, 2)).reshape(num_l, 3)
  return jnp.concatenate([jnp.cos(rad_padded), jnp.sin(rad_padded)], axis=-1)


def _compute_virtual_cb_unit(
  n_xyz: Float[Array, "L 3"],
  ca_xyz: Float[Array, "L 3"],
  c_xyz: Float[Array, "L 3"],
) -> Float[Array, "L 3"]:
  """Compute upstream FAMPNN tetrahedral unit vector ``-bisector*sqrt(1/3) - perp*sqrt(2/3)``."""
  u = _normalize_zero_safe(n_xyz - ca_xyz, axis=-1)
  c = _normalize_zero_safe(c_xyz - ca_xyz, axis=-1)
  b = _normalize_zero_safe(u - c, axis=-1)
  n = _normalize_zero_safe(jnp.cross(u, c), axis=-1)
  return -b * math.sqrt(1.0 / 3.0) - n * math.sqrt(2.0 / 3.0)


class GVPLayerNorm(eqx.Module):
  """Combined scalar LayerNorm (``eps=1e-4``) and RMS vector norm scaling.

  Matches ``fampnn/model/gvp/layers.py::LayerNorm`` where scalar channels use
  ``nn.LayerNorm(s_dim, eps=1e-4)`` and vector channels are divided by their
  root-mean-square norm across the ``V`` vector channels.

  Attributes
  ----------
  scalar_norm : eqx.nn.LayerNorm
    Scalar channel LayerNorm with ``eps=1e-4``.
  s_dim : int
    Scalar feature dimension.
  v_dim : int
    Vector feature count.
  """

  scalar_norm: eqx.nn.LayerNorm
  s_dim: int = eqx.field(static=True)
  v_dim: int = eqx.field(static=True)

  def __init__(self, dims: tuple[int, int]) -> None:
    """Initialize ``GVPLayerNorm``.

    Parameters
    ----------
    dims : tuple[int, int]
      Tuple ``(s_dim, v_dim)`` of scalar and vector channel counts.

    Returns
    -------
    None
      Initializes instance state.
    """
    s_dim, v_dim = dims
    self.scalar_norm = eqx.nn.LayerNorm(s_dim, eps=1e-4)
    self.s_dim = s_dim
    self.v_dim = v_dim

  def __call__(
    self,
    s: Float[Array, "... S"],
    v: Float[Array, "... V 3"] | None = None,
  ) -> tuple[Float[Array, "... S"], Float[Array, "... V 3"] | None]:
    """Normalize scalar and vector features.

    Parameters
    ----------
    s : Float[Array, "... S"]
      Scalar features.
    v : Float[Array, "... V 3"] | None, optional
      Vector features (when ``v_dim > 0``).

    Returns
    -------
    tuple[Float[Array, "... S"], Float[Array, "... V 3"] | None]
      Normalized ``(s_norm, v_norm)``.
    """
    s_out = _layer_norm(self.scalar_norm, s)
    if self.v_dim == 0 or v is None:
      return s_out, None
    vn = jnp.sum(v * v, axis=-1, keepdims=True) + 1e-4
    nonzero_mask = (vn > 2e-4).astype(v.dtype)
    vn = jnp.sum(vn * nonzero_mask, axis=-2, keepdims=True) / (
      1e-4 + jnp.sum(nonzero_mask, axis=-2, keepdims=True)
    )
    vn = jnp.sqrt(vn + 1e-4)
    v_out = nonzero_mask * (v / vn)
    return s_out, v_out


class GVP(eqx.Module):
  """Geometric Vector Perceptron layer operating on ``(scalar, vector)`` tuples.

  Matches ``fampnn/model/gvp/layers.py::GVP``:
  - Vector projection ``wh`` (``vi -> max(vi, vo)``, ``use_bias=False``)
  - When ``vo > 0``: ``wu`` (``max(vi, vo) -> vo``, ``use_bias=False``)
  - When ``vector_gate=True``: ``wg`` (``so -> vo``, ``use_bias=True``)
  - Scalar projection ``ws`` (``si + max(vi, vo) -> so``, ``use_bias=True``)

  Attributes
  ----------
  wh : eqx.nn.Linear | None
    Vector hidden projection ``(h, vi)`` applied along the vector channel axis.
  ws : eqx.nn.Linear
    Scalar projection ``(so, si + h)``.
  wu : eqx.nn.Linear | None
    Vector output projection ``(vo, h)`` when ``vo > 0``.
  wg : eqx.nn.Linear | None
    Vector gating linear ``(vo, so)`` when ``vector_gate=True`` and ``vo > 0``.
  si : int
    Input scalar dimension.
  vi : int
    Input vector count.
  so : int
    Output scalar dimension.
  vo : int
    Output vector count.
  has_scalar_act : bool
    Whether to apply ReLU to scalar output.
  has_vector_act : bool
    Whether to apply sigmoid gating to vector output.
  vector_gate : bool
    Whether vector gating uses scalar-derived ``sigmoid(wg(s))``.
  """

  wh: eqx.nn.Linear | None
  ws: eqx.nn.Linear
  wu: eqx.nn.Linear | None
  wg: eqx.nn.Linear | None
  si: int = eqx.field(static=True)
  vi: int = eqx.field(static=True)
  so: int = eqx.field(static=True)
  vo: int = eqx.field(static=True)
  has_scalar_act: bool = eqx.field(static=True, default=True)
  has_vector_act: bool = eqx.field(static=True, default=True)
  vector_gate: bool = eqx.field(static=True, default=True)

  def __init__(
    self,
    in_dims: tuple[int, int],
    out_dims: tuple[int, int],
    *,
    has_scalar_act: bool = True,
    has_vector_act: bool = True,
    vector_gate: bool = True,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``GVP``.

    Parameters
    ----------
    in_dims : tuple[int, int]
      Input ``(si, vi)`` dimensions.
    out_dims : tuple[int, int]
      Output ``(so, vo)`` dimensions.
    has_scalar_act : bool, default=True
      Whether to apply ReLU on scalar outputs.
    has_vector_act : bool, default=True
      Whether to apply sigmoid gating on vector outputs.
    vector_gate : bool, default=True
      Whether to use scalar-controlled vector gating ``wg``.
    key : jax.Array | None, optional
      PRNG key for weight initialization.

    Returns
    -------
    None
      Initializes instance state.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    si, vi = in_dims
    so, vo = out_dims
    self.si = si
    self.vi = vi
    self.so = so
    self.vo = vo
    self.has_scalar_act = has_scalar_act
    self.has_vector_act = has_vector_act
    self.vector_gate = vector_gate

    if vi > 0:
      h_dim = max(vi, vo)
      self.wh = eqx.nn.Linear(vi, h_dim, use_bias=False, key=k1)
      self.ws = eqx.nn.Linear(h_dim + si, so, use_bias=True, key=k2)
      if vo > 0:
        self.wu = eqx.nn.Linear(h_dim, vo, use_bias=False, key=k3)
        self.wg = (
          eqx.nn.Linear(so, vo, use_bias=True, key=k4)
          if (vector_gate and has_vector_act)
          else None
        )
      else:
        self.wu = None
        self.wg = None
    else:
      self.wh = None
      self.ws = eqx.nn.Linear(si, so, use_bias=True, key=k2)
      self.wu = None
      self.wg = None

  def __call__(
    self,
    s: Float[Array, "... S_in"],
    v: Float[Array, "... V_in 3"] | None = None,
  ) -> tuple[Float[Array, "... S_out"], Float[Array, "... V_out 3"] | None]:
    """Run GVP forward pass on ``(s, v)``.

    Parameters
    ----------
    s : Float[Array, "... S_in"]
      Input scalar features.
    v : Float[Array, "... V_in 3"] | None, optional
      Input vector features of shape ``(..., vi, 3)``.

    Returns
    -------
    tuple[Float[Array, "... S_out"], Float[Array, "... V_out 3"] | None]
      Output ``(s_out, v_out)``.
    """
    if self.vi > 0 and self.wh is not None and v is not None:
      # v has shape (..., vi, 3); wh.weight has shape (h_dim, vi)
      vh = jnp.einsum("...ic,hi->...hc", v, self.wh.weight)
      vn = _safe_norm(vh, axis=-1)
      s_in = jnp.concatenate([s, vn], axis=-1)
      s_out = _linear(self.ws, s_in)
      if self.vo > 0 and self.wu is not None:
        vu = jnp.einsum("...hc,oh->...oc", vh, self.wu.weight)
        if self.has_vector_act:
          if self.vector_gate and self.wg is not None:
            gate = jax.nn.sigmoid(_linear(self.wg, jax.nn.sigmoid(s_out)))
            v_out = vu * gate[..., None]
          else:
            v_out = vu * jax.nn.sigmoid(_safe_norm(vu, axis=-1, keepdims=True))
        else:
          v_out = vu
      else:
        v_out = None
    else:
      s_out = _linear(self.ws, s)
      v_out = (
        jnp.zeros((*s.shape[:-1], self.vo, 3), dtype=s.dtype)
        if self.vo > 0
        else None
      )

    if self.has_scalar_act:
      s_out = jax.nn.relu(s_out)
    return s_out, v_out


class GVPGraphEmbedding(eqx.Module):
  """FAMPNN GVP graph embedding constructing initial node/edge scalar and vector features.

  Matches ``fampnn/model/gvp/features.py::GVPGraphEmbedding``:
  - Node scalar input: ``6 (dihedrals) + 21 (aatype) + 14 (atom14_mask) = 41``
  - Node vector input: ``3 (forward/backward CA unit vectors + virtual CB unit vector) + 14 (atom14 - CA displacements) = 17``
  - Edge scalar input: ``16 (CA-CA RBFs) + 16 (sinusoidal positional encodings) = 32``
  - Edge vector input: ``14 (source j atom14 - CA_i displacements)``

  Attributes
  ----------
  W_v : tuple[GVPLayerNorm, GVP, GVPLayerNorm]
    Node embedding pipeline ``(41, 17) -> (128, 16)``.
  W_e : tuple[GVPLayerNorm, GVP, GVPLayerNorm]
    Edge embedding pipeline ``(32, 14) -> (32, 1)``.
  top_k : int
    Number of kNN neighbors (default ``30``).
  num_rbf : int
    Number of RBF centers (default ``16``).
  num_positional_embeddings : int
    Sinusoidal positional embedding dimension (default ``16``).
  """

  W_v_norm0: GVPLayerNorm
  W_v_gvp: GVP
  W_v_norm1: GVPLayerNorm
  W_e_norm0: GVPLayerNorm
  W_e_gvp: GVP
  W_e_norm1: GVPLayerNorm
  embed_mpnn_node: eqx.nn.Linear | None = None
  embed_mpnn_edge: eqx.nn.Linear | None = None
  top_k: int = eqx.field(static=True, default=30)
  num_rbf: int = eqx.field(static=True, default=16)
  num_positional_embeddings: int = eqx.field(static=True, default=16)

  def __init__(
    self,
    node_in_dim: tuple[int, int] = (41, 17),
    node_h_dim: tuple[int, int] = (128, 16),
    edge_in_dim: tuple[int, int] = (32, 14),
    edge_h_dim: tuple[int, int] = (32, 1),
    top_k: int = 30,
    num_rbf: int = 16,
    num_positional_embeddings: int = 16,
    *,
    has_mpnn_proj: bool = False,
    mpnn_node_dim: int = 128,
    mpnn_edge_dim: int = 512,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``GVPGraphEmbedding``.

    Parameters
    ----------
    node_in_dim : tuple[int, int]
      Parameter ``node_in_dim``.
    node_h_dim : tuple[int, int]
      Parameter ``node_h_dim``.
    edge_in_dim : tuple[int, int]
      Parameter ``edge_in_dim``.
    edge_h_dim : tuple[int, int]
      Parameter ``edge_h_dim``.
    top_k : int
      Parameter ``top_k``.
    num_rbf : int
      Parameter ``num_rbf``.
    num_positional_embeddings : int
      Parameter ``num_positional_embeddings``.
    has_mpnn_proj : bool
      Parameter ``has_mpnn_proj``.
    mpnn_node_dim : int
      Parameter ``mpnn_node_dim``.
    mpnn_edge_dim : int
      Parameter ``mpnn_edge_dim``.
    key : jax.Array | None
      Parameter ``key``.

    Returns
    -------
    None
      Result of ``__init__``.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    self.W_v_norm0 = GVPLayerNorm(node_in_dim)
    self.W_v_gvp = GVP(
      node_in_dim,
      node_h_dim,
      has_scalar_act=False,
      has_vector_act=False,
      key=k1,
    )
    self.W_v_norm1 = GVPLayerNorm(node_h_dim)

    self.W_e_norm0 = GVPLayerNorm(edge_in_dim)
    self.W_e_gvp = GVP(
      edge_in_dim,
      edge_h_dim,
      has_scalar_act=False,
      has_vector_act=False,
      key=k2,
    )
    self.W_e_norm1 = GVPLayerNorm(edge_h_dim)
    if has_mpnn_proj:
      self.embed_mpnn_node = eqx.nn.Linear(mpnn_node_dim, node_h_dim[0], key=k3)
      self.embed_mpnn_edge = eqx.nn.Linear(mpnn_edge_dim, edge_h_dim[0], key=k4)
    else:
      self.embed_mpnn_node = None
      self.embed_mpnn_edge = None
    self.top_k = top_k
    self.num_rbf = num_rbf
    self.num_positional_embeddings = num_positional_embeddings

  def __call__(  # noqa: PLR0915
    self,
    atom14_xyz: Float[Array, "L 14 3"],
    atom14_mask: Float[Array, "L 14"],
    aatype_onehot: Float[Array, "L 21"],
    mask: Float[Array, " L"],
    residue_idx: Int[Array, " L"],
    chain_idx: Int[Array, " L"],
    *,
    e_idx: Int[Array, "L K"] | None = None,
    mpnn_node: Float[Array, "L H"] | None = None,
    mpnn_edge: Float[Array, "L K E"] | None = None,
  ) -> tuple[
    tuple[Float[Array, "L Sv"], Float[Array, "L Vv 3"]],
    tuple[Float[Array, "L K Se"], Float[Array, "L K Ve 3"]],
    Int[Array, "L K"],
  ]:
    """Compute initial GVP node features, edge features, and ``E_idx``.

    Parameters
    ----------
    atom14_xyz : Float[Array, 'L 14 3']
      Parameter ``atom14_xyz``.
    atom14_mask : Float[Array, 'L 14']
      Parameter ``atom14_mask``.
    aatype_onehot : Float[Array, 'L 21']
      Parameter ``aatype_onehot``.
    mask : Float[Array, ' L']
      Parameter ``mask``.
    residue_idx : Int[Array, ' L']
      Parameter ``residue_idx``.
    chain_idx : Int[Array, ' L']
      Parameter ``chain_idx``.
    e_idx : Int[Array, 'L K'] | None, optional
      Optional precomputed kNN neighbor indices ``(L, K)``.
    mpnn_node : Float[Array, 'L H'] | None
      Parameter ``mpnn_node``.
    mpnn_edge : Float[Array, 'L K E'] | None
      Parameter ``mpnn_edge``.

    Returns
    -------
    tuple[tuple[Float[Array, 'L Sv'], Float[Array, 'L Vv 3']], tuple[Float[Array, 'L K Se'], Float[Array, 'L K Ve 3']], Int[Array, 'L K']]
      Result of ``__call__``.
    """
    del residue_idx, chain_idx
    dtype = self.W_v_gvp.ws.weight.dtype
    num_l = atom14_xyz.shape[0]
    xyz_raw = jnp.nan_to_num(atom14_xyz.astype(dtype), nan=0.0)
    m14 = atom14_mask.astype(dtype)
    xyz = jnp.where(m14[..., None] > 0, xyz_raw, xyz_raw[:, 1:2, :])

    n_xyz = xyz[:, 0, :]
    ca_xyz = xyz[:, 1, :]
    c_xyz = xyz[:, 2, :]
    o_xyz = xyz[:, 3, :]

    dihedrals = _compute_backbone_dihedrals(n_xyz, ca_xyz, c_xyz)
    if self.W_v_gvp.si == 6:
      node_s_raw = dihedrals
    else:
      node_s_raw = jnp.concatenate([dihedrals, aatype_onehot.astype(dtype), m14], axis=-1)

    fwd = jnp.concatenate([ca_xyz[1:] - ca_xyz[:-1], jnp.zeros((1, 3), dtype=dtype)], axis=0)
    bwd = jnp.concatenate([jnp.zeros((1, 3), dtype=dtype), ca_xyz[:-1] - ca_xyz[1:]], axis=0)
    u_fwd = _normalize_zero_safe(fwd, axis=-1)
    u_bwd = _normalize_zero_safe(bwd, axis=-1)
    u_cb = _compute_virtual_cb_unit(n_xyz, ca_xyz, c_xyz)
    orient_v = jnp.stack([u_fwd, u_bwd, u_cb], axis=1)

    non_ca_xyz = jnp.concatenate([xyz[:, :1, :], xyz[:, 2:, :]], axis=1)
    intra_13 = _normalize_zero_safe(ca_xyz[:, None, :] - non_ca_xyz, axis=-1)
    impute_cb = (m14[:, 4] <= 0) & (mask.astype(dtype) > 0)
    cb_slot_3 = jnp.where(impute_cb[:, None], u_cb, intra_13[:, 3, :])
    intra_13 = intra_13.at[:, 3, :].set(cb_slot_3)

    if self.W_v_gvp.vi == 15:
      node_v_raw = jnp.concatenate([orient_v[:, :2, :], intra_13], axis=1)
    else:
      atom_disp = _normalize_zero_safe(ca_xyz[:, None, :] - xyz, axis=-1)
      node_v_raw = jnp.concatenate([orient_v, atom_disp], axis=1)

    ns1, nv1 = self.W_v_gvp(node_s_raw, node_v_raw)
    h_v_s, h_v_v = self.W_v_norm1(ns1, nv1)
    if h_v_v is None:
      h_v_v = node_v_raw
    if self.embed_mpnn_node is not None and mpnn_node is not None:
      h_v_s = h_v_s + _linear(self.embed_mpnn_node, mpnn_node.astype(dtype))

    if e_idx is None:
      _, e_idx = compute_fampnn_knn(ca_xyz, mask.astype(dtype), top_k=self.top_k)
    e_dist = jnp.sqrt(jnp.sum((ca_xyz[:, None, :] - ca_xyz[e_idx]) ** 2, axis=-1) + 1e-8)
    rbf_ca = compute_fampnn_rbf(e_dist, d_min=0.0, d_max=20.0, num_rbf=self.num_rbf)

    # Sinusoidal relative positional encoding (data.py:427-442: i - j index offset, [cos, sin])
    d_offset = jnp.arange(num_l, dtype=dtype)[:, None] - e_idx.astype(dtype)
    freqs = jnp.exp(
      jnp.arange(0, self.num_positional_embeddings, 2, dtype=dtype)
      * -(math.log(10000.0) / self.num_positional_embeddings),
    )
    angles = d_offset[..., None] * freqs[None, None, :]
    pos_emb = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    edge_s_raw = jnp.concatenate([rbf_ca, pos_emb], axis=-1)

    xyz_j = xyz[e_idx]
    if self.W_e_gvp.vi == 56:
      disp_56 = jnp.concatenate(
        [
          _normalize_zero_safe(ca_xyz[:, None, None, :] - xyz_j, axis=-1),
          _normalize_zero_safe(n_xyz[:, None, None, :] - xyz_j, axis=-1),
          _normalize_zero_safe(c_xyz[:, None, None, :] - xyz_j, axis=-1),
          _normalize_zero_safe(o_xyz[:, None, None, :] - xyz_j, axis=-1),
        ],
        axis=-2,
      )
      edge_v_raw = _normalize_zero_safe(disp_56, axis=-1)
    else:
      edge_v_raw = _normalize_zero_safe(ca_xyz[:, None, None, :] - xyz_j, axis=-1)

    es1, ev1 = self.W_e_gvp(edge_s_raw, edge_v_raw)
    h_e_s, h_e_v = self.W_e_norm1(es1, ev1)
    if h_e_v is None:
      h_e_v = edge_v_raw
    if self.embed_mpnn_edge is not None and mpnn_edge is not None:
      h_e_s = h_e_s + _linear(self.embed_mpnn_edge, mpnn_edge.astype(dtype))

    return (h_v_s, h_v_v), (h_e_s, h_e_v), e_idx


class GVPConvLayer(eqx.Module):
  """FAMPNN GVP message-passing layer with PyG ``source_to_target`` scatter-mean aggregation."""

  conv_gvp0: GVP
  conv_gvp1: GVP
  conv_gvp2: GVP
  norm0: GVPLayerNorm
  ff_gvp0: GVP
  ff_gvp1: GVP
  norm1: GVPLayerNorm

  def __init__(
    self,
    node_dims: tuple[int, int] = (128, 16),
    edge_dims: tuple[int, int] = (32, 1),
    *,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``GVPConvLayer``.

    Parameters
    ----------
    node_dims : tuple[int, int]
      Parameter ``node_dims``.
    edge_dims : tuple[int, int]
      Parameter ``edge_dims``.
    key : jax.Array | None
      Parameter ``key``.

    Returns
    -------
    None
      Result of ``__init__``.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 5)
    ns, nv = node_dims
    es, ev = edge_dims
    msg_in = (2 * ns + es, 2 * nv + ev)
    ff_mid = (4 * ns, 2 * nv)

    self.conv_gvp0 = GVP(msg_in, node_dims, key=keys[0])
    self.conv_gvp1 = GVP(node_dims, node_dims, key=keys[1])
    self.conv_gvp2 = GVP(
      node_dims,
      node_dims,
      has_scalar_act=False,
      has_vector_act=False,
      key=keys[2],
    )
    self.norm0 = GVPLayerNorm(node_dims)

    self.ff_gvp0 = GVP(node_dims, ff_mid, key=keys[3])
    self.ff_gvp1 = GVP(
      ff_mid,
      node_dims,
      has_scalar_act=False,
      has_vector_act=False,
      key=keys[4],
    )
    self.norm1 = GVPLayerNorm(node_dims)

  def __call__(
    self,
    h_v: tuple[Float[Array, "L Sv"], Float[Array, "L Vv 3"]],
    h_e: tuple[Float[Array, "L K Se"], Float[Array, "L K Ve 3"]],
    e_idx: Int[Array, "L K"],
    mask_attend: Float[Array, "L K"] | None = None,
  ) -> tuple[Float[Array, "L Sv"], Float[Array, "L Vv 3"]]:
    """Run one GVPConvLayer step with PyG ``source_to_target`` scatter-mean into ``dest = e_idx``.

    Parameters
    ----------
    h_v : tuple[Float[Array, 'L Sv'], Float[Array, 'L Vv 3']]
      Parameter ``h_v``.
    h_e : tuple[Float[Array, 'L K Se'], Float[Array, 'L K Ve 3']]
      Parameter ``h_e``.
    e_idx : Int[Array, 'L K']
      Parameter ``e_idx``.
    mask_attend : Float[Array, 'L K'] | None
      Parameter ``mask_attend``.

    Returns
    -------
    tuple[Float[Array, 'L Sv'], Float[Array, 'L Vv 3']]
      Result of ``__call__``.
    """
    s_v, v_v = h_v
    s_e, v_e = h_e
    num_l, num_k = e_idx.shape
    dtype = s_v.dtype

    s_src = jnp.broadcast_to(s_v[:, None, :], (num_l, num_k, s_v.shape[-1]))
    v_src = jnp.broadcast_to(v_v[:, None, :, :], (num_l, num_k, v_v.shape[-2], 3))
    s_dst = s_v[e_idx]
    v_dst = v_v[e_idx]

    msg_s_in = jnp.concatenate([s_src, s_e, s_dst], axis=-1)
    msg_v_in = jnp.concatenate([v_src, v_e, v_dst], axis=-2)

    ms, mv = self.conv_gvp0(msg_s_in, msg_v_in)
    ms, mv = self.conv_gvp1(ms, mv)
    ms, mv = self.conv_gvp2(ms, mv)
    if mv is None:
      mv = v_v

    if mask_attend is not None:
      valid_e = mask_attend.astype(dtype)
    else:
      valid_e = jnp.ones((num_l, num_k), dtype=dtype)

    ms = ms * valid_e[..., None]
    mv = mv * valid_e[..., None, None]

    # Scatter-mean into dest = e_idx (matching PyG MessagePassing flow="source_to_target")
    agg_s = jnp.zeros_like(s_v).at[e_idx].add(ms)
    agg_v = jnp.zeros_like(v_v).at[e_idx].add(mv)
    deg = jnp.zeros((num_l,), dtype=dtype).at[e_idx].add(valid_e)
    safe_deg = jnp.maximum(deg, 1.0)

    dh_s = agg_s / safe_deg[:, None]
    dh_v = agg_v / safe_deg[:, None, None]

    s_mid, v_mid = self.norm0(s_v + dh_s, v_v + dh_v)
    if v_mid is None:
      v_mid = v_v

    fs, fv = self.ff_gvp0(s_mid, v_mid)
    fs, fv = self.ff_gvp1(fs, fv)
    if fv is None:
      fv = v_mid

    s_out, v_out = self.norm1(s_mid + fs, v_mid + fv)
    if v_out is None:
      v_out = v_mid
    return s_out, v_out


class GVPEncoder(eqx.Module):
  """GVP encoder producing node features for ``FAEncoder``."""

  embed_graph: GVPGraphEmbedding
  encoder_layers: tuple[GVPConvLayer, ...]
  norm_gvp_out: eqx.nn.LayerNorm
  w_out_norm: GVPLayerNorm | None = None
  w_out_gvp: GVP | None = None
  w_out_linear: eqx.nn.Linear | None = None
  rotate_out: bool = eqx.field(static=True, default=True)

  def __init__(
    self,
    node_h_dim: tuple[int, int] = (128, 16),
    edge_h_dim: tuple[int, int] = (32, 1),
    num_layers: int = 3,
    *,
    node_in_dim: tuple[int, int] = (41, 17),
    edge_in_dim: tuple[int, int] = (32, 14),
    has_mpnn_proj: bool = False,
    w_out_mode: str = "norm",
    rotate_out: bool = True,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``GVPEncoder``.

    Parameters
    ----------
    node_h_dim : tuple[int, int]
      Parameter ``node_h_dim``.
    edge_h_dim : tuple[int, int]
      Parameter ``edge_h_dim``.
    num_layers : int
      Parameter ``num_layers``.
    node_in_dim : tuple[int, int]
      Parameter ``node_in_dim``.
    edge_in_dim : tuple[int, int]
      Parameter ``edge_in_dim``.
    has_mpnn_proj : bool
      Parameter ``has_mpnn_proj``.
    w_out_mode : str
      Parameter ``w_out_mode``.
    rotate_out : bool
      Parameter ``rotate_out``.
    key : jax.Array | None
      Parameter ``key``.

    Returns
    -------
    None
      Result of ``__init__``.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 3 + num_layers)
    self.embed_graph = GVPGraphEmbedding(
      node_in_dim=node_in_dim,
      node_h_dim=node_h_dim,
      edge_in_dim=edge_in_dim,
      edge_h_dim=edge_h_dim,
      has_mpnn_proj=has_mpnn_proj,
      key=keys[0],
    )
    self.encoder_layers = tuple(
      GVPConvLayer(node_dims=node_h_dim, edge_dims=edge_h_dim, key=k)
      for k in keys[1 : 1 + num_layers]
    )
    out_dim = node_h_dim[0] + 3 * node_h_dim[1]
    self.norm_gvp_out = eqx.nn.LayerNorm(out_dim, eps=1e-5)
    if w_out_mode == "gvp":
      self.w_out_norm = GVPLayerNorm(node_h_dim)
      self.w_out_gvp = GVP(node_h_dim, (node_h_dim[0], 0), key=keys[-2])
      self.w_out_linear = None
    elif w_out_mode == "linear":
      self.w_out_norm = None
      self.w_out_gvp = None
      self.w_out_linear = eqx.nn.Linear(out_dim, node_h_dim[0], key=keys[-1])
    else:
      self.w_out_norm = None
      self.w_out_gvp = None
      self.w_out_linear = None
    self.rotate_out = rotate_out

  def __call__(
    self,
    atom14_xyz: Float[Array, "L 14 3"],
    atom14_mask: Float[Array, "L 14"],
    aatype_onehot: Float[Array, "L 21"],
    mask: Float[Array, " L"],
    residue_idx: Int[Array, " L"],
    chain_idx: Int[Array, " L"],
    *,
    e_idx: Int[Array, "L K"] | None = None,
    mpnn_node: Float[Array, "L H"] | None = None,
    mpnn_edge: Float[Array, "L K E"] | None = None,
  ) -> Float[Array, "L Out"]:
    """Run GVP graph embedding, GVPConvLayers, and output projection.

    Parameters
    ----------
    atom14_xyz : Float[Array, 'L 14 3']
      Parameter ``atom14_xyz``.
    atom14_mask : Float[Array, 'L 14']
      Parameter ``atom14_mask``.
    aatype_onehot : Float[Array, 'L 21']
      Parameter ``aatype_onehot``.
    mask : Float[Array, ' L']
      Parameter ``mask``.
    residue_idx : Int[Array, ' L']
      Parameter ``residue_idx``.
    chain_idx : Int[Array, ' L']
      Parameter ``chain_idx``.
    e_idx : Int[Array, 'L K'] | None
      Parameter ``e_idx``.
    mpnn_node : Float[Array, 'L H'] | None
      Parameter ``mpnn_node``.
    mpnn_edge : Float[Array, 'L K E'] | None
      Parameter ``mpnn_edge``.

    Returns
    -------
    Float[Array, 'L Out']
      Result of ``__call__``.
    """
    h_v, h_e, e_idx = self.embed_graph(
      atom14_xyz=atom14_xyz,
      atom14_mask=atom14_mask,
      aatype_onehot=aatype_onehot,
      mask=mask,
      residue_idx=residue_idx,
      chain_idx=chain_idx,
      e_idx=e_idx,
      mpnn_node=mpnn_node,
      mpnn_edge=mpnn_edge,
    )
    dtype = h_v[0].dtype
    mask_f = mask.astype(dtype)
    mask_2d = mask_f[:, None] * mask_f[None, :]
    mask_attend = jnp.take_along_axis(mask_2d, e_idx, axis=1)

    for layer in self.encoder_layers:
      h_v = layer(h_v, h_e, e_idx, mask_attend=mask_attend)

    s_v, v_v = h_v
    if self.w_out_norm is not None and self.w_out_gvp is not None:
      s_norm, v_norm = self.w_out_norm(s_v, v_v)
      s_out, _ = self.w_out_gvp(s_norm, v_norm)
      return s_out * mask_f[:, None]

    if self.rotate_out:
      rot = get_rotation_frames(atom14_xyz[:, 0, :], atom14_xyz[:, 1, :], atom14_xyz[:, 2, :])
      v_v = jnp.einsum("lvi,lij->lvj", v_v, rot.astype(dtype))

    v_flat = v_v.reshape(s_v.shape[0], -1)
    combined = jnp.concatenate([s_v, v_flat], axis=-1)
    if self.w_out_linear is not None:
      return _linear(self.w_out_linear, combined) * mask_f[:, None]
    return _layer_norm(self.norm_gvp_out, combined) * mask_f[:, None]
