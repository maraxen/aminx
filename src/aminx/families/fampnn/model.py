"""FAMPNN full-atom encoder (FAEncoder), sequence categorical head, and FAMPNNModel.

Ports ``EncLayer``, ``EdgeUpdatingDecLayer``, ``DecLayer``, ``FAEncoder``, and
``FAMPNN`` from ``richardshuai/fampnn`` (``fampnn/model/fampnn.py``).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from aminx.families.fampnn.constants import AMINX_TO_FAMPNN_ARR
from aminx.families.fampnn.features import (
  FAMPNNProteinFeatures,
  SidechainProteinFeatures,
  _layer_norm,
  _linear,
  atom37_to_atom14,
)
from aminx.families.fampnn.gvp import GVPEncoder
from aminx.families.fampnn.sidechain_diffusion import (
  SidechainConfidenceModule,
  SidechainDiffusionModule,
)


def _cat_neighbors_nodes(
  h_nodes: Float[Array, "L C"],
  h_neighbors: Float[Array, "L K E"],
  e_idx: Int[Array, "L K"],
) -> Float[Array, "L K E_plus_C"]:
  """Concatenate ``h_neighbors`` with gathered neighbor nodes ``h_nodes[e_idx]``."""
  return jnp.concatenate([h_neighbors, h_nodes[e_idx]], axis=-1)


class FAMPNNPositionWiseFFN(eqx.Module):
  """Two-layer position-wise feed-forward network with exact-erf GELU."""

  W_in: eqx.nn.Linear
  W_out: eqx.nn.Linear

  def __init__(
    self,
    num_hidden: int = 128,
    num_ff: int = 512,
    *,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``FAMPNNPositionWiseFFN``.

    Parameters
    ----------
    num_hidden : int
      Parameter ``num_hidden``.
    num_ff : int
      Parameter ``num_ff``.
    key : jax.Array | None
      Parameter ``key``.

    Returns
    -------
    None
      Result of ``__init__``.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key, 2)
    self.W_in = eqx.nn.Linear(num_hidden, num_ff, key=k1)
    self.W_out = eqx.nn.Linear(num_ff, num_hidden, key=k2)

  def __call__(self, h_v: Float[Array, "... H"]) -> Float[Array, "... H"]:
    """Apply ``W_out(gelu(W_in(h_v), approximate=False))``.

    Parameters
    ----------
    h_v : Float[Array, '... H']
      Parameter ``h_v``.

    Returns
    -------
    Float[Array, '... H']
      Result of ``__call__``.
    """
    return _linear(self.W_out, jax.nn.gelu(_linear(self.W_in, h_v), approximate=False))


class FAMPNNEncLayer(eqx.Module):
  """Bidirectional MPNN encoder layer updating both ``h_V`` and ``h_E``."""

  W1: eqx.nn.Linear
  W2: eqx.nn.Linear
  W3: eqx.nn.Linear
  W11: eqx.nn.Linear
  W12: eqx.nn.Linear
  W13: eqx.nn.Linear
  norm1: eqx.nn.LayerNorm
  norm2: eqx.nn.LayerNorm
  norm3: eqx.nn.LayerNorm
  dense: FAMPNNPositionWiseFFN
  scale: float = eqx.field(static=True, default=30.0)

  def __init__(
    self,
    num_hidden: int = 128,
    num_in: int = 256,
    *,
    scale: float = 30.0,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``FAMPNNEncLayer``.

    Parameters
    ----------
    num_hidden : int
      Parameter ``num_hidden``.
    num_in : int
      Parameter ``num_in``.
    scale : float
      Parameter ``scale``.
    key : jax.Array | None
      Parameter ``key``.

    Returns
    -------
    None
      Result of ``__init__``.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 7)
    total_in = num_hidden + num_in
    self.W1 = eqx.nn.Linear(total_in, num_hidden, key=keys[0])
    self.W2 = eqx.nn.Linear(num_hidden, num_hidden, key=keys[1])
    self.W3 = eqx.nn.Linear(num_hidden, num_hidden, key=keys[2])
    self.W11 = eqx.nn.Linear(total_in, num_hidden, key=keys[3])
    self.W12 = eqx.nn.Linear(num_hidden, num_hidden, key=keys[4])
    self.W13 = eqx.nn.Linear(num_hidden, num_hidden, key=keys[5])
    self.norm1 = eqx.nn.LayerNorm(num_hidden)
    self.norm2 = eqx.nn.LayerNorm(num_hidden)
    self.norm3 = eqx.nn.LayerNorm(num_hidden)
    self.dense = FAMPNNPositionWiseFFN(num_hidden, num_hidden * 4, key=keys[6])
    self.scale = scale

  def __call__(
    self,
    h_v: Float[Array, "L H"],
    h_e: Float[Array, "L K H"],
    e_idx: Int[Array, "L K"],
    mask_v: Float[Array, " L"],
    mask_attend: Float[Array, "L K"],
  ) -> tuple[Float[Array, "L H"], Float[Array, "L K H"]]:
    """Run one bidirectional encoder layer step.

    Parameters
    ----------
    h_v : Float[Array, 'L H']
      Parameter ``h_v``.
    h_e : Float[Array, 'L K H']
      Parameter ``h_e``.
    e_idx : Int[Array, 'L K']
      Parameter ``e_idx``.
    mask_v : Float[Array, ' L']
      Parameter ``mask_v``.
    mask_attend : Float[Array, 'L K']
      Parameter ``mask_attend``.

    Returns
    -------
    tuple[Float[Array, 'L H'], Float[Array, 'L K H']]
      Result of ``__call__``.
    """
    num_k = e_idx.shape[1]
    h_v_exp = jnp.broadcast_to(h_v[:, None, :], (*h_e.shape[:-1], h_v.shape[-1]))
    h_ev = jnp.concatenate([h_v_exp, _cat_neighbors_nodes(h_v, h_e, e_idx)], axis=-1)

    h_msg = _linear(
      self.W3,
      jax.nn.gelu(
        _linear(self.W2, jax.nn.gelu(_linear(self.W1, h_ev), approximate=False)),
        approximate=False,
      ),
    )
    dh = jnp.sum(mask_attend[..., None] * h_msg, axis=-2) / self.scale
    h_v = _layer_norm(self.norm1, h_v + dh)
    h_v = _layer_norm(self.norm2, h_v + self.dense(h_v)) * mask_v[:, None]

    h_v_exp2 = jnp.broadcast_to(h_v[:, None, :], (h_v.shape[0], num_k, h_v.shape[-1]))
    h_ev_edge = jnp.concatenate([h_v_exp2, _cat_neighbors_nodes(h_v, h_e, e_idx)], axis=-1)
    h_e_msg = _linear(
      self.W13,
      jax.nn.gelu(
        _linear(self.W12, jax.nn.gelu(_linear(self.W11, h_ev_edge), approximate=False)),
        approximate=False,
      ),
    )
    h_e = _layer_norm(self.norm3, h_e + h_e_msg)
    return h_v, h_e


class FAMPNNEdgeUpdatingDecLayer(eqx.Module):
  """FAMPNN ``FAEncoder`` decoder layer that updates both ``h_V`` and ``h_E`` / ``h_ESV``."""

  W1: eqx.nn.Linear
  W2: eqx.nn.Linear
  W3: eqx.nn.Linear
  W11: eqx.nn.Linear
  W12: eqx.nn.Linear
  W13: eqx.nn.Linear
  norm1: eqx.nn.LayerNorm
  norm2: eqx.nn.LayerNorm
  norm3: eqx.nn.LayerNorm
  dense: FAMPNNPositionWiseFFN
  edge_update_full: bool = eqx.field(static=True, default=False)
  scale: float = eqx.field(static=True, default=30.0)

  def __init__(
    self,
    num_hidden: int = 128,
    num_in: int = 384,
    *,
    edge_update_full: bool = False,
    scale: float = 30.0,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``FAMPNNEdgeUpdatingDecLayer``.

    Parameters
    ----------
    num_hidden : int
      Parameter ``num_hidden``.
    num_in : int
      Parameter ``num_in``.
    edge_update_full : bool
      Parameter ``edge_update_full``.
    scale : float
      Parameter ``scale``.
    key : jax.Array | None
      Parameter ``key``.

    Returns
    -------
    None
      Result of ``__init__``.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 7)
    self.W1 = eqx.nn.Linear(num_hidden + num_in, num_hidden, key=keys[0])
    self.W2 = eqx.nn.Linear(num_hidden, num_hidden, key=keys[1])
    self.W3 = eqx.nn.Linear(num_hidden, num_hidden, key=keys[2])
    if edge_update_full:
      self.W11 = eqx.nn.Linear(num_hidden * 2 + num_in, num_hidden, key=keys[3])
      self.W12 = eqx.nn.Linear(num_hidden, num_hidden, key=keys[4])
      self.W13 = eqx.nn.Linear(num_hidden, num_in, key=keys[5])
      self.norm3 = eqx.nn.LayerNorm(num_in)
    else:
      self.W11 = eqx.nn.Linear(num_hidden * 3, num_hidden, key=keys[3])
      self.W12 = eqx.nn.Linear(num_hidden, num_hidden, key=keys[4])
      self.W13 = eqx.nn.Linear(num_hidden, num_hidden, key=keys[5])
      self.norm3 = eqx.nn.LayerNorm(num_hidden)
    self.norm1 = eqx.nn.LayerNorm(num_hidden)
    self.norm2 = eqx.nn.LayerNorm(num_hidden)
    self.dense = FAMPNNPositionWiseFFN(num_hidden, num_hidden * 4, key=keys[6])
    self.edge_update_full = edge_update_full
    self.scale = scale

  def __call__(
    self,
    h_v: Float[Array, "L H"],
    h_e: Float[Array, "L K E"],
    h_s: Float[Array, "L H"],
    e_idx: Int[Array, "L K"],
    mask_v: Float[Array, " L"],
    mask_attend: Float[Array, "L K"],
  ) -> tuple[Float[Array, "L H"], Float[Array, "L K E"]]:
    """Run one edge-updating decoder layer step.

    Parameters
    ----------
    h_v : Float[Array, 'L H']
      Parameter ``h_v``.
    h_e : Float[Array, 'L K E']
      Parameter ``h_e``.
    h_s : Float[Array, 'L H']
      Parameter ``h_s``.
    e_idx : Int[Array, 'L K']
      Parameter ``e_idx``.
    mask_v : Float[Array, ' L']
      Parameter ``mask_v``.
    mask_attend : Float[Array, 'L K']
      Parameter ``mask_attend``.

    Returns
    -------
    tuple[Float[Array, 'L H'], Float[Array, 'L K E']]
      Result of ``__call__``.
    """
    num_k = e_idx.shape[1]
    if self.edge_update_full:
      h_esv = h_e
    else:
      h_es = _cat_neighbors_nodes(h_s, h_e, e_idx)
      h_esv = _cat_neighbors_nodes(h_v, h_es, e_idx)
    h_v_exp = jnp.broadcast_to(h_v[:, None, :], (*h_esv.shape[:-1], h_v.shape[-1]))
    h_in = jnp.concatenate([h_v_exp, h_esv], axis=-1)

    h_msg = _linear(
      self.W3,
      jax.nn.gelu(
        _linear(self.W2, jax.nn.gelu(_linear(self.W1, h_in), approximate=False)),
        approximate=False,
      ),
    )
    dh = jnp.sum(mask_attend[..., None] * h_msg, axis=-2) / self.scale
    h_v = _layer_norm(self.norm1, h_v + dh)
    h_v = _layer_norm(self.norm2, h_v + self.dense(h_v)) * mask_v[:, None]

    # Edge update using updated h_v
    h_v_exp2 = jnp.broadcast_to(h_v[:, None, :], (h_v.shape[0], num_k, h_v.shape[-1]))
    h_ev_edge = jnp.concatenate([h_v_exp2, _cat_neighbors_nodes(h_v, h_e, e_idx)], axis=-1)
    h_e_msg = _linear(
      self.W13,
      jax.nn.gelu(
        _linear(self.W12, jax.nn.gelu(_linear(self.W11, h_ev_edge), approximate=False)),
        approximate=False,
      ),
    )
    h_e = _layer_norm(self.norm3, h_e + h_e_msg)
    return h_v, h_e


class FAMPNNDecLayer(eqx.Module):
  """Standard node-only decoder layer used inside ``SidechainConfidenceModule`` (``512 -> 128``)."""

  W1: eqx.nn.Linear
  W2: eqx.nn.Linear
  W3: eqx.nn.Linear
  norm1: eqx.nn.LayerNorm
  norm2: eqx.nn.LayerNorm
  dense: FAMPNNPositionWiseFFN
  scale: float = eqx.field(static=True, default=30.0)

  def __init__(
    self,
    num_hidden: int = 128,
    num_in: int = 384,
    *,
    scale: float = 30.0,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``FAMPNNDecLayer``.

    Parameters
    ----------
    num_hidden : int
      Parameter ``num_hidden``.
    num_in : int
      Parameter ``num_in``.
    scale : float
      Parameter ``scale``.
    key : jax.Array | None
      Parameter ``key``.

    Returns
    -------
    None
      Result of ``__init__``.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)
    self.W1 = eqx.nn.Linear(num_hidden + num_in, num_hidden, key=keys[0])
    self.W2 = eqx.nn.Linear(num_hidden, num_hidden, key=keys[1])
    self.W3 = eqx.nn.Linear(num_hidden, num_hidden, key=keys[2])
    self.norm1 = eqx.nn.LayerNorm(num_hidden)
    self.norm2 = eqx.nn.LayerNorm(num_hidden)
    self.dense = FAMPNNPositionWiseFFN(num_hidden, num_hidden * 4, key=keys[3])
    self.scale = scale

  def __call__(
    self,
    h_v: Float[Array, "L H"],
    h_esv: Float[Array, "L K M"],
    mask_v: Float[Array, " L"],
    mask_attend: Float[Array, "L K"],
  ) -> Float[Array, "L H"]:
    """Run one node-only decoder layer step.

    Parameters
    ----------
    h_v : Float[Array, 'L H']
      Parameter ``h_v``.
    h_esv : Float[Array, 'L K M']
      Parameter ``h_esv``.
    mask_v : Float[Array, ' L']
      Parameter ``mask_v``.
    mask_attend : Float[Array, 'L K']
      Parameter ``mask_attend``.

    Returns
    -------
    Float[Array, 'L H']
      Result of ``__call__``.
    """
    h_v_exp = jnp.broadcast_to(h_v[:, None, :], (*h_esv.shape[:-1], h_v.shape[-1]))
    h_in = jnp.concatenate([h_v_exp, h_esv], axis=-1)
    h_msg = _linear(
      self.W3,
      jax.nn.gelu(
        _linear(self.W2, jax.nn.gelu(_linear(self.W1, h_in), approximate=False)),
        approximate=False,
      ),
    )
    dh = jnp.sum(mask_attend[..., None] * h_msg, axis=-2) / self.scale
    h_v = _layer_norm(self.norm1, h_v + dh)
    return _layer_norm(self.norm2, h_v + self.dense(h_v)) * mask_v[:, None]


class FAEncoder(eqx.Module):
  """FAMPNN full-atom encoder combining ``FAMPNNProteinFeatures``, ``GVPEncoder``, ``EncLayer``s, and ``EdgeUpdatingDecLayer``s."""

  features: FAMPNNProteinFeatures
  gvp_encoder: GVPEncoder
  W_v: eqx.nn.Linear
  W_e: eqx.nn.Linear
  W_s: eqx.nn.Embedding
  encoder_layers: tuple[FAMPNNEncLayer, ...]
  decoder_layers: tuple[FAMPNNEdgeUpdatingDecLayer, ...]
  sidechain_features: SidechainProteinFeatures | None = None
  W_e2: eqx.nn.Linear | None = None

  def __init__(
    self,
    hidden_dim: int = 128,
    num_encoder_layers: int = 5,
    num_decoder_layers: int = 5,
    *,
    use_sidechain_features: bool = False,
    num_gvp_layers: int = 3,
    gvp_w_out_mode: str = "norm",
    gvp_rotate_out: bool = True,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``FAEncoder``.

    Parameters
    ----------
    hidden_dim : int
      Parameter ``hidden_dim``.
    num_encoder_layers : int
      Parameter ``num_encoder_layers``.
    num_decoder_layers : int
      Parameter ``num_decoder_layers``.
    use_sidechain_features : bool
      Parameter ``use_sidechain_features``.
    num_gvp_layers : int
      Parameter ``num_gvp_layers``.
    gvp_w_out_mode : str
      Parameter ``gvp_w_out_mode``.
    gvp_rotate_out : bool
      Parameter ``gvp_rotate_out``.
    key : jax.Array | None
      Parameter ``key``.

    Returns
    -------
    None
      Result of ``__init__``.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 7 + num_encoder_layers + num_decoder_layers)
    self.features = FAMPNNProteinFeatures(edge_features=hidden_dim, key=keys[0])
    self.gvp_encoder = GVPEncoder(
      edge_h_dim=(512, 1) if use_sidechain_features else (32, 1),
      num_layers=num_gvp_layers,
      node_in_dim=(6, 15) if use_sidechain_features else (41, 17),
      edge_in_dim=(32, 56) if use_sidechain_features else (32, 14),
      has_mpnn_proj=use_sidechain_features,
      w_out_mode=gvp_w_out_mode,
      rotate_out=gvp_rotate_out,
      key=keys[1],
    )
    self.W_v = eqx.nn.Linear(176, hidden_dim, key=keys[2])
    self.W_e = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[3])
    self.W_s = eqx.nn.Embedding(21, hidden_dim, key=keys[4])
    if use_sidechain_features:
      self.sidechain_features = SidechainProteinFeatures(edge_features=hidden_dim, key=keys[5])
      self.W_e2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[6])
    else:
      self.sidechain_features = None
      self.W_e2 = None

    enc_keys = keys[7 : 7 + num_encoder_layers]
    dec_keys = keys[7 + num_encoder_layers :]
    self.encoder_layers = tuple(
      FAMPNNEncLayer(num_hidden=hidden_dim, key=k) for k in enc_keys
    )
    dec_in = hidden_dim * 4 if use_sidechain_features else hidden_dim * 3
    self.decoder_layers = tuple(
      FAMPNNEdgeUpdatingDecLayer(
        num_hidden=hidden_dim,
        num_in=dec_in,
        edge_update_full=use_sidechain_features,
        key=k,
      )
      for k in dec_keys
    )

  def __call__(
    self,
    atom37_xyz: Float[Array, "L 37 3"],
    atom37_mask: Float[Array, "L 37"],
    aatype: Int[Array, " L"],
    seq_mask: Float[Array, " L"],
    mask: Float[Array, " L"],
    residue_idx: Int[Array, " L"],
    chain_idx: Int[Array, " L"],
  ) -> tuple[Float[Array, "L 128"], Float[Array, "L K E"], Int[Array, "L K"]]:
    """Run ``FAEncoder`` forward pass.

    Parameters
    ----------
    atom37_xyz : Float[Array, 'L 37 3']
      Parameter ``atom37_xyz``.
    atom37_mask : Float[Array, 'L 37']
      Parameter ``atom37_mask``.
    aatype : Int[Array, ' L']
      Parameter ``aatype``.
    seq_mask : Float[Array, ' L']
      Parameter ``seq_mask``.
    mask : Float[Array, ' L']
      Parameter ``mask``.
    residue_idx : Int[Array, ' L']
      Parameter ``residue_idx``.
    chain_idx : Int[Array, ' L']
      Parameter ``chain_idx``.

    Returns
    -------
    tuple[Float[Array, 'L 128'], Float[Array, 'L K E'], Int[Array, 'L K']]
      Result of ``__call__``.
    """
    dtype = self.W_v.weight.dtype
    aa_clamped = jnp.clip(aatype, 0, 20)
    vis_seq = seq_mask.astype(dtype)
    aatype_oh = jax.nn.one_hot(aa_clamped, 21, dtype=dtype) * vis_seq[:, None]

    atom14_xyz, atom14_mask = atom37_to_atom14(atom37_xyz, atom37_mask, aa_clamped)
    sc_gate = jnp.concatenate(
      [
        jnp.ones((atom14_mask.shape[0], 4), dtype=dtype),
        jnp.broadcast_to(vis_seq[:, None], (atom14_mask.shape[0], 10)),
      ],
      axis=-1,
    )
    atom14_mask_gated = atom14_mask.astype(dtype) * sc_gate
    atom14_xyz_gated = atom14_xyz.astype(dtype) * atom14_mask_gated[..., None]

    e_init, e_idx = self.features(atom37_xyz, mask, residue_idx, chain_idx)
    mask_f = mask.astype(dtype)
    mask_2d = mask_f[:, None] * mask_f[None, :]
    mask_attend = jnp.take_along_axis(mask_2d, e_idx, axis=1)

    if self.sidechain_features is not None and self.W_e2 is not None:
      h_v = jnp.zeros((atom37_xyz.shape[0], self.W_e.out_features), dtype=dtype)
      h_e = _linear(self.W_e, e_init)
      for enc_layer in self.encoder_layers:
        h_v, h_e = enc_layer(h_v, h_e, e_idx, mask_f, mask_attend)
      h_s = self.W_s.weight[aa_clamped] * vis_seq[:, None]
      e2_init, _ = self.sidechain_features(atom14_xyz_gated, atom14_mask_gated, mask, residue_idx, chain_idx)
      h_e2 = _linear(self.W_e2, e2_init)
      h_es = jnp.concatenate([_cat_neighbors_nodes(h_s, h_e, e_idx), h_e2], axis=-1)
      h_esv = _cat_neighbors_nodes(h_v, h_es, e_idx)
      for dec_layer in self.decoder_layers:
        h_v, h_esv = dec_layer(h_v, h_esv, h_s, e_idx, mask_f, mask_attend)
      h_v = self.gvp_encoder(
        atom14_xyz_gated,
        atom14_mask_gated,
        aatype_oh,
        mask,
        residue_idx,
        chain_idx,
        mpnn_node=h_v,
        mpnn_edge=h_esv,
      )
      return h_v, h_esv, e_idx

    gvp_feat = self.gvp_encoder(
      atom14_xyz_gated,
      atom14_mask_gated,
      aatype_oh,
      mask,
      residue_idx,
      chain_idx,
    )

    h_v = _linear(self.W_v, gvp_feat)
    h_e = _linear(self.W_e, e_init)

    for enc_layer in self.encoder_layers:
      h_v, h_e = enc_layer(h_v, h_e, e_idx, mask_f, mask_attend)

    h_s = self.W_s.weight[aa_clamped] * vis_seq[:, None]
    for dec_layer in self.decoder_layers:
      h_v, h_e = dec_layer(h_v, h_e, h_s, e_idx, mask_f, mask_attend)

    return h_v, h_e, e_idx


class FAMPNNForwardOutput(NamedTuple):
  """Output of :class:`FAMPNNModel` forward pass."""

  logits_fampnn: Float[Array, "L 21"]
  logits_aminx: Float[Array, "L 21"]
  h_v: Float[Array, "L 128"]
  h_e: Float[Array, "L K E"]
  e_idx: Int[Array, "L K"]


class FAMPNNModel(eqx.Module):
  """Top-level FAMPNN joint categorical sequence and continuous sidechain diffusion model."""

  fa_encoder: FAEncoder
  mlp_seq_0: eqx.nn.Linear
  mlp_seq_2: eqx.nn.Linear
  scn_diffusion: SidechainDiffusionModule
  scn_confidence: SidechainConfidenceModule | None = None
  W_out: eqx.nn.Linear | None = None

  def __init__(
    self,
    hidden_dim: int = 128,
    *,
    has_confidence: bool = True,
    num_encoder_layers: int = 5,
    num_decoder_layers: int = 5,
    use_sidechain_features: bool = False,
    num_gvp_layers: int = 3,
    gvp_w_out_mode: str = "norm",
    gvp_rotate_out: bool = True,
    has_w_out: bool = False,
    scn_depth: int = 3,
    has_f_embedder: bool = False,
    conf_num_bins: int = 64,
    conf_has_upstream_heads: bool = False,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``FAMPNNModel``.

    Parameters
    ----------
    hidden_dim : int
      Parameter ``hidden_dim``.
    has_confidence : bool
      Parameter ``has_confidence``.
    num_encoder_layers : int
      Parameter ``num_encoder_layers``.
    num_decoder_layers : int
      Parameter ``num_decoder_layers``.
    use_sidechain_features : bool
      Parameter ``use_sidechain_features``.
    num_gvp_layers : int
      Parameter ``num_gvp_layers``.
    gvp_w_out_mode : str
      Parameter ``gvp_w_out_mode``.
    gvp_rotate_out : bool
      Parameter ``gvp_rotate_out``.
    has_w_out : bool
      Parameter ``has_w_out``.
    scn_depth : int
      Parameter ``scn_depth``.
    has_f_embedder : bool
      Parameter ``has_f_embedder``.
    conf_num_bins : int
      Parameter ``conf_num_bins``.
    conf_has_upstream_heads : bool
      Parameter ``conf_has_upstream_heads``.
    key : jax.Array | None
      Parameter ``key``.

    Returns
    -------
    None
      Result of ``__init__``.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 7)
    self.fa_encoder = FAEncoder(
      hidden_dim=hidden_dim,
      num_encoder_layers=num_encoder_layers,
      num_decoder_layers=num_decoder_layers,
      use_sidechain_features=use_sidechain_features,
      num_gvp_layers=num_gvp_layers,
      gvp_w_out_mode=gvp_w_out_mode,
      gvp_rotate_out=gvp_rotate_out,
      key=keys[0],
    )
    self.mlp_seq_0 = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1])
    self.mlp_seq_2 = eqx.nn.Linear(hidden_dim, 20, key=keys[2])
    self.W_out = eqx.nn.Linear(hidden_dim, 21, key=keys[6]) if has_w_out else None
    self.scn_diffusion = SidechainDiffusionModule(
      depth=scn_depth,
      cond_dim=hidden_dim,
      has_f_embedder=has_f_embedder,
      key=keys[3],
    )
    if has_confidence:
      c_enc_keys = jax.random.split(keys[4], 3)
      c_dec_keys = jax.random.split(keys[5], 3)
      conf_enc = tuple(FAMPNNEncLayer(num_hidden=hidden_dim, key=k) for k in c_enc_keys)
      conf_dec: tuple[eqx.Module, ...]
      if conf_has_upstream_heads:
        conf_dec = tuple(
          FAMPNNEdgeUpdatingDecLayer(
            num_hidden=hidden_dim,
            num_in=hidden_dim * 4,
            edge_update_full=True,
            key=k,
          )
          for k in c_dec_keys
        )
      else:
        conf_dec = tuple(FAMPNNDecLayer(num_hidden=hidden_dim, key=k) for k in c_dec_keys)
      self.scn_confidence = SidechainConfidenceModule(
        encoder_layers=conf_enc,
        decoder_layers=conf_dec,
        hidden_dim=hidden_dim,
        num_bins=conf_num_bins,
        has_upstream_heads=conf_has_upstream_heads,
        key=keys[4],
      )
    else:
      self.scn_confidence = None

  def __call__(
    self,
    atom37_xyz: Float[Array, "L 37 3"],
    atom37_mask: Float[Array, "L 37"],
    aatype: Int[Array, " L"],
    seq_mask: Float[Array, " L"],
    mask: Float[Array, " L"],
    residue_idx: Int[Array, " L"],
    chain_idx: Int[Array, " L"],
  ) -> FAMPNNForwardOutput:
    """Run ``FAEncoder`` and sequence head to predict 21-token sequence logits.

    Parameters
    ----------
    atom37_xyz : Float[Array, 'L 37 3']
      Parameter ``atom37_xyz``.
    atom37_mask : Float[Array, 'L 37']
      Parameter ``atom37_mask``.
    aatype : Int[Array, ' L']
      Parameter ``aatype``.
    seq_mask : Float[Array, ' L']
      Parameter ``seq_mask``.
    mask : Float[Array, ' L']
      Parameter ``mask``.
    residue_idx : Int[Array, ' L']
      Parameter ``residue_idx``.
    chain_idx : Int[Array, ' L']
      Parameter ``chain_idx``.

    Returns
    -------
    FAMPNNForwardOutput
      Result of ``__call__``.
    """
    h_v, h_e, e_idx = self.fa_encoder(
      atom37_xyz=atom37_xyz,
      atom37_mask=atom37_mask,
      aatype=aatype,
      seq_mask=seq_mask,
      mask=mask,
      residue_idx=residue_idx,
      chain_idx=chain_idx,
    )
    if self.W_out is not None:
      logits_fampnn = _linear(self.W_out, h_v)
    else:
      h_seq = jax.nn.gelu(_linear(self.mlp_seq_0, h_v), approximate=False)
      logits_20 = _linear(self.mlp_seq_2, h_seq)
      pad_x = jnp.full((logits_20.shape[0], 1), -1e9, dtype=logits_20.dtype)
      logits_fampnn = jnp.concatenate([logits_20, pad_x], axis=-1)
    logits_aminx = jnp.take(logits_fampnn, AMINX_TO_FAMPNN_ARR, axis=-1)
    return FAMPNNForwardOutput(
      logits_fampnn=logits_fampnn,
      logits_aminx=logits_aminx,
      h_v=h_v,
      h_e=h_e,
      e_idx=e_idx,
    )
