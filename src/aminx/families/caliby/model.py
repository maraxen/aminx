"""Caliby 3D-coordinate Potts MPNN encoder/decoder trunk and unified CalibyModel.

Ports ``EncLayer``, ``DecLayer``, ``CalibyMPNN``, and ``CalibyModel`` from
``ProteinDesignLab/caliby`` (``caliby/model/lr_potts_model/model.py``).
"""

from __future__ import annotations

from typing import NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from aminx.families.caliby.features import CalibyTokenFeatures, _layer_norm, _linear
from aminx.families.caliby.potts import CalibyGraphPotts


def _cat_neighbors_nodes(
  h_nodes: Float[Array, "L C"],
  h_neighbors: Float[Array, "L K E"],
  e_idx: Int[Array, "L K"],
) -> Float[Array, "L K E_plus_C"]:
  """Concatenate ``h_neighbors`` with gathered neighbor nodes ``h_nodes[e_idx]``."""
  return jnp.concatenate([h_neighbors, h_nodes[e_idx]], axis=-1)


class CalibyPositionWiseFFN(eqx.Module):
  """Two-layer position-wise feed-forward network with GELU activation.

  Attributes
  ----------
  W_in : eqx.nn.Linear
    Input projection ``num_hidden -> num_ff``.
  W_out : eqx.nn.Linear
    Output projection ``num_ff -> num_hidden``.
  """

  W_in: eqx.nn.Linear
  W_out: eqx.nn.Linear

  def __init__(self, num_hidden: int, num_ff: int, *, key: jax.Array | None = None) -> None:
    """Initialize ``CalibyPositionWiseFFN``.

    Parameters
    ----------
    num_hidden : int
      Input and output hidden dimension.
    num_ff : int
      Inner feed-forward dimension (typically ``4 * num_hidden``).
    key : jax.Array
      PRNG key for weight initialization.

    Returns
    -------
    None
      Initializes instance state.
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
    h_v : Float[Array, "... H"]
      Input hidden states.

    Returns
    -------
    Float[Array, "... H"]
      Transformed hidden states.
    """
    return _linear(self.W_out, jax.nn.gelu(_linear(self.W_in, h_v), approximate=False))


class CalibyEncLayer(eqx.Module):
  """Caliby bidirectional MPNN encoder layer updating both nodes ``h_V`` and edges ``h_E``.

  Attributes
  ----------
  W1 : eqx.nn.Linear
    Node message first linear ``3 * H -> H``.
  W2 : eqx.nn.Linear
    Node message second linear ``H -> H``.
  W3 : eqx.nn.Linear
    Node message third linear ``H -> H``.
  W11 : eqx.nn.Linear
    Edge update first linear ``3 * H -> H``.
  W12 : eqx.nn.Linear
    Edge update second linear ``H -> H``.
  W13 : eqx.nn.Linear
    Edge update third linear ``H -> H``.
  norm1 : eqx.nn.LayerNorm
    LayerNorm after node message residual.
  norm2 : eqx.nn.LayerNorm
    LayerNorm after node FFN residual.
  norm3 : eqx.nn.LayerNorm
    LayerNorm after edge message residual.
  dense : CalibyPositionWiseFFN
    Node position-wise FFN (``H -> 4*H -> H``).
  scale : float
    Message aggregation divisor (default ``30.0``).
  """

  W1: eqx.nn.Linear
  W2: eqx.nn.Linear
  W3: eqx.nn.Linear
  W11: eqx.nn.Linear
  W12: eqx.nn.Linear
  W13: eqx.nn.Linear
  norm1: eqx.nn.LayerNorm
  norm2: eqx.nn.LayerNorm
  norm3: eqx.nn.LayerNorm
  dense: CalibyPositionWiseFFN
  scale: float = eqx.field(static=True, default=30.0)

  def __init__(
    self,
    num_hidden: int = 128,
    num_in: int = 256,
    *,
    scale: float = 30.0,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``CalibyEncLayer``.

    Parameters
    ----------
    num_hidden : int, default=128
      Node and edge hidden dimension.
    num_in : int, default=256
      Sum of neighbor node + edge dimensions (``num_hidden + num_in = 384``).
    scale : float, default=30.0
      Neighbor message sum divisor.
    key : jax.Array
      PRNG key for weight initialization.

    Returns
    -------
    None
      Initializes instance state.
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
    self.dense = CalibyPositionWiseFFN(num_hidden, num_hidden * 4, key=keys[6])
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
    h_v : Float[Array, "L H"]
      Node hidden states ``(L, H)``.
    h_e : Float[Array, "L K H"]
      Edge hidden states ``(L, K, H)``.
    e_idx : Int[Array, "L K"]
      Neighbor index table ``(L, K)``.
    mask_v : Float[Array, " L"]
      Valid node mask ``(L,)``.
    mask_attend : Float[Array, "L K"]
      Valid edge attention mask ``(L, K)``.

    Returns
    -------
    tuple[Float[Array, "L H"], Float[Array, "L K H"]]
      Updated ``(h_v, h_e)``.
    """
    num_k = e_idx.shape[1]
    h_v_expand = jnp.broadcast_to(h_v[:, None, :], (*h_e.shape[:-1], h_v.shape[-1]))
    h_ev = _cat_neighbors_nodes(h_v, h_e, e_idx)
    h_ev = jnp.concatenate([h_v_expand, h_ev], axis=-1)

    h_message = _linear(
      self.W3,
      jax.nn.gelu(
        _linear(self.W2, jax.nn.gelu(_linear(self.W1, h_ev), approximate=False)),
        approximate=False,
      ),
    )
    h_message = mask_attend[..., None] * h_message
    dh = jnp.sum(h_message, axis=-2) / self.scale
    h_v = _layer_norm(self.norm1, h_v + dh)

    dh_ffn = self.dense(h_v)
    h_v = _layer_norm(self.norm2, h_v + dh_ffn)
    h_v = mask_v[:, None] * h_v

    # Edge update using updated h_v
    h_v_expand2 = jnp.broadcast_to(h_v[:, None, :], (h_v.shape[0], num_k, h_v.shape[-1]))
    h_ev_edge = _cat_neighbors_nodes(h_v, h_e, e_idx)
    h_ev_edge = jnp.concatenate([h_v_expand2, h_ev_edge], axis=-1)

    h_e_msg = _linear(
      self.W13,
      jax.nn.gelu(
        _linear(self.W12, jax.nn.gelu(_linear(self.W11, h_ev_edge), approximate=False)),
        approximate=False,
      ),
    )
    h_e = _layer_norm(self.norm3, h_e + h_e_msg)
    return h_v, h_e


class CalibyDecLayer(eqx.Module):
  """Caliby decoder layer updating node states ``h_V`` and optionally edge states ``h_E``.

  Attributes
  ----------
  W1 : eqx.nn.Linear
    First message linear ``num_hidden + num_in -> num_hidden``.
  W2 : eqx.nn.Linear
    Second message linear ``num_hidden -> num_hidden``.
  W3 : eqx.nn.Linear
    Third message linear ``num_hidden -> num_hidden``.
  norm1 : eqx.nn.LayerNorm
    LayerNorm after message residual.
  norm2 : eqx.nn.LayerNorm
    LayerNorm after FFN residual.
  dense : CalibyPositionWiseFFN
    Position-wise FFN (``H -> 4*H -> H``).
  W11 : eqx.nn.Linear | None
    Optional first edge-update linear ``2*num_hidden + num_in -> num_hidden``.
  W12 : eqx.nn.Linear | None
    Optional second edge-update linear ``num_hidden -> num_hidden``.
  W13 : eqx.nn.Linear | None
    Optional third edge-update linear ``num_hidden -> num_in``.
  norm3 : eqx.nn.LayerNorm | None
    Optional LayerNorm after edge update ``(num_in,)``.
  update_edges : bool
    Whether this decoder layer updates ``h_E`` via ``W11..W13, norm3``.
  scale : float
    Message aggregation divisor (default ``30.0``).
  """

  W1: eqx.nn.Linear
  W2: eqx.nn.Linear
  W3: eqx.nn.Linear
  norm1: eqx.nn.LayerNorm
  norm2: eqx.nn.LayerNorm
  dense: CalibyPositionWiseFFN
  W11: eqx.nn.Linear | None = None
  W12: eqx.nn.Linear | None = None
  W13: eqx.nn.Linear | None = None
  norm3: eqx.nn.LayerNorm | None = None
  update_edges: bool = eqx.field(static=True, default=False)
  scale: float = eqx.field(static=True, default=30.0)

  def __init__(
    self,
    num_hidden: int = 128,
    num_in: int = 256,
    *,
    update_edges: bool = False,
    scale: float = 30.0,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``CalibyDecLayer``.

    Parameters
    ----------
    num_hidden : int, default=128
      Node hidden dimension.
    num_in : int, default=256
      Message input dimension (``256`` or ``384``).
    update_edges : bool, default=False
      Whether to initialize ``W11, W12, W13, norm3`` for edge updates.
    scale : float, default=30.0
      Neighbor message sum divisor.
    key : jax.Array
      PRNG key for weight initialization.

    Returns
    -------
    None
      Initializes instance state.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 7)
    total_in = num_hidden + num_in
    self.W1 = eqx.nn.Linear(total_in, num_hidden, key=keys[0])
    self.W2 = eqx.nn.Linear(num_hidden, num_hidden, key=keys[1])
    self.W3 = eqx.nn.Linear(num_hidden, num_hidden, key=keys[2])
    self.norm1 = eqx.nn.LayerNorm(num_hidden)
    self.norm2 = eqx.nn.LayerNorm(num_hidden)
    self.dense = CalibyPositionWiseFFN(num_hidden, num_hidden * 4, key=keys[3])
    if update_edges:
      self.W11 = eqx.nn.Linear(num_hidden * 2 + num_in, num_hidden, key=keys[4])
      self.W12 = eqx.nn.Linear(num_hidden, num_hidden, key=keys[5])
      self.W13 = eqx.nn.Linear(num_hidden, num_in, key=keys[6])
      self.norm3 = eqx.nn.LayerNorm(num_in)
    else:
      self.W11 = None
      self.W12 = None
      self.W13 = None
      self.norm3 = None
    self.update_edges = update_edges
    self.scale = scale

  def __call__(
    self,
    h_v: Float[Array, "L H"],
    h_e: Float[Array, "L K M"],
    mask_v: Float[Array, " L"],
    mask_attend: Float[Array, "L K"] | None = None,
  ) -> Float[Array, "L H"]:
    """Run one node-only decoder layer step.

    Parameters
    ----------
    h_v : Float[Array, "L H"]
      Node hidden states ``(L, H)``.
    h_e : Float[Array, "L K M"]
      Combined edge + neighbor-node features ``(L, K, M)``.
    mask_v : Float[Array, " L"]
      Valid node mask ``(L,)``.
    mask_attend : Float[Array, "L K"] | None, optional
      Optional edge attention mask ``(L, K)``.

    Returns
    -------
    Float[Array, "L H"]
      Updated node hidden states ``(L, H)``.
    """
    h_v_expand = jnp.broadcast_to(h_v[:, None, :], (*h_e.shape[:-1], h_v.shape[-1]))
    h_ev = jnp.concatenate([h_v_expand, h_e], axis=-1)

    h_message = _linear(
      self.W3,
      jax.nn.gelu(
        _linear(self.W2, jax.nn.gelu(_linear(self.W1, h_ev), approximate=False)),
        approximate=False,
      ),
    )
    if mask_attend is not None:
      h_message = mask_attend[..., None] * h_message
    dh = jnp.sum(h_message, axis=-2) / self.scale
    h_v = _layer_norm(self.norm1, h_v + dh)

    dh_ffn = self.dense(h_v)
    h_v = _layer_norm(self.norm2, h_v + dh_ffn)
    return mask_v[:, None] * h_v

  def forward_with_edges(
    self,
    h_v: Float[Array, "L H"],
    h_e: Float[Array, "L K M"],
    e_idx: Int[Array, "L K"],
    mask_v: Float[Array, " L"],
    mask_attend: Float[Array, "L K"] | None = None,
  ) -> tuple[Float[Array, "L H"], Float[Array, "L K M"]]:
    """Run one decoder layer step with optional edge update when ``update_edges=True``.

    Parameters
    ----------
    h_v : Float[Array, "L H"]
      Input node representations.
    h_e : Float[Array, "L K M"]
      Input edge representations.
    e_idx : Int[Array, "L K"]
      kNN neighbor index table.
    mask_v : Float[Array, " L"]
      Valid token mask.
    mask_attend : Float[Array, "L K"] | None, optional
      Pairwise attention mask.

    Returns
    -------
    tuple[Float[Array, "L H"], Float[Array, "L K M"]]
      Updated ``(h_v, h_e)`` representations.
    """
    h_v = self(h_v, h_e, mask_v, mask_attend)
    if (
      self.update_edges
      and self.W11 is not None
      and self.W12 is not None
      and self.W13 is not None
      and self.norm3 is not None
    ):
      h_ev = _cat_neighbors_nodes(h_v, h_e, e_idx)
      h_v_expand = jnp.broadcast_to(h_v[:, None, :], (*h_ev.shape[:-1], h_v.shape[-1]))
      h_ev_full = jnp.concatenate([h_v_expand, h_ev], axis=-1)
      h_e_msg = _linear(
        self.W13,
        jax.nn.gelu(
          _linear(self.W12, jax.nn.gelu(_linear(self.W11, h_ev_full), approximate=False)),
          approximate=False,
        ),
      )
      h_e = _layer_norm(self.norm3, h_e + h_e_msg)
    return h_v, h_e


class CalibyMPNN(eqx.Module):
  """Caliby MPNN encoder-decoder stack.

  Supports both the synthetic test mode and the upstream ``CalibyMPNN`` mode
  (with ``W_s``, ``decoder_layers`` edge updates, and optional ``W_out``).

  Attributes
  ----------
  features : CalibyTokenFeatures
    Token and pairwise RBF featurizer.
  W_e : eqx.nn.Linear
    Initial edge projection ``edge_features -> hidden_dim``.
  W_s : eqx.nn.Embedding | None
    Optional sequence embedding ``(32, hidden_dim)`` when ``use_sequence=True``.
  W_out : eqx.nn.Linear | None
    Optional output projection ``(hidden_dim -> 32)``.
  encoder_layers : tuple[CalibyEncLayer, ...]
    Tuple of ``num_encoder_layers`` encoder blocks.
  decoder_layers : tuple[CalibyDecLayer, ...]
    Tuple of ``num_decoder_layers`` decoder blocks.
  use_sequence : bool
    Whether sequence tokens ``S`` are embedded and concatenated into decoder messages.
  update_decoder_edges : bool
    Whether decoder layers update ``h_ESV`` in place.
  return_decoder_edges : bool
    Whether to return the 384-channel ``h_ESV`` as ``h_e`` (for ``parameterization="factor"``).
  """

  features: CalibyTokenFeatures
  W_e: eqx.nn.Linear
  W_s: eqx.nn.Embedding | None
  encoder_layers: tuple[CalibyEncLayer, ...]
  decoder_layers: tuple[CalibyDecLayer, ...]
  W_out: eqx.nn.Linear | None = None
  use_sequence: bool = eqx.field(static=True, default=False)
  update_decoder_edges: bool = eqx.field(static=True, default=False)
  return_decoder_edges: bool = eqx.field(static=True, default=False)

  def __init__(
    self,
    node_features: int = 128,
    edge_features: int = 128,
    hidden_dim: int = 128,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 3,
    k_neighbors: int = 48,
    num_positional_embeddings: int = 16,
    num_rbf: int = 16,
    model_version: int = 0,
    *,
    use_sequence: bool = False,
    update_decoder_edges: bool = False,
    update_last_decoder_edge: bool = True,
    return_decoder_edges: bool = False,
    has_w_out: bool = False,
    vocab: int = 32,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``CalibyMPNN``.

    Parameters
    ----------
    node_features : int, default=128
      Node feature dimension.
    edge_features : int, default=128
      Edge feature dimension.
    hidden_dim : int, default=128
      Hidden layer dimension.
    num_encoder_layers : int, default=3
      Number of encoder layers.
    num_decoder_layers : int, default=3
      Number of decoder layers.
    k_neighbors : int, default=48
      Number of kNN neighbors.
    num_positional_embeddings : int, default=16
      Positional embedding dimension.
    num_rbf : int, default=16
      Number of RBF bins.
    model_version : int, default=0
      Feature layout version (``0`` or ``1``).
    use_sequence : bool, default=False
      Whether sequence embeddings ``W_s`` are included.
    update_decoder_edges : bool, default=False
      Whether decoder layers update ``h_ESV`` edges.
    update_last_decoder_edge : bool, default=True
      Whether the last decoder layer updates edges.
    return_decoder_edges : bool, default=False
      Whether ``__call__`` returns ``h_ESV`` as ``h_e``.
    has_w_out : bool, default=False
      Whether ``W_out`` is allocated.
    vocab : int, default=32
      Token vocabulary size.
    key : jax.Array | None, optional
      PRNG key for weight initialization.

    Returns
    -------
    None
      Initializes instance state.
    """
    if key is None:
      key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4 + num_encoder_layers + num_decoder_layers)
    self.features = CalibyTokenFeatures(
      edge_features=edge_features,
      node_features=node_features,
      num_positional_embeddings=num_positional_embeddings,
      num_rbf=num_rbf,
      top_k=k_neighbors,
      model_version=model_version,
      key=keys[0],
    )
    self.W_e = eqx.nn.Linear(edge_features, hidden_dim, key=keys[1])
    self.W_s = eqx.nn.Embedding(vocab, hidden_dim, key=keys[2]) if use_sequence else None
    self.W_out = eqx.nn.Linear(hidden_dim, vocab, key=keys[3]) if has_w_out else None
    self.use_sequence = use_sequence
    self.update_decoder_edges = update_decoder_edges
    self.return_decoder_edges = return_decoder_edges

    enc_keys = keys[4 : 4 + num_encoder_layers]
    dec_keys = keys[4 + num_encoder_layers :]
    self.encoder_layers = tuple(
      CalibyEncLayer(hidden_dim, hidden_dim * 2, key=k) for k in enc_keys
    )
    dec_num_in = hidden_dim * 3 if use_sequence else hidden_dim * 2
    self.decoder_layers = tuple(
      CalibyDecLayer(
        hidden_dim,
        dec_num_in,
        update_edges=(
          update_decoder_edges
          and (update_last_decoder_edge or i < num_decoder_layers - 1)
        ),
        key=k,
      )
      for i, k in enumerate(dec_keys)
    )

  def __call__(
    self,
    atom23_xyz: Float[Array, "L 23 3"],
    atom23_mask: Float[Array, "L 23"],
    token_mask: Float[Array, " L"],
    residue_idx: Int[Array, " L"],
    token_idx: Int[Array, " L"],
    asym_id: Int[Array, " L"],
    entity_id: Int[Array, " L"],
    sym_id: Int[Array, " L"],
    *,
    aatype: Int[Array, " L"] | None = None,
    molecule_type_encoding: Float[Array, "L 5"] | None = None,
    is_ligand: Bool[Array, " L"] | None = None,
  ) -> tuple[Float[Array, "L H"], Float[Array, "L K E"], Int[Array, "L K"], Float[Array, "L K"]]:
    """Run ``CalibyTokenFeatures``, encoder layers, and decoder layers.

    Parameters
    ----------
    atom23_xyz : Float[Array, "L 23 3"]
      Atom23 coordinates ``(L, 23, 3)``.
    atom23_mask : Float[Array, "L 23"]
      Atom23 presence mask ``(L, 23)``.
    token_mask : Float[Array, " L"]
      Valid token mask ``(L,)``.
    residue_idx : Int[Array, " L"]
      Residue sequence numbers ``(L,)``.
    token_idx : Int[Array, " L"]
      Token indices ``(L,)``.
    asym_id : Int[Array, " L"]
      Chain IDs ``(L,)``.
    entity_id : Int[Array, " L"]
      Entity IDs ``(L,)``.
    sym_id : Int[Array, " L"]
      Symmetry IDs ``(L,)``.
    aatype : Int[Array, " L"] | None, optional
      Optional sequence token indices ``(L,)``.
    molecule_type_encoding : Float[Array, "L 5"] | None, optional
      Optional molecule type one-hot encoding ``(L, 5)``.
    is_ligand : Bool[Array, " L"] | None, optional
      Optional ligand token indicator ``(L,)``.

    Returns
    -------
    tuple[Float[Array, "L H"], Float[Array, "L K E"], Int[Array, "L K"], Float[Array, "L K"]]
      Tuple ``(h_v, h_e, e_idx, mask_attend)``.
    """
    dtype = self.W_e.weight.dtype
    mask_v = token_mask.astype(dtype)
    v_init, e_init, e_idx = self.features(
      atom23_xyz=atom23_xyz,
      atom23_mask=atom23_mask,
      residue_idx=residue_idx,
      token_idx=token_idx,
      asym_id=asym_id,
      entity_id=entity_id,
      sym_id=sym_id,
      molecule_type_encoding=molecule_type_encoding,
      is_ligand=is_ligand,
    )
    h_v = v_init
    h_e = _linear(self.W_e, e_init)

    if self.use_sequence and self.W_s is not None:
      default_tok = self.W_s.num_embeddings - 1 if self.return_decoder_edges else 0
      seq_tokens = (
        jnp.full((h_v.shape[0],), default_tok, dtype=jnp.int32)
        if aatype is None
        else aatype
      )
      h_s = self.W_s.weight[jnp.clip(seq_tokens, 0, self.W_s.num_embeddings - 1)]
      if self.update_decoder_edges:
        h_v = h_v + h_s
    else:
      h_s = None

    mask_2d = mask_v[:, None] * mask_v[None, :]
    mask_attend = jnp.take_along_axis(mask_2d, e_idx, axis=1)

    for enc_layer in self.encoder_layers:
      h_v, h_e = enc_layer(h_v, h_e, e_idx, mask_v, mask_attend)

    if self.use_sequence and h_s is not None:
      h_es = _cat_neighbors_nodes(h_s, h_e, e_idx)
      if self.update_decoder_edges:
        h_esv = _cat_neighbors_nodes(h_v, h_es, e_idx)
        for dec_layer in self.decoder_layers:
          h_v, h_esv = dec_layer.forward_with_edges(h_v, h_esv, e_idx, mask_v, mask_attend)
        if self.return_decoder_edges:
          return h_v, h_esv, e_idx, mask_attend
      else:
        for dec_layer in self.decoder_layers:
          h_esv = _cat_neighbors_nodes(h_v, h_es, e_idx)
          h_v = dec_layer(h_v, h_esv, mask_v, mask_attend)
    else:
      for dec_layer in self.decoder_layers:
        h_ev = _cat_neighbors_nodes(h_v, h_e, e_idx)
        h_v = dec_layer(h_v, h_ev, mask_v, mask_attend)

    return h_v, h_e, e_idx, mask_attend


class CalibyPottsOutput(NamedTuple):
  """Output of :class:`CalibyModel` Potts forward pass.

  Attributes
  ----------
  h_32 : Float[Array, "L 32"]
    Full 32-token AF3 site fields.
  j_32 : Float[Array, "L L 32 32"]
    Full 32-token AF3 symmetric couplings.
  h_v : Float[Array, "L H"]
    Final decoder node representations.
  h_e : Float[Array, "L K E"]
    Final encoder/decoder edge representations.
  e_idx : Int[Array, "L K"]
    kNN neighbor indices.
  """

  h_32: Float[Array, "L 32"]
  j_32: Float[Array, "L L 32 32"]
  h_v: Float[Array, "L H"]
  h_e: Float[Array, "L K E"]
  e_idx: Int[Array, "L K"]


class CalibyModel(eqx.Module):
  """Top-level Caliby Potts model combining :class:`CalibyMPNN` and :class:`CalibyGraphPotts`.

  Attributes
  ----------
  mpnn : CalibyMPNN
    3-encoder + 3-decoder MPNN trunk.
  potts_decoder : CalibyGraphPotts
    Low-rank symmetric Potts parameter head.
  """

  mpnn: CalibyMPNN
  potts_decoder: CalibyGraphPotts

  def __init__(
    self,
    node_features: int = 128,
    edge_features: int = 128,
    hidden_dim: int = 128,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 3,
    k_neighbors: int = 48,
    num_positional_embeddings: int = 16,
    num_rbf: int = 16,
    model_version: int = 0,
    num_states: int = 32,
    num_factors: int = 64,
    *,
    parameterization: str = "factorized",
    use_sequence: bool = False,
    update_decoder_edges: bool = False,
    has_w_out: bool = False,
    has_bg: bool = False,
    bg_dim: int = 1,
    key: jax.Array | None = None,
  ) -> None:
    """Initialize ``CalibyModel``.

    Parameters
    ----------
    node_features : int, default=128
      Node feature dimension.
    edge_features : int, default=128
      Edge feature dimension.
    hidden_dim : int, default=128
      Hidden layer dimension.
    num_encoder_layers : int, default=3
      Number of encoder layers.
    num_decoder_layers : int, default=3
      Number of decoder layers.
    k_neighbors : int, default=48
      Number of kNN neighbors.
    num_positional_embeddings : int, default=16
      Positional embedding dimension.
    num_rbf : int, default=16
      Number of RBF bins.
    model_version : int, default=0
      Feature layout version (``0`` or ``1``).
    num_states : int, default=32
      Token vocabulary size.
    num_factors : int, default=32
      Number of low-rank Potts factors.
    parameterization : str, default="factorized"
      Potts parameterization (``"factor"`` or ``"factorized"``).
    use_sequence : bool, default=False
      Whether sequence embeddings ``W_s`` are included.
    update_decoder_edges : bool, default=False
      Whether decoder layers update ``h_ESV`` edges.
    has_w_out : bool, default=False
      Whether ``W_out`` is allocated.
    has_bg : bool, default=False
      Whether background Potts projections ``W_h_bg`` and ``W_J_bg`` are allocated.
    bg_dim : int, default=1
      Output dimension for ``W_h_bg`` and ``W_J_bg``.
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
    is_factor = parameterization == "factor"
    self.mpnn = CalibyMPNN(
      node_features=node_features,
      edge_features=edge_features,
      hidden_dim=hidden_dim,
      num_encoder_layers=num_encoder_layers,
      num_decoder_layers=num_decoder_layers,
      k_neighbors=k_neighbors,
      num_positional_embeddings=num_positional_embeddings,
      num_rbf=num_rbf,
      model_version=model_version,
      use_sequence=use_sequence or is_factor,
      update_decoder_edges=update_decoder_edges or is_factor,
      update_last_decoder_edge=True,
      return_decoder_edges=is_factor,
      has_w_out=has_w_out,
      key=k1,
    )
    self.potts_decoder = CalibyGraphPotts(
      d_node=hidden_dim,
      d_edge=hidden_dim * 3 if is_factor else hidden_dim,
      num_states=num_states,
      num_factors=num_factors,
      parameterization=parameterization,
      has_bg=has_bg,
      bg_dim=bg_dim,
      key=k2,
    )

  def __call__(
    self,
    atom23_xyz: Float[Array, "L 23 3"],
    atom23_mask: Float[Array, "L 23"],
    token_mask: Float[Array, " L"],
    residue_idx: Int[Array, " L"],
    token_idx: Int[Array, " L"],
    asym_id: Int[Array, " L"],
    entity_id: Int[Array, " L"],
    sym_id: Int[Array, " L"],
    *,
    molecule_type_encoding: Float[Array, "L 5"] | None = None,
    is_ligand: Bool[Array, " L"] | None = None,
  ) -> CalibyPottsOutput:
    """Compute 32-state Potts parameters ``(h_32, j_32)`` from 3D coordinates.

    Parameters
    ----------
    atom23_xyz : Float[Array, "L 23 3"]
      All-atom coordinates in AF3 23-atom representation.
    atom23_mask : Float[Array, "L 23"]
      Atom validity mask.
    token_mask : Float[Array, " L"]
      Token validity mask.
    residue_idx : Int[Array, " L"]
      Residue sequence numbers.
    token_idx : Int[Array, " L"]
      Token indices.
    asym_id : Int[Array, " L"]
      Chain IDs.
    entity_id : Int[Array, " L"]
      Entity IDs.
    sym_id : Int[Array, " L"]
      Symmetry IDs.
    molecule_type_encoding : Float[Array, "L 5"] | None, optional
      Optional molecule type one-hot encoding.
    is_ligand : Bool[Array, " L"] | None, optional
      Optional ligand token mask.

    Returns
    -------
    CalibyPottsOutput
      NamedTuple ``(h_32, j_32, h_v, h_e, e_idx)``.
    """
    h_v, h_e, e_idx, mask_attend = self.mpnn(
      atom23_xyz=atom23_xyz,
      atom23_mask=atom23_mask,
      token_mask=token_mask,
      residue_idx=residue_idx,
      token_idx=token_idx,
      asym_id=asym_id,
      entity_id=entity_id,
      sym_id=sym_id,
      molecule_type_encoding=molecule_type_encoding,
      is_ligand=is_ligand,
    )
    h_32, j_32 = self.potts_decoder(
      h_v=h_v,
      h_e=h_e,
      e_idx=e_idx,
      mask_i=token_mask.astype(h_v.dtype),
      mask_ij=mask_attend,
    )
    return CalibyPottsOutput(
      h_32=cast("Float[Array, 'L 32']", h_32),
      j_32=cast("Float[Array, 'L L 32 32']", j_32),
      h_v=h_v,
      h_e=h_e,
      e_idx=e_idx,
    )
