"""Ligand-conditioned MPNN (:class:`PrxteinLigandMPNN`) and ligand stack helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from aminx.model.capabilities import (
  PRXTEIN_LIGAND_MPNN_CAPABILITIES,
  ModelCapabilities,
)
from aminx.model.decoder import Decoder, DecoderLayer
from aminx.model.dropout import Dropout
from aminx.model.encoder import Encoder
from aminx.model.ligand_features import ProteinFeaturesLigand
from aminx.model.ligand_tiling import map_chunks_axis0, map_chunks_axis0_multi

if TYPE_CHECKING:
  from aminx.types.arrays import PRNGKeyArray


class PrxteinLigandMPNN(eqx.Module):
  """Ligand-conditioned protein sequence design model.

  Extends ProteinMPNN with ligand atom context via dual-encoder cross-attention.
  The ``context_encoder`` and ``y_context_encoder`` modules process ligand features
  in parallel and cross-attend to protein features.

  Parameters
  ----------
  features : ProteinFeaturesLigand
    Feature extraction module (backbone + ligand).
  encoder : Encoder
    Protein backbone encoder.
  decoder : Decoder
    Protein decoder.
  context_encoder : tuple[DecoderLayer, ...]
    Protein cross-attention layers attending to ligand features.
  y_context_encoder : tuple[DecoderLayer, ...]
    Ligand encoder layers processing ligand context.
  w_v : eqx.nn.Linear
    Projection for node features.
  w_c : eqx.nn.Linear
    Projection for protein context.
  w_nodes_y : eqx.nn.Linear
    Projection for ligand node features.
  w_edges_y : eqx.nn.Linear
    Projection for ligand edge features.
  v_c : eqx.nn.Linear
    Protein-ligand context fusion layer.
  v_c_norm : eqx.nn.LayerNorm
    Layer normalization after fusion.
  w_s_embed : eqx.nn.Embedding
    Sequence embedding (amino acid token to latent).
  w_out : eqx.nn.Linear
    Output logits projection.
  dropout : Dropout
    Dropout layer.
  node_features_dim : int
    Node feature dimension. Static (not a JAX array).
  edge_features_dim : int
    Edge feature dimension. Static (not a JAX array).
  hidden_features_dim : int
    Hidden layer dimension. Static (not a JAX array).
  num_decoder_layers : int
    Number of decoder layers. Static (not a JAX array).
  ligand_mpnn_use_side_chain_context : bool
    Whether to include side-chain context. Static (not a JAX array).
  capabilities : ModelCapabilities
    Model capabilities descriptor. Static (not a JAX array).

  References
  ----------
  .. [LigandMPNN] Dauparas, J., et al. "Atomic context-conditioned protein
     sequence design using LigandMPNN." *Nature Methods* 22(4):717-723 (2025).
     https://doi.org/10.1038/s41592-025-02626-1

  .. [LigandMPNN-code] Dauparas, J. LigandMPNN source code (commit 3870631).
     https://github.com/dauparas/LigandMPNN

  """

  features: ProteinFeaturesLigand
  encoder: Encoder
  decoder: Decoder

  context_encoder: tuple[DecoderLayer, ...]
  y_context_encoder: tuple[DecoderLayer, ...]

  w_v: eqx.nn.Linear
  w_c: eqx.nn.Linear
  w_nodes_y: eqx.nn.Linear
  w_edges_y: eqx.nn.Linear
  v_c: eqx.nn.Linear
  v_c_norm: eqx.nn.LayerNorm

  w_s_embed: eqx.nn.Embedding
  w_out: eqx.nn.Linear
  dropout: Dropout

  node_features_dim: int = eqx.field(static=True)
  edge_features_dim: int = eqx.field(static=True)
  hidden_features_dim: int = eqx.field(static=True)
  num_decoder_layers: int = eqx.field(static=True)
  ligand_mpnn_use_side_chain_context: bool = eqx.field(static=True)
  capabilities: ModelCapabilities = eqx.field(static=True, default=PRXTEIN_LIGAND_MPNN_CAPABILITIES)

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    hidden_features: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    k_neighbors: int,
    num_context_layers: int = 2,
    num_positional_embeddings: int = 16,
    num_amino_acids: int = 21,
    vocab_size: int = 21,
    dropout_rate: float = 0.1,
    ligand_l_chunk: int = 16,
    *,
    ligand_mpnn_use_side_chain_context: bool = False,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize PrxteinLigandMPNN model.

    Parameters
    ----------
    node_features : int
        Node feature dimension.
    edge_features : int
        Edge feature dimension.
    hidden_features : int
        Hidden layer dimension.
    num_encoder_layers : int
        Number of encoder layers.
    num_decoder_layers : int
        Number of decoder layers.
    k_neighbors : int
        Number of top-k neighbors for message passing.
    num_context_layers : int
        Number of ligand context encoder layers. Default: 2.
    num_positional_embeddings : int
        Dimension of positional embeddings. Default: 16.
    num_amino_acids : int
        Vocabulary size for amino acid tokens. Default: 21.
    vocab_size : int
        Output vocabulary size (should match num_amino_acids). Default: 21.
    dropout_rate : float
        Dropout rate. Default: 0.1.
    ligand_l_chunk : int
        Chunk size for ligand processing. Default: 16.
    ligand_mpnn_use_side_chain_context : bool
        Whether to include side-chain context. Default: False.
    key : PRNGKeyArray
        PRNG key for weight initialization.

    """
    keys = jax.random.split(key, 5)
    self.node_features_dim = node_features
    self.edge_features_dim = edge_features
    self.hidden_features_dim = hidden_features
    self.num_decoder_layers = num_decoder_layers
    self.ligand_mpnn_use_side_chain_context = ligand_mpnn_use_side_chain_context

    self.features = ProteinFeaturesLigand(
      node_features=node_features,
      edge_features=edge_features,
      k_neighbors=k_neighbors,
      num_positional_embeddings=num_positional_embeddings,
      use_side_chains=ligand_mpnn_use_side_chain_context,
      ligand_l_chunk=ligand_l_chunk,
      key=keys[0],
    )

    self.encoder = Encoder(
      node_features=node_features,
      edge_features=edge_features,
      hidden_features=hidden_features,
      num_layers=num_encoder_layers,
      dropout_rate=dropout_rate,
      key=keys[1],
    )

    self.decoder = Decoder(
      node_features=node_features,
      edge_features=edge_features,
      hidden_features=hidden_features,
      num_layers=num_decoder_layers,
      dropout_rate=dropout_rate,
      key=keys[2],
    )

    context_keys = jax.random.split(keys[3], num_context_layers)
    y_context_keys = jax.random.split(keys[4], num_context_layers)

    # Extra keys for projections
    proj_keys = jax.random.split(jax.random.fold_in(key, 100), 7)

    self.context_encoder = tuple(
      DecoderLayer(
        node_features,
        node_features * 2,
        hidden_features,
        dropout_rate=dropout_rate,
        key=k,
      )
      for k in context_keys
    )
    # y_context_encoder takes num_in = hidden_dim
    self.y_context_encoder = tuple(
      DecoderLayer(node_features, node_features, hidden_features, dropout_rate=dropout_rate, key=k)
      for k in y_context_keys
    )

    self.w_v = eqx.nn.Linear(node_features, node_features, key=proj_keys[0])
    self.w_c = eqx.nn.Linear(node_features, node_features, key=proj_keys[1])
    self.w_nodes_y = eqx.nn.Linear(node_features, node_features, key=proj_keys[2])
    self.w_edges_y = eqx.nn.Linear(node_features, node_features, key=proj_keys[3])
    self.v_c = eqx.nn.Linear(node_features, node_features, key=proj_keys[4])
    self.v_c_norm = eqx.nn.LayerNorm(node_features)

    self.dropout = Dropout(dropout_rate)

    self.w_s_embed = eqx.nn.Embedding(vocab_size, node_features, key=proj_keys[5])
    self.w_out = eqx.nn.Linear(node_features, num_amino_acids, key=proj_keys[6])

  @classmethod
  def stage_schema(cls) -> dict[str, type | None]:
    """Return the canonical stage names and type signatures for LigandMPNN pipeline.

    Returns
    -------
    dict[str, type | None]
        Mapping of stage name to expected type (e.g., "encode" → LigandEncodeFn).

    """
    from aminx.types.stages import (
      ARLogitTransformFn,
      ConditionalDecodeFn,
      EncoderStateFn,
      FeaturizeFn,
      LigandEncodeFn,
      LogitTransformFn,
      UnconditionalDecodeFn,
    )

    return {
      "featurize": FeaturizeFn,
      "encode": LigandEncodeFn,
      "decode": ConditionalDecodeFn | UnconditionalDecodeFn,
      "logit_transform": LogitTransformFn,
      "ar_logit_transform": ARLogitTransformFn,
      "encoder_state_fn": EncoderStateFn | None,
    }

  def __call__(
    self,
    coords: jax.Array,
    mask: jax.Array,
    residue_index: jax.Array,
    chain_index: jax.Array,
    *,
    prng_key: PRNGKeyArray | None = None,
    backbone_noise: float | jax.Array = 0.0,
    backbone_noise_mode: str = "direct",
    structure_mapping: jax.Array | None = None,
    y: jax.Array | None = None,
    y_t: jax.Array | None = None,
    y_m: jax.Array | None = None,
    atom_37: jax.Array | None = None,
    atom_37_mask: jax.Array | None = None,
    chain_mask: jax.Array | None = None,
    inference: bool = True,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Forward pass: featurize protein structure with ligand context, then encode.

    Parameters
    ----------
    coords : jax.Array
        Protein backbone + side-chain atom coordinates. Shape ``(L, 4, 3)``.
    mask : jax.Array
        Residue mask. Shape ``(L,)``.
    residue_index : jax.Array
        Residue sequence indices. Shape ``(L,)``.
    chain_index : jax.Array
        Chain assignment indices. Shape ``(L,)``.
    prng_key : PRNGKeyArray | None
        PRNG key for encoder stochasticity (optional). Default: PRNGKey(0).
    backbone_noise : float | jax.Array
        Backbone noise standard deviation. Default: 0.0.
    backbone_noise_mode : str
        Noise injection mode (not used by LigandMPNN features). Default: "direct".
    structure_mapping : jax.Array | None
        Optional structure isolation mask. Default: None.
    y : jax.Array | None
        Ligand atom coordinates. Shape ``(M, 4, 3)`` or None. Default: None.
    y_t : jax.Array | None
        Ligand atom type tokens. Shape ``(M, 4)`` or None. Default: None.
    y_m : jax.Array | None
        Ligand atom mask. Shape ``(M, 4)`` or None. Default: None.
    atom_37 : jax.Array | None
        Side-chain atom coordinates (full 37-atom set). Shape ``(L, 37, 3)`` or None.
        Only used when use_side_chains=True. Default: None.
    atom_37_mask : jax.Array | None
        Side-chain atom validity mask. Shape ``(L, 37)`` or None.
        Only used when use_side_chains=True. Default: None.
    chain_mask : jax.Array | None
        Chain mask to gate side chains (1=designable, 0=fixed). Shape ``(L,)`` or None.
        Only used when use_side_chains=True. Default: None.
    inference : bool
        Not used; accepted for API uniformity. Default: True.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Tuple of (node_features, edge_features, neighbor_indices):
        - node_features: Shape ``(L, D)`` — encoded residue features.
        - edge_features: Shape ``(L, K, D_edge)`` — neighbor edge context.
        - neighbor_indices: Shape ``(L, K)`` — neighbor indices.

    """
    if prng_key is None:
      prng_key = jax.random.PRNGKey(0)
    keys = jax.random.split(prng_key, 2)

    # Default ligand arrays to zero-context per residue when not provided.
    # The features function expects (L, atom_context_num, 3) — not (0, 4, 3).
    # A zero-length leading axis from the bundle's no-ligand default also lands here.
    if y is None or y.shape[0] == 0:
      n_res = coords.shape[0]
      n_ctx = self.features.atom_context_num
      y = jnp.zeros((n_res, n_ctx, 3))
      y_t = jnp.zeros((n_res, n_ctx), dtype=jnp.int32)
      y_m = jnp.zeros((n_res, n_ctx))

    V, E, E_idx, Y_nodes, Y_edges, Y_m_out = self.features(
      _key=keys[0],
      structure_coordinates=coords,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      ligand_coords=y,
      ligand_atom_types=y_t,
      ligand_mask=y_m,
      backbone_noise=backbone_noise,
      structure_mapping=structure_mapping,
      atom_37=atom_37,
      atom_37_mask=atom_37_mask,
      chain_mask=chain_mask,
    )

    h_V = jnp.zeros((E.shape[0], self.node_features_dim))
    h_E = E

    mask_2d = mask[:, None] * mask[None, :]
    mask_attend = jnp.take_along_axis(mask_2d, E_idx.astype(jnp.int32), axis=1)

    for layer in self.encoder.layers:
      h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend, inference=True)

    h_V_C = jax.vmap(self.w_c)(h_V)
    h_E_context = jax.vmap(jax.vmap(self.w_v))(V)

    lig_chunk = self.features.ligand_l_chunk
    Y_m_edges = Y_m_out[..., None] * Y_m_out[..., None, :]

    if lig_chunk <= 0:
      Y_nodes = jax.vmap(jax.vmap(self.w_nodes_y))(Y_nodes)
      Y_edges = jax.vmap(jax.vmap(jax.vmap(self.w_edges_y)))(Y_edges)

      for i in range(len(self.context_encoder)):
        Y_nodes = jax.vmap(
          lambda node, edge, mask_l, mask_e: self.y_context_encoder[i](
            node,
            edge,
            mask_l,
            attention_mask=mask_e,
            inference=True,
          ),
        )(Y_nodes, Y_edges, Y_m_out, Y_m_edges)

        h_E_context_cat = jnp.concatenate([h_E_context, Y_nodes], axis=-1)
        h_V_C = self.context_encoder[i](
          h_V_C,
          h_E_context_cat,
          mask,
          attention_mask=Y_m_out,
          inference=True,
        )
    else:
      Y_nodes = map_chunks_axis0(
        Y_nodes,
        chunk_size=lig_chunk,
        fn=lambda s: jax.vmap(jax.vmap(self.w_nodes_y))(s),
      )
      Y_edges = map_chunks_axis0(
        Y_edges,
        chunk_size=lig_chunk,
        fn=lambda s: jax.vmap(jax.vmap(jax.vmap(self.w_edges_y)))(s),
      )

      for i in range(len(self.context_encoder)):
        y_layer = self.y_context_encoder[i]
        ctx_layer = self.context_encoder[i]

        def slab_fn(
          Yn: jax.Array,
          Ye: jax.Array,
          Ymm: jax.Array,
          Yme: jax.Array,
          hv: jax.Array,
          hec: jax.Array,
          msk: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
          Yn_out = jax.vmap(
            lambda node, edge, mask_l, mask_e: y_layer(
              node,
              edge,
              mask_l,
              attention_mask=mask_e,
              inference=True,
            ),
          )(Yn, Ye, Ymm, Yme)
          he_cat = jnp.concatenate([hec, Yn_out], axis=-1)
          hv_out = ctx_layer(hv, he_cat, msk, attention_mask=Ymm, inference=True)
          return Yn_out, hv_out

        Y_nodes, h_V_C = map_chunks_axis0_multi(
          slab_fn,
          lig_chunk,
          (Y_nodes, Y_edges, Y_m_out, Y_m_edges, h_V_C, h_E_context, mask),
        )

    h_V_C = jax.vmap(self.v_c)(h_V_C)
    h_V = h_V + jax.vmap(self.v_c_norm)(self.dropout(h_V_C, key=keys[1], inference=True))

    return h_V, h_E, E_idx
