"""LASErMPNN GATv2, GVP, and equivariant layer-norm.

Upstream (``utils/model_generics.py`` at ``e70f2c6d``) stores edges as lists and
reduces them with ``scatter`` / ``scatter_softmax``. These modules take a dense
padded neighbour axis ``(N, K)`` plus a boolean mask. Padded slots do not enter
the softmax or the sum.

Math:
    $$a_{ik} = \\mathrm{softmax}_{k \\in \\mathcal{N}(i)}(\\ell_{ik})$$
    $$a_{ik} = 0 \\text{ when } \\mathcal{N}(i) = \\emptyset$$
    $$h_i = \\sum_k m_{ik} a_{ik} v_{ik}$$

Pseudocode:
    masked = where(mask, logits, -inf)
    shift = where(isfinite(max(masked)), max(masked), 0)
    exp = where(mask, exp(logits - shift), 0)
    weights = where(sum(exp) > 0, exp / sum(exp), 0)
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

_NORM_EPS = 1e-8
_LEAKY_SLOPE = 0.2
_XAVIER_FAN_OUT = 1


def norm_no_nan(
  values: Float[Array, "..."],
  axis: int,
  *,
  keepdims: bool = False,
  sqrt: bool = True,
) -> Float[Array, "..."]:
  """L2 norm clamped above ``1e-8``, matching upstream ``_norm_no_nan``.

  ``sqrt=False`` returns the clamped sum of squares.
  """
  squared = jnp.sum(jnp.square(values), axis=axis, keepdims=keepdims)
  clamped = jnp.clip(squared, min=_NORM_EPS)
  if sqrt:
    return jnp.sqrt(clamped)
  return clamped


def masked_softmax(
  logits: Float[Array, "n k *c"],
  mask: Bool[Array, "n k"],
) -> Float[Array, "n k *c"]:
  """Softmax over the neighbour axis. An empty row (no valid ``k``) is 0.

  ``mask`` is broadcast over every axis after ``K``. Fully masked rows do not
  subtract ``-inf`` from itself, so the result is 0 rather than NaN. A zero-width
  neighbour axis is empty for every row.
  """
  if logits.shape[1] == 0:
    return jnp.zeros_like(logits)
  mask_b = jnp.reshape(mask, mask.shape + (1,) * (logits.ndim - mask.ndim))
  filled = jnp.where(mask_b, logits, -jnp.inf)
  row_max = jnp.max(filled, axis=1, keepdims=True)
  row_max = jnp.where(jnp.isfinite(row_max), row_max, jnp.zeros_like(row_max))
  shifted = jnp.where(mask_b, logits - row_max, jnp.zeros_like(logits))
  exp = jnp.where(mask_b, jnp.exp(shifted), jnp.zeros_like(logits))
  denom = jnp.sum(exp, axis=1, keepdims=True)
  safe = jnp.where(denom > 0, denom, jnp.ones_like(denom))
  return jnp.where(denom > 0, exp / safe, jnp.zeros_like(exp))


def apply_linear(layer: eqx.nn.Linear, features: Float[Array, "... f"]) -> Float[Array, "... o"]:
  """Batched ``x @ W.T + b`` without ``vmap``.

  ``eqx.nn.Linear`` contracts ``weight @ x`` and only accepts a single vector.
  PyTorch ``nn.Linear`` broadcasts over every leading axis; the weight layout
  ``(out, in)`` is the same.
  """
  transformed = jnp.matmul(features, jnp.swapaxes(layer.weight, -1, -2))
  if layer.bias is None:
    return transformed
  return transformed + layer.bias


def apply_layer_norm(
  layer: eqx.nn.LayerNorm,
  features: Float[Array, "... f"],
) -> Float[Array, "... f"]:
  """PyTorch ``LayerNorm`` over the last axis, no ``vmap``.

  Every norm in this package is constructed as ``LayerNorm(width)``, so the
  normalised axis is the feature axis.
  """
  mean = jnp.mean(features, axis=-1, keepdims=True)
  variance = jnp.var(features, axis=-1, keepdims=True)
  inv = jax.lax.rsqrt(variance + layer.eps)
  normalised = (features - mean) * inv
  if layer.weight is not None:
    normalised = normalised * layer.weight
  if layer.bias is not None:
    normalised = normalised + layer.bias
  return normalised


def _as_vectors(vectors: Float[Array, "n *v 3"]) -> Float[Array, "n v 3"]:
  """Match ``EquivariantData.to_tuple``: a single ``(N, 3)`` vector is ``V=1``."""
  if vectors.ndim == 2:
    return vectors[:, None, :]
  return vectors


def _linear_at(layers: tuple[eqx.nn.Linear | None, ...], index: int) -> eqx.nn.Linear:
  layer = layers[index]
  if not isinstance(layer, eqx.nn.Linear):
    msg = f"expected Linear at sequential index {index}"
    raise TypeError(msg)
  return layer


def _node_pair(dim: int | tuple[int, int]) -> tuple[int, int]:
  if isinstance(dim, int):
    return dim, dim
  return dim


def _output_dim(dim: int | tuple[int, int], output: int | None) -> int:
  if isinstance(dim, int):
    if output is not None:
      msg = "output_node_embedding_dim is only valid for a pair of input dims"
      raise ValueError(msg)
    return dim
  if output is None:
    msg = "output_node_embedding_dim is required when node dims differ"
    raise ValueError(msg)
  return output


class DenseMLP(eqx.Module):
  """Three-linear MLP matching ``nn.Sequential`` indices 0, 3, and 6.

  Dropout slots (1, 4) and GELU slots (2, 5) have no parameters. Their
  ``mlp_dropout`` is 0 in upstream ``DenseMLP``.
  """

  layers: tuple[eqx.nn.Linear | None, ...]
  # Static so a filter_jit rebuild keeps it. A plain attribute is dropped, and
  # the dropout replay then misses every recorded mask.
  proofread_path: str = eqx.field(static=True)

  def __init__(
    self,
    input_dim: int,
    latent_dim: int,
    output_dim: int,
    *,
    key: PRNGKeyArray,
  ) -> None:
    keys = jax.random.split(key, 3)
    self.layers = (
      eqx.nn.Linear(input_dim, latent_dim, key=keys[0]),
      None,
      None,
      eqx.nn.Linear(latent_dim, latent_dim, key=keys[1]),
      None,
      None,
      eqx.nn.Linear(latent_dim, output_dim, key=keys[2]),
    )
    self.proofread_path = ""

  def __call__(self, features: Float[Array, "*batch f"]) -> Float[Array, "*batch o"]:
    hidden = jax.nn.gelu(apply_linear(_linear_at(self.layers, 0), features), approximate=False)
    hidden = jax.nn.gelu(apply_linear(_linear_at(self.layers, 3), hidden), approximate=False)
    return apply_linear(_linear_at(self.layers, 6), hidden)


class DenseResidualNodeUpdate(eqx.Module):
  """ProteinMPNN residual node update. Sequential linears sit at indices 0 and 2."""

  node_norm1: eqx.nn.LayerNorm
  layers: tuple[eqx.nn.Linear | None, ...]
  node_norm2: eqx.nn.LayerNorm

  def __init__(self, input_dim: int, *, key: PRNGKeyArray) -> None:
    keys = jax.random.split(key, 2)
    self.node_norm1 = eqx.nn.LayerNorm(input_dim)
    self.layers = (
      eqx.nn.Linear(input_dim, 2 * input_dim, key=keys[0]),
      None,
      eqx.nn.Linear(2 * input_dim, input_dim, key=keys[1]),
    )
    self.node_norm2 = eqx.nn.LayerNorm(input_dim)

  def __call__(
    self,
    prev: Float[Array, "n h"],
    update: Float[Array, "n h"],
  ) -> Float[Array, "n h"]:
    """Eval dropout is the identity, matching an ``nn.Dropout`` in eval mode."""
    hidden = apply_layer_norm(self.node_norm1, prev + update)
    delta = apply_linear(_linear_at(self.layers, 0), hidden)
    delta = jax.nn.gelu(delta, approximate=False)
    delta = apply_linear(_linear_at(self.layers, 2), delta)
    return apply_layer_norm(self.node_norm2, hidden + delta)


class AttentionAggregation(eqx.Module):
  """Mean over heads when ``atten_head_aggr_layers`` is 0 (every LASEr checkpoint)."""

  linears: tuple[eqx.nn.Linear, ...]

  def __init__(
    self,
    n_layers: int,
    node_dim: int,
    num_heads: int,
    *,
    key: PRNGKeyArray,
  ) -> None:
    if n_layers <= 0:
      self.linears = ()
      return
    keys = jax.random.split(key, n_layers)
    built: list[eqx.nn.Linear] = [
      eqx.nn.Linear(node_dim * num_heads, node_dim, key=keys[0]),
    ]
    built.extend(eqx.nn.Linear(node_dim, node_dim, key=keys[i]) for i in range(1, n_layers))
    self.linears = tuple(built)

  def __call__(self, update: Float[Array, "n heads d"]) -> Float[Array, "n d"]:
    if len(self.linears) == 0:
      return jnp.mean(update, axis=1)
    flat = jnp.reshape(update, (update.shape[0], update.shape[1] * update.shape[2]))
    started = False
    out = flat
    for linear in self.linears:
      if started:
        out = jax.nn.gelu(out, approximate=False)
      out = apply_linear(linear, out)
      started = True
    return out


class GVP(eqx.Module):
  """Geometric vector perceptron (drorlab GVP), vector gate optional.

  Absent linears are ``None``: no scalar channels omits ``ws`` / ``wsv``, and
  ``vectors_out == 0`` omits ``wv`` / ``wsv`` even when ``vector_gate`` is set.
  """

  wh: eqx.nn.Linear | None
  ws: eqx.nn.Linear | None
  wv: eqx.nn.Linear | None
  wsv: eqx.nn.Linear | None
  scalar_in: int = eqx.field(static=True)
  vector_in: int = eqx.field(static=True)
  scalar_out: int = eqx.field(static=True)
  vector_out: int = eqx.field(static=True)
  vector_gate: bool = eqx.field(static=True)
  apply_scalar_act: bool = eqx.field(static=True)

  def __init__(
    self,
    in_dims: tuple[int, int],
    out_dims: tuple[int, int],
    *,
    vector_gate: bool = False,
    key: PRNGKeyArray,
  ) -> None:
    si, vi = in_dims
    so, vo = out_dims
    self.scalar_in = si
    self.vector_in = vi
    self.scalar_out = so
    self.vector_out = vo
    self.vector_gate = vector_gate
    self.apply_scalar_act = bool(si and so)
    keys = jax.random.split(key, 4)
    self.wh = None
    self.ws = None
    self.wv = None
    self.wsv = None
    if vi:
      h_dim = max(vi, vo)
      self.wh = eqx.nn.Linear(vi, h_dim, use_bias=False, key=keys[0])
      if si and so:
        self.ws = eqx.nn.Linear(h_dim + si, so, key=keys[1])
      if vo:
        self.wv = eqx.nn.Linear(h_dim, vo, use_bias=False, key=keys[2])
        if vector_gate:
          self.wsv = eqx.nn.Linear(so, vo, key=keys[3])
    else:
      self.ws = eqx.nn.Linear(si, so, key=keys[1])

  def __call__(
    self,
    scalars: Float[Array, "n s"],
    vectors: Float[Array, "n *v 3"],
  ) -> tuple[Float[Array, "n so"], Float[Array, "n vo 3"]]:
    """Return ``(scalars, vectors)``. Scalar GELU runs after the vector gate."""
    vectors = _as_vectors(vectors)
    if self.wh is None or self.vector_in == 0:
      if self.ws is None:
        msg = "GVP has no scalar linear"
        raise RuntimeError(msg)
      scalars = apply_linear(self.ws, scalars)
      vectors_out = jnp.zeros(
        (scalars.shape[0], self.vector_out, 3),
        dtype=scalars.dtype,
      )
    else:
      spatial = jnp.swapaxes(vectors, -1, -2)
      vh = apply_linear(self.wh, spatial)
      vn = norm_no_nan(vh, -2, keepdims=False, sqrt=True)
      if self.ws is not None:
        scalars = apply_linear(self.ws, jnp.concatenate((scalars, vn), axis=-1))
      if self.wv is None:
        vectors_out = jnp.zeros(
          (scalars.shape[0], self.vector_out, 3),
          dtype=scalars.dtype,
        )
      else:
        vectors_out = jnp.swapaxes(apply_linear(self.wv, vh), -1, -2)
        if self.vector_gate:
          if self.wsv is None:
            msg = "vector gate requires wsv"
            raise RuntimeError(msg)
          gate = jax.nn.sigmoid(apply_linear(self.wsv, jax.nn.sigmoid(scalars)))
          vectors_out = vectors_out * gate[..., None]
        else:
          vectors_out = vectors_out * jax.nn.sigmoid(
            norm_no_nan(vectors_out, -1, keepdims=True, sqrt=True),
          )
    if self.apply_scalar_act:
      scalars = jax.nn.gelu(scalars, approximate=False)
    return scalars, vectors_out


class EquivariantLayerNorm(eqx.Module):
  """Layer-norm scalars and divide vectors by the RMS of their norms.

  ``vector_only`` leaves scalars untouched and has no ``scalar_norm`` parameters.
  """

  scalar_norm: eqx.nn.LayerNorm | None
  vector_only: bool = eqx.field(static=True)
  vector_channels: int = eqx.field(static=True)

  def __init__(self, dims: tuple[int, int], *, vector_only: bool = False) -> None:
    scalar_dim, vector_channels = dims
    self.vector_only = vector_only
    self.vector_channels = vector_channels
    self.scalar_norm = None if vector_only else eqx.nn.LayerNorm(scalar_dim)

  def __call__(
    self,
    scalars: Float[Array, "n s"],
    vectors: Float[Array, "n *v 3"],
  ) -> tuple[Float[Array, "n s"], Float[Array, "n v 3"]]:
    vectors = _as_vectors(vectors)
    if self.scalar_norm is not None:
      scalars = apply_layer_norm(self.scalar_norm, scalars)
    if self.vector_channels == 0:
      return scalars, vectors
    squared = norm_no_nan(vectors, -1, keepdims=True, sqrt=False)
    scale = jnp.sqrt(jnp.mean(squared, axis=-2, keepdims=True))
    return scalars, vectors / scale


class _VDropout(eqx.Module):
  """Vector-channel dropout. ``dummy_param`` is the empty upstream Parameter."""

  drop_rate: float = eqx.field(static=True)
  dummy_param: Float[Array, "0"]

  def __init__(self, drop_rate: float) -> None:
    self.drop_rate = drop_rate
    self.dummy_param = jnp.zeros((0,))


class EquivariantDropout(eqx.Module):
  """Combined scalar and vector dropout. Eval (``inference=True``) is the identity."""

  vdropout: _VDropout
  # Static so a filter_jit rebuild keeps it. A plain attribute is dropped, and
  # the dropout replay then misses every recorded mask.
  proofread_path: str = eqx.field(static=True)

  def __init__(self, drop_rate: float) -> None:
    self.vdropout = _VDropout(drop_rate)
    self.proofread_path = ""

  def __call__(
    self,
    scalars: Float[Array, "n s"],
    vectors: Float[Array, "n v 3"],
    *,
    inference: bool = True,
  ) -> tuple[Float[Array, "n s"], Float[Array, "n v 3"]]:
    if inference or self.vdropout.drop_rate == 0.0:
      return scalars, vectors
    msg = "LASEr equivariant dropout at train time is not used by the layer port"
    raise RuntimeError(msg)


class HomoGATv2(eqx.Module):
  """Single-node-type GATv2 over a dense ``(N, K)`` neighbourhood."""

  gatW: eqx.nn.Linear  # noqa: N815  upstream Parameter name
  gatA: Float[Array, "heads atten 1"]  # noqa: N815  upstream Parameter name
  final_atten_aggr: AttentionAggregation
  dense_node_update_layers: DenseMLP | None
  linear_edge_updates: DenseMLP | None
  edge_norm: eqx.nn.LayerNorm | None
  vectors_update_layer: GVP
  equivariant_layer_norm: EquivariantLayerNorm
  num_heads: int = eqx.field(static=True)
  atten_dim: int = eqx.field(static=True)
  node_dim: int = eqx.field(static=True)
  source_dim: int = eqx.field(static=True)
  sink_dim: int = eqx.field(static=True)
  edge_dim: int = eqx.field(static=True)
  update_edges: bool = eqx.field(static=True)
  use_mlp_node_update: bool = eqx.field(static=True)
  dropout_rate: float = eqx.field(static=True)
  # Static so a filter_jit rebuild keeps it. A plain attribute is dropped, and
  # the dropout replay then misses every recorded mask.
  proofread_path: str = eqx.field(static=True)

  def __init__(
    self,
    node_embedding_dim: int | tuple[int, int],
    edge_embedding_dim: int,
    num_attention_heads: int,
    dropout: float,
    *,
    update_edges: bool,
    use_mlp_node_update: bool,
    atten_head_aggr_layers: int,
    num_vectors: int,
    atten_dimension_upscale_factor: int | None,
    output_node_embedding_dim: int | None = None,
    key: PRNGKeyArray,
  ) -> None:
    if num_attention_heads <= 0:
      msg = "num_attention_heads must be positive"
      raise ValueError(msg)
    source_dim, sink_dim = _node_pair(node_embedding_dim)
    node_dim = _output_dim(node_embedding_dim, output_node_embedding_dim)
    if use_mlp_node_update:
      if atten_dimension_upscale_factor is None:
        msg = "atten_dimension_upscale_factor is required when use_mlp_node_update is set"
        raise ValueError(msg)
      atten_dim = atten_dimension_upscale_factor * node_dim
    else:
      atten_dim = node_dim
    self.num_heads = num_attention_heads
    self.atten_dim = atten_dim
    self.node_dim = node_dim
    self.source_dim = source_dim
    self.sink_dim = sink_dim
    self.edge_dim = edge_embedding_dim
    self.update_edges = update_edges
    self.use_mlp_node_update = use_mlp_node_update
    self.dropout_rate = dropout
    keys = jax.random.split(key, 8)
    feat_dim = source_dim + sink_dim + edge_embedding_dim
    self.gatW = eqx.nn.Linear(
      feat_dim,
      num_attention_heads * atten_dim,
      use_bias=False,
      key=keys[0],
    )
    fan = atten_dim + _XAVIER_FAN_OUT
    limit = (6.0 / fan) ** 0.5
    self.gatA = jax.random.uniform(
      keys[1],
      (num_attention_heads, atten_dim, 1),
      minval=-limit,
      maxval=limit,
    )
    self.final_atten_aggr = AttentionAggregation(
      atten_head_aggr_layers,
      node_dim,
      num_attention_heads,
      key=keys[2],
    )
    self.dense_node_update_layers = None
    if use_mlp_node_update:
      self.dense_node_update_layers = DenseMLP(feat_dim, node_dim, node_dim, key=keys[3])
    self.linear_edge_updates = None
    self.edge_norm = None
    if update_edges:
      self.linear_edge_updates = DenseMLP(
        feat_dim,
        edge_embedding_dim,
        edge_embedding_dim,
        key=keys[4],
      )
      self.edge_norm = eqx.nn.LayerNorm(edge_embedding_dim)
    self.vectors_update_layer = GVP(
      (node_dim, num_vectors),
      (node_dim, num_vectors),
      vector_gate=True,
      key=keys[5],
    )
    self.equivariant_layer_norm = EquivariantLayerNorm((node_dim, num_vectors))
    self.proofread_path = ""

  def messages(
    self,
    source: Float[Array, "n k fs"],
    sink: Float[Array, "n k fh"],
    edge_attr: Float[Array, "n k e"],
  ) -> tuple[Float[Array, "n k heads d"], Float[Array, "n k heads"]]:
    """Per-edge values and pre-softmax scores. No neighbour reduction."""
    features = jnp.concatenate((source, edge_attr, sink), axis=-1)
    projected = apply_linear(self.gatW, features)
    projected = jnp.reshape(projected, (*projected.shape[:-1], self.num_heads, self.atten_dim))
    activated = jax.nn.leaky_relu(projected, negative_slope=_LEAKY_SLOPE)
    scores = jnp.einsum("nkha,hac->nkh", activated, self.gatA)
    values = projected
    if self.dense_node_update_layers is not None:
      updated = self.dense_node_update_layers(features)
      values = jnp.repeat(updated[:, :, None, :], self.num_heads, axis=2)
    return values, scores

  def edge_update(
    self,
    source: Float[Array, "n k fs"],
    sink: Float[Array, "n k fh"],
    edge_attr: Float[Array, "n k e"],
    mask: Bool[Array, "n k"],
  ) -> Float[Array, "n k e"]:
    """Dense edge MLP. ``update_edges=False`` returns ``edge_attr`` unchanged."""
    if self.linear_edge_updates is None or self.edge_norm is None:
      return edge_attr
    features = jnp.concatenate((source, edge_attr, sink), axis=-1)
    # Upstream applies ``self.dropout`` here. Eval mode is the identity for any p.
    delta = self.linear_edge_updates(features)
    updated = apply_layer_norm(self.edge_norm, edge_attr + delta)
    return jnp.where(mask[..., None], updated, jnp.zeros_like(updated))

  def __call__(
    self,
    scalars: Float[Array, "n h"],
    vectors: Float[Array, "n v 3"],
    edge_attr: Float[Array, "n k e"],
    neighbours: Int[Array, "n k"],
    mask: Bool[Array, "n k"],
    *,
    inference: bool = True,
  ) -> tuple[Float[Array, "n h"], Float[Array, "n v 3"], Float[Array, "n k e"]]:
    """One homogeneous step. ``inference`` keeps attention dropout as the identity."""
    del inference
    source = scalars[neighbours]
    sink = jnp.broadcast_to(
      scalars[:, None, :],
      (scalars.shape[0], neighbours.shape[1], scalars.shape[-1]),
    )
    values, scores = self.messages(source, sink, edge_attr)
    weights = masked_softmax(scores, mask)
    mixed = values * weights[..., None]
    mixed = jnp.where(mask[..., None, None], mixed, jnp.zeros_like(mixed))
    pooled = jnp.sum(mixed, axis=1)
    updated = self.final_atten_aggr(pooled)
    scalars_out, vectors_out = self.equivariant_layer_norm(
      *self.vectors_update_layer(updated, vectors),
    )
    source_out = scalars_out[neighbours]
    sink_out = jnp.broadcast_to(
      scalars_out[:, None, :],
      (scalars_out.shape[0], neighbours.shape[1], scalars_out.shape[-1]),
    )
    edges_out = self.edge_update(source_out, sink_out, edge_attr, mask)
    return scalars_out, vectors_out, edges_out


class HeteroGATv2(eqx.Module):
  """GATv2 over several node types into one sink type.

  Subgraphs are concatenated on ``K`` and softmax-normalised jointly, which is
  the dense form of ``scatter_softmax`` on the concatenated edge list. Each
  subgraph's own ``HomoGATv2`` is built with ``update_edges=False``, so edge
  attributes pass through unchanged, matching upstream ``HeteroGATv2``.
  """

  subgats: tuple[HomoGATv2, ...]
  dense_residual_node_update: DenseResidualNodeUpdate | None
  final_atten_aggr: AttentionAggregation
  vectors_update_layer: GVP
  equivariant_layer_norm: EquivariantLayerNorm
  use_residual_node_update: bool = eqx.field(static=True)
  compute_edge_updates: bool = eqx.field(static=True)
  num_vectors: int = eqx.field(static=True)
  node_dim: int = eqx.field(static=True)
  # Static so a filter_jit rebuild keeps it. A plain attribute is dropped, and
  # the dropout replay then misses every recorded mask.
  proofread_path: str = eqx.field(static=True)

  def __init__(
    self,
    num_subgraphs: int,
    node_embedding_dim: int | tuple[int | tuple[int, int], ...],
    edge_embedding_dim: int | tuple[int, ...],
    atten_head_aggr_layers: int,
    num_attention_heads: int,
    dropout: float,
    *,
    num_vectors: int,
    use_mlp_node_update: bool,
    use_residual_node_update: bool,
    compute_edge_updates: bool,
    output_node_embedding_dim: int | None = None,
    output_edge_embedding_dim: int | None = None,
    atten_dimension_upscale_factor: int | None = None,
    key: PRNGKeyArray,
  ) -> None:
    del output_edge_embedding_dim
    if isinstance(node_embedding_dim, int):
      node_dim = node_embedding_dim
    else:
      if output_node_embedding_dim is None:
        msg = "output_node_embedding_dim is required for a per-subgraph node dim"
        raise ValueError(msg)
      node_dim = output_node_embedding_dim
    self.node_dim = node_dim
    self.num_vectors = num_vectors
    self.use_residual_node_update = use_residual_node_update
    self.compute_edge_updates = compute_edge_updates
    keys = jax.random.split(key, num_subgraphs + 3)
    built: list[HomoGATv2] = []
    for index in range(num_subgraphs):
      if isinstance(node_embedding_dim, int):
        node_i: int | tuple[int, int] = node_embedding_dim
        out_i = None
      else:
        raw = node_embedding_dim[index]
        node_i = raw if isinstance(raw, int) else (int(raw[0]), int(raw[1]))
        out_i = None if isinstance(node_i, int) else node_dim
      if isinstance(edge_embedding_dim, int):
        edge_i = edge_embedding_dim
      else:
        edge_i = edge_embedding_dim[index]
      built.append(
        HomoGATv2(
          node_i,
          edge_i,
          num_attention_heads,
          dropout,
          update_edges=False,
          use_mlp_node_update=use_mlp_node_update,
          atten_head_aggr_layers=0,
          num_vectors=num_vectors,
          atten_dimension_upscale_factor=atten_dimension_upscale_factor,
          output_node_embedding_dim=out_i,
          key=keys[index],
        ),
      )
    self.subgats = tuple(built)
    self.dense_residual_node_update = None
    if use_residual_node_update:
      self.dense_residual_node_update = DenseResidualNodeUpdate(node_dim, key=keys[-3])
    self.final_atten_aggr = AttentionAggregation(
      atten_head_aggr_layers,
      node_dim,
      num_attention_heads,
      key=keys[-2],
    )
    self.vectors_update_layer = GVP(
      (node_dim, num_vectors),
      (node_dim, num_vectors),
      vector_gate=True,
      key=keys[-1],
    )
    self.equivariant_layer_norm = EquivariantLayerNorm((node_dim, num_vectors), vector_only=True)
    self.proofread_path = ""

  def __call__(
    self,
    sink_scalars: Float[Array, "n h"],
    sink_vectors: Float[Array, "n v 3"],
    source_scalars: tuple[Float[Array, "ns hs"], ...],
    neighbours: tuple[Int[Array, "n k"], ...],
    edge_attr: tuple[Float[Array, "n k e"], ...],
    mask: tuple[Bool[Array, "n k"], ...],
    source_is_sink: tuple[bool, ...],
    *,
    inference: bool = True,
  ) -> tuple[Float[Array, "n h"], Float[Array, "n v 3"], tuple[Float[Array, "n k e"], ...]]:
    """Joint attention over every subgraph, then one GVP and vector-only norm."""
    del inference
    values: list[Float[Array, "n k heads d"]] = []
    scores: list[Float[Array, "n k heads"]] = []
    masks: list[Bool[Array, "n k"]] = []
    for subgat, src, neigh, attr, row_mask, is_sink in zip(
      self.subgats,
      source_scalars,
      neighbours,
      edge_attr,
      mask,
      source_is_sink,
      strict=True,
    ):
      if src.ndim == 3:
        gathered = src
      else:
        table = sink_scalars if is_sink else src
        gathered = table[neigh]
      sink_feat = jnp.broadcast_to(
        sink_scalars[:, None, :],
        (sink_scalars.shape[0], neigh.shape[1], sink_scalars.shape[-1]),
      )
      value, score = subgat.messages(gathered, sink_feat, attr)
      values.append(value)
      scores.append(score)
      masks.append(row_mask)
    combined_mask = jnp.concatenate(tuple(masks), axis=1)
    weights = masked_softmax(jnp.concatenate(tuple(scores), axis=1), combined_mask)
    mixed = jnp.concatenate(tuple(values), axis=1) * weights[..., None]
    mixed = jnp.where(combined_mask[..., None, None], mixed, jnp.zeros_like(mixed))
    pooled = jnp.sum(mixed, axis=1)
    updated = self.final_atten_aggr(pooled)
    if self.dense_residual_node_update is not None:
      updated = self.dense_residual_node_update(sink_scalars, updated)
    scalars_out, vectors_out = self.equivariant_layer_norm(
      *self.vectors_update_layer(updated, sink_vectors),
    )
    edges_out: list[Float[Array, "n k e"]] = []
    for subgat, src, neigh, attr, row_mask, is_sink in zip(
      self.subgats,
      source_scalars,
      neighbours,
      edge_attr,
      mask,
      source_is_sink,
      strict=True,
    ):
      src_table = scalars_out if is_sink else src
      sink_feat = jnp.broadcast_to(
        scalars_out[:, None, :],
        (scalars_out.shape[0], neigh.shape[1], scalars_out.shape[-1]),
      )
      if self.compute_edge_updates:
        edges_out.append(subgat.edge_update(src_table[neigh], sink_feat, attr, row_mask))
      else:
        edges_out.append(attr)
    return scalars_out, vectors_out, tuple(edges_out)


class DenseGVP(eqx.Module):
  """Three gated GVPs. Optional norms sit between them; eval dropout is identity."""

  l1: GVP
  l2: GVP
  l3: GVP
  dropout: EquivariantDropout
  norm1: EquivariantLayerNorm | None
  norm2: EquivariantLayerNorm | None

  def __init__(
    self,
    input_dim: tuple[int, int],
    latent_dim: tuple[int, int],
    output_dim: tuple[int, int],
    dropout: float,
    *,
    intermediate_norm: bool = False,
    vector_only_norm: bool = False,
    key: PRNGKeyArray,
  ) -> None:
    if intermediate_norm and vector_only_norm:
      msg = "intermediate_norm and vector_only_norm are mutually exclusive"
      raise ValueError(msg)
    keys = jax.random.split(key, 3)
    self.l1 = GVP(input_dim, latent_dim, vector_gate=True, key=keys[0])
    self.l2 = GVP(latent_dim, latent_dim, vector_gate=True, key=keys[1])
    self.l3 = GVP(latent_dim, output_dim, vector_gate=True, key=keys[2])
    self.dropout = EquivariantDropout(dropout)
    self.norm1 = None
    self.norm2 = None
    if intermediate_norm or vector_only_norm:
      self.norm1 = EquivariantLayerNorm(latent_dim, vector_only=vector_only_norm)
      self.norm2 = EquivariantLayerNorm(latent_dim, vector_only=vector_only_norm)

  def __call__(
    self,
    scalars: Float[Array, "n s"],
    vectors: Float[Array, "n v 3"],
    *,
    inference: bool = True,
  ) -> tuple[Float[Array, "n so"], Float[Array, "n vo 3"]]:
    scalars, vectors = self.l1(scalars, vectors)
    if self.norm1 is not None:
      scalars, vectors = self.norm1(scalars, vectors)
    scalars, vectors = self.dropout(scalars, vectors, inference=inference)
    scalars, vectors = self.l2(scalars, vectors)
    if self.norm2 is not None:
      scalars, vectors = self.norm2(scalars, vectors)
    scalars, vectors = self.dropout(scalars, vectors, inference=inference)
    return self.l3(scalars, vectors)
