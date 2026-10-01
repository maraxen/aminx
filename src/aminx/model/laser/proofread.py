"""LASEr proofreading: unconditional logits, and the conditional dropout ensemble.

Unconditional proofreading is one decoder forward in which every protein edge
is masked, matching ``return_unconditional_probabilities`` (nothing has been
decoded yet). Conditional proofreading samples under injected scalar-dropout
masks and reduces with ``ddof=1`` over decoding orders, then averages over
dropout reps. Vector dropout stays off unless a caller turns it on: upstream
sets ``nn.Dropout`` to train and leaves ``_VDropout`` in eval.

The reduction order is fixed. With two orders and one dropout rep,
``std = |p1 - p2| / sqrt(2)``. ``ddof=0`` is ``|p1 - p2| / 2``. A standard
deviation over reps instead of orders is a different axis. ``n_orders=1``
leaves the standard deviation all-NaN; that NaN is the ddof=1 answer for one
sample.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from aminx.model.laser.decoder import LaserDecoder, _masked_first_order
from aminx.model.laser.encoders import LaserEncoder, encode_structure
from aminx.model.laser.graphs import GraphStructure, pack_edges, scatter_edge_values
from aminx.model.laser.joint_decode import LaserJointDecode, decode_order
from aminx.model.laser.layers import (
  DenseMLP,
  DenseResidualNodeUpdate,
  EquivariantDropout,
  HeteroGATv2,
  HomoGATv2,
  _linear_at,
  apply_layer_norm,
  apply_linear,
  masked_softmax,
)

_MASK_CLASS = 21
_CHI_BINS = 72
_DROPOUT_P = 0.1


class _Plan:
  """Host-side mask cursor. Read only under ``jax.disable_jit``."""

  def __init__(
    self,
    *,
    scalar: bool,
    vector: bool,
    masks: Mapping[str, Sequence[np.ndarray]] | None,
    drop_p: float,
  ) -> None:
    self.scalar = scalar
    self.vector = vector
    self.drop_p = drop_p
    self.calls: dict[str, int] = {}
    self.masks = {} if masks is None else {key: list(rows) for key, rows in masks.items()}
    self.chi_ids: set[int] = set()

  def take(self, path: str, like: jax.Array) -> jax.Array:
    """Next keep-mask for ``path``, broadcast onto ``like``."""
    rows = self.masks.get(path, [])
    call = self.calls.get(path, 0)
    self.calls[path] = call + 1
    if call < len(rows):
      keep = jnp.asarray(rows[call], dtype=like.dtype)
    elif path.endswith(".vdropout") and self.vector:
      # No injected vector mask means drop every component. Oracle eval never
      # drops vectors, so this is the "vector dropout on" control.
      keep = jnp.zeros(like.shape, dtype=like.dtype)
    else:
      keep = jnp.ones(like.shape, dtype=like.dtype)
    return _fit_keep(keep, like)

  def scale(self, path: str, value: jax.Array) -> jax.Array:
    if not self.scalar:
      return value
    keep = self.take(path, value)
    return value * keep / jnp.asarray(1.0 - self.drop_p, dtype=value.dtype)


_PLAN: ContextVar[_Plan | None] = ContextVar("laser_proofread_dropout", default=None)
_PATCHED = False
_ORIGINALS: dict[str, Any] = {}


def _fit_keep(keep: jax.Array, value: jax.Array) -> jax.Array:
  """Place a torch keep-mask on a dense tensor of the same number of elements.

  Valid-edge masks are shorter than a padded neighbour axis. They fill the
  leading True-sized prefix in C order and the padded tail stays kept, so a
  mask recorded on the sparse edge list still lines up with the dense slots
  the layer port already uses for those edges.
  """
  if keep.shape == value.shape:
    return keep
  if keep.size == value.size:
    return jnp.reshape(keep, value.shape)
  flat = jnp.ones(value.size, dtype=value.dtype)
  n = min(int(keep.size), int(value.size))
  flat = flat.at[:n].set(jnp.reshape(keep, (-1,))[:n])
  return jnp.reshape(flat, value.shape)


def reduce_proofread(
  probs: Float[Array, "dropouts orders *c 21"],
  *,
  ddof: int = 1,
  std_over: str = "orders",
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Mean and std of a ``(dropout, order, ..., 21)`` ensemble.

  Mean is always over orders, then over dropout reps. The standard deviation
  is over orders (``ddof`` as given) and then averaged over reps. ``std_over
  == "reps"`` takes that standard deviation over dropout reps instead, which
  is the reduction-swap control. A single order leaves the std all-NaN.
  """
  if std_over not in {"orders", "reps"}:
    msg = "std_over must be 'orders' or 'reps'"
    raise ValueError(msg)
  stacked = jnp.asarray(probs)
  order_mean = jnp.mean(stacked, axis=1)
  proofread_mean = jnp.mean(order_mean, axis=0)
  std_axis = 1 if std_over == "orders" else 0
  # ddof=1 of one sample is NaN. Do not replace it: n_orders=1 is defined that way.
  spread = jnp.std(stacked, axis=std_axis, ddof=ddof)
  # After the std, axis 0 is the axis that was not reduced: dropout reps when
  # std ran over orders, and orders when std ran over reps.
  proofread_std = jnp.mean(spread, axis=0)
  return proofread_mean, proofread_std, proofread_mean + proofread_std


def closed_form_std(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
  """``|p1 - p2| / sqrt(2)``, the ddof=1 std of two samples, in float64."""
  left = np.asarray(p1, dtype=np.float64)
  right = np.asarray(p2, dtype=np.float64)
  return np.abs(left - right) / np.sqrt(2.0)


def _all_masked_edges(
  decoder: LaserDecoder,
  prot_s: jax.Array,  # signature matches ``_teacher_edges``; the masked branch ignores it
  encoder_s: jax.Array,
  pr_edges: jax.Array,
  pr_neighbours: jax.Array,
  pr_mask: jax.Array,
  sequence_indices: jax.Array,
  chi_flat: jax.Array,
  decoding_order: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Teacher edges with the unmasked branch forced off.

  ``_teacher_edges`` selects that branch from the decoding order. Unconditional
  proofreading has no decoded predecessor, so every edge stays on the mask
  token. The order argument is unused on purpose: a real permutation would
  reveal earlier residues.
  """
  del decoding_order, prot_s
  source = pr_neighbours
  table = decoder.sequence_label_embedding.weight
  mask_ids = jnp.full(sequence_indices.shape, _MASK_CLASS)
  masked_sequence = table[mask_ids]
  seq_m = masked_sequence[source]
  chi_m = jnp.zeros_like(chi_flat[source])
  node_m = encoder_s[source]
  features = jnp.concatenate((seq_m, chi_m, pr_edges, node_m), axis=-1)
  unmasked = jnp.zeros(pr_mask.shape, dtype=jnp.bool_)
  order = _masked_first_order(unmasked, pr_mask)
  index = jnp.broadcast_to(order[:, :, None], features.shape)
  features = jnp.take_along_axis(features, index, axis=1)
  mask = jnp.take_along_axis(pr_mask, order, axis=1)
  return features, mask


class _UnconditionalView:
  """Decoder view whose teacher edges never reveal a predecessor."""

  def __init__(self, decoder: LaserDecoder) -> None:
    self._decoder = decoder

  def __getattr__(self, name: str) -> Any:  # noqa: ANN401
    return getattr(self._decoder, name)

  def _teacher_edges(self, *args: Any, **kwargs: Any) -> tuple[jax.Array, jax.Array]:  # noqa: ANN401
    return _all_masked_edges(self._decoder, *args, **kwargs)


def unconditional_logits(
  encoder: LaserEncoder,
  decoder: LaserDecoder,
  backbone: np.ndarray,
  ligand_coords: np.ndarray,
  ligand_atomic_numbers: np.ndarray,
  ligand_subbatch: np.ndarray,
  period_index: np.ndarray,
  group_index: np.ndarray,
  structure: GraphStructure,
  sequence_indices: np.ndarray,
  chi_angles: np.ndarray,
) -> np.ndarray:
  """Sequence logits with every residue treated as first, shape ``(L, 21)``."""
  encoded = encode_structure(
    encoder,
    backbone,
    ligand_coords,
    ligand_atomic_numbers,
    ligand_subbatch,
    period_index,
    group_index,
    structure,
  )
  n_res = int(encoded.prot_scalars.shape[0])
  packed_pr = pack_edges(n_res, encoded.pr_pr_idx)
  packed_lp = pack_edges(n_res, encoded.lig_pr_idx)
  pr_edges = scatter_edge_values(n_res, packed_pr, encoded.pr_pr_eattr)
  lp_edges = scatter_edge_values(n_res, packed_lp, encoded.lig_pr_eattr)
  dtype = encoded.prot_scalars.dtype
  order = jnp.arange(n_res, dtype=jnp.int32)
  view = _UnconditionalView(decoder)
  logits, _chi, _bank, _order_lp = LaserDecoder.__call__(
    cast("LaserDecoder", view),
    jnp.asarray(encoded.prot_scalars),
    jnp.asarray(encoded.prot_vectors),
    jnp.asarray(encoded.lig_scalars),
    jnp.asarray(pr_edges),
    jnp.asarray(packed_pr.neighbours),
    jnp.asarray(packed_pr.mask),
    jnp.asarray(lp_edges),
    jnp.asarray(packed_lp.neighbours),
    jnp.asarray(packed_lp.mask),
    order,
    jnp.asarray(sequence_indices, dtype=jnp.int32),
    jnp.asarray(chi_angles, dtype=dtype),
  )
  return np.asarray(logits)


def focus_rows(
  first_shell: np.ndarray,
  resnums: np.ndarray,
  selection_string: str | None,
) -> np.ndarray:
  """Rows proofread reports. A selection replaces the first-shell set."""
  shell = np.asarray(first_shell, dtype=bool)
  if selection_string is None or selection_string == "":
    rows = np.flatnonzero(shell)
    if rows.size == 0:
      return np.asarray([0], dtype=np.int32)
    return rows.astype(np.int32)
  wanted = {int(part) for part in selection_string.split(",") if part}
  rows = np.flatnonzero(np.isin(np.asarray(resnums), list(wanted)))
  if rows.size == 0:
    msg = f"selection_string {selection_string!r} matched no residue"
    raise ValueError(msg)
  return rows.astype(np.int32)


def _homo_call(
  self: HomoGATv2,
  scalars: jax.Array,
  vectors: jax.Array,
  edge_attr: jax.Array,
  neighbours: jax.Array,
  mask: jax.Array,
  *,
  inference: bool = True,
  path: str,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  del inference
  plan = _PLAN.get()
  source = scalars[neighbours]
  sink = jnp.broadcast_to(
    scalars[:, None, :],
    (scalars.shape[0], neighbours.shape[1], scalars.shape[-1]),
  )
  values, scores = self.messages(source, sink, edge_attr)
  weights = masked_softmax(scores, mask)
  if plan is not None:
    weights = plan.scale(path, weights)
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
  if plan is not None and self.linear_edge_updates is not None and self.edge_norm is not None:
    # The edge MLP itself is p=0. The GAT dropout hits the residual delta.
    delta = self.linear_edge_updates(
      jnp.concatenate((source_out, edge_attr, sink_out), axis=-1),
    )
    delta = plan.scale(path, delta)
    updated_edges = apply_layer_norm(self.edge_norm, edge_attr + delta)
    edges_out = jnp.where(mask[..., None], updated_edges, jnp.zeros_like(updated_edges))
  return scalars_out, vectors_out, edges_out


def _hetero_call(
  self: HeteroGATv2,
  sink_scalars: jax.Array,
  sink_vectors: jax.Array,
  source_scalars: tuple[jax.Array, ...],
  neighbours: tuple[jax.Array, ...],
  edge_attr: tuple[jax.Array, ...],
  mask: tuple[jax.Array, ...],
  source_is_sink: tuple[bool, ...],
  *,
  inference: bool = True,
  path: str,
) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...]]:
  del inference
  plan = _PLAN.get()
  values: list[jax.Array] = []
  scores: list[jax.Array] = []
  masks: list[jax.Array] = []
  for subgat, src, neigh, attr, row_mask, is_sink in zip(
    self.subgats,
    source_scalars,
    neighbours,
    edge_attr,
    mask,
    source_is_sink,
    strict=True,
  ):
    gathered = src if src.ndim == 3 else (sink_scalars if is_sink else src)[neigh]
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
  if plan is not None:
    weights = plan.scale(path + ".dropout", weights)
  mixed = jnp.concatenate(tuple(values), axis=1) * weights[..., None]
  mixed = jnp.where(combined_mask[..., None, None], mixed, jnp.zeros_like(mixed))
  pooled = jnp.sum(mixed, axis=1)
  updated = self.final_atten_aggr(pooled)
  if self.dense_residual_node_update is not None:
    updated = _residual_call(
      self.dense_residual_node_update,
      sink_scalars,
      updated,
      path=path + ".dense_residual_node_update.dropout",
    )
  scalars_out, vectors_out = self.equivariant_layer_norm(
    *self.vectors_update_layer(updated, sink_vectors),
  )
  edges_out: list[jax.Array] = []
  for index, (subgat, src, neigh, attr, row_mask, is_sink) in enumerate(
    zip(self.subgats, source_scalars, neighbours, edge_attr, mask, source_is_sink, strict=True),
  ):
    src_table = scalars_out if is_sink else src
    sink_feat = jnp.broadcast_to(
      scalars_out[:, None, :],
      (scalars_out.shape[0], neigh.shape[1], scalars_out.shape[-1]),
    )
    if self.compute_edge_updates and subgat.linear_edge_updates is not None:
      features = jnp.concatenate((src_table[neigh], attr, sink_feat), axis=-1)
      delta = subgat.linear_edge_updates(features)
      if plan is not None:
        delta = plan.scale(path + f".subgats.{index}.dropout", delta)
      if subgat.edge_norm is not None:
        revised = apply_layer_norm(subgat.edge_norm, attr + delta)
        edges_out.append(jnp.where(row_mask[..., None], revised, jnp.zeros_like(revised)))
      else:
        edges_out.append(attr)
    else:
      edges_out.append(attr)
  return scalars_out, vectors_out, tuple(edges_out)


def _residual_call(
  self: DenseResidualNodeUpdate,
  prev: jax.Array,
  update: jax.Array,
  *,
  path: str,
) -> jax.Array:
  plan = _PLAN.get()
  if plan is not None and plan.scalar:
    update = plan.scale(path, update)
  hidden = apply_layer_norm(self.node_norm1, prev + update)
  delta = apply_linear(_linear_at(self.layers, 0), hidden)
  delta = jax.nn.gelu(delta, approximate=False)
  delta = apply_linear(_linear_at(self.layers, 2), delta)
  if plan is not None and plan.scalar:
    delta = plan.scale(path, delta)
  return apply_layer_norm(self.node_norm2, hidden + delta)


def _install_patches(encoder: LaserEncoder, decoder: LaserDecoder) -> None:
  """Bind path-tagged forwards. The originals are restored when the plan exits.

  A warmed jit would keep the unpatched trace, so proofreading runs this under
  ``jax.disable_jit`` and each parity arm is a fresh process.
  """
  global _PATCHED  # noqa: PLW0603
  if _PATCHED:
    return
  _ORIGINALS["homo"] = HomoGATv2.__call__
  _ORIGINALS["hetero"] = HeteroGATv2.__call__
  _ORIGINALS["eqdrop"] = EquivariantDropout.__call__
  _ORIGINALS["mlp"] = DenseMLP.__call__
  plan_holder = _PLAN

  def homo(
    self: HomoGATv2,
    scalars: jax.Array,
    vectors: jax.Array,
    edge_attr: jax.Array,
    neighbours: jax.Array,
    mask: jax.Array,
    *,
    inference: bool = True,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    path = getattr(self, "_proofread_path", "homo.dropout")
    if plan_holder.get() is None:
      return _ORIGINALS["homo"](
        self,
        scalars,
        vectors,
        edge_attr,
        neighbours,
        mask,
        inference=inference,
      )
    return _homo_call(
      self,
      scalars,
      vectors,
      edge_attr,
      neighbours,
      mask,
      inference=inference,
      path=path,
    )

  def hetero(
    self: HeteroGATv2,
    sink_scalars: jax.Array,
    sink_vectors: jax.Array,
    source_scalars: tuple[jax.Array, ...],
    neighbours: tuple[jax.Array, ...],
    edge_attr: tuple[jax.Array, ...],
    mask: tuple[jax.Array, ...],
    source_is_sink: tuple[bool, ...],
    *,
    inference: bool = True,
  ) -> tuple[jax.Array, jax.Array, tuple[jax.Array, ...]]:
    path = getattr(self, "_proofread_path", "hetero")
    if plan_holder.get() is None:
      return _ORIGINALS["hetero"](
        self,
        sink_scalars,
        sink_vectors,
        source_scalars,
        neighbours,
        edge_attr,
        mask,
        source_is_sink,
        inference=inference,
      )
    return _hetero_call(
      self,
      sink_scalars,
      sink_vectors,
      source_scalars,
      neighbours,
      edge_attr,
      mask,
      source_is_sink,
      inference=inference,
      path=path,
    )

  def eqdrop(
    self: EquivariantDropout,
    scalars: jax.Array,
    vectors: jax.Array,
    *,
    inference: bool = True,
  ) -> tuple[jax.Array, jax.Array]:
    plan = plan_holder.get()
    if plan is None:
      return _ORIGINALS["eqdrop"](self, scalars, vectors, inference=inference)
    path = getattr(self, "_proofread_path", "equivariant.sdropout")
    # Scalar dropout is the nn.Dropout child. Vector dropout stays off unless
    # the plan asks for it; upstream leaves _VDropout in eval.
    if plan.scalar:
      scalars = plan.scale(path, scalars)
    if plan.vector:
      keep = plan.take(path + ".vdropout", vectors[..., 0])
      vectors = vectors * keep[..., None] / jnp.asarray(1.0 - plan.drop_p, dtype=vectors.dtype)
    return scalars, vectors

  def mlp(self: DenseMLP, features: jax.Array) -> jax.Array:
    plan = plan_holder.get()
    if plan is None or id(self) not in plan.chi_ids:
      return _ORIGINALS["mlp"](self, features)
    path = getattr(self, "_proofread_path", "chi")
    hidden = apply_linear(_linear_at(self.layers, 0), features)
    hidden = plan.scale(path + ".layers.1", hidden)
    hidden = jax.nn.gelu(hidden, approximate=False)
    hidden = apply_linear(_linear_at(self.layers, 3), hidden)
    hidden = plan.scale(path + ".layers.4", hidden)
    hidden = jax.nn.gelu(hidden, approximate=False)
    return apply_linear(_linear_at(self.layers, 6), hidden)

  # Class-level swap so every layer instance sees the mask cursor.
  HomoGATv2.__call__ = homo  # ty: ignore[invalid-assignment]
  HeteroGATv2.__call__ = hetero  # ty: ignore[invalid-assignment]
  EquivariantDropout.__call__ = eqdrop  # ty: ignore[invalid-assignment]
  DenseMLP.__call__ = mlp  # ty: ignore[invalid-assignment]
  _stamp(encoder, decoder)
  _PATCHED = True


def _stamp(encoder: LaserEncoder, decoder: LaserDecoder) -> None:
  for index, gat in enumerate(encoder.ligand_encoder.gat_layers):
    object.__setattr__(gat, "_proofread_path", f"ligand_encoder.gat_layers.{index}.dropout")
  object.__setattr__(
    encoder.ligand_encoder_output_gvp.dropout,
    "_proofread_path",
    "ligand_encoder_output_gvp.dropout.sdropout",
  )
  for index, layer in enumerate(encoder.protein_encoder_layers):
    object.__setattr__(layer.hetgat, "_proofread_path", f"protein_encoder_layers.{index}.hetgat")
  for index, layer in enumerate(decoder.protein_decoder_layers):
    object.__setattr__(layer.hetgat, "_proofread_path", f"protein_decoder_layers.{index}.hetgat")
  for index, layer in enumerate(decoder.chi_prediction_layers):
    object.__setattr__(layer, "_proofread_path", f"chi_prediction_layers.{index}")


def _chi_ids(decoder: LaserDecoder, joint_offsets: tuple[DenseMLP, ...]) -> set[int]:
  ids = {id(layer) for layer in decoder.chi_prediction_layers}
  ids.update(id(layer) for layer in joint_offsets)
  for index, layer in enumerate(joint_offsets):
    object.__setattr__(layer, "_proofread_path", f"chi_offset_prediction_layers.{index}")
  return ids


@contextmanager
def proofread_dropout(
  encoder: LaserEncoder,
  decoder: LaserDecoder,
  joint_offsets: tuple[DenseMLP, ...],
  *,
  scalar: bool,
  vector: bool = False,
  masks: Mapping[str, Sequence[np.ndarray]] | None = None,
  drop_p: float = _DROPOUT_P,
) -> Iterator[_Plan]:
  """Apply injected scalar dropout for one eager decode. Vector dropout defaults off."""
  _install_patches(encoder, decoder)
  plan = _Plan(scalar=scalar, vector=vector, masks=masks, drop_p=drop_p)
  plan.chi_ids = _chi_ids(decoder, joint_offsets)
  token = _PLAN.set(plan)
  try:
    with jax.disable_jit():
      yield plan
  finally:
    _PLAN.reset(token)


def softmax_rows(logits: np.ndarray) -> np.ndarray:
  """Softmax over the amino-acid axis. Stored proofread probabilities."""
  rows = np.asarray(logits, dtype=np.float64)
  shifted = rows - np.max(rows, axis=-1, keepdims=True)
  exp = np.exp(shifted)
  return exp / np.sum(exp, axis=-1, keepdims=True)


def _uniform_draws(draws: np.ndarray, n_res: int) -> np.ndarray:
  """One cell's inverse-CDF stream, shaped the way the decode indexes it.

  Upstream allocates ``length * 5 + 4``: five draws per residue (the sequence
  class, then four chi bins) and four tail slots the step cursor never reads.
  A shorter buffer is zero-padded inside the decode, and a zero uniform is the
  first class at every site, so the length check is what keeps that placeholder
  from sampling a frozen context.
  """
  array = np.asarray(draws)
  if array.ndim == 2 and int(array.shape[0]) == 1:
    array = array[0]
  need = n_res * 5 + 4
  if array.ndim != 1 or int(array.shape[0]) != need:
    msg = f"uniform draws length {array.size} != {need} (length*5+4)"
    raise ValueError(msg)
  return array


def conditional_focus_probs(
  encoder: LaserEncoder,
  decoder: LaserDecoder,
  joint: LaserJointDecode,
  backbone: np.ndarray,
  ligand_coords: np.ndarray,
  ligand_atomic_numbers: np.ndarray,
  ligand_subbatch: np.ndarray,
  period_index: np.ndarray,
  group_index: np.ndarray,
  structure: GraphStructure,
  sequence_indices: np.ndarray,
  chi_angles: np.ndarray,
  focus: int,
  orders: Sequence[np.ndarray],
  mask_sets: Sequence[Sequence[Mapping[str, Sequence[np.ndarray]] | None]],
  uniforms: Sequence[Sequence[np.ndarray]],
  *,
  scalar: bool,
  vector: bool = False,
  repack_all: bool = True,
) -> np.ndarray:
  """``(n_dropouts, n_orders, 21)`` softmax rows at ``focus``.

  Each order is one decode with only that residue designable. Masks are the
  injected scalar-dropout keeps for that rep and order. ``uniforms`` lines up
  with ``mask_sets``: one ``length * 5 + 4`` buffer per (dropout, order) cell,
  consumed in that order. The decode runs eager so the host cursor can hand a
  different mask to each call; a warmed jit would replay the first mask.
  """
  n_res = int(np.asarray(sequence_indices).shape[0])
  if len(uniforms) != len(mask_sets):
    msg = f"{len(uniforms)} uniform blocks != {len(mask_sets)} dropout reps"
    raise ValueError(msg)
  stacked: list[np.ndarray] = []
  offsets = tuple(joint.chi_offset_prediction_layers)
  for drop_masks, drop_draws in zip(mask_sets, uniforms, strict=True):
    if len(drop_draws) != len(orders):
      msg = f"{len(drop_draws)} uniform rows != {len(orders)} decoding orders"
      raise ValueError(msg)
    order_rows: list[np.ndarray] = []
    for order, masks, draws in zip(orders, drop_masks, drop_draws, strict=True):
      chain = np.ones((n_res,), dtype=bool)
      chain[int(focus)] = False
      with proofread_dropout(
        encoder,
        decoder,
        offsets,
        scalar=scalar,
        vector=vector,
        masks=masks,
      ):
        decoded = decode_order(
          encoder,
          decoder,
          joint,
          backbone,
          ligand_coords,
          ligand_atomic_numbers,
          ligand_subbatch,
          period_index,
          group_index,
          structure,
          sequence_indices,
          chi_angles,
          chain,
          np.zeros((n_res,), dtype=bool),
          np.asarray(order, dtype=np.int32),
          _uniform_draws(draws, n_res),
          sequence_temperature=1.0,
          chi_temperature=1.0,
          disabled_residues=("X",),
          repack_all=repack_all,
        )
      order_rows.append(softmax_rows(np.asarray(decoded.sequence_logits)[int(focus)]))
    stacked.append(np.stack(order_rows, axis=0))
  return np.stack(stacked, axis=0)
