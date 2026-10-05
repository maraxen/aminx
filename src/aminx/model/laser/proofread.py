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

import zlib
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
    edges: Mapping[str, Sequence[np.ndarray | None]] | None = None,
    dropout_key: jax.Array | None = None,
  ) -> None:
    self.scalar = scalar
    self.vector = vector
    self.drop_p = drop_p
    # Production has no injected catalog and draws its own keeps from this key.
    # The parity vehicle passes None, which is what makes a mask miss raise
    # there instead of being papered over by a fresh draw.
    self.dropout_key = dropout_key
    self.calls: dict[str, int] = {}
    # Calls satisfied by a draw rather than a recorded row, per path. Keeps
    # ``reconcile`` from reading a drawn call as an unrecorded replay.
    self.drawn: dict[str, int] = {}
    self.masks = {} if masks is None else {key: list(rows) for key, rows in masks.items()}
    # Parallel to ``masks``. ``None`` is an MLP keep; a ``(3, E)`` array is
    # source, sink, and neighbour-block for a sparse GAT keep.
    self.edges = {} if edges is None else {key: list(rows) for key, rows in edges.items()}
    self.chi_ids: set[int] = set()
    # Hetero edge updates look up ``subgats.{i}.dropout``. Upstream records no
    # such path, so a non-zero count means those calls were live misses.
    self.subgat_dropout_lookups = 0

  def take(
    self,
    path: str,
    like: jax.Array,
    *,
    neighbours: jax.Array | tuple[jax.Array, ...] | None = None,
    valid: jax.Array | tuple[jax.Array, ...] | None = None,
  ) -> jax.Array:
    """Next keep-mask for ``path``, placed onto ``like``.

    A miss is resolved by the two callers' opposite expectations. The parity
    vehicle injects a full catalog and passes no key, so a miss there is a
    broken replay and raises -- substituting ones while still dividing by
    ``1 - p`` scales the activation by ``1/0.9`` and is indistinguishable from
    a correct replay. Production injects nothing and passes a key, so a miss
    is the normal case and draws a fresh bernoulli keep at ``1 - p``.

    ``.vdropout`` stays the vector-dropout control: upstream never injects
    those masks. A recorded ``(3, E)`` edge index scatters by identity;
    without one, only element-for-element shapes fit.
    """
    rows = self.masks.get(path, [])
    call = self.calls.get(path, 0)
    self.calls[path] = call + 1
    edge: np.ndarray | None = None
    if call < len(rows):
      keep = jnp.asarray(rows[call], dtype=like.dtype)
      edge_rows = self.edges.get(path, [])
      if call < len(edge_rows):
        edge = edge_rows[call]
    elif path.endswith(".vdropout"):
      # Vector-on drops every component. Vector-off keeps every component.
      # Both are controls, not a silent scalar-mask miss.
      if self.vector:
        keep = jnp.zeros(like.shape, dtype=like.dtype)
      else:
        keep = jnp.ones(like.shape, dtype=like.dtype)
    elif self.dropout_key is not None:
      keep = self._draw(path, call, like)
      self.drawn[path] = self.drawn.get(path, 0) + 1
    else:
      msg = f"no injected dropout mask for {path} call {call} and no dropout key to draw one from"
      raise RuntimeError(msg)
    if edge is not None:
      return _scatter_keep(
        keep,
        edge,
        like,
        neighbours,
        valid,
        path=path,
        call=call,
      )
    return _fit_keep(keep, like)

  def _draw(self, path: str, call: int, like: jax.Array) -> jax.Array:
    """A fresh bernoulli keep at ``1 - drop_p``, shaped like ``like``.

    Each call site gets its own stream, keyed by the path and the call index,
    so two calls on the same layer do not share a mask. ``crc32`` is the hash,
    not ``hash()``: the builtin is salted per process, which would make a run
    unreproducible from its seed alone.
    """
    if self.dropout_key is None:  # pragma: no cover - guarded by the caller
      msg = f"no dropout key to draw {path} call {call}"
      raise RuntimeError(msg)
    stream = zlib.crc32(path.encode("utf-8")) & 0x7FFFFFFF
    key = jax.random.fold_in(jax.random.fold_in(self.dropout_key, stream), call)
    keep = jax.random.bernoulli(key, 1.0 - self.drop_p, shape=like.shape)
    return jnp.asarray(keep, dtype=like.dtype)

  def scale(
    self,
    path: str,
    value: jax.Array,
    *,
    neighbours: jax.Array | tuple[jax.Array, ...] | None = None,
    valid: jax.Array | tuple[jax.Array, ...] | None = None,
  ) -> jax.Array:
    if not self.scalar:
      return value
    keep = self.take(path, value, neighbours=neighbours, valid=valid)
    return value * keep / jnp.asarray(1.0 - self.drop_p, dtype=value.dtype)

  def consumption(self) -> dict[str, tuple[int, int]]:
    """``path -> (consumed, recorded)`` for every path either side touched."""
    paths = set(self.masks) | set(self.calls)
    return {
      path: (self.calls.get(path, 0), len(self.masks.get(path, []))) for path in sorted(paths)
    }

  def reconcile(self) -> None:
    """Demand each recorded scalar mask was consumed once.

    Scalar-off leaves the injected catalog unread on purpose, so it does not
    reconcile. ``.vdropout`` is the vector control and is not a scalar mask.
    Drawn calls are discounted: production records nothing and draws every
    keep, so counting those as unconsumed rows would fail every real decode.
    """
    if not self.scalar:
      return
    problems: list[str] = []
    seen: set[str] = set()
    for path, rows in self.masks.items():
      if path.endswith(".vdropout"):
        continue
      seen.add(path)
      replayed = self.calls.get(path, 0) - self.drawn.get(path, 0)
      if replayed != len(rows):
        problems.append(f"{path} call {replayed}: consumed {replayed}, recorded {len(rows)}")
    for path, used in self.calls.items():
      if path.endswith(".vdropout") or path in seen:
        continue
      replayed = used - self.drawn.get(path, 0)
      if replayed != 0:
        problems.append(f"{path} call {replayed}: consumed {replayed}, recorded 0")
    if problems:
      msg = "dropout replay reconciliation failed: " + "; ".join(problems)
      raise RuntimeError(msg)


_PLAN: ContextVar[_Plan | None] = ContextVar("laser_proofread_dropout", default=None)
_PATCHED = False
#: Nesting depth of live ``proofread_dropout`` contexts. The class patches are process-global, so
#: an inner context exiting must NOT unpatch an outer one still running; only the outermost exit
#: restores. Counted rather than boolean because ``_PATCHED`` alone cannot tell one live context
#: from two.
_PATCH_DEPTH = 0
_ORIGINALS: dict[str, Any] = {}


def _fit_keep(keep: jax.Array, value: jax.Array) -> jax.Array:
  """Place a keep whose elements already correspond one-for-one with ``value``.

  Exact shape is that correspondence. So is an equal element count at rank
  <= 2: the MLP masks ``(1, 256)`` and ``(115, 256)`` are the chi and
  residual-node tensors, sometimes with a leading singleton, and a reshape
  hits the same elements. A length-1 keep is one draw broadcast onto every
  element, including a padded attention grid: the consumption fixture records
  ``ones((1,))`` and does not build an edge list.

  A NODE-indexed keep may also be SHORTER along the leading axis, and that is
  still element-for-element. Upstream's ligand scalars are ``(31, 256)`` for
  31 real ligand nodes while aminx pads that axis to ``(32, 256)``; row i is
  node i on both sides and the padded tail is not a node at all, so it stays
  kept. This branch requires the trailing dims to match EXACTLY, which is what
  distinguishes it from the sparse case -- a GAT edge axis shares no trailing
  shape with the padded ``(node, slot)`` grid, so it cannot reach here.

  A sparse GAT mask is none of these. Its edge axis is not a prefix of that
  grid in any sense; ``_scatter_keep`` places it by edge identity.
  """
  if keep.shape == value.shape:
    return keep
  if keep.ndim <= 2 and value.ndim <= 2 and keep.size == value.size:
    return jnp.reshape(keep, value.shape)
  if keep.ndim <= 2 and keep.size == 1:
    return jnp.broadcast_to(jnp.reshape(keep, ()), value.shape)
  if (
    keep.ndim == value.ndim
    and keep.ndim >= 1
    and keep.shape[1:] == value.shape[1:]
    and 0 < keep.shape[0] < value.shape[0]
  ):
    # Node-indexed prefix: real nodes first, padding after. Padded rows stay
    # kept, matching an eval-mode dropout that never touches them.
    ones = jnp.ones(value.shape, dtype=value.dtype)
    return ones.at[: keep.shape[0]].set(jnp.asarray(keep, dtype=value.dtype))
  msg = (
    f"keep shape {keep.shape} ({keep.size} elements) and value shape "
    f"{value.shape} ({value.size} elements) are not element-for-element; "
    "a sparse edge mask needs an edge index"
  )
  raise RuntimeError(msg)


def _grids(grid: jax.Array | tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
  # jax.Array satisfies enough of the tuple protocol that a bare isinstance
  # does not narrow the return to ``tuple[Array, ...]``.
  if isinstance(grid, tuple):
    return cast("tuple[jax.Array, ...]", grid)
  return (grid,)


def _dense_slots(
  edge: np.ndarray,
  value: jax.Array,
  neighbours: jax.Array | tuple[jax.Array, ...],
  valid: jax.Array | tuple[jax.Array, ...],
  *,
  path: str,
  call: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Map each ``(source, sink, block)`` column onto one free dense slot."""
  identity = np.asarray(edge)
  if identity.ndim != 2 or int(identity.shape[0]) != 3:
    msg = f"{path} call {call}: edge identity shape {identity.shape} is not (3, E)"
    raise RuntimeError(msg)
  source = identity[0].astype(np.int64, copy=False)
  sink = identity[1].astype(np.int64, copy=False)
  block = identity[2].astype(np.int64, copy=False)
  neigh_grids = _grids(neighbours)
  valid_grids = _grids(valid)
  if len(neigh_grids) != len(valid_grids):
    msg = f"{path} call {call}: {len(neigh_grids)} neighbour grids and {len(valid_grids)} masks"
    raise RuntimeError(msg)
  neigh_np = [np.asarray(grid) for grid in neigh_grids]
  valid_np = [np.asarray(grid).astype(bool, copy=False) for grid in valid_grids]
  widths = [int(grid.shape[1]) for grid in neigh_np]
  if sum(widths) != int(value.shape[1]):
    msg = f"{path} call {call}: neighbour width {sum(widths)} != value slots {int(value.shape[1])}"
    raise RuntimeError(msg)
  offsets = np.cumsum([0, *widths[:-1]]).astype(np.int64, copy=False) if widths else np.zeros((0,))
  slots = np.empty((int(source.shape[0]),), dtype=np.int64)
  used = [np.zeros(grid.shape, dtype=bool) for grid in neigh_np]
  for index in range(int(source.shape[0])):
    which = int(block[index])
    row = int(sink[index])
    if which < 0 or which >= len(neigh_np):
      msg = f"{path} call {call}: edge {index} block {which} outside {len(neigh_np)} grids"
      raise RuntimeError(msg)
    n_nodes = int(neigh_np[which].shape[0])
    if row < 0 or row >= n_nodes:
      msg = f"{path} call {call}: edge {index} sink {row} outside {n_nodes} nodes"
      raise RuntimeError(msg)
    free = valid_np[which][row] & ~used[which][row]
    hits = np.flatnonzero(free & (neigh_np[which][row] == int(source[index])))
    if int(hits.size) == 0:
      msg = (
        f"{path} call {call}: edge {index} source {int(source[index])} "
        f"sink {row} block {which} has no dense slot"
      )
      raise RuntimeError(msg)
    choice = int(hits[0])
    used[which][row, choice] = True
    slots[index] = int(offsets[which]) + choice
  return sink, slots


def _paint_keep(
  placed: np.ndarray,
  keep_np: np.ndarray,
  sink: np.ndarray,
  slots: np.ndarray,
  *,
  path: str,
  call: int,
) -> None:
  """Write an attention ``(heads, edges, 1)`` or residual ``(edges, feat)`` keep."""
  n_edges = int(sink.shape[0])
  attention = keep_np.ndim == 3 and int(keep_np.shape[1]) == n_edges and int(keep_np.shape[-1]) == 1
  if attention:
    heads = int(keep_np.shape[0])
    if placed.ndim != 3 or int(placed.shape[-1]) != heads:
      msg = (
        f"{path} call {call}: attention keep {keep_np.shape} does not match value {placed.shape}"
      )
      raise RuntimeError(msg)
    placed[sink, slots, :] = np.reshape(keep_np, (heads, n_edges)).T
    return
  residual = keep_np.ndim == 2 and int(keep_np.shape[0]) == n_edges
  if residual:
    if placed.ndim != 3 or int(placed.shape[-1]) != int(keep_np.shape[1]):
      msg = f"{path} call {call}: edge keep {keep_np.shape} does not match value {placed.shape}"
      raise RuntimeError(msg)
    placed[sink, slots, :] = keep_np
    return
  msg = f"{path} call {call}: keep shape {keep_np.shape} does not align with {n_edges} edges"
  raise RuntimeError(msg)


def _scatter_keep(
  keep: jax.Array,
  edge: np.ndarray,
  value: jax.Array,
  neighbours: jax.Array | tuple[jax.Array, ...] | None,
  valid: jax.Array | tuple[jax.Array, ...] | None,
  *,
  path: str,
  call: int,
) -> jax.Array:
  """Scatter a sparse keep onto ``value`` by ``(source, sink, block)``.

  Padded slots stay kept: they are not edges, and the neighbour mask zeros
  them. A recorded edge with no free slot raises. Matching the edge count
  is not enough, so this never fills a C-order prefix.
  """
  if neighbours is None or valid is None:
    msg = f"{path} call {call}: edge index has no dense neighbourhood to scatter onto"
    raise RuntimeError(msg)
  sink, slots = _dense_slots(edge, value, neighbours, valid, path=path, call=call)
  # Host array in the activation dtype. A float64 scratch would promote the
  # attention multiply under the eager decode.
  placed = np.ones(value.shape, dtype=np.dtype(value.dtype))
  _paint_keep(placed, np.asarray(keep), sink, slots, path=path, call=call)
  return jnp.asarray(placed, dtype=value.dtype)


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
) -> tuple[jax.Array, jax.Array, jax.Array]:
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
  # Same slot order as the features. Padding moves to the end; the replay
  # looks source ids up here, not in the knn grid.
  neighbours = jnp.take_along_axis(pr_neighbours, order, axis=1)
  return features, mask, neighbours


class _UnconditionalView:
  """Decoder view whose teacher edges never reveal a predecessor."""

  def __init__(self, decoder: LaserDecoder) -> None:
    self._decoder = decoder

  def __getattr__(self, name: str) -> Any:  # noqa: ANN401
    return getattr(self._decoder, name)

  def _teacher_edges(self, *args: Any, **kwargs: Any) -> tuple[jax.Array, jax.Array, jax.Array]:  # noqa: ANN401
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
    weights = plan.scale(path, weights, neighbours=neighbours, valid=mask)
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
    delta = plan.scale(path, delta, neighbours=neighbours, valid=mask)
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
    weights = plan.scale(path + ".dropout", weights, neighbours=neighbours, valid=mask)
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
        plan.subgat_dropout_lookups += 1
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


def _install_patches() -> None:
  """Bind path-tagged forwards. The originals are restored when the plan exits.

  A warmed jit would keep the unpatched trace, so proofreading runs this under
  ``jax.disable_jit`` and each parity arm is a fresh process.
  """
  global _PATCHED, _PATCH_DEPTH  # noqa: PLW0603
  _PATCH_DEPTH += 1
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
    path = self.proofread_path
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
    if path == "":
      # filter_jit used to drop a non-field stamp, and this call fell through
      # to a name upstream never recorded.
      msg = "no proofread path on HomoGATv2"
      raise RuntimeError(msg)
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
    path = self.proofread_path
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
    if path == "":
      msg = "no proofread path on HeteroGATv2"
      raise RuntimeError(msg)
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
    path = self.proofread_path
    if path == "":
      msg = "no proofread path on EquivariantDropout"
      raise RuntimeError(msg)
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
    # Identity is not stable across filter_jit: the rebuilt module is a new
    # object, so chi heads are marked by the static path instead of ``id``.
    path = self.proofread_path
    if plan is None or path == "":
      return _ORIGINALS["mlp"](self, features)
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
  _PATCHED = True


def _remove_patches() -> None:
  """Put the four production methods back when the outermost plan exits.

  ``_install_patches`` has always documented that "the originals are restored when the plan
  exits", but nothing restored them: ``proofread_dropout``'s ``finally`` reset only the ``_PLAN``
  ContextVar, so once any proofread context had been entered the four class patches stayed on the
  classes for the life of the process. Nothing looked broken because each patched method
  delegates to ``_ORIGINALS[...]`` when no plan is active, which preserves behaviour -- it just
  preserves it through an extra Python frame and a ContextVar lookup on four hot methods, for
  every later caller in that process, forever.

  It also blinded a regression test. ``tests/model/test_laser_attention_dropout_2404.py`` asserts
  that ``HomoGATv2.__call__``/``HeteroGATv2.__call__`` are still the ones defined in
  ``aminx.model.laser.layers``, precisely because (its own docstring) "the proofread patches
  replace those methods, so a parity vehicle on the patched path cannot see this defect". Any
  earlier test in the same worker that entered a proofread context made debt #2404's guard
  unable to reach the production path at all.

  That is how this surfaced: the a10->a11 xtrax pin had been breaking collection of these tests,
  and fixing the pin let them run beside a proofread test for the first time.
  """
  global _PATCHED, _PATCH_DEPTH  # noqa: PLW0603
  _PATCH_DEPTH = max(0, _PATCH_DEPTH - 1)
  if _PATCH_DEPTH > 0 or not _PATCHED:
    return
  HomoGATv2.__call__ = _ORIGINALS["homo"]  # ty: ignore[invalid-assignment]
  HeteroGATv2.__call__ = _ORIGINALS["hetero"]  # ty: ignore[invalid-assignment]
  EquivariantDropout.__call__ = _ORIGINALS["eqdrop"]  # ty: ignore[invalid-assignment]
  DenseMLP.__call__ = _ORIGINALS["mlp"]  # ty: ignore[invalid-assignment]
  _PATCHED = False


def _stamp(encoder: LaserEncoder, decoder: LaserDecoder) -> None:
  for index, gat in enumerate(encoder.ligand_encoder.gat_layers):
    object.__setattr__(gat, "proofread_path", f"ligand_encoder.gat_layers.{index}.dropout")
  object.__setattr__(
    encoder.ligand_encoder_output_gvp.dropout,
    "proofread_path",
    "ligand_encoder_output_gvp.dropout.sdropout",
  )
  for index, layer in enumerate(encoder.protein_encoder_layers):
    object.__setattr__(layer.hetgat, "proofread_path", f"protein_encoder_layers.{index}.hetgat")
  for index, layer in enumerate(decoder.protein_decoder_layers):
    object.__setattr__(layer.hetgat, "proofread_path", f"protein_decoder_layers.{index}.hetgat")
  for index, layer in enumerate(decoder.chi_prediction_layers):
    object.__setattr__(layer, "proofread_path", f"chi_prediction_layers.{index}")


def _chi_ids(decoder: LaserDecoder, joint_offsets: tuple[DenseMLP, ...]) -> set[int]:
  ids = {id(layer) for layer in decoder.chi_prediction_layers}
  ids.update(id(layer) for layer in joint_offsets)
  for index, layer in enumerate(joint_offsets):
    object.__setattr__(layer, "proofread_path", f"chi_offset_prediction_layers.{index}")
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
  edges: Mapping[str, Sequence[np.ndarray | None]] | None = None,
  drop_p: float = _DROPOUT_P,
  dropout_key: jax.Array | None = None,
) -> Iterator[_Plan]:
  """Apply scalar dropout for one eager decode. Vector dropout defaults off.

  Pass ``masks`` to replay a recorded catalog (the parity vehicle) or
  ``dropout_key`` to draw fresh keeps (production). Neither makes every call a
  miss, and a miss with no key raises rather than silently running dropout-free
  while still rescaling by ``1 / (1 - p)``.
  """
  _install_patches()
  # The install/remove pair straddles everything below, so a raise in _stamp or _Plan cannot
  # leave the classes patched and the depth counter stranded above zero.
  try:
    # Restamp on every entry. The class patch is process-global while a context is live, but
    # each encoder carries its own static paths.
    _stamp(encoder, decoder)
    plan = _Plan(
      scalar=scalar,
      vector=vector,
      masks=masks,
      drop_p=drop_p,
      edges=edges,
      dropout_key=dropout_key,
    )
    plan.chi_ids = _chi_ids(decoder, joint_offsets)
    token = _PLAN.set(plan)
    try:
      with jax.disable_jit():
        yield plan
      # One decode is done. Leftover masks are calls the replay never made.
      plan.reconcile()
    finally:
      _PLAN.reset(token)
  finally:
    _remove_patches()


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
  edge_sets: Sequence[Sequence[Mapping[str, Sequence[np.ndarray | None]] | None]] | None = None,
  *,
  scalar: bool,
  vector: bool = False,
  repack_all: bool = True,
  dropout_key: jax.Array | None = None,
) -> np.ndarray:
  """``(n_dropouts, n_orders, 21)`` softmax rows at ``focus``.

  Each order is one decode with only that residue designable. Masks are the
  injected scalar-dropout keeps for that rep and order. ``uniforms`` lines up
  with ``mask_sets``: one ``length * 5 + 4`` buffer per (dropout, order) cell.
  Upstream ``sample`` consumes that buffer in call order, and the calls follow
  ``decoding_order``, five draws per residue. The first draw of a reversed
  order therefore belongs to the last residue; indexing the buffer by residue
  number assigns it to residue 0 instead. The decode runs eager so the host
  cursor can hand a different mask to each call; a warmed jit would replay
  the first mask.
  """
  n_res = int(np.asarray(sequence_indices).shape[0])
  if len(uniforms) != len(mask_sets):
    msg = f"{len(uniforms)} uniform blocks != {len(mask_sets)} dropout reps"
    raise ValueError(msg)
  if edge_sets is not None and len(edge_sets) != len(mask_sets):
    msg = f"{len(edge_sets)} edge blocks != {len(mask_sets)} dropout reps"
    raise ValueError(msg)
  stacked: list[np.ndarray] = []
  offsets = tuple(joint.chi_offset_prediction_layers)
  for drop_index, (drop_masks, drop_draws) in enumerate(zip(mask_sets, uniforms, strict=True)):
    if len(drop_draws) != len(orders):
      msg = f"{len(drop_draws)} uniform rows != {len(orders)} decoding orders"
      raise ValueError(msg)
    drop_edges = None if edge_sets is None else edge_sets[drop_index]
    if drop_edges is not None and len(drop_edges) != len(orders):
      msg = f"{len(drop_edges)} edge rows != {len(orders)} decoding orders"
      raise ValueError(msg)
    order_rows: list[np.ndarray] = []
    for order_index, (order, masks, draws) in enumerate(
      zip(orders, drop_masks, drop_draws, strict=True),
    ):
      chain = np.ones((n_res,), dtype=bool)
      chain[int(focus)] = False
      edges = None if drop_edges is None else drop_edges[order_index]
      # Each (rep, order) cell gets its own stream. Sharing one key would make
      # every dropout rep identical, and the reduction over reps would collapse
      # to zero standard deviation -- a pass that measures nothing.
      cell_key = (
        None
        if dropout_key is None
        else jax.random.fold_in(
          jax.random.fold_in(dropout_key, drop_index),
          order_index,
        )
      )
      with proofread_dropout(
        encoder,
        decoder,
        offsets,
        scalar=scalar,
        vector=vector,
        masks=masks,
        edges=edges,
        dropout_key=cell_key,
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
