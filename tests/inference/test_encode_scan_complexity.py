"""Encode scan complexity (debt #1981).

``make_encode_fn(..., use_rolling_state=True)`` scans once per conformational
state. Per-iteration work must be independent of S (criterion A) and at most
linear in L (criterion B). Work is a jaxpr proxy, not wall time.

The must-fire mutant re-runs the whole model ``S`` times inside each state and
keeps copy 0. Folding only ``model.encoder`` left the S work ratio at 1.001 on
this 64-d network, because the feature stage dominates the proxy, so criterion
A accepts it. That fold is guarded separately: an encoder mark's output numel
inside the scan is constant on the clean path and tracks S when only the
encoder is folded.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from aminx.inference.encode import make_encode_fn
from aminx.model.encoder import Encoder
from tests.helpers.aminx_small import bundle as make_bundle
from tests.helpers.aminx_small import make_guard_model
from tests.helpers.loop_scaling import (
  A_MUTANT_MIN,
  A_THRESHOLD,
  B_THRESHOLD,
  ENCODER_MARK,
  LoopRow,
  assert_mark_numel_independent,
  assert_work_at_most_linear,
  assert_work_independent,
  mark_array,
  measure_rows,
  pick_fixed_scan_largest,
  pick_trips_scaling,
  primitive_numel_in_loops,
  work_ratios,
)

_MODEL = make_guard_model()
_SENTINEL = {"encode": 0, "encoder_fold": 0}


class FoldEncoder(eqx.Module):
  """Re-run the whole per-state model ``n_fold`` times and keep copy 0."""

  wrapped: Any
  n_fold: int = eqx.field(static=True)

  def __call__(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    _SENTINEL["encode"] += 1

    def one(index: jax.Array) -> tuple[Any, Any, Any]:
      shifted = args[0] + index.astype(args[0].dtype)
      return self.wrapped(shifted, *args[1:], **kwargs)

    nodes, edges, neighbors = jax.vmap(one)(jnp.arange(self.n_fold))
    scale = jnp.float32(0)
    return nodes[0] + scale * jnp.sum(nodes), edges[0] + scale * jnp.sum(edges), neighbors[0]


class FoldOnlyEncoder(eqx.Module):
  """Re-run only the encoder stack ``n_fold`` times inside one state."""

  wrapped: Any
  n_fold: int = eqx.field(static=True)

  def __call__(
    self,
    edge_features: jax.Array,
    neighbor_indices: jax.Array,
    mask: jax.Array,
    initial_node_features: jax.Array | None = None,
    scale: float = 30.0,
    *,
    inference: bool = False,
    key: jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array]:
    _SENTINEL["encoder_fold"] += 1

    def one(index: jax.Array) -> tuple[jax.Array, jax.Array]:
      shifted = edge_features + index.astype(edge_features.dtype)
      return self.wrapped(
        shifted,
        neighbor_indices,
        mask,
        initial_node_features,
        scale,
        inference=inference,
        key=key,
      )

    nodes, edges = jax.vmap(one)(jnp.arange(self.n_fold))
    keep = jnp.float32(0)
    return nodes[0] + keep * jnp.sum(nodes), edges[0] + keep * jnp.sum(edges)


def _trace(
  extent: int, *, length: int, n_states: int, mutant: bool | str
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
  """``extent`` is the swept axis; the other of length and S is the fixed one.

  ``n_states == 0`` means the extent is S and ``length`` stays fixed.
  """
  if n_states == 0:
    n_states = extent
  else:
    length = extent
  model: Any = _MODEL
  if mutant == "encoder":
    folded = FoldOnlyEncoder(wrapped=_MODEL.encoder, n_fold=n_states)
    model = eqx.tree_at(lambda item: item.encoder, _MODEL, folded)
  elif mutant:
    model = FoldEncoder(wrapped=model, n_fold=n_states)
  encode = make_encode_fn(model, use_rolling_state=True)
  built, config = make_bundle(length, n_states=n_states)
  key = jax.random.PRNGKey(0)

  def fn(bundle_arr: Any, key_arr: jax.Array, config_arr: Any) -> Any:
    return encode(bundle_arr, key_arr, config_arr)

  return fn, (built, key, config)


def _rows(
  *, length: int, n_states: int, mutant: bool | str, extents: tuple[int, ...]
) -> list[LoopRow]:
  return measure_rows(
    lambda n: _trace(n, length=length, n_states=n_states, mutant=mutant),
    extents,
  )


def test_encode_scan_work_is_independent_of_states() -> None:
  extents = (2, 4, 8)
  rows = _rows(length=24, n_states=0, mutant=False, extents=extents)
  # One scan trip per conformational state. The probe logged trips=(2, 4, 8).
  row = pick_trips_scaling(rows, expected_trips=extents)
  assert row.primitive == "scan"
  assert row.trips == list(extents)
  assert_work_independent(rows)
  assert max(work_ratios(row)) < A_THRESHOLD


def test_encode_scan_work_is_at_most_linear_in_length() -> None:
  rows = _rows(length=0, n_states=2, mutant=False, extents=(32, 64, 128))
  # S is held at 2, so the state scan's trip count stays 2 as L grows.
  row = pick_fixed_scan_largest(rows, expected_trips=(2, 2, 2))
  assert row.primitive == "scan"
  assert row.trips == [2, 2, 2]
  assert_work_at_most_linear([row])
  assert max(work_ratios(row)) < B_THRESHOLD


def test_encode_scan_state_fold_must_fire() -> None:
  _SENTINEL["encode"] = 0
  clean = _rows(length=24, n_states=0, mutant=False, extents=(2, 4, 8))
  mutant = _rows(length=24, n_states=0, mutant=True, extents=(2, 4, 8))
  assert _SENTINEL["encode"] > 0
  expected = (2, 4, 8)
  clean_row = pick_trips_scaling(clean, expected_trips=expected)
  mutant_row = pick_trips_scaling(mutant, expected_trips=expected)
  clean_ratios = work_ratios(clean_row)
  mutant_ratios = work_ratios(mutant_row)
  assert max(clean_ratios) < A_THRESHOLD
  assert min(mutant_ratios) >= A_MUTANT_MIN
  assert min(mutant_ratios) > max(clean_ratios)
  with pytest.raises(AssertionError, match="work ratio"):
    assert_work_independent(mutant)


def _mark_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
  """Leave ``ENCODER_MARK`` on the encoder return so a vmap over S is visible."""
  original = Encoder.__call__

  def marked(self: Encoder, *args: Any, **kwargs: Any) -> tuple[jax.Array, jax.Array]:
    nodes, edges = original(self, *args, **kwargs)
    # Sum first: the mark's output is a scalar on the clean path and (S,) under
    # the encoder vmap, so its numel tracks S without adding an (S, L, H) term
    # to the work proxy.
    bump = mark_array(jnp.sum(nodes))
    return nodes + jnp.float32(0) * jnp.sum(bump), edges

  monkeypatch.setattr(Encoder, "__call__", marked)


def _mark_numel(*, mutant: bool | str, extents: tuple[int, ...]) -> list[int]:
  counts: list[int] = []
  for extent in extents:
    fn, args = _trace(extent, length=24, n_states=0, mutant=mutant)
    counts.append(primitive_numel_in_loops(fn, args, ENCODER_MARK))
  return counts


def test_encode_scan_encoder_only_fold_must_fire(monkeypatch: pytest.MonkeyPatch) -> None:
  """Folding only the encoder keeps the S work ratio under A; the mark numel does not."""
  _mark_encoder(monkeypatch)
  extents = (2, 4, 8)
  _SENTINEL["encoder_fold"] = 0
  clean = _rows(length=24, n_states=0, mutant=False, extents=extents)
  mutant = _rows(length=24, n_states=0, mutant="encoder", extents=extents)
  assert _SENTINEL["encoder_fold"] > 0
  clean_row = pick_trips_scaling(clean, expected_trips=extents)
  mutant_row = pick_trips_scaling(mutant, expected_trips=extents)
  # Feature work dominates, so criterion A still passes. The guard below is what fires.
  assert max(work_ratios(clean_row)) < A_THRESHOLD
  assert max(work_ratios(mutant_row)) < A_THRESHOLD
  clean_counts = _mark_numel(mutant=False, extents=extents)
  mutant_counts = _mark_numel(mutant="encoder", extents=extents)
  assert_mark_numel_independent(clean_counts)
  clean_ratios = [
    large / small for small, large in zip(clean_counts, clean_counts[1:], strict=False)
  ]
  mutant_ratios = [
    large / small for small, large in zip(mutant_counts, mutant_counts[1:], strict=False)
  ]
  assert max(clean_ratios) < A_THRESHOLD
  assert min(mutant_ratios) >= A_MUTANT_MIN
  assert min(mutant_ratios) > max(clean_ratios)
  with pytest.raises(AssertionError, match="numel ratio"):
    assert_mark_numel_independent(mutant_counts)
