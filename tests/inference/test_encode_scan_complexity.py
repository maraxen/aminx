"""Encode scan complexity (debt #1981).

``make_encode_fn(..., use_rolling_state=True)`` scans once per conformational
state. Per-iteration work must be independent of S (criterion A) and at most
linear in L (criterion B). Work is a jaxpr proxy, not wall time.

The must-fire mutant re-runs the whole model ``S`` times inside each state and
keeps copy 0. Folding only ``model.encoder`` left the S ratio at 1.001 on this
64-d network, because the feature stage dominates the proxy; folding the model
call is the same whole-body recompute and is what the graded probe measured.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from aminx.inference.encode import make_encode_fn
from tests.helpers.aminx_small import bundle as make_bundle
from tests.helpers.aminx_small import make_guard_model
from tests.helpers.loop_scaling import (
  A_MUTANT_MIN,
  A_THRESHOLD,
  B_THRESHOLD,
  LoopRow,
  assert_work_at_most_linear,
  assert_work_independent,
  measure_rows,
  pick_fixed_scan_largest,
  pick_trips_scaling,
  work_ratios,
)

_MODEL = make_guard_model()
_SENTINEL = {"encode": 0}


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


def _trace(
  extent: int, *, length: int, n_states: int, mutant: bool
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
  """``extent`` is the swept axis; the other of length and S is the fixed one.

  ``n_states == 0`` means the extent is S and ``length`` stays fixed.
  """
  if n_states == 0:
    n_states = extent
  else:
    length = extent
  model: Any = _MODEL
  if mutant:
    model = FoldEncoder(wrapped=model, n_fold=n_states)
  encode = make_encode_fn(model, use_rolling_state=True)
  built, config = make_bundle(length, n_states=n_states)
  key = jax.random.PRNGKey(0)

  def fn(bundle_arr: Any, key_arr: jax.Array, config_arr: Any) -> Any:
    return encode(bundle_arr, key_arr, config_arr)

  return fn, (built, key, config)


def _rows(*, length: int, n_states: int, mutant: bool, extents: tuple[int, ...]) -> list[LoopRow]:
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
