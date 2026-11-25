"""Toy jaxprs for the shared loop-scaling helper (debt #1981, Task 1).

Work is a deterministic jaxpr proxy, not wall time.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from xtrax.profiling.loop_scaling import LoopStructureMismatchError

from tests.helpers.aminx_small import bundle, make_ar_model, make_guard_model, structure
from tests.helpers.loop_scaling import (
  A_THRESHOLD,
  B_THRESHOLD,
  ENCODER_MARK,
  LoopRow,
  assert_no_primitive_in_loops,
  assert_work_at_most_linear,
  assert_work_independent,
  count_primitive_in_loops,
  fold_over,
  loop_work_by_extent,
  mark_array,
  measure_rows,
  pick_fixed_scan_largest,
  pick_trips_scaling,
  trips_scale,
  work_ratios,
)


def _o1_scan(xs: jax.Array) -> jax.Array:
  def body(carry: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return carry + x, x

  carry, _ys = jax.lax.scan(body, jnp.float32(0), xs)
  return carry


def _fold_scan(xs: jax.Array) -> jax.Array:
  width = xs.shape[1]

  def body(carry: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    recomputed = xs @ jnp.ones((width, width), dtype=xs.dtype)
    updated = carry + jnp.sum(x) + jnp.float32(0) * jnp.sum(recomputed)
    return updated, x

  carry, _ys = jax.lax.scan(body, jnp.float32(0), xs)
  return carry


def _quadratic_fori(xs: jax.Array) -> jax.Array:
  def body(_index: jax.Array, carry: jax.Array) -> jax.Array:
    gram = xs @ xs.T
    return carry + jnp.sum(gram)

  return jax.lax.fori_loop(0, 4, body, jnp.float32(0))


def _maybe_two_scans(xs: jax.Array) -> jax.Array:
  def body(carry: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return carry + x, x

  carry, ys = jax.lax.scan(body, jnp.float32(0), xs)
  if xs.shape[0] >= 16:

    def body2(carry2: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
      return carry2 + y, y

    carry, _ys = jax.lax.scan(body2, carry, ys)
  return carry


def _searchsorted(xs: jax.Array) -> jax.Array:
  return jnp.searchsorted(jnp.arange(xs.shape[0]), xs)


def test_o1_scan_work_is_independent() -> None:
  rows = loop_work_by_extent(
    _o1_scan,
    lambda n: (jnp.ones((n,), dtype=jnp.float32),),
    (8, 16, 32),
  )
  assert len(rows) == 1
  assert rows[0].primitive == "scan"
  assert trips_scale(rows[0].trips)
  for ratio in work_ratios(rows[0]):
    assert ratio == pytest.approx(1.0)
  assert_work_independent(rows)


def test_recomputation_scan_ratio_is_about_two() -> None:
  rows = loop_work_by_extent(
    _fold_scan,
    lambda n: (jnp.ones((n, 16), dtype=jnp.float32),),
    (8, 16, 32),
  )
  ratios = work_ratios(rows[0])
  assert all(1.7 <= ratio < 2.5 for ratio in ratios)
  with pytest.raises(AssertionError, match="work ratio"):
    assert_work_independent(rows)


def test_fixed_trip_quadratic_fori_ratio_is_about_four() -> None:
  rows = loop_work_by_extent(
    _quadratic_fori,
    lambda n: (jnp.ones((n, 8), dtype=jnp.float32),),
    (16, 32, 64),
  )
  row = pick_fixed_scan_largest(rows, expected_trips=(4, 4, 4))
  assert row.trips == [4, 4, 4]
  ratios = work_ratios(row)
  assert all(ratio > B_THRESHOLD for ratio in ratios)
  assert all(3.5 < ratio < 4.5 for ratio in ratios)
  with pytest.raises(AssertionError, match="work ratio"):
    assert_work_at_most_linear([row])


def test_path_mismatch_across_extents_raises() -> None:
  with pytest.raises(LoopStructureMismatchError):
    loop_work_by_extent(
      _maybe_two_scans,
      lambda n: (jnp.ones((n,), dtype=jnp.float32),),
      (8, 16, 32),
    )


def test_searchsorted_scan_is_exempt_from_criterion_a() -> None:
  rows = loop_work_by_extent(
    _searchsorted,
    lambda n: (jnp.arange(n),),
    (32, 64, 128),
  )
  assert len(rows) == 1
  assert not trips_scale(rows[0].trips)
  # The body carries O(L) work, so the ratio sits near 2. Criterion A would
  # fire if the trip test did not exempt it.
  assert all(ratio >= A_THRESHOLD for ratio in work_ratios(rows[0]))
  assert_work_independent(rows, require_any=False)


def test_fold_over_keeps_element_zero_and_is_visible_to_the_tracer() -> None:
  def total(xs: jax.Array) -> jax.Array:
    return jnp.sum(xs)

  folded = fold_over(total, n=4, arg_index=0)
  xs = jnp.arange(6, dtype=jnp.float32)
  assert float(folded(xs)) == pytest.approx(float(total(xs)))

  def scan_fold(xs: jax.Array) -> jax.Array:
    def body(carry: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
      recomputed = fold_over(lambda rows: rows @ rows.T, xs.shape[0], 0)(xs)
      return carry + jnp.sum(x) + jnp.float32(0) * jnp.sum(recomputed), x

    carry, _ys = jax.lax.scan(body, jnp.float32(0), xs)
    return carry

  rows = measure_rows(
    lambda n: (scan_fold, (jnp.ones((n, 4), dtype=jnp.float32),)),
    (8, 16),
  )
  assert work_ratios(rows[0])[0] > A_THRESHOLD


def test_marker_primitive_is_seen_only_inside_loops() -> None:
  def outside(xs: jax.Array) -> jax.Array:
    return mark_array(jnp.sum(xs))

  xs = jnp.ones((4,), dtype=jnp.float32)
  assert count_primitive_in_loops(outside, (xs,), ENCODER_MARK) == 0
  assert_no_primitive_in_loops(outside, (xs,), ENCODER_MARK)

  def inside(xs: jax.Array) -> jax.Array:
    def body(carry: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
      return carry + mark_array(x), x

    carry, _ys = jax.lax.scan(body, jnp.float32(0), xs)
    return carry

  assert count_primitive_in_loops(inside, (xs,), ENCODER_MARK) == 1
  with pytest.raises(AssertionError, match=ENCODER_MARK):
    assert_no_primitive_in_loops(inside, (xs,), ENCODER_MARK)


def test_work_independent_checks_every_trip_scaling_loop() -> None:
  """A clean outer loop must not hide a trip-scaling inner loop whose work grows."""
  clean = LoopRow("outer", "scan", [2, 4, 8], [10, 10, 10])
  dirty = LoopRow("inner", "scan", [2, 4, 8], [10, 20, 40])
  assert_work_independent([clean])
  with pytest.raises(AssertionError, match="inner"):
    assert_work_independent([clean, dirty])


def test_pick_trips_scaling_refuses_a_different_loop() -> None:
  """Wrong counts, and a heavier loop that stopped scaling, both fail the pick."""
  pair = LoopRow("pair", "scan", [5, 10, 20], [40, 40, 40])
  nested = LoopRow("nested", "scan", [3, 3, 3], [5, 5, 5])
  assert pick_trips_scaling([pair, nested], expected_trips=(5, 10, 20)).path == "pair"
  with pytest.raises(LookupError, match="refusing to select a different loop"):
    pick_trips_scaling([pair, nested], expected_trips=(2, 4, 8))

  stopped = LoopRow("outer", "scan", [4, 4, 4], [100, 100, 100])
  inner = LoopRow("inner", "scan", [8, 16, 32], [10, 20, 40])
  with pytest.raises(LookupError, match="refusing to substitute"):
    pick_trips_scaling([stopped, inner], expected_trips=(8, 16, 32))


def test_pick_fixed_scan_refuses_a_loop_whose_trips_changed() -> None:
  """A scaling scan with more work is not replaced by a smaller fixed scan."""
  outer = LoopRow("outer", "scan", [3, 3, 3], [100, 90, 80])
  scaling = LoopRow("inner", "scan", [32, 64, 128], [10, 20, 40])
  picked = pick_fixed_scan_largest([outer, scaling], expected_trips=(3, 3, 3))
  assert picked.path == "outer"
  assert picked.trips == [3, 3, 3]

  changed = LoopRow("outer", "scan", [3, 6, 12], [100, 200, 400])
  smaller = LoopRow("inner", "scan", [3, 3, 3], [10, 10, 10])
  with pytest.raises(LookupError, match="whose trips changed"):
    pick_fixed_scan_largest([changed, smaller], expected_trips=(3, 3, 3))
  with pytest.raises(LookupError, match="refusing to select a different loop"):
    pick_fixed_scan_largest([outer, smaller], expected_trips=(2, 2, 2))


def test_ar_fixture_copy_and_guard_model_shapes() -> None:
  ar = make_ar_model()
  guard = make_guard_model()
  assert ar.num_decoder_layers == 3
  assert guard.num_decoder_layers == 2
  assert float(jnp.max(jnp.abs(ar.w_s_embed.weight))) > float(jnp.max(jnp.abs(guard.w_s_embed.weight)))
  multi = structure(6, n_states=2)
  assert multi["coords"].shape == (2, 6, 4, 3)
  single = structure(6)
  assert single["coords"].shape == (6, 4, 3)
  built, _config = bundle(6)
  assert built.geometry.coords.shape[1] == 6
