"""Jaxpr loop-scaling guards shared by the debt #1981 complexity tests.

``loop_bodies`` only traces. Work is an explicit proxy (one matmul term plus
output sizes), so these helpers compare ratios of the same loop across extents.
They never execute the program and never measure ``while`` trip counts.

Criterion A: a loop that iterates once per element of an axis must do
extent-independent work per iteration (ratio < 1.5). Loops whose trip count
grows slower than that (``jnp.searchsorted``) are exempt.

Criterion B: a fixed-trip loop over an L-sized state must grow at most linearly
(ratio < 3.0), and its trip count must not start depending on L.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import batching
from xtrax.profiling.jaxpr import sub_jaxprs
from xtrax.profiling.loop_scaling import LoopStructureMismatchError, loop_bodies

#: Per-iteration work ratio at or above which criterion A fails. Same split
#: xtrax uses between "independent" (~1) and "recomputes the extent" (~2).
A_THRESHOLD = 1.5
#: Per-iteration work ratio at or above which criterion B fails. Sits between
#: linear (~2) and quadratic (~4).
B_THRESHOLD = 3.0
#: A must-fire mutant has to clear the threshold with this margin.
A_MUTANT_MIN = 1.7
#: A criterion-B mutant has to land strictly above this.
B_MUTANT_MIN = 3.5
#: Trip-count ratio at or above which a loop is treated as iterating over the swept axis.
TRIP_SCALING_RATIO = 1.5

#: Identity primitive a test inserts where a full model call stands in for the encoder.
ENCODER_MARK = "aminx_encoder_mark"

_encoder_mark_p = Primitive(ENCODER_MARK)


def _encoder_mark_impl(value: jax.Array) -> jax.Array:
  return value


_encoder_mark_p.def_impl(_encoder_mark_impl)
_encoder_mark_p.def_abstract_eval(lambda value: value)
# Encode (and a chunked pair map) vmaps the model call. The mark is an identity,
# so it keeps the batch axis of its input.
batching.defvectorized(_encoder_mark_p)


def mark_array(value: jax.Array) -> jax.Array:
  """Return ``value`` unchanged, leaving ``ENCODER_MARK`` in the jaxpr.

  Multiplying the mark by zero still keeps the primitive, so a later trace can
  see that the marked call ran inside a loop.
  """
  return _encoder_mark_p.bind(value)


@dataclass(frozen=True)
class LoopRow:
  """One loop body compared at every requested extent, in extent order."""

  path: str
  primitive: str
  trips: list[int | None]
  work: list[int]


def _rows_from_bodies(found: list[list[Any]], extents: Sequence[int]) -> list[LoopRow]:
  paths = [[body.path for body in row] for row in found]
  if any(row != paths[0] for row in paths[1:]):
    msg = f"loop structure differs across extents {list(extents)}: {paths}"
    raise LoopStructureMismatchError(msg)
  rows: list[LoopRow] = []
  for index, body in enumerate(found[0]):
    rows.append(
      LoopRow(
        path=str(body.path),
        primitive=str(body.primitive),
        trips=[row[index].trip_count for row in found],
        work=[int(row[index].iteration_work) for row in found],
      )
    )
  return rows


def loop_work_by_extent(
  fn: Callable[..., Any],
  make_args: Callable[[int], tuple[Any, ...]],
  extents: Sequence[int],
) -> list[LoopRow]:
  """Trace ``fn(*make_args(extent))`` at each extent and align loops by path.

  ``jax.clear_caches`` runs first: a cached ``filter_jit`` jaxpr would hide a
  monkeypatched body of the same shape. Path mismatch across extents is an
  error, not a skip. The extent has to arrive as an argument of ``fn``. A
  no-argument wrapper that closes over the extent is cached by ``make_jaxpr``
  and every extent then reports the first trace (``measure_rows`` avoids that
  by tracing a fresh callable per extent).

  Args:
    fn: Program to trace. Never executed.
    make_args: Maps an extent to the positional arguments of ``fn``.
    extents: Two or more increasing sizes of the swept axis.

  Returns:
    One row per loop, outermost first, the same order ``loop_bodies`` uses.
  """
  if len(extents) < 2:
    msg = f"need at least two extents, got {list(extents)}"
    raise ValueError(msg)
  jax.clear_caches()
  found = [loop_bodies(fn, *make_args(int(extent))) for extent in extents]
  return _rows_from_bodies(found, extents)


def per_extent(
  trace_at: Callable[[int], tuple[Callable[..., Any], tuple[Any, ...]]],
) -> tuple[Callable[..., Any], Callable[[int], tuple[Any, ...]]]:
  """Adapt a per-extent ``(fn, args)`` builder to ``loop_work_by_extent``.

  Mutants close over the extent-sized tensor, so the traced callable has to be
  rebuilt at every extent (the shape used by the graded probe). ``make_args``
  stores that callable; ``fn`` forwards to it. Call ``make_args`` before ``fn``.
  """
  box: dict[str, Callable[..., Any]] = {}

  def fn(*args: Any, **kwargs: Any) -> Any:
    return box["fn"](*args, **kwargs)

  def make_args(extent: int) -> tuple[Any, ...]:
    built, args = trace_at(int(extent))
    box["fn"] = built
    return args

  return fn, make_args


def measure_rows(
  trace_at: Callable[[int], tuple[Callable[..., Any], tuple[Any, ...]]],
  extents: Sequence[int],
) -> list[LoopRow]:
  """Trace a fresh callable from ``trace_at`` at each extent.

  ``make_jaxpr`` caches a no-argument function, so a mutant that closes over
  the extent-sized tensor has to be a new function object at every extent (the
  shape the graded probe traces).
  """
  if len(extents) < 2:
    msg = f"need at least two extents, got {list(extents)}"
    raise ValueError(msg)
  jax.clear_caches()
  found: list[list[Any]] = []
  for extent in extents:
    fn, args = trace_at(int(extent))
    found.append(loop_bodies(fn, *args))
  return _rows_from_bodies(found, extents)


def work_ratios(row: LoopRow) -> list[float]:
  """Per-iteration work ratios between consecutive extents."""
  ratios: list[float] = []
  for small, large in zip(row.work, row.work[1:], strict=False):
    if small <= 0 or large <= 0:
      msg = f"non-positive iteration work {row.work} at {row.path}"
      raise ValueError(msg)
    ratios.append(large / small)
  return ratios


def trips_scale(trips: Sequence[int | None], threshold: float = TRIP_SCALING_RATIO) -> bool:
  """Whether ``trips`` grow with the swept axis.

  A ``None`` count (a ``while``, or an unmeasured trip count) counts as
  scaling. A static count scales only when every consecutive ratio reaches
  ``threshold``. A binary-search scan (one extra trip when the extent doubles)
  stays below the threshold and is exempt from criterion A.
  """
  if len(trips) < 2:
    return False
  saw = False
  for small, large in zip(trips, trips[1:], strict=False):
    saw = True
    if small is None or large is None:
      continue
    if small <= 0 or large / small < threshold:
      return False
  return saw


def scaling_summary(rows: Sequence[LoopRow]) -> str:
  """One block of path, trips, work and ratios for an assertion message."""
  lines: list[str] = []
  for row in rows:
    lines.append(
      f"{row.path or '<root>'} {row.primitive} trips={row.trips} work={row.work} "
      f"ratios={work_ratios(row)}"
    )
  return "\n".join(lines)


def independence_violations(
  rows: Sequence[LoopRow],
  *,
  only_trips_scaling: bool = True,
  threshold: float = A_THRESHOLD,
  require_any: bool = True,
) -> list[str]:
  """Criterion A failures: trip-scaling loops whose per-iteration work grows.

  Args:
    rows: Loops aligned across the same extents.
    only_trips_scaling: Skip loops whose trip count does not scale. That is the
      ``searchsorted`` exemption.
    threshold: A ratio at or above this fails.
    require_any: When true, zero trip-scaling loops is itself a violation, so a
      refactor that deletes the loop cannot pass silently.

  Returns:
    Human-readable violations. Empty means the rows pass criterion A.
  """
  violations: list[str] = []
  checked = 0
  for row in rows:
    if only_trips_scaling and not trips_scale(row.trips, threshold):
      continue
    checked += 1
    for ratio in work_ratios(row):
      if ratio >= threshold:
        violations.append(
          f"{row.path or '<root>'} {row.primitive} work ratio {ratio:.4f} "
          f">= {threshold} trips={row.trips} work={row.work}"
        )
  if require_any and checked == 0:
    violations.append("no trip-scaling loop; the guard would be vacuous")
  return violations


def assert_work_independent(
  rows: Sequence[LoopRow],
  *,
  only_trips_scaling: bool = True,
  threshold: float = A_THRESHOLD,
  require_any: bool = True,
) -> None:
  """Assert criterion A. Raises ``AssertionError`` listing every failing loop."""
  violations = independence_violations(
    rows,
    only_trips_scaling=only_trips_scaling,
    threshold=threshold,
    require_any=require_any,
  )
  assert not violations, scaling_summary(rows) + "\n" + "\n".join(violations)


def linearity_violations(
  rows: Sequence[LoopRow],
  *,
  threshold: float = B_THRESHOLD,
) -> list[str]:
  """Criterion B failures: fixed-trip work above ``threshold``, or trips that move.

  Returns:
    Human-readable violations. Empty means the rows pass criterion B.
  """
  if not rows:
    return ["no loop; the guard would be vacuous"]
  violations: list[str] = []
  for row in rows:
    if any(trip != row.trips[0] for trip in row.trips):
      violations.append(f"trips must not depend on L: {row.path} trips={row.trips}")
    for ratio in work_ratios(row):
      if ratio >= threshold:
        violations.append(
          f"{row.path or '<root>'} {row.primitive} work ratio {ratio:.4f} "
          f">= {threshold} trips={row.trips} work={row.work}"
        )
  return violations


def assert_work_at_most_linear(
  rows: Sequence[LoopRow],
  *,
  threshold: float = B_THRESHOLD,
) -> None:
  """Assert criterion B. Raises ``AssertionError`` listing every failing loop."""
  violations = linearity_violations(rows, threshold=threshold)
  assert not violations, scaling_summary(rows) + "\n" + "\n".join(violations)


def pick_outermost_while(rows: Sequence[LoopRow]) -> LoopRow:
  """First ``while`` (outermost; rows are parent-before-child)."""
  for row in rows:
    if row.primitive == "while":
      return row
  msg = f"no while loop among {[row.primitive for row in rows]}"
  raise LookupError(msg)


def _expected_trip_list(expected_trips: Sequence[int]) -> list[int]:
  return [int(trip) for trip in expected_trips]


def pick_trips_scaling(rows: Sequence[LoopRow], *, expected_trips: Sequence[int]) -> LoopRow:
  """Outermost static-doubling loop, which must have ``expected_trips``.

  The guarded loop is named by its trip counts. Two substitutions are refused.
  A loop that still doubles, but at the wrong counts, is not replaced by an
  inner loop that happens to have ``expected_trips``. A heavier loop listed
  earlier (it contains the candidate) whose trips no longer double is not
  skipped in favour of that inner loop.
  """
  expected = _expected_trip_list(expected_trips)
  chosen: LoopRow | None = None
  for row in rows:
    if any(trip is None or trip <= 0 for trip in row.trips):
      continue
    if not trips_scale(row.trips):
      continue
    ratios = [
      large / small
      for small, large in zip(row.trips, row.trips[1:], strict=False)
      if small is not None and large is not None and small > 0
    ]
    if ratios and all(ratio >= TRIP_SCALING_RATIO for ratio in ratios):
      chosen = row
      break
  if chosen is None:
    msg = f"no trip-scaling loop among {[row.trips for row in rows]}"
    raise LookupError(msg)
  if list(chosen.trips) != expected:
    msg = (
      f"outermost trip-scaling loop {chosen.path} has trips {chosen.trips}, "
      f"not {expected}; refusing to select a different loop"
    )
    raise LookupError(msg)
  for row in rows:
    if row is chosen:
      break
    if row.primitive in {"scan", "while"} and row.work[0] > chosen.work[0]:
      msg = (
        f"refusing to substitute {chosen.path} for heavier loop {row.path} "
        f"trips={row.trips} work={row.work[0]}"
      )
      raise LookupError(msg)
  return chosen


def pick_fixed_scan_largest(rows: Sequence[LoopRow], *, expected_trips: Sequence[int]) -> LoopRow:
  """Largest fixed-trip scan, which must have ``expected_trips``.

  Scans whose trip count changes across extents are not candidates. If one of
  those has at least as much per-iteration work as the selected scan, the
  intended loop changed shape and this raises instead of returning a smaller
  fixed scan that still has constant trips.
  """
  expected = _expected_trip_list(expected_trips)
  best: LoopRow | None = None
  best_work = -1
  varying: list[LoopRow] = []
  for row in rows:
    if row.primitive != "scan" or row.trips[0] is None:
      continue
    if any(trip != row.trips[0] for trip in row.trips):
      varying.append(row)
      continue
    if row.work[0] > best_work:
      best = row
      best_work = row.work[0]
  if best is None:
    msg = "no fixed-trip scan"
    raise LookupError(msg)
  if list(best.trips) != expected:
    msg = (
      f"largest fixed scan {best.path} has trips {best.trips}, not {expected}; "
      "refusing to select a different loop"
    )
    raise LookupError(msg)
  heavier = [row for row in varying if row.work[0] >= best.work[0]]
  if heavier:
    detail = ", ".join(f"{row.path} trips={row.trips} work={row.work[0]}" for row in heavier)
    msg = (
      "refusing to substitute a smaller fixed scan for a loop whose trips changed: "
      f"{detail}"
    )
    raise LookupError(msg)
  return best


def _count_primitive(jaxpr: Any, primitive: str) -> int:
  total = 0
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == primitive:
      total += 1
    for sub in sub_jaxprs(eqn.params):
      total += _count_primitive(sub, primitive)
  return total


def _count_in_loops(jaxpr: Any, primitive: str) -> int:
  total = 0
  for eqn in jaxpr.eqns:
    if eqn.primitive.name in {"scan", "while"}:
      for sub in sub_jaxprs(eqn.params):
        total += _count_primitive(sub, primitive)
    else:
      for sub in sub_jaxprs(eqn.params):
        total += _count_in_loops(sub, primitive)
  return total


def count_primitive_in_loops(
  fn: Callable[..., Any],
  args: tuple[Any, ...],
  primitive: str,
) -> int:
  """How many times ``primitive`` appears inside any ``scan`` or ``while`` body.

  The walk enters ``pjit`` and ``cond`` so a marked call nested in the pair-map
  kernel is visible. Marks outside every loop are ignored.

  ``jax.clear_caches`` runs first so a jaxpr cached by an earlier test cannot
  change this count.
  """
  jax.clear_caches()
  closed = jax.make_jaxpr(fn)(*args)
  jaxpr = closed.jaxpr
  return _count_in_loops(jaxpr, primitive)


def assert_no_primitive_in_loops(
  fn: Callable[..., Any],
  args: tuple[Any, ...],
  primitive: str,
) -> None:
  """Assert ``primitive`` does not run inside a loop body."""
  count = count_primitive_in_loops(fn, args, primitive)
  assert count == 0, f"primitive {primitive} ran inside a loop ({count} times)"


def fold_over(fn: Callable[..., Any], n: int, arg_index: int) -> Callable[..., Any]:
  """Re-run ``fn`` on ``n`` perturbed copies and keep element 0.

  This is the shape of the autoregressive bug, and the only mutation that moves
  the work ratio (A12). A small injected einsum changes the proxy by about a
  tenth of a percent and does not clear the must-fire margin. The index is
  added into argument ``arg_index`` so the ``vmap`` reaches the body: a ``vmap``
  that ignores its index is deleted by ``make_jaxpr`` when only element 0 is
  kept, and the fold then costs nothing.

  Args:
    fn: Body to recompute. Called with the same positional arguments.
    n: Number of copies. Use the swept extent.
    arg_index: Which positional argument receives ``+ index``.

  Returns:
    A function with ``fn``'s positional signature.
  """

  def folded(*args: Any) -> Any:
    def one(index: jax.Array) -> Any:
      replaced = list(args)

      def shift(leaf: Any) -> Any:
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.number):
          return leaf + index.astype(leaf.dtype)
        return leaf

      replaced[arg_index] = jax.tree.map(shift, replaced[arg_index])
      return fn(*replaced)

    outs = jax.vmap(one)(jnp.arange(n))
    return jax.tree.map(lambda leaf: leaf[0], outs)

  return folded
