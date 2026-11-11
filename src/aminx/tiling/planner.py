from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from xtrax.tiling import AxisSpec, BatchPlanner, ChunkedMap, MemoryBudget, Vmap

from aminx.tiling.memory_budget import resolve_memory_budget_bytes

if TYPE_CHECKING:
  from collections.abc import Sequence

  from xtrax.tiling import AxisDecision

  from aminx.tiling.strategy import AxisStrategy


def estimate_memory_theoretical(
  decisions: Sequence[AxisDecision],
  base_shape_bytes: float,
  activation_multiplier: float,
) -> float:
  """Estimate peak memory for a set of axis decisions.

  Vmap-strategy axes contribute their full cardinality to the memory
  product; all other strategies (ChunkedMap, Scan, DedupGather) contribute
  only their decided tile size (d.batch_size) -- one tile is live at a
  time.

  Checks type(d.strategy).__name__ == "Vmap" rather than d.batch_size == 0:
  xtrax.tiling.BatchPlanner's joint-budget mode sets batch_size=spec.default_batch_size
  even for Vmap decisions, never 0.

  activation_multiplier must be supplied by the caller (no default).
  """
  product = 1
  for d in decisions:
    if type(d.strategy).__name__ == "Vmap":
      product *= d.spec.cardinality
    else:
      product *= d.batch_size
  return base_shape_bytes * product * activation_multiplier


def plan_axis_strategy(
  axis_template: AxisSpec,
  cardinality: int,
  batch_size_override: int | None,
  *,
  activation_bytes_per_element: float,
  headroom: float = 0.80,
) -> AxisStrategy:
  """Resolve a Vmap/ChunkedMap strategy for one axis via xtrax's BatchPlanner.

  Args:
    axis_template: Axis template from :mod:`aminx.tiling.axes`; only its
      ``cardinality`` is replaced.
    cardinality: True element count on this axis for this call.
    batch_size_override: Fixed ChunkedMap batch_size. Bypasses the planner when > 0.
    activation_bytes_per_element: Live bytes one element of this axis holds.
    headroom: Fraction of the device limit the plan may occupy.

  Returns:
    A :class:`Vmap` or :class:`ChunkedMap` strategy.

  Raises:
    TypeError: If BatchPlanner returns a strategy other than Vmap/ChunkedMap.
  """
  if batch_size_override is not None and batch_size_override > 0:
    return ChunkedMap(batch_size=batch_size_override)

  axis = dataclasses.replace(axis_template, cardinality=cardinality)
  budget = MemoryBudget(
    bytes=resolve_memory_budget_bytes(headroom=headroom),
    estimate=lambda decisions: int(
      estimate_memory_theoretical(decisions, activation_bytes_per_element, 1.0),
    ),
  )
  plan = BatchPlanner(budget=budget).plan([axis])
  xtrax_strategy = plan.decisions[0].strategy
  if isinstance(xtrax_strategy, (ChunkedMap, Vmap)):
    return xtrax_strategy
  msg = f"plan_axis_strategy: unexpected BatchPlanner decision strategy {type(xtrax_strategy)}"
  raise TypeError(msg)
