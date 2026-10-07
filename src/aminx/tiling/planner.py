from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from xtrax.tiling import AxisSpec, BatchPlanner, MemoryBudget
from xtrax.tiling import ChunkedMap as XtraxChunkedMap
from xtrax.tiling import Vmap as XtraxVmap

from aminx.tiling.memory_budget import resolve_memory_budget_bytes
from aminx.tiling.strategy import SafeMap, Vmap

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
  product; all other strategies (SafeMap, Scan, DedupGather) contribute
  only their decided tile size (d.batch_size) -- one tile is live at a
  time.

  Checks type(d.strategy).__name__ == "Vmap" rather than d.batch_size == 0:
  xtrax.tiling.BatchPlanner's joint-budget mode (which this function is
  used with, as a MemoryBudget.estimate callable -- see host/plan.py's
  _plan_with_joint_budget) sets batch_size=spec.default_batch_size even for
  Vmap decisions, never 0 (EPIC #1541 T-PLANNER.2 finding, 2026-07-06). A
  d.batch_size == 0 check -- this module's own retired local BatchPlanner's
  convention -- would silently underestimate memory for every Vmap axis
  under xtrax's convention.

  activation_multiplier must be supplied by the caller (no default).

  Note: this is the one survivor of aminx.tiling.planner's original
  BatchPlanner/AxisSpec/AxisDecision/BatchPlan/ceil_to_granularity/
  resolve_safe_map_tile -- those migrated to xtrax.tiling.BatchPlanner's
  joint-budget mode (see host/plan.py's _plan_with_joint_budget and
  .praxia/docs/specs/260706_epic1541-planner-joint-budget-migration.md).
  This function's own math is unchanged and is now invoked through xtrax's
  engine rather than this module's retired copy of the demotion loop.
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
  """Resolve a Vmap/SafeMap strategy for one axis via xtrax's BatchPlanner.

  Composable-JAX primitive (see the ``using-xtrax`` skill and the
  ``feedback_xtrax_composable_primitives`` memory): never hand-roll
  ``jax.vmap``/``lax.map`` chunking at a call site. An explicit
  ``batch_size_override`` bypasses the planner with a fixed SafeMap tile (a
  guardrail for callers who already know their memory ceiling); otherwise the
  planner auto-demotes Vmap -> SafeMap when the device memory budget would be
  exceeded.

  Promoted here from ``aminx.sampling.conditional_logits._plan_axis_strategy``
  (2026-08-12). It had three consumers -- ``conditional_logits`` itself,
  ``mbr_consensus`` (which imported the private name across modules), and now
  the categorical Jacobian -- which is one too many for a leading underscore.
  ``conditional_logits._plan_axis_strategy`` remains as an alias so existing
  imports and tests keep working; the behavior is byte-identical.

  Args:
    axis_template: Axis template from :mod:`aminx.tiling.axes`; only its
      ``cardinality`` is replaced.
    cardinality: True element count on this axis for this call.
    batch_size_override: Fixed SafeMap tile. Bypasses the planner when > 0.
    activation_bytes_per_element: Live bytes one element of this axis holds.
      Drives the demotion decision, so an order-of-magnitude estimate is enough
      but a wildly wrong one is not.
    headroom: Fraction of the device limit the plan may occupy.

  Returns:
    An aminx-native :class:`Vmap` or :class:`SafeMap` strategy.

  Raises:
    TypeError: If BatchPlanner returns a strategy other than Vmap/SafeMap.
  """
  if batch_size_override is not None and batch_size_override > 0:
    return SafeMap(tile=batch_size_override)

  axis = dataclasses.replace(axis_template, cardinality=cardinality)
  budget = MemoryBudget(
    bytes=resolve_memory_budget_bytes(headroom=headroom),
    estimate=lambda decisions: int(
      estimate_memory_theoretical(decisions, activation_bytes_per_element, 1.0),
    ),
  )
  plan = BatchPlanner(budget=budget).plan([axis])
  xtrax_strategy = plan.decisions[0].strategy
  # make_axis_dispatch_via_xtrax expects aminx-native strategy objects (it
  # translates to xtrax-native internally via _strategy_to_xtrax) -- BatchPlanner
  # itself is xtrax-native, so translate its decision back before returning.
  if isinstance(xtrax_strategy, XtraxChunkedMap):
    return SafeMap(tile=xtrax_strategy.batch_size)
  if isinstance(xtrax_strategy, XtraxVmap):
    return Vmap()
  msg = f"plan_axis_strategy: unexpected BatchPlanner decision strategy {type(xtrax_strategy)}"
  raise TypeError(msg)
