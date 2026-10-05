"""Regression tests for host.plan._plan_with_joint_budget (EPIC #1541).

Originally written as T-PLANNER.GATE's old-vs-new parity check, comparing
the retired local aminx.tiling.planner.BatchPlanner against this wrapper
around xtrax.tiling.BatchPlanner's joint-budget mode. The gate passed (4/4
scenarios matched, including the deliberate infeasible-budget behavior
change) and the local planner was deleted; these assertions are kept as
permanent regression coverage for the behaviors the gate specifically
existed to verify, since they aren't redundant with any other test:

- The heterogeneous-axis-forces-SafeMap wrapper fix (_plan_with_joint_budget's
  entire reason for existing over a bare xtrax.tiling.BatchPlanner(budget=...)
  call) -- confirmed empirically (2026-07-06) that xtrax's engine alone
  assigns Vmap to a heterogeneous axis whenever the joint estimate fits.
- Fail-loud PlanBudgetInfeasibleError (replacing the old silent
  budget_exceeded=True) -- see 260706_epic1541-planner-joint-budget-migration.md.
- Demotion order and stopping-at-first-fit behavior on the real axes.py
  registry.
"""

from __future__ import annotations

import dataclasses

import pytest

from aminx.host.plan import PlanBudgetInfeasibleError, _plan_with_joint_budget
from aminx.tiling.axes import N_NOISES, N_SAMPLES, N_STRUCTURES, N_TEMPERATURES
from aminx.tiling.planner import estimate_memory_theoretical


def _axes(n_structures=32, n_samples=128, n_temperatures=8, n_noises=8):
    return [
        dataclasses.replace(N_STRUCTURES, cardinality=n_structures),
        dataclasses.replace(N_SAMPLES, cardinality=n_samples),
        dataclasses.replace(N_TEMPERATURES, cardinality=n_temperatures),
        dataclasses.replace(N_NOISES, cardinality=n_noises),
    ]


def _estimate_fn(ds):
    return estimate_memory_theoretical(ds, 1.0, 2.5)


def _plan(axes, budget_bytes):
    return _plan_with_joint_budget(axes, budget_bytes=budget_bytes, estimate_fn=_estimate_fn)


def _shape(plan) -> dict[str, tuple[str, int]]:
    return {d.spec.name: (type(d.strategy).__name__, d.batch_size) for d in plan.decisions}


def test_heterogeneous_axis_always_safemap_regardless_of_budget():
    """n_structures (heterogeneous=True) must never be Vmap, even when the
    joint estimate comfortably fits budget -- the exact scenario where
    xtrax's bare BatchPlanner(budget=...) was confirmed to assign Vmap
    incorrectly (T-PLANNER.2 finding)."""
    plan = _plan(_axes(), budget_bytes=int(8e9))
    shape = _shape(plan)
    assert shape["n_structures"][0] == "ChunkedMap"


def test_comfortable_budget_keeps_homogeneous_axes_vmap():
    """Everything except n_structures stays Vmap when the budget fits easily."""
    plan = _plan(_axes(), budget_bytes=int(8e9))
    shape = _shape(plan)
    assert shape["n_samples"][0] == "Vmap"
    assert shape["n_temperatures"][0] == "Vmap"
    assert shape["n_noises"][0] == "Vmap"


def test_tight_budget_demotes_exactly_one_axis_in_order():
    """A budget that fits after demoting only the first candidate (n_samples,
    first in the remaining-axes list after n_structures is pre-fixed) stops
    there -- n_temperatures/n_noises stay Vmap."""
    plan = _plan(_axes(), budget_bytes=5000)
    shape = _shape(plan)
    assert shape["n_samples"] == ("ChunkedMap", 1)
    assert shape["n_temperatures"][0] == "Vmap"
    assert shape["n_noises"][0] == "Vmap"


def test_infeasible_budget_raises_with_useful_message():
    """A budget nothing can fit, even fully demoted, raises
    PlanBudgetInfeasibleError (translated from xtrax's plain Exception into
    aminx's TilingError hierarchy) -- not a silently-returned over-budget
    plan, per the deliberate tightening this migration introduces."""
    with pytest.raises(PlanBudgetInfeasibleError):
        _plan(_axes(), budget_bytes=1)


def test_small_cardinalities_need_no_demotion():
    plan = _plan(_axes(n_structures=4, n_samples=4, n_temperatures=1, n_noises=1), budget_bytes=int(8e9))
    shape = _shape(plan)
    assert shape["n_samples"][0] == "Vmap"
    assert shape["n_temperatures"][0] == "Vmap"
    assert shape["n_noises"][0] == "Vmap"
    assert shape["n_structures"][0] == "ChunkedMap"
