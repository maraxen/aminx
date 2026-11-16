# tests/host/test_plan_topology_validator.py
from __future__ import annotations

import pytest

from aminx.host.plan import PlanTopologyError, _validate_plan_topology
from aminx.tiling.axes import N_NOISES, N_SAMPLES, N_STRUCTURES, N_TEMPERATURES
from aminx.types.boundaries import AxisBoundary
from aminx.types.stages import StageSet
from xtrax.tiling import AxisDecision, BatchPlan, ChunkedMap, Scan, Vmap


def _make_plan(decisions):
    # xtrax.tiling.BatchPlan only needs decisions (EPIC #1541 T-PLANNER.GATE
    # retired the richer local BatchPlan/its axes_by_index/budget_exceeded
    # fields, unused by _validate_plan_topology, which only reads .decisions)
    return BatchPlan(decisions=decisions)


def _vmap_decision(axis):
    return AxisDecision(spec=axis, batch_size=0, reasoning="test", strategy=Vmap())


def _chunked_map_decision(axis, batch_size=1):
    return AxisDecision(spec=axis, batch_size=batch_size, reasoning="test", strategy=ChunkedMap(batch_size=batch_size))


def _scan_decision(axis):
    return AxisDecision(spec=axis, batch_size=1,
    reasoning="test",
    strategy=Scan(init=None, transition=lambda c, x: (c, x), ordered_sinks=True),)


# --- ordered sink on Vmap axis ---


def test_validator_rejects_ordered_sink_on_vmap_axis():
    class OrderedSink:
        ordered = True

        def __call__(self, x) -> None:
            pass

    plan = _make_plan([_vmap_decision(N_NOISES)])
    stage_set = StageSet(
        axis_boundaries={"n_noises": AxisBoundary(sink=OrderedSink())}
    )
    with pytest.raises(PlanTopologyError, match="ordered.*Vmap"):
        _validate_plan_topology(plan, stage_set)


def test_validator_rejects_ordered_tap_on_vmap_axis():
    class OrderedTap:
        ordered = True

        def __call__(self, x):
            return x

    plan = _make_plan([_vmap_decision(N_NOISES)])
    stage_set = StageSet(
        axis_boundaries={"n_noises": AxisBoundary(tap=OrderedTap())}
    )
    with pytest.raises(PlanTopologyError, match="ordered.*Vmap"):
        _validate_plan_topology(plan, stage_set)


def test_validator_allows_unordered_sink_on_vmap_axis():
    class UnorderedSink:
        ordered = False

        def __call__(self, x) -> None:
            pass

    plan = _make_plan([_vmap_decision(N_NOISES)])
    stage_set = StageSet(
        axis_boundaries={"n_noises": AxisBoundary(sink=UnorderedSink())}
    )
    _validate_plan_topology(plan, stage_set)  # must not raise


def test_validator_allows_ordered_sink_on_scan_axis():
    class OrderedSink:
        ordered = True

        def __call__(self, x) -> None:
            pass

    plan = _make_plan([_scan_decision(N_NOISES)])
    stage_set = StageSet(
        axis_boundaries={"n_noises": AxisBoundary(sink=OrderedSink())}
    )
    _validate_plan_topology(plan, stage_set)  # must not raise


def test_validator_allows_ordered_sink_on_chunked_map_axis():
    class OrderedSink:
        ordered = True

        def __call__(self, x) -> None:
            pass

    plan = _make_plan([_chunked_map_decision(N_NOISES)])
    stage_set = StageSet(
        axis_boundaries={"n_noises": AxisBoundary(sink=OrderedSink())}
    )
    _validate_plan_topology(plan, stage_set)  # must not raise


# --- Scan on heterogeneous axis ---


def test_validator_rejects_scan_on_heterogeneous_axis():
    """Manually construct an invalid AxisDecision (bypassing CarrySpec guard)."""
    bad_decision = AxisDecision(spec=N_STRUCTURES, batch_size=1,
    reasoning="bad",
    strategy=Scan(init=None, transition=lambda c, x: (c, x)),)
    plan = _make_plan([bad_decision])
    stage_set = StageSet()
    with pytest.raises(PlanTopologyError, match="heterogeneous.*Scan"):
        _validate_plan_topology(plan, stage_set)


# --- Clean topology passes ---


def test_validator_passes_for_default_plan():
    plan = _make_plan([
        _chunked_map_decision(N_STRUCTURES),
        _vmap_decision(N_NOISES),
        _vmap_decision(N_TEMPERATURES),
        _vmap_decision(N_SAMPLES),
    ])
    stage_set = StageSet()
    _validate_plan_topology(plan, stage_set)  # no raise


def test_plan_topology_error_is_tiling_error_subclass():
    """PlanTopologyError is a TilingError (library-side error base)."""
    from aminx.tiling.errors import TilingError

    assert issubclass(PlanTopologyError, TilingError)
