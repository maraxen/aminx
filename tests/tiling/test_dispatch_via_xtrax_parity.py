"""Parity suite for make_axis_dispatch_via_xtrax vs make_axis_dispatch.

EPIC #1541 / T2.4 (`.praxia/docs/specs/260611_aminx-xtrax-refactor.md`).
Mirrors tests/tiling/test_dispatch.py's assertions exactly, but against the
xtrax-backed `make_axis_dispatch_via_xtrax`, plus direct side-by-side checks.
This is the shadow-parity evidence for flipping `factory.py` (and friends)
from `make_axis_dispatch` to `make_axis_dispatch_via_xtrax` once T2.GATE
passes -- if this suite passes, the xtrax-backed path is behaviorally
indistinguishable from the current one for every case the original suite
covers.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from aminx.host.plan import PlanTopologyError
from aminx.tiling.dispatch import (
    _DISPATCH_HETEROGENEOUS_AXES,
    DispatchRejected,
    make_axis_dispatch,
    make_axis_dispatch_via_xtrax,
)
from aminx.tiling.errors import TilingError
from aminx.tiling.strategy import DedupGather, SafeMap, Scan, Vmap


class TestMakeAxisDispatchViaXtraxHappyPath:
    """Mirrors TestMakeAxisDispatchHappyPath in test_dispatch.py."""

    def test_vmap_dispatch_returns_vmap_iterator(self) -> None:
        result = make_axis_dispatch_via_xtrax(Vmap())
        assert result is not None
        assert type(result).__name__ == "VmapIterator"

    def test_safemap_dispatch_returns_safemap_iterator(self) -> None:
        result = make_axis_dispatch_via_xtrax(SafeMap(tile=4))
        assert result is not None
        assert type(result).__name__ == "ChunkedMapIterator"  # xtrax's iterator (xtrax #3644)
        assert result.tile == 4


class TestMakeAxisDispatchViaXtraxRejectPath:
    """Mirrors TestMakeAxisDispatchRejectPath in test_dispatch.py."""

    def test_scan_on_state_axis_raises_dispatch_rejected(self) -> None:
        scan_strategy = Scan(init=0, transition=lambda c, x: (c, x))

        with pytest.raises(DispatchRejected) as exc_info:
            make_axis_dispatch_via_xtrax(scan_strategy, axis="state")

        error_msg = str(exc_info.value)
        assert "state axis" in error_msg.lower()
        assert "heterogeneous" in error_msg.lower()

    def test_dispatch_rejected_is_still_a_tilingerror(self) -> None:
        """The translated exception (aminx's own DispatchRejected) preserves
        the TilingError contract -- callers doing `except TilingError` keep
        working even though xtrax's own DispatchRejected does not subclass it.
        """
        assert issubclass(DispatchRejected, TilingError)
        assert issubclass(PlanTopologyError, TilingError)

    def test_make_axis_dispatch_via_xtrax_rejects_dedup_gather(self) -> None:
        dg = DedupGather(
            unique_indices=np.array([0, 1], dtype=np.int32),
            index_map=np.array([0, 1, 0, 1], dtype=np.int32),
            k=2,
            k_bucket=2,
        )
        with pytest.raises(DispatchRejected):
            make_axis_dispatch_via_xtrax(dg)


class TestSideBySideParity:
    """Direct comparison: same input, same observable outcome, both paths."""

    def test_vmap_produces_same_iterator_type_on_both_paths(self) -> None:
        legacy = make_axis_dispatch(Vmap())
        via_xtrax = make_axis_dispatch_via_xtrax(Vmap())
        assert type(legacy).__name__ == type(via_xtrax).__name__ == "VmapIterator"

    def test_safemap_produces_same_tile_value_on_both_paths(self) -> None:
        legacy = make_axis_dispatch(SafeMap(tile=7))
        via_xtrax = make_axis_dispatch_via_xtrax(SafeMap(tile=7))
        assert legacy.tile == via_xtrax.tile == 7

    def test_scan_on_state_rejected_on_both_paths(self) -> None:
        scan_strategy = Scan(init=0, transition=lambda c, x: (c, x))
        with pytest.raises(DispatchRejected):
            make_axis_dispatch(scan_strategy, axis="state")
        with pytest.raises(DispatchRejected):
            make_axis_dispatch_via_xtrax(scan_strategy, axis="state")

    def test_scan_on_non_state_axis_accepted_on_both_paths(self) -> None:
        """Confirms the heterogeneous_axes translation covers the non-rejection
        case too, not just the rejection case -- Scan on a non-heterogeneous
        axis must be accepted on both paths, not just "doesn't crash."
        """
        scan_strategy = Scan(init=0, transition=lambda c, x: (c, x))
        legacy = make_axis_dispatch(scan_strategy, axis="wave")
        via_xtrax = make_axis_dispatch_via_xtrax(scan_strategy, axis="wave")
        assert type(legacy).__name__ == type(via_xtrax).__name__ == "JaxScanIterator"

    def test_dedup_gather_rejected_on_both_paths(self) -> None:
        dg = DedupGather(
            unique_indices=np.array([0, 1], dtype=np.int32),
            index_map=np.array([0, 1, 0, 1], dtype=np.int32),
            k=2,
            k_bucket=2,
        )
        with pytest.raises(DispatchRejected):
            make_axis_dispatch(dg)
        with pytest.raises(DispatchRejected):
            make_axis_dispatch_via_xtrax(dg)


class TestHeterogeneousAxesSingleSourceOfTruth:
    """Both dispatch paths must derive rejection from _DISPATCH_HETEROGENEOUS_AXES.

    The legacy path used to hardcode `axis == "state"` while only the xtrax path
    read the constant, so adding an axis to the constant would have started
    rejecting on one path and not the other. These tests are parameterized over
    the constant's *contents* rather than the literal "state", so adding an axis
    extends coverage automatically instead of silently escaping it.
    """

    @pytest.mark.parametrize("axis", sorted(_DISPATCH_HETEROGENEOUS_AXES))
    def test_every_declared_heterogeneous_axis_rejects_scan_on_both_paths(
        self, axis: str
    ) -> None:
        scan_strategy = Scan(init=0, transition=lambda c, x: (c, x))
        with pytest.raises(DispatchRejected):
            make_axis_dispatch(scan_strategy, axis=axis)
        with pytest.raises(DispatchRejected):
            make_axis_dispatch_via_xtrax(scan_strategy, axis=axis)

    @pytest.mark.parametrize("axis", ["wave", "samples", "n_structures", "batch"])
    def test_undeclared_axes_accept_scan_on_both_paths(self, axis: str) -> None:
        """An axis absent from the constant must be accepted by both paths.

        `n_structures` is included deliberately: it appears in host/plan.py's
        separate _HETEROGENEOUS_AXIS_NAMES, and this pins that the two constants
        are genuinely independent rather than one silently shadowing the other.
        """
        assert axis not in _DISPATCH_HETEROGENEOUS_AXES
        scan_strategy = Scan(init=0, transition=lambda c, x: (c, x))
        legacy = make_axis_dispatch(scan_strategy, axis=axis)
        via_xtrax = make_axis_dispatch_via_xtrax(scan_strategy, axis=axis)
        assert type(legacy).__name__ == type(via_xtrax).__name__ == "JaxScanIterator"

    def test_constant_is_what_the_xtrax_path_forwards(self) -> None:
        """Guard the wiring itself: the xtrax path must forward this constant.

        A rename or a stray literal in `make_axis_dispatch_via_xtrax` would leave
        every test above passing while the two paths drift apart again.
        """
        source = inspect.getsource(make_axis_dispatch_via_xtrax)
        assert "heterogeneous_axes=set(_DISPATCH_HETEROGENEOUS_AXES)" in source
