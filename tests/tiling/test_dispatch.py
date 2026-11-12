"""Test suite for make_axis_dispatch library-side factory contract (Task S6-A2).

This module verifies the dispatch logic that converts AxisStrategy instances
into typed iterators, with rejection logic for invalid axis/strategy pairs.
"""

from __future__ import annotations

import pytest

from aminx.host.plan import PlanTopologyError
from aminx.tiling.dispatch import DispatchRejected, make_axis_dispatch
from aminx.tiling.errors import TilingError
from aminx.tiling.strategy import ChunkedMap, Scan, Vmap


class TestMakeAxisDispatchHappyPath:
    """Happy path: valid strategy/axis pairs dispatch to correct iterator."""

    def test_vmap_dispatch_returns_vmap_iterator(self) -> None:
        """make_axis_dispatch(Vmap()) returns a VmapIterator."""
        result = make_axis_dispatch(Vmap())
        assert result is not None
        # Verify it's the right type by checking the class name.
        assert type(result).__name__ == "VmapIterator"

    def test_chunked_map_dispatch_returns_chunked_map_iterator(self) -> None:
        """make_axis_dispatch(ChunkedMap(batch_size=4)) returns a ChunkedMapIterator with tile=4."""
        result = make_axis_dispatch(ChunkedMap(batch_size=4))
        assert result is not None
        assert type(result).__name__ == "ChunkedMapIterator"
        # ChunkedMapIterator has a static tile field.
        assert result.tile == 4

    def test_scan_on_homogeneous_axis_returns_jax_scan_iterator(self) -> None:
        """make_axis_dispatch(Scan(...), axis='wave') returns a JaxScanIterator."""
        import jax.numpy as jnp

        scan_strategy = Scan(init=jnp.int32(0), transition=lambda c, x: (c + x, c + x))
        result = make_axis_dispatch(scan_strategy, axis="wave")
        assert type(result).__name__ == "JaxScanIterator"
        final_carry, ys = result(scan_strategy.transition, scan_strategy.init, jnp.array([1, 2, 3]))
        assert int(final_carry) == 6
        assert list(map(int, ys)) == [1, 3, 6]


class TestMakeAxisDispatchRejectPath:
    """Reject path: Scan on heterogeneous (state) axis raises DispatchRejected."""

    def test_scan_on_state_axis_raises_dispatch_rejected(self) -> None:
        """make_axis_dispatch(Scan(...), axis="state") raises DispatchRejected."""
        # Create a minimal Scan strategy.
        scan_strategy = Scan(init=0, transition=lambda c, x: (c, x))

        with pytest.raises(DispatchRejected) as exc_info:
            make_axis_dispatch(scan_strategy, axis="state")

        error_msg = str(exc_info.value)
        # Error message must mention both "state axis" and "heterogeneous".
        assert "state axis" in error_msg.lower()
        assert "heterogeneous" in error_msg.lower()

    def test_dispatch_rejected_and_plan_topology_error_share_base(self) -> None:
        """DispatchRejected and PlanTopologyError both subclass TilingError.

        This enforces the library/app dependency boundary: both errors have a
        common base (TilingError) in the library, allowing callers to ``except
        TilingError`` to catch both, without making tiling/ depend on host/.
        """
        assert issubclass(DispatchRejected, TilingError)
        assert issubclass(PlanTopologyError, TilingError)

    def test_make_axis_dispatch_rejects_dedup_gather(self) -> None:
        """make_axis_dispatch(DedupGather(...)) raises DispatchRejected.

        DedupGather is handled by _dispatch_axis in kernel_dispatch.py, not by
        make_axis_dispatch (which maps to iterator types). Passing DedupGather
        here must not fall through to TypeError.
        """
        from aminx.tiling.strategy import DedupGather
        import numpy as np

        dg = DedupGather(
            unique_indices=np.array([0, 1], dtype=np.int32),
            index_map=np.array([0, 1, 0, 1], dtype=np.int32),
            k=2,
            k_bucket=2,
            dedup_fn=lambda xs, idx: xs[idx],
            gather_fn=lambda ys, idx: ys[idx],
        )
        with pytest.raises(DispatchRejected):
            make_axis_dispatch(dg)
