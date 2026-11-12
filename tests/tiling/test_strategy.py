# tests/tiling/test_strategy.py
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.tiling.strategy import AxisStrategy, ChunkedMap, DedupGather, Scan, Vmap


def test_vmap_is_frozen_dataclass():
    v = Vmap()
    assert v == Vmap()  # equality by value


def test_chunked_map_stores_batch_size():
    s = ChunkedMap(batch_size=4)
    assert s.batch_size == 4
    assert s == ChunkedMap(batch_size=4)
    assert s != ChunkedMap(batch_size=8)


def test_scan_stores_init_and_transition():
    init = jnp.zeros((5, 32))

    def transition(carry, x):
        return carry + x, carry

    s = Scan(init=init, transition=transition)
    assert s.ordered_sinks is True  # default


def test_scan_ordered_sinks_configurable():
    s = Scan(init=None, transition=lambda c, x: (c, x), ordered_sinks=False)
    assert s.ordered_sinks is False


def test_axis_strategy_union_isinstance():
    assert isinstance(Vmap(), (Vmap, ChunkedMap, Scan, DedupGather))
    assert isinstance(ChunkedMap(batch_size=2), (Vmap, ChunkedMap, Scan, DedupGather))
    assert isinstance(Scan(init=None, transition=lambda c, x: (c, x)), (Vmap, ChunkedMap, Scan, DedupGather))


def test_scan_transition_protocol_conformance():
    from aminx.types.tiling_protocols import ScanTransition

    # A function with (carry, x) -> (carry, y) signature satisfies the protocol
    def my_transition(carry: int, x: int) -> tuple[int, int]:
        return carry + x, carry

    assert isinstance(my_transition, ScanTransition)


def _dedup_fn(xs, unique_indices):
    return xs[unique_indices]


def _gather_fn(ys_unique, index_map):
    return ys_unique[index_map]


def test_dedup_gather_stores_fields():
    unique_indices = np.array([0, 1], dtype=np.int32)
    index_map = np.array([0, 1, 0, 1], dtype=np.int32)
    dg = DedupGather(
        unique_indices=unique_indices,
        index_map=index_map,
        k=2,
        k_bucket=2,
        dedup_fn=_dedup_fn,
        gather_fn=_gather_fn,
    )
    assert dg.k == 2
    assert dg.k_bucket == 2
    assert np.array_equal(dg.unique_indices, unique_indices)
    assert np.array_equal(dg.index_map, index_map)


def test_dedup_gather_in_union():
    unique_indices = np.array([0], dtype=np.int32)
    index_map = np.array([0, 0], dtype=np.int32)
    dg = DedupGather(
        unique_indices=unique_indices,
        index_map=index_map,
        k=1,
        k_bucket=1,
        dedup_fn=_dedup_fn,
        gather_fn=_gather_fn,
    )
    assert isinstance(dg, DedupGather)
    assert isinstance(dg, (Vmap, ChunkedMap, Scan, DedupGather))


def test_batch_planner_dedup_gather_satisfies_axis_strategy():
    """BatchPlanner(dedup_specs=[...]) emits the exact DedupGather class in AxisStrategy."""
    from xtrax.tiling import AxisSpec, BatchPlanner
    from xtrax.tiling.dedup import DedupSpec  # noqa: TID251

    spec = AxisSpec(
        name="n_structures",
        cardinality=4,
        default_batch_size=1,
        heterogeneous=True,
        dedup_eligible=True,
    )
    dedup = DedupSpec(
        axis_name="n_structures",
        unique_indices=np.array([0, 1], dtype=np.int32),
        index_map=np.array([0, 1, 0, 1], dtype=np.int32),
        k=2,
    )
    plan = BatchPlanner(dedup_specs=[dedup]).plan([spec])
    strategy = plan.decisions[0].strategy
    assert isinstance(strategy, DedupGather)
    assert isinstance(strategy, AxisStrategy)
