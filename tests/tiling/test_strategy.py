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


def test_dedup_gather_stores_fields():
    unique_indices = np.array([0, 1], dtype=np.int32)
    index_map = np.array([0, 1, 0, 1], dtype=np.int32)
    dg = DedupGather(unique_indices=unique_indices, index_map=index_map, k=2, k_bucket=2)
    assert dg.k == 2
    assert dg.k_bucket == 2
    assert np.array_equal(dg.unique_indices, unique_indices)
    assert np.array_equal(dg.index_map, index_map)


def test_dedup_gather_in_union():
    unique_indices = np.array([0], dtype=np.int32)
    index_map = np.array([0, 0], dtype=np.int32)
    dg = DedupGather(unique_indices=unique_indices, index_map=index_map, k=1, k_bucket=1)
    assert isinstance(dg, DedupGather)
    assert isinstance(dg, (Vmap, ChunkedMap, Scan, DedupGather))
