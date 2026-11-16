"""Tests for DedupGather strategy (backlog #930, task_id 260603_het-batch-dedup).

get_k_bucket sourced from xtrax.tiling.dedup (EPIC #1541 T-PLANNER.GATE):
aminx.tiling.dedup was retired once the planner migrated to xtrax's
joint-budget BatchPlanner; DedupGather itself (aminx.tiling.strategy) is
unaffected -- still constructed directly here, not via DedupSpec.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import io_callback
import pytest

from aminx.tiling.strategy import DedupGather
from xtrax.tiling.dedup import get_k_bucket  # noqa: TID251 -- submodule-only, not re-exported from xtrax.tiling's public __init__


def _make_dedup_gather(
    *,
    unique_indices: np.ndarray,
    index_map: np.ndarray,
    k: int,
    k_bucket: int,
) -> DedupGather:
    return DedupGather(
        unique_indices=unique_indices,
        index_map=index_map,
        k=k,
        k_bucket=k_bucket,
        dedup_fn=lambda xs, idx: jax.tree.map(lambda x: x[idx], xs),
        gather_fn=lambda ys, idx: jax.tree.map(lambda y: y[idx], ys),
    )


def test_dedup_gather_bit_identical() -> None:
    """DedupGather with K=2 uniques in N=4 elements produces bit-identical output to naive map.

    Positions 0,2 share unique_index=0; positions 1,3 share unique_index=1.
    body is deterministic (x * 2.0), so result must equal naive jax.lax.map(body, xs).
    """
    xs = jnp.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    unique_indices = np.array([0, 1], dtype=np.int32)
    index_map = np.array([0, 1, 0, 1], dtype=np.int32)
    k = 2
    k_bucket = get_k_bucket(k)  # expected 2

    strategy = _make_dedup_gather(
        unique_indices=unique_indices,
        index_map=index_map,
        k=k,
        k_bucket=k_bucket,
    )

    def body(x: jax.Array) -> jax.Array:
        return x * 2.0

    @jax.jit
    def run_dedup(xs: jax.Array) -> jax.Array:
        unique_idx = jnp.asarray(strategy.unique_indices, dtype=jnp.int32)
        idx_map = jnp.asarray(strategy.index_map, dtype=jnp.int32)
        xs_unique = strategy.dedup_fn(xs, unique_idx)
        ys_unique = jax.lax.map(body, xs_unique)
        return strategy.gather_fn(ys_unique, idx_map)

    result = run_dedup(xs)
    naive = jax.lax.map(body, xs)
    assert jnp.array_equal(result, naive), (
        f"DedupGather result does not match naive map.\n"
        f"result={result}\nnaive={naive}"
    )


def test_dedup_gather_k_not_n_lax_map() -> None:
    """jax.lax.map is called k_bucket times (not N=6) when K=3 unique structures exist.

    N=6 positions: pairs (0,1), (2,3), (4,5) share unique indices 0,1,2.
    k_bucket = get_k_bucket(3) = 4 (next power-of-two bucket after 3).
    The body should execute exactly k_bucket times, not N times.
    """
    N = 6
    xs = jnp.stack([
        jnp.array([float(i)] * 4, dtype=jnp.float32) for i in range(N)
    ])
    unique_indices = np.array([0, 2, 4], dtype=np.int32)
    index_map = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    k = 3
    k_bucket = get_k_bucket(k)  # expected 4

    # Pad unique_indices to k_bucket with edge-padding
    unique_indices_padded = np.pad(
        unique_indices, (0, k_bucket - k), mode="edge"
    )

    strategy = _make_dedup_gather(
        unique_indices=unique_indices_padded,
        index_map=index_map,
        k=k,
        k_bucket=k_bucket,
    )

    _counter: list[int] = [0]

    def _count_fn(x: jax.Array) -> jax.Array:
        _counter[0] += 1
        return x

    def body(x: jax.Array) -> jax.Array:
        io_callback(_count_fn, x, x, ordered=False)
        return x * 2.0

    @jax.jit
    def run_dedup(xs: jax.Array) -> jax.Array:
        unique_idx = jnp.asarray(strategy.unique_indices, dtype=jnp.int32)
        idx_map = jnp.asarray(strategy.index_map, dtype=jnp.int32)
        xs_unique = strategy.dedup_fn(xs, unique_idx)
        ys_unique = jax.lax.map(body, xs_unique)
        return strategy.gather_fn(ys_unique, idx_map)

    _counter[0] = 0
    _result = run_dedup(xs)
    assert _counter[0] == k_bucket, (
        f"Expected body to execute {k_bucket} times (k_bucket, not N={N}), "
        f"but got {_counter[0]}."
    )


def test_dedup_gather_jit_compiles() -> None:
    """DedupGather dispatch survives jax.jit without TracerError.

    After JIT: xs_unique.shape[0] == k_bucket (inner axis is deduplicated, not N).
    """
    xs = jnp.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    unique_indices = np.array([0, 1], dtype=np.int32)
    index_map = np.array([0, 1, 0, 1], dtype=np.int32)
    k = 2
    k_bucket = get_k_bucket(k)

    strategy = _make_dedup_gather(
        unique_indices=unique_indices,
        index_map=index_map,
        k=k,
        k_bucket=k_bucket,
    )

    def body(x: jax.Array) -> jax.Array:
        return x * 2.0

    @jax.jit
    def run_dedup(xs: jax.Array) -> tuple[jax.Array, jax.Array]:
        unique_idx = jnp.asarray(strategy.unique_indices, dtype=jnp.int32)
        idx_map = jnp.asarray(strategy.index_map, dtype=jnp.int32)
        xs_unique = strategy.dedup_fn(xs, unique_idx)
        ys_unique = jax.lax.map(body, xs_unique)
        result = strategy.gather_fn(ys_unique, idx_map)
        return result, xs_unique

    # Must not raise TracerError
    result, xs_unique = run_dedup(xs)
    assert xs_unique.shape[0] == k_bucket, (
        f"xs_unique.shape[0]={xs_unique.shape[0]} but expected k_bucket={k_bucket}"
    )


def test_dedup_gather_output_shape() -> None:
    """DedupGather output shape[0] equals N (not K).

    Even though computation is performed on K=2 unique elements, the result
    must be gathered back to N=4 positions.
    """
    N = 4
    xs = jnp.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    unique_indices = np.array([0, 1], dtype=np.int32)
    index_map = np.array([0, 1, 0, 1], dtype=np.int32)
    k = 2
    k_bucket = get_k_bucket(k)

    strategy = _make_dedup_gather(
        unique_indices=unique_indices,
        index_map=index_map,
        k=k,
        k_bucket=k_bucket,
    )

    def body(x: jax.Array) -> jax.Array:
        return x * 2.0

    @jax.jit
    def run_dedup(xs: jax.Array) -> jax.Array:
        unique_idx = jnp.asarray(strategy.unique_indices, dtype=jnp.int32)
        idx_map = jnp.asarray(strategy.index_map, dtype=jnp.int32)
        xs_unique = strategy.dedup_fn(xs, unique_idx)
        ys_unique = jax.lax.map(body, xs_unique)
        return strategy.gather_fn(ys_unique, idx_map)

    result = run_dedup(xs)
    assert result.shape[0] == N, (
        f"Output shape[0]={result.shape[0]}, expected N={N}. "
        "DedupGather must gather back to full N positions."
    )
