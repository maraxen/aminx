# tests/tiling/test_iterator.py
"""Tests for MapIterator and ScanIterator protocols + concrete implementations.

Covers:
- Protocol conformance via isinstance(concrete, RuntimeCheckableProtocol)
- Numerical equivalence: Vmap, ChunkedMap, JaxScan
- Type distinctness (oracle REC-1): switching iterator type triggers re-JIT
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from xtrax.tiling import ChunkedMapIterator

from aminx.tiling.iterator import (
    JaxScanIterator,
    MapIterator,
    ScanIterator,
    VmapIterator,
)


# ============================================================================
# Protocol Conformance Tests
# ============================================================================


def test_vmap_iterator_is_map_iterator():
    """VmapIterator satisfies MapIterator protocol."""
    it = VmapIterator()
    assert isinstance(it, MapIterator)


def test_chunked_map_iterator_is_map_iterator():
    """ChunkedMapIterator satisfies MapIterator protocol."""
    it = ChunkedMapIterator(tile=2)
    assert isinstance(it, MapIterator)


def test_jax_scan_iterator_is_scan_iterator():
    """JaxScanIterator satisfies ScanIterator protocol."""
    it = JaxScanIterator()
    assert isinstance(it, ScanIterator)


# ============================================================================
# Numerical Equivalence Tests — MapIterator
# ============================================================================


def test_vmap_iterator_numerical_equivalence():
    """VmapIterator correctly applies a function across elements."""
    it = VmapIterator()
    fn = lambda x: x * 2
    xs = jnp.arange(4)
    result = it(fn, xs)
    expected = jnp.array([0, 2, 4, 6])
    assert jnp.allclose(result, expected)


def test_vmap_iterator_with_in_axes():
    """VmapIterator respects in_axes argument."""
    it = VmapIterator()
    fn = lambda x: x * 2
    xs = jnp.arange(4)
    result = it(fn, xs, in_axes=0)
    expected = jnp.arange(4) * 2
    assert jnp.allclose(result, expected)


def test_chunked_map_iterator_numerical_equivalence():
    """ChunkedMapIterator correctly applies a function across elements."""
    it = ChunkedMapIterator(tile=2)
    fn = lambda x: x * 2
    xs = jnp.arange(4)
    result = it(fn, xs)
    expected = jnp.array([0, 2, 4, 6])
    assert jnp.allclose(result, expected)


def test_chunked_map_iterator_respects_tile():
    """ChunkedMapIterator uses tile parameter for chunking."""
    it = ChunkedMapIterator(tile=2)
    assert it.tile == 2
    # Verify it can be called with different tile sizes
    it_large = ChunkedMapIterator(tile=10)
    assert it_large.tile == 10


def test_chunked_map_iterator_with_in_axes():
    """ChunkedMapIterator respects in_axes argument."""
    it = ChunkedMapIterator(tile=2)
    fn = lambda x: x * 2
    xs = jnp.arange(4)
    result = it(fn, xs, in_axes=0)
    expected = jnp.arange(4) * 2
    assert jnp.allclose(result, expected)


# ============================================================================
# Numerical Equivalence Tests — ScanIterator
# ============================================================================


def test_jax_scan_iterator_numerical_equivalence():
    """JaxScanIterator correctly carries state through iterations."""
    it = JaxScanIterator()
    fn = lambda c, x: (c + x, c + x)  # Carry: accumulated sum, Output: running total
    init = jnp.array(0)
    xs = jnp.arange(4)
    final_carry, ys = it(fn, init, xs)
    assert final_carry == jnp.array(6)
    assert jnp.allclose(ys, jnp.array([0, 1, 3, 6]))


def test_jax_scan_iterator_with_complex_carry():
    """JaxScanIterator handles structured carry correctly."""
    it = JaxScanIterator()
    def fn(carry, x):
        s, p = carry
        return (s + x, p * x), (s + x, p * x)

    init = (jnp.array(0), jnp.array(1))
    xs = jnp.array([1, 2, 3])
    final_carry, ys = it(fn, init, xs)

    expected_carry_sum = jnp.array(6)
    expected_carry_prod = jnp.array(6)
    assert final_carry[0] == expected_carry_sum
    assert final_carry[1] == expected_carry_prod


# ============================================================================
# Type Distinctness Tests (oracle REC-1)
# ============================================================================


def test_vmap_vs_chunked_map_different_types():
    """Different iterator types are distinct (per oracle REC-1)."""
    it_vmap = VmapIterator()
    it_chunked = ChunkedMapIterator(tile=4)

    assert type(it_vmap) != type(it_chunked)
    assert isinstance(it_vmap, MapIterator)
    assert isinstance(it_chunked, MapIterator)


def test_vmap_vs_jax_scan_different_types():
    """MapIterator vs ScanIterator are fundamentally different types."""
    it_map = VmapIterator()
    it_scan = JaxScanIterator()

    assert type(it_map) != type(it_scan)
    assert isinstance(it_map, MapIterator)
    assert isinstance(it_scan, ScanIterator)


def test_treedef_invariant_re_jit_on_strategy_switch():
    """REC-1: switching iterator strategy must change PyTree structure → re-JIT."""
    vmap_struct = jax.tree_util.tree_structure(VmapIterator())
    chunked_struct = jax.tree_util.tree_structure(ChunkedMapIterator(tile=4))
    jaxscan_struct = jax.tree_util.tree_structure(JaxScanIterator())
    assert vmap_struct != chunked_struct
    assert vmap_struct != jaxscan_struct
    assert chunked_struct != jaxscan_struct


# ============================================================================
# Iterator Equality Tests
# ============================================================================


def test_iterator_equality_by_value():
    """Iterators with same parameters are equal."""
    it1 = ChunkedMapIterator(tile=4)
    it2 = ChunkedMapIterator(tile=4)
    assert it1 == it2


def test_iterator_inequality_different_params():
    """Iterators with different parameters are not equal."""
    it1 = ChunkedMapIterator(tile=2)
    it2 = ChunkedMapIterator(tile=4)
    assert it1 != it2


def test_vmap_iterator_equality():
    """VmapIterator instances are always equal (no parameters)."""
    it1 = VmapIterator()
    it2 = VmapIterator()
    assert it1 == it2


def test_jax_scan_iterator_equality():
    """JaxScanIterator instances are always equal (no parameters)."""
    it1 = JaxScanIterator()
    it2 = JaxScanIterator()
    assert it1 == it2
