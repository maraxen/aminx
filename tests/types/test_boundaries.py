# tests/types/test_boundaries.py
from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from aminx.types.boundaries import AxisBoundary, Fuse, Sink, Tap


# --- Protocol conformance checks ---

def test_fuse_protocol_duck_typing():
    class MeanFuse:
        def __call__(self, stacked):
            return jnp.mean(stacked, axis=0)

    f = MeanFuse()
    assert isinstance(f, Fuse)


def test_tap_protocol_requires_ordered_attr():
    class GoodTap:
        ordered = True
        def __call__(self, x):
            return x

    class BadTap:  # missing `ordered`
        def __call__(self, x):
            return x

    assert isinstance(GoodTap(), Tap)
    assert not isinstance(BadTap(), Tap)


def test_sink_protocol_requires_ordered_attr():
    class GoodSink:
        ordered = False
        def __call__(self, x) -> None:
            pass

    assert isinstance(GoodSink(), Sink)


# --- AxisBoundary construction ---

def test_axis_boundary_defaults_all_none():
    b = AxisBoundary()
    assert b.fuse is None
    assert b.tap is None
    assert b.sink is None


def test_axis_boundary_accepts_fuse():
    class IdentityFuse:
        def __call__(self, stacked):
            return stacked[0]

    b = AxisBoundary(fuse=IdentityFuse())
    assert b.fuse is not None


def test_axis_boundary_accepts_tap_and_sink():
    class MySink:
        ordered = True
        def __call__(self, x) -> None:
            pass

    class MyTap:
        ordered = False
        def __call__(self, x):
            return x

    b = AxisBoundary(tap=MyTap(), sink=MySink())
    assert b.tap is not None
    assert b.sink is not None


def test_axis_boundary_is_an_xtrax_struct_that_equinox_filters_accept():
    """Since xtrax 0.4.1b5 AxisBoundary is an xtrax.Struct, not an eqx.Module; it is
    still a pytree, so equinox's filters partition and recombine it."""
    import xtrax

    b = AxisBoundary()
    assert isinstance(b, xtrax.Struct)
    arrays, static = eqx.partition(b, eqx.is_array)
    assert eqx.combine(arrays, static) == b


def test_axis_boundary_has_no_traced_leaves():
    """AxisBoundary contains no JAX arrays — all fields are static callables."""
    b = AxisBoundary()
    leaves = jax.tree_util.tree_leaves(b)
    # All fields are static=True so no traced leaves
    assert leaves == []


def test_axis_boundary_jit_static():
    """AxisBoundary can be captured as a static closure in jit."""
    boundary = AxisBoundary()

    @jax.jit
    def fn(x):
        if boundary.fuse is not None:
            return boundary.fuse(x)
        return x

    result = fn(jnp.ones(3))
    assert result.shape == (3,)
