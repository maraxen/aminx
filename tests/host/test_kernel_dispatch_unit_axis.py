"""_dispatch_axis must not hand XLA a size-1 vmap axis (aminx #2391).

On XLA GPU (TITAN RTX, jax/jaxlib 0.10.2) a *jitted* size-1 batch axis around a matmul whose
intermediate is square -- the encoder's 512-wide ``dense`` at max_length=512 -- miscompiles to
activations ~0.2 off, with no error. ``scripts/parity/repro_gpu_nested_vmap_mlp.py`` reproduces
it with no aminx code. The runner's default noise/temperature axes are exactly size 1, so the
Vmap branch runs the body on the lone element instead and puts the axis back.

The GPU fault itself cannot be asserted here (CPU is correct), so these tests pin the
*property that avoids it*: results are unchanged, and no size-1 batched ``dot_general`` is emitted.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.host.kernel_dispatch import _dispatch_axis, _is_unit_axis
from aminx.tiling.strategy import Vmap


def _outer_product(a):
    return a @ a.T


def _batched_dot_generals(jaxpr) -> list:
    """dot_general equations (recursing into sub-jaxprs) that carry batch dimensions."""
    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            (_, _), (lhs_batch, _) = eqn.params["dimension_numbers"]
            if lhs_batch:
                found.append(eqn)
        for value in eqn.params.values():
            for item in value if isinstance(value, (list, tuple)) else [value]:
                sub = getattr(item, "jaxpr", item)  # ClosedJaxpr -> Jaxpr; a bare Jaxpr passes through
                if hasattr(sub, "eqns"):
                    found.extend(_batched_dot_generals(sub))
    return found


@pytest.mark.parametrize("n", [1, 2, 5])
def test_matches_plain_vmap(n):
    xs = jnp.arange(n * 12, dtype=jnp.float32).reshape(n, 3, 4)
    np.testing.assert_array_equal(_dispatch_axis(Vmap(), _outer_product, xs), jax.vmap(_outer_product)(xs))


def test_unit_axis_matches_plain_vmap_for_pytrees():
    xs = {"a": jnp.arange(6, dtype=jnp.float32).reshape(1, 6), "b": (jnp.ones((1, 2, 2)), jnp.arange(1))}

    def body(t):
        return {"s": t["a"].sum() + t["b"][0].sum(), "i": t["b"][1] + 1}

    got, want = _dispatch_axis(Vmap(), body, xs), jax.vmap(body)(xs)
    assert jax.tree_util.tree_structure(got) == jax.tree_util.tree_structure(want)
    for g, w in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want), strict=True):
        assert g.shape == w.shape
        assert g.dtype == w.dtype
        np.testing.assert_array_equal(g, w)


def test_unit_axis_runs_body_on_the_unbatched_element():
    seen = []

    def body(x):
        seen.append(x.shape)
        return x * 2

    out = _dispatch_axis(Vmap(), body, jnp.ones((1, 4, 3)))
    assert seen == [(4, 3)]
    assert out.shape == (1, 4, 3)


def test_unit_axis_emits_no_batched_dot_but_larger_axes_still_vmap():
    unit = jax.make_jaxpr(lambda x: _dispatch_axis(Vmap(), _outer_product, x))(jnp.ones((1, 3, 4)))
    many = jax.make_jaxpr(lambda x: _dispatch_axis(Vmap(), _outer_product, x))(jnp.ones((3, 3, 4)))
    assert _batched_dot_generals(unit.jaxpr) == []
    assert _batched_dot_generals(many.jaxpr), "size>1 must still be a real vmap (batched dot_general)"


def test_unit_axis_holds_under_jit_and_nesting():
    """The failing GPU shape was jit around nested size-1 axes; the CPU result must stay identical."""
    xs = jnp.arange(24, dtype=jnp.float32).reshape(1, 1, 3, 8)

    def inner(a):
        return _dispatch_axis(Vmap(), _outer_product, a)

    out = jax.jit(lambda x: _dispatch_axis(Vmap(), inner, x))(xs)
    np.testing.assert_allclose(out, jax.vmap(jax.vmap(_outer_product))(xs), rtol=1e-6)


@pytest.mark.parametrize(
    ("xs", "expected"),
    [
        (jnp.ones((1, 3)), True),
        ({"a": jnp.ones((1,)), "b": jnp.ones((1, 4))}, True),
        (jnp.ones((2, 3)), False),
        ({"a": jnp.ones((1, 3)), "b": jnp.ones((2, 3))}, False),
        (jnp.float32(1.0), False),
        ({}, False),
    ],
    ids=["unit", "unit-pytree", "size2", "mixed", "scalar", "empty"],
)
def test_is_unit_axis(xs, expected):
    assert _is_unit_axis(xs) is expected
