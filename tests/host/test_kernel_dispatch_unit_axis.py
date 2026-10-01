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

from aminx.host.kernel_dispatch import _dispatch_axis
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


def test_unit_axis_runs_body_on_a_concrete_element_not_a_batch_tracer():
    """Under jax.vmap the body sees a BatchTracer (and the same shape), so shape alone cannot tell the guard from vmap."""
    seen = []

    def body(x):
        seen.append(isinstance(x, jax.core.Tracer))
        return x * 2

    out = _dispatch_axis(Vmap(), body, jnp.ones((1, 4, 3)))
    assert seen == [False], "the guard must call the body on the concrete element"
    assert out.shape == (1, 4, 3)
    seen.clear()
    _dispatch_axis(Vmap(), body, jnp.ones((2, 4, 3)))
    assert seen == [True], "control: a real vmap hands the body a tracer"


def _dot_lhs_shapes(jaxpr) -> list:
    """lhs operand shapes of every dot_general, recursing into sub-jaxprs."""
    shapes = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            shapes.append(tuple(eqn.invars[0].aval.shape))
        for value in eqn.params.values():
            for item in value if isinstance(value, (list, tuple)) else [value]:
                sub = getattr(item, "jaxpr", item)
                if hasattr(sub, "eqns"):
                    shapes.extend(_dot_lhs_shapes(sub))
    return shapes


def test_closed_over_weights_matmul_has_no_leading_unit_axis():
    """The encoder's `dense` is x @ W with W closed over: a size-1 vmap gives lhs (1, L, C) with NO batch dimension,
    which a 'batched dot_general' check cannot see. Compare against the control (plain vmap) in the same test."""
    w = jnp.ones((3, 3))

    def body(x):
        return x @ w

    guarded = jax.make_jaxpr(lambda x: _dispatch_axis(Vmap(), body, x))(jnp.ones((1, 4, 3)))
    plain = jax.make_jaxpr(jax.vmap(body))(jnp.ones((1, 4, 3)))
    assert _dot_lhs_shapes(plain.jaxpr) == [(1, 4, 3)], "control: plain vmap leaves a leading size-1 operand axis"
    assert _dot_lhs_shapes(guarded.jaxpr) == [(4, 3)], "guard must hand XLA the unbatched operand"


def test_xtrax_vmap_strategy_is_guarded_too():
    """Dispatch is by class NAME, and the production planner emits xtrax-native strategies."""
    from xtrax.tiling import Vmap as XtraxVmap

    out = jax.make_jaxpr(lambda x: _dispatch_axis(XtraxVmap(), lambda v: v @ jnp.ones((3, 3)), x))(jnp.ones((1, 4, 3)))
    assert _dot_lhs_shapes(out.jaxpr) == [(4, 3)]


def test_non_array_output_leaves_still_work_on_a_unit_axis():
    """jax.vmap broadcasts a Python scalar leaf; the guard must not raise on it only when the axis has size 1."""
    got = _dispatch_axis(Vmap(), lambda x: {"v": x.sum(), "flag": 1.0}, jnp.ones((1, 3)))
    want = jax.vmap(lambda x: {"v": x.sum(), "flag": 1.0})(jnp.ones((1, 3)))
    assert got["flag"].shape == want["flag"].shape == (1,)
    np.testing.assert_array_equal(got["v"], want["v"])


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
