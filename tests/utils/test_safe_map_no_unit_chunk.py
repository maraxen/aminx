"""safe_map must never compile a vmapped chunk of size 1 (aminx #2391).

On one XLA GPU stack (TITAN RTX, jax 0.10.2, CUDA 12.9) a jitted size-1 batch around a matmul whose
intermediate is square returned wrong activations silently (scripts/parity/repro_gpu_nested_vmap_mlp.py;
the GPU fault itself is not reproducible on CPU CI). A chunk of 1 arises three ways -- a size-1 axis,
``batch_size == 1`` and a remainder of 1 -- and each is pinned here by what XLA is handed, with a
plain-``lax.map``/``vmap`` control that DOES show the leading size-1 operand, so the check can fail.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.utils.safe_map import safe_map

W = jnp.arange(9.0).reshape(3, 3) / 10.0  # closed over, like the encoder's weights


def _body(x):
    return {"y": x @ W, "s": x.sum()}


def _dot_lhs_shapes(jaxpr) -> list:
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


def _leading_unit(shapes) -> list:
    return [s for s in shapes if len(s) == 3 and s[0] == 1]


@pytest.mark.parametrize(
    ("n", "batch_size"),
    list(itertools.product([1, 2, 3, 4, 5, 6, 7, 8, 9], [None, 0, 1, 2, 3, 4, 100])),
)
def test_matches_plain_vmap(n, batch_size):
    xs = jnp.arange(n * 12, dtype=jnp.float32).reshape(n, 4, 3)
    got, want = safe_map(_body, xs, batch_size), jax.vmap(_body)(xs)
    assert jax.tree_util.tree_structure(got) == jax.tree_util.tree_structure(want)
    for g, w in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want), strict=True):
        assert g.shape == w.shape
        np.testing.assert_allclose(g, w, rtol=1e-6)


@pytest.mark.parametrize(
    ("n", "batch_size", "why"),
    [
        (1, None, "size-1 axis"),
        (1, 4, "size-1 axis, batch larger"),
        (3, 1, "batch_size == 1"),
        (5, 2, "remainder of 1"),
        (7, 3, "remainder of 1, batch 3"),
    ],
)
def test_no_vmapped_chunk_of_one(n, batch_size, why):
    jaxpr = jax.make_jaxpr(lambda x: safe_map(_body, x, batch_size))(jnp.ones((n, 4, 3)))
    assert _leading_unit(_dot_lhs_shapes(jaxpr.jaxpr)) == [], why


def test_control_plain_constructs_do_show_a_chunk_of_one():
    """The check can fail: the constructs safe_map avoids really do hand XLA a leading size-1 operand."""
    vmap1 = jax.make_jaxpr(jax.vmap(_body))(jnp.ones((1, 4, 3)))
    map_bs1 = jax.make_jaxpr(lambda x: jax.lax.map(_body, x, batch_size=1))(jnp.ones((3, 4, 3)))
    map_rem1 = jax.make_jaxpr(lambda x: jax.lax.map(_body, x, batch_size=2))(jnp.ones((5, 4, 3)))
    for jaxpr in (vmap1, map_bs1, map_rem1):
        assert _leading_unit(_dot_lhs_shapes(jaxpr.jaxpr))


@pytest.mark.parametrize(("n", "batch_size"), [(4, 2), (6, 3), (5, 3), (8, 3), (3, 100), (2, None)])
def test_larger_chunks_are_still_batched(n, batch_size):
    """Only chunks of 1 are avoided; real batching (the memory/speed reason to vmap) is kept."""
    jaxpr = jax.make_jaxpr(lambda x: safe_map(_body, x, batch_size))(jnp.ones((n, 4, 3)))
    shapes = _dot_lhs_shapes(jaxpr.jaxpr)
    assert any(len(s) == 3 and s[0] >= 2 for s in shapes), shapes


def test_jit_and_pytree_inputs():
    xs = {"a": jnp.arange(30.0).reshape(5, 6), "b": jnp.arange(5.0)}

    def body(t):
        return t["a"].sum() + t["b"]

    np.testing.assert_allclose(jax.jit(lambda t: safe_map(body, t, 2))(xs), jax.vmap(body)(xs))


def test_python_scalar_output_leaf_on_a_unit_axis():
    got = safe_map(lambda x: {"v": x.sum(), "flag": 1.0}, jnp.ones((1, 3)))
    assert got["flag"].shape == (1,)


def test_empty_pytree_still_rejected():
    with pytest.raises(ValueError, match="empty PyTree"):
        safe_map(lambda x: x, {}, 2)
