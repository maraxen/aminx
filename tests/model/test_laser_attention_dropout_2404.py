"""Debt #2404: production LASEr GAT attention dropout honours ``inference``.

``HomoGATv2.__call__`` and ``HeteroGATv2.__call__`` used to accept ``inference``
and discard it, so ``inference=False`` was the identity. The observable that
would have caught that is a difference, at a non-zero drop rate, between
``inference=True`` and ``inference=False``, and between two ``inference=False``
passes whose keys were split from one fixed key. This calls the production
methods in ``aminx.model.laser.layers``. The proofread patches replace those
methods, so a parity vehicle on the patched path cannot see this defect.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from aminx.model.laser.layers import HeteroGATv2, HomoGATv2

_N = 6
_K = 5
_H = 8
_E = 4
_HEADS = 2
_V = 2
_DROP = 0.5


def _inputs(key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    keys = jax.random.split(key, 5)
    scalars = jax.random.normal(keys[0], (_N, _H))
    vectors = jax.random.normal(keys[1], (_N, _V, 3))
    edges = jax.random.normal(keys[2], (_N, _K, _E))
    neighbours = jax.random.randint(keys[3], (_N, _K), 0, _N)
    mask = jnp.ones((_N, _K), dtype=jnp.bool_)
    mask = mask.at[:, -1].set(False)
    return scalars, vectors, edges, neighbours, mask


def _assert_production(method: object) -> None:
    module = getattr(method, "__module__", "")
    assert module == "aminx.model.laser.layers", module


def test_homo_inference_false_drops_attention() -> None:
    """Two train-mode passes under one fixed key differ, and both differ from eval.

    The observable that would have caught the discarded ``inference`` argument is
    ``inference=False`` changing the scalars relative to ``inference=True``, and
    the two ``inference=False`` keys disagreeing. Before the fix every pass
    matched.
    """
    _assert_production(HomoGATv2.__call__)
    module = HomoGATv2(
        _H,
        _E,
        _HEADS,
        _DROP,
        update_edges=False,
        use_mlp_node_update=False,
        atten_head_aggr_layers=0,
        num_vectors=_V,
        atten_dimension_upscale_factor=None,
        key=jax.random.PRNGKey(0),
    )
    scalars, vectors, edges, neighbours, mask = _inputs(jax.random.PRNGKey(1))
    fixed = jax.random.PRNGKey(2)
    key_a, key_b = jax.random.split(fixed)
    eval_out, _, _ = module(
        scalars,
        vectors,
        edges,
        neighbours,
        mask,
        inference=True,
        key=fixed,
    )
    train_a, _, _ = module(
        scalars,
        vectors,
        edges,
        neighbours,
        mask,
        inference=False,
        key=key_a,
    )
    train_b, _, _ = module(
        scalars,
        vectors,
        edges,
        neighbours,
        mask,
        inference=False,
        key=key_b,
    )
    assert not jnp.array_equal(train_a, train_b)
    assert not jnp.array_equal(train_a, eval_out)


def test_hetero_inference_false_drops_attention() -> None:
    """Joint hetero attention drops when ``inference=False``.

    The observable that would have caught the discarded ``inference`` argument is
    the same one as the homogeneous layer: two keys split from a fixed key
    disagree, and either disagrees with ``inference=True``.
    """
    _assert_production(HeteroGATv2.__call__)
    module = HeteroGATv2(
        1,
        _H,
        _E,
        0,
        _HEADS,
        _DROP,
        num_vectors=_V,
        use_mlp_node_update=False,
        use_residual_node_update=False,
        compute_edge_updates=False,
        key=jax.random.PRNGKey(0),
    )
    scalars, vectors, edges, neighbours, mask = _inputs(jax.random.PRNGKey(1))
    fixed = jax.random.PRNGKey(2)
    key_a, key_b = jax.random.split(fixed)
    eval_out, _, _ = module(
        scalars,
        vectors,
        (scalars,),
        (neighbours,),
        (edges,),
        (mask,),
        (True,),
        inference=True,
        key=fixed,
    )
    train_a, _, _ = module(
        scalars,
        vectors,
        (scalars,),
        (neighbours,),
        (edges,),
        (mask,),
        (True,),
        inference=False,
        key=key_a,
    )
    train_b, _, _ = module(
        scalars,
        vectors,
        (scalars,),
        (neighbours,),
        (edges,),
        (mask,),
        (True,),
        inference=False,
        key=key_b,
    )
    assert not jnp.array_equal(train_a, train_b)
    assert not jnp.array_equal(train_a, eval_out)
