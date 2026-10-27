"""Tests for closed-form algebraic inverse of DifferentiableTRW on trees."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.potts.trw import (
    DifferentiableTRW,
    project_to_zero_sum_gauge,
    tree_marginals_to_potts,
)


def _compute_exact_tree_marginals(
    h: jax.Array,
    J: jax.Array,
    edges: list[tuple[int, int]],
    n: int,
    q: int,
    iters: int = 50,
) -> tuple[jax.Array, dict[tuple[int, int], jax.Array]]:
    """Compute exact node and edge marginals on a tree via belief propagation."""
    W = jnp.zeros((n, n))
    for u, v in edges:
        W = W.at[u, v].set(1.0).at[v, u].set(1.0)

    # Message passing with unit weights (rho = 1 on tree edges)
    m = jnp.ones((n, n, q)) / float(q)
    for _ in range(iters):
        total_in = jnp.prod(m, axis=0)
        m_j_to_i = jnp.transpose(m, (1, 0, 2))
        ratio = total_in[:, None, :] / (m_j_to_i + 1e-12)
        energy = J + h[:, None, :, None]
        exp_energy = jnp.exp(jnp.clip(energy, -30.0, 30.0))
        msg_val = jnp.sum(exp_energy * ratio[..., None], axis=2)
        msg_val = jnp.where(W[..., None] > 0, msg_val, 1.0 / float(q))
        msg_sum = jnp.sum(msg_val, axis=-1, keepdims=True)
        m = jnp.where(msg_sum > 0, msg_val / msg_sum, 1.0 / float(q))

    total_in = jnp.prod(m, axis=0)
    unnorm_node = jnp.exp(jnp.clip(h, -30.0, 30.0)) * total_in
    node_marginals = unnorm_node / jnp.sum(unnorm_node, axis=-1, keepdims=True)

    edge_marginals = {}
    for u, v in edges:
        prod_u = total_in[u] / (m[v, u] + 1e-12)
        prod_v = total_in[v] / (m[u, v] + 1e-12)
        unnorm_edge = (
            jnp.exp(jnp.clip(h[u, :, None] + h[v, None, :] + J[u, v], -30.0, 30.0))
            * prod_u[:, None]
            * prod_v[None, :]
        )
        p_uv = unnorm_edge / jnp.sum(unnorm_edge)
        edge_marginals[(u, v)] = p_uv

    return node_marginals, edge_marginals


def test_chain_tree_round_trip():
    n, q = 6, 21
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    key = jax.random.PRNGKey(101)
    k1, k2 = jax.random.split(key)
    h_raw = jax.random.normal(k1, (n, q)) * 0.4
    h_true = h_raw - jnp.mean(h_raw, axis=-1, keepdims=True)

    J_true = jnp.zeros((n, n, q, q))
    W = jnp.zeros((n, n))
    for u, v in edges:
        W = W.at[u, v].set(1.0).at[v, u].set(1.0)
        k2, subk = jax.random.split(k2)
        j_sub = jax.random.normal(subk, (q, q)) * 0.4
        j_zero = (
            j_sub
            - jnp.mean(j_sub, axis=0, keepdims=True)
            - jnp.mean(j_sub, axis=1, keepdims=True)
            + jnp.mean(j_sub)
        )
        J_true = J_true.at[u, v].set(j_zero).at[v, u].set(j_zero.T)

    p_nodes, p_edges = _compute_exact_tree_marginals(h_true, J_true, edges, n, q)

    # Edge-list interface
    edge_arr = jnp.array(edges, dtype=jnp.int32)
    p_edge_arr = jnp.stack([p_edges[e] for e in edges], axis=0)

    h_rec, J_rec = tree_marginals_to_potts(
        p_nodes, p_edge_arr, edges=edge_arr, zero_sum_gauge=True, dense_output=True
    )

    np.testing.assert_allclose(h_rec, h_true, atol=1e-5)
    for u, v in edges:
        np.testing.assert_allclose(J_rec[u, v], J_true[u, v], atol=1e-5)
        np.testing.assert_allclose(J_rec[v, u], J_true[v, u], atol=1e-5)


def test_star_tree_round_trip():
    n, q = 5, 21
    # Star centered at 0: 0-1, 0-2, 0-3, 0-4
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]

    key = jax.random.PRNGKey(202)
    k1, k2 = jax.random.split(key)
    h_raw = jax.random.normal(k1, (n, q)) * 0.3
    h_true = h_raw - jnp.mean(h_raw, axis=-1, keepdims=True)

    J_true = jnp.zeros((n, n, q, q))
    W = jnp.zeros((n, n))
    for u, v in edges:
        W = W.at[u, v].set(1.0).at[v, u].set(1.0)
        k2, subk = jax.random.split(k2)
        j_sub = jax.random.normal(subk, (q, q)) * 0.3
        j_zero = (
            j_sub
            - jnp.mean(j_sub, axis=0, keepdims=True)
            - jnp.mean(j_sub, axis=1, keepdims=True)
            + jnp.mean(j_sub)
        )
        J_true = J_true.at[u, v].set(j_zero).at[v, u].set(j_zero.T)

    p_nodes, p_edges = _compute_exact_tree_marginals(h_true, J_true, edges, n, q)

    # Dense interface
    p_dense = jnp.zeros((n, n, q, q))
    for u, v in edges:
        p_dense = p_dense.at[u, v].set(p_edges[(u, v)])
        p_dense = p_dense.at[v, u].set(p_edges[(u, v)].T)

    h_rec, J_rec = tree_marginals_to_potts(
        p_nodes, p_dense, zero_sum_gauge=True, dense_output=True
    )

    np.testing.assert_allclose(h_rec, h_true, atol=1e-5)
    for u, v in edges:
        np.testing.assert_allclose(J_rec[u, v], J_true[u, v], atol=1e-5)
        np.testing.assert_allclose(J_rec[v, u], J_true[v, u], atol=1e-5)


def test_random_tree_round_trip():
    n, q = 8, 21
    # Spanning tree on 8 nodes
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (5, 6), (5, 7)]

    key = jax.random.PRNGKey(303)
    k1, k2 = jax.random.split(key)
    h_raw = jax.random.normal(k1, (n, q)) * 0.3
    h_true = h_raw - jnp.mean(h_raw, axis=-1, keepdims=True)

    J_true = jnp.zeros((n, n, q, q))
    for u, v in edges:
        k2, subk = jax.random.split(k2)
        j_sub = jax.random.normal(subk, (q, q)) * 0.3
        j_zero = (
            j_sub
            - jnp.mean(j_sub, axis=0, keepdims=True)
            - jnp.mean(j_sub, axis=1, keepdims=True)
            + jnp.mean(j_sub)
        )
        J_true = J_true.at[u, v].set(j_zero).at[v, u].set(j_zero.T)

    p_nodes, p_edges = _compute_exact_tree_marginals(h_true, J_true, edges, n, q)
    edge_arr = jnp.array(edges, dtype=jnp.int32)
    p_edge_arr = jnp.stack([p_edges[e] for e in edges], axis=0)

    # Edge list output format
    h_rec, j_edges_rec = tree_marginals_to_potts(
        p_nodes, p_edge_arr, edges=edge_arr, zero_sum_gauge=True, dense_output=False
    )

    np.testing.assert_allclose(h_rec, h_true, atol=1e-5)
    for idx, (u, v) in enumerate(edges):
        np.testing.assert_allclose(j_edges_rec[idx], J_true[u, v], atol=1e-5)


def test_zero_sum_gauge_properties():
    n, q = 4, 10
    edges = [(0, 1), (1, 2), (2, 3)]
    edge_arr = jnp.array(edges, dtype=jnp.int32)

    # Arbitrary non-zero-sum marginals
    key = jax.random.PRNGKey(404)
    k1, k2 = jax.random.split(key)
    p_nodes = jax.nn.softmax(jax.random.normal(k1, (n, q)), axis=-1)
    p_edges = jax.nn.softmax(jax.random.normal(k2, (len(edges), q, q)).reshape(len(edges), -1), axis=-1).reshape(len(edges), q, q)

    h_rec, J_rec = tree_marginals_to_potts(
        p_nodes, p_edges, edges=edge_arr, zero_sum_gauge=True, dense_output=True
    )

    # Check zero-sum conditions
    np.testing.assert_allclose(jnp.sum(h_rec, axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(jnp.sum(J_rec, axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(jnp.sum(J_rec, axis=-2), 0.0, atol=1e-5)


def test_invalid_inputs_raise():
    p_nodes = jnp.ones((5, 10)) / 10.0
    p_edges = jnp.ones((4, 10, 10)) / 100.0

    # 3D marginals without edges
    with pytest.raises(ValueError, match="edges of shape"):
        tree_marginals_to_potts(p_nodes, p_edges, edges=None)

    # Invalid dimension
    with pytest.raises(ValueError, match="edge_marginals must be 3D"):
        tree_marginals_to_potts(p_nodes, jnp.ones((5, 10)))
