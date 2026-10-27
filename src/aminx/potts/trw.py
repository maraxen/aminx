"""Differentiable TRW inference and algebraic inverse on tree topologies.

This module provides:
- ``DifferentiableTRW``: TRW belief propagation on Potts MRFs (re-exported from ``_trw``).
- ``tree_marginals_to_potts``: Closed-form algebraic inverse mapping single-site and
  pairwise tree marginals back to Potts fields h and couplings J in zero-sum gauge.

References
----------
.. [TRW] Wainwright, M.J., Jaakkola, T.S., and Willsky, A.S. (2003).
   "Tree-reweighted belief propagation algorithms and approximate ML estimation
   by pseudo-moment matching." AISTATS 2003.
   https://doi.org/10.7551/mitpress/3968.003.0085
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from aminx.potts._trw import (
    DifferentiableTRW,
    lcp_regularizer,
    rho_from_l_pinv,
    rho_uniform,
    spectral_rho_dense,
)

__all__ = [
    "DifferentiableTRW",
    "lcp_regularizer",
    "project_to_zero_sum_gauge",
    "rho_from_l_pinv",
    "rho_uniform",
    "spectral_rho_dense",
    "tree_marginals_to_potts",
]


def project_to_zero_sum_gauge(
    h: Float[Array, "n q"],
    J: Float[Array, "n n q q"],  # noqa: N803
    W: Float[Array, "n n"] | None = None,  # noqa: N803
) -> tuple[Float[Array, "n q"], Float[Array, "n n q q"]]:
    """Project Potts fields h and dense couplings J to zero-sum gauge.

    Gauge conditions:
        sum_a h_i(a) = 0 for all i
        sum_a J_ij(a, b) = 0 for all i, j, b
        sum_b J_ij(a, b) = 0 for all i, j, a

    The row and column means of J are absorbed into h to preserve state energies.

    Parameters
    ----------
    h : Float[Array, "n q"]
        Single-site fields of shape (n, q).
    J : Float[Array, "n n q q"]
        Dense pairwise couplings of shape (n, n, q, q).
    W : Float[Array, "n n"] | None, optional
        Adjacency or weight matrix masking active edges. Defaults to all off-diagonal pairs.

    Returns
    -------
    tuple[Float[Array, "n q"], Float[Array, "n n q q"]]
        Gauge-projected (h, J) satisfying zero-sum constraints.
    """
    n, q = h.shape
    if W is None:
        mask = jnp.ones((n, n), dtype=h.dtype)
    else:
        mask = (W > 0).astype(h.dtype)

    # Zero out diagonal
    mask = mask * (1.0 - jnp.eye(n, dtype=h.dtype))

    # Masked J
    J_masked = J * mask[..., None, None]  # noqa: N806

    # Row, col, and grand averages of J_ij
    j_row_mean = jnp.mean(J_masked, axis=-1, keepdims=True)  # (n, n, q, 1)
    j_col_mean = jnp.mean(J_masked, axis=-2, keepdims=True)  # (n, n, 1, q)
    j_grand_mean = jnp.mean(J_masked, axis=(-1, -2), keepdims=True)  # (n, n, 1, 1)

    # Zero-sum J on active edges
    J_zero = (J_masked - j_row_mean - j_col_mean + j_grand_mean) * mask[..., None, None]  # noqa: N806

    # Absorb row/col averages into h
    # For undirected graphs, J_ij(a, b) and J_ji(b, a) both represent the edge.
    # If directed edge list / symmetric J:
    delta_h = jnp.sum(j_row_mean.squeeze(-1) * mask[..., None], axis=1)
    h_absorbed = h + delta_h

    # Zero-sum center h
    h_zero = h_absorbed - jnp.mean(h_absorbed, axis=-1, keepdims=True)

    return h_zero, J_zero


def tree_marginals_to_potts(
    node_marginals: Float[Array, "n q"],
    edge_marginals: Float[Array, "..."],
    edges: Int[Array, "E 2"] | None = None,
    *,
    zero_sum_gauge: bool = True,
    dense_output: bool = True,
    eps: float = 1e-12,
) -> tuple[Float[Array, "n q"], Array]:
    r"""Invert tree marginals to Potts parameters (h, J) in closed form.

    For a tree-structured MRF, the joint probability factorizes as:
    .. math::
        p(x) = \prod_{i \in V} p_i(x_i)^{1 - d_i} \prod_{(i, j) \in E} p_{ij}(x_i, x_j)
             = \prod_{i \in V} p_i(x_i) \prod_{(i, j) \in E} \frac{p_{ij}(x_i, x_j)}{p_i(x_i) p_j(x_j)}

    The corresponding raw potentials are:
    .. math::
        J_{ij}^{\mathrm{raw}}(a, b) = \ln \frac{p_{ij}(a, b)}{p_i(a) p_j(b)}
        h_i^{\mathrm{raw}}(a) = \ln p_i(a)

    When projected to zero-sum gauge:
    .. math::
        J_{ij}(a, b) = J_{ij}^{\mathrm{raw}}(a, b) - \bar{J}_{ij}(a, \cdot) - \bar{J}_{ij}(\cdot, b) + \bar{J}_{ij}(\cdot, \cdot)
        h_i(a) = \left( h_i^{\mathrm{raw}}(a) + \sum_{j \in N(i)} \bar{J}_{ij}(a, \cdot) \right) - \mathrm{mean}_a(\dots)

    Parameters
    ----------
    node_marginals : Float[Array, "n q"]
        Node marginals p_i(a) of shape (n, q).
    edge_marginals : Float[Array, "..."]
        Pairwise marginals p_ij(a, b), either dense (n, n, q, q) or edge-list (E, q, q).
    edges : Int[Array, "E 2"] | None, optional
        Edge indices (E, 2) when edge_marginals is (E, q, q), or optional edge selection for dense.
    zero_sum_gauge : bool, default=True
        If True (default), project (h, J) into zero-sum gauge.
    dense_output : bool, default=True
        If True (default), return J with shape (n, n, q, q). If False and edges is provided, return (E, q, q).
    eps : float, default=1e-12
        Small epsilon for clipping and numerical stability in log.

    Returns
    -------
    tuple[Float[Array, "n q"], Array]
        Tuple of (h, J) single-site and pairwise Potts potentials.
    """
    n, q = node_marginals.shape
    p_i = jnp.clip(node_marginals, eps, 1.0)
    p_i = p_i / jnp.sum(p_i, axis=-1, keepdims=True)

    if edge_marginals.ndim == 3:
        # Edge list format (E, q, q)
        if edges is None:
            raise ValueError("edges of shape (E, 2) must be provided when edge_marginals is 3D (E, q, q)")
        e_edges = edges.shape[0]
        p_uv = jnp.clip(edge_marginals, eps, 1.0)
        p_uv = p_uv / jnp.sum(p_uv, axis=(-1, -2), keepdims=True)

        u = edges[:, 0]
        v = edges[:, 1]

        p_u = p_i[u]  # (E, q)
        p_v = p_i[v]  # (E, q)
        denom = p_u[:, :, None] * p_v[:, None, :] + eps
        j_raw = jnp.log(jnp.clip(p_uv / denom, eps, None))

        if zero_sum_gauge:
            j_row_mean = jnp.mean(j_raw, axis=-1, keepdims=True)  # (E, q, 1)
            j_col_mean = jnp.mean(j_raw, axis=-2, keepdims=True)  # (E, 1, q)
            j_grand_mean = jnp.mean(j_raw, axis=(-1, -2), keepdims=True)  # (E, 1, 1)
            j_zero = j_raw - j_row_mean - j_col_mean + j_grand_mean

            h_raw = jnp.log(p_i)
            # Accumulate absorbed row means to u and col means to v
            h_acc = h_raw.at[u].add(j_row_mean.squeeze(-1))
            h_acc = h_acc.at[v].add(j_col_mean.squeeze(-2))
            h_out = h_acc - jnp.mean(h_acc, axis=-1, keepdims=True)

            if dense_output:
                J_dense = jnp.zeros((n, n, q, q), dtype=node_marginals.dtype)  # noqa: N806
                J_dense = J_dense.at[u, v].set(j_zero)
                J_dense = J_dense.at[v, u].set(jnp.swapaxes(j_zero, -1, -2))
                return h_out, J_dense
            return h_out, j_zero
        # Without gauge projection: raw factorization
        d = jnp.zeros((n,), dtype=node_marginals.dtype)
        d = d.at[u].add(1.0)
        d = d.at[v].add(1.0)
        h_out = (1.0 - d[:, None]) * jnp.log(p_i)
        if dense_output:
            J_dense = jnp.zeros((n, n, q, q), dtype=node_marginals.dtype)  # noqa: N806
            J_dense = J_dense.at[u, v].set(j_raw)
            J_dense = J_dense.at[v, u].set(jnp.swapaxes(j_raw, -1, -2))
            return h_out, J_dense
        return h_out, j_raw

    if edge_marginals.ndim == 4:
        # Dense format (n, n, q, q)
        p_uv = jnp.clip(edge_marginals, eps, 1.0)
        p_uv = p_uv / jnp.maximum(jnp.sum(p_uv, axis=(-1, -2), keepdims=True), eps)

        if edges is not None:
            mask = jnp.zeros((n, n), dtype=node_marginals.dtype)
            mask = mask.at[edges[:, 0], edges[:, 1]].set(1.0)
            mask = mask.at[edges[:, 1], edges[:, 0]].set(1.0)
        else:
            # Mask edges where marginal sum is substantial
            marginal_mass = jnp.sum(edge_marginals, axis=(-1, -2))
            mask = (marginal_mass > 0.5).astype(node_marginals.dtype)
            mask = mask * (1.0 - jnp.eye(n, dtype=node_marginals.dtype))

        denom = p_i[:, None, :, None] * p_i[None, :, None, :] + eps
        j_raw = jnp.log(jnp.clip(p_uv / denom, eps, None)) * mask[..., None, None]

        if zero_sum_gauge:
            j_row_mean = jnp.mean(j_raw, axis=-1, keepdims=True)  # (n, n, q, 1)
            j_col_mean = jnp.mean(j_raw, axis=-2, keepdims=True)  # (n, n, 1, q)
            j_grand_mean = jnp.mean(j_raw, axis=(-1, -2), keepdims=True)  # (n, n, 1, 1)
            j_zero = (j_raw - j_row_mean - j_col_mean + j_grand_mean) * mask[..., None, None]

            # Enforce symmetry if undirected
            j_zero = 0.5 * (j_zero + jnp.transpose(j_zero, (1, 0, 3, 2)))

            h_raw = jnp.log(p_i)
            delta_h = jnp.sum(j_row_mean.squeeze(-1) * mask[..., None], axis=1)
            h_acc = h_raw + delta_h
            h_out = h_acc - jnp.mean(h_acc, axis=-1, keepdims=True)

            if not dense_output and edges is not None:
                return h_out, j_zero[edges[:, 0], edges[:, 1]]
            return h_out, j_zero
        d = jnp.sum(mask, axis=1)
        h_out = (1.0 - d[:, None]) * jnp.log(p_i)
        if not dense_output and edges is not None:
            return h_out, j_raw[edges[:, 0], edges[:, 1]]
        return h_out, j_raw
    raise ValueError(f"edge_marginals must be 3D (E, q, q) or 4D (n, n, q, q), got shape {edge_marginals.shape}")
