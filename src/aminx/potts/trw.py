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

import jax
import jax.numpy as jnp
import numpy as np
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
        mask = ((W > 0) | (W.T > 0)).astype(h.dtype)

    # Zero out diagonal
    mask = mask * (1.0 - jnp.eye(n, dtype=h.dtype))

    # Masked J; symmetrize undirected couplings if only one direction is populated
    J_masked = J * mask[..., None, None]  # noqa: N806
    J_T = jnp.transpose(J_masked, (1, 0, 3, 2))  # noqa: N806
    has_fwd = jnp.any(jnp.abs(J_masked) > 0, axis=(-1, -2), keepdims=True)
    has_rev = jnp.any(jnp.abs(J_T) > 0, axis=(-1, -2), keepdims=True)
    J_sym = jnp.where(  # noqa: N806
        has_fwd & has_rev,
        0.5 * (J_masked + J_T),
        jnp.where(has_fwd, J_masked, J_T),
    ) * mask[..., None, None]

    # Row, col, and grand averages of J_ij
    j_row_mean = jnp.mean(J_sym, axis=-1, keepdims=True)  # (n, n, q, 1)
    j_col_mean = jnp.mean(J_sym, axis=-2, keepdims=True)  # (n, n, 1, q)
    j_grand_mean = jnp.mean(J_sym, axis=(-1, -2), keepdims=True)  # (n, n, 1, 1)

    # Zero-sum J on active edges
    J_zero = (J_sym - j_row_mean - j_col_mean + j_grand_mean) * mask[..., None, None]  # noqa: N806

    # Absorb row averages into h (since J_sym is symmetric, j_col_mean[j, i] == j_row_mean[i, j])
    delta_h = jnp.sum(j_row_mean.squeeze(-1) * mask[..., None], axis=1)
    h_absorbed = h + delta_h

    # Zero-sum center h
    h_zero = h_absorbed - jnp.mean(h_absorbed, axis=-1, keepdims=True)

    return h_zero, J_zero


def _validate_edges(edges: Array, n: int, expected_e: int | None = None) -> None:
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (E, 2), got {edges.shape}")
    if expected_e is not None and edges.shape[0] != expected_e:
        raise ValueError(
            f"edges count ({edges.shape[0]}) must match edge_marginals first dimension ({expected_e})"
        )
    if isinstance(edges, jax.core.Tracer):
        return
    edges_np = np.asarray(edges)
    if edges_np.size > 0:
        if np.any(edges_np < 0) or np.any(edges_np >= n):
            raise ValueError(f"edge node indices must be in [0, {n}), got {edges_np.tolist()}")
        if np.any(edges_np[:, 0] == edges_np[:, 1]):
            raise ValueError("self-loops (u == v) are not allowed in tree edges")
        seen: set[tuple[int, int]] = set()
        for u_idx, v_idx in edges_np.tolist():
            canon = (min(int(u_idx), int(v_idx)), max(int(u_idx), int(v_idx)))
            if canon in seen:
                raise ValueError(f"duplicate edge {canon} in edges")
            seen.add(canon)


def tree_marginals_to_potts(
    node_marginals: Float[Array, "n q"],
    edge_marginals: Float[Array, "..."],
    edges: Int[Array, "E 2"] | None = None,
    *,
    zero_sum_gauge: bool = True,
    dense_output: bool = True,
    eps: float = 1e-12,
) -> tuple[Float[Array, "n q"], Array]:
    """Invert tree marginals to Potts parameters (h, J) in closed form.

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
    if node_marginals.ndim != 2:
        raise ValueError(f"node_marginals must be 2D (n, q), got shape {node_marginals.shape}")
    n, q = node_marginals.shape
    pair_eps = min(eps * eps, 1e-30)
    p_i = jnp.clip(node_marginals, eps, 1.0)
    p_i = p_i / jnp.sum(p_i, axis=-1, keepdims=True)
    log_p_i = jnp.log(p_i)

    if edge_marginals.ndim == 3:
        # Edge list format (E, q, q)
        if edges is None:
            raise ValueError("edges of shape (E, 2) must be provided when edge_marginals is 3D (E, q, q)")
        if edge_marginals.shape[1:] != (q, q):
            raise ValueError(
                f"edge_marginals state dimensions {edge_marginals.shape[1:]} must match (q, q)=({q}, {q})"
            )
        _validate_edges(edges, n, expected_e=edge_marginals.shape[0])
        p_uv = jnp.clip(edge_marginals, pair_eps, 1.0)
        p_uv = p_uv / jnp.sum(p_uv, axis=(-1, -2), keepdims=True)

        u = edges[:, 0]
        v = edges[:, 1]

        log_p_u = log_p_i[u]  # (E, q)
        log_p_v = log_p_i[v]  # (E, q)
        j_raw = jnp.log(p_uv) - log_p_u[:, :, None] - log_p_v[:, None, :]

        if zero_sum_gauge:
            j_row_mean = jnp.mean(j_raw, axis=-1, keepdims=True)  # (E, q, 1)
            j_col_mean = jnp.mean(j_raw, axis=-2, keepdims=True)  # (E, 1, q)
            j_grand_mean = jnp.mean(j_raw, axis=(-1, -2), keepdims=True)  # (E, 1, 1)
            j_zero = j_raw - j_row_mean - j_col_mean + j_grand_mean

            h_raw = log_p_i
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
        # Without gauge projection: raw factorization log p(x) = sum_i log p_i(x_i) + sum_{(u,v)} log(p_uv / (p_u p_v))
        h_out = log_p_i
        if dense_output:
            J_dense = jnp.zeros((n, n, q, q), dtype=node_marginals.dtype)  # noqa: N806
            J_dense = J_dense.at[u, v].set(j_raw)
            J_dense = J_dense.at[v, u].set(jnp.swapaxes(j_raw, -1, -2))
            return h_out, J_dense
        return h_out, j_raw

    if edge_marginals.ndim == 4:
        # Dense format (n, n, q, q)
        if edge_marginals.shape != (n, n, q, q):
            raise ValueError(
                f"4D edge_marginals must have shape ({n}, {n}, {q}, {q}), got {edge_marginals.shape}"
            )
        if edges is not None:
            _validate_edges(edges, n, expected_e=None)

        # Symmetrize if only one direction of an undirected edge was populated
        em_t = jnp.transpose(edge_marginals, (1, 0, 3, 2))
        mass_fwd = jnp.sum(edge_marginals, axis=(-1, -2), keepdims=True) > 0.5
        mass_rev = jnp.sum(em_t, axis=(-1, -2), keepdims=True) > 0.5
        em_sym = jnp.where(
            mass_fwd & mass_rev,
            0.5 * (edge_marginals + em_t),
            jnp.where(mass_fwd, edge_marginals, em_t),
        )

        p_uv = jnp.clip(em_sym, pair_eps, 1.0)
        p_uv = p_uv / jnp.maximum(jnp.sum(p_uv, axis=(-1, -2), keepdims=True), pair_eps)

        if edges is not None:
            mask = jnp.zeros((n, n), dtype=node_marginals.dtype)
            mask = mask.at[edges[:, 0], edges[:, 1]].set(1.0)
            mask = mask.at[edges[:, 1], edges[:, 0]].set(1.0)
        else:
            # Mask edges where marginal sum is substantial
            marginal_mass = jnp.sum(em_sym, axis=(-1, -2))
            mask = (marginal_mass > 0.5).astype(node_marginals.dtype)
            mask = mask * (1.0 - jnp.eye(n, dtype=node_marginals.dtype))

        j_raw = (jnp.log(p_uv) - log_p_i[:, None, :, None] - log_p_i[None, :, None, :]) * mask[..., None, None]

        if zero_sum_gauge:
            j_row_mean = jnp.mean(j_raw, axis=-1, keepdims=True)  # (n, n, q, 1)
            j_col_mean = jnp.mean(j_raw, axis=-2, keepdims=True)  # (n, n, 1, q)
            j_grand_mean = jnp.mean(j_raw, axis=(-1, -2), keepdims=True)  # (n, n, 1, 1)
            j_zero = (j_raw - j_row_mean - j_col_mean + j_grand_mean) * mask[..., None, None]

            # Enforce symmetry if undirected
            j_zero = 0.5 * (j_zero + jnp.transpose(j_zero, (1, 0, 3, 2)))

            h_raw = log_p_i
            delta_h = jnp.sum(j_row_mean.squeeze(-1) * mask[..., None], axis=1)
            h_acc = h_raw + delta_h
            h_out = h_acc - jnp.mean(h_acc, axis=-1, keepdims=True)

            if not dense_output and edges is not None:
                return h_out, j_zero[edges[:, 0], edges[:, 1]]
            return h_out, j_zero
        h_out = log_p_i
        if not dense_output and edges is not None:
            return h_out, j_raw[edges[:, 0], edges[:, 1]]
        return h_out, j_raw
    raise ValueError(f"edge_marginals must be 3D (E, q, q) or 4D (n, n, q, q), got shape {edge_marginals.shape}")
