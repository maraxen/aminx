"""Graph utilities for the Aminx model.

aminx.utils.graph
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int

if TYPE_CHECKING:
  from aminx.types.arrays import (
    NeighborIndices,
    ResidueIndex,
  )

NeighborOffsets = Int[Array, "num_residues num_neighbors"]


def compute_neighbor_offsets(
  residue_indices: ResidueIndex,
  neighbor_indices: NeighborIndices,
  *,
  cyclic: bool | jax.Array = False,
  chain_index: jax.Array | None = None,
  mask: jax.Array | None = None,
) -> jax.Array:
  """Compute offsets between residues for neighbor indices.

  Parameters
  ----------
  residue_indices : ResidueIndex
    Residue indices for each residue ``(L,)``.
  neighbor_indices : NeighborIndices
    Indices of neighboring residues ``(L, K)``.
  cyclic : bool | jax.Array
    When True, wrap intra-chain offsets across each chain's valid cycle
    span ``L_c`` via shortest signed modular distance. When False (default),
    return linear offsets ``r_i - r_j`` bit-for-bit.
  chain_index : jax.Array | None
    Optional chain ID per residue ``(L,)``. Defaults to a single chain when
    ``None``.
  mask : jax.Array | None
    Optional validity mask per residue ``(L,)``. Positions with ``mask <= 0``
    are excluded from per-chain cycle span calculation and keep linear offsets.

  Returns
  -------
  jax.Array
    A 2D array of offsets ``(L, K)`` where each row corresponds to a residue
    and each column corresponds to a neighbor.
  """
  linear_offset = residue_indices[:, None] - residue_indices[None, :]
  if isinstance(cyclic, bool) and not cyclic:
    return jnp.take_along_axis(linear_offset, neighbor_indices, axis=1)

  r = jnp.asarray(residue_indices, dtype=jnp.int32)
  valid = (
    jnp.ones_like(r, dtype=jnp.bool_)
    if mask is None
    else (jnp.asarray(mask) > 0)
  )
  chains = (
    jnp.zeros_like(r, dtype=jnp.int32)
    if chain_index is None
    else jnp.asarray(chain_index, dtype=jnp.int32)
  )
  same_valid_chain = (
    (chains[:, None] == chains[None, :]) & valid[:, None] & valid[None, :]
  )

  sentinel = jnp.asarray(1_000_000_000, dtype=jnp.int32)
  r_for_min = jnp.where(same_valid_chain, r[None, :], sentinel)
  r_for_max = jnp.where(same_valid_chain, r[None, :], -sentinel)
  r_min = jnp.min(r_for_min, axis=1, keepdims=True)
  r_max = jnp.max(r_for_max, axis=1, keepdims=True)
  span = jnp.where(valid[:, None], r_max - r_min + 1, 1)
  cycle_len = jnp.maximum(span, 1)

  half = cycle_len // 2
  wrapped = ((linear_offset + half) % cycle_len) - half
  is_even_antipode = (
    ((cycle_len % 2) == 0) & (wrapped == -half) & (linear_offset > 0)
  )
  wrapped = jnp.where(is_even_antipode, half, wrapped)
  cyclic_offset = jnp.where(same_valid_chain, wrapped, linear_offset)
  cyc = jnp.asarray(cyclic, dtype=jnp.bool_)
  cyc_pair = (cyc[:, None] & cyc[None, :]) if cyc.ndim == 1 else cyc
  chosen_offset = jnp.where(
    cyc_pair,
    cyclic_offset,
    linear_offset,
  )
  return jnp.take_along_axis(chosen_offset, neighbor_indices, axis=1)

