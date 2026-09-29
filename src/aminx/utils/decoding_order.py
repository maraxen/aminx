"""Utilities for decoding order generation.

aminx.utils.decoding_order
"""

from __future__ import annotations

from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from aminx.types.arrays import (
  DecodingOrder,
)

from .autoregression import get_decoding_step_map

DecodingOrderInputs = tuple[PRNGKeyArray, int, jnp.ndarray | None]  # Added tie_group_map
DecodingOrderOutputs = tuple[DecodingOrder, PRNGKeyArray]


class DecodingOrderFn(Protocol):
  """Draw a decoding order.

  ``chain_mask`` is optional. Callers that omit it, and the all-designed
  default, must keep the historical permutation for the same key.
  """

  def __call__(
    self,
    prng_key: PRNGKeyArray,
    num_residues: int,
    tie_group_map: jnp.ndarray | None = None,
    num_groups: int | None = None,
    *,
    chain_mask: jnp.ndarray | None = None,
  ) -> DecodingOrderOutputs:
    """Return ``(decoding_order, next_key)``."""
    ...


def design_chain_mask(
  residue_mask: jnp.ndarray,
  fixed_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
  """Build a design mask with 1 = designed (ProteinMPNN ``chain_mask``).

  A position is designed when it is a valid residue and not fixed.
  ``residue_mask`` uses 1 = valid (0 = padded). ``fixed_mask`` uses 1 = fixed,
  the sampling and conditioning convention. Rank-2 inputs contribute the first
  state. ``fixed_mask is None`` means nothing is fixed.
  """
  residue = residue_mask[0] if residue_mask.ndim == 2 else residue_mask
  residue = residue.astype(jnp.float32)
  if fixed_mask is None:
    return residue
  fixed = fixed_mask[0] if fixed_mask.ndim == 2 else fixed_mask
  return residue * (1.0 - fixed.astype(jnp.float32))


def _stable_undesigned_first(order: jnp.ndarray, designed: jnp.ndarray) -> jnp.ndarray:
  """Move non-designed entries ahead of designed ones, keeping relative order.

  ``designed`` is aligned with ``order`` (one flag per entry). An all-designed
  flag vector leaves ``order`` unchanged, so the same key still yields the
  historical permutation.
  """
  n = order.shape[0]
  keys = designed.astype(jnp.int32) * jnp.int32(n) + jnp.arange(n, dtype=jnp.int32)
  return order[jnp.argsort(keys)]


def _group_is_designed(
  chain_mask: jnp.ndarray,
  tie_group_map: jnp.ndarray,
  num_groups: int,
) -> jnp.ndarray:
  """Return a (num_groups,) bool vector.

  A tie group is designed if any member is designed (``chain_mask > 0``).
  Members of that group are decoded with the designed block, including members
  whose own mask is 0. A group with no members is treated as designed so unused
  ids (``num_groups`` larger than the number of occupied ids) do not jump ahead
  under an all-designed mask.
  """
  member = (chain_mask.astype(jnp.float32) > 0).astype(jnp.int32)
  ones = jnp.ones((tie_group_map.shape[0],), dtype=jnp.int32)
  counts = jnp.zeros((num_groups,), dtype=jnp.int32).at[tie_group_map].add(ones)
  designed_sum = jnp.zeros((num_groups,), dtype=jnp.int32).at[tie_group_map].add(member)
  return (designed_sum > 0) | (counts == 0)


@partial(jax.jit, static_argnames=("num_residues", "num_groups"))
def random_decoding_order(
  prng_key: PRNGKeyArray,
  num_residues: int,
  tie_group_map: jnp.ndarray | None = None,
  num_groups: int | None = None,
  *,
  chain_mask: jnp.ndarray | None = None,
) -> DecodingOrderOutputs:
  """Return a random decoding order, optionally respecting tied positions.

  Args:
    prng_key: PRNG key for randomness.
    num_residues: Total number of residues.
    tie_group_map: Optional (N,) array mapping each position to a group ID.
                   Positions with the same group ID are tied and will be
                   decoded together in the same step.
    num_groups: Number of unique groups in tie_group_map. Required if
                tie_group_map is provided. Should equal tie_group_map.max() + 1
                when groups are normalized to [0, 1, ..., num_groups-1].
    chain_mask: Optional (N,) design mask, 1 = designed, 0 = fixed, not
                designed, or padded (ProteinMPNN ``chain_mask``). Omitted means
                every position is designed, which is bit-identical to the
                historical permutation for the same key. When given, non-designed
                positions are decoded first. Within each block the order is the
                relative order of one uniform permutation, so each block is
                itself uniform. A tie group is designed if any member is
                designed; that whole group is decoded in the designed block.
                Groups with no members stay in the designed block.

  Returns:
    Tuple of (decoding_order, next_key) where decoding_order respects ties.

  Example:
    >>> key = jax.random.PRNGKey(0)
    >>> # Without ties: standard random order
    >>> order, key = random_decoding_order(key, 5)
    >>>
    >>> # With ties: positions in same group stay together
    >>> tie_map = jnp.array([0, 1, 0, 2, 1])  # Groups: {0: [0,2], 1: [1,4], 2: [3]}
    >>> order, key = random_decoding_order(key, 5, tie_map, num_groups=3)

  """
  current_key, next_key = jax.random.split(prng_key)

  if num_residues < 0:
    msg = f"num_residues must be non-negative, but got {num_residues}"
    raise TypeError(msg)

  if tie_group_map is None:
    # Standard random order without ties. ``chain_mask is None`` skips the
    # partition so this branch stays bit-identical to the historical permutation.
    decoding_order = jax.random.permutation(current_key, jnp.arange(0, num_residues))
    decoding_order = jnp.asarray(decoding_order, dtype=jnp.int32)
    if chain_mask is not None:
      designed = chain_mask.astype(jnp.float32)[decoding_order] > 0
      decoding_order = _stable_undesigned_first(decoding_order, designed)
    return decoding_order, next_key

  if num_groups is None:
    msg = "num_groups must be provided when tie_group_map is not None"
    raise ValueError(msg)

  # With tied positions: generate random order over groups (vectorized)
  group_order = jax.random.permutation(current_key, jnp.arange(num_groups))
  if chain_mask is not None:
    group_designed = _group_is_designed(chain_mask, tie_group_map, num_groups)
    group_order = _stable_undesigned_first(group_order, group_designed[group_order])

  # Map groups to decoding steps
  decoding_step_map = get_decoding_step_map(tie_group_map, group_order, num_groups)

  # Create decoding order: sort by step, then by position within step
  # This ensures positions in the same group are adjacent
  decoding_order = jnp.argsort(decoding_step_map)

  return jnp.asarray(decoding_order, dtype=jnp.int32), next_key


def single_decoding_order(
  key: PRNGKeyArray,
  num_residues: int,
  tie_group_map: jnp.ndarray | None = None,
  num_groups: int | None = None,
  *,
  chain_mask: jnp.ndarray | None = None,
) -> DecodingOrderOutputs:
  """Generate a single decoding order (identity permutation).

  Args:
      key: Random key (unused, for API compatibility).
      num_residues: Number of residues.
      tie_group_map: Optional (N,) array mapping each position to a group ID.
          Currently ignored for single decoding order.
      num_groups: Number of unique groups (unused, for API compatibility).
      chain_mask: Optional design mask. Ignored; the identity order has no
          random blocks to partition.

  Returns:
      decoding_order: (N,) array, [0, 1, ..., N-1].
      key: Same random key (unchanged).

  """
  del tie_group_map, num_groups, chain_mask  # Unused for single order
  decoding_order = jnp.arange(0, num_residues, dtype=jnp.int32)
  return decoding_order, key
