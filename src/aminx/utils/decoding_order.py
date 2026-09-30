"""Utilities for decoding order generation.

aminx.utils.decoding_order
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from aminx.types.arrays import (
  DecodingOrder,
)

from .autoregression import get_decoding_step_map

DecodingOrderInputs = tuple[PRNGKeyArray, int, jnp.ndarray | None]  # Added tie_group_map
DecodingOrderOutputs = tuple[DecodingOrder, PRNGKeyArray]
DecodingOrderFn = Callable[
  [PRNGKeyArray, int, jnp.ndarray | None, int | None],
  DecodingOrderOutputs,
]


@partial(jax.jit, static_argnames=("num_residues", "num_groups"))
def random_decoding_order(
  prng_key: PRNGKeyArray,
  num_residues: int,
  tie_group_map: jnp.ndarray | None = None,
  num_groups: int | None = None,
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

  Returns:
    Tuple of (decoding_order, next_key) where decoding_order respects ties.
    ``decoding_order`` is an ORDER array: ``decoding_order[t]`` is the position decoded at
    step ``t`` (not each position's rank). Build masks from it with
    :func:`aminx.utils.autoregression.ar_mask_from_decoding_order`, not by passing it to
    ``generate_ar_mask``, which reads its argument as a rank array when untied.

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
    # Standard random order without ties
    decoding_order = jax.random.permutation(current_key, jnp.arange(0, num_residues))
    return jnp.asarray(decoding_order, dtype=jnp.int32), next_key

  if num_groups is None:
    msg = "num_groups must be provided when tie_group_map is not None"
    raise ValueError(msg)

  # With tied positions: generate random order over groups (vectorized)
  group_order = jax.random.permutation(current_key, jnp.arange(num_groups))

  # Map groups to decoding steps
  decoding_step_map = get_decoding_step_map(tie_group_map, group_order, num_groups)

  # Sort by step, then by position within step, so positions in the same group
  # are adjacent and in ascending position order.
  #
  # Both keys are stated explicitly rather than leaning on sort stability.
  # `decoding_step_map` maps tie GROUPS to steps, so every position in a group
  # shares a step value EXACTLY -- ties by construction, not near-ties -- and a
  # single-key `jnp.argsort(decoding_step_map)` delivers "then by position" only
  # if the backend happens to sort stably. IREE does not: measured 260911
  # against iree-base-compiler 3.11, an argsort over 64 slots holding 4 shuffled
  # tie groups disagreed with eager JAX at 44 of 64 positions once compiled.
  # IREE's answer is a valid sort every time -- it is the tie order, not the
  # sort, that differs -- so the failure is silent, and an eager-vs-eager test
  # cannot see it because JAX's own sort IS stable.
  #
  # Sorting on (step, index) with `num_keys=2` is a strict total order: no two
  # entries compare equal, because no two indices are equal, so every correct
  # sort must return this permutation whether or not it is stable. `is_stable`
  # is passed False deliberately -- it is genuinely irrelevant now, and asking
  # for it would imply otherwise. Same fix as `model.features.top_k` (PR #156).
  index = jax.lax.broadcasted_iota(jnp.int32, decoding_step_map.shape, 0)
  _, decoding_order = jax.lax.sort(
    (decoding_step_map, index),
    dimension=0,
    is_stable=False,
    num_keys=2,
  )

  return jnp.asarray(decoding_order, dtype=jnp.int32), next_key


@jax.jit
def random_design_order(
  prng_key: PRNGKeyArray,
  tie_group_map: jnp.ndarray,
  fixed_mask: jnp.ndarray | None = None,
) -> DecodingOrder:
  """The default sampling order: fixed positions first, then a uniform random order of groups.

  Mirrors reference ProteinMPNN/LigandMPNN, which decode in
  ``argsort((chain_mask + 1e-4) * |randn|)``: known (fixed) positions come first, so every
  designed position is conditioned on them, and the designed positions follow in a uniformly
  random order. Tie groups are drawn as units -- one random score per GROUP, so group order
  is uniform over groups regardless of their size (drawing per position would favour large
  groups) -- and a group's members are adjacent, in position order.

  A group counts as fixed if any member is fixed (the sampler writes the fixed token to the
  whole group).

  Args:
    prng_key: Key for the order draw.
    tie_group_map: (L,) tie group id per position, ids in ``[0, L)``. Use ``arange(L)`` for
      untied.
    fixed_mask: Optional (L,) mask, > 0.5 where the token is fixed.

  Returns:
    (L,) int32 ORDER array (``order[t]`` = position decoded at step ``t``).
  """
  seq_len = tie_group_map.shape[0]
  tie = jnp.asarray(tie_group_map, dtype=jnp.int32)
  group_score = jax.random.uniform(prng_key, (seq_len,))
  if fixed_mask is None:
    group_fixed = jnp.zeros((seq_len,), dtype=jnp.int32)
  else:
    fixed = (jnp.asarray(fixed_mask) > 0.5).astype(jnp.int32)  # noqa: PLR2004
    group_fixed = jnp.zeros((seq_len,), dtype=jnp.int32).at[tie].max(fixed)
  designable = 1 - group_fixed[tie]
  positions = jnp.arange(seq_len, dtype=jnp.int32)
  # Three explicit keys (designable, group score, position) give a strict total order, so the
  # result does not depend on sort stability (IREE's sort is not stable).
  *_, order = jax.lax.sort(
    (designable, group_score[tie], positions),
    dimension=0,
    is_stable=False,
    num_keys=3,
  )
  return order.astype(jnp.int32)


def single_decoding_order(
  key: PRNGKeyArray,
  num_residues: int,
  tie_group_map: jnp.ndarray | None = None,
  num_groups: int | None = None,
) -> DecodingOrderOutputs:
  """Generate a single decoding order (identity permutation).

  Args:
      key: Random key (unused, for API compatibility).
      num_residues: Number of residues.
      tie_group_map: Optional (N,) array mapping each position to a group ID.
          Currently ignored for single decoding order.
      num_groups: Number of unique groups (unused, for API compatibility).

  Returns:
      decoding_order: (N,) array, [0, 1, ..., N-1].
      key: Same random key (unchanged).

  """
  del tie_group_map, num_groups  # Unused for single order
  decoding_order = jnp.arange(0, num_residues, dtype=jnp.int32)
  return decoding_order, key
