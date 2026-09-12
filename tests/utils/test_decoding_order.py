"""Tests for decoding order utilities."""

import chex
import jax
import jax.numpy as jnp
import pytest

from aminx.utils.decoding_order import random_decoding_order


def test_random_decoding_order_properties():
  """Test properties of the random_decoding_order function.

  Raises:
      AssertionError: If output shapes, types, or values are incorrect.

  """
  key = jax.random.PRNGKey(42)
  num_residues = 10

  decoding_order, next_key = random_decoding_order(key, num_residues)

  # Check output shapes
  chex.assert_shape(decoding_order, (num_residues,))
  chex.assert_shape(next_key, key.shape)

  # Check output dtype
  chex.assert_type(decoding_order, jnp.int32)

  # Check that the PRNG key was consumed and changed
  assert not jnp.all(key == next_key), "PRNGKey was not updated."

  # Check that the output is a permutation of the input range
  # The sorted order should be identical to jnp.arange
  sorted_order = jnp.sort(decoding_order)
  expected_order = jnp.arange(num_residues, dtype=jnp.int32)
  chex.assert_trees_all_equal(sorted_order, expected_order)


def test_random_decoding_order_is_deterministic_with_same_key():
  """Test that the same key produces the same decoding order."""
  key = jax.random.PRNGKey(0)
  num_residues = 50

  order1, _ = random_decoding_order(key, num_residues)
  order2, _ = random_decoding_order(key, num_residues)

  chex.assert_trees_all_equal(order1, order2)


@pytest.mark.parametrize("num_residues", [1, 10, 100])
def test_random_decoding_order_with_various_lengths(num_residues):
  """Test the function with different sequence lengths."""
  key = jax.random.PRNGKey(num_residues)
  decoding_order, _ = random_decoding_order(key, num_residues)

  chex.assert_shape(decoding_order, (num_residues,))
  assert jnp.unique(decoding_order).shape[0] == num_residues

def test_random_decoding_order_with_zero_length():
  """Test the function with zero-length sequences."""
  key = jax.random.PRNGKey(0)
  decoding_order, _ = random_decoding_order(key, 0)

  chex.assert_shape(decoding_order, (0,))
  chex.assert_trees_all_equal(decoding_order, jnp.array([], dtype=jnp.int32))

def test_random_decoding_order_with_negative_length():
  """Test the function with negative sequence lengths."""
  key = jax.random.PRNGKey(0)

  with pytest.raises(TypeError):
    random_decoding_order(key, -5)

  with pytest.raises(TypeError):
    random_decoding_order(key, -1)

def test_random_decoding_order_from_array_shape():
  """Test the function with an array shape input."""
  key = jax.random.PRNGKey(0)
  arr = jnp.zeros((5, 4, 3))  # Example shape
  num_residues = arr.shape[0]

  decoding_order, _ = random_decoding_order(key, num_residues)

  chex.assert_shape(decoding_order, (num_residues,))
  assert jnp.unique(decoding_order).shape[0] == num_residues


# ---------------------------------------------------------------------------
# Tie order must be stated, not inherited from the backend's sort stability.
#
# `decoding_step_map` maps tie GROUPS to steps, so every position in a group
# shares a step value exactly. A single-key argsort delivers the documented
# "then by position within step" only if the backend sorts stably, and IREE
# does not -- measured 260911, an argsort over 64 slots holding 4 shuffled tie
# groups disagreed with eager JAX at 44 of 64 positions once compiled, while
# still returning a valid sort.
#
# These tests are STRUCTURAL on purpose. An eager JAX comparison cannot detect
# the problem, because JAX's own sort IS stable, so eager and eager always
# agree. The claim has to be made about the `sort` that is actually emitted.
# ---------------------------------------------------------------------------

TIE_MAP = jnp.array([2, 0, 1, 0, 2, 1, 0, 2], dtype=jnp.int32)
N_GROUPS = 3


def _sort_params(fn, *args):
  """Params of every `sort` primitive in fn's jaxpr, sub-jaxprs included.

  Recursion is mandatory rather than defensive: `random_decoding_order` is
  `jax.jit`-decorated, so its `sort` lives inside a nested sub-jaxpr and a
  top-level-only walk finds nothing at all and reports a confident all-clear.
  """
  found = []

  def walk(jaxpr):
    for eqn in jaxpr.eqns:
      if eqn.primitive.name == "sort":
        found.append(dict(eqn.params))
      for value in eqn.params.values():
        for candidate in value if isinstance(value, (list, tuple)) else (value,):
          inner = getattr(candidate, "jaxpr", candidate)
          if hasattr(inner, "eqns"):
            walk(inner)

  walk(jax.make_jaxpr(fn)(*args).jaxpr)
  return found


def test_tied_decoding_order_sorts_on_two_keys():
  """The emitted sort must carry the position index as a second key.

  With the index as a sort key the ordering is a strict total order -- no two
  entries can compare equal -- so every correct sort returns the same
  permutation and stability stops mattering.
  """
  n = int(TIE_MAP.shape[0])
  params = _sort_params(
    lambda k, tm: random_decoding_order(k, n, tm, N_GROUPS),
    jax.random.PRNGKey(0),
    TIE_MAP,
  )
  assert params, (
    "no sort primitive found in random_decoding_order's jaxpr -- either the "
    "implementation stopped sorting, or _sort_params stopped recursing."
  )
  assert any(p.get("num_keys", 1) >= 2 for p in params), (
    f"the tied decoding order was computed by sorts with num_keys="
    f"{[p.get('num_keys') for p in params]}. With a single key its tie order "
    "is whatever the backend's stability happens to give, and IREE does not "
    "honour stable-sort tie order, so the documented 'then by position within "
    "step' silently stops holding once compiled."
  )


def test_single_key_argsort_would_not_satisfy_the_guard():
  """Control: proves the assertion above discriminates.

  This is the implementation the guard exists to rule out.
  """
  params = _sort_params(lambda x: jnp.argsort(x), TIE_MAP)
  assert [p.get("num_keys") for p in params] == [1]


def test_tied_positions_are_adjacent_and_ascending_within_a_step():
  """The documented contract, asserted directly rather than assumed."""
  n = int(TIE_MAP.shape[0])
  for seed in (0, 1, 2, 3):
    order, _ = random_decoding_order(jax.random.PRNGKey(seed), n, TIE_MAP, N_GROUPS)
    order = jnp.asarray(order)
    groups = TIE_MAP[order]
    # every group occupies one contiguous run
    changes = int(jnp.sum(groups[1:] != groups[:-1]))
    assert changes == N_GROUPS - 1, f"seed={seed}: groups are not contiguous"
    # within each run, positions ascend
    for g in range(N_GROUPS):
      positions = order[groups == g]
      assert bool(jnp.all(jnp.diff(positions) > 0)), (
        f"seed={seed}, group={g}: tied positions are not in ascending order"
      )


def test_tied_order_is_still_a_permutation():
  n = int(TIE_MAP.shape[0])
  order, _ = random_decoding_order(jax.random.PRNGKey(7), n, TIE_MAP, N_GROUPS)
  assert jnp.unique(order).shape[0] == n
  chex.assert_type(order, jnp.int32)
