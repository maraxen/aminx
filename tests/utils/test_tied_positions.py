"""Tests for tied positions functionality in decoding order and AR masks."""

import chex
import jax
import jax.numpy as jnp
import pytest

from aminx.utils.autoregression import generate_ar_mask, get_decoding_step_map
from aminx.utils.decoding_order import random_decoding_order, single_decoding_order


class TestDecodingOrderWithTies:
  """Test decoding order generation with tied positions."""

  def test_random_decoding_order_without_ties(self):
    """Test that random_decoding_order works without ties (baseline)."""
    key = jax.random.PRNGKey(42)
    num_residues = 5
    order, _ = random_decoding_order(key, num_residues, tie_group_map=None)

    # Should be a permutation of [0, 1, 2, 3, 4]
    chex.assert_shape(order, (num_residues,))
    assert set(order.tolist()) == set(range(num_residues))

  def test_random_decoding_order_with_ties_groups_together(self):
    """Test that tied positions are grouped together in decoding order."""
    key = jax.random.PRNGKey(42)
    num_residues = 6
    # Group structure: {0: [0, 2, 4], 1: [1, 5], 2: [3]}
    tie_group_map = jnp.array([0, 1, 0, 2, 0, 1])
    num_groups = int(tie_group_map.max()) + 1

    order, _ = random_decoding_order(key, num_residues, tie_group_map, num_groups)

    # Verify all positions are present
    assert set(order.tolist()) == set(range(num_residues))

    # Check that tied positions have the same step
    step_map = get_decoding_step_map(tie_group_map, jnp.arange(num_groups))

    # Positions in same group should have same step
    assert step_map[0] == step_map[2] == step_map[4]  # Group 0
    assert step_map[1] == step_map[5]  # Group 1
    # Position 3 is alone in group 2

  def test_single_decoding_order_without_ties(self):
    """Test single_decoding_order without ties returns identity."""
    key = jax.random.PRNGKey(0)
    num_residues = 5
    order, _ = single_decoding_order(key, num_residues, tie_group_map=None)

    # Should be [0, 1, 2, 3, 4]
    expected = jnp.arange(num_residues)
    chex.assert_trees_all_equal(order, expected)

  def test_single_decoding_order_with_ties_groups_by_id(self):
    """Test single_decoding_order returns identity (no reordering for ties)."""
    key = jax.random.PRNGKey(0)
    num_residues = 6
    # Group IDs: positions sorted by group
    tie_group_map = jnp.array([1, 0, 2, 0, 1, 2])
    num_groups = int(tie_group_map.max()) + 1

    order, _ = single_decoding_order(key, num_residues, tie_group_map, num_groups)

    # Single decoding order should just return identity permutation
    # even with ties (it's "single" = deterministic identity order)
    expected = jnp.arange(num_residues)
    chex.assert_trees_all_equal(order, expected)

  def test_tied_positions_deterministic_with_same_seed(self):
    """Test that tied decoding order is deterministic with same seed."""
    tie_group_map = jnp.array([0, 1, 0, 2, 1])
    num_groups = int(tie_group_map.max()) + 1

    key1 = jax.random.PRNGKey(42)
    order1, _ = random_decoding_order(key1, 5, tie_group_map, num_groups)

    key2 = jax.random.PRNGKey(42)
    order2, _ = random_decoding_order(key2, 5, tie_group_map, num_groups)

    chex.assert_trees_all_equal(order1, order2)

  def test_tied_positions_different_with_different_seed(self):
    """Test that tied decoding order varies with different seeds."""
    tie_group_map = jnp.array([0, 1, 0, 2, 1])
    num_groups = int(tie_group_map.max()) + 1

    key1 = jax.random.PRNGKey(42)
    order1, _ = random_decoding_order(key1, 5, tie_group_map, num_groups)

    key2 = jax.random.PRNGKey(123)
    order2, _ = random_decoding_order(key2, 5, tie_group_map, num_groups)

    # Orders should be different (with very high probability)
    assert not jnp.array_equal(order1, order2)


class TestARMaskWithTies:
  """Test autoregressive mask generation with tied positions."""

  def test_ar_mask_without_ties_baseline(self):
    """Test AR mask generation without ties (baseline)."""
    decoding_order = jnp.array([0, 1, 2])
    mask = generate_ar_mask(decoding_order, tie_group_map=None)

    # Strictly lower-triangular: causal AND self-excluding (was [[1,0,0],[1,1,0],[1,1,1]]).
    expected = jnp.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    chex.assert_trees_all_equal(mask, expected)

  def test_ar_mask_with_ties_allows_group_attention(self):
    """Test that tied positions can attend to each other."""
    # Decoding order: [0, 1, 2, 3, 4]
    # Tie groups: positions 0 and 2 are tied (group 0)
    decoding_order = jnp.array([0, 1, 2, 3, 4])
    tie_group_map = jnp.array([0, 1, 0, 2, 3])
    num_groups = int(tie_group_map.max()) + 1

    mask = generate_ar_mask(decoding_order, tie_group_map=tie_group_map, num_groups=num_groups)

    # Positions 0 and 2 should be able to attend to each other
    # since they're in the same group (group 0)
    assert mask[0, 2] == 1
    assert mask[2, 0] == 1
    # ...but not to THEMSELVES.
    assert mask[0, 0] == 0
    assert mask[2, 2] == 0

    # Group decoding order: group 0 appears first, then group 1, then 2, 3
    # Position 1 (group 1) CAN attend to group 0 (positions 0, 2)
    assert mask[1, 0] == 1
    assert mask[1, 2] == 1
    # But position 1 should NOT attend to later groups (2, 3)
    assert mask[1, 3] == 0
    assert mask[1, 4] == 0
    # And group 0 should NOT attend to later groups
    assert mask[0, 1] == 0
    assert mask[0, 3] == 0

  def test_ar_mask_respects_group_ordering(self):
    """Test that AR mask respects the ordering of tie groups."""
    # Simple case: two groups
    decoding_order = jnp.array([0, 1, 2, 3])
    tie_group_map = jnp.array([0, 0, 1, 1])  # Two pairs
    num_groups = int(tie_group_map.max()) + 1

    mask = generate_ar_mask(decoding_order, tie_group_map=tie_group_map, num_groups=num_groups)

    # Within group 0 (positions 0, 1): should attend to each other
    assert mask[0, 1] == 1
    assert mask[1, 0] == 1
    # ...but not to themselves.
    assert mask[0, 0] == 0
    assert mask[1, 1] == 0

    # Within group 1 (positions 2, 3): should attend to each other
    assert mask[2, 3] == 1
    assert mask[3, 2] == 1
    # ...but not to themselves.
    assert mask[2, 2] == 0
    assert mask[3, 3] == 0

    # Group 1 should attend to group 0 (earlier in order)
    assert mask[2, 0] == 1
    assert mask[2, 1] == 1
    assert mask[3, 0] == 1
    assert mask[3, 1] == 1

    # Group 0 should NOT attend to group 1 (later in order)
    assert mask[0, 2] == 0
    assert mask[0, 3] == 0
    assert mask[1, 2] == 0
    assert mask[1, 3] == 0

  def test_ar_mask_with_ties_and_chains(self):
    """Test AR mask with both tied positions and chain masking."""
    decoding_order = jnp.array([0, 1, 2, 3])
    tie_group_map = jnp.array([0, 0, 1, 1])
    num_groups = int(tie_group_map.max()) + 1
    chain_idx = jnp.array([0, 0, 1, 1])  # Two chains

    mask = generate_ar_mask(
      decoding_order, chain_idx=chain_idx, tie_group_map=tie_group_map, num_groups=num_groups,
    )

    # Within same chain and same group: should attend to the PARTNER, not to self
    assert mask[0, 1] == 1
    assert mask[2, 3] == 1
    assert mask[0, 0] == 0
    assert mask[2, 2] == 0

    # Cross-chain: should NOT attend even if in valid AR position
    assert mask[2, 0] == 0
    assert mask[2, 1] == 0
    assert mask[3, 0] == 0
    assert mask[3, 1] == 0

  def test_ar_mask_shape_and_type(self):
    """Test AR mask has correct shape and type."""
    decoding_order = jnp.array([0, 1, 2, 3, 4])
    tie_group_map = jnp.array([0, 1, 0, 2, 1])
    num_groups = int(tie_group_map.max()) + 1

    mask = generate_ar_mask(decoding_order, tie_group_map=tie_group_map, num_groups=num_groups)

    chex.assert_shape(mask, (5, 5))
    chex.assert_type(mask, jnp.int8)

  def test_ar_mask_no_position_attends_to_itself(self):
    """No position may read its own slot -- inverted 2026-08-27.

    This test previously asserted the opposite, and its name said so. The diagonal was
    believed to be harmless self-attention over an inert placeholder. It was not: undrawn
    slots held token index 0, which is ALANINE in MPNN_ALPHABET (the unknown token X is
    index 20), embedded via `one_hot @ w_s_weight` as a real nonzero row. Both LigandMPNN
    and ColabDesign self-exclude. Measured downstream: mean JSD 0.0296 over designable
    positions, with an alanine-specific probability shift.
    """
    decoding_order = jnp.array([0, 1, 2, 3])
    tie_group_map = jnp.array([0, 1, 0, 2])
    num_groups = int(tie_group_map.max()) + 1

    mask = generate_ar_mask(decoding_order, tie_group_map=tie_group_map, num_groups=num_groups)

    assert jnp.all(jnp.diag(mask) == 0)
    # Non-degeneracy: an all-zero mask would also pass the line above while removing
    # every position's context, which is a separate and worse defect class.
    assert mask.sum() > 0


@pytest.mark.parametrize(
  "tie_group_map,expected_num_groups",
  [
    (jnp.array([0, 0, 0]), 1),
    (jnp.array([0, 1, 2]), 3),
    (jnp.array([0, 1, 0, 1]), 2),
    (jnp.array([0, 0, 1, 1, 2, 2]), 3),
  ],
)
def test_get_decoding_step_map_num_groups(tie_group_map, expected_num_groups):
  """Test that get_decoding_step_map creates correct number of steps."""
  unique_groups = jnp.unique(tie_group_map)
  group_order = unique_groups  # Use identity ordering for testing

  step_map = get_decoding_step_map(tie_group_map, group_order)

  # Number of unique steps should equal number of groups
  num_steps = len(jnp.unique(step_map))
  assert num_steps == expected_num_groups


def test_integration_tied_decoding_and_mask():
  """Integration test: decoding order and AR mask work together."""
  key = jax.random.PRNGKey(42)
  num_residues = 6
  tie_group_map = jnp.array([0, 1, 0, 2, 1, 2])
  num_groups = int(tie_group_map.max()) + 1

  # Generate tied decoding order
  order, _ = random_decoding_order(key, num_residues, tie_group_map, num_groups)

  # Generate AR mask
  mask = generate_ar_mask(order, tie_group_map=tie_group_map, num_groups=num_groups)

  # Verify basic properties
  assert mask.shape == (num_residues, num_residues)
  # Self-EXCLUSION (see tests/inference/test_fixed_position_self_visibility.py docstring)
  assert jnp.all(jnp.diag(mask) == 0)
  assert mask.sum() > 0  # ...but context is not empty

  # Verify group-based autoregressive structure:
  # Positions in same group can attend to each other
  for i in range(num_residues):
    for j in range(num_residues):
      if i == j:
        # A position never reads its own slot, regardless of tie group (2026-08-27).
        assert mask[i, j] == 0, f"Position {i} attends to itself"
      elif tie_group_map[i] == tie_group_map[j]:
        # Same group, different position: mutual visibility is UNCHANGED by that fix.
        assert mask[i, j] == 1, f"Position {i} and {j} in same group should attend"
