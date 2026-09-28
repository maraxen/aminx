"""Regression test for Debt #119: tie-group fusion with mixed tied/untied positions.

Verifies that _apply_tie_group_fuse correctly handles a MIXED tie map where
some positions are tied together and others are untied (singletons).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_apply_tie_group_fuse_mixed_groups_untied_unchanged() -> None:
  """Verify untied positions keep their original logits exactly.

  When a tie map has mixed groups (some tied, some singleton), untied
  (singleton) positions should pass through _apply_tie_group_fuse unchanged.
  This ensures the untied path is value-preserving (no spurious numerical changes).
  """
  from aminx.inference.decode._base import _ConditionalDecodeBase
  from aminx.inference.logits import make_stage_set

  # _apply_tie_group_fuse is a static method of _ConditionalDecodeBase
  _apply_tie_group_fuse = _ConditionalDecodeBase._apply_tie_group_fuse

  # Create test logits: L=6, V=21
  rng = np.random.default_rng(42)
  logits = jnp.asarray(rng.normal(size=(6, 21)).astype(np.float32))

  # Mixed tie map: positions [0, 1] tied, position [2] untied (singleton),
  # positions [3, 4] tied, position [5] untied (singleton).
  # Compact representation: [0, 0, 1, 2, 2, 3]
  tie_group_map = jnp.asarray([0, 0, 1, 2, 2, 3], dtype=jnp.int32)

  stage_set = make_stage_set()
  fused_logits = _apply_tie_group_fuse(logits, stage_set, tie_group_map)
  fused_np = np.asarray(fused_logits)

  # Untied positions (2 and 5) should equal their pre-fusion logits exactly
  np.testing.assert_array_equal(
    fused_np[2],
    np.asarray(logits[2]),
    err_msg="Untied position 2 should be unchanged",
  )
  np.testing.assert_array_equal(
    fused_np[5],
    np.asarray(logits[5]),
    err_msg="Untied position 5 should be unchanged",
  )


def test_apply_tie_group_fuse_mixed_groups_tied_shared() -> None:
  """Verify tied positions share the same logits within each group.

  When a tie map has mixed groups, positions in the same tied group should
  receive the fused (shared) logits via the configured tie_group_fuse strategy.
  """
  from aminx.inference.decode._base import _ConditionalDecodeBase
  from aminx.inference.logits import make_stage_set

  # _apply_tie_group_fuse is a static method of _ConditionalDecodeBase
  _apply_tie_group_fuse = _ConditionalDecodeBase._apply_tie_group_fuse

  # Create test logits: L=6, V=21
  rng = np.random.default_rng(42)
  logits = jnp.asarray(rng.normal(size=(6, 21)).astype(np.float32))

  # Mixed tie map: positions [0, 1] tied, position [2] untied,
  # positions [3, 4] tied, position [5] untied.
  tie_group_map = jnp.asarray([0, 0, 1, 2, 2, 3], dtype=jnp.int32)

  stage_set = make_stage_set()
  fused_logits = _apply_tie_group_fuse(logits, stage_set, tie_group_map)
  fused_np = np.asarray(fused_logits)

  # Tied positions within the same group should have identical logits
  # Group 1: [0, 1]
  np.testing.assert_allclose(
    fused_np[0],
    fused_np[1],
    rtol=1e-5,
    atol=1e-5,
    err_msg="Positions 0 and 1 (tied group) should have identical logits",
  )

  # Group 2: [3, 4]
  np.testing.assert_allclose(
    fused_np[3],
    fused_np[4],
    rtol=1e-5,
    atol=1e-5,
    err_msg="Positions 3 and 4 (tied group) should have identical logits",
  )
