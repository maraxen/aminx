"""Empty-neighbour masked softmax.

Upstream ``scatter_softmax`` is undefined on a sink with no edges (NaN). The
dense port returns 0 for that row.
"""

from __future__ import annotations

import jax.numpy as jnp

from aminx.model.laser.layers import masked_softmax


def test_masked_softmax_empty_row_is_zero() -> None:
  """A row with no valid neighbours is 0, including the all-masked batch row."""
  logits = jnp.array([[1.5, -0.25, 0.5], [4.0, 4.0, 4.0]], dtype=jnp.float32)
  mask = jnp.array([[True, False, True], [False, False, False]])
  weights = masked_softmax(logits, mask)
  assert bool(jnp.all(jnp.isfinite(weights)))
  assert weights.shape == logits.shape
  empty = weights[1]
  assert float(jnp.max(jnp.abs(empty))) == 0.0
  kept = weights[0]
  assert float(kept[1]) == 0.0
  total = float(kept[0] + kept[2])
  assert abs(total - 1.0) < 1e-6
  # The two valid logits softmax against each other, not against the pad.
  pair = jnp.exp(logits[0, jnp.array([0, 2])] - jnp.max(logits[0, jnp.array([0, 2])]))
  pair = pair / jnp.sum(pair)
  assert float(jnp.max(jnp.abs(kept[jnp.array([0, 2])] - pair))) < 1e-6
