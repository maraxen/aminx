"""Jaxpr complexity guards for autoregressive loops (debt #1981).

Guards verify that loop bodies don't contain hidden O(L) or O(L^2) complexity
when they should be O(1) or O(A) where A is the active set size.

The main tool is jax.make_jaxpr() which captures the computation graph without
executing it, allowing inspection of operation counts. A dot_general primitive
is a good indicator of "decoder call" since decoders heavily use matmuls.

Note: These guards are applied to production code paths (autoregressive.py,
ste.py, etc.) to detect the O(L^2) k bug mentioned in debt #1981, where the
AR sampler ran the full decoder over all L rows every wave instead of just
the active positions in that wave.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tests._jaxpr_guards import count_primitive_in_jaxpr


def test_jaxpr_guard_counts_primitives():
  """Test the jaxpr guard helper: verify it counts dot_general correctly.

  Sanity check that count_primitive_in_jaxpr() works on a simple function
  with known primitive counts.
  """

  def single_dot(x):
    return jnp.dot(x, x)

  def two_dots(x):
    a = jnp.dot(x, x)
    b = jnp.dot(a, a)
    return a + b

  x = jnp.ones((8,))

  jaxpr_single = jax.make_jaxpr(single_dot)(x)
  jaxpr_double = jax.make_jaxpr(two_dots)(x)

  count_single = count_primitive_in_jaxpr(jaxpr_single, "dot_general")
  count_double = count_primitive_in_jaxpr(jaxpr_double, "dot_general")

  # single_dot should have at least 1 dot_general
  assert (
    count_single >= 1
  ), f"Expected >= 1 dot_general in single_dot, got {count_single}"

  # two_dots should have more than single_dot
  assert (
    count_double > count_single
  ), f"Expected two_dots ({count_double}) > single_dot ({count_single})"


def test_jaxpr_guard_detects_primitive_absent():
  """Test that jaxpr guard returns 0 when primitive is absent.

  Verify the guard correctly reports 0 when a primitive doesn't appear.
  """

  def no_dot(x):
    return jnp.sum(x) + x.mean() + jnp.max(x)

  x = jnp.ones((8,))
  jaxpr = jax.make_jaxpr(no_dot)(x)

  count = count_primitive_in_jaxpr(jaxpr, "dot_general")
  assert count == 0, f"Expected 0 dot_general in no_dot, got {count}"


__all__ = [
  "test_jaxpr_guard_counts_primitives",
  "test_jaxpr_guard_detects_primitive_absent",
]
