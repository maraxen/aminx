"""Shared utilities for jaxpr complexity guards (debt #1981).

This module provides helpers to assert loop-body complexity invariants
by tracing jaxpr at different input scales and counting primitive operations.

Key invariant: A loop's per-iteration cost must match its intended complexity.
E.g., a wave loop decoding A positions per wave should NOT have per-wave cost O(L)
where L is total sequence length — it should be O(A) and independent of L.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp


def count_primitive_in_jaxpr(jaxpr: jax.core.ClosedJaxpr, primitive_name: str) -> int:
  """Count occurrences of a named primitive in a jaxpr.

  Parameters
  ----------
  jaxpr : jax.core.ClosedJaxpr
      The jaxpr to search.
  primitive_name : str
      Name of the primitive to count (e.g., 'dot_general', 'transpose').

  Returns
  -------
  int
      Count of matching primitives in jaxpr.literals + eqns.
  """

  def _count_eqns(eqn_list):
    return sum(1 for eqn in eqn_list if eqn.primitive.name == primitive_name)

  count = _count_eqns(jaxpr.jaxpr.eqns)
  # Also recurse into nested jaxprs in literals
  for lit in jaxpr.literals:
    if isinstance(lit, jax.core.ClosedJaxpr):
      count += count_primitive_in_jaxpr(lit, primitive_name)
  return count


def get_jaxpr_primitive_counts(
  fn: Callable,
  args: tuple[Any, ...],
) -> dict[str, int]:
  """Trace a function and return a dict of primitive counts.

  Parameters
  ----------
  fn : Callable
      Function to trace. Should be traceable (e.g., JAX operations).
  args : tuple
      Arguments to trace with.

  Returns
  -------
  dict[str, int]
      Dict mapping primitive name to count.
      Only includes primitives that appear at least once.
  """
  jaxpr = jax.make_jaxpr(fn)(*args)

  # Collect all primitives from the main jaxpr
  primitives: dict[str, int] = {}

  def _walk_eqns(eqn_list):
    for eqn in eqn_list:
      name = eqn.primitive.name
      primitives[name] = primitives.get(name, 0) + 1

  _walk_eqns(jaxpr.jaxpr.eqns)

  # Recurse into literals that might be jaxprs
  def _walk_literals(lit):
    if isinstance(lit, jax.core.ClosedJaxpr):
      _walk_eqns(lit.jaxpr.eqns)
      for sub_lit in lit.literals:
        _walk_literals(sub_lit)

  for lit in jaxpr.literals:
    _walk_literals(lit)

  return primitives


def assert_jaxpr_complexity_scales(
  fn: Callable,
  small_args: tuple[Any, ...],
  large_args: tuple[Any, ...],
  scale_name: str,
  should_scale: bool = False,
  primitive: str = "dot_general",
  min_scale_ratio: float = 2.0,
) -> None:
  """Assert that a primitive's count scales (or doesn't) with input size.

  Parameters
  ----------
  fn : Callable
      Function to trace.
  small_args : tuple
      Arguments with small scale (e.g., L=32).
  large_args : tuple
      Arguments with large scale (e.g., L=64).
  scale_name : str
      Human-readable name of the scaled dimension (e.g., "L").
  should_scale : bool, default False
      If False, assert count stays roughly constant.
      If True, assert count grows with ratio >= min_scale_ratio.
  primitive : str, default 'dot_general'
      Name of primitive to count.
  min_scale_ratio : float, default 2.0
      Minimum ratio (large/small) to consider "scaling".
      Only used if should_scale=True.

  Raises
  ------
  AssertionError
      If complexity does not match the expected behavior.
  """
  small_jaxpr = jax.make_jaxpr(fn)(*small_args)
  large_jaxpr = jax.make_jaxpr(fn)(*large_args)

  small_count = count_primitive_in_jaxpr(small_jaxpr, primitive)
  large_count = count_primitive_in_jaxpr(large_jaxpr, primitive)

  if should_scale:
    ratio = large_count / max(small_count, 1)
    assert (
      ratio >= min_scale_ratio
    ), f"{primitive} count did not scale as expected (ratio {ratio} < {min_scale_ratio}): small={small_count}, large={large_count}"
  else:
    # Should NOT scale: counts should be similar
    relative_change = abs(large_count - small_count) / max(small_count, large_count, 1)
    assert (
      relative_change < 0.5
    ), f"{primitive} count grew with {scale_name} when it shouldn't (small={small_count}, large={large_count}, change={relative_change})"


__all__ = [
  "count_primitive_in_jaxpr",
  "get_jaxpr_primitive_counts",
  "assert_jaxpr_complexity_scales",
]
