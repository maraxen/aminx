"""Intentionally wrong kernels exercised by the port_selftest wave."""

from __future__ import annotations

import jax
from jaxtyping import Array, Float


def sign_flipped_kernel(x: Float[Array, " n"]) -> Float[Array, " n"]:
  """Negate the sealed ``2x`` kernel. Tier 2 numeric parity must reject it."""
  return -(x * 2)


def retrace_per_call(x: Float[Array, " n"], *, max_traces: int) -> Float[Array, " n"]:
  """JIT a fresh nested kernel on every call so the trace budget is exceeded.

  ``max_traces`` is the wave cap. The nested function shares one chex footprint
  across calls, so the second call traces again and fails the budget.
  """
  import chex

  @chex.assert_max_traces(n=max_traces)
  def kernel(y: Float[Array, " n"]) -> Float[Array, " n"]:
    return -y

  return jax.jit(kernel)(x)
