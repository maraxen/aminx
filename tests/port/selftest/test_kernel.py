"""port_selftest wave: a sign-flipped kernel fails tier 2; a retrace fails tier 5.

Both tests pass. Each one proves the tier check rejects the bad kernel
(``pytest.raises``), so later tiers are not blocked by an earlier failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from port.selftest.kernel import retrace_per_call, sign_flipped_kernel

pytestmark = pytest.mark.port_wave("port_selftest")


def _host_input() -> np.ndarray:
  return np.arange(4, dtype=np.float64)


@pytest.mark.tier_1
def test_tier_1_sign_flip_keeps_shape(oracle: object) -> None:
  """Tier 1 is shape-only, so the sign flip is not yet a failure."""
  reference = np.asarray(oracle.reference_output())  # type: ignore[attr-defined]
  flipped = np.asarray(sign_flipped_kernel(_host_input()))
  assert flipped.shape == reference.shape


@pytest.mark.tier_2
def test_tier_2_sign_flipped_kernel_fails(oracle: object) -> None:
  """Float64 parity rejects ``sign_flipped_kernel`` against the sealed oracle."""
  reference = np.asarray(oracle.reference_output())  # type: ignore[attr-defined]
  flipped = np.asarray(sign_flipped_kernel(_host_input()))
  with pytest.raises(AssertionError):
    np.testing.assert_allclose(flipped, reference, rtol=1e-10, atol=1e-12)


@pytest.mark.tier_3
def test_tier_3_sign_flipped_kernel_fails_f32(oracle: object) -> None:
  """The f32 policy also rejects the sign-flipped kernel."""
  reference = np.asarray(oracle.reference_output(), dtype=np.float32)  # type: ignore[attr-defined]
  flipped = np.asarray(sign_flipped_kernel(_host_input().astype(np.float32)))
  with pytest.raises(AssertionError):
    np.testing.assert_allclose(flipped, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.tier_5
def test_tier_5_retrace_per_call_kernel_fails(max_traces: int) -> None:
  """A kernel that traces again on every call exceeds ``max_traces``."""
  import chex
  import jax.numpy as jnp

  chex.clear_trace_counter()
  sample = jnp.ones((4,), dtype=jnp.float32)
  retrace_per_call(sample, max_traces=max_traces)
  with pytest.raises(AssertionError, match="traced"):
    retrace_per_call(sample, max_traces=max_traces)
