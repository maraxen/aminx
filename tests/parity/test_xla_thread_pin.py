"""XLA CPU thread-pin check (T8 step 0).

The matmul runs in a subprocess so XLA reads ``XLA_FLAGS`` at backend init.
jax 0.10.2 still schedules the 2048² f32 matmul across the CPU pool when
``XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"``
is set (``intra_op_parallelism_threads`` is an ``xla.ExecutionOptions`` field, not
an XLA debug flag, and the eigen flag does not pin the thunk/OneDNN pool). The
pinned CPU/wall ratio stays above 1.2, so this test is an expected failure and
the bench records ``native_threads = "unpinned"`` and prints no native ratio.
"""

from __future__ import annotations

import pytest

from scripts.browser_validation.layer_c_bench import thread_pin_status


@pytest.mark.xfail(
  strict=True,
  reason=(
    "jax 0.10.2 does not pin a 2048^2 f32 matmul: CPU/wall stays above 1.2 under "
    "XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1' "
    "(eigen flag and ExecutionOptions intra_op field do not cap the CPU pool). "
    "native_threads = 'unpinned'; no native ratio is printed."
  ),
)
def test_xla_thread_pin() -> None:
  status = thread_pin_status()
  assert status["pinned_ratio"] <= 1.2
  assert status["unpinned_ratio"] > 1.5
