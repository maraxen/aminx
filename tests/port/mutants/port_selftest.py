"""Mutant context managers for the port_selftest wave.

The self-test kernels are closed over by the tier tests; these managers exist
so AMINX_PORT_MUTANT can name a registered function at session start.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def selftest_noop() -> Iterator[None]:
  """Leave the sign-flipped kernel unchanged (mutant run still passes tier checks)."""
  yield


@contextmanager
def selftest_raise() -> Iterator[None]:
  """Registered name for the RuntimeError mutant outcome fixture."""
  yield
