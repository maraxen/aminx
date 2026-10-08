# REFERENCE: DO NOT MODIFY
"""Sealed oracle for the laser_decode_step wave.

Loads ``laser_decode_step/oracle_{f64,f32}.npz`` from ``AMINX_LASER_ORACLE_DIR``.
``injected_uniform`` names the dump key whose draws make the categorical
samples deterministic. A stochastic wave whose oracle module lacks that
attribute fails in ``tests/port/conftest.py``.
"""

from __future__ import annotations

import numpy as np

from port.reference.laser_decode_step.sealed import OracleAbsentError, load_npz

__all__ = ["OracleAbsentError", "injected_uniform", "load"]

injected_uniform = "uniforms"


def load(precision: str) -> np.lib.npyio.NpzFile:
  """Return the dump for ``precision`` (``f64`` or ``f32``)."""
  relative = {
    "f64": "laser_decode_step/oracle_f64.npz",
    "f32": "laser_decode_step/oracle_f32.npz",
  }[precision]
  return load_npz(relative)
