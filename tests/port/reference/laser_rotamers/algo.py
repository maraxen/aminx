# REFERENCE: DO NOT MODIFY
"""Sealed oracle for the laser_rotamers wave.

Loads ``laser_rotamers/oracle_{f64,f32}.npz`` from ``AMINX_LASER_ORACLE_DIR``.
"""

from __future__ import annotations

import numpy as np

from port.reference.laser_rotamers.sealed import OracleAbsentError, load_npz

__all__ = ["OracleAbsentError", "load"]


def load(precision: str) -> np.lib.npyio.NpzFile:
  """Return the dump for ``precision`` (``f64`` or ``f32``)."""
  relative = {
    "f64": "laser_rotamers/oracle_f64.npz",
    "f32": "laser_rotamers/oracle_f32.npz",
  }[precision]
  return load_npz(relative)
