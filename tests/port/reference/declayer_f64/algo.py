# REFERENCE: DO NOT MODIFY
"""Sealed oracle for the declayer_f64 wave.

Loads ``declayer_f64.npz`` from ``AMINX_A1_ORACLE_DIR`` and raises unless the
bytes match ``oracles.sha256``. The fixture is f64 only.
"""

from __future__ import annotations

import numpy as np

from port.reference.a1_potts.sealed import OracleAbsentError, load_npz, sha_table

__all__ = ["OracleAbsentError", "load"]

_REL = "declayer_f64.npz"
_SHA256 = "62e55be135cd13a4c80463e42cbf06589357f693573f253a56232b8ce5b2d3fd"
if sha_table()[_REL] != _SHA256:
  msg = "declayer_f64 pin drifted from oracles.sha256"
  raise RuntimeError(msg)


def load(precision: str) -> np.lib.npyio.NpzFile:
  """Return the pinned f64 dump. ``f32`` is absent and raises ``OracleAbsentError``."""
  if precision != "f64":
    msg = f"declayer_f64 has no {precision} dump"
    raise OracleAbsentError(msg)
  return load_npz(_REL)
