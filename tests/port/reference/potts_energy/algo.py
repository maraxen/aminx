# REFERENCE: DO NOT MODIFY
"""Sealed oracle for the potts_energy wave.

Loads ``potts_energy/oracle_{f64,f32}.npz`` from ``AMINX_A1_ORACLE_DIR`` and
raises unless the bytes match ``oracles.sha256``.
"""

from __future__ import annotations

import numpy as np

from port.reference.a1_potts.sealed import OracleAbsentError, load_npz, sha_table

__all__ = ["OracleAbsentError", "load"]

_PINS = {
  "potts_energy/oracle_f64.npz": "94c4036a9b7647eecf2ca3538470c4e1bab2b3fd284d074ff90118526a3e5d5c",
  "potts_energy/oracle_f32.npz": "9bb53449a2eb236ea94affff96bd0228345765496328ce795ee808332f14e26f",
}
_TABLE = sha_table()
for _rel, _digest in _PINS.items():
  if _TABLE[_rel] != _digest:
    msg = f"potts_energy pin drifted from oracles.sha256: {_rel}"
    raise RuntimeError(msg)


def load(precision: str) -> np.lib.npyio.NpzFile:
  """Return the pinned dump for ``precision`` (``f64`` or ``f32``)."""
  relative = {
    "f64": "potts_energy/oracle_f64.npz",
    "f32": "potts_energy/oracle_f32.npz",
  }[precision]
  return load_npz(relative)
