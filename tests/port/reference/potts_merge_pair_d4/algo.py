# REFERENCE: DO NOT MODIFY
"""Sealed oracle for the potts_merge_pair_d4 wave.

Loads ``potts_merge_pair_d4/oracle_{f64,f32}.npz`` from ``AMINX_A1_ORACLE_DIR``
and raises unless the bytes match ``oracles.sha256``.
"""

from __future__ import annotations

import numpy as np

from port.reference.a1_potts.sealed import OracleAbsentError, load_npz, sha_table

__all__ = ["OracleAbsentError", "load"]

_PINS = {
  "potts_merge_pair_d4/oracle_f64.npz": (
    "4e30f9546c8d04e9d30fb2abb128b2b36a67fb311eeb674cff2373138f18c07e"
  ),
  "potts_merge_pair_d4/oracle_f32.npz": (
    "70b20636f350cf6f262453874780bafad140ebebc897a4aa73752bdac79deeb5"
  ),
}
_TABLE = sha_table()
for _rel, _digest in _PINS.items():
  if _TABLE[_rel] != _digest:
    msg = f"potts_merge_pair_d4 pin drifted from oracles.sha256: {_rel}"
    raise RuntimeError(msg)


def load(precision: str) -> np.lib.npyio.NpzFile:
  """Return the pinned dump for ``precision`` (``f64`` or ``f32``)."""
  relative = {
    "f64": "potts_merge_pair_d4/oracle_f64.npz",
    "f32": "potts_merge_pair_d4/oracle_f32.npz",
  }[precision]
  return load_npz(relative)
