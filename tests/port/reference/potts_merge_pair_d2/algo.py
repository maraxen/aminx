# REFERENCE: DO NOT MODIFY
"""Sealed oracle for the potts_merge_pair_d2 wave.

Loads ``potts_merge_pair_d2/oracle_{f64,f32}.npz`` from ``AMINX_A1_ORACLE_DIR``
and raises unless the bytes match ``oracles.sha256``.
"""

from __future__ import annotations

import numpy as np

from port.reference.a1_potts.sealed import OracleAbsentError, load_npz, sha_table

__all__ = ["OracleAbsentError", "load"]

_PINS = {
  "potts_merge_pair_d2/oracle_f64.npz": (
    "657ee24bdd6152050440c12110987b9582da062cf26d8c2e3e3290555dc84ca2"
  ),
  "potts_merge_pair_d2/oracle_f32.npz": (
    "09d291fa52848dd58b1de2b341e6620d65f2b71abd8da548aba4712470636c7c"
  ),
}
_TABLE = sha_table()
for _rel, _digest in _PINS.items():
  if _TABLE[_rel] != _digest:
    msg = f"potts_merge_pair_d2 pin drifted from oracles.sha256: {_rel}"
    raise RuntimeError(msg)


def load(precision: str) -> np.lib.npyio.NpzFile:
  """Return the pinned dump for ``precision`` (``f64`` or ``f32``)."""
  relative = {
    "f64": "potts_merge_pair_d2/oracle_f64.npz",
    "f32": "potts_merge_pair_d2/oracle_f32.npz",
  }[precision]
  return load_npz(relative)
