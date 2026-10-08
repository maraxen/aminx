# REFERENCE: DO NOT MODIFY
"""Sealed oracle for the potts_head wave.

Loads ``potts_head/oracle_{f64,f32}.npz`` from ``AMINX_A1_ORACLE_DIR`` and
raises unless the bytes match ``tests/port/reference/a1_potts/oracles.sha256``.
"""

from __future__ import annotations

import numpy as np

from port.reference.a1_potts.sealed import OracleAbsentError, load_npz, sha_table

__all__ = ["OracleAbsentError", "load"]

_PINS = {
  "potts_head/oracle_f64.npz": "4bad82488791105f28ff66518611cd28d4b0d46c5007a5e36e0b18a9c84423c9",
  "potts_head/oracle_f32.npz": "4fea049f8d26ea3b1a253b498329621e8641875dc7ace3b59ab0f18ad24a032a",
}
_TABLE = sha_table()
for _rel, _digest in _PINS.items():
  if _TABLE[_rel] != _digest:
    msg = f"potts_head pin drifted from oracles.sha256: {_rel}"
    raise RuntimeError(msg)


def load(precision: str) -> np.lib.npyio.NpzFile:
  """Return the pinned dump for ``precision`` (``f64`` or ``f32``)."""
  relative = {
    "f64": "potts_head/oracle_f64.npz",
    "f32": "potts_head/oracle_f32.npz",
  }[precision]
  return load_npz(relative)
