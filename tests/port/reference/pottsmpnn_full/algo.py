# REFERENCE: DO NOT MODIFY
"""Sealed oracle for the pottsmpnn_full wave.

Loads ``pottsmpnn_full/oracle_{f64,f32}.npz`` from ``AMINX_A1_ORACLE_DIR`` and
raises unless the bytes match ``oracles.sha256``.
"""

from __future__ import annotations

import numpy as np

from port.reference.a1_potts.sealed import OracleAbsentError, load_npz, sha_table

__all__ = ["OracleAbsentError", "load"]

_PINS = {
  "pottsmpnn_full/oracle_f64.npz": (
    "0fe956b3e4b4e151df4eea5924f6975a7fdf020abacec41d03b6cfcffb981e85"
  ),
  "pottsmpnn_full/oracle_f32.npz": (
    "245bd1908c23f280a6c662a16c0c70521f8dd772732abebb320bb7846ff14edc"
  ),
}
_TABLE = sha_table()
for _rel, _digest in _PINS.items():
  if _TABLE[_rel] != _digest:
    msg = f"pottsmpnn_full pin drifted from oracles.sha256: {_rel}"
    raise RuntimeError(msg)


def load(precision: str) -> np.lib.npyio.NpzFile:
  """Return the pinned dump for ``precision`` (``f64`` or ``f32``)."""
  relative = {
    "f64": "pottsmpnn_full/oracle_f64.npz",
    "f32": "pottsmpnn_full/oracle_f32.npz",
  }[precision]
  return load_npz(relative)
