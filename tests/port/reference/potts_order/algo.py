"""Sealed oracle for the potts_order wave.

Loads ``potts_order/order_f64.npz`` from ``AMINX_A1_ORACLE_DIR``. A missing
file raises ``OracleAbsentError`` so the wave skips before
``dump_potts_oracles.py --only order`` has appended the sha line. Once the
file is present, the sha line in ``oracles.sha256`` has to match.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from port.reference.a1_potts.sealed import OracleAbsentError, oracle_root, sha_table

__all__ = ["OracleAbsentError", "load"]

_RELATIVE = "potts_order/order_f64.npz"


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def load(precision: str) -> np.lib.npyio.NpzFile:
  """Return the order dump. Only ``f64`` exists; anything else is absent."""
  if precision != "f64":
    msg = f"potts_order dump is f64 only, got {precision}"
    raise OracleAbsentError(msg)
  root = oracle_root()
  path = root / _RELATIVE
  if not root.is_dir() or not path.is_file():
    msg = f"oracle dump absent: {path}"
    raise OracleAbsentError(msg)
  expected = sha_table().get(_RELATIVE)
  if expected is None:
    msg = f"no sha256 entry for {_RELATIVE}"
    raise RuntimeError(msg)
  actual = _file_sha256(path)
  if actual != expected:
    msg = f"sealed npz hash mismatch: {actual} != {expected}"
    raise RuntimeError(msg)
  return np.load(path)
