# REFERENCE: DO NOT MODIFY
"""Sealed oracle for the port_selftest wave.

Loads ``fixture.npz`` and refuses to run if its SHA-256 changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

_NPZ = Path(__file__).with_name("fixture.npz")
_SHA256 = "523804d2c4b406bedd95b048d346895b9fd30a101b47f8a507867ef7172a3528"


def _sealed() -> np.lib.npyio.NpzFile:
  raw = _NPZ.read_bytes()
  actual = hashlib.sha256(raw).hexdigest()
  if actual != _SHA256:
    msg = f"sealed npz hash mismatch: {actual} != {_SHA256}"
    raise RuntimeError(msg)
  return np.load(_NPZ)


def reference_output() -> np.ndarray:
  """Sealed ``y = 2x`` vector."""
  with _sealed() as data:
    return np.array(data["y"], copy=True)
