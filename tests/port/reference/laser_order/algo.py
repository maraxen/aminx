"""Sealed oracle for the laser_order wave.

Loads ``laser_order/order_f64.npz`` from ``AMINX_LASER_ORACLE_DIR``. A missing
file raises ``OracleAbsentError``. When ``oracle_manifest.toml`` records the
new dump, the hash has to match; a present file with no sha entry fails.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

import numpy as np

from port.reference.laser_score.sealed import OracleAbsentError, oracle_root

__all__ = ["OracleAbsentError", "load"]

_RELATIVE = "laser_order/order_f64.npz"


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _manifest_sha(root: Path, relative: str) -> str | None:
  manifest = root / "oracle_manifest.toml"
  if not manifest.is_file():
    return None
  document = tomllib.loads(manifest.read_text(encoding="utf-8"))
  rows = document.get("dump", [])
  if not isinstance(rows, list):
    return None
  for row in rows:
    if not isinstance(row, dict):
      continue
    recorded = str(row.get("path", "")).replace("\\", "/")
    if recorded == relative:
      sha = row.get("sha256")
      return str(sha) if isinstance(sha, str) else None
  return None


def load(precision: str) -> np.lib.npyio.NpzFile:
  """Return the order dump. Only ``f64`` exists; anything else is absent."""
  if precision != "f64":
    msg = f"laser_order dump is f64 only, got {precision}"
    raise OracleAbsentError(msg)
  root = oracle_root()
  path = root / _RELATIVE
  if not root.is_dir() or not path.is_file():
    msg = f"oracle dump absent: {path}"
    raise OracleAbsentError(msg)
  expected = _manifest_sha(root, _RELATIVE)
  if expected is None:
    msg = f"no sha256 entry for {_RELATIVE}"
    raise RuntimeError(msg)
  actual = _file_sha256(path)
  if actual != expected:
    msg = f"sealed npz hash mismatch: {actual} != {expected}"
    raise RuntimeError(msg)
  return np.load(path)
