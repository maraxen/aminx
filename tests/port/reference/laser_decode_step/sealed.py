# REFERENCE: DO NOT MODIFY
"""Loader for the sealed laser_decode_step npz dumps.

Dumps are not in the repo. They live under ``AMINX_LASER_ORACLE_DIR`` (default
``~/projects/aminx-oracles/dumps/b3_laser_owned``) as
``laser_decode_step/oracle_{f64,f32}.npz``. A missing file raises
``OracleAbsentError`` so the wave skips. When ``oracle_manifest.toml`` sits
next to the dumps, the file hash must match it.
"""

from __future__ import annotations

import hashlib
import os
import tomllib
from pathlib import Path

import numpy as np


class OracleAbsentError(FileNotFoundError):
  """The laser dump directory or a pinned npz is not on disk."""


def oracle_root() -> Path:
  """Directory named by ``AMINX_LASER_ORACLE_DIR``."""
  raw = os.environ.get(
    "AMINX_LASER_ORACLE_DIR",
    "~/projects/aminx-oracles/dumps/b3_laser_owned",
  )
  return Path(raw).expanduser()


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
    path = str(row.get("path", ""))
    if path.replace("\\", "/") == relative:
      sha = row.get("sha256")
      return str(sha) if isinstance(sha, str) else None
  return None


def load_npz(relative: str) -> np.lib.npyio.NpzFile:
  """Load one dump. Missing path skips; a manifest hash mismatch fails."""
  root = oracle_root()
  path = root / relative
  if not root.is_dir() or not path.is_file():
    msg = f"oracle dump absent: {path}"
    raise OracleAbsentError(msg)
  expected = _manifest_sha(root, relative)
  if expected is not None:
    actual = _file_sha256(path)
    if actual != expected:
      msg = f"sealed npz hash mismatch: {actual} != {expected}"
      raise RuntimeError(msg)
  return np.load(path)
