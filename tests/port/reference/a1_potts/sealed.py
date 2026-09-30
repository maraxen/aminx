# REFERENCE: DO NOT MODIFY
"""Shared loader for sealed A1 Potts npz dumps.

Dumps are not in the repo. They live under ``AMINX_A1_ORACLE_DIR`` (default
``~/projects/aminx-oracles/dumps/a1_potts``) and are pinned by ``oracles.sha256``.
A missing directory or file raises ``OracleAbsentError`` so tests can skip.
A hash mismatch raises ``RuntimeError``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

_SHA_FILE = Path(__file__).with_name("oracles.sha256")


class OracleAbsentError(FileNotFoundError):
  """The A1 dump directory or a pinned npz is not on disk."""


def oracle_root() -> Path:
  """Directory named by ``AMINX_A1_ORACLE_DIR``."""
  raw = os.environ.get("AMINX_A1_ORACLE_DIR", "~/projects/aminx-oracles/dumps/a1_potts")
  return Path(raw).expanduser()


def sha_table() -> dict[str, str]:
  """Map ``oracles.sha256`` relative paths to hex digests."""
  table: dict[str, str] = {}
  for line in _SHA_FILE.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if not stripped:
      continue
    digest, name = stripped.split(maxsplit=1)
    table[name.removeprefix("./")] = digest
  return table


def _file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def load_npz(relative: str) -> np.lib.npyio.NpzFile:
  """Load one pinned dump. Missing path skips; a bad hash fails."""
  expected = sha_table().get(relative)
  if expected is None:
    msg = f"no sha256 entry for {relative}"
    raise RuntimeError(msg)
  root = oracle_root()
  path = root / relative
  if not root.is_dir() or not path.is_file():
    msg = f"oracle dump absent: {path}"
    raise OracleAbsentError(msg)
  actual = _file_sha256(path)
  if actual != expected:
    msg = f"sealed npz hash mismatch: {actual} != {expected}"
    raise RuntimeError(msg)
  return np.load(path)
