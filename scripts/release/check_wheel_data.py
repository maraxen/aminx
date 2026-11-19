"""Refuse a wheel or source tree whose packaged data files are Git LFS pointers or unreadable.

The 0.5.3 PyPI wheel shipped ``aminx/families/laser_mpnn/ideal_geometry.npz`` as a 130-byte LFS
pointer: the release workflow checked out without ``lfs: true``, so ``uv build`` packaged the
pointer and ``runner.sample(model_family="lasermpnn")`` could not build coordinates (debt #3500).
This check runs on the built wheel before publishing and, through
``tests/test_package_data.py``, on the source tree in CI.

Every non-Python file under ``aminx/`` is checked: it must not start with the LFS pointer header;
``.npz`` / ``.npy`` must load with ``allow_pickle=False``; ``.json`` must parse.

Usage: ``python scripts/release/check_wheel_data.py dist/aminx-*.whl [more wheels or dirs]``.
Exit 0 when every file passes, 1 otherwise.

task_id: 261118_aminx-campaign-teacher-forced-scoring
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger("check_wheel_data")

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_CODE_SUFFIXES = {".py", ".pyc", ".pyi", ".typed"}


def check_bytes(name: str, data: bytes) -> str | None:
  """Return a failure reason for one packaged file, or ``None`` if it is a real, readable file."""
  if data.startswith(LFS_POINTER_PREFIX):
    return f"{name}: Git LFS pointer ({len(data)} bytes), not the object"
  suffix = Path(name).suffix
  try:
    if suffix in {".npz", ".npy"}:
      import numpy as np  # noqa: PLC0415

      loaded = np.load(io.BytesIO(data), allow_pickle=False)
      if suffix == ".npz":
        _ = [loaded[key].shape for key in loaded.files]
    elif suffix == ".json":
      json.loads(data)
  except Exception as exc:  # noqa: BLE001 -- any failure to read is a packaging defect
    return f"{name}: unreadable {suffix} ({type(exc).__name__}: {exc})"
  return None


def _wheel_files(path: Path) -> Iterator[tuple[str, bytes]]:
  with zipfile.ZipFile(path) as wheel:
    for name in wheel.namelist():
      if name.startswith("aminx/") and not name.endswith("/") and Path(name).suffix not in _CODE_SUFFIXES:
        yield name, wheel.read(name)


def _tree_files(root: Path) -> Iterator[tuple[str, bytes]]:
  for file in sorted(root.rglob("*")):
    if file.is_file() and file.suffix not in _CODE_SUFFIXES and "__pycache__" not in file.parts:
      yield str(file.relative_to(root.parent)), file.read_bytes()


def check(paths: list[Path]) -> list[str]:
  """Check every packaged data file in each wheel (``.whl``) or package directory."""
  failures: list[str] = []
  for path in paths:
    files = _wheel_files(path) if path.suffix == ".whl" else _tree_files(path)
    n = 0
    for name, data in files:
      n += 1
      if (reason := check_bytes(name, data)) is not None:
        failures.append(reason)
    logger.info("%s: %d data files checked", path, n)
    if n == 0:
      failures.append(f"{path}: no packaged data files found (wrong path?)")
  return failures


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("paths", nargs="+", type=Path)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  failures = check(args.paths)
  for failure in failures:
    logger.error(failure)
  sys.exit(1 if failures else 0)


if __name__ == "__main__":
  main()
