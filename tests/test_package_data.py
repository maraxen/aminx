"""Packaged data files must be real objects, not Git LFS pointers (debt #3500).

The 0.5.3 wheel shipped LASErMPNN's ``ideal_geometry.npz`` as a 130-byte LFS pointer. This test
checks every file declared in ``[tool.setuptools.package-data]`` in the source tree CI builds from
(CI checks out with ``lfs: true``), with the same checker the release workflow runs on the built
wheel, plus a positive control that the checker flags a pointer.

task_id: 261118_aminx-campaign-teacher-forced-scoring
"""

from __future__ import annotations

import importlib.util
import io
import tomllib
import zipfile
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _checker():  # noqa: ANN202
  spec = importlib.util.spec_from_file_location("check_wheel_data", _ROOT / "scripts/release/check_wheel_data.py")
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _declared_data_files() -> list[Path]:
  config = tomllib.loads((_ROOT / "pyproject.toml").read_text())
  patterns = config["tool"]["setuptools"]["package-data"]["aminx"]
  files: list[Path] = []
  for pattern in patterns:
    files.extend(sorted((_ROOT / "src/aminx").glob(pattern)))
  return files


def test_declared_package_data_are_real_readable_files() -> None:
  checker = _checker()
  files = _declared_data_files()
  assert any(f.name == "ideal_geometry.npz" for f in files)
  failures = [r for f in files if (r := checker.check_bytes(str(f), f.read_bytes())) is not None]
  assert not failures, failures


def test_checker_flags_a_pointer_and_accepts_a_real_npz(tmp_path: Path) -> None:
  """Positive control: the checker must fail on exactly the 0.5.3 defect."""
  checker = _checker()
  pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 11282\n"
  assert checker.check_bytes("aminx/families/laser_mpnn/ideal_geometry.npz", pointer) is not None
  buf = io.BytesIO()
  np.savez(buf, x=np.arange(3))
  assert checker.check_bytes("aminx/x.npz", buf.getvalue()) is None

  wheel = tmp_path / "aminx-0.0.0-py3-none-any.whl"
  with zipfile.ZipFile(wheel, "w") as zf:
    zf.writestr("aminx/__init__.py", "")
    zf.writestr("aminx/families/laser_mpnn/ideal_geometry.npz", pointer)
  assert checker.check([wheel])


if __name__ == "__main__":  # pragma: no cover
  pytest.main([__file__, "-v"])
