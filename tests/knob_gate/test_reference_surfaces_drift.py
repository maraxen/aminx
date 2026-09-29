"""Regenerate the upstream knob surface and fail when the committed file drifted."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_POTTS = Path("/home/marielle/repos/PottsMPNN")
_LASER = Path("/home/marielle/repos/LASErMPNN")
_SCRIPT = _REPO / "scripts/redsox/extract_upstream_knobs.py"
_OUT = _REPO / "tests/knob_gate/reference_surfaces.py"


def test_reference_surfaces_match_pinned_upstreams() -> None:
  """Skip when the pinned checkouts are not on this machine; otherwise --check."""
  missing = [str(path) for path in (_POTTS, _LASER) if not path.is_dir()]
  if missing:
    pytest.skip("upstream roots absent: " + ", ".join(missing))
  completed = subprocess.run(
    [
      sys.executable,
      str(_SCRIPT),
      "--potts-root",
      str(_POTTS),
      "--laser-root",
      str(_LASER),
      "--out",
      str(_OUT),
      "--check",
    ],
    check=False,
    capture_output=True,
    text=True,
  )
  assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
