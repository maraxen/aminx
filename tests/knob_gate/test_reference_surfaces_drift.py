"""Regenerate the upstream knob surface and fail when the committed file drifted."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts/redsox/extract_upstream_knobs.py"
_OUT = _REPO / "tests/knob_gate/reference_surfaces.py"


def _upstream_root(env_var: str, default: str) -> Path:
  """Same resolution every parity vehicle uses (e.g. potts_energy_parity.py:54).

  Hardcoding one user's home made this test skip on every other machine --
  including titanix, which is the host that runs the gate.
  """
  return Path(os.environ.get(env_var, default)).expanduser()


def test_reference_surfaces_match_pinned_upstreams() -> None:
  """Skip when the pinned checkouts are not on this machine; otherwise --check."""
  potts = _upstream_root("AMINX_POTTS_ROOT", "~/repos/PottsMPNN")
  laser = _upstream_root("AMINX_LASER_ROOT", "~/repos/LASErMPNN")
  missing = [str(path) for path in (potts, laser) if not path.is_dir()]
  if missing:
    pytest.skip("upstream roots absent: " + ", ".join(missing))
  completed = subprocess.run(
    [
      sys.executable,
      str(_SCRIPT),
      "--potts-root",
      str(potts),
      "--laser-root",
      str(laser),
      "--out",
      str(_OUT),
      "--check",
    ],
    check=False,
    capture_output=True,
    text=True,
  )
  assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
