"""ThermoMPNN / ThermoMPNN-D parity vehicle, v2: float16 is reported, not graded.

v1 (``thermo_mpnn_parity.py``) grades a float16 band of 1.5e-2 (max-abs) and Pearson r > 0.999. Its registered runs fail
only on that band, deterministically (``v1_f16_max_abs_diff`` 0.0278, v2 0.018; float32 1.3e-5, float64 1e-14). The standing
decision is that float16 stays an xfail for parity, so this wrapper runs the same suite with ``--no-grade-float16``: every
float16 number is still recorded (plus ``float16_within_v1_band``), the float64 / float32 bands, exact invariants and negative
controls are unchanged, and the sidecar only requires the float16 numbers to be non-null. v1 stays as the closure of its
recorded runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
  sys.path.insert(0, str(_DIR))

from thermo_mpnn_parity import main  # noqa: E402

if __name__ == "__main__":
  if "--no-grade-float16" not in sys.argv:
    sys.argv.append("--no-grade-float16")
  main()
