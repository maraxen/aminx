"""FAMPNN parity vehicle v3: ``fampnn_parity.py`` under band rule ``floor`` with non-finite, shape and crash accounting.

Same bands, oracle dumps, fixtures and nine negative controls as v2 (``fampnn_parity_v2.py``). What changed is the
verdict logic in ``fampnn_parity.py`` (final review of PR #269): non-finite values and shape mismatches are counted
separately per arm and fail a clean arm; a control trips only on a FINITE over-band diff or a discrete flip;
a crashed control is recorded as ``n_crashed`` and never counts as a trip.

Sidecar: ``fampnn_parity_v3.bth.toml``. Everything else is ``fampnn_parity.py``.
"""

from __future__ import annotations

import sys

from fampnn_parity import main

if __name__ == "__main__":
  if "--band-rule" not in sys.argv:
    sys.argv += ["--band-rule", "floor"]
  raise SystemExit(main())
