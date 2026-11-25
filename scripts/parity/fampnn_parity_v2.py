"""FAMPNN parity vehicle v2: ``fampnn_parity.py`` under band rule ``floor``.

The first vehicle (sidecar ``fampnn_parity.bth.toml``, registered run recorded FAIL on one held-out float64 loop check)
graded the loop stages against a flat 2e-6 band. The diagnosis (``fampnn_diag_floor.py``,
``dump_fampnn_oracle_f64tables.py``) showed that band had no principled value: the deviation is the rounding of
torch's float32 cos/sin tables, amplified by the loop. v2 therefore grades

  identity arm  aminx fed upstream's own torch float32 tables, every stage, 1e-9 absolute (observed ~1e-12);
  shipped arm   aminx's correctly rounded tables, loop-stage bands = the sealed upstream's own sensitivity to the
                same tables (|sealed f64 - f64-table oracle|, pooled per checkpoint and stage), 1e-7 floor;
  the parse/enc/scn1 float64 bands (1e-7), the float32 arm and the nine negative controls are unchanged.

Sidecar: ``fampnn_parity_v2.bth.toml``. Everything else is ``fampnn_parity.py``.
"""

from __future__ import annotations

import sys

from fampnn_parity import main

if __name__ == "__main__":
  if "--band-rule" not in sys.argv:
    sys.argv += ["--band-rule", "floor"]
  raise SystemExit(main())
