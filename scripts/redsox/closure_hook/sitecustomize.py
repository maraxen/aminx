"""Record which repo files a vehicle process actually loaded.

Why this exists. ``tests/knob_gate/_closure.py`` decides a ledger row is still valid
from a STATIC import closure plus reviewed declarations (``closure_edges.toml``). A
declaration is a claim; this hook is how a real run checks it. ``verify_wave.py``
compares the file written here with the row's closure, so a vehicle that loads a file
the closure missed is reported rather than trusted.

It is a ``sitecustomize`` rather than code in each vehicle on purpose: a vehicle is a
scoped file in its own closure, so editing it would invalidate the row it is meant to
protect. ``launch_wave.sh`` puts this directory on ``PYTHONPATH`` and sets:

    AMINX_CLOSURE_OUT     where to write the JSON
    AMINX_CLOSURE_SCRIPT  basename of the vehicle script (``potts_ar_decode.py``)
    AMINX_CLOSURE_REPO    the checkout root; only files under it are recorded

It does nothing unless all three are set, writes only from the process whose
``sys.argv[0]`` is the vehicle (a vehicle that spawns an oracle interpreter inherits
the environment, and that child must not overwrite the file), and can never fail the
run: every error is swallowed, because a recording problem must not cost hours of
measurement.
"""

from __future__ import annotations

import atexit
import json
import os
import sys


def _record() -> None:
  try:
    out = os.environ["AMINX_CLOSURE_OUT"]
    script = os.environ["AMINX_CLOSURE_SCRIPT"]
    root = os.path.realpath(os.environ["AMINX_CLOSURE_REPO"])
    if not sys.argv or os.path.basename(sys.argv[0]) != script:
      return
    files: set[str] = set()
    # The script itself is added explicitly. Measured 261008 (titanix, python 3.13):
    # walking sys.modules here did not yield the vehicle script although
    # __main__.__file__ was set while it ran. The cause was not investigated.
    candidates: list[str | None] = [sys.argv[0]]
    candidates.extend(getattr(m, "__file__", None) for m in list(sys.modules.values()))
    for path in candidates:
      if not path:
        continue
      real = os.path.realpath(path)
      if real.startswith(root + os.sep):
        files.add(os.path.relpath(real, root).replace(os.sep, "/"))
    tmp = f"{out}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
      json.dump({"script": script, "loaded": sorted(files)}, handle, indent=1)
    os.replace(tmp, out)
  except Exception:  # noqa: BLE001, S110 - recording must never fail a run
    pass


if all(os.environ.get(k) for k in ("AMINX_CLOSURE_OUT", "AMINX_CLOSURE_SCRIPT", "AMINX_CLOSURE_REPO")):
  atexit.register(_record)
