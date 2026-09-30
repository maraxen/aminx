# ruff: noqa: S101
"""Grade branch_manifest against the gate's outcome jsonl.

``check_branch_coverage`` groups records by ``(mutant, nodeid)``. The clean
verdict reads only ``mutant is None``. A manifest row R is judged on
``mutant == R.id``, and only on that row's vehicle nodeids. An empty
manifest is ``instrument_invalid``; this test expects ``pass``, so T0.4's
empty manifest fails the gate instead of reading as covered.
"""

from __future__ import annotations

import os
from pathlib import Path

from knob_gate._coverage import check_branch_coverage

_HERE = Path(__file__).resolve().parent


def test_branch_coverage() -> None:
  outcomes = os.environ["AMINX_REDSOX_OUTCOMES_READ"]
  verdict = check_branch_coverage(
    _HERE / "branch_manifest.toml",
    outcomes,
    _HERE / "sidecar_ledger.toml",
  )
  assert verdict == "pass", verdict
