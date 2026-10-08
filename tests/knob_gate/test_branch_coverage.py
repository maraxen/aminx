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

import pytest
from knob_gate._coverage import check_branch_coverage

_HERE = Path(__file__).resolve().parent

#: ``scripts/redsox/run_gate.py`` is the only writer of this variable (line 234), so this test
#: grades a gate run's own outcomes and has nothing to read outside one. Reading it unconditionally
#: made every ordinary suite run red with a bare ``KeyError``, which says nothing about the gate --
#: it only hides whatever else that run found. Skipping is safe in the direction that matters: the
#: gate harness always exports it, so this never skips where it is supposed to grade.
_NEEDS_GATE = pytest.mark.skipif(
  "AMINX_REDSOX_OUTCOMES_READ" not in os.environ,
  reason="grades a redsox gate run's own outcomes.jsonl; AMINX_REDSOX_OUTCOMES_READ is exported "
  "only by scripts/redsox/run_gate.py, so there is nothing to grade outside a gate run",
)


@_NEEDS_GATE
def test_branch_coverage() -> None:
  outcomes = os.environ["AMINX_REDSOX_OUTCOMES_READ"]
  verdict = check_branch_coverage(
    _HERE / "branch_manifest.toml",
    outcomes,
    _HERE / "sidecar_ledger.toml",
  )
  assert verdict == "pass", verdict
