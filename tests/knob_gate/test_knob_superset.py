# ruff: noqa: S101, PT018
"""Redsox knob-superset harness (spec §6.6).

Live rows are everything that is not an exclusion. Empty ``targets`` or
empty ``parity_test_ids`` fail here; T0.4 does not special-case the TODO
skeleton into a pass.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import fields, is_dataclass, make_dataclass
from pathlib import Path
from typing import cast

import pytest
from knob_gate import reference_surfaces
from knob_gate._outcomes import passed_nodeids

from aminx.run.options import LaserOptions, PottsMPNNOptions
from aminx.run.specs import RunSpecification, SamplingSpecification, ScoringSpecification

pytestmark = pytest.mark.redsox_gate

_GATE = Path(__file__).resolve().parent
_ALIAS = _GATE / "alias_map.toml"
_REASONS = frozenset(
  {
    "io_only",
    "device",
    "visualization",
    "training_only",
    "checkpoint_derived",
    "no_op",
    "internal",
    "duplicate",
  },
)

ROWS = cast("list[dict[str, object]]", tomllib.loads(_ALIAS.read_text(encoding="utf-8"))["row"])
REFS = [cls for cls in vars(reference_surfaces).values() if is_dataclass(cls)]
REF_F = {field.name for cls in REFS for field in fields(cls)}
TGT = (
  RunSpecification,
  SamplingSpecification,
  ScoringSpecification,
  PottsMPNNOptions,
  LaserOptions,
)
TGT_F = {field.name for cls in TGT for field in fields(cls)}
LIVE = [row for row in ROWS if row["equivalence"] != "exclusion"]
DEFERRED_IDS = {
  line.strip()
  for line in (_GATE / "DEFERRED_IDS.txt").read_text(encoding="utf-8").splitlines()
  if line.strip()
}
NEW_FIELDS = (
  {field.name for field in fields(PottsMPNNOptions)}
  | {field.name for field in fields(LaserOptions)}
  | {"omit_aa", "omit_aa_per_position", "output_kind"}
)


def test_rows_bijective() -> None:
  refs = [row["ref"] for row in ROWS]
  assert len(refs) == len(set(refs))
  assert set(refs) == REF_F


def test_superset() -> None:
  # Imported here so a missing redsox install fails this test instead of
  # aborting collection of the rest of tests/knob_gate.
  from redsox.checkers.superset import check_superset  # noqa: PLC0415

  # A live row with no targets used to surface here as a bare
  # "IndexError: list index out of range" from the [0] below, which names
  # neither the row nor how many there are. That is the alias skeleton still
  # being unfinished -- the condition this test exists to catch -- so it should
  # read as a work list, not as a crash. Same pass/fail, more signal.
  unmapped = [str(row["ref"]) for row in LIVE if not row.get("targets")]
  assert not unmapped, (
    f"{len(unmapped)} of {len(LIVE)} live rows have empty targets; every row must be "
    f"classified identical/semantic/divergence with a target, or excluded with a "
    f"reason (spec 6.3). First 5: {unmapped[:5]}"
  )

  alias = {str(row["ref"]): cast("list[str]", row["targets"])[0] for row in LIVE}
  filtered = [
    make_dataclass(
      cls.__name__,
      [(field.name, object) for field in fields(cls) if field.name in alias],
    )
    for cls in REFS
  ]
  result = check_superset(TGT, filtered, alias)
  assert result["verdict"] == "SUPERSET-HYPOTHESIS-PASS", result
  for row in LIVE:
    targets = cast("list[str]", row["targets"])
    assert targets and set(targets) <= TGT_F, row


def test_exclusions() -> None:
  for row in ROWS:
    if row["equivalence"] != "exclusion":
      continue
    reason = str(row["reason"])
    targets = row.get("targets") or []
    assert not targets
    assert reason in _REASONS or reason.startswith("deferred:")
    if reason.startswith("deferred:"):
      assert reason[9:] in DEFERRED_IDS


def test_parity_ids_passed() -> None:
  passed = passed_nodeids(os.environ["AMINX_REDSOX_OUTCOMES_READ"])

  # This used to assert per row and die on the first offender with a bare dict
  # dump, which names one row and neither the scale nor the shape of what is
  # left. The three conditions below are the same pass/fail, reported as three
  # work lists: rows with no test at all, rows naming a test that did not pass,
  # and Options fields no passing knob test reaches.
  unwired = [str(row["ref"]) for row in LIVE if not row["parity_test_ids"]]
  assert not unwired, (
    f"{len(unwired)} of {len(LIVE)} live rows have no parity_test_ids; every live "
    f"row must name a knob test that passed in this gate's own outcomes "
    f"(spec 6.3). First 5: {unwired[:5]}"
  )

  not_passed = {
    str(row["ref"]): sorted(set(cast("list[str]", row["parity_test_ids"])) - passed)
    for row in LIVE
    if not set(cast("list[str]", row["parity_test_ids"])) <= passed
  }
  assert not not_passed, (
    f"{len(not_passed)} live rows name a test that did not pass in this gate's "
    f"outcomes ({len(passed)} nodeids passed). An id missing from the outcomes "
    f"file counts as not passed, so check it was collected by gate_ids first. "
    f"First 5: {dict(list(not_passed.items())[:5])}"
  )

  covered = {
    target
    for row in LIVE
    for target in cast("list[str]", row["targets"])
    if any(
      nodeid.split("::")[-1].startswith("test_knob_semantics_") and nodeid in passed
      for nodeid in cast("list[str]", row["parity_test_ids"])
    )
  }
  uncovered = sorted(NEW_FIELDS - covered)
  assert not uncovered, (
    f"{len(uncovered)} of {len(NEW_FIELDS)} new Options fields are not reached by "
    f"any passing test_knob_semantics_* test: {uncovered}"
  )


def test_u1_reachability_present() -> None:
  """U1 is presence-only until redsox grows a behavioural check."""
  from redsox.checkers import reachability  # noqa: PLC0415

  assert callable(reachability.check_reachability)
