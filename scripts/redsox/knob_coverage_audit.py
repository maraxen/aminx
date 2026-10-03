#!/usr/bin/env python3
"""Classify WHY each new Options field is or is not reachable by the gate's ``uncovered`` check.

``test_parity_ids_passed`` (``tests/knob_gate/test_knob_superset.py:148-156``) fails
the gate when any field in ``NEW_FIELDS`` is not reached by a passing
``test_knob_semantics_*`` nodeid named on a live alias row. The assertion prints
the field names. It does not say WHY each one is unreached, and the three
mechanisms have completely different remedies:

  NO_REFERENCE_FIELD  No row names the field, live or exclusion. ``test_rows_bijective``
                      pins ``{row.ref} == REF_F``, the set of UPSTREAM reference-surface
                      field names, so a row can only exist for a field that has an
                      upstream analogue. An aminx-only knob therefore cannot be named
                      by any row, and no amount of test-writing will cover it. The
                      remedy is a scope decision (narrow NEW_FIELDS, or add an
                      aminx-only row kind), never a test.

  PREFIX_MISMATCH     A live row names the field and names a test, but that test's
                      function name does not begin with ``test_knob_semantics_``.
                      Renaming would satisfy the gate, so this is the one mechanism
                      where a green gate is one edit away -- which is exactly why it
                      must be reported separately rather than folded in. Whether the
                      rename is honest depends on whether the test pins knob SEMANTICS
                      or merely records that the knob is inert.

  MISSING_TEST        A live row names the field and lists no test at all. This is the
                      only mechanism that is genuine test debt.

So the question this script answers is the one that decides whether the gate
blocker is a test backlog or a decision backlog. Those are not interchangeable:
a test backlog is work, a decision backlog is a question, and reporting one as
the other either invents busywork or silently parks the gate.

NOTE ON PROVENANCE: an exploratory probe of the prefixed-id condition alone was run
in-conversation before this script existed, and reported 3 fields. That probe produced
no classification and no control, and none of its numbers are cited; this run is the
tracked measurement.

THE NEGATIVE CONTROL IS THE POINT. ``n_missing_test == 0`` is the expected result, and
a classifier that can only ever return zero would produce it whether or not it works.
So ``--control`` injects a synthetic field carrying a live row with an empty
``parity_test_ids`` and asserts the classifier labels it MISSING_TEST. If the control
does not fire, the zero is not evidence of anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger("knob_coverage_audit")

_PREFIX = "test_knob_semantics_"

NO_REFERENCE_FIELD = "NO_REFERENCE_FIELD"
PREFIX_MISMATCH = "PREFIX_MISMATCH"
MISSING_TEST = "MISSING_TEST"
EXCLUDED_ONLY = "EXCLUDED_ONLY"
COVERABLE = "COVERABLE"

# A synthetic reference field name that cannot collide with a real upstream one.
_CONTROL_REF = "__synthetic_control_ref__"
_CONTROL_TARGET = "__synthetic_control_field__"


@dataclass(frozen=True)
class Verdict:
  """Why one field is or is not reachable by the gate's ``uncovered`` condition."""

  field: str
  mechanism: str
  live_rows: int
  all_rows: int
  prefixed_ids: tuple[str, ...]
  other_ids: tuple[str, ...]


def classify(field: str, rows: list[dict[str, Any]]) -> Verdict:
  """Classify one field against the alias rows.

  Mirrors the gate's own reading of the alias map: a row is live when its
  ``equivalence`` is not ``exclusion``, and only live rows can contribute
  coverage.
  """
  naming = [r for r in rows if field in (r.get("targets") or [])]
  live = [r for r in naming if r["equivalence"] != "exclusion"]
  ids = [i for r in live for i in (r.get("parity_test_ids") or [])]
  prefixed = tuple(sorted({i for i in ids if i.split("::")[-1].startswith(_PREFIX)}))
  other = tuple(sorted({i for i in ids if i.split("::")[-1].startswith(_PREFIX) is False}))

  if prefixed:
    mechanism = COVERABLE
  elif not naming:
    mechanism = NO_REFERENCE_FIELD
  elif not live:
    mechanism = EXCLUDED_ONLY
  elif other:
    mechanism = PREFIX_MISMATCH
  else:
    mechanism = MISSING_TEST

  return Verdict(
    field=field,
    mechanism=mechanism,
    live_rows=len(live),
    all_rows=len(naming),
    prefixed_ids=prefixed,
    other_ids=other,
  )


def new_fields() -> set[str]:
  """``NEW_FIELDS`` exactly as ``test_knob_superset`` builds it."""
  from aminx.run.options import LaserOptions, PottsMPNNOptions  # noqa: PLC0415

  return (
    {f.name for f in fields(PottsMPNNOptions)}
    | {f.name for f in fields(LaserOptions)}
    | {"omit_aa", "omit_aa_per_position", "output_kind"}
  )


def run_control(rows: list[dict[str, Any]]) -> bool:
  """Inject a field that IS genuine test debt and check the classifier says so.

  The synthetic row is live and names the synthetic field with no tests, which is
  the MISSING_TEST shape. A classifier that cannot return MISSING_TEST would pass
  the real audit vacuously.
  """
  spiked = [
    *rows,
    {
      "ref": _CONTROL_REF,
      "targets": [_CONTROL_TARGET],
      "equivalence": "identical",
      "parity_test_ids": [],
    },
  ]
  verdict = classify(_CONTROL_TARGET, spiked)
  logger.info("control: synthetic field classified %s", verdict.mechanism)
  if verdict.mechanism != MISSING_TEST:
    logger.error("control FAILED: expected %s, got %s", MISSING_TEST, verdict.mechanism)
    return False

  # And the control must NOT be detected when the row names a prefixed test --
  # otherwise the classifier returns MISSING_TEST indiscriminately.
  spiked[-1]["parity_test_ids"] = [f"tests/knob_semantics/x.py::{_PREFIX}synthetic"]
  inverse = classify(_CONTROL_TARGET, spiked)
  logger.info("control inverse: with a prefixed test it classified %s", inverse.mechanism)
  return inverse.mechanism == COVERABLE


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--alias-map",
    type=Path,
    default=Path("tests/knob_gate/alias_map.toml"),
    help="the gate's alias map",
  )
  parser.add_argument("--out", type=Path, required=True, help="results JSON")
  parser.add_argument(
    "--control",
    action="store_true",
    help="run the negative control (required for the result to mean anything)",
  )
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

  rows = cast(
    "list[dict[str, Any]]",
    tomllib.loads(args.alias_map.read_text(encoding="utf-8"))["row"],
  )
  logger.info("alias map: %d rows (%d live)", len(rows), sum(1 for r in rows if r["equivalence"] != "exclusion"))

  control_detected = run_control(rows) if args.control else None
  if args.control and not control_detected:
    logger.error("the negative control did not fire; the audit below is not evidence")

  verdicts = [classify(f, rows) for f in sorted(new_fields())]
  by_mechanism: dict[str, list[str]] = {}
  for v in verdicts:
    by_mechanism.setdefault(v.mechanism, []).append(v.field)

  unreached = [v for v in verdicts if v.mechanism != COVERABLE]
  for v in unreached:
    logger.info(
      "%-18s %s (live rows=%d, all rows=%d, other ids=%s)",
      v.mechanism, v.field, v.live_rows, v.all_rows, list(v.other_ids),
    )

  result = {
    "n_new_fields": len(verdicts),
    "n_coverable": len(by_mechanism.get(COVERABLE, [])),
    "n_unreached": len(unreached),
    "n_missing_test": len(by_mechanism.get(MISSING_TEST, [])),
    "n_no_reference_field": len(by_mechanism.get(NO_REFERENCE_FIELD, [])),
    "n_prefix_mismatch": len(by_mechanism.get(PREFIX_MISMATCH, [])),
    "n_excluded_only": len(by_mechanism.get(EXCLUDED_ONLY, [])),
    "control_detected": control_detected,
    "by_mechanism": {k: sorted(v) for k, v in sorted(by_mechanism.items())},
    "unreached": [
      {
        "field": v.field,
        "mechanism": v.mechanism,
        "live_rows": v.live_rows,
        "all_rows": v.all_rows,
        "other_ids": list(v.other_ids),
      }
      for v in unreached
    ],
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  logger.info(
    "%d of %d fields reachable; %d unreached (%d genuine test debt); control=%s -> %s",
    result["n_coverable"], result["n_new_fields"], result["n_unreached"],
    result["n_missing_test"], control_detected, args.out,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
