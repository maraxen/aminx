#!/usr/bin/env python3
"""Measure whether the gate's ``uncovered`` blocker is closed, and whether it closed honestly.

REVISION 2. Revision 1 asked a different question and its answer has expired.

Revision 1 asked whether the blocker was test debt or decision debt, and
pre-registered ``pass`` as ``control_detected AND n_missing_test = 0 AND
n_unreached > 0``. It measured 3 unreached fields, zero of them missing a test
(bathos fa2fac73), so the blocker was entirely structural and went to the user
as a scope decision. Commit ``1d11963c`` implemented that decision, which makes
revision 1's ``n_unreached > 0`` clause FALSE by construction. A run under those
criteria could now only report ``inconclusive``-by-accident or a vacuous miss,
so this is a new pre-registration with new criteria rather than an edited
outcome on the old one. Revision 1's record stands as it was written; nothing
in it is retracted.

WHAT REVISION 2 MEASURES

Two numbers, and the second is the one that matters.

  n_unreached == 0            Every field in ``NEW_FIELDS`` is reached by a
                              correctly-prefixed test named on a live row.
                              This is the blocker being closed.

  n_inert_unsafeguarded == 0  Every field credited by an INERTNESS test is also
                              recorded ``not_plumbed`` in the plumbing lint
                              allowlist. This is the blocker having closed
                              honestly.

The second exists because the first is cheap to fake. ``kind = "inert"`` lets a
knob be covered by a test proving it reaches nothing, which is the correct thing
to record for a field like ``emit_dense_hJ`` -- declared and read nowhere -- and
is also one short step from "mark it inert and move on". The safeguard is that
an inert credit must be backed by an allowlist row that
``test_allowlist_has_no_fixed_entries`` DELETES the moment the field becomes
plumbed. So the credit cannot outlive the defect: wire the knob and the
allowlist row goes, the credit goes with it, and the inertness test fires. A
run that reports ``n_unreached = 0`` without checking that would be reporting
the easy half.

WHAT IT DOES NOT MEASURE, STATED PLAINLY

This script reads the alias map. It does not run pytest. The gate's own
condition is stricter in one specific way: it requires the named nodeid to
appear in ``passed_nodeids(AMINX_REDSOX_OUTCOMES_READ)``, this run's own
outcomes file, which exists only after a gate run. So ``n_unreached = 0`` here
is NECESSARY but NOT SUFFICIENT for ``test_parity_ids_passed`` to pass: it says
every field is named by a correctly-prefixed test, not that every such test
passed. The gate run is the complement, and the two together are the claim.
Revision 1 had the identical limitation and did not say so; naming it is part
of this revision.

THE MECHANISMS, AND ONE RENAME

  COVERABLE        A live row naming the field names a test whose function name
                   begins with the prefix THAT ROW'S OWN ``kind`` demands.

  KIND_MISMATCH    NEW IN REVISION 2, and the mechanism that would appear if
                   the two prefixes were being accepted interchangeably: a live
                   row names a test prefixed for the OTHER kind. A semantics row
                   credited by an inertness test would certify semantics for a
                   knob proven to have none; an inert row credited by a
                   semantics test is the false claim the marker exists to avoid.

  PREFIX_MISMATCH  A live row names a test prefixed for neither kind. Renaming
                   satisfies the gate, so it stays reported separately: whether
                   the rename is honest depends on what the test actually
                   proves.

  MISSING_TEST     A live row names the field and lists no test at all. Still
                   the only mechanism that is genuine test debt.

  NO_ROW           No row names the field, live or exclusion. RENAMED from
                   revision 1's NO_REFERENCE_FIELD, because the remedy changed:
                   ``test_rows_bijective`` used to pin ``{row.ref} == REF_F``
                   flat, so a field with no upstream analogue was structurally
                   unnameable and the only fix was a scope decision. Since
                   ``1d11963c`` an aminx-only field CAN be named, by a row
                   spelled ``__aminx_only__<field>`` with ``upstream = false``.
                   So this is now ordinary work -- add the row -- and the old
                   name would assert a dead end that no longer exists.

  EXCLUDED_ONLY    Only exclusion rows name it; exclusions cannot cover.

THE CONTROLS ARE THREE GROUPS AND ALL OF THEM MUST FIRE

``n_unreached = 0`` is the expected result, and an instrument that can only ever
return zero would produce it whether or not it works. Each group is a synthetic
row that the classifier MUST label a particular way:

  1. missing_test        A live row with no ids must be MISSING_TEST, and the
                         same row with a correctly-prefixed id must be
                         COVERABLE. Inherited from revision 1.

  2. kind_discriminated  THE SHARP ONE. A ``semantics`` row naming only an
                         inertness test must NOT be COVERABLE, and an ``inert``
                         row naming only a semantics test must NOT be COVERABLE.
                         Both must be KIND_MISMATCH. If either returns
                         COVERABLE the classifier is unioning the two prefixes,
                         which is precisely the cheap way to drive
                         ``n_unreached`` to zero, and every count below would be
                         worthless.

  3. inert_safeguarded   An inert row whose target is absent from the allowlist
                         must count as unsafeguarded, and the same row with the
                         target present must not. The allowlist set is a
                         parameter for exactly this reason.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger("knob_coverage_audit")

_SEMANTICS_PREFIX = "test_knob_semantics_"
_INERT_PREFIX = "test_knob_inert_"

NO_ROW = "NO_ROW"
KIND_MISMATCH = "KIND_MISMATCH"
PREFIX_MISMATCH = "PREFIX_MISMATCH"
MISSING_TEST = "MISSING_TEST"
EXCLUDED_ONLY = "EXCLUDED_ONLY"
COVERABLE = "COVERABLE"

# Synthetic names that cannot collide with a real upstream field or Options field.
_CONTROL_REF = "__synthetic_control_ref__"
_CONTROL_TARGET = "__synthetic_control_field__"


def _kind(row: dict[str, Any]) -> str:
  """``kind`` defaults to ``semantics``, exactly as the gate reads it."""
  return str(row.get("kind", "semantics"))


def _expected_prefix(row: dict[str, Any]) -> str:
  """The prefix THIS row's kind demands. The whole point of revision 2."""
  return _INERT_PREFIX if _kind(row) == "inert" else _SEMANTICS_PREFIX


def _func(nodeid: str) -> str:
  return nodeid.split("::")[-1]


@dataclass(frozen=True)
class Verdict:
  """Why one field is or is not reachable, and whether its credit is safeguarded."""

  field: str
  mechanism: str
  live_rows: int
  all_rows: int
  kinds: tuple[str, ...]
  matched_ids: tuple[str, ...]
  wrong_kind_ids: tuple[str, ...]
  unprefixed_ids: tuple[str, ...]
  inert_credited: bool
  inert_unsafeguarded: bool


def classify(field: str, rows: list[dict[str, Any]], not_plumbed: frozenset[str]) -> Verdict:
  """Classify one field against the alias rows and the plumbing allowlist.

  Mirrors the gate's reading on every point that matters: a row is live when its
  ``equivalence`` is not ``exclusion``; only live rows contribute coverage; the
  accepted prefix is chosen per row from that row's own ``kind``; and the
  safeguard check ranges over ALL rows naming the field, not only live ones,
  because ``test_row_markers_are_declared`` does the same.
  """
  naming = [r for r in rows if field in (r.get("targets") or [])]
  live = [r for r in naming if r["equivalence"] != "exclusion"]

  matched: set[str] = set()
  wrong_kind: set[str] = set()
  unprefixed: set[str] = set()
  credited_inert = False
  for row in live:
    want = _expected_prefix(row)
    other = _SEMANTICS_PREFIX if want == _INERT_PREFIX else _INERT_PREFIX
    for nodeid in cast("list[str]", row.get("parity_test_ids") or []):
      name = _func(nodeid)
      if name.startswith(want):
        matched.add(nodeid)
        credited_inert = credited_inert or _kind(row) == "inert"
      elif name.startswith(other):
        wrong_kind.add(nodeid)
      else:
        unprefixed.add(nodeid)

  if matched:
    mechanism = COVERABLE
  elif not naming:
    mechanism = NO_ROW
  elif not live:
    mechanism = EXCLUDED_ONLY
  elif wrong_kind:
    mechanism = KIND_MISMATCH
  elif unprefixed:
    mechanism = PREFIX_MISMATCH
  else:
    mechanism = MISSING_TEST

  # The safeguard. Any row marking this field inert obliges the allowlist to
  # record it as not_plumbed; the gate asserts exactly this, over all rows.
  marked_inert = any(_kind(row) == "inert" for row in naming)

  return Verdict(
    field=field,
    mechanism=mechanism,
    live_rows=len(live),
    all_rows=len(naming),
    kinds=tuple(sorted({_kind(row) for row in live})),
    matched_ids=tuple(sorted(matched)),
    wrong_kind_ids=tuple(sorted(wrong_kind)),
    unprefixed_ids=tuple(sorted(unprefixed)),
    inert_credited=credited_inert,
    inert_unsafeguarded=marked_inert and field not in not_plumbed,
  )


def new_fields() -> set[str]:
  """``NEW_FIELDS`` exactly as ``test_knob_superset`` builds it."""
  from aminx.run.options import LaserOptions, PottsMPNNOptions  # noqa: PLC0415

  return (
    {f.name for f in fields(PottsMPNNOptions)}
    | {f.name for f in fields(LaserOptions)}
    | {"omit_aa", "omit_aa_per_position", "output_kind"}
  )


def read_not_plumbed(path: Path) -> frozenset[str]:
  """Fields the plumbing lint records as inert, as the gate's ``_not_plumbed`` reads them."""
  raw = tomllib.loads(path.read_text(encoding="utf-8"))
  entries = cast("list[dict[str, Any]]", raw.get("entries", []))
  return frozenset(str(entry["field"]) for entry in entries)


def _spike(rows: list[dict[str, Any]], **over: Any) -> list[dict[str, Any]]:
  row: dict[str, Any] = {
    "ref": _CONTROL_REF,
    "targets": [_CONTROL_TARGET],
    "equivalence": "identical",
    "parity_test_ids": [],
  }
  row.update(over)
  return [*rows, row]


def run_controls(rows: list[dict[str, Any]]) -> dict[str, bool]:
  """Three groups of synthetic rows, each of which must be labelled one exact way.

  Returns the per-check results so a failure names which direction broke rather
  than collapsing to a bare False.
  """
  empty = frozenset[str]()
  sem = f"tests/knob_semantics/x.py::{_SEMANTICS_PREFIX}synthetic"
  inert = f"tests/knob_semantics/x.py::{_INERT_PREFIX}synthetic"

  def mech(**over: Any) -> str:
    return classify(_CONTROL_TARGET, _spike(rows, **over), empty).mechanism

  checks = {
    # 1. missing_test, both directions (revision 1's control).
    "no_ids_is_missing_test": mech() == MISSING_TEST,
    "semantics_id_is_coverable": mech(parity_test_ids=[sem]) == COVERABLE,
    # 2. THE SHARP ONE: the prefixes are discriminated, not unioned.
    "semantics_row_rejects_inert_test": mech(parity_test_ids=[inert]) == KIND_MISMATCH,
    "inert_row_rejects_semantics_test": mech(kind="inert", parity_test_ids=[sem])
    == KIND_MISMATCH,
    "inert_row_accepts_inert_test": mech(kind="inert", parity_test_ids=[inert]) == COVERABLE,
    "unprefixed_id_is_prefix_mismatch": mech(
      parity_test_ids=["tests/knob_semantics/x.py::test_something_else"],
    )
    == PREFIX_MISMATCH,
    # 3. the safeguard fires, and only when the allowlist is missing the field.
    "inert_without_allowlist_is_flagged": classify(
      _CONTROL_TARGET,
      _spike(rows, kind="inert", parity_test_ids=[inert]),
      empty,
    ).inert_unsafeguarded,
    "inert_with_allowlist_is_not_flagged": not classify(
      _CONTROL_TARGET,
      _spike(rows, kind="inert", parity_test_ids=[inert]),
      frozenset({_CONTROL_TARGET}),
    ).inert_unsafeguarded,
  }
  for name, ok in checks.items():
    logger.info("control %-36s %s", name, "OK" if ok else "FAILED")
  return checks


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--alias-map",
    type=Path,
    default=Path("tests/knob_gate/alias_map.toml"),
    help="the gate's alias map",
  )
  parser.add_argument(
    "--allowlist",
    type=Path,
    default=Path("tests/lint/options_plumbing_allowlist.toml"),
    help="the plumbing lint allowlist that safeguards every inert credit",
  )
  parser.add_argument("--out", type=Path, required=True, help="results JSON")
  parser.add_argument(
    "--control",
    action="store_true",
    help="run the controls (required for the counts to mean anything)",
  )
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

  rows = cast(
    "list[dict[str, Any]]",
    tomllib.loads(args.alias_map.read_text(encoding="utf-8"))["row"],
  )
  not_plumbed = read_not_plumbed(args.allowlist)
  logger.info(
    "alias map: %d rows (%d live, %d inert, %d aminx-only); allowlist: %d not_plumbed",
    len(rows),
    sum(1 for r in rows if r["equivalence"] != "exclusion"),
    sum(1 for r in rows if _kind(r) == "inert"),
    sum(1 for r in rows if not bool(r.get("upstream", True))),
    len(not_plumbed),
  )

  controls = run_controls(rows) if args.control else {}
  controls_detected = all(controls.values()) if args.control else None
  if args.control and not controls_detected:
    logger.error(
      "controls that FAILED: %s -- the counts below are not evidence",
      sorted(name for name, ok in controls.items() if not ok),
    )

  verdicts = [classify(f, rows, not_plumbed) for f in sorted(new_fields())]
  by_mechanism: dict[str, list[str]] = {}
  for v in verdicts:
    by_mechanism.setdefault(v.mechanism, []).append(v.field)

  unreached = [v for v in verdicts if v.mechanism != COVERABLE]
  for v in unreached:
    logger.info(
      "%-16s %s (live=%d all=%d kinds=%s wrong_kind=%s unprefixed=%s)",
      v.mechanism, v.field, v.live_rows, v.all_rows,
      list(v.kinds), list(v.wrong_kind_ids), list(v.unprefixed_ids),
    )
  inert = [v for v in verdicts if v.inert_credited or v.inert_unsafeguarded]
  for v in inert:
    logger.info(
      "inert credit %s: safeguarded=%s ids=%s",
      v.field, not v.inert_unsafeguarded, list(v.matched_ids),
    )

  result: dict[str, Any] = {
    "n_new_fields": len(verdicts),
    "n_coverable": len(by_mechanism.get(COVERABLE, [])),
    "n_unreached": len(unreached),
    "n_missing_test": len(by_mechanism.get(MISSING_TEST, [])),
    "n_kind_mismatch": len(by_mechanism.get(KIND_MISMATCH, [])),
    "n_prefix_mismatch": len(by_mechanism.get(PREFIX_MISMATCH, [])),
    "n_no_row": len(by_mechanism.get(NO_ROW, [])),
    "n_excluded_only": len(by_mechanism.get(EXCLUDED_ONLY, [])),
    "n_inert_credited": sum(1 for v in verdicts if v.inert_credited),
    "n_inert_unsafeguarded": sum(1 for v in verdicts if v.inert_unsafeguarded),
    "n_aminx_only_rows": sum(1 for r in rows if not bool(r.get("upstream", True))),
    "controls_detected": controls_detected,
    "controls": controls,
    "by_mechanism": {k: sorted(v) for k, v in sorted(by_mechanism.items())},
    "unreached": [
      {
        "field": v.field,
        "mechanism": v.mechanism,
        "live_rows": v.live_rows,
        "all_rows": v.all_rows,
        "kinds": list(v.kinds),
        "wrong_kind_ids": list(v.wrong_kind_ids),
        "unprefixed_ids": list(v.unprefixed_ids),
      }
      for v in unreached
    ],
    "inert": [
      {
        "field": v.field,
        "credited": v.inert_credited,
        "unsafeguarded": v.inert_unsafeguarded,
        "matched_ids": list(v.matched_ids),
      }
      for v in inert
    ],
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  payload = json.dumps(result, indent=2) + "\n"
  args.out.write_text(payload, encoding="utf-8")

  # ALSO emit to $BTH_RESULTS_PATH, which is how bathos actually finds a result.
  # Revision 2's first run (94e0bfde) recorded outcome 'unknown' with
  # output_paths = [] despite every count being present in --out: bathos reads
  # the emission from $BTH_RESULTS_PATH, else an adjacent
  # <stem>.bth-results.json, else a single path REGISTERED via `bth run
  # --output-file` (runner.py:_read_result_emission). A bare `--out` is none of
  # those, so the sidecar's own [outcomes] conditions were never evaluated --
  # an ungraded run that still exits 0, which is the shape the verify-by-record
  # rule exists to catch. Writing both makes the grading independent of
  # remembering a flag at the call site.
  bth_results = os.environ.get("BTH_RESULTS_PATH")
  if bth_results:
    Path(bth_results).parent.mkdir(parents=True, exist_ok=True)
    Path(bth_results).write_text(payload, encoding="utf-8")
    logger.info("emitted result to BTH_RESULTS_PATH=%s", bth_results)
  else:
    logger.warning(
      "BTH_RESULTS_PATH is unset; this run will not be graded by its sidecar "
      "unless the path is registered with `bth run --output-file`",
    )

  logger.info(
    "%d of %d fields reachable; %d unreached (%d test debt, %d kind mismatch); "
    "%d inert credits, %d unsafeguarded; controls=%s -> %s",
    result["n_coverable"], result["n_new_fields"], result["n_unreached"],
    result["n_missing_test"], result["n_kind_mismatch"],
    result["n_inert_credited"], result["n_inert_unsafeguarded"],
    controls_detected, args.out,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
