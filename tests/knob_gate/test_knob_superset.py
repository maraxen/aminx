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

_PLUMBING_ALLOWLIST = _GATE.parent / "lint" / "options_plumbing_allowlist.toml"
_EQUIVALENCES = frozenset(
  {"identical", "semantic", "divergence", "exclusion", "aminx_extension"},
)
_KINDS = frozenset({"semantics", "inert"})
_SEMANTICS_PREFIX = "test_knob_semantics_"
_INERT_PREFIX = "test_knob_inert_"
_AMINX_ONLY_PREFIX = "__aminx_only__"

ROWS = cast("list[dict[str, object]]", tomllib.loads(_ALIAS.read_text(encoding="utf-8"))["row"])
REFS = [cls for cls in vars(reference_surfaces).values() if is_dataclass(cls)]
REF_F = {field.name for cls in REFS for field in fields(cls)}


def _kind(row: dict[str, object]) -> str:
  """``kind`` defaults to ``semantics``; see the alias_map.toml header."""
  return str(row.get("kind", "semantics"))


def _is_upstream(row: dict[str, object]) -> bool:
  """``upstream`` defaults to true; false means no reference surface names it."""
  return bool(row.get("upstream", True))


def _not_plumbed() -> frozenset[str]:
  """Fields the plumbing lint records as inert.

  This is the safeguard behind ``kind = "inert"``: a knob may only be credited
  by an inertness test while the repo separately, explicitly records it as
  not-plumbed debt. ``test_allowlist_has_no_fixed_entries`` deletes a row the
  moment its field becomes plumbed, so the credit cannot outlive the defect.
  """
  raw = tomllib.loads(_PLUMBING_ALLOWLIST.read_text(encoding="utf-8"))
  entries = cast("list[dict[str, object]]", raw.get("entries", []))
  return frozenset(str(entry["field"]) for entry in entries)
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


AMINX_ONLY = {str(row["ref"]) for row in ROWS if not _is_upstream(row)}


def test_rows_bijective() -> None:
  """One row per reference field, PLUS the declared aminx-only rows.

  The bijection used to be ``set(refs) == REF_F`` flat, which made an
  aminx-only Options field structurally unnameable: a row could exist only
  where an upstream analogue did. ``emit_etab`` was the proof that this was a
  gate construction and not a coverage hole -- implemented, tested, and
  invisible. Aminx-only rows are admitted by exemption, not by relaxation: they
  must be spelled ``__aminx_only__<field>`` and must not collide with a real
  reference field, so the exemption cannot quietly absorb a missing upstream
  row.
  """
  refs = [str(row["ref"]) for row in ROWS]
  assert len(refs) == len(set(refs))
  assert set(refs) == REF_F | AMINX_ONLY, {
    "missing_upstream_rows": sorted(REF_F - set(refs)),
    "unexpected_rows": sorted(set(refs) - REF_F - AMINX_ONLY),
  }
  shadowed = sorted(AMINX_ONLY & REF_F)
  assert not shadowed, (
    f"{len(shadowed)} rows are marked upstream = false but name a real "
    f"reference field: {shadowed}. A field with an upstream analogue is not "
    f"aminx-only, and the marker would hide a genuine mapping."
  )
  misnamed = sorted(ref for ref in AMINX_ONLY if not ref.startswith(_AMINX_ONLY_PREFIX))
  assert not misnamed, (
    f"{len(misnamed)} aminx-only refs are not spelled {_AMINX_ONLY_PREFIX}<field>: "
    f"{misnamed}. The convention is what keeps the exemption auditable by eye."
  )


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

  # Upstream rows only. The superset hypothesis maps a reference surface onto
  # the aminx surface, and an aminx-only field has no upstream side to map
  # from -- its ref is not a field of any REF class, so it would be a key in
  # `alias` that `filtered` can never contain.
  alias = {
    str(row["ref"]): cast("list[str]", row["targets"])[0]
    for row in LIVE
    if _is_upstream(row)
  }
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


def test_row_markers_are_declared() -> None:
  """Both optional markers use a declared vocabulary, and inert is safeguarded.

  Four separate conditions, reported separately, because they fail for
  unrelated reasons and a combined message would name the wrong one.
  """
  bad_equivalence = sorted(
    {
      f"{row['ref']}={row['equivalence']!r}"
      for row in ROWS
      if str(row["equivalence"]) not in _EQUIVALENCES
    },
  )
  assert not bad_equivalence, (
    f"{len(bad_equivalence)} rows use an undeclared equivalence; add it to "
    f"_EQUIVALENCES deliberately rather than relying on 'anything that is not "
    f"exclusion is live'. First 5: {bad_equivalence[:5]}"
  )

  bad_kind = sorted(
    {f"{row['ref']}={_kind(row)!r}" for row in ROWS if _kind(row) not in _KINDS},
  )
  assert not bad_kind, (
    f"{len(bad_kind)} rows use an undeclared kind (expected one of "
    f"{sorted(_KINDS)}): {bad_kind[:5]}"
  )

  # An inert row must name ONLY inert tests. A test_knob_semantics_ id on an
  # inert row is the exact false claim the marker exists to avoid.
  mismarked = {
    str(row["ref"]): sorted(
      nodeid
      for nodeid in cast("list[str]", row["parity_test_ids"])
      if not nodeid.split("::")[-1].startswith(_INERT_PREFIX)
    )
    for row in ROWS
    if _kind(row) == "inert" and row.get("parity_test_ids")
  }
  mismarked = {ref: ids for ref, ids in mismarked.items() if ids}
  assert not mismarked, (
    f"{len(mismarked)} inert rows name a test that is not {_INERT_PREFIX}*: "
    f"{mismarked}. An inert knob has no semantics to pin, so crediting a "
    f"semantics test for it would certify something that does not exist."
  )

  # THE SAFEGUARD. Scoped to the fields the gate actually demands coverage for:
  # fixed_mask is inert too but is a RunSpecification field, not an Options
  # field, so it is not in NEW_FIELDS and the Options allowlist cannot list it.
  not_plumbed = _not_plumbed()
  unlisted = sorted(
    {
      target
      for row in ROWS
      if _kind(row) == "inert"
      for target in cast("list[str]", row.get("targets") or [])
      if target in NEW_FIELDS and target not in not_plumbed
    },
  )
  assert not unlisted, (
    f"{len(unlisted)} fields are credited by an inertness test but are NOT "
    f"listed not_plumbed in {_PLUMBING_ALLOWLIST.name}: {unlisted}. Without "
    f"that row the inert credit has nothing policing it -- "
    f"test_allowlist_has_no_fixed_entries is what deletes the credit when the "
    f"knob gets plumbed. Either list the field or stop marking the row inert."
  )


@pytest.mark.skipif(
  "AMINX_REDSOX_OUTCOMES_READ" not in os.environ,
  reason="asserts every live row's parity_test_ids passed in THIS gate's own outcomes.jsonl "
  "(spec 6.3); AMINX_REDSOX_OUTCOMES_READ is exported only by scripts/redsox/run_gate.py, so "
  "outside a gate run there are no outcomes to check the rows against",
)
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

  # A field is covered by a passing test whose prefix matches the row's kind:
  # test_knob_semantics_ pins a knob's SEMANTICS, test_knob_inert_ PROVES it
  # has none. The second is only accepted on a row marked kind = "inert",
  # which test_row_markers_are_declared requires to be allowlisted
  # not_plumbed -- so an inertness credit is always backed by tracked debt
  # that a separate test retires the moment the knob is wired.
  covered = {
    target
    for row in LIVE
    for target in cast("list[str]", row["targets"])
    if any(
      nodeid.split("::")[-1].startswith(
        _INERT_PREFIX if _kind(row) == "inert" else _SEMANTICS_PREFIX,
      )
      and nodeid in passed
      for nodeid in cast("list[str]", row["parity_test_ids"])
    )
  }
  uncovered = sorted(NEW_FIELDS - covered)
  assert not uncovered, (
    f"{len(uncovered)} of {len(NEW_FIELDS)} new Options fields are not reached by "
    f"any passing {_SEMANTICS_PREFIX}* test (or {_INERT_PREFIX}* on a row marked "
    f"kind = 'inert'): {uncovered}"
  )


def test_u1_reachability_present() -> None:
  """U1 is presence-only until redsox grows a behavioural check."""
  from redsox.checkers import reachability  # noqa: PLC0415

  assert callable(reachability.check_reachability)
