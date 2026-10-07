"""Every committed bathos sidecar must be structurally capable of grading a run.

WHY THIS EXISTS. A sidecar is a PRE-registration: it states, before the run,
which outcome each result maps to. A sidecar that cannot grade is not a weaker
pre-registration, it is none at all -- the run completes, exits 0, prints a
confident payload, and records ``outcome: unknown``. That exact sequence
already happened here (run 2b056576), and the console gave no hint.

Measured 261006 with bathos's own validator over the 90 committed sidecars:
**74 valid, 11 invalid, 5 that do not parse at all.** Nothing in the repo ran
that check, so the 16 accumulated silently across several sprints.

WHAT THIS CHECKS, and deliberately what it does not. The failures bathos finds
split in two:

  * STRUCTURAL -- no ``[outcomes]`` section, an outcome missing ``decision`` or
    ``reasoning``, no residual catch-all, a file that is not valid TOML. These
    need nothing but the file itself, so they are checked here, in the ordinary
    suite, with no bathos or duckdb import.
  * SEMANTIC -- the DuckDB conditions binding against ``[result_schema]``: a
    type mismatch, a parse error, a column the schema never declares. These
    need duckdb and bathos's own binder, so they stay with
    ``bth validate-sidecar`` and are NOT reimplemented here. A second,
    divergent copy of a SQL binder would be worse than no copy.

ON SEVERITY, stated because it is easy to overstate. A sidecar failing the
SEMANTIC half can still grade at run time: bathos binds conditions against the
emitted payload, not against ``[result_schema]``, so ``laser_layers_tolerance``
-- which fails static validation -- really did grade a run ``pass``
(2747efa8). What such a sidecar loses is the ability to be CHECKED BEFORE a
run, which is when a typo'd column is cheap to fix. The STRUCTURAL failures
this file checks are the harder kind: a sidecar with no ``[outcomes]`` section
has nothing to bind at all.

The registry below is asserted EXACT, in both directions, so fixing one without
removing its entry fails just as loudly as a new one appearing.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]

#: Sidecars that fail a STRUCTURAL check today, with why. Pinned so the set
#: cannot grow silently. Most predate this sprint; the scoped ones can only be
#: repaired in the post-merge re-wave (tests/knob_gate/_coverage.py:21-28).
_KNOWN_INVALID: dict[str, str] = {
  ".bth.toml": "repo-root sidecar: not a run sidecar; has no [outcomes]",
  "scripts/benchmarks/bench_temperature_array.py.bth.toml": (
    "not valid TOML: declares ('result_schema','cells','item') twice"
  ),
  "scripts/browser_validation/parity.bth.toml": "no [outcomes] section",
  "scripts/validate/poe_energy_sanity.bth.toml": (
    "not valid TOML: expected '=' after a key at line 22"
  ),
  "scripts/recapture/caliby_recapture.bth.toml": "no [outcomes] section. SCOPED -> re-wave",
  "scripts/ebm/benchmarks/decoy_benchmark.bth.toml": "no [outcomes] section",
  "scripts/ebm/benchmarks/heterogeneous_batch_benchmark.bth.toml": "no [outcomes] section",
  "scripts/ebm/benchmarks/langevin_annealing_benchmark.bth.toml": "no [outcomes] section",
  "scripts/ebm/benchmarks/langevin_benchmark.bth.toml": "no [outcomes] section",
  "scripts/benchmarks/bench_dedup_hetero.py.bth.toml": "outcomes missing 'decision'",
  "scripts/benchmarks/bench_mixed_length.py.bth.toml": "outcomes missing 'decision'",
  "scripts/benchmarks/sprint23_capability_analysis.py.bth.toml": "outcomes missing 'decision'",
  "scripts/recapture/pottsmpnn_to_eqx.bth.toml": "outcomes missing 'decision'. SCOPED -> re-wave",
  "scripts/spikes/dedup_encode_kvn_spike.bth.toml": "outcomes missing 'decision'",
  "scripts/analysis/alias_claim_reachability.bth.toml": (
    "no outcome sets is_residual=true, so a result matching none of its four "
    "branches grades 'unknown'. Unscoped and a one-line fix, but it edits an "
    "already-run PRE-registration, so it belongs to whoever authored it as a "
    "deliberate act rather than as a side effect of adding this guard."
  ),
}

#: Required on every outcome branch. ``condition`` says WHEN the branch fires,
#: ``decision`` says what is then done, ``reasoning`` says why that follows.
#: A branch missing any of the three cannot be acted on without re-deriving it.
_REQUIRED = ("condition", "decision", "reasoning")


def sidecars() -> list[Path]:
  out = [p for p in sorted(_REPO.rglob("*.bth.toml")) if ".venv" not in p.parts]
  if not out:
    msg = f"no sidecars found under {_REPO}; the guard would be vacuous"
    raise AssertionError(msg)
  return out


def structural_problems(path: Path) -> list[str]:
  """Everything wrong with a sidecar that the file alone can reveal."""
  try:
    doc: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
  except tomllib.TOMLDecodeError as exc:
    return [f"not valid TOML: {exc}"]
  except OSError as exc:
    return [f"unreadable: {exc}"]

  outcomes = doc.get("outcomes")
  if not isinstance(outcomes, dict) or not outcomes:
    return ["no [outcomes] section: this sidecar can never grade a run"]

  problems: list[str] = []
  residual = False
  for name, body in sorted(outcomes.items()):
    if not isinstance(body, dict):
      problems.append(f"outcome {name!r} is not a table")
      continue
    for field in _REQUIRED:
      value = body.get(field)
      if not isinstance(value, str) or not value.strip():
        problems.append(f"outcome {name!r} is missing {field!r}")
    if body.get("is_residual") is True:
      residual = True
  if not residual:
    problems.append(
      "no outcome sets is_residual=true: a result matching no branch grades "
      "'unknown' instead of a named refusal",
    )
  return problems


def test_no_sidecar_newly_fails_a_structural_check() -> None:
  """The set of broken sidecars is pinned, and asserted EXACT both ways."""
  found = {
    str(path.relative_to(_REPO)): structural_problems(path)
    for path in sidecars()
  }
  broken = {name: problems for name, problems in found.items() if problems}

  appeared = sorted(set(broken) - set(_KNOWN_INVALID))
  assert not appeared, (
    "these sidecars newly fail a structural check. A sidecar that cannot grade "
    "records outcome 'unknown' at exit 0, which reads exactly like a pass:\n  "
    + "\n  ".join(f"{n}: {'; '.join(broken[n])}" for n in appeared)
  )

  fixed = sorted(set(_KNOWN_INVALID) - set(broken))
  assert not fixed, (
    "these sidecars are listed in _KNOWN_INVALID but now pass. Delete their "
    f"entries so the registry keeps stating what is really broken: {fixed}"
  )


def test_registry_entries_name_real_files_and_give_reasons() -> None:
  """A registry that can name anything is an opt-out, not a record."""
  wrong = []
  for name, reason in sorted(_KNOWN_INVALID.items()):
    if not (_REPO / name).is_file():
      wrong.append(f"{name}: no such file; remove the entry")
    if not reason.strip():
      wrong.append(f"{name}: entry needs a reason")
  assert not wrong, "\n  ".join(wrong)


def test_the_checker_actually_refuses(tmp_path: Path) -> None:
  """Negative controls. A check that only passes is not a check."""
  good = tmp_path / "good.bth.toml"
  good.write_text(
    "[outcomes.pass]\n"
    'condition = "x > 0"\ndecision = "ship"\nreasoning = "because"\n'
    "[outcomes.other]\n"
    'condition = "x <= 0"\ndecision = "stop"\nreasoning = "because"\n'
    "is_residual = true\n",
    encoding="utf-8",
  )
  assert structural_problems(good) == []

  # 1. a file that is not TOML at all
  bad = tmp_path / "bad.bth.toml"
  bad.write_text("[outcomes\nthis is not toml", encoding="utf-8")
  assert any("not valid TOML" in p for p in structural_problems(bad))

  # 2. no [outcomes] section -- the sidecar can never grade
  empty = tmp_path / "empty.bth.toml"
  empty.write_text('[metadata]\ntitle = "x"\n', encoding="utf-8")
  assert structural_problems(empty) == [
    "no [outcomes] section: this sidecar can never grade a run",
  ]

  # 3. each required field, dropped one at a time
  for field in _REQUIRED:
    lines = [
      "[outcomes.only]",
      *[f'{f} = "v"' for f in _REQUIRED if f != field],
      "is_residual = true",
    ]
    partial = tmp_path / f"missing_{field}.bth.toml"
    partial.write_text("\n".join(lines) + "\n", encoding="utf-8")
    problems = structural_problems(partial)
    assert any(f"missing {field!r}" in p for p in problems), (field, problems)

  # 4. a present-but-blank field is not a field
  blank = tmp_path / "blank.bth.toml"
  blank.write_text(
    '[outcomes.only]\ncondition = "x"\ndecision = "   "\nreasoning = "y"\n'
    "is_residual = true\n",
    encoding="utf-8",
  )
  assert any("missing 'decision'" in p for p in structural_problems(blank))

  # 5. no residual catch-all
  no_residual = tmp_path / "no_residual.bth.toml"
  no_residual.write_text(
    '[outcomes.only]\ncondition = "x"\ndecision = "d"\nreasoning = "r"\n',
    encoding="utf-8",
  )
  assert any("is_residual" in p for p in structural_problems(no_residual))

  # 6. an outcome that is not a table at all
  scalar = tmp_path / "scalar.bth.toml"
  scalar.write_text('[outcomes]\npass = "yes"\n', encoding="utf-8")
  assert any("not a table" in p for p in structural_problems(scalar))
