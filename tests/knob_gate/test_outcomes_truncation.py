# ruff: noqa: S101
"""A truncated outcomes file must refuse to grade, and say why.

``conftest`` appends one JSON object per line as tests run, so an interrupted
run -- a kill, a crash, a full disk -- leaves a half-written final line. Every
reader of that file then died inside ``json/decoder.py`` with
``JSONDecodeError: Unterminated string``, naming neither the file nor the cause.

FAILING CLOSED IS CORRECT HERE AND IS NOT CHANGED. Skipping the bad line would
drop whatever record it held, and that record may well be a PASS -- turning a
disk-full event into a gate failure that reads like a port defect. So the
refusal stands; only the message improves.

Seen 261003: titanix hit a per-user disk quota mid-run and left exactly this.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knob_gate._outcomes import passed_nodeids

_RECORD = {
  "nodeid": "tests/knob_semantics/test_x.py::test_y",
  "wave": "__nonport__",
  "when": "call",
  "outcome": "passed",
  "wasxfail": False,
  "mutant": None,
  "exc_type": None,
}


def test_truncated_line_refuses_to_grade(tmp_path: Path) -> None:
  """The error names the file, the line, the likely cause and the action."""
  path = tmp_path / "outcomes.jsonl"
  path.write_text(
    json.dumps(_RECORD) + "\n" + json.dumps(_RECORD)[:45], encoding="utf-8",
  )
  with pytest.raises(ValueError) as caught:
    passed_nodeids(path)
  message = str(caught.value)
  assert "outcomes.jsonl" in message, message
  assert ":2 " in message, f"must name the offending line: {message}"
  assert "disk filled" in message, message
  assert "Re-run the wave" in message, message


def test_whole_file_of_good_records_still_grades(tmp_path: Path) -> None:
  """The guard must not make well-formed files fail.

  The nodeid is deliberately one that does not exist, so this asserts the file
  PARSES rather than that a particular id passed -- `declared_wave` reads the
  real tree and would not resolve it.
  """
  path = tmp_path / "outcomes.jsonl"
  path.write_text(json.dumps(_RECORD) + "\n", encoding="utf-8")
  assert isinstance(passed_nodeids(path), set)


def test_blank_lines_are_not_corruption(tmp_path: Path) -> None:
  """Empty and whitespace-only lines are skipped, as before."""
  path = tmp_path / "outcomes.jsonl"
  path.write_text(json.dumps(_RECORD) + "\n\n   \n", encoding="utf-8")
  assert isinstance(passed_nodeids(path), set)


def test_missing_file_is_still_an_empty_set(tmp_path: Path) -> None:
  """Absent is not corrupt. This path is how a run with no outcomes reads."""
  assert passed_nodeids(tmp_path / "nope.jsonl") == set()
