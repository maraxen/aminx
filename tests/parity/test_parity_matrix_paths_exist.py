"""Verify every parity matrix ``code_paths`` entry resolves on disk.

The manifest (`tests/parity/parity_matrix.json`) is hand-maintained and drifts as
modules move (AC-14). This sweep fails loudly on any stale ``code_paths`` entry
instead of letting a dangling reference sit silently until a human notices.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from aminx.parity.matrix import manifest_path, project_root


def _raw_paths() -> list[dict[str, object]]:
  payload = json.loads(manifest_path().read_text(encoding="utf-8"))
  return list(payload["paths"])


def test_every_code_path_exists() -> None:
  """Every literal or glob entry in ``code_paths`` must resolve under the repo root."""
  root = project_root()
  missing: list[str] = []
  for entry in _raw_paths():
    entry_id = str(entry["id"])
    for code_path in entry["code_paths"]:
      candidate = root / code_path
      if "*" in code_path:
        if not glob.glob(str(candidate)):
          missing.append(f"{entry_id}: {code_path} (glob matched nothing)")
        continue
      if not candidate.exists():
        missing.append(f"{entry_id}: {code_path} (no such file)")
  assert not missing, "Stale parity_matrix.json code_paths:\n" + "\n".join(missing)


def test_autoregressive_sampling_notes_non_discriminating_token_agreement() -> None:
  """`autoregressive-sampling` must flag that token agreement doesn't discriminate."""
  entry = next(e for e in _raw_paths() if e["id"] == "autoregressive-sampling")
  note = str(entry.get("note", ""))
  assert "non-discriminating" in note
  assert "layer_a_sampling" in note


def test_tied_positions_notes_non_discriminating_token_agreement() -> None:
  """`tied-positions-and-multi-state` must flag the same teacher-forced-lane caveat."""
  entry = next(e for e in _raw_paths() if e["id"] == "tied-positions-and-multi-state")
  note = str(entry.get("note", ""))
  assert "non-discriminating" in note
  assert "layer_a_sampling" in note


def test_no_duplicate_path_ids() -> None:
  """Guard against a copy-paste `id` collision hiding a stale entry."""
  ids = [str(e["id"]) for e in _raw_paths()]
  assert len(ids) == len(set(ids))


def test_manifest_path_helper_matches_file_on_disk() -> None:
  """`manifest_path()` should point at the file this test just parsed."""
  assert manifest_path() == Path(__file__).with_name("parity_matrix.json")
