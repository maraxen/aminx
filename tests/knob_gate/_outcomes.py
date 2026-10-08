"""Read redsox outcome jsonl the way spec §6.6 defines a passed nodeid.

Only ``mutant is None`` records count. An id passes when its call phase
passed under the wave the test declares, and none of those clean records
failed or were xfail. ``skipped`` does not count as a pass.
"""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from types import ModuleType


def _gate_common() -> ModuleType:
  path = Path(__file__).resolve().parents[2] / "scripts" / "redsox" / "gate_common.py"
  spec = importlib.util.spec_from_file_location("aminx_gate_common", path)
  if spec is None or spec.loader is None:
    msg = f"unable to load {path}"
    raise RuntimeError(msg)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def passed_nodeids(outcomes_path: str | Path) -> set[str]:
  """Nodeids whose clean (mutant is None) records passed under the declared wave."""
  path = Path(outcomes_path)
  if not path.is_file():
    return set()
  common = _gate_common()
  repo = common.repo_root(Path(__file__))
  grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
  for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    if not line.strip():
      continue
    try:
      raw = json.loads(line)
    except json.JSONDecodeError as exc:
      # FAIL CLOSED, LOUDLY. A truncated record means the evidence is
      # incomplete, and skipping it would drop an id that may well have passed
      # -- turning a disk-full event into a manufactured gate failure that
      # looks like a port defect. So this still refuses to grade; it just says
      # which file and line, and why, instead of a bare JSONDecodeError.
      #
      # Seen 261003: titanix hit a per-user disk quota mid-run and conftest's
      # appender left a half-written line, after which every reader of this
      # file died pointing at json/decoder.py.
      msg = (
        f"{path}:{number} is not valid JSON, so the outcomes are incomplete and "
        f"must not be graded. A partial final line usually means the writing run "
        f"was interrupted or the disk filled. Re-run the wave rather than "
        f"deleting the line. ({exc})"
      )
      raise ValueError(msg) from exc
    if not isinstance(raw, dict):
      continue
    mutant = raw.get("mutant")
    if mutant not in (None, ""):
      continue
    nodeid = raw.get("nodeid")
    if not isinstance(nodeid, str) or not nodeid:
      continue
    grouped[nodeid].append(raw)
  passed: set[str] = set()
  for nodeid, rows in grouped.items():
    if any(bool(row.get("wasxfail")) for row in rows):
      continue
    if any(row.get("outcome") in {"failed", "error"} for row in rows):
      continue
    declared = common.declared_wave(repo, nodeid)
    call_passed = any(
      row.get("when") == "call" and row.get("outcome") == "passed" and row.get("wave") == declared
      for row in rows
    )
    if call_passed:
      passed.add(nodeid)
  return passed
