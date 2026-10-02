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
  for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    raw = json.loads(line)
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
