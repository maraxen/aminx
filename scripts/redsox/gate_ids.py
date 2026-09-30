#!/usr/bin/env python3
"""Step 0 of the redsox gate: emit per-wave id and file lists.

The union is every alias-map ``parity_test_ids`` entry plus every test
collected under ``tests/knob_semantics``, ``tests/port``, and ``tests/golden``.
``__nonport__`` is exactly the ids outside ``tests/port/**``. Collection uses
``-o addopts=""`` so the pyproject deselect of ``parity_heavy`` cannot shrink
the list to whatever the default marker expression happens to leave.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

import pytest
from gate_common import NONPORT, declared_wave, nodeid_path, repo_root, target_wave_names

logger = logging.getLogger(__name__)

_COLLECT_ROOTS = ("tests/knob_semantics", "tests/port", "tests/golden")


class _Collector:
  """Record ``(wave, nodeid, relative path)`` at collection finish."""

  def __init__(self) -> None:
    self.rows: list[tuple[str, str, str]] = []

  def pytest_collection_finish(self, session: pytest.Session) -> None:
    for item in session.items:
      relative = nodeid_path(item.nodeid)
      mark = item.get_closest_marker("port_wave")
      wave = str(mark.args[0]) if mark is not None and mark.args else NONPORT
      # Path wins over a stray marker: __nonport__ must be exactly the
      # ids outside tests/port, or that wave would load port's conftest.
      if not relative.startswith("tests/port/"):
        wave = NONPORT
      self.rows.append((wave, item.nodeid, relative))


def _parity_ids(repo: Path) -> list[str]:
  path = repo / "tests" / "knob_gate" / "alias_map.toml"
  data = tomllib.loads(path.read_text(encoding="utf-8"))
  rows = data.get("row", [])
  if not isinstance(rows, list):
    return []
  found: list[str] = []
  for row in rows:
    if not isinstance(row, dict):
      continue
    raw_ids = row.get("parity_test_ids", [])
    if not isinstance(raw_ids, list):
      continue
    found.extend(nodeid for nodeid in raw_ids if isinstance(nodeid, str) and nodeid)
  return found


def _collect(repo: Path, extra_files: list[str]) -> list[tuple[str, str, str]]:
  """Collect with addopts cleared. Exit 5 (nothing collected) is not a crash."""
  roots = [root for root in _COLLECT_ROOTS if (repo / root).exists()]
  args = [*roots, *extra_files, "-o", "addopts=", "--collect-only", "-q"]
  collector = _Collector()
  previous = Path.cwd()
  os.chdir(repo)
  try:
    code = int(pytest.main(args, plugins=[collector]))
  finally:
    os.chdir(previous)
  # 5 is pytest's "no tests collected". The gate grades that as n_ids=0.
  # An import or usage error must still abort step 0.
  if code not in (0, 5):
    msg = f"pytest collection failed with exit {code}"
    raise RuntimeError(msg)
  return collector.rows


def _write_lists(
  repo: Path, out: Path, waves: dict[str, list[str]], files_of: dict[str, str]
) -> int:
  names = [NONPORT, *target_wave_names(repo)]
  for wave in sorted(waves):
    if wave not in names:
      names.append(wave)
  total = 0
  seen: set[str] = set()
  for wave in names:
    nodeids = sorted(set(waves.get(wave, [])))
    file_names = sorted({files_of[nodeid] for nodeid in nodeids})
    (out / f"ids_{wave}.txt").write_text(
      "".join(f"{nodeid}\n" for nodeid in nodeids), encoding="utf-8"
    )
    (out / f"files_{wave}.txt").write_text(
      "".join(f"{name}\n" for name in file_names),
      encoding="utf-8",
    )
    for nodeid in nodeids:
      if nodeid not in seen:
        seen.add(nodeid)
        total += 1
    logger.info("wave %s ids=%s files=%s", wave, len(nodeids), len(file_names))
  nonport = (out / f"ids_{NONPORT}.txt").read_text(encoding="utf-8").splitlines()
  leaked = [nodeid for nodeid in nonport if nodeid_path(nodeid).startswith("tests/port/")]
  if leaked:
    msg = f"__nonport__ contains port ids: {leaked[:5]}"
    raise RuntimeError(msg)
  return total


def emit(repo: Path, out: Path) -> int:
  """Write ``ids_<W>.txt`` and ``files_<W>.txt``. Return the unique id count."""
  parity = _parity_ids(repo)
  extra: list[str] = []
  known_roots = tuple(root for root in _COLLECT_ROOTS if (repo / root).exists())
  for nodeid in parity:
    relative = nodeid_path(nodeid)
    if any(relative == root or relative.startswith(root + "/") for root in known_roots):
      continue
    if relative not in extra:
      extra.append(relative)
  collected = _collect(repo, extra)
  files_of: dict[str, str] = {}
  waves: dict[str, list[str]] = defaultdict(list)
  for wave, nodeid, relative in collected:
    if relative.startswith("tests/port/") and wave == NONPORT:
      msg = f"collected port id {nodeid} has no port_wave"
      raise RuntimeError(msg)
    files_of[nodeid] = relative
    waves[wave].append(nodeid)
  for nodeid in parity:
    if any(nodeid in bucket for bucket in waves.values()):
      continue
    relative = nodeid_path(nodeid)
    wave = declared_wave(repo, nodeid)
    files_of[nodeid] = relative
    waves[wave].append(nodeid)
  return _write_lists(repo, out, waves, files_of)


def _parse(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--out", type=Path, required=True, help="directory for ids_<W>.txt and files_<W>.txt"
  )
  parser.add_argument(
    "--log", type=Path, default=None, help="log file (default: <out>/gate_ids.log)"
  )
  return parser.parse_args(argv)


def _configure_logging(path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()],
  )


def main(argv: list[str] | None = None) -> None:
  args = _parse(sys.argv[1:] if argv is None else argv)
  out = args.out.resolve()
  out.mkdir(parents=True, exist_ok=True)
  log_path = args.log.resolve() if args.log is not None else out / "gate_ids.log"
  _configure_logging(log_path)
  repo = repo_root(Path(__file__))
  try:
    count = emit(repo, out)
  except Exception:
    logger.exception("gate_ids failed")
    raise SystemExit(1) from None
  logger.info("unique ids %s", count)


if __name__ == "__main__":
  main()
