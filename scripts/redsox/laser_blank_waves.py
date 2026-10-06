#!/usr/bin/env python3
"""Run the two LASEr port waves that the gate has never exercised.

``laser_layers`` and ``laser_encoder`` contribute zero assertions to every gate
run so far. Measured 261006: their ``sealed.py`` defaults ``AMINX_LASER_ORACLE_DIR``
to ``~/projects/aminx-oracles/dumps/laser``, which does not exist on the gate
host, while ``laser_score`` and ``laser_decode_step`` default to
``dumps/b3_laser_owned``, which does. The dumps are present and sealed; only the
default path is stale. See ``.praxia/docs/plans/261005_rewave-composition-scoped-batch.md``
section 1a-i and debt #2494.

This runs exactly those two waves with the oracle directory pointed at the real
dumps, and grades whether their parity assertions hold at the bands their target
files already declare. It changes no scoped file: ``scripts/redsox/`` is not in
``_SCOPED_PREFIXES`` (``tests/knob_gate/_coverage.py:21``), so no ledger row is
invalidated by this script existing or running.

WHAT IS AND IS NOT NEW EVIDENCE. That each wave collects 63 pairs is already
measured and is used here only as a precondition -- if it does not hold, the
environment is wrong and nothing is graded. The finding is whether the collected
cases PASS, which no run has ever established for either wave.

Resumable: each wave writes its own report plus a completion stamp, and a rerun
skips a wave whose stamp matches the current (wave, oracle dir, commit) key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WAVES = ("laser_layers", "laser_encoder")
_EXPECTED_PAIRS = 63  # 3 checkpoints x 21 fixtures, shared with laser_score
_BOGUS_ORACLE = "/nonexistent/aminx-oracles/dumps/__absent__"
_WAVE_TIMEOUT_S = 3600


class _Refused(RuntimeError):
  """The instrument refuses to grade."""


def _repo_root() -> Path:
  here = Path(__file__).resolve()
  for parent in here.parents:
    if (parent / "pyproject.toml").is_file():
      return parent
  msg = "no pyproject.toml above this script"
  raise _Refused(msg)


def _parse_junit(text: str) -> dict[str, Any]:
  """Counts and failing ids from a junit xml document.

  Counts come from the <testsuite> attributes, which pytest always writes.
  ``collected`` is ``tests``; a wave whose parameter set is empty reports a
  single skipped placeholder per test function, so ``collected > 0`` is NOT by
  itself evidence that the oracle resolved -- that is what ``n_pairs`` is for.
  """
  root = ET.fromstring(text)  # noqa: S314 - pytest's own output, not untrusted input
  suite = root if root.tag == "testsuite" else root.find("testsuite")
  if suite is None:
    msg = "junit xml has no <testsuite> element"
    raise _Refused(msg)
  get = lambda name: int(suite.get(name, "0"))  # noqa: E731
  collected = get("tests")
  errors, failures, skipped = get("errors"), get("failures"), get("skipped")
  failing: list[str] = []
  for case in suite.iter("testcase"):
    bad = [child for child in case if child.tag in {"failure", "error"}]
    if bad:
      name = f"{case.get('classname', '')}::{case.get('name', '')}"
      failing.append(name)
  return {
    "collected": collected,
    "failed": failures + errors,
    "skipped": skipped,
    "passed": collected - failures - errors - skipped,
    "failing_ids": sorted(failing)[:40],
  }


def _self_test() -> None:
  """Synthetic checks, three of which must REFUSE. Runs before pytest is touched."""
  ok = """<testsuite name="pytest" errors="0" failures="0" skipped="0" tests="4">
    <testcase classname="t" name="a"/><testcase classname="t" name="b"/>
    <testcase classname="t" name="c"/><testcase classname="t" name="d"/>
  </testsuite>"""
  got = _parse_junit(ok)
  assert got["collected"] == 4, got
  assert got["passed"] == 4, got
  assert got["failed"] == 0, got

  # A failure must be counted AND named. A parser that counts but cannot name
  # the case is useless for deciding what to do about it.
  bad = """<testsuite name="pytest" errors="0" failures="1" skipped="0" tests="2">
    <testcase classname="t" name="a"/>
    <testcase classname="t" name="b"><failure message="x">boom</failure></testcase>
  </testsuite>"""
  got = _parse_junit(bad)
  assert got["failed"] == 1, got
  assert got["passed"] == 1, got
  assert got["failing_ids"] == ["t::b"], got

  # An ERROR is a failure too. Counting only <failure> would grade a wave that
  # blew up in setup as clean -- which is this whole investigation's failure mode.
  err = """<testsuite name="pytest" errors="1" failures="0" skipped="0" tests="1">
    <testcase classname="t" name="a"><error message="x">boom</error></testcase>
  </testsuite>"""
  got = _parse_junit(err)
  assert got["failed"] == 1, got
  assert got["failing_ids"] == ["t::a"], got

  # The blank-wave shape: all skipped, nothing failed. Must NOT read as healthy.
  blank = """<testsuite name="pytest" errors="0" failures="0" skipped="4" tests="4">
    <testcase classname="t" name="a"><skipped/></testcase>
    <testcase classname="t" name="b"><skipped/></testcase>
    <testcase classname="t" name="c"><skipped/></testcase>
    <testcase classname="t" name="d"><skipped/></testcase>
  </testsuite>"""
  got = _parse_junit(blank)
  assert got["failed"] == 0, got
  assert got["passed"] == 0, got
  assert got["skipped"] == 4, got

  # --- three that must refuse ---
  for payload, label in (
    ("<notasuite/>", "no testsuite element"),
    ("<root><other/></root>", "wrong child"),
  ):
    try:
      _parse_junit(payload)
    except _Refused:
      pass
    else:  # pragma: no cover
      msg = f"parser accepted {label}"
      raise AssertionError(msg)
  try:
    _parse_junit("not xml at all")
  except ET.ParseError:
    pass
  else:  # pragma: no cover
    msg = "parser accepted non-xml"
    raise AssertionError(msg)

  logger.info("self-test passed (9 checks, 3 refusals)")


def _n_pairs(repo: Path, wave: str, oracle_dir: str) -> int:
  """Pairs the wave's own loader resolves. 0 means the oracle is unreachable."""
  code = (
    "import sys; sys.path.append('tests')\n"
    f"from port.reference.{wave}.sealed import load_npz, OracleAbsentError\n"
    "try:\n"
    f"    d = load_npz('{wave}/oracle_f64.npz')\n"
    "    print(len(d['checkpoint_ids']) * len(d['fixture_names']))\n"
    "except OracleAbsentError:\n"
    "    print(0)\n"
  )
  env = {**os.environ, "AMINX_LASER_ORACLE_DIR": oracle_dir}
  proc = subprocess.run(  # noqa: S603
    ["uv", "run", "--no-sync", "python3", "-c", code],  # noqa: S607
    cwd=repo,
    env=env,
    capture_output=True,
    text=True,
    timeout=600,
    check=False,
  )
  if proc.returncode != 0:
    msg = f"pair probe failed for {wave}: {proc.stderr[-600:]}"
    raise _Refused(msg)
  return int(proc.stdout.strip().splitlines()[-1])


def _run_wave(repo: Path, wave: str, oracle_dir: str, work: Path) -> dict[str, Any]:
  """One pytest session for one wave. ``-o addopts=''`` defeats parity_heavy."""
  xml = work / f"{wave}.junit.xml"
  env = {
    **os.environ,
    "AMINX_LASER_ORACLE_DIR": oracle_dir,
    "AMINX_PORT_WAVE": wave,
  }
  runner = ["uv", "run", "--frozen", "--extra=dev", "pytest"]
  argv = [
    *runner,
    "-o",
    "addopts=",
    f"tests/port/test_{wave}.py",
    f"--junitxml={xml}",
    "-q",
    "--tb=line",
  ]
  logger.info("running wave %s", wave)
  proc = subprocess.run(  # noqa: S603
    argv, cwd=repo, env=env, capture_output=True, text=True,
    timeout=_WAVE_TIMEOUT_S, check=False,
  )
  if not xml.is_file():
    msg = f"{wave}: pytest wrote no junit xml (rc={proc.returncode}): {proc.stderr[-600:]}"
    raise _Refused(msg)
  report = _parse_junit(xml.read_text(encoding="utf-8"))
  report["returncode"] = proc.returncode
  report["wave"] = wave
  return report


def _key(wave: str, oracle_dir: str, commit: str) -> str:
  raw = f"{wave}\0{oracle_dir}\0{commit}".encode()
  return hashlib.sha256(raw).hexdigest()


def _commit(repo: Path) -> str:
  proc = subprocess.run(  # noqa: S603
    ["git", "rev-parse", "HEAD"],  # noqa: S607
    cwd=repo, capture_output=True, text=True, check=False,
  )
  return proc.stdout.strip() or "unknown"


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--work-dir", type=Path, required=False)
  parser.add_argument("--out", type=Path, required=False)
  parser.add_argument(
    "--oracle-dir",
    default=str(Path("~/projects/aminx-oracles/dumps/b3_laser_owned").expanduser()),
  )
  parser.add_argument("--self-test-only", action="store_true")
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  _self_test()
  if args.self_test_only:
    return 0

  if args.work_dir is None:
    parser.error("--work-dir is required unless --self-test-only")
  repo = _repo_root()
  work: Path = args.work_dir
  work.mkdir(parents=True, exist_ok=True)
  commit = _commit(repo)

  # NEGATIVE CONTROL, first: with a bogus oracle dir the loader must resolve ZERO
  # pairs. Without this, "63 pairs" is just a number the probe printed; with it,
  # the probe is shown to discriminate reachable from unreachable.
  control_zero = {w: _n_pairs(repo, w, _BOGUS_ORACLE) for w in _WAVES}
  live_pairs = {w: _n_pairs(repo, w, args.oracle_dir) for w in _WAVES}
  controls_detected = all(v == 0 for v in control_zero.values()) and all(
    v == _EXPECTED_PAIRS for v in live_pairs.values()
  )
  logger.info("control(bogus)=%s live=%s detected=%s", control_zero, live_pairs, controls_detected)

  reports: dict[str, Any] = {}
  if controls_detected:
    for wave in _WAVES:
      stamp = work / f"{_key(wave, args.oracle_dir, commit)}.done"
      body = work / f"{_key(wave, args.oracle_dir, commit)}.json"
      if stamp.is_file() and body.is_file():
        logger.info("reusing completed wave %s", wave)
        reports[wave] = {**json.loads(body.read_text(encoding="utf-8")), "reused": True}
        continue
      report = _run_wave(repo, wave, args.oracle_dir, work)
      body.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
      stamp.write_text(commit + "\n", encoding="utf-8")  # body first, stamp second
      reports[wave] = {**report, "reused": False}

  n_failed = sum(int(r.get("failed", 0)) for r in reports.values())
  n_collected = sum(int(r.get("collected", 0)) for r in reports.values())
  n_passed = sum(int(r.get("passed", 0)) for r in reports.values())
  result = {
    "controls_detected": controls_detected,
    "control_pairs_bogus_dir": control_zero,
    "live_pairs": live_pairs,
    # Flat scalars as well as the maps: the sidecar's condition language cannot
    # index into an object, so every field a criterion names must be a scalar.
    "live_pairs_laser_layers": live_pairs["laser_layers"],
    "live_pairs_laser_encoder": live_pairs["laser_encoder"],
    "control_pairs_laser_layers": control_zero["laser_layers"],
    "control_pairs_laser_encoder": control_zero["laser_encoder"],
    "n_collected": n_collected,
    "n_passed": n_passed,
    "n_failed": n_failed,
    "per_wave": reports,
    "oracle_dir": args.oracle_dir,
    "commit": commit,
  }

  payload = json.dumps(result, indent=2) + "\n"
  if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")
  # Bathos grades from $BTH_RESULTS_PATH; a bare --out records outcome 'unknown'.
  bth_results = os.environ.get("BTH_RESULTS_PATH")
  if bth_results:
    Path(bth_results).write_text(payload, encoding="utf-8")
    logger.info("emitted result to BTH_RESULTS_PATH=%s", bth_results)
  else:
    logger.warning("BTH_RESULTS_PATH unset; this run will NOT be graded by its sidecar")
  sys.stdout.write(payload)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
