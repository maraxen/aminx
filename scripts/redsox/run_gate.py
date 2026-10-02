#!/usr/bin/env python3
"""Drive the redsox gate (spec §6.6 steps 0, 1, 1b, and 2).

Writes ``{rc, step2_passed, n_ids, n_mutant_runs}`` to ``$BTH_RESULTS_PATH``
and exits 0 whenever the tree was graded, including a fail verdict. A missing
results path, a step-0 crash, or an unexpected exception is a harness crash
and exits non-zero. Step 1b's pytest status is not folded into ``rc``.

Invoke as ``bth run --project-slug aminx -- uv run --no-sync python3
scripts/redsox/run_gate.py`` so bathos can see this file's sidecar. A
``bash -c`` argv records ``script_path`` as bash and the outcome stays empty.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tomllib
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from gate_common import NONPORT, declared_wave, nodeid_path, repo_root, target_wave_names

logger = logging.getLogger(__name__)

_UV = ("uv", "run", "--no-sync")


def _parse(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--log",
    type=Path,
    default=None,
    help="log file for pytest output (default: $OUT/run_gate.log)",
  )
  return parser.parse_args(argv)


def _configure_logging(path: Path) -> None:
  """Attach the run log. basicConfig already owns stderr; this adds the file."""
  path.parent.mkdir(parents=True, exist_ok=True)
  handler = logging.FileHandler(path, encoding="utf-8")
  handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
  logging.getLogger().addHandler(handler)


def _require_results_path() -> Path:
  # Outcomes are scored from this file. A previous graded run exited 0 with a
  # pass-looking payload and still recorded outcome=unknown because the JSON
  # never landed on $BTH_RESULTS_PATH.
  raw = os.environ.get("BTH_RESULTS_PATH")
  if not raw:
    logger.error("BTH_RESULTS_PATH is unset; refusing to grade without a results file")
    raise SystemExit(2)
  path = Path(raw)
  path.parent.mkdir(parents=True, exist_ok=True)
  return path


def _gate_out(repo: Path) -> Path:
  override = os.environ.get("AMINX_GATE_OUT")
  if override:
    return Path(override).resolve()
  stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
  return (repo / "outputs" / "gate" / stamp).resolve()


def _child_env(out: Path) -> dict[str, str]:
  env = dict(os.environ)
  env["AMINX_PORT_AUDITS_PATH"] = str(out / "port_audits.jsonl")
  return env


def _run(argv: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> int:
  logger.info("exec %s", " ".join(argv))
  completed = subprocess.run(  # noqa: S603 -- argv is the spec's fixed uv/pytest list
    argv,
    cwd=cwd,
    env=env,
    capture_output=True,
    text=True,
    check=False,
  )
  with log.open("a", encoding="utf-8") as handle:
    handle.write(f"\n$ {' '.join(argv)}\n")
    handle.write(completed.stdout)
    handle.write(completed.stderr)
    handle.write(f"\nexit {completed.returncode}\n")
  logger.info("exit %s (output in %s)", completed.returncode, log)
  return completed.returncode


def _read_lines(path: Path) -> list[str]:
  if not path.is_file():
    msg = f"missing gate list {path}"
    raise RuntimeError(msg)
  return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _count_ids(out: Path, waves: list[str]) -> int:
  seen: set[str] = set()
  for wave in waves:
    seen.update(_read_lines(out / f"ids_{wave}.txt"))
  return len(seen)


def _id_waves(out: Path, waves: list[str]) -> dict[str, str]:
  found: dict[str, str] = {}
  for wave in waves:
    for nodeid in _read_lines(out / f"ids_{wave}.txt"):
      found[nodeid] = wave
  return found


def _pytest(files: list[str]) -> list[str]:
  return [*_UV, "pytest", "-o", "addopts=", *files]


def _step1(
  repo: Path,
  out: Path,
  waves: list[str],
  base_env: dict[str, str],
  log: Path,
) -> int:
  # A reused OUT must not mix an earlier invocation into this grade.
  outcomes = out / "outcomes.jsonl"
  if outcomes.exists():
    outcomes.unlink()
  rc = 0
  for wave in waves:
    files = _read_lines(out / f"files_{wave}.txt")
    # An empty argv would drop pytest onto pyproject testpaths and collect
    # the whole suite, which then looks like a clean run of the wrong tests.
    if not files:
      logger.error("wave %s has no files; refusing an unbounded pytest", wave)
      rc = 1
      continue
    env = dict(base_env)
    env["AMINX_PORT_WAVE"] = wave
    env["AMINX_REDSOX_SELECT"] = str(out / f"ids_{wave}.txt")
    env["AMINX_REDSOX_OUTCOMES"] = str(outcomes)
    env.pop("AMINX_PORT_MUTANT", None)
    code = _run(_pytest(files), cwd=repo, env=env, log=log)
    if code != 0:
      rc = 1
  return rc


def _as_dict(value: object) -> dict[str, object] | None:
  """Narrow a TOML value the way ty accepts ``dict[str, object]``."""
  if not isinstance(value, dict):
    return None
  return cast(dict[str, object], value)


def _branch_rows(repo: Path) -> list[dict[str, object]]:
  path = repo / "tests" / "knob_gate" / "branch_manifest.toml"
  if not path.is_file():
    return []
  data = tomllib.loads(path.read_text(encoding="utf-8"))
  rows = data.get("branch", [])
  if not isinstance(rows, list):
    return []
  typed: list[dict[str, object]] = []
  for row in rows:
    parsed = _as_dict(row)
    if parsed is not None:
      typed.append(parsed)
  return typed


def _step1b(
  repo: Path,
  out: Path,
  waves: list[str],
  base_env: dict[str, str],
  log: Path,
) -> int:
  """One pytest per (row, wave). Status is not folded into ``rc``."""
  id_waves = _id_waves(out, waves)
  runs = 0
  select_root = out / "mutant_select"
  for row in _branch_rows(repo):
    vehicle = _as_dict(row.get("vehicle"))
    if vehicle is None or vehicle.get("kind") != "pytest":
      continue
    row_id = row.get("id")
    nodeids = vehicle.get("nodeids")
    if not isinstance(row_id, str) or not row_id:
      msg = "pytest branch row is missing id"
      raise RuntimeError(msg)
    if not isinstance(nodeids, list) or not nodeids:
      msg = f"pytest branch row {row_id} is missing nodeids"
      raise RuntimeError(msg)
    grouped: dict[str, list[str]] = defaultdict(list)
    for nodeid in nodeids:
      if not isinstance(nodeid, str) or not nodeid:
        msg = f"pytest branch row {row_id} has a non-string nodeid"
        raise RuntimeError(msg)
      wave = id_waves.get(nodeid, declared_wave(repo, nodeid))
      grouped[wave].append(nodeid)
    for wave, group in sorted(grouped.items()):
      select = select_root / f"{row_id}__{wave}.txt"
      select.parent.mkdir(parents=True, exist_ok=True)
      select.write_text("".join(f"{nodeid}\n" for nodeid in group), encoding="utf-8")
      files = sorted({nodeid_path(nodeid) for nodeid in group})
      env = dict(base_env)
      env["AMINX_PORT_WAVE"] = wave
      env["AMINX_PORT_MUTANT"] = row_id
      env["AMINX_REDSOX_SELECT"] = str(select)
      env["AMINX_REDSOX_OUTCOMES"] = str(out / "outcomes.jsonl")
      _run(_pytest(files), cwd=repo, env=env, log=log)
      runs += 1
  return runs


def _step2(repo: Path, out: Path, base_env: dict[str, str], log: Path) -> bool:
  env = dict(base_env)
  # Step 2 reads the jsonl. The outcome hooks stay off only while the wave
  # variable is unset; leaving a parent value in the environment would
  # deselect the harness and append a second copy of the records.
  env.pop("AMINX_PORT_WAVE", None)
  env.pop("AMINX_PORT_MUTANT", None)
  env.pop("AMINX_REDSOX_SELECT", None)
  env.pop("AMINX_REDSOX_OUTCOMES", None)
  env["AMINX_REDSOX_OUTCOMES_READ"] = str(out / "outcomes.jsonl")
  argv = [*_UV, "pytest", "-o", "addopts=", "tests/knob_gate", "-q"]
  return _run(argv, cwd=repo, env=env, log=log) == 0


def _write_results(path: Path, payload: dict[str, object]) -> None:
  path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
  logger.info("wrote %s %s", path, payload)


def main(argv: list[str] | None = None) -> None:
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
  )
  args = _parse(sys.argv[1:] if argv is None else argv)
  results_path = _require_results_path()
  repo = repo_root(Path(__file__))
  out = _gate_out(repo)
  out.mkdir(parents=True, exist_ok=True)
  log_path = args.log.resolve() if args.log is not None else out / "run_gate.log"
  _configure_logging(log_path)
  logger.info("repo %s out %s", repo, out)
  base_env = _child_env(out)
  # Step 0 must see every wave. A leaked AMINX_PORT_WAVE would deselect the rest.
  collect_env = dict(base_env)
  collect_env.pop("AMINX_PORT_WAVE", None)
  collect_env.pop("AMINX_PORT_MUTANT", None)
  collect_env.pop("AMINX_REDSOX_SELECT", None)
  step0 = [
    *_UV,
    "python3",
    "scripts/redsox/gate_ids.py",
    "--out",
    str(out),
  ]
  if _run(step0, cwd=repo, env=collect_env, log=log_path) != 0:
    logger.error("step 0 failed; not grading")
    raise SystemExit(1)
  waves = [NONPORT, *target_wave_names(repo)]
  emitted = sorted(
    path.name.removeprefix("ids_").removesuffix(".txt") for path in out.glob("ids_*.txt")
  )
  # Step 1's loop is exactly these waves. An id filed under any other name
  # would never be executed, and a missing list would make `cat` fail open.
  if emitted != sorted(waves):
    logger.error("id lists %s do not match waves %s", emitted, waves)
    raise SystemExit(1)
  try:
    rc = _step1(repo, out, waves, base_env, log_path)
    n_ids = _count_ids(out, waves)
    n_mutant_runs = _step1b(repo, out, waves, base_env, log_path)
    step2_passed = _step2(repo, out, base_env, log_path)
  except Exception:
    logger.exception("gate failed before it could grade")
    raise SystemExit(1) from None
  _write_results(
    results_path,
    {
      "n_ids": n_ids,
      "n_mutant_runs": n_mutant_runs,
      "rc": rc,
      "step2_passed": step2_passed,
    },
  )


if __name__ == "__main__":
  main()
