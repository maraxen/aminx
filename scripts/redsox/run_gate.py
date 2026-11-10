#!/usr/bin/env python3
"""Drive the redsox gate (spec §6.6 steps 0, 1, 1b, and 2).

Writes ``{rc, step2_passed, n_ids, n_mutant_runs, family, covers_whole_gate}``
to ``$BTH_RESULTS_PATH`` and exits 0 whenever the tree was graded, including a
fail verdict. ``family`` is ``"all"`` or the single family a ``--family`` run
graded; only ``covers_whole_gate = true`` is a whole-gate verdict. A missing
results path, a step-0 crash, or an unexpected exception is a harness crash
and exits non-zero. Step 1b's pytest status is not folded into ``rc``.

Invoke as ``bth run --project-slug aminx -- uv run --no-sync python3
scripts/redsox/run_gate.py`` so bathos can see this file's sidecar. A
``bash -c`` argv records ``script_path`` as bash and the outcome stays empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import tomllib
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
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
  parser.add_argument(
    "--family",
    type=str,
    default=None,
    help="evaluate a specific family shard (e.g. 'potts')",
  )
  parser.add_argument(
    "--all",
    action="store_true",
    help="evaluate all families in aggregate (default)",
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


def _chunk_timeout_s() -> float:
  raw = os.environ.get("AMINX_GATE_CHUNK_TIMEOUT_S")
  if raw:
    return float(raw)
  return 3600.0


def _terminate_pg(pid: int) -> None:
  try:
    os.killpg(pid, signal.SIGTERM)
  except ProcessLookupError:
    return
  except PermissionError:
    return
  try:
    os.killpg(pid, signal.SIGKILL)
  except (ProcessLookupError, PermissionError):
    return


def _run(
  argv: list[str],
  *,
  cwd: Path,
  env: dict[str, str],
  log: Path,
  timeout_s: float | None = None,
) -> int:
  logger.info("exec %s", " ".join(argv))
  limit = _chunk_timeout_s() if timeout_s is None else timeout_s
  proc = subprocess.Popen(  # noqa: S603 -- argv is the spec's fixed uv/pytest list
    argv,
    cwd=cwd,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    start_new_session=True,
  )

  def _forward_signal(signum: int, _frame: FrameType | None) -> None:
    _terminate_pg(proc.pid)
    raise SystemExit(128 + signum)

  old_term = signal.signal(signal.SIGTERM, _forward_signal)
  old_int = signal.signal(signal.SIGINT, _forward_signal)
  try:
    try:
      stdout, stderr = proc.communicate(timeout=limit)
      returncode = proc.returncode
    except subprocess.TimeoutExpired:
      logger.error("chunk timed out after %ss: %s", limit, " ".join(argv))
      _terminate_pg(proc.pid)
      stdout, stderr = proc.communicate()
      returncode = 124
  finally:
    signal.signal(signal.SIGTERM, old_term)
    signal.signal(signal.SIGINT, old_int)

  with log.open("a", encoding="utf-8") as handle:
    handle.write(f"\n$ {' '.join(argv)}\n")
    handle.write(stdout or "")
    handle.write(stderr or "")
    handle.write(f"\nexit {returncode}\n")
  logger.info("exit %s (output in %s)", returncode, log)
  return returncode


def _head_sha(repo: Path) -> str:
  completed = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=repo,
    capture_output=True,
    text=True,
    check=False,
  )
  if completed.returncode == 0:
    return completed.stdout.strip()
  return "unknown"


def _unit_sha256(
  repo: Path,
  select_file: Path,
  files: list[str],
  wave: str,
  mutant: str | None,
) -> str:
  hasher = hashlib.sha256()
  hasher.update(f"head:{_head_sha(repo)}\n".encode())
  hasher.update(f"wave:{wave}\n".encode())
  hasher.update(f"mutant:{mutant or ''}\n".encode())
  hasher.update(b"select:\n")
  if select_file.is_file():
    hasher.update(select_file.read_bytes())
  hasher.update(b"\nfiles:\n")
  for rel in sorted(files):
    hasher.update(f"{rel}:".encode())
    candidate = repo / rel
    if candidate.is_file():
      hasher.update(hashlib.sha256(candidate.read_bytes()).hexdigest().encode())
    else:
      hasher.update(b"missing")
    hasher.update(b"\n")
  return hasher.hexdigest()


def _valid_jsonl_file(path: Path) -> bool:
  if not path.is_file():
    return False
  try:
    text = path.read_text(encoding="utf-8")
  except OSError:
    return False
  for line in text.splitlines():
    if not line.strip():
      continue
    try:
      parsed = json.loads(line)
    except json.JSONDecodeError:
      return False
    if not isinstance(parsed, dict):
      return False
  return True


def _unit_is_reusable(
  stamp_path: Path,
  outcomes_path: Path,
  expected_sha256: str,
  *,
  require_zero_rc: bool,
) -> tuple[bool, int]:
  if not stamp_path.is_file() or not _valid_jsonl_file(outcomes_path):
    return False, 1
  try:
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return False, 1
  if not isinstance(stamp, dict):
    return False, 1
  if stamp.get("sha256") != expected_sha256:
    return False, 1
  rc = stamp.get("rc")
  if not isinstance(rc, int) or isinstance(rc, bool) or rc == 124:
    return False, 1
  if require_zero_rc and rc != 0:
    return False, rc
  return True, rc


def _write_unit_stamp(stamp_path: Path, sha256: str, rc: int) -> None:
  stamp_path.parent.mkdir(parents=True, exist_ok=True)
  tmp = stamp_path.with_suffix(".stamp.tmp")
  payload = {"rc": rc, "sha256": sha256}
  tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
  tmp.replace(stamp_path)


def _assemble_outcomes(out: Path, unit_outcome_files: list[Path]) -> None:
  """Atomically rebuild ``out/outcomes.jsonl`` from completed per-unit files in order."""
  tmp = out / "outcomes.jsonl.tmp"
  target = out / "outcomes.jsonl"
  with tmp.open("w", encoding="utf-8") as dst:
    for unit_file in unit_outcome_files:
      if not unit_file.is_file():
        continue
      content = unit_file.read_text(encoding="utf-8")
      if content:
        dst.write(content)
        if not content.endswith("\n"):
          dst.write("\n")
  tmp.replace(target)


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
  units_dir = out / "units"
  units_dir.mkdir(parents=True, exist_ok=True)
  unit_files: list[Path] = []
  rc = 0
  for wave in waves:
    files = _read_lines(out / f"files_{wave}.txt")
    # An empty argv would drop pytest onto pyproject testpaths and collect
    # the whole suite, which then looks like a clean run of the wrong tests.
    if not files:
      logger.error("wave %s has no files; refusing an unbounded pytest", wave)
      rc = 1
      continue
    select_path = out / f"ids_{wave}.txt"
    unit_id = f"step1__{wave}"
    unit_outcomes = units_dir / f"{unit_id}.outcomes.jsonl"
    unit_stamp = units_dir / f"{unit_id}.stamp.json"
    digest = _unit_sha256(repo, select_path, files, wave, None)
    reusable, _ = _unit_is_reusable(
      unit_stamp,
      unit_outcomes,
      digest,
      require_zero_rc=True,
    )
    if reusable:
      logger.info("reusing completed wave %s (sha256=%s)", wave, digest[:12])
      unit_files.append(unit_outcomes)
      continue
    unit_stamp.unlink(missing_ok=True)
    unit_outcomes.unlink(missing_ok=True)
    env = dict(base_env)
    env["AMINX_PORT_WAVE"] = wave
    env["AMINX_REDSOX_SELECT"] = str(select_path)
    env["AMINX_REDSOX_OUTCOMES"] = str(unit_outcomes)
    env.pop("AMINX_PORT_MUTANT", None)
    code = _run(_pytest(files), cwd=repo, env=env, log=log)
    if not unit_outcomes.exists():
      unit_outcomes.write_text("", encoding="utf-8")
    if _valid_jsonl_file(unit_outcomes) and code != 124:
      _write_unit_stamp(unit_stamp, digest, code)
    if code != 0:
      rc = 1
    unit_files.append(unit_outcomes)
  _assemble_outcomes(out, unit_files)
  return rc


def _as_dict(value: object) -> dict[str, object] | None:
  """Narrow a TOML value the way ty accepts ``dict[str, object]``."""
  if not isinstance(value, dict):
    return None
  return cast("dict[str, object]", value)


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
  family: str | None = None,
) -> int:
  """One pytest per (row, wave). Status is not folded into ``rc``."""
  id_waves = _id_waves(out, waves)
  runs = 0
  select_root = out / "mutant_select"
  units_dir = out / "units"
  units_dir.mkdir(parents=True, exist_ok=True)
  step1_files = [units_dir / f"step1__{wave}.outcomes.jsonl" for wave in waves]
  step1b_files: list[Path] = []
  for row in _branch_rows(repo):
    if family is not None:
      row_family = row.get("family")
      if row_family is not None and row_family != family:
        continue
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
      unit_id = f"step1b__{row_id}__{wave}"
      unit_outcomes = units_dir / f"{unit_id}.outcomes.jsonl"
      unit_stamp = units_dir / f"{unit_id}.stamp.json"
      digest = _unit_sha256(repo, select, files, wave, row_id)
      reusable, _ = _unit_is_reusable(
        unit_stamp,
        unit_outcomes,
        digest,
        require_zero_rc=False,
      )
      if reusable:
        logger.info("reusing completed mutant %s__%s (sha256=%s)", row_id, wave, digest[:12])
        step1b_files.append(unit_outcomes)
        runs += 1
        continue
      unit_stamp.unlink(missing_ok=True)
      unit_outcomes.unlink(missing_ok=True)
      env = dict(base_env)
      env["AMINX_PORT_WAVE"] = wave
      env["AMINX_PORT_MUTANT"] = row_id
      env["AMINX_REDSOX_SELECT"] = str(select)
      env["AMINX_REDSOX_OUTCOMES"] = str(unit_outcomes)
      code = _run(_pytest(files), cwd=repo, env=env, log=log)
      if not unit_outcomes.exists():
        unit_outcomes.write_text("", encoding="utf-8")
      if _valid_jsonl_file(unit_outcomes) and code != 124:
        _write_unit_stamp(unit_stamp, digest, code)
      step1b_files.append(unit_outcomes)
      runs += 1
  _assemble_outcomes(out, [*step1_files, *step1b_files])
  return runs


def _step2(repo: Path, out: Path, base_env: dict[str, str], log: Path, family: str | None = None) -> bool:
  env = dict(base_env)
  # Step 2 reads the jsonl. The outcome hooks stay off only while the wave
  # variable is unset; leaving a parent value in the environment would
  # deselect the harness and append a second copy of the records.
  env.pop("AMINX_PORT_WAVE", None)
  env.pop("AMINX_PORT_MUTANT", None)
  env.pop("AMINX_REDSOX_SELECT", None)
  env.pop("AMINX_REDSOX_OUTCOMES", None)
  if family is not None:
    env["AMINX_REDSOX_FAMILY"] = family
  else:
    env.pop("AMINX_REDSOX_FAMILY", None)
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
  family = None if args.all else args.family
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
    n_mutant_runs = _step1b(repo, out, waves, base_env, log_path, family=family)
    step2_passed = _step2(repo, out, base_env, log_path, family=family)
  except Exception:
    logger.exception("gate failed before it could grade")
    raise SystemExit(1) from None
  # A --family run grades only that family's branch rows and step-2 surfaces. Record the scope in
  # the payload so a pass from a single-family run cannot be read as a whole-gate pass.
  scope = "all" if family is None else family
  if family is not None:
    logger.warning(
      "family-scoped run: this pass covers family %r only, not the whole gate",
      family,
    )
  _write_results(
    results_path,
    {
      "family": scope,
      "covers_whole_gate": family is None,
      "n_ids": n_ids,
      "n_mutant_runs": n_mutant_runs,
      "rc": rc,
      "step2_passed": step2_passed,
    },
  )


if __name__ == "__main__":
  main()
