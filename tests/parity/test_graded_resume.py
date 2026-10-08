# ruff: noqa: S101
"""Resume cache for the graded Potts scripts. No torch, JAX, or checkpoints."""

from __future__ import annotations

import importlib.util
import io
import logging
import re
import sys
from pathlib import Path
from types import ModuleType

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "parity" / "graded_resume.py"


def _load() -> ModuleType:
  spec = importlib.util.spec_from_file_location("graded_resume_under_test", _SCRIPT)
  if spec is None or spec.loader is None:
    msg = f"cannot load {_SCRIPT}"
    raise RuntimeError(msg)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


resume = _load()

_ATTEMPT = re.compile(r"attempt (\d+) units reused (\d+) units remaining (\d+)")


def test_completed_unit_is_reused_on_second_pass(tmp_path: Path) -> None:
  """A stamped unit is skipped; the fake compute runs once."""
  cache = tmp_path / "units"
  units = resume.build_units(
    ["a", "b"],
    arm_id="clean",
    checkpoint_sha256="checkpoint",
    input_sha256="input",
    script_sha256="script",
  )
  calls: list[str] = []

  def compute(unit: resume.Unit) -> dict[str, object]:
    calls.append(unit.unit_id)
    return {"unit_id": unit.unit_id, "n": len(calls)}

  _first, first = resume.run_units(units, compute, cache, resume=True)
  _second, second = resume.run_units(units, compute, cache, resume=True)
  assert first.n_units == 2
  assert first.n_computed == 2
  assert first.n_reused == 0
  assert second.n_units == 2
  assert second.n_reused == 2
  assert second.n_computed == 0
  assert calls == ["a", "b"]


def test_script_sha256_or_arm_id_invalidates_cache(tmp_path: Path) -> None:
  """A clean-arm stamp is not a hit for another script hash or another arm."""
  cache = tmp_path / "units"
  calls: list[str] = []

  def compute(unit: resume.Unit) -> dict[str, str]:
    calls.append(f"{unit.arm_id}:{unit.script_sha256}")
    return {"unit_id": unit.unit_id}

  clean = resume.build_units(
    ["pdb"],
    arm_id="clean",
    checkpoint_sha256="checkpoint",
    input_sha256="input",
    script_sha256="script-a",
  )
  resume.run_units(clean, compute, cache, resume=True)
  other_script = resume.build_units(
    ["pdb"],
    arm_id="clean",
    checkpoint_sha256="checkpoint",
    input_sha256="input",
    script_sha256="script-b",
  )
  _payloads, script_stats = resume.run_units(other_script, compute, cache, resume=True)
  assert script_stats.n_reused == 0
  assert script_stats.n_computed == 1
  mutant = resume.build_units(
    ["pdb"],
    arm_id="skip_transpose_merge_pair",
    checkpoint_sha256="checkpoint",
    input_sha256="input",
    script_sha256="script-a",
  )
  _payloads, arm_stats = resume.run_units(mutant, compute, cache, resume=True)
  assert arm_stats.n_reused == 0
  assert arm_stats.n_computed == 1
  _payloads, again = resume.run_units(clean, compute, cache, resume=True)
  assert again.n_reused == 1
  assert again.n_computed == 0
  assert calls == [
    "clean:script-a",
    "clean:script-b",
    "skip_transpose_merge_pair:script-a",
  ]


def test_truncated_or_unstamped_unit_is_recomputed(tmp_path: Path) -> None:
  """A partial body or a missing stamp is not trusted."""
  cache = tmp_path / "units"
  units = resume.build_units(
    ["trunc", "nostamp"],
    arm_id="clean",
    checkpoint_sha256="checkpoint",
    input_sha256="input",
    script_sha256="script",
  )
  calls: list[str] = []

  def compute(unit: resume.Unit) -> dict[str, str]:
    calls.append(unit.unit_id)
    return {"unit_id": unit.unit_id}

  resume.run_units(units, compute, cache, resume=True)
  trunc_body, trunc_stamp = resume.unit_paths(cache, units[0].cache_key)
  assert trunc_stamp.is_file()
  trunc_body.write_text("{truncated", encoding="utf-8")
  _body, stamp = resume.unit_paths(cache, units[1].cache_key)
  stamp.unlink()
  calls.clear()
  _payloads, stats = resume.run_units(units, compute, cache, resume=True)
  assert stats.n_reused == 0
  assert stats.n_computed == 2
  assert calls == ["trunc", "nostamp"]


def test_relaunch_makes_progress_after_partial_kill(tmp_path: Path) -> None:
  """The first attempt is killed after one unit; the next attempt finishes the rest."""
  cache = tmp_path / "units"
  counter = tmp_path / "counter.txt"
  die_once = tmp_path / "die-once"
  die_once.write_text("1", encoding="utf-8")
  units = resume.build_units(
    ["a", "b", "c"],
    arm_id="clean",
    checkpoint_sha256="checkpoint",
    input_sha256="input",
    script_sha256="script",
  )
  command = [
    sys.executable,
    str(_SCRIPT),
    "--cache-dir",
    str(cache),
    "--unit-ids",
    "a,b,c",
    "--arm-id",
    "clean",
    "--checkpoint-sha256",
    "checkpoint",
    "--input-sha256",
    "input",
    "--script-sha256",
    "script",
    "--counter",
    str(counter),
    "--die-once",
    str(die_once),
    "--resume",
  ]
  logger = logging.getLogger("test_graded_resume_relaunch")
  logger.handlers.clear()
  logger.setLevel(logging.INFO)
  logger.propagate = False
  buffer = io.StringIO()
  logger.addHandler(logging.StreamHandler(buffer))
  launched = resume.relaunch_subprocess(
    command,
    directory=cache,
    units=units,
    resume=True,
    max_attempts=3,
    timeout_s=30,
    logger=logger,
  )
  assert launched.returncode == 0
  assert launched.n_units == 3
  assert launched.n_reused == 0
  assert launched.n_computed == 3
  counted = counter.read_text(encoding="utf-8").split()
  assert counted.count("a") == 1
  assert counted.count("b") == 1
  assert counted.count("c") == 1
  matches = [
    (int(attempt), int(reused), int(remaining))
    for attempt, reused, remaining in _ATTEMPT.findall(buffer.getvalue())
  ]
  assert matches[0] == (1, 0, 3)
  assert matches[1] == (2, 1, 2)
