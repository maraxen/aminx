# ruff: noqa: S101
"""Pin preempt-safe per-unit completion stamps, sha256 reuse, and chunk timeouts in ``run_gate.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_run_gate() -> ModuleType:
  redsox_dir = Path(__file__).resolve().parents[2] / "scripts" / "redsox"
  if str(redsox_dir) not in sys.path:
    sys.path.insert(0, str(redsox_dir))
  path = redsox_dir / "run_gate.py"
  spec = importlib.util.spec_from_file_location("aminx_run_gate", path)
  if spec is None or spec.loader is None:
    msg = f"unable to load {path}"
    raise RuntimeError(msg)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_step1_persists_per_wave_stamps_and_reuses_on_matching_sha256(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  run_gate = _load_run_gate()
  repo = tmp_path / "repo"
  (repo / "tests").mkdir(parents=True)
  (repo / "tests" / "test_a.py").write_text("def test_a(): pass\n", encoding="utf-8")
  (repo / "tests" / "test_b.py").write_text("def test_b(): pass\n", encoding="utf-8")

  out = tmp_path / "out"
  out.mkdir(parents=True)
  waves = ["wave_a", "wave_b"]
  (out / "files_wave_a.txt").write_text("tests/test_a.py\n", encoding="utf-8")
  (out / "ids_wave_a.txt").write_text("tests/test_a.py::test_a\n", encoding="utf-8")
  (out / "files_wave_b.txt").write_text("tests/test_b.py\n", encoding="utf-8")
  (out / "ids_wave_b.txt").write_text("tests/test_b.py::test_b\n", encoding="utf-8")

  executed_waves: list[str] = []

  def fake_run(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: Path,
    timeout_s: float | None = None,
  ) -> int:
    _ = argv, cwd, log, timeout_s
    wave = env["AMINX_PORT_WAVE"]
    executed_waves.append(wave)
    outcomes_file = Path(env["AMINX_REDSOX_OUTCOMES"])
    record = {
      "nodeid": f"tests/test_{wave[-1]}.py::test_{wave[-1]}",
      "wave": wave,
      "when": "call",
      "outcome": "passed",
      "wasxfail": False,
      "mutant": None,
      "exc_type": None,
    }
    outcomes_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return 0

  monkeypatch.setattr(run_gate, "_run", fake_run)
  monkeypatch.setattr(run_gate, "_head_sha", lambda _repo: "deadbeef1234")

  log_path = out / "run_gate.log"
  rc1 = run_gate._step1(repo, out, waves, {}, log_path)
  assert rc1 == 0
  assert executed_waves == ["wave_a", "wave_b"]

  # Assembled outcomes.jsonl has both waves in deterministic order
  assembled = [
    json.loads(line)
    for line in (out / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]
  assert [row["wave"] for row in assembled] == ["wave_a", "wave_b"]

  # Second invocation with identical inputs reuses both stamps without calling _run
  executed_waves.clear()
  rc2 = run_gate._step1(repo, out, waves, {}, log_path)
  assert rc2 == 0
  assert executed_waves == []

  # Modifying wave_b's test file invalidates only wave_b's sha256; wave_a is still reused
  (repo / "tests" / "test_b.py").write_text("def test_b(): assert 1 == 1\n", encoding="utf-8")
  rc3 = run_gate._step1(repo, out, waves, {}, log_path)
  assert rc3 == 0
  assert executed_waves == ["wave_b"]


def test_step1_reruns_corrupted_or_failed_wave_only(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  run_gate = _load_run_gate()
  repo = tmp_path / "repo"
  (repo / "tests").mkdir(parents=True)
  (repo / "tests" / "test_a.py").write_text("def test_a(): pass\n", encoding="utf-8")
  (repo / "tests" / "test_b.py").write_text("def test_b(): pass\n", encoding="utf-8")

  out = tmp_path / "out"
  out.mkdir(parents=True)
  waves = ["wave_a", "wave_b"]
  for w in waves:
    sfx = w[-1]
    (out / f"files_{w}.txt").write_text(f"tests/test_{sfx}.py\n", encoding="utf-8")
    (out / f"ids_{w}.txt").write_text(f"tests/test_{sfx}.py::test_{sfx}\n", encoding="utf-8")

  executed: list[str] = []

  def fake_run(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: Path,
    timeout_s: float | None = None,
  ) -> int:
    _ = argv, cwd, log, timeout_s
    wave = env["AMINX_PORT_WAVE"]
    executed.append(wave)
    Path(env["AMINX_REDSOX_OUTCOMES"]).write_text(
      json.dumps({"nodeid": f"id_{wave}", "wave": wave, "outcome": "passed"}) + "\n",
      encoding="utf-8",
    )
    return 0

  monkeypatch.setattr(run_gate, "_run", fake_run)
  monkeypatch.setattr(run_gate, "_head_sha", lambda _repo: "deadbeef1234")

  log_path = out / "run_gate.log"
  assert run_gate._step1(repo, out, waves, {}, log_path) == 0
  assert executed == ["wave_a", "wave_b"]

  # Corrupt wave_a's outcomes file as if truncated mid-write during preemption
  (out / "units" / "step1__wave_a.outcomes.jsonl").write_text('{"nodeid": "trun', encoding="utf-8")
  executed.clear()
  assert run_gate._step1(repo, out, waves, {}, log_path) == 0
  assert executed == ["wave_a"]


def test_run_enforces_chunk_timeout(tmp_path: Path) -> None:
  run_gate = _load_run_gate()
  log_path = tmp_path / "timeout.log"
  rc = run_gate._run(
    [sys.executable, "-c", "import time; time.sleep(10)"],
    cwd=tmp_path,
    env={},
    log=log_path,
    timeout_s=0.2,
  )
  assert rc == 124
  assert "exit 124" in log_path.read_text(encoding="utf-8")
