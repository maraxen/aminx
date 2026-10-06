"""Opt-in CLI telemetry naming and event files."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from aminx.agent.telemetry import cli_telemetry_enabled, command_name

_SCRIPT = (
  "import sys\n"
  "sys.argv = ['aminx', 'spec', '--help']\n"
  "from aminx.cli import main\n"
  "main()\n"
)


def test_command_name_skips_option_values() -> None:
  """Group options that take values do not hide the following command."""
  argv = ["run", "--backbone-noise", "0.1", "sample", "--inputs", "x.pdb"]
  assert command_name(argv) == "run.sample"


def test_command_name_spec_validate() -> None:
  """A spec subcommand is group.command."""
  assert command_name(["spec", "validate"]) == "spec.validate"


def test_command_name_help_is_the_group_only() -> None:
  """``spec --help`` names the group and does not include the flag."""
  assert command_name(["aminx", "spec", "--help"]) == "spec"


def test_command_name_unknown() -> None:
  """Tokens that are not a registered group become ``unknown``."""
  assert command_name(["nope", "sample"]) == "unknown"
  assert command_name(["--help"]) == "unknown"


def test_cli_telemetry_enabled_values() -> None:
  """Enablement is the documented set, compared case-insensitively."""
  assert cli_telemetry_enabled({}) is False
  assert cli_telemetry_enabled({"CISTERNAL_TELEMETRY": "aminx"}) is True
  assert cli_telemetry_enabled({"CISTERNAL_TELEMETRY": "ALL"}) is True
  assert cli_telemetry_enabled({"CISTERNAL_TELEMETRY": "yes"}) is True
  assert cli_telemetry_enabled({"CISTERNAL_TELEMETRY": "1"}) is True
  assert cli_telemetry_enabled({"CISTERNAL_TELEMETRY": "true"}) is True
  assert cli_telemetry_enabled({"CISTERNAL_TELEMETRY": " True "}) is True
  assert cli_telemetry_enabled({"CISTERNAL_TELEMETRY": "no"}) is False
  assert cli_telemetry_enabled({"CISTERNAL_TELEMETRY": "bathos"}) is False


def _run(tmp_path: Path, telemetry: str | None) -> subprocess.CompletedProcess[str]:
  """Run ``aminx spec --help`` with telemetry pointed at ``tmp_path``."""
  env = os.environ.copy()
  env.pop("CISTERNAL_TELEMETRY", None)
  env["CISTERNAL_LOG_DIR"] = str(tmp_path)
  if telemetry is not None:
    env["CISTERNAL_TELEMETRY"] = telemetry
  return subprocess.run(
    [sys.executable, "-c", _SCRIPT],
    check=False,
    capture_output=True,
    text=True,
    env=env,
    timeout=180,
  )


def test_unset_telemetry_writes_no_events_file(tmp_path: Path) -> None:
  """With CISTERNAL_TELEMETRY unset, no events file is created."""
  completed = _run(tmp_path, None)
  assert completed.returncode == 0, completed.stderr
  assert list(tmp_path.glob("events*.jsonl")) == []


def test_enabled_telemetry_records_the_spec_span(tmp_path: Path) -> None:
  """CISTERNAL_TELEMETRY=aminx writes a span named aminx.cli.spec."""
  pytest.importorskip("cisternal")
  completed = _run(tmp_path, "aminx")
  assert completed.returncode == 0, completed.stderr
  files = list(tmp_path.glob("events*.jsonl"))
  assert files
  text = "\n".join(path.read_text(encoding="utf-8") for path in files)
  assert "aminx.cli.spec" in text


def test_failed_init_still_runs_the_command(
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  """With telemetry on, a cisternal.init() failure warns and the command still runs."""
  cisternal = pytest.importorskip("cisternal")
  from aminx.agent.telemetry import run_cli

  def _broken() -> None:
    msg = "events dir not writable"
    raise OSError(msg)

  monkeypatch.setenv("CISTERNAL_TELEMETRY", "aminx")
  monkeypatch.setattr(cisternal, "init", _broken)
  calls: list[int] = []
  run_cli(lambda: calls.append(1), argv=["aminx", "spec", "validate"])
  assert calls == [1]
  assert "events dir not writable" in capsys.readouterr().err
