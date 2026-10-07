"""G-LAUNCH: publish pins the MCP command and refuses a bad launch.

These tests never call the Claude CLI and never write outside ``tmp_path``.
``claude_install`` is replaced with a recorder. ``CISTERNAL_PLUGIN_MARKETPLACE``
and ``XDG_CONFIG_HOME`` point at tmp.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("cisternal")
pytest.importorskip("fastmcp")

from aminx.agent.config import LaunchConfig
from aminx.agent.plugin import info, launch_command, main, publish


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
  """Point marketplace and XDG config at tmp and clear launch env vars."""
  monkeypatch.delenv("AMINX_MCP_LAUNCH", raising=False)
  monkeypatch.delenv("AMINX_MCP_PYTHON", raising=False)
  xdg = tmp_path / "xdg"
  xdg.mkdir(exist_ok=True)
  monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
  monkeypatch.setenv("CISTERNAL_PLUGIN_MARKETPLACE", str(tmp_path / "market"))
  return xdg


def _empty_start(tmp_path: Path) -> Path:
  """A directory with no ``pyproject.toml`` of its own."""
  start = tmp_path / "start"
  start.mkdir(exist_ok=True)
  return start


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
  """Capture ``claude_install`` calls."""
  calls: list[dict[str, Any]] = []

  def _record(*args: Any, **kwargs: Any) -> None:
    calls.append({"args": args, "kwargs": kwargs})

  monkeypatch.setattr("cisternal.plugin.app.claude_install", _record)
  return calls


@pytest.fixture(autouse=True)
def _on_index(monkeypatch: pytest.MonkeyPatch) -> None:
  """Treat the bundle version as published unless a test says otherwise."""

  def _yes(version: str, timeout: float = 10.0) -> bool:
    del version, timeout
    return True

  monkeypatch.setattr("aminx.agent.plugin.version_on_index", _yes)


def _assert_venv(result: dict[str, Any], source: str, calls: list[dict[str, Any]]) -> None:
  """The record and the installed server both use this interpreter."""
  command = [sys.executable, "-m", "aminx.agent.mcp"]
  assert result["launch_mode"] == "venv"
  assert result["launch_source"] == source
  assert result["command"] == command
  assert Path(result["command"][0]).is_absolute()
  assert len(calls) == 1
  bundle = calls[0]["args"][0]
  assert len(bundle.mcp_servers) == 1
  assert tuple(bundle.mcp_servers[0].command) == tuple(command)
  assert bundle.mcp_servers[0].launch == "path"


def test_publish_default_uvx(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  recorder: list[dict[str, Any]],
) -> None:
  """With no config, publish records the uvx pin and source ``default``."""
  _isolate(monkeypatch, tmp_path)
  result = publish(environ={}, start=_empty_start(tmp_path))
  assert result["launch_source"] == "default"
  assert result["launch_mode"] == "uvx"
  assert result["command"][0:2] == ["uvx", "--from"]
  assert result["command"][3] == "aminx-mcp"
  pinned = result["command"][2]
  assert pinned.startswith("aminx[agent]==")
  assert "+" not in pinned.split("==", 1)[1]
  assert len(recorder) == 1
  bundle = recorder[0]["args"][0]
  assert len(bundle.mcp_servers) == 1
  assert tuple(bundle.mcp_servers[0].command) == tuple(result["command"])


def test_argument_selects_venv(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  recorder: list[dict[str, Any]],
) -> None:
  """An explicit ``venv`` argument wins."""
  _isolate(monkeypatch, tmp_path)
  result = publish("venv", environ={}, start=_empty_start(tmp_path))
  _assert_venv(result, "argument", recorder)


def test_env_selects_venv(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  recorder: list[dict[str, Any]],
) -> None:
  """``AMINX_MCP_LAUNCH=venv`` alone selects the local interpreter."""
  _isolate(monkeypatch, tmp_path)
  result = publish(environ={"AMINX_MCP_LAUNCH": "venv"}, start=_empty_start(tmp_path))
  _assert_venv(result, "$AMINX_MCP_LAUNCH", recorder)


def test_pyproject_selects_venv(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  recorder: list[dict[str, Any]],
) -> None:
  """``[tool.aminx.agent] mcp_launch`` alone selects venv."""
  _isolate(monkeypatch, tmp_path)
  start = _empty_start(tmp_path).resolve()
  pyproject = start / "pyproject.toml"
  pyproject.write_text('[tool.aminx.agent]\nmcp_launch = "venv"\n', encoding="utf-8")
  result = publish(environ={}, start=start)
  _assert_venv(result, f"[tool.aminx.agent] in {pyproject}", recorder)


def test_xdg_selects_venv(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  recorder: list[dict[str, Any]],
) -> None:
  """The XDG ``aminx/config.toml`` alone selects venv."""
  xdg = _isolate(monkeypatch, tmp_path)
  config = xdg / "aminx" / "config.toml"
  config.parent.mkdir()
  config.write_text('[agent]\nmcp_launch = "venv"\n', encoding="utf-8")
  result = publish(environ={}, start=_empty_start(tmp_path))
  _assert_venv(result, str(config), recorder)


def _assert_main_error(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  recorder: list[dict[str, Any]],
  capsys: pytest.CaptureFixture[str],
  argv: list[str],
) -> str:
  """``main`` exits 1 with ``error:`` and does not install."""
  _isolate(monkeypatch, tmp_path)
  assert main(argv) == 1
  err = capsys.readouterr().err
  assert "error:" in err
  assert recorder == []
  return err


def test_bogus_launch_mode(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  recorder: list[dict[str, Any]],
  capsys: pytest.CaptureFixture[str],
) -> None:
  """``AMINX_MCP_LAUNCH=bogus`` fails before install."""
  _isolate(monkeypatch, tmp_path)
  monkeypatch.setenv("AMINX_MCP_LAUNCH", "bogus")
  assert main(["publish"]) == 1
  assert "error:" in capsys.readouterr().err
  assert recorder == []


def test_malformed_xdg_config(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  recorder: list[dict[str, Any]],
  capsys: pytest.CaptureFixture[str],
) -> None:
  """A malformed XDG config fails before install."""
  xdg = _isolate(monkeypatch, tmp_path)
  config = xdg / "aminx" / "config.toml"
  config.parent.mkdir()
  config.write_text("[\n", encoding="utf-8")
  assert main(["publish"]) == 1
  err = capsys.readouterr().err
  assert "error:" in err
  assert recorder == []


def test_venv_python_import_fails(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  recorder: list[dict[str, Any]],
  capsys: pytest.CaptureFixture[str],
) -> None:
  """A ``--python`` that exits 1 fails before install."""
  script = tmp_path / "fake-python"
  script.write_text("#!/bin/sh\necho 'import failed' >&2\nexit 1\n", encoding="utf-8")
  script.chmod(0o755)
  _assert_main_error(
    monkeypatch,
    tmp_path,
    recorder,
    capsys,
    ["publish", "--launch", "venv", "--python", str(script)],
  )


def test_uvx_unpublished(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  recorder: list[dict[str, Any]],
  capsys: pytest.CaptureFixture[str],
) -> None:
  """``uvx`` refuses a version the index does not have."""

  def _missing(version: str, timeout: float = 10.0) -> bool:
    del version, timeout
    return False

  monkeypatch.setattr("aminx.agent.plugin.version_on_index", _missing)
  err = _assert_main_error(monkeypatch, tmp_path, recorder, capsys, ["publish"])
  assert "--launch venv" in err


def test_uvx_index_unreachable(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  recorder: list[dict[str, Any]],
  capsys: pytest.CaptureFixture[str],
) -> None:
  """``uvx`` refuses to publish when the index cannot be reached."""

  def _offline(version: str, timeout: float = 10.0) -> None:
    del version, timeout
    return None

  monkeypatch.setattr("aminx.agent.plugin.version_on_index", _offline)
  err = _assert_main_error(monkeypatch, tmp_path, recorder, capsys, ["publish"])
  assert "could not be reached" in err
  assert "uvx needs it too" in err


def test_dry_run_does_not_install(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  recorder: list[dict[str, Any]],
  capsys: pytest.CaptureFixture[str],
) -> None:
  """``--dry-run`` returns the record and never calls the installer."""
  _isolate(monkeypatch, tmp_path)
  assert main(["publish", "--dry-run"]) == 0
  assert recorder == []
  payload = json.loads(capsys.readouterr().out)
  assert payload["dry_run"] is True
  assert payload["launch_mode"] == "uvx"


def test_launch_command_drops_local_suffix() -> None:
  """A ``+local`` version suffix is omitted from the uvx pin."""
  cfg = LaunchConfig(mode="uvx", python=None, source="argument")
  assert launch_command(cfg, "1.2.3+gabcdef") == (
    "uvx",
    "--from",
    "aminx[agent]==1.2.3",
    "aminx-mcp",
  )


def test_info_reports_resolution_without_writing(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """``info`` returns the launch and marketplace keys and writes nothing."""

  def _boom(version: str, timeout: float = 10.0) -> bool:
    del version, timeout
    msg = "info must not query the index"
    raise AssertionError(msg)

  monkeypatch.setattr("aminx.agent.plugin.version_on_index", _boom)
  _isolate(monkeypatch, tmp_path)
  start = _empty_start(tmp_path)
  before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
  result = info(start=start, environ={})
  after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
  assert before == after
  assert not (tmp_path / "market").exists()
  assert set(result) >= {
    "launch_mode",
    "launch_source",
    "command",
    "python",
    "marketplace",
    "marketplace_source",
    "bundle_source",
  }
  assert result["launch_mode"] == "uvx"
  assert result["launch_source"] == "default"
  assert result["command"][0] == "uvx"
  assert result["python"] is None
  assert result["marketplace_source"] == "$CISTERNAL_PLUGIN_MARKETPLACE"
  assert result["bundle_source"]


def test_dry_run_with_no_marketplace_writes_no_config(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  recorder: list[dict[str, Any]],
  capsys: pytest.CaptureFixture[str],
) -> None:
  """With nothing configured, ``--dry-run`` reports that and leaves the user config alone.

  ``resolve_marketplace_root`` writes a default entry into
  ``$XDG_CONFIG_HOME/cisternal/config.toml``; a dry run must not reach it.
  """
  xdg = _isolate(monkeypatch, tmp_path)
  monkeypatch.delenv("CISTERNAL_PLUGIN_MARKETPLACE", raising=False)
  monkeypatch.chdir(_empty_start(tmp_path))
  assert main(["publish", "--dry-run"]) == 0
  assert recorder == []
  assert list(xdg.rglob("*")) == []
  payload = json.loads(capsys.readouterr().out)
  assert payload["dry_run"] is True
  assert payload["marketplace"] is None
  assert payload["marketplace_source"]
