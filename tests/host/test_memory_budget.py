"""Precedence tests for the layered memory-budget resolver."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import xtrax.tiling.estimators as estimators  # noqa: TID251

import aminx.host.memory_budget as memory_budget
from aminx.host.memory_budget import (
  MEMORY_BUDGET_ENV,
  memory_budget_source,
  resolve_memory_budget_bytes,
)

_GIB = 4 * 1024**3


def _isolated(tmp_path: Path, **extra: str) -> tuple[Path, dict[str, str]]:
  start = tmp_path / "start"
  start.mkdir(parents=True)
  # Nearest pyproject, with no budget key, so a parent checkout cannot decide the layer.
  (start / "pyproject.toml").write_text("", encoding="utf-8")
  xdg = tmp_path / "xdg"
  xdg.mkdir(parents=True)
  return start, {"XDG_CONFIG_HOME": str(xdg), **extra}


def _write_pyproject(start: Path, text: str) -> None:
  (start / "pyproject.toml").write_text(text, encoding="utf-8")


def _write_user_config(environ: dict[str, str], text: str) -> None:
  path = Path(environ["XDG_CONFIG_HOME"]) / "aminx" / "config.toml"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def _patch_device(monkeypatch: pytest.MonkeyPatch, fn: object) -> None:
  monkeypatch.setattr(estimators, "device_memory_budget", fn)


def test_argument_alone_ignores_headroom(tmp_path: Path) -> None:
  start, environ = _isolated(tmp_path)
  value, source = memory_budget_source(1000, headroom=0.25, start=start, environ=environ)
  assert value == 1000
  assert source == "argument"
  assert resolve_memory_budget_bytes(1000, headroom=0.25, start=start, environ=environ) == 1000


def test_env_alone(tmp_path: Path) -> None:
  start, environ = _isolated(tmp_path, **{MEMORY_BUDGET_ENV: "2000"})
  value, source = memory_budget_source(headroom=0.25, start=start, environ=environ)
  assert value == 2000
  assert source == f"${MEMORY_BUDGET_ENV}"


def test_pyproject_alone(tmp_path: Path) -> None:
  start, environ = _isolated(tmp_path)
  _write_pyproject(start, "[tool.aminx]\nmemory_budget_bytes = 3000\n")
  value, source = memory_budget_source(headroom=0.25, start=start, environ=environ)
  assert value == 3000
  assert source == f"{(start / 'pyproject.toml').resolve()}:[tool.aminx]"


def test_user_config_alone(tmp_path: Path) -> None:
  start, environ = _isolated(tmp_path)
  _write_user_config(environ, "[runtime]\nmemory_budget_bytes = 4000\n")
  value, source = memory_budget_source(headroom=0.25, start=start, environ=environ)
  config = Path(environ["XDG_CONFIG_HOME"]) / "aminx" / "config.toml"
  assert value == 4000
  assert source == f"{config}:[runtime]"


def test_argument_beats_env_pyproject_and_user_config(tmp_path: Path) -> None:
  start, environ = _isolated(tmp_path, **{MEMORY_BUDGET_ENV: "2000"})
  _write_pyproject(start, "[tool.aminx]\nmemory_budget_bytes = 3000\n")
  _write_user_config(environ, "[runtime]\nmemory_budget_bytes = 4000\n")
  value, source = memory_budget_source(1000, headroom=0.25, start=start, environ=environ)
  assert (value, source) == (1000, "argument")


def test_env_beats_pyproject_and_user_config(tmp_path: Path) -> None:
  start, environ = _isolated(tmp_path, **{MEMORY_BUDGET_ENV: "2000"})
  _write_pyproject(start, "[tool.aminx]\nmemory_budget_bytes = 3000\n")
  _write_user_config(environ, "[runtime]\nmemory_budget_bytes = 4000\n")
  value, source = memory_budget_source(headroom=0.25, start=start, environ=environ)
  assert value == 2000
  assert source == f"${MEMORY_BUDGET_ENV}"


def test_pyproject_beats_user_config(tmp_path: Path) -> None:
  start, environ = _isolated(tmp_path)
  _write_pyproject(start, "[tool.aminx]\nmemory_budget_bytes = 3000\n")
  _write_user_config(environ, "[runtime]\nmemory_budget_bytes = 4000\n")
  value, source = memory_budget_source(headroom=0.25, start=start, environ=environ)
  assert value == 3000
  assert source.endswith(":[tool.aminx]")


def test_env_none_skips_to_pyproject(tmp_path: Path) -> None:
  start, environ = _isolated(tmp_path, **{MEMORY_BUDGET_ENV: "none"})
  _write_pyproject(start, "[tool.aminx]\nmemory_budget_bytes = 3000\n")
  value, source = memory_budget_source(headroom=0.25, start=start, environ=environ)
  assert value == 3000
  assert source.endswith(":[tool.aminx]")


def test_env_empty_skips(tmp_path: Path) -> None:
  start, environ = _isolated(tmp_path, **{MEMORY_BUDGET_ENV: "  "})
  _write_pyproject(start, "[tool.aminx]\nmemory_budget_bytes = 3000\n")
  value, _source = memory_budget_source(start=start, environ=environ)
  assert value == 3000


def test_malformed_toml_raises(tmp_path: Path) -> None:
  start, environ = _isolated(tmp_path)
  _write_pyproject(start, "this is not [ toml")
  with pytest.raises(ValueError, match="pyproject.toml"):
    memory_budget_source(start=start, environ=environ)

  start2, environ2 = _isolated(tmp_path / "user")
  _write_user_config(environ2, "runtime = [\n")
  with pytest.raises(ValueError, match="config.toml"):
    memory_budget_source(start=start2, environ=environ2)


def test_non_positive_and_non_int_raise(tmp_path: Path) -> None:
  start, environ = _isolated(tmp_path)
  _write_pyproject(start, "[tool.aminx]\nmemory_budget_bytes = 0\n")
  with pytest.raises(ValueError, match=r"\[tool.aminx\]"):
    memory_budget_source(start=start, environ=environ)

  _write_pyproject(start, "[tool.aminx]\nmemory_budget_bytes = 1.5\n")
  with pytest.raises(ValueError, match=r"\[tool.aminx\]"):
    memory_budget_source(start=start, environ=environ)

  bad_env = {**environ, MEMORY_BUDGET_ENV: "abc"}
  with pytest.raises(ValueError, match=MEMORY_BUDGET_ENV):
    memory_budget_source(start=start, environ=bad_env)

  zero_env = {**environ, MEMORY_BUDGET_ENV: "0"}
  with pytest.raises(ValueError, match=MEMORY_BUDGET_ENV):
    memory_budget_source(start=start, environ=zero_env)


def test_device_layer_applies_headroom(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  start, environ = _isolated(tmp_path)

  def fake_budget(fraction: float = 0.9, device: object = None) -> int:
    del device
    return int(8000 * fraction)

  _patch_device(monkeypatch, fake_budget)
  value, source = memory_budget_source(headroom=0.25, start=start, environ=environ)
  assert value == 2000
  assert source == "device"


def test_device_failure_uses_default_and_logs_once(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  caplog: pytest.LogCaptureFixture,
) -> None:
  start, environ = _isolated(tmp_path)

  def boom(fraction: float = 0.9, device: object = None) -> int:
    del fraction, device
    raise RuntimeError("no bytes_limit")

  _patch_device(monkeypatch, boom)
  memory_budget._default_warned[0] = False
  with caplog.at_level(logging.INFO, logger="aminx.host.memory_budget"):
    first, source = memory_budget_source(headroom=0.5, start=start, environ=environ)
    second, source2 = memory_budget_source(headroom=0.5, start=start, environ=environ)
  assert first == int(_GIB * 0.5)
  assert second == first
  assert source == "default"
  assert source2 == "default"
  # INFO, not WARNING: the default layer is the normal CPU path.
  records = [record for record in caplog.records if record.name == "aminx.host.memory_budget"]
  assert [record.levelno for record in records] == [logging.INFO]
  assert "default" in records[0].getMessage()
  assert MEMORY_BUDGET_ENV in records[0].getMessage()


def test_headroom_not_applied_to_configured_layers(tmp_path: Path) -> None:
  start, environ = _isolated(tmp_path, **{MEMORY_BUDGET_ENV: "2000"})
  _write_pyproject(start, "[tool.aminx]\nmemory_budget_bytes = 3000\n")
  _write_user_config(environ, "[runtime]\nmemory_budget_bytes = 4000\n")
  assert memory_budget_source(9000, headroom=0.25, start=start, environ=environ)[0] == 9000
  no_arg = {key: value for key, value in environ.items() if key != MEMORY_BUDGET_ENV}
  assert memory_budget_source(headroom=0.25, start=start, environ=environ)[0] == 2000
  assert memory_budget_source(headroom=0.25, start=start, environ=no_arg)[0] == 3000
