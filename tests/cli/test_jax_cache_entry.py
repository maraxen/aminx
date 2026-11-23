"""``aminx.cli.main`` applies xtrax's persistent JAX cache before dispatch (debt #3527)."""

from __future__ import annotations

import importlib.metadata
import os
import re
from pathlib import Path

import jax
import pytest
from xtrax import jax_cache

import aminx.agent.telemetry as telemetry
import aminx.cli as cli


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
  """Record calls to ``configure`` and the stubbed app dispatch, in order."""
  log: list[str] = []
  monkeypatch.setattr(jax_cache, "configure", lambda *a, **k: log.append("configure"))
  monkeypatch.setattr(telemetry, "run_cli", lambda app: log.append("run_cli"))
  return log


def test_main_configures_jax_cache_before_dispatch(events: list[str]) -> None:
  cli.main()
  assert events == ["configure", "run_cli"]


def test_main_applies_env_dir_namespaced_and_created(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  monkeypatch.chdir(tmp_path)
  monkeypatch.setattr(telemetry, "run_cli", lambda app: None)
  cache = tmp_path / "xla"
  monkeypatch.setenv("XTRAX_JAX_CACHE_DIR", str(cache))

  cli.main()

  expected = cache / f"jaxlib-{importlib.metadata.version('jaxlib')}"
  assert jax.config.jax_compilation_cache_dir == str(expected)
  assert expected.is_dir()


def test_main_off_clears_preset_jax_dir(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  monkeypatch.chdir(tmp_path)
  monkeypatch.setattr(telemetry, "run_cli", lambda app: None)
  jax.config.update("jax_compilation_cache_dir", str(tmp_path / "preset"))
  monkeypatch.setenv("XTRAX_JAX_CACHE_DIR", "none")

  cli.main()

  assert jax.config.jax_compilation_cache_dir is None


def test_main_fails_loud_on_malformed_user_config(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  monkeypatch.chdir(tmp_path)
  dispatched: list[str] = []
  monkeypatch.setattr(telemetry, "run_cli", lambda app: dispatched.append("run_cli"))
  cfg = Path(os.environ["XDG_CONFIG_HOME"]) / "xtrax" / "config.toml"
  cfg.parent.mkdir(parents=True)
  cfg.write_text("[tool.xtrax\n", encoding="utf-8")

  with pytest.raises(ValueError, match=re.escape("config.toml")):
    cli.main()
  assert dispatched == []
