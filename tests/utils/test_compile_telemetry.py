"""Compile-cache provenance reported by ``compile_telemetry`` follows xtrax's configured setting."""

from __future__ import annotations

import pytest

from pathlib import Path

import jax
from xtrax import jax_cache

from aminx.utils.compile_telemetry import jax_compilation_cache_setting


pytestmark = pytest.mark.usefixtures("jax_cache_isolated")


def test_namespaced_dir_reported_with_explicit_source(tmp_path: Path) -> None:
  setting = jax_cache.configure(path=tmp_path / "xla")

  record = jax_compilation_cache_setting()

  assert record == {"enabled": True, "dir": str(setting.dir), "source": "explicit argument"}
  assert setting.dir is not None
  assert setting.dir.name.startswith("jaxlib-")


def test_disabled_cache_reports_no_dir(tmp_path: Path) -> None:
  # A sentinel flag makes this fail if configure(path="none") stops clearing it.
  jax.config.update("jax_compilation_cache_dir", str(tmp_path / "sentinel"))
  jax_cache.configure(path="none")

  record = jax_compilation_cache_setting()

  assert jax.config.jax_compilation_cache_dir is None
  assert record["enabled"] is False
  assert record["dir"] is None


def test_dir_changed_behind_xtrax_reports_fallback_source(tmp_path: Path) -> None:
  jax_cache.configure(path=tmp_path / "xla")
  other = tmp_path / "elsewhere"
  jax.config.update("jax_compilation_cache_dir", str(other))

  record = jax_compilation_cache_setting()

  assert record["dir"] == str(other)
  assert record["source"] == "jax.config (not set by xtrax jax_cache)"
