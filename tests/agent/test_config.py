"""Layer precedence for ``aminx.agent.config``.

The config module is loaded from its file so collection does not execute
``aminx/__init__.py``. ``start`` is a tmp directory and ``XDG_CONFIG_HOME``
points at tmp, so the repo pyproject and the user config are never read.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "src" / "aminx" / "agent" / "config.py"


def _load_config() -> ModuleType:
  """Load ``config.py`` without importing the jax-heavy package init."""
  spec = importlib.util.spec_from_file_location("aminx_agent_config", _CONFIG_PATH)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


_config = _load_config()


class _Launch(Protocol):
  """Attributes tests read off ``LaunchConfig``."""

  mode: str
  python: Path | None
  source: str


class _Layout:
  """Tmp project root and XDG config home for one test."""

  def __init__(self, root: Path) -> None:
    self.start = root / "start"
    self.xdg = root / "xdg"
    self.start.mkdir()
    self.xdg.mkdir()

  @property
  def config_path(self) -> Path:
    """Path of the user ``aminx/config.toml``."""
    return self.xdg / "aminx" / "config.toml"


@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Layout:
  """Isolate project discovery and ``XDG_CONFIG_HOME`` under ``tmp_path``."""
  isolated = _Layout(tmp_path)
  monkeypatch.setenv("XDG_CONFIG_HOME", str(isolated.xdg))
  return isolated


def _resolve(
  layout: _Layout,
  mode: str | None = None,
  python: str | Path | None = None,
  *,
  environ: Mapping[str, str] | None = None,
  start: Path | None = None,
) -> _Launch:
  return cast(
    _Launch,
    _config.resolve_launch(
      mode,
      python,
      start=layout.start if start is None else start,
      environ={} if environ is None else environ,
    ),
  )


def _source(
  layout: _Layout,
  mode: str | None = None,
  python: str | Path | None = None,
  *,
  environ: Mapping[str, str] | None = None,
  start: Path | None = None,
) -> str:
  source = _config.mcp_launch_source(
    mode,
    python,
    start=layout.start if start is None else start,
    environ={} if environ is None else environ,
  )
  assert isinstance(source, str)
  return source


def test_argument_alone_selects_venv(layout: _Layout) -> None:
  """An explicit mode and interpreter win and report ``argument``."""
  python = layout.start / "python"
  cfg = _resolve(layout, "venv", python)
  assert cfg.mode == "venv"
  assert cfg.python == python
  assert cfg.source == "argument"
  assert _source(layout, "venv", python) == "argument"


def test_env_alone_selects_venv(layout: _Layout) -> None:
  """``AMINX_MCP_LAUNCH`` selects venv and reports ``$AMINX_MCP_LAUNCH``."""
  python = layout.start / "python"
  environ = {"AMINX_MCP_LAUNCH": "venv", "AMINX_MCP_PYTHON": str(python)}
  cfg = _resolve(layout, environ=environ)
  assert cfg.mode == "venv"
  assert cfg.python == python
  assert cfg.source == "$AMINX_MCP_LAUNCH"
  assert _source(layout, environ=environ) == "$AMINX_MCP_LAUNCH"


def test_pyproject_alone_selects_venv(layout: _Layout) -> None:
  """``[tool.aminx.agent]`` selects venv and names the pyproject path."""
  python = layout.start / "python"
  pyproject = layout.start.resolve() / "pyproject.toml"
  pyproject.write_text(
    f"[tool.aminx.agent]\nmcp_launch = 'venv'\nmcp_python = '{python}'\n",
    encoding="utf-8",
  )
  cfg = _resolve(layout)
  assert cfg.mode == "venv"
  assert cfg.python == python
  assert cfg.source == f"[tool.aminx.agent] in {pyproject}"
  assert _source(layout) == cfg.source


def test_xdg_alone_selects_venv(layout: _Layout) -> None:
  """The XDG ``[agent]`` table selects venv and reports the config path."""
  python = layout.xdg / "python"
  layout.config_path.parent.mkdir()
  layout.config_path.write_text(
    f"[agent]\nmcp_launch = 'venv'\nmcp_python = '{python}'\n",
    encoding="utf-8",
  )
  cfg = _resolve(layout)
  assert cfg.mode == "venv"
  assert cfg.python == python
  assert cfg.source == str(layout.config_path)
  assert _source(layout) == str(layout.config_path)


def test_argument_beats_env(layout: _Layout) -> None:
  """An explicit mode wins over ``AMINX_MCP_LAUNCH``."""
  cfg = _resolve(layout, "venv", environ={"AMINX_MCP_LAUNCH": "uvx"})
  assert cfg.mode == "venv"
  assert cfg.source == "argument"
  assert cfg.python == Path(sys.executable)


def test_env_beats_pyproject(layout: _Layout) -> None:
  """``AMINX_MCP_LAUNCH`` wins over ``[tool.aminx.agent]``."""
  (layout.start / "pyproject.toml").write_text(
    "[tool.aminx.agent]\nmcp_launch = 'uvx'\n",
    encoding="utf-8",
  )
  cfg = _resolve(layout, environ={"AMINX_MCP_LAUNCH": "venv"})
  assert cfg.mode == "venv"
  assert cfg.source == "$AMINX_MCP_LAUNCH"
  assert cfg.python == Path(sys.executable)


def test_pyproject_beats_xdg(layout: _Layout) -> None:
  """The nearest pyproject wins over the XDG file."""
  (layout.start / "pyproject.toml").write_text(
    "[tool.aminx.agent]\nmcp_launch = 'venv'\n",
    encoding="utf-8",
  )
  layout.config_path.parent.mkdir()
  layout.config_path.write_text("[agent]\nmcp_launch = 'uvx'\n", encoding="utf-8")
  cfg = _resolve(layout)
  assert cfg.mode == "venv"
  assert cfg.source == f"[tool.aminx.agent] in {layout.start.resolve() / 'pyproject.toml'}"
  assert cfg.python == Path(sys.executable)


def test_xdg_beats_default(layout: _Layout) -> None:
  """The XDG file wins over the built-in ``uvx`` default."""
  layout.config_path.parent.mkdir()
  layout.config_path.write_text("[agent]\nmcp_launch = 'venv'\n", encoding="utf-8")
  cfg = _resolve(layout)
  assert cfg.mode == "venv"
  assert cfg.source == str(layout.config_path)
  assert cfg.python == Path(sys.executable)
  assert _source(layout) == str(layout.config_path)


def test_default_is_uvx_without_python(layout: _Layout) -> None:
  """With no config, the mode is ``uvx``, the interpreter is unset, source is default."""
  cfg = _resolve(layout)
  assert cfg.mode == "uvx"
  assert cfg.python is None
  assert cfg.source == "default"
  assert _source(layout) == "default"


def test_venv_without_python_uses_sys_executable(layout: _Layout) -> None:
  """``venv`` with no interpreter at any layer uses ``sys.executable``."""
  cfg = _resolve(layout, "venv")
  assert cfg.mode == "venv"
  assert cfg.python == Path(sys.executable)
  assert cfg.source == "argument"


def test_uvx_mode_clears_python(layout: _Layout) -> None:
  """An explicit ``uvx`` mode does not keep an interpreter."""
  cfg = _resolve(layout, "uvx", layout.start / "python")
  assert cfg.mode == "uvx"
  assert cfg.python is None
  assert cfg.source == "argument"


def test_relative_pyproject_python_resolves_against_pyproject_dir(layout: _Layout) -> None:
  """A relative ``mcp_python`` is resolved from the pyproject directory."""
  nested = layout.start / "pkg" / "sub"
  nested.mkdir(parents=True)
  (layout.start / "pyproject.toml").write_text(
    "[tool.aminx.agent]\nmcp_launch = 'venv'\nmcp_python = 'rel/bin/python'\n",
    encoding="utf-8",
  )
  cfg = _resolve(layout, start=nested)
  assert cfg.mode == "venv"
  assert cfg.python == (layout.start.resolve() / "rel" / "bin" / "python").resolve()
  assert cfg.source == f"[tool.aminx.agent] in {layout.start.resolve() / 'pyproject.toml'}"


def test_bogus_env_mode_names_the_variable(layout: _Layout) -> None:
  """``AMINX_MCP_LAUNCH=bogus`` raises ``ValueError`` naming that variable."""
  with pytest.raises(ValueError, match="AMINX_MCP_LAUNCH") as exc_info:
    _resolve(layout, environ={"AMINX_MCP_LAUNCH": "bogus"})
  message = str(exc_info.value)
  assert "uvx" in message
  assert "venv" in message
  with pytest.raises(ValueError, match="AMINX_MCP_LAUNCH"):
    _source(layout, environ={"AMINX_MCP_LAUNCH": "bogus"})


def test_malformed_xdg_toml_names_the_file(layout: _Layout) -> None:
  """A broken XDG config raises ``ValueError`` naming that file."""
  layout.config_path.parent.mkdir()
  layout.config_path.write_text("mcp_launch = [\n", encoding="utf-8")
  with pytest.raises(ValueError, match=re.escape(str(layout.config_path))):
    _resolve(layout)
