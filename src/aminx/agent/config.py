"""Layered resolver for the MCP launch mode.

Resolution order, first hit wins, with ``mode`` and ``python`` chosen
independently. ``LaunchConfig.source`` names the layer that decided the
mode:

1. Explicit arguments (``argument``).
2. ``AMINX_MCP_LAUNCH`` / ``AMINX_MCP_PYTHON`` (``$AMINX_MCP_LAUNCH``).
3. ``[tool.aminx.agent]`` in the nearest ``pyproject.toml`` at or above
   ``start``.
4. ``${XDG_CONFIG_HOME:-~/.config}/aminx/config.toml`` table ``[agent]``.
5. Default mode ``uvx`` (``default``).

``uvx`` never carries an interpreter. ``venv`` with no interpreter from any
layer uses ``sys.executable``. Files are read with ``tomllib`` and never
written. A malformed TOML file or a non-string value raises ``ValueError``
instead of being skipped.
"""

from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NoReturn, cast

LAUNCH_MODES: tuple[str, ...] = ("uvx", "venv")

_ENV_LAUNCH = "AMINX_MCP_LAUNCH"
_ENV_PYTHON = "AMINX_MCP_PYTHON"
_ENV_SOURCE = "$AMINX_MCP_LAUNCH"


@dataclass(frozen=True, slots=True)
class LaunchConfig:
  """Resolved MCP launch mode.

  Attributes
  ----------
  mode : str
    ``uvx`` or ``venv``.
  python : Path or None
    Interpreter for ``venv``. Always ``None`` for ``uvx``.
  source : str
    Layer that selected ``mode``.
  """

  mode: str
  python: Path | None
  source: str


@dataclass(frozen=True, slots=True)
class _Partial:
  """Mode and interpreter collected before defaults are applied."""

  mode: str | None = None
  python: Path | None = None
  source: str | None = None


def resolve_launch(
  mode: str | None = None,
  python: str | Path | None = None,
  *,
  start: Path | None = None,
  environ: Mapping[str, str] | None = None,
) -> LaunchConfig:
  """Resolve the MCP launch mode and interpreter.

  Parameters
  ----------
  mode : str or None, optional
    Explicit launch mode. ``None`` defers to a lower layer.
  python : str or Path or None, optional
    Explicit interpreter. Ignored when the resolved mode is ``uvx``.
  start : Path or None, optional
    Directory walked upward for ``pyproject.toml``. Defaults to the
    current working directory.
  environ : mapping of str to str or None, optional
    Environment for ``AMINX_MCP_LAUNCH`` and ``AMINX_MCP_PYTHON``.
    ``None`` uses ``os.environ``. ``XDG_CONFIG_HOME`` is read from this
    mapping when the key is present, otherwise from ``os.environ``.

  Returns
  -------
  LaunchConfig
    Resolved mode, interpreter, and the layer that decided the mode.

  Raises
  ------
  ValueError
    Unknown mode, malformed TOML, or a non-string config value.
  """
  root = _start_dir(Path.cwd() if start is None else start)
  env = os.environ if environ is None else environ
  partial = _Partial()

  if mode is not None:
    partial = _consider_mode(partial, mode, "argument", "argument")
  if python is not None:
    partial = _consider_python(partial, python, base=root, origin="argument")
  if _satisfied(partial):
    return _finish(partial)

  if _ENV_LAUNCH in env:
    partial = _consider_mode(partial, env[_ENV_LAUNCH], _ENV_SOURCE, _ENV_SOURCE)
  if _ENV_PYTHON in env:
    partial = _consider_python(partial, env[_ENV_PYTHON], base=root, origin=_ENV_SOURCE)
  if _satisfied(partial):
    return _finish(partial)

  pyproject = _nearest_pyproject(root)
  if pyproject is not None:
    table = _nested_table(_read_toml(pyproject), pyproject, ("tool", "aminx", "agent"))
    if table is not None:
      origin = f"[tool.aminx.agent] in {pyproject}"
      if "mcp_launch" in table:
        partial = _consider_mode(partial, table["mcp_launch"], origin, origin)
      if "mcp_python" in table:
        partial = _consider_python(
          partial,
          table["mcp_python"],
          base=pyproject.parent,
          origin=origin,
        )
  if _satisfied(partial):
    return _finish(partial)

  config_path = _user_config_path(environ)
  if config_path.is_file():
    table = _nested_table(_read_toml(config_path), config_path, ("agent",))
    if table is not None:
      origin = str(config_path)
      if "mcp_launch" in table:
        partial = _consider_mode(partial, table["mcp_launch"], origin, origin)
      if "mcp_python" in table:
        partial = _consider_python(
          partial,
          table["mcp_python"],
          base=config_path.parent,
          origin=origin,
        )

  return _finish(partial)


def mcp_launch_source(
  mode: str | None = None,
  python: str | Path | None = None,
  *,
  start: Path | None = None,
  environ: Mapping[str, str] | None = None,
) -> str:
  """Return the layer that decided the launch mode.

  Parameters
  ----------
  mode : str or None, optional
    Explicit launch mode. See ``resolve_launch``.
  python : str or Path or None, optional
    Explicit interpreter. See ``resolve_launch``.
  start : Path or None, optional
    Directory walked upward for ``pyproject.toml``.
  environ : mapping of str to str or None, optional
    Environment mapping. See ``resolve_launch``.

  Returns
  -------
  str
    ``LaunchConfig.source`` for the same arguments.

  Raises
  ------
  ValueError
    Unknown mode, malformed TOML, or a non-string config value.
  """
  return resolve_launch(mode, python, start=start, environ=environ).source


def _start_dir(start: Path) -> Path:
  """Return ``start`` as a directory."""
  resolved = start.expanduser().resolve()
  if resolved.is_dir():
    return resolved
  return resolved.parent


def _satisfied(partial: _Partial) -> bool:
  """Return whether both mode and, for venv, an interpreter are known."""
  if partial.mode == "uvx":
    return True
  return partial.mode == "venv" and partial.python is not None


def _finish(partial: _Partial) -> LaunchConfig:
  """Apply the uvx/venv interpreter rules and the default mode."""
  mode = "uvx" if partial.mode is None else partial.mode
  source = "default" if partial.source is None else partial.source
  if mode == "uvx":
    python: Path | None = None
  elif partial.python is None:
    python = Path(sys.executable)
  else:
    python = partial.python
  return LaunchConfig(mode=mode, python=python, source=source)


def _consider_mode(partial: _Partial, value: object, origin: str, source: str) -> _Partial:
  """Validate ``value`` and keep the first mode."""
  mode = _coerce_mode(value, origin)
  if partial.mode is not None:
    return partial
  return replace(partial, mode=mode, source=source)


def _consider_python(partial: _Partial, value: object, *, base: Path, origin: str) -> _Partial:
  """Validate ``value`` and keep the first interpreter."""
  path = _coerce_python(value, base=base, origin=origin)
  if partial.python is not None:
    return partial
  return replace(partial, python=path)


def _invalid(message: str) -> NoReturn:
  """Raise ``ValueError``.

  Config failures are ``ValueError`` (unknown mode, malformed TOML, non-string
  values), including cases where the bad value has the wrong Python type.
  """
  raise ValueError(message)


def _coerce_mode(value: object, origin: str) -> str:
  """Return a launch mode or raise ``ValueError`` naming ``origin``."""
  if not isinstance(value, str):
    _invalid(f"non-string mcp_launch from {origin}: got {type(value).__name__}")
  mode = value.strip()
  if mode not in LAUNCH_MODES:
    allowed = ", ".join(LAUNCH_MODES)
    raise ValueError(f"unknown launch mode {value!r} from {origin}; allowed modes: {allowed}")
  return mode


def _coerce_python(value: object, *, base: Path, origin: str) -> Path:
  """Return an interpreter path, resolving relative paths against ``base``."""
  if isinstance(value, Path):
    raw = value.expanduser()
  elif isinstance(value, str):
    raw = Path(value).expanduser()
  else:
    _invalid(f"non-string mcp_python from {origin}: got {type(value).__name__}")
  if raw.is_absolute():
    return raw
  return (base / raw).resolve()


def _nearest_pyproject(start: Path) -> Path | None:
  """Return the nearest ``pyproject.toml`` at or above ``start``."""
  for directory in (start, *start.parents):
    candidate = directory / "pyproject.toml"
    if candidate.is_file():
      return candidate
  return None


def _user_config_path(environ: Mapping[str, str] | None) -> Path:
  """Return ``$XDG_CONFIG_HOME/aminx/config.toml`` or the ``~/.config`` default."""
  if environ is not None and "XDG_CONFIG_HOME" in environ:
    raw = environ["XDG_CONFIG_HOME"]
  else:
    raw = os.environ.get("XDG_CONFIG_HOME", "")
  if raw:
    return Path(raw).expanduser() / "aminx" / "config.toml"
  return Path.home() / ".config" / "aminx" / "config.toml"


def _read_toml(path: Path) -> Mapping[str, object]:
  """Parse ``path`` or raise ``ValueError`` naming the file."""
  try:
    loaded = tomllib.loads(path.read_text(encoding="utf-8"))
  except tomllib.TOMLDecodeError as exc:
    raise ValueError(f"malformed TOML in {path}: {exc}") from exc
  if not isinstance(loaded, dict):
    _invalid(f"malformed TOML in {path}: top-level value is not a table")
  return cast("dict[str, object]", loaded)


def _nested_table(
  data: Mapping[str, object],
  path: Path,
  keys: tuple[str, ...],
) -> Mapping[str, object] | None:
  """Walk nested TOML tables. Missing tables are a miss; wrong types raise."""
  current: object = data
  walked: list[str] = []
  for key in keys:
    if not isinstance(current, Mapping):
      parent = ".".join(walked)
      _invalid(f"malformed TOML in {path}: {parent} is not a table")
    walked.append(key)
    mapping = cast("Mapping[str, object]", current)
    if key not in mapping:
      return None
    current = mapping[key]
  if not isinstance(current, Mapping):
    _invalid(f"malformed TOML in {path}: {'.'.join(keys)} is not a table")
  return cast("Mapping[str, object]", current)


__all__ = ["LAUNCH_MODES", "LaunchConfig", "mcp_launch_source", "resolve_launch"]
