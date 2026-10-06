"""Layered memory-budget resolution for host and tiling planners.

Layers 1-4 are absolute byte counts. Layers 5-6 apply ``headroom``, matching
``bytes_limit * headroom`` on a device that reports a limit and the documented
4 GiB fallback otherwise.
"""

from __future__ import annotations

import logging
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

MEMORY_BUDGET_ENV = "AMINX_MEMORY_BUDGET_BYTES"
_DEFAULT_LIMIT_BYTES = 4 * 1024**3
_default_warned = [False]


def resolve_memory_budget_bytes(
  explicit: int | None = None,
  *,
  headroom: float = 1.0,
  start: Path | None = None,
  environ: Mapping[str, str] | None = None,
) -> int:
  """Resolve the memory budget in bytes.

  Parameters
  ----------
  explicit : int or None, optional
    Absolute byte count. When set, it wins and ``headroom`` is not applied.
  headroom : float, optional
    Multiplier for the device limit and the 4 GiB default. Not applied to
    an explicit argument, the environment variable, or TOML values.
  start : pathlib.Path or None, optional
    Directory to walk upward from when looking for ``pyproject.toml``.
    Defaults to the current working directory.
  environ : mapping or None, optional
    Environment to read. Defaults to :data:`os.environ`.

  Returns
  -------
  int
    Budget in bytes.

  Raises
  ------
  ValueError
    A configured value is malformed, not an int, or not positive. The
    message names the variable or file.

  """
  value, _source = memory_budget_source(
    explicit,
    headroom=headroom,
    start=start,
    environ=environ,
  )
  return value


def memory_budget_source(
  explicit: int | None = None,
  *,
  headroom: float = 1.0,
  start: Path | None = None,
  environ: Mapping[str, str] | None = None,
) -> tuple[int, str]:
  """Resolve the memory budget and name the layer that decided it.

  Parameters
  ----------
  explicit : int or None, optional
    Absolute byte count. Source ``"argument"`` when set.
  headroom : float, optional
    Applied only to the device layer and the 4 GiB default.
  start : pathlib.Path or None, optional
    Start of the ``pyproject.toml`` walk. Defaults to the current directory.
  environ : mapping or None, optional
    Environment to read. Defaults to :data:`os.environ`.

  Returns
  -------
  tuple[int, str]
    ``(bytes, source)``. ``source`` is ``"argument"``,
    ``"$AMINX_MEMORY_BUDGET_BYTES"``, ``"<path>:[tool.aminx]"``,
    ``"<path>:[runtime]"``, ``"device"``, or ``"default"``.

  Raises
  ------
  ValueError
    A configured value is malformed, not an int, or not positive.

  """
  if explicit is not None:
    return _positive_int(explicit, where="argument"), "argument"

  env = os.environ if environ is None else environ
  from_env = _env_budget(env)
  if from_env is not None:
    return from_env, f"${MEMORY_BUDGET_ENV}"

  origin = Path.cwd() if start is None else start
  from_project = _pyproject_budget(origin)
  if from_project is not None:
    return from_project

  from_user = _user_config_budget(env)
  if from_user is not None:
    return from_user

  from_device = _device_budget(headroom)
  if from_device is not None:
    return from_device, "device"

  value = int(_DEFAULT_LIMIT_BYTES * headroom)
  _warn_default_once(value, headroom)
  return value, "default"


def _positive_int(value: object, *, where: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    msg = f"{where}: memory_budget_bytes must be a positive int, got {value!r}"
    raise ValueError(msg)
  return value


def _env_budget(env: Mapping[str, str]) -> int | None:
  if MEMORY_BUDGET_ENV not in env:
    return None
  raw = env[MEMORY_BUDGET_ENV]
  stripped = raw.strip()
  if stripped == "" or stripped.lower() == "none":
    return None
  where = f"${MEMORY_BUDGET_ENV}"
  try:
    parsed = int(stripped)
  except ValueError as exc:
    msg = f"{where}: memory_budget_bytes must be a positive int, got {raw!r}"
    raise ValueError(msg) from exc
  return _positive_int(parsed, where=where)


def _load_toml(path: Path) -> dict[str, object]:
  try:
    return tomllib.loads(path.read_text(encoding="utf-8"))
  except tomllib.TOMLDecodeError as exc:
    msg = f"{path}: malformed TOML ({exc})"
    raise ValueError(msg) from exc


def _pyproject_budget(start: Path) -> tuple[int, str] | None:
  current = start.resolve()
  if current.is_file():
    current = current.parent
  for directory in (current, *current.parents):
    path = directory / "pyproject.toml"
    if not path.is_file():
      continue
    data = _load_toml(path)
    tool = data.get("tool")
    aminx = tool.get("aminx") if isinstance(tool, dict) else None
    if not isinstance(aminx, dict) or "memory_budget_bytes" not in aminx:
      return None
    where = f"{path}:[tool.aminx]"
    return _positive_int(aminx["memory_budget_bytes"], where=where), where
  return None


def _user_config_path(env: Mapping[str, str]) -> Path:
  xdg = env.get("XDG_CONFIG_HOME", "").strip()
  base = Path(xdg) if xdg else Path.home() / ".config"
  return base / "aminx" / "config.toml"


def _user_config_budget(env: Mapping[str, str]) -> tuple[int, str] | None:
  path = _user_config_path(env)
  if not path.is_file():
    return None
  data = _load_toml(path)
  runtime = data.get("runtime")
  if not isinstance(runtime, dict) or "memory_budget_bytes" not in runtime:
    return None
  where = f"{path}:[runtime]"
  return _positive_int(runtime["memory_budget_bytes"], where=where), where


def _device_budget(headroom: float) -> int | None:
  import xtrax.tiling.estimators as estimators  # noqa: PLC0415, PLR0402, TID251

  try:
    return int(estimators.device_memory_budget(fraction=headroom))
  except Exception:  # noqa: BLE001 -- missing stats (and any probe failure) fall through to the default
    return None


def _warn_default_once(value: int, headroom: float) -> None:
  if _default_warned[0]:
    return
  _default_warned[0] = True
  # INFO, not WARNING: CPU devices never report bytes_limit, so this is the normal CPU path;
  # memory_budget_source() reports the deciding layer for anyone who needs it.
  logger.info(
    "memory budget source is default (%s bytes = 4 GiB * headroom %s). "
    "Set it with an explicit argument, $%s, [tool.aminx] memory_budget_bytes "
    "in the nearest pyproject.toml, or [runtime] memory_budget_bytes in "
    "${XDG_CONFIG_HOME:-~/.config}/aminx/config.toml.",
    value,
    headroom,
    MEMORY_BUDGET_ENV,
  )
