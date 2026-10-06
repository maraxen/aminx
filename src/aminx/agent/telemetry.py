"""Opt-in CLI telemetry.

Spans are emitted only when ``CISTERNAL_TELEMETRY`` is one of
:data:`ENABLE_VALUES`. cisternal is imported inside the functions that need
it, so a disabled CLI does not import it.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache

ENABLE_VALUES: frozenset[str] = frozenset({"aminx", "all", "1", "true", "yes"})

# The listener thread can exit on its stop flag before it has written the
# events already queued. A short pause lets it drain; shutdown then joins it
# so the process does not wait on that non-daemon thread.
_DRAIN_SECONDS = 0.3


def cli_telemetry_enabled(environ: Mapping[str, str] | None = None) -> bool:
  """Return whether CLI telemetry should run.

  Parameters
  ----------
  environ : mapping or None, optional
    Environment to read. ``None`` reads :data:`os.environ`.

  Returns
  -------
  bool
    True when ``CISTERNAL_TELEMETRY``, compared case-insensitively after
    stripping, is in :data:`ENABLE_VALUES`.
  """
  env = os.environ if environ is None else environ
  raw = env.get("CISTERNAL_TELEMETRY", "")
  return raw.strip().lower() in ENABLE_VALUES


def command_name(argv: Sequence[str]) -> str:
  """Return ``group`` or ``group.command`` from registered Typer names.

  The scan looks for a registered group name, then for a registered command
  of that group later in ``argv``. Option tokens and their values are never
  copied into the result. Paths and sequences are not included.

  Parameters
  ----------
  argv : sequence of str
    Argument vector, including an optional program name.

  Returns
  -------
  str
    ``group.command``, the group alone when no command follows, or
    ``unknown``.
  """
  tree = _cli_tree()
  group: str | None = None
  group_at: int | None = None
  for index, token in enumerate(argv):
    if token in tree:
      group = token
      group_at = index
      break
  if group is None or group_at is None:
    return "unknown"
  commands = tree[group]
  for token in argv[group_at + 1 :]:
    if token in commands:
      return f"{group}.{token}"
  return group


def run_cli(app: Callable[[], object], argv: Sequence[str] | None = None) -> None:
  """Run ``app``, wrapping it in a CLI span when telemetry is enabled.

  Parameters
  ----------
  app : callable
    Zero-argument CLI entry point, usually the Typer app.
  argv : sequence of str or None, optional
    Argument vector used only to name the span. ``None`` uses
    :data:`sys.argv`.

  Returns
  -------
  None
  """
  if not cli_telemetry_enabled():
    app()
    return
  try:
    import cisternal  # noqa: PLC0415
  except ImportError:
    app()
    return
  tokens = list(sys.argv if argv is None else argv)
  cisternal.init()
  try:
    with cisternal.span(f"aminx.cli.{command_name(tokens)}"):
      try:
        app()
      except SystemExit as exc:
        cisternal.emit_event("aminx.cli.exit", code=_exit_code(exc))
        raise
  finally:
    _shutdown_cisternal()


def _exit_code(exc: SystemExit) -> int:
  """Return the integer status carried by ``exc``."""
  code = exc.code
  if code is None or code == 0:
    return 0
  if isinstance(code, int):
    return code
  return 1


def _shutdown_cisternal() -> None:
  """Drain queued events and stop the cisternal listener thread."""
  try:
    from cisternal.telemetry import shutdown_pipeline  # noqa: PLC0415
  except ImportError:
    return
  time.sleep(_DRAIN_SECONDS)
  shutdown_pipeline()


@lru_cache(maxsize=1)
def _cli_tree() -> dict[str, frozenset[str]]:
  """Return registered Typer group names mapped to their command names."""
  from aminx.cli import app  # noqa: PLC0415

  tree: dict[str, frozenset[str]] = {}
  for group in app.registered_groups:
    group_name = _cli_label(group)
    sub = getattr(group, "typer_instance", None)
    if group_name is None or sub is None:
      continue
    commands: set[str] = set()
    for info in getattr(sub, "registered_commands", ()):
      label = _cli_label(info)
      if label is not None:
        commands.add(label)
    tree[group_name] = frozenset(commands)
  return tree


def _cli_label(info: object) -> str | None:
  """Return a Typer command or group name."""
  name = getattr(info, "name", None)
  if isinstance(name, str) and name:
    return name
  callback = getattr(info, "callback", None)
  raw = getattr(callback, "__name__", None)
  if isinstance(raw, str) and raw:
    return raw.replace("_", "-")
  return None


__all__ = [
  "ENABLE_VALUES",
  "cli_telemetry_enabled",
  "command_name",
  "run_cli",
]
