"""Publish and describe the aminx Claude plugin.

``python -m aminx.agent.plugin publish`` loads the cisternal bundle, pins the
MCP server command to the resolved launch mode, and installs into the shared
marketplace. ``info`` prints that resolution and does not write.

``cisternal`` is imported only after :func:`aminx.agent.require_agent_extra`
inside a function, so importing this module does not load the agent extra.
``PLUGIN_SPEC`` is the same object, built on first access.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aminx.agent import require_agent_extra
from aminx.agent.config import LaunchConfig, resolve_launch

if TYPE_CHECKING:
  from cisternal.assets.bundle import AssetBundle
  from cisternal.plugin.app import PluginSpec

_SPEC_FIELDS: dict[str, Any] = {
  "name": "aminx",
  "package": "aminx",
  "registry": "aminx",
  "imports": ("aminx.agent.tools",),
  "distribution": "aminx",
  "cli": "python -m aminx.agent.plugin",
}
_PLUGIN_SPEC: PluginSpec | None = None


def __getattr__(name: str) -> PluginSpec:
  """Build :data:`PLUGIN_SPEC` on first access.

  Parameters
  ----------
  name : str
    Module attribute. Only ``PLUGIN_SPEC`` is served here.

  Returns
  -------
  PluginSpec
    The aminx plugin spec.

  Raises
  ------
  AttributeError
    ``name`` is not ``PLUGIN_SPEC``.
  ImportError
    The agent extra is not installed.
  """
  if name == "PLUGIN_SPEC":
    return plugin_spec()
  msg = f"module {__name__!r} has no attribute {name!r}"
  raise AttributeError(msg)


def plugin_spec() -> PluginSpec:
  """Return the cisternal plugin spec for aminx.

  Returns
  -------
  PluginSpec
    Name, package, registry, and the ``aminx.agent.tools`` import.

  Raises
  ------
  ImportError
    The agent extra is not installed.
  """
  global _PLUGIN_SPEC  # noqa: PLW0603
  require_agent_extra()
  if _PLUGIN_SPEC is None:
    from cisternal.plugin import PluginSpec as _PluginSpec  # noqa: PLC0415

    _PLUGIN_SPEC = _PluginSpec(**_SPEC_FIELDS)
  return _PLUGIN_SPEC


def _public_version(version: str) -> str:
  """Drop a ``+local`` suffix that no index can resolve."""
  return version.split("+", 1)[0]


def _interpreter(python: Path) -> str:
  """Return an absolute interpreter path."""
  if python.is_absolute():
    return str(python)
  return str(python.resolve())


def launch_command(cfg: LaunchConfig, version: str) -> tuple[str, ...]:
  """Argv for the MCP server under ``cfg``.

  Parameters
  ----------
  cfg : LaunchConfig
    Resolved launch mode. ``venv`` reads ``cfg.python``.
  version : str
    Bundle version. A ``+local`` suffix is omitted from the ``uvx`` pin.

  Returns
  -------
  tuple of str
    ``uvx --from aminx[agent]==<version> aminx-mcp``, or
    ``<absolute python> -m aminx.agent.mcp``.

  Raises
  ------
  ValueError
    ``cfg.mode`` is not ``uvx`` or ``venv``, or ``venv`` has no interpreter.
  """
  if cfg.mode == "uvx":
    public = _public_version(version)
    return ("uvx", "--from", f"aminx[agent]=={public}", "aminx-mcp")
  if cfg.mode == "venv":
    if cfg.python is None:
      msg = "venv launch has no interpreter"
      raise ValueError(msg)
    return (_interpreter(cfg.python), "-m", "aminx.agent.mcp")
  msg = f"unknown launch mode {cfg.mode!r}"
  raise ValueError(msg)


def version_on_index(version: str, *, timeout: float = 10.0) -> bool | None:
  """Whether ``aminx==version`` is on PyPI.

  Parameters
  ----------
  version : str
    Version segment of ``https://pypi.org/pypi/aminx/<version>/json``.
  timeout : float, optional
    Socket timeout in seconds.

  Returns
  -------
  bool or None
    ``True`` on HTTP 200, ``False`` on HTTP 404, ``None`` when the index
    cannot be reached or returns another status.
  """
  url = f"https://pypi.org/pypi/aminx/{version}/json"
  try:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
      return response.status == 200
  except urllib.error.HTTPError as exc:
    if exc.code == 404:
      return False
    return None
  except (urllib.error.URLError, TimeoutError, OSError):
    return None


def check_venv_python(python: Path) -> str | None:
  """Try ``import aminx.agent.mcp`` with ``python``.

  Parameters
  ----------
  python : Path
    Interpreter to run.

  Returns
  -------
  str or None
    ``None`` when the import exits 0. Otherwise an error that includes the
    tail of stderr.
  """
  try:
    result = subprocess.run(  # noqa: S603
      [str(python), "-c", "import aminx.agent.mcp"],
      capture_output=True,
      text=True,
      timeout=120,
      check=False,
    )
  except (OSError, subprocess.TimeoutExpired) as exc:
    return f"interpreter {python} could not import aminx.agent.mcp: {exc}"
  if result.returncode == 0:
    return None
  tail = (result.stderr or "").strip()[-400:]
  return f"interpreter {python} could not import aminx.agent.mcp (exit {result.returncode}): {tail}"


def _require_plugin_error() -> type[Exception]:
  """Import :class:`cisternal.plugin.PluginError` after the extra check."""
  require_agent_extra()
  from cisternal.plugin import PluginError  # noqa: PLC0415

  return PluginError


def prepare(
  launch: str | None = None,
  python: str | Path | None = None,
  *,
  start: Path | None = None,
  environ: Mapping[str, str] | None = None,
  allow_unpublished: bool = False,
) -> tuple[AssetBundle, dict[str, Any]]:
  """Load the bundle and pin its MCP command to the resolved launch.

  Parameters
  ----------
  launch : str or None, optional
    Explicit ``uvx`` or ``venv``. ``None`` defers to a lower layer.
  python : str or Path or None, optional
    Explicit interpreter for ``venv``.
  start : Path or None, optional
    Directory walked for ``pyproject.toml``.
  environ : mapping or None, optional
    Environment for launch resolution. ``None`` uses :data:`os.environ`.
  allow_unpublished : bool, optional
    When true, ``uvx`` proceeds if the version is missing or the index is down.

  Returns
  -------
  tuple of AssetBundle and dict
    The rewritten bundle and a record of the launch decision.

  Raises
  ------
  PluginError
    ``uvx`` cannot see the version on the index, or the ``venv`` interpreter
    cannot import ``aminx.agent.mcp``.
  ValueError
    The launch configuration is malformed.
  ImportError
    The agent extra is not installed.
  """
  plugin_error = _require_plugin_error()
  from cisternal.plugin import load_bundle  # noqa: PLC0415

  cfg = resolve_launch(launch, python, start=start, environ=environ)
  bundle, source = load_bundle(plugin_spec())
  if len(bundle.mcp_servers) != 1:
    msg = f"expected one MCP server, found {len(bundle.mcp_servers)}"
    raise plugin_error(msg)
  server = bundle.mcp_servers[0]
  version = bundle.metadata.version
  public = _public_version(version)
  if cfg.mode == "uvx":
    indexed = version_on_index(public)
    if indexed is False and not allow_unpublished:
      msg = f"aminx {public} is not on the index; pass --launch venv"
      raise plugin_error(msg)
    if indexed is None and not allow_unpublished:
      msg = f"the PyPI index could not be reached for aminx {public} (uvx needs it too)"
      raise plugin_error(msg)
  else:
    if cfg.python is None:
      msg = "venv launch has no interpreter"
      raise plugin_error(msg)
    failure = check_venv_python(cfg.python)
    if failure is not None:
      raise plugin_error(failure)
  command = launch_command(cfg, version)
  rewritten = replace(server, command=command, launch="path", uvx_from="")
  bundle = replace(bundle, mcp_servers=(rewritten,))
  record: dict[str, Any] = {
    "launch_mode": cfg.mode,
    "launch_source": cfg.source,
    "python": None if cfg.python is None else _interpreter(cfg.python),
    "command": list(command),
    "bundle_source": source.describe(),
    "aminx_version": version,
  }
  return bundle, record


def publish(
  launch: str | None = None,
  python: str | Path | None = None,
  *,
  marketplace: str | Path | None = None,
  scope: str = "user",
  claude_bin: str = "claude",
  prune_shadowed: bool = False,
  dry_run: bool = False,
  allow_unpublished: bool = False,
  start: Path | None = None,
  environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
  """Install the plugin into the shared marketplace.

  Parameters
  ----------
  launch : str or None, optional
    Explicit launch mode.
  python : str or Path or None, optional
    Explicit ``venv`` interpreter.
  marketplace : str or Path or None, optional
    Marketplace root. ``None`` resolves the shared root.
  scope : str, optional
    Claude install scope: ``user``, ``project``, or ``local``.
  claude_bin : str, optional
    Claude Code executable.
  prune_shadowed : bool, optional
    Ask cisternal to move shadowed user-level skills aside.
  dry_run : bool, optional
    Resolve and return the record without writing.
  allow_unpublished : bool, optional
    Allow ``uvx`` when the index lacks the version or is unreachable.
  start : Path or None, optional
    Directory walked for ``pyproject.toml``.
  environ : mapping or None, optional
    Environment for launch resolution.

  Returns
  -------
  dict
    The prepare record plus ``marketplace``. Dry runs also set ``dry_run``.

  Raises
  ------
  PluginError
    Launch validation failed, or the Claude install failed.
  ValueError
    Launch or marketplace configuration is malformed.
  ImportError
    The agent extra is not installed.
  """
  bundle, record = prepare(
    launch,
    python,
    start=start,
    environ=environ,
    allow_unpublished=allow_unpublished,
  )
  require_agent_extra()
  from cisternal.plugin import marketplace_root_source, resolve_marketplace_root  # noqa: PLC0415
  from cisternal.plugin.app import claude_install  # noqa: PLC0415

  if dry_run:
    # resolve_marketplace_root writes a default entry to the user's cisternal
    # config when none exists; a dry run must not write anything.
    found, found_source = marketplace_root_source(marketplace)
    return {
      "dry_run": True,
      **record,
      "marketplace": None if found is None else str(found),
      "marketplace_source": found_source,
    }
  root = resolve_marketplace_root(marketplace)
  claude_install(
    bundle,
    record,
    marketplace=root,
    scope=scope,
    claude_bin=claude_bin,
    require_installed=False,
    prune_shadowed=prune_shadowed,
  )
  return {**record, "marketplace": str(root)}


def info(
  *,
  start: Path | None = None,
  environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
  """Describe the launch and marketplace that publish would use.

  Parameters
  ----------
  start : Path or None, optional
    Directory walked for ``pyproject.toml``.
  environ : mapping or None, optional
    Environment for launch resolution.

  Returns
  -------
  dict
    Launch mode and source, the command, the interpreter, the marketplace
    root and its source, and the bundle source. Nothing is written.

  Raises
  ------
  ValueError
    The launch configuration is malformed.
  ImportError
    The agent extra is not installed.
  """
  require_agent_extra()
  import importlib.metadata  # noqa: PLC0415

  from cisternal.plugin import locate_bundle, marketplace_root_source  # noqa: PLC0415

  cfg = resolve_launch(start=start, environ=environ)
  try:
    version = importlib.metadata.version("aminx")
  except importlib.metadata.PackageNotFoundError:
    version = "0.0.0"
  command = launch_command(cfg, version)
  root, marketplace_source = marketplace_root_source()
  return {
    "launch_mode": cfg.mode,
    "launch_source": cfg.source,
    "command": list(command),
    "python": None if cfg.python is None else _interpreter(cfg.python),
    "marketplace": None if root is None else str(root),
    "marketplace_source": marketplace_source,
    "bundle_source": locate_bundle(plugin_spec()).describe(),
  }


def _parser() -> argparse.ArgumentParser:
  """Build the ``publish`` / ``info`` argument parser."""
  parser = argparse.ArgumentParser(prog="python -m aminx.agent.plugin")
  sub = parser.add_subparsers(dest="command", required=True)
  publish_parser = sub.add_parser("publish")
  publish_parser.add_argument("--launch", choices=["uvx", "venv"])
  publish_parser.add_argument("--python", dest="python_path")
  publish_parser.add_argument("--marketplace")
  publish_parser.add_argument("--scope", choices=["user", "project", "local"], default="user")
  publish_parser.add_argument("--prune-shadowed", action="store_true")
  publish_parser.add_argument("--dry-run", action="store_true")
  publish_parser.add_argument("--allow-unpublished", action="store_true")
  sub.add_parser("info")
  return parser


def _is_plugin_error(exc: BaseException) -> bool:
  """Return whether ``exc`` is cisternal's :class:`PluginError`."""
  return type(exc).__name__ == "PluginError" and type(exc).__module__.startswith("cisternal")


def _emit_error(exc: BaseException) -> int:
  """Print ``error:`` and return 1."""
  print(f"error: {exc}", file=sys.stderr)  # noqa: T201
  return 1


def main(argv: list[str] | None = None) -> int:
  """Run ``publish`` or ``info`` and print a JSON object.

  Parameters
  ----------
  argv : list of str or None, optional
    Arguments, excluding the program name. ``None`` reads :data:`sys.argv`.

  Returns
  -------
  int
    ``0`` after printing JSON. ``1`` after ``error:`` on stderr for
    :class:`PluginError`, :class:`ValueError`, or :class:`ImportError`.
  """
  args = _parser().parse_args(argv)
  try:
    if args.command == "info":
      payload = info()
    else:
      payload = publish(
        args.launch,
        args.python_path,
        marketplace=args.marketplace,
        scope=args.scope,
        prune_shadowed=args.prune_shadowed,
        dry_run=args.dry_run,
        allow_unpublished=args.allow_unpublished,
      )
  except (ValueError, ImportError) as exc:
    return _emit_error(exc)
  except Exception as exc:
    if not _is_plugin_error(exc):
      raise
    return _emit_error(exc)
  print(json.dumps(payload))  # noqa: T201
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
