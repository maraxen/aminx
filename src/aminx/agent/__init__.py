"""Agent-facing layer for coding agents.

Hosts the MCP server, the plugin bundle, launch configuration, and provenance
capture that make aminx usable through cisternal. Importing this package does
not import ``cisternal``, ``fastmcp``, or ``cyclopts``; those packages belong
to the optional ``agent`` extra and are loaded only by later entry points.
"""

from __future__ import annotations

import importlib.util

AGENT_EXTRA_MODULES: tuple[str, ...] = ("cisternal", "fastmcp", "cyclopts")

_AGENT_EXTRA_GUIDANCE = (
  'aminx agent features need the agent extra: pip install "aminx[agent]" (or uv sync --extra agent)'
)


def require_agent_extra() -> None:
  """Raise if the optional agent extra is not installed.

  ``cisternal`` and ``fastmcp`` are the runtime dependencies of the MCP
  entry points. ``cyclopts`` is part of the same extra (see
  ``AGENT_EXTRA_MODULES``) but is not required until a CLI port exists.

  Returns
  -------
  None
    Returns when both ``cisternal`` and ``fastmcp`` can be found.

  Raises
  ------
  ImportError
    When either module is missing. The message is the install guidance for
    ``aminx[agent]``.
  """
  for name in ("cisternal", "fastmcp"):
    if importlib.util.find_spec(name) is None:
      raise ImportError(_AGENT_EXTRA_GUIDANCE)


__all__ = ["AGENT_EXTRA_MODULES", "require_agent_extra"]
