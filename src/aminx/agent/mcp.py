"""Stdio MCP server for the aminx agent tools.

``python -m aminx.agent.mcp`` and the ``aminx-mcp`` console script both call
:func:`main`. ``cisternal`` and ``fastmcp`` are imported only after the agent
extra is confirmed present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aminx.agent import require_agent_extra

if TYPE_CHECKING:
  from fastmcp import FastMCP

_INSTRUCTIONS = (
  "Aminx is a ProteinMPNN and LigandMPNN inverse-folding model that designs "
  "and scores amino-acid sequences for protein backbones. Use sample to "
  "design sequences, score to evaluate sequences, inspect for logits and "
  "features, and jacobian for per-residue score gradients. spec_emit and "
  "spec_validate build and check run specifications without running the "
  "model, and list_checkpoints names the packaged weight files. Inputs are "
  "local structure file paths. Each run result includes the specification, "
  "provenance, and an optional npz side file when arrays are too large to "
  "return inline."
)


def build_server() -> FastMCP:
  """Build the FastMCP server with the aminx registry wired in.

  Returns
  -------
  FastMCP
    Server named ``aminx`` with the v1 tools and cisternal middleware.

  Raises
  ------
  ImportError
    The ``agent`` extra is not installed.
  """
  require_agent_extra()
  import cisternal  # noqa: PLC0415
  from cisternal.adapters.base import PassthroughAdapter  # noqa: PLC0415
  from cisternal.adapters.v3_middleware import CisternalMiddleware  # noqa: PLC0415
  from fastmcp import FastMCP as FastMCPServer  # noqa: PLC0415

  from aminx.agent import tools  # noqa: PLC0415

  server = FastMCPServer("aminx", instructions=_INSTRUCTIONS)
  cisternal.wire(
    server,
    registry=tools.REGISTRY,
    expected=list(tools.TOOL_NAMES),
  )
  server.add_middleware(CisternalMiddleware(adapter=PassthroughAdapter(), reraise=True))
  return server


def main() -> None:
  """Initialize cisternal and serve aminx over stdio."""
  require_agent_extra()
  import cisternal  # noqa: PLC0415

  cisternal.init()
  build_server().run()


if __name__ == "__main__":
  main()
