"""S7-06: the real aminx-mcp process answers list_tools over stdio."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastmcp")
pytest.importorskip("cisternal")

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from aminx.agent.tools import TOOL_NAMES


def test_stdio_server_lists_tools(tmp_path: Path) -> None:
  """A subprocess started with ``python -m aminx.agent.mcp`` lists the v1 tools."""
  env = dict(os.environ)
  env["CISTERNAL_LOG_DIR"] = str(tmp_path)
  transport = StdioTransport(
    command=sys.executable,
    args=["-m", "aminx.agent.mcp"],
    env=env,
  )

  async def body() -> set[str]:
    async with Client(transport) as client:
      listed = await client.list_tools()
      return {tool.name for tool in listed}

  assert asyncio.run(body()) == set(TOOL_NAMES)
