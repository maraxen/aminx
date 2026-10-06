"""G-SURFACE: the wired MCP tool list, schemas, and negative controls."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastmcp")
pytest.importorskip("cisternal")

import cisternal
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from aminx.agent.mcp import build_server
from aminx.agent.tools import REGISTRY, TOOL_NAMES

_PDB = Path(__file__).resolve().parents[1] / "data" / "1ubq.pdb"
_SAMPLE_PROPERTIES = {
  "inputs",
  "num_samples",
  "temperature",
  "random_seed",
  "checkpoint_id",
  "chain_id",
  "options",
  "output_dir",
  "inline_cap",
}


def _schema(tool: object) -> dict[str, Any]:
  """Return a tool's JSON input schema."""
  schema = getattr(tool, "input_schema", None)
  if schema is None:
    schema = getattr(tool, "inputSchema", None)
  if not isinstance(schema, dict):
    msg = "tool has no input schema"
    raise AssertionError(msg)
  return schema


def _data(result: object) -> dict[str, Any]:
  """Return the structured payload of a tool call."""
  data = getattr(result, "data", result)
  if isinstance(data, dict):
    return data
  dump = getattr(data, "model_dump", None)
  if callable(dump):
    dumped = dump()
    if isinstance(dumped, dict):
      return dumped
  msg = f"unexpected tool payload type {type(data).__name__}"
  raise AssertionError(msg)


def test_list_tools_matches_tool_names() -> None:
  """The wired server exposes exactly the v1 tool names."""

  async def body() -> set[str]:
    async with Client(build_server()) as client:
      listed = await client.list_tools()
      return {tool.name for tool in listed}

  assert asyncio.run(body()) == set(TOOL_NAMES)


def test_sample_and_score_schemas() -> None:
  """Sample and score schemas expose the typed parameters and required fields."""

  async def body() -> dict[str, dict[str, Any]]:
    async with Client(build_server()) as client:
      listed = await client.list_tools()
      return {tool.name: _schema(tool) for tool in listed}

  schemas = asyncio.run(body())
  sample = schemas["sample"]
  score = schemas["score"]
  assert set(sample["properties"]) >= _SAMPLE_PROPERTIES
  assert set(sample.get("required", [])) == {"inputs"}
  assert set(score.get("required", [])) == {"inputs", "sequences"}


def test_ghost_expected_name_raises() -> None:
  """wire() rejects an expected tool name that was never registered."""
  with pytest.raises(cisternal.CisternalWireError):
    cisternal.wire(
      FastMCP("neg"),
      registry=REGISTRY,
      expected=[*TOOL_NAMES, "ghost"],
    )


def test_unknown_option_raises_before_the_model() -> None:
  """An unknown options key is a ToolError and does not invoke the runner."""

  async def body() -> None:
    async with Client(build_server()) as client:
      with pytest.raises(ToolError, match="bogus"):
        await client.call_tool(
          "sample",
          {"inputs": [str(_PDB)], "options": {"bogus": 1}},
        )

  asyncio.run(body())


def test_duplicate_typed_option_names_the_key() -> None:
  """A key passed as a parameter and inside options names that key."""

  async def body() -> None:
    async with Client(build_server()) as client:
      with pytest.raises(ToolError, match="num_samples"):
        await client.call_tool(
          "sample",
          {
            "inputs": [str(_PDB)],
            "num_samples": 2,
            "options": {"num_samples": 3},
          },
        )

  asyncio.run(body())


def test_spec_validate_unknown_class_is_not_ok() -> None:
  """A document with an unknown spec class returns ok false."""

  async def body() -> dict[str, Any]:
    async with Client(build_server()) as client:
      result = await client.call_tool("spec_validate", {"spec": {"_spec_class": "Nope"}})
      return _data(result)

  payload = asyncio.run(body())
  assert payload["ok"] is False
  assert payload["errors"]


def test_spec_emit_sample_class() -> None:
  """spec_emit for sample returns a SamplingSpecification document."""

  async def body() -> dict[str, Any]:
    async with Client(build_server()) as client:
      result = await client.call_tool(
        "spec_emit",
        {"kind": "sample", "inputs": [str(_PDB)]},
      )
      return _data(result)

  payload = asyncio.run(body())
  assert payload["spec"]["_spec_class"] == "SamplingSpecification"


def test_list_checkpoints_includes_default() -> None:
  """The packaged checkpoint list includes the default ProteinMPNN id."""

  async def body() -> dict[str, Any]:
    async with Client(build_server()) as client:
      result = await client.call_tool("list_checkpoints", {})
      return _data(result)

  payload = asyncio.run(body())
  assert payload["default_checkpoint"] == "proteinmpnn_v_48_020"
  assert "proteinmpnn_v_48_020" in str(payload)


def test_known_checkpoint_ids_match_source_tree() -> None:
  """KNOWN_CHECKPOINT_IDS lists exactly the checkpoints tracked in src/aminx/model_params."""
  from aminx.agent.tools import KNOWN_CHECKPOINT_IDS  # noqa: PLC0415

  model_params = Path(__file__).resolve().parents[2] / "src" / "aminx" / "model_params"
  on_disk = {path.name.removesuffix(".eqx.zst") for path in model_params.glob("*.eqx.zst")}
  assert set(KNOWN_CHECKPOINT_IDS) == on_disk


def test_list_checkpoints_without_packaged_weights(monkeypatch: pytest.MonkeyPatch) -> None:
  """A wheel install (no packaged weights) still lists every checkpoint, marked unpackaged."""
  from aminx.agent import tools  # noqa: PLC0415

  monkeypatch.setattr(tools, "_packaged_checkpoint_ids", set)

  async def body() -> dict[str, Any]:
    async with Client(build_server()) as client:
      return _data(await client.call_tool("list_checkpoints", {}))

  entries = asyncio.run(body())["checkpoints"]
  assert {entry["checkpoint_id"] for entry in entries} == set(tools.KNOWN_CHECKPOINT_IDS)
  assert all(entry["packaged"] is False for entry in entries)


def test_options_may_set_a_field_whose_typed_parameter_is_unset() -> None:
  """Unset typed parameters do not collide with the same key inside options."""

  async def body() -> dict[str, Any]:
    async with Client(build_server()) as client:
      result = await client.call_tool(
        "spec_emit",
        {"kind": "sample", "inputs": [str(_PDB)], "options": {"temperature": 0.3}},
      )
      return _data(result)

  assert asyncio.run(body())["spec"]["temperature"] in (0.3, [0.3])


def test_sample_options_temperature_does_not_collide() -> None:
  """sample accepts temperature inside options when the typed parameter is unset."""

  async def body() -> None:
    async with Client(build_server()) as client:
      with pytest.raises(ToolError, match="bogus"):
        await client.call_tool(
          "sample",
          {"inputs": [str(_PDB)], "options": {"temperature": 0.3, "bogus": 1}},
        )

  # Fails on `bogus` (unknown key), not on a temperature collision: the collision check runs
  # first, so reaching the unknown-key error proves temperature was accepted.
  asyncio.run(body())
