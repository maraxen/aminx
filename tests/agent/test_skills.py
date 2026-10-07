"""G-SKILLS harness: CLI verbs and MCP tool names in manifest skills.

CLI checks import the Typer app only. MCP name checks import
``aminx.agent.tools`` after skipping when cisternal or fastmcp is absent.
"""

from __future__ import annotations

import re
import shlex
import tomllib
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO / ".praxia" / "manifest.toml"
_TOOL_REF = re.compile(r"aminx-mcp:([A-Za-z_][A-Za-z0-9_]*)")


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


@lru_cache(maxsize=1)
def _cli_tree() -> dict[str, frozenset[str]]:
  """Return registered Typer groups mapped to their command names."""
  from aminx.cli import app

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


def _bash_aminx_lines(text: str) -> list[str]:
  """Return ``aminx ...`` lines inside ``bash`` fences."""
  lines: list[str] = []
  in_bash = False
  for raw in text.splitlines():
    stripped = raw.strip()
    if stripped.startswith("```"):
      in_bash = False if in_bash else stripped.startswith("```bash")
      continue
    if in_bash and stripped.startswith("aminx "):
      lines.append(stripped)
  return lines


def _bash_problems(text: str) -> list[str]:
  """Return problems for ``aminx`` lines whose group or command is unknown."""
  problems: list[str] = []
  tree = _cli_tree()
  from aminx.cli import app
  from typer.testing import CliRunner

  runner = CliRunner()
  for line in _bash_aminx_lines(text):
    tokens = [token for token in shlex.split(line) if not token.startswith("-")]
    group: str | None = None
    group_at: int | None = None
    for index, token in enumerate(tokens):
      if token in tree:
        group = token
        group_at = index
        break
    command: str | None = None
    if group is not None and group_at is not None:
      for token in tokens[group_at + 1 :]:
        if token in tree[group]:
          command = token
          break
    if group is None or command is None:
      problems.append(f"unknown command in: {line}")
      continue
    result = runner.invoke(app, [group, command, "--help"])
    if result.exit_code != 0:
      problems.append(f"aminx {group} {command} --help failed in: {line}")
  return problems


def _tool_problems(text: str) -> list[str]:
  """Return problems for ``aminx-mcp:<name>`` refs missing from ``TOOL_NAMES``."""
  names = _TOOL_REF.findall(text)
  if not names:
    return []
  pytest.importorskip("cisternal")
  pytest.importorskip("fastmcp")
  from aminx.agent.tools import TOOL_NAMES

  known = set(TOOL_NAMES)
  return [f"unknown MCP tool aminx-mcp:{name}" for name in names if name not in known]


def check_skill_text(text: str) -> list[str]:
  """Return CLI and MCP-tool problems in one skill body.

  Parameters
  ----------
  text : str
    Skill markdown, including frontmatter.

  Returns
  -------
  list of str
    Empty when every ``aminx`` bash line resolves and every ``aminx-mcp``
    name is a registered tool.
  """
  return [*_bash_problems(text), *_tool_problems(text)]


def _manifest_skills() -> list[dict[str, Any]]:
  """Return ``[[plugin.skills]]`` entries from the repo manifest."""
  document = tomllib.loads(_MANIFEST.read_text(encoding="utf-8"))
  plugin = document.get("plugin")
  if not isinstance(plugin, dict):
    return []
  skills = plugin.get("skills")
  if not isinstance(skills, list):
    return []
  return [entry for entry in skills if isinstance(entry, dict)]


def duplicate_trigger_problems(skills: list[Mapping[str, Any]]) -> list[str]:
  """Return problems where one trigger string is used by two skills.

  Parameters
  ----------
  skills : list of mapping
    Manifest skill entries with ``name`` and ``triggers``.

  Returns
  -------
  list of str
    One entry per repeated trigger.
  """
  owner: dict[str, str] = {}
  problems: list[str] = []
  for skill in skills:
    name = str(skill.get("name") or "")
    triggers = skill.get("triggers") or []
    if not isinstance(triggers, list):
      continue
    for trigger in triggers:
      text = str(trigger)
      previous = owner.get(text)
      if previous is not None:
        problems.append(f"trigger {text!r} appears in {previous!r} and {name!r}")
      else:
        owner[text] = name
  return problems


def test_manifest_skills_resolve() -> None:
  """Every manifest skill path, command, tool name, and trigger is valid."""
  skills = _manifest_skills()
  assert skills, "manifest lists no skills"
  for skill in skills:
    rel = skill.get("path")
    assert isinstance(rel, str) and rel, skill
    path = _REPO / rel
    assert path.is_file(), rel
    problems = check_skill_text(path.read_text(encoding="utf-8"))
    assert problems == [], problems
  assert duplicate_trigger_problems(skills) == []


def test_check_skill_text_negative_control() -> None:
  """A bogus run verb and a bogus MCP tool are both reported."""
  text = "\n".join(
    [
      "```bash",
      "aminx run frobnicate --inputs x.pdb",
      "```",
      "Use aminx-mcp:teleport when lost.",
    ]
  )
  problems = check_skill_text(text)
  joined = "\n".join(problems)
  assert "aminx run frobnicate --inputs x.pdb" in joined
  assert "aminx-mcp:teleport" in joined


def test_duplicate_triggers_are_reported() -> None:
  """The same trigger string in two skills is a problem."""
  skills: list[Mapping[str, Any]] = [
    {"name": "one", "triggers": ["aminx", "shared phrase"]},
    {"name": "two", "triggers": ["shared phrase"]},
  ]
  problems = duplicate_trigger_problems(skills)
  assert len(problems) == 1
  assert "shared phrase" in problems[0]
  assert "one" in problems[0]
  assert "two" in problems[0]
