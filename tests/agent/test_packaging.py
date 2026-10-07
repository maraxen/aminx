"""G-LOCK (static) and G-IMPORT checks for the agent extra.

Loaded from source files so collection does not execute ``aminx/__init__.py``.
The import-time probe is the one place a test runs ``import aminx``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[2]
_INIT = _REPO / "src" / "aminx" / "agent" / "__init__.py"
_GUIDANCE = (
  'aminx agent features need the agent extra: pip install "aminx[agent]"'
  " (or uv sync --extra agent)"
)


def _load_init() -> ModuleType:
  """Load ``aminx.agent`` without importing the jax-heavy package init."""
  spec = importlib.util.spec_from_file_location("aminx_agent_init", _INIT)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def _dist_name(requirement: str) -> str:
  """Return the distribution name of a PEP 508 requirement string."""
  name = requirement.split(";", 1)[0].strip()
  name = name.split("[", 1)[0]
  for separator in ("===", "==", "!=", "~=", ">=", "<=", ">", "<"):
    if separator in name:
      name = name.split(separator, 1)[0]
  return name.strip().lower().replace("_", "-")


_agent = _load_init()


def heavy_modules_in_importtime(stderr: str) -> set[str]:
  """Return agent-extra top-level modules named in importtime text.

  Parameters
  ----------
  stderr : str
    ``python -X importtime`` stderr, or synthetic text in that shape.

  Returns
  -------
  set of str
    Subset of ``AGENT_EXTRA_MODULES`` whose top-level name was imported.
    ``fastmcp.server`` counts as ``fastmcp``.
  """
  watched = set(_agent.AGENT_EXTRA_MODULES)
  found: set[str] = set()
  for line in stderr.splitlines():
    module = line.rsplit("|", 1)[-1].strip() if "|" in line else line.strip()
    if not module or " " in module:
      continue
    top = module.split(".", 1)[0]
    if top in watched:
      found.add(top)
  return found


def test_agent_extra_pins_cisternal_and_base_deps_do_not() -> None:
  """The agent extra lower-bounds cisternal; base deps omit the extra."""
  with (_REPO / "pyproject.toml").open("rb") as handle:
    data = tomllib.load(handle)
  project = data["project"]
  extras = project["optional-dependencies"]
  assert "agent" in extras
  agent = extras["agent"]
  assert isinstance(agent, list)
  cisternal = [req for req in agent if _dist_name(str(req)) == "cisternal"]
  assert len(cisternal) == 1
  assert ">=0.1.1a15" in str(cisternal[0]).replace(" ", "")
  base = project["dependencies"]
  assert isinstance(base, list)
  names = {_dist_name(str(req)) for req in base}
  assert names.isdisjoint({"cisternal", "fastmcp", "cyclopts"})
  assert _agent.AGENT_EXTRA_MODULES == ("cisternal", "fastmcp", "cyclopts")


def test_import_aminx_does_not_import_agent_extra_modules() -> None:
  """``import aminx`` importtime lists no cisternal, fastmcp, or cyclopts."""
  result = subprocess.run(
    [sys.executable, "-X", "importtime", "-c", "import aminx"],
    capture_output=True,
    text=True,
    check=False,
  )
  assert result.returncode == 0, result.stderr
  assert heavy_modules_in_importtime(result.stderr) == set()


def test_importtime_negative_control_detects_synthetic_heavy_modules() -> None:
  """The import gate flags synthetic cisternal and fastmcp.server lines."""
  synthetic = "\n".join(
    (
      "import time: self [us] | cumulative | imported package",
      "import time:       100 |        100 |   cisternal",
      "import time:        40 |        140 |     fastmcp.server",
    ),
  )
  assert heavy_modules_in_importtime(synthetic) == {"cisternal", "fastmcp"}


def test_require_agent_extra_names_the_install_command(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A missing cisternal spec raises the agent-extra install guidance."""
  real = importlib.util.find_spec

  def _find_spec(name: str, package: str | None = None) -> object:
    if name == "cisternal":
      return None
    return real(name, package)

  monkeypatch.setattr(importlib.util, "find_spec", _find_spec)
  with pytest.raises(ImportError, match=r"aminx\[agent\]") as exc_info:
    _agent.require_agent_extra()
  assert str(exc_info.value) == _GUIDANCE
