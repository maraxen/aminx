"""Provenance capture tiers: cisternal, builtin git, and unknown.

``provenance.py`` is loaded from its file so collection does not execute
``aminx/__init__.py``. Git tests skip when ``git`` is not on ``PATH``.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_PROVENANCE = Path(__file__).resolve().parents[2] / "src" / "aminx" / "agent" / "provenance.py"
_SHA = re.compile(r"^[0-9a-f]{40}$")


def _load_provenance() -> ModuleType:
  """Load ``provenance.py`` without importing the jax-heavy package init."""
  spec = importlib.util.spec_from_file_location("aminx_agent_provenance", _PROVENANCE)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


_provenance = _load_provenance()


def _require_git() -> None:
  """Skip the test when ``git`` cannot be executed."""
  if shutil.which("git") is None:
    pytest.skip("git not on PATH")


def _git(repo: Path, *args: str) -> None:
  """Run git in ``repo`` with a fixed author identity."""
  env = os.environ.copy()
  env["GIT_AUTHOR_NAME"] = "aminx-test"
  env["GIT_AUTHOR_EMAIL"] = "aminx-test@example.com"
  env["GIT_COMMITTER_NAME"] = "aminx-test"
  env["GIT_COMMITTER_EMAIL"] = "aminx-test@example.com"
  subprocess.run(
    ["git", "-C", str(repo), *args],
    check=True,
    capture_output=True,
    text=True,
    env=env,
  )


def _init_repo(tmp_path: Path) -> Path:
  """Create a one-commit repository and return its path."""
  _require_git()
  repo = tmp_path / "repo"
  repo.mkdir()
  (repo / "tracked.txt").write_text("alpha\n", encoding="utf-8")
  _git(repo, "init")
  _git(repo, "add", "tracked.txt")
  _git(repo, "commit", "-m", "init")
  return repo


def test_clean_repo_has_sha_and_dirty_tree_is_detected(tmp_path: Path) -> None:
  """A commit yields a 40-hex sha; editing the file sets ``dirty`` (negative)."""
  repo = _init_repo(tmp_path)
  clean = _provenance.capture(root=repo)
  assert isinstance(clean["sha"], str)
  assert _SHA.fullmatch(clean["sha"])
  assert clean["dirty"] is False
  source = clean["provenance_source"]
  assert source == "builtin" or (isinstance(source, str) and source.startswith("cisternal:"))
  (repo / "tracked.txt").write_text("alpha\nbeta\n", encoding="utf-8")
  dirty = _provenance.capture(root=repo)
  assert dirty["dirty"] is True
  assert isinstance(dirty["sha"], str)
  assert _SHA.fullmatch(dirty["sha"])


def test_non_repo_is_unknown(tmp_path: Path) -> None:
  """A directory that is not a git repo yields no sha and source ``unknown``."""
  _require_git()
  state = _provenance.capture(root=tmp_path)
  assert state["sha"] is None
  assert state["dirty"] is None
  assert state["branch"] is None
  assert state["provenance_source"] == "unknown"


def test_builtin_tier_when_cisternal_provenance_is_unavailable(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Forcing ``cisternal.provenance`` to ``None`` uses the builtin git tier."""
  repo = _init_repo(tmp_path)
  monkeypatch.setitem(sys.modules, "cisternal.provenance", None)
  state = _provenance.capture(root=repo)
  assert state["provenance_source"] == "builtin"
  assert isinstance(state["sha"], str)
  assert _SHA.fullmatch(state["sha"])
  assert state["dirty"] is False
