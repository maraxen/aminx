# ruff: noqa: S101
"""stale_rows forwards --head into the uv.lock comparison.

Pre-fix, assess() called row_is_stale without head, so a what-if against another ref compared the
working-tree uv.lock. Not run on this box; the remote suite is its first execution.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _load_stale_rows() -> ModuleType:
  path = _REPO / "scripts" / "redsox" / "stale_rows.py"
  spec = importlib.util.spec_from_file_location("stale_rows_under_test", path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def test_head_reaches_the_uv_lock_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
  stale_rows = _load_stale_rows()
  seen: dict[str, object] = {}

  def fake_row_is_stale(
    paths: list[str],
    closure: object,
    *,
    repo: Path | None = None,
    git_hash: str | None = None,
    head: str | None = None,
    **_kwargs: object,
  ) -> bool:
    seen.update(paths=list(paths), git_hash=git_hash, head=head)
    return True

  monkeypatch.setattr(stale_rows, "_diff_names", lambda _base, _head: ["uv.lock"])
  monkeypatch.setattr(stale_rows, "default_closure_for", lambda _slug: None)
  monkeypatch.setattr(stale_rows, "row_is_stale", fake_row_is_stale)
  stale_rows.assess("potts_x", None, run_commit="abc123", head="refX")
  assert seen == {"paths": ["uv.lock"], "git_hash": "abc123", "head": "refX"}
