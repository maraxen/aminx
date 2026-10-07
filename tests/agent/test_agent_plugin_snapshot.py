"""G-SNAPSHOT: ``cisternal assets snapshot --check`` and a missing-skill validate.

``python -m cisternal.cli`` does not dispatch (the module has no ``__main__``
guard). The console script is ``cisternal=cisternal.cli:app``, so these tests
call that cyclopts app in-process with the same argv.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("cisternal")

_REPO = Path(__file__).resolve().parents[2]


def invoke_cisternal(argv: list[str]) -> int:
  """Run ``cisternal.cli:app`` and return the process exit code.

  Parameters
  ----------
  argv : list of str
    Arguments after the program name, for example
    ``["assets", "snapshot", "--check", ...]``.

  Returns
  -------
  int
    ``0`` when the command returns, or the ``SystemExit`` code.
  """
  from cisternal.cli import app

  try:
    result = app(argv)
  except SystemExit as exc:
    code = exc.code
    if code is None or code == 0:
      return 0
    if isinstance(code, int):
      return code
    return 1
  if isinstance(result, int):
    return result
  return 0


def test_snapshot_check(monkeypatch: pytest.MonkeyPatch) -> None:
  """The packaged snapshot matches ``.praxia/manifest.toml``."""
  monkeypatch.chdir(_REPO)
  code = invoke_cisternal(
    [
      "assets",
      "snapshot",
      "--check",
      "--manifest",
      ".praxia/manifest.toml",
      "--out",
      "src/aminx/agent_plugin.json",
    ]
  )
  assert code == 0


def test_validate_missing_skill_exits_nonzero(tmp_path: Path) -> None:
  """A manifest whose skill file is absent fails ``assets validate``."""
  dest = tmp_path / ".praxia"
  dest.mkdir()
  shutil.copy(_REPO / ".praxia" / "manifest.toml", dest / "manifest.toml")
  code = invoke_cisternal(
    ["assets", "validate", "--manifest", str(dest / "manifest.toml")]
  )
  assert code != 0
