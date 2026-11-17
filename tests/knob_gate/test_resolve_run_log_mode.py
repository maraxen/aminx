"""default_resolve_run must resolve log-mode runs (bathos >= 0.15 writes no cool-tier parquet)."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _coverage  # noqa: E402

RUN_ID = "3f42635a-18a9-4bd3-aa6b-f2b037936258"


def _fake_bth(tmp_path: Path, payload: dict | None, *, rc: int = 0) -> Path:
  """A `bth` that records its argv and prints ``payload`` as JSON."""
  bindir = tmp_path / "bin"
  bindir.mkdir()
  script = bindir / "bth"
  out = json.dumps(payload) if payload is not None else "not json"
  script.write_text(
    "#!/usr/bin/env bash\n"
    f"printf '%s\\n' \"$@\" > {tmp_path / 'argv.txt'}\n"
    f"printf '%s' '{out}'\n"
    f"exit {rc}\n",
    encoding="utf-8",
  )
  script.chmod(script.stat().st_mode | stat.S_IEXEC)
  return bindir


@pytest.fixture
def empty_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  cat = tmp_path / "catalog"
  cat.mkdir()
  monkeypatch.setattr(_coverage, "default_catalog_dir", lambda: cat)
  return cat


def test_falls_back_to_bth_sql_when_no_parquet(
  tmp_path: Path, empty_catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  row = [
    "completed",
    "pass",
    False,
    "365d174da8293d65149620af90abd9ef42234c4f",
    "e213b3d0",
    json.dumps(["uv", "run", "--mutants", "a,b"]),
    json.dumps(["/x/caliby_parity_v2/365d174d/branch_controls.json"]),
  ]
  monkeypatch.setenv("PATH", f"{_fake_bth(tmp_path, {'rows': [row]})}:/usr/bin:/bin")
  run = _coverage.default_resolve_run(RUN_ID)
  assert run is not None
  assert run["outcome"] == "pass" and run["status"] == "completed" and run["git_dirty"] is False
  assert run["argv"][-1] == "a,b"
  assert run["output_paths"] == ["/x/caliby_parity_v2/365d174d/branch_controls.json"]
  argv = (tmp_path / "argv.txt").read_text(encoding="utf-8")
  assert RUN_ID in argv and "--catalog-dir" in argv


@pytest.mark.parametrize("payload", [{"rows": []}, {"rows": [["too", "short"]]}, None])
def test_no_row_or_bad_shape_is_none(
  tmp_path: Path, empty_catalog: Path, monkeypatch: pytest.MonkeyPatch, payload: dict | None
) -> None:
  monkeypatch.setenv("PATH", f"{_fake_bth(tmp_path, payload)}:/usr/bin:/bin")
  assert _coverage.default_resolve_run(RUN_ID) is None


def test_nonzero_exit_and_missing_binary_are_none(
  tmp_path: Path, empty_catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv("PATH", f"{_fake_bth(tmp_path, {'rows': []}, rc=2)}:/usr/bin:/bin")
  assert _coverage.default_resolve_run(RUN_ID) is None
  monkeypatch.setenv("PATH", "/nonexistent")
  assert _coverage.default_resolve_run(RUN_ID) is None


def test_non_uuid_id_never_reaches_sql(
  tmp_path: Path, empty_catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv("PATH", f"{_fake_bth(tmp_path, {'rows': []})}:/usr/bin:/bin")
  assert _coverage.default_resolve_run("x' or '1'='1") is None
  assert not (tmp_path / "argv.txt").exists()
