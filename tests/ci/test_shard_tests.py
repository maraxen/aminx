from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "ci"))

import select_affected_tests  # noqa: E402
import shard_tests  # noqa: E402


def _make_tree(root: Path, sizes: dict[str, int]) -> None:
    for name, size in sizes.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x" * size)


def test_shards_partition_every_file_exactly_once(tmp_path: Path) -> None:
    _make_tree(tmp_path, {f"d{i % 3}/test_f{i}.py": 10 * (i + 1) for i in range(17)})
    files = shard_tests.discover(tmp_path)
    shards = shard_tests.shard(files, 4)
    flat = [p for s in shards for p in s]
    assert sorted(flat) == files
    assert len(flat) == len(set(flat))


def test_shards_are_balanced_and_deterministic(tmp_path: Path) -> None:
    _make_tree(tmp_path, {f"test_f{i}.py": 100 for i in range(12)})
    files = shard_tests.discover(tmp_path)
    first = shard_tests.shard(files, 4)
    assert first == shard_tests.shard(files, 4)
    assert {len(s) for s in first} == {3}


def test_discover_skips_non_tests(tmp_path: Path) -> None:
    _make_tree(tmp_path, {"test_a.py": 1, "b_test.py": 1, "helper.py": 1, "conftest.py": 1})
    assert [p.name for p in shard_tests.discover(tmp_path)] == ["b_test.py", "test_a.py"]


def test_cli_writes_targets(tmp_path: Path) -> None:
    _make_tree(tmp_path / "t", {"test_a.py": 5, "test_b.py": 5})
    out = tmp_path / "targets.txt"
    rc = shard_tests.main(["--index", "0", "--total", "1", "--root", str(tmp_path / "t"), "--write-pytest-args", str(out)])
    assert rc == 0
    assert len(out.read_text().splitlines()) == 2


def test_empty_shard_exits_zero_with_empty_file(tmp_path: Path) -> None:
    _make_tree(tmp_path / "t", {"test_a.py": 5, "test_b.py": 5})
    out = tmp_path / "targets.txt"
    # Two files across four shards: shards 2 and 3 are necessarily empty.
    rc = shard_tests.main(["--index", "3", "--total", "4", "--root", str(tmp_path / "t"), "--write-pytest-args", str(out)])
    assert rc == 0
    assert out.read_text() == ""


def test_zero_discovered_files_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "t").mkdir()
    out = tmp_path / "targets.txt"
    rc = shard_tests.main(["--index", "0", "--total", "1", "--root", str(tmp_path / "t"), "--write-pytest-args", str(out)])
    assert rc == shard_tests._EXIT_NO_FILES
    assert not out.exists()


def test_relative_root_resolves_against_repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "targets.txt"
    rc = shard_tests.main(["--index", "0", "--total", "1", "--root", "tests/ci", "--write-pytest-args", str(out)])
    assert rc == 0
    lines = out.read_text().splitlines()
    assert lines and all(Path(line).is_relative_to(_REPO / "tests" / "ci") for line in lines)


def test_discovery_agrees_with_affected_test_selector() -> None:
    sharded = {p.relative_to(_REPO).as_posix() for p in shard_tests.discover(_REPO / "tests")}
    selected = set(select_affected_tests.discover_test_files(_REPO))
    assert sharded == selected
