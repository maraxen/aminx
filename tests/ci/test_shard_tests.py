from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "ci"))

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
