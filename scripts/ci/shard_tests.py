"""Split the test files into balanced, deterministic shards for the CI matrix.

File size stands in for runtime (greedy longest-first onto the lightest shard). Every test file
lands in exactly one shard, so the shards together run the same set as one full-suite run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def discover(root: Path) -> list[Path]:
    """Return every pytest-collectable test module under ``root``, sorted."""
    found = set(root.rglob("test_*.py")) | set(root.rglob("*_test.py"))
    return sorted(p for p in found if "__pycache__" not in p.parts)


def shard(files: list[Path], total: int) -> list[list[Path]]:
    """Assign ``files`` to ``total`` shards, heaviest file first onto the lightest shard."""
    if total < 1:
        raise ValueError("total shards must be >= 1")
    shards: list[list[Path]] = [[] for _ in range(total)]
    loads = [0] * total
    for path in sorted(files, key=lambda p: (-p.stat().st_size, str(p))):
        i = min(range(total), key=lambda k: (loads[k], k))
        shards[i].append(path)
        loads[i] += path.stat().st_size
    return [sorted(s) for s in shards]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True, help="0-based shard index")
    parser.add_argument("--total", type=int, required=True, help="number of shards")
    parser.add_argument("--root", type=Path, default=Path("tests"))
    parser.add_argument("--write-pytest-args", type=Path, required=True, help="file to write one path per line")
    args = parser.parse_args(argv)

    if not 0 <= args.index < args.total:
        parser.error("--index must satisfy 0 <= index < total")
    selected = shard(discover(args.root), args.total)[args.index]
    if not selected:
        print(f"shard {args.index}/{args.total} is empty", file=sys.stderr)
        return 1
    args.write_pytest_args.write_text("".join(f"{p}\n" for p in selected))
    print(f"shard {args.index}/{args.total}: {len(selected)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
