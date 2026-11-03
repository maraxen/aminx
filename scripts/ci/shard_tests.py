"""Split the test files into balanced, deterministic shards for the CI matrix.

File size stands in for runtime (greedy longest-first onto the lightest shard). Every test file
lands in exactly one shard, so the shards together run the same set as one full-suite run.

Limitation: no recorded per-test or per-file runtime exists in the repository (no committed timing
file, no pytest durations artifact), so file size is only a proxy. A shard can run much longer than
its peers when a few large files are slow. Replace the weight in ``shard`` once a timing source exists.

Discovery uses the same name rule as ``scripts/ci/select_affected_tests.is_test_file``
(``test_*.py`` and ``*_test.py``). ``tests/ci/test_shard_tests.py`` asserts the two agree on this repo.

Exit codes: 0 when the shard was written, including an empty shard (its file is empty and CI skips
pytest). 2 when no test files are discovered at all, so a wrong ``--root`` fails loudly. A relative
``--root`` resolves against the repository root, not the current directory.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXIT_NO_FILES = 2

log = logging.getLogger("shard_tests")


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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=int, required=True, help="0-based shard index")
    parser.add_argument("--total", type=int, required=True, help="number of shards")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("tests"),
        help="directory to scan; a relative path resolves against the repository root",
    )
    parser.add_argument("--write-pytest-args", type=Path, required=True, help="file to write one path per line")
    args = parser.parse_args(argv)

    if not 0 <= args.index < args.total:
        parser.error("--index must satisfy 0 <= index < total")
    root = args.root if args.root.is_absolute() else _REPO_ROOT / args.root
    discovered = discover(root)
    if not discovered:
        log.error("no test files discovered under %s; refusing to write an empty plan", root)
        return _EXIT_NO_FILES
    selected = shard(discovered, args.total)[args.index]
    args.write_pytest_args.write_text("".join(f"{p}\n" for p in selected))
    if not selected:
        log.info("shard %d/%d is empty (%d files discovered); nothing to run", args.index, args.total, len(discovered))
        return 0
    log.info("shard %d/%d: %d of %d files", args.index, args.total, len(selected), len(discovered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
