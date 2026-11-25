"""Decide how the unit-test matrix is split for one CI run.

Run by the `select` job in .github/workflows/ci.yml. Standard library only, so it runs before any
`uv sync`. It reads the selector's `--json` output on pull requests and nothing on pushes.

Modes:
- push-full: a push to main. Every shard runs the full suite (scripts/ci/shard_tests.py).
- pr-full: a pull request whose diff selection covers at least `threshold` of the test files, or
  where the selector fell back to the full suite (mode "full", or no test files known). Every shard
  runs the full suite, without coverage.
- pr-selected: a pull request with a small selection. One job runs the selected tests.

Outputs, written to `--github-output` when given (otherwise printed):
- mode: one of the three modes above.
- shards: JSON array of shard indices, e.g. [0,1,2,3] or [0].
- shard_total: len(shards). In a full mode this is the `--total` every shard passes to the sharder,
  and it is also the number of matrix legs. The aggregate job checks the two agree.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("choose_shards")

MODE_PUSH_FULL = "push-full"
MODE_PR_FULL = "pr-full"
MODE_PR_SELECTED = "pr-selected"


def choose(
    event: str,
    selector: dict | None,
    full_shard_count: int,
    threshold: float,
) -> tuple[str, list[int]]:
    """Return ``(mode, shard_indices)`` for one run.

    ``selector`` is the parsed `select_affected_tests.py --json` payload. It is required for
    pull requests and ignored for pushes. ``threshold`` is the fraction of test files at or above
    which a pull request runs the full suite as shards.
    """
    if full_shard_count < 1:
        raise ValueError("full_shard_count must be >= 1")
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must satisfy 0 < threshold <= 1, got {threshold}")
    full = list(range(full_shard_count))
    if event != "pull_request":
        return MODE_PUSH_FULL, full
    if selector is None:
        raise ValueError("a pull_request run needs the selector JSON")
    try:
        mode = selector["mode"]
        selected = len(selector["selected_tests"])
        total = int(selector["total_tests"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"selector JSON is missing a required field: {exc}") from exc
    if mode == "full" or total == 0:
        log.info("selector mode=%s total=%d: full suite as shards", mode, total)
        return MODE_PR_FULL, full
    fraction = selected / total
    if fraction >= threshold:
        log.info("selected %d of %d files (%.1f%% >= %.1f%%): full suite as shards",
                 selected, total, 100 * fraction, 100 * threshold)
        return MODE_PR_FULL, full
    log.info("selected %d of %d files (%.1f%% < %.1f%%): one selected job",
             selected, total, 100 * fraction, 100 * threshold)
    return MODE_PR_SELECTED, [0]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", "push"))
    parser.add_argument("--selector-json", type=Path, help="selector --json output (pull requests)")
    parser.add_argument("--full-shard-count", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--github-output", type=Path, help="append key=value lines here")
    args = parser.parse_args(argv)

    selector = None
    if args.selector_json is not None:
        selector = json.loads(args.selector_json.read_text(encoding="utf-8"))
    mode, shards = choose(args.event, selector, args.full_shard_count, args.threshold)
    lines = [f"mode={mode}", f"shards={json.dumps(shards)}", f"shard_total={len(shards)}"]
    text = "\n".join(lines) + "\n"
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
