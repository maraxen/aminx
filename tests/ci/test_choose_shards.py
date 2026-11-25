from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "ci"))

import choose_shards  # noqa: E402


def _selector(mode: str, selected: int, total: int) -> dict:
    return {"mode": mode, "selected_tests": [f"t{i}" for i in range(selected)], "total_tests": total}


def test_push_always_runs_full_suite_in_every_shard() -> None:
    assert choose_shards.choose("push", None, 4, 0.5) == ("push-full", [0, 1, 2, 3])


def test_pull_request_small_selection_runs_one_job() -> None:
    assert choose_shards.choose("pull_request", _selector("selected", 20, 435), 4, 0.5) == (
        "pr-selected",
        [0],
    )


def test_pull_request_broad_selection_shards_the_full_suite() -> None:
    # #255's selection was 361 of 435 files.
    assert choose_shards.choose("pull_request", _selector("selected", 361, 435), 4, 0.5) == (
        "pr-full",
        [0, 1, 2, 3],
    )


def test_threshold_is_inclusive() -> None:
    assert choose_shards.choose("pull_request", _selector("selected", 50, 100), 4, 0.5)[0] == "pr-full"
    assert choose_shards.choose("pull_request", _selector("selected", 49, 100), 4, 0.5)[0] == "pr-selected"


def test_selector_whole_suite_mode_shards() -> None:
    assert choose_shards.choose("pull_request", _selector("full", 0, 435), 4, 0.5) == (
        "pr-full",
        [0, 1, 2, 3],
    )


def test_no_known_files_falls_back_to_full_shards() -> None:
    assert choose_shards.choose("pull_request", _selector("selected", 0, 0), 4, 0.5)[0] == "pr-full"


def test_pull_request_without_selector_json_is_an_error() -> None:
    with pytest.raises(ValueError):
        choose_shards.choose("pull_request", None, 4, 0.5)


def test_selector_json_missing_fields_is_an_error() -> None:
    with pytest.raises(ValueError):
        choose_shards.choose("pull_request", {"mode": "selected"}, 4, 0.5)


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
def test_threshold_out_of_range_is_an_error(threshold: float) -> None:
    with pytest.raises(ValueError):
        choose_shards.choose("push", None, 4, threshold)


def test_shard_total_matches_matrix_length(tmp_path: Path) -> None:
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(_selector("selected", 400, 435)))
    out = tmp_path / "out.txt"
    rc = choose_shards.main(
        ["--event", "pull_request", "--selector-json", str(sel), "--full-shard-count", "4",
         "--threshold", "0.5", "--github-output", str(out)]
    )
    assert rc == 0
    fields = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert fields["mode"] == "pr-full"
    assert json.loads(fields["shards"]) == [0, 1, 2, 3]
    assert int(fields["shard_total"]) == len(json.loads(fields["shards"])) == 4
