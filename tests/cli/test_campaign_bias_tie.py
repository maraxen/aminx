"""`campaign plan` / `ramp-plan` expose --bias and --tie-group-map (debt #2119).

Both were settable only through the Python API -- the same gap `--fixed-arm` had before it
existed. The flags carry their own referent (a `.npy` path, loaded and shape-checked at the CLI),
land on the base spec, and so reach every row's `sampling_spec`, which is all a worker reads.

task_id: 260930_resolve-issues-debt
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from aminx.cli import app
from aminx.host.campaign import main as campaign_main
from aminx.run.specs import SamplingSpecification

_CKPT = "ligandmpnn_v_32_020_25"
_L = 512


@pytest.fixture
def arrays(tmp_path: Path) -> dict[str, Path]:
  bias = np.zeros((_L, 21), dtype=np.float32)
  bias[3, 5] = 1.5
  tie = np.arange(_L, dtype=np.int32) // 2
  np.save(tmp_path / "bias.npy", bias)
  np.save(tmp_path / "tie.npy", tie)
  return {"bias": tmp_path / "bias.npy", "tie": tmp_path / "tie.npy"}


def _plan_args(tmp_path: Path, *extra: str) -> list[str]:
  return [
    "campaign", "plan",
    "--inputs", "fake.pdb",
    "--campaign-id", "bt",
    "--manifest-path", str(tmp_path / "m.json"),
    "--output-root", str(tmp_path / "out"),
    "--designs-per-library-type", "1",
    "--samples-chunk-size", "1",
    "--checkpoint-id", _CKPT,
    *extra,
  ]


def _rows(manifest: Path) -> list[dict[str, Any]]:
  return json.loads(manifest.read_text(encoding="utf-8"))["rows"]


def _assert_reaches_worker(rows: list[dict[str, Any]]) -> None:
  assert rows
  for row in rows:
    worker = SamplingSpecification(**json.loads(json.dumps(row["sampling_spec"])))
    assert np.asarray(worker.bias).shape == (_L, 21)
    assert float(np.asarray(worker.bias)[3, 5]) == pytest.approx(1.5)
    tie = np.asarray(worker.tie_group_map)
    assert tie.shape == (_L,)
    assert tie[0] == tie[1] != tie[2]


def test_plan_bias_and_tie_group_reach_every_row(tmp_path: Path, arrays: dict[str, Path]) -> None:
  result = CliRunner().invoke(
    app,
    _plan_args(tmp_path, "--bias", str(arrays["bias"]), "--tie-group-map", str(arrays["tie"])),
  )
  assert result.exit_code == 0, result.output
  _assert_reaches_worker(_rows(tmp_path / "m.json"))


def test_tie_group_alias_is_accepted(tmp_path: Path, arrays: dict[str, Path]) -> None:
  result = CliRunner().invoke(app, _plan_args(tmp_path, "--tie-group", str(arrays["tie"])))
  assert result.exit_code == 0, result.output
  tie = np.asarray(_rows(tmp_path / "m.json")[0]["sampling_spec"]["tie_group_map"])
  assert tie.shape == (_L,)


def test_default_carries_neither(tmp_path: Path) -> None:
  """Back-compat: without the flags the rows are exactly as before."""
  result = CliRunner().invoke(app, _plan_args(tmp_path))
  assert result.exit_code == 0, result.output
  for row in _rows(tmp_path / "m.json"):
    assert row["sampling_spec"]["bias"] is None
    assert row["sampling_spec"]["tie_group_map"] is None


def test_ramp_plan_bias_and_tie_group_reach_every_stage(
  tmp_path: Path, arrays: dict[str, Path],
) -> None:
  result = CliRunner().invoke(
    app,
    [
      "campaign", "ramp-plan",
      "--inputs", "fake.pdb",
      "--campaign-id", "bt",
      "--manifest-dir", str(tmp_path / "manifests"),
      "--output-root", str(tmp_path / "out"),
      "--stage-designs-per-library-type", "1,2",
      "--samples-chunk-size", "1",
      "--checkpoint-id", _CKPT,
      "--bias", str(arrays["bias"]),
      "--tie-group-map", str(arrays["tie"]),
    ],
  )
  assert result.exit_code == 0, result.output
  manifests = sorted((tmp_path / "manifests").glob("*.manifest.json"))
  assert len(manifests) == 2
  for manifest in manifests:
    _assert_reaches_worker(_rows(manifest))


def test_argparse_entrypoint_also_carries_them(
  tmp_path: Path, arrays: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
  """`python -m aminx.host.campaign plan` has no --checkpoint-id (so it cannot finish a plan on
  its own); capture the base spec it hands to the planner instead."""
  captured: dict[str, Any] = {}
  monkeypatch.setattr(
    "aminx.host.campaign.write_campaign_manifest", lambda **kw: captured.update(kw),
  )
  code = campaign_main(
    [
      "plan",
      "--inputs", "fake.pdb",
      "--campaign-id", "bt",
      "--manifest-path", str(tmp_path / "m.json"),
      "--output-root", str(tmp_path / "out"),
      "--designs-per-library-type", "1",
      "--samples-chunk-size", "1",
      "--bias", str(arrays["bias"]),
      "--tie-group", str(arrays["tie"]),
    ],
  )
  assert code == 0
  base = captured["base_spec"]
  assert np.asarray(base.bias).shape == (_L, 21)
  assert np.asarray(base.tie_group_map).shape == (_L,)


@pytest.mark.parametrize(
  ("flag", "array", "match"),
  [
    ("--bias", np.zeros((4, 21), dtype=np.float32), "expected \\(512, 21\\)"),
    ("--bias", np.full((_L, 21), np.nan, dtype=np.float32), "non-finite"),
    ("--tie-group-map", np.zeros((4,), dtype=np.int32), "expected \\(512,\\)"),
    ("--tie-group-map", np.full((_L,), 0.5, dtype=np.float32), "integer group ids"),
    ("--tie-group-map", np.full((_L,), -1, dtype=np.int32), "negative"),
  ],
)
def test_bad_arrays_are_refused_at_the_cli(
  tmp_path: Path, flag: str, array: np.ndarray, match: str,
) -> None:
  path = tmp_path / "bad.npy"
  np.save(path, array)
  result = CliRunner().invoke(app, _plan_args(tmp_path, flag, str(path)))
  assert result.exit_code != 0
  assert isinstance(result.exception, ValueError)
  assert re.search(match, str(result.exception))
  assert not (tmp_path / "m.json").exists()


def test_missing_file_is_refused(tmp_path: Path) -> None:
  result = CliRunner().invoke(app, _plan_args(tmp_path, "--bias", str(tmp_path / "nope.npy")))
  assert result.exit_code != 0
  assert "existing .npy file" in str(result.exception)
