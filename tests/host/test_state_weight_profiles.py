"""A state-weight profile must reach the spec, or it is a label with no meaning (#111).

`--state-weight-profiles equal,weighted` was decorative -- the same bug `fixed_policies` had,
one line above it in the same loop. The planner hashed the profile NAME into the row, wrote it
to the row, checked it for non-emptiness, and never resolved it to a weight vector:

    for state_weight_profile in state_weight_profiles:     # <- the axis
        spec_variant = replace(base_spec, ...)             # <- state_weights NEVER set

Two profiles therefore produced two row-sets, differently labelled and identically weighted:
duplicate work counted as an ablation. The fix shape is `fixed_arms`': the declaration carries
its own referent (label -> weights), and a bare name aminx cannot resolve raises.

task_id: 260930_resolve-issues-debt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from aminx.cli import app
from aminx.host.campaign import (
  parse_state_weight_profiles,
  plan_campaign_manifest,
)
from aminx.run.specs import SamplingSpecification

pytestmark = pytest.mark.knob_differential

_CKPT = "ligandmpnn_v_32_020_25"


def _plan(base: SamplingSpecification | None = None, **kwargs: Any) -> list[dict[str, Any]]:
  return plan_campaign_manifest(
    base_spec=base or SamplingSpecification(inputs=["ref.pdb"], checkpoint_id=_CKPT),
    campaign_id="swp",
    output_root="/tmp/swp",
    designs_per_library_type=1,
    samples_chunk_size=1,
    **kwargs,
  )


def _by_profile(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
  out: dict[str, list[dict[str, Any]]] = {}
  for row in rows:
    out.setdefault(row["state_weight_profile"], []).append(row)
  return out


def test_two_profiles_produce_DIFFERENT_state_weights() -> None:
  """THE bug. Different profiles must differ on the spec, not just on the label."""
  rows = _plan(state_weight_profiles={"equal": None, "pocket_heavy": [0.7, 0.3]})
  groups = _by_profile(rows)

  assert {r["sampling_spec"]["state_weights"] is None for r in groups["equal"]} == {True}
  for row in groups["pocket_heavy"]:
    np.testing.assert_allclose(row["sampling_spec"]["state_weights"], [0.7, 0.3], rtol=1e-6)


def test_profile_weights_survive_to_the_worker_spec() -> None:
  """Plan -> manifest JSON -> the worker's reconstruction (`SamplingSpecification(**payload)`)."""
  rows = _plan(state_weight_profiles={"a": [0.25, 0.75], "b": "0.9|0.1"})
  seen: dict[str, np.ndarray] = {}
  for row in rows:
    payload = json.loads(json.dumps(row["sampling_spec"]))  # the real JSON boundary
    worker = SamplingSpecification(**payload)
    seen[row["state_weight_profile"]] = np.asarray(worker.state_weights)
  np.testing.assert_allclose(seen["a"], [0.25, 0.75], rtol=1e-6)
  np.testing.assert_allclose(seen["b"], [0.9, 0.1], rtol=1e-6)


def test_npy_profile_is_loaded(tmp_path: Path) -> None:
  path = tmp_path / "w.npy"
  np.save(path, np.array([0.5, 0.25, 0.25], dtype=np.float32))
  rows = _plan(state_weight_profiles={"from_file": str(path)})
  np.testing.assert_allclose(rows[0]["sampling_spec"]["state_weights"], [0.5, 0.25, 0.25])


def test_default_profile_changes_nothing() -> None:
  """Back-compat: the default ("equal",) carries no weights and keeps every row hash."""
  default_rows = _plan()
  explicit_rows = _plan(state_weight_profiles={"equal": None})
  assert [r["manifest_row_hash"] for r in default_rows] == [
    r["manifest_row_hash"] for r in explicit_rows
  ]
  assert all(r["sampling_spec"]["state_weights"] is None for r in default_rows)
  assert {r["state_weight_profile"] for r in default_rows} == {"equal"}


def test_a_weighted_profile_hashes_differently_from_equal_and_from_other_weights() -> None:
  rows = _plan(
    state_weight_profiles={"equal": None, "a": [0.7, 0.3], "b": [0.3, 0.7]},
  )
  hashes = [r["manifest_row_hash"] for r in rows]
  assert len(hashes) == len(set(hashes))


def test_same_label_different_vector_does_not_collide_on_hash() -> None:
  """A reused label with a different vector is a different row (and output path)."""
  a = _plan(state_weight_profiles={"w": [0.7, 0.3]})
  b = _plan(state_weight_profiles={"w": [0.6, 0.4]})
  assert {r["manifest_row_hash"] for r in a}.isdisjoint({r["manifest_row_hash"] for r in b})


def test_base_spec_weights_are_kept_for_the_equal_profile() -> None:
  """The pre-existing route (a caller setting `state_weights` directly) still works."""
  base = SamplingSpecification(
    inputs=["ref.pdb"], checkpoint_id=_CKPT, state_weights=np.array([0.4, 0.6], dtype=np.float32),
  )
  rows = _plan(base)
  np.testing.assert_allclose(rows[0]["sampling_spec"]["state_weights"], [0.4, 0.6])


def test_a_bare_name_aminx_cannot_resolve_RAISES() -> None:
  """The old behaviour: a label with no meaning. Now refused, naming the fix."""
  with pytest.raises(ValueError, match="bare name aminx cannot resolve"):
    _plan(state_weight_profiles=("equal", "weighted"))


@pytest.mark.parametrize(
  ("profiles", "match"),
  [
    ({"equal": [0.7, 0.3]}, "reserved"),
    ({"a": [0.7, 0.3], "b": [0.7, 0.3]}, "resolve to the same weights"),
    ({"a": [-0.1, 1.1]}, "non-negative"),
    ({"a": [0.0, 0.0]}, "not all zero"),
    ({"a": [float("nan"), 1.0]}, "finite"),
    ({"a": [[0.5, 0.5]]}, "1-D"),
    ({"a": "x|y"}, "not a list of numbers"),
    ({"": [0.5, 0.5]}, "empty label"),
    ({}, "empty"),
  ],
)
def test_invalid_profiles_raise(profiles: dict[str, Any], match: str) -> None:
  with pytest.raises(ValueError, match=match):
    _plan(state_weight_profiles=profiles)


def test_parse_state_weight_profiles_flags() -> None:
  assert parse_state_weight_profiles(None, None) == {"equal": None}
  assert parse_state_weight_profiles("equal", ["h=0.7|0.3"]) == {"equal": None, "h": "0.7|0.3"}
  # Declarations alone: NO implicit equal baseline.
  assert parse_state_weight_profiles(None, ["h=0.7|0.3"]) == {"h": "0.7|0.3"}
  with pytest.raises(ValueError, match="has no weights"):
    parse_state_weight_profiles("equal,weighted", None)
  with pytest.raises(ValueError, match="LABEL=WEIGHTS"):
    parse_state_weight_profiles(None, ["nolabel"])
  with pytest.raises(ValueError, match="twice"):
    parse_state_weight_profiles(None, ["h=1|1", "h=2|1"])


def _cli_plan_args(tmp_path: Path, *extra: str) -> list[str]:
  return [
    "campaign", "plan",
    "--inputs", "fake.pdb",
    "--campaign-id", "swp",
    "--manifest-path", str(tmp_path / "m.json"),
    "--output-root", str(tmp_path / "out"),
    "--designs-per-library-type", "1",
    "--samples-chunk-size", "1",
    "--checkpoint-id", _CKPT,
    *extra,
  ]


def test_cli_profiles_reach_the_manifest(tmp_path: Path) -> None:
  result = CliRunner().invoke(
    app,
    _cli_plan_args(
      tmp_path, "--state-weight-profiles", "equal", "--state-weight-profile", "heavy=0.8|0.2",
    ),
  )
  assert result.exit_code == 0, result.output
  groups = _by_profile(json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))["rows"])
  assert set(groups) == {"equal", "heavy"}
  assert groups["equal"][0]["sampling_spec"]["state_weights"] is None
  np.testing.assert_allclose(
    groups["heavy"][0]["sampling_spec"]["state_weights"], [0.8, 0.2], rtol=1e-6,
  )


def test_cli_bare_name_fails_loudly(tmp_path: Path) -> None:
  result = CliRunner().invoke(
    app, _cli_plan_args(tmp_path, "--state-weight-profiles", "equal,weighted"),
  )
  assert result.exit_code != 0
  assert "has no weights" in str(result.exception)
  assert not (tmp_path / "m.json").exists()
