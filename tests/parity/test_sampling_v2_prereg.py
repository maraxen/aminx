"""Drafted v2 sampling sidecars: shard contract, merge finding, calibrate budget."""

# ruff: noqa: S101

from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path

import bathos.sidecar as bathos_sidecar
import jax
import jax.numpy as jnp
import numpy as np

import scripts.browser_validation.layer_a_sampling as las
import scripts.browser_validation.layer_a_sampling_merge as merge
import scripts.browser_validation.layer_a_sampling_shard as shard
import scripts.browser_validation.layer_a_sampling_validate as lasv

_BV = Path(__file__).resolve().parents[2] / "scripts" / "browser_validation"
_VALIDATE = _BV / "layer_a_sampling_validate.bth.toml"
_MERGE = _BV / "layer_a_sampling_merge.bth.toml"
_CALIBRATE = _BV / "layer_a_sampling_calibrate.bth.toml"


def _schema_keys(path: Path) -> set[str]:
  document = tomllib.loads(path.read_text())
  schema = document["result_schema"]
  assert isinstance(schema, dict)
  return set(schema)


def _outcome(path: Path, result: dict[str, object]) -> str:
  sidecar = bathos_sidecar.parse_sidecar(path)
  return bathos_sidecar.evaluate_outcome(sidecar, result)


def _finite(result: dict[str, object]) -> None:
  for value in result.values():
    if isinstance(value, float):
      assert math.isfinite(value)


def test_shard_contract_pass_and_incomplete() -> None:
  good = lasv.shard_schema_fields(
    units_assigned=4,
    units_drawn=4,
    prereq_ok=True,
    git_clean=True,
    n_shards=2,
    shard_index=1,
    records_path="records/shard.records.json",
    git_hash="abc123",
  )
  assert _outcome(_VALIDATE, good) == "pass"
  short = dict(good)
  short["units_drawn"] = 3
  assert _outcome(_VALIDATE, short) == "incomplete"


def test_shard_partial_emits_schema_and_exits_zero(tmp_path: Path) -> None:
  partial = {
    "partial": True,
    "git_hash": "abc123",
    "params": {"sampling": {"n_required": 1}},
    "shard_index": 0,
    "n_shards": 2,
    "lanes": ["P07@1.0"],
    "n_control_replicates": 0,
    "records": [
      {
        "lane": "P07@1.0",
        "fixture": "fx",
        "arm": "main/A1",
        "replicate": 0,
        "draw_index": 0,
      },
    ],
    "prereq_ok": False,
    "git_clean": True,
  }
  out = tmp_path / "layer_a_sampling_validate.json"
  flat, code = lasv.finalize_shard_partial(
    partial,
    out,
    units_assigned=2,
    units_drawn=1,
    prereq_ok=False,
    git_clean=True,
    n_shards=2,
    shard_index=0,
    git_hash="abc123",
  )
  assert code == 0
  assert _schema_keys(_VALIDATE) <= set(flat)
  assert "records" not in flat
  _finite(flat)
  records_path = Path(str(flat["records_path"]))
  stored = json.loads(records_path.read_text())
  assert stored["records"]
  assert _outcome(_VALIDATE, flat) == "incomplete"


def _passing_finding() -> dict[str, object]:
  return {
    "merge_refused": False,
    "tf_max_abs": 0.0,
    "tf_bar": las.TF_BAR,
    "n_lanes": 4,
    "n_lanes_equiv": 4,
    "n_per_arm": 4,
    "n_required": 2,
    "sigma_hat": 0.01,
    "excess_js_ub_max_ratio": 0.5,
    "main_js_vs_ref": 0.01,
    "posctl_detected": 18,
    "negctl_fp": 3,
    "omitted_aa_count": 0,
    "x_token_count_aminx": 0,
    "x_token_count_reference": 0,
    "p09_tied_positions": 7,
    "p09_tied_positions_prereg": 7,
    "p09_fused_tf_max_abs": 0.0,
    "p09_fusion_ctrl_detected": True,
    "n_skipped": 0,
    "prereq_ok": True,
    "reference_commit": "26ec57ac",
    "git_hash": "abc123",
    "git_clean": True,
    "prereg_sha256": "deadbeef",
  }


def test_merge_finding_outcomes() -> None:
  passing = _passing_finding()
  assert _schema_keys(_MERGE) <= set(passing)
  assert _outcome(_MERGE, passing) == "pass"

  tost = dict(passing)
  tost["n_lanes_equiv"] = 3
  assert _outcome(_MERGE, tost) == "fail"

  js_over = dict(passing)
  js_over["excess_js_ub_max_ratio"] = 1.0
  assert _outcome(_MERGE, js_over) == "fail"

  posctl = dict(passing)
  posctl["posctl_detected"] = 17
  assert _outcome(_MERGE, posctl) == "ctrl_blind"

  negctl = dict(passing)
  negctl["negctl_fp"] = 4
  assert _outcome(_MERGE, negctl) == "ctrl_blind"

  x_token = dict(passing)
  x_token["x_token_count_aminx"] = 1
  assert _outcome(_MERGE, x_token) == "fail"

  refused = dict(passing)
  refused["merge_refused"] = True
  assert _outcome(_MERGE, refused) == "merge_refused"


def _sampler(key: jax.Array) -> np.ndarray:
  return np.asarray(jax.random.uniform(key, (2,), dtype=jnp.float32))


def _write_tiny_partials(directory: Path) -> list[Path]:
  units = shard.enumerate_work_units(
    lanes=("P07@1.0",),
    allocation={"fx": 1},
    n_control_replicates=0,
  )
  protocol = {
    "lanes": ["P07@1.0"],
    "allocation": {"fx": 1},
    "n_control_replicates": 0,
    "control_scope": "once",
    "control_lane": las.CONTROL_LANE,
  }
  paths: list[Path] = []
  plan = shard.plan_shards(units, 2)
  for index in (0, 1):
    records = shard.run_assigned_units(plan[index], _sampler, shard_index=index)
    partial = shard.make_partial(
      git_hash="abc123",
      params={"sampling": {"n_required": 1}},
      shard_index=index,
      n_shards=2,
      records=records,
      lanes=["P07@1.0"],
      n_control_replicates=0,
      protocol=protocol,
    )
    for record in partial["records"]:
      assert isinstance(record, dict)
      record["value"] = np.asarray(record["value"]).tolist()
    record_path = directory / f"shard-{index}.records.json"
    record_path.write_text(json.dumps(partial))
    wrapper = directory / f"shard-{index}.json"
    wrapper.write_text(json.dumps({"records_path": str(record_path)}))
    paths.append(wrapper)
  return paths


def test_merge_script_emits_schema(tmp_path: Path) -> None:
  wrappers = _write_tiny_partials(tmp_path)
  out = tmp_path / "merged.json"
  code = merge.main([str(path) for path in wrappers] + ["--out", str(out)])
  assert code == 0
  emitted = json.loads(out.read_text())
  assert isinstance(emitted, dict)
  assert _schema_keys(_MERGE) <= set(emitted)
  assert emitted["merge_refused"] is False
  _finite(emitted)

  broken = json.loads(Path(json.loads(wrappers[0].read_text())["records_path"]).read_text())
  records = broken["records"]
  assert isinstance(records, list)
  assert records
  records.pop()
  record_path = Path(json.loads(wrappers[0].read_text())["records_path"])
  record_path.write_text(json.dumps(broken))
  refused_out = tmp_path / "refused.json"
  refused_code = merge.main([str(path) for path in wrappers] + ["--out", str(refused_out)])
  assert refused_code == 0
  refused = json.loads(refused_out.read_text())
  assert refused["merge_refused"] is True
  assert _schema_keys(_MERGE) <= set(refused)
  _finite(refused)
  assert _outcome(_MERGE, refused) == "merge_refused"


def _calibrate_pass() -> dict[str, object]:
  return {
    "sigma_hat": 0.01,
    "n_required": 4,
    "beta": 0.2,
    "p09_qualifying_groups": 1,
    "p09_tied_positions": 3,
    "p09_fusion_ctrl_eps": 0.001,
    "min_effect": 0.01,
    "measured_half_effect": 0.02,
    "n_not_advanced": 0,
    "n_skipped": 0,
    "controls_total": 2,
    "controls_sized": 2,
    "params_written": True,
    "params_section_sha256": "abc",
    "fixture_set": "A",
    "git_hash": "abc123",
    "git_clean": True,
    "budget_wall_hours": 1.0,
    "projected_peak_rss_gib": 2.0,
  }


def test_calibrate_sidecar_names_budget_rule() -> None:
  document = tomllib.loads(_CALIBRATE.read_text())
  hypothesis = document["experiment"]["hypothesis"]
  assert "V2_LANES" in hypothesis
  assert "n_shards=2" in hypothesis
  passing = _calibrate_pass()
  assert _schema_keys(_CALIBRATE) <= set(passing)
  assert _outcome(_CALIBRATE, passing) == "pass"
  over = dict(passing)
  over["budget_wall_hours"] = 17.0
  assert _outcome(_CALIBRATE, over) == "budget_exceeded"
  rss = dict(passing)
  rss["projected_peak_rss_gib"] = 49.0
  assert _outcome(_CALIBRATE, rss) == "budget_exceeded"
