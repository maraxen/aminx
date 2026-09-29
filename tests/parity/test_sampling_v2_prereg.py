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
import pytest

import scripts.browser_validation.layer_a_sampling as las
import scripts.browser_validation.layer_a_sampling_calibrate as lasc
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
  # T10e amendment: n_lanes is now 3 (P07@0.1 dropped from V2_LANES, run e091a33e).
  return {
    "merge_refused": False,
    "tf_max_abs": 0.0,
    "tf_bar": las.TF_BAR,
    "n_lanes": 3,
    "n_lanes_equiv": 3,
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
  tost["n_lanes_equiv"] = 2
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
    "checkpoint_dir": "",
    "checkpoint_units_resumed": [],
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


# --------------------------------------------------------------------------------------
# T10e regression guards: negative/non-finite margin must not "size" against the
# UNCOMPUTED_SENTINEL, and the extended FUSION_EPS_CANDIDATES grid must size a
# linear-in-eps control (the run-e091a33e measured relationship) in its new low range,
# never falling back to the grid's last candidate when nothing sizes.
# --------------------------------------------------------------------------------------


def test_search_beta_for_lane_nonpositive_margin_is_unsized(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A lane whose own margin is `<= 0` or non-finite (P07@0.1 measured -8.5e-5 on run
  e091a33e) must be declared UNSIZED immediately, `tried == []` -- never by dividing by
  that margin and comparing the result against `UNCOMPUTED_SENTINEL`, which the pre-fix
  code did (any beta then "sized" trivially on the FIRST candidate, spending real draws
  on a search that could never mean anything). Monkeypatches `draw_iut_arms` to raise if
  called at all, proving the short-circuit happens before a single real draw."""

  def _must_not_be_called(*_args: object, **_kwargs: object) -> None:
    msg = "draw_iut_arms must not run when margin is <= 0 or non-finite"
    raise AssertionError(msg)

  monkeypatch.setattr(lasc.las, "draw_iut_arms", _must_not_be_called)
  for bad_margin in (-8.5e-5, 0.0, float("nan"), float("-inf")):
    beta, sized, tried = lasc._search_beta_for_lane(  # noqa: SLF001
      None,
      [],
      "P07@1.0",
      bad_margin,
      "redcheck",
    )
    assert beta is None
    assert sized is False
    assert tried == []


def _linear_fusion_control(
  _jax_model: object,
  _batch: object,
  _decoding_order: object,
  *,
  eps: float,
) -> dict[str, object]:
  """Fake control mirroring the measured run-e091a33e relationship on 3HTN:
  `ratio_to_bar` linear in `eps`, ~28,600*eps."""
  ratio = 28_600.0 * eps
  return {"eps": eps, "effect": ratio * las.TF_BAR, "ratio_to_bar": ratio, "detected": ratio > 1.0}


def test_search_fusion_eps_sizes_in_new_low_range(monkeypatch: pytest.MonkeyPatch) -> None:
  """With `FUSION_EPS_CANDIDATES` extended down to 1e-5, a fusion control whose
  `ratio_to_bar` is linear in eps (~28,600*eps, the measured run-e091a33e relationship)
  now sizes -- at eps=1e-4 (ratio~2.86, inside [2x, 10x]) -- instead of bottoming out
  unsized at the pre-amendment grid's floor of 0.01 (ratio~286, already over 10x)."""
  monkeypatch.setattr(lasc.las, "_fusion_sized_control", _linear_fusion_control)
  eps, sized = lasc._search_fusion_eps(None, None, None)  # noqa: SLF001
  assert sized is True
  assert eps == pytest.approx(1e-4)
  lo, hi = lasc.las.SIZING_RATIO_RANGE
  assert lo <= 28_600.0 * eps <= hi


def test_search_fusion_eps_returns_zero_when_never_sized(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """When NOTHING in the grid sizes, `_search_fusion_eps` must return `(0.0, False)` --
  never the grid's last candidate, which is indistinguishable from a genuinely sized
  value to a caller that reads only `p09_fusion_ctrl_eps` without its paired `sized`."""

  def _never_sizes(*_args: object, **_kwargs: object) -> dict[str, object]:
    return {"eps": 0.0, "effect": 0.0, "ratio_to_bar": 0.0, "detected": False}

  monkeypatch.setattr(lasc.las, "_fusion_sized_control", _never_sizes)
  eps, sized = lasc._search_fusion_eps(None, None, None)  # noqa: SLF001
  assert eps == 0.0
  assert sized is False


# --------------------------------------------------------------------------------------
# T10g: checkpoint store (atomic write/load round trip, fingerprint mismatch ->
# recompute, missing -> recompute, toy stage-runner load-instead-of-recompute) + the
# `_seed_for` determinism fix (identity-derived, never call-order dependent).
# --------------------------------------------------------------------------------------


class _CallCounter:
  """A `compute` callable that counts its own invocations and returns a fixed value --
  used to prove a `CheckpointStore.get_or_compute` call did or did not actually run it."""

  def __init__(self, value: dict[str, object]) -> None:
    self.value = value
    self.calls = 0

  def __call__(self) -> dict[str, object]:
    self.calls += 1
    return dict(self.value)


def test_checkpoint_store_missing_checkpoint_recomputes(tmp_path: Path) -> None:
  """No checkpoint file yet -> `compute` runs, and a checkpoint file is written."""
  store = lasc.CheckpointStore(tmp_path, {"git_hash": "abc", "fixture_set": "A"})
  counter = _CallCounter({"x": 1.5})
  value = store.get_or_compute("unit_a", {"lane": "P07@1.0"}, counter)
  assert value == {"x": 1.5}
  assert counter.calls == 1
  files = list(tmp_path.glob("*.json"))
  assert len(files) == 1


def test_checkpoint_store_atomic_round_trip_resumes_without_recompute(tmp_path: Path) -> None:
  """A SECOND store (simulating a fresh process after a crash/resume), same
  `checkpoint_dir` and same identity, LOADS the checkpoint instead of recomputing --
  and the loaded value is byte-identical (JSON round-trip) to what was computed."""
  base_fp = {"git_hash": "abc", "fixture_set": "A", "smoke": False, "selected_lanes": ["P07@1.0"]}
  store1 = lasc.CheckpointStore(tmp_path, base_fp)
  counter1 = _CallCounter({"margin": 0.123456789, "n_draws": 4000, "allocation": {"5L33": 1000}})
  value1 = store1.get_or_compute("lanemargin_P07@1.0_n1000_pilotmargin0", {"n_required": 1000}, counter1)
  assert counter1.calls == 1

  # A NEW store instance -- as a resumed run would construct, in a new process.
  store2 = lasc.CheckpointStore(tmp_path, base_fp)
  counter2 = _CallCounter({"margin": 999.0, "n_draws": 1, "allocation": {}})  # must NOT be used
  value2 = store2.get_or_compute("lanemargin_P07@1.0_n1000_pilotmargin0", {"n_required": 1000}, counter2)
  assert counter2.calls == 0, "resumed unit must not recompute"
  assert value2 == value1
  assert store2.units_resumed == ["lanemargin_P07@1.0_n1000_pilotmargin0"]


def test_checkpoint_store_fingerprint_mismatch_recomputes(
  tmp_path: Path,
  caplog: pytest.LogCaptureFixture,
) -> None:
  """A checkpoint written under one identity (e.g. one git_hash) is NEVER silently reused
  under a different one -- it is recomputed, and the mismatch is logged WARNING."""
  store1 = lasc.CheckpointStore(tmp_path, {"git_hash": "abc", "fixture_set": "A"})
  counter1 = _CallCounter({"x": 1.0})
  store1.get_or_compute("unit_a", {}, counter1)
  assert counter1.calls == 1

  store2 = lasc.CheckpointStore(tmp_path, {"git_hash": "DIFFERENT", "fixture_set": "A"})
  counter2 = _CallCounter({"x": 2.0})
  with caplog.at_level("WARNING"):
    value2 = store2.get_or_compute("unit_a", {}, counter2)
  assert counter2.calls == 1, "a fingerprint mismatch must recompute, never silently reuse"
  assert value2 == {"x": 2.0}
  assert any("fingerprint mismatch" in rec.message for rec in caplog.records)


def test_checkpoint_store_disabled_is_pure_passthrough(tmp_path: Path) -> None:
  """`checkpoint_dir=None` -- the default -- always calls `compute`, writes nothing, and
  every call site's behavior is unchanged from before T10g."""
  store = lasc.CheckpointStore(None, {"git_hash": "abc"})
  counter = _CallCounter({"x": 1.0})
  store.get_or_compute("unit_a", {}, counter)
  store.get_or_compute("unit_a", {}, counter)
  assert counter.calls == 2
  assert list(tmp_path.iterdir()) == []


def test_checkpoint_store_toy_stage_runner_resumes_every_unit(tmp_path: Path) -> None:
  """A toy multi-unit "stage runner" (mirroring `run_full`'s null-replicate loop): the
  FIRST pass computes `n` units; a SECOND pass, with a fresh store over the same dir and
  identity, must load every one of them without recomputing any -- proving
  load-instead-of-recompute at the granularity `run_full` actually uses (one unit per
  null replicate)."""
  base_fp = {"git_hash": "abc", "fixture_set": "A", "smoke": False}

  def _run_stage(store: lasc.CheckpointStore, counters: list[_CallCounter]) -> list[dict[str, object]]:
    results = []
    for replicate in range(5):
      counter = _CallCounter({"tost_pass": True, "excess_js_ub": 0.001 * replicate})
      counters.append(counter)
      unit = f"nullreplicate_d0_r{replicate}_n1500"
      results.append(store.get_or_compute(unit, {"replicate": replicate, "n": 1500}, counter))
    return results

  store1 = lasc.CheckpointStore(tmp_path, base_fp)
  counters1: list[_CallCounter] = []
  first_pass = _run_stage(store1, counters1)
  assert all(c.calls == 1 for c in counters1)

  store2 = lasc.CheckpointStore(tmp_path, base_fp)
  counters2: list[_CallCounter] = []
  second_pass = _run_stage(store2, counters2)
  assert all(c.calls == 0 for c in counters2), "every unit must resume, none recomputed"
  assert second_pass == first_pass
  assert len(store2.units_resumed) == 5


def test_checkpoint_store_atomic_write_uses_tmp_then_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """`_atomic_write_json` never leaves a partially-written checkpoint file visible under
  its final name -- it writes a `.tmp<pid>` sibling and `os.replace`s it into place."""
  seen_tmp_paths: list[Path] = []
  real_replace = lasc.os.replace

  def _spy_replace(src: object, dst: object) -> None:
    seen_tmp_paths.append(Path(src))
    assert Path(src).is_file(), "tmp file must exist and be fully written before replace"
    real_replace(src, dst)

  monkeypatch.setattr(lasc.os, "replace", _spy_replace)
  target = tmp_path / "unit.json"
  lasc._atomic_write_json(target, {"fingerprint": "abc", "value": {"x": 1}})
  assert target.is_file()
  assert len(seen_tmp_paths) == 1
  assert seen_tmp_paths[0] != target
  assert not seen_tmp_paths[0].exists(), "tmp file must be gone after os.replace"
  with target.open() as fh:
    assert json.load(fh) == {"fingerprint": "abc", "value": {"x": 1}}


def test_seed_for_is_identity_derived_hashlib_based() -> None:
  """`_seed_for` is a PURE function of its string argument -- not of call order, and not
  of Python's (per-process-randomized) builtin `hash()`. Pins the hashlib-based formula
  directly (a regression back to `hash()` would fail this even within one process, since
  the formula would no longer match)."""
  import hashlib

  name = "P07@1.0testA1"
  digest = hashlib.sha256(name.encode("utf-8")).digest()
  expected = (las._SEED_BASE + (int.from_bytes(digest[:8], "big") % 10_000)) & 0xFFFFFFFF  # noqa: SLF001
  assert las._seed_for(name) == expected  # noqa: SLF001

  # Order independence: seed_for(tagA) does not change once other tags are looked up
  # in between -- proving no shared/advancing state underlies the derivation.
  first = las._seed_for("tagA")  # noqa: SLF001
  _ = las._seed_for("tagB")  # noqa: SLF001
  _ = las._seed_for("tagC")  # noqa: SLF001
  second = las._seed_for("tagA")  # noqa: SLF001
  assert first == second
