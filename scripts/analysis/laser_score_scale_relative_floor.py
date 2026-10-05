"""Measure the laser_score float32 SCALE-RELATIVE parity floor (debt #2445).

Pre-registered in ``laser_score_scale_relative_floor.bth.toml``, committed before this file
existed. Read that sidecar for the hypothesis and the outcome criteria; this module only
implements them.

THE RULE. ``max|aminx - upstream| <= tol * max|ref|``, evaluated per (pair, field), where
``max|ref|`` is that pair's field maximum absolute reference magnitude. One tolerance, not
numpy's ``atol + rtol*|ref|``. This is option 3 of
``.praxia/docs/decisions/261003_laser-score-tier3-f32-design.md`` and the same reading already
applied to the decode padding test (``f97d99b6``).

WHY THE RULE CHANGE IS THE EXPERIMENT. Run ``e47bfaab`` graded
``noise_reaches_defect_scale`` for this wave under the ELEMENT-WISE rule: the worst violations
sit on near-zero elements, which makes ``atol`` bind and pushes it to 0.61 of an injected 1e-2
defect. Nothing about the port is wrong -- f64 agrees to <8e-13 and the two f32 implementations
agree with each other ~200x more tightly than either agrees with f64. So the question is whether
judging each field against its own scale restores the margin that treating near-zero elements
individually destroyed.

WHAT MAKES THIS INSTRUMENT HONEST, since the obvious version is not. Under a single
scale-relative statistic, admitting the same pairs the floor was derived from is TRUE BY
CONSTRUCTION -- the floor is the maximum of exactly the quantity admission tests -- so plain
"all admitted" is a check that cannot fail. It is deliberately not a criterion here. The real
admission is LEAVE-ONE-OUT over pairs: derive ``tol`` from 62 pairs and require the 63rd
admitted, for all 63 choices. That fails exactly when the worst pair is an outlier rather than a
ceiling, and it costs no extra forward passes.

The rule also has its own version of the failure it cures: a field whose ``max|ref|`` is at or
near zero has no scale to be relative to. Declared as the ``scale_is_degenerate`` outcome with a
threshold fixed in the sidecar before the run, not discovered while reading results.

MEASUREMENT IS REUSED, NOT RE-EXPRESSED. ``_measure_unit`` and ``_control`` come from
``laser_f32_tolerance_floor``, so the deviations are produced by the test module's own ``_run``
with its own ``_rows_over`` recorded, and the injected defect is the identical 1e-2. Changing
only the derivation rule is what makes this run's margin comparable to ``e47bfaab``'s.

Invocation:
  bth run --project-slug aminx --output-paths <json> -- \
    uv run --no-sync python3 scripts/analysis/laser_score_scale_relative_floor.py \
      --work-dir <dir> --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

from laser_f32_tolerance_floor import (  # noqa: E402
  FieldStats,
  OracleAbsent,
  _add_tests_to_path,  # noqa: PLC2701
  _ceil_one_significant_figure,  # noqa: PLC2701
  _measure_unit,  # noqa: PLC2701
  _repo_root,  # noqa: PLC2701
)

_LOG = logging.getLogger("laser_score_scale_relative_floor")

_WAVE = "laser_score"
_HEADROOM = 10.0
_CONTROL_GRADED = 1e-2
_CONTROL_SCALE_MARGIN = 10.0
#: A (pair, field) is degenerate when its max|ref| is this fraction of the largest field scale in
#: that pair or smaller. Fixed in the sidecar before the run.
_DEGENERATE_REL = 1e-6


def _scale_ratio(stats: FieldStats) -> float:
  """``max|err| / max|ref|`` for one (pair, field). Infinite when there is no scale."""
  if stats.max_ref_magnitude <= 0.0:
    return math.inf
  return stats.max_abs / stats.max_ref_magnitude


def _by_pair(stats: list[FieldStats]) -> dict[str, list[FieldStats]]:
  grouped: dict[str, list[FieldStats]] = {}
  for s in stats:
    grouped.setdefault(s.pair, []).append(s)
  return grouped


def _degenerate(stats: list[FieldStats]) -> list[str]:
  """``pair::field`` for every entry with no scale to be relative to.

  Judged WITHIN a pair: a field is degenerate when its own reference scale is at or below
  ``_DEGENERATE_REL`` of the largest field scale in that same pair. Comparing across pairs would
  make a legitimately small-scale pair look degenerate next to a large-scale one.
  """
  found: list[str] = []
  for pair, group in _by_pair(stats).items():
    largest = max((s.max_ref_magnitude for s in group), default=0.0)
    if largest <= 0.0:
      found.extend(f"{pair}::{s.field}" for s in group)
      continue
    found.extend(
      f"{pair}::{s.field}"
      for s in group
      if s.max_ref_magnitude <= _DEGENERATE_REL * largest
    )
  return sorted(found)


def _floor(stats: list[FieldStats]) -> tuple[float, str, str, float]:
  """``(floor, pair, field, second_worst)`` over every entry."""
  ranked = sorted(
    ((_scale_ratio(s), s.pair, s.field) for s in stats),
    key=lambda row: row[0],
    reverse=True,
  )
  if not ranked:
    msg = "no FieldStats to derive a floor from"
    raise RuntimeError(msg)
  worst, pair, field_name = ranked[0]
  second = ranked[1][0] if len(ranked) > 1 else 0.0
  return worst, pair, field_name, second


def _tol_from(floor: float) -> float:
  """The pre-registered derivation. Not a choice made after seeing numbers."""
  return _ceil_one_significant_figure(_HEADROOM * floor)


def _per_field_floor(stats: list[FieldStats]) -> dict[str, float]:
  per: dict[str, float] = {}
  for s in stats:
    per[s.field] = max(per.get(s.field, 0.0), _scale_ratio(s))
  return per


def _leave_one_out(stats: list[FieldStats]) -> tuple[int, str, float]:
  """``(pairs not admitted, worst held-out pair, its ratio to its own band)``.

  For each pair, derive the band from every OTHER pair and require the held-out pair to satisfy
  the rule at that band. A ratio above 1 means the band derived without that pair does not cover
  it, i.e. one pair's maximum does not characterise the population.
  """
  grouped = _by_pair(stats)
  pairs = sorted(grouped)
  not_admitted = 0
  worst_pair = ""
  worst_ratio = 0.0
  for held in pairs:
    others = [s for pair, group in grouped.items() if pair != held for s in group]
    if not others:
      continue
    tol = _tol_from(_floor(others)[0])
    held_worst = max(_scale_ratio(s) for s in grouped[held])
    ratio = held_worst / tol if tol > 0.0 else math.inf
    if ratio > worst_ratio:
      worst_ratio = ratio
      worst_pair = held
    if held_worst > tol:
      not_admitted += 1
  return not_admitted, worst_pair, worst_ratio


def _control_scale_relative(stats: list[FieldStats], tol: float) -> tuple[int, float]:
  """``(entries ADMITTING an injected 1e-2 relative defect, worst band/defect ratio)``.

  Analytic, exactly as ``laser_f32_tolerance_floor._control`` is: the defect is
  ``_CONTROL_GRADED * max|ref|`` and the band is ``tol * max|ref|``, both per field, so no model
  re-run is needed. Under this rule the magnitudes cancel and the comparison reduces to ``tol``
  against ``_CONTROL_GRADED`` -- which is a real bar that a large floor fails, not a tautology,
  but it is worth knowing it is scale-free. Entries with no scale are skipped; they are caught by
  ``_degenerate`` instead.
  """
  admitted = 0
  worst_ratio = 0.0
  for s in stats:
    magnitude = s.max_ref_magnitude
    if magnitude <= 0.0:
      continue
    defect = _CONTROL_GRADED * magnitude
    band = tol * magnitude
    if defect <= band:
      admitted += 1
    worst_ratio = max(worst_ratio, band / defect if defect > 0.0 else math.inf)
  return admitted, worst_ratio


def _grade(payload: dict[str, Any]) -> str:
  """The sidecar's outcome table, in the sidecar's order. First match wins."""
  if not payload["self_test_passed"]:
    return "instrument_unverified"
  if payload["oracle_absent"]:
    return "oracle_absent"
  if not payload["f64_passes"]:
    return "defect_not_tolerance"
  if payload["degenerate_fields"]:
    return "scale_is_degenerate"
  if not payload["f32_violated_elementwise"]:
    return "elementwise_was_fine"
  if not payload["below_control_scale"]:
    return "noise_reaches_defect_scale"
  if not payload["loo_all_admitted"]:
    return "band_does_not_generalise"
  if not payload["control_rejected"]:
    return "fail"
  return "pass"


def _synth(
  pair: str,
  field_name: str,
  max_abs: float,
  max_ref_magnitude: float,
  n_violating: int = 0,
) -> FieldStats:
  """A FieldStats carrying only the fields this rule reads. Self-test use only."""
  return FieldStats(
    pair=pair,
    field=field_name,
    n_elements=1,
    max_abs=max_abs,
    max_abs_ref=max_ref_magnitude,
    max_ref_magnitude=max_ref_magnitude,
    max_rel_unconditioned=0.0,
    max_rel_conditioned=0.0,
    n_conditioned=0,
    n_violating=n_violating,
    worst_violation_ratio=0.0,
    worst_violation_ref=0.0,
    worst_violation_abs=0.0,
    worst_violation_atol_share=0.0,
    abs_excess=0.0,
    rel_excess=0.0,
    zero_mag_worst=0.0,
    n_zero_mag=0,
  )


def _base_payload(**over: Any) -> dict[str, Any]:
  """A grading payload that is a `pass` unless a key is overridden."""
  payload = {
    "self_test_passed": True,
    "oracle_absent": [],
    "f64_passes": True,
    "degenerate_fields": [],
    "f32_violated_elementwise": True,
    "below_control_scale": True,
    "loo_all_admitted": True,
    "control_rejected": True,
  }
  payload.update(over)
  return payload


def _self_test() -> dict[str, Any]:
  """Verify the instrument on synthetic ground truth before any oracle is touched.

  Includes the four negative controls the sidecar declares: each must force a specific non-pass
  verdict. An instrument not shown capable of refusing is not an instrument.
  """
  failed: list[str] = []
  checks = 0

  def check(name: str, condition: bool) -> None:  # noqa: FBT001
    nonlocal checks
    checks += 1
    if not condition:
      failed.append(name)

  # Rounding, hand-computed.
  check("round_1.43e-6", _ceil_one_significant_figure(1.43e-6) == 2e-6)
  check("round_exact_1e-3", _ceil_one_significant_figure(1e-3) == 1e-3)
  check("round_3.6e-5", _ceil_one_significant_figure(3.6e-5) == 4e-5)

  # Scale ratio, hand-computed.
  check("ratio_basic", _scale_ratio(_synth("p", "f", 2.0, 100.0)) == 0.02)
  check("ratio_identical", _scale_ratio(_synth("p", "f", 0.0, 100.0)) == 0.0)
  check("ratio_no_scale_is_inf", math.isinf(_scale_ratio(_synth("p", "f", 1.0, 0.0))))

  # Derivation: 10x headroom then one significant figure.
  check("tol_from_floor", _tol_from(3.6e-5) == 4e-4)

  # Floor picks the maximum and reports the runner-up.
  uniform = [_synth(f"p{i}", "f", 1.0, 1000.0) for i in range(5)]
  floor, _, _, second = _floor(uniform)
  check("floor_uniform", floor == 1e-3)
  check("floor_second_uniform", second == 1e-3)

  # Leave-one-out ADMITS a uniform population: every held-out pair matches the others.
  not_adm, _, worst = _leave_one_out(uniform)
  check("loo_admits_uniform", not_adm == 0)
  check("loo_uniform_ratio_below_1", worst <= 1.0)

  # NEGATIVE CONTROL 1 (on the derivation): a planted outlier must NOT be admitted by a band
  # derived without it. 1e4x above its peers clears the 10x headroom and the rounding.
  outlier = [*uniform, _synth("p_bad", "f", 1e4, 1000.0)]
  not_adm_bad, worst_pair, _ = _leave_one_out(outlier)
  check("loo_refuses_outlier", not_adm_bad >= 1)
  check("loo_names_outlier", worst_pair == "p_bad")

  # Degenerate-scale detector, at and either side of the threshold.
  pair_ok = [_synth("p", "big", 1.0, 1.0), _synth("p", "small", 1.0, 1e-5)]
  check("degenerate_absent_above_threshold", _degenerate(pair_ok) == [])
  pair_bad = [_synth("p", "big", 1.0, 1.0), _synth("p", "tiny", 1.0, 1e-7)]
  check("degenerate_found_below_threshold", _degenerate(pair_bad) == ["p::tiny"])
  pair_zero = [_synth("p", "big", 1.0, 1.0), _synth("p", "zero", 1.0, 0.0)]
  check("degenerate_found_at_zero", _degenerate(pair_zero) == ["p::zero"])

  # Control margin, pinned at three tolerances including the boundary.
  stats_one = [_synth("p", "f", 0.0, 50.0)]
  adm_tight, ratio_tight = _control_scale_relative(stats_one, 1e-3)
  check("control_rejected_at_1e-3", adm_tight == 0)
  check("control_margin_at_1e-3", ratio_tight <= 1.0 / _CONTROL_SCALE_MARGIN)
  adm_edge, ratio_edge = _control_scale_relative(stats_one, 1e-2)
  check("control_admitted_at_1e-2", adm_edge == 1)
  check("control_margin_fails_at_1e-2", ratio_edge > 1.0 / _CONTROL_SCALE_MARGIN)
  adm_wide, _ = _control_scale_relative(stats_one, 1.0)
  check("control_admitted_at_1.0", adm_wide == 1)

  # NaN handling comes from the shared _stats; assert the shared contract still holds.
  from laser_f32_tolerance_floor import _stats  # noqa: PLC0415, PLC2701

  # atol is a tiny nonzero rather than 0 only to keep _stats's band strictly positive, which
  # avoids a 0/0 RuntimeWarning in ITS ratio line. The contract under test is max_abs, which no
  # tolerance affects, so this does not weaken either check -- and editing the shared module to
  # silence the warning would change its script hash and invalidate every resume unit keyed on it.
  tiny = 1e-30
  shared_nan = _stats("p", "f", np.array([np.nan, 1.0]), np.array([np.nan, 1.0]), 0.0, tiny)
  check("shared_nan_is_not_error", shared_nan.max_abs == 0.0)
  one_sided = _stats("p", "f", np.array([np.nan]), np.array([1.0]), 0.0, tiny)
  check("one_sided_nan_is_infinite", math.isinf(one_sided.max_abs))

  # NEGATIVE CONTROLS 2-5 (on the grader): each must force its own non-pass.
  check("grade_clean_is_pass", _grade(_base_payload()) == "pass")
  check(
    "grade_f64_violation",
    _grade(_base_payload(f64_passes=False)) == "defect_not_tolerance",
  )
  check(
    "grade_clean_elementwise",
    _grade(_base_payload(f32_violated_elementwise=False)) == "elementwise_was_fine",
  )
  check(
    "grade_outlier",
    _grade(_base_payload(loo_all_admitted=False)) == "band_does_not_generalise",
  )
  check(
    "grade_degenerate",
    _grade(_base_payload(degenerate_fields=["p::z"])) == "scale_is_degenerate",
  )
  check(
    "grade_noise_reaches_defect",
    _grade(_base_payload(below_control_scale=False)) == "noise_reaches_defect_scale",
  )
  check(
    "grade_instrument",
    _grade(_base_payload(self_test_passed=False)) == "instrument_unverified",
  )

  return {"passed": not failed, "n_checks": checks, "failed": failed}


def _unit_key(script_sha: str, pair: str, precision: str) -> str:
  raw = f"{script_sha}|{_WAVE}|{pair}|{precision}".encode()
  return hashlib.sha256(raw).hexdigest()


def _load_unit(work_dir: Path, key: str) -> list[FieldStats] | None:
  """Reuse a unit only when its body AND its stamp are both present."""
  body = work_dir / f"{key}.json"
  stamp = work_dir / f"{key}.stamp"
  if not (body.is_file() and stamp.is_file()):
    return None
  try:
    rows = json.loads(body.read_text(encoding="utf-8"))
    return [FieldStats(**row) for row in rows]
  except (OSError, ValueError, TypeError):
    # A body that no longer matches the dataclass is recomputed, never coerced.
    return None


def _save_unit(work_dir: Path, key: str, stats: list[FieldStats]) -> None:
  """Body first, stamp second, so a crash between them leaves an unstamped body."""
  work_dir.mkdir(parents=True, exist_ok=True)
  (work_dir / f"{key}.json").write_text(
    json.dumps([asdict(s) for s in stats], indent=2),
    encoding="utf-8",
  )
  (work_dir / f"{key}.stamp").write_text("done\n", encoding="utf-8")


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Measure the laser_score float32 scale-relative parity floor.",
  )
  parser.add_argument("--out", type=Path, required=True, help="result JSON path")
  parser.add_argument("--work-dir", type=Path, required=True, help="resume directory")
  parser.add_argument(
    "--self-test-only",
    action="store_true",
    help="run the instrument self-test and exit without touching an oracle",
  )
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

  script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  self_test = _self_test()
  if args.self_test_only:
    print(json.dumps(self_test, indent=2))  # noqa: T201
    return 0 if self_test["passed"] else 1

  payload: dict[str, Any] = {
    "self_test_passed": bool(self_test["passed"]),
    "self_test_n_checks": int(self_test["n_checks"]),
    "self_test_failed": list(self_test["failed"]),
    "oracle_absent": [],
    "wave": _WAVE,
    "headroom": _HEADROOM,
    "float32_eps": float(np.finfo(np.float32).eps),
    "degenerate_scale_threshold": _DEGENERATE_REL,
    "control_graded_relative": _CONTROL_GRADED,
    "control_scale_margin": _CONTROL_SCALE_MARGIN,
    "script_sha256": script_sha,
    "work_dir": str(args.work_dir),
    "n_reused": 0,
    "n_computed": 0,
  }

  if not self_test["passed"]:
    # Do not touch an oracle with an instrument that is not known to be able to refuse.
    payload["verdict"] = "instrument_unverified"
    _emit(args.out, payload)
    return 1

  root = _repo_root()
  _add_tests_to_path(root)

  try:
    measured = _collect(args.work_dir, script_sha, payload)
  except OracleAbsent as exc:
    payload["oracle_absent"] = [str(exc)]
    payload["verdict"] = "oracle_absent"
    _emit(args.out, payload)
    return 0

  f64, f32 = measured
  payload["n_pairs"] = len({s.pair for s in f32})
  payload["n_fields"] = len({s.field for s in f32})
  payload["f64_passes"] = all(s.n_violating == 0 for s in f64)
  payload["f32_violated_elementwise"] = any(s.n_violating for s in f32)
  payload["degenerate_fields"] = _degenerate(f32)

  floor, floor_pair, floor_field, second = _floor(f32)
  payload["floor_scale_relative"] = floor
  payload["floor_pair"] = floor_pair
  payload["floor_field"] = floor_field
  payload["second_worst_scale_relative"] = second
  payload["floor_scale_relative_per_field"] = _per_field_floor(f32)

  if math.isfinite(floor) and floor > 0.0:
    tol = _tol_from(floor)
  else:
    # An infinite or zero floor cannot produce a band; degenerate_fields or the
    # elementwise premise will carry the verdict.
    tol = math.inf if math.isinf(floor) else 0.0
  payload["proposed_tol"] = tol

  not_adm, loo_pair, loo_ratio = _leave_one_out(f32)
  payload["n_loo_not_admitted"] = not_adm
  payload["loo_all_admitted"] = not_adm == 0
  payload["loo_worst_pair"] = loo_pair
  payload["loo_worst_ratio"] = loo_ratio

  n_control_admitted, control_ratio = _control_scale_relative(f32, tol)
  payload["control_rejected"] = n_control_admitted == 0
  payload["control_scale_ratio"] = control_ratio
  payload["below_control_scale"] = control_ratio <= 1.0 / _CONTROL_SCALE_MARGIN

  payload["verdict"] = _grade(payload)
  _emit(args.out, payload)
  return 0


def _collect(
  work_dir: Path,
  script_sha: str,
  payload: dict[str, Any],
) -> tuple[list[FieldStats], list[FieldStats]]:
  """Measure every (pair, precision), reusing stamped units."""
  module = importlib.import_module(f"port.test_{_WAVE}")
  algo = importlib.import_module(f"port.reference.{_WAVE}.algo")
  pairs = list(module._PAIRS)  # noqa: SLF001
  if not pairs:
    msg = f"{_WAVE} oracle absent: no (checkpoint, fixture) pairs"
    raise OracleAbsent(msg)

  out: dict[str, list[FieldStats]] = {"f64": [], "f32": []}
  for precision in ("f64", "f32"):
    try:
      dump = algo.load(precision)
    except algo.OracleAbsentError as exc:
      raise OracleAbsent(str(exc)) from exc
    try:
      for checkpoint, fixture in pairs:
        pair = f"{checkpoint}__{fixture}"
        key = _unit_key(script_sha, pair, precision)
        cached = _load_unit(work_dir, key)
        if cached is not None:
          payload["n_reused"] += 1
          stats = cached
        else:
          stats = _measure_unit(module, dump, precision, checkpoint, fixture)
          _save_unit(work_dir, key, stats)
          payload["n_computed"] += 1
        out[precision].extend(stats)
    finally:
      dump.close()
  return out["f64"], out["f32"]


def _emit(out: Path, payload: dict[str, Any]) -> None:
  """Write the result to --out AND to $BTH_RESULTS_PATH when bathos set one.

  bathos grades only from $BTH_RESULTS_PATH, an adjacent .bth-results.json, or a registered
  --output-paths; a bare --out records outcome 'unknown' at exit 0.
  """
  text = json.dumps(payload, indent=2, sort_keys=True)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(text, encoding="utf-8")
  results = os.environ.get("BTH_RESULTS_PATH")
  if results:
    Path(results).write_text(text, encoding="utf-8")
  _LOG.info("verdict %s", payload.get("verdict"))


if __name__ == "__main__":
  raise SystemExit(main())
