"""Measure the laser_layers float32 parity floor, under BOTH candidate rules.

Pre-registered in ``laser_layers_f32_floor.bth.toml``, committed before this file ran.
Read that sidecar for the hypothesis and the outcome criteria; this module only
implements them.

WHY THIS RUN EXISTS, AND WHY IT IS NOT A WIDENING. ``tests/port/targets/laser_layers.toml``
says, in ``f32_atol_basis``, written before the f32 tier had ever executed:

    NOT measured. 1e-6 is the house pairing for rtol=1e-5 ... The f32 tier had not run
    when this was written, because the f64 tier gated it. If the f32 tier shows a
    deviation above this bound, measure it and amend separately rather than widening
    this value.

Graded run ``0f08f15a`` (the blank-waves vehicle) ran that tier for the first time and it
failed on all 63 pairs -- max abs deviation 3.933907e-06 against ``atol=1e-06+rtol=1e-05``
-- while tiers 1, 2 (f64) and 5 passed. So the condition named above has fired, and the
response it prescribes is this measurement. The band is NOT moved by this script.

TWO RULES, MEASURED SIDE BY SIDE, BECAUSE THE CHEAP ANSWER MAY BE THE RIGHT ONE.

  element-wise   max|got-ref|  <=  atol + rtol*|ref|      (the rule laser_layers has now)
  scale-relative max|got-ref|  <=  tol * max|ref|          (the rule the user chose for
                                                            laser_score on 261006)

``laser_score`` needed the second because no element-wise band could separate its noise
from a real defect: the worst violations sat on near-zero elements, which makes ``atol``
bind. That is a property of ``laser_score``'s magnitudes, NOT a general fact about LASEr
f32, and assuming it here would be assuming the answer. ``laser_layers`` is only ~4x over
its ``atol``, so an element-wise amendment may well clear the 10x defect margin on its
own. If it does, nothing changes form, nothing needs deciding, and the pre-registration's
"amend separately rather than widening" is satisfied by a measured ``atol``. This run
reports which rules clear the margin; it does not pick one.

WHAT MAKES THE FIELD ATTRIBUTION SOUND, since the obvious version is not.
``laser_f32_tolerance_floor._measure_unit`` recovers field names from ``module._FLOATS``,
a declared tuple. ``tests/port/test_laser_layers.py`` HAS NO ``_FLOATS``: its fields are
discovered per pair inside ``_run``, as ``sorted`` layer ids crossed with the npz's own
key order, and ``_rows_over`` is handed only ``(got, ref, rtol, atol)`` -- it never sees
the key. A recorder alone therefore cannot say which field a deviation belongs to, and
pairing recorded calls with a re-derived name list by position would silently mis-attribute
if either order ever drifted.

So positional pairing is VERIFIED, not assumed: for every recorded call, the ``ref`` array
is checked to be element-for-element identical to ``dump[name]`` for the name claimed at
that position. A re-ordering, an inserted comparison or a renamed key breaks that equality
and raises, rather than producing a plausible number against the wrong field.

Invocation:
  bth run --project-slug aminx --output-paths <json> -- \
    uv run --no-sync python3 scripts/analysis/laser_layers_f32_floor.py \
      --work-dir <dir> --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
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
  _control,  # noqa: PLC2701
  _Recorder,  # noqa: PLC2701
  _repo_root,  # noqa: PLC2701
  _stats,  # noqa: PLC2701
)

_LOG = logging.getLogger("laser_layers_f32_floor")

_WAVE = "laser_layers"
#: The proposed band must sit this far below the injected defect's size.
_HEADROOM = 10.0
#: Injected defect size. Inherited UNCHANGED from laser_f32_tolerance_floor and
#: laser_score_scale_relative_floor so this run's margin is directly comparable to
#: e47bfaab's and 3af3be02's. Not scaled to the measured floor -- a floor-scaled
#: control is near-circular when the band is also derived from the floor.
_CONTROL_GRADED = 1e-2
_CONTROL_SCALE_MARGIN = 10.0
#: A (pair, field) is degenerate when its max|ref| is this fraction of the largest field
#: scale in that pair, or smaller. Fixed in the sidecar before the run.
_DEGENERATE_REL = 1e-6


def _field_names(dump: Any, checkpoint: str, fixture: str) -> list[str]:
  """The npz keys ``_run`` compares, in the exact order it compares them.

  This mirrors ``tests/port/test_laser_layers.py::_run`` line for line: layer ids are
  the sorted set of ``__out__``-bearing suffixes under the pair's base, and within each
  layer the out keys come in the npz's own ``files`` order, which ``_run`` does NOT sort.
  Reproducing that order is only half the guarantee; ``_measure_unit`` then verifies each
  pairing against the dump rather than trusting it.
  """
  base = f"{checkpoint}__{fixture}__layer__"
  layer_ids = sorted(
    {
      key[len(base) :].split("__out__", 1)[0]
      for key in dump.files
      if key.startswith(base) and "__out__" in key
    },
  )
  names: list[str] = []
  for layer_id in layer_ids:
    prefix = base + layer_id + "__"
    names.extend(key for key in dump.files if key.startswith(prefix + "out__"))
  return names


def _measure_unit(
  module: Any,
  dump: Any,
  precision: str,
  checkpoint: str,
  fixture: str,
) -> list[FieldStats]:
  """Run one (pair, precision) through the test module's own ``_run``.

  The prediction path is not re-expressed here: ``_run`` loads the checkpoint, builds the
  modules and produces the arrays exactly as the wave does. Only the comparison is
  observed.
  """
  rtol = float(module._RTOL[precision])  # noqa: SLF001
  atol = float(module._ATOL[precision])  # noqa: SLF001
  names = _field_names(dump, checkpoint, fixture)

  original = module._rows_over  # noqa: SLF001
  recorder = _Recorder(original)
  module._rows_over = recorder  # noqa: SLF001
  try:
    try:
      module._run(dump, precision, checkpoint, fixture, numeric=True)  # noqa: SLF001
    except AssertionError as exc:
      # Expected whenever the band is violated. The arrays are already recorded.
      _LOG.info("%s %s %s: %s", precision, checkpoint, fixture, str(exc).splitlines()[0])
  finally:
    module._rows_over = original  # noqa: SLF001

  if len(recorder.calls) != len(names):
    msg = (
      f"recorded {len(recorder.calls)} comparisons but enumerated {len(names)} out keys "
      f"for {checkpoint}__{fixture} at {precision}; test_laser_layers._run's plumbing "
      f"changed and this measurement would no longer describe it"
    )
    raise RuntimeError(msg)

  # THE POSITIONAL PAIRING IS VERIFIED, NOT ASSUMED. Without _FLOATS there is no declared
  # field order to lean on, so each recorded reference array is checked against the dump
  # entry for the name claimed at that position.
  for index, (name, (_got, ref)) in enumerate(zip(names, recorder.calls, strict=True)):
    expected = np.asarray(dump[name])
    actual = np.asarray(ref)
    if expected.shape != actual.shape or not np.array_equal(
      expected, actual, equal_nan=True,
    ):
      msg = (
        f"comparison {index} for {checkpoint}__{fixture} at {precision} does not match "
        f"dump key {name!r}: the recorded reference differs from the dump entry, so the "
        f"call order and the enumerated key order have diverged"
      )
      raise RuntimeError(msg)

  pair = f"{checkpoint}__{fixture}"
  return [
    _stats(pair, name, got, ref, rtol, atol)
    for name, (got, ref) in zip(names, recorder.calls, strict=True)
  ]


def _scale_ratio(stat: FieldStats) -> float:
  """``max|got-ref| / max|ref|`` for one (pair, field), with no scale handled."""
  if stat.max_ref_magnitude > 0.0:
    return stat.max_abs / stat.max_ref_magnitude
  return 0.0 if stat.max_abs == 0.0 else float("inf")


def _by_pair(stats: list[FieldStats]) -> dict[str, list[FieldStats]]:
  out: dict[str, list[FieldStats]] = {}
  for stat in stats:
    out.setdefault(stat.pair, []).append(stat)
  return out


def _degenerate(stats: list[FieldStats]) -> list[str]:
  """Fields with no scale to be relative to, per pair.

  The scale-relative rule's own version of the failure it cures: dividing by a near-zero
  maximum recreates the near-zero denominator that made ``atol`` bind element-wise.
  """
  bad: list[str] = []
  for pair, group in _by_pair(stats).items():
    largest = max((s.max_ref_magnitude for s in group), default=0.0)
    if largest <= 0.0:
      bad.extend(f"{pair}::{s.field}" for s in group)
      continue
    threshold = _DEGENERATE_REL * largest
    bad.extend(f"{pair}::{s.field}" for s in group if s.max_ref_magnitude <= threshold)
  return sorted(bad)


def _scale_floor(stats: list[FieldStats]) -> tuple[float, str, str, float]:
  """Worst scale-relative ratio, the (pair, field) carrying it, and the runner-up."""
  ranked = sorted(
    ((_scale_ratio(s), s.pair, s.field) for s in stats), reverse=True,
  )
  if not ranked:
    return 0.0, "", "", 0.0
  worst, pair, field_name = ranked[0]
  second = ranked[1][0] if len(ranked) > 1 else 0.0
  return float(worst), pair, field_name, float(second)


def _leave_one_out(stats: list[FieldStats]) -> tuple[int, str, float]:
  """Derive the scale-relative band from 62 pairs; require the 63rd admitted.

  Plain "all admitted" is TRUE BY CONSTRUCTION under a single scale-relative statistic --
  the floor is the maximum of exactly the quantity admission tests -- so it is not a
  criterion. This is the version that can fail, and it fails exactly when the worst pair
  is an outlier rather than a ceiling.
  """
  grouped = _by_pair(stats)
  worst_pair = ""
  worst_ratio = 0.0
  n_bad = 0
  for held_out, group in grouped.items():
    rest = [s for pair, members in grouped.items() if pair != held_out for s in members]
    if not rest:
      continue
    floor, _, _, _ = _scale_floor(rest)
    if floor <= 0.0:
      continue
    tol = _ceil_one_significant_figure(_HEADROOM * floor)
    held_worst = max((_scale_ratio(s) for s in group), default=0.0)
    if not held_worst <= tol:
      n_bad += 1
    ratio = held_worst / tol if tol > 0.0 else float("inf")
    if ratio > worst_ratio:
      worst_ratio = ratio
      worst_pair = held_out
  return n_bad, worst_pair, float(worst_ratio)


def _elementwise_proposal(stats: list[FieldStats], rtol: float) -> float:
  """The smallest ``atol`` that admits everything at the CURRENT ``rtol``, times headroom.

  ``abs_excess`` is ``max(err - rtol*|ref|)``, i.e. exactly the atol each field needs, so
  the maximum over fields is the floor and no search is required. ``rtol`` is deliberately
  held at its current value: the pre-registration says amend the unmeasured ``atol``, and
  moving both terms at once would make the result unattributable.
  """
  floor = max((s.abs_excess for s in stats), default=0.0)
  if floor <= 0.0:
    return 0.0
  return _ceil_one_significant_figure(_HEADROOM * floor)


def _grade(payload: dict[str, Any]) -> str:
  """Outcome names and their order are fixed in the sidecar, before the run."""
  if not payload["self_test_passed"]:
    return "instrument_unverified"
  if payload["oracle_absent"]:
    return "oracle_absent"
  if not payload["f64_passes"]:
    return "defect_not_tolerance"
  if not payload["f32_violated_elementwise"]:
    return "elementwise_was_fine"
  if payload["degenerate_fields"]:
    return "scale_is_degenerate"
  if payload["elementwise_clears_margin"]:
    return "elementwise_suffices"
  if not payload["below_control_scale"]:
    return "noise_reaches_defect_scale"
  if not payload["loo_all_admitted"]:
    return "band_does_not_generalise"
  if not payload["control_rejected"]:
    return "fail"
  return "scale_relative_needed"


def _base_payload(**over: Any) -> dict[str, Any]:
  payload: dict[str, Any] = {
    "self_test_passed": True,
    "oracle_absent": [],
    "f64_passes": True,
    "f32_violated_elementwise": True,
    "degenerate_fields": [],
    "elementwise_clears_margin": False,
    "below_control_scale": True,
    "loo_all_admitted": True,
    "control_rejected": True,
  }
  payload.update(over)
  return payload


def _synth(pair: str, field_name: str, got: Any, ref: Any, rtol: float, atol: float):
  return _stats(pair, field_name, np.asarray(got, dtype=np.float64),
                np.asarray(ref, dtype=np.float64), rtol, atol)


def _self_test() -> dict[str, Any]:
  """Synthetic checks, run BEFORE any oracle is touched.

  Several must REFUSE. A grader that cannot return a non-pass is not an instrument, and a
  floor measured by one is the #2432 revision-1 failure repeated.
  """
  failed: list[str] = []
  checks = 0

  def check(name: str, condition: bool) -> None:
    nonlocal checks
    checks += 1
    if not condition:
      failed.append(name)

  # --- rounding, inherited but re-pinned here so a shared-module change is caught ---
  check("round_up_1sf", _ceil_one_significant_figure(1.43e-6) == 2e-6)
  check("round_up_exact", _ceil_one_significant_figure(2e-6) == 2e-6)

  # --- scale-relative arithmetic on a hand-computed case ---
  ref = np.array([1.0, -4.0, 2.0])
  got = np.array([1.0, -4.0, 2.0 + 8e-6])
  stat = _synth("p", "f", got, ref, 1e-5, 1e-6)
  # Relative, not absolute: 2.0 + 8e-6 is not representable, so the realised deviation is
  # 7.99999999978e-06 and an absolute 1e-18 window would be testing float dust.
  check(
    "scale_ratio_hand",
    abs(_scale_ratio(stat) - (8e-6 / 4.0)) <= 1e-9 * (8e-6 / 4.0),
  )
  check("identity_ratio_zero", _scale_ratio(_synth("p", "f", ref, ref, 1e-5, 1e-6)) == 0.0)

  # --- the element-wise proposal is exactly the atol the data needs ---
  # The case above does NOT violate: rtol*|ref| = 2e-5 already covers its 8e-6 error, so
  # abs_excess clamps to 0 and there is nothing to propose. Pin that, then measure a case
  # that genuinely violates. Getting this backwards is how a floor gets derived from data
  # that was never over the band.
  check("elementwise_floor_zero_when_admitted", stat.abs_excess == 0.0)
  check("elementwise_proposal_zero_when_admitted", _elementwise_proposal([stat], 1e-5) == 0.0)

  # err = 5e-5 at |ref| = 1.0 against rtol=1e-5, atol=1e-6: band is 1.1e-5, so this is
  # over. abs_excess = 5e-5 - 1e-5 = 4e-5 is exactly the atol it needs at that rtol.
  bad = _synth("p", "f", np.array([1.0 + 5e-5]), np.array([1.0]), 1e-5, 1e-6)
  need = 5e-5 - 1e-5
  check("violating_case_is_violating", bad.n_violating == 1)
  check("elementwise_floor_is_abs_excess", abs(bad.abs_excess - need) <= 1e-9 * need)
  proposal = _elementwise_proposal([bad], 1e-5)
  # Pinned against the MEASURED floor, not against the idealised 4e-5, and the difference
  # is not pedantry. DISCLOSED PROPERTY OF THE SHARED ROUNDING: the realised abs_excess
  # here is 4.0000000000105516e-05, a hair above 4e-5, and _ceil_one_significant_figure
  # rounds 10x that UP to 5e-4 rather than 4e-4 -- float dust moves the proposal by 25%.
  # It always errs WIDE, never tight, and the helper is inherited unchanged from #2432 and
  # used by two already-graded runs, so it is recorded here rather than altered. Asserting
  # against the idealised value instead would have hidden it.
  check(
    "elementwise_proposal_rounds_up",
    proposal == _ceil_one_significant_figure(_HEADROOM * bad.abs_excess),
  )
  check("rounding_errs_wide_not_tight", proposal >= _HEADROOM * bad.abs_excess)
  # And it must actually admit what it was derived from.
  check("elementwise_proposal_admits", bad.abs_excess <= proposal)

  # --- degenerate scale detection, at and either side of the threshold ---
  big = _synth("q", "big", np.array([1.0]), np.array([1.0]), 1e-5, 1e-6)
  tiny = _synth("q", "tiny", np.array([1e-9]), np.array([1e-9]), 1e-5, 1e-6)
  at = _synth("q", "at", np.array([1e-6]), np.array([1e-6]), 1e-5, 1e-6)
  over = _synth("q", "over", np.array([1e-5]), np.array([1e-5]), 1e-5, 1e-6)
  check("degenerate_catches_tiny", "q::tiny" in _degenerate([big, tiny]))
  check("degenerate_at_threshold", "q::at" in _degenerate([big, at]))
  check("degenerate_spares_over", "q::over" not in _degenerate([big, over]))
  check("degenerate_empty_when_uniform", _degenerate([big]) == [])

  # --- leave-one-out: admits a uniform population, REFUSES a planted outlier ---
  uniform = [
    _synth(f"p{i}", "f", np.array([1.0 + 1e-7]), np.array([1.0]), 1e-5, 1e-6)
    for i in range(5)
  ]
  n_bad, _, _ = _leave_one_out(uniform)
  check("loo_admits_uniform", n_bad == 0)
  outlier = [*uniform, _synth("odd", "f", np.array([1.0 + 1e-2]), np.array([1.0]), 1e-5, 1e-6)]
  n_bad_out, worst_pair, _ = _leave_one_out(outlier)
  check("loo_refuses_outlier", n_bad_out >= 1 and worst_pair == "odd")

  # --- shared vs one-sided NaN ---
  nan_both = _synth("p", "f", np.array([np.nan, 1.0]), np.array([np.nan, 1.0]), 1e-5, 1e-6)
  check("shared_nan_is_not_error", _scale_ratio(nan_both) == 0.0)
  nan_one = _synth("p", "f", np.array([np.nan, 1.0]), np.array([0.5, 1.0]), 1e-5, 1e-6)
  check("one_sided_nan_is_infinite", _scale_ratio(nan_one) == float("inf"))

  # --- FIVE NEGATIVE CONTROLS ON THE GRADER ITSELF. None may propose a band. ---
  check(
    "grades_instrument_unverified",
    _grade(_base_payload(self_test_passed=False)) == "instrument_unverified",
  )
  check(
    "grades_defect_not_tolerance",
    _grade(_base_payload(f64_passes=False)) == "defect_not_tolerance",
  )
  check(
    "grades_elementwise_was_fine",
    _grade(_base_payload(f32_violated_elementwise=False)) == "elementwise_was_fine",
  )
  check(
    "grades_scale_is_degenerate",
    _grade(_base_payload(degenerate_fields=["p::f"])) == "scale_is_degenerate",
  )
  check(
    "grades_band_does_not_generalise",
    _grade(_base_payload(loo_all_admitted=False)) == "band_does_not_generalise",
  )
  check(
    "grades_noise_reaches_defect_scale",
    _grade(_base_payload(below_control_scale=False)) == "noise_reaches_defect_scale",
  )
  # The two PASS-shaped outcomes are distinct and the cheap one wins when available.
  check(
    "grades_elementwise_suffices",
    _grade(_base_payload(elementwise_clears_margin=True)) == "elementwise_suffices",
  )
  check("grades_scale_relative_needed", _grade(_base_payload()) == "scale_relative_needed")
  # elementwise_suffices must NOT mask a degenerate scale or an f64 defect.
  check(
    "degenerate_outranks_elementwise",
    _grade(_base_payload(elementwise_clears_margin=True, degenerate_fields=["p::f"]))
    == "scale_is_degenerate",
  )
  check(
    "f64_outranks_elementwise",
    _grade(_base_payload(elementwise_clears_margin=True, f64_passes=False))
    == "defect_not_tolerance",
  )

  return {
    "self_test_passed": not failed,
    "self_test_n_checks": checks,
    "self_test_failed": failed,
  }


def _unit_key(script_sha: str, pair: str, precision: str) -> str:
  raw = f"{script_sha}|{_WAVE}|{pair}|{precision}".encode()
  return hashlib.sha256(raw).hexdigest()


def _load_unit(work_dir: Path, key: str) -> list[FieldStats] | None:
  stamp = work_dir / f"unit_{key}.done"
  body = work_dir / f"unit_{key}.json"
  if not (stamp.is_file() and body.is_file()):
    return None
  try:
    rows = json.loads(body.read_text())
    return [FieldStats(**row) for row in rows]
  except (json.JSONDecodeError, TypeError, ValueError):
    # A body that no longer matches the dataclass is recomputed, never coerced.
    return None


def _save_unit(work_dir: Path, key: str, stats: list[FieldStats]) -> None:
  """Body first, stamp second, so a crash between them recomputes rather than trusts."""
  (work_dir / f"unit_{key}.json").write_text(
    json.dumps([asdict(s) for s in stats], indent=2),
  )
  (work_dir / f"unit_{key}.done").write_text("done\n")


def _collect(
  module: Any,
  algo: Any,
  pairs: list[tuple[str, str]],
  work_dir: Path,
  script_sha: str,
) -> tuple[dict[str, list[FieldStats]], list[str], list[str]]:
  measured: dict[str, list[FieldStats]] = {"f64": [], "f32": []}
  reused: list[str] = []
  computed: list[str] = []
  for precision in ("f64", "f32"):
    try:
      dump = algo.load(precision)
    except algo.OracleAbsentError as exc:  # pragma: no cover - environment
      raise OracleAbsent(str(exc)) from exc
    try:
      for checkpoint, fixture in pairs:
        pair = f"{checkpoint}__{fixture}"
        key = _unit_key(script_sha, pair, precision)
        cached = _load_unit(work_dir, key)
        if cached is not None:
          measured[precision].extend(cached)
          reused.append(f"{precision}:{pair}")
          continue
        stats = _measure_unit(module, dump, precision, checkpoint, fixture)
        _save_unit(work_dir, key, stats)
        measured[precision].extend(stats)
        computed.append(f"{precision}:{pair}")
        _LOG.info("%s %s: %d fields", precision, pair, len(stats))
    finally:
      dump.close()
  return measured, reused, computed


def _emit(out: Path, payload: dict[str, Any]) -> None:
  out.parent.mkdir(parents=True, exist_ok=True)
  text = json.dumps(payload, indent=2, sort_keys=True)
  out.write_text(text)
  results = os.environ.get("BTH_RESULTS_PATH")
  if results:
    Path(results).write_text(text)
    _LOG.info("emitted result to BTH_RESULTS_PATH=%s", results)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--work-dir", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
  )
  args.work_dir.mkdir(parents=True, exist_ok=True)

  script_sha = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
  self_test = _self_test()
  if not self_test["self_test_passed"]:
    payload = {
      **self_test,
      "verdict": "instrument_unverified",
      "wave": _WAVE,
      "script_sha256": script_sha,
    }
    _emit(args.out, payload)
    _LOG.error("instrument unverified: %s", self_test["self_test_failed"])
    return 1
  _LOG.info("self-test passed (%d checks)", self_test["self_test_n_checks"])

  root = _repo_root()
  _add_tests_to_path(root)
  module = importlib.import_module(f"port.test_{_WAVE}")
  algo = importlib.import_module(f"port.reference.{_WAVE}.algo")

  pairs = list(module._PAIRS)  # noqa: SLF001
  payload: dict[str, Any] = {
    **self_test,
    "wave": _WAVE,
    "script_sha256": script_sha,
    "work_dir": str(args.work_dir),
    "oracle_absent": [],
    "headroom": _HEADROOM,
    "control_graded_relative": _CONTROL_GRADED,
    "control_scale_margin": _CONTROL_SCALE_MARGIN,
    "degenerate_scale_threshold": _DEGENERATE_REL,
    "current_rtol_f32": float(module._RTOL["f32"]),  # noqa: SLF001
    "current_atol_f32": float(module._ATOL["f32"]),  # noqa: SLF001
  }

  if not pairs:
    payload.update({"verdict": "oracle_absent", "oracle_absent": [_WAVE]})
    _emit(args.out, payload)
    return 0

  try:
    measured, reused, computed = _collect(module, algo, pairs, args.work_dir, script_sha)
  except OracleAbsent as exc:
    payload.update({"verdict": "oracle_absent", "oracle_absent": [str(exc)]})
    _emit(args.out, payload)
    return 0

  f64 = measured["f64"]
  f32 = measured["f32"]
  rtol32 = float(module._RTOL["f32"])  # noqa: SLF001

  # The f64 precondition is RE-MEASURED in this tree, never inherited. A floor measured
  # over a defect describes the defect.
  f64_passes = not any(s.n_violating for s in f64)
  f32_violated = any(s.n_violating for s in f32)

  degenerate = _degenerate(f32)
  floor_scale, floor_pair, floor_field, second = _scale_floor(f32)
  proposed_scale = _ceil_one_significant_figure(_HEADROOM * floor_scale) if floor_scale > 0 else 0.0
  proposed_atol = _elementwise_proposal(f32, rtol32)
  floor_abs = max((s.abs_excess for s in f32), default=0.0)

  n_loo_bad, loo_pair, loo_ratio = _leave_one_out(f32)

  # Element-wise margin: the proposed atol must stay _HEADROOM below the injected defect.
  n_control_admitted, control_ratio = _control(f32, rtol32, proposed_atol, _CONTROL_GRADED)
  elementwise_clears = bool(
    proposed_atol > 0.0 and n_control_admitted == 0 and control_ratio <= 1.0 / _HEADROOM,
  )

  # Scale-relative margin, on the same injected defect size.
  control_scale_ratio = (
    proposed_scale / _CONTROL_GRADED if _CONTROL_GRADED > 0 else float("inf")
  )
  below_control_scale = control_scale_ratio <= 1.0 / _CONTROL_SCALE_MARGIN
  scale_control_rejected = bool(proposed_scale > 0.0 and proposed_scale < _CONTROL_GRADED)

  payload.update(
    {
      "f64_passes": f64_passes,
      "f32_violated_elementwise": f32_violated,
      "degenerate_fields": degenerate,
      "floor_scale_relative": floor_scale,
      "floor_pair": floor_pair,
      "floor_field": floor_field,
      "second_worst_scale_relative": second,
      "proposed_scale_rel": proposed_scale,
      "floor_abs_excess": floor_abs,
      "proposed_atol_elementwise": proposed_atol,
      "elementwise_clears_margin": elementwise_clears,
      "n_control_admitted_elementwise": n_control_admitted,
      "control_band_defect_ratio_elementwise": control_ratio,
      "below_control_scale": below_control_scale,
      "control_scale_ratio": control_scale_ratio,
      "control_rejected": scale_control_rejected,
      "loo_all_admitted": n_loo_bad == 0,
      "n_loo_not_admitted": n_loo_bad,
      "loo_worst_pair": loo_pair,
      "loo_worst_ratio": loo_ratio,
      "n_pairs": len(pairs),
      "n_fields": len({s.field for s in f32}),
      "n_reused": len(reused),
      "n_computed": len(computed),
      "max_abs_f32": max((s.max_abs for s in f32), default=0.0),
      "max_abs_f64": max((s.max_abs for s in f64), default=0.0),
    },
  )
  payload["verdict"] = _grade(payload)

  _emit(args.out, payload)
  _LOG.info("verdict=%s", payload["verdict"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
