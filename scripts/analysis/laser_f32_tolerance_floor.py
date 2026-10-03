"""Measure the float32 parity floor for the laser_score and laser_decode_step waves.

WHY THIS EXISTS

The redsox gate exits ``rc = 1`` on exactly two waves, both at tier 3:
``tests/port/test_laser_score.py::test_tier_3_f32`` and
``tests/port/test_laser_decode_step.py::test_tier_3_f32``. Both assert
``rtol=1e-4, atol=1e-7`` against a sealed upstream dump, and both target TOMLs
say so in writing::

    [tolerance_basis]
    f32_atol_basis = "house-paired and NOT measured"

So the band that is failing was never measured. Debt #2445. The standing rule
is that a pre-registered band is never loosened to make a run pass, and this
script does not loosen one -- it measures the floor, which is the only thing
that can legitimately close #2445, exactly as #2432 was closed by
``potts_jit_eager_f32_floor.py``.

WHAT MAKES THIS DIFFERENT FROM #2432, AND WHY ONE BAR IS A HARD PRECONDITION

#2432 compared an eager pass against a jitted pass on IDENTICAL float32
inputs. Nothing but XLA reassociation could differ, so a measured floor was
automatically a statement about float32 and not about the model.

That is NOT true here. This compares aminx against upstream LASErMPNN, where a
deviation can perfectly well be a real algorithmic defect, and a floor measured
over a defect would launder it into a tolerance. The discriminator is the f64
tier: both waves are expected to pass ``test_tier_2_f64`` at
``rtol=1e-8, atol=1e-11``, which is ~1e4 times tighter than the f32 band being
discussed. Math that agrees to 1e-8 in float64 and disagrees at 1e-4 in float32
is accumulating differently, not computing something else.

So ``f64_passes`` is a PRECONDITION, re-measured here rather than assumed. If
f64 does not pass, this run proposes nothing and returns
``defect_not_tolerance``: the right response then is to fix the port, and
widening an f32 band would be hiding it.

WHICH TERM BINDS IS MEASURED, NOT ASSUMED

numpy's band is ``atol + rtol*|ref|``. In #2432 atol was obviously the binding
term -- the rtol contribution was 0.1% of the band -- and only atol moved. Here
it is genuinely unknown in advance: ``atol=1e-7`` is below one float32 ulp for
any ``|ref| > ~1``, so on large logits the rtol term should dominate, while on
near-zero entries atol should. The rule below reads the binding term off the
measurement and moves ONLY that term. Moving a parameter that is not violated
would be widening for its own sake.

THE RULE, FIXED BEFORE THIS RUN

Per wave, independently. The two waves get their own numbers: they are
different computations, and a shared band would be the looser of the two.

    V       = the elements that violate the CURRENT f32 band
    e*      = the element of V with the largest violation ratio |err| / band
              -- the element the test's own failure message reports
    binding = "atol" if atol >= rtol*|ref(e*)| else "rtol"

    if binding == "atol":
        A = max ABSOLUTE deviation over every pair, field and element
        proposed_atol = ceil_one_significant_figure(10 * A)
        proposed_rtol = unchanged
    else:
        R = max RELATIVE deviation over elements with |ref| > 0.1
        proposed_rtol = ceil_one_significant_figure(10 * R)
        proposed_atol = unchanged

The conditioning threshold 0.1 and the one-significant-figure rounding are both
inherited deliberately from #2432 revision 2. Unconditioned relative deviation
is an ill-conditioned statistic -- it divides by a magnitude that is sometimes
near zero, so it measures its own denominator; #2432 revision 1 returned a
relative floor of 5.1e-02 against an absolute floor of 1.43e-06 for exactly
that reason, and the pre-registered scale guard refusing to proceed is the only
thing that stopped an rtol of 1.0 being written into a test. Rounding to one
significant figure rather than the next power of ten is from the same lesson: a
power of ten turned a 1.43e-06 floor into 1e-04, a 70x jump that would have
left everything below 1e-04 unchecked.

Note that the conditioned floor R is used to PROPOSE, while admission below is
checked against every element including the unconditioned ones. If the
conditioned floor cannot admit the full population, this run fails and says so
rather than conditioning the admission check to match.

ADMISSION IS EXACT, NOT RECONSTRUCTED FROM MAXIMA

#2432 recorded that reconstructing numpy's verdict from recorded maxima is
unsound, because the max-absolute and max-relative elements need not be the
same element, and re-ran the comparison per candidate band instead. Re-running
a LASEr forward pass per candidate band is far more expensive, so this script
records a sufficient statistic that makes admission exact in one pass.

Because exactly one term moves and the other stays at its current value, there
are only two shapes, and each has a closed form over the elements of a field:

    atol moves, rtol fixed at rtol_cur:
        admitted(atol_new) iff atol_new >= max_i( err_i - rtol_cur*|ref_i| )
        recorded as abs_excess

    rtol moves, atol fixed at atol_cur:
        admitted(rtol_new) iff rtol_new >= max_{|ref_i|>0}( (err_i - atol_cur)/|ref_i| )
                            and max_{|ref_i|=0}( err_i ) <= atol_cur
        recorded as rel_excess and zero_mag_worst

Both are maxima over every element, so neither depends on which element
happened to be worst by some other statistic. The second condition is a real
failure mode with its own counter: an element whose reference is exactly zero
cannot be admitted by any rtol, so if one of those violates atol_cur, no rtol
closes this wave.

THE BARS, THREE OF WHICH CAN FAIL

  1. f64_passes -- every f64 element inside rtol=1e-8, atol=1e-11. A hard
     precondition, see above. Failing it means the port is wrong.
  2. f32_violated -- at least one element outside the current f32 band. If the
     band is NOT violated then it was fine and the gate's rc=1 came from
     somewhere else: do not touch the tolerance, go find the real cause.
  3. all_admitted -- every element of every pair and field inside the PROPOSED
     band, by the exact criterion above. A band fitted to the single worst
     element but excluding another is not a floor.
  4. control_rejected -- a fixed relative defect injected at the
     largest-magnitude element of each float field is REJECTED at the proposed
     band. A tolerance that admits everything is not a test, and a positive
     control that can only pass is not a check. The defect size is fixed in
     advance and NOT scaled to the measured floor, because a floor-scaled
     control is near-circular when the band is derived from the floor.
  5. below_control_scale -- the proposed band at the injected element stays at
     least 10x below that defect's absolute size, so the band cannot swallow
     the thing it exists to catch.

WHY THE GRADED CONTROL IS 1e-2 RELATIVE AND THE 1e-3 ONE IS ONLY REPORTED

#2432 graded on a 1e-3 relative defect. That cannot be the bar here, and the
reason is arithmetic rather than convenience: against a 1e-3 relative defect, a
band of ``rtol*|ref|`` clears a 10x margin only when ``rtol <= 1e-4``. The
CURRENT rtol is already exactly 1e-4, so any rtol this run could propose would
fail a 1e-3 margin -- which would make the bar unsatisfiable by construction in
the rtol branch and would be demanding of the new band something the old band
never had.

The graded defect is therefore 1e-2 relative, the order of a genuine
algorithmic divergence in logits, with the same 10x margin. The 1e-3 control is
still measured and reported as ``control_1e3_margin`` so the weaker guarantee
is visible rather than quietly dropped, and the sidecar states that this run
does not claim a 1e-3 defect would be caught with margin.

HOW THE DEVIATIONS ARE OBTAINED, AND WHY NOT BY REIMPLEMENTING THE COMPARISON

The arrays are taken from the test module's OWN ``_run``, by wrapping its
``_rows_over`` with a recorder that delegates to the original. Nothing about
the checkpoint loading, the feature plumbing, the decoding-order feed or the
dump key layout is re-expressed here. That is deliberate: a second
implementation of the comparison is a second thing that can be wrong, and a
measurement defect is precisely how #2432 revision 1 produced a number that
could not be true. The recorder asserts it saw exactly one call per declared
float field, in order, so if the test's plumbing changes this script fails
loudly instead of silently measuring something else.

RESUME

One unit is one (wave, pair, precision) triple. Each unit's JSON body is
written before its stamp, so a crash between them leaves an unstamped body that
is recomputed rather than trusted. The cache key includes the sha256 of this
script's bytes, so editing the rule recomputes everything rather than reusing a
previous revision's units.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_LOG = logging.getLogger("laser_f32_tolerance_floor")

_WAVES = ("laser_score", "laser_decode_step")
_CONDITION_THRESHOLD = 0.1
_HEADROOM = 10.0
_CONTROL_GRADED = 1e-2
_CONTROL_REPORTED = 1e-3
_CONTROL_SCALE_MARGIN = 10.0
_F32_EPS = float(np.finfo(np.float32).eps)


def _repo_root() -> Path:
  here = Path(__file__).resolve()
  for parent in here.parents:
    if (parent / "pyproject.toml").is_file():
      return parent
  msg = f"no pyproject.toml above {here}"
  raise RuntimeError(msg)


def _add_tests_to_path(root: Path) -> None:
  """Put ``tests`` LAST on sys.path.

  ``tests/`` holds directories named ``io``, ``types``, ``model`` and ``data``.
  Prepending it would shadow the stdlib modules of the same name for anything
  imported lazily afterwards. Appending still resolves ``port`` -- nothing else
  on the path provides it -- while leaving stdlib and site-packages to win every
  collision.
  """
  tests = str(root / "tests")
  if tests not in sys.path:
    sys.path.append(tests)


def _ceil_one_significant_figure(value: float) -> float:
  """Round UP to one significant figure. 1.43e-06 -> 2e-06, not 1e-04."""
  if value <= 0.0 or not math.isfinite(value):
    msg = f"cannot round {value!r} to one significant figure"
    raise ValueError(msg)
  exponent = math.floor(math.log10(value))
  scale = 10.0**exponent
  return float(math.ceil(value / scale) * scale)


class OracleAbsent(RuntimeError):
  """The sealed dump for a wave is not present in this checkout."""


@dataclass
class FieldStats:
  """Per (pair, field) deviation statistics at one precision.

  ``abs_excess``, ``rel_excess`` and ``zero_mag_worst`` are the sufficient
  statistics that make admission at a proposed band exact; see the module
  docstring. Everything else is reporting.
  """

  pair: str
  field: str
  n_elements: int
  max_abs: float
  max_abs_ref: float
  max_ref_magnitude: float
  max_rel_unconditioned: float
  max_rel_conditioned: float
  n_conditioned: int
  n_violating: int
  worst_violation_ratio: float
  worst_violation_ref: float
  worst_violation_abs: float
  worst_violation_atol_share: float
  abs_excess: float
  rel_excess: float
  zero_mag_worst: float
  n_zero_mag: int


@dataclass
class WaveResult:
  """Everything measured and derived for one wave."""

  wave: str
  current_rtol_f32: float
  current_atol_f32: float
  current_rtol_f64: float
  current_atol_f64: float
  n_pairs: int
  n_fields: int
  f32: list[FieldStats] = field(default_factory=list)
  f64: list[FieldStats] = field(default_factory=list)
  f64_passes: bool = False
  f32_violated: bool = False
  floor_abs: float = 0.0
  floor_rel_unconditioned: float = 0.0
  floor_rel_conditioned: float = 0.0
  floor_abs_in_eps: float = 0.0
  binding_term: str = ""
  binding_ref: float = 0.0
  binding_atol_share: float = 0.0
  proposed_rtol: float = 0.0
  proposed_atol: float = 0.0
  all_admitted: bool = False
  n_not_admitted: int = 0
  n_zero_mag_blocked: int = 0
  control_rejected: bool = False
  n_control_admitted: int = 0
  below_control_scale: bool = False
  control_scale_ratio: float = 0.0
  control_1e3_margin: float = 0.0
  reused_units: list[str] = field(default_factory=list)
  computed_units: list[str] = field(default_factory=list)


class _Recorder:
  """Wrap ``_rows_over`` so every compared array pair is captured.

  Delegates to the original so ``_run`` behaves exactly as the test does,
  including raising its own AssertionError with its own message.
  """

  def __init__(self, original: Any) -> None:
    self._original = original
    self.calls: list[tuple[np.ndarray, np.ndarray]] = []

  def __call__(
    self,
    got: np.ndarray,
    ref: np.ndarray,
    rtol: float,
    atol: float,
  ) -> tuple[int, int, float]:
    self.calls.append((np.asarray(got), np.asarray(ref)))
    return self._original(got, ref, rtol, atol)  # type: ignore[no-any-return]


def _stats(
  pair: str,
  field_name: str,
  got: np.ndarray,
  ref: np.ndarray,
  rtol: float,
  atol: float,
) -> FieldStats:
  """Deviation statistics for one compared array pair.

  Shared NaNs are not errors, matching ``_rows_over``. A NaN on one side only is
  an infinite deviation and shows up as such rather than being dropped.
  """
  got64 = np.asarray(got, dtype=np.float64).ravel()
  ref64 = np.asarray(ref, dtype=np.float64).ravel()
  both_nan = np.isnan(got64) & np.isnan(ref64)
  nan_mismatch = np.isnan(got64) != np.isnan(ref64)
  err = np.abs(got64 - ref64)
  err = np.where(both_nan, 0.0, err)
  err = np.where(nan_mismatch, np.inf, err)
  mag = np.abs(np.where(np.isnan(ref64), 0.0, ref64))

  band = atol + rtol * mag
  violating = err > band
  ratio = np.where(band > 0.0, err / band, np.inf)

  worst_abs_idx = int(np.argmax(err)) if err.size else 0
  nonzero = mag > 0.0
  conditioned = mag > _CONDITION_THRESHOLD
  rel_uncond = float(np.max(err[nonzero] / mag[nonzero])) if np.any(nonzero) else 0.0
  rel_cond = (
    float(np.max(err[conditioned] / mag[conditioned])) if np.any(conditioned) else 0.0
  )

  # Sufficient statistics for exact admission at a proposed band.
  abs_excess = float(np.max(err - rtol * mag, initial=0.0))
  rel_excess = (
    float(np.max((err[nonzero] - atol) / mag[nonzero], initial=0.0))
    if np.any(nonzero)
    else 0.0
  )
  zero_mag = ~nonzero
  zero_mag_worst = float(np.max(err[zero_mag], initial=0.0)) if np.any(zero_mag) else 0.0

  if np.any(violating):
    worst = int(np.argmax(np.where(violating, ratio, -np.inf)))
    worst_ratio = float(ratio[worst])
    worst_ref = float(mag[worst])
    worst_abs = float(err[worst])
    denom = atol + rtol * worst_ref
    worst_share = float(atol / denom) if denom > 0.0 else 1.0
  else:
    worst_ratio = 0.0
    worst_ref = 0.0
    worst_abs = 0.0
    worst_share = 0.0

  return FieldStats(
    pair=pair,
    field=field_name,
    n_elements=int(err.size),
    max_abs=float(np.max(err, initial=0.0)),
    max_abs_ref=float(mag[worst_abs_idx]) if mag.size else 0.0,
    max_ref_magnitude=float(np.max(mag, initial=0.0)),
    max_rel_unconditioned=rel_uncond,
    max_rel_conditioned=rel_cond,
    n_conditioned=int(np.sum(conditioned)),
    n_violating=int(np.sum(violating)),
    worst_violation_ratio=worst_ratio,
    worst_violation_ref=worst_ref,
    worst_violation_abs=worst_abs,
    worst_violation_atol_share=worst_share,
    abs_excess=max(abs_excess, 0.0),
    rel_excess=max(rel_excess, 0.0),
    zero_mag_worst=zero_mag_worst,
    n_zero_mag=int(np.sum(zero_mag)),
  )


def _measure_unit(
  module: Any,
  dump: Any,
  precision: str,
  checkpoint: str,
  fixture: str,
) -> list[FieldStats]:
  """Run one (pair, precision) through the test module's own ``_run``."""
  floats = tuple(module._FLOATS)  # noqa: SLF001 -- deliberate, see module docstring
  rtol = float(module._RTOL[precision])  # noqa: SLF001
  atol = float(module._ATOL[precision])  # noqa: SLF001
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

  if len(recorder.calls) != len(floats):
    msg = (
      f"recorded {len(recorder.calls)} comparisons for {len(floats)} declared float "
      f"fields in {module.__name__}; the test's plumbing changed and this "
      f"measurement would no longer describe it"
    )
    raise RuntimeError(msg)

  pair = f"{checkpoint}__{fixture}"
  return [
    _stats(pair, name, got, ref, rtol, atol)
    for name, (got, ref) in zip(floats, recorder.calls, strict=True)
  ]


def _admission(
  stats: list[FieldStats],
  binding: str,
  rtol_new: float,
  atol_new: float,
  atol_cur: float,
) -> tuple[int, int]:
  """Entries NOT admitted at the proposed band, and entries no rtol can admit.

  Exact, by the closed forms in the module docstring: which form applies is
  fixed by which term moved.
  """
  outside = 0
  blocked = 0
  for s in stats:
    if binding == "atol":
      if s.abs_excess > atol_new:
        outside += 1
    else:
      if s.zero_mag_worst > atol_cur:
        blocked += 1
        outside += 1
      elif s.rel_excess > rtol_new:
        outside += 1
  return outside, blocked


def _control(
  stats: list[FieldStats],
  rtol: float,
  atol: float,
  defect_relative: float,
) -> tuple[int, float]:
  """Inject a relative defect at each field's largest-magnitude element.

  Returns (entries that ADMIT the defect, worst band/defect ratio). The band
  must be strictly below the defect for it to be rejected at all, and
  ``_CONTROL_SCALE_MARGIN`` times below for the margin bar.
  """
  admitted = 0
  worst_ratio = 0.0
  for s in stats:
    magnitude = s.max_ref_magnitude
    if magnitude <= 0.0:
      continue
    defect = defect_relative * magnitude
    band = atol + rtol * magnitude
    if defect <= band:
      admitted += 1
    worst_ratio = max(worst_ratio, band / defect if defect > 0.0 else math.inf)
  return admitted, worst_ratio


def _derive(result: WaveResult) -> None:
  """Apply the pre-registered rule. Sets the proposed band and every bar."""
  result.f64_passes = all(s.n_violating == 0 for s in result.f64)
  violating = [s for s in result.f32 if s.n_violating]
  result.f32_violated = bool(violating)
  result.floor_abs = max((s.max_abs for s in result.f32), default=0.0)
  result.floor_abs_in_eps = result.floor_abs / _F32_EPS if _F32_EPS else 0.0
  result.floor_rel_unconditioned = max(
    (s.max_rel_unconditioned for s in result.f32),
    default=0.0,
  )
  result.floor_rel_conditioned = max(
    (s.max_rel_conditioned for s in result.f32),
    default=0.0,
  )
  if not (result.f64_passes and result.f32_violated):
    return

  worst = max(violating, key=lambda s: s.worst_violation_ratio)
  result.binding_ref = worst.worst_violation_ref
  result.binding_atol_share = worst.worst_violation_atol_share
  atol = result.current_atol_f32
  rtol = result.current_rtol_f32
  if atol >= rtol * worst.worst_violation_ref:
    result.binding_term = "atol"
    result.proposed_atol = _ceil_one_significant_figure(_HEADROOM * result.floor_abs)
    result.proposed_rtol = rtol
  else:
    result.binding_term = "rtol"
    if result.floor_rel_conditioned <= 0.0:
      msg = (
        f"rtol binds but no element has |ref| > {_CONDITION_THRESHOLD}: the "
        "conditioned relative floor is undefined and no rtol can be derived "
        "from this measurement"
      )
      raise RuntimeError(msg)
    result.proposed_rtol = _ceil_one_significant_figure(
      _HEADROOM * result.floor_rel_conditioned,
    )
    result.proposed_atol = atol

  outside, blocked = _admission(
    result.f32,
    result.binding_term,
    result.proposed_rtol,
    result.proposed_atol,
    result.current_atol_f32,
  )
  result.n_not_admitted = outside
  result.n_zero_mag_blocked = blocked
  result.all_admitted = outside == 0

  admitted, ratio = _control(
    result.f32,
    result.proposed_rtol,
    result.proposed_atol,
    _CONTROL_GRADED,
  )
  result.n_control_admitted = admitted
  result.control_rejected = admitted == 0
  result.control_scale_ratio = ratio
  result.below_control_scale = ratio * _CONTROL_SCALE_MARGIN <= 1.0
  _, reported = _control(
    result.f32,
    result.proposed_rtol,
    result.proposed_atol,
    _CONTROL_REPORTED,
  )
  result.control_1e3_margin = reported


def _grade(result: WaveResult) -> str:
  if not result.f64_passes:
    return "defect_not_tolerance"
  if not result.f32_violated:
    return "tolerance_was_fine"
  if not result.below_control_scale:
    return "noise_reaches_defect_scale"
  if result.all_admitted and result.control_rejected:
    return "pass"
  return "fail"


def _unit_key(script_sha: str, wave: str, pair: str, precision: str) -> str:
  raw = f"{script_sha}|{wave}|{pair}|{precision}"
  return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_unit(work_dir: Path, key: str) -> list[FieldStats] | None:
  body = work_dir / f"unit_{key}.json"
  stamp = work_dir / f"unit_{key}.done"
  if not (body.is_file() and stamp.is_file()):
    return None
  try:
    payload = json.loads(body.read_text())
  except (OSError, json.JSONDecodeError):
    return None
  try:
    return [FieldStats(**entry) for entry in payload]
  except TypeError:
    # The stats schema changed under a reused work-dir; recompute rather than
    # trust a body that no longer matches the dataclass.
    return None


def _save_unit(work_dir: Path, key: str, stats: list[FieldStats]) -> None:
  """Body first, stamp second, so a crash between them is recomputed."""
  (work_dir / f"unit_{key}.json").write_text(
    json.dumps([asdict(s) for s in stats], indent=2),
  )
  (work_dir / f"unit_{key}.done").write_text("done\n")


def _run_wave(wave: str, work_dir: Path, script_sha: str) -> WaveResult:
  module = importlib.import_module(f"port.test_{wave}")
  algo = importlib.import_module(f"port.reference.{wave}.algo")
  pairs = list(module._PAIRS)  # noqa: SLF001
  if not pairs:
    msg = f"{wave} oracle absent: no (checkpoint, fixture) pairs"
    raise OracleAbsent(msg)

  result = WaveResult(
    wave=wave,
    current_rtol_f32=float(module._RTOL["f32"]),  # noqa: SLF001
    current_atol_f32=float(module._ATOL["f32"]),  # noqa: SLF001
    current_rtol_f64=float(module._RTOL["f64"]),  # noqa: SLF001
    current_atol_f64=float(module._ATOL["f64"]),  # noqa: SLF001
    n_pairs=len(pairs),
    n_fields=len(module._FLOATS),  # noqa: SLF001
  )

  for precision in ("f64", "f32"):
    try:
      dump = algo.load(precision)
    except algo.OracleAbsentError as exc:  # pragma: no cover - environment
      raise OracleAbsent(str(exc)) from exc
    try:
      for checkpoint, fixture in pairs:
        pair = f"{checkpoint}__{fixture}"
        key = _unit_key(script_sha, wave, pair, precision)
        cached = _load_unit(work_dir, key)
        if cached is not None:
          result.reused_units.append(f"{precision}:{pair}")
          stats = cached
        else:
          stats = _measure_unit(module, dump, precision, checkpoint, fixture)
          _save_unit(work_dir, key, stats)
          result.computed_units.append(f"{precision}:{pair}")
        getattr(result, precision).extend(stats)
    finally:
      dump.close()

  _derive(result)
  return result


def _self_test() -> dict[str, Any]:
  """Verify the instrument on synthetic ground truth before trusting a number.

  Every check below has a hand-computed expected value, and two of them are
  NEGATIVE controls that must come back refusing to propose: a grader that can
  only return ``pass`` grades nothing. The sidecar makes this mandatory -- a
  measured floor from an unverified instrument is what produced #2432
  revision 1.
  """
  checks: dict[str, bool] = {}
  rtol, atol = 1e-4, 1e-7

  # Rounding: up to one significant figure, never to the next power of ten.
  checks["round_1p43e6_is_2e6"] = _ceil_one_significant_figure(1.43e-06) == 2e-06
  checks["round_exact_is_unchanged"] = _ceil_one_significant_figure(1e-06) == 1e-06
  checks["round_9p1e3_is_1e2"] = _ceil_one_significant_figure(9.1e-03) == 1e-02

  # Identical arrays deviate by nothing and have no excess.
  same = _stats("p", "f", np.ones(8), np.ones(8), rtol, atol)
  checks["identical_has_no_deviation"] = (
    same.max_abs == 0.0
    and same.n_violating == 0
    and same.abs_excess == 0.0
    and same.rel_excess == 0.0
  )

  # A known single deviation: ref 1.0, got 1.0+3e-4, band 1.001e-4.
  one = _stats("p", "f", np.array([1.0 + 3e-4]), np.array([1.0]), rtol, atol)
  checks["known_deviation_violates"] = one.n_violating == 1
  checks["known_abs_excess"] = math.isclose(one.abs_excess, 2e-4, rel_tol=1e-9)
  checks["known_rel_excess"] = math.isclose(one.rel_excess, 3e-4 - 1e-7, rel_tol=1e-9)

  # The closed forms admit and reject exactly at the boundary they claim.
  checks["rtol_form_admits_at_floor"] = _admission([one], "rtol", 3e-4, atol, atol) == (0, 0)
  checks["rtol_form_rejects_below"] = _admission([one], "rtol", 2e-4, atol, atol) == (1, 0)
  checks["atol_form_admits_at_floor"] = _admission([one], "atol", rtol, 2e-4, atol) == (0, 0)
  checks["atol_form_rejects_below"] = _admission([one], "atol", rtol, 1e-4, atol) == (1, 0)

  # A reference of exactly zero cannot be admitted by any rtol.
  zero = _stats("p", "f", np.array([1e-5]), np.array([0.0]), rtol, atol)
  checks["zero_mag_blocks_any_rtol"] = _admission([zero], "rtol", 1.0, atol, atol) == (1, 1)

  # Shared NaN is not an error; a one-sided NaN is an infinite deviation.
  nans = _stats("p", "f", np.array([np.nan, 1.0]), np.array([np.nan, np.nan]), rtol, atol)
  checks["shared_nan_is_not_an_error"] = nans.n_violating == 1
  checks["one_sided_nan_is_infinite"] = math.isinf(nans.max_abs)

  # The control margin bites exactly where pre-registered: band <= defect/10.
  # The bar is band <= defect/10 exactly, and the atol term counts toward it:
  # at rtol = defect/10 the band is already defect/10 + atol, so the margin
  # fails by that contribution alone. Pinned here so the strictness is
  # deliberate rather than discovered in a run.
  unit = _stats("p", "f", np.array([1.0]), np.array([1.0]), rtol, atol)
  _, tight = _control([unit], 5e-4, atol, _CONTROL_GRADED)
  _, boundary = _control([unit], 1e-3, atol, _CONTROL_GRADED)
  _, loose = _control([unit], 2e-3, atol, _CONTROL_GRADED)
  checks["margin_holds_at_5e4"] = tight * _CONTROL_SCALE_MARGIN <= 1.0
  checks["margin_fails_at_1e3_by_atol"] = boundary * _CONTROL_SCALE_MARGIN > 1.0
  checks["margin_fails_at_2e3"] = loose * _CONTROL_SCALE_MARGIN > 1.0
  admits, _ = _control([unit], 1.0, atol, _CONTROL_GRADED)
  checks["wide_band_admits_the_defect"] = admits == 1

  # NEGATIVE CONTROL 1: an f64 violation must refuse to propose anything.
  bad64 = WaveResult(
    wave="synthetic",
    current_rtol_f32=rtol,
    current_atol_f32=atol,
    current_rtol_f64=1e-8,
    current_atol_f64=1e-11,
    n_pairs=1,
    n_fields=1,
  )
  bad64.f64 = [_stats("p", "f", np.array([1.0 + 1e-3]), np.array([1.0]), 1e-8, 1e-11)]
  bad64.f32 = [one]
  _derive(bad64)
  checks["f64_violation_grades_defect"] = _grade(bad64) == "defect_not_tolerance"
  checks["f64_violation_proposes_nothing"] = bad64.proposed_atol == 0.0

  # NEGATIVE CONTROL 2: a clean f32 must refuse to widen a band that holds.
  clean = WaveResult(
    wave="synthetic",
    current_rtol_f32=rtol,
    current_atol_f32=atol,
    current_rtol_f64=1e-8,
    current_atol_f64=1e-11,
    n_pairs=1,
    n_fields=1,
  )
  clean.f64 = [_stats("p", "f", np.ones(4), np.ones(4), 1e-8, 1e-11)]
  clean.f32 = [same]
  _derive(clean)
  checks["clean_f32_grades_tolerance_was_fine"] = _grade(clean) == "tolerance_was_fine"
  checks["clean_f32_proposes_nothing"] = clean.proposed_rtol == 0.0

  failed = sorted(name for name, ok in checks.items() if not ok)
  return {
    "self_test_passed": not failed,
    "self_test_n_checks": len(checks),
    "self_test_failed": failed,
    "self_test_checks": checks,
  }


def _overall(verdicts: dict[str, str]) -> str:
  """Worst verdict across waves, residuals taking precedence over a bare fail."""
  for name in ("defect_not_tolerance", "noise_reaches_defect_scale", "tolerance_was_fine"):
    if name in verdicts.values():
      return name
  return "pass" if set(verdicts.values()) == {"pass"} else "fail"


def main() -> int:
  parser = argparse.ArgumentParser(description="Measure the LASEr f32 parity floor.")
  parser.add_argument("--out", type=Path, required=True, help="result JSON path")
  parser.add_argument("--work-dir", type=Path, required=True, help="resume directory")
  parser.add_argument(
    "--wave",
    action="append",
    choices=_WAVES,
    help="restrict to one wave (repeatable); default is both",
  )
  parser.add_argument(
    "--self-test-only",
    action="store_true",
    help="run the synthetic instrument checks and stop, touching no oracle",
  )
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

  root = _repo_root()
  _add_tests_to_path(root)
  script_sha = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
  args.work_dir.mkdir(parents=True, exist_ok=True)

  instrument = _self_test()
  if not instrument["self_test_passed"]:
    _LOG.error("instrument self-test FAILED: %s", instrument["self_test_failed"])

  waves = tuple(args.wave) if args.wave else _WAVES
  results: list[WaveResult] = []
  absent: list[str] = []
  if instrument["self_test_passed"] and not args.self_test_only:
    for wave in waves:
      try:
        results.append(_run_wave(wave, args.work_dir, script_sha))
      except OracleAbsent as exc:
        _LOG.error("%s: %s", wave, exc)
        absent.append(wave)

  verdicts = {r.wave: _grade(r) for r in results}
  if not instrument["self_test_passed"]:
    overall = "instrument_unverified"
  elif args.self_test_only:
    overall = "self_test_only"
  elif absent:
    overall = "oracle_absent"
  else:
    overall = _overall(verdicts)

  payload: dict[str, Any] = {
    "verdict": overall,
    **instrument,
    "per_wave_verdict": verdicts,
    "oracle_absent": absent,
    "waves": [asdict(r) for r in results],
    "f64_passes": bool(results) and all(r.f64_passes for r in results),
    "f32_violated": bool(results) and all(r.f32_violated for r in results),
    "all_admitted": bool(results) and all(r.all_admitted for r in results),
    "control_rejected": bool(results) and all(r.control_rejected for r in results),
    "below_control_scale": bool(results) and all(r.below_control_scale for r in results),
    "float32_eps": _F32_EPS,
    "condition_threshold": _CONDITION_THRESHOLD,
    "headroom": _HEADROOM,
    "control_graded_relative": _CONTROL_GRADED,
    "control_reported_relative": _CONTROL_REPORTED,
    "control_scale_margin": _CONTROL_SCALE_MARGIN,
    "n_reused": sum(len(r.reused_units) for r in results),
    "n_computed": sum(len(r.computed_units) for r in results),
    "script_sha256": script_sha,
    "work_dir": str(args.work_dir),
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
  _LOG.info("verdict=%s per_wave=%s -> %s", overall, verdicts, args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
