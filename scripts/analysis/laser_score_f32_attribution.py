"""Attribute laser_score's float32 disagreement: is aminx's f32 worse than upstream's?

WHY THIS EXISTS (debt #2445). Graded run e47bfaab returned
``noise_reaches_defect_scale`` for laser_score: aminx-f32 and upstream-f32
disagree by up to 5.4e-4 (chi_logits), too close to a 1e-2 defect for any band
to separate them, while f64 agrees to 7.8e-13. Its pre-registered decision was
"escalate, do not widen", and the obvious next step -- hunt for a numerical
instability in aminx's f32 path -- presumes aminx is the worse of the two.

That presumption is testable. The tier-3 test compares TWO float32
implementations, and both carry float32 error. This script measures each one
against the truth separately:

    truth      aminx in float64, at the f32 dump's own decoding order
    e_aminx    max |aminx_f32    - truth|   per field, per pair
    e_upstream max |upstream_f32 - truth|   per field, per pair   (from the dump)

and reports, per field, ``ratio = E_aminx / E_upstream`` where ``E`` is the max
over pairs, each pair's error first divided by that pair's ``max|truth|`` so
large-magnitude fixtures do not dominate.

WHY aminx-f64 IS A VALID TRUTH. The f32 and f64 dumps use the same decoding
order for every pair (63/63, checked), so aminx-f64 at that order can be
compared directly with upstream-f64 from the f64 dump. ``truth_valid`` requires
that to agree within 1e-10 of scale on every field and pair -- re-measured
here, not assumed from the tier-2 result.

WHAT EACH ANSWER MEANS
* ratio <= 2 on every rateable field: aminx's f32 is no worse than upstream's.
  Then a stability hunt in aminx has no target, and laser_score's tier-3 margin
  failure is a property of comparing two f32 implementations -- a test-design
  question for the user, not a port defect.
* ratio > 2 on any field: aminx's f32 path is genuinely worse there, and the
  earliest such field in depth is where a stability fix starts.

RESUMABLE. One unit per (pair), body then stamp, keyed by script sha + pair;
a crash loses at most the pair in flight. 63 units, 2 forward passes each.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_LOG = logging.getLogger("laser_score_f32_attribution")

#: ratio above which aminx's f32 error is called worse than upstream's.
_WORSE_RATIO = 2.0
#: truth proxy must match upstream-f64 within this fraction of field scale.
_TRUTH_BAR = 1e-10
#: below this (as a fraction of scale) upstream's f32 error counts as zero, so
#: no ratio can be formed for that field.
_ZERO_ERR = 1e-12


def _repo_root() -> Path:
  here = Path(__file__).resolve()
  for parent in here.parents:
    if (parent / "pyproject.toml").is_file():
      return parent
  msg = f"no pyproject.toml above {here}"
  raise RuntimeError(msg)


def _add_tests_to_path(root: Path) -> None:
  """Append (never prepend) ``tests`` so its ``io``/``types`` dirs shadow nothing."""
  tests = str(root / "tests")
  if tests not in sys.path:
    sys.path.append(tests)


def _scaled_err(got: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
  """(max |got - ref| / max |ref|, max |ref|) over elements finite in both."""
  got = np.asarray(got, dtype=np.float64)
  ref = np.asarray(ref, dtype=np.float64)
  ok = np.isfinite(got) & np.isfinite(ref)
  if not ok.any():
    return 0.0, 0.0
  scale = float(np.max(np.abs(ref[ok])))
  if scale == 0.0:
    return float(np.max(np.abs(got[ok] - ref[ok]))), 0.0
  return float(np.max(np.abs(got[ok] - ref[ok]))) / scale, scale


@dataclass
class PairErrors:
  pair: str
  e_aminx: dict[str, float]
  e_upstream: dict[str, float]
  e_between: dict[str, float]
  truth_vs_up64: dict[str, float]


@dataclass
class Result:
  verdict: str
  self_test_passed: bool
  self_test_failed: list[str]
  truth_valid: bool
  truth_worst: float
  n_pairs: int
  oracle_present: bool
  any_rateable: bool
  aminx_worse: bool
  ratios: dict[str, float | None] = field(default_factory=dict)
  e_aminx_max: dict[str, float] = field(default_factory=dict)
  e_upstream_max: dict[str, float] = field(default_factory=dict)
  e_between_max: dict[str, float] = field(default_factory=dict)
  worse_fields: list[str] = field(default_factory=list)
  unrateable_fields: list[str] = field(default_factory=list)
  n_reused: int = 0
  n_computed: int = 0
  script_sha256: str = ""


def _aggregate(pairs: list[PairErrors], fields: tuple[str, ...]) -> dict[str, Any]:
  e_a = {f: max((p.e_aminx[f] for p in pairs), default=0.0) for f in fields}
  e_u = {f: max((p.e_upstream[f] for p in pairs), default=0.0) for f in fields}
  e_b = {f: max((p.e_between[f] for p in pairs), default=0.0) for f in fields}
  truth_worst = max((v for p in pairs for v in p.truth_vs_up64.values()), default=0.0)
  ratios: dict[str, float | None] = {}
  unrateable: list[str] = []
  for f in fields:
    if e_u[f] <= _ZERO_ERR:
      ratios[f] = None
      if e_a[f] > _ZERO_ERR:
        unrateable.append(f)
    else:
      ratios[f] = e_a[f] / e_u[f]
  worse = [f for f, r in ratios.items() if r is not None and r > _WORSE_RATIO]
  return {
    "e_aminx_max": e_a,
    "e_upstream_max": e_u,
    "e_between_max": e_b,
    "truth_worst": truth_worst,
    "ratios": ratios,
    "worse_fields": worse,
    "unrateable_fields": unrateable,
  }


def _verdict(self_ok: bool, agg: dict[str, Any], n_pairs: int) -> str:
  if not self_ok:
    return "instrument_unverified"
  if n_pairs == 0:
    return "oracle_absent"
  if agg["truth_worst"] > _TRUTH_BAR:
    return "truth_invalid"
  rateable = [r for r in agg["ratios"].values() if r is not None]
  if not rateable:
    return "inconclusive"
  if agg["worse_fields"]:
    return "aminx_worse"
  return "aminx_no_worse_than_upstream"


def _self_test() -> tuple[bool, list[str]]:
  """Synthetic ground truth for every branch, including two that must NOT pass."""
  failed: list[str] = []
  rng = np.random.default_rng(0)
  truth = rng.normal(size=(20, 5)) * 10.0
  fields = ("x",)

  def mk(a: float, u: float, t: float = 0.0) -> PairErrors:
    return PairErrors(
      pair="p",
      e_aminx={"x": _scaled_err(truth + a * rng.normal(size=truth.shape), truth)[0]},
      e_upstream={"x": _scaled_err(truth + u * rng.normal(size=truth.shape), truth)[0]},
      e_between={"x": 0.0},
      truth_vs_up64={"x": t},
    )

  checks = {
    "equal_noise_is_parity": (_verdict(True, _aggregate([mk(1e-4, 1e-4)], fields), 1), "aminx_no_worse_than_upstream"),
    "aminx_100x_worse_is_flagged": (_verdict(True, _aggregate([mk(1e-2, 1e-4)], fields), 1), "aminx_worse"),
    "aminx_better_is_parity": (_verdict(True, _aggregate([mk(1e-5, 1e-3)], fields), 1), "aminx_no_worse_than_upstream"),
    "bad_truth_refuses": (_verdict(True, _aggregate([mk(1e-4, 1e-4, t=1e-6)], fields), 1), "truth_invalid"),
    "zero_upstream_error_is_unrateable": (_verdict(True, _aggregate([mk(1e-4, 0.0)], fields), 1), "inconclusive"),
    "instrument_failure_dominates": (_verdict(False, _aggregate([mk(1e-4, 1e-4)], fields), 1), "instrument_unverified"),
    "no_pairs_is_oracle_absent": (_verdict(True, _aggregate([], fields), 0), "oracle_absent"),
  }
  for name, (got, want) in checks.items():
    if got != want:
      failed.append(f"{name}: got {got}, want {want}")
  # NaN handling: chi_log_prob carries NaN where a chi angle is absent.
  a = np.array([1.0, np.nan, 3.0])
  if _scaled_err(a, np.array([1.0, np.nan, 3.0]))[0] != 0.0:
    failed.append("nan_masked: NaN positions must be ignored, not propagated")
  return not failed, failed


def _unit_key(script_sha: str, pair: str) -> str:
  return hashlib.sha256(f"{script_sha}|{pair}".encode()).hexdigest()[:16]


def _measure_pair(module: Any, d32: Any, d64: Any, checkpoint: str, fixture: str) -> PairErrors:
  import jax  # noqa: PLC0415 -- after sys.path is set
  from port.a1_compare import x64_context  # noqa: PLC0415

  prefix = f"{checkpoint}__{fixture}__"
  order = np.asarray(d32[prefix + "decoding_order"])
  if not np.array_equal(order, np.asarray(d64[prefix + "decoding_order"])):
    msg = f"{prefix}: f32 and f64 dumps disagree on decoding_order; truth would be invalid"
    raise RuntimeError(msg)
  fields = tuple(module._FLOATS)  # noqa: SLF001
  with x64_context("f64"), jax.default_matmul_precision("highest"):
    truth = module._predict(checkpoint, fixture, np.dtype(np.float64), order)  # noqa: SLF001
    truth = {f: np.asarray(truth[f], dtype=np.float64) for f in fields}
  with x64_context("f32"), jax.default_matmul_precision("highest"):
    mine = module._predict(checkpoint, fixture, np.dtype(np.float32), order)  # noqa: SLF001
    mine = {f: np.asarray(mine[f]) for f in fields}
  return PairErrors(
    pair=f"{checkpoint}__{fixture}",
    e_aminx={f: _scaled_err(mine[f], truth[f])[0] for f in fields},
    e_upstream={f: _scaled_err(np.asarray(d32[prefix + f]), truth[f])[0] for f in fields},
    e_between={f: _scaled_err(mine[f], np.asarray(d32[prefix + f]))[0] for f in fields},
    truth_vs_up64={f: _scaled_err(truth[f], np.asarray(d64[prefix + f]))[0] for f in fields},
  )


def main() -> int:
  parser = argparse.ArgumentParser(description="Attribute laser_score f32 error: aminx vs upstream.")
  parser.add_argument("--work-dir", type=Path, required=True, help="per-pair resume directory")
  parser.add_argument("--out", type=Path, required=True, help="result JSON path")
  parser.add_argument("--self-test-only", action="store_true", help="synthetic checks only")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

  root = _repo_root()
  _add_tests_to_path(root)
  script_sha = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
  args.work_dir.mkdir(parents=True, exist_ok=True)

  self_ok, self_failed = _self_test()
  if not self_ok:
    _LOG.error("instrument self-test FAILED: %s", self_failed)

  pairs: list[PairErrors] = []
  fields: tuple[str, ...] = ()
  reused = computed = 0
  if self_ok and not args.self_test_only:
    module = importlib.import_module("port.test_laser_score")
    algo = importlib.import_module("port.reference.laser_score.algo")
    fields = tuple(module._FLOATS)  # noqa: SLF001
    d32, d64 = algo.load("f32"), algo.load("f64")
    try:
      for checkpoint, fixture in module._PAIRS:  # noqa: SLF001
        pair = f"{checkpoint}__{fixture}"
        key = _unit_key(script_sha, pair)
        body, stamp = args.work_dir / f"unit_{key}.json", args.work_dir / f"unit_{key}.done"
        if body.is_file() and stamp.is_file():
          pairs.append(PairErrors(**json.loads(body.read_text())))
          reused += 1
          continue
        errors = _measure_pair(module, d32, d64, checkpoint, fixture)
        body.write_text(json.dumps(asdict(errors)))
        stamp.write_text("done\n")
        pairs.append(errors)
        computed += 1
        _LOG.info("%s chi_logits e_aminx=%.3e e_upstream=%.3e", pair,
                  errors.e_aminx.get("chi_logits", 0.0), errors.e_upstream.get("chi_logits", 0.0))
    finally:
      d32.close()
      d64.close()

  agg = _aggregate(pairs, fields)
  result = Result(
    verdict="self_test_only" if (self_ok and args.self_test_only) else _verdict(self_ok, agg, len(pairs)),
    self_test_passed=self_ok,
    self_test_failed=self_failed,
    truth_valid=agg["truth_worst"] <= _TRUTH_BAR,
    truth_worst=agg["truth_worst"],
    n_pairs=len(pairs),
    oracle_present=bool(pairs),
    any_rateable=any(r is not None for r in agg["ratios"].values()),
    aminx_worse=bool(agg["worse_fields"]),
    ratios=agg["ratios"],
    e_aminx_max=agg["e_aminx_max"],
    e_upstream_max=agg["e_upstream_max"],
    e_between_max=agg["e_between_max"],
    worse_fields=agg["worse_fields"],
    unrateable_fields=agg["unrateable_fields"],
    n_reused=reused,
    n_computed=computed,
    script_sha256=script_sha,
  )
  payload = json.dumps(asdict(result), indent=2, sort_keys=True)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(payload)
  bth_results = os.environ.get("BTH_RESULTS_PATH")
  if bth_results:
    Path(bth_results).parent.mkdir(parents=True, exist_ok=True)
    Path(bth_results).write_text(payload)
  _LOG.info("verdict=%s ratios=%s -> %s", result.verdict, result.ratios, args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
