"""Attribute the ``laser_encoder`` float32 disagreement: port divergence, or f32 itself?

WHY THIS EXISTS. Graded run 91dfbf20 measured the laser_encoder f32 floor and
returned ``noise_reaches_defect_scale``: a band wide enough to admit aminx's f32
encoder output sits only ~4.07x below an injected 1e-2 relative defect, against
a pre-registered 10x bar. Its pre-registered decision is "Escalate. Do NOT
widen. float32 noise in this wave is within 10x of a defect worth catching,
which is a numerical-stability question about the port, not a test-tolerance
question."

That decision names the port. This run checks whether naming the port is
actually warranted, because there are two very different worlds behind the same
number:

  A. aminx-f32 disagrees with upstream-f32 by about as much as upstream-f32
     disagrees with upstream-f64. Then the disagreement is the size of float32
     rounding itself, the tier is asking f32 to be more reproducible than f32
     is, and there is no aminx stability defect to chase.

  B. aminx-f32 disagrees with upstream-f32 by far MORE than upstream's own f32
     rounding error. Then the two implementations genuinely accumulate
     differently and the escalation points somewhere real in aminx.

THE SHAPE OF THIS INSTRUMENT IS NOT NEW. Run 18b97f4f did the same attribution
for laser_score and found aminx-f32 and upstream-f32 agree with EACH OTHER far
more tightly than either agrees with an f64 truth -- world A -- which is why
laser_score's tier-3 became a design question rather than a bug hunt. Whether
laser_encoder is in the same world is NOT assumed here; it is measured.

THE TWO QUANTITIES, per (pair, field):

  D_port     = max |aminx_f32 - upstream_f32|
  D_upstream = max |upstream_f32 - upstream_f64|
  R          = D_port / D_upstream

D_port is NOT recomputed. It is read from the persisted units of graded run
91dfbf20, whose per-unit cache keys are DERIVED here rather than guessed: the
key is sha256(script_sha | wave | pair | precision)[:16] and every input is
recorded in that run's result JSON, so the f32 units can be addressed exactly
and the f64 units cannot be mistaken for them. If a derived key is missing the
run stops rather than silently averaging over whatever files happen to be there.

D_upstream needs NO model and NO aminx code: it is two sealed reference arrays
subtracted. That is the whole reason this measurement is cheap.

WHY THE THRESHOLD IS 2 AND NOT 1. Two independent float32 implementations that
are EACH within e of a double-precision truth may differ from each other by up
to 2e. So R <= 2 is the region where the port's disagreement is fully explained
by float32 rounding, and demanding R <= 1 would fail implementations that are
doing nothing wrong. Fixed before the run.

ZERO-BASELINE FIELDS. Where upstream's f32 and f64 arrays are bitwise equal,
D_upstream is 0 and R is undefined -- that field carries no float32 rounding at
this pair and cannot attribute anything. Those entries are COUNTED and excluded
from the ratio rather than being given an infinite R, and the count is reported
so an attribution resting on a handful of comparable entries is visible as such.

This run proposes no band and amends no file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

from laser_f32_tolerance_floor import (  # noqa: E402
  OracleAbsent,
  _add_tests_to_path,
  _repo_root,
)

_LOG = logging.getLogger("laser_encoder_f32_attribution")

_WAVE = "laser_encoder"

#: R at or below this is fully explained by float32 rounding. See the module
#: docstring: two f32 paths each within e of truth may differ by up to 2e.
_NOISE_RATIO = 2.0

#: R above this says the port diverges by an order of magnitude more than
#: upstream's own f32 error, which is a real stability lead.
_DIVERGENCE_RATIO = 10.0


def _unit_key(script_sha: str, wave: str, pair: str, precision: str) -> str:
  """Reproduce laser_f32_tolerance_floor._unit_key EXACTLY.

  Imported rather than re-implemented would be better, but the shared module's
  copy is the authority and this one is checked against it in the self-test, so
  a drift in either is a loud failure instead of a silent mis-addressing.
  """
  raw = f"{script_sha}|{wave}|{pair}|{precision}"
  return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _upstream_rounding(ref32: np.ndarray, ref64: np.ndarray) -> float:
  """max |upstream_f32 - upstream_f64|, with shared NaNs treated as equal."""
  a = np.asarray(ref32, dtype=np.float64).ravel()
  b = np.asarray(ref64, dtype=np.float64).ravel()
  if a.shape != b.shape:
    msg = f"upstream f32/f64 shapes differ: {a.shape} vs {b.shape}"
    raise RuntimeError(msg)
  if a.size == 0:
    return 0.0
  both_nan = np.isnan(a) & np.isnan(b)
  nan_mismatch = np.isnan(a) != np.isnan(b)
  err = np.abs(a - b)
  err = np.where(both_nan, 0.0, err)
  err = np.where(nan_mismatch, np.inf, err)
  return float(np.max(err))


def _grade(payload: dict[str, Any]) -> str:
  if not payload["self_test_passed"]:
    return "instrument_unverified"
  if payload["units_absent"]:
    return "units_absent"
  if payload["oracle_absent"]:
    return "oracle_absent"
  if payload["n_comparable"] == 0:
    return "no_comparable_entries"
  worst = payload["worst_ratio"]
  if worst <= _NOISE_RATIO:
    return "within_f32_noise"
  if worst <= _DIVERGENCE_RATIO:
    return "larger_than_upstream_rounding"
  return "port_exceeds_upstream_f32"


def _self_test() -> dict[str, Any]:
  """Synthetic checks, run before any dump or unit is read."""
  failed: list[str] = []
  checks = 0

  # 1. The key derivation matches the shared module's, so the f32 units this
  #    run loads are the ones graded run 91dfbf20 actually wrote.
  checks += 1
  shared = importlib.import_module("laser_f32_tolerance_floor")
  if _unit_key("abc", "w", "p", "f32") != shared._unit_key("abc", "w", "p", "f32"):  # noqa: SLF001
    failed.append("unit_key_drifted_from_shared_module")

  # 2. f32 and f64 keys differ, so the two precisions cannot be confused.
  checks += 1
  if _unit_key("abc", "w", "p", "f32") == _unit_key("abc", "w", "p", "f64"):
    failed.append("unit_key_ignores_precision")

  # 3. Hand-computed rounding baseline.
  checks += 1
  if _upstream_rounding(np.array([1.0, 2.0]), np.array([1.0, 2.25])) != 0.25:
    failed.append("upstream_rounding_wrong")

  # 4. Shared NaNs are not an error; a one-sided NaN is infinite.
  checks += 1
  if _upstream_rounding(np.array([np.nan]), np.array([np.nan])) != 0.0:
    failed.append("shared_nan_not_zero")
  checks += 1
  if not np.isinf(_upstream_rounding(np.array([np.nan]), np.array([1.0]))):
    failed.append("one_sided_nan_not_infinite")

  # 5. An empty field yields a zero baseline rather than raising -- this wave
  #    has one: lig_vectors is shape (N, 0, 3) for every pair.
  checks += 1
  if _upstream_rounding(np.zeros((3, 0)), np.zeros((3, 0))) != 0.0:
    failed.append("empty_field_not_zero")

  # 6-9. GRADER NEGATIVE CONTROLS. The grader must be able to return each of
  #      its three measured verdicts, and must refuse when nothing compares.
  base = {
    "self_test_passed": True,
    "units_absent": [],
    "oracle_absent": [],
    "n_comparable": 5,
  }
  checks += 1
  if _grade({**base, "worst_ratio": 1.5}) != "within_f32_noise":
    failed.append("grader_misses_within_f32_noise")
  checks += 1
  if _grade({**base, "worst_ratio": 5.0}) != "larger_than_upstream_rounding":
    failed.append("grader_misses_larger_than_upstream_rounding")
  checks += 1
  if _grade({**base, "worst_ratio": 50.0}) != "port_exceeds_upstream_f32":
    failed.append("grader_misses_port_exceeds")
  checks += 1
  if _grade({**base, "n_comparable": 0, "worst_ratio": 0.0}) != "no_comparable_entries":
    failed.append("grader_does_not_refuse_without_comparable_entries")

  # 10. Boundaries are inclusive exactly where the docstring says.
  checks += 1
  if _grade({**base, "worst_ratio": _NOISE_RATIO}) != "within_f32_noise":
    failed.append("noise_boundary_exclusive")
  checks += 1
  if _grade({**base, "worst_ratio": _DIVERGENCE_RATIO}) != "larger_than_upstream_rounding":
    failed.append("divergence_boundary_exclusive")

  # 11. A missing unit outranks a measured ratio: the run must not report an
  #     attribution computed over a subset it could not fully address.
  checks += 1
  if _grade({**base, "units_absent": ["x"], "worst_ratio": 1.0}) != "units_absent":
    failed.append("units_absent_does_not_outrank")

  return {
    "self_test_passed": not failed,
    "self_test_n_checks": checks,
    "self_test_failed": failed,
  }


def _emit(out: Path, payload: dict[str, Any]) -> None:
  out.parent.mkdir(parents=True, exist_ok=True)
  text = json.dumps(payload, indent=2, sort_keys=True)
  out.write_text(text)
  results = os.environ.get("BTH_RESULTS_PATH")
  if results:
    Path(results).write_text(text)
    _LOG.info("emitted result to BTH_RESULTS_PATH=%s", results)


def main() -> int:
  parser = argparse.ArgumentParser(description="Attribute laser_encoder f32 deviation.")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument(
    "--floor-result",
    type=Path,
    required=True,
    help="result.json of the graded floor run whose f32 units supply D_port",
  )
  parser.add_argument(
    "--floor-work-dir",
    type=Path,
    required=True,
    help="work dir of that run, holding its unit_<key>.json files",
  )
  parser.add_argument("--self-test-only", action="store_true")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  root = _repo_root()
  _add_tests_to_path(root)

  self_test = _self_test()
  base: dict[str, Any] = {
    **self_test,
    "wave": _WAVE,
    "script_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
    "noise_ratio_threshold": _NOISE_RATIO,
    "divergence_ratio_threshold": _DIVERGENCE_RATIO,
    "units_absent": [],
    "oracle_absent": [],
    "n_comparable": 0,
    "worst_ratio": 0.0,
  }
  if not self_test["self_test_passed"]:
    _emit(args.out, {**base, "verdict": "instrument_unverified"})
    _LOG.error("instrument unverified: %s", self_test["self_test_failed"])
    return 1
  _LOG.info("self-test passed (%d checks)", self_test["self_test_n_checks"])
  if args.self_test_only:
    _emit(args.out, {**base, "verdict": "self_test_only"})
    return 0

  floor = json.loads(args.floor_result.read_text())
  floor_sha = floor["script_sha256"]
  base["floor_run_script_sha256"] = floor_sha
  base["floor_run_verdict"] = floor["verdict"]

  module = importlib.import_module(f"port.test_{_WAVE}")
  algo = importlib.import_module(f"port.reference.{_WAVE}.algo")
  pairs = [f"{c}__{f}" for c, f in module._PAIRS]  # noqa: SLF001
  floats = tuple(module._FLOATS)  # noqa: SLF001

  # D_port, addressed by DERIVED key -- never by scanning the directory.
  d_port: dict[tuple[str, str], float] = {}
  missing: list[str] = []
  for pair in pairs:
    key = _unit_key(floor_sha, _WAVE, pair, "f32")
    body = args.floor_work_dir / f"unit_{key}.json"
    stamp = args.floor_work_dir / f"unit_{key}.done"
    if not (body.is_file() and stamp.is_file()):
      missing.append(f"f32:{pair}")
      continue
    for stat in json.loads(body.read_text()):
      d_port[(pair, stat["field"])] = float(stat["max_abs"])
  if missing:
    _emit(args.out, {**base, "verdict": "units_absent", "units_absent": missing[:20]})
    _LOG.error("%d f32 units unaddressable; refusing to attribute", len(missing))
    return 1

  # D_upstream, from the sealed dumps alone.
  try:
    dump32 = algo.load("f32")
    dump64 = algo.load("f64")
  except algo.OracleAbsentError as exc:
    _emit(args.out, {**base, "verdict": "oracle_absent", "oracle_absent": [str(exc)]})
    return 1
  except OracleAbsent as exc:  # pragma: no cover - environment
    _emit(args.out, {**base, "verdict": "oracle_absent", "oracle_absent": [str(exc)]})
    return 1

  rows: list[dict[str, Any]] = []
  n_zero_baseline = 0
  try:
    for pair in pairs:
      for name in floats:
        key = f"{pair}__{name}"
        if key not in dump32.files or key not in dump64.files:
          msg = f"dump missing {key}"
          raise RuntimeError(msg)
        baseline = _upstream_rounding(dump32[key], dump64[key])
        port = d_port[(pair, name)]
        if baseline <= 0.0:
          n_zero_baseline += 1
          continue
        rows.append(
          {
            "pair": pair,
            "field": name,
            "d_port": port,
            "d_upstream": baseline,
            "ratio": port / baseline,
          },
        )
  finally:
    dump32.close()
    dump64.close()

  by_field: dict[str, dict[str, float]] = {}
  for row in rows:
    slot = by_field.setdefault(
      row["field"], {"worst_ratio": 0.0, "d_port": 0.0, "d_upstream": 0.0, "n": 0},
    )
    slot["n"] += 1
    slot["worst_ratio"] = max(slot["worst_ratio"], row["ratio"])
    slot["d_port"] = max(slot["d_port"], row["d_port"])
    slot["d_upstream"] = max(slot["d_upstream"], row["d_upstream"])

  worst = max((r["ratio"] for r in rows), default=0.0)
  worst_row = max(rows, key=lambda r: r["ratio"]) if rows else {}
  payload = {
    **base,
    "n_pairs": len(pairs),
    "n_fields": len(floats),
    "n_comparable": len(rows),
    "n_zero_baseline": n_zero_baseline,
    "worst_ratio": worst,
    "worst_entry": worst_row,
    "median_ratio": float(np.median([r["ratio"] for r in rows])) if rows else 0.0,
    "n_above_noise": sum(1 for r in rows if r["ratio"] > _NOISE_RATIO),
    "n_above_divergence": sum(1 for r in rows if r["ratio"] > _DIVERGENCE_RATIO),
    "per_field": by_field,
  }
  payload["verdict"] = _grade(payload)
  _emit(args.out, payload)
  _LOG.info(
    "verdict=%s worst_ratio=%.4g over %d comparable entries (%d zero-baseline)",
    payload["verdict"],
    worst,
    len(rows),
    n_zero_baseline,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
