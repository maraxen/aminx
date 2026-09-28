"""Synthetic checks for within-session stratified bootstrap CIs (T8 step 1)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.browser_validation import bench_stats
from scripts.browser_validation.layer_c_bench import (
  cite_bucket,
  claim_ratio_pairs,
  make_probe_record,
)

_SEED = 260928
_MU = float(np.log(10.0))
_SIGMA = 0.25


def _reference_p50_ci(
  repeats: list[np.ndarray],
  *,
  seed: int,
  n_resamples: int = 1000,
) -> tuple[float, float, float]:
  """Independent copy of the stratified linear-interpolation bootstrap."""
  rng = np.random.default_rng(seed)
  parts = []
  for repeat in repeats:
    size = repeat.shape[0]
    index = rng.integers(0, size, size=(n_resamples, size))
    parts.append(repeat[index])
  stats = np.percentile(np.concatenate(parts, axis=1), 50.0, axis=1, method="linear")
  pooled = np.concatenate(repeats)
  point = float(np.percentile(pooled, 50.0, method="linear"))
  low = float(np.percentile(stats, 2.5, method="linear"))
  high = float(np.percentile(stats, 97.5, method="linear"))
  return point, low, high


def _lognormal_repeats(
  rng: np.random.Generator, per_repeat: int, n_repeats: int, shift: float = 0.0
):
  return [np.exp(rng.normal(_MU + shift, _SIGMA, size=per_repeat)) for _ in range(n_repeats)]


def test_p50_ci_coverage_known_quantile() -> None:
  """95% CI covers the lognormal p50 in >= 0.9 of 200 replicates at n = 90."""
  fingerprint_rng = np.random.default_rng(7)
  fingerprint = [
    np.exp(fingerprint_rng.normal(_MU, _SIGMA, size=30)),
    np.exp(fingerprint_rng.normal(_MU + 1.5, 0.05, size=30)),
    np.exp(fingerprint_rng.normal(_MU - 1.2, 0.05, size=30)),
  ]
  got = bench_stats.p50_ci(fingerprint, seed=12345)
  ref_point, ref_low, ref_high = _reference_p50_ci(fingerprint, seed=12345)
  assert got.scope == "within-session"
  assert got.point == pytest.approx(ref_point)
  assert got.low == pytest.approx(ref_low)
  assert got.high == pytest.approx(ref_high)

  covered = 0
  true_p50 = float(np.exp(_MU))
  for child in np.random.SeedSequence(_SEED).spawn(200):
    rng = np.random.default_rng(child)
    repeats = _lognormal_repeats(rng, 30, 3)
    interval = bench_stats.p50_ci(repeats, seed=int(rng.integers(0, 2**31 - 1)))
    assert interval.scope == "within-session"
    if interval.low <= true_p50 <= interval.high:
      covered += 1
  assert covered / 200 >= 0.9


def test_identical_distribution_ratio_contains_one() -> None:
  """Identical lognormals: the p50 ratio CI contains 1 in >= 0.9 of 200 replicates."""
  contained = 0
  for index, child in enumerate(np.random.SeedSequence(_SEED + 1).spawn(200)):
    rng = np.random.default_rng(child)
    left = _lognormal_repeats(rng, 30, 3)
    right = _lognormal_repeats(rng, 30, 3)
    interval = bench_stats.ratio_p50_ci(left, right, seed=10_000 + index)
    assert interval.scope == "within-session"
    if bench_stats.ci_contains(interval, 1.0):
      contained += 1
  assert contained / 200 >= 0.9


def test_five_percent_shift_detected() -> None:
  """A +5% scale shift at n = 90 x 3 is excluded from 1 in >= 0.9 of 200 replicates.

  The shifted arm is the base draw multiplied by 1.05, which is
  lognormal(mu + log 1.05, sigma = 0.25). Shared within-repeat indices keep that
  exact scale in every resample.
  """
  detected = 0
  for index, child in enumerate(np.random.SeedSequence(_SEED + 2).spawn(200)):
    rng = np.random.default_rng(child)
    base = _lognormal_repeats(rng, 90, 3)
    shifted = [sample * 1.05 for sample in base]
    interval = bench_stats.ratio_p50_ci(shifted, base, seed=20_000 + index)
    assert interval.scope == "within-session"
    if bench_stats.ci_excludes(interval, 1.0):
      detected += 1
  assert detected / 200 >= 0.9


def _record(
  *,
  probe_id: str,
  route: str,
  threads: str,
  seconds: float | None,
  git_sha: str,
  bucket: int = 256,
) -> object:
  return make_probe_record(
    probe_id=probe_id,
    path="p04",
    bucket=bucket,
    route=route,
    session_id="sess-unit",
    threads=threads,
    total_step_seconds=seconds,
    git_sha=git_sha,
  )


def test_dirty_git_sha_refused() -> None:
  from xtrax.profiling.claims import ClaimValidityError

  record = _record(
    probe_id="p04_L256_t1",
    route="ort_wasm_t1",
    threads="1",
    seconds=0.01,
    git_sha="abc123-dirty",
  )
  with pytest.raises(ClaimValidityError):
    cite_bucket([record], 256)


def test_missing_total_step_seconds_refused() -> None:
  from xtrax.profiling.claims import ClaimValidityError

  record = _record(
    probe_id="p04_L256_t1",
    route="ort_wasm_t1",
    threads="1",
    seconds=None,
    git_sha="abc123",
  )
  with pytest.raises(ClaimValidityError):
    cite_bucket([record], 256)


def test_unequal_threads_pair_yields_no_sentence() -> None:
  native = _record(
    probe_id="p04_L256_native",
    route="native_jax_cpu",
    threads="1",
    seconds=0.01,
    git_sha="abc123",
  )
  wasm_tn = _record(
    probe_id="p04_L256_tn",
    route="ort_wasm_tN",
    threads="4",
    seconds=0.02,
    git_sha="abc123",
  )
  assert claim_ratio_pairs([native, wasm_tn], native_pinned=True) == []
