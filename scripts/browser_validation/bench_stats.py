"""Within-session p50/p90 summaries and stratified bootstrap CIs (T8 step 1).

Bootstrap resamples are stratified by repeat: each repeat is drawn with replacement
at its own size, then concatenated. Every interval is within-session by construction
(`WithinSessionCI.scope`). The percentile interpolation lives in one place
(`PERCENTILE_INTERPOLATION`) so a swap is a single edit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Single interpolation site. The coverage test locks this to "linear".
PERCENTILE_INTERPOLATION = "linear"

N_RESAMPLES = 1000
CI_LEVEL = 0.95
WITHIN_SESSION = "within-session"


@dataclass(frozen=True)
class WithinSessionCI:
  """A percentile bootstrap interval computed inside one session."""

  point: float
  low: float
  high: float
  level: float
  n_resamples: int
  seed: int
  scope: str


def percentile(values: np.ndarray, q: float) -> float:
  """Scalar percentile of `values` using `PERCENTILE_INTERPOLATION`."""
  return float(
    np.percentile(np.asarray(values, dtype=np.float64), q, method=PERCENTILE_INTERPOLATION)
  )


def _percentile_axis(values: np.ndarray, q: float, axis: int) -> np.ndarray:
  return np.percentile(values, q, axis=axis, method=PERCENTILE_INTERPOLATION)


def as_repeats(samples: list[np.ndarray] | tuple[np.ndarray, ...]) -> list[np.ndarray]:
  """Coerce one array per repeat. Empty input or an empty stratum is an error."""
  repeats: list[np.ndarray] = []
  for sample in samples:
    arr = np.asarray(sample, dtype=np.float64).reshape(-1)
    if arr.size == 0:
      msg = "repeat stratum is empty"
      raise ValueError(msg)
    if not np.isfinite(arr).all():
      msg = "repeat stratum contains a non-finite timing"
      raise ValueError(msg)
    repeats.append(arr)
  if not repeats:
    msg = "no repeats"
    raise ValueError(msg)
  return repeats


def _repeat_resample_indices(
  repeat_sizes: list[int],
  rng: np.random.Generator,
  n_resamples: int,
) -> list[np.ndarray]:
  """Stratified by repeat: resample each repeat with replacement at its own size."""
  return [rng.integers(0, size, size=(n_resamples, size)) for size in repeat_sizes]


def _apply_indices(repeats: list[np.ndarray], indices: list[np.ndarray]) -> np.ndarray:
  parts = [repeat[idx] for repeat, idx in zip(repeats, indices, strict=True)]
  return np.concatenate(parts, axis=1)


def _pooled(repeats: list[np.ndarray]) -> np.ndarray:
  return np.concatenate(repeats)


def bootstrap_percentile_ci(
  samples: list[np.ndarray] | tuple[np.ndarray, ...],
  q: float,
  *,
  seed: int,
  n_resamples: int = N_RESAMPLES,
  level: float = CI_LEVEL,
) -> WithinSessionCI:
  """Within-session stratified bootstrap CI for the `q` percentile (q in 0..100)."""
  repeats = as_repeats(samples)
  if n_resamples < 2:
    msg = "n_resamples must be >= 2"
    raise ValueError(msg)
  if not 0.0 < level < 1.0:
    msg = "level must lie in (0, 1)"
    raise ValueError(msg)
  rng = np.random.default_rng(seed)
  indices = _repeat_resample_indices([int(r.shape[0]) for r in repeats], rng, n_resamples)
  stats = _percentile_axis(_apply_indices(repeats, indices), q, axis=1)
  alpha = 1.0 - level
  return WithinSessionCI(
    point=percentile(_pooled(repeats), q),
    low=percentile(stats, 100.0 * alpha / 2.0),
    high=percentile(stats, 100.0 * (1.0 - alpha / 2.0)),
    level=level,
    n_resamples=n_resamples,
    seed=seed,
    scope=WITHIN_SESSION,
  )


def p50_ci(
  samples: list[np.ndarray] | tuple[np.ndarray, ...],
  *,
  seed: int,
  n_resamples: int = N_RESAMPLES,
) -> WithinSessionCI:
  return bootstrap_percentile_ci(samples, 50.0, seed=seed, n_resamples=n_resamples)


def p90_ci(
  samples: list[np.ndarray] | tuple[np.ndarray, ...],
  *,
  seed: int,
  n_resamples: int = N_RESAMPLES,
) -> WithinSessionCI:
  return bootstrap_percentile_ci(samples, 90.0, seed=seed, n_resamples=n_resamples)


def _paired_p50s(
  left: list[np.ndarray],
  right: list[np.ndarray],
  *,
  seed: int,
  n_resamples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Shared within-repeat indices applied to both arms. Returns pooled and bootstrap p50s."""
  if len(left) != len(right):
    msg = "paired arms must have the same number of repeats"
    raise ValueError(msg)
  for arm_l, arm_r in zip(left, right, strict=True):
    if arm_l.shape[0] != arm_r.shape[0]:
      msg = "paired repeats must have equal length"
      raise ValueError(msg)
  rng = np.random.default_rng(seed)
  indices = _repeat_resample_indices([int(repeat.shape[0]) for repeat in left], rng, n_resamples)
  left_p = _percentile_axis(_apply_indices(left, indices), 50.0, axis=1)
  right_p = _percentile_axis(_apply_indices(right, indices), 50.0, axis=1)
  return _pooled(left), _pooled(right), left_p, right_p


def _ci_from_stats(
  point: float,
  stats: np.ndarray,
  *,
  seed: int,
  n_resamples: int,
  level: float,
) -> WithinSessionCI:
  alpha = 1.0 - level
  return WithinSessionCI(
    point=float(point),
    low=percentile(stats, 100.0 * alpha / 2.0),
    high=percentile(stats, 100.0 * (1.0 - alpha / 2.0)),
    level=level,
    n_resamples=n_resamples,
    seed=seed,
    scope=WITHIN_SESSION,
  )


def ratio_p50_ci(
  numerator: list[np.ndarray] | tuple[np.ndarray, ...],
  denominator: list[np.ndarray] | tuple[np.ndarray, ...],
  *,
  seed: int,
  n_resamples: int = N_RESAMPLES,
  level: float = CI_LEVEL,
) -> WithinSessionCI:
  """Within-session CI for the ratio of p50s. Indices are shared across the two arms."""
  pooled_l, pooled_r, left_p, right_p = _paired_p50s(
    as_repeats(numerator),
    as_repeats(denominator),
    seed=seed,
    n_resamples=n_resamples,
  )
  denom = percentile(pooled_r, 50.0)
  if denom <= 0.0 or np.any(right_p <= 0.0):
    msg = "p50 ratio denominator is non-positive"
    raise ValueError(msg)
  point = percentile(pooled_l, 50.0) / denom
  return _ci_from_stats(point, left_p / right_p, seed=seed, n_resamples=n_resamples, level=level)


def diff_p50_ci(
  left: list[np.ndarray] | tuple[np.ndarray, ...],
  right: list[np.ndarray] | tuple[np.ndarray, ...],
  *,
  seed: int,
  n_resamples: int = N_RESAMPLES,
  level: float = CI_LEVEL,
) -> WithinSessionCI:
  """Within-session CI for p50(left) - p50(right), same shared-index stratification."""
  pooled_l, pooled_r, left_p, right_p = _paired_p50s(
    as_repeats(left),
    as_repeats(right),
    seed=seed,
    n_resamples=n_resamples,
  )
  point = percentile(pooled_l, 50.0) - percentile(pooled_r, 50.0)
  return _ci_from_stats(point, left_p - right_p, seed=seed, n_resamples=n_resamples, level=level)


def ci_excludes(interval: WithinSessionCI, value: float) -> bool:
  return interval.high < value or interval.low > value


def ci_contains(interval: WithinSessionCI, value: float) -> bool:
  return interval.low <= value <= interval.high
