# ruff: noqa: S101
"""Synthetic ground truth for the distributional-protocol statistics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "parity" / "sample_dist_stats.py"


def _load() -> Any:
  spec = importlib.util.spec_from_file_location("sample_dist_stats_under_test", _SCRIPT)
  if spec is None or spec.loader is None:
    msg = f"cannot load {_SCRIPT}"
    raise RuntimeError(msg)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


stats = _load()


def _filled(token: int, n: int, length: int) -> np.ndarray:
  return np.full((n, length), token, dtype=np.int64)


def _column_counts(n: int, length: int, n_ones: int) -> np.ndarray:
  """Binary samples with exactly ``n_ones`` ones in every column."""
  column = np.zeros(n, dtype=np.int64)
  column[:n_ones] = 1
  return np.broadcast_to(column[:, None], (n, length)).copy()


def test_spec_defaults_match_the_protocol() -> None:
  """B, δ, the pilot clip, and δ_χ are the spec's values."""
  assert stats.DEFAULT_N_BOOT == 2000
  assert stats.DEFAULT_DELTA == 0.02
  assert stats.DELTA_LO == 0.01
  assert stats.DELTA_HI == 0.05
  assert stats.DELTA_CHI == 0.05
  assert stats.CHI_BINS == 36


def test_a_tv_identical_is_zero_and_disjoint_is_one() -> None:
  """(a) Identical empirical distributions have TV 0; disjoint supports have TV 1."""
  alphabet = 5
  length = 4
  n = 12
  # Same counts, different row order: identical distributions, not the same array.
  base = np.arange(n, dtype=np.int64)[:, None] % alphabet
  base = np.broadcast_to(base, (n, length)).copy()
  shuffled = base[::-1].copy()
  mask = np.array([True, False, True, False])
  tv = stats.tv_per_position(base, shuffled, alphabet)
  assert tv.shape == (length,)
  assert np.all(tv == 0.0)
  assert stats.mean_tv(base, shuffled, alphabet, mask) == 0.0

  left = _filled(0, n, length)
  right = _filled(1, n, length)
  disjoint = stats.tv_per_position(left, right, alphabet)
  assert np.all(disjoint == 1.0)
  assert stats.mean_tv(left, right, alphabet, np.ones(length, dtype=bool)) == 1.0

  # Mask selects the position whose TV is known.
  mixed_x = np.column_stack([_filled(0, n, 1), _filled(0, n, 1)])
  mixed_y = np.column_stack([_filled(0, n, 1), _filled(1, n, 1)])
  assert stats.mean_tv(mixed_x, mixed_y, alphabet, np.array([True, False])) == 0.0
  assert stats.mean_tv(mixed_x, mixed_y, alphabet, np.array([False, True])) == 1.0
  assert stats.mean_tv(mixed_x, mixed_y, alphabet, np.array([True, True])) == 0.5

  # Hand distribution: p = (1/2, 1/2) vs (3/4, 1/4) → TV = 1/4.
  known_x = np.array([[0], [0], [1], [1]], dtype=np.int64)
  known_y = np.array([[0], [0], [0], [1]], dtype=np.int64)
  assert stats.tv_per_position(known_x, known_y, 2)[0] == 0.25


def test_b_delta_cancels_plugin_bias() -> None:
  """(b) Equal-n iid draws from one law: raw TV stays biased, Δ_s is ~0."""
  rng = np.random.default_rng(7)
  n, length, alphabet = 400, 250, 2
  mask = np.ones(length, dtype=bool)
  gaps: list[float] = []
  biases: list[float] = []
  for _ in range(4):
    u1 = rng.integers(0, alphabet, size=(n, length), dtype=np.int64)
    u2 = rng.integers(0, alphabet, size=(n, length), dtype=np.int64)
    a = rng.integers(0, alphabet, size=(n, length), dtype=np.int64)
    d_a = stats.mean_tv(a, u1, alphabet, mask)
    d_u2 = stats.mean_tv(u2, u1, alphabet, mask)
    gaps.append(stats.delta_s(a, u1, u2, alphabet, mask))
    biases.append(d_a)
    biases.append(d_u2)
  # Bernoulli plug-in TV is about 0.028 here; the contrast is a few thousandths.
  assert float(np.mean(biases)) > 0.02
  assert abs(float(np.mean(gaps))) < 0.01
  assert max(abs(gap) for gap in gaps) < 0.03


def test_c_positive_control_passes_and_sidecar_passes() -> None:
  """(c) A drawn from U1's law grades pass at δ = 0.02."""
  n = 16
  # Designed column: A, U1, and U2 are the same point mass. The masked column
  # shifts only A. Ignoring the mask makes Δ = 1/2, which cannot pass.
  mask = np.array([True, False])
  reference = np.column_stack([_filled(0, n, 1), _filled(0, n, 1)])
  aminx = np.column_stack([_filled(0, n, 1), _filled(1, n, 1)])
  structures = 3
  grade = stats.grade_condition(
    [aminx] * structures,
    [reference] * structures,
    [reference] * structures,
    [mask] * structures,
    alphabet_size=2,
    delta=0.02,
    n_boot=32,
    seed=0,
  )
  assert grade.verdict == "pass"
  assert grade.mean_delta == 0.0
  assert grade.upper_95 == 0.0
  assert grade.max_delta == 0.0
  assert grade.max_delta < 2.0 * grade.delta
  assert stats.grade_sidecar([grade]).verdict == "pass"


def test_d_negative_control_must_fail() -> None:
  """(d) A shift of TV >= 0.1 fails at δ = 0.02."""
  n, length = 16, 2
  mask = np.ones(length, dtype=bool)
  u = _filled(0, n, length)
  a = _filled(1, n, length)
  assert stats.mean_tv(a, u, 2, mask) == 1.0
  grade = stats.grade_condition(
    [a, a],
    [u, u],
    [u, u],
    [mask, mask],
    alphabet_size=2,
    delta=0.02,
    n_boot=24,
    seed=1,
  )
  assert grade.mean_delta == 1.0
  assert grade.lower_90 > 0.02
  assert grade.verdict == "fail"
  positive = stats.grade_condition(
    [u],
    [u],
    [u],
    [mask],
    alphabet_size=2,
    delta=0.02,
    n_boot=8,
    seed=1,
  )
  assert stats.grade_sidecar([positive, grade]).verdict == "fail"


def test_e_near_margin_shift_does_not_pass() -> None:
  """(e) A shift just above 2δ is inconclusive or fail, never pass."""
  # One hot token out of 16 against a point mass: plug-in TV = 1/16 = 0.0625.
  # That is above 2δ = 0.04, so the max gate blocks pass, and below the gross
  # TV>=0.1 negative control. Resamples often draw zero hot tokens, so the
  # 90% lower bound stays at 0 and the cell is not a forced fail.
  n = 16
  length = 1
  mask = np.ones(length, dtype=bool)
  u = _filled(0, n, length)
  a = _column_counts(n, length, n_ones=1)
  point = stats.delta_s(a, u, u, 2, mask)
  assert point == np.float64(1.0 / 16.0)
  grade = stats.grade_condition(
    [a, a, a],
    [u, u, u],
    [u, u, u],
    [mask, mask, mask],
    alphabet_size=2,
    delta=0.02,
    n_boot=40,
    seed=2,
  )
  assert grade.max_delta == np.float64(1.0 / 16.0)
  assert grade.verdict in ("inconclusive", "fail")
  assert grade.verdict != "pass"
  near = stats.CellGrade(
    grade.verdict,
    grade.mean_delta,
    grade.upper_95,
    grade.lower_90,
    grade.max_delta,
    0.02,
  )
  mixed = stats.grade_sidecar([stats.CellGrade("pass", 0.0, 0.0, 0.0, 0.0, 0.02), near])
  assert mixed.verdict == grade.verdict


def test_confirmatory_bounds_and_sidecar_rule() -> None:
  """Pass, fail, and sidecar rollup use the spec's strict inequalities."""
  delta = 0.02
  assert stats.confirmatory_verdict(0.019, 0.0, 0.039, delta) == "pass"
  # max_s Δ_s < 2δ is strict.
  assert stats.confirmatory_verdict(0.019, 0.0, 0.04, delta) == "inconclusive"
  # 95% upper bound < δ is strict.
  assert stats.confirmatory_verdict(0.02, 0.0, 0.0, delta) == "inconclusive"
  # 90% lower bound > δ is strict, and it is not a pass.
  assert stats.confirmatory_verdict(0.03, 0.02, 0.03, delta) == "inconclusive"
  assert stats.confirmatory_verdict(0.03, 0.021, 0.03, delta) == "fail"
  # Pass is decided first when both descriptions could apply.
  assert stats.confirmatory_verdict(0.019, 0.021, 0.01, delta) == "pass"

  def cell(verdict: str) -> Any:
    return stats.CellGrade(verdict, 0.0, 0.0, 0.0, 0.0, delta)

  assert stats.grade_sidecar([cell("pass"), cell("pass")]).verdict == "pass"
  assert stats.grade_sidecar([cell("pass"), cell("fail")]).verdict == "fail"
  assert stats.grade_sidecar([cell("pass"), cell("inconclusive")]).verdict == "inconclusive"
  assert stats.grade_sidecar([cell("fail"), cell("inconclusive")]).verdict == "fail"
  assert stats.grade_sidecar([cell("inconclusive")]).verdict == "inconclusive"


def test_f_pilot_picks_smallest_m_inside_delta_clip() -> None:
  """(f) Null pilot clips δ into [0.01, 0.05] and keeps the smallest qualifying m."""
  n = 8
  # Position 1 is undesigned and distorted. Including it would move δ off the floor.
  mask = np.array([True, False])
  quiet = np.column_stack([_filled(0, n, 1), _filled(0, n, 1)])
  loud = np.column_stack([_filled(0, n, 1), _filled(1, n, 1)])
  # Shifts grow with m. 0 < 0.01 <= 1/8 <= 1/4, all dyadic, so the cut is exact.
  c_small = quiet
  c_mid = np.column_stack([_column_counts(n, 1, n_ones=1), _filled(0, n, 1)])
  c_large = np.column_stack([_column_counts(n, 1, n_ones=2), _filled(0, n, 1)])
  controls = {1.5: c_large, 1.1: c_small, 1.25: c_mid}
  result = stats.derive_pilot(
    [quiet, quiet],
    [quiet, loud],
    [quiet, quiet],
    [mask, mask],
    [controls, controls],
    alphabet_size=2,
    n_boot=16,
    seed=3,
  )
  assert result.h_hat == 0.0
  assert result.q_hat == 0.0
  assert result.delta == 0.01
  assert 0.01 <= result.delta <= 0.05
  assert result.chosen_m == 1.25
  assert result.escalate is False
  assert stats.resolve_pilot(result, repilot=False) == "ok"
  assert stats.resolve_pilot(result, repilot=True) == "ok"
  by_m = dict(result.delta_neg)
  assert by_m[1.1] < result.delta + 2.0 * result.h_hat
  assert by_m[1.25] >= result.delta + 2.0 * result.h_hat
  assert by_m[1.5] > by_m[1.25] > by_m[1.1]

  none = stats.derive_pilot(
    [quiet],
    [quiet],
    [quiet],
    [mask],
    [{1.1: quiet, 1.25: quiet, 1.5: quiet}],
    alphabet_size=2,
    n_boot=8,
    seed=4,
  )
  assert none.chosen_m is None
  assert none.escalate is True
  assert stats.resolve_pilot(none, repilot=False) == "escalate"
  assert stats.resolve_pilot(none, repilot=True) == "instrument_invalid"

  # Two structures, null contrasts 0 and 1. q̂ is the max, not the mean, and
  # ĥ is the half-width of the pooled mean (here the constant 1/2).
  zeros_s = np.column_stack([_filled(0, n, 1), _filled(0, n, 1)])
  ones_s = np.column_stack([_filled(1, n, 1), _filled(0, n, 1)])
  pilot_controls = {1.1: quiet, 1.25: c_mid, 1.5: c_large}
  hot = stats.derive_pilot(
    [zeros_s, zeros_s],
    [zeros_s, zeros_s],
    [zeros_s, ones_s],
    [mask, mask],
    [pilot_controls, pilot_controls],
    alphabet_size=2,
    n_boot=12,
    seed=5,
  )
  assert hot.h_hat == 0.0
  assert hot.q_hat == 1.0
  assert hot.delta == 0.05
  assert hot.chosen_m == 1.25
  assert hot.escalate is True
  assert stats.resolve_pilot(hot, repilot=True) == "instrument_invalid"


def test_g_shim_check_flags_distorted_u1() -> None:
  """(g) Shimmed U1 that leaves U2 raises instrument_invalid; a matched shim does not."""
  n, length = 10, 2
  mask = np.ones(length, dtype=bool)
  zeros = _filled(0, n, length)
  ones = _filled(1, n, length)
  # U1 and U3 agree (both shimmed) and disagree with unshimmed U2.
  distorted = stats.shim_check(
    [ones, ones],
    [zeros, zeros],
    [ones, ones],
    [mask, mask],
    alphabet_size=2,
    h_hat=0.0,
  )
  assert distorted.mean_gap == 1.0
  assert distorted.instrument_invalid is True
  # Strict inequality: a gap equal to ĥ is not instrument_invalid.
  tied = stats.shim_check([ones], [zeros], [ones], [mask], alphabet_size=2, h_hat=1.0)
  assert tied.mean_gap == 1.0
  assert tied.instrument_invalid is False
  healthy = stats.shim_check([zeros], [zeros], [zeros], [mask], alphabet_size=2, h_hat=0.0)
  assert healthy.mean_gap == 0.0
  assert healthy.instrument_invalid is False


def test_h_bootstrap_is_reproducible_and_shares_u1() -> None:
  """(h) A fixed seed repeats, and both Δ terms use one U1 row resample."""
  rng = np.random.default_rng(11)
  n, length, alphabet = 18, 5, 4
  a = rng.integers(0, alphabet, size=(n, length), dtype=np.int64)
  u1 = rng.integers(0, alphabet, size=(n, length), dtype=np.int64)
  u2 = rng.integers(0, alphabet, size=(n, length), dtype=np.int64)
  # Row 0 differs from every other U1 row, so the reference resample is visible.
  u1[0] = 0
  u1[1:] = 1
  mask = np.ones(length, dtype=bool)
  kwargs: dict[str, Any] = {
    "alphabet_size": alphabet,
    "mask": mask,
    "n_boot": 30,
    "seed": 99,
  }
  first = stats.bootstrap_delta_s(a, u1, u2, **kwargs)
  second = stats.bootstrap_delta_s(a, u1, u2, **kwargs)
  assert np.array_equal(first.delta, second.delta)
  assert np.array_equal(first.indices_u1, second.indices_u1)
  assert np.array_equal(first.indices_left, second.indices_left)
  assert np.array_equal(first.indices_right, second.indices_right)

  def manual(idx_u1: np.ndarray) -> np.ndarray:
    out = np.empty(kwargs["n_boot"], dtype=np.float64)
    for b in range(kwargs["n_boot"]):
      out[b] = stats.delta_s(
        a[first.indices_left[b]],
        u1[idx_u1[b]],
        u2[first.indices_right[b]],
        alphabet,
        mask,
      )
    return out

  shared = manual(first.indices_u1)
  assert np.allclose(shared, first.delta)
  flipped = np.where(first.indices_u1 == 0, 1, 0)
  alternatives = (manual(flipped), manual(np.zeros_like(first.indices_u1)))
  assert any(not np.allclose(alt, first.delta) for alt in alternatives)


def test_chi1_tv_uses_the_has_chi1_table_and_36_bins() -> None:
  """χ1 TV keeps positions whose modal AA has χ1, and only samples of that AA."""
  # AA 0 has no χ1. AA 1 and AA 3 do. Nothing here is hardcoded chemistry.
  has_chi1 = np.array([False, True, False, True])
  modal = np.array([0, 1, 3], dtype=np.int64)
  aa_x = np.array(
    [
      [0, 1, 3],
      [1, 1, 3],
      [2, 0, 3],
      [3, 2, 3],
    ],
    dtype=np.int64,
  )
  aa_y = np.array(
    [
      [0, 1, 3],
      [0, 1, 3],
      [0, 1, 3],
      [1, 0, 3],
    ],
    dtype=np.int64,
  )
  # Position 1, modal AA 1: X rows 0 and 1 contribute bin 35; Y rows 0..2 contribute bin 0.
  # Decoy rows sit on bin 0 and must not pull the conditional TV below 1.
  chi_x = np.array(
    [
      [0, 35, 4],
      [0, 35, 4],
      [0, 0, 4],
      [0, 0, 4],
    ],
    dtype=np.int64,
  )
  chi_y = np.array(
    [
      [10, 0, 4],
      [10, 0, 4],
      [10, 0, 4],
      [10, 0, 4],
    ],
    dtype=np.int64,
  )
  tv = stats.tv_chi1_per_position(aa_x, chi_x, aa_y, chi_y, modal, has_chi1)
  assert np.isnan(tv[0])
  assert tv[1] == 1.0
  assert tv[2] == 0.0
  assert stats.mean_tv_chi1(aa_x, chi_x, aa_y, chi_y, modal, has_chi1) == 0.5
  designed = np.array([False, True, False])
  assert stats.mean_tv_chi1(aa_x, chi_x, aa_y, chi_y, modal, has_chi1, designed) == 1.0

  # The table is data. Marking AA 0 as having χ1 brings position 0 into the mean.
  forced = np.array([True, True, False, True])
  forced_tv = stats.tv_chi1_per_position(aa_x, chi_x, aa_y, chi_y, modal, forced)
  assert np.isfinite(forced_tv[0])

  chi_bad = chi_x.copy()
  chi_bad[0, 1] = 36
  with pytest.raises(ValueError, match="chi bins"):
    stats.tv_chi1_per_position(aa_x, chi_bad, aa_y, chi_y, modal, has_chi1)
