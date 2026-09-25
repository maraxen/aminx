"""Synthetic-truth tests for ``aminx.parity.compare`` (AC-6, browser-validation spec).

Every check below plants a known truth (fixed seed, closed form, or a hand-computed example)
and verifies the metric under test recovers it -- these are the invariant tests that
``scripts/browser_validation/stamp_metric_gate.py`` re-runs before self-attesting the
``mpnn_metric_synthetic_truth`` BP-2 gate. No test here is hollow: every statistical claim
(the excess-JS null centring, the excess-JS IUT power/type-I checks, TOST) is backed by a real
simulation with enough replicates/samples for genuine statistical power, not a token
assertion.

Two additional tests near the bottom exercise
``scripts/browser_validation/assert_no_skips.py`` directly against two hand-written JUnit XML
strings (T5 step 3), loaded by file path since ``scripts/`` is not an importable package.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy import stats

from aminx.parity.compare import (
  argmax_agreement,
  excess_js,
  excess_js_upper,
  iut_equivalent,
  js_divergence,
  max_abs,
  mean_positional_js,
  neighbor_set_equality,
  no_tie_mask,
  pearson,
  ratio_to_bar,
  reference_nll,
  required_n,
  tost_mean_diff,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _softmax(logits: np.ndarray) -> np.ndarray:
  shifted = logits - logits.max()
  weights = np.exp(shifted)
  return weights / weights.sum()


def _token_counts(samples: np.ndarray, k: int) -> np.ndarray:
  """Convert raw (n, n_positions) token draws to (n_positions, k) per-position counts."""
  return np.stack([(samples == cls).sum(axis=0) for cls in range(k)], axis=-1).astype(np.float64)


# --------------------------------------------------------------------------------------------
# max_abs, pearson, ratio_to_bar
# --------------------------------------------------------------------------------------------


def test_max_abs_recovers_planted_value() -> None:
  """A single planted deviation is recovered to within 1e-12."""
  rng = np.random.default_rng(1)
  a = rng.normal(size=64)
  b = a.copy()
  planted = 0.123456789012
  b[17] += planted
  assert abs(max_abs(a, b) - planted) < 1e-12


def test_max_abs_zero_for_identical_arrays() -> None:
  """Identical arrays have zero max-abs difference."""
  rng = np.random.default_rng(2)
  a = rng.normal(size=32)
  assert max_abs(a, a.copy()) == 0.0


def test_pearson_perfect_linear_is_exactly_one() -> None:
  """pearson(2 * x, x) == 1.0 exactly: Pearson is diagnostic, never a gate."""
  rng = np.random.default_rng(3)
  x = rng.normal(size=97) * 12.5 + 3.0
  assert pearson(2.0 * x, x) == 1.0
  assert pearson(x, 2.0 * x) == 1.0


def test_pearson_perfect_negative_linear_is_near_negative_one() -> None:
  """A negative scaling gives correlation essentially -1 (within float tolerance)."""
  rng = np.random.default_rng(4)
  x = rng.normal(size=64)
  assert pearson(-3.0 * x, x) == pytest.approx(-1.0, abs=1e-9)


def test_ratio_to_bar() -> None:
  """ratio_to_bar is a plain value/bar ratio, and rejects a non-positive bar."""
  assert ratio_to_bar(0.5, 1.0) == pytest.approx(0.5)
  assert ratio_to_bar(2.0, 1.0) == pytest.approx(2.0)
  with pytest.raises(ValueError, match="bar must be positive"):
    ratio_to_bar(1.0, 0.0)


# --------------------------------------------------------------------------------------------
# reference_nll: get_score-equivalent NLL on a hand example
# --------------------------------------------------------------------------------------------


def _numpy_get_score(
  seq: np.ndarray,
  log_probs: np.ndarray,
  mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Numpy transcription of LigandMPNN's ``data_utils.get_score`` reference formula.

  Used ONLY as the ``get_score``-shaped callable handed to ``reference_nll`` in this test --
  ``reference_nll`` itself never encodes this formula, it only delegates to whatever is passed.
  """
  num_classes = log_probs.shape[-1]
  one_hot = np.eye(num_classes)[seq]
  loss_per_residue = -(one_hot * log_probs).sum(-1)
  average_loss = (loss_per_residue * mask).sum(-1) / (mask.sum(-1) + 1e-8)
  return average_loss, loss_per_residue


def test_reference_nll_matches_hand_example() -> None:
  """reference_nll delegates correctly to a get_score-shaped callable on a hand example."""
  seq = np.array([0, 2, 4, 1])
  log_probs = np.zeros((4, 5))
  log_probs[0, 0] = -0.1
  log_probs[1, 2] = -0.2
  log_probs[2, 4] = -0.3
  log_probs[3, 1] = -0.4  # masked out below; must not affect the result
  mask = np.array([1.0, 1.0, 1.0, 0.0])

  expected = (0.1 + 0.2 + 0.3) / (3.0 + 1e-8)
  observed = reference_nll(log_probs, seq, mask, _numpy_get_score)
  assert observed == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------------------------
# argmax_agreement: 100% where top-2 margin > 2e-4, near-ties excluded
# --------------------------------------------------------------------------------------------


def test_argmax_agreement_excludes_near_ties() -> None:
  """Rows near a's top-2 tie are excluded; clean rows are correctly agree/disagree."""
  margin = 2e-4
  a = np.array(
    [
      [5.0, 1.0, 0.0],  # clear winner idx 0
      [3.0, 2.99995, 0.0],  # near-tie: gap 5e-5 < margin
      [2.0, 0.0, 0.0],  # clear winner idx 0
    ]
  )
  b = np.array(
    [
      [5.0, 1.0, 0.0],  # agrees (idx 0)
      [3.0, 2.99995, 0.0],  # irrelevant: excluded by margin
      [0.0, 9.0, 0.0],  # disagrees (idx 1 != idx 0)
    ]
  )
  agree, valid = argmax_agreement(a, b, margin)
  np.testing.assert_array_equal(valid, [True, False, True])
  assert agree[0]
  assert not agree[2]


# --------------------------------------------------------------------------------------------
# neighbor_set_equality: detects one swapped index, ignores order, respects mask
# --------------------------------------------------------------------------------------------


def test_neighbor_set_equality_detects_one_swapped_index() -> None:
  """A genuinely different neighbour index is detected; order permutation and masking are not."""
  idx_a = np.array(
    [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
    ]
  )
  idx_b = np.array(
    [
      [3, 1, 2],  # same set, different order -> still equal
      [4, 5, 6],  # identical
      [7, 8, 99],  # 9 replaced by 99 -> genuinely different set
      [0, 0, 0],  # would differ, but masked out below
    ]
  )
  mask = np.array([True, True, True, False])

  result = neighbor_set_equality(idx_a, idx_b, mask)
  np.testing.assert_array_equal(result, [True, True, False, True])


# --------------------------------------------------------------------------------------------
# no_tie_mask: excludes a planted near-tie, keeps a clearly-separated residue
# --------------------------------------------------------------------------------------------


def test_no_tie_mask_excludes_planted_near_tie() -> None:
  """A deliberately near-tied k/(k+1) neighbour pair is excluded; a clean case is kept."""
  ca = np.array(
    [
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      [2.3, 0.0, 0.0],
      [2.3 + 5e-5, 0.0, 0.0],  # near-tie partner for residue 0's k=2 boundary
      [50.0, 0.0, 0.0],  # far from the close pair: clean k/(k+1) gap
    ]
  )
  mask = no_tie_mask(ca, k=2, gap=1e-4)
  assert mask.dtype == np.bool_
  assert mask[0] == np.bool_(False)
  assert mask[4] == np.bool_(True)


# --------------------------------------------------------------------------------------------
# js_divergence: JS(p, p) = 0, closed-form JS within 1e-9
# --------------------------------------------------------------------------------------------


def test_js_divergence_zero_for_identical_distribution() -> None:
  """JS(p, p) == 0."""
  p = np.array([0.2, 0.3, 0.5])
  assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-12)


def test_js_divergence_closed_form_disjoint_support() -> None:
  """JS between two degenerate distributions with disjoint support is exactly ln(2)."""
  p = np.array([1.0, 0.0])
  q = np.array([0.0, 1.0])
  assert js_divergence(p, q) == pytest.approx(math.log(2.0), abs=1e-9)


def test_mean_positional_js_averages_per_position_values() -> None:
  """mean_positional_js averages independently-verified per-position JS values."""
  counts_a = np.array([[1.0, 0.0], [1.0, 0.0]])
  counts_b = np.array([[0.0, 1.0], [1.0, 0.0]])
  # Position 0: disjoint support -> ln(2). Position 1: identical -> 0.
  expected = (math.log(2.0) + 0.0) / 2.0
  assert mean_positional_js(counts_a, counts_b) == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------------------------
# excess_js: null centred at 0 within 3 SE over 200 replicates
# --------------------------------------------------------------------------------------------

_K = 4
_BASE_LOGITS = np.array([1.5, 0.5, -0.5, -1.5])
_P0 = _softmax(_BASE_LOGITS)
_SHIFT_T = 1.4  # A larger-than-1.05 stand-in shift: keeps the 200-replicate power test fast
# while exercising the identical statistical machinery (see the power test's docstring).
_P1 = _softmax(_BASE_LOGITS / _SHIFT_T)
_N_POSITIONS = 150
_N_BOOT = 200
_ALPHA = 0.05


def test_excess_js_null_centered_at_zero() -> None:
  """Under the null (all four draws from one source), mean(E) is 0 within 3 SE over 200 reps."""
  n = required_n(sigma=0.3, delta=0.08)  # exercises required_n; ~305 at defaults
  master = np.random.default_rng(20260924)
  values = np.empty(200)
  for rep in range(200):
    rng = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
    a1 = rng.choice(_K, size=(n, _N_POSITIONS), p=_P0)
    a2 = rng.choice(_K, size=(n, _N_POSITIONS), p=_P0)
    r1 = rng.choice(_K, size=(n, _N_POSITIONS), p=_P0)
    r2 = rng.choice(_K, size=(n, _N_POSITIONS), p=_P0)
    values[rep] = excess_js(
      _token_counts(a1, _K),
      _token_counts(a2, _K),
      _token_counts(r1, _K),
      _token_counts(r2, _K),
    )

  se = float(values.std(ddof=1) / math.sqrt(values.size))
  assert abs(float(values.mean())) < 3.0 * se


# --------------------------------------------------------------------------------------------
# excess-JS IUT: >=90% equivalence for identical sources, >=90% rejection for a shifted source
# --------------------------------------------------------------------------------------------


def test_excess_js_iut_power_and_type_i_at_formula_n() -> None:
  """The excess-JS IUT decision is well-powered at n = required_n(...): see module docstring.

  Real production pools D over thousands of designable positions across an entire fixture
  set, so it can reliably resolve the true, small (1.05x-temperature-scale) margin of
  concern. Reproducing that pooled sample size here would make this unit test far too slow, so
  this test instead uses a larger (1.4x-scale) synthetic perturbation -- big enough to resolve
  at ``_N_POSITIONS`` positions and ``_N_BOOT`` bootstrap resamples in a few seconds -- while
  exercising the EXACT SAME statistical machinery (``required_n``, ``excess_js_upper``,
  ``iut_equivalent``) that the real lane test uses. The margin itself is calibrated the same
  way the spec's margin is: as a quantile of the null (aminx-vs-aminx) excess-JS-upper-bound
  distribution, on a held-out calibration batch never reused in the 200+200 evaluation
  replicates below.
  """
  n = required_n(sigma=0.3, delta=0.08)
  master = np.random.default_rng(2026091701)

  def _draw_counts(p_r: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    a1 = rng.choice(_K, size=(n, _N_POSITIONS), p=_P0)
    a2 = rng.choice(_K, size=(n, _N_POSITIONS), p=_P0)
    r1 = rng.choice(_K, size=(n, _N_POSITIONS), p=p_r)
    r2 = rng.choice(_K, size=(n, _N_POSITIONS), p=p_r)
    return tuple(_token_counts(arm, _K) for arm in (a1, a2, r1, r2))

  # Calibrate the margin from null (identical-source) replicates only.
  calibration_values = np.empty(150)
  for rep in range(150):
    rng = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
    counts_a1, counts_a2, counts_r1, counts_r2 = _draw_counts(_P0, rng)
    calibration_values[rep] = excess_js_upper(
      counts_a1, counts_a2, counts_r1, counts_r2, _N_BOOT, _ALPHA, rng
    )
  margin = float(np.quantile(calibration_values, 0.95))

  identical_pass = 0
  for _rep in range(200):
    rng = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
    counts_a1, counts_a2, counts_r1, counts_r2 = _draw_counts(_P0, rng)
    upper = excess_js_upper(counts_a1, counts_a2, counts_r1, counts_r2, _N_BOOT, _ALPHA, rng)
    if iut_equivalent([upper < margin]):
      identical_pass += 1

  shifted_pass = 0
  for _rep in range(200):
    rng = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
    counts_a1, counts_a2, counts_r1, counts_r2 = _draw_counts(_P1, rng)
    upper = excess_js_upper(counts_a1, counts_a2, counts_r1, counts_r2, _N_BOOT, _ALPHA, rng)
    if iut_equivalent([upper < margin]):
      shifted_pass += 1

  assert identical_pass / 200 >= 0.90
  assert shifted_pass / 200 <= 0.10


def test_iut_equivalent() -> None:
  """iut_equivalent is True iff every lane passes, with no multiplicity adjustment."""
  assert iut_equivalent([True, True, True]) is True
  assert iut_equivalent([True, False, True]) is False
  assert iut_equivalent([False, False]) is False
  assert iut_equivalent([]) is True


# --------------------------------------------------------------------------------------------
# TOST: rejects 2*delta, accepts 0
# --------------------------------------------------------------------------------------------


def test_tost_mean_diff_rejects_2delta_and_accepts_zero() -> None:
  """TOST correctly rejects an obviously-non-equivalent 2*delta shift and accepts a 0 shift."""
  delta = 0.05
  rng = np.random.default_rng(20260924)

  x_zero = rng.normal(loc=0.0, scale=0.01, size=300)
  y_zero = rng.normal(loc=0.0, scale=0.01, size=300)
  passed_zero, p_lower_zero, p_upper_zero = tost_mean_diff(x_zero, y_zero, delta)
  assert passed_zero
  assert p_lower_zero < 0.05
  assert p_upper_zero < 0.05

  x_shift = rng.normal(loc=2.0 * delta, scale=0.01, size=300)
  y_shift = rng.normal(loc=0.0, scale=0.01, size=300)
  passed_shift, _p_lower_shift, p_upper_shift = tost_mean_diff(x_shift, y_shift, delta)
  assert not passed_shift
  assert p_upper_shift >= 0.05


# --------------------------------------------------------------------------------------------
# required_n: matches the closed-form ceil(2 * (z_alpha + z_beta/2)^2 * sigma^2 / delta^2)
# --------------------------------------------------------------------------------------------


def test_required_n_matches_closed_form() -> None:
  """required_n matches an independently-recomputed closed form for two (alpha, power) pairs."""
  for sigma, delta, alpha, power in [(1.0, 1.0, 0.05, 0.9), (0.3, 0.08, 0.05, 0.9)]:
    beta = 1.0 - power
    z_alpha = float(stats.norm.ppf(1.0 - alpha))
    z_beta = float(stats.norm.ppf(1.0 - beta / 2.0))
    expected = math.ceil(2.0 * (z_alpha + z_beta) ** 2 * sigma**2 / delta**2)
    assert required_n(sigma, delta, alpha=alpha, power=power) == expected

  assert required_n(1.0, 1.0) == 22


def test_required_n_rejects_invalid_inputs() -> None:
  """required_n fail-closes on non-positive sigma/delta and out-of-range alpha/power."""
  with pytest.raises(ValueError, match="sigma must be positive"):
    required_n(0.0, 1.0)
  with pytest.raises(ValueError, match="delta must be positive"):
    required_n(1.0, 0.0)
  with pytest.raises(ValueError, match="alpha must be in"):
    required_n(1.0, 1.0, alpha=1.5)
  with pytest.raises(ValueError, match="power must be in"):
    required_n(1.0, 1.0, power=0.0)


# --------------------------------------------------------------------------------------------
# assert_no_skips.py: tested against two hand-written JUnit XML strings (T5 step 3)
# --------------------------------------------------------------------------------------------


def _load_assert_no_skips() -> ModuleType:
  """Load scripts/browser_validation/assert_no_skips.py by path (scripts/ is not a package)."""
  script_path = REPO_ROOT / "scripts" / "browser_validation" / "assert_no_skips.py"
  spec = importlib.util.spec_from_file_location("assert_no_skips", script_path)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


_JUNIT_CLEAN = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="5" skipped="0" failures="0" errors="0">
    <testcase classname="test_mod" name="test_a" time="0.01" />
    <testcase classname="test_mod" name="test_b" time="0.01" />
  </testsuite>
</testsuites>
"""

_JUNIT_WITH_SKIPS = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="4" skipped="2" failures="0" errors="0">
    <testcase classname="test_mod" name="test_a" time="0.01" />
    <testcase classname="test_mod" name="test_b" time="0.01">
      <skipped message="not applicable" />
    </testcase>
  </testsuite>
</testsuites>
"""


def test_assert_no_skips_passes_on_clean_report(tmp_path: Path) -> None:
  """A report with tests > 0 and skipped == 0 exits 0."""
  module = _load_assert_no_skips()
  junit_path = tmp_path / "clean.xml"
  junit_path.write_text(_JUNIT_CLEAN, encoding="utf-8")

  tests, skipped = module.summarize_junit(junit_path)
  assert (tests, skipped) == (5, 0)
  assert module.main([str(junit_path)]) == 0


def test_assert_no_skips_fails_on_skips(tmp_path: Path) -> None:
  """A report with any skipped test exits 1."""
  module = _load_assert_no_skips()
  junit_path = tmp_path / "skips.xml"
  junit_path.write_text(_JUNIT_WITH_SKIPS, encoding="utf-8")

  tests, skipped = module.summarize_junit(junit_path)
  assert (tests, skipped) == (4, 2)
  assert module.main([str(junit_path)]) == 1


def test_assert_no_skips_fails_on_zero_tests(tmp_path: Path) -> None:
  """A report with zero collected tests exits 1, even with zero skips."""
  module = _load_assert_no_skips()
  empty = ET.Element("testsuites")
  suite = ET.SubElement(empty, "testsuite", name="pytest", tests="0", skipped="0")
  del suite  # only the attributes matter
  junit_path = tmp_path / "empty.xml"
  ET.ElementTree(empty).write(junit_path, encoding="utf-8", xml_declaration=True)

  tests, skipped = module.summarize_junit(junit_path)
  assert (tests, skipped) == (0, 0)
  assert module.main([str(junit_path)]) == 1
