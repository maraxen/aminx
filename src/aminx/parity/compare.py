"""Comparison-metric primitives for the browser-validation parity harness (AC-6).

Numpy/scipy only -- deliberately framework-agnostic (no torch, no JAX) so this module can be
imported by any harness stage regardless of which side of a comparison it is instrumenting.

Three families of primitive live here:

- **Pointwise/exact comparisons**: ``max_abs``, ``pearson``, ``ratio_to_bar``, ``reference_nll``,
  ``argmax_agreement``, ``neighbor_set_equality``, ``no_tie_mask``.
- **Distributional comparisons for sampled sequences**: ``js_divergence``, ``mean_positional_js``,
  ``excess_js``, ``excess_js_upper`` -- the excess-JS equivalence machinery described in the
  browser-validation spec's "Sampling statistics" section.
- **Equivalence-test plumbing**: ``iut_equivalent``, ``tost_mean_diff``, ``required_n``.

``reference_nll`` is the one function that touches a caller-supplied reference formula
(``get_score``-shaped), and it deliberately never retypes that formula: it only adapts the
calling convention, so a reference implementation lifted by AST (see
``tests/parity/reference_utils.load_reference_function``) can be dropped in unmodified.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
  from collections.abc import Callable, Sequence


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
  """Compute the maximum absolute elementwise difference between ``a`` and ``b``."""
  a_arr = np.asarray(a, dtype=np.float64)
  b_arr = np.asarray(b, dtype=np.float64)
  if a_arr.shape != b_arr.shape:
    msg = "max_abs inputs must share the same shape"
    raise ValueError(msg)
  return float(np.max(np.abs(a_arr - b_arr)))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
  """Compute the Pearson correlation coefficient between flattened ``a`` and ``b``.

  Uses a single ``sqrt(sum(a_c**2) * sum(b_c**2))`` in the denominator (rather than a product
  of two separate square roots) specifically so that a perfect linear relationship -- e.g.
  ``pearson(2 * x, x)`` -- recovers exactly 1.0 rather than a value one ULP off; this was
  verified against 5000 random arrays before being relied on by the synthetic-truth test.
  Pearson is diagnostic only and is never a gate (see the bars table): it is not used to
  decide pass/fail anywhere in this module.
  """
  a_arr = np.asarray(a, dtype=np.float64).ravel()
  b_arr = np.asarray(b, dtype=np.float64).ravel()
  if a_arr.shape != b_arr.shape:
    msg = "pearson inputs must share the same flattened shape"
    raise ValueError(msg)
  if a_arr.size == 0:
    msg = "pearson inputs must be non-empty"
    raise ValueError(msg)

  a_centered = a_arr - a_arr.mean()
  b_centered = b_arr - b_arr.mean()
  sum_aa = float(np.sum(a_centered * a_centered))
  sum_bb = float(np.sum(b_centered * b_centered))
  denominator = math.sqrt(sum_aa * sum_bb)
  if denominator == 0.0:
    return 1.0 if np.allclose(a_centered, b_centered) else 0.0
  numerator = float(np.sum(a_centered * b_centered))
  return float(np.clip(numerator / denominator, -1.0, 1.0))


def ratio_to_bar(value: float, bar: float) -> float:
  """Compute ``value / bar``: how a metric sits relative to its pre-registered bar.

  A ratio below 1.0 means ``value`` is within the bar; at or above 1.0 means it meets or
  exceeds it. ``bar`` must be strictly positive -- a bar is always a positive tolerance or
  count, never zero or negative.
  """
  if bar <= 0.0:
    msg = f"bar must be positive, got {bar!r}"
    raise ValueError(msg)
  return float(value) / float(bar)


def reference_nll(
  log_probs: object,
  seq: object,
  mask: object,
  get_score: Callable[[object, object, object], tuple[object, object]],
) -> float:
  """Compute the reference masked-average NLL by delegating to the caller's ``get_score``.

  ``get_score`` is a reference aggregator such as LigandMPNN's ``data_utils.get_score``
  (``get_score(S, log_probs, mask) -> (average_loss, loss_per_residue)``), typically lifted by
  AST from the reference checkout so its formula is never retyped here (a transcribed copy
  would be a second definition that can silently stop matching the thing it is supposed to be
  a reference for). This function only adapts the calling convention -- ``(log_probs, seq,
  mask)`` here versus the reference's own ``(seq, log_probs, mask)`` order -- and extracts a
  plain float from whatever tensor/array type ``get_score`` returns.

  Only the first returned element (the averaged loss) is used.
  """
  average_loss, _loss_per_residue = get_score(seq, log_probs, mask)
  return float(np.asarray(average_loss).reshape(-1)[0])


def argmax_agreement(
  a: np.ndarray,
  b: np.ndarray,
  margin: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Compare per-row argmax between ``a`` and ``b``, excluding ``a``'s near-ties.

  ``a`` is treated as the reference side: a row's top-1-vs-top-2 margin is measured on ``a``,
  and rows with margin at or below ``margin`` are excluded (near-ties are not exact-match
  material, matching the browser-validation bars table: "100% where top-2 margin > 2e-4").

  Args:
    a: Reference scores, shape ``(n, k)`` with ``k >= 2``.
    b: Comparison scores, shape ``(n, k)``.
    margin: Minimum top-2 gap (on ``a``) required for a row to count.

  Returns:
    ``(agree, valid)``: boolean arrays of shape ``(n,)``. ``agree[i]`` is
    ``argmax(a[i]) == argmax(b[i])``; ``valid[i]`` is whether row ``i`` clears the margin.
    Exact agreement should be checked as ``np.all(agree[valid])``.
  """
  a_arr = np.asarray(a, dtype=np.float64)
  b_arr = np.asarray(b, dtype=np.float64)
  if a_arr.shape != b_arr.shape:
    msg = "argmax_agreement inputs must share the same shape"
    raise ValueError(msg)
  if a_arr.ndim != 2 or a_arr.shape[-1] < 2:
    msg = "argmax_agreement expects (n, k) arrays with k >= 2"
    raise ValueError(msg)

  sorted_desc = np.sort(a_arr, axis=-1)[..., ::-1]
  top_gap = sorted_desc[..., 0] - sorted_desc[..., 1]
  valid = top_gap > margin
  agree = np.argmax(a_arr, axis=-1) == np.argmax(b_arr, axis=-1)
  return agree, valid


def neighbor_set_equality(
  idx_a: np.ndarray,
  idx_b: np.ndarray,
  mask: np.ndarray,
) -> np.ndarray:
  """Compare per-residue neighbour SETS (order-independent) between ``idx_a`` and ``idx_b``.

  Args:
    idx_a: Neighbour indices, shape ``(L, k)``.
    idx_b: Neighbour indices, shape ``(L, k)``.
    mask: Per-residue validity mask, shape ``(L,)``. Masked-out residues are reported as
      equal (``True``) unconditionally, since they were never evaluated.

  Returns:
    Boolean array of shape ``(L,)``: ``True`` where the set of neighbour indices agrees (or
    the residue is masked out).
  """
  idx_a_arr = np.asarray(idx_a)
  idx_b_arr = np.asarray(idx_b)
  valid = np.asarray(mask, dtype=bool)
  if idx_a_arr.shape != idx_b_arr.shape:
    msg = "neighbor_set_equality inputs must share the same shape"
    raise ValueError(msg)
  if valid.shape != idx_a_arr.shape[:1]:
    msg = "mask must have shape (L,) matching the leading axis of idx_a/idx_b"
    raise ValueError(msg)

  result = np.ones(idx_a_arr.shape[0], dtype=bool)
  for i in range(idx_a_arr.shape[0]):
    if not valid[i]:
      continue
    result[i] = set(idx_a_arr[i].tolist()) == set(idx_b_arr[i].tolist())
  return result


def no_tie_mask(ca: np.ndarray, k: int, gap: float = 1e-4) -> np.ndarray:
  """Flag residues whose k-th vs (k+1)-th nearest-CA-neighbour distance gap is safe.

  Excludes (returns ``False`` for) residues whose k-th and (k+1)-th nearest-neighbour
  distances differ by ``gap`` or less: a near-tie at the k-NN boundary means two
  implementations could legitimately disagree on which residue is the k-th neighbour, so an
  EXACT-tier comparison should not be evaluated there.

  Args:
    ca: CA coordinates, shape ``(L, 3)``.
    k: Neighbour rank to test the boundary of (must satisfy ``1 <= k < L - 1``).
    gap: Minimum safe gap (Angstrom) between the k-th and (k+1)-th distances.

  Returns:
    Boolean array of shape ``(L,)``: ``True`` where the residue is NOT near a k-NN tie.
  """
  ca_arr = np.asarray(ca, dtype=np.float64)
  if ca_arr.ndim != 2 or ca_arr.shape[-1] != 3:
    msg = "no_tie_mask expects CA coordinates with shape (L, 3)"
    raise ValueError(msg)
  n_residues = ca_arr.shape[0]
  if not 1 <= k < n_residues - 1:
    msg = f"k must satisfy 1 <= k < L - 1 (L={n_residues}), got k={k}"
    raise ValueError(msg)

  diffs = ca_arr[:, None, :] - ca_arr[None, :, :]
  distances = np.sqrt(np.sum(diffs * diffs, axis=-1))
  np.fill_diagonal(distances, np.inf)
  sorted_distances = np.sort(distances, axis=-1)
  kth = sorted_distances[:, k - 1]
  k_plus_1th = sorted_distances[:, k]
  return (k_plus_1th - kth) > gap


def js_divergence(p: Sequence[float] | np.ndarray, q: Sequence[float] | np.ndarray) -> float:
  """Compute the Jensen-Shannon divergence (base e / natural log) between ``p`` and ``q``.

  ``p`` and ``q`` are normalized internally, so raw counts are accepted directly.
  """
  p_arr = np.asarray(p, dtype=np.float64)
  q_arr = np.asarray(q, dtype=np.float64)
  if p_arr.shape != q_arr.shape:
    msg = "js_divergence inputs must share the same shape"
    raise ValueError(msg)
  if p_arr.ndim != 1:
    msg = "js_divergence expects 1-D vectors; use mean_positional_js for (n_positions, k) tables"
    raise ValueError(msg)
  return float(_js_divergence_rows(p_arr[None, :], q_arr[None, :])[0])


def _js_divergence_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
  """Batched JS divergence over the last axis; broadcasts over any leading batch axes."""
  p_norm = p / p.sum(axis=-1, keepdims=True)
  q_norm = q / q.sum(axis=-1, keepdims=True)
  mixture = 0.5 * (p_norm + q_norm)
  with np.errstate(divide="ignore", invalid="ignore"):
    kl_p = np.sum(np.where(p_norm > 0, p_norm * np.log(p_norm / mixture), 0.0), axis=-1)
    kl_q = np.sum(np.where(q_norm > 0, q_norm * np.log(q_norm / mixture), 0.0), axis=-1)
  return 0.5 * (kl_p + kl_q)


def mean_positional_js(counts_a: np.ndarray, counts_b: np.ndarray) -> float | np.ndarray:
  """Compute D(X, Y): the mean over positions of the per-position JS divergence.

  ``counts_a``/``counts_b`` are per-position token-count tables of shape ``(..., n_positions,
  k)``: unbatched callers pass ``(n_positions, k)`` and get back a scalar ``float``; batched
  callers (see ``excess_js_upper``) pass a leading batch axis, e.g. ``(n_boot, n_positions,
  k)``, and get back an array of shape ``(n_boot,)``.
  """
  counts_a_arr = np.asarray(counts_a, dtype=np.float64)
  counts_b_arr = np.asarray(counts_b, dtype=np.float64)
  if counts_a_arr.shape != counts_b_arr.shape:
    msg = "mean_positional_js inputs must share the same shape"
    raise ValueError(msg)
  if counts_a_arr.ndim < 2:
    msg = "mean_positional_js expects (..., n_positions, k) count tables"
    raise ValueError(msg)

  per_position = _js_divergence_rows(counts_a_arr, counts_b_arr)
  result = per_position.mean(axis=-1)
  return float(result) if result.ndim == 0 else result


def excess_js(
  a1: np.ndarray,
  a2: np.ndarray,
  r1: np.ndarray,
  r2: np.ndarray,
) -> float | np.ndarray:
  """Compute E = D(a1, r1) - 1/2 * [D(a1, a2) + D(r1, r2)].

  ``a1``/``a2`` are two independent aminx-side count draws, ``r1``/``r2`` two independent
  reference-side draws (same per-position shape, same ``n`` per arm). Subtracting the average
  of the two same-source terms cancels the plug-in bias floor common to all three JS
  estimates, so ``E`` is unbiased at 0 when all four draws come from the same source.
  Shapes/typing follow ``mean_positional_js``: batched inputs give a batched result.
  """
  d_a1_r1 = mean_positional_js(a1, r1)
  d_a1_a2 = mean_positional_js(a1, a2)
  d_r1_r2 = mean_positional_js(r1, r2)
  return d_a1_r1 - 0.5 * (d_a1_a2 + d_r1_r2)


def _bootstrap_counts(counts: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
  """Multinomial-resample a ``(n_positions, k)`` count table ``n_boot`` times.

  Resampling ``n`` items with replacement from an empirical ``k``-category sample and
  recounting is exactly a ``Multinomial(n, p_hat)`` draw, so this reproduces a per-position
  sequence bootstrap without needing the underlying per-sequence draws.
  """
  counts_arr = np.asarray(counts, dtype=np.float64)
  if counts_arr.ndim != 2:
    msg = "_bootstrap_counts expects a (n_positions, k) count table"
    raise ValueError(msg)
  totals = counts_arr.sum(axis=-1)
  probabilities = counts_arr / totals[..., None]
  n_positions, k = counts_arr.shape
  totals_batched = np.broadcast_to(totals.astype(np.int64), (n_boot, n_positions))
  probabilities_batched = np.broadcast_to(probabilities, (n_boot, n_positions, k))
  return rng.multinomial(totals_batched, probabilities_batched).astype(np.float64)


def excess_js_upper(
  a1: np.ndarray,
  a2: np.ndarray,
  r1: np.ndarray,
  r2: np.ndarray,
  n_boot: int,
  alpha: float,
  rng: np.random.Generator,
) -> float:
  """Compute a one-sided bootstrap upper bound on ``excess_js(a1, a2, r1, r2)``.

  Each arm's ``(n_positions, k)`` count table is multinomial-resampled ``n_boot`` times at its
  own observed per-position totals (see ``_bootstrap_counts``), ``excess_js`` is recomputed on
  every resample, and the ``(1 - alpha)`` quantile of the resulting distribution is returned
  as the one-sided upper confidence bound.
  """
  if n_boot <= 0:
    msg = "n_boot must be positive"
    raise ValueError(msg)
  if not 0.0 < alpha < 1.0:
    msg = "alpha must be in (0, 1)"
    raise ValueError(msg)

  boots = [_bootstrap_counts(arr, n_boot, rng) for arr in (a1, a2, r1, r2)]
  values = excess_js(*boots)
  return float(np.quantile(np.asarray(values), 1.0 - alpha))


def iut_equivalent(lane_results: Sequence[bool]) -> bool:
  """Combine per-lane equivalence decisions via intersection-union: True iff every lane passes.

  No multiplicity adjustment is applied (matches the spec: "Equivalence iff every lane passes
  (IUT; no multiplicity adjustment)"). An empty sequence of lanes is vacuously equivalent.
  """
  return all(bool(passed) for passed in lane_results)


def tost_mean_diff(
  x: np.ndarray,
  y: np.ndarray,
  delta: float,
  alpha: float = 0.05,
) -> tuple[bool, float, float]:
  """Two one-sided tests (TOST) for equivalence of ``mean(x) - mean(y)`` within ``+/- delta``.

  Uses Welch's t-test (unequal variance, unpaired) for each one-sided hypothesis:
  ``H0_lower: diff <= -delta`` vs ``H0_upper: diff >= delta``. Equivalence is declared iff
  BOTH null hypotheses are rejected at level ``alpha``.

  Returns:
    ``(passed, p_lower, p_upper)``.
  """
  if delta <= 0.0:
    msg = "delta must be positive"
    raise ValueError(msg)
  if not 0.0 < alpha < 1.0:
    msg = "alpha must be in (0, 1)"
    raise ValueError(msg)

  x_arr = np.asarray(x, dtype=np.float64)
  y_arr = np.asarray(y, dtype=np.float64)
  n_x, n_y = x_arr.size, y_arr.size
  if n_x < 2 or n_y < 2:
    msg = "tost_mean_diff needs at least 2 observations per arm"
    raise ValueError(msg)

  diff = float(x_arr.mean() - y_arr.mean())
  var_x, var_y = float(x_arr.var(ddof=1)), float(y_arr.var(ddof=1))
  se = math.sqrt(var_x / n_x + var_y / n_y)
  if se == 0.0:
    passed = -delta < diff < delta
    return passed, (0.0 if passed else 1.0), (0.0 if passed else 1.0)

  df = (var_x / n_x + var_y / n_y) ** 2 / (
    (var_x / n_x) ** 2 / (n_x - 1) + (var_y / n_y) ** 2 / (n_y - 1)
  )

  # H0_lower: diff <= -delta, Ha_lower: diff > -delta -- reject for large diff.
  t_lower = (diff - (-delta)) / se
  p_lower = float(1.0 - stats.t.cdf(t_lower, df))
  # H0_upper: diff >= delta, Ha_upper: diff < delta -- reject for small diff.
  t_upper = (diff - delta) / se
  p_upper = float(stats.t.cdf(t_upper, df))

  passed = p_lower < alpha and p_upper < alpha
  return passed, p_lower, p_upper


def required_n(sigma: float, delta: float, alpha: float = 0.05, power: float = 0.9) -> int:
  """Compute the per-arm TOST sample size ceil(2 * (z_{1-alpha} + z_{1-beta/2})^2 * sigma^2 / delta^2).

  ``beta = 1 - power`` is the target Type-II error rate.
  """
  if sigma <= 0.0:
    msg = "sigma must be positive"
    raise ValueError(msg)
  if delta <= 0.0:
    msg = "delta must be positive"
    raise ValueError(msg)
  if not 0.0 < alpha < 1.0:
    msg = "alpha must be in (0, 1)"
    raise ValueError(msg)
  if not 0.0 < power < 1.0:
    msg = "power must be in (0, 1)"
    raise ValueError(msg)

  beta = 1.0 - power
  z_alpha = float(stats.norm.ppf(1.0 - alpha))
  z_beta = float(stats.norm.ppf(1.0 - beta / 2.0))
  n = 2.0 * (z_alpha + z_beta) ** 2 * sigma**2 / delta**2
  return math.ceil(n)
