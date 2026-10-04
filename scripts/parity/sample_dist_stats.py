"""Distributional-protocol statistics for Potts and LASEr sample parity.

Implements the statistics in ``.praxia/docs/specs/260929_pottsmpnn-lasermpnn-xtrax-composition.md``
from ``**Distributional protocol.**`` through ``**Shim check**``. Inputs are integer
sample arrays. No sampling, models, or I/O.

Plug-in total variation is biased up at finite n. ``Δ_s`` subtracts two equal-n
plug-ins against the same reference, which cancels that bias to first order.
Bootstrap intervals resample sequences (rows) inside each run. Confirmatory
``Δ_s`` draws the U1 rows once and reuses that draw in both terms.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Spec defaults and cutoffs. Not free parameters.
DEFAULT_N_BOOT: int = 2000
DEFAULT_DELTA: float = 0.02
DELTA_LO: float = 0.01
DELTA_HI: float = 0.05
DELTA_CHI: float = 0.05
CHI_BINS: int = 36

# Percentile bootstrap. Two-sided 90% CI is the 5th and 95th percentiles
# (5% in each tail). The 95% one-sided upper bound is that same 95th percentile.
# q̂ is the 97.5th percentile of the per-structure null replicates.
_Q_90_LO: float = 0.05
_Q_90_HI: float = 0.95
_Q_95_UPPER: float = 0.95
_Q_97_5: float = 0.975

Verdict = Literal["pass", "fail", "inconclusive"]
PilotStatus = Literal["ok", "escalate", "instrument_invalid"]


@dataclass(frozen=True)
class DeltaBootstrap:
  """Bootstrap replicates of one structure's ``Δ``.

  ``indices_u1`` is the single U1 row resample. Both total-variation terms use it.
  ``indices_left`` resamples the run in the first term (A, U3, or C_m) and
  ``indices_right`` resamples U2. Draws are in that order.
  """

  delta: NDArray[np.float64]
  indices_left: NDArray[np.int64]
  indices_u1: NDArray[np.int64]
  indices_right: NDArray[np.int64]


@dataclass(frozen=True)
class CellGrade:
  """Confirmatory grade for one (condition, temperature) cell."""

  verdict: Verdict
  mean_delta: float
  upper_95: float
  lower_90: float
  max_delta: float
  delta: float


@dataclass(frozen=True)
class SidecarGrade:
  """Sidecar rollup across (condition, temperature) cells."""

  verdict: Verdict


@dataclass(frozen=True)
class PilotDerivation:
  """Pilot ``δ``, half-width, null percentile, and negative-control temperature factor."""

  h_hat: float
  q_hat: float
  delta: float
  delta_neg: tuple[tuple[float, float], ...]
  chosen_m: float | None
  escalate: bool


@dataclass(frozen=True)
class ShimCheck:
  """Shim-distortion check against the pilot half-width."""

  mean_gap: float
  h_hat: float
  instrument_invalid: bool


def tv_per_position(
  x: ArrayLike,
  y: ArrayLike,
  alphabet_size: int,
) -> NDArray[np.float64]:
  """Per-position total variation of two integer sample arrays.

  Spec: per designed position ``i``, ``TV_i(X,Y) = ½ Σ_a |p̂_X(a) − p̂_Y(a)|``.
  ``x`` and ``y`` are shaped ``(n, L)``. Sample sizes may differ; each empirical
  distribution uses its own row count. Returns shape ``(L,)``.
  """
  left = _as_tokens(x)
  right = _as_tokens(y)
  _same_length(left, right)
  px = _frequencies(left, alphabet_size)
  py = _frequencies(right, alphabet_size)
  return 0.5 * np.abs(px - py).sum(axis=1)


def mean_tv(
  x: ArrayLike,
  y: ArrayLike,
  alphabet_size: int,
  mask: ArrayLike,
) -> float:
  """Mean per-position total variation over designed positions.

  Spec: ``D(X,Y) = mean_i TV_i`` over the designed-position mask.
  """
  tv = tv_per_position(x, y, alphabet_size)
  designed = _as_mask(mask, int(tv.shape[0]))
  return float(tv[designed].mean())


def delta_s(
  a: ArrayLike,
  u1: ArrayLike,
  u2: ArrayLike,
  alphabet_size: int,
  mask: ArrayLike,
) -> float:
  """One structure's confirmatory or control contrast.

  Spec: ``Δ_s = D(A,U1) − D(U2,U1)``. Equal ``n`` cancels plug-in bias to first
  order. The same contrast with ``A`` replaced by ``U3`` is the null
  ``Δ⁰_s = D(U3,U1) − D(U2,U1)``, and with ``A`` replaced by ``C_m`` it is the
  summand of ``Δ_neg(m)``.
  """
  left = _as_tokens(a)
  reference = _as_tokens(u1)
  right = _as_tokens(u2)
  _require_equal_n(left, reference, right)
  return mean_tv(left, reference, alphabet_size, mask) - mean_tv(
    right, reference, alphabet_size, mask
  )


def bootstrap_delta_s(
  a: ArrayLike,
  u1: ArrayLike,
  u2: ArrayLike,
  alphabet_size: int,
  mask: ArrayLike,
  *,
  n_boot: int = DEFAULT_N_BOOT,
  seed: int = 0,
) -> DeltaBootstrap:
  """Bootstrap ``Δ_s`` by resampling sequences within each run.

  Spec: bootstrap ``B=2000`` resampling sequences within A, U1, and U2
  independently. Confirmatory rule: the U1 bootstrap resample is shared by both
  terms of ``Δ_s``. ``seed`` builds a ``numpy.random.Generator``; index draws
  are left run, then U1, then U2.
  """
  rng = np.random.default_rng(seed)
  return _bootstrap_delta(a, u1, u2, alphabet_size, mask, n_boot=n_boot, rng=rng)


def confirmatory_verdict(
  upper_95: float,
  lower_90: float,
  max_delta: float,
  delta: float,
) -> Verdict:
  """Grade one pooled cell from its bootstrap bounds and the worst structure.

  Spec: **pass** = 95% one-sided upper bound < ``δ`` AND ``max_s Δ_s < 2δ``;
  **fail** = 90% CI lower bound > ``δ``; else **inconclusive**.
  """
  if upper_95 < delta and max_delta < 2.0 * delta:
    return "pass"
  if lower_90 > delta:
    return "fail"
  return "inconclusive"


def grade_condition(
  a: Sequence[ArrayLike],
  u1: Sequence[ArrayLike],
  u2: Sequence[ArrayLike],
  masks: Sequence[ArrayLike],
  alphabet_size: int,
  *,
  delta: float = DEFAULT_DELTA,
  n_boot: int = DEFAULT_N_BOOT,
  seed: int = 0,
) -> CellGrade:
  """Grade one (condition, temperature) across structures.

  Spec: pooled ``Δ̄ = mean_s Δ_s``, with the pass/fail rule in
  ``confirmatory_verdict``. Structures are a fixed set. Each structure's
  sequences are resampled independently, and that structure's U1 draw is shared
  by both terms. Structures are visited in list order under one generator.
  """
  _matching_counts("grade_condition", a, u1, u2, masks)
  rng = np.random.default_rng(seed)
  points: list[float] = []
  replicates: list[NDArray[np.float64]] = []
  for left, reference, right, mask in zip(a, u1, u2, masks, strict=True):
    points.append(delta_s(left, reference, right, alphabet_size, mask))
    draw = _bootstrap_delta(
      left,
      reference,
      right,
      alphabet_size,
      mask,
      n_boot=n_boot,
      rng=rng,
    )
    replicates.append(draw.delta)
  point = np.asarray(points, dtype=np.float64)
  pooled = np.stack(replicates, axis=0).mean(axis=0)
  upper_95 = _quantile(pooled, _Q_95_UPPER)
  lower_90 = _quantile(pooled, _Q_90_LO)
  mean_delta = float(point.mean())
  max_delta = float(point.max())
  return CellGrade(
    verdict=confirmatory_verdict(upper_95, lower_90, max_delta, delta),
    mean_delta=mean_delta,
    upper_95=upper_95,
    lower_90=lower_90,
    max_delta=max_delta,
    delta=delta,
  )


def grade_sidecar(cells: Sequence[CellGrade]) -> SidecarGrade:
  """Roll cell grades up to the sidecar.

  Spec: sidecar pass iff all (c, T) pass; fail if any fail; else inconclusive.
  """
  if len(cells) == 0:
    raise ValueError("sidecar grade requires at least one (condition, T) cell")
  verdicts = [cell.verdict for cell in cells]
  if any(verdict == "fail" for verdict in verdicts):
    return SidecarGrade("fail")
  if all(verdict == "pass" for verdict in verdicts):
    return SidecarGrade("pass")
  return SidecarGrade("inconclusive")


def derive_pilot(
  u1: Sequence[ArrayLike],
  u2: Sequence[ArrayLike],
  u3: Sequence[ArrayLike],
  masks: Sequence[ArrayLike],
  controls: Sequence[Mapping[float, ArrayLike]],
  alphabet_size: int,
  *,
  n_boot: int = DEFAULT_N_BOOT,
  seed: int = 0,
) -> PilotDerivation:
  """Derive ``δ`` and the smallest negative-control factor ``m``.

  Spec: ``Δ⁰_s = D(U3,U1) − D(U2,U1)``; ``ĥ`` is the half-width of the two-sided
  90% bootstrap CI (``B=2000``) of ``mean_s Δ⁰_s`` (unscaled); ``q̂_s`` is the
  max over pilot structures of the 97.5th bootstrap percentile of ``Δ⁰_s``;
  ``δ = clip(max(2ĥ, q̂_s/2), 0.01, 0.05)``;
  ``Δ_neg(m) = mean_s[D(C_m,U1) − D(U2,U1)]``; choose the smallest ``m`` with
  ``Δ_neg(m) ≥ δ + 2ĥ``. Escalation signal when ``max(2ĥ, q̂_s/2) > 0.05`` or
  no ``m`` qualifies. The U1 resample is shared inside each null replicate.
  """
  _matching_counts("derive_pilot", u1, u2, u3, masks, controls)
  rng = np.random.default_rng(seed)
  replicates: list[NDArray[np.float64]] = []
  for reference, unshimmed, null, mask in zip(u1, u2, u3, masks, strict=True):
    draw = _bootstrap_delta(
      null,
      reference,
      unshimmed,
      alphabet_size,
      mask,
      n_boot=n_boot,
      rng=rng,
    )
    replicates.append(draw.delta)
  stack = np.stack(replicates, axis=0)
  pooled = stack.mean(axis=0)
  h_hat = 0.5 * (_quantile(pooled, _Q_90_HI) - _quantile(pooled, _Q_90_LO))
  q_hat = max(_quantile(row, _Q_97_5) for row in stack)
  unclipped = max(2.0 * h_hat, q_hat / 2.0)
  delta = float(min(max(unclipped, DELTA_LO), DELTA_HI))
  ms = _candidate_ms(controls)
  delta_neg: list[tuple[float, float]] = []
  for m in ms:
    per_structure = [
      delta_s(table[m], reference, unshimmed, alphabet_size, mask)
      for table, reference, unshimmed, mask in zip(controls, u1, u2, masks, strict=True)
    ]
    delta_neg.append((m, float(np.mean(np.asarray(per_structure, dtype=np.float64)))))
  threshold = delta + 2.0 * h_hat
  chosen_m: float | None = None
  for m, value in delta_neg:
    if value >= threshold:
      chosen_m = m
      break
  escalate = unclipped > DELTA_HI or chosen_m is None
  return PilotDerivation(
    h_hat=h_hat,
    q_hat=q_hat,
    delta=delta,
    delta_neg=tuple(delta_neg),
    chosen_m=chosen_m,
    escalate=escalate,
  )


def resolve_pilot(result: PilotDerivation, *, repilot: bool) -> PilotStatus:
  """Map a pilot derivation to ok, escalate, or ``instrument_invalid``.

  Spec: if ``max(2ĥ, q̂_s/2) > 0.05`` or no ``m`` qualifies, re-pilot once at
  ``n=4000``; still failing yields ``instrument_invalid``. ``repilot=True`` means
  this derivation is already that second pilot. This function does not sample.
  """
  if not result.escalate:
    return "ok"
  if repilot:
    return "instrument_invalid"
  return "escalate"


def shim_check(
  u1: Sequence[ArrayLike],
  u2: Sequence[ArrayLike],
  u3: Sequence[ArrayLike],
  masks: Sequence[ArrayLike],
  alphabet_size: int,
  h_hat: float,
) -> ShimCheck:
  """Flag a shim that moves upstream samples.

  Spec: every (c, T), ``mean_s[D(U2,U1) − D(U3,U1)] > ĥ`` yields
  ``instrument_invalid`` (shim distorts sampling). ``ĥ`` is the value committed
  from the pilot, not recomputed here.
  """
  _matching_counts("shim_check", u1, u2, u3, masks)
  gaps = [
    mean_tv(unshimmed, reference, alphabet_size, mask)
    - mean_tv(null, reference, alphabet_size, mask)
    for reference, unshimmed, null, mask in zip(u1, u2, u3, masks, strict=True)
  ]
  mean_gap = float(np.mean(np.asarray(gaps, dtype=np.float64)))
  return ShimCheck(
    mean_gap=mean_gap,
    h_hat=h_hat,
    instrument_invalid=mean_gap > h_hat,
  )


def tv_chi1_per_position(
  aa_x: ArrayLike,
  chi_x: ArrayLike,
  aa_y: ArrayLike,
  chi_y: ArrayLike,
  modal_aa: ArrayLike,
  has_chi1: ArrayLike,
) -> NDArray[np.float64]:
  """LASEr χ1 total variation on 36 bins of 10 degrees.

  Spec: positions where upstream's modal amino acid has χ1, restricted to
  samples with that amino acid, TV over 36×10° bins. ``has_chi1[aa]`` is the
  caller-supplied table; this module does not hardcode which residues have χ1.
  ``chi_*`` are integer bin indices in ``0 .. 35``. Positions that fail the
  modal-χ1 screen, and positions where either run has no sample of the modal
  amino acid, are NaN.
  """
  aa_left, chi_left = _as_chi_pair(aa_x, chi_x)
  aa_right, chi_right = _as_chi_pair(aa_y, chi_y)
  _same_length(aa_left, aa_right)
  length = int(aa_left.shape[1])
  modal = _as_modal(modal_aa, length)
  table = _as_chi_table(has_chi1)
  if int(modal.max()) >= int(table.shape[0]):
    raise ValueError("modal amino-acid id is outside the has_chi1 table")
  out = np.full(length, np.nan, dtype=np.float64)
  for position in range(length):
    amino = int(modal[position])
    if not bool(table[amino]):
      continue
    keep_left = aa_left[:, position] == amino
    keep_right = aa_right[:, position] == amino
    if not bool(keep_left.any()) or not bool(keep_right.any()):
      continue
    bins_left = chi_left[keep_left, position]
    bins_right = chi_right[keep_right, position]
    _require_bins(bins_left)
    _require_bins(bins_right)
    out[position] = float(tv_per_position(bins_left[:, None], bins_right[:, None], CHI_BINS)[0])
  return out


def mean_tv_chi1(
  aa_x: ArrayLike,
  chi_x: ArrayLike,
  aa_y: ArrayLike,
  chi_y: ArrayLike,
  modal_aa: ArrayLike,
  has_chi1: ArrayLike,
  mask: ArrayLike | None = None,
) -> float:
  """Mean χ1 total variation over positions that pass the χ1 screen.

  Spec default for this distance is ``δ_χ = 0.05`` (``DELTA_CHI``). ``mask`` is
  an optional designed-position mask applied on top of the χ1 screen.
  """
  tv = tv_chi1_per_position(aa_x, chi_x, aa_y, chi_y, modal_aa, has_chi1)
  valid = np.isfinite(tv)
  if mask is not None:
    designed = np.asarray(mask, dtype=np.bool_)
    if designed.shape != valid.shape:
      raise ValueError("mask must have shape (L,)")
    valid = valid & designed
  if not bool(valid.any()):
    raise ValueError("no positions contribute to chi1 total variation")
  return float(tv[valid].mean())


def _bootstrap_delta(
  left: ArrayLike,
  u1: ArrayLike,
  right: ArrayLike,
  alphabet_size: int,
  mask: ArrayLike,
  *,
  n_boot: int,
  rng: np.random.Generator,
) -> DeltaBootstrap:
  if n_boot < 1:
    raise ValueError("n_boot must be positive")
  samples_left = _as_tokens(left)
  reference = _as_tokens(u1)
  samples_right = _as_tokens(right)
  _require_equal_n(samples_left, reference, samples_right)
  _same_length(samples_left, reference)
  _same_length(samples_right, reference)
  designed = _as_mask(mask, int(reference.shape[1]))
  _frequencies(samples_left, alphabet_size)
  _frequencies(reference, alphabet_size)
  _frequencies(samples_right, alphabet_size)
  idx_left = _draw_rows(int(samples_left.shape[0]), n_boot, rng)
  idx_u1 = _draw_rows(int(reference.shape[0]), n_boot, rng)
  idx_right = _draw_rows(int(samples_right.shape[0]), n_boot, rng)
  positions = np.flatnonzero(designed)
  total_left = np.zeros(n_boot, dtype=np.float64)
  total_right = np.zeros(n_boot, dtype=np.float64)
  for position in positions:
    # One U1 frequency per replicate, shared by both terms.
    freq_u1 = _resampled_freq_1d(reference[:, position], idx_u1, alphabet_size)
    freq_left = _resampled_freq_1d(samples_left[:, position], idx_left, alphabet_size)
    freq_right = _resampled_freq_1d(samples_right[:, position], idx_right, alphabet_size)
    total_left += _tv_batch(freq_left, freq_u1)
    total_right += _tv_batch(freq_right, freq_u1)
  scale = float(positions.shape[0])
  return DeltaBootstrap(
    delta=(total_left - total_right) / scale,
    indices_left=idx_left,
    indices_u1=idx_u1,
    indices_right=idx_right,
  )


def _draw_rows(n_rows: int, n_boot: int, rng: np.random.Generator) -> NDArray[np.int64]:
  return rng.integers(0, n_rows, size=(n_boot, n_rows), dtype=np.int64)


def _resampled_freq_1d(
  tokens: NDArray[np.int64],
  indices: NDArray[np.int64],
  alphabet_size: int,
) -> NDArray[np.float64]:
  drawn = tokens[indices]
  n_boot, n_rows = drawn.shape
  offsets = (np.arange(n_boot, dtype=np.int64) * alphabet_size)[:, None]
  flat = np.reshape(drawn + offsets, -1)
  counts = np.bincount(flat, minlength=n_boot * alphabet_size)
  freq = counts.reshape(n_boot, alphabet_size).astype(np.float64)
  return freq / float(n_rows)


def _tv_batch(px: NDArray[np.float64], py: NDArray[np.float64]) -> NDArray[np.float64]:
  return 0.5 * np.abs(px - py).sum(axis=1)


def _frequencies(samples: NDArray[np.int64], alphabet_size: int) -> NDArray[np.float64]:
  if alphabet_size < 1:
    raise ValueError("alphabet_size must be positive")
  n_rows, length = samples.shape
  if n_rows == 0 or length == 0:
    raise ValueError("sample array is empty")
  if int(samples.min()) < 0 or int(samples.max()) >= alphabet_size:
    raise ValueError("tokens must lie in [0, alphabet_size)")
  offsets = (np.arange(length, dtype=np.int64) * alphabet_size)[None, :]
  flat = np.reshape(samples + offsets, -1)
  counts = np.bincount(flat, minlength=length * alphabet_size)
  freq = counts.reshape(length, alphabet_size).astype(np.float64)
  return freq / float(n_rows)


def _as_tokens(samples: ArrayLike) -> NDArray[np.int64]:
  array = np.asarray(samples)
  if array.ndim != 2:
    raise ValueError("sample arrays must have shape (n, L)")
  if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
    raise ValueError("sample arrays must be integer tokens")
  return array.astype(np.int64, copy=False)


def _as_mask(mask: ArrayLike, length: int) -> NDArray[np.bool_]:
  designed = np.asarray(mask, dtype=np.bool_)
  if designed.shape != (length,):
    raise ValueError("designed-position mask must have shape (L,)")
  if not bool(designed.any()):
    raise ValueError("designed-position mask selects no positions")
  return designed


def _as_modal(modal_aa: ArrayLike, length: int) -> NDArray[np.int64]:
  modal = np.asarray(modal_aa)
  if modal.ndim != 1 or int(modal.shape[0]) != length:
    raise ValueError("modal_aa must have shape (L,)")
  if modal.dtype == np.bool_ or not np.issubdtype(modal.dtype, np.integer):
    raise ValueError("modal_aa must be integer amino-acid ids")
  if int(modal.min()) < 0:
    raise ValueError("modal_aa ids must be non-negative")
  return modal.astype(np.int64, copy=False)


def _as_chi_table(has_chi1: ArrayLike) -> NDArray[np.bool_]:
  table = np.asarray(has_chi1, dtype=np.bool_)
  if table.ndim != 1 or int(table.shape[0]) == 0:
    raise ValueError("has_chi1 must be a non-empty boolean table indexed by amino acid")
  return table


def _as_chi_pair(
  amino: ArrayLike,
  chi: ArrayLike,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
  aa = _as_tokens(amino)
  bins = _as_tokens(chi)
  if aa.shape != bins.shape:
    raise ValueError("amino-acid and chi-bin arrays must share shape (n, L)")
  return aa, bins


def _require_bins(bins: NDArray[np.int64]) -> None:
  if int(bins.min()) < 0 or int(bins.max()) >= CHI_BINS:
    raise ValueError(f"chi bins must lie in [0, {CHI_BINS})")


def _same_length(left: NDArray[np.int64], right: NDArray[np.int64]) -> None:
  if int(left.shape[1]) != int(right.shape[1]):
    raise ValueError("sample arrays must share L")


def _require_equal_n(
  left: NDArray[np.int64],
  reference: NDArray[np.int64],
  right: NDArray[np.int64],
) -> None:
  n_left = int(left.shape[0])
  n_reference = int(reference.shape[0])
  n_right = int(right.shape[0])
  if n_left != n_reference or n_right != n_reference:
    raise ValueError("Δ requires equal n so plug-in bias cancels to first order")


def _matching_counts(label: str, *groups: Sequence[object]) -> None:
  counts = {len(group) for group in groups}
  if len(counts) != 1:
    raise ValueError(f"{label} sequences must cover the same structures")
  if counts.pop() == 0:
    raise ValueError(f"{label} requires at least one structure")


def _candidate_ms(controls: Sequence[Mapping[float, ArrayLike]]) -> list[float]:
  key_sets = [set(map(float, table)) for table in controls]
  if any(keys != key_sets[0] for keys in key_sets):
    raise ValueError("every pilot structure must use the same candidate m list")
  return sorted(key_sets[0])


def _quantile(values: NDArray[np.float64], probability: float) -> float:
  return float(np.quantile(values, probability, method="linear"))
