"""Is GPU-batched LASEr sampling equivalent in chi1, not just in sequence?

WHY THIS IS A SEPARATE QUESTION. Run 04e881a5 established that batching leaves
the SEQUENCE distribution alone (cross 0.0613 against a floor edge of 0.0650).
It said nothing about side-chain torsions: its payload kept only
``chi1_distinct_bins``, a count of how many of the 36 bins were occupied, and
discarded the control arm's chi entirely. "36 bins were produced" is satisfied
by any distribution that is not degenerate, including a badly wrong one. So the
chi1 half of the sampler is, as of that run, UNMEASURED rather than confirmed.

WHAT IT COMPARES. The same already-completed CPU reference unit (U2, unshimmed,
4jnj-1_prot, T=1.0, min_p=0, n=1000), which stores ``chi1_degrees`` alongside
its sequences, against GPU-batched draws from ``_sample_batched`` -- imported
from laser_batch_equivalence rather than copied, so this is literally the code
path that run graded, not a lookalike.

THE DISTANCE IS REUSED, NOT REINVENTED. ``stats.mean_tv_chi1`` is the confirm's
own chi1 estimator: positions where the upstream modal amino acid has chi1,
restricted to the samples that actually carry that amino acid, total variation
over 36 bins of 10 degrees. Conditioning on the modal amino acid is what keeps
this a question about TORSIONS rather than about composition -- without it, an
arm that simply drew more glycine would score as a chi1 difference.

ONE DEVIATION FROM THE CONFIRM, STATED ON PURPOSE. The confirm screens on U1
(its shimmed upstream anchor). There is no U1 here, so the screen is taken from
the CPU reference, which is the upstream arm in this comparison. Same role,
different arm, and it is named here rather than left for a reader to infer.

THE TRAP THIS DESIGN HAS AND THE SEQUENCE VERSION DID NOT. Total variation is
biased upward at small n, which already falsified the sequence instrument once
(see ``_group_size`` in laser_batch_equivalence). Equalising the number of
DESIGNS per arm does not equalise n here: ``mean_tv_chi1`` keeps only the
samples whose amino acid matches the modal one, so a position's effective n is
its modal-amino-acid COUNT, which differs per position and per arm. Two arms
with identical torsion distributions but slightly different composition would
therefore show a nonzero chi1 TV, and it would look like a torsion difference.

That is handled three ways rather than assumed away:
  1. the floor is measured from reference-against-itself at the same group
     size, so ordinary sampling noise is priced in;
  2. ``count_matched_cross`` recomputes the cross after truncating both sides
     to a common per-position count, which removes the asymmetry entirely --
     reported as a ROBUSTNESS CHECK, never as the graded number, because
     choosing between two estimators after seeing them is how a result gets
     talked into existence;
  3. ``modal_count_ratio_min`` reports the worst per-position count asymmetry,
     so a reader can see how much room the effect had.

The graded verdict is driven by the reused estimator alone.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve()
_PARITY_DIR = _HERE.parents[1] / "parity"
if str(_PARITY_DIR) not in sys.path:
  sys.path.insert(0, str(_PARITY_DIR))
if str(_HERE.parent) not in sys.path:
  sys.path.insert(0, str(_HERE.parent))

import sample_dist_stats as stats  # noqa: E402

_LOG = logging.getLogger("laser_batch_chi1_equivalence")

#: Resamples behind both the floor and the cross. Matches the sequence run.
_N_RESAMPLE = 200
#: The floor edge: a cross at or below this sits inside the reference's own
#: noise. Pre-registered, and never widened to admit a result.
_FLOOR_QUANTILE = 0.975
#: Control multiplier on temperature. 1.5x already fired at 2.5x the floor edge
#: on sequences, so it is a control with known power rather than a hope.
_CONTROL_M = 1.5


def _chi1_bins(degrees: np.ndarray) -> np.ndarray:
  """Degrees -> ``CHI_BINS`` equal bins, mirroring the pilot exactly.

  Non-finite entries map to ``CHI_BINS`` itself, an out-of-range sentinel. They
  must never reach the estimator: a residue with no chi1 has no torsion to
  compare, and ``stats`` rejects an out-of-range bin rather than quietly
  folding it into bin 0.
  """
  values = np.asarray(degrees, dtype=np.float64)
  bins = np.full(values.shape, stats.CHI_BINS, dtype=np.int64)
  finite = np.isfinite(values)
  if bool(finite.any()):
    width = 360.0 / float(stats.CHI_BINS)
    shifted = np.mod(values[finite] + 180.0, 360.0)
    drawn = np.floor(shifted / width).astype(np.int64)
    bins[finite] = np.clip(drawn, 0, stats.CHI_BINS - 1)
  return bins


def _degrees(rows: list[list[float | None]]) -> np.ndarray:
  """``null`` means the residue has no chi1, which is NaN, not zero."""
  return np.asarray(
    [[np.nan if value is None else float(value) for value in row] for row in rows],
    dtype=np.float64,
  )


def _modal_aa(sequences: np.ndarray, alphabet: int) -> np.ndarray:
  tokens = np.asarray(sequences, dtype=np.int64)
  length = int(tokens.shape[1])
  modal = np.empty(length, dtype=np.int64)
  for position in range(length):
    counts = np.bincount(tokens[:, position], minlength=alphabet)
    modal[position] = int(np.argmax(counts[:alphabet]))
  return modal


def _group_size(*sample_sets: np.ndarray) -> int:
  """The one group size every comparison in a run must use.

  THIS IS LOAD-BEARING. Comparing a floor measured at one group size against a
  cross measured at another manufactures a difference out of sample size alone.
  The sequence instrument produced exactly that false verdict before its own
  ``_group_size`` existed; this is the same guard, carried over deliberately.
  """
  return min(len(s) for s in sample_sets) // 2


def _chi1_tv(
  aa_left: np.ndarray,
  chi_left: np.ndarray,
  aa_right: np.ndarray,
  chi_right: np.ndarray,
  modal: np.ndarray,
  table: np.ndarray,
  mask: np.ndarray,
) -> float:
  """The reused estimator. No local variant, no tuning."""
  return float(
    stats.mean_tv_chi1(aa_left, chi_left, aa_right, chi_right, modal, table, mask),
  )


def _count_matched_chi1_tv(
  aa_left: np.ndarray,
  chi_left: np.ndarray,
  aa_right: np.ndarray,
  chi_right: np.ndarray,
  modal: np.ndarray,
  table: np.ndarray,
  mask: np.ndarray,
  rng: np.random.Generator,
) -> tuple[float, float]:
  """Robustness check: equalise the per-position count before comparing.

  Returns ``(mean_tv, worst_count_ratio)``. Reported, never graded -- see the
  module docstring for why the graded number must be fixed in advance.
  """
  length = int(aa_left.shape[1])
  per_position: list[float] = []
  worst = 1.0
  for position in range(length):
    amino = int(modal[position])
    if not bool(table[amino]) or not bool(mask[position]):
      continue
    keep_left = np.flatnonzero(aa_left[:, position] == amino)
    keep_right = np.flatnonzero(aa_right[:, position] == amino)
    if keep_left.size == 0 or keep_right.size == 0:
      continue
    worst = min(worst, min(keep_left.size, keep_right.size) / max(keep_left.size, keep_right.size))
    take = min(keep_left.size, keep_right.size)
    left = chi_left[rng.permutation(keep_left)[:take], position]
    right = chi_right[rng.permutation(keep_right)[:take], position]
    per_position.append(
      float(stats.tv_per_position(left[:, None], right[:, None], stats.CHI_BINS)[0]),
    )
  if not per_position:
    return float("nan"), worst
  return float(np.mean(per_position)), worst


def _within_tv(
  aa: np.ndarray,
  chi: np.ndarray,
  modal: np.ndarray,
  table: np.ndarray,
  mask: np.ndarray,
  half: int,
  rng: np.random.Generator,
) -> list[float]:
  """The floor: the reference against ITSELF, at the shared group size."""
  out: list[float] = []
  for _ in range(_N_RESAMPLE):
    order = rng.permutation(len(aa))
    left, right = order[:half], order[half : 2 * half]
    out.append(_chi1_tv(aa[left], chi[left], aa[right], chi[right], modal, table, mask))
  return out


def _cross_tv(
  aa_a: np.ndarray,
  chi_a: np.ndarray,
  aa_b: np.ndarray,
  chi_b: np.ndarray,
  modal: np.ndarray,
  table: np.ndarray,
  mask: np.ndarray,
  half: int,
  rng: np.random.Generator,
) -> list[float]:
  """Two different arms, at the SAME group size the floor used."""
  out: list[float] = []
  for _ in range(_N_RESAMPLE):
    left = rng.permutation(len(aa_a))[:half]
    right = rng.permutation(len(aa_b))[:half]
    out.append(_chi1_tv(aa_a[left], chi_a[left], aa_b[right], chi_b[right], modal, table, mask))
  return out


def _grade(payload: dict[str, Any]) -> str:
  if not payload["self_test_passed"]:
    return "instrument_unverified"
  if payload.get("refusal"):
    return str(payload["refusal"])
  if not payload["control_fired"]:
    return "control_did_not_fire"
  return "chi1_batched_equivalent" if payload["within_floor"] else "chi1_batched_not_equivalent"


def _self_test() -> dict[str, Any]:
  """Synthetic ground truth. A positive control alone proves nothing."""
  rng = np.random.default_rng(20261006)
  failed: list[str] = []
  checks = 0

  length, n = 8, 400
  alphabet = 21
  table = np.zeros(alphabet, dtype=np.bool_)
  table[3] = True  # exactly one amino acid carries chi1 in the synthetic world
  mask = np.ones(length, dtype=np.bool_)
  modal = np.full(length, 3, dtype=np.int64)

  aa = np.full((n, length), 3, dtype=np.int64)
  chi_a = rng.integers(0, stats.CHI_BINS, size=(n, length))
  chi_b = rng.integers(0, stats.CHI_BINS, size=(n, length))
  # A genuinely different torsion distribution: concentrated on few bins.
  chi_far = rng.integers(0, 4, size=(n, length))

  half = _group_size(aa, aa)
  floor = _within_tv(aa, chi_a, modal, table, mask, half, rng)
  edge = float(np.quantile(floor, _FLOOR_QUANTILE))

  # 1. same distribution must sit INSIDE the floor
  checks += 1
  same = float(statistics.median(_cross_tv(aa, chi_a, aa, chi_b, modal, table, mask, half, rng)))
  if same > edge:
    failed.append(f"same_distribution_above_floor({same:.4f}>{edge:.4f})")

  # 2. a different distribution must sit ABOVE it, or the test cannot detect
  #    the very thing it exists to detect
  checks += 1
  diff = float(statistics.median(_cross_tv(aa, chi_a, aa, chi_far, modal, table, mask, half, rng)))
  if diff <= edge:
    failed.append(f"different_distribution_inside_floor({diff:.4f}<={edge:.4f})")

  # 3. the small-n bias is real and this is the demonstration of it
  checks += 1
  small = float(np.quantile(_within_tv(aa, chi_a, modal, table, mask, 25, rng), _FLOOR_QUANTILE))
  if small <= edge * 1.5:
    failed.append(f"small_n_not_biased_upward({small:.4f} vs {edge:.4f})")

  # 4. _group_size follows the SMALLEST set, never the first
  checks += 1
  if _group_size(np.zeros((100, 2)), np.zeros((1000, 2))) != 50:
    failed.append("group_size_not_minimum")

  # 5. the modal screen must EXCLUDE an amino acid with no chi1. Flip the
  #    table off and the estimator should have nothing left to average.
  checks += 1
  empty_table = np.zeros(alphabet, dtype=np.bool_)
  try:
    _chi1_tv(aa, chi_a, aa, chi_b, modal, empty_table, mask)
  except ValueError:
    pass
  else:
    failed.append("screen_admits_residue_without_chi1")

  # 6. a residue whose chi1 is undefined must not be silently binned
  checks += 1
  if int(_chi1_bins(np.array([np.nan]))[0]) != stats.CHI_BINS:
    failed.append("nan_chi1_not_sentinel")

  # 7. binning must be stable at the wrap point, not fold 180 onto 0 twice
  checks += 1
  if int(_chi1_bins(np.array([-180.0]))[0]) != 0 or int(_chi1_bins(np.array([179.9]))[0]) != 35:
    failed.append("bin_edges_wrong")

  # 8. count matching must actually equalise: a deliberately lopsided pair
  #    must report a worst ratio well below 1
  checks += 1
  lopsided = aa.copy()
  lopsided[: n // 2, 0] = 5  # half the samples stop carrying the modal acid
  _matched, worst = _count_matched_chi1_tv(
    lopsided, chi_a, aa, chi_b, modal, table, mask, rng,
  )
  if not worst < 0.75:
    failed.append(f"count_matching_did_not_detect_asymmetry(worst={worst:.3f})")

  # 9. grading order: an unverified instrument outranks every other verdict
  checks += 1
  base = {"self_test_passed": True, "refusal": None, "control_fired": True, "within_floor": True}
  if _grade({**base, "self_test_passed": False}) != "instrument_unverified":
    failed.append("unverified_does_not_outrank")
  checks += 1
  if _grade({**base, "refusal": "cuda_unavailable"}) != "cuda_unavailable":
    failed.append("refusal_does_not_outrank")
  checks += 1
  if _grade({**base, "control_fired": False}) != "control_did_not_fire":
    failed.append("dead_control_does_not_outrank")
  checks += 1
  if _grade({**base, "within_floor": False}) != "chi1_batched_not_equivalent":
    failed.append("outside_floor_not_graded_negative")

  return {
    "self_test_passed": not failed,
    "self_test_n_checks": checks,
    "self_test_failed": failed,
    "self_test_floor_edge": round(edge, 6),
    "self_test_same_cross": round(same, 6),
    "self_test_different_cross": round(diff, 6),
  }


def _emit(out: Path, payload: dict[str, Any]) -> None:
  out.parent.mkdir(parents=True, exist_ok=True)
  text = json.dumps(payload, indent=2, sort_keys=True)
  out.write_text(text, encoding="utf-8")
  results = os.environ.get("BTH_RESULTS_PATH")
  if results:
    Path(results).write_text(text, encoding="utf-8")
    _LOG.info("emitted result to BTH_RESULTS_PATH=%s", results)


def main() -> int:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--reference-unit", type=Path, help="completed U2 unit JSON")
  parser.add_argument("--laser-root")
  parser.add_argument("--checkpoint")
  parser.add_argument("--pdb")
  parser.add_argument("--device", default="cuda:2")
  parser.add_argument("--copies", type=int, default=50)
  parser.add_argument("--n", type=int, default=1000)
  parser.add_argument("--min-p", type=float, default=0.0)
  parser.add_argument("--seed", type=int, default=20261006)
  parser.add_argument("--self-test-only", action="store_true")
  args = parser.parse_args()

  base = _self_test()
  _LOG.info(
    "self-test %s (%d checks) floor<=%.4f same=%.4f different=%.4f",
    "passed" if base["self_test_passed"] else f"FAILED {base['self_test_failed']}",
    base["self_test_n_checks"],
    base["self_test_floor_edge"],
    base["self_test_same_cross"],
    base["self_test_different_cross"],
  )
  if args.self_test_only:
    # A smoke run, not a finding: "self_test_only" matches no registered
    # outcome on purpose, so such a run can never be mistaken for a graded one.
    verdict = "self_test_only" if base["self_test_passed"] else "instrument_unverified"
    _emit(args.out, {**base, "refusal": None, "verdict": verdict})
    return 0 if base["self_test_passed"] else 1

  if not args.reference_unit or not args.reference_unit.is_file():
    _emit(args.out, {**base, "refusal": "reference_absent", "verdict": "reference_absent"})
    _LOG.error("reference unit missing: %s", args.reference_unit)
    return 1

  import torch  # noqa: PLC0415

  if not torch.cuda.is_available():
    _emit(args.out, {**base, "refusal": "cuda_unavailable", "verdict": "cuda_unavailable"})
    return 1

  import laser_batch_equivalence as seqrun  # noqa: PLC0415
  from laser_sample_dist_pilot import _chi_table  # noqa: PLC0415

  reference = seqrun._load_reference(args.reference_unit)
  ref_aa = np.asarray(reference["sequences"], dtype=np.int64)
  ref_chi = _chi1_bins(_degrees(reference["chi1_degrees"]))
  mask = np.asarray(reference["mask"], dtype=np.bool_)
  temperature = float(reference["sample_temperature"])
  table, alphabet = _chi_table()
  table = np.asarray(table, dtype=np.bool_)
  modal = _modal_aa(ref_aa, alphabet)

  screened = int(sum(1 for p in range(len(modal)) if table[int(modal[p])] and mask[p]))
  if screened == 0:
    _emit(args.out, {**base, "refusal": "chi_screen_empty", "verdict": "chi_screen_empty"})
    _LOG.error("no position survives the modal-chi1 screen")
    return 1

  gpu_aa, gpu_chi_deg, gpu_info = seqrun._sample_batched(args, temperature, args.n, args.seed)
  ctrl_aa, ctrl_chi_deg, ctrl_info = seqrun._sample_batched(
    args, temperature * _CONTROL_M, args.n, args.seed + 1,
  )
  gpu_chi = _chi1_bins(gpu_chi_deg[:, :, 0])
  ctrl_chi = _chi1_bins(ctrl_chi_deg[:, :, 0])

  rng = np.random.default_rng(args.seed)
  half = _group_size(ref_aa, gpu_aa, ctrl_aa)
  floor = _within_tv(ref_aa, ref_chi, modal, table, mask, half, rng)
  edge = float(np.quantile(floor, _FLOOR_QUANTILE))
  cross = _cross_tv(gpu_aa, gpu_chi, ref_aa, ref_chi, modal, table, mask, half, rng)
  control = _cross_tv(ctrl_aa, ctrl_chi, ref_aa, ref_chi, modal, table, mask, half, rng)
  cross_median = float(statistics.median(cross))
  control_median = float(statistics.median(control))
  matched, worst_ratio = _count_matched_chi1_tv(
    gpu_aa, gpu_chi, ref_aa, ref_chi, modal, table, mask, rng,
  )

  payload = {
    **base,
    "refusal": None,
    "reference_unit": str(args.reference_unit),
    "structure": reference.get("structure"),
    "condition": reference.get("condition"),
    "temperature": temperature,
    "control_m": _CONTROL_M,
    "n_reference": int(len(ref_aa)),
    "n_gpu": int(len(gpu_aa)),
    "group_size": half,
    "positions_screened": screened,
    "positions_total": int(len(modal)),
    "floor_median": round(float(statistics.median(floor)), 6),
    "floor_edge": round(edge, 6),
    "cross_median": round(cross_median, 6),
    "control_median": round(control_median, 6),
    "count_matched_cross": None if not np.isfinite(matched) else round(float(matched), 6),
    "modal_count_ratio_min": round(float(worst_ratio), 4),
    "within_floor": bool(cross_median <= edge),
    "control_fired": bool(control_median > edge),
    "gpu_per_design_s": gpu_info.get("per_design_s"),
    "control_per_design_s": ctrl_info.get("per_design_s"),
  }
  payload["verdict"] = _grade(payload)
  _emit(args.out, payload)
  _LOG.info(
    "verdict=%s floor<=%.4f cross=%.4f control=%.4f matched=%s worst_ratio=%.3f",
    payload["verdict"],
    edge,
    cross_median,
    control_median,
    payload["count_matched_cross"],
    worst_ratio,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
