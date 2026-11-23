"""Does GPU + batched LASEr sampling draw the SAME distribution as CPU unbatched?

WHY THIS EXISTS. The distributional confirm costs ~4.02 s/design because
``laser_sample_dist_pilot._sample_once`` draws ONE design per forward pass --
it calls ``output_batch_data`` with the default ``num_copies=1`` while upstream
documents ``num_copies > 1`` as its own parallel-design facility
(``run_inference.py:305``) -- and because the oracle venv pins CPU torch. A
throwaway spike measured GPU + ``num_copies=50`` at 0.091 s/design, roughly
30-45x faster.

THAT NUMBER IS WORTHLESS ON ITS OWN. Batching changes which samples a given
seed produces. A sampler that is 40x faster and draws from a DIFFERENT
distribution is not a speedup, it is a defect -- and it is a defect that a
timing run cannot see. This run asks the question the timing run could not.

WHAT IT COMPARES. The already-completed CPU reference unit (U2, unshimmed,
1000 designs) against designs drawn GPU-batched at the SAME temperature,
min_p and structure, graded by the confirm's OWN statistic
(``sample_dist_stats.mean_tv`` and ``mean_tv_chi1``) rather than a reinvented
one, so the answer is in the units the confirm already grades in.

THE FLOOR IS THE WHOLE POINT. Two independent 500-sample draws from the SAME
distribution do NOT give TV = 0; they give the sampling floor at that n. So the
question is never "is the TV small" -- it is "is the cross-TV inside the floor
that the reference shows against ITSELF". The floor is therefore measured here,
from random 500/500 splits of the CPU reference, not assumed.

NEGATIVE CONTROL, REQUIRED TO FIRE. A metric too blunt to separate two
distributions would report "equivalent" for anything, including a broken
sampler. So a deliberately-different GPU batch, drawn at a raised temperature,
must land ABOVE the floor. If it does not, this run reports NO CONCLUSION about
equivalence rather than a reassuring one.

WHAT THIS DELIBERATELY DOES NOT ANSWER. The SHIMMED arm (U1). The shim injects
a fixed uniform stream and asserts it consumed exactly
``_DRAWS_PER_RESIDUE * length`` draws; batching needs a ``(copies, ...)`` stream
and a batch-aware cursor, which is real work and a separate question. Nothing
here licenses running a shimmed arm batched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_PARITY_DIR = Path(__file__).resolve().parents[1] / "parity"
if str(_PARITY_DIR) not in sys.path:
  sys.path.insert(0, str(_PARITY_DIR))

import sample_dist_stats as stats  # noqa: E402

_LOG = logging.getLogger("laser_batch_equivalence")

#: Amino-acid alphabet size the confirm grades sequence TV over.
_ALPHABET = 21
#: Resamples used to turn a single TV number into a distribution with a band.
_N_RESAMPLE = 200
#: The floor's upper edge. Pre-registered BEFORE the run: a cross-TV at or below
#: the 97.5th percentile of the reference-against-itself TV is inside the noise
#: this many samples can resolve, and is not evidence of a different distribution.
_FLOOR_QUANTILE = 0.975


def _median(values: list[float]) -> float:
  return float(statistics.median(values))


def _resampled_tv(
  left: np.ndarray,
  right: np.ndarray,
  mask: np.ndarray,
  rng: np.random.Generator,
  *,
  n_resample: int,
  half: int,
) -> list[float]:
  """TV between random ``half``-sized subsets of two sample sets.

  ``half`` is supplied by the caller, never derived here, so that the floor and
  the cross are forced to the SAME group size. See ``_group_size``.
  """
  out: list[float] = []
  for _ in range(n_resample):
    i = rng.permutation(len(left))[:half]
    j = rng.permutation(len(right))[:half]
    out.append(stats.mean_tv(left[i], right[j], _ALPHABET, mask))
  return out


def _within_tv(
  samples: np.ndarray,
  mask: np.ndarray,
  rng: np.random.Generator,
  *,
  n_resample: int,
  half: int,
) -> list[float]:
  """TV of a sample set against ITSELF, over random disjoint ``half``-sized groups.

  This is the floor. The two groups are disjoint by construction, so they are
  genuinely independent draws from one distribution.
  """
  out: list[float] = []
  for _ in range(n_resample):
    order = rng.permutation(len(samples))
    out.append(
      stats.mean_tv(samples[order[:half]], samples[order[half : 2 * half]], _ALPHABET, mask)
    )
  return out


def _group_size(*sample_sets: np.ndarray) -> int:
  """The one group size every comparison in a run must use.

  THIS IS LOAD-BEARING, and getting it wrong is the failure this run is most
  likely to make. Total variation between two empirical distributions is biased
  UPWARD at small n -- two halves of one distribution look more different the
  fewer samples each holds. So a floor measured at 500-vs-500 and a cross
  measured at 50-vs-50 differ for a reason that has nothing to do with the
  samplers, and the run would report a difference it manufactured itself.
  Deriving the size once, from the SMALLEST set in the comparison, is what stops
  that; a smoke run at n=100 against a 1000-sample reference produced exactly
  that false "not equivalent" before this existed.
  """
  return min(len(s) for s in sample_sets) // 2


def _grade(payload: dict[str, Any]) -> str:
  if not payload["self_test_passed"]:
    return "instrument_unverified"
  if payload.get("refusal"):
    return str(payload["refusal"])
  if not payload["control_fired"]:
    return "control_did_not_fire"
  return "batched_equivalent" if payload["within_floor"] else "batched_not_equivalent"


def _self_test() -> dict[str, Any]:
  """Synthetic ground truth, before any real sample is read.

  The decisive pair is checks 2 and 3: the metric must NOT separate two draws
  from one distribution, and MUST separate two draws from different ones. A
  check that can only pass is not a check.
  """
  failed: list[str] = []
  checks = 0
  rng = np.random.default_rng(0)
  length, n = 40, 1000
  mask = np.ones(length, dtype=bool)

  # One distribution, drawn twice, and a second, clearly different one.
  p_a = rng.dirichlet(np.ones(_ALPHABET), size=length)
  p_b = rng.dirichlet(np.ones(_ALPHABET) * 0.3, size=length)

  def draw(probability: np.ndarray, rows: int, generator: np.random.Generator) -> np.ndarray:
    return np.stack(
      [
        generator.choice(_ALPHABET, size=rows, p=probability[i])
        for i in range(probability.shape[0])
      ],
      axis=1,
    )

  same_1 = draw(p_a, n, rng)
  same_2 = draw(p_a, n, rng)
  different = draw(p_b, n, rng)

  half = _group_size(same_1, same_2, different)
  floor = _within_tv(same_1, mask, np.random.default_rng(1), n_resample=60, half=half)
  edge = float(np.quantile(floor, _FLOOR_QUANTILE))

  # 1. A set against itself is exactly zero -- the metric has no constant offset.
  checks += 1
  if stats.mean_tv(same_1, same_1, _ALPHABET, mask) != 0.0:
    failed.append("self_distance_not_zero")

  # 2. NEGATIVE CONTROL. Two draws from ONE distribution must sit inside the
  #    floor. If this fails the metric false-alarms and would call a correct
  #    batched sampler broken.
  checks += 1
  same_cross = _median(
    _resampled_tv(same_1, same_2, mask, np.random.default_rng(2), n_resample=60, half=half)
  )
  if same_cross > edge:
    failed.append(f"false_alarm_on_identical_distributions:{same_cross:.4f}>{edge:.4f}")

  # 3. POSITIVE CONTROL, the one that proves the instrument fires at all. Two
  #    genuinely different distributions must land ABOVE the floor.
  checks += 1
  diff_cross = _median(
    _resampled_tv(same_1, different, mask, np.random.default_rng(3), n_resample=60, half=half)
  )
  if diff_cross <= edge:
    failed.append(f"blind_to_different_distributions:{diff_cross:.4f}<={edge:.4f}")

  # 3b. THE SIZE-BIAS TRAP, pinned. A smoke run produced a confident, entirely
  #     false "not equivalent" by measuring the floor at 500-vs-500 and the
  #     cross at 50-vs-50. This asserts the bias is real and large, so that any
  #     future change which lets the floor and the cross drift to different
  #     group sizes fails here rather than in a published verdict.
  checks += 1
  small = _median(_within_tv(same_1, mask, np.random.default_rng(4), n_resample=60, half=25))
  large = _median(_within_tv(same_1, mask, np.random.default_rng(5), n_resample=60, half=half))
  if not small > large * 1.5:
    failed.append(f"size_bias_not_demonstrated:small={small:.4f} large={large:.4f}")

  # 3c. _group_size must take the SMALLEST set, not the first or the largest.
  checks += 1
  if _group_size(np.zeros((1000, 4)), np.zeros((100, 4))) != 50:
    failed.append("group_size_does_not_follow_smallest_set")

  # 4. Symmetry: TV is a metric, and an asymmetric implementation would make the
  #    verdict depend on argument order.
  checks += 1
  forward = stats.mean_tv(same_1, different, _ALPHABET, mask)
  backward = stats.mean_tv(different, same_1, _ALPHABET, mask)
  if abs(forward - backward) > 1e-12:
    failed.append(f"asymmetric:{forward}!={backward}")

  # 5. The mask is honoured: a position excluded from the mask must not move the
  #    mean, however wildly it differs.
  checks += 1
  spiked = different.copy()
  spiked[:, 0] = 0
  partial = np.ones(length, dtype=bool)
  partial[0] = False
  if stats.mean_tv(same_1, different, _ALPHABET, partial) != stats.mean_tv(
    same_1, spiked, _ALPHABET, partial
  ):
    failed.append("mask_not_honoured")

  # 6-10. Grader controls: every verdict must be reachable.
  base = {"self_test_passed": True, "refusal": None, "control_fired": True}
  checks += 1
  if _grade({**base, "within_floor": True}) != "batched_equivalent":
    failed.append("grader_misses_equivalent")
  checks += 1
  if _grade({**base, "within_floor": False}) != "batched_not_equivalent":
    failed.append("grader_misses_not_equivalent")
  checks += 1
  if _grade({**base, "control_fired": False, "within_floor": True}) != "control_did_not_fire":
    failed.append("grader_ignores_dead_control")
  checks += 1
  if _grade({**base, "refusal": "cuda_unavailable", "within_floor": True}) != "cuda_unavailable":
    failed.append("refusal_does_not_outrank")
  checks += 1
  if _grade({**base, "self_test_passed": False, "within_floor": True}) != "instrument_unverified":
    failed.append("unverified_does_not_outrank_all")

  return {
    "self_test_passed": not failed,
    "self_test_n_checks": checks,
    "self_test_failed": failed,
    "self_test_floor_edge": round(edge, 6),
    "self_test_same_cross": round(same_cross, 6),
    "self_test_different_cross": round(diff_cross, 6),
  }


def _load_reference(path: Path) -> dict[str, Any]:
  payload = json.loads(path.read_text())["payload"]
  if payload.get("shim"):
    msg = (
      f"{path} is a SHIMMED unit (run={payload.get('run')}). The shim is a "
      "different sampler and is out of scope here; pass the unshimmed U2 unit."
    )
    raise ValueError(msg)
  return payload


def _sample_batched(
  args: argparse.Namespace,
  temperature: float,
  total: int,
  seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
  """Draw ``total`` designs GPU-batched, ``args.copies`` per forward pass."""
  import torch

  sys.path.insert(0, args.laser_root)
  sys.path.insert(0, str(Path(args.laser_root).parent))
  from LASErMPNN.run_inference import (  # noqa: PLC0415
    ProteinComplexData,
    get_protein_hierview,
    load_model_from_parameter_dict,
  )

  model, params = load_model_from_parameter_dict(args.checkpoint, args.device, strict=False)
  model.eval()
  model_params = params["model_params"]

  view = get_protein_hierview(args.pdb)
  data = ProteinComplexData(view, args.pdb, verbose=False)
  batch = data.output_batch_data(fix_beta=False, num_copies=args.copies)
  # to_device BEFORE construct_graphs, matching run_inference.sample_model:524:
  # construct_graphs runs the model's rotamer_builder over these tensors, so
  # moving afterwards leaves them on two devices and raises.
  batch.to_device(model.device)
  batch.construct_graphs(
    model.rotamer_builder,
    model.ligand_featurizer,
    **model_params["graph_structure"],
    protein_training_noise=0.0,
    ligand_training_noise=0.0,
    subgraph_only_dropout_rate=0.0,
    num_adjacent_residues_to_drop=0,
    build_hydrogens=bool(model_params["build_hydrogens"]),
  )

  native_seq = batch.sequence_indices.detach().clone()
  native_chi = batch.chi_angles.detach().clone()
  native_mask = batch.chain_mask.detach().clone()
  kwargs = {
    "sequence_sample_temperature": temperature,
    "chi_angle_sample_temperature": temperature,
    "disabled_residues": ["X"],
    "disable_pbar": True,
    "seq_min_p": args.min_p,
    "chi_min_p": args.min_p,
    "ignore_chain_mask_zeros": False,
    "repack_all": False,
  }

  torch.manual_seed(seed)
  seqs: list[np.ndarray] = []
  chis: list[np.ndarray] = []
  n_calls = (total + args.copies - 1) // args.copies
  started = time.perf_counter()
  with torch.no_grad():
    for _ in range(n_calls):
      batch.sequence_indices = native_seq.clone()
      batch.chi_angles = native_chi.clone()
      batch.chain_mask = native_mask.clone()
      batch.generate_decoding_order(stack_tensors=True)
      sampled = model.sample(batch, **kwargs)
      seqs.append(
        sampled.sampled_sequence_indices.detach().cpu().numpy().reshape(args.copies, -1)
      )
      # sampled_chi_degrees, not sampled_chi_angles: upstream reports DEGREES
      # here, so no radian conversion is applied downstream.
      chis.append(sampled.sampled_chi_degrees.detach().cpu().numpy().reshape(args.copies, -1, 4))
  elapsed = time.perf_counter() - started

  seq = np.concatenate(seqs, axis=0)[:total]
  chi = np.concatenate(chis, axis=0)[:total]
  # The within-batch independence control again, on THIS run's own draws: a
  # batch whose copies were not sampled independently would carry one design's
  # worth of information while counting as `copies`.
  first = seqs[0]
  unique_in_first = len({tuple(int(v) for v in row) for row in first})
  info = {
    "n_calls": n_calls,
    "elapsed_s": round(elapsed, 2),
    "per_design_s": round(elapsed / max(len(seq), 1), 4),
    "unique_in_first_batch": unique_in_first,
    "copies": args.copies,
    "first_batch_independent": unique_in_first == args.copies,
  }
  return seq, chi, info


def _chi_bins(chi: np.ndarray) -> np.ndarray:
  """χ1 to the confirm's 36 bins of 10°, matching sample_dist_stats.CHI_BINS.

  Upstream already reports DEGREES (``sampled_chi_degrees``), so nothing is
  converted. Residues without a χ1 carry NaN; those are dropped rather than
  floored, since ``int(nan)`` is undefined and would silently become a bin.
  """
  degrees = chi[:, :, 0].astype(np.float64)
  finite = degrees[np.isfinite(degrees)]
  if finite.size == 0:
    return np.empty(0, dtype=np.int64)
  width = 360.0 / stats.CHI_BINS
  return (np.floor((finite % 360.0) / width).astype(np.int64)) % stats.CHI_BINS


def _emit(out: Path, payload: dict[str, Any]) -> None:
  out.parent.mkdir(parents=True, exist_ok=True)
  text = json.dumps(payload, indent=2, sort_keys=True)
  out.write_text(text)
  results = os.environ.get("BTH_RESULTS_PATH")
  if results:
    Path(results).write_text(text)
    _LOG.info("emitted result to BTH_RESULTS_PATH=%s", results)


def main() -> int:
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
  parser.add_argument(
    "--control-temperature-scale",
    type=float,
    default=1.5,
    help="the deliberately-different draw the metric must be able to detect",
  )
  parser.add_argument("--self-test-only", action="store_true")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  self_test = _self_test()
  base: dict[str, Any] = {
    **self_test,
    "script_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
    "refusal": None,
    "control_fired": False,
    "within_floor": False,
  }
  if not self_test["self_test_passed"]:
    _emit(args.out, {**base, "verdict": "instrument_unverified"})
    _LOG.error("instrument unverified: %s", self_test["self_test_failed"])
    return 1
  _LOG.info(
    "self-test passed (%d checks): same=%.4f  floor_edge=%.4f  different=%.4f",
    self_test["self_test_n_checks"],
    self_test["self_test_same_cross"],
    self_test["self_test_floor_edge"],
    self_test["self_test_different_cross"],
  )
  if args.self_test_only:
    _emit(args.out, {**base, "verdict": "self_test_only"})
    return 0

  if not args.reference_unit or not args.reference_unit.is_file():
    _emit(args.out, {**base, "refusal": "reference_absent", "verdict": "reference_absent"})
    _LOG.error("reference unit missing: %s", args.reference_unit)
    return 1
  reference = _load_reference(args.reference_unit)

  import torch

  if args.device.startswith("cuda") and not torch.cuda.is_available():
    # A cuda request that silently lands on CPU would compare CPU against CPU
    # and report a reassuring "equivalent" about a question never asked.
    _emit(args.out, {**base, "refusal": "cuda_unavailable", "verdict": "cuda_unavailable"})
    _LOG.error("cuda requested but torch.cuda.is_available() is False")
    return 1

  ref_seq = np.asarray(reference["sequences"], dtype=np.int64)
  mask = np.asarray(reference["mask"], dtype=bool)
  temperature = float(reference["sample_temperature"])

  gpu_seq, gpu_chi, gpu_info = _sample_batched(args, temperature, args.n, args.seed)
  ctrl_seq, _ctrl_chi, ctrl_info = _sample_batched(
    args,
    temperature * args.control_temperature_scale,
    args.n,
    args.seed + 1,
  )

  rng = np.random.default_rng(args.seed)
  # One group size for the floor AND both crosses, derived from the smallest
  # set. See _group_size: this is what keeps the comparison honest.
  half = _group_size(ref_seq, gpu_seq, ctrl_seq)
  floor = _within_tv(ref_seq, mask, rng, n_resample=_N_RESAMPLE, half=half)
  cross = _resampled_tv(ref_seq, gpu_seq, mask, rng, n_resample=_N_RESAMPLE, half=half)
  control = _resampled_tv(ref_seq, ctrl_seq, mask, rng, n_resample=_N_RESAMPLE, half=half)
  edge = float(np.quantile(floor, _FLOOR_QUANTILE))

  payload = {
    **base,
    "reference_unit": str(args.reference_unit),
    "structure": reference.get("structure"),
    "temperature": temperature,
    "min_p": args.min_p,
    "n_reference": int(len(ref_seq)),
    "n_gpu": int(len(gpu_seq)),
    "gpu": gpu_info,
    "control": ctrl_info,
    "group_size": half,
    "floor_median": round(_median(floor), 6),
    "floor_edge_q97_5": round(edge, 6),
    "cross_median": round(_median(cross), 6),
    "control_median": round(_median(control), 6),
    "control_fired": _median(control) > edge,
    "within_floor": _median(cross) <= edge,
    # A sanity field, not a graded quantity: it records that χ1 was actually
    # produced and spans more than one bin, so a later χ1 equivalence run has
    # something to work with. χ1 equivalence itself is NOT measured here.
    "chi1_distinct_bins": int(np.unique(_chi_bins(gpu_chi)).size),
  }
  payload["verdict"] = _grade(payload)
  _emit(args.out, payload)
  _LOG.info(
    "verdict=%s  floor<=%.4f  cross=%.4f  control=%.4f (fired=%s)",
    payload["verdict"],
    edge,
    payload["cross_median"],
    payload["control_median"],
    payload["control_fired"],
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
