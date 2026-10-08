"""Are the sealed Potts/LASEr oracle neighbour indices thread-dependent? (aminx debt #2584)

Multi-threaded float reductions in a pairwise-distance computation flip ``topk`` where two
neighbours are near-tied, and the thread partition varies per process. The Potts and LASEr
dumpers set no thread count until 9b520400 (261007), so every existing seal was made at default
threads, and aminx's tests compare the recomputed neighbour tensors against those seals exactly.

Two measurements, neither of which re-runs a dumper:

A. CROSS-DUMP IDENTITY. The dump root holds several independently produced sets of the same
   cells (separate processes, default threads). Any neighbour-index array present in two sets
   of the same dtype must be identical; a differing slot is a flip that actually happened.
B. TIE MARGIN (Potts only: LASEr dumps carry no coordinates). From each cell's stored ``X`` and
   ``mask`` recompute the Calpha distances in float64 and take the smallest RELATIVE gap
   between adjacent sorted distances within the neighbour list. A cell whose smallest gap is
   far above float32 reduction noise cannot flip at any thread count; one below it is exposed.

Instrument checks, run before the verdict is read (a measurement that can only pass is not a
check): the recomputed neighbours must equal the dumped ``E_idx`` wherever the margin is clear
(this validates the recipe), a synthetic exact tie must be flagged exposed, a synthetic spread
layout must not be, and a single flipped slot must be detected by the comparator.

Read-only on the dumps. Usage (graded run, on the host that holds the dumps):

    bth run --project-slug aminx -- uv run --no-sync python3 \
        scripts/parity/oracle_idx_stability.py --dumps-root ~/projects/aminx-oracles/dumps
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
from typing import Any

logger = logging.getLogger("oracle_idx_stability")

K_NEIGHBORS = 48
EPS = 1e-6  # ProteinMPNN's _dist eps: D = sqrt(sum(dX^2) + eps)
# float32 eps is 1.19e-7; a squared-distance sum plus sqrt carries a few ulp of reduction
# noise. 1e-5 relative is ~84 ulp: deliberately generous, so "exposed" errs toward flagging.
REL_TIE = 1e-5

POTTS_SETS = ("a1_potts", "a1_potts_v1")
LASER_SETS = ("b1_laser", "b2_laser_prehook", "b2_laser_rerun", "b3_laser_owned")
POTTS_WAVES = (
  "potts_head",
  "potts_energy",
  "potts_ar_decode",
  "potts_refine",
  "potts_merge_pair_d2",
  "potts_merge_pair_d4",
  "pottsmpnn_full",
)
LASER_WAVES = ("laser_encoder", "laser_score")
IDX_SUFFIXES = ("E_idx", "pr_pr_idx", "lig_pr_idx")


def neighbors_and_margin(ca: np.ndarray, mask: np.ndarray, k: int = K_NEIGHBORS) -> tuple[np.ndarray, float]:
  """ProteinMPNN-recipe neighbours (self included) and the smallest relative adjacent gap.

  ``ca`` is ``(L, 3)``; ``mask`` is ``(L,)``. Distances are float64, so the result is the
  geometric answer that a float32 multi-threaded reduction approximates.
  """
  ca = np.asarray(ca, dtype=np.float64)
  mask = np.asarray(mask, dtype=np.float64)
  length = ca.shape[0]
  kk = min(k, length)
  mask2d = mask[:, None] * mask[None, :]
  dist = np.sqrt(((ca[:, None, :] - ca[None, :, :]) ** 2).sum(-1) + EPS)
  adjusted = dist + (1.0 - mask2d) * dist.max()
  order = np.argsort(adjusted, axis=1, kind="stable")
  top = np.take_along_axis(adjusted, order, axis=1)[:, : min(kk + 1, length)]
  gaps = np.diff(top, axis=1) / np.maximum(top[:, 1:], 1e-12)
  return order[:, :kk], float(gaps.min()) if gaps.size else float("inf")


def count_slot_differences(a: np.ndarray, b: np.ndarray) -> int:
  if a.shape != b.shape:
    return int(max(a.size, b.size))
  return int((a != b).sum())


def _idx_keys(npz: np.lib.npyio.NpzFile) -> list[str]:
  return [k for k in npz.files if k.endswith(IDX_SUFFIXES)]


def cross_dump(root: Path, sets: tuple[str, ...], waves: tuple[str, ...]) -> dict[str, int]:
  """Compare every neighbour-index array present in two or more sets, per dtype file."""
  compared = differing = slots = 0
  for wave in waves:
    for dtype in ("oracle_f32.npz", "oracle_f64.npz"):
      loaded = []
      for name in sets:
        path = root / name / wave / dtype
        if path.is_file():
          loaded.append((name, np.load(path, allow_pickle=False)))
      for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
          (na, za), (nb, zb) = loaded[i], loaded[j]
          for key in sorted(set(_idx_keys(za)) & set(_idx_keys(zb))):
            n = count_slot_differences(za[key], zb[key])
            compared += 1
            if n:
              differing += 1
              slots += n
              logger.info("DIFF %s %s/%s vs %s: %d slots in %s", wave, dtype, na, nb, n, key)
  return {"compared": compared, "differing": differing, "slots": slots}


def potts_margins(root: Path) -> dict[str, float | int]:
  """Tie margin per Potts cell with stored coordinates, and the recipe check."""
  n_cells = n_exposed = n_recipe_bad = n_clear = 0
  min_gap = float("inf")
  for name in POTTS_SETS:
    for wave in POTTS_WAVES:
      path = root / name / wave / "oracle_f64.npz"
      if not path.is_file():
        continue
      z = np.load(path, allow_pickle=False)
      for key in sorted(k for k in z.files if k.endswith("__X")):
        prefix = key[: -len("X")]
        if prefix + "E_idx" not in z.files or prefix + "mask" not in z.files:
          continue
        x, mask, dumped = z[key], z[prefix + "mask"], z[prefix + "E_idx"]
        if x.ndim != 4 or x.shape[0] != 1:
          continue
        got, gap = neighbors_and_margin(x[0, :, 1, :], mask[0])
        n_cells += 1
        min_gap = min(min_gap, gap)
        if gap < REL_TIE:
          n_exposed += 1
          logger.info("EXPOSED %s/%s %s min_rel_gap=%.3e", name, wave, prefix, gap)
        else:
          n_clear += 1
          if not np.array_equal(got, dumped[0]):
            n_recipe_bad += 1
            logger.info("RECIPE-MISMATCH %s/%s %s", name, wave, prefix)
  return {
    "n_cells": n_cells,
    "n_exposed": n_exposed,
    "n_clear": n_clear,
    "n_recipe_bad": n_recipe_bad,
    "min_rel_gap": min_gap if n_cells else -1.0,
  }


def instrument_controls() -> dict[str, bool]:
  """Synthetic ground truth for the two measurements (see module docstring)."""
  rng = np.random.default_rng(0)
  spread = rng.normal(size=(12, 3)) * 10.0
  ones = np.ones(12)
  _, spread_gap = neighbors_and_margin(spread, ones)
  tied = spread.copy()
  tied[5] = tied[3] + 0.0  # residue 5 sits exactly on residue 3: equal distances to everyone
  _, tied_gap = neighbors_and_margin(tied, ones)
  idx, _ = neighbors_and_margin(spread, ones)
  flipped = idx.copy()
  flipped[4, 2], flipped[4, 3] = flipped[4, 3], flipped[4, 2]
  return {
    "control_spread_clear": spread_gap >= REL_TIE,
    "control_tie_flagged": tied_gap < REL_TIE,
    "control_flip_detected": count_slot_differences(idx, flipped) == 2,
    "control_identical_clean": count_slot_differences(idx, idx.copy()) == 0,
  }


def grade(results: dict[str, Any]) -> str:
  controls_ok = all(results[k] for k in results if str(k).startswith("control_"))
  enough = results["n_arrays_compared"] > 0 and results["n_cells"] > 0
  if not controls_ok or not enough or results["n_recipe_bad"] > 0:
    return "inconclusive"
  if results["n_arrays_differing"] == 0 and results["n_exposed"] == 0:
    return "pass"
  return "fail"


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dumps-root", type=Path, required=True)
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  root = args.dumps_root.expanduser()

  results: dict[str, Any] = dict(instrument_controls())
  potts = cross_dump(root, POTTS_SETS, POTTS_WAVES)
  laser = cross_dump(root, LASER_SETS, LASER_WAVES)
  margins = potts_margins(root)
  results.update(
    {
      "n_arrays_compared": potts["compared"] + laser["compared"],
      "n_arrays_differing": potts["differing"] + laser["differing"],
      "n_slots_differing": potts["slots"] + laser["slots"],
      "n_potts_arrays_compared": potts["compared"],
      "n_laser_arrays_compared": laser["compared"],
      **margins,
    }
  )
  results["clean"] = grade(results)
  logger.info("%s", json.dumps(results, indent=1, default=str))

  env = os.environ.get("BTH_RESULTS_PATH")
  if not env:
    msg = "oracle_idx_stability requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  Path(env).write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
