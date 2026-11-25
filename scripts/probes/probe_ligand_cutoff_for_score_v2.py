"""Pre-registered probe v2 for Clause 1: ligand_cutoff_for_score.

Supersedes probe_ligand_cutoff_for_score.py (v1), which is kept unmodified on record.
v1 run 6a6abe5e FAILED because its expectation used CA distance (CA at x = 0, 5, 7.5, 12
from a ligand atom at the origin). The library gates on the virtual C-beta distance, and CB
sits at CA + (0.5533, -0.7569, -1.1830) for these synthetic residues, so the CA-based
expectation was wrong, not the library. v2 places CA so that the CB distances land on
known values and checks every gating decision against CB distance.

Geometry (hand-computed, see _expected_cb_xyz):
  CA = (x, 0, 0), N = CA - (1.45, 0, 0), C = CA + (0.5, 1.4, 0), ligand atom at the origin.
  CB offset from CA = (0.55330366, -0.75694452, -1.18295065).
  CB distance d(x) = sqrt((x + 0.55330366)^2 + 1.97233723).
  x = 0.0     -> CB distance 1.50946   (inside 8.0 A)
  x = 5.0     -> CB distance 5.72813   (inside)
  x = 7.2209  -> CB distance 7.90004   (inside; straddle pair, low side)
  x = 7.4240  -> CB distance 8.09998   (outside; straddle pair, high side)
  x = 12.0    -> CB distance 12.63162  (outside)
Expected mask is [1, 1, 1, 0, 0] and expected proximal count is 3.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT / "src"))

from aminx.scoring.ligand_cutoff import (
  compute_ligand_cutoff_mask,
  compute_loss_xy,
  compute_virtual_c_beta,
)


logger = logging.getLogger(__name__)

# Distance cutoff in Angstroms, written as a literal (not imported) so the expectation below
# does not depend on the library default under test.
CUTOFF_ANGSTROMS: float = 8.0

# CA x-positions of the five synthetic residues. Each CB distance is given beside it.
CA_X_POSITIONS: tuple[float, ...] = (0.0, 5.0, 7.2209, 7.4240, 12.0)
STRADDLE_INSIDE_INDEX: int = 2  # CB distance about 7.90 A, expected inside the cutoff
STRADDLE_OUTSIDE_INDEX: int = 3  # CB distance about 8.10 A, expected outside the cutoff


def _expected_cb_xyz(ca_x: float) -> np.ndarray:
  """Virtual C-beta for the synthetic residue with CA at (ca_x, 0, 0), in plain numpy.

  Same empirical formula as ProteinMPNN/LigandMPNN (CB = -0.58273431 a + 0.56802827 b
  - 0.54067466 c + CA), written out component by component:
    b = CA - N = (1.45, 0, 0)
    c = C - CA = (0.5, 1.4, 0)
    a = b x c  = (b_y c_z - b_z c_y, b_z c_x - b_x c_z, b_x c_y - b_y c_x)
               = (0*0 - 0*1.4, 0*0.5 - 1.45*0, 1.45*1.4 - 0*0.5)
               = (0, 0, 2.03)
    CB - CA    = (0.56802827*1.45 - 0.54067466*0.5, -0.54067466*1.4, -0.58273431*2.03)
               = (0.55330366, -0.75694452, -1.18295065)
  """
  b = (1.45, 0.0, 0.0)
  c = (0.5, 1.4, 0.0)
  a = (
    b[1] * c[2] - b[2] * c[1],
    b[2] * c[0] - b[0] * c[2],
    b[0] * c[1] - b[1] * c[0],
  )
  offset_x = -0.58273431 * a[0] + 0.56802827 * b[0] - 0.54067466 * c[0]
  offset_y = -0.58273431 * a[1] + 0.56802827 * b[1] - 0.54067466 * c[1]
  offset_z = -0.58273431 * a[2] + 0.56802827 * b[2] - 0.54067466 * c[2]
  return np.array([ca_x + offset_x, offset_y, offset_z], dtype=np.float64)


def _expected_cb_distance(ca_x: float) -> float:
  """CB distance from the CB of the residue at ca_x to a ligand atom at the origin."""
  cb = _expected_cb_xyz(ca_x)
  # Euclidean norm written out: sqrt(x^2 + y^2 + z^2).
  return float(np.sqrt(cb[0] * cb[0] + cb[1] * cb[1] + cb[2] * cb[2]))


def run_probe() -> dict[str, object]:
  """Execute probe verifying CB-distance ligand gating and auxiliary loss_XY."""
  # 1. Virtual C-beta test (same synthetic N, CA, C layout as v1)
  n = jnp.array([0.0, 0.0, 0.0])
  ca = jnp.array([1.46, 0.0, 0.0])
  c = jnp.array([1.96, 1.42, 0.0])
  coords = jnp.stack([n, ca, c], axis=0)[None, ...]  # (1, 3, 3)

  b = ca - n
  c_vec = c - ca
  a = jnp.cross(b, c_vec)
  expected_cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca
  computed_cb = compute_virtual_c_beta(coords)[0]
  cb_diff = float(jnp.max(jnp.abs(computed_cb - expected_cb)))

  # 2. CB-distance gating test
  # Expected CB distances and mask come only from _expected_cb_distance (plain numpy).
  l_len = len(CA_X_POSITIONS)
  expected_cb_distances = np.array([_expected_cb_distance(x) for x in CA_X_POSITIONS])
  expected_mask = np.array(
    [1.0 if d < CUTOFF_ANGSTROMS else 0.0 for d in expected_cb_distances], dtype=np.float32
  )
  expected_num_proximal = int(np.sum(expected_mask))

  res_coords = jnp.zeros((l_len, 4, 3))
  ca_x = jnp.array(CA_X_POSITIONS, dtype=jnp.float32)
  res_coords = res_coords.at[:, 1, 0].set(ca_x)
  # Backbone offsets: N = CA - (1.45, 0, 0), C = CA + (0.5, 1.4, 0)
  res_coords = res_coords.at[:, 0, :].set(res_coords[:, 1, :] - jnp.array([1.45, 0.0, 0.0]))
  res_coords = res_coords.at[:, 2, :].set(res_coords[:, 1, :] + jnp.array([0.5, 1.4, 0.0]))

  # Ligand atom at origin: (0.0, 0.0, 0.0). Second atom is masked out and far away.
  lig_coords = jnp.array([[0.0, 0.0, 0.0], [99.0, 99.0, 99.0]])
  lig_mask = jnp.array([1.0, 0.0])
  res_mask = jnp.ones(l_len)

  mask_8a, _d_xy = compute_ligand_cutoff_mask(
    res_coords, lig_coords, lig_mask, res_mask, cutoff=CUTOFF_ANGSTROMS
  )
  library_mask = np.asarray(mask_8a)
  # Full-mask agreement with the CB-distance expectation (this also fixes the proximal count).
  cutoff_gating_passed = bool(
    np.array_equal(library_mask, expected_mask) and int(np.sum(library_mask)) == expected_num_proximal
  )

  # Straddle check: the two residues either side of 8.0 A in CB distance must lie on opposite
  # sides in the expectation, and the library must place them on those same sides.
  inside_d = float(expected_cb_distances[STRADDLE_INSIDE_INDEX])
  outside_d = float(expected_cb_distances[STRADDLE_OUTSIDE_INDEX])
  straddle_passed = bool(
    inside_d < CUTOFF_ANGSTROMS < outside_d
    and library_mask[STRADDLE_INSIDE_INDEX] == 1.0
    and library_mask[STRADDLE_OUTSIDE_INDEX] == 0.0
  )

  # Negative control: when all ligand atoms are masked out, mask_xy must be all zeros.
  mask_empty, _ = compute_ligand_cutoff_mask(
    res_coords, lig_coords, jnp.array([0.0, 0.0]), res_mask, cutoff=CUTOFF_ANGSTROMS
  )
  empty_control_passed = bool(jnp.all(mask_empty == 0.0))

  # 3. Loss_XY calculation test
  logits = jnp.zeros((l_len, 21))
  logits = logits.at[0, 1].set(5.0)
  logits = logits.at[1, 2].set(5.0)
  seq = jnp.array([1, 2, 3, 4, 5])
  loss_xy = float(compute_loss_xy(logits, seq, mask_8a))
  loss_xy_finite = bool(loss_xy > 0.0 and not np.isnan(loss_xy) and not np.isinf(loss_xy))

  # Known answer, by hand: two proximal residues, both given UNIFORM logits over the 21-token
  # vocabulary, so each contributes NLL = ln(21) and the mask-weighted mean is
  # ln(21) * 2 / (2 + 1e-8). Residues outside the mask are given a confidently WRONG prediction
  # (they must not move the answer). The expected value is computed here from the formula, not
  # from the library.
  known_logits = jnp.zeros((l_len, 21))
  known_logits = known_logits.at[2, 5].set(40.0).at[3, 5].set(40.0)  # masked-out, token 5 wrong
  known_mask = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0])
  known_seq = jnp.array([1, 2, 3, 4, 5])
  known_expected = float(np.log(21.0) * 2.0 / (2.0 + 1e-8))
  known_value = float(compute_loss_xy(known_logits, known_seq, known_mask))
  loss_xy_known_answer_error = abs(known_value - known_expected)
  loss_xy_valid = bool(loss_xy_finite and loss_xy_known_answer_error < 1e-5)

  probe_passed = bool(
    cb_diff < 1e-6
    and cutoff_gating_passed
    and straddle_passed
    and empty_control_passed
    and loss_xy_valid
  )

  return {
    "probe_name": "ligand_cutoff_for_score_v2",
    "clause": "ligand_cutoff_for_score",
    "status": "PASS" if probe_passed else "FAIL",
    "cb_max_abs_diff": cb_diff,
    "cutoff_gating_passed": cutoff_gating_passed,
    "straddle_passed": straddle_passed,
    "straddle_cb_distance_inside": inside_d,
    "straddle_cb_distance_outside": outside_d,
    "empty_control_passed": empty_control_passed,
    "loss_xy_valid": loss_xy_valid,
    "loss_xy_value": loss_xy,
    "loss_xy_known_answer_error": loss_xy_known_answer_error,
    "num_proximal_residues": int(np.sum(library_mask)),
    "expected_num_proximal_residues": expected_num_proximal,
  }


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
  parser = argparse.ArgumentParser(description="Probe v2 for ligand_cutoff_for_score clause.")
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("outputs/probes/probe_ligand_cutoff_for_score_v2.json"),
    help="Output JSON path.",
  )
  args = parser.parse_args()

  result = run_probe()

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  bth_results = os.environ.get("BTH_RESULTS_PATH")
  if bth_results:
    bth_path = Path(bth_results)
    bth_path.parent.mkdir(parents=True, exist_ok=True)
    bth_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  logger.info("Probe ligand_cutoff_for_score v2: %s", result["status"])


if __name__ == "__main__":
  main()
