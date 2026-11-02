"""Pre-registered probe for Clause 1: ligand_cutoff_for_score."""

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
  DEFAULT_LIGAND_CUTOFF_ANGSTROMS,
  compute_ligand_cutoff_mask,
  compute_loss_xy,
  compute_virtual_c_beta,
)


logger = logging.getLogger(__name__)


def run_probe() -> dict[str, object]:
  """Execute probe verifying ligand cutoff distance gating and auxiliary loss_XY."""
  # 1. Virtual C-beta test
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

  # 2. Distance gating test
  # 4 residues: pos 0 at 0A, pos 1 at 5A, pos 2 at 7.5A, pos 3 at 12A
  l_len = 4
  res_coords = jnp.zeros((l_len, 4, 3))
  res_coords = res_coords.at[:, 1, 0].set(jnp.array([0.0, 5.0, 7.5, 12.0]))
  # Backbone offsets
  res_coords = res_coords.at[:, 0, :].set(res_coords[:, 1, :] - jnp.array([1.45, 0.0, 0.0]))
  res_coords = res_coords.at[:, 2, :].set(res_coords[:, 1, :] + jnp.array([0.5, 1.4, 0.0]))

  # Ligand atom at origin: (0.0, 0.0, 0.0)
  lig_coords = jnp.array([[0.0, 0.0, 0.0], [99.0, 99.0, 99.0]])
  lig_mask = jnp.array([1.0, 0.0])  # second atom masked
  res_mask = jnp.ones(l_len)

  mask_8a, d_xy = compute_ligand_cutoff_mask(
    res_coords, lig_coords, lig_mask, res_mask, cutoff=DEFAULT_LIGAND_CUTOFF_ANGSTROMS
  )
  # Residues 0, 1, 2 are within 8.0A, residue 3 is at ~12A > 8.0A
  cutoff_gating_passed = bool(
    mask_8a[0] == 1.0 and mask_8a[1] == 1.0 and mask_8a[2] == 1.0 and mask_8a[3] == 0.0
  )

  # Negative control: when all ligand atoms are masked out, mask_xy must be all zeros
  mask_empty, _ = compute_ligand_cutoff_mask(
    res_coords, lig_coords, jnp.array([0.0, 0.0]), res_mask, cutoff=8.0
  )
  empty_control_passed = bool(jnp.all(mask_empty == 0.0))

  # 3. Loss_XY calculation test
  logits = jnp.zeros((l_len, 21))
  logits = logits.at[0, 1].set(5.0)
  logits = logits.at[1, 2].set(5.0)
  seq = jnp.array([1, 2, 3, 4])
  loss_xy = float(compute_loss_xy(logits, seq, mask_8a))
  loss_xy_valid = bool(loss_xy > 0.0 and not np.isnan(loss_xy) and not np.isinf(loss_xy))

  probe_passed = bool(
    cb_diff < 1e-6 and cutoff_gating_passed and empty_control_passed and loss_xy_valid
  )

  return {
    "probe_name": "ligand_cutoff_for_score",
    "clause": "ligand_cutoff_for_score",
    "status": "PASS" if probe_passed else "FAIL",
    "cb_max_abs_diff": cb_diff,
    "cutoff_gating_passed": cutoff_gating_passed,
    "empty_control_passed": empty_control_passed,
    "loss_xy_valid": loss_xy_valid,
    "loss_xy_value": loss_xy,
    "num_proximal_residues": int(jnp.sum(mask_8a)),
  }


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
  parser = argparse.ArgumentParser(description="Probe for ligand_cutoff_for_score clause.")
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("outputs/probes/probe_ligand_cutoff_for_score.json"),
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

  logger.info("Probe ligand_cutoff_for_score: %s", result["status"])


if __name__ == "__main__":
  main()
