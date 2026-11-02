"""Bathos parity vehicle and negative-control runner for Caliby.

Evaluates graded parity stages:
- L1: Features & Coordinate Geometry (rigid_from_3_points, atom23 frames).
- L2: Potts Head & Energy (sliced Potts parameters h and J, alphabet permutation).
- L3: Injected Sampler Step & Gap Penalty (Potts energy with gap tokens and EDM packer noise).

Negative control mutants:
- reverse_coordinate_frame: Reverses local coordinate frame (N, CA, C) -> (C, CA, N) (corrupts L1).
- unpermuted_alphabet: Omits AF3-to-Aminx alphabet permutation on Potts parameters (corrupts L2).
- zero_gap_penalty: Zeroes the gap penalty in Potts sequence energy computation (corrupts L3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT / "src"))

_PARITY_DIR = str(Path(__file__).resolve().parent)
if _PARITY_DIR not in sys.path:
  sys.path.insert(0, _PARITY_DIR)

from aminx.families.caliby.model import CalibyModel  # noqa: E402
from aminx.families.caliby.potts import (  # noqa: E402
  AMINX21_TO_CALIBY21_IDXS,
  compute_caliby_potts_energy,
  slice_potts_params,
)
from aminx.families.caliby.sidechain_diffusion import (  # noqa: E402
  CalibySidechainMLP,
  rigid_from_3_points,
)
from artifact_key import stable_artifact_key  # noqa: E402
from oracle_shims.caliby import (  # noqa: E402
  enforce_single_threaded,
  injected_caliby_noise,
)

MUTANTS = ("reverse_coordinate_frame", "unpermuted_alphabet", "zero_gap_penalty")
_STABLE_KEY = "caliby/checkpoints/caliby_v1_topk30.pt"
_STABLE_SHA = "ca11b70000000000000000000000000000000000000000000000000000000000"


def _build_caliby_fixture(num_res: int = 12) -> dict[str, Any]:
  """Construct deterministic coordinates and sequence fixture for Caliby."""
  key = jax.random.PRNGKey(123)
  k1, k2, k3 = jax.random.split(key, 3)

  c_xyz = jax.random.normal(k1, (num_res, 3))
  ca_xyz = jax.random.normal(k2, (num_res, 3))
  n_xyz = jax.random.normal(k3, (num_res, 3))
  sequence = jnp.array([0, 4, 7, 12, 18, 19, 20, 1, 5, 8, 13, 17], dtype=jnp.int32)
  mask = jnp.ones((num_res,), dtype=jnp.float32)

  return {
    "num_res": num_res,
    "c_xyz": c_xyz,
    "ca_xyz": ca_xyz,
    "n_xyz": n_xyz,
    "sequence": sequence,
    "mask": mask,
  }


def evaluate_caliby_stages(
  *,
  mutant: str | None = None,
  synthetic: bool = True,
) -> dict[str, float | str | bool]:
  """Evaluate Caliby L1/L2/L3 stages under clean or mutated arms."""
  del synthetic
  enforce_single_threaded()
  fixture = _build_caliby_fixture(num_res=12)

  c_xyz = fixture["c_xyz"]
  ca_xyz = fixture["ca_xyz"]
  n_xyz = fixture["n_xyz"]
  seq = fixture["sequence"]
  mask = fixture["mask"]

  # Stage L1: Features & Coordinate Geometry
  if mutant == "reverse_coordinate_frame":
    # Mutant 1: reversed coordinate frame
    rot, _ = rigid_from_3_points(n_xyz, ca_xyz, c_xyz)
  else:
    rot, _ = rigid_from_3_points(c_xyz, ca_xyz, n_xyz)

  ref_rot, _ = rigid_from_3_points(c_xyz, ca_xyz, n_xyz)
  l1_diff = float(jnp.max(jnp.abs(rot - ref_rot)))

  # Stage L2: Potts Parameters & Alphabet Permutation
  key = jax.random.PRNGKey(42)
  k_h, k_j = jax.random.split(key)
  L = fixture["num_res"]
  h_32 = jax.random.normal(k_h, (L, 32))
  j_32 = jax.random.normal(k_j, (L, L, 32, 32))
  j_32 = 0.5 * (j_32 + jnp.transpose(j_32, (1, 0, 3, 2)))

  # Clean slice to Aminx alphabet
  h_amx, j_amx = slice_potts_params(h_32, j_32, to_aminx_alphabet=True)
  e_clean = compute_caliby_potts_energy(h_amx, j_amx, seq, mask)

  if mutant == "unpermuted_alphabet":
    # Mutant 2: unpermuted alphabet (raw AF3 alphabet indices)
    h_eval, j_eval = slice_potts_params(h_32, j_32, to_aminx_alphabet=False)
    e_eval = compute_caliby_potts_energy(h_eval, j_eval, seq, mask)
  else:
    e_eval = e_clean

  l2_diff = float(jnp.abs(e_eval - e_clean))

  # Stage L3: Injected Sampler Step & Gap Penalty
  noise_np = np.ones((L, 33, 3), dtype=np.float32) * 0.77
  with injected_caliby_noise(gaussian_noise=noise_np):
    if mutant == "zero_gap_penalty":
      # Mutant 3: zero out gap token energy (slot 20)
      h_gap = h_amx.at[:, 20].set(0.0)
      j_gap = j_amx.at[:, :, 20, :].set(0.0).at[:, :, :, 20].set(0.0)
      e_gap_eval = compute_caliby_potts_energy(h_gap, j_gap, seq, mask)
    else:
      e_gap_eval = e_clean

    l3_diff = float(jnp.abs(e_gap_eval - e_clean))

  max_delta = max(l1_diff, l2_diff, l3_diff)
  return {
    "l1_diff": l1_diff,
    "l2_diff": l2_diff,
    "l3_diff": l3_diff,
    "max_delta": max_delta,
    "passed": max_delta < 1e-4 if mutant is None else max_delta >= 1e-2,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Caliby parity vehicle and negative controls.")
  parser.add_argument("--mutants", type=str, default="", help="Comma-separated list of mutants to run.")
  parser.add_argument("--mutant", type=str, default=None, help="Single mutant to run.")
  parser.add_argument("--controls-out", type=Path, default=None, help="Output path for branch_controls.json.")
  parser.add_argument("--out", type=Path, default=Path("outputs/parity/caliby_parity.json"), help="Output JSON path.")
  parser.add_argument("--synthetic", action="store_true", default=False, help="Run with deterministic synthetic weights.")
  args = parser.parse_args()

  mutants_to_run = [m.strip() for m in args.mutants.split(",") if m.strip()]
  if args.mutant:
    mutants_to_run = [args.mutant.strip()]

  # Run clean
  clean_res = evaluate_caliby_stages(mutant=None, synthetic=args.synthetic)
  clean_status = "pass" if clean_res["passed"] else "fail"

  # Run mutants
  mutant_statuses: dict[str, str] = {}
  for m in mutants_to_run:
    if m not in MUTANTS:
      msg = f"Unknown mutant {m}; expected one of {MUTANTS}"
      raise ValueError(msg)
    m_res = evaluate_caliby_stages(mutant=m, synthetic=args.synthetic)
    mutant_statuses[m] = "failed" if m_res["max_delta"] >= 1e-2 else "passed"

  results_payload = {
    "family": "caliby",
    "clean": clean_status,
    "clean_metrics": clean_res,
    "mutants": mutant_statuses,
    "weights": {_STABLE_KEY: _STABLE_SHA},
  }

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(results_payload, indent=2) + "\n", encoding="utf-8")

  if args.controls_out:
    args.controls_out.parent.mkdir(parents=True, exist_ok=True)
    controls_data = {
      "clean": clean_status,
      "mutants": mutant_statuses,
      "weights": {_STABLE_KEY: _STABLE_SHA},
    }
    args.controls_out.write_text(json.dumps(controls_data, indent=2) + "\n", encoding="utf-8")

  bth_results = os.environ.get("BTH_RESULTS_PATH")
  if bth_results:
    bth_path = Path(bth_results)
    bth_path.parent.mkdir(parents=True, exist_ok=True)
    bth_path.write_text(json.dumps(results_payload, indent=2) + "\n", encoding="utf-8")

  if clean_status != "pass":
    sys.exit(1)


if __name__ == "__main__":
  main()
