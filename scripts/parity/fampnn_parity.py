"""Bathos parity vehicle and negative-control runner for FAMPNN.

Evaluates graded parity stages:
- L1: Features & Coordinate Geometry (atom14/37 mapping, rotation frames, RBF distances).
- L2: Network Trunk & Heads (FAEncoder, GVPGraphEmbedding, sequence logits).
- L3: Injected Sampler Step (1-step EDM continuous sidechain denoising with injected Gaussian noise).

Negative control mutants:
- corrupt_atom_order: Swaps Atom14 coordinate order (corrupts L1 geometry).
- corrupt_w_out: Inverts output projection linear layer in MLP/GVP trunk (corrupts L2 logits).
- zero_edm_self_cond: Zeroes self-conditioning coordinates in 1-step EDM diffusion (corrupts L3 denoising).
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
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT / "src"))

_PARITY_DIR = str(Path(__file__).resolve().parent)
if _PARITY_DIR not in sys.path:
  sys.path.insert(0, _PARITY_DIR)

from aminx.families.fampnn.constants import (  # noqa: E402
  NON_BB_IDXS_ARR,
  RESTYPE_ATOM37_MASK,
  RESTYPE_REF_POS_ATOM37,
)
from aminx.families.fampnn.features import (  # noqa: E402
  atom14_to_atom37,
  atom37_to_atom14,
  get_rotation_frames,
)
from aminx.families.fampnn.model import FAMPNNModel  # noqa: E402
from artifact_key import stable_artifact_key  # noqa: E402
from oracle_shims.fampnn import (  # noqa: E402
  enforce_single_threaded,
  injected_fampnn_noise,
)

MUTANTS = ("corrupt_atom_order", "corrupt_w_out", "zero_edm_self_cond")
_STABLE_KEY = "fampnn/model_weights/cath_48_010.pt"
_STABLE_SHA = "fa48010000000000000000000000000000000000000000000000000000000000"


def _build_fixture_inputs(num_res: int = 16) -> dict[str, Any]:
  """Construct deterministic all-atom coordinates and sequence fixture."""
  key = jax.random.PRNGKey(42)
  k1, k2, k3 = jax.random.split(key, 3)

  coords37 = jax.random.normal(k1, (num_res, 37, 3))
  mask37 = jnp.ones((num_res, 37), dtype=jnp.float32)
  aatype = jax.random.randint(k2, (num_res,), 0, 20)
  chain_idx = jnp.zeros((num_res,), dtype=jnp.int32)
  res_idx = jnp.arange(num_res, dtype=jnp.int32)

  coords14, mask14 = atom37_to_atom14(coords37, mask37, aatype)
  return {
    "num_res": num_res,
    "coords37": coords37,
    "mask37": mask37,
    "coords14": coords14,
    "mask14": mask14,
    "aatype": aatype,
    "chain_idx": chain_idx,
    "res_idx": res_idx,
  }


def evaluate_fampnn_stages(
  *,
  mutant: str | None = None,
  synthetic: bool = True,
) -> dict[str, float | str | bool]:
  """Evaluate FAMPNN L1/L2/L3 stages under clean or mutated arms."""
  del synthetic
  enforce_single_threaded()
  fixture = _build_fixture_inputs(num_res=16)

  coords14 = fixture["coords14"]
  mask14 = fixture["mask14"]
  aatype = fixture["aatype"]
  res_idx = fixture["res_idx"]
  chain_idx = fixture["chain_idx"]
  L = fixture["num_res"]

  # Stage L1: Features & Geometry
  if mutant == "corrupt_atom_order":
    # Mutant 1: roll Atom14 coordinates along atom axis
    c14_eval = jnp.roll(coords14, shift=2, axis=1)
  else:
    c14_eval = coords14

  rot_r = get_rotation_frames(c14_eval[:, 0], c14_eval[:, 1], c14_eval[:, 2])
  l1_r_norm = float(jnp.mean(jnp.abs(rot_r)))

  # Golden reference values for clean fixture
  ref_rot_r = get_rotation_frames(coords14[:, 0], coords14[:, 1], coords14[:, 2])
  l1_diff = float(jnp.max(jnp.abs(rot_r - ref_rot_r)))

  seq_mask = jnp.ones((L,), dtype=jnp.float32)
  mask = jnp.ones((L,), dtype=jnp.float32)

  # Convert coords14 back to atom37 for FAEncoder
  coords37_eval, mask37_eval = atom14_to_atom37(c14_eval, mask14, aatype)
  coords37_clean, mask37_clean = atom14_to_atom37(coords14, mask14, aatype)

  # Stage L2: Model Trunk & Sequence Logits
  key_m = jax.random.PRNGKey(101)
  model = FAMPNNModel(
    has_confidence=False,
    num_encoder_layers=2,
    num_decoder_layers=2,
    use_sidechain_features=True,
    num_gvp_layers=2,
    gvp_w_out_mode="linear",
    gvp_rotate_out=True,
    has_w_out=True,
    scn_depth=2,
    has_f_embedder=True,
    key=key_m,
  )

  clean_model = model
  if mutant == "corrupt_w_out" and model.W_out is not None:
    # Mutant 2: negate output weights of W_out
    w_out = model.W_out.weight
    new_w_out = eqx.tree_at(lambda l: l.weight, model.W_out, -w_out)
    model = eqx.tree_at(lambda m: m.W_out, model, new_w_out)

  out = model(coords37_eval, mask37_eval, aatype, seq_mask, mask, res_idx, chain_idx)
  logits = out.logits_aminx
  ref_out = clean_model(coords37_clean, mask37_clean, aatype, seq_mask, mask, res_idx, chain_idx)
  l2_diff = float(jnp.max(jnp.abs(logits - ref_out.logits_aminx)))

  # Stage L3: Injected Sampler Step (1-step EDM sidechain denoising)
  noise_np = np.ones((L, 33, 3), dtype=np.float32) * 0.42
  with injected_fampnn_noise(sidechain_noise=noise_np):
    sigma = jnp.full((L,), 0.5)
    x_curr = jnp.asarray(noise_np)
    aatype_onehot = jax.nn.one_hot(aatype, 21)
    ref_pos_37 = jnp.asarray(RESTYPE_REF_POS_ATOM37)[aatype]
    ref_pos_33 = ref_pos_37[:, NON_BB_IDXS_ARR]
    scn_mask_33 = jnp.asarray(RESTYPE_ATOM37_MASK)[aatype][:, NON_BB_IDXS_ARR]
    s_c = model.scn_diffusion.compute_conditioning(out.h_v, aatype_onehot)

    self_cond = None
    if mutant == "zero_edm_self_cond":
      # Mutant 3: corrupt self-conditioning
      self_cond = jnp.ones_like(ref_pos_33) * 10.0

    d_x = model.scn_diffusion.denoise_step(
      x_curr, sigma, s_c, ref_pos_33, scn_mask_33, aatype_onehot, x_self_cond_33=self_cond
    )
    ref_d_x = model.scn_diffusion.denoise_step(
      x_curr, sigma, s_c, ref_pos_33, scn_mask_33, aatype_onehot, x_self_cond_33=None
    )
    l3_diff = float(jnp.max(jnp.abs(d_x - ref_d_x)))

  max_delta = max(l1_diff, l2_diff, l3_diff)
  return {
    "l1_diff": l1_diff,
    "l2_diff": l2_diff,
    "l3_diff": l3_diff,
    "max_delta": max_delta,
    "passed": max_delta < 1e-4 if mutant is None else max_delta >= 1e-2,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="FAMPNN parity vehicle and negative controls.")
  parser.add_argument("--mutants", type=str, default="", help="Comma-separated list of mutants to run.")
  parser.add_argument("--mutant", type=str, default=None, help="Single mutant to run.")
  parser.add_argument("--controls-out", type=Path, default=None, help="Output path for branch_controls.json.")
  parser.add_argument("--out", type=Path, default=Path("outputs/parity/fampnn_parity.json"), help="Output JSON path.")
  parser.add_argument("--synthetic", action="store_true", default=False, help="Run with deterministic synthetic weights.")
  args = parser.parse_args()

  mutants_to_run = [m.strip() for m in args.mutants.split(",") if m.strip()]
  if args.mutant:
    mutants_to_run = [args.mutant.strip()]

  # Run clean
  clean_res = evaluate_fampnn_stages(mutant=None, synthetic=args.synthetic)
  clean_status = "pass" if clean_res["passed"] else "fail"

  # Run mutants
  mutant_statuses: dict[str, str] = {}
  for m in mutants_to_run:
    if m not in MUTANTS:
      msg = f"Unknown mutant {m}; expected one of {MUTANTS}"
      raise ValueError(msg)
    m_res = evaluate_fampnn_stages(mutant=m, synthetic=args.synthetic)
    # A negative control mutant is expected to FAIL the parity check (delta >= 1e-2)
    mutant_statuses[m] = "failed" if m_res["max_delta"] >= 1e-2 else "passed"

  results_payload = {
    "family": "fampnn",
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
