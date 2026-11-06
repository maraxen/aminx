#!/usr/bin/env python3
"""Tracked diagnostic script for aminx 0.4.0 lmpnn__base Lane T divergence.

Pre-registered under scripts/analysis/diagnose_lmpnn_lane_t.bth.toml.

Performs:
1. Exact reproduction of Lane T for cell lmpnn__base on 1BC8 Chain C.
2. Full per-position |logprob diff| extraction across all 93 designed positions.
3. Multi-axis clustering:
   - Spatial axis: Euclidean distance from Calpha and Cbeta to ligand atom(s).
   - Temporal axis: Decode order rank accumulation.
   - Boundary axis: Distance to C-terminus / padding boundary.
4. Controlled bisection probe:
   - Incremental caching (run_incremental) vs full recompute (run_full) via --force-incremental.
5. Emits structured results to BTH_RESULTS_PATH and --out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

# --------------------------------------------------------------------------- Constants
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
PDB_NAME = "1BC8.pdb"
CHAIN = "C"
ORDER_SEED = 7
DIST_SEED = 5
N_PAD = 96
N_TEACHER = 4
TOL_LOGPROB = 1e-3


def _lsm(x: np.ndarray) -> np.ndarray:
  """Log-softmax along last axis in float64."""
  x = x.astype(np.float64)
  m = x.max(-1, keepdims=True)
  return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def _order_fn(order: np.ndarray, length: int):
  """Persistent static order closure for aminx sampler."""
  import jax.numpy as jnp

  def fn(key, n, tie_map=None, num_groups=None):
    tail = jnp.arange(length, n, dtype=jnp.int32)
    return jnp.concatenate([jnp.asarray(order, dtype=jnp.int32), tail]), key

  return fn


def setup_ref_oracle(ref_root: Path):
  """Load upstream LigandMPNN model and oracle context."""
  if str(ref_root) not in sys.path:
    sys.path.insert(0, str(ref_root))
  import torch
  torch.cuda.is_available = lambda: False
  import model_utils

  checkpoint_path = ref_root / "model_params" / "ligandmpnn_v_32_020_25.pt"
  checkpoint = torch.load(checkpoint_path, map_location="cpu")
  model = model_utils.ProteinMPNN(
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=checkpoint["num_edges"],
    atom_context_num=checkpoint["atom_context_num"],
    model_type="ligand_mpnn",
    ligand_mpnn_use_side_chain_context=checkpoint.get("use_side_chains", 0),
  )
  model.load_state_dict(checkpoint["model_state_dict"])
  model.eval()
  return torch, model_utils, model


def patch_force_incremental(mode: str):
  """Intercept make_decode_fn in aminx to enforce incremental mode."""
  if mode == "auto":
    return
  import aminx.host.plan as host_plan
  import aminx.inference.decode.factory as decode_factory
  from aminx.inference.decode.mode import AutoregressiveConfig

  orig_make_decode_fn = decode_factory.make_decode_fn

  def patched_make_decode_fn(*args, **kwargs):
    ar_cfg = kwargs.get("autoregressive_config")
    if ar_cfg is None:
      kwargs["autoregressive_config"] = AutoregressiveConfig(incremental=mode)
    else:
      from dataclasses import replace
      kwargs["autoregressive_config"] = replace(ar_cfg, incremental=mode)
    return orig_make_decode_fn(*args, **kwargs)

  decode_factory.make_decode_fn = patched_make_decode_fn
  host_plan.make_decode_fn = patched_make_decode_fn


def main() -> int:
  parser = argparse.ArgumentParser(description="Diagnose lmpnn__base Lane T divergence")
  parser.add_argument("--ref-root", type=Path, default=Path("/home/marielle/projects/aminx/reference_ligandmpnn_clone"))
  parser.add_argument("--fixture-dir", type=Path, default=None, help="Directory containing pre-dumped ref/ligand npz")
  parser.add_argument("--out", type=Path, default=Path("outputs/analysis/diagnose_lmpnn_lane_t.json"))
  parser.add_argument("--force-incremental", choices=["auto", "off", "force"], default="auto")
  args = parser.parse_args()

  args.out.parent.mkdir(parents=True, exist_ok=True)

  # Apply incremental probe patch if requested
  if args.force_incremental != "auto":
    patch_force_incremental(args.force_incremental)

  # Setup reference context
  os.environ["REFERENCE_PATH"] = str(args.ref_root)

  # Find and import e2e_run_api_parity
  repo_root = Path(__file__).resolve().parent.parent.parent
  parity_script_dir = repo_root / "scripts" / "parity"
  if str(parity_script_dir) not in sys.path:
    sys.path.insert(0, str(parity_script_dir))
  import e2e_run_api_parity as e2e

  ctx = e2e.ref_context("lmpnn", "base", batch=1)
  fd = ctx["fd"]
  order = ctx["out"]["decoding_order"][0].numpy()
  length = fd["mask"].shape[1]
  designed = e2e._designed(fd)
  n_designed = int(designed.sum())

  # Run aminx sampling under the injected reference order
  kw = e2e.aminx_kwargs("lmpnn", ctx, temperature=0.1)
  seqs, logits = e2e.aminx_sample(kw, num_samples=N_TEACHER, seed=DIST_SEED, order=order, length=length)

  # Teacher-force reference oracle on aminx's sequences
  torch, model_utils, ref_model = e2e._import_reference_torch()
  model = ctx["model"]

  per_sample_diffs = []
  per_sample_ref_lp = []
  worst_overall = 0.0

  for i in range(N_TEACHER):
    fd2 = dict(fd)
    fd2["S"] = torch.as_tensor(seqs[i : i + 1].astype(np.int64))
    fd2["batch_size"] = 1
    fd2["randn"] = ctx["randn_row"]
    with torch.no_grad():
      ref_lp = model.score(fd2, use_sequence=True)["log_probs"].numpy()[0]
    per_sample_ref_lp.append(ref_lp)
    diff = np.abs(_lsm(logits[i])[designed, :20] - ref_lp[designed, :20])
    per_sample_diffs.append(diff)
    worst_overall = max(worst_overall, float(diff.max()))

  # Shape: (N_TEACHER, 93, 20)
  all_diffs = np.stack(per_sample_diffs, axis=0)
  # Max over samples and tokens: (93,)
  pos_max_diff = all_diffs.max(axis=(0, 2))
  pos_mean_diff = all_diffs.mean(axis=(0, 2))

  # Rank mapping
  rank_of_pos = np.zeros(length, dtype=int)
  for r, p in enumerate(order):
    rank_of_pos[p] = r

  # Spatial coordinates and distance to ligand
  # Extract Ca and Cb coordinates from fd["X"]
  X_torch = fd["X"][0] # (93, 4, 3): N, Ca, C, O
  Ca = X_torch[:, 1, :].numpy()
  N = X_torch[:, 0, :].numpy()
  C = X_torch[:, 2, :].numpy()
  b = Ca - N
  c = C - Ca
  a = np.cross(b, c, axis=-1)
  Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

  # Ligand coordinates: Zn atom is in fd["Y"]
  # Y has shape (1, 93, 25, 3). For 1BC8, the 1 ligand atom is at index 0 of the window where Y_m > 0
  Y = fd["Y"][0].numpy() # (93, 25, 3)
  Y_m = fd["Y_m"][0].numpy() # (93, 25)
  
  # For each residue, compute min Euclidean distance to any valid ligand atom
  lig_atom_coords = []
  for p in range(length):
    valid = Y_m[p] > 0.5
    if np.any(valid):
      lig_atom_coords.append(Y[p, valid])
  if lig_atom_coords:
    all_lig_atoms = np.unique(np.concatenate(lig_atom_coords, axis=0), axis=0) # (num_unique_atoms, 3)
  else:
    all_lig_atoms = np.array([[-8.319, -18.103, 2.886]])

  ca_lig_dist = np.min(np.linalg.norm(Ca[:, None, :] - all_lig_atoms[None, :, :], axis=-1), axis=-1)
  cb_lig_dist = np.min(np.linalg.norm(Cb[:, None, :] - all_lig_atoms[None, :, :], axis=-1), axis=-1)
  min_lig_dist = np.minimum(ca_lig_dist, cb_lig_dist)

  # Distance to padding boundary (index 92)
  dist_to_tail = (length - 1) - np.arange(length)

  # Correlations
  rank_corr, rank_pval = spearmanr(rank_of_pos, pos_max_diff)
  lig_corr, lig_pval = spearmanr(min_lig_dist, pos_max_diff)
  tail_corr, tail_pval = spearmanr(dist_to_tail, pos_max_diff)

  # Identify exceeding positions
  exceeding_indices = np.where(pos_max_diff > TOL_LOGPROB)[0].tolist()
  num_exceeding = len(exceeding_indices)

  # Identify first rank where divergence exceeds tolerance
  first_divergent_rank = length
  first_divergent_pos = -1
  for r in range(length):
    p = order[r]
    if pos_max_diff[p] > TOL_LOGPROB:
      first_divergent_rank = r
      first_divergent_pos = int(p)
      break

  # Shell analysis (<5A, 5-8A, 8-12A, >12A)
  shell_stats = {}
  shells = [
    ("<5A", min_lig_dist < 5.0),
    ("5-8A", (min_lig_dist >= 5.0) & (min_lig_dist < 8.0)),
    ("8-12A", (min_lig_dist >= 8.0) & (min_lig_dist < 12.0)),
    (">12A", min_lig_dist >= 12.0),
  ]
  for label, mask in shells:
    if np.any(mask):
      shell_stats[label] = {
        "count": int(mask.sum()),
        "max_diff": float(pos_max_diff[mask].max()),
        "mean_diff": float(pos_max_diff[mask].mean()),
        "num_exceeding": int((pos_max_diff[mask] > TOL_LOGPROB).sum()),
      }

  # Temporal quartile analysis (Q1: ranks 0-22, Q2: 23-45, Q3: 46-68, Q4: 69-92)
  quartile_stats = {}
  for q in range(4):
    q_mask = (rank_of_pos >= q * 23) & (rank_of_pos < (q + 1) * 23 if q < 3 else length)
    quartile_stats[f"Q{q+1}"] = {
      "count": int(q_mask.sum()),
      "max_diff": float(pos_max_diff[q_mask].max()),
      "mean_diff": float(pos_max_diff[q_mask].mean()),
      "num_exceeding": int((pos_max_diff[q_mask] > TOL_LOGPROB).sum()),
    }

  # Build detailed per-position report
  per_pos_details = []
  for p in range(length):
    per_pos_details.append({
      "position": int(p),
      "rank": int(rank_of_pos[p]),
      "ligand_distance_A": round(float(min_lig_dist[p]), 3),
      "max_diff": round(float(pos_max_diff[p]), 6),
      "mean_diff": round(float(pos_mean_diff[p]), 6),
      "exceeds_tol": bool(pos_max_diff[p] > TOL_LOGPROB),
    })

  # Bisection comparison: if running auto, run with off to detect incremental diff
  incremental_diff_detected = False
  inc_vs_full_max = 0.0
  if args.force_incremental == "auto":
    # Run full recompute comparison
    patch_force_incremental("off")
    _, full_logits = e2e.aminx_sample(kw, num_samples=1, seed=DIST_SEED, order=order, length=length)
    # Compare step 0 aminx sample logits between auto and off
    inc_diff = np.abs(logits[0] - full_logits[0])
    inc_vs_full_max = float(inc_diff.max())
    incremental_diff_detected = bool(inc_vs_full_max > 1e-4)

  results: dict[str, Any] = {
    "max_abs_logprob": float(worst_overall),
    "mean_abs_logprob": float(all_diffs.mean()),
    "num_positions_exceeding_tol": num_exceeding,
    "total_designed_positions": n_designed,
    "fraction_positions_exceeding_tol": float(num_exceeding / n_designed),
    "first_divergent_rank": int(first_divergent_rank),
    "first_divergent_position": int(first_divergent_pos),
    "rank_correlation": float(rank_corr) if not np.isnan(rank_corr) else 0.0,
    "rank_correlation_pvalue": float(rank_pval) if not np.isnan(rank_pval) else 1.0,
    "ligand_dist_correlation": float(lig_corr) if not np.isnan(lig_corr) else 0.0,
    "ligand_dist_correlation_pvalue": float(lig_pval) if not np.isnan(lig_pval) else 1.0,
    "tail_dist_correlation": float(tail_corr) if not np.isnan(tail_corr) else 0.0,
    "tail_dist_correlation_pvalue": float(tail_pval) if not np.isnan(tail_pval) else 1.0,
    "incremental_diff_detected": incremental_diff_detected,
    "incremental_vs_full_max_diff": inc_vs_full_max,
    "force_incremental_mode": args.force_incremental,
    "exceeding_positions": exceeding_indices,
    "shell_stats": shell_stats,
    "quartile_stats": quartile_stats,
    "per_position_details": per_pos_details,
  }

  with open(args.out, "w") as f:
    json.dump(results, f, indent=2)

  bth_path = os.environ.get("BTH_RESULTS_PATH")
  if bth_path:
    with open(bth_path, "w") as f:
      json.dump(results, f, indent=2)

  print(f"[Lane T Diagnosis] Mode={args.force_incremental}")
  print(f"[Lane T Diagnosis] max|diff|={worst_overall:.6f}, positions > 1e-3: {num_exceeding}/{n_designed}")
  print(f"[Lane T Diagnosis] first divergent rank: {first_divergent_rank} (position {first_divergent_pos})")
  print(f"[Lane T Diagnosis] rank Spearman r: {rank_corr:.3f} (p={rank_pval:.3e})")
  print(f"[Lane T Diagnosis] ligand distance Spearman r: {lig_corr:.3f} (p={lig_pval:.3e})")
  print(f"[Lane T Diagnosis] incremental vs full diff: {inc_vs_full_max:.6f} (detected={incremental_diff_detected})")

  return 0


if __name__ == "__main__":
  sys.exit(main())
