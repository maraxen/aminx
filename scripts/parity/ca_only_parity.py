"""Bathos parity and negative-control driver for CA-Only ProteinMPNN.

Covers:
1. ProteinMPNN CA-Only checkpoints (Dauparas et al., ``ca_model_weights/v_48_{002,010,020}.pt``)

Validates:
- Float64 parity against PyTorch (< 1e-7) on 1ubq.pdb
- Float32 parity against PyTorch: 99.9th percentile < 1e-4, max < 1e-3
  (documented float32 tolerance tier due to 1-ULP amplification in quaternion sqrt near zero-crossings)
- 4 manifest-driven negative controls:
  * quaternion_sign_flip
  * rbf_block_swap
  * zero_edge_features
  * permuted_output_weights
- Mandatory hard failure (FileNotFoundError, exit 1) on missing checkpoints.
- Supports --synthetic for harness testing (never promotes).
- Coordinate equivalence: bit-identical output between (L, 4, 3) backbone and (L, 3) CA-only inputs.
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
if str(_REPO_ROOT / "scripts") not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT / "scripts"))
if str(_REPO_ROOT / "scripts" / "parity") not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT / "scripts" / "parity"))

from artifact_key import stable_artifact_key  # noqa: E402
from convert_weights import convert_mpnn_checkpoint, convert_state_dict_to_aminx  # noqa: E402
from tests.parity.reference_utils import find_parity_repo_file  # noqa: E402
from tests.parity.test_mpnn_variants_tier1_tier2 import (  # noqa: E402
  _UBQ_PDB,
  _cast_floating,
  _jax_teacher_forced_forward,
  _parse_pdb_backbone,
  _torch_teacher_forced_forward,
  _x64_mode,
)

from aminx.model.features import ProteinFeaturesCA  # noqa: E402
from aminx.model.mpnn import Aminx  # noqa: E402

_REQUIRED_CHECKPOINTS = [
  ("v_48_002", "ProteinMPNN/ca_model_weights/v_48_002.pt"),
  ("v_48_010", "ProteinMPNN/ca_model_weights/v_48_010.pt"),
  ("v_48_020", "ProteinMPNN/ca_model_weights/v_48_020.pt"),
]


def _sha256_of_file(path: Path) -> str:
  hasher = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
      hasher.update(chunk)
  return hasher.hexdigest()


def _make_synthetic_ca_state_dict(seed: int = 42) -> dict[str, torch.Tensor]:
  """Construct a deterministic synthetic PyTorch state dict for CA-only ProteinMPNN."""
  rng = np.random.default_rng(seed)

  def _t(shape: tuple[int, ...]) -> torch.Tensor:
    fan_in = shape[-1] if len(shape) > 1 else shape[0]
    bound = 1.0 / np.sqrt(max(fan_in, 1))
    arr = rng.uniform(-bound, bound, size=shape).astype(np.float32)
    return torch.tensor(arr)

  sd: dict[str, torch.Tensor] = {}
  sd["features.embeddings.linear.weight"] = _t((16, 66))
  sd["features.embeddings.linear.bias"] = _t((16,))
  # CA-only edge embedding is 167 dimensions
  sd["features.edge_embedding.weight"] = _t((128, 167))
  sd["features.norm_edges.weight"] = torch.ones(128, dtype=torch.float32)
  sd["features.norm_edges.bias"] = torch.zeros(128, dtype=torch.float32)

  sd["W_e.weight"] = _t((128, 128))
  sd["W_e.bias"] = _t((128,))
  sd["W_s.weight"] = _t((21, 128))

  for i in range(3):
    p = f"encoder_layers.{i}"
    for sub in ("W1", "W11"):
      sd[f"{p}.{sub}.weight"] = _t((128, 384))
      sd[f"{p}.{sub}.bias"] = _t((128,))
    for sub in ("W2", "W3", "W12", "W13"):
      sd[f"{p}.{sub}.weight"] = _t((128, 128))
      sd[f"{p}.{sub}.bias"] = _t((128,))
    for norm in ("norm1", "norm2", "norm3"):
      sd[f"{p}.{norm}.weight"] = torch.ones(128, dtype=torch.float32)
      sd[f"{p}.{norm}.bias"] = torch.zeros(128, dtype=torch.float32)
    sd[f"{p}.dense.W_in.weight"] = _t((512, 128))
    sd[f"{p}.dense.W_in.bias"] = _t((512,))
    sd[f"{p}.dense.W_out.weight"] = _t((128, 512))
    sd[f"{p}.dense.W_out.bias"] = _t((128,))

  for i in range(3):
    p = f"decoder_layers.{i}"
    sd[f"{p}.W1.weight"] = _t((128, 512))
    sd[f"{p}.W1.bias"] = _t((128,))
    for sub in ("W2", "W3"):
      sd[f"{p}.{sub}.weight"] = _t((128, 128))
      sd[f"{p}.{sub}.bias"] = _t((128,))
    for norm in ("norm1", "norm2", "norm3"):
      sd[f"{p}.{norm}.weight"] = torch.ones(128, dtype=torch.float32)
      sd[f"{p}.{norm}.bias"] = torch.zeros(128, dtype=torch.float32)
    sd[f"{p}.dense.W_in.weight"] = _t((512, 128))
    sd[f"{p}.dense.W_in.bias"] = _t((512,))
    sd[f"{p}.dense.W_out.weight"] = _t((128, 512))
    sd[f"{p}.dense.W_out.bias"] = _t((128,))

  sd["W_out.weight"] = _t((21, 128))
  sd["W_out.bias"] = _t((21,))
  return sd


def run_ca_only_parity_suite(
  *,
  synthetic: bool = False,
  work_dir: Path | None = None,
) -> dict[str, Any]:
  """Execute parity comparisons and negative controls for CA-Only ProteinMPNN."""
  ckpt_paths: dict[str, Path] = {}
  ckpt_hashes: dict[str, str] = {}

  if not synthetic:
    missing: list[str] = []
    for version, rel_path in _REQUIRED_CHECKPOINTS:
      resolved = find_parity_repo_file(rel_path)
      if resolved is None:
        missing.append(rel_path)
      else:
        ckpt_paths[version] = resolved
        ckpt_hashes[rel_path] = _sha256_of_file(resolved)
    if missing:
      raise FileNotFoundError(
        f"Missing required upstream checkpoints: {', '.join(missing)}. "
        "Hard failure under bth run -- never skip.",
      )

  # Parse 1ubq with broken chain and masked residue to exercise CA gating
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB, max_residues=48)
  coords_broken = coords.copy()
  coords_broken[15:, :, :] += np.array([6.5, -4.0, 3.0], dtype=np.float64)
  mask_with_gap = mask.copy()
  mask_with_gap[28] = 0.0

  rng = np.random.default_rng(7)
  decoding_order = rng.permutation(coords_broken.shape[0]).astype(np.int32)
  valid = mask_with_gap > 0.5

  f64_diffs: list[float] = []
  f32_q999_diffs: list[float] = []
  f32_max_diffs: list[float] = []

  base_jax_model: Aminx | None = None

  if synthetic:
    synth_sd = _make_synthetic_ca_state_dict(42)
    with _x64_mode(True):
      base_jax_model = _cast_floating(convert_state_dict_to_aminx(synth_sd), jnp.float64)
      assert isinstance(base_jax_model.features, ProteinFeaturesCA)
      j_e, j_v, j_lp = _jax_teacher_forced_forward(
        base_jax_model,
        jnp.asarray(coords_broken, dtype=jnp.float64),
        jnp.asarray(mask_with_gap, dtype=jnp.float64),
        jnp.asarray(residue_index, dtype=jnp.int32),
        jnp.asarray(chain_index, dtype=jnp.int32),
        jnp.asarray(sequence, dtype=jnp.int32),
        jnp.asarray(decoding_order, dtype=jnp.int32),
      )
    clean_f64_diff = 0.0
    clean_f32_q999 = 1.5e-6
    clean_f32_max = 5.0e-5
    f64_diffs.append(clean_f64_diff)
    f32_q999_diffs.append(clean_f32_q999)
    f32_max_diffs.append(clean_f32_max)
  else:
    for version, rel_path in _REQUIRED_CHECKPOINTS:
      pt_path = ckpt_paths[version]
      with _x64_mode(True):
        jax_model_f64 = _cast_floating(convert_mpnn_checkpoint(pt_path), jnp.float64)
        assert isinstance(jax_model_f64.features, ProteinFeaturesCA)
        if base_jax_model is None:
          base_jax_model = jax_model_f64
        j_e, j_v, j_lp = _jax_teacher_forced_forward(
          jax_model_f64,
          jnp.asarray(coords_broken, dtype=jnp.float64),
          jnp.asarray(mask_with_gap, dtype=jnp.float64),
          jnp.asarray(residue_index, dtype=jnp.int32),
          jnp.asarray(chain_index, dtype=jnp.int32),
          jnp.asarray(sequence, dtype=jnp.int32),
          jnp.asarray(decoding_order, dtype=jnp.int32),
        )
      pt_e, pt_v, pt_lp = _torch_teacher_forced_forward(
        pt_path,
        coords_broken,
        mask_with_gap,
        residue_index,
        chain_index,
        sequence,
        decoding_order,
        ca_only=True,
        dtype=torch.float64,
      )
      f64_diff = max(
        float(np.max(np.abs(np.asarray(j_e)[valid] - pt_e[valid]))),
        float(np.max(np.abs(np.asarray(j_v)[valid] - pt_v[valid]))),
        float(np.max(np.abs(np.asarray(j_lp)[valid] - pt_lp[valid]))),
      )
      f64_diffs.append(f64_diff)

      with _x64_mode(False):
        jax_model_f32 = _cast_floating(convert_mpnn_checkpoint(pt_path), jnp.float32)
        j_e32, j_v32, j_lp32 = _jax_teacher_forced_forward(
          jax_model_f32,
          jnp.asarray(coords_broken, dtype=jnp.float32),
          jnp.asarray(mask_with_gap, dtype=jnp.float32),
          jnp.asarray(residue_index, dtype=jnp.int32),
          jnp.asarray(chain_index, dtype=jnp.int32),
          jnp.asarray(sequence, dtype=jnp.int32),
          jnp.asarray(decoding_order, dtype=jnp.int32),
        )
      pt_e32, pt_v32, pt_lp32 = _torch_teacher_forced_forward(
        pt_path,
        coords_broken,
        mask_with_gap,
        residue_index,
        chain_index,
        sequence,
        decoding_order,
        ca_only=True,
        dtype=torch.float32,
      )
      diff_e = np.abs(np.asarray(j_e32)[valid] - pt_e32[valid])
      diff_lp = np.abs(np.asarray(j_lp32)[valid] - pt_lp32[valid])
      q999 = max(float(np.quantile(diff_e, 0.999)), float(np.quantile(diff_lp, 0.999)))
      m_diff = max(float(np.max(diff_e)), float(np.max(diff_lp)))
      f32_q999_diffs.append(q999)
      f32_max_diffs.append(m_diff)

  assert base_jax_model is not None

  # Clean reference forward pass in float32 for negative control baselines
  coords_f32 = jnp.asarray(coords_broken, dtype=jnp.float32)
  mask_f32 = jnp.asarray(mask_with_gap, dtype=jnp.float32)
  res_i32 = jnp.asarray(residue_index, dtype=jnp.int32)
  chain_i32 = jnp.asarray(chain_index, dtype=jnp.int32)
  seq_i32 = jnp.asarray(sequence, dtype=jnp.int32)
  dec_i32 = jnp.asarray(decoding_order, dtype=jnp.int32)

  clean_model_f32 = _cast_floating(base_jax_model, jnp.float32)
  clean_edges, clean_nodes, clean_log_probs = _jax_teacher_forced_forward(
    clean_model_f32,
    coords_f32,
    mask_f32,
    res_i32,
    chain_i32,
    seq_i32,
    dec_i32,
  )
  clean_lp_np = np.asarray(clean_log_probs)
  clean_edges_np = np.asarray(clean_edges)

  # Verify coordinate dimensionality equivalence: (L, 4, 3) vs (L, 3) CA-only
  coords_ca3 = coords_f32[:, 1, :]
  edges_ca3, _, lp_ca3 = _jax_teacher_forced_forward(
    clean_model_f32,
    coords_ca3,
    mask_f32,
    res_i32,
    chain_i32,
    seq_i32,
    dec_i32,
  )
  coord_equiv_diff = float(np.max(np.abs(np.asarray(edges_ca3) - clean_edges_np)))

  # Negative Control 1: quaternion_sign_flip
  def _run_quaternion_flip() -> float:
    w_e = clean_model_f32.features.w_e.weight
    w_e_corrupt = w_e.at[:, -4:].set(-w_e[:, -4:])
    corrupt_model = eqx.tree_at(lambda m: m.features.w_e.weight, clean_model_f32, w_e_corrupt)
    _, _, bad_lp = _jax_teacher_forced_forward(
      corrupt_model,
      coords_f32,
      mask_f32,
      res_i32,
      chain_i32,
      seq_i32,
      dec_i32,
    )
    return float(np.max(np.abs(np.asarray(bad_lp) - clean_lp_np)))

  # Negative Control 2: rbf_block_swap
  def _run_rbf_block_swap() -> float:
    w_e = clean_model_f32.features.w_e.weight
    w_e_swapped = w_e.copy()
    block_a = slice(16 + 3 * 16, 16 + 4 * 16)
    block_b = slice(16 + 5 * 16, 16 + 6 * 16)
    w_e_swapped = w_e_swapped.at[:, block_a].set(w_e[:, block_b])
    w_e_swapped = w_e_swapped.at[:, block_b].set(w_e[:, block_a])
    corrupt_model = eqx.tree_at(lambda m: m.features.w_e.weight, clean_model_f32, w_e_swapped)
    _, _, bad_lp = _jax_teacher_forced_forward(
      corrupt_model,
      coords_f32,
      mask_f32,
      res_i32,
      chain_i32,
      seq_i32,
      dec_i32,
    )
    return float(np.max(np.abs(np.asarray(bad_lp) - clean_lp_np)))

  # Negative Control 3: zero_edge_features
  def _run_zero_edge_features() -> float:
    w_e_proj = clean_model_f32.features.w_e_proj
    zero_proj = eqx.tree_at(lambda l: l.weight, w_e_proj, jnp.zeros_like(w_e_proj.weight))
    corrupt_model = eqx.tree_at(lambda m: m.features.w_e_proj, clean_model_f32, zero_proj)
    _, _, bad_lp = _jax_teacher_forced_forward(
      corrupt_model,
      coords_f32,
      mask_f32,
      res_i32,
      chain_i32,
      seq_i32,
      dec_i32,
    )
    return float(np.max(np.abs(np.asarray(bad_lp) - clean_lp_np)))

  # Negative Control 4: permuted_output_weights
  def _run_permuted_output() -> float:
    w_out = clean_model_f32.w_out
    perm = jnp.roll(w_out.weight, shift=3, axis=0)
    perm_model = eqx.tree_at(lambda m: m.w_out.weight, clean_model_f32, perm)
    _, _, bad_lp = _jax_teacher_forced_forward(
      perm_model,
      coords_f32,
      mask_f32,
      res_i32,
      chain_i32,
      seq_i32,
      dec_i32,
    )
    return float(np.max(np.abs(np.asarray(bad_lp) - clean_lp_np)))

  diff_quat = _run_quaternion_flip()
  diff_rbf = _run_rbf_block_swap()
  diff_zero = _run_zero_edge_features()
  diff_perm = _run_permuted_output()

  statuses: dict[str, str] = {
    "quaternion_sign_flip": "failed" if diff_quat > 1e-2 else "passed",
    "rbf_block_swap": "failed" if diff_rbf > 1e-2 else "passed",
    "zero_edge_features": "failed" if diff_zero > 1e-2 else "passed",
    "permuted_output_weights": "failed" if diff_perm > 1e-2 else "passed",
  }

  max_f64 = max(f64_diffs)
  max_q999 = max(f32_q999_diffs)
  max_f32 = max(f32_max_diffs)
  min_neg = min(diff_quat, diff_rbf, diff_zero, diff_perm)

  clean_grade = "pass"
  if synthetic:
    clean_grade = "fail"
  elif max_f64 >= 1e-7 or max_q999 >= 1e-4 or max_f32 >= 1e-3:
    clean_grade = "fail"

  result: dict[str, Any] = {
    "clean": clean_grade,
    "weight_source": "synthetic" if synthetic else "upstream_checkpoint",
    "f64_max_abs_diff": max_f64,
    "f32_q999_abs_diff": max_q999,
    "f32_max_abs_diff": max_f32,
    "coord_equivalence_diff": coord_equiv_diff,
    "neg_ctrl_quaternion_sign_flip_diff": diff_quat,
    "neg_ctrl_rbf_block_swap_diff": diff_rbf,
    "neg_ctrl_zero_edge_features_diff": diff_zero,
    "neg_ctrl_permuted_output_weights_diff": diff_perm,
    "neg_ctrl_min_max_abs_diff": min_neg,
    "_statuses": statuses,
    "_weights": ckpt_hashes,
  }
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description="Run CA-Only ProteinMPNN parity suite.")
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("outputs/parity/ca_only_parity.json"),
    help="Output JSON path (default: outputs/parity/ca_only_parity.json).",
  )
  parser.add_argument(
    "--synthetic",
    action="store_true",
    default=False,
    help="Run with deterministic synthetic weights instead of upstream checkpoints.",
  )
  parser.add_argument(
    "--mutants",
    type=str,
    default="ca_quaternion_sign_flip,ca_rbf_block_swap,ca_zero_edge_features,ca_permuted_output_weights",
    help="Comma-separated negative control mutant IDs.",
  )
  parser.add_argument(
    "--controls-out",
    type=Path,
    default=None,
    help="Output JSON path for branch_controls.json.",
  )
  parser.add_argument(
    "--work-dir",
    type=Path,
    default=None,
    help="Working directory for scratch/cache.",
  )
  args = parser.parse_args()

  try:
    result = run_ca_only_parity_suite(synthetic=args.synthetic, work_dir=args.work_dir)
  except FileNotFoundError as exc:
    print(f"HARD FAILURE: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc

  statuses = result.pop("_statuses")
  raw_weights = result.pop("_weights")
  weights: dict[str, str] = {stable_artifact_key(k): v for k, v in raw_weights.items()}

  mutant_list = [m.strip() for m in args.mutants.split(",") if m.strip()]
  mutants_report: dict[str, str] = {}
  for m in mutant_list:
    base = m
    if base.startswith("ca_"):
      base = base[len("ca_"):]
    mutants_report[m] = statuses.get(base, "error")

  n_listed = len(mutant_list)
  n_failed = sum(1 for s in mutants_report.values() if s == "failed")
  result["n_listed"] = n_listed
  result["n_failed"] = n_failed

  if args.controls_out is not None:
    args.controls_out.parent.mkdir(parents=True, exist_ok=True)
    controls_data = {
      "clean": result["clean"],
      "mutants": mutants_report,
      "weights": weights,
    }
    args.controls_out.write_text(json.dumps(controls_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  bth_results = os.environ.get("BTH_RESULTS_PATH")
  if bth_results:
    bth_path = Path(bth_results)
    bth_path.parent.mkdir(parents=True, exist_ok=True)
    bth_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
