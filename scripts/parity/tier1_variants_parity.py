"""Bathos parity and negative-control driver for Tier 1 MPNN variants.

Covers:
1. AbMPNN (Zenodo 8164693, ``abmpnn.pt``)
2. HyperMPNN (Meiler Lab, ``retrained_models/v48_{002,010,020,030}_hyper.pt``)
3. CyclicMPNN (ParisaH-Lab, ``cyclicmpnn_weights/cyclicmpnn_48_010.pt``)

Validates:
- Float64 parity against PyTorch (< 1e-7) on 1ubq.pdb
- Float32 parity against PyTorch (< 1e-4)
- 5 manifest-driven negative controls:
  * reversed_hidden
  * zero_edge_features
  * permuted_output_weights
  * shifted_decoding_order
  * linear_offsets_on_cyclic
- Mandatory hard failure (FileNotFoundError, exit 1) on missing checkpoints.
- Supports --synthetic for harness testing (never promotes).
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

from aminx.model.decoder import (  # noqa: E402
  conditional_decoder_layer_edge_features,
  pack_conditional_decoder_static_edges,
)
from aminx.model.mpnn import Aminx  # noqa: E402
from aminx.utils.autoregression import ar_mask_from_decoding_order  # noqa: E402

_REQUIRED_CHECKPOINTS = [
  ("abmpnn", "AbMPNN/abmpnn.pt"),
  ("hypermpnn_002", "HyperMPNN/retrained_models/v48_002_epoch240_hyper.pt"),
  ("hypermpnn_010", "HyperMPNN/retrained_models/v48_010_epoch300_hyper.pt"),
  ("hypermpnn_020", "HyperMPNN/retrained_models/v48_020_epoch300_hyper.pt"),
  ("hypermpnn_030", "HyperMPNN/retrained_models/v48_030_epoch300_hyper.pt"),
  ("cyclicmpnn", "CyclicMPNN/cyclicmpnn_weights/cyclicmpnn_48_010.pt"),
]


def _sha256_of_file(path: Path) -> str:
  hasher = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
      hasher.update(chunk)
  return hasher.hexdigest()


def _make_synthetic_mpnn_state_dict(seed: int = 42) -> dict[str, torch.Tensor]:
  """Construct a deterministic synthetic PyTorch state dict for ProteinMPNN."""
  rng = np.random.default_rng(seed)

  def _t(shape: tuple[int, ...]) -> torch.Tensor:
    fan_in = shape[-1] if len(shape) > 1 else shape[0]
    bound = 1.0 / np.sqrt(max(fan_in, 1))
    arr = rng.uniform(-bound, bound, size=shape).astype(np.float32)
    return torch.tensor(arr)

  sd: dict[str, torch.Tensor] = {}
  sd["features.embeddings.linear.weight"] = _t((16, 66))
  sd["features.embeddings.linear.bias"] = _t((16,))
  sd["features.edge_embedding.weight"] = _t((128, 416))
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


def run_tier1_parity_suite(
  *,
  synthetic: bool = False,
  work_dir: Path | None = None,
) -> dict[str, Any]:
  """Execute parity comparisons and negative controls for Tier 1 variants."""
  ckpt_paths: dict[str, Path] = {}
  ckpt_hashes: dict[str, str] = {}

  if not synthetic:
    missing: list[str] = []
    for variant, rel_path in _REQUIRED_CHECKPOINTS:
      resolved = find_parity_repo_file(rel_path)
      if resolved is None:
        missing.append(rel_path)
      else:
        ckpt_paths[variant] = resolved
        ckpt_hashes[rel_path] = _sha256_of_file(resolved)
    if missing:
      raise FileNotFoundError(
        f"Missing required upstream checkpoints: {', '.join(missing)}. "
        "Hard failure under bth run -- never skip.",
      )

  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(
    _UBQ_PDB,
    max_residues=36,
  )
  rng = np.random.default_rng(42)
  decoding_order = rng.permutation(coords.shape[0]).astype(np.int32)

  f64_diffs: list[float] = []
  f32_diffs: list[float] = []

  base_jax_model: Aminx | None = None

  if synthetic:
    synth_sd = _make_synthetic_mpnn_state_dict(42)
    with _x64_mode(True):
      base_jax_model = _cast_floating(convert_state_dict_to_aminx(synth_sd), jnp.float64)
      j_e, j_v, j_lp = _jax_teacher_forced_forward(
        base_jax_model,
        jnp.asarray(coords, dtype=jnp.float64),
        jnp.asarray(mask, dtype=jnp.float64),
        jnp.asarray(residue_index, dtype=jnp.int32),
        jnp.asarray(chain_index, dtype=jnp.int32),
        jnp.asarray(sequence, dtype=jnp.int32),
        jnp.asarray(decoding_order, dtype=jnp.int32),
        use_potts_ca_dist=False,
      )
    clean_f64_diff = 0.0
    clean_f32_diff = 1.2e-6
    f64_diffs.append(clean_f64_diff)
    f32_diffs.append(clean_f32_diff)
  else:
    for variant, rel_path in _REQUIRED_CHECKPOINTS:
      pt_path = ckpt_paths[variant]
      with _x64_mode(True):
        jax_model_f64 = _cast_floating(convert_mpnn_checkpoint(pt_path), jnp.float64)
        if base_jax_model is None:
          base_jax_model = jax_model_f64
        j_e, j_v, j_lp = _jax_teacher_forced_forward(
          jax_model_f64,
          jnp.asarray(coords, dtype=jnp.float64),
          jnp.asarray(mask, dtype=jnp.float64),
          jnp.asarray(residue_index, dtype=jnp.int32),
          jnp.asarray(chain_index, dtype=jnp.int32),
          jnp.asarray(sequence, dtype=jnp.int32),
          jnp.asarray(decoding_order, dtype=jnp.int32),
          use_potts_ca_dist=False,
        )
      pt_e, pt_v, pt_lp = _torch_teacher_forced_forward(
        pt_path,
        coords,
        mask,
        residue_index,
        chain_index,
        sequence,
        decoding_order,
        ca_only=False,
        dtype=torch.float64,
      )
      f64_diffs.append(float(np.max(np.abs(np.asarray(j_lp) - pt_lp))))

      with _x64_mode(False):
        jax_model_f32 = _cast_floating(convert_mpnn_checkpoint(pt_path), jnp.float32)
        j_e32, j_v32, j_lp32 = _jax_teacher_forced_forward(
          jax_model_f32,
          jnp.asarray(coords, dtype=jnp.float32),
          jnp.asarray(mask, dtype=jnp.float32),
          jnp.asarray(residue_index, dtype=jnp.int32),
          jnp.asarray(chain_index, dtype=jnp.int32),
          jnp.asarray(sequence, dtype=jnp.int32),
          jnp.asarray(decoding_order, dtype=jnp.int32),
          use_potts_ca_dist=False,
        )
      pt_e32, pt_v32, pt_lp32 = _torch_teacher_forced_forward(
        pt_path,
        coords,
        mask,
        residue_index,
        chain_index,
        sequence,
        decoding_order,
        ca_only=False,
        dtype=torch.float32,
      )
      f32_diffs.append(float(np.max(np.abs(np.asarray(j_lp32) - pt_lp32))))

  assert base_jax_model is not None

  # Clean reference forward pass for negative control baselines
  coords_f32 = jnp.asarray(coords, dtype=jnp.float32)
  mask_f32 = jnp.asarray(mask, dtype=jnp.float32)
  res_i32 = jnp.asarray(residue_index, dtype=jnp.int32)
  chain_i32 = jnp.asarray(chain_index, dtype=jnp.int32)
  seq_i32 = jnp.asarray(sequence, dtype=jnp.int32)
  dec_i32 = jnp.asarray(decoding_order, dtype=jnp.int32)

  clean_model_f32 = _cast_floating(base_jax_model, jnp.float32)
  _, _, clean_log_probs = _jax_teacher_forced_forward(
    clean_model_f32,
    coords_f32,
    mask_f32,
    res_i32,
    chain_i32,
    seq_i32,
    dec_i32,
    use_potts_ca_dist=False,
  )
  clean_lp_np = np.asarray(clean_log_probs)

  # Negative Control 1: reversed_hidden
  def _run_reversed_hidden() -> float:
    edge_f, neigh_idx, _, _ = clean_model_f32.features(
      jax.random.PRNGKey(0),
      coords_f32,
      mask_f32,
      res_i32,
      chain_i32,
      backbone_noise=jnp.asarray(0.0, dtype=coords_f32.dtype),
    )
    node_feat, enc_edges = clean_model_f32.encoder(
      edge_f,
      neigh_idx,
      mask_f32,
      inference=True,
    )
    rev_nodes = node_feat[::-1]
    one_hot_seq = jax.nn.one_hot(seq_i32, 21, dtype=jnp.float32)
    ar_mask = ar_mask_from_decoding_order(dec_i32)
    seq_edges, mask_bw, masked_ne = pack_conditional_decoder_static_edges(
      rev_nodes,
      enc_edges,
      neigh_idx,
      one_hot_seq,
      clean_model_f32.w_s_embed.weight,
      ar_mask,
      mask_f32,
    )
    h_v = rev_nodes
    for layer in clean_model_f32.decoder.layers:
      layer_edge = conditional_decoder_layer_edge_features(
        h_v,
        seq_edges,
        neigh_idx,
        mask_bw,
        masked_ne,
      )
      h_v = layer(h_v, layer_edge, mask_f32, inference=True)
    logits = h_v @ clean_model_f32.w_out.weight.T + clean_model_f32.w_out.bias
    lp = np.asarray(jax.nn.log_softmax(logits, axis=-1))
    return float(np.max(np.abs(lp - clean_lp_np)))

  # Negative Control 2: zero_edge_features
  def _run_zero_edges() -> float:
    edge_f, neigh_idx, _, _ = clean_model_f32.features(
      jax.random.PRNGKey(0),
      coords_f32,
      mask_f32,
      res_i32,
      chain_i32,
      backbone_noise=jnp.asarray(0.0, dtype=coords_f32.dtype),
    )
    node_feat, _ = clean_model_f32.encoder(
      edge_f,
      neigh_idx,
      mask_f32,
      inference=True,
    )
    zero_enc_edges = jnp.zeros_like(edge_f)
    one_hot_seq = jax.nn.one_hot(seq_i32, 21, dtype=jnp.float32)
    ar_mask = ar_mask_from_decoding_order(dec_i32)
    seq_edges, mask_bw, masked_ne = pack_conditional_decoder_static_edges(
      node_feat,
      zero_enc_edges,
      neigh_idx,
      one_hot_seq,
      clean_model_f32.w_s_embed.weight,
      ar_mask,
      mask_f32,
    )
    h_v = node_feat
    for layer in clean_model_f32.decoder.layers:
      layer_edge = conditional_decoder_layer_edge_features(
        h_v,
        seq_edges,
        neigh_idx,
        mask_bw,
        masked_ne,
      )
      h_v = layer(h_v, layer_edge, mask_f32, inference=True)
    logits = h_v @ clean_model_f32.w_out.weight.T + clean_model_f32.w_out.bias
    lp = np.asarray(jax.nn.log_softmax(logits, axis=-1))
    return float(np.max(np.abs(lp - clean_lp_np)))

  # Negative Control 3: permuted_output_weights
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
      use_potts_ca_dist=False,
    )
    return float(np.max(np.abs(np.asarray(bad_lp) - clean_lp_np)))

  # Negative Control 4: shifted_decoding_order
  def _run_shifted_order() -> float:
    shifted_dec = jnp.roll(dec_i32, shift=1)
    _, _, bad_lp = _jax_teacher_forced_forward(
      clean_model_f32,
      coords_f32,
      mask_f32,
      res_i32,
      chain_i32,
      seq_i32,
      shifted_dec,
      use_potts_ca_dist=False,
    )
    return float(np.max(np.abs(np.asarray(bad_lp) - clean_lp_np)))

  # Negative Control 5: linear_offsets_on_cyclic
  def _run_linear_offsets_on_cyclic() -> float:
    def _forward_edges(ef: jax.Array, ni: jax.Array) -> np.ndarray:
      node_f, enc_e = clean_model_f32.encoder(ef, ni, mask_f32, inference=True)
      one_hot_s = jax.nn.one_hot(seq_i32, 21, dtype=coords_f32.dtype)
      ar_m = ar_mask_from_decoding_order(dec_i32.astype(jnp.int32))
      seq_e, m_bw, m_ne = pack_conditional_decoder_static_edges(
        node_f,
        enc_e,
        ni,
        one_hot_s,
        clean_model_f32.w_s_embed.weight,
        ar_m,
        mask_f32,
      )
      h = node_f
      for dec_layer in clean_model_f32.decoder.layers:
        le = conditional_decoder_layer_edge_features(
          h,
          seq_e,
          ni,
          m_bw,
          m_ne,
        )
        h = dec_layer(h, le, mask_f32, inference=True)
      logits = h @ clean_model_f32.w_out.weight.T + clean_model_f32.w_out.bias
      return np.asarray(jax.nn.log_softmax(logits, axis=-1))

    ef_cyclic, ni_cyclic, _, _ = clean_model_f32.features(
      jax.random.PRNGKey(0),
      coords_f32,
      mask_f32,
      res_i32,
      chain_i32,
      backbone_noise=jnp.asarray(0.0, dtype=coords_f32.dtype),
      cyclic=True,
    )
    ef_linear, ni_linear, _, _ = clean_model_f32.features(
      jax.random.PRNGKey(0),
      coords_f32,
      mask_f32,
      res_i32,
      chain_i32,
      backbone_noise=jnp.asarray(0.0, dtype=coords_f32.dtype),
      cyclic=False,
    )
    cyclic_lp = _forward_edges(ef_cyclic, ni_cyclic)
    linear_lp = _forward_edges(ef_linear, ni_linear)
    return float(np.max(np.abs(cyclic_lp - linear_lp)))

  diff_rev = _run_reversed_hidden()
  diff_zero = _run_zero_edges()
  diff_perm = _run_permuted_output()
  diff_shift = _run_shifted_order()
  diff_cyclic = _run_linear_offsets_on_cyclic()

  statuses: dict[str, str] = {
    "reversed_hidden": "failed" if diff_rev > 1e-2 else "passed",
    "zero_edge_features": "failed" if diff_zero > 1e-2 else "passed",
    "permuted_output_weights": "failed" if diff_perm > 1e-2 else "passed",
    "shifted_decoding_order": "failed" if diff_shift > 1e-2 else "passed",
    "linear_offsets_on_cyclic": "failed" if diff_cyclic > 1e-3 else "passed",
  }

  max_f64 = max(f64_diffs)
  max_f32 = max(f32_diffs)
  min_neg = min(diff_rev, diff_zero, diff_perm, diff_shift, diff_cyclic)

  clean_grade = "pass"
  if synthetic:
    clean_grade = "fail"
  elif max_f64 >= 1e-7 or max_f32 >= 1e-4:
    clean_grade = "fail"

  result: dict[str, Any] = {
    "clean": clean_grade,
    "weight_source": "synthetic" if synthetic else "upstream_checkpoint",
    "f64_max_abs_diff": max_f64,
    "f32_max_abs_diff": max_f32,
    "neg_ctrl_reversed_hidden_diff": diff_rev,
    "neg_ctrl_zero_edge_features_diff": diff_zero,
    "neg_ctrl_permuted_output_weights_diff": diff_perm,
    "neg_ctrl_shifted_decoding_order_diff": diff_shift,
    "neg_ctrl_linear_offsets_on_cyclic_diff": diff_cyclic,
    "neg_ctrl_min_max_abs_diff": min_neg,
    "_statuses": statuses,
    "_weights": ckpt_hashes,
  }
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description="Run Tier 1 MPNN variants parity suite.")
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("outputs/parity/tier1_variants_parity.json"),
    help="Output JSON path (default: outputs/parity/tier1_variants_parity.json).",
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
    default="tier1_reversed_hidden,tier1_zero_edge_features,tier1_permuted_output_weights,tier1_shifted_decoding_order,tier1_linear_offsets_on_cyclic",
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
    result = run_tier1_parity_suite(synthetic=args.synthetic, work_dir=args.work_dir)
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
    if base.startswith("tier1_"):
      base = base[len("tier1_"):]
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
