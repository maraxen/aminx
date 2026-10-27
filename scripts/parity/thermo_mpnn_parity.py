"""Bathos parity and negative-control driver for ThermoMPNN (v1/v2) and ThermoMPNN-D.

Executes:
1. Single-mutant ``ThermoMPNN`` v1 and v2 parity in ``float64`` and ``float32``
   against the PyTorch reference on ``tests/data/1ubq.pdb``.
2. Wild-type self-mutation identity check (``ddg[i, S[i]] == 0.0``).
3. Double-mutant ``ThermoMPNNDouble`` parity in ``float64`` and ``float32`` across
   neighbor-connected and distant residue pairs.
4. Siamese mutation-slot order invariance check.
5. Four negative controls (reversing ``all_mpnn_hid``, omitting ``wt - mut``
   embedding subtraction, zeroing edge features, permuting output rows).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
if str(_REPO_ROOT / "tests" / "families") not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT / "tests" / "families"))

from test_thermo_mpnn_parity import (  # noqa: E402
  _UBQ_PDB,
  _corrupt_double_omit_seq_subtraction,
  _corrupt_double_zero_edges,
  _corrupt_single_unreversed_hidden,
  _parse_pdb_backbone,
  _run_torch_double_reference,
  _run_torch_single_reference,
  _select_neighbor_and_distant_mutations,
  _x64_mode,
  resolve_thermo_mpnn_parity_inputs,
)

from aminx.families.thermo_mpnn import (  # noqa: E402
  convert_thermo_mpnn_double_state_dict,
  convert_thermo_mpnn_state_dict,
)


def run_parity_suite(*, synthetic: bool = False) -> dict[str, float | int | str]:
  """Run the ThermoMPNN / ThermoMPNN-D parity and negative-control suite."""
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  v1_sd, v2_single_sd, v2_double_sd, weight_meta = resolve_thermo_mpnn_parity_inputs(
    synthetic=synthetic,
  )

  with _x64_mode(True):
    v1_model_f64 = convert_thermo_mpnn_state_dict(v1_sd)
    v2_model_f64 = convert_thermo_mpnn_state_dict(v2_single_sd)
    double_model_f64 = convert_thermo_mpnn_double_state_dict(v2_double_sd)

    mut_positions, mut_aas, n_neighbor_pairs, n_distant_pairs = (
      _select_neighbor_and_distant_mutations(
        coords,
        mask,
        residue_index,
        chain_index,
        sequence,
        double_model_f64,
      )
    )

    coords_f64 = jnp.asarray(coords, dtype=jnp.float64)
    mask_f64 = jnp.asarray(mask, dtype=jnp.float64)
    res_i32 = jnp.asarray(residue_index, dtype=jnp.int32)
    chain_i32 = jnp.asarray(chain_index, dtype=jnp.int32)
    seq_i32 = jnp.asarray(sequence, dtype=jnp.int32)
    pos_i32 = jnp.asarray(mut_positions, dtype=jnp.int32)
    aas_i32 = jnp.asarray(mut_aas, dtype=jnp.int32)

    jax_v1_f64 = np.asarray(v1_model_f64(coords_f64, mask_f64, res_i32, chain_i32, seq_i32))
    jax_v2_f64 = np.asarray(v2_model_f64(coords_f64, mask_f64, res_i32, chain_i32, seq_i32))
    jax_double_f64 = np.asarray(
      double_model_f64(coords_f64, mask_f64, res_i32, chain_i32, seq_i32, pos_i32, aas_i32),
    )
    jax_double_swapped_f64 = np.asarray(
      double_model_f64(
        coords_f64,
        mask_f64,
        res_i32,
        chain_i32,
        seq_i32,
        pos_i32[:, ::-1],
        aas_i32[:, ::-1],
      ),
    )

    # Negative controls in float64
    neg_hid = np.asarray(
      _corrupt_single_unreversed_hidden(
        v1_model_f64,
        coords_f64,
        mask_f64,
        res_i32,
        chain_i32,
        seq_i32,
      ),
    )
    neg_seq = np.asarray(
      _corrupt_double_omit_seq_subtraction(
        double_model_f64,
        coords_f64,
        mask_f64,
        res_i32,
        chain_i32,
        seq_i32,
        pos_i32,
        aas_i32,
      ),
    )
    neg_edges = np.asarray(
      _corrupt_double_zero_edges(
        double_model_f64,
        coords_f64,
        mask_f64,
        res_i32,
        chain_i32,
        seq_i32,
        pos_i32,
        aas_i32,
      ),
    )
    last_linear = v1_model_f64.mlp_layers[-1]
    permuted_last = eqx.tree_at(
      lambda mod: mod.weight,
      last_linear,
      jnp.roll(last_linear.weight, shift=1, axis=0),
    )
    permuted_model = eqx.tree_at(
      lambda mod: mod.mlp_layers[-1],
      v1_model_f64,
      permuted_last,
    )
    neg_perm = np.asarray(permuted_model(coords_f64, mask_f64, res_i32, chain_i32, seq_i32))

  torch_v1_f64 = _run_torch_single_reference(
    v1_sd,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    dtype=torch.float64,
  )
  torch_v2_f64 = _run_torch_single_reference(
    v2_single_sd,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    dtype=torch.float64,
  )
  torch_double_f64 = _run_torch_double_reference(
    v2_double_sd,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    mut_positions,
    mut_aas,
    dtype=torch.float64,
  )

  with _x64_mode(False):
    v1_model_f32 = convert_thermo_mpnn_state_dict(v1_sd)
    v2_model_f32 = convert_thermo_mpnn_state_dict(v2_single_sd)
    double_model_f32 = convert_thermo_mpnn_double_state_dict(v2_double_sd)

    coords_f32 = jnp.asarray(coords, dtype=jnp.float32)
    mask_f32 = jnp.asarray(mask, dtype=jnp.float32)
    res_i32 = jnp.asarray(residue_index, dtype=jnp.int32)
    chain_i32 = jnp.asarray(chain_index, dtype=jnp.int32)
    seq_i32 = jnp.asarray(sequence, dtype=jnp.int32)
    pos_i32 = jnp.asarray(mut_positions, dtype=jnp.int32)
    aas_i32 = jnp.asarray(mut_aas, dtype=jnp.int32)

    jax_v1_f32 = np.asarray(v1_model_f32(coords_f32, mask_f32, res_i32, chain_i32, seq_i32))
    jax_v2_f32 = np.asarray(v2_model_f32(coords_f32, mask_f32, res_i32, chain_i32, seq_i32))
    jax_double_f32 = np.asarray(
      double_model_f32(coords_f32, mask_f32, res_i32, chain_i32, seq_i32, pos_i32, aas_i32),
    )

  torch_v1_f32 = _run_torch_single_reference(
    v1_sd,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    dtype=torch.float32,
  )
  torch_v2_f32 = _run_torch_single_reference(
    v2_single_sd,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    dtype=torch.float32,
  )
  torch_double_f32 = _run_torch_double_reference(
    v2_double_sd,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    mut_positions,
    mut_aas,
    dtype=torch.float32,
  )

  v1_f64_max_abs_diff = float(np.max(np.abs(jax_v1_f64 - torch_v1_f64)))
  v2_f64_max_abs_diff = float(np.max(np.abs(jax_v2_f64 - torch_v2_f64)))
  double_f64_max_abs_diff = float(np.max(np.abs(jax_double_f64 - torch_double_f64)))

  v1_f32_max_abs_diff = float(np.max(np.abs(jax_v1_f32 - torch_v1_f32)))
  v2_f32_max_abs_diff = float(np.max(np.abs(jax_v2_f32 - torch_v2_f32)))
  double_f32_max_abs_diff = float(np.max(np.abs(jax_double_f32 - torch_double_f32)))

  wt_v1 = jax_v1_f64[np.arange(sequence.shape[0]), sequence]
  wt_v2 = jax_v2_f64[np.arange(sequence.shape[0]), sequence]
  wt_self_max_abs_diff = float(max(np.max(np.abs(wt_v1)), np.max(np.abs(wt_v2))))

  order_invariance_max_abs_diff = float(np.max(np.abs(jax_double_f64 - jax_double_swapped_f64)))

  neg_ctrl_reversed_hidden_diff = float(np.max(np.abs(neg_hid - torch_v1_f64)))
  neg_ctrl_omit_seq_sub_diff = float(np.max(np.abs(neg_seq - torch_double_f64)))
  neg_ctrl_zero_edges_diff = float(np.max(np.abs(neg_edges - torch_double_f64)))
  neg_ctrl_permuted_out_diff = float(np.max(np.abs(neg_perm - torch_v1_f64)))
  neg_ctrl_min_max_abs_diff = float(
    min(
      neg_ctrl_reversed_hidden_diff,
      neg_ctrl_omit_seq_sub_diff,
      neg_ctrl_zero_edges_diff,
      neg_ctrl_permuted_out_diff,
    ),
  )

  return {
    "pdb_id": "1ubq",
    "num_residues": int(sequence.shape[0]),
    "n_neighbor_pairs": int(n_neighbor_pairs),
    "n_distant_pairs": int(n_distant_pairs),
    **weight_meta,
    "v1_f64_max_abs_diff": v1_f64_max_abs_diff,
    "v2_f64_max_abs_diff": v2_f64_max_abs_diff,
    "double_f64_max_abs_diff": double_f64_max_abs_diff,
    "v1_f32_max_abs_diff": v1_f32_max_abs_diff,
    "v2_f32_max_abs_diff": v2_f32_max_abs_diff,
    "double_f32_max_abs_diff": double_f32_max_abs_diff,
    "wt_self_max_abs_diff": wt_self_max_abs_diff,
    "order_invariance_max_abs_diff": order_invariance_max_abs_diff,
    "neg_ctrl_reversed_hidden_diff": neg_ctrl_reversed_hidden_diff,
    "neg_ctrl_omit_seq_sub_diff": neg_ctrl_omit_seq_sub_diff,
    "neg_ctrl_zero_edges_diff": neg_ctrl_zero_edges_diff,
    "neg_ctrl_permuted_out_diff": neg_ctrl_permuted_out_diff,
    "neg_ctrl_min_max_abs_diff": neg_ctrl_min_max_abs_diff,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Run ThermoMPNN & ThermoMPNN-D parity suite.")
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("outputs/parity/thermo_mpnn_parity.json"),
    help="Output JSON path (default: outputs/parity/thermo_mpnn_parity.json).",
  )
  parser.add_argument(
    "--synthetic",
    action="store_true",
    default=False,
    help=(
      "Run with deterministic synthetic weights instead of upstream checkpoints "
      "(records weight_source='synthetic' and cannot grade as 'pass' for promotion)."
    ),
  )
  args = parser.parse_args()

  try:
    result = run_parity_suite(synthetic=args.synthetic)
  except FileNotFoundError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  bth_results = os.environ.get("BTH_RESULTS_PATH")
  if bth_results:
    bth_path = Path(bth_results)
    bth_path.parent.mkdir(parents=True, exist_ok=True)
    bth_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
