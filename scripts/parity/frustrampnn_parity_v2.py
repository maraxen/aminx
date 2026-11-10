"""Bathos parity vehicle and negative controls for FrustraMPNN against sealed oracle dumps (v2).

Governed by scripts/parity/frustrampnn_parity_v2.bth.toml.

Evaluates:
1. Anti-self-trust pinned hash verification against external registry and dump pin files:
   - frustrampnn_1ubq_oracle.npz
   - megascale_default_la-true_esm-False_split1.ckpt
   Missing stem or hash mismatch raises RuntimeError immediately.
2. Clean arms:
   - float32 (tolerance < 1e-4)
   - float64 (tolerance < 1e-7 via precision cast)
3. Stage-by-stage forward decomposition against sealed oracle dump:
   - s0_prep: driver.prepare_structure(1ubq.pdb)
   - s1_features: ProteinMPNN raw features (76, 48, 128)
   - s2_encoder: Encoder output (76, 128)
   - s3_decoder: Autoregressive decoder hidden states (3, 76, 128)
   - s4_latent: lin_input concatenated latent vector (76, 512)
   - s5_attention: LightAttention1D output (76, 512)
   - s6_mlp_head: both_out MLP output (76, 21)
   - s7_frustration: frustration_scores (76, 21)
   - s8_entry: Public runner.score(inputs="1ubq.pdb", model_family="frustrampnn", output_kind="frustration")
4. Four negative control mutants (mapped to tests/knob_gate/branch_manifest.toml):
   - reversed_hidden: inverts decoder layer concatenation order
   - refinement_bypass: bypasses LightAttention1D with identity skip
   - permuted_output_weights: permutes readout MLP output layer weights
   - subtract_mut_inverted: inverts subtract_mut flag to force wild-type subtraction
5. Strict finite-trip and crash-free requirements:
   - Mutant tripped iff deviation is strictly finite and > 1e-2.
   - Any exception, NaN, Inf, or shape mismatch is marked crashed.
   - n_crashed == 0 required for promotion.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# Enable float64 support in JAX
jax.config.update("jax_enable_x64", val=True)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "src", Path(__file__).resolve().parent):
  if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))


import aminx.families.frustra_mpnn.model as frustra_model
from aminx.families.frustra_mpnn.driver import FrustraMPNNDriver
from aminx.families.frustra_mpnn.model import (
  FrustraMPNN,
  FrustraStageOutputs,
  LightAttention1D,
)
from aminx.host.runner import score as runner_score

log = logging.getLogger("frustrampnn_parity")

BANDS = {
  "float64": 1e-7,
  "float32": 1e-4,
}

CLEAN_ARMS = ("float32", "float64")

MUTANTS = (
  "reversed_hidden",
  "refinement_bypass",
  "permuted_output_weights",
  "subtract_mut_inverted",
)

MUTANT_MIN_POWER = 1e-2
CANONICAL_CKPT = "megascale_default_la-true_esm-False_split1"
ORACLE_STEM = "frustrampnn_1ubq_oracle"


def sha256_file(path: Path) -> str:
  """Compute SHA-256 hex digest of file."""
  hasher = hashlib.sha256()
  with path.open("rb") as f:
    while chunk := f.read(65536):
      hasher.update(chunk)
  return hasher.hexdigest()


def stage_band(name: str, dtype: str) -> float:
  """Return tolerance band for stage and precision arm."""
  if name == "s0_prep":
    return 1e-4
  return BANDS.get(dtype, 1e-4)


# ---------------------------------------------------------------------------------------------
# Anti-self-trust Verification
# ---------------------------------------------------------------------------------------------
def verify_anti_self_trust(
  oracle_dir: Path,
  weights_dir: Path,
  pinned_dumps_path: Path,
) -> tuple[str, str, Path, dict[str, str]]:
  """Fail-loud anti-self-trust verification of oracle dump and weights against external pins."""
  if not pinned_dumps_path.is_file():
    msg = f"Pinned dumps file not found: {pinned_dumps_path} (refusing to self-trust)"
    raise RuntimeError(msg)

  pinned = json.loads(pinned_dumps_path.read_text(encoding="utf-8"))
  if ORACLE_STEM not in pinned and f"{ORACLE_STEM}.npz" not in pinned:
    msg = (
      f"{ORACLE_STEM}: missing from pinned-dumps file {pinned_dumps_path} (refusing to self-trust)"
    )
    raise RuntimeError(msg)

  expected_oracle_sha = pinned.get(ORACLE_STEM) or pinned.get(f"{ORACLE_STEM}.npz")

  # Inspect tests/knob_gate/checkpoint_registry.json
  registry_path = _REPO_ROOT / "tests" / "knob_gate" / "checkpoint_registry.json"
  if not registry_path.is_file():
    msg = f"Checkpoint registry not found: {registry_path} (refusing to self-trust)"
    raise RuntimeError(msg)

  registry_data = json.loads(registry_path.read_text(encoding="utf-8"))
  entries = registry_data.get("entries")
  if not entries:
    msg = f"Checkpoint registry {registry_path} contains no entries (refusing to self-trust)"
    raise RuntimeError(msg)

  expected_ckpt_sha: str | None = None
  reg_oracle_sha: str | None = None
  for entry in entries:
    art_path = entry.get("artifact_path", "")
    if CANONICAL_CKPT in art_path:
      expected_ckpt_sha = entry.get("sha256")
    elif ORACLE_STEM in art_path:
      reg_oracle_sha = entry.get("sha256")

  if expected_ckpt_sha is None:
    msg = (
      f"Canonical checkpoint {CANONICAL_CKPT} missing from checkpoint registry {registry_path} "
      "(refusing to self-trust)"
    )
    raise RuntimeError(msg)

  if reg_oracle_sha is None:
    msg = (
      f"Oracle dump {ORACLE_STEM} missing from checkpoint registry {registry_path} "
      "(refusing to self-trust)"
    )
    raise RuntimeError(msg)

  if reg_oracle_sha != expected_oracle_sha:
    msg = (
      f"Checkpoint registry sha256 ({reg_oracle_sha}) does not match "
      f"pinned dumps sha256 ({expected_oracle_sha}) (refusing to self-trust)"
    )
    raise RuntimeError(msg)

  npz_candidates = [
    oracle_dir / f"{ORACLE_STEM}.npz",
    _REPO_ROOT / "tests" / "data" / "fixtures" / f"{ORACLE_STEM}.npz",
  ]
  npz_file = next((p for p in npz_candidates if p.is_file()), None)
  if npz_file is None:
    msg = f"Oracle dump {ORACLE_STEM}.npz not found in candidate paths (refusing to self-trust)"
    raise RuntimeError(msg)

  oracle_sha = sha256_file(npz_file)
  if oracle_sha != expected_oracle_sha:
    msg = (
      f"Oracle dump SHA256 mismatch for {npz_file}: "
      f"computed {oracle_sha} != pinned {expected_oracle_sha} (refusing to self-trust)"
    )
    raise RuntimeError(msg)

  ckpt_candidates = [
    weights_dir / f"{CANONICAL_CKPT}.ckpt",
    weights_dir / "FrustraMPNN" / "model_weights" / f"{CANONICAL_CKPT}.ckpt",
    _REPO_ROOT / "tests" / "data" / "fixtures" / f"{CANONICAL_CKPT}.ckpt",
    _REPO_ROOT / "outputs" / "FrustraMPNN" / "model_weights" / f"{CANONICAL_CKPT}.ckpt",
    Path.home() / "repos" / "FrustraMPNN" / "model_weights" / f"{CANONICAL_CKPT}.ckpt",
    Path("/tmp/repos/frustraMPNN/model_weights") / f"{CANONICAL_CKPT}.ckpt",
  ]
  for env_var in ("AMINX_PARITY_REPOS", "AMINX_FRUSTRAMPNN_ROOT"):
    if env_val := os.environ.get(env_var):
      cand_dir = Path(env_val)
      ckpt_candidates.extend([
        cand_dir / "FrustraMPNN" / "model_weights" / f"{CANONICAL_CKPT}.ckpt",
        cand_dir / "model_weights" / f"{CANONICAL_CKPT}.ckpt",
        cand_dir / f"{CANONICAL_CKPT}.ckpt",
      ])
  ckpt_file = next((p for p in ckpt_candidates if p.is_file()), None)
  if ckpt_file is None:
    msg = f"Checkpoint {CANONICAL_CKPT}.ckpt not found in candidate paths (refusing to self-trust)"
    raise RuntimeError(msg)

  ckpt_sha = sha256_file(ckpt_file)
  if ckpt_sha != expected_ckpt_sha:
    msg = (
      f"Checkpoint SHA256 mismatch for {ckpt_file}: "
      f"computed {ckpt_sha} != pinned {expected_ckpt_sha} (refusing to self-trust)"
    )
    raise RuntimeError(msg)

  return oracle_sha, ckpt_sha, ckpt_file, pinned


# ---------------------------------------------------------------------------------------------
# Mutant Application Context Manager
# ---------------------------------------------------------------------------------------------
@contextlib.contextmanager
def apply_mutant(
  mutant: str | None,
  model: FrustraMPNN,
) -> Iterator[FrustraMPNN]:
  """Context-manager applying targeted negative control mutations."""
  if mutant is None or mutant == "clean":
    yield model
    return

  if mutant == "reversed_hidden":
    # 1. reversed_hidden: Invert decoder hidden concatenation order ([dec0, dec1, dec2] instead of [dec2, dec1, dec0])
    orig_trunk_fn = frustra_model.frustra_mpnn_trunk

    def corrupted_trunk(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
      tr = orig_trunk_fn(*args, **kwargs)
      # tr.all_mpnn_hid has [dec2, dec1, dec0]
      corrupted = (tr.all_mpnn_hid[2], tr.all_mpnn_hid[1], tr.all_mpnn_hid[0])
      return tr._replace(all_mpnn_hid=corrupted)

    frustra_model.frustra_mpnn_trunk = corrupted_trunk
    try:
      yield model
    finally:
      frustra_model.frustra_mpnn_trunk = orig_trunk_fn

  elif mutant == "refinement_bypass":
    # 2. refinement_bypass: Bypass LightAttention1D with identity skip
    orig_call = LightAttention1D.__call__
    LightAttention1D.__call__ = lambda _self, x: x  # type: ignore[assignment]
    try:
      yield model
    finally:
      LightAttention1D.__call__ = orig_call  # type: ignore[assignment]

  elif mutant == "permuted_output_weights":
    # 3. permuted_output_weights: Channel permutation / roll on both_out.layers[2].weight
    w = model.both_out.layers[2].weight
    w_perm = jnp.roll(w, shift=1, axis=0)
    mutated_model = eqx.tree_at(lambda m: m.both_out.layers[2].weight, model, w_perm)
    yield mutated_model

  elif mutant == "subtract_mut_inverted":
    # 4. subtract_mut_inverted: Invert subtract_mut boolean (False -> True)
    orig_subtract_mut = model.subtract_mut
    object.__setattr__(model, "subtract_mut", not orig_subtract_mut)
    try:
      yield model
    finally:
      object.__setattr__(model, "subtract_mut", orig_subtract_mut)

  else:
    msg = f"Unknown mutant identifier: {mutant!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------------------------
# Arm Execution and Stage Evaluation
# ---------------------------------------------------------------------------------------------
def evaluate_stages(
  model: FrustraMPNN,
  struct: dict[str, Any],
  oracle: dict[str, np.ndarray],
  pdb_path: Path,
  dtype: str,
  is_clean: bool = True,
) -> dict[str, float]:
  """Evaluate intermediate stages s0-s8 against sealed oracle dump."""
  target_dtype = jnp.float64 if dtype == "float64" else jnp.float32
  coords = jnp.asarray(struct["coords"], dtype=target_dtype)
  mask = jnp.asarray(struct["mask"], dtype=target_dtype)
  res_idx = jnp.asarray(struct["residue_index"])
  chain_idx = jnp.asarray(struct["chain_index"])
  seq = jnp.asarray(struct["sequence"])

  # Execute forward pass
  stages: FrustraStageOutputs = model.forward_stages(
    coords,
    mask,
    res_idx,
    chain_idx,
    seq,
  )

  if dtype == "float64" and stages.frustration_scores.dtype != jnp.float64:
    msg = f"float64 arm produced unexpected dtype {stages.frustration_scores.dtype}"
    raise ValueError(msg)

  out: dict[str, float] = {}

  def _ref(name: str) -> np.ndarray:
    if dtype == "float64" and f"f64_{name}" in oracle:
      return oracle[f"f64_{name}"]
    return oracle[name]

  # s0_prep: Structure prep invariants & feature checks against upstream featurizer X
  out["s0_prep"] = float(np.max(np.abs(struct["coords"] - _ref("feat/X"))))

  # Genuine stage-by-stage forward decomposition against sealed PyTorch oracle dump
  out["s1_features"] = float(np.max(np.abs(np.asarray(stages.features) - _ref("features"))))
  out["s2_encoder"] = float(np.max(np.abs(np.asarray(stages.encoder_out) - _ref("encoder_out"))))
  out["s3_decoder"] = float(np.max(np.abs(np.asarray(stages.decoder_hidden) - _ref("decoder_hidden"))))
  out["s4_latent"] = float(np.max(np.abs(np.asarray(stages.lin_input) - _ref("lin_input"))))
  out["s5_attention"] = float(
    np.max(np.abs(np.asarray(stages.light_attention_out) - _ref("light_attention_out"))),
  )
  out["s6_mlp_head"] = float(np.max(np.abs(np.asarray(stages.both_out) - _ref("both_out"))))
  out["s7_frustration"] = float(
    np.max(np.abs(np.asarray(stages.frustration_scores) - _ref("frustration_scores"))),
  )

  # s8_entry: Public runner.score integration (evaluated only on clean float32 arm)
  if is_clean and dtype == "float32":
    runner_res = runner_score(
      inputs=str(pdb_path),
      model_family="frustrampnn",
      output_kind="frustration",
    )
    runner_arr = np.asarray(runner_res["structures"]["0"]["arrays"]["frustration_scores"])
    out["s8_entry"] = float(np.max(np.abs(runner_arr - oracle["frustration_scores"])))

  return out


def summarize_arm(arm: dict[str, float], dtype: str) -> dict[str, Any]:
  """Summarize stage differences and determine in-band compliance."""
  band = BANDS.get(dtype, 1e-4)
  fails = {k: v for k, v in arm.items() if not (v <= stage_band(k, dtype))}
  over_finite = sorted(k for k, v in fails.items() if math.isfinite(v))
  nonfinite = sorted(k for k, v in fails.items() if not math.isfinite(v))
  worst = max((v for k, v in arm.items() if math.isfinite(v)), default=0.0)

  return {
    "n_stages": len(arm),
    "n_out_of_band": len(fails),
    "out_of_band": sorted(fails),
    "max_stage_diff": worst,
    "n_over_band_finite": len(over_finite),
    "over_band_finite": over_finite,
    "n_nonfinite": len(nonfinite),
    "nonfinite": nonfinite,
    "in_band": len(fails) == 0,
    "band": band,
  }


# ---------------------------------------------------------------------------------------------
# Main CLI Driver
# ---------------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
  """Parse command line arguments."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--oracle-dir",
    type=Path,
    default=_REPO_ROOT / "tests" / "data" / "fixtures",
    help="Directory containing sealed oracle dump .npz.",
  )
  parser.add_argument(
    "--weights-dir",
    type=Path,
    default=_REPO_ROOT / "tests" / "data" / "fixtures",
    help="Directory containing model checkpoint.",
  )
  parser.add_argument(
    "--pinned-dumps",
    type=Path,
    default=_REPO_ROOT / "tests" / "knob_gate" / "frustrampnn_oracle_dumps.sha256.json",
    help="Path to pinned sha256 JSON file.",
  )
  parser.add_argument(
    "--mutants",
    type=str,
    default=",".join(MUTANTS),
    help="Comma-separated list of mutants to execute.",
  )
  parser.add_argument(
    "--arms",
    type=str,
    default=",".join(CLEAN_ARMS),
    help="Comma-separated list of clean arms to evaluate.",
  )
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("outputs/parity/frustrampnn_parity.json"),
    help="Destination path for vehicle summary JSON.",
  )
  parser.add_argument(
    "--controls-out",
    type=Path,
    default=None,
    help="Destination path for branch_controls.json.",
  )
  return parser.parse_args()


def main() -> int:  # noqa: PLR0915
  """Execute FrustraMPNN parity vehicle v2."""
  args = parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s")

  log.info("Starting FrustraMPNN Parity Vehicle v2")

  # 1. Anti-self-trust Verification
  oracle_sha, ckpt_sha, ckpt_file, _pinned = verify_anti_self_trust(
    args.oracle_dir,
    args.weights_dir,
    args.pinned_dumps,
  )
  log.info("Anti-self-trust checks passed: oracle SHA=%s", oracle_sha)

  oracle_npz_path = args.oracle_dir / f"{ORACLE_STEM}.npz"
  if not oracle_npz_path.is_file():
    oracle_npz_path = _REPO_ROOT / "tests" / "data" / "fixtures" / f"{ORACLE_STEM}.npz"
  with np.load(oracle_npz_path) as z:
    oracle = {k: z[k] for k in z.files}

  pdb_path = _REPO_ROOT / "tests" / "data" / "1ubq.pdb"
  driver = FrustraMPNNDriver()
  struct = driver.prepare_structure(pdb_path)

  wanted_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
  mutant_list = [m.strip() for m in args.mutants.split(",") if m.strip()]

  # 2. Execute Clean Arms
  clean_stages: dict[str, dict[str, float]] = {}
  clean_summary: dict[str, dict[str, Any]] = {}

  for arm in wanted_arms:
    dtype = arm
    model = driver.load_model(
      CANONICAL_CKPT,
      local_path=ckpt_file,
      dtype=dtype,
    )
    stages = evaluate_stages(model, struct, oracle, pdb_path, dtype, is_clean=True)
    clean_stages[arm] = stages
    clean_summary[arm] = summarize_arm(stages, dtype)
    log.info("Clean arm %s: %s", arm, clean_summary[arm])

  # Extract precision metrics
  max_diff_f32 = clean_summary.get("float32", {}).get("max_stage_diff", 0.0)
  max_diff_f64 = clean_summary.get("float64", {}).get("max_stage_diff", 0.0)
  clean_ok = all(s["in_band"] for s in clean_summary.values())

  # 3. Execute Mutants (on primary float32 model)
  base_model = driver.load_model(
    CANONICAL_CKPT,
    local_path=ckpt_file,
    dtype="float32",
  )

  mutants: dict[str, Any] = {}
  min_over_band_diffs: list[float] = []

  for m in mutant_list:
    try:
      with apply_mutant(m, base_model) as mutated_model:
        m_stages = evaluate_stages(mutated_model, struct, oracle, pdb_path, "float32", is_clean=False)
        m_summary = summarize_arm(m_stages, "float32")

      over_finite = m_summary["n_over_band_finite"] > 0
      nonfinite = m_summary["n_nonfinite"] > 0
      worst_diff = m_summary["max_stage_diff"]

      # Tripped = strictly finite over-band difference > MUTANT_MIN_POWER
      tripped = over_finite and (not nonfinite) and (worst_diff > MUTANT_MIN_POWER)
      crashed = nonfinite

      if tripped and math.isfinite(worst_diff):
        min_over_band_diffs.append(worst_diff)

      mutants[m] = {
        "status": "failed" if tripped else "survived",
        "crashed": crashed,
        "max_stage_diff": worst_diff,
        "stages": m_stages,
        "summary": m_summary,
      }
      log.info("Mutant %s: %s (worst_diff=%.4f)", m, mutants[m]["status"], worst_diff)

    except Exception:
      log.exception("Mutant %s crashed", m)
      mutants[m] = {
        "status": "survived",
        "crashed": True,
        "max_stage_diff": 0.0,
      }

  n_failed = sum(1 for m in mutants.values() if m["status"] == "failed")
  n_crashed = sum(1 for m in mutants.values() if m["crashed"])
  neg_ctrl_min_diff = min(min_over_band_diffs) if min_over_band_diffs else 0.0

  # Compute decomposition justification (upstream float32 vs float64 floor)
  stage_key_map = {
    "s0_prep": "feat/X",
    "s1_features": "features",
    "s2_encoder": "encoder_out",
    "s3_decoder": "decoder_hidden",
    "s4_latent": "lin_input",
    "s5_attention": "light_attention_out",
    "s6_mlp_head": "both_out",
    "s7_frustration": "frustration_scores",
  }
  floor_up_f32_vs_f64 = {
    stage: float(np.max(np.abs(oracle[k] - oracle[f"f64_{k}"])))
    for stage, k in stage_key_map.items()
    if k in oracle and f"f64_{k}" in oracle
  }

  # 4. Construct Final Results Payload Matching [result_schema]
  result = {
    "clean": "pass" if clean_ok else "fail",
    "weight_source": "upstream_checkpoint",
    "oracle_verified": 1,
    "f64_max_abs_diff": float(max_diff_f64),
    "f32_max_abs_diff": float(max_diff_f32),
    "n_arms": len(wanted_arms),
    "n_clean_arms_in_band": sum(1 for s in clean_summary.values() if s["in_band"]),
    "n_listed": len(mutant_list),
    "n_failed": n_failed,
    "n_crashed": n_crashed,
    "neg_ctrl_min_diff": float(neg_ctrl_min_diff),
    "oracle_dump_sha256": {ORACLE_STEM: oracle_sha},
    "clean_summary": clean_summary,
    "clean_stages": clean_stages,
    "decomposition": {
      "floor_up_f32_vs_f64": floor_up_f32_vs_f64,
      "max_upstream_floor": max(floor_up_f32_vs_f64.values(), default=0.0),
    },
    "mutants": {m: v["status"] for m, v in mutants.items()},
    "mutant_detail": mutants,
    "not_compared": [
      "sampling bit-for-bit token reproduction through runner.sample (requires RNG draw injection hook)",
      "esm-2 co-embedding (deferred under debt #3222)",
    ],
  }

  def _sanitize(v: Any) -> Any:  # noqa: ANN401
    if isinstance(v, float) and not math.isfinite(v):
      return None
    if isinstance(v, dict):
      return {k: _sanitize(val) for k, val in v.items()}
    if isinstance(v, list):
      return [_sanitize(val) for val in v]
    return v

  clean_result = _sanitize(result)

  # 5. Emit Outputs
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(clean_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  log.info("Wrote vehicle summary to %s", args.out)

  if args.controls_out is not None:
    args.controls_out.parent.mkdir(parents=True, exist_ok=True)
    controls_data = {
      "clean": clean_result["clean"],
      "mutants": clean_result["mutants"],
      "weights": {
        "FrustraMPNN/model_weights/megascale_default_la-true_esm-False_split1.ckpt": ckpt_sha,
      },
    }
    args.controls_out.write_text(
      json.dumps(controls_data, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
    )
    log.info("Wrote branch controls to %s", args.controls_out)

  bth_results = os.environ.get("BTH_RESULTS_PATH")
  if bth_results:
    Path(bth_results).parent.mkdir(parents=True, exist_ok=True)
    Path(bth_results).write_text(
      json.dumps(clean_result, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
    )
    log.info("Wrote BTH results payload to %s", bth_results)

  log.info(
    "Vehicle execution complete: clean=%s, f64=%.2e, f32=%.2e, failed=%d/%d, crashed=%d",
    clean_result["clean"],
    clean_result["f64_max_abs_diff"],
    clean_result["f32_max_abs_diff"],
    n_failed,
    len(mutant_list),
    n_crashed,
  )

  return 0 if (clean_ok and n_failed == len(mutant_list) and n_crashed == 0) else 1


if __name__ == "__main__":
  sys.exit(main())
