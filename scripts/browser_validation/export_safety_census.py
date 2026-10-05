"""Export-safety census of real aminx callables (Phase 0, T1, AC-1).

Builds the seven AC-1 callables with SHIPPED weights (no network -- packaged
resources under ``aminx.model_params``), traces each independently with
``jax.make_jaxpr``, and for every callable that traces runs
``xtrax.export.check_export_safety`` for both the NATIVE and WASM32 targets.
Two positive controls (a bare ``random_decoding_order`` call and a bare
``lax.top_k`` call) confirm the instrument actually detects known-unsafe ops.

A callable is recorded ``trace_ok=False`` only when ``jax.make_jaxpr`` of the
REAL callable, built with real loaded weights, genuinely raises -- the
exception is recorded verbatim. Nothing is swallowed into a generic
"unavailable" message.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, cast

import equinox as eqx
import jax
import jax.lax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROTEIN_CKPT = "proteinmpnn_v_48_020"
LIGAND_CKPT = "ligandmpnn_v_32_010_25"
PACKER_CKPT = "ligandmpnn_sc_v_32_002_16"

AC1_NAMES = [
  "P04_unconditional_kernel",
  "P05_conditional_kernel",
  "P06_score_sequence_pinned_order",
  "P07_ar_sampling_kernel_host_wave",
  "legacy_aminx_sampling_sample",
  "P11_ligandmpnn_conditional_kernel",
  "P14_packer",
]


def check_xtrax_rules() -> bool:
  """True iff xtrax.export.safety carries the two rule constants this census depends on."""
  from xtrax.export import safety

  has_permutation = hasattr(safety, "_RANDOM_PERMUTATION_RULE")
  has_threefry = hasattr(safety, "_UNBATCHED_THREEFRY_RULE")
  logger.info(
    "_RANDOM_PERMUTATION_RULE=%s _UNBATCHED_THREEFRY_RULE=%s",
    has_permutation,
    has_threefry,
  )
  return has_permutation and has_threefry


def synthetic_coords(key: PRNGKeyArray, length: int) -> jax.Array:
  """Well-separated synthetic backbone coords (N, CA, C, O), random within +/-10 Angstrom."""
  return jax.random.uniform(key, (length, 4, 3), minval=-10.0, maxval=10.0)


def trace_callable(name: str, fn: Callable[[], Any]) -> tuple[bool, str]:
  """Trace fn (zero-arg) with jax.make_jaxpr; return (trace_ok, exception text or '')."""
  try:
    jax.make_jaxpr(fn)()
  except Exception as e:  # noqa: BLE001 - the exception text itself is the finding
    logger.warning("%s: trace_ok=False (%s: %s)", name, type(e).__name__, e)
    return False, f"{type(e).__name__}: {e}"
  logger.info("%s: trace_ok=True", name)
  return True, ""


def safety_checks(fn: Callable[[], Any]) -> list[dict[str, Any]]:
  """Run check_export_safety for NATIVE and WASM32; each entry names its blockers."""
  from xtrax.export import NATIVE, WASM32, check_export_safety

  results = []
  for target_name, target in (("NATIVE", NATIVE), ("WASM32", WASM32)):
    blockers = check_export_safety([], {}, [], fn, target)
    results.append(
      {
        "target": target_name,
        "blockers": [{"rule": b.rule, "axis": b.axis, "detail": b.detail} for b in blockers],
      }
    )
  return results


def control_random_decoding_order(length: int) -> tuple[bool, list[str]]:
  """Bare random_decoding_order must be flagged random-permutation (NATIVE)."""
  from xtrax.export import NATIVE, check_export_safety

  from aminx.utils.decoding_order import random_decoding_order

  key = jax.random.key(0)

  def fn() -> Any:
    return random_decoding_order(key, length)

  blockers = check_export_safety([], {}, [], fn, NATIVE)
  rules = [b.rule for b in blockers]
  return "random-permutation" in rules, rules


def control_top_k(length: int) -> tuple[bool, list[str]]:
  """Bare lax.top_k must be flagged unlegalizable-op (NATIVE)."""
  from xtrax.export import NATIVE, check_export_safety

  x = jnp.arange(length, dtype=jnp.float32)

  def fn() -> Any:
    return jax.lax.top_k(x, 8)

  blockers = check_export_safety([], {}, [], fn, NATIVE)
  rules = [b.rule for b in blockers]
  return "unlegalizable-op" in rules, rules


def build_callables(bucket: int) -> dict[str, tuple[Callable[[], Any] | None, str]]:
  """Build the seven AC-1 callables. Each value is (zero-arg fn, build-failure text|"")."""
  from aminx.inference import score_conditional, score_unconditional
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set
  from aminx.inference.sample_autoregressive import kernel as sample_autoregressive_kernel
  from aminx.io.weights import get_topology_for_checkpoint, load_model
  from aminx.sampling.sample import make_sample_sequences
  from aminx.scoring.score import make_score_fn
  from aminx.types.bundles import PackerBundle, WaveScheduleBundle
  from aminx.types.configs import InferenceConfig
  from aminx.utils.decoding_order import single_decoding_order

  key = jax.random.key(42)
  coords = synthetic_coords(jax.random.key(1), bucket)
  mask = jnp.ones((bucket,))
  residue_index = jnp.arange(bucket)
  chain_index = jnp.zeros((bucket,), dtype=jnp.int32)
  sequence = jnp.zeros((bucket,), dtype=jnp.int32)
  stage_set = make_stage_set()

  out: dict[str, tuple[Callable[[], Any] | None, str]] = {}

  def _fail(name: str, exc: Exception) -> None:
    logger.error("%s: failed to build (%s: %s)", name, type(exc).__name__, exc)
    out[name] = (None, f"{type(exc).__name__}: {exc}")

  try:
    protein_model = eqx.nn.inference_mode(load_model(checkpoint_id=PROTEIN_CKPT), value=True)
  except Exception as e:  # noqa: BLE001
    protein_model = None
    for name in AC1_NAMES[:5]:
      _fail(name, e)

  if protein_model is not None:
    try:
      bundle, config = build_inference_bundle(
        coords=coords,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        mode="score_unconditional",
      )
      out["P04_unconditional_kernel"] = (
        lambda: score_unconditional.kernel(protein_model, key, bundle, config, stage_set),
        "",
      )
    except Exception as e:  # noqa: BLE001
      _fail("P04_unconditional_kernel", e)

    try:
      bundle, config = build_inference_bundle(
        coords=coords,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        sequence=sequence,
        mode="score_conditional",
      )
      out["P05_conditional_kernel"] = (
        lambda: score_conditional.kernel(protein_model, key, bundle, config, stage_set),
        "",
      )
    except Exception as e:  # noqa: BLE001
      _fail("P05_conditional_kernel", e)

    try:
      # make_score_fn's return type annotation (ScoreFn) names an aspirational
      # (bundle, config, stage_set) protocol; the CONCRETE function it actually
      # returns (score_sequence, scoring/score.py:115) takes the raw-array
      # signature used below -- cast to Any so ty checks the real call, not the
      # stale protocol.
      score_fn = cast(
        "Callable[..., Any]",
        make_score_fn(protein_model, decoding_order_fn=single_decoding_order),
      )
      out["P06_score_sequence_pinned_order"] = (
        lambda: score_fn(
          key,
          sequence,
          coords,
          mask,
          residue_index,
          chain_index,
          multi_state_strategy="arithmetic_mean",
        ),
        "",
      )
    except Exception as e:  # noqa: BLE001
      _fail("P06_score_sequence_pinned_order", e)

    try:
      # Step 2: WaveScheduleBundle.from_tie_groups(arange(L), host_order); every position
      # is its own tie group (arange), decoded in fixed N-to-C host order (arange).
      host_order = jnp.arange(bucket)
      wave = WaveScheduleBundle.from_tie_groups(jnp.arange(bucket), host_order)
      bundle, config = build_inference_bundle(
        coords=coords,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        wave=wave,
        mode="sample_ar",
      )
      out["P07_ar_sampling_kernel_host_wave"] = (
        lambda: sample_autoregressive_kernel(protein_model, key, bundle, config, stage_set),
        "",
      )
    except Exception as e:  # noqa: BLE001
      _fail("P07_ar_sampling_kernel_host_wave", e)

    try:
      # Same protocol/concrete mismatch as make_score_fn above: SamplerFn names
      # (bundle, config, stage_set) but the "temperature"-strategy concrete
      # function (sample.py's sample_sequences) takes raw arrays.
      legacy_sample_fn = cast("Callable[..., Any]", make_sample_sequences(protein_model))
      out["legacy_aminx_sampling_sample"] = (
        lambda: legacy_sample_fn(key, coords, mask, residue_index, chain_index),
        "",
      )
    except Exception as e:  # noqa: BLE001
      _fail("legacy_aminx_sampling_sample", e)

  try:
    ligand_model = eqx.nn.inference_mode(load_model(checkpoint_id=LIGAND_CKPT), value=True)
    n_atoms = int(get_topology_for_checkpoint(LIGAND_CKPT)["atom_context_num"])
    lig_coords = jax.random.uniform(
      jax.random.key(2),
      (bucket, n_atoms, 3),
      minval=-10.0,
      maxval=10.0,
    )  # (L, A, 3)
    lig_types = jnp.zeros((bucket, n_atoms), dtype=jnp.int32)
    lig_mask = jnp.ones((bucket, n_atoms))
    bundle, config = build_inference_bundle(
      coords=coords,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      sequence=sequence,
      ligand_coords=lig_coords,
      ligand_atom_types=lig_types,
      ligand_mask=lig_mask,
      mode="score_conditional",
    )
    out["P11_ligandmpnn_conditional_kernel"] = (
      lambda: score_conditional.kernel(ligand_model, key, bundle, config, stage_set),
      "",
    )
  except Exception as e:  # noqa: BLE001
    _fail("P11_ligandmpnn_conditional_kernel", e)

  try:
    packer_model = eqx.nn.inference_mode(load_model(checkpoint_id=PACKER_CKPT), value=True)
    n_atoms = int(get_topology_for_checkpoint(PACKER_CKPT)["atom_context_num"])
    packer_bundle = PackerBundle(
      sequence=sequence,
      backbone_coords=jax.random.uniform(
        jax.random.key(3),
        (bucket, 14, 3),
        minval=-10.0,
        maxval=10.0,
      ),
      backbone_mask=jnp.ones((bucket, 14)),
      ligand_coords=jax.random.uniform(
        jax.random.key(4),
        (bucket, n_atoms, 3),
        minval=-10.0,
        maxval=10.0,
      ),
      ligand_mask=jnp.ones((bucket, n_atoms)),
      ligand_atom_types=jnp.zeros((bucket, n_atoms)),
      mask=mask,
      residue_index=residue_index,
      chain_labels=chain_index,
    )
    packer_config = InferenceConfig()
    out["P14_packer"] = (lambda: packer_model(key, packer_bundle, packer_config), "")
  except Exception as e:  # noqa: BLE001
    _fail("P14_packer", e)

  return out


def _write_result(out_path: Path, result: dict[str, Any]) -> None:
  """Write result to --out and, if set, to $BTH_RESULTS_PATH (the reliable outcome-eval path)."""
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w") as f:
    json.dump(result, f, indent=2, default=str)
  results_path = os.environ.get("BTH_RESULTS_PATH")
  if results_path:
    with Path(results_path).open("w") as f:
      json.dump(result, f, indent=2, default=str)


def run_census(bucket: int) -> dict[str, Any]:
  """Build, trace, and safety-check every AC-1 callable; run both controls."""
  import xtrax

  callables = build_callables(bucket)
  rows: list[dict[str, Any]] = []
  n_untraceable = 0
  n_with_blockers = 0

  for name in AC1_NAMES:
    fn, build_error = callables.get(
      name,
      (None, "callable was never built (missing from build_callables output)"),
    )
    if fn is None:
      n_untraceable += 1
      rows.append({"name": name, "trace_ok": False, "exception": build_error, "blockers": None})
      continue

    trace_ok, exc = trace_callable(name, fn)
    if not trace_ok:
      n_untraceable += 1
      rows.append({"name": name, "trace_ok": False, "exception": exc, "blockers": None})
      continue

    blockers = safety_checks(fn)
    if any(entry["blockers"] for entry in blockers):
      n_with_blockers += 1
    rows.append({"name": name, "trace_ok": True, "exception": "", "blockers": blockers})

  perm_detected, perm_rules = control_random_decoding_order(bucket)
  topk_detected, topk_rules = control_top_k(bucket)
  controls_detected = int(perm_detected) + int(topk_detected)

  return {
    "n_callables": len(AC1_NAMES),
    "n_untraceable": n_untraceable,
    "n_with_blockers": n_with_blockers,
    "controls_total": 2,
    "controls_detected": controls_detected,
    "xtrax_file": xtrax.__file__,
    "rules_present": True,  # only reachable past the main() gate, which exits 2 otherwise
    "callables": rows,
    "controls": {
      "random_decoding_order": {"detected": perm_detected, "rules_detected": perm_rules},
      "top_k": {"detected": topk_detected, "rules_detected": topk_rules},
    },
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON.")
  parser.add_argument(
    "--bucket",
    type=int,
    default=128,
    help="Sequence length L for all callables.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Check the environment (xtrax rules present) and write a stub result without building "
    "or tracing any callable.",
  )
  args = parser.parse_args(argv)

  import xtrax

  logger.info(
    "xtrax.__file__=%s xtrax.__version__=%s",
    xtrax.__file__,
    getattr(xtrax, "__version__", "unknown"),
  )

  rules_present = check_xtrax_rules()
  if not rules_present:
    logger.error(
      "xtrax.export.safety is missing _RANDOM_PERMUTATION_RULE and/or _UNBATCHED_THREEFRY_RULE "
      "-- this xtrax build cannot run the export-safety census. Rerun with "
      "PYTHONPATH=/home/marielle/projects/xtrax/src so the correct rule set is importable.",
    )
    return 2

  if args.dry_run:
    result = {
      "n_callables": len(AC1_NAMES),
      "n_untraceable": 0,
      "n_with_blockers": 0,
      "controls_total": 2,
      "controls_detected": 0,
      "xtrax_file": xtrax.__file__,
      "rules_present": rules_present,
      "callables": [],
      "dry_run": True,
    }
    _write_result(args.out, result)
    logger.info("dry-run complete; wrote stub result to %s", args.out)
    return 0

  result = run_census(args.bucket)
  _write_result(args.out, result)
  logger.info(
    "census complete: n_untraceable=%d n_with_blockers=%d controls_detected=%d/%d",
    result["n_untraceable"],
    result["n_with_blockers"],
    result["controls_detected"],
    result["controls_total"],
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
