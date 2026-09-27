"""Layer (b) build: P03/P04 ONNX export ladder + RNG audit, jaxpr + ONNX (T2, step 4).

For each path in `{p03, p04}` and each bucket in `--buckets` (default `EXPORT_BUCKETS`,
D-C), builds the RNG-free export wrapper (`aminx.export.make_p03_featurize` /
`make_p04_unconditional`, T1) closed over the pinned checkpoint and converts it through
`jax2onnx.to_onnx` on synthetic inputs of exactly that bucket's length (this task proves
CONVERSION and RNG-freedom; T3a is where numeric parity against real fixtures is graded).
Every clean converted artifact is run through BOTH RNG audits: the jaxpr walker
(`aminx.export.rng_audit.find_rng_primitives`, T1) and the ONNX-graph walker
(`onnx_audit.find_onnx_rng_ops`, this task).

Also builds, per bucket, the numeric perturbation/index-swap control artifacts T3a will
later calibrate and grade (`p04_L{b}_perturbed.onnx`, `p03_L{b}_ebias.onnx`,
`p03_L{b}_swap.onnx`; deltas here are BUILD-time placeholders, T3a re-sizes them), and,
separately from those, the 5 AC-B2 RNG-audit self-check controls this run's own outcome
gates on (`controls_total`/`controls_detected` in the sidecar): three ONNX-level RNG
plants at increasing structural depth (`If` subgraph, `Loop` subgraph, local
`FunctionProto`), one direct top-level ONNX RNG op, and one deliberately RNG-carrying
("poisoned") jaxpr -- proving the jaxpr-side walker fires independently of the ONNX-side
one. (Interpretive note, documented per the fixer's Close-out "deviations": the spec
names "the five AC-B2 planted controls" without enumerating all five in what this task
received verbatim; this 3-ONNX-depth + 1-direct-ONNX + 1-jaxpr split was chosen to give
both walkers -- jaxpr AND ONNX, the task's own title -- independent, exactly-once
coverage, reaching the stated total of 5.)

`--dry-run` builds and audits all 5 AC-B2 controls (self-contained; no checkpoint load,
no jax2onnx conversion) as a wiring smoke test, and reports zero clean artifacts.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from onnx import TensorProto, helper

if TYPE_CHECKING:
  from onnx import ModelProto

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import layer_a_common as lac  # noqa: E402
import layer_b_common as lbc  # noqa: E402
from jax2onnx_spike import _extract_primitive  # noqa: E402
from onnx_audit import find_onnx_rng_ops  # noqa: E402

CHECKPOINT_ID = "proteinmpnn_v_48_020"  # V10, matches aminx.export.PINNED_CHECKPOINT_ID
DEFAULT_BUCKETS: tuple[int, ...] = (128, 256, 512, 1024)
PATHS: tuple[str, ...] = ("p03", "p04")

#: Build-time control-perturbation magnitudes (T3a calibrates the real deltas).
BUILD_DELTA_P04_BIAS = 5e-4
BUILD_DELTA_P03_EBIAS = 1e-4

CONTROLS_TOTAL = 5


# ----------------------------------------------------------------------------------
# Synthetic inputs (no real fixture needed -- this task proves conversion, not parity)
# ----------------------------------------------------------------------------------


def _synthetic_inputs(bucket: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Deterministic, non-degenerate synthetic backbone of length `bucket` (all real)."""
  ca_spacing = 3.8
  base = np.arange(bucket, dtype=np.float32) * ca_spacing
  jitter = np.sin(np.arange(bucket, dtype=np.float32) * 0.7).astype(np.float32)

  coords = np.zeros((bucket, 4, 3), dtype=np.float32)
  coords[:, 0, 0] = base - 1.0  # N
  coords[:, 1, 0] = base  # CA
  coords[:, 2, 0] = base + 1.0  # C
  coords[:, 3, 0] = base + 1.5  # O
  coords[:, :, 1] = jitter[:, None] * 0.3

  mask = np.ones((bucket,), dtype=np.float32)
  residue_index = np.arange(bucket, dtype=np.int32)
  chain_index = np.zeros((bucket,), dtype=np.int32)
  return coords, mask, residue_index, chain_index


def _specs(inputs: tuple[np.ndarray, ...]) -> list[Any]:
  return [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in inputs]


# ----------------------------------------------------------------------------------
# AC-B2: the 5 planted RNG-audit self-check controls
# ----------------------------------------------------------------------------------


def _placeholder_base_model() -> ModelProto:
  """A trivial clean ONNX model to graft planted subgraphs/functions onto in --dry-run
  (no checkpoint load, no jax2onnx conversion needed for the audit self-check)."""
  x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
  y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
  node = helper.make_node("Identity", ["x"], ["y"])
  graph = helper.make_graph([node], "placeholder", [x], [y])
  return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _plant_if_subgraph_rng(base: ModelProto) -> ModelProto:
  m = copy.deepcopy(base)
  then_out = helper.make_tensor_value_info("audit_if_rng", TensorProto.FLOAT, [1])
  rng_node = helper.make_node(
    "RandomUniform", [], ["audit_if_rng"], shape=[1], name="AuditPlantIfRNG"
  )
  then_graph = helper.make_graph([rng_node], "audit_if_then", [], [then_out])

  else_out = helper.make_tensor_value_info("audit_if_rng", TensorProto.FLOAT, [1])
  const_tensor = helper.make_tensor("audit_if_const", TensorProto.FLOAT, [1], [0.0])
  const_node = helper.make_node(
    "Constant", [], ["audit_if_rng"], value=const_tensor, name="AuditPlantIfElseConst"
  )
  else_graph = helper.make_graph([const_node], "audit_if_else", [], [else_out])

  cond_init = helper.make_tensor("audit_if_cond", TensorProto.BOOL, [], [True])
  if_node = helper.make_node(
    "If",
    ["audit_if_cond"],
    ["audit_if_out"],
    then_branch=then_graph,
    else_branch=else_graph,
    name="AuditPlantIf",
  )
  m.graph.node.append(if_node)
  m.graph.initializer.append(cond_init)
  return m


def _plant_loop_subgraph_rng(base: ModelProto) -> ModelProto:
  m = copy.deepcopy(base)
  iter_in = helper.make_tensor_value_info("audit_loop_iter", TensorProto.INT64, [])
  cond_in = helper.make_tensor_value_info("audit_loop_cond_in", TensorProto.BOOL, [])
  cond_out = helper.make_tensor_value_info("audit_loop_cond_out", TensorProto.BOOL, [])
  rng_out = helper.make_tensor_value_info("audit_loop_rng", TensorProto.FLOAT, [1])

  rng_node = helper.make_node(
    "RandomNormal", [], ["audit_loop_rng"], shape=[1], name="AuditPlantLoopRNG"
  )
  cond_identity = helper.make_node("Identity", ["audit_loop_cond_in"], ["audit_loop_cond_out"])
  body = helper.make_graph(
    [rng_node, cond_identity], "audit_loop_body", [iter_in, cond_in], [cond_out, rng_out]
  )

  trip_count = helper.make_tensor("audit_loop_trip_count", TensorProto.INT64, [], [1])
  cond0 = helper.make_tensor("audit_loop_cond0", TensorProto.BOOL, [], [True])
  loop_node = helper.make_node(
    "Loop",
    ["audit_loop_trip_count", "audit_loop_cond0"],
    ["audit_loop_out"],
    body=body,
    name="AuditPlantLoop",
  )
  m.graph.node.append(loop_node)
  m.graph.initializer.extend([trip_count, cond0])
  return m


def _plant_function_rng(base: ModelProto) -> ModelProto:
  m = copy.deepcopy(base)
  domain = "bv.test"
  fn_node = helper.make_node("RandomUniform", [], ["fn_rng_out"], shape=[1], name="AuditPlantFnRNG")
  function = helper.make_function(
    domain,
    "AuditPlantedFn",
    inputs=[],
    outputs=["fn_rng_out"],
    nodes=[fn_node],
    opset_imports=[helper.make_opsetid("", 18)],
  )
  call_node = helper.make_node(
    "AuditPlantedFn", [], ["audit_fn_call_out"], domain=domain, name="AuditPlantFnCall"
  )
  m.graph.node.append(call_node)
  m.functions.append(function)
  m.opset_import.append(helper.make_opsetid(domain, 1))
  return m


def _plant_direct_multinomial(base: ModelProto) -> ModelProto:
  m = copy.deepcopy(base)
  dummy = helper.make_tensor("audit_logits_dummy", TensorProto.FLOAT, [1, 2], [0.0, 0.0])
  node = helper.make_node(
    "Multinomial",
    ["audit_logits_dummy"],
    ["audit_multinomial_out"],
    sample_size=1,
    name="AuditPlantMultinomial",
  )
  m.graph.node.append(node)
  m.graph.initializer.append(dummy)
  return m


def _poisoned_jaxpr() -> Any:
  """A deliberately RNG-carrying jaxpr (the 5th AC-B2 control): proves the jaxpr-side
  walker (`aminx.export.rng_audit.find_rng_primitives`) fires independently of the
  ONNX-side one -- neither walker is trusted on the strength of the OTHER's controls."""

  def poisoned(x: jax.Array) -> jax.Array:
    key = jax.random.PRNGKey(0)
    return x + jax.random.normal(key, x.shape)

  return jax.make_jaxpr(poisoned)(jnp.zeros((4,), dtype=jnp.float32))


def run_ac_b2_controls() -> dict[str, Any]:
  """Build and audit all 5 AC-B2 planted RNG controls. Returns `{detected, total, detail}`."""
  from aminx.export.rng_audit import find_rng_primitives

  base = _placeholder_base_model()
  onnx_plants = {
    "if_subgraph": _plant_if_subgraph_rng(base),
    "loop_subgraph": _plant_loop_subgraph_rng(base),
    "local_function": _plant_function_rng(base),
    "direct_multinomial": _plant_direct_multinomial(base),
  }

  detail: dict[str, bool] = {}
  for name, model_proto in onnx_plants.items():
    findings = find_onnx_rng_ops(model_proto)
    detail[name] = len(findings) > 0
    logger.info("AC-B2 control %r: findings=%s detected=%s", name, findings, detail[name])

  poisoned_findings = find_rng_primitives(_poisoned_jaxpr())
  detail["poisoned_jaxpr"] = len(poisoned_findings) > 0
  logger.info(
    "AC-B2 control %r: findings=%s detected=%s",
    "poisoned_jaxpr",
    poisoned_findings,
    detail["poisoned_jaxpr"],
  )

  detected = sum(1 for v in detail.values() if v)
  return {"detected": detected, "total": CONTROLS_TOTAL, "detail": detail}


# ----------------------------------------------------------------------------------
# Clean wrapper conversion + numeric-perturbation control artifacts
# ----------------------------------------------------------------------------------


def _convert_to_onnx(
  fn: Any, inputs: tuple[np.ndarray, ...], model_name: str, out_path: Path
) -> dict[str, Any]:
  import jax2onnx

  specs = _specs(inputs)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  try:
    jax2onnx.to_onnx(
      fn, specs, model_name=model_name, output_path=str(out_path), return_mode="file"
    )
  except Exception as e:  # noqa: BLE001 -- the error text itself is the finding
    logger.warning("%s: conversion failed: %s: %s", model_name, type(e).__name__, e)
    return {
      "converted": False,
      "error": f"{type(e).__name__}: {e}",
      "primitive": _extract_primitive(e),
    }
  return {"converted": True, "path": out_path}


def _rng_audit_artifact(fn: Any, inputs: tuple[np.ndarray, ...], onnx_path: Path) -> list[str]:
  """Run BOTH RNG audits (jaxpr + ONNX) on a converted artifact; return combined findings."""
  import onnx as onnx_mod

  from aminx.export.rng_audit import find_rng_primitives

  jaxpr_hits = find_rng_primitives(jax.make_jaxpr(fn)(*inputs))
  onnx_hits = find_onnx_rng_ops(onnx_mod.load(str(onnx_path)))
  return [f"jaxpr:{h}" for h in jaxpr_hits] + onnx_hits


def _perturb_bias(model: Any, get_bias: Any, delta: float) -> Any:
  """Return a copy of `model` with `get_bias(model)[0] += delta` (a real weight edit,
  `eqx.tree_at`), mirroring `jax2onnx_spike.perturb_w_out_bias`'s pattern."""
  import equinox as eqx

  bias = get_bias(model)
  new_bias = bias.at[0].add(delta)
  return eqx.tree_at(get_bias, model, new_bias)


def _perturb_all_channels(model: Any, get_bias: Any, delta: float) -> Any:
  import equinox as eqx

  bias = get_bias(model)
  return eqx.tree_at(get_bias, model, bias + delta)


def _make_p03_swap_variant(p03_fn: Any) -> Any:
  """Swap neighbour slots k-2/k-1 of real row 0 in `p03_fn`'s OWN traced output (D-C:
  the index-control artifact T3a will detect via neighbour-index mismatch)."""

  def swapped(coords: Any, mask: Any, residue_index: Any, chain_index: Any) -> Any:
    idx, feats = p03_fn(coords, mask, residue_index, chain_index)
    k = idx.shape[-1]
    perm = jnp.arange(k).at[k - 2].set(k - 1).at[k - 1].set(k - 2)
    idx = idx.at[0].set(idx[0][perm])
    feats = feats.at[0].set(feats[0][perm])
    return idx, feats

  return swapped


def build_path_bucket(
  path: str,
  bucket: int,
  model: Any,
  stage_set: Any,
  artifacts_dir: Path,
) -> dict[str, Any]:
  """Build + convert + audit the clean artifact and its control artifacts for one
  `(path, bucket)` cell. Returns a dict with `converted`, `rng_hits`, `error`, `primitive`,
  `manifest_rows` (list of rows for `layer_b_common.manifest_append`)."""
  from aminx.export import make_p03_featurize, make_p04_unconditional

  inputs = _synthetic_inputs(bucket)
  manifest_rows: list[dict[str, Any]] = []

  if path == "p03":
    fn = make_p03_featurize(model)
  else:
    fn = make_p04_unconditional(model, stage_set)

  clean_name = f"{path}_L{bucket}"
  clean_path = artifacts_dir / f"{clean_name}.onnx"
  clean_result = _convert_to_onnx(fn, inputs, clean_name, clean_path)

  cell: dict[str, Any] = {"converted": clean_result["converted"], "rng_hits": []}
  if not clean_result["converted"]:
    cell["error"] = clean_result["error"]
    cell["primitive"] = clean_result["primitive"]
    return cell

  cell["rng_hits"] = _rng_audit_artifact(fn, inputs, clean_path)
  manifest_rows.append(
    {
      "path": f"{clean_name}.onnx",
      "sha256": lbc.artifact_sha256(clean_path),
      "role": "clean",
      "export_path": path,
      "bucket": bucket,
    }
  )

  # Numeric perturbation/index-swap control artifacts (T3a calibrates + grades these;
  # this task only builds them at fixed build-time deltas).
  if path == "p04":
    perturbed_model = _perturb_bias(model, lambda m: m.w_out.bias, BUILD_DELTA_P04_BIAS)
    perturbed_fn = make_p04_unconditional(perturbed_model, stage_set)
    perturbed_path = artifacts_dir / f"p04_L{bucket}_perturbed.onnx"
    r = _convert_to_onnx(perturbed_fn, inputs, f"p04_L{bucket}_perturbed", perturbed_path)
    if r["converted"]:
      manifest_rows.append(
        {
          "path": f"p04_L{bucket}_perturbed.onnx",
          "sha256": lbc.artifact_sha256(perturbed_path),
          "role": "control_perturbed_bias",
          "export_path": path,
          "bucket": bucket,
          "delta": BUILD_DELTA_P04_BIAS,
        }
      )
  else:
    ebias_model = _perturb_all_channels(
      model, lambda m: m.features.w_e_proj.bias, BUILD_DELTA_P03_EBIAS
    )
    ebias_fn = make_p03_featurize(ebias_model)
    ebias_path = artifacts_dir / f"p03_L{bucket}_ebias.onnx"
    r = _convert_to_onnx(ebias_fn, inputs, f"p03_L{bucket}_ebias", ebias_path)
    if r["converted"]:
      manifest_rows.append(
        {
          "path": f"p03_L{bucket}_ebias.onnx",
          "sha256": lbc.artifact_sha256(ebias_path),
          "role": "control_ebias",
          "export_path": path,
          "bucket": bucket,
          "delta": BUILD_DELTA_P03_EBIAS,
        }
      )

    swap_fn = _make_p03_swap_variant(fn)
    swap_path = artifacts_dir / f"p03_L{bucket}_swap.onnx"
    r = _convert_to_onnx(swap_fn, inputs, f"p03_L{bucket}_swap", swap_path)
    if r["converted"]:
      manifest_rows.append(
        {
          "path": f"p03_L{bucket}_swap.onnx",
          "sha256": lbc.artifact_sha256(swap_path),
          "role": "control_swap",
          "export_path": path,
          "bucket": bucket,
        }
      )

  cell["manifest_rows"] = manifest_rows
  return cell


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------


def _pkg_version(name: str) -> str:
  try:
    return pkg_version(name)
  except PackageNotFoundError:
    return "not-installed"


def run(args: argparse.Namespace) -> dict[str, Any]:
  buckets = tuple(int(b) for b in args.buckets.split(","))
  artifacts_base = Path(args.artifacts_dir)

  controls = run_ac_b2_controls()

  versions = {
    "jax": _pkg_version("jax"),
    "jax2onnx": _pkg_version("jax2onnx"),
    "onnx": _pkg_version("onnx"),
    "onnxruntime": _pkg_version("onnxruntime"),
    "bathos_overlay": lbc.bathos_overlay_provenance(),
  }

  if args.dry_run:
    return {
      "n_artifacts": 0,
      "n_converted": 0,
      "n_rng_clean": 0,
      "controls_total": controls["total"],
      "controls_detected": controls["detected"],
      "converted_by_path": {p: True for p in PATHS},  # vacuously true (nothing built)
      "opsets": {},
      "versions": versions,
      "artifacts": {},
      "_ac_b2_detail": controls["detail"],  # extra diagnostic, not schema-required
    }

  from aminx.export import PINNED_CHECKPOINT_ID
  from aminx.inference.logits import make_stage_set
  from aminx.io.weights import load_model

  if CHECKPOINT_ID != PINNED_CHECKPOINT_ID:
    msg = (
      f"CHECKPOINT_ID={CHECKPOINT_ID!r} != "
      f"aminx.export.PINNED_CHECKPOINT_ID={PINNED_CHECKPOINT_ID!r}"
    )
    raise ValueError(msg)
  lbc.assert_checkpoint_pinned(CHECKPOINT_ID)  # raises on mismatch; caller exits 3

  model = load_model(checkpoint_id=CHECKPOINT_ID)
  stage_set = make_stage_set()

  h12 = args.artifacts_subdir or "unpinned"
  artifacts_dir = artifacts_base / h12
  artifacts_dir.mkdir(parents=True, exist_ok=True)

  n_artifacts = 0
  n_converted = 0
  n_rng_clean = 0
  converted_by_path: dict[str, bool] = {}
  opsets: dict[str, Any] = {}
  all_manifest_rows: list[dict[str, Any]] = []
  errors: dict[str, Any] = {}

  for path in PATHS:
    path_all_converted = True
    for bucket in buckets:
      n_artifacts += 1
      cell = build_path_bucket(path, bucket, model, stage_set, artifacts_dir)
      key = f"{path}_L{bucket}"
      if cell["converted"]:
        n_converted += 1
        if not cell["rng_hits"]:
          n_rng_clean += 1
        else:
          logger.error("%s: clean artifact carries RNG findings: %s", key, cell["rng_hits"])
        all_manifest_rows.extend(cell.get("manifest_rows", []))
        import onnx as onnx_mod  # noqa: PLC0415

        onnx_path = artifacts_dir / f"{key}.onnx"
        opsets[key] = onnx_mod.load(str(onnx_path)).opset_import[0].version
      else:
        path_all_converted = False
        errors[key] = {"error": cell["error"], "primitive": cell["primitive"]}
    converted_by_path[path] = path_all_converted

  if all_manifest_rows:
    lbc.manifest_append(all_manifest_rows, artifacts_dir / "artifact_manifest.json")

  artifacts = {row["path"]: row for row in all_manifest_rows}

  return {
    "n_artifacts": n_artifacts,
    "n_converted": n_converted,
    "n_rng_clean": n_rng_clean,
    "controls_total": controls["total"],
    "controls_detected": controls["detected"],
    "converted_by_path": converted_by_path,
    "opsets": opsets,
    "versions": versions,
    "artifacts": artifacts,
    "_ac_b2_detail": controls["detail"],  # extra diagnostic, not schema-required
    "_errors": errors,  # extra diagnostic, not schema-required
    "_artifact_subdir": h12,
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON.")
  parser.add_argument(
    "--artifacts-dir", required=True, type=Path, help="Base artifact directory (D-G)."
  )
  parser.add_argument(
    "--artifacts-subdir",
    default=None,
    help="Subdir under --artifacts-dir to write into (D-G artifact_subdir; default: git HEAD H12).",
  )
  parser.add_argument(
    "--buckets",
    default=",".join(str(b) for b in DEFAULT_BUCKETS),
    help="Comma-separated bucket ladder.",
  )
  parser.add_argument(
    "--dry-run", action="store_true", help="AC-B2 controls only; no checkpoint/conversion."
  )
  args = parser.parse_args(argv)

  if args.artifacts_subdir is None and not args.dry_run:
    import subprocess

    args.artifacts_subdir = subprocess.check_output(
      ["git", "rev-parse", "--short=12", "HEAD"], cwd=_SCRIPT_DIR, text=True
    ).strip()

  try:
    result = run(args)
    exit_code = 0
  except (FileNotFoundError, ValueError) as exc:
    # Integrity refusal (checkpoint sha mismatch, missing manifest row): exit 3
    # (result-emission rule #2). Still write a schema-valid result.
    logger.error("layer_b_build: integrity refusal: %s", exc)
    result = {
      "n_artifacts": 0,
      "n_converted": 0,
      "n_rng_clean": 0,
      "controls_total": CONTROLS_TOTAL,
      "controls_detected": 0,
      "converted_by_path": {},
      "opsets": {},
      "versions": {},
      "artifacts": {},
      "_integrity_error": str(exc),
    }
    exit_code = 3

  lac.emit(result, args.out)
  logger.info(
    "layer_b_build: n_artifacts=%s n_converted=%s n_rng_clean=%s controls=%s/%s exit_code=%d",
    result.get("n_artifacts"),
    result.get("n_converted"),
    result.get("n_rng_clean"),
    result.get("controls_detected"),
    result.get("controls_total"),
    exit_code,
  )
  return exit_code


if __name__ == "__main__":
  sys.exit(main())
