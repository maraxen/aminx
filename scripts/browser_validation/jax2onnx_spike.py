"""Phase 0, T2: jax2onnx -> ONNX Runtime (CPU EP) spike, incl. the multi-key sort tie test (AC-2).

Attempts to convert the REAL P05 conditional-logits kernel
(``aminx.inference.score_conditional.kernel`` on the shipped
``proteinmpnn_v_48_020`` checkpoint, L = 128, structure parsed from
``$REFERENCE_PATH/inputs/1BC8.pdb`` by aminx's own parser, noise 0) via
``jax2onnx.to_onnx``, then executes the result on ONNX Runtime's CPU
execution provider and compares against the JAX baseline: log-prob max-abs
and EXACT neighbor indices. A genuine positive control (``model.w_out.bias[0]
+= 5e-4``) must be detected in the converted+executed graph.

Independently of whether P05 converts, this also converts
``aminx.model.features.top_k`` ALONE on a tie-rich cubic-lattice input
(``_lattice.py``) and checks that ORT-CPU and JAX select EXACTLY the same
neighbor indices -- the model's ``top_k`` breaks ties by folding the index
into the sort key (a strict total order), and this is the only way to find
out whether ONNX Runtime's execution of that multi-key sort preserves the
same tie order.

A failure to convert P05 is recorded verbatim (error text + offending JAX
primitive, when identifiable) rather than swallowed -- this is itself the
finding this spike exists to make. ORT Python's CPU EP is not ORT Web; a
"converted" result here is NOT browser evidence (see T3).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Callable, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from _lattice import cubic_lattice_ca, lattice_neighbor_logits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROTEIN_CKPT = "proteinmpnn_v_48_020"
K_NEIGHBORS = 48
DEFAULT_REFERENCE_PATH = "/home/marielle/projects/aminx/reference_ligandmpnn_clone"
CONTROL_BIAS_DELTA = 5e-4
CONTROL_MIN_MAX_ABS = 2e-4
ORT_MAX_ABS_BAR = 1e-4

# Fixed, legacy-style (raw uint32 array) PRNG key: jax2onnx's constant-folding
# path calls `np.asarray` on every closed-over jaxpr constant, which raises on
# a new-style typed key (`jax.random.key`) -- see the recorded conversion
# error below for the analogous failure on `jax.random.split` itself. Using
# the legacy representation clears that specific failure and lets the REAL
# blocker (the `random_split` primitive itself) surface.
_FIXED_KEY = jax.random.PRNGKey(0)


def _reference_path() -> Path:
  """$REFERENCE_PATH, defaulting to the pinned local LigandMPNN clone."""
  return Path(os.environ.get("REFERENCE_PATH", DEFAULT_REFERENCE_PATH))


def _sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _package_version(name: str) -> str:
  try:
    return pkg_version(name)
  except PackageNotFoundError:
    return "unavailable"


def build_p05_inputs(bucket: int) -> dict[str, Any]:
  """Parse 1BC8.pdb with aminx's own parser and pad to `bucket` (noise 0).

  Returns a dict of the five raw arrays `aminx.inference.bundle_builder
  .build_inference_bundle` needs for `mode="score_conditional"`:
  coords (bucket, 4, 3), mask, residue_index, chain_index, sequence
  (all (bucket,)).
  """
  from proxide.core.containers import atom_order

  from aminx.io.parsing import parse_structure

  pdb_path = _reference_path() / "inputs" / "1BC8.pdb"
  protein = parse_structure(str(pdb_path), k_neighbors=K_NEIGHBORS)

  # proxide's `coordinates` is atom37; aminx's bundle wants the (N, CA, C, O)
  # 4-atom backbone slice, in that order (matches
  # `aminx.utils.coordinates.compute_backbone_coordinates`'s own indexing).
  backbone_idx = [atom_order["N"], atom_order["CA"], atom_order["C"], atom_order["O"]]
  coords = jnp.asarray(protein.coordinates)[:, backbone_idx, :]
  mask = jnp.asarray(protein.mask).astype(jnp.float32)
  residue_index = jnp.asarray(protein.residue_index)
  chain_index = jnp.asarray(protein.chain_index)
  sequence = jnp.asarray(protein.aatype)

  real_length = coords.shape[0]
  if real_length > bucket:
    msg = f"1BC8.pdb has {real_length} real residues, which exceeds --bucket {bucket}"
    raise ValueError(msg)

  n_pad = bucket - real_length
  coords_p = jnp.pad(coords, ((0, n_pad), (0, 0), (0, 0)), constant_values=0.0)
  sequence_p = jnp.pad(sequence, (0, n_pad), constant_values=0)
  mask_p = jnp.pad(mask, (0, n_pad), constant_values=False)
  residue_index_p = jnp.pad(residue_index, (0, n_pad), constant_values=int(residue_index[-1]))
  chain_index_p = jnp.pad(chain_index, (0, n_pad), constant_values=int(chain_index[-1]))

  return {
    "coords": coords_p,
    "mask": mask_p,
    "residue_index": residue_index_p,
    "chain_index": chain_index_p,
    "sequence": sequence_p,
    "real_length": real_length,
  }


def make_p05_export_fn(model: Any, stage_set: Any) -> Callable[..., tuple[jax.Array, jax.Array]]:
  """Build the pure (coords, mask, residue_index, chain_index, sequence) -> (logits, nbr_idx) fn.

  Reimplements `aminx.inference.score_conditional.kernel`'s body (encode then
  score_from_encoding, same key split) so the second output -- neighbor
  indices, needed for the EXACT-match check -- can be exposed without a
  second, possibly-diverging encode() call.
  """
  from aminx.inference import score_conditional
  from aminx.inference.bundle_builder import build_inference_bundle

  def fn(
    coords: jax.Array,
    mask: jax.Array,
    residue_index: jax.Array,
    chain_index: jax.Array,
    sequence: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    bundle, config = build_inference_bundle(
      coords=coords,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      sequence=sequence,
      mode="score_conditional",
    )
    k_enc, k_dec = jax.random.split(_FIXED_KEY)
    enc = score_conditional.encode(model, k_enc, bundle, config)
    logits = score_conditional.score_from_encoding(model, k_dec, enc, bundle, config, stage_set)
    return cast("jax.Array", logits), enc.neighbor_indices

  return fn


def perturb_w_out_bias(model: Any, delta: float = CONTROL_BIAS_DELTA) -> Any:
  """Return a copy of `model` with `w_out.bias[0] += delta` (genuine weight perturbation)."""
  new_bias = model.w_out.bias.at[0].add(delta)
  return eqx.tree_at(lambda m: m.w_out.bias, model, new_bias)


def _extract_primitive(exc: Exception) -> str:
  """Best-effort extraction of the offending JAX primitive name from a jax2onnx error."""
  text = str(exc)
  match = re.search(r"primitive '([^']+)'", text)
  if match:
    return match.group(1)
  match = re.search(r"for '([a-zA-Z0-9_.]+)' not implemented", text)
  if match:
    return match.group(1)
  return "unknown"


def onnx_inputs_for(arrays: dict[str, jax.Array], names: list[str]) -> list[Any]:
  """jax.ShapeDtypeStruct spec list, in `names` order, from a dict of concrete arrays."""
  return [jax.ShapeDtypeStruct(arrays[n].shape, arrays[n].dtype) for n in names]


def run_p05_conversion(
  bucket: int,
  out_dir: Path,
) -> dict[str, Any]:
  """Attempt P05 conversion + ORT-CPU execution + control detection.

  Returns a dict with `converted`, and either the failure fields
  (`conversion_error`, `conversion_primitive`) or the success fields
  (`ort_max_abs`, `nbr_exact`, `control_detected`, `control_max_abs`), plus
  the artifact paths it actually wrote.
  """
  import jax2onnx
  import onnxruntime as ort

  from aminx.inference.logits import make_stage_set
  from aminx.io.weights import load_model

  inputs = build_p05_inputs(bucket)
  input_names = ["coords", "mask", "residue_index", "chain_index", "sequence"]

  model = eqx.nn.inference_mode(load_model(checkpoint_id=PROTEIN_CKPT), value=True)
  stage_set = make_stage_set()

  fn = make_p05_export_fn(model, stage_set)
  jax_logits, jax_neighbor_indices = fn(*(inputs[n] for n in input_names))

  # Write the JAX-side inputs/outputs regardless of ONNX conversion outcome:
  # they are the reproducibility record for Phase 2's RNG-outside-the-graph
  # wrapper redesign, whether or not this v1 kernel converts today.
  inputs_npz_path = out_dir / "p05_inputs.npz"
  np.savez(
    inputs_npz_path,
    coords=np.asarray(inputs["coords"]),
    mask=np.asarray(inputs["mask"]),
    residue_index=np.asarray(inputs["residue_index"]),
    chain_index=np.asarray(inputs["chain_index"]),
    sequence=np.asarray(inputs["sequence"]),
    real_length=np.asarray(inputs["real_length"]),
  )
  outputs_npz_path = out_dir / "p05_jax_outputs.npz"
  np.savez(
    outputs_npz_path,
    logits=np.asarray(jax_logits),
    neighbor_indices=np.asarray(jax_neighbor_indices),
  )

  onnx_inputs = onnx_inputs_for(inputs, input_names)
  onnx_path = out_dir / "p05_L128.onnx"
  result: dict[str, Any] = {
    "artifact_paths": {
      "p05_inputs.npz": inputs_npz_path,
      "p05_jax_outputs.npz": outputs_npz_path,
    },
  }

  try:
    jax2onnx.to_onnx(
      fn, onnx_inputs, model_name="p05_L128", output_path=str(onnx_path), return_mode="file"
    )
  except Exception as e:  # noqa: BLE001 - the exception text itself is the finding
    logger.warning("P05 conversion failed: %s: %s", type(e).__name__, e)
    result["converted"] = False
    result["conversion_error"] = f"{type(e).__name__}: {e}"
    result["conversion_primitive"] = _extract_primitive(e)
    return result

  logger.info("P05 converted successfully to %s", onnx_path)
  result["converted"] = True
  result["artifact_paths"]["p05_L128.onnx"] = onnx_path

  sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  ort_input_names = [i.name for i in sess.get_inputs()]
  feed = {
    name: np.asarray(inputs[key]) for name, key in zip(ort_input_names, input_names, strict=True)
  }
  ort_logits, ort_neighbor_indices = sess.run(None, feed)

  ort_max_abs = float(np.max(np.abs(np.asarray(ort_logits) - np.asarray(jax_logits))))
  nbr_exact = bool(
    np.array_equal(np.asarray(ort_neighbor_indices), np.asarray(jax_neighbor_indices))
  )
  result["ort_max_abs"] = ort_max_abs
  result["nbr_exact"] = nbr_exact
  logger.info("P05 ORT-CPU vs JAX: max_abs=%s nbr_exact=%s", ort_max_abs, nbr_exact)

  # Control: a genuine perturbation of model.w_out.bias[0], converted and
  # executed the SAME way, compared against the UNPERTURBED JAX baseline.
  perturbed_model = perturb_w_out_bias(model)
  perturbed_fn = make_p05_export_fn(perturbed_model, stage_set)
  perturbed_onnx_path = out_dir / "p05_L128_perturbed.onnx"
  try:
    jax2onnx.to_onnx(
      perturbed_fn,
      onnx_inputs,
      model_name="p05_L128_perturbed",
      output_path=str(perturbed_onnx_path),
      return_mode="file",
    )
    sess_perturbed = ort.InferenceSession(
      str(perturbed_onnx_path), providers=["CPUExecutionProvider"]
    )
    perturbed_ort_logits, _ = sess_perturbed.run(None, feed)
    control_max_abs = float(
      np.max(np.abs(np.asarray(perturbed_ort_logits) - np.asarray(jax_logits)))
    )
    control_detected = control_max_abs >= CONTROL_MIN_MAX_ABS
    result["artifact_paths"]["p05_L128_perturbed.onnx"] = perturbed_onnx_path
  except Exception as e:  # noqa: BLE001
    logger.warning("Perturbed-control conversion/execution failed: %s: %s", type(e).__name__, e)
    control_max_abs = None
    control_detected = False
    result["control_error"] = f"{type(e).__name__}: {e}"

  result["control_max_abs"] = control_max_abs
  result["control_detected"] = control_detected
  logger.info(
    "Control (w_out.bias[0] += %s): max_abs=%s detected=%s",
    CONTROL_BIAS_DELTA,
    control_max_abs,
    control_detected,
  )

  return result


def run_tie_lattice_conversion(bucket: int, out_dir: Path) -> dict[str, Any]:
  """Convert `aminx.model.features.top_k` ALONE on a tie-rich cubic-lattice input.

  Independent of whether P05 converts -- this exercises only the multi-key
  sort primitive, not the RNG-carrying featurizer/encoder around it.
  """
  import jax2onnx
  import onnxruntime as ort

  from aminx.model.features import top_k

  coords = cubic_lattice_ca(bucket, spacing=1.0)
  x = lattice_neighbor_logits(coords)
  k = min(K_NEIGHBORS, bucket)

  def fn(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return top_k(x, k)

  jax_values, jax_indices = fn(x)

  io_npz_path = out_dir / "topk_lattice_io.npz"
  np.savez(
    io_npz_path,
    coords=np.asarray(coords),
    x=np.asarray(x),
    k=np.asarray(k),
    jax_values=np.asarray(jax_values),
    jax_indices=np.asarray(jax_indices),
  )

  result: dict[str, Any] = {"artifact_paths": {"topk_lattice_io.npz": io_npz_path}}

  onnx_path = out_dir / "topk_lattice.onnx"
  try:
    jax2onnx.to_onnx(
      fn,
      [jax.ShapeDtypeStruct(x.shape, x.dtype)],
      model_name="topk_lattice",
      output_path=str(onnx_path),
      return_mode="file",
    )
  except Exception as e:  # noqa: BLE001
    logger.warning("Tie-lattice top_k conversion failed: %s: %s", type(e).__name__, e)
    result["tie_converted"] = False
    result["tie_conversion_error"] = f"{type(e).__name__}: {e}"
    result["tie_indices_identical"] = False
    return result

  result["tie_converted"] = True
  result["artifact_paths"]["topk_lattice.onnx"] = onnx_path

  sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  ort_values, ort_indices = sess.run(None, {sess.get_inputs()[0].name: np.asarray(x)})
  tie_indices_identical = bool(np.array_equal(np.asarray(ort_indices), np.asarray(jax_indices)))
  result["tie_indices_identical"] = tie_indices_identical
  logger.info("Tie-lattice top_k: ORT-CPU vs JAX indices identical=%s", tie_indices_identical)
  return result


def run_p07_conversion_attempt(bucket: int) -> dict[str, Any]:
  """Optional, non-gating: attempt converting the P07 host-order wrapper.

  `aminx.types.bundles.WaveScheduleBundle.from_tie_groups` is documented
  HOST-SIDE ONLY (it calls `.tolist()` on its inputs), so it cannot itself
  accept a traced decoding-order input; the wave schedule is built once,
  host-side, from a fixed `arange(bucket)` order and closed over, and only
  `backbone_noise` is exposed as a traced (scalar) input. Whatever the result,
  this does not gate the sidecar outcome.
  """
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set
  from aminx.inference.sample_autoregressive import kernel as sample_autoregressive_kernel
  from aminx.io.weights import load_model
  from aminx.types.bundles import WaveScheduleBundle

  result: dict[str, Any] = {
    "attempted": True,
    "note": (
      "WaveScheduleBundle.from_tie_groups is host-side only (.tolist() internally) and "
      "cannot accept a traced decoding-order input; host_order is baked in as a fixed "
      "arange(bucket) and only backbone_noise is exposed as a traced input."
    ),
  }
  try:
    import jax2onnx

    model = eqx.nn.inference_mode(load_model(checkpoint_id=PROTEIN_CKPT), value=True)
    stage_set = make_stage_set()
    inputs = build_p05_inputs(bucket)
    host_order = jnp.arange(bucket)
    wave = WaveScheduleBundle.from_tie_groups(jnp.arange(bucket), host_order)

    def fn(
      coords: jax.Array,
      mask: jax.Array,
      residue_index: jax.Array,
      chain_index: jax.Array,
      backbone_noise: jax.Array,
    ) -> Any:
      bundle, config = build_inference_bundle(
        coords=coords,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        wave=wave,
        backbone_noise=backbone_noise,
        mode="sample_ar",
      )
      return sample_autoregressive_kernel(model, _FIXED_KEY, bundle, config, stage_set)

    onnx_inputs = [
      jax.ShapeDtypeStruct(inputs["coords"].shape, inputs["coords"].dtype),
      jax.ShapeDtypeStruct(inputs["mask"].shape, inputs["mask"].dtype),
      jax.ShapeDtypeStruct(inputs["residue_index"].shape, inputs["residue_index"].dtype),
      jax.ShapeDtypeStruct(inputs["chain_index"].shape, inputs["chain_index"].dtype),
      jax.ShapeDtypeStruct((), jnp.float32),
    ]
    jax2onnx.to_onnx(fn, onnx_inputs, model_name="p07_host_order")
    result["converted"] = True
  except Exception as e:  # noqa: BLE001
    logger.info("P07 host-order wrapper conversion (non-gating): %s: %s", type(e).__name__, e)
    result["converted"] = False
    result["error"] = f"{type(e).__name__}: {e}"
    result["primitive"] = _extract_primitive(e)
  return result


def _write_result(out_path: Path, result: dict[str, Any]) -> None:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w") as f:
    json.dump(result, f, indent=2, default=str)
  results_path = os.environ.get("BTH_RESULTS_PATH")
  if results_path:
    with Path(results_path).open("w") as f:
      json.dump(result, f, indent=2, default=str)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON.")
  parser.add_argument("--bucket", type=int, default=128, help="Padded sequence length L.")
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Check the environment and write a stub result without converting or running anything.",
  )
  args = parser.parse_args(argv)

  versions = {
    "jax": jax.__version__,
    "equinox": eqx.__version__,
    "numpy": np.__version__,
    "jax2onnx": _package_version("jax2onnx"),
    "onnxruntime": _package_version("onnxruntime"),
    "onnx": _package_version("onnx"),
  }
  logger.info("versions: %s", versions)

  if args.dry_run:
    result = {
      "dry_run": True,
      "versions": versions,
      "converted": False,
      "note": "ORT Python CPU EP is not ORT Web; this is not browser evidence.",
    }
    _write_result(args.out, result)
    logger.info("dry-run complete; wrote stub result to %s", args.out)
    return 0

  out_dir = args.out.parent
  out_dir.mkdir(parents=True, exist_ok=True)

  p05_result = run_p05_conversion(args.bucket, out_dir)
  tie_result = run_tie_lattice_conversion(args.bucket, out_dir)
  p07_result = run_p07_conversion_attempt(args.bucket)

  artifact_paths: dict[str, Path] = {}
  artifact_paths.update(p05_result.pop("artifact_paths", {}))
  artifact_paths.update(tie_result.pop("artifact_paths", {}))

  artifacts: dict[str, Any] = {}
  for name, path in artifact_paths.items():
    if path.is_file():
      artifacts[name] = {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}

  result = {
    "versions": versions,
    "bucket": args.bucket,
    "k_neighbors": K_NEIGHBORS,
    "converted": p05_result["converted"],
    "ort_max_abs": p05_result.get("ort_max_abs"),
    "nbr_exact": p05_result.get("nbr_exact"),
    "control_max_abs": p05_result.get("control_max_abs"),
    "control_detected": p05_result.get("control_detected", False),
    "tie_converted": tie_result["tie_converted"],
    "tie_indices_identical": tie_result["tie_indices_identical"],
    "p07_conversion": p07_result,
    "artifacts": artifacts,
    "note": "ORT Python CPU EP is not ORT Web; this is not browser evidence.",
  }
  if "conversion_error" in p05_result:
    result["conversion_error"] = p05_result["conversion_error"]
    result["conversion_primitive"] = p05_result["conversion_primitive"]
  if "control_error" in p05_result:
    result["control_error"] = p05_result["control_error"]
  if "tie_conversion_error" in tie_result:
    result["tie_conversion_error"] = tie_result["tie_conversion_error"]

  _write_result(args.out, result)
  logger.info(
    "jax2onnx spike complete: converted=%s ort_max_abs=%s nbr_exact=%s control_detected=%s "
    "tie_indices_identical=%s",
    result["converted"],
    result["ort_max_abs"],
    result["nbr_exact"],
    result["control_detected"],
    result["tie_indices_identical"],
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
