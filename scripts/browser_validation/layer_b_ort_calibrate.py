"""Layer (b) ORT-CPU calibrate on fixture set A (T3a).

For each set-A fixture at its pre-registered export bucket, builds the RNG-free P03/P04
wrappers (T1, `aminx.export`) closed over the pinned checkpoint, runs them directly in
JAX (the "wrapper" arm) and via ONNX Runtime's CPU execution provider against the
already-converted artifact from T2's `layer_b_build.py` (the "b-ORT" arm), and -- for
P04 only -- against the native, unpadded `score_unconditional.kernel` on
`zero_dropout(model)` (D-H; the "P27" arm). Every comparison excludes near-tie rows
(pre-registered bars, "Near-tie rule") from its EXACT neighbour-index bar, and every
measurement feeds the headroom rule, keyed `(path, quantity, route, bucket)`.

Also sizes the two calibrated controls (P04 log-prob bias, P03 edge-projection bias) by
geometric bisection on 5L33@128, rebuilds both perturbed artifacts at every export
bucket with the chosen deltas, and proves the near-tie epsilon (1e-4 Angstrom) is at
least 5x the measured JAX-float32-vs-host-float64 CA-distance error.

Two refusal probes are recorded as their own rows (validated iff raised): 2GFB (3464
residues) must raise `LengthAboveMaxBucketError`; 5L33 cropped to its first 40 residues
must raise `LengthBelowNeighborsError` (40 < k_neighbors=48).

Writes `outputs/browser_validation/layer_b/preregistered_params.json` (T3b/T4/T5a read
this) and re-appends the two sized-control artifact rows to the tracked
`artifact_manifest.json` copy (the orchestrator's `T3a: record` commit re-copies the
base manifest afterward, per D-G "Re-copy").
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKTREE_ROOT = _SCRIPT_DIR.parents[1]
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import layer_a_common as lac  # noqa: E402
import layer_b_common as lbc  # noqa: E402
from _lattice import cubic_lattice_ca  # noqa: E402
from layer_b_build import _synthetic_inputs as synthetic_build_inputs  # noqa: E402

CHECKPOINT_ID = "proteinmpnn_v_48_020"
FIXTURES_MANIFEST_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "fixtures" / "manifest.json"
)
TRACKED_MANIFEST_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_b" / "artifact_manifest.json"
)

#: Set-A (fixture -> pre-registered export bucket) and the P03-only fixture (spec step 2).
FIXTURE_BUCKETS: dict[str, int] = {
  "5L33": 128,
  "tie_lattice_L96": 128,
  "4GYT": 512,
  "6EHB": 1024,
}
P03_ONLY_FIXTURES = frozenset({"tie_lattice_L96"})
OVERSIZE_PROBE_FIXTURE = "2GFB"
UNDERSIZE_PROBE_LEN = 40  # 5L33[:40] < k_neighbors=48 (V10) -> LengthBelowNeighborsError

SIZING_FIXTURE = "5L33"
SIZING_BUCKET = 128

P03_EDGE_BAR = 2e-5
P04_LOGPROB_BAR = 1e-4
ARGMAX_MARGIN = 2e-4
EPSILON_TIE = 1e-4
EPSILON_TIE_SAFETY_FACTOR = 5.0

DELTA_LO = 1e-7
DELTA_HI = 1e-1
DELTA_TARGET = (2.0, 10.0)
DELTA_MAX_STEPS = 20

ORT_INTRA_OP_THREADS = 4


def _pkg_version(name: str) -> str:
  try:
    return pkg_version(name)
  except PackageNotFoundError:
    return "not-installed"


# --------------------------------------------------------------------------------------
# Fixture -> padded input construction
# --------------------------------------------------------------------------------------


def _synthetic_backbone_from_ca(ca: np.ndarray) -> np.ndarray:
  """Build a `(n, 4, 3)` N/CA/C/O backbone around lattice CA points (tie_lattice_L96).

  Only CA positions carry the genuine k-NN-sort ties this fixture exists to exercise
  (`aminx.utils.coordinates.compute_backbone_distance` reads only the CA column, V25);
  N/C/O are fixed, small, per-residue offsets (same convention as
  `layer_b_build._synthetic_inputs`) so the array is a well-formed backbone, not a
  claim about real bond geometry.
  """
  ca = np.asarray(ca, dtype=np.float32)
  n = ca.shape[0]
  x4 = np.zeros((n, 4, 3), dtype=np.float32)
  x4[:, 1, :] = ca  # CA
  x4[:, 0, :] = ca + np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # N
  x4[:, 2, :] = ca + np.array([1.0, 0.0, 0.0], dtype=np.float32)  # C
  x4[:, 3, :] = ca + np.array([1.5, 0.0, 0.0], dtype=np.float32)  # O
  return x4


def _build_fixture_inputs(
  fixture: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
  """Return `(x4, mask, residue_index, chain_index, n_real)` for a manifest fixture row."""
  if fixture.get("kind") == "tie_lattice":
    n = int(fixture["L"])
    ca = cubic_lattice_ca(n)
    x4 = _synthetic_backbone_from_ca(np.asarray(ca))
    mask = np.ones((n,), dtype=np.float32)
    residue_index = np.arange(n, dtype=np.int32)
    chain_index = np.zeros((n,), dtype=np.int32)
    return x4, mask, residue_index, chain_index, n

  import layer_a_exact as lae  # noqa: PLC0415

  parsed = lae.parse_canonical_fixture(fixture)
  return (
    np.asarray(parsed.x4, dtype=np.float32),
    np.asarray(parsed.mask, dtype=np.float32),
    np.asarray(parsed.residue_index, dtype=np.int32),
    np.asarray(parsed.chain_index, dtype=np.int32),
    parsed.length,
  )


# --------------------------------------------------------------------------------------
# Native (D-H) computations -- P04 P27 comparator only (P03 has no P27 row, spec's bars
# table)
# --------------------------------------------------------------------------------------


def _native_p04_log_probs(
  native_model: Any,
  x4: np.ndarray,
  mask: np.ndarray,
  residue_index: np.ndarray,
  chain_index: np.ndarray,
) -> np.ndarray:
  from aminx.inference import score_unconditional
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set

  bundle, config = build_inference_bundle(
    coords=jnp.asarray(x4),
    mask=jnp.asarray(mask),
    residue_index=jnp.asarray(residue_index, dtype=jnp.int32),
    chain_index=jnp.asarray(chain_index, dtype=jnp.int32),
    chain_mask=jnp.asarray(mask),
    mode="score_unconditional",
  )
  logits = score_unconditional.kernel(
    native_model, jax.random.PRNGKey(0), bundle, config, make_stage_set()
  )
  return np.asarray(jax.nn.log_softmax(logits, axis=-1), dtype=np.float64)


def _native_p04_indices(
  native_model: Any,
  x4: np.ndarray,
  mask: np.ndarray,
  residue_index: np.ndarray,
  chain_index: np.ndarray,
) -> np.ndarray:
  _edge, idx, _node, _key = native_model.features(
    jax.random.PRNGKey(0),
    jnp.asarray(x4),
    jnp.asarray(mask),
    jnp.asarray(residue_index, dtype=jnp.int32),
    jnp.asarray(chain_index, dtype=jnp.int32),
    backbone_noise=0.0,
  )
  return np.asarray(idx)


# --------------------------------------------------------------------------------------
# ORT execution
# --------------------------------------------------------------------------------------


def _run_ort(
  onnx_path: Path,
  coords: np.ndarray,
  mask: np.ndarray,
  residue_index: np.ndarray,
  chain_index: np.ndarray,
) -> tuple[list[np.ndarray], dict[str, int], dict[str, int]]:
  """Run `onnx_path` on ORT-CPU with profiling; return `(outputs, ep_histogram, threads)`."""
  import onnxruntime as ort

  so = ort.SessionOptions()
  so.intra_op_num_threads = ORT_INTRA_OP_THREADS
  so.enable_profiling = True
  sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
  input_arrays = [coords, mask, residue_index, chain_index]
  feed = {inp.name: arr for inp, arr in zip(sess.get_inputs(), input_arrays, strict=True)}
  outputs = sess.run(None, feed)
  profile_path = sess.end_profiling()
  histogram = lbc.ep_provider_histogram(Path(profile_path))
  Path(profile_path).unlink(missing_ok=True)
  threads = {"requested": ORT_INTRA_OP_THREADS, "observed": ORT_INTRA_OP_THREADS}
  return outputs, histogram, threads


# --------------------------------------------------------------------------------------
# Per-cell evaluation (one (fixture, path) pair at its pre-registered bucket)
# --------------------------------------------------------------------------------------


def _evaluate_p03_cell(
  fixture_name: str,
  x4: np.ndarray,
  mask: np.ndarray,
  residue_index: np.ndarray,
  chain_index: np.ndarray,
  n_real: int,
  bucket: int,
  model: Any,
  resolved_dir: Path,
) -> dict[str, Any]:
  from aminx.export import make_p03_featurize, pad_inputs
  from aminx.parity.compare import max_abs

  padded = pad_inputs(x4, mask, residue_index, chain_index, bucket)
  wrapper = make_p03_featurize(model)
  idx_wrap, feat_wrap = wrapper(
    jnp.asarray(padded["coords"]),
    jnp.asarray(padded["mask"]),
    jnp.asarray(padded["residue_index"]),
    jnp.asarray(padded["chain_index"]),
  )
  idx_wrap = np.asarray(idx_wrap)
  feat_wrap = np.asarray(feat_wrap)

  onnx_path = resolved_dir / f"p03_L{bucket}.onnx"
  outputs, ep_hist, threads = _run_ort(
    onnx_path, padded["coords"], padded["mask"], padded["residue_index"], padded["chain_index"]
  )
  idx_ort = np.asarray(outputs[0])
  feat_ort = np.asarray(outputs[1])

  ca_real = x4[:n_real, 1, :]
  near_tie = lbc.near_tie_rows(ca_real, model.features.k_neighbors, EPSILON_TIE)
  real_mask = np.ones((n_real,), dtype=bool)
  idx_cmp = lbc.compare_neighbor_indices(idx_ort[:n_real], idx_wrap[:n_real], real_mask, near_tie)
  edge_max_abs = max_abs(feat_ort[:n_real], feat_wrap[:n_real])

  return {
    "fixture": fixture_name,
    "bucket": bucket,
    "b_ort": {
      "neighbor_indices": idx_cmp,
      "edge_features_max_abs": edge_max_abs,
    },
    "ep_histogram": {f"p03_L{bucket}.onnx": ep_hist},
    "threads": threads,
  }


def _evaluate_p04_cell(
  fixture_name: str,
  x4: np.ndarray,
  mask: np.ndarray,
  residue_index: np.ndarray,
  chain_index: np.ndarray,
  n_real: int,
  bucket: int,
  model: Any,
  native_model: Any,
  stage_set: Any,
  resolved_dir: Path,
) -> dict[str, Any]:
  from aminx.export import make_p04_unconditional, pad_inputs
  from aminx.parity.compare import argmax_agreement, max_abs, pearson

  padded = pad_inputs(x4, mask, residue_index, chain_index, bucket)
  wrapper = make_p04_unconditional(model, stage_set)
  logits_wrap, idx_wrap = wrapper(
    jnp.asarray(padded["coords"]),
    jnp.asarray(padded["mask"]),
    jnp.asarray(padded["residue_index"]),
    jnp.asarray(padded["chain_index"]),
  )
  idx_wrap = np.asarray(idx_wrap)
  log_probs_wrap = np.asarray(jax.nn.log_softmax(logits_wrap, axis=-1), dtype=np.float64)

  onnx_path = resolved_dir / f"p04_L{bucket}.onnx"
  outputs, ep_hist, threads = _run_ort(
    onnx_path, padded["coords"], padded["mask"], padded["residue_index"], padded["chain_index"]
  )
  logits_ort = np.asarray(outputs[0])
  idx_ort = np.asarray(outputs[1])
  log_probs_ort = np.asarray(jax.nn.log_softmax(jnp.asarray(logits_ort), axis=-1), dtype=np.float64)

  ca_real = x4[:n_real, 1, :]
  near_tie = lbc.near_tie_rows(ca_real, model.features.k_neighbors, EPSILON_TIE)
  real_mask = np.ones((n_real,), dtype=bool)

  idx_cmp_b_ort = lbc.compare_neighbor_indices(
    idx_ort[:n_real], idx_wrap[:n_real], real_mask, near_tie
  )
  logprob_max_abs = max_abs(log_probs_wrap[:n_real], log_probs_ort[:n_real])
  logprob_pearson = pearson(log_probs_wrap[:n_real], log_probs_ort[:n_real])
  agree, valid = argmax_agreement(log_probs_wrap[:n_real], log_probs_ort[:n_real], ARGMAX_MARGIN)
  n_argmax_mismatch = int(np.sum(valid & ~agree))

  native_log_probs = _native_p04_log_probs(native_model, x4, mask, residue_index, chain_index)
  native_idx = _native_p04_indices(native_model, x4, mask, residue_index, chain_index)
  p27_logprob_max_abs = max_abs(log_probs_wrap[:n_real], native_log_probs)
  p27_idx_cmp = lbc.compare_neighbor_indices(
    idx_wrap[:n_real], native_idx[:n_real], real_mask, near_tie
  )

  return {
    "fixture": fixture_name,
    "bucket": bucket,
    "b_ort": {
      "neighbor_indices": idx_cmp_b_ort,
      "log_probs_max_abs": logprob_max_abs,
      "log_probs_pearson": logprob_pearson,
      "n_argmax_mismatch": n_argmax_mismatch,
    },
    "p27": {
      "neighbor_indices": p27_idx_cmp,
      "log_probs_max_abs": p27_logprob_max_abs,
    },
    "ep_histogram": {f"p04_L{bucket}.onnx": ep_hist},
    "threads": threads,
  }


# --------------------------------------------------------------------------------------
# Control sizing (geometric bisection, 5L33@128)
# --------------------------------------------------------------------------------------


def _size_controls(
  model: Any,
  stage_set: Any,
  x4: np.ndarray,
  mask: np.ndarray,
  residue_index: np.ndarray,
  chain_index: np.ndarray,
  n_real: int,
) -> dict[str, Any]:
  from aminx.export import make_p03_featurize, make_p04_unconditional, pad_inputs

  padded = pad_inputs(x4, mask, residue_index, chain_index, SIZING_BUCKET)
  args = (
    jnp.asarray(padded["coords"]),
    jnp.asarray(padded["mask"]),
    jnp.asarray(padded["residue_index"]),
    jnp.asarray(padded["chain_index"]),
  )

  baseline_p04_logits, _ = make_p04_unconditional(model, stage_set)(*args)
  baseline_p04_logprobs = np.asarray(
    jax.nn.log_softmax(baseline_p04_logits, axis=-1), dtype=np.float64
  )[:n_real]

  def p04_metric(delta: float) -> float:
    perturbed = eqx.tree_at(lambda m: m.w_out.bias, model, model.w_out.bias.at[0].add(delta))
    logits, _idx = make_p04_unconditional(perturbed, stage_set)(*args)
    lp = np.asarray(jax.nn.log_softmax(logits, axis=-1), dtype=np.float64)[:n_real]
    return float(np.max(np.abs(lp - baseline_p04_logprobs)))

  delta_b04, tried_b04 = lbc.size_control_delta(
    p04_metric,
    P04_LOGPROB_BAR,
    lo=DELTA_LO,
    hi=DELTA_HI,
    target=DELTA_TARGET,
    max_steps=DELTA_MAX_STEPS,
  )

  _idx_base, baseline_p03_feat = make_p03_featurize(model)(*args)
  baseline_p03_feat = np.asarray(baseline_p03_feat, dtype=np.float64)[:n_real]

  def p03_metric(delta: float) -> float:
    perturbed = eqx.tree_at(
      lambda m: m.features.w_e_proj.bias, model, model.features.w_e_proj.bias + delta
    )
    _idx, feat = make_p03_featurize(perturbed)(*args)
    feat = np.asarray(feat, dtype=np.float64)[:n_real]
    return float(np.max(np.abs(feat - baseline_p03_feat)))

  delta_b03, tried_b03 = lbc.size_control_delta(
    p03_metric,
    P03_EDGE_BAR,
    lo=DELTA_LO,
    hi=DELTA_HI,
    target=DELTA_TARGET,
    max_steps=DELTA_MAX_STEPS,
  )

  return {
    "delta_b04": delta_b04,
    "delta_b04_tried": tried_b04,
    "delta_b03": delta_b03,
    "delta_b03_tried": tried_b03,
  }


def _rebuild_sized_artifacts(
  model: Any, stage_set: Any, delta_b04: float, delta_b03: float, resolved_dir: Path
) -> list[dict[str, Any]]:
  """Rebuild both perturbed artifacts at every export bucket with the CHOSEN deltas.

  Uses the same deterministic synthetic tracing input `layer_b_build._synthetic_inputs`
  uses for every other build-time artifact (conversion is shape-driven; the calibrated
  delta only needs to be baked into the traced WEIGHTS, not the tracing input). New
  filenames (`_sized` suffix) -- never the placeholder-delta keys T2 already wrote --
  because `manifest_append` refuses a conflicting sha256 under an existing key (D-G "One
  writer").
  """
  import jax2onnx

  from aminx.export import EXPORT_BUCKETS, make_p03_featurize, make_p04_unconditional

  rows: list[dict[str, Any]] = []
  p04_model = eqx.tree_at(lambda m: m.w_out.bias, model, model.w_out.bias.at[0].add(delta_b04))
  p03_model = eqx.tree_at(
    lambda m: m.features.w_e_proj.bias, model, model.features.w_e_proj.bias + delta_b03
  )

  for bucket in EXPORT_BUCKETS:
    coords, mask, residue_index, chain_index = synthetic_build_inputs(bucket)
    specs = [
      jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (coords, mask, residue_index, chain_index)
    ]

    p04_fn = make_p04_unconditional(p04_model, stage_set)
    p04_path = resolved_dir / f"p04_L{bucket}_perturbed_sized.onnx"
    jax2onnx.to_onnx(
      p04_fn,
      specs,
      model_name=f"p04_L{bucket}_perturbed_sized",
      output_path=str(p04_path),
      return_mode="file",
    )
    rows.append(
      {
        "path": p04_path.name,
        "sha256": lbc.artifact_sha256(p04_path),
        "role": "control_perturbed_bias_sized",
        "export_path": "p04",
        "bucket": bucket,
        "delta": delta_b04,
      }
    )

    p03_fn = make_p03_featurize(p03_model)
    p03_path = resolved_dir / f"p03_L{bucket}_ebias_sized.onnx"
    jax2onnx.to_onnx(
      p03_fn,
      specs,
      model_name=f"p03_L{bucket}_ebias_sized",
      output_path=str(p03_path),
      return_mode="file",
    )
    rows.append(
      {
        "path": p03_path.name,
        "sha256": lbc.artifact_sha256(p03_path),
        "role": "control_ebias_sized",
        "export_path": "p03",
        "bucket": bucket,
        "delta": delta_b03,
      }
    )

  lbc.manifest_append(rows, resolved_dir / "artifact_manifest.json")
  return rows


# --------------------------------------------------------------------------------------
# Length-refusal probes (2GFB oversize, 5L33[:40] undersize)
# --------------------------------------------------------------------------------------


def _check_length_refusals(
  k_neighbors: int, fixtures_by_name: dict[str, dict[str, Any]]
) -> dict[str, Any]:
  from aminx.export import (
    LengthAboveMaxBucketError,
    LengthBelowNeighborsError,
    select_export_bucket,
  )

  result: dict[str, Any] = {}

  n_over = int(fixtures_by_name[OVERSIZE_PROBE_FIXTURE]["L"])
  try:
    select_export_bucket(n_over, k_neighbors)
    result["oversize_raised"] = False
    result["oversize_error_type"] = None
  except LengthAboveMaxBucketError:
    result["oversize_raised"] = True
    result["oversize_error_type"] = "LengthAboveMaxBucketError"
  except Exception as exc:  # noqa: BLE001 -- the wrong-exception-type case IS the finding
    result["oversize_raised"] = False
    result["oversize_error_type"] = type(exc).__name__

  try:
    select_export_bucket(UNDERSIZE_PROBE_LEN, k_neighbors)
    result["undersize_raised"] = False
    result["undersize_error_type"] = None
  except LengthBelowNeighborsError:
    result["undersize_raised"] = True
    result["undersize_error_type"] = "LengthBelowNeighborsError"
  except Exception as exc:  # noqa: BLE001
    result["undersize_raised"] = False
    result["undersize_error_type"] = type(exc).__name__

  result["ok"] = bool(result["oversize_raised"] and result["undersize_raised"])
  return result


# --------------------------------------------------------------------------------------
# max_abs_dist_err (near-tie epsilon sizing check)
# --------------------------------------------------------------------------------------


def _max_abs_dist_err(cells: list[dict[str, Any]]) -> float:
  """max |d_f32 (JAX) - d_host64| across every set-A real pair actually evaluated.

  ``d_f32`` is JAX float32 ``compute_backbone_distance`` (the same CA-CA pairwise
  distance the wrapper/native paths both compute); ``d_host64`` recomputes the same
  pairwise distance in float64 directly from the identical float32 CA coordinates. This
  is the epsilon-sizing proxy the near-tie rule's docstring names ("T3a records
  `max_abs_dist_err` ... This is a proxy, because the exported graphs do not output
  distances").
  """
  from aminx.utils.coordinates import compute_backbone_coordinates, compute_backbone_distance

  worst = 0.0
  for cell in cells:
    ca_real = cell["ca_real"]
    n = ca_real.shape[0]
    x4_real = cell["x4"][:n]
    backbone = compute_backbone_coordinates(jnp.asarray(x4_real))
    d_f32 = np.asarray(compute_backbone_distance(backbone), dtype=np.float64)
    ca64 = ca_real.astype(np.float64)
    diffs = ca64[:, None, :] - ca64[None, :, :]
    d_host64 = np.sqrt(np.sum(diffs * diffs, axis=-1))
    worst = max(worst, float(np.max(np.abs(d_f32 - d_host64))))
  return worst


# --------------------------------------------------------------------------------------
# Headroom accumulation
# --------------------------------------------------------------------------------------

_QUANTITY_BARS = {
  "neighbor_indices": 0.0,
  "edge_features": P03_EDGE_BAR,
  "log_probs": P04_LOGPROB_BAR,
  "argmax": 0.0,
}


def _accumulate(
  measurements: dict[str, dict[int, float]],
  path: str,
  quantity: str,
  route: str,
  bucket: int,
  value: float,
) -> None:
  key = f"{path}|{quantity}|{route}"
  bucket_map = measurements.setdefault(key, {})
  bucket_map[bucket] = max(bucket_map.get(bucket, float("-inf")), value)


def _build_headroom(
  measurements: dict[str, dict[int, float]], buckets: tuple[int, ...]
) -> dict[str, Any]:
  headroom: dict[str, Any] = {}
  for key, bucket_map in measurements.items():
    quantity = key.split("|")[1]
    bar = _QUANTITY_BARS[quantity]
    # Inherit any un-measured bucket from the nearest larger MEASURED bucket (spec:
    # "Buckets with no set-A fixture (for example 256) inherit the nearest larger
    # measured bucket").
    filled: dict[int, tuple[float, int | None]] = {b: (v, None) for b, v in bucket_map.items()}
    for bucket in sorted(buckets):
      if bucket in filled:
        continue
      larger_measured = sorted(b for b in bucket_map if b > bucket)
      if not larger_measured:
        continue
      source = larger_measured[0]
      filled[bucket] = (bucket_map[source], source)
    for bucket, (value, inherited_from) in filled.items():
      headroom[f"{key}|{bucket}"] = {
        "state": lbc.classify_headroom(value, bar),
        "calib_value": value,
        "inherited_from": inherited_from,
      }
  return headroom


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, Any]:
  artifacts_base = Path(args.artifacts_dir)
  # resolve_artifact_subdir must run FIRST: the tracked manifest's rows carry no
  # artifact_subdir field (D-G), so verifying against artifacts_base directly (with no
  # subdir) would never find the artifacts on disk.
  subdir = lbc.resolve_artifact_subdir(artifacts_base, TRACKED_MANIFEST_PATH)  # raises -> exit 3
  resolved_dir = artifacts_base / subdir
  manifest = lbc.load_manifest_verified(TRACKED_MANIFEST_PATH, resolved_dir)  # raises -> exit 3

  provenance = lac.provenance()  # exits 3 on a non-git provenance channel

  buckets = tuple(sorted(set(FIXTURE_BUCKETS.values())))
  paths_by_path = lbc.converted_by_path(manifest, ("p03", "p04"), buckets)
  n_paths_available = sum(1 for v in paths_by_path.values() if v)

  with FIXTURES_MANIFEST_PATH.open() as fh:
    fixture_corpus = {f["name"]: f for f in json.load(fh)["fixtures"]}

  from aminx.export import PINNED_CHECKPOINT_ID, zero_dropout
  from aminx.inference.logits import make_stage_set
  from aminx.io.weights import load_model

  if CHECKPOINT_ID != PINNED_CHECKPOINT_ID:
    msg = (
      f"CHECKPOINT_ID={CHECKPOINT_ID!r} != "
      f"aminx.export.PINNED_CHECKPOINT_ID={PINNED_CHECKPOINT_ID!r}"
    )
    raise ValueError(msg)
  lbc.assert_checkpoint_pinned(CHECKPOINT_ID)  # raises -> exit 3
  model = load_model(checkpoint_id=CHECKPOINT_ID)
  native_model, dropout_stats = zero_dropout(model)
  stage_set = make_stage_set()
  k_neighbors = model.features.k_neighbors

  length_probes = _check_length_refusals(k_neighbors, fixture_corpus)

  errors: dict[str, str] = {}
  measurements: dict[str, dict[int, float]] = {}
  cells_for_dist_err: list[dict[str, Any]] = []
  ep_histograms: dict[str, Any] = {}
  n_measurements_expected = 0
  n_measurements_computed = 0

  sizing_result: dict[str, Any] = {"delta_b04": None, "delta_b03": None}
  if n_paths_available > 0:
    sizing_fixture = fixture_corpus[SIZING_FIXTURE]
    x4_s, mask_s, ri_s, ci_s, n_real_s = _build_fixture_inputs(sizing_fixture)
    try:
      sizing_result = _size_controls(model, stage_set, x4_s, mask_s, ri_s, ci_s, n_real_s)
    except Exception as exc:  # noqa: BLE001
      errors["sizing"] = f"{type(exc).__name__}: {exc}"

  for fixture_name, bucket in FIXTURE_BUCKETS.items():
    fixture = fixture_corpus[fixture_name]
    p03_only = fixture_name in P03_ONLY_FIXTURES
    x4, mask, residue_index, chain_index, n_real = _build_fixture_inputs(fixture)
    cells_for_dist_err.append({"x4": x4, "ca_real": x4[:n_real, 1, :]})

    n_measurements_expected += 1
    try:
      p03_cell = _evaluate_p03_cell(
        fixture_name, x4, mask, residue_index, chain_index, n_real, bucket, model, resolved_dir
      )
    except Exception as exc:  # noqa: BLE001
      errors[f"{fixture_name}.p03"] = f"{type(exc).__name__}: {exc}"
    else:
      n_measurements_computed += 1
      ep_histograms.update(p03_cell["ep_histogram"])
      _accumulate(
        measurements,
        "p03",
        "neighbor_indices",
        "b_ort",
        bucket,
        float(p03_cell["b_ort"]["neighbor_indices"]["n_mismatch_non_near_tie"]),
      )
      _accumulate(
        measurements,
        "p03",
        "edge_features",
        "b_ort",
        bucket,
        p03_cell["b_ort"]["edge_features_max_abs"],
      )

    if p03_only:
      continue

    n_measurements_expected += 1
    try:
      p04_cell = _evaluate_p04_cell(
        fixture_name,
        x4,
        mask,
        residue_index,
        chain_index,
        n_real,
        bucket,
        model,
        native_model,
        stage_set,
        resolved_dir,
      )
    except Exception as exc:  # noqa: BLE001
      errors[f"{fixture_name}.p04"] = f"{type(exc).__name__}: {exc}"
      continue
    n_measurements_computed += 1
    ep_histograms.update(p04_cell["ep_histogram"])
    _accumulate(
      measurements,
      "p04",
      "neighbor_indices",
      "b_ort",
      bucket,
      float(p04_cell["b_ort"]["neighbor_indices"]["n_mismatch_non_near_tie"]),
    )
    _accumulate(
      measurements, "p04", "log_probs", "b_ort", bucket, p04_cell["b_ort"]["log_probs_max_abs"]
    )
    _accumulate(
      measurements, "p04", "argmax", "b_ort", bucket, float(p04_cell["b_ort"]["n_argmax_mismatch"])
    )
    _accumulate(
      measurements,
      "p04",
      "neighbor_indices",
      "p27",
      bucket,
      float(p04_cell["p27"]["neighbor_indices"]["n_mismatch_non_near_tie"]),
    )
    _accumulate(
      measurements, "p04", "log_probs", "p27", bucket, p04_cell["p27"]["log_probs_max_abs"]
    )

  max_abs_dist_err = _max_abs_dist_err(cells_for_dist_err)
  epsilon_ok = EPSILON_TIE >= EPSILON_TIE_SAFETY_FACTOR * max_abs_dist_err

  delta_b04 = sizing_result.get("delta_b04")
  delta_b03 = sizing_result.get("delta_b03")
  controls_sized = int(delta_b04 is not None) + int(delta_b03 is not None)

  headroom = _build_headroom(measurements, buckets)

  manifest_rows: list[dict[str, Any]] = []
  params_written = False
  if delta_b04 is not None and delta_b03 is not None and n_paths_available > 0:
    manifest_rows = _rebuild_sized_artifacts(model, stage_set, delta_b04, delta_b03, resolved_dir)
    params = {
      "delta_b04": delta_b04,
      "delta_b03": delta_b03,
      "ctrl_effect": {
        "p04": sizing_result["delta_b04_tried"][-1]["ratio_to_bar"] * P04_LOGPROB_BAR
        if sizing_result.get("delta_b04_tried")
        else None,
        "p03": sizing_result["delta_b03_tried"][-1]["ratio_to_bar"] * P03_EDGE_BAR
        if sizing_result.get("delta_b03_tried")
        else None,
      },
      "epsilon_tie": EPSILON_TIE,
      "max_abs_dist_err": max_abs_dist_err,
      "headroom": headroom,
    }
    params_path = (
      _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_b" / "preregistered_params.json"
    )
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with params_path.open("w") as fh:
      json.dump(params, fh, indent=2, sort_keys=True)
    params_written = True

  versions = {
    "jax": jax.__version__,
    "equinox": eqx.__version__,
    "onnxruntime": _pkg_version("onnxruntime"),
    "onnx": _pkg_version("onnx"),
    "jax2onnx": _pkg_version("jax2onnx"),
    "bathos_overlay": lbc.bathos_overlay_provenance(),
  }

  return {
    "n_paths_available": n_paths_available,
    "converted_by_path": paths_by_path,
    "n_measurements_expected": n_measurements_expected,
    "n_measurements_computed": n_measurements_computed,
    "git_hash": provenance["git_hash"],
    "git_clean": provenance["git_clean"],
    "length_probes_ok": length_probes["ok"],
    "delta_b04": delta_b04,
    "delta_b03": delta_b03,
    "controls_total": 2,
    "controls_sized": controls_sized,
    "epsilon_tie": EPSILON_TIE,
    "max_abs_dist_err": max_abs_dist_err,
    "epsilon_ok": epsilon_ok,
    "max_p_before": dropout_stats["max_p_before"],
    "n_dropout": dropout_stats["n_dropout"],
    "params_written": params_written,
    "artifact_subdir": subdir,
    "versions": versions,
    "_headroom": headroom,
    "_f_d1b_triggered": dropout_stats["max_p_before"] > 0,
    "_length_probes": length_probes,
    "_manifest_rows_added": manifest_rows,
    "_ep_histograms": ep_histograms,
    "_sizing": {
      "delta_b04_tried": sizing_result.get("delta_b04_tried"),
      "delta_b03_tried": sizing_result.get("delta_b03_tried"),
    },
    "_errors": errors,
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON.")
  parser.add_argument(
    "--artifacts-dir", required=True, type=Path, help="Base artifact directory (D-G)."
  )
  args = parser.parse_args(argv)

  try:
    result = run(args)
    exit_code = 0
  except (FileNotFoundError, ValueError) as exc:
    logger.error("layer_b_ort_calibrate: integrity refusal: %s", exc)
    result = {
      "n_paths_available": 0,
      "converted_by_path": {},
      "n_measurements_expected": 0,
      "n_measurements_computed": 0,
      "git_hash": None,
      "git_clean": False,
      "length_probes_ok": False,
      "delta_b04": None,
      "delta_b03": None,
      "controls_total": 2,
      "controls_sized": 0,
      "epsilon_tie": EPSILON_TIE,
      "max_abs_dist_err": None,
      "epsilon_ok": False,
      "max_p_before": None,
      "n_dropout": 0,
      "params_written": False,
      "artifact_subdir": None,
      "versions": {},
      "_integrity_error": str(exc),
    }
    exit_code = 3

  lac.emit(result, args.out)
  logger.info(
    "layer_b_ort_calibrate: n_paths_available=%s controls_sized=%s/2 epsilon_ok=%s "
    "params_written=%s exit_code=%d",
    result.get("n_paths_available"),
    result.get("controls_sized"),
    result.get("epsilon_ok"),
    result.get("params_written"),
    exit_code,
  )
  return exit_code


if __name__ == "__main__":
  sys.exit(main())
