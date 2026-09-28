"""Layer (c) browser (ORT Web wasm EP, headless Chromium) calibrate on fixture set A (T5a).

For each set-A fixture at its pre-registered export bucket (same set as T3a: 5L33@128,
tie_lattice_L96@128 [P03 only], 4GYT@512, 6EHB@1024), runs the ALREADY-CONVERTED, tracked
T2 ONNX artifacts (`outputs/browser_validation/layer_b/artifact_manifest.json`, D-G) a
SECOND time -- through onnxruntime-web's wasm execution provider inside headless
Chromium, `numThreads = 1` (`browser/layer_c/`) -- and compares against the SAME artifact
run through ONNX Runtime's native CPU execution provider (the "c" bars row: "ORT Web vs
ORT-CPU, same artifact"). Neighbour indices are EXACT (near-tie excluded, same 1e-4
Angstrom epsilon T3a proved sufficient); P03 edge features and P04 log-probs are TOL
against the layer-(c) bars (bar/10 of the layer-(b) bars, ODQ-B8). Every measurement also
records its gap against the JAX wrapper (the layer-(b) bars, reported only -- never
gated, per the bars table's "the transitive gap vs native is reported against the
layer-(b) bars").

Also sizes the two layer-(c) controls (P04 log-prob bias, P03 edge-projection bias) by
the SAME geometric-bisection procedure T3a used, but targeting the layer-(c) bars
(1e-5, 2e-6) rather than T3a's `ctrl_effect` (spec: "sized against the c bars, not taken
from T3a's ctrl_effect"), and rebuilds both perturbed artifacts at every export bucket
with the chosen deltas -- appended to the SAME tracked manifest T2/T3a write to (D-G "One
writer"). Detection of these controls in the browser happens in T5b (validate), not here.

Writes `outputs/browser_validation/layer_c/preregistered_params.json`
(`{delta_c04, delta_c03, ctrl_effect_c, headroom_c}`, T5b reads this).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
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
import layer_c_common as lcc  # noqa: E402
from layer_b_ort_calibrate import (  # noqa: E402
  CHECKPOINT_ID,
  FIXTURE_BUCKETS,
  P03_ONLY_FIXTURES,
  SIZING_BUCKET,
  SIZING_FIXTURE,
  _build_fixture_inputs,
  _run_ort,
)

# Layer (b)'s tracked manifest -- layer (c) is a second CONSUMER of the same D-G store
# and the same base directory ($BV_ARTIFACTS), never a second producer of clean
# artifacts. Its own sized-control rows are appended to this SAME manifest (D-G "One
# writer" -- new keys only, `_c` suffixed, never colliding with layer (b)'s own rows).
TRACKED_MANIFEST_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_b" / "artifact_manifest.json"
)
PARAMS_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_c" / "preregistered_params.json"
)

# Layer (c) bars (bars table: "c | P04 log-probs / P03 edge features | TOL | <= 1e-5 /
# <= 2e-6 (bar/10, ODQ-B8)"; "c: ORT Web vs ORT-CPU, same artifact | indices/argmax |
# EXACT | 0").
P04_LOGPROB_BAR_C = 1e-5
P03_EDGE_BAR_C = 2e-6
ARGMAX_MARGIN = 2e-4
EPSILON_TIE = 1e-4  # T3a proved this sufficient (>= 5x max_abs_dist_err); not re-derived.

# Layer (b) bars, for the "transitive gap vs native" REPORTED-ONLY field (bars table
# footnote). Never gates this task's outcome.
P04_LOGPROB_BAR_B = 1e-4
P03_EDGE_BAR_B = 2e-5

DELTA_LO = 1e-7
DELTA_HI = 1e-1
DELTA_TARGET = (2.0, 10.0)
DELTA_MAX_STEPS = 20

NUM_THREADS_PARITY = 1


def _pkg_version(name: str) -> str:
  try:
    return pkg_version(name)
  except PackageNotFoundError:
    return "not-installed"


# --------------------------------------------------------------------------------------
# Cell construction (one browser cell per (fixture, path) pair at its pre-registered
# bucket)
# --------------------------------------------------------------------------------------


def _build_cell(
  *,
  path: str,
  fixture_name: str,
  bucket: int,
  onnx_path: Path,
  padded: dict[str, np.ndarray],
  site_dir: Path,
) -> dict[str, Any]:
  from aminx.export import EXPORT_BUCKETS  # noqa: PLC0415, F401 -- import-site validity check

  name = f"{path}_{fixture_name}_L{bucket}"
  data_dir = site_dir / "data"
  inputs = [
    lcc.write_raw_input(padded["coords"], data_dir / f"{name}_in0.bin", "float32"),
    lcc.write_raw_input(padded["mask"], data_dir / f"{name}_in1.bin", "float32"),
    lcc.write_raw_input(padded["residue_index"], data_dir / f"{name}_in2.bin", "int32"),
    lcc.write_raw_input(padded["chain_index"], data_dir / f"{name}_in3.bin", "int32"),
  ]
  if path == "p03":
    outputs = [
      {"label": "neighbor_indices", "dtype": "int32"},
      {"label": "edge_features", "dtype": "float32"},
    ]
  else:
    outputs = [
      {"label": "logits", "dtype": "float32"},
      {"label": "neighbor_indices", "dtype": "int32"},
    ]
  return {
    "name": name,
    "onnx": f"models/{onnx_path.name}",
    "_onnx_src": onnx_path,
    "inputs": inputs,
    "outputs": outputs,
  }


# --------------------------------------------------------------------------------------
# Control sizing (geometric bisection, 5L33@128, against the layer-(c) bars)
# --------------------------------------------------------------------------------------


def _size_controls_c(
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

  delta_c04, tried_c04 = lbc.size_control_delta(
    p04_metric,
    P04_LOGPROB_BAR_C,
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

  delta_c03, tried_c03 = lbc.size_control_delta(
    p03_metric,
    P03_EDGE_BAR_C,
    lo=DELTA_LO,
    hi=DELTA_HI,
    target=DELTA_TARGET,
    max_steps=DELTA_MAX_STEPS,
  )

  return {
    "delta_c04": delta_c04,
    "delta_c04_tried": tried_c04,
    "delta_c03": delta_c03,
    "delta_c03_tried": tried_c03,
  }


def _rebuild_sized_artifacts_c(
  model: Any, stage_set: Any, delta_c04: float, delta_c03: float, resolved_dir: Path
) -> list[dict[str, Any]]:
  """Rebuild both layer-(c) perturbed artifacts at every export bucket (D-G "Re-copy").

  New `_c`-suffixed filenames -- never T2/T3a's existing keys (D-G "One writer" refuses a
  conflicting sha256 under an existing key).
  """
  import jax2onnx
  from layer_b_build import _synthetic_inputs as synthetic_build_inputs

  from aminx.export import EXPORT_BUCKETS, make_p03_featurize, make_p04_unconditional

  rows: list[dict[str, Any]] = []
  p04_model = eqx.tree_at(lambda m: m.w_out.bias, model, model.w_out.bias.at[0].add(delta_c04))
  p03_model = eqx.tree_at(
    lambda m: m.features.w_e_proj.bias, model, model.features.w_e_proj.bias + delta_c03
  )

  for bucket in EXPORT_BUCKETS:
    coords, mask, residue_index, chain_index = synthetic_build_inputs(bucket)
    specs = [
      jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (coords, mask, residue_index, chain_index)
    ]

    p04_fn = make_p04_unconditional(p04_model, stage_set)
    p04_path = resolved_dir / f"p04_L{bucket}_perturbed_sized_c.onnx"
    jax2onnx.to_onnx(
      p04_fn,
      specs,
      model_name=f"p04_L{bucket}_perturbed_sized_c",
      output_path=str(p04_path),
      return_mode="file",
    )
    rows.append(
      {
        "path": p04_path.name,
        "sha256": lbc.artifact_sha256(p04_path),
        "role": "control_perturbed_bias_sized_c",
        "export_path": "p04",
        "bucket": bucket,
        "delta": delta_c04,
      }
    )

    p03_fn = make_p03_featurize(p03_model)
    p03_path = resolved_dir / f"p03_L{bucket}_ebias_sized_c.onnx"
    jax2onnx.to_onnx(
      p03_fn,
      specs,
      model_name=f"p03_L{bucket}_ebias_sized_c",
      output_path=str(p03_path),
      return_mode="file",
    )
    rows.append(
      {
        "path": p03_path.name,
        "sha256": lbc.artifact_sha256(p03_path),
        "role": "control_ebias_sized_c",
        "export_path": "p03",
        "bucket": bucket,
        "delta": delta_c03,
      }
    )

  lbc.manifest_append(rows, resolved_dir / "artifact_manifest.json")
  return rows


# --------------------------------------------------------------------------------------
# Headroom accumulation (route = "ort_wasm", per the task spec)
# --------------------------------------------------------------------------------------

_QUANTITY_BARS_C = {
  "neighbor_indices": 0.0,
  "edge_features": P03_EDGE_BAR_C,
  "log_probs": P04_LOGPROB_BAR_C,
  "argmax": 0.0,
}


def _accumulate(
  measurements: dict[str, dict[int, float]], path: str, quantity: str, bucket: int, value: float
) -> None:
  key = f"{path}|{quantity}|ort_wasm"
  bucket_map = measurements.setdefault(key, {})
  bucket_map[bucket] = max(bucket_map.get(bucket, float("-inf")), value)


def _build_headroom(
  measurements: dict[str, dict[int, float]], buckets: tuple[int, ...]
) -> dict[str, Any]:
  headroom: dict[str, Any] = {}
  for key, bucket_map in measurements.items():
    quantity = key.split("|")[1]
    bar = _QUANTITY_BARS_C[quantity]
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
  from aminx.export import (
    PINNED_CHECKPOINT_ID,
    make_p03_featurize,
    make_p04_unconditional,
    pad_inputs,
  )
  from aminx.inference.logits import make_stage_set
  from aminx.io.weights import load_model
  from aminx.parity.compare import argmax_agreement, max_abs

  artifacts_base = Path(args.artifacts_dir)
  subdir = lbc.resolve_artifact_subdir(artifacts_base, TRACKED_MANIFEST_PATH)  # raises -> exit 3
  resolved_dir = artifacts_base / subdir
  manifest = lbc.load_manifest_verified(TRACKED_MANIFEST_PATH, resolved_dir)  # raises -> exit 3

  provenance = lac.provenance()  # exits 3 on a non-git provenance channel

  buckets = tuple(sorted(set(FIXTURE_BUCKETS.values())))
  paths_by_path = lbc.converted_by_path(manifest, ("p03", "p04"), buckets)
  n_paths_available = sum(1 for v in paths_by_path.values() if v)

  fixtures_manifest_path = (
    _WORKTREE_ROOT / "outputs" / "browser_validation" / "fixtures" / "manifest.json"
  )
  with fixtures_manifest_path.open() as fh:
    fixture_corpus = {f["name"]: f for f in json.load(fh)["fixtures"]}

  if CHECKPOINT_ID != PINNED_CHECKPOINT_ID:
    msg = (
      f"CHECKPOINT_ID={CHECKPOINT_ID!r} != "
      f"aminx.export.PINNED_CHECKPOINT_ID={PINNED_CHECKPOINT_ID!r}"
    )
    raise ValueError(msg)
  lbc.assert_checkpoint_pinned(CHECKPOINT_ID)  # raises -> exit 3
  model = load_model(checkpoint_id=CHECKPOINT_ID)
  stage_set = make_stage_set()
  k_neighbors = model.features.k_neighbors

  errors: dict[str, str] = {}
  n_measurements_expected = 0

  node_bin = lcc.find_node(args.node_bin)
  if node_bin is None:
    errors["node"] = "no node binary found (checked --node-bin, $NODE_BIN, $PATH, nvm)"

  with tempfile.TemporaryDirectory(prefix="layer_c_calibrate_site_") as tmp:
    site_dir = Path(tmp) / "site"
    results_dir = Path(tmp) / "results"

    cells: list[dict[str, Any]] = []
    cell_meta: dict[str, dict[str, Any]] = {}  # name -> {path, fixture, bucket, n_real, ...}
    model_files: dict[str, Path] = {}

    for fixture_name, bucket in FIXTURE_BUCKETS.items():
      fixture = fixture_corpus[fixture_name]
      p03_only = fixture_name in P03_ONLY_FIXTURES
      x4, mask, residue_index, chain_index, n_real = _build_fixture_inputs(fixture)
      padded = pad_inputs(x4, mask, residue_index, chain_index, bucket)

      paths_here = ["p03"] if p03_only else ["p03", "p04"]
      for path in paths_here:
        n_measurements_expected += 1
        onnx_path = resolved_dir / f"{path}_L{bucket}.onnx"
        model_files[onnx_path.name] = onnx_path
        cell = _build_cell(
          path=path,
          fixture_name=fixture_name,
          bucket=bucket,
          onnx_path=onnx_path,
          padded=padded,
          site_dir=site_dir,
        )
        cells.append(cell)
        cell_meta[cell["name"]] = {
          "path": path,
          "fixture": fixture_name,
          "bucket": bucket,
          "n_real": n_real,
          "ca_real": x4[:n_real, 1, :],
          "padded": padded,
        }

    site_cells = [{k: v for k, v in cell.items() if not k.startswith("_")} for cell in cells]
    lcc.assemble_site(site_cells, model_files, site_dir)

    measurements: dict[str, dict[int, float]] = {}
    n_measurements_computed = 0
    harness: dict[str, Any] = {"harnessOk": False, "result": None}
    if node_bin is not None:
      try:
        harness = lcc.run_browser(
          site_dir=site_dir,
          out_dir=results_dir,
          node_bin=node_bin,
          num_threads=NUM_THREADS_PARITY,
          isolate=True,
        )
      except RuntimeError as exc:
        errors["browser"] = str(exc)

    harness_ok = bool(harness.get("harnessOk"))
    browser_result = harness.get("result") or {}
    cross_origin_isolated = bool(browser_result.get("crossOriginIsolated"))
    browser_cells = browser_result.get("cells", {})

    cell_details: dict[str, Any] = {}
    if harness_ok:
      for cell_name, meta in cell_meta.items():
        browser_cell = browser_cells.get(cell_name)
        if not browser_cell or not browser_cell.get("ok"):
          errors[f"{cell_name}.browser"] = (browser_cell or {}).get(
            "error"
          ) or "missing/failed in browser result"
          continue
        try:
          padded = meta["padded"]
          onnx_path = resolved_dir / f"{meta['path']}_L{meta['bucket']}.onnx"
          cpu_outputs, _ep_hist, _threads = _run_ort(
            onnx_path,
            padded["coords"],
            padded["mask"],
            padded["residue_index"],
            padded["chain_index"],
          )
          n_real = meta["n_real"]
          near_tie = lbc.near_tie_rows(meta["ca_real"], k_neighbors, EPSILON_TIE)
          real_mask = np.ones((n_real,), dtype=bool)

          if meta["path"] == "p03":
            idx_cpu, feat_cpu = np.asarray(cpu_outputs[0]), np.asarray(cpu_outputs[1])
            idx_browser = lcc.read_raw_output(
              results_dir / f"{cell_name}__neighbor_indices.bin", "int32", idx_cpu.shape
            )
            feat_browser = lcc.read_raw_output(
              results_dir / f"{cell_name}__edge_features.bin", "float32", feat_cpu.shape
            )
            idx_cmp = lbc.compare_neighbor_indices(
              idx_browser[:n_real], idx_cpu[:n_real], real_mask, near_tie
            )
            edge_max_abs = max_abs(feat_browser[:n_real], feat_cpu[:n_real])

            args_jax = (
              jnp.asarray(padded["coords"]),
              jnp.asarray(padded["mask"]),
              jnp.asarray(padded["residue_index"]),
              jnp.asarray(padded["chain_index"]),
            )
            _idx_wrap, feat_wrap = make_p03_featurize(model)(*args_jax)
            feat_wrap = np.asarray(feat_wrap, dtype=np.float64)
            transitive_max_abs = max_abs(feat_browser[:n_real], feat_wrap[:n_real])

            cell_details[cell_name] = {
              "neighbor_indices": idx_cmp,
              "edge_features_max_abs": edge_max_abs,
              "transitive_edge_features_max_abs_vs_wrapper": transitive_max_abs,
              "transitive_within_b_bar": transitive_max_abs <= P03_EDGE_BAR_B,
            }
            n_measurements_computed += 1
            _accumulate(
              measurements,
              "p03",
              "neighbor_indices",
              meta["bucket"],
              float(idx_cmp["n_mismatch_non_near_tie"]),
            )
            _accumulate(measurements, "p03", "edge_features", meta["bucket"], edge_max_abs)
          else:
            logits_cpu, idx_cpu = np.asarray(cpu_outputs[0]), np.asarray(cpu_outputs[1])
            logprobs_cpu = np.asarray(
              jax.nn.log_softmax(jnp.asarray(logits_cpu), axis=-1), dtype=np.float64
            )
            logits_browser = lcc.read_raw_output(
              results_dir / f"{cell_name}__logits.bin", "float32", logits_cpu.shape
            )
            idx_browser = lcc.read_raw_output(
              results_dir / f"{cell_name}__neighbor_indices.bin", "int32", idx_cpu.shape
            )
            logprobs_browser = np.asarray(
              jax.nn.log_softmax(jnp.asarray(logits_browser), axis=-1), dtype=np.float64
            )

            idx_cmp = lbc.compare_neighbor_indices(
              idx_browser[:n_real], idx_cpu[:n_real], real_mask, near_tie
            )
            logprob_max_abs = max_abs(logprobs_browser[:n_real], logprobs_cpu[:n_real])
            agree, valid = argmax_agreement(
              logprobs_browser[:n_real], logprobs_cpu[:n_real], ARGMAX_MARGIN
            )
            n_argmax_mismatch = int(np.sum(valid & ~agree))

            args_jax = (
              jnp.asarray(padded["coords"]),
              jnp.asarray(padded["mask"]),
              jnp.asarray(padded["residue_index"]),
              jnp.asarray(padded["chain_index"]),
            )
            logits_wrap, _idx_wrap = make_p04_unconditional(model, stage_set)(*args_jax)
            logprobs_wrap = np.asarray(jax.nn.log_softmax(logits_wrap, axis=-1), dtype=np.float64)
            transitive_max_abs = max_abs(logprobs_browser[:n_real], logprobs_wrap[:n_real])

            cell_details[cell_name] = {
              "neighbor_indices": idx_cmp,
              "log_probs_max_abs": logprob_max_abs,
              "n_argmax_mismatch": n_argmax_mismatch,
              "transitive_log_probs_max_abs_vs_wrapper": transitive_max_abs,
              "transitive_within_b_bar": transitive_max_abs <= P04_LOGPROB_BAR_B,
            }
            n_measurements_computed += 1
            _accumulate(
              measurements,
              "p04",
              "neighbor_indices",
              meta["bucket"],
              float(idx_cmp["n_mismatch_non_near_tie"]),
            )
            _accumulate(measurements, "p04", "log_probs", meta["bucket"], logprob_max_abs)
            _accumulate(measurements, "p04", "argmax", meta["bucket"], float(n_argmax_mismatch))
        except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
          errors[f"{cell_name}.compare"] = f"{type(exc).__name__}: {exc}"

  headroom = _build_headroom(measurements, buckets)

  sizing_result: dict[str, Any] = {"delta_c04": None, "delta_c03": None}
  if n_paths_available > 0:
    sizing_fixture = fixture_corpus[SIZING_FIXTURE]
    x4_s, mask_s, ri_s, ci_s, n_real_s = _build_fixture_inputs(sizing_fixture)
    try:
      sizing_result = _size_controls_c(model, stage_set, x4_s, mask_s, ri_s, ci_s, n_real_s)
    except Exception as exc:  # noqa: BLE001
      errors["sizing"] = f"{type(exc).__name__}: {exc}"

  delta_c04 = sizing_result.get("delta_c04")
  delta_c03 = sizing_result.get("delta_c03")
  controls_sized = int(delta_c04 is not None) + int(delta_c03 is not None)

  manifest_rows: list[dict[str, Any]] = []
  params_written = False
  if delta_c04 is not None and delta_c03 is not None and n_paths_available > 0:
    try:
      manifest_rows = _rebuild_sized_artifacts_c(
        model, stage_set, delta_c04, delta_c03, resolved_dir
      )
      params = {
        "delta_c04": delta_c04,
        "delta_c03": delta_c03,
        "ctrl_effect_c": {
          "p04": sizing_result["delta_c04_tried"][-1]["ratio_to_bar"] * P04_LOGPROB_BAR_C
          if sizing_result.get("delta_c04_tried")
          else None,
          "p03": sizing_result["delta_c03_tried"][-1]["ratio_to_bar"] * P03_EDGE_BAR_C
          if sizing_result.get("delta_c03_tried")
          else None,
        },
        "headroom_c": headroom,
      }
      PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
      with PARAMS_PATH.open("w") as fh:
        json.dump(params, fh, indent=2, sort_keys=True)
      params_written = True
    except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
      # A failure while rebuilding the sized artifacts (jax2onnx export, ONNX save, or a
      # write/IO failure -- e.g. the shared D-G $BV_ARTIFACTS store is temporarily
      # unwritable) must not crash the script with an uncaught traceback and a bare,
      # un-graded exit code -- the result-emission rule requires every
      # non-integrity-refusal path to exit 0 with a result JSON. Matches
      # _size_controls_c's own `except Exception` above. Recorded as a residual
      # (ctrl_unsized via `NOT params_written`, never pass) rather than silently
      # treated as params_written.
      errors["manifest_rebuild"] = f"{type(exc).__name__}: {exc}"

  versions = {
    "jax": jax.__version__,
    "equinox": eqx.__version__,
    "onnxruntime": _pkg_version("onnxruntime"),
    "onnx": _pkg_version("onnx"),
    "jax2onnx": _pkg_version("jax2onnx"),
    "bathos_overlay": lbc.bathos_overlay_provenance(),
    "chromium": harness.get("chromiumVersion"),
    "playwright_test": harness.get("playwrightVersion"),
    "onnxruntime_web": browser_result.get("ortWebVersion"),
    "node_toolchain": lcc.node_env_report(node_bin) if node_bin else {},
  }

  return {
    "n_paths_available": n_paths_available,
    "converted_by_path": paths_by_path,
    "n_measurements_expected": n_measurements_expected,
    "n_measurements_computed": n_measurements_computed,
    "git_hash": provenance["git_hash"],
    "git_clean": provenance["git_clean"],
    "harness_ok": harness_ok,
    "cross_origin_isolated": cross_origin_isolated,
    "delta_c04": delta_c04,
    "delta_c03": delta_c03,
    "controls_total": 2,
    "controls_sized": controls_sized,
    "params_written": params_written,
    "artifact_subdir": subdir,
    "versions": versions,
    "_cell_details": cell_details,
    "_headroom_c": headroom,
    "_manifest_rows_added": manifest_rows,
    "_sizing": {
      "delta_c04_tried": sizing_result.get("delta_c04_tried"),
      "delta_c03_tried": sizing_result.get("delta_c03_tried"),
    },
    "_evidence": {
      "sessionId": browser_result.get("sessionId"),
      "userAgent": browser_result.get("userAgent"),
      "threadAssertion": browser_result.get("threadAssertion"),
      "webgpu": browser_result.get("webgpu"),
    },
    "_errors": errors,
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON.")
  parser.add_argument(
    "--artifacts-dir", required=True, type=Path, help="Base artifact directory (D-G)."
  )
  parser.add_argument("--node-bin", default=None, help="Explicit node binary (else discovered).")
  args = parser.parse_args(argv)

  try:
    result = run(args)
    exit_code = 0
  except (FileNotFoundError, ValueError) as exc:
    logger.error("layer_c_calibrate: integrity refusal: %s", exc)
    result = {
      "n_paths_available": 0,
      "converted_by_path": {},
      "n_measurements_expected": 0,
      "n_measurements_computed": 0,
      "git_hash": None,
      "git_clean": False,
      "harness_ok": False,
      "cross_origin_isolated": False,
      "delta_c04": None,
      "delta_c03": None,
      "controls_total": 2,
      "controls_sized": 0,
      "params_written": False,
      "artifact_subdir": None,
      "versions": {},
      "_integrity_error": str(exc),
    }
    exit_code = 3

  lac.emit(result, args.out)
  logger.info(
    "layer_c_calibrate: n_paths_available=%s harness_ok=%s controls_sized=%s/2 "
    "params_written=%s exit_code=%d",
    result.get("n_paths_available"),
    result.get("harness_ok"),
    result.get("controls_sized"),
    result.get("params_written"),
    exit_code,
  )
  return exit_code


if __name__ == "__main__":
  sys.exit(main())
