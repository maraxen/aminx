"""Layer (b) IREE route: safety, native vmfb parity, WASM32 codegen (T4).

For both RNG-free P03/P04 export wrappers (T1, `aminx.export.make_p03_featurize` /
`make_p04_unconditional`) and every export bucket in `--buckets` (default
`aminx.export.EXPORT_BUCKETS`), this script:

1. Traces the wrapper independently with `jax.make_jaxpr` (`trace_ok`) and runs
   `xtrax.export.check_export_safety` for both the NATIVE and WASM32 targets -- both must
   return `[]` (D-D: "native executed parity plus WASM32 compiled as CODEGEN_ONLY").
2. Exports the wrapper to StableHLO (`jax.export.export(jax.jit(fn))(*specs).mlir_module()`)
   and compiles it for NATIVE (`compile_for_target(..., NATIVE)`), then runs it against
   held-out fixture set B (the SAME buckets T3a/T3b used) via `run_native_vmfb` and
   compares the result against the wrapper's own JAX output at that bucket -- the "b-IREE"
   comparator in the pre-registered bars table (artifact vs native wrapper; neighbour
   indices EXACT with the near-tie exclusion, P03 edge features / P04 log-probs TOL, P04
   argmax EXACT). Every gated row is checked against the T3a-committed calibration
   headroom state, keyed `route="b_ort"` (b-IREE has no calibrate pass of its own, so it
   INHERITS the b-ORT route's headroom state for the same `(path, quantity, bucket)` key
   and records `inherited_route=True` -- a stated limitation, not a bar change; see the
   pre-registered bars text, "Headroom rule").
3. Compiles the SAME StableHLO for WASM32 (`compile_for_target(..., WASM32)`) and records
   its sha256 with `verification="codegen_only"` -- there is no emsdk IREE runtime to
   execute it (D-D, parent ODQ-1 fallback), so nothing beyond compilation is claimed.
4. Re-detects the two T1 census controls (`random_decoding_order` -> `random-permutation`,
   `lambda x: jax.lax.top_k(x, 8)` -> `unlegalizable-op`) as a standing self-test that the
   `check_export_safety` instrument still fires on known-unsafe ops (reused verbatim from
   `export_safety_census.py`).
5. Compiles and executes the three T3a-sized planted control variants (P04 `delta_b04` on
   `w_out.bias[0]`, P03 `delta_b03` on `features.w_e_proj.bias`, and the T2-built
   neighbour-slot swap) at `CONTROL_BUCKET` and confirms each is detected, mirroring
   `layer_b_ort_validate.py`'s control-detection logic but through the native vmfb instead
   of an ONNX Runtime session.

Every vmfb row this script builds (clean NATIVE + WASM32 per (path, bucket), plus the
three NATIVE-only planted-control vmfbs) is appended to the tracked artifact manifest via
`layer_b_common.manifest_append` with `route="iree"`, into the SAME `artifact_subdir` T2's
build already verified (D-G: "Later producers ... write into the same artifact_subdir").

`rings = "not_available"` throughout (V8: the `xtrax.export.rings` bucket ladder used by
the R0-R3 tracks is unavailable on the installed `xtrax==0.4.0a10`).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
import layer_a_exact as lae  # noqa: E402
import layer_b_common as lbc  # noqa: E402
from export_safety_census import (  # noqa: E402
  control_random_decoding_order,
  control_top_k,
)

CHECKPOINT_ID = "proteinmpnn_v_48_020"  # V10, matches aminx.export.PINNED_CHECKPOINT_ID
FIXTURES_MANIFEST_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "fixtures" / "manifest.json"
)
TRACKED_MANIFEST_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_b" / "artifact_manifest.json"
)
PARAMS_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_b" / "preregistered_params.json"
)

DEFAULT_BUCKETS: tuple[int, ...] = (128, 256, 512, 1024)  # aminx.export.EXPORT_BUCKETS

#: Set-B (fixture, own-export-bucket) and every ladder bucket >= own (same coverage as
#: T3b's layer_b_ort_validate.py, "set-B inputs" per this task's step 2).
OWN_BUCKET: dict[str, int] = {"6MRR": 128, "1BC8": 128, "3HTN": 512, "4YOW": 1024}
SET_B_BUCKETS: dict[str, tuple[int, ...]] = {
  "6MRR": (128, 256, 512, 1024),
  "1BC8": (128, 256, 512, 1024),
  "3HTN": (512, 1024),
  "4YOW": (1024,),
}
CONTROL_FIXTURE_CANDIDATES: tuple[str, ...] = ("6MRR", "1BC8")
CONTROL_BUCKET = 128
SIZING_FIXTURE = "5L33"  # same fixture T3a's own control-sizing search used
SIZING_BUCKET = 128

#: b bars (pre-registered bars table; b-ORT and b-IREE share the same bars).
P03_EDGE_BAR = 2e-5
P04_LOGPROB_BAR = 1e-4
ARGMAX_MARGIN = 2e-4
EPSILON_TIE = 1e-4

CONTROLS_TOTAL = 3  # the three T3a-sized/T2-built planted variants (b-IREE)
CENSUS_CONTROLS_TOTAL = 2  # random_decoding_order + top_k (T1 census re-detection)

DIFFERENTIAL_KNOB = "AMINX_BV_LAYERB_IREE_VALIDATE_PERTURB"
DIFFERENTIAL_METRIC = "ctrl_ratio_min"
EXIT_DIFFERENTIAL_KNOB_MISMATCH = 2

RINGS = "not_available"  # V8: no rings module / default ladder on installed xtrax==0.4.0a10


# --------------------------------------------------------------------------------------
# Step 1: IREE import check (spec step 1) -- BLOCKED, not a crash, if it fails.
# --------------------------------------------------------------------------------------


def _check_iree_import() -> tuple[bool, str]:
  try:
    import iree.compiler  # noqa: F401
    import iree.runtime  # noqa: F401
    from xtrax.export import (  # noqa: F401
      NATIVE,
      WASM32,
      check_export_safety,
      compile_for_target,
      run_native_vmfb,
    )
  except Exception as exc:  # noqa: BLE001 -- the exception text itself is the finding
    return False, f"{type(exc).__name__}: {exc}"
  return True, ""


# --------------------------------------------------------------------------------------
# Synthetic inputs for the safety/trace/compile step (conversion proof, not parity --
# mirrors layer_b_build.py's own `_synthetic_inputs`).
# --------------------------------------------------------------------------------------


def _synthetic_inputs(bucket: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  ca_spacing = 3.8
  base = np.arange(bucket, dtype=np.float32) * ca_spacing
  jitter = np.sin(np.arange(bucket, dtype=np.float32) * 0.7).astype(np.float32)

  coords = np.zeros((bucket, 4, 3), dtype=np.float32)
  coords[:, 0, 0] = base - 1.0
  coords[:, 1, 0] = base
  coords[:, 2, 0] = base + 1.0
  coords[:, 3, 0] = base + 1.5
  coords[:, :, 1] = jitter[:, None] * 0.3

  mask = np.ones((bucket,), dtype=np.float32)
  residue_index = np.arange(bucket, dtype=np.int32)
  chain_index = np.zeros((bucket,), dtype=np.int32)
  return coords, mask, residue_index, chain_index


def _specs(inputs: tuple[np.ndarray, ...]) -> list[Any]:
  return [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in inputs]


def _flatten_vmfb_output(raw: Any) -> list[np.ndarray]:
  """`run_native_vmfb` returns a bare value for a 1-output entry point, a tuple otherwise."""
  items = raw if isinstance(raw, (list, tuple)) else [raw]
  return [np.asarray(item) for item in items]


# --------------------------------------------------------------------------------------
# Headroom lookup, inheriting the b-ORT route's state (spec step 2, "IREE has no
# calibrate pass ... IREE keys are inherited from the route='ort' key" -- the committed
# params file's own route label for that comparator is "b_ort").
# --------------------------------------------------------------------------------------


def _headroom_lookup(
  headroom: dict[str, Any], key_prefix: str, bucket: int, buckets_ladder: tuple[int, ...]
) -> tuple[str, int | None]:
  key = f"{key_prefix}|{bucket}"
  entry = headroom.get(key)
  if entry is not None:
    return entry["state"], None
  for candidate in sorted(b for b in buckets_ladder if b > bucket):
    entry = headroom.get(f"{key_prefix}|{candidate}")
    if entry is not None:
      return entry["state"], candidate
  return "advanced", None


def _gate_state(
  headroom: dict[str, Any], path: str, quantity: str, bucket: int, buckets_ladder: tuple[int, ...]
) -> tuple[str, int | None]:
  # Inherit from the b-ORT route's own headroom key (spec step 2: "inherited_route").
  return _headroom_lookup(headroom, f"{path}|{quantity}|b_ort", bucket, buckets_ladder)


def _new_tally() -> dict[str, Any]:
  return {
    "n_rows": 0,
    "n_excluded_not_advanced": 0,
    "n_low_headroom": 0,
    "n_over_bar": 0,
    "n_tie_unstable": 0,
    "worst_path": None,
    "worst_ratio": 0.0,
    "rows": [],
  }


def _record_index_row(
  tally: dict[str, Any],
  headroom: dict[str, Any],
  path: str,
  quantity: str,
  bucket: int,
  cmp: dict[str, Any],
  label: str,
  buckets_ladder: tuple[int, ...],
) -> None:
  state, inherited = _gate_state(headroom, path, quantity, bucket, buckets_ladder)
  row: dict[str, Any] = {
    "label": label,
    "path": path,
    "quantity": quantity,
    "route": "iree",
    "inherited_route": True,
    "bucket": bucket,
    "metric_kind": "index",
    "n_mismatch_total": cmp["n_mismatch_total"],
    "n_mismatch_non_near_tie": cmp["n_mismatch_non_near_tie"],
    "n_near_tie_mismatch": cmp["n_near_tie_mismatch"],
    "headroom_state": state,
    "inherited_from_bucket": inherited,
  }
  tally["n_rows"] += 1
  if state == "not_advanced":
    row["status"] = "excluded_not_advanced"
    tally["n_excluded_not_advanced"] += 1
  else:
    if state == "low_headroom":
      tally["n_low_headroom"] += 1
    if cmp["n_mismatch_total"] == 0:
      row["status"] = "ok"
    elif cmp["all_mismatches_near_tie"]:
      row["status"] = "tie_unstable"
      tally["n_tie_unstable"] += 1
    else:
      row["status"] = "fail"
      tally["n_over_bar"] += 1
  tally["rows"].append(row)


def _record_tol_row(
  tally: dict[str, Any],
  headroom: dict[str, Any],
  path: str,
  quantity: str,
  bucket: int,
  value: float,
  bar: float,
  label: str,
  buckets_ladder: tuple[int, ...],
) -> None:
  state, inherited = _gate_state(headroom, path, quantity, bucket, buckets_ladder)
  row: dict[str, Any] = {
    "label": label,
    "path": path,
    "quantity": quantity,
    "route": "iree",
    "inherited_route": True,
    "bucket": bucket,
    "metric_kind": "tol",
    "value": value,
    "bar": bar,
    "headroom_state": state,
    "inherited_from_bucket": inherited,
  }
  tally["n_rows"] += 1
  if state == "not_advanced":
    row["status"] = "excluded_not_advanced"
    tally["n_excluded_not_advanced"] += 1
  else:
    if state == "low_headroom":
      tally["n_low_headroom"] += 1
    if value <= bar:
      row["status"] = "ok"
    else:
      row["status"] = "fail"
      tally["n_over_bar"] += 1
    ratio = (value / bar) if bar > 0 else float("inf")
    if ratio > tally["worst_ratio"]:
      tally["worst_ratio"] = ratio
      tally["worst_path"] = label
  tally["rows"].append(row)


def _record_argmax_row(
  tally: dict[str, Any],
  headroom: dict[str, Any],
  path: str,
  bucket: int,
  n_mismatch: int,
  label: str,
  buckets_ladder: tuple[int, ...],
) -> None:
  state, inherited = _gate_state(headroom, path, "argmax", bucket, buckets_ladder)
  row: dict[str, Any] = {
    "label": label,
    "path": path,
    "quantity": "argmax",
    "route": "iree",
    "inherited_route": True,
    "bucket": bucket,
    "metric_kind": "argmax",
    "n_mismatch": n_mismatch,
    "headroom_state": state,
    "inherited_from_bucket": inherited,
  }
  tally["n_rows"] += 1
  if state == "not_advanced":
    row["status"] = "excluded_not_advanced"
    tally["n_excluded_not_advanced"] += 1
  else:
    if state == "low_headroom":
      tally["n_low_headroom"] += 1
    if n_mismatch == 0:
      row["status"] = "ok"
    else:
      row["status"] = "fail"
      tally["n_over_bar"] += 1
  tally["rows"].append(row)


# --------------------------------------------------------------------------------------
# Differential pre-flight arm -- pure-JAX (no IREE compile), reproducing T3a/T3b's own
# ctrl_ratio_min formula so [differential].min_effect (from params) is the SAME number
# this run's "on" arm should reproduce, without tripling the (expensive) IREE compile
# workload. See layer_b_ort_validate.py's `_ctrl_ratio_min`/`run_differential_arm` for
# the twin implementation this one deliberately mirrors bit-for-bit.
# --------------------------------------------------------------------------------------


def _ctrl_ratio_min(
  model: Any,
  stage_set: Any,
  batch: Any,
  delta_b04: float,
  delta_b03: float,
  *,
  perturb: bool,
) -> float:
  from aminx.export import make_p03_featurize, make_p04_unconditional, pad_inputs
  from aminx.parity.compare import max_abs

  n_real = batch.length
  padded = pad_inputs(batch.x4, batch.mask, batch.residue_index, batch.chain_index, SIZING_BUCKET)
  args_j = (
    jnp.asarray(padded["coords"]),
    jnp.asarray(padded["mask"]),
    jnp.asarray(padded["residue_index"]),
    jnp.asarray(padded["chain_index"]),
  )

  baseline_logits, _ = make_p04_unconditional(model, stage_set)(*args_j)
  baseline_logprobs = np.asarray(jax.nn.log_softmax(baseline_logits, axis=-1), dtype=np.float64)[
    :n_real
  ]
  _, baseline_feat = make_p03_featurize(model)(*args_j)
  baseline_feat = np.asarray(baseline_feat, dtype=np.float64)[:n_real]

  d04 = delta_b04 if perturb else 0.0
  d03 = delta_b03 if perturb else 0.0

  p04_model = eqx.tree_at(lambda m: m.w_out.bias, model, model.w_out.bias.at[0].add(d04))
  logits2, _ = make_p04_unconditional(p04_model, stage_set)(*args_j)
  logprobs2 = np.asarray(jax.nn.log_softmax(logits2, axis=-1), dtype=np.float64)[:n_real]
  ratio_p04 = max_abs(logprobs2, baseline_logprobs) / P04_LOGPROB_BAR

  p03_model = eqx.tree_at(
    lambda m: m.features.w_e_proj.bias, model, model.features.w_e_proj.bias + d03
  )
  _, feat2 = make_p03_featurize(p03_model)(*args_j)
  feat2 = np.asarray(feat2, dtype=np.float64)[:n_real]
  ratio_p03 = max_abs(feat2, baseline_feat) / P03_EDGE_BAR

  return min(ratio_p04, ratio_p03)


def run_differential_arm(mode: dict[str, Any], args: argparse.Namespace) -> int:
  if mode["knob"] != DIFFERENTIAL_KNOB:
    print(
      f"layer_b_iree: BTH_DIFFERENTIAL_KNOB={mode['knob']!r}, expected {DIFFERENTIAL_KNOB!r} "
      "(this script's sidecar knob)",
      file=sys.stderr,
    )
    return EXIT_DIFFERENTIAL_KNOB_MISMATCH

  perturb = mode["value"] == "1"
  with PARAMS_PATH.open() as fh:
    params = json.load(fh)
  delta_b04 = float(params["delta_b04"])
  delta_b03 = float(params["delta_b03"])

  from aminx.inference.logits import make_stage_set
  from aminx.io.weights import load_model

  model = load_model(checkpoint_id=CHECKPOINT_ID)
  stage_set = make_stage_set()

  with FIXTURES_MANIFEST_PATH.open() as fh:
    fixture_corpus = {f["name"]: f for f in json.load(fh)["fixtures"]}
  batch = lae.build_exact_batch(fixture_corpus[SIZING_FIXTURE], None)

  metric_value = _ctrl_ratio_min(model, stage_set, batch, delta_b04, delta_b03, perturb=perturb)
  result = {
    DIFFERENTIAL_METRIC: metric_value,
    "differential_phase": mode["phase"],
    "differential_value": mode["value"],
  }
  lac.emit(result, args.out)
  logger.info(
    "layer_b_iree: differential arm value=%s ctrl_ratio_min=%.6g", mode["value"], metric_value
  )
  return 0


# --------------------------------------------------------------------------------------
# Per-(path, bucket) safety + compile
# --------------------------------------------------------------------------------------


def _wrapper_for(path: str, model: Any, stage_set: Any) -> Any:
  from aminx.export import make_p03_featurize, make_p04_unconditional

  return make_p03_featurize(model) if path == "p03" else make_p04_unconditional(model, stage_set)


def build_path_bucket(
  path: str,
  bucket: int,
  model: Any,
  stage_set: Any,
  resolved_dir: Path,
  iree_mod: Any,
) -> dict[str, Any]:
  """Trace + safety-check + compile (NATIVE and WASM32) one (path, bucket) cell.

  Returns `{trace_ok, native_blockers, wasm32_blockers, manifest_rows, compile_seconds,
  native_vmfb_path, error}`. `native_vmfb_path` is `None` if compilation failed.
  """
  NATIVE = iree_mod["NATIVE"]
  WASM32 = iree_mod["WASM32"]
  check_export_safety = iree_mod["check_export_safety"]
  compile_for_target = iree_mod["compile_for_target"]

  fn = _wrapper_for(path, model, stage_set)
  inputs = _synthetic_inputs(bucket)
  specs = _specs(inputs)

  cell: dict[str, Any] = {
    "trace_ok": False,
    "native_blockers": None,
    "wasm32_blockers": None,
    "manifest_rows": [],
    "compile_seconds": {},
    "native_vmfb_path": None,
    "error": None,
  }

  try:
    jax.make_jaxpr(fn)(*specs)
    cell["trace_ok"] = True
  except Exception as exc:  # noqa: BLE001 -- the exception text is the finding
    cell["error"] = f"trace: {type(exc).__name__}: {exc}"
    return cell

  try:
    native_blockers = check_export_safety([], {}, specs, fn, NATIVE)
    wasm32_blockers = check_export_safety([], {}, specs, fn, WASM32)
  except Exception as exc:  # noqa: BLE001
    cell["error"] = f"check_export_safety: {type(exc).__name__}: {exc}"
    return cell
  cell["native_blockers"] = [
    {"rule": b.rule, "axis": b.axis, "detail": b.detail} for b in native_blockers
  ]
  cell["wasm32_blockers"] = [
    {"rule": b.rule, "axis": b.axis, "detail": b.detail} for b in wasm32_blockers
  ]
  if native_blockers or wasm32_blockers:
    return cell

  try:
    t0 = time.perf_counter()
    exported = jax.export.export(jax.jit(fn))(*specs)
    mlir_text = exported.mlir_module()
    native_path = resolved_dir / f"{path}_L{bucket}.native.vmfb"
    native_result = compile_for_target(mlir_text, NATIVE, out_path=native_path)
    cell["compile_seconds"]["native"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    wasm32_path = resolved_dir / f"{path}_L{bucket}.wasm32.vmfb"
    wasm32_result = compile_for_target(mlir_text, WASM32, out_path=wasm32_path)
    cell["compile_seconds"]["wasm32"] = time.perf_counter() - t1
  except Exception as exc:  # noqa: BLE001
    cell["error"] = f"compile: {type(exc).__name__}: {exc}"
    return cell

  cell["native_vmfb_path"] = native_result.path
  cell["manifest_rows"] = [
    {
      "path": native_result.path.name,
      "sha256": lbc.artifact_sha256(native_result.path),
      "role": "clean",
      "route": "iree",
      "export_path": path,
      "bucket": bucket,
      "target": "native",
      "verification": "executed",
    },
    {
      "path": wasm32_result.path.name,
      "sha256": lbc.artifact_sha256(wasm32_result.path),
      "role": "clean",
      "route": "iree",
      "export_path": path,
      "bucket": bucket,
      "target": "wasm32",
      "verification": "codegen_only",
    },
  ]
  return cell


# --------------------------------------------------------------------------------------
# Planted control compile + detect (P04 delta_b04, P03 delta_b03, P03 index swap) at
# CONTROL_BUCKET, mirroring layer_b_ort_validate.py's control-detection logic but through
# the native vmfb.
# --------------------------------------------------------------------------------------


def _make_p03_swap_variant(p03_fn: Any) -> Any:
  """Swap the last two neighbour slots of real row 0 only.

  Two earlier attempts failed to compile under IREE's llvm-cpu backend
  (measured 260928, `layer_b_iree.py` smoke runs):

  1. ``.at[0].set(...)`` (a dynamic-update-slice writing the function's own
     output) -- rejected as "write affecting operations on global resources
     are restricted to workgroup distributed contexts".
  2. ``jnp.where`` with an explicitly `jnp.broadcast_to`-shaped mask over an
     all-rows static gather (`idx[:, perm]`) -- this compiled in an isolated
     toy repro (`/tmp/swap_test.py`, bare `ShapeDtypeStruct` inputs) but
     FAILED again once `idx`/`feats` are the real `p03_fn` wrapper's output
     (`/tmp/swap_test3.py`, same real checkpoint + L=128/k=48 as this
     script): `vector.transfer_read` mask-type/operand-type mismatch
     (`vector<16xi1>` vs `vector<1x16xi1>`) -- an IREE vectorizer bug
     specific to masking an intermediate value produced by the wrapper's own
     top_k/sort chain, not reproducible on a bare function argument of the
     same shape.

  This version uses only static slicing + `jnp.concatenate` (row 0 permuted,
  rows 1..L-1 untouched, concatenated back) -- no scatter, no masked select,
  so nothing triggers either IREE bug. Verified numerically correct and
  IREE-compilable against the real wrapper at L=128/k=48
  (`/tmp/swap_test4.py`, 260928).
  """

  def swapped(coords: Any, mask: Any, residue_index: Any, chain_index: Any) -> Any:
    idx, feats = p03_fn(coords, mask, residue_index, chain_index)
    k = idx.shape[-1]
    perm = jnp.arange(k).at[k - 2].set(k - 1).at[k - 1].set(k - 2)
    idx = jnp.concatenate([idx[:1, perm], idx[1:]], axis=0)
    feats = jnp.concatenate([feats[:1, perm], feats[1:]], axis=0)
    return idx, feats

  return swapped


def run_controls(
  model: Any,
  stage_set: Any,
  delta_b04: float,
  delta_b03: float,
  fixture_corpus: dict[str, Any],
  resolved_dir: Path,
  iree_mod: Any,
) -> dict[str, Any]:
  """Build + compile + run the three planted control vmfbs; return `{detail,
  control_fixture, manifest_rows}`."""
  from aminx.export import make_p03_featurize, make_p04_unconditional, pad_inputs

  NATIVE = iree_mod["NATIVE"]
  compile_for_target = iree_mod["compile_for_target"]
  run_native_vmfb = iree_mod["run_native_vmfb"]

  control_fixture: str | None = None
  fixture_cells: dict[str, dict[str, Any]] = {}

  for candidate in CONTROL_FIXTURE_CANDIDATES:
    fixture_row = fixture_corpus.get(candidate)
    if fixture_row is None:
      continue
    batch = lae.build_exact_batch(fixture_row, None)
    n_real = batch.length
    if n_real == 0:
      continue
    ca_real = batch.x4[:n_real, 1, :]
    k_neighbors = model.features.k_neighbors
    near_tie = lbc.near_tie_rows(ca_real, k_neighbors, EPSILON_TIE)
    fixture_cells[candidate] = {"batch": batch, "n_real": n_real, "near_tie": near_tie}
    if control_fixture is None and not near_tie[0]:
      control_fixture = candidate

  detail: dict[str, Any] = {}
  manifest_rows: list[dict[str, Any]] = []

  if control_fixture is None:
    return {"detail": detail, "control_fixture": None, "manifest_rows": manifest_rows}

  cell = fixture_cells[control_fixture]
  batch = cell["batch"]
  n_real = cell["n_real"]
  padded = pad_inputs(batch.x4, batch.mask, batch.residue_index, batch.chain_index, CONTROL_BUCKET)
  padded_j = (
    jnp.asarray(padded["coords"]),
    jnp.asarray(padded["mask"]),
    jnp.asarray(padded["residue_index"]),
    jnp.asarray(padded["chain_index"]),
  )
  padded_np = (padded["coords"], padded["mask"], padded["residue_index"], padded["chain_index"])

  # Wrapper baselines are computed ONCE, outside every try block, so a failure in one
  # control's compile/execute never leaves a later control referencing an unset name.
  p03_wrap = make_p03_featurize(model)
  idx_wrap, feat_wrap = p03_wrap(*padded_j)
  idx_wrap = np.asarray(idx_wrap)
  feat_wrap = np.asarray(feat_wrap, dtype=np.float64)[:n_real]

  # ---- P04 delta_b04 control ----
  try:
    p04_wrap = make_p04_unconditional(model, stage_set)
    logits_wrap, _idx_wrap = p04_wrap(*padded_j)
    logprobs_wrap = np.asarray(jax.nn.log_softmax(logits_wrap, axis=-1), dtype=np.float64)[:n_real]

    p04_model = eqx.tree_at(lambda m: m.w_out.bias, model, model.w_out.bias.at[0].add(delta_b04))
    p04_pert_fn = make_p04_unconditional(p04_model, stage_set)
    specs = _specs(padded_np)
    exported = jax.export.export(jax.jit(p04_pert_fn))(*specs)
    vmfb_path = resolved_dir / f"p04_L{CONTROL_BUCKET}_perturbed_sized.native.vmfb"
    result = compile_for_target(exported.mlir_module(), NATIVE, out_path=vmfb_path)
    raw = run_native_vmfb(vmfb_path, *padded_np)
    logits_pert, _idx_pert = _flatten_vmfb_output(raw)
    logprobs_pert = np.asarray(
      jax.nn.log_softmax(jnp.asarray(logits_pert), axis=-1), dtype=np.float64
    )[:n_real]
    from aminx.parity.compare import max_abs

    ratio = max_abs(logprobs_pert, logprobs_wrap) / P04_LOGPROB_BAR
    detail["p04_perturbed_sized"] = {"ratio_to_bar": ratio, "detected": ratio > 1.0}
    manifest_rows.append(
      {
        "path": vmfb_path.name,
        "sha256": lbc.artifact_sha256(result.path),
        "role": "control_perturbed_sized",
        "route": "iree",
        "export_path": "p04",
        "bucket": CONTROL_BUCKET,
        "target": "native",
        "verification": "executed",
        "delta": delta_b04,
      }
    )
  except Exception as exc:  # noqa: BLE001
    detail["p04_perturbed_sized"] = {"error": f"{type(exc).__name__}: {exc}", "detected": False}

  # ---- P03 delta_b03 (ebias) control ----
  try:
    p03_model = eqx.tree_at(
      lambda m: m.features.w_e_proj.bias, model, model.features.w_e_proj.bias + delta_b03
    )
    p03_pert_fn = make_p03_featurize(p03_model)
    specs = _specs(padded_np)
    exported = jax.export.export(jax.jit(p03_pert_fn))(*specs)
    vmfb_path = resolved_dir / f"p03_L{CONTROL_BUCKET}_ebias_sized.native.vmfb"
    result = compile_for_target(exported.mlir_module(), NATIVE, out_path=vmfb_path)
    raw = run_native_vmfb(vmfb_path, *padded_np)
    _idx_pert, feat_pert = _flatten_vmfb_output(raw)
    feat_pert = np.asarray(feat_pert, dtype=np.float64)[:n_real]

    from aminx.parity.compare import max_abs

    ratio = max_abs(feat_pert, feat_wrap) / P03_EDGE_BAR
    detail["p03_ebias_sized"] = {"ratio_to_bar": ratio, "detected": ratio > 1.0}
    manifest_rows.append(
      {
        "path": vmfb_path.name,
        "sha256": lbc.artifact_sha256(result.path),
        "role": "control_ebias_sized",
        "route": "iree",
        "export_path": "p03",
        "bucket": CONTROL_BUCKET,
        "target": "native",
        "verification": "executed",
        "delta": delta_b03,
      }
    )
  except Exception as exc:  # noqa: BLE001
    detail["p03_ebias_sized"] = {"error": f"{type(exc).__name__}: {exc}", "detected": False}

  # ---- P03 swap control (positional, real row 0 only) ----
  try:
    swap_fn = _make_p03_swap_variant(p03_wrap)
    specs = _specs(padded_np)
    exported = jax.export.export(jax.jit(swap_fn))(*specs)
    vmfb_path = resolved_dir / f"p03_L{CONTROL_BUCKET}_swap.native.vmfb"
    result = compile_for_target(exported.mlir_module(), NATIVE, out_path=vmfb_path)
    raw = run_native_vmfb(vmfb_path, *padded_np)
    idx_swap, _feat_swap = _flatten_vmfb_output(raw)
    idx_swap = np.asarray(idx_swap)

    row0_matches = np.array_equal(idx_swap[0], idx_wrap[0])
    other_rows_match = np.array_equal(idx_swap[1:n_real], idx_wrap[1:n_real])
    row0_near_tie = bool(cell["near_tie"][0])
    detected = (not row0_matches) and other_rows_match and not row0_near_tie
    detail["planted_swap"] = {
      "row0_positional_match": bool(row0_matches),
      "other_rows_positional_match": bool(other_rows_match),
      "row0_near_tie": row0_near_tie,
      "detected": bool(detected),
    }
    manifest_rows.append(
      {
        "path": vmfb_path.name,
        "sha256": lbc.artifact_sha256(result.path),
        "role": "control_swap",
        "route": "iree",
        "export_path": "p03",
        "bucket": CONTROL_BUCKET,
        "target": "native",
        "verification": "executed",
      }
    )
  except Exception as exc:  # noqa: BLE001
    detail["planted_swap"] = {"error": f"{type(exc).__name__}: {exc}", "detected": False}

  return {"detail": detail, "control_fixture": control_fixture, "manifest_rows": manifest_rows}


# --------------------------------------------------------------------------------------
# Main run
# --------------------------------------------------------------------------------------


def _empty_result(
  git_hash: str | None, git_clean: bool, iree_import_ok: bool, reason: str
) -> dict[str, Any]:
  return {
    "iree_import_ok": iree_import_ok,
    "_iree_import_error": reason,
    "git_hash": git_hash,
    "git_clean": git_clean,
    "trace_ok_all": False,
    "safety_clean_all": False,
    "census_controls_total": CENSUS_CONTROLS_TOTAL,
    "census_controls_detected": 0,
    "controls_total": CONTROLS_TOTAL,
    "controls_detected": 0,
    "n_rows": 0,
    "n_skipped": 0,
    "n_excluded_not_advanced": 0,
    "n_low_headroom": 0,
    "n_over_bar": 0,
    "n_tie_unstable": 0,
    "worst_ratio_to_bar": None,
    "worst_ratio_to_bar_available": False,
    "worst_path": None,
    "max_p_before": 0.0,
    "n_dropout": 0,
    "artifact_subdir": None,
    "rings": RINGS,
    "versions": {},
    "control_fixture": None,
    "ctrl_ratio_min": 0.0,
  }


def _pkg_version(name: str) -> str:
  try:
    return pkg_version(name)
  except PackageNotFoundError:
    return "not-installed"


def run(args: argparse.Namespace) -> dict[str, Any]:
  import iree.compiler  # noqa: F401
  import iree.runtime  # noqa: F401
  from xtrax.export import NATIVE, WASM32, check_export_safety, compile_for_target, run_native_vmfb

  iree_mod = {
    "NATIVE": NATIVE,
    "WASM32": WASM32,
    "check_export_safety": check_export_safety,
    "compile_for_target": compile_for_target,
    "run_native_vmfb": run_native_vmfb,
  }

  from aminx.inference.logits import make_stage_set
  from aminx.io.weights import load_model
  from aminx.parity.compare import argmax_agreement, max_abs

  buckets = tuple(int(b) for b in args.buckets.split(",")) if args.buckets else DEFAULT_BUCKETS
  artifacts_base = Path(args.artifacts_dir)

  # D-G: resolve + verify the manifest T2 already built, before writing anything new.
  subdir = lbc.resolve_artifact_subdir(artifacts_base, TRACKED_MANIFEST_PATH)  # raises -> exit 3
  resolved_dir = artifacts_base / subdir
  lbc.load_manifest_verified(TRACKED_MANIFEST_PATH, resolved_dir)  # raises -> exit 3 (integrity)

  with PARAMS_PATH.open() as fh:
    params = json.load(fh)
  headroom: dict[str, Any] = params.get("headroom", {})
  delta_b04 = float(params["delta_b04"])
  delta_b03 = float(params["delta_b03"])

  provenance = lac.provenance(PARAMS_PATH)  # exits 3 on a non-git provenance channel

  lbc.assert_checkpoint_pinned(CHECKPOINT_ID)  # raises -> exit 3
  model = load_model(checkpoint_id=CHECKPOINT_ID)
  stage_set = make_stage_set()

  with FIXTURES_MANIFEST_PATH.open() as fh:
    fixture_corpus = {f["name"]: f for f in json.load(fh)["fixtures"]}

  result = _empty_result(provenance["git_hash"], provenance["git_clean"], True, "")
  result["artifact_subdir"] = subdir

  # Diagnostic baseline (perturb=False), always 0.0 -- present so [differential].metric
  # is a declared result_schema field (bathos validate-sidecar requirement), mirroring
  # layer_a_exact_validate's sentinel_ratio_to_bar / layer_b_ort_validate's ctrl_ratio_min.
  sizing_batch = lae.build_exact_batch(fixture_corpus[SIZING_FIXTURE], None)
  result["ctrl_ratio_min"] = _ctrl_ratio_min(
    model, stage_set, sizing_batch, delta_b04, delta_b03, perturb=False
  )

  # ---- Census controls (T1 self-test the instrument still fires) ----
  perm_detected, perm_rules = control_random_decoding_order(CONTROL_BUCKET)
  topk_detected, topk_rules = control_top_k(CONTROL_BUCKET)
  census_controls_detected = int(perm_detected) + int(topk_detected)
  result["census_controls_detected"] = census_controls_detected
  result["_census_controls_detail"] = {
    "random_decoding_order": {"detected": perm_detected, "rules_detected": perm_rules},
    "top_k": {"detected": topk_detected, "rules_detected": topk_rules},
  }

  # ---- Per-(path, bucket): trace_ok + safety + compile ----
  all_manifest_rows: list[dict[str, Any]] = []
  compile_seconds: dict[str, float] = {}
  cells: dict[tuple[str, int], dict[str, Any]] = {}
  errors: dict[str, str] = {}
  n_skipped = 0
  trace_ok_all = True
  safety_clean_all = True

  for path in ("p03", "p04"):
    for bucket in buckets:
      cell = build_path_bucket(path, bucket, model, stage_set, resolved_dir, iree_mod)
      cells[(path, bucket)] = cell
      # trace_ok/check_export_safety failures are graded as `unsafe` (spec step 3's own
      # declaration order puts `incomplete` BEFORE `unsafe`), so they do NOT increment
      # n_skipped -- only the dedicated trace_ok_all/safety_clean_all flags record them.
      # n_skipped is reserved for genuine execution-level anomalies (a compile that fails
      # AFTER safety passed, a comparison/fixture exception) that make `incomplete` fire.
      if not cell["trace_ok"]:
        trace_ok_all = False
        errors[f"{path}.L{bucket}.trace"] = cell["error"] or "trace_ok=False"
        continue
      if cell["native_blockers"] or cell["wasm32_blockers"]:
        safety_clean_all = False
        errors[f"{path}.L{bucket}.safety"] = json.dumps(
          {"native": cell["native_blockers"], "wasm32": cell["wasm32_blockers"]}
        )
        continue
      if cell["error"] is not None:
        errors[f"{path}.L{bucket}.compile"] = cell["error"]
        n_skipped += 1
        continue
      all_manifest_rows.extend(cell["manifest_rows"])
      for target, seconds in cell["compile_seconds"].items():
        compile_seconds[f"{path}.L{bucket}.{target}"] = seconds

  result["trace_ok_all"] = trace_ok_all
  result["safety_clean_all"] = safety_clean_all

  # ---- Numeric parity: set-B fixtures vs the compiled NATIVE vmfb (b-IREE) ----
  tally = _new_tally()
  for fixture_name, fixture_buckets in SET_B_BUCKETS.items():
    fixture_row = fixture_corpus.get(fixture_name)
    if fixture_row is None:
      errors[f"{fixture_name}.fatal"] = "fixture not found in manifest"
      n_skipped += 1
      continue
    try:
      batch = lae.build_exact_batch(fixture_row, None)
      n_real = batch.length
      ca_real = batch.x4[:n_real, 1, :]
      near_tie = lbc.near_tie_rows(ca_real, model.features.k_neighbors, EPSILON_TIE)
      real_mask = np.ones((n_real,), dtype=bool)

      for bucket in fixture_buckets:
        from aminx.export import pad_inputs

        padded = pad_inputs(batch.x4, batch.mask, batch.residue_index, batch.chain_index, bucket)
        padded_np = (
          padded["coords"],
          padded["mask"],
          padded["residue_index"],
          padded["chain_index"],
        )
        padded_j = tuple(jnp.asarray(a) for a in padded_np)

        p03_cell = cells.get(("p03", bucket))
        if p03_cell is not None and p03_cell["native_vmfb_path"] is not None:
          try:
            fn = _wrapper_for("p03", model, stage_set)
            idx_wrap, feat_wrap = fn(*padded_j)
            idx_wrap = np.asarray(idx_wrap)
            feat_wrap = np.asarray(feat_wrap, dtype=np.float64)

            raw = run_native_vmfb(p03_cell["native_vmfb_path"], *padded_np)
            idx_iree, feat_iree = _flatten_vmfb_output(raw)
            feat_iree = feat_iree.astype(np.float64)

            cmp_idx = lbc.compare_neighbor_indices(
              idx_iree[:n_real], idx_wrap[:n_real], real_mask, near_tie
            )
            _record_index_row(
              tally,
              headroom,
              "p03",
              "neighbor_indices",
              bucket,
              cmp_idx,
              f"{fixture_name}.L{bucket}.bIREE.p03.idx",
              DEFAULT_BUCKETS,
            )
            edge_max_abs = max_abs(feat_iree[:n_real], feat_wrap[:n_real])
            _record_tol_row(
              tally,
              headroom,
              "p03",
              "edge_features",
              bucket,
              edge_max_abs,
              P03_EDGE_BAR,
              f"{fixture_name}.L{bucket}.bIREE.p03.edge",
              DEFAULT_BUCKETS,
            )
          except Exception as exc:  # noqa: BLE001
            n_skipped += 1
            errors[f"{fixture_name}.L{bucket}.p03.b_iree"] = f"{type(exc).__name__}: {exc}"

        p04_cell = cells.get(("p04", bucket))
        if p04_cell is not None and p04_cell["native_vmfb_path"] is not None:
          try:
            fn = _wrapper_for("p04", model, stage_set)
            logits_wrap, idx_wrap = fn(*padded_j)
            idx_wrap = np.asarray(idx_wrap)
            logprobs_wrap = np.asarray(jax.nn.log_softmax(logits_wrap, axis=-1), dtype=np.float64)

            raw = run_native_vmfb(p04_cell["native_vmfb_path"], *padded_np)
            logits_iree, idx_iree = _flatten_vmfb_output(raw)
            logprobs_iree = np.asarray(
              jax.nn.log_softmax(jnp.asarray(logits_iree), axis=-1), dtype=np.float64
            )

            cmp_idx = lbc.compare_neighbor_indices(
              idx_iree[:n_real], idx_wrap[:n_real], real_mask, near_tie
            )
            _record_index_row(
              tally,
              headroom,
              "p04",
              "neighbor_indices",
              bucket,
              cmp_idx,
              f"{fixture_name}.L{bucket}.bIREE.p04.idx",
              DEFAULT_BUCKETS,
            )
            logprob_max_abs = max_abs(logprobs_iree[:n_real], logprobs_wrap[:n_real])
            _record_tol_row(
              tally,
              headroom,
              "p04",
              "log_probs",
              bucket,
              logprob_max_abs,
              P04_LOGPROB_BAR,
              f"{fixture_name}.L{bucket}.bIREE.p04.logprobs",
              DEFAULT_BUCKETS,
            )
            agree, valid = argmax_agreement(
              logprobs_wrap[:n_real], logprobs_iree[:n_real], ARGMAX_MARGIN
            )
            n_argmax_mismatch = int(np.sum(valid & ~agree))
            _record_argmax_row(
              tally,
              headroom,
              "p04",
              bucket,
              n_argmax_mismatch,
              f"{fixture_name}.L{bucket}.bIREE.p04.argmax",
              DEFAULT_BUCKETS,
            )
          except Exception as exc:  # noqa: BLE001
            n_skipped += 1
            errors[f"{fixture_name}.L{bucket}.p04.b_iree"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 -- a fixture-level failure is a skip, not a crash
      n_skipped += 1
      errors[f"{fixture_name}.fatal"] = f"{type(exc).__name__}: {exc}"

  # ---- Planted controls (P04 delta_b04, P03 delta_b03, P03 swap) ----
  controls = run_controls(
    model, stage_set, delta_b04, delta_b03, fixture_corpus, resolved_dir, iree_mod
  )
  controls_detected = sum(1 for c in controls["detail"].values() if c.get("detected"))
  all_manifest_rows.extend(controls["manifest_rows"])

  if all_manifest_rows:
    lbc.manifest_append(all_manifest_rows, resolved_dir / "artifact_manifest.json")

  versions = {
    "jax": jax.__version__,
    "equinox": eqx.__version__,
    "iree_base_compiler": _pkg_version("iree-base-compiler"),
    "iree_base_runtime": _pkg_version("iree-base-runtime"),
    "xtrax": _pkg_version("xtrax"),
    "bathos_overlay": lbc.bathos_overlay_provenance(),
  }

  dropout_stats = {"max_p_before": 0.0, "n_dropout": 0}
  try:
    from aminx.export import zero_dropout

    _, dropout_stats = zero_dropout(model)
  except Exception as exc:  # noqa: BLE001 -- diagnostic only, never fatal
    errors["zero_dropout"] = f"{type(exc).__name__}: {exc}"

  result.update(
    {
      "census_controls_detected": census_controls_detected,
      "controls_total": CONTROLS_TOTAL,
      "controls_detected": controls_detected,
      "n_rows": tally["n_rows"],
      "n_skipped": n_skipped,
      "n_excluded_not_advanced": tally["n_excluded_not_advanced"],
      "n_low_headroom": tally["n_low_headroom"],
      "n_over_bar": tally["n_over_bar"],
      "n_tie_unstable": tally["n_tie_unstable"],
      "worst_ratio_to_bar": tally["worst_ratio"] if tally["worst_path"] else None,
      "worst_ratio_to_bar_available": tally["worst_path"] is not None,
      "worst_path": tally["worst_path"],
      "max_p_before": dropout_stats["max_p_before"],
      "n_dropout": dropout_stats["n_dropout"],
      "rings": RINGS,
      "versions": versions,
      "control_fixture": controls["control_fixture"],
      "_rows": tally["rows"],
      "_controls_detail": controls["detail"],
      "_compile_seconds": compile_seconds,
      "_errors": errors,
      "delta_b04": delta_b04,
      "delta_b03": delta_b03,
    }
  )
  return result


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON.")
  parser.add_argument(
    "--artifacts-dir", required=True, type=Path, help="Base artifact directory (D-G)."
  )
  parser.add_argument(
    "--buckets",
    default=",".join(str(b) for b in DEFAULT_BUCKETS),
    help="Comma-separated bucket ladder (default: aminx.export.EXPORT_BUCKETS).",
  )
  args = parser.parse_args(argv)

  mode = lac.differential_mode()
  if mode["active"]:
    return run_differential_arm(mode, args)

  iree_ok, iree_reason = _check_iree_import()
  if not iree_ok:
    logger.error("layer_b_iree: step-1 IREE import check failed: %s", iree_reason)
    result = _empty_result(None, False, False, iree_reason)
    lac.emit(result, args.out)
    return 0  # result-emission rule: BLOCKED is still exit 0, the outcome label carries it

  try:
    result = run(args)
    exit_code = 0
  except (FileNotFoundError, ValueError) as exc:
    logger.error("layer_b_iree: integrity refusal: %s", exc)
    result = _empty_result(None, False, True, "")
    result["_integrity_error"] = str(exc)
    exit_code = 3

  lac.emit(result, args.out)
  logger.info(
    "layer_b_iree: iree_import_ok=%s trace_ok_all=%s safety_clean_all=%s "
    "census_controls=%s/%s controls=%s/%s n_over_bar=%s n_tie_unstable=%s "
    "n_excluded_not_advanced=%s n_low_headroom=%s n_skipped=%s exit_code=%d",
    result.get("iree_import_ok"),
    result.get("trace_ok_all"),
    result.get("safety_clean_all"),
    result.get("census_controls_detected"),
    result.get("census_controls_total"),
    result.get("controls_detected"),
    result.get("controls_total"),
    result.get("n_over_bar"),
    result.get("n_tie_unstable"),
    result.get("n_excluded_not_advanced"),
    result.get("n_low_headroom"),
    result.get("n_skipped"),
    exit_code,
  )
  return exit_code


if __name__ == "__main__":
  sys.exit(main())
