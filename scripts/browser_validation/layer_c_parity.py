"""Layer (c) browser (ORT Web wasm EP, headless Chromium) validate on fixture set B (T5b).

Reuses the T5a harness (`browser/layer_c/{index.html,parity.mjs,run_parity.mjs}`,
`layer_c_common.py`, frozen -- T6 is the next task allowed to edit `parity.mjs`) and the
T5a-committed layer-(c) params (`outputs/browser_validation/layer_c/preregistered_params.json`:
`delta_c04`, `delta_c03`, `ctrl_effect_c`, `headroom_c`) to run the SAME (fixture, bucket)
pairs T3b's b-ORT arm used (`layer_b_ort_validate.SET_B_BUCKETS`/`OWN_BUCKET`) -- held-out
set B (6MRR, 1BC8, 3HTN, 4YOW), at every export bucket >= each fixture's own -- through the
already-converted, tracked T2 ONNX artifacts a SECOND time: onnxruntime-web's wasm execution
provider inside one real headless-Chromium session (numThreads = 1, cross-origin-isolated),
compared against ONNX Runtime's native CPU execution provider on the SAME artifact (the "c"
bars row: "ORT Web vs ORT-CPU, same artifact"). Neighbour indices and P04 argmax are EXACT
(near-tie rows excluded, same 1e-4 Angstrom epsilon T3a/T5a proved sufficient); P03 edge
features and P04 log-probs are TOL against the layer-(c) bars (bar/10 of the layer-(b) bars,
ODQ-B8). Every measurement also records its transitive gap against the JAX wrapper, reported
against the layer-(b) bars (never gated, per the bars table footnote). Every b-ORT-style row
is gated against the T5a-committed `headroom_c` state (`path|quantity|ort_wasm|bucket`); a
`not_advanced` cell is reported `excluded_not_advanced` and never gated, capping the outcome
at `partial_headroom`.

Six planted controls (`controls_total = 6`) are run through the SAME browser session, on the
first fixture in `("6MRR", "1BC8")` whose real row 0 is not a near-tie row, at BOTH bucket 128
and bucket 1024: the T5a-sized `p04_L{128,1024}_perturbed_sized_c.onnx` (P04 log-prob bias,
must exceed 1e-5 vs the JAX wrapper baseline), the T5a-sized `p03_L{128,1024}_ebias_sized_c.onnx`
(P03 edge-projection bias, must exceed 2e-6), and the T2-built `p03_L{128,1024}_swap.onnx` (a
real-row-0-slot index swap, detected via a POSITIONAL comparison against the wrapper, same
convention as `layer_b_ort_validate`'s `planted_swap`). A comparator self-test (identical
arrays -> 0 mismatches) is also run and reported, but is a sanity check, never counted in
`controls_total`.

The SAME site is run twice through `run_parity.mjs` (`layer_c_common.run_browser`): once
isolated (the primary evidence) and once with `--no-isolation` (the negative control, which
must report `crossOriginIsolated === false`). Both invocations' outcome feed `isolation_ok`.
Writes `outputs/browser_validation/layer_c/evidence/ort-wasm__chromium.json` (AC-C1: UA,
browser version, ORT Web version, execution provider, per-context wasm-init thread assertion
-- `parity.mjs`'s own `threadAssertion.readbackAfterInit` signal, T5a's frozen mechanism --
`crossOriginIsolated`, and the WebGPU capability probe).

A `[differential]` pre-flight arm (`AMINX_BV_LAYERC_PARITY_PERTURB`) reproduces T5a's own
sizing computation in pure JAX (no browser) on `5L33@128`, applying/withholding
`delta_c04`/`delta_c03` and reporting `ctrl_ratio_min = min(p04_ratio_to_bar, p03_ratio_to_bar)`
against the layer-(c) bars.
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
import layer_a_exact as lae  # noqa: E402
import layer_b_common as lbc  # noqa: E402
import layer_c_common as lcc  # noqa: E402
from layer_b_ort_calibrate import _run_ort  # noqa: E402
from layer_b_ort_validate import (  # noqa: E402
  CONTROL_FIXTURE_CANDIDATES,
  SET_B_BUCKETS,
  SIZING_BUCKET,
  SIZING_FIXTURE,
  _headroom_lookup,
)

CHECKPOINT_ID = "proteinmpnn_v_48_020"
FIXTURES_MANIFEST_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "fixtures" / "manifest.json"
)
TRACKED_MANIFEST_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_b" / "artifact_manifest.json"
)
PARAMS_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_c" / "preregistered_params.json"
)
EVIDENCE_PATH = (
  _WORKTREE_ROOT
  / "outputs"
  / "browser_validation"
  / "layer_c"
  / "evidence"
  / "ort-wasm__chromium.json"
)

# Layer (c) bars (bars table: "c | P04 log-probs / P03 edge features | TOL | <= 1e-5 /
# <= 2e-6 (bar/10, ODQ-B8)"; "c: ORT Web vs ORT-CPU, same artifact | indices/argmax | EXACT | 0").
P04_LOGPROB_BAR_C = 1e-5
P03_EDGE_BAR_C = 2e-6
ARGMAX_MARGIN = 2e-4
EPSILON_TIE = 1e-4

# Layer (b) bars, for the "transitive gap vs native" REPORTED-ONLY field (bars table
# footnote: "the transitive gap vs native is reported against the layer-(b) bars"). Never
# gates this task's outcome.
P04_LOGPROB_BAR_B = 1e-4
P03_EDGE_BAR_B = 2e-5

NUM_THREADS_PARITY = 1
CONTROL_BUCKETS: tuple[int, ...] = (128, 1024)
CONTROLS_TOTAL = 6

DIFFERENTIAL_KNOB = "AMINX_BV_LAYERC_PARITY_PERTURB"
DIFFERENTIAL_METRIC = "ctrl_ratio_min"
EXIT_DIFFERENTIAL_KNOB_MISMATCH = 2


def _pkg_version(name: str) -> str:
  try:
    return pkg_version(name)
  except PackageNotFoundError:
    return "not-installed"


def _jnp_padded(padded: dict[str, np.ndarray]) -> tuple[Any, Any, Any, Any]:
  return (
    jnp.asarray(padded["coords"]),
    jnp.asarray(padded["mask"]),
    jnp.asarray(padded["residue_index"]),
    jnp.asarray(padded["chain_index"]),
  )


# --------------------------------------------------------------------------------------
# Cell construction
# --------------------------------------------------------------------------------------


def _build_cell(
  *, path: str, name: str, onnx_path: Path, padded: dict[str, np.ndarray], site_dir: Path
) -> dict[str, Any]:
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
# Headroom-gated row accumulation (route is always "ort_wasm" for layer c)
# --------------------------------------------------------------------------------------


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


def _gate_state_c(
  headroom_c: dict[str, Any], path: str, quantity: str, bucket: int, buckets_ladder: tuple[int, ...]
) -> tuple[str, int | None]:
  return _headroom_lookup(headroom_c, f"{path}|{quantity}|ort_wasm", bucket, buckets_ladder)


def _record_index_row(
  tally: dict[str, Any],
  headroom_c: dict[str, Any],
  path: str,
  quantity: str,
  bucket: int,
  cmp: dict[str, Any],
  label: str,
  buckets_ladder: tuple[int, ...],
) -> None:
  state, inherited = _gate_state_c(headroom_c, path, quantity, bucket, buckets_ladder)
  row: dict[str, Any] = {
    "label": label,
    "path": path,
    "quantity": quantity,
    "bucket": bucket,
    "metric_kind": "index",
    "n_mismatch_total": cmp["n_mismatch_total"],
    "n_mismatch_non_near_tie": cmp["n_mismatch_non_near_tie"],
    "n_near_tie_mismatch": cmp["n_near_tie_mismatch"],
    "headroom_state": state,
    "inherited_from": inherited,
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
  headroom_c: dict[str, Any],
  path: str,
  quantity: str,
  bucket: int,
  value: float,
  bar: float,
  label: str,
  buckets_ladder: tuple[int, ...],
) -> None:
  state, inherited = _gate_state_c(headroom_c, path, quantity, bucket, buckets_ladder)
  row: dict[str, Any] = {
    "label": label,
    "path": path,
    "quantity": quantity,
    "bucket": bucket,
    "metric_kind": "tol",
    "value": value,
    "bar": bar,
    "headroom_state": state,
    "inherited_from": inherited,
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
  headroom_c: dict[str, Any],
  path: str,
  bucket: int,
  n_mismatch: int,
  label: str,
  buckets_ladder: tuple[int, ...],
) -> None:
  state, inherited = _gate_state_c(headroom_c, path, "argmax", bucket, buckets_ladder)
  row: dict[str, Any] = {
    "label": label,
    "path": path,
    "quantity": "argmax",
    "bucket": bucket,
    "metric_kind": "argmax",
    "n_mismatch": n_mismatch,
    "headroom_state": state,
    "inherited_from": inherited,
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
# Differential pre-flight arm (bathos [differential] pre-flight, sidecar knob)
# --------------------------------------------------------------------------------------


def _ctrl_ratio_min_c(
  model: Any, stage_set: Any, batch: Any, delta_c04: float, delta_c03: float, *, perturb: bool
) -> float:
  """`min(P04 log-prob ratio-to-bar, P03 edge ratio-to-bar)` on `batch` @ `SIZING_BUCKET`,
  against the LAYER-C bars -- mirrors `layer_b_ort_validate._ctrl_ratio_min`, but sized
  against the layer-(c) bars and the T5a-committed `delta_c04`/`delta_c03`, and it is pure
  JAX (no browser) in both the differential pre-flight arm and the main run's diagnostic
  baseline.
  """
  from aminx.export import make_p03_featurize, make_p04_unconditional, pad_inputs
  from aminx.parity.compare import max_abs

  n_real = batch.length
  padded = pad_inputs(batch.x4, batch.mask, batch.residue_index, batch.chain_index, SIZING_BUCKET)
  args_j = _jnp_padded(padded)

  baseline_logits, _ = make_p04_unconditional(model, stage_set)(*args_j)
  baseline_logprobs = np.asarray(jax.nn.log_softmax(baseline_logits, axis=-1), dtype=np.float64)[
    :n_real
  ]
  _, baseline_feat = make_p03_featurize(model)(*args_j)
  baseline_feat = np.asarray(baseline_feat, dtype=np.float64)[:n_real]

  d04 = delta_c04 if perturb else 0.0
  d03 = delta_c03 if perturb else 0.0

  p04_model = eqx.tree_at(lambda m: m.w_out.bias, model, model.w_out.bias.at[0].add(d04))
  logits2, _ = make_p04_unconditional(p04_model, stage_set)(*args_j)
  logprobs2 = np.asarray(jax.nn.log_softmax(logits2, axis=-1), dtype=np.float64)[:n_real]
  ratio_p04 = max_abs(logprobs2, baseline_logprobs) / P04_LOGPROB_BAR_C

  p03_model = eqx.tree_at(
    lambda m: m.features.w_e_proj.bias, model, model.features.w_e_proj.bias + d03
  )
  _, feat2 = make_p03_featurize(p03_model)(*args_j)
  feat2 = np.asarray(feat2, dtype=np.float64)[:n_real]
  ratio_p03 = max_abs(feat2, baseline_feat) / P03_EDGE_BAR_C

  return min(ratio_p04, ratio_p03)


def run_differential_arm(mode: dict[str, Any], args: argparse.Namespace) -> int:
  """One `bth run` differential arm: apply/withhold the T5a-sized deltas on 5L33@128."""
  from aminx.inference.logits import make_stage_set
  from aminx.io.weights import load_model

  if mode["knob"] != DIFFERENTIAL_KNOB:
    print(
      f"layer_c_parity: BTH_DIFFERENTIAL_KNOB={mode['knob']!r}, expected "
      f"{DIFFERENTIAL_KNOB!r} (this script's sidecar knob)",
      file=sys.stderr,
    )
    return EXIT_DIFFERENTIAL_KNOB_MISMATCH

  perturb = mode["value"] == "1"
  with PARAMS_PATH.open() as fh:
    params = json.load(fh)
  delta_c04 = float(params["delta_c04"])
  delta_c03 = float(params["delta_c03"])

  model = load_model(checkpoint_id=CHECKPOINT_ID)
  stage_set = make_stage_set()

  with FIXTURES_MANIFEST_PATH.open() as fh:
    fixture_corpus = {f["name"]: f for f in json.load(fh)["fixtures"]}
  batch = lae.build_exact_batch(fixture_corpus[SIZING_FIXTURE], None)

  metric_value = _ctrl_ratio_min_c(model, stage_set, batch, delta_c04, delta_c03, perturb=perturb)
  result = {
    DIFFERENTIAL_METRIC: metric_value,
    "differential_phase": mode["phase"],
    "differential_value": mode["value"],
  }
  lac.emit(result, args.out)
  logger.info(
    "layer_c_parity: differential arm value=%s ctrl_ratio_min=%.6g", mode["value"], metric_value
  )
  return 0


# --------------------------------------------------------------------------------------
# Evidence JSON (AC-C1)
# --------------------------------------------------------------------------------------


def _write_evidence(
  harness: dict[str, Any], browser_result: dict[str, Any], noiso_result: dict[str, Any]
) -> bool:
  thread_assertion = browser_result.get("threadAssertion") or {}
  readback = thread_assertion.get("readbackAfterInit")
  threads_observed_available = readback is not None
  contexts = [
    {
      "threads_requested": NUM_THREADS_PARITY,
      "threads_observed": readback,
      "threads_observed_available": threads_observed_available,
    }
  ]
  evidence = {
    "cross_origin_isolated": bool(browser_result.get("crossOriginIsolated")),
    "execution_provider": browser_result.get("executionProvider") or "wasm",
    "browser_version": harness.get("chromiumVersion"),
    "playwright_version": harness.get("playwrightVersion"),
    "ort_web_version": browser_result.get("ortWebVersion"),
    "user_agent": browser_result.get("userAgent"),
    "session_id": browser_result.get("sessionId"),
    "contexts": contexts,
    "webgpu": browser_result.get("webgpu"),
    "no_isolation_check": {"cross_origin_isolated": bool(noiso_result.get("crossOriginIsolated"))},
  }
  EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
  with EVIDENCE_PATH.open("w") as fh:
    json.dump(evidence, fh, indent=2, sort_keys=True)
  return True


# --------------------------------------------------------------------------------------
# Main run
# --------------------------------------------------------------------------------------


def _empty_result(git_hash: str | None, git_clean: bool) -> dict[str, Any]:
  return {
    "git_hash": git_hash,
    "git_clean": git_clean,
    "n_paths_available": 0,
    "converted_by_path": {},
    "controls_total": CONTROLS_TOTAL,
    "controls_detected": 0,
    "n_rows": 0,
    "n_skipped": 0,
    "n_excluded_not_advanced": 0,
    "n_low_headroom": 0,
    "n_over_bar": 0,
    "n_tie_unstable": 0,
    "harness_ok": False,
    "cross_origin_isolated": False,
    "no_isolation_cross_origin_isolated": False,
    "isolation_ok": False,
    "worst_ratio_to_bar": None,
    "worst_ratio_to_bar_available": False,
    "worst_path": None,
    "artifact_subdir": None,
    "versions": {},
    "comparator_self_test_pass": False,
    "control_fixture": None,
    "ctrl_ratio_min": 0.0,
    "evidence_written": False,
  }


def run(args: argparse.Namespace) -> dict[str, Any]:
  from aminx.export import (
    EXPORT_BUCKETS,
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

  with PARAMS_PATH.open() as fh:
    params_c = json.load(fh)
  headroom_c: dict[str, Any] = params_c.get("headroom_c", {})
  delta_c04 = float(params_c["delta_c04"])
  delta_c03 = float(params_c["delta_c03"])

  provenance = lac.provenance(PARAMS_PATH)  # exits 3 on a non-git provenance channel

  paths_by_path = lbc.converted_by_path(manifest, ("p03", "p04"), EXPORT_BUCKETS)
  n_paths_available = sum(1 for v in paths_by_path.values() if v)

  result = _empty_result(provenance["git_hash"], provenance["git_clean"])
  result["converted_by_path"] = paths_by_path
  result["n_paths_available"] = n_paths_available
  result["artifact_subdir"] = subdir

  if n_paths_available == 0:
    result["_note"] = "T2 BLOCKED table: neither path fully converted; nothing measured."
    return result

  if CHECKPOINT_ID != PINNED_CHECKPOINT_ID:
    msg = (
      f"CHECKPOINT_ID={CHECKPOINT_ID!r} != aminx.export.PINNED_CHECKPOINT_ID="
      f"{PINNED_CHECKPOINT_ID!r}"
    )
    raise ValueError(msg)
  lbc.assert_checkpoint_pinned(CHECKPOINT_ID)  # raises -> exit 3
  model = load_model(checkpoint_id=CHECKPOINT_ID)
  stage_set = make_stage_set()
  k_neighbors = model.features.k_neighbors

  with FIXTURES_MANIFEST_PATH.open() as fh:
    fixture_corpus = {f["name"]: f for f in json.load(fh)["fixtures"]}

  # Diagnostic baseline (perturb=False) of the SAME metric the [differential] pre-flight arm
  # reports -- always ~0 on the unperturbed model; declared here only so the sidecar's
  # `[differential].metric` key is a declared `result_schema` field.
  sizing_batch = lae.build_exact_batch(fixture_corpus[SIZING_FIXTURE], None)
  result["ctrl_ratio_min"] = _ctrl_ratio_min_c(
    model, stage_set, sizing_batch, delta_c04, delta_c03, perturb=False
  )

  node_bin = lcc.find_node(args.node_bin)
  errors: dict[str, str] = {}
  if node_bin is None:
    errors["node"] = "no node binary found (checked --node-bin, $NODE_BIN, $PATH, nvm)"

  tally = _new_tally()
  n_skipped = 0

  with tempfile.TemporaryDirectory(prefix="layer_c_parity_site_") as tmp:
    site_dir = Path(tmp) / "site"
    results_dir = Path(tmp) / "results"
    results_dir_noiso = Path(tmp) / "results_noiso"

    cells: list[dict[str, Any]] = []
    cell_meta: dict[str, dict[str, Any]] = {}
    model_files: dict[str, Path] = {}
    fixture_geo: dict[str, dict[str, Any]] = {}

    # ---- Main comparison cells: the T3b validate pairs (same (fixture, bucket) set) ----
    for fixture_name, buckets in SET_B_BUCKETS.items():
      fixture_row = fixture_corpus[fixture_name]
      batch = lae.build_exact_batch(fixture_row, None)
      x4, mask, ri, ci, n_real = (
        batch.x4,
        batch.mask,
        batch.residue_index,
        batch.chain_index,
        (batch.length),
      )
      ca_real = x4[:n_real, 1, :]
      near_tie = lbc.near_tie_rows(ca_real, k_neighbors, EPSILON_TIE)
      fixture_geo[fixture_name] = {"near_tie": near_tie, "n_real": n_real}

      for bucket in buckets:
        padded = pad_inputs(x4, mask, ri, ci, bucket)
        for path in ("p03", "p04"):
          onnx_path = resolved_dir / f"{path}_L{bucket}.onnx"
          model_files[onnx_path.name] = onnx_path
          name = f"{path}_{fixture_name}_L{bucket}"
          cells.append(
            _build_cell(path=path, name=name, onnx_path=onnx_path, padded=padded, site_dir=site_dir)
          )
          cell_meta[name] = {
            "kind": "main",
            "path": path,
            "fixture": fixture_name,
            "bucket": bucket,
            "n_real": n_real,
            "padded": padded,
          }

    # ---- Control fixture selection (same rule as layer_b_ort_validate) ----
    control_fixture: str | None = None
    for candidate in CONTROL_FIXTURE_CANDIDATES:
      geo = fixture_geo.get(candidate)
      if geo is None or geo["n_real"] == 0:
        continue
      if not geo["near_tie"][0]:
        control_fixture = candidate
        break

    control_cell_names: dict[tuple[str, int], str] = {}
    if control_fixture is not None:
      for bucket in CONTROL_BUCKETS:
        base_p04 = cell_meta.get(f"p04_{control_fixture}_L{bucket}")
        base_p03 = cell_meta.get(f"p03_{control_fixture}_L{bucket}")
        if base_p04 is not None:
          onnx_path = resolved_dir / f"p04_L{bucket}_perturbed_sized_c.onnx"
          model_files[onnx_path.name] = onnx_path
          name = f"ctrl_p04_perturbed_c_L{bucket}"
          cells.append(
            _build_cell(
              path="p04",
              name=name,
              onnx_path=onnx_path,
              padded=base_p04["padded"],
              site_dir=site_dir,
            )
          )
          cell_meta[name] = {
            "kind": "ctrl_p04_perturbed",
            "bucket": bucket,
            "n_real": base_p04["n_real"],
          }
          control_cell_names[("p04_perturbed", bucket)] = name
        if base_p03 is not None:
          onnx_path = resolved_dir / f"p03_L{bucket}_ebias_sized_c.onnx"
          model_files[onnx_path.name] = onnx_path
          name = f"ctrl_p03_ebias_c_L{bucket}"
          cells.append(
            _build_cell(
              path="p03",
              name=name,
              onnx_path=onnx_path,
              padded=base_p03["padded"],
              site_dir=site_dir,
            )
          )
          cell_meta[name] = {
            "kind": "ctrl_p03_ebias",
            "bucket": bucket,
            "n_real": base_p03["n_real"],
          }
          control_cell_names[("p03_ebias", bucket)] = name

          onnx_path = resolved_dir / f"p03_L{bucket}_swap.onnx"
          model_files[onnx_path.name] = onnx_path
          name = f"ctrl_p03_swap_L{bucket}"
          cells.append(
            _build_cell(
              path="p03",
              name=name,
              onnx_path=onnx_path,
              padded=base_p03["padded"],
              site_dir=site_dir,
            )
          )
          cell_meta[name] = {
            "kind": "ctrl_p03_swap",
            "bucket": bucket,
            "n_real": base_p03["n_real"],
          }
          control_cell_names[("p03_swap", bucket)] = name

    site_cells = [{k: v for k, v in cell.items() if not k.startswith("_")} for cell in cells]
    lcc.assemble_site(site_cells, model_files, site_dir)

    harness: dict[str, Any] = {"harnessOk": False, "result": None}
    harness_noiso: dict[str, Any] = {"harnessOk": False, "result": None}
    if node_bin is not None:
      try:
        harness = lcc.run_browser(
          site_dir=site_dir,
          out_dir=results_dir,
          node_bin=node_bin,
          num_threads=NUM_THREADS_PARITY,
          isolate=True,
          timeout_s=args.timeout_s,
        )
      except RuntimeError as exc:
        errors["browser"] = str(exc)
      try:
        harness_noiso = lcc.run_browser(
          site_dir=site_dir,
          out_dir=results_dir_noiso,
          node_bin=node_bin,
          num_threads=NUM_THREADS_PARITY,
          isolate=False,
          timeout_s=args.timeout_s,
        )
      except RuntimeError as exc:
        errors["browser_noiso"] = str(exc)

    harness_ok = bool(harness.get("harnessOk"))
    browser_result = harness.get("result") or {}
    cross_origin_isolated = bool(browser_result.get("crossOriginIsolated"))
    browser_cells = browser_result.get("cells", {})

    noiso_harness_ok = bool(harness_noiso.get("harnessOk"))
    noiso_result = harness_noiso.get("result") or {}
    no_isolation_cross_origin_isolated = bool(noiso_result.get("crossOriginIsolated"))
    isolation_ok = (
      cross_origin_isolated and noiso_harness_ok and not no_isolation_cross_origin_isolated
    )

    evidence_written = False
    if harness_ok:
      try:
        evidence_written = _write_evidence(harness, browser_result, noiso_result)
      except OSError as exc:  # noqa: BLE001 -- recorded, not swallowed
        errors["evidence_write"] = f"{type(exc).__name__}: {exc}"

    fixture_wrap_cache: dict[tuple[str, int, str], dict[str, Any]] = {}

    if harness_ok:
      for name, meta in cell_meta.items():
        if meta["kind"] != "main":
          continue
        browser_cell = browser_cells.get(name)
        if not browser_cell or not browser_cell.get("ok"):
          errors[f"{name}.browser"] = (browser_cell or {}).get(
            "error"
          ) or "missing/failed in browser result"
          n_skipped += 1
          continue
        try:
          path = meta["path"]
          fixture_name = meta["fixture"]
          bucket = meta["bucket"]
          padded = meta["padded"]
          n_real = meta["n_real"]
          near_tie = fixture_geo[fixture_name]["near_tie"]
          real_mask = np.ones((n_real,), dtype=bool)
          onnx_path = resolved_dir / f"{path}_L{bucket}.onnx"
          cpu_outputs, _ep_hist, _threads = _run_ort(
            onnx_path,
            padded["coords"],
            padded["mask"],
            padded["residue_index"],
            padded["chain_index"],
          )
          args_jax = _jnp_padded(padded)

          if path == "p03":
            idx_cpu = np.asarray(cpu_outputs[0])
            feat_cpu = np.asarray(cpu_outputs[1], dtype=np.float64)
            idx_browser = lcc.read_raw_output(
              results_dir / f"{name}__neighbor_indices.bin", "int32", idx_cpu.shape
            )
            feat_browser = lcc.read_raw_output(
              results_dir / f"{name}__edge_features.bin", "float32", feat_cpu.shape
            ).astype(np.float64)

            idx_cmp = lbc.compare_neighbor_indices(
              idx_browser[:n_real], idx_cpu[:n_real], real_mask, near_tie
            )
            _record_index_row(
              tally,
              headroom_c,
              "p03",
              "neighbor_indices",
              bucket,
              idx_cmp,
              f"{fixture_name}.L{bucket}.c.p03.idx",
              EXPORT_BUCKETS,
            )
            edge_max_abs = max_abs(feat_browser[:n_real], feat_cpu[:n_real])
            _record_tol_row(
              tally,
              headroom_c,
              "p03",
              "edge_features",
              bucket,
              edge_max_abs,
              P03_EDGE_BAR_C,
              f"{fixture_name}.L{bucket}.c.p03.edge",
              EXPORT_BUCKETS,
            )

            idx_wrap, feat_wrap = make_p03_featurize(model)(*args_jax)
            idx_wrap = np.asarray(idx_wrap)
            feat_wrap = np.asarray(feat_wrap, dtype=np.float64)
            transitive_max_abs = max_abs(feat_browser[:n_real], feat_wrap[:n_real])
            fixture_wrap_cache[(fixture_name, bucket, "p03")] = {"idx": idx_wrap, "feat": feat_wrap}
            tally["rows"][-1]["transitive_edge_features_max_abs_vs_wrapper"] = transitive_max_abs
            tally["rows"][-1]["transitive_within_b_bar"] = transitive_max_abs <= P03_EDGE_BAR_B
          else:
            logits_cpu, idx_cpu = np.asarray(cpu_outputs[0]), np.asarray(cpu_outputs[1])
            logprobs_cpu = np.asarray(
              jax.nn.log_softmax(jnp.asarray(logits_cpu), axis=-1), dtype=np.float64
            )
            logits_browser = lcc.read_raw_output(
              results_dir / f"{name}__logits.bin", "float32", logits_cpu.shape
            )
            idx_browser = lcc.read_raw_output(
              results_dir / f"{name}__neighbor_indices.bin", "int32", idx_cpu.shape
            )
            logprobs_browser = np.asarray(
              jax.nn.log_softmax(jnp.asarray(logits_browser), axis=-1), dtype=np.float64
            )

            idx_cmp = lbc.compare_neighbor_indices(
              idx_browser[:n_real], idx_cpu[:n_real], real_mask, near_tie
            )
            _record_index_row(
              tally,
              headroom_c,
              "p04",
              "neighbor_indices",
              bucket,
              idx_cmp,
              f"{fixture_name}.L{bucket}.c.p04.idx",
              EXPORT_BUCKETS,
            )
            logprob_max_abs = max_abs(logprobs_browser[:n_real], logprobs_cpu[:n_real])
            _record_tol_row(
              tally,
              headroom_c,
              "p04",
              "log_probs",
              bucket,
              logprob_max_abs,
              P04_LOGPROB_BAR_C,
              f"{fixture_name}.L{bucket}.c.p04.logprobs",
              EXPORT_BUCKETS,
            )
            agree, valid = argmax_agreement(
              logprobs_browser[:n_real], logprobs_cpu[:n_real], ARGMAX_MARGIN
            )
            n_argmax_mismatch = int(np.sum(valid & ~agree))
            _record_argmax_row(
              tally,
              headroom_c,
              "p04",
              bucket,
              n_argmax_mismatch,
              f"{fixture_name}.L{bucket}.c.p04.argmax",
              EXPORT_BUCKETS,
            )

            logits_wrap, idx_wrap = make_p04_unconditional(model, stage_set)(*args_jax)
            idx_wrap = np.asarray(idx_wrap)
            logprobs_wrap = np.asarray(jax.nn.log_softmax(logits_wrap, axis=-1), dtype=np.float64)
            transitive_max_abs = max_abs(logprobs_browser[:n_real], logprobs_wrap[:n_real])
            fixture_wrap_cache[(fixture_name, bucket, "p04")] = {
              "idx": idx_wrap,
              "logprobs": logprobs_wrap,
            }
            tally["rows"][-1]["transitive_log_probs_max_abs_vs_wrapper"] = transitive_max_abs
            tally["rows"][-1]["transitive_within_b_bar"] = transitive_max_abs <= P04_LOGPROB_BAR_B
        except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
          n_skipped += 1
          errors[f"{name}.compare"] = f"{type(exc).__name__}: {exc}"

    # ---- Controls: browser-executed perturbed/swap artifact vs the JAX wrapper baseline ----
    controls_detail: dict[str, Any] = {}
    comparator_self_test_pass = False
    if harness_ok and control_fixture is not None:
      for bucket in CONTROL_BUCKETS:
        base_p04 = fixture_wrap_cache.get((control_fixture, bucket, "p04"))
        base_p03 = fixture_wrap_cache.get((control_fixture, bucket, "p03"))
        n_real_c = fixture_geo[control_fixture]["n_real"]

        name_p04 = control_cell_names.get(("p04_perturbed", bucket))
        cell_p04 = browser_cells.get(name_p04) if name_p04 else None
        key_p04 = f"p04_perturbed_c_L{bucket}"
        if base_p04 is not None and cell_p04 and cell_p04.get("ok"):
          try:
            logits_pert = lcc.read_raw_output(
              results_dir / f"{name_p04}__logits.bin", "float32", (base_p04["logprobs"].shape)
            )
            logprobs_pert = np.asarray(
              jax.nn.log_softmax(jnp.asarray(logits_pert), axis=-1), dtype=np.float64
            )
            ratio = (
              max_abs(logprobs_pert[:n_real_c], base_p04["logprobs"][:n_real_c]) / P04_LOGPROB_BAR_C
            )
            controls_detail[key_p04] = {"ratio_to_bar": ratio, "detected": ratio > 1.0}
          except Exception as exc:  # noqa: BLE001
            controls_detail[key_p04] = {"error": f"{type(exc).__name__}: {exc}", "detected": False}
        else:
          controls_detail[key_p04] = {
            "error": "browser cell missing/failed or no baseline",
            "detected": False,
          }

        name_p03e = control_cell_names.get(("p03_ebias", bucket))
        cell_p03e = browser_cells.get(name_p03e) if name_p03e else None
        key_p03e = f"p03_ebias_c_L{bucket}"
        if base_p03 is not None and cell_p03e and cell_p03e.get("ok"):
          try:
            feat_pert = lcc.read_raw_output(
              results_dir / f"{name_p03e}__edge_features.bin", "float32", base_p03["feat"].shape
            ).astype(np.float64)
            ratio = max_abs(feat_pert[:n_real_c], base_p03["feat"][:n_real_c]) / P03_EDGE_BAR_C
            controls_detail[key_p03e] = {"ratio_to_bar": ratio, "detected": ratio > 1.0}
          except Exception as exc:  # noqa: BLE001
            controls_detail[key_p03e] = {"error": f"{type(exc).__name__}: {exc}", "detected": False}
        else:
          controls_detail[key_p03e] = {
            "error": "browser cell missing/failed or no baseline",
            "detected": False,
          }

        name_swap = control_cell_names.get(("p03_swap", bucket))
        cell_swap = browser_cells.get(name_swap) if name_swap else None
        key_swap = f"p03_swap_L{bucket}"
        if base_p03 is not None and cell_swap and cell_swap.get("ok"):
          try:
            idx_swap = lcc.read_raw_output(
              results_dir / f"{name_swap}__neighbor_indices.bin", "int32", base_p03["idx"].shape
            )
            idx_ref = base_p03["idx"]
            row0_matches = np.array_equal(idx_swap[0], idx_ref[0])
            other_rows_match = np.array_equal(idx_swap[1:n_real_c], idx_ref[1:n_real_c])
            row0_near_tie = bool(fixture_geo[control_fixture]["near_tie"][0])
            detected = (not row0_matches) and other_rows_match and not row0_near_tie
            controls_detail[key_swap] = {
              "row0_positional_match": bool(row0_matches),
              "other_rows_positional_match": bool(other_rows_match),
              "row0_near_tie": row0_near_tie,
              "detected": bool(detected),
            }
          except Exception as exc:  # noqa: BLE001
            controls_detail[key_swap] = {"error": f"{type(exc).__name__}: {exc}", "detected": False}
        else:
          controls_detail[key_swap] = {
            "error": "browser cell missing/failed or no baseline",
            "detected": False,
          }

      base_p03_128 = fixture_wrap_cache.get((control_fixture, 128, "p03"))
      if base_p03_128 is not None:
        real_mask_c = np.ones((fixture_geo[control_fixture]["n_real"],), dtype=bool)
        self_cmp = lbc.compare_neighbor_indices(
          base_p03_128["idx"][: fixture_geo[control_fixture]["n_real"]],
          base_p03_128["idx"][: fixture_geo[control_fixture]["n_real"]],
          real_mask_c,
          fixture_geo[control_fixture]["near_tie"],
        )
        comparator_self_test_pass = self_cmp["n_mismatch_total"] == 0

    controls_detected = sum(1 for c in controls_detail.values() if c.get("detected"))

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

  result.update(
    {
      "controls_total": CONTROLS_TOTAL,
      "controls_detected": controls_detected,
      "n_rows": tally["n_rows"],
      "n_skipped": n_skipped,
      "n_excluded_not_advanced": tally["n_excluded_not_advanced"],
      "n_low_headroom": tally["n_low_headroom"],
      "n_over_bar": tally["n_over_bar"],
      "n_tie_unstable": tally["n_tie_unstable"],
      "harness_ok": harness_ok,
      "cross_origin_isolated": cross_origin_isolated,
      "no_isolation_cross_origin_isolated": no_isolation_cross_origin_isolated,
      "isolation_ok": isolation_ok,
      "worst_ratio_to_bar": tally["worst_ratio"] if tally["worst_path"] else None,
      "worst_ratio_to_bar_available": tally["worst_path"] is not None,
      "worst_path": tally["worst_path"],
      "versions": versions,
      "comparator_self_test_pass": comparator_self_test_pass,
      "control_fixture": control_fixture,
      "evidence_written": evidence_written,
      "_rows": tally["rows"],
      "_controls_detail": controls_detail,
      "_errors": errors,
      "delta_c04": delta_c04,
      "delta_c03": delta_c03,
    }
  )
  return result


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON.")
  parser.add_argument(
    "--artifacts-dir", required=True, type=Path, help="Base artifact directory (D-G)."
  )
  parser.add_argument("--node-bin", default=None, help="Explicit node binary (else discovered).")
  parser.add_argument(
    "--timeout-s", type=float, default=300.0, help="Per-browser-invocation timeout (seconds)."
  )
  args = parser.parse_args(argv)

  mode = lac.differential_mode()
  if mode["active"]:
    return run_differential_arm(mode, args)

  try:
    result = run(args)
    exit_code = 0
  except (FileNotFoundError, ValueError) as exc:
    logger.error("layer_c_parity: integrity refusal: %s", exc)
    result = _empty_result(None, False)
    result["_integrity_error"] = str(exc)
    exit_code = 3

  lac.emit(result, args.out)
  logger.info(
    "layer_c_parity: n_paths_available=%s harness_ok=%s isolation_ok=%s controls=%s/%s "
    "n_over_bar=%s n_tie_unstable=%s n_excluded_not_advanced=%s n_low_headroom=%s "
    "n_skipped=%s exit_code=%d",
    result.get("n_paths_available"),
    result.get("harness_ok"),
    result.get("isolation_ok"),
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
