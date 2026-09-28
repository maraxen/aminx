"""Layer (b) ORT-CPU validate on fixture set B (T3b).

Runs the RNG-free P03/P04 export wrappers (T1) and the already-converted, sha256-
verified T2 ONNX artifacts (via ONNX Runtime's CPU execution provider) against the
HELD-OUT fixture set B (6MRR, 1BC8, 3HTN, 4YOW), at every export bucket >= each
fixture's own bucket (spec step 1). Three comparator families are measured per real
fixture:

- **W** (wrapper vs native, D-H; UNPADDED, at the fixture's real length): proves the
  RNG-free wrapper's own decomposition doesn't drift from the native kernel, decoupled
  from any export-padding effect. P04's native arm is
  `layer_a_exact._score_aminx_unconditional(zero_dropout(model)[0], batch)` (log-probs);
  P03's native arm is `native_model.features(PRNGKey(0), ..., backbone_noise=0.0)`.
  Bars: neighbour indices EXACT 0 (near-tie excluded); edge features / P04 log-probs TOL
  <= 1e-6.
- **P27** (P04 only; wrapper PADDED to the fixture's own export bucket vs native
  UNPADDED): the export-padding effect. Bars: indices EXACT 0 (near-tie excluded);
  log-probs TOL <= 1e-4 nats.
- **b-ORT** (ORT-executed artifact vs the JAX wrapper, both padded to the SAME bucket,
  for every bucket >= the fixture's own): P03 indices EXACT / edge features TOL <= 2e-5;
  P04 indices EXACT / log-probs TOL <= 1e-4 / argmax EXACT (top-2 margin > 2e-4).

Every EXACT (neighbour-index) comparison excludes near-tie rows (pre-registered bars,
"Near-tie rule", epsilon = 1e-4 Angstrom, `layer_b_common.near_tie_rows`); a mismatch
confined to near-tie rows is `tie_unstable`, not `fail`. Every b-ORT/P27 cell is gated
against the T3a-committed headroom state
(`outputs/browser_validation/layer_b/preregistered_params.json`'s `headroom` dict, keyed
`path|quantity|route|bucket`); a bucket T3a never measured (256) inherits the nearest
LARGER measured bucket's state. A `not_advanced` cell is reported `excluded_not_advanced`
and never gated; any `low_headroom`/`excluded_not_advanced` cell caps the run's outcome at
`partial_headroom`. The two refusal probes (2GFB oversize, 5L33[:40] undersize, same as
T3a) are re-checked and gate `length_probes_ok`.

Three planted controls (`controls_total = 3`) are checked once, on the first fixture in
`("6MRR", "1BC8")` whose real row 0 is not a near-tie row, at bucket 128: the T3a-sized
`p04_L128_perturbed_sized.onnx` (P04 log-prob bias, must exceed 1e-4), the T3a-sized
`p03_L128_ebias_sized.onnx` (P03 edge-projection bias, must exceed 2e-5), and the T2-built
`p03_L128_swap.onnx` (a real-row-0-slot index swap; detected via a POSITIONAL, not
set-based, comparison against the wrapper -- the swap permutes row 0's own slot order
without changing its neighbour SET, so the set-based `compare_neighbor_indices` used
everywhere else would not see it). A comparator self-test (identical arrays -> 0
mismatches) is also run and reported, but per the pre-registered bars text is a sanity
check, never counted in `controls_total`.

**Differential pre-flight.** The sidecar's `[differential]` block (knob
`AMINX_BV_LAYERB_ORT_VALIDATE_PERTURB`) makes `bth run` re-execute this exact argv twice
more (`BTH_DIFFERENTIAL_{KNOB,VALUE,PHASE}` set); this script recognises that via
`layer_a_common.differential_mode()` and, in that mode, applies (value=="1") or withholds
(value=="0") the T3a-committed `delta_b04`/`delta_b03` perturbations on `5L33@128` (the
same sizing fixture/bucket T3a itself searched on) and reports
`ctrl_ratio_min = min(p04_ratio_to_bar, p03_ratio_to_bar)` to `$BTH_RESULTS_PATH` only
(never `--out`, `layer_a_common.emit`'s own differential-phase skip). The sidecar's
`min_effect` is pinned to `min(ctrl_effect.p04 / 1e-4, ctrl_effect.p03 / 2e-5)` from the
T3a-committed `preregistered_params.json` (effect in bar units) -- the SAME formula, so
the "on" arm's `ctrl_ratio_min` should reproduce `min_effect` under the identical
deterministic JAX computation.
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
import layer_a_exact as lae  # noqa: E402
import layer_b_common as lbc  # noqa: E402
from layer_b_ort_calibrate import _check_length_refusals  # noqa: E402
from layer_b_ort_calibrate import _run_ort as run_ort  # noqa: E402

CHECKPOINT_ID = "proteinmpnn_v_48_020"
FIXTURES_MANIFEST_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "fixtures" / "manifest.json"
)
TRACKED_MANIFEST_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_b" / "artifact_manifest.json"
)
PARAMS_PATH = (
  _WORKTREE_ROOT / "outputs" / "browser_validation" / "layer_b" / "preregistered_params.json"
)

#: Set-B (fixture, own-export-bucket) and every ladder bucket >= own (spec step 1 table).
OWN_BUCKET: dict[str, int] = {"6MRR": 128, "1BC8": 128, "3HTN": 512, "4YOW": 1024}
SET_B_BUCKETS: dict[str, tuple[int, ...]] = {
  "6MRR": (128, 256, 512, 1024),
  "1BC8": (128, 256, 512, 1024),
  "3HTN": (512, 1024),
  "4YOW": (1024,),
}
CONTROL_FIXTURE_CANDIDATES: tuple[str, ...] = ("6MRR", "1BC8")
CONTROL_BUCKET = 128

OVERSIZE_PROBE_FIXTURE = "2GFB"
UNDERSIZE_PROBE_LEN = 40  # 5L33[:40] < k_neighbors=48 -> LengthBelowNeighborsError
SIZING_FIXTURE = "5L33"  # same fixture T3a's own control-sizing search used
SIZING_BUCKET = 128

W_BAR = 1e-6  # W: wrapper vs native, UNPADDED (bars table)
P03_EDGE_BAR = 2e-5
P04_LOGPROB_BAR = 1e-4
ARGMAX_MARGIN = 2e-4
EPSILON_TIE = 1e-4

CONTROLS_TOTAL = 3

DIFFERENTIAL_KNOB = "AMINX_BV_LAYERB_ORT_VALIDATE_PERTURB"
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
# Headroom lookup with "inherit from nearest larger MEASURED bucket" (headroom rule)
# --------------------------------------------------------------------------------------


def _headroom_lookup(
  headroom: dict[str, Any], key_prefix: str, bucket: int, buckets_ladder: tuple[int, ...]
) -> tuple[str, int | None]:
  """Return `(state, inherited_from)` for `f"{key_prefix}|{bucket}"`.

  If T3a never measured `bucket` for this key (e.g. 256, absent from T3a's own
  FIXTURE_BUCKETS), inherit from the nearest LARGER bucket in `buckets_ladder` that DOES
  have a recorded state (headroom rule: "Buckets with no set-A fixture ... inherit the
  nearest larger measured bucket"). Falls back to `"advanced"` (no exclusion) only if no
  larger bucket has a state either -- not expected given T3a's coverage up to 1024.
  """
  key = f"{key_prefix}|{bucket}"
  entry = headroom.get(key)
  if entry is not None:
    return entry["state"], None
  for candidate in sorted(b for b in buckets_ladder if b > bucket):
    entry = headroom.get(f"{key_prefix}|{candidate}")
    if entry is not None:
      return entry["state"], candidate
  return "advanced", None


# --------------------------------------------------------------------------------------
# Row accumulation
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


def _gate_state(
  tally: dict[str, Any],
  headroom: dict[str, Any],
  route: str,
  path: str,
  quantity: str,
  bucket: int,
  buckets_ladder: tuple[int, ...],
) -> tuple[str, int | None]:
  if route in ("b_ort", "p27"):
    return _headroom_lookup(headroom, f"{path}|{quantity}|{route}", bucket, buckets_ladder)
  return "advanced", None  # W is a new (T3b-only) comparator; not in T3a's headroom system.


def _record_index_row(
  tally: dict[str, Any],
  headroom: dict[str, Any],
  route: str,
  path: str,
  quantity: str,
  bucket: int,
  cmp: dict[str, Any],
  label: str,
  buckets_ladder: tuple[int, ...],
) -> None:
  state, inherited = _gate_state(tally, headroom, route, path, quantity, bucket, buckets_ladder)
  row: dict[str, Any] = {
    "label": label,
    "path": path,
    "quantity": quantity,
    "route": route,
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
  headroom: dict[str, Any],
  route: str,
  path: str,
  quantity: str,
  bucket: int,
  value: float,
  bar: float,
  label: str,
  buckets_ladder: tuple[int, ...],
) -> None:
  state, inherited = _gate_state(tally, headroom, route, path, quantity, bucket, buckets_ladder)
  row: dict[str, Any] = {
    "label": label,
    "path": path,
    "quantity": quantity,
    "route": route,
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
  headroom: dict[str, Any],
  route: str,
  path: str,
  bucket: int,
  n_mismatch: int,
  label: str,
  buckets_ladder: tuple[int, ...],
) -> None:
  state, inherited = _gate_state(tally, headroom, route, path, "argmax", bucket, buckets_ladder)
  row: dict[str, Any] = {
    "label": label,
    "path": path,
    "quantity": "argmax",
    "route": route,
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
# EP (op-set/execution-provider) report -- AC-B3: "opset imports, op-type histogram,
# get_providers(), per-node provider from one profiled run"
# --------------------------------------------------------------------------------------


def _build_ep_report(onnx_path: Path, ep_hist: dict[str, int]) -> dict[str, Any]:
  import onnx as onnx_mod
  import onnxruntime as ort

  model_proto = onnx_mod.load(str(onnx_path))
  opset_imports = [{"domain": oi.domain, "version": oi.version} for oi in model_proto.opset_import]
  op_type_histogram: dict[str, int] = {}
  for node in model_proto.graph.node:
    op_type_histogram[node.op_type] = op_type_histogram.get(node.op_type, 0) + 1
  return {
    "artifact": onnx_path.name,
    "opset_imports": opset_imports,
    "op_type_histogram": op_type_histogram,
    "available_providers": ort.get_available_providers(),
    "session_providers": ["CPUExecutionProvider"],  # the only provider `_run_ort` requests
    "provider_node_histogram": ep_hist,
  }


# --------------------------------------------------------------------------------------
# Differential pre-flight arm (bathos [differential] pre-flight, sidecar knob)
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
  """`min(P04 log-prob ratio-to-bar, P03 edge ratio-to-bar)` on `batch` @ `SIZING_BUCKET`.

  Applies (`perturb=True`) or withholds (`perturb=False`, both deltas 0.0) the
  T3a-committed `delta_b04`/`delta_b03` and compares against the SAME baseline (the
  unperturbed model). Used both by the `[differential]` pre-flight arm (`perturb` set
  from the knob) and by the main run (`perturb=False`, a diagnostic 0.0 baseline value
  in `result_schema`, mirroring `layer_a_exact_validate`'s `sentinel_ratio_to_bar`).
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
  """One `bth run` differential arm: apply/withhold the T3a-sized deltas on 5L33@128."""
  from aminx.inference.logits import make_stage_set
  from aminx.io.weights import load_model

  if mode["knob"] != DIFFERENTIAL_KNOB:
    print(
      f"layer_b_ort_validate: BTH_DIFFERENTIAL_KNOB={mode['knob']!r}, expected "
      f"{DIFFERENTIAL_KNOB!r} (this script's sidecar knob)",
      file=sys.stderr,
    )
    return EXIT_DIFFERENTIAL_KNOB_MISMATCH

  perturb = mode["value"] == "1"
  with PARAMS_PATH.open() as fh:
    params = json.load(fh)
  delta_b04 = float(params["delta_b04"])
  delta_b03 = float(params["delta_b03"])

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
    "layer_b_ort_validate: differential arm value=%s ctrl_ratio_min=%.6g",
    mode["value"],
    metric_value,
  )
  return 0


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
    "length_probes_ok": False,
    "worst_ratio_to_bar": None,
    "worst_ratio_to_bar_available": False,
    "worst_path": None,
    "max_p_before": 0.0,
    "n_dropout": 0,
    "artifact_subdir": None,
    "versions": {},
    "comparator_self_test_pass": False,
    "control_fixture": None,
    "ctrl_ratio_min": 0.0,
  }


def run(args: argparse.Namespace) -> dict[str, Any]:
  from aminx.export import (
    EXPORT_BUCKETS,
    PINNED_CHECKPOINT_ID,
    make_p03_featurize,
    make_p04_unconditional,
    pad_inputs,
    zero_dropout,
  )
  from aminx.inference.logits import make_stage_set
  from aminx.io.weights import load_model
  from aminx.parity.compare import argmax_agreement, max_abs

  artifacts_base = Path(args.artifacts_dir)
  subdir = lbc.resolve_artifact_subdir(artifacts_base, TRACKED_MANIFEST_PATH)  # raises -> exit 3
  resolved_dir = artifacts_base / subdir
  manifest = lbc.load_manifest_verified(TRACKED_MANIFEST_PATH, resolved_dir)  # raises -> exit 3

  with PARAMS_PATH.open() as fh:
    params = json.load(fh)
  headroom: dict[str, Any] = params.get("headroom", {})
  delta_b04 = float(params["delta_b04"])
  delta_b03 = float(params["delta_b03"])

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
  native_model, dropout_stats = zero_dropout(model)
  stage_set = make_stage_set()
  k_neighbors = model.features.k_neighbors

  with FIXTURES_MANIFEST_PATH.open() as fh:
    fixture_corpus = {f["name"]: f for f in json.load(fh)["fixtures"]}

  # Diagnostic baseline (perturb=False) of the SAME metric the [differential] pre-flight
  # arm reports -- always 0.0 on the unperturbed model, present here only so the sidecar's
  # `[differential].metric` key is a declared `result_schema` field (bathos
  # `validate-sidecar` requirement), mirroring `layer_a_exact_validate`'s
  # `sentinel_ratio_to_bar` field.
  sizing_batch = lae.build_exact_batch(fixture_corpus[SIZING_FIXTURE], None)
  result["ctrl_ratio_min"] = _ctrl_ratio_min(
    model, stage_set, sizing_batch, delta_b04, delta_b03, perturb=False
  )

  length_probes = _check_length_refusals(k_neighbors, fixture_corpus)

  tally = _new_tally()
  errors: dict[str, str] = {}
  n_skipped = 0
  ep_report: dict[str, Any] | None = None
  fixture_cache: dict[str, dict[str, Any]] = {}

  for fixture_name, buckets in SET_B_BUCKETS.items():
    own_bucket = OWN_BUCKET[fixture_name]
    try:
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
      real_mask = np.ones((n_real,), dtype=bool)

      # ---- W row: wrapper vs native, UNPADDED (D-H; T1 synthetic, T3b real) ----
      unpadded_j = (
        jnp.asarray(x4),
        jnp.asarray(mask),
        jnp.asarray(ri, dtype=jnp.int32),
        jnp.asarray(ci, dtype=jnp.int32),
      )
      idx_wrap_p03_u, feat_wrap_p03_u = make_p03_featurize(model)(*unpadded_j)
      idx_wrap_p03_u = np.asarray(idx_wrap_p03_u)
      feat_wrap_p03_u = np.asarray(feat_wrap_p03_u, dtype=np.float64)

      edge_native, idx_native, _node_native, _key_native = native_model.features(
        jax.random.PRNGKey(0), *unpadded_j, backbone_noise=0.0
      )
      idx_native = np.asarray(idx_native)
      edge_native = np.asarray(edge_native, dtype=np.float64)

      cmp_w_p03 = lbc.compare_neighbor_indices(
        idx_native[:n_real], idx_wrap_p03_u[:n_real], real_mask, near_tie
      )
      _record_index_row(
        tally,
        headroom,
        "w",
        "p03",
        "neighbor_indices",
        own_bucket,
        cmp_w_p03,
        f"{fixture_name}.W.p03.idx",
        EXPORT_BUCKETS,
      )
      w_p03_edge_max_abs = max_abs(edge_native[:n_real], feat_wrap_p03_u[:n_real])
      _record_tol_row(
        tally,
        headroom,
        "w",
        "p03",
        "edge_features",
        own_bucket,
        w_p03_edge_max_abs,
        W_BAR,
        f"{fixture_name}.W.p03.edge",
        EXPORT_BUCKETS,
      )

      logits_wrap_p04_u, idx_wrap_p04_u = make_p04_unconditional(model, stage_set)(*unpadded_j)
      idx_wrap_p04_u = np.asarray(idx_wrap_p04_u)
      wrap_logprobs_u = np.asarray(jax.nn.log_softmax(logits_wrap_p04_u, axis=-1), dtype=np.float64)
      native_logprobs = np.asarray(
        lae._score_aminx_unconditional(native_model, batch),
        dtype=np.float64,  # noqa: SLF001
      )

      cmp_w_p04 = lbc.compare_neighbor_indices(
        idx_native[:n_real], idx_wrap_p04_u[:n_real], real_mask, near_tie
      )
      _record_index_row(
        tally,
        headroom,
        "w",
        "p04",
        "neighbor_indices",
        own_bucket,
        cmp_w_p04,
        f"{fixture_name}.W.p04.idx",
        EXPORT_BUCKETS,
      )
      w_p04_logprob_max_abs = max_abs(native_logprobs[:n_real], wrap_logprobs_u[:n_real])
      _record_tol_row(
        tally,
        headroom,
        "w",
        "p04",
        "log_probs",
        own_bucket,
        w_p04_logprob_max_abs,
        W_BAR,
        f"{fixture_name}.W.p04.logprobs",
        EXPORT_BUCKETS,
      )

      # ---- P27 row: wrapper PADDED to own bucket vs native UNPADDED (P04 only, D-H) ----
      padded_own = pad_inputs(x4, mask, ri, ci, own_bucket)
      logits_wrap_p04_pad, idx_wrap_p04_pad = make_p04_unconditional(model, stage_set)(
        *_jnp_padded(padded_own)
      )
      idx_wrap_p04_pad = np.asarray(idx_wrap_p04_pad)
      logprobs_wrap_p27 = np.asarray(
        jax.nn.log_softmax(logits_wrap_p04_pad, axis=-1), dtype=np.float64
      )[:n_real]

      cmp_p27 = lbc.compare_neighbor_indices(
        idx_wrap_p04_pad[:n_real], idx_native[:n_real], real_mask, near_tie
      )
      _record_index_row(
        tally,
        headroom,
        "p27",
        "p04",
        "neighbor_indices",
        own_bucket,
        cmp_p27,
        f"{fixture_name}.P27.p04.idx",
        EXPORT_BUCKETS,
      )
      p27_logprob_max_abs = max_abs(logprobs_wrap_p27, native_logprobs[:n_real])
      _record_tol_row(
        tally,
        headroom,
        "p27",
        "p04",
        "log_probs",
        own_bucket,
        p27_logprob_max_abs,
        P04_LOGPROB_BAR,
        f"{fixture_name}.P27.p04.logprobs",
        EXPORT_BUCKETS,
      )

      # ---- b-ORT rows: ORT artifact vs wrapper, both padded to the SAME bucket ----
      bort_cache: dict[int, dict[str, Any]] = {}
      for bucket in buckets:
        padded = pad_inputs(x4, mask, ri, ci, bucket)
        padded_j = _jnp_padded(padded)

        idx_wrap_p03_b, feat_wrap_p03_b = make_p03_featurize(model)(*padded_j)
        idx_wrap_p03_b = np.asarray(idx_wrap_p03_b)
        feat_wrap_p03_b = np.asarray(feat_wrap_p03_b, dtype=np.float64)

        logits_wrap_p04_b, idx_wrap_p04_b = make_p04_unconditional(model, stage_set)(*padded_j)
        idx_wrap_p04_b = np.asarray(idx_wrap_p04_b)
        logprobs_wrap_p04_b = np.asarray(
          jax.nn.log_softmax(logits_wrap_p04_b, axis=-1), dtype=np.float64
        )

        bort_cache[bucket] = {
          "padded": padded,
          "idx_wrap_p03": idx_wrap_p03_b,
          "feat_wrap_p03": feat_wrap_p03_b,
          "idx_wrap_p04": idx_wrap_p04_b,
          "logprobs_wrap_p04": logprobs_wrap_p04_b,
        }

        try:
          onnx_p03 = resolved_dir / f"p03_L{bucket}.onnx"
          outputs_p03, ep_hist_p03, _threads_p03 = run_ort(
            onnx_p03,
            padded["coords"],
            padded["mask"],
            padded["residue_index"],
            padded["chain_index"],
          )
        except Exception as exc:  # noqa: BLE001 -- recorded as a skip, never swallowed
          n_skipped += 1
          errors[f"{fixture_name}.L{bucket}.p03.b_ort"] = f"{type(exc).__name__}: {exc}"
        else:
          idx_ort_p03 = np.asarray(outputs_p03[0])
          feat_ort_p03 = np.asarray(outputs_p03[1], dtype=np.float64)
          cmp_bort_p03 = lbc.compare_neighbor_indices(
            idx_ort_p03[:n_real], idx_wrap_p03_b[:n_real], real_mask, near_tie
          )
          _record_index_row(
            tally,
            headroom,
            "b_ort",
            "p03",
            "neighbor_indices",
            bucket,
            cmp_bort_p03,
            f"{fixture_name}.L{bucket}.bORT.p03.idx",
            EXPORT_BUCKETS,
          )
          edge_max_abs = max_abs(feat_ort_p03[:n_real], feat_wrap_p03_b[:n_real])
          _record_tol_row(
            tally,
            headroom,
            "b_ort",
            "p03",
            "edge_features",
            bucket,
            edge_max_abs,
            P03_EDGE_BAR,
            f"{fixture_name}.L{bucket}.bORT.p03.edge",
            EXPORT_BUCKETS,
          )
          if ep_report is None:
            ep_report = _build_ep_report(onnx_p03, ep_hist_p03)

        try:
          onnx_p04 = resolved_dir / f"p04_L{bucket}.onnx"
          outputs_p04, ep_hist_p04, _threads_p04 = run_ort(
            onnx_p04,
            padded["coords"],
            padded["mask"],
            padded["residue_index"],
            padded["chain_index"],
          )
        except Exception as exc:  # noqa: BLE001
          n_skipped += 1
          errors[f"{fixture_name}.L{bucket}.p04.b_ort"] = f"{type(exc).__name__}: {exc}"
        else:
          logits_ort_p04 = np.asarray(outputs_p04[0])
          idx_ort_p04 = np.asarray(outputs_p04[1])
          logprobs_ort_p04 = np.asarray(
            jax.nn.log_softmax(jnp.asarray(logits_ort_p04), axis=-1), dtype=np.float64
          )
          cmp_bort_p04 = lbc.compare_neighbor_indices(
            idx_ort_p04[:n_real], idx_wrap_p04_b[:n_real], real_mask, near_tie
          )
          _record_index_row(
            tally,
            headroom,
            "b_ort",
            "p04",
            "neighbor_indices",
            bucket,
            cmp_bort_p04,
            f"{fixture_name}.L{bucket}.bORT.p04.idx",
            EXPORT_BUCKETS,
          )
          logprob_max_abs = max_abs(logprobs_ort_p04[:n_real], logprobs_wrap_p04_b[:n_real])
          _record_tol_row(
            tally,
            headroom,
            "b_ort",
            "p04",
            "log_probs",
            bucket,
            logprob_max_abs,
            P04_LOGPROB_BAR,
            f"{fixture_name}.L{bucket}.bORT.p04.logprobs",
            EXPORT_BUCKETS,
          )
          agree, valid = argmax_agreement(
            logprobs_wrap_p04_b[:n_real], logprobs_ort_p04[:n_real], ARGMAX_MARGIN
          )
          n_argmax_mismatch = int(np.sum(valid & ~agree))
          _record_argmax_row(
            tally,
            headroom,
            "b_ort",
            "p04",
            bucket,
            n_argmax_mismatch,
            f"{fixture_name}.L{bucket}.bORT.p04.argmax",
            EXPORT_BUCKETS,
          )
          if ep_report is None:
            ep_report = _build_ep_report(onnx_p04, ep_hist_p04)

      fixture_cache[fixture_name] = {"near_tie": near_tie, "n_real": n_real, "bort": bort_cache}
    except Exception as exc:  # noqa: BLE001 -- a fixture-level failure is a skip, not a crash
      n_skipped += 1
      errors[f"{fixture_name}.fatal"] = f"{type(exc).__name__}: {exc}"

  # ---- Controls (3, counted) + comparator self-test (sanity, uncounted) ----
  control_fixture: str | None = None
  for candidate in CONTROL_FIXTURE_CANDIDATES:
    info = fixture_cache.get(candidate)
    if info is None or info["n_real"] == 0:
      continue
    if not info["near_tie"][0]:
      control_fixture = candidate
      break

  controls_detail: dict[str, Any] = {}
  comparator_self_test_pass = False
  if control_fixture is not None:
    info = fixture_cache[control_fixture]
    cell = info["bort"].get(CONTROL_BUCKET)
    if cell is not None:
      padded = cell["padded"]
      n_real_c = info["n_real"]

      try:
        outputs, _eh, _th = run_ort(
          resolved_dir / f"p04_L{CONTROL_BUCKET}_perturbed_sized.onnx",
          padded["coords"],
          padded["mask"],
          padded["residue_index"],
          padded["chain_index"],
        )
        logprobs_pert = np.asarray(
          jax.nn.log_softmax(jnp.asarray(outputs[0]), axis=-1), dtype=np.float64
        )
        ratio = (
          max_abs(logprobs_pert[:n_real_c], cell["logprobs_wrap_p04"][:n_real_c]) / P04_LOGPROB_BAR
        )
        controls_detail["p04_perturbed_sized"] = {"ratio_to_bar": ratio, "detected": ratio > 1.0}
      except Exception as exc:  # noqa: BLE001
        controls_detail["p04_perturbed_sized"] = {"error": str(exc), "detected": False}

      try:
        outputs, _eh, _th = run_ort(
          resolved_dir / f"p03_L{CONTROL_BUCKET}_ebias_sized.onnx",
          padded["coords"],
          padded["mask"],
          padded["residue_index"],
          padded["chain_index"],
        )
        feat_pert = np.asarray(outputs[1], dtype=np.float64)
        ratio = max_abs(feat_pert[:n_real_c], cell["feat_wrap_p03"][:n_real_c]) / P03_EDGE_BAR
        controls_detail["p03_ebias_sized"] = {"ratio_to_bar": ratio, "detected": ratio > 1.0}
      except Exception as exc:  # noqa: BLE001
        controls_detail["p03_ebias_sized"] = {"error": str(exc), "detected": False}

      try:
        outputs, _eh, _th = run_ort(
          resolved_dir / f"p03_L{CONTROL_BUCKET}_swap.onnx",
          padded["coords"],
          padded["mask"],
          padded["residue_index"],
          padded["chain_index"],
        )
        idx_swap = np.asarray(outputs[0])
        idx_ref = cell["idx_wrap_p03"]
        row0_matches = np.array_equal(idx_swap[0], idx_ref[0])
        other_rows_match = np.array_equal(idx_swap[1:n_real_c], idx_ref[1:n_real_c])
        detected = (not row0_matches) and other_rows_match and not info["near_tie"][0]
        controls_detail["planted_swap"] = {
          "row0_positional_match": bool(row0_matches),
          "other_rows_positional_match": bool(other_rows_match),
          "row0_near_tie": bool(info["near_tie"][0]),
          "detected": bool(detected),
        }
      except Exception as exc:  # noqa: BLE001
        controls_detail["planted_swap"] = {"error": str(exc), "detected": False}

      real_mask_c = np.ones((n_real_c,), dtype=bool)
      self_cmp = lbc.compare_neighbor_indices(
        cell["idx_wrap_p03"][:n_real_c],
        cell["idx_wrap_p03"][:n_real_c],
        real_mask_c,
        info["near_tie"],
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
      "length_probes_ok": bool(length_probes["ok"]),
      "worst_ratio_to_bar": tally["worst_ratio"] if tally["worst_path"] else None,
      "worst_ratio_to_bar_available": tally["worst_path"] is not None,
      "worst_path": tally["worst_path"],
      "max_p_before": dropout_stats["max_p_before"],
      "n_dropout": dropout_stats["n_dropout"],
      "versions": versions,
      "comparator_self_test_pass": comparator_self_test_pass,
      "control_fixture": control_fixture,
      "_rows": tally["rows"],
      "_ep_report": ep_report,
      "_controls_detail": controls_detail,
      "_length_probes": length_probes,
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
  args = parser.parse_args(argv)

  mode = lac.differential_mode()
  if mode["active"]:
    return run_differential_arm(mode, args)

  try:
    result = run(args)
    exit_code = 0
  except (FileNotFoundError, ValueError) as exc:
    logger.error("layer_b_ort_validate: integrity refusal: %s", exc)
    result = _empty_result(None, False)
    result["_integrity_error"] = str(exc)
    exit_code = 3

  lac.emit(result, args.out)
  logger.info(
    "layer_b_ort_validate: n_paths_available=%s controls=%s/%s n_over_bar=%s "
    "n_tie_unstable=%s n_excluded_not_advanced=%s n_low_headroom=%s n_skipped=%s exit_code=%d",
    result.get("n_paths_available"),
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
