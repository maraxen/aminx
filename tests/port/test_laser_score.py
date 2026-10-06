"""laser_score wave.

One item per (checkpoint, fixture). Ids are read from the dump at collection
time; an absent oracle yields no pairs and the wave skips.

The scoring path is fed the dump's ``decoding_order``. Order log-probs are an
invariant of the no-generator branch: every entry is exactly 1.

The two tiers use DIFFERENT FORMS of comparison, deliberately.

f64 (tier 2) is element-wise at rtol=1e-8, atol=1e-11. It is the correctness
check: it asks whether the port computes the same function as upstream.

f32 (tier 3) is SCALE-RELATIVE at 4e-4 -- ``max|got-ref| <= tol * max|ref|``,
reduced per field per pair. This is option 3 of
decisions/261003_laser-score-tier3-f32-design.md, chosen by the user
2026-10-06. It exists because run 18b97f4f showed the two f32 implementations
make the SAME error (they agree with each other ~200x more tightly than either
agrees with an f64 truth), so no element-wise band could both admit that shared
f32 noise and still reject a real 1% defect.

Both bounds are pre-registered and measured. A miss reports the measured
deviation and does not widen them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from port.a1_compare import assert_shape_dtype, names, open_dump, x64_context
from port.reference.laser_score.algo import OracleAbsentError, load as load_score_dump

from aminx.families.laser_mpnn.featurize import featurize
from aminx.model.laser.decoder import LaserDecoder, binned_degree_basis, score_structure
from aminx.model.laser.encoders import LaserEncoder
from aminx.model.laser.graphs import GraphStructure

pytestmark = [pytest.mark.port_wave("laser_score"), pytest.mark.parity_heavy]

# f64 keeps the ELEMENT-WISE envelope. It is the actual correctness check, and
# decisions/261003_laser-score-tier3-f32-design.md is explicit that no option
# there touches it. f64 is a parity dtype, never a production one.
_RTOL = {"f64": 1e-8}
# House pairing for the declared rtol. Not a measured deviation.
_ATOL = {"f64": 1e-11}
# f32 is SCALE-RELATIVE (option 3, chosen by the user 2026-10-06). The rule is a
# REDUCTION per field per pair, not an element-wise envelope:
#
#     max|got - ref|  <=  tol * max|ref|
#
# The change of FORM is the point, not the change of value. Run 18b97f4f
# measured that aminx-f32 and upstream-f32 agree with each other ~200x more
# tightly than either agrees with an f64 truth: they make the SAME f32 error, so
# this tier compares two implementations of one approximation. An element-wise
# band wide enough to admit that shared error could not separate it from a real
# 1% defect, which is why widening the old rule was refused rather than tuned.
#
# The band is measured, not chosen: graded run 3af3be02
# (scripts/analysis/laser_score_scale_relative_floor.py @ 0e642d44,
# completed/pass/exit 0/clean) gives floor_scale_relative = 3.6297e-5 over
# n_pairs = 63 -- the same 63 pairs this tier asserts over, not a sample of
# them -- with headroom = 10.0, control_rejected = true (the negative control
# still fails) and loo_all_admitted = true (loo_worst_ratio 0.121, so no single
# pair props the floor up).
_SCALE_REL = {"f32": 4e-4}
_CHECKPOINTS = {
  "nothing_heldout": "model_weights/laser_weights_0p1A_nothing_heldout.pt",
  "noise_ligandmpnn_split": "model_weights/laser_weights_0p1A_noise_ligandmpnn_split.pt",
  "soluble_65000": "model_weights/soluble_weights_no_heldout_drop_clusters_optstep_65000.pt",
}
_ENCODER_PREFIXES = (
  "ligand_encoder.",
  "ligand_encoder_output_gvp.",
  "backbone_frame_vec_input_layer.",
  "protein_encoder_layers.",
  "prot_prot_rbf_encoding.",
  "lig_prot_rbf_encoding.",
  "prot_prot_edge_input_layer.",
  "lig_prot_edge_input_layer.",
)
_DECODER_PREFIXES = (
  "protein_decoder_layers.",
  "sequence_label_embedding.",
  "sequence_output_layer.",
  "chi_prediction_layers.",
  "chi_vector_update_layers.",
  "chi_vector_layer_norms.",
)
_EXACT = (
  "chi_bin",
  "decoding_order",
  "pr_pr_idx",
  "sequence_indices",
  "first_shell",
)
_FLOATS = (
  "sequence_logits",
  "chi_logits",
  "chi_log_prob",
  "seq_log_prob",
  "decoding_order_log_probs",
  "encoder_scalars",
  "encoder_edges",
  "ligand_scalars",
  "lig_prot_edges",
)
# chi_angles is an INPUT to teacher forcing, not a scoring output. The dump
# routes it to the laser_encoder and laser_rotamers waves, never to this one,
# so comparing it here asserts a key laser_score never carried. It stays gated
# by the encoder wave and by B0's featurize parity.
_REPO = Path(__file__).resolve().parents[2]


def _laser_root() -> Path:
  return Path(os.environ.get("AMINX_LASER_ROOT", "~/repos/LASErMPNN")).expanduser()


def _fixture_path(name: str) -> Path:
  if name == "4jnj-1_prot":
    return _laser_root() / "example_pdbs" / "4jnj-1_prot.pdb"
  return _REPO / "tests" / "fixtures" / "laser" / f"{name}.pdb"


_BLOB_CACHE: dict[str, dict[str, Any]] = {}
_FEATURE_CACHE: dict[tuple[str, np.dtype, int, float], Any] = {}


def _blob(checkpoint: str) -> dict[str, Any]:
  cached = _BLOB_CACHE.get(checkpoint)
  if cached is not None:
    return cached
  torch = pytest.importorskip("torch")
  root = _laser_root()
  if not root.is_dir():
    pytest.skip(f"LASErMPNN checkout absent: {root}")
  path = root / _CHECKPOINTS[checkpoint]
  if not path.is_file():
    pytest.skip(f"checkpoint absent: {path}")
  blob = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(blob, dict):
    msg = f"{path} is not a checkpoint dict"
    raise TypeError(msg)
  _BLOB_CACHE[checkpoint] = blob
  return blob


def _graph_structure(blob: dict[str, Any]) -> GraphStructure:
  params = blob["params"]
  if not isinstance(params, dict):
    msg = "checkpoint params is not a dict"
    raise TypeError(msg)
  model_params = params["model_params"]
  if not isinstance(model_params, dict):
    msg = "checkpoint model_params is not a dict"
    raise TypeError(msg)
  raw = model_params["graph_structure"]
  if not isinstance(raw, dict):
    msg = "checkpoint graph_structure is not a dict"
    raise TypeError(msg)
  return GraphStructure(
    pr_pr_knn_graph_k=int(raw["pr_pr_knn_graph_k"]),
    lig_pr_knn_graph_k=int(raw["lig_pr_knn_graph_k"]),
    lig_lig_knn_graph_k=int(raw["lig_lig_knn_graph_k"]),
    lig_pr_distance_cutoff=float(raw["lig_pr_distance_cutoff"]),
  )


def _set_path(tree: Any, parts: list[str], value: jax.Array) -> Any:
  head, *rest = parts
  if head.isdigit():
    index = int(head)
    seq = list(tree)
    seq[index] = _set_path(seq[index], rest, value) if rest else value
    return tuple(seq)
  child = _set_path(getattr(tree, head), rest, value) if rest else value
  return eqx.tree_at(lambda obj, name=head: getattr(obj, name), tree, child)


def _load(module: Any, state: dict[str, Any], dtype: np.dtype, prefixes: tuple[str, ...]) -> Any:
  for key, value in state.items():
    text = str(key)
    if not text.startswith(prefixes):
      continue
    # D_mu is a derived linspace, not a learned weight. See the encoder wave.
    if dtype == np.float64 and text.endswith("D_mu"):
      continue
    array = np.asarray(value.detach().cpu().numpy())
    if np.issubdtype(array.dtype, np.floating):
      array = array.astype(dtype, copy=False)
    module = _set_path(module, text.split("."), jnp.asarray(array))
  return module


def _features(fixture: str, dtype: np.dtype, structure: GraphStructure) -> Any:
  key = (
    fixture,
    np.dtype(dtype),
    structure.lig_pr_knn_graph_k,
    structure.lig_pr_distance_cutoff,
  )
  cached = _FEATURE_CACHE.get(key)
  if cached is not None:
    return cached
  path = _fixture_path(fixture)
  if not path.is_file():
    pytest.skip(f"LASEr fixture absent: {path}")
  # First-shell uses the checkpoint cutoff. Sequence and chi do not.
  features = featurize(
    path,
    dtype=dtype,
    lig_pr_knn_k=structure.lig_pr_knn_graph_k,
    lig_pr_distance_cutoff=structure.lig_pr_distance_cutoff,
  )
  _FEATURE_CACHE[key] = features
  return features


def _derived(
  sequence_logits: np.ndarray,
  chi_logits: np.ndarray,
  sequence_indices: np.ndarray,
  chi_angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Match ``dump_laser_oracles._score_arrays`` on the ported logits."""
  seq_log = np.asarray(jax.nn.log_softmax(jnp.asarray(sequence_logits), axis=-1))
  seq_lp = seq_log[np.arange(seq_log.shape[0]), sequence_indices]
  basis = np.asarray(jnp.nan_to_num(binned_degree_basis(jnp.asarray(chi_angles))))
  chi_bin = np.argmax(basis, axis=-1).astype(np.int64)
  chi_log = np.asarray(jax.nn.log_softmax(jnp.asarray(chi_logits), axis=-1))
  rows = np.arange(chi_log.shape[0])[:, None]
  angles = np.arange(chi_log.shape[1])[None, :]
  picked = chi_log[rows, angles, chi_bin]
  chi_lp = np.where(np.isnan(chi_angles), np.nan, picked)
  return seq_lp, chi_bin, np.asarray(chi_lp, dtype=chi_logits.dtype)


def _predict(
  checkpoint: str,
  fixture: str,
  dtype: np.dtype,
  decoding_order: np.ndarray,
) -> dict[str, np.ndarray]:
  blob = _blob(checkpoint)
  state = blob["model_state_dict"]
  if not isinstance(state, dict):
    msg = "model_state_dict is not a dict"
    raise TypeError(msg)
  encoder = _load(LaserEncoder(key=jax.random.key(0)), state, dtype, _ENCODER_PREFIXES)
  decoder = _load(LaserDecoder(key=jax.random.key(1)), state, dtype, _DECODER_PREFIXES)
  structure = _graph_structure(blob)
  features = _features(fixture, dtype, structure)
  period = np.asarray(state["ligand_featurizer.atomic_number_idx_to_period_idx"])
  group = np.asarray(state["ligand_featurizer.atomic_number_idx_to_group_idx"])
  scored = score_structure(
    encoder,
    decoder,
    features.backbone_coords,
    features.ligand_coords,
    features.ligand_atomic_numbers,
    features.ligand_subbatch_indices,
    period,
    group,
    structure,
    decoding_order,
    features.sequence_indices,
    features.chi_angles,
  )
  sequence = np.asarray(features.sequence_indices)
  chi = np.asarray(features.chi_angles, dtype=np.float64 if dtype == np.float64 else np.float32)
  seq_lp, chi_bin, chi_lp = _derived(
    scored.sequence_logits,
    scored.chi_logits,
    sequence,
    chi,
  )
  return {
    "sequence_logits": scored.sequence_logits,
    "chi_logits": scored.chi_logits,
    "chi_log_prob": chi_lp,
    "chi_bin": chi_bin,
    "seq_log_prob": seq_lp,
    "decoding_order": np.asarray(decoding_order),
    "decoding_order_log_probs": scored.decoding_order_log_probs,
    "encoder_scalars": scored.encoder_scalars,
    "encoder_edges": scored.encoder_edges,
    "ligand_scalars": scored.ligand_scalars,
    "lig_prot_edges": scored.lig_prot_edges,
    "pr_pr_idx": scored.pr_pr_idx,
    "sequence_indices": sequence,
    "first_shell": np.asarray(features.first_shell_ligand_contact_mask),
  }


def _rows_over(
  got: np.ndarray,
  ref: np.ndarray,
  rtol: float,
  atol: float,
) -> tuple[int, int, float]:
  """Rows exceeding ``atol + rtol*|ref|``. Shared NaNs are not errors."""
  got64 = got.astype(np.float64)
  ref64 = ref.astype(np.float64)
  both_nan = np.isnan(got64) & np.isnan(ref64)
  nan_mismatch = np.isnan(got64) != np.isnan(ref64)
  error = np.abs(got64 - ref64)
  error = np.where(both_nan, 0.0, error)
  error = np.where(nan_mismatch, np.inf, error)
  finite_ref = np.where(np.isnan(ref64), 0.0, ref64)
  limit = atol + rtol * np.abs(finite_ref)
  bad = error > limit
  worst = float(np.max(error)) if error.size else 0.0
  if not np.isfinite(worst):
    worst = float(np.max(np.abs(got64 - ref64), initial=0.0))
  if bad.ndim == 0:
    return int(bad), 1, worst
  row_bad = np.any(bad, axis=tuple(range(1, bad.ndim)))
  return int(np.sum(row_bad)), int(row_bad.shape[0]), worst


def _scale_relative(got: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
  """``max|got-ref|``, the scale ``max|ref|``, and their ratio.

  This is the option-3 rule's measurement. It is a reduction over the whole
  field, so unlike ``_rows_over`` it does not ask each element to sit inside an
  envelope -- it asks whether the field's largest deviation is small *relative
  to the field's own magnitude*. NaN handling matches ``_rows_over`` so the two
  rules disagree only about form: shared NaNs are not errors, and a NaN on one
  side only is infinite error.

  A zero scale is not silently forgiving. If ``max|ref|`` is 0 the only
  deviation the rule can admit is exactly 0, and the ratio is reported as inf
  for any nonzero deviation so the failure message says why.
  """
  got64 = got.astype(np.float64)
  ref64 = ref.astype(np.float64)
  both_nan = np.isnan(got64) & np.isnan(ref64)
  nan_mismatch = np.isnan(got64) != np.isnan(ref64)
  error = np.abs(got64 - ref64)
  error = np.where(both_nan, 0.0, error)
  error = np.where(nan_mismatch, np.inf, error)
  max_abs = float(np.max(error)) if error.size else 0.0
  finite_ref = np.where(np.isnan(ref64), 0.0, ref64)
  scale = float(np.max(np.abs(finite_ref))) if finite_ref.size else 0.0
  if scale > 0.0:
    ratio = max_abs / scale
  else:
    ratio = 0.0 if max_abs == 0.0 else float("inf")
  return max_abs, scale, ratio


def _run(
  data: np.lib.npyio.NpzFile,
  precision: str,
  checkpoint: str,
  fixture: str,
  *,
  numeric: bool,
) -> int:
  compared = 0
  failures: list[str] = []
  worst = 0.0
  dtype = np.dtype(np.float64 if precision == "f64" else np.float32)
  rtol = _RTOL.get(precision)
  atol = _ATOL.get(precision)
  scale_rel = _SCALE_REL.get(precision)
  # A precision must be defined by exactly ONE rule. Without this, a half-done
  # migration that left a stale _RTOL[precision] beside a new _SCALE_REL entry
  # would silently apply whichever branch happened to be checked first, and the
  # emitted verdict would name a band the test did not use -- the exact defect
  # tests/lint/test_port_tolerances_match_targets.py exists to prevent.
  if (rtol is None and atol is None) == (scale_rel is None):
    msg = (
      f"laser_score {precision}: exactly one of the element-wise band "
      f"(_RTOL/_ATOL) and the scale-relative band (_SCALE_REL) must define it"
    )
    raise AssertionError(msg)
  prefix = f"{checkpoint}__{fixture}__"
  order_key = prefix + "decoding_order"
  if order_key not in data.files:
    msg = f"laser_score dump has no {order_key}"
    raise AssertionError(msg)
  decoding_order = np.asarray(data[order_key])
  uniforms_key = prefix + "order_uniforms"
  if uniforms_key not in data.files:
    msg = f"laser_score dump has no {uniforms_key}"
    raise AssertionError(msg)
  uniforms = np.asarray(data[uniforms_key])
  with x64_context(precision), jax.default_matmul_precision("highest"):
    predicted = _predict(checkpoint, fixture, dtype, decoding_order)
    if uniforms.dtype != np.float64 or uniforms.shape != (decoding_order.shape[0] + 8,):
      failures.append(
        f"{uniforms_key}: expected float64 length {decoding_order.shape[0] + 8}, "
        f"got {uniforms.dtype} {uniforms.shape}",
      )
    compared += 1
    for field in _EXACT + _FLOATS:
      key = prefix + field
      if key not in data.files:
        msg = f"laser_score dump has no {key}"
        raise AssertionError(msg)
      got = predicted[field]
      ref = np.asarray(data[key])
      label = f"laser_score {precision} {key}"
      if got.shape != ref.shape or got.dtype != ref.dtype:
        assert_shape_dtype(got, ref, label)
      if field in _EXACT:
        if not np.array_equal(got, ref):
          n_bad = int(np.sum(got != ref))
          failures.append(f"{key}: {n_bad} entries differ (exact)")
          worst = max(worst, 1.0)
      else:
        if field == "decoding_order_log_probs" and not np.array_equal(
          got,
          np.ones(got.shape, dtype=got.dtype),
        ):
          failures.append(f"{key}: not exactly 1 (no decoding-order generator)")
          worst = max(worst, float(np.max(np.abs(got.astype(np.float64) - 1.0))))
        if numeric and scale_rel is not None:
          max_abs, scale, ratio = _scale_relative(got, ref)
          worst = max(worst, max_abs)
          # Negated rather than `ratio > scale_rel` so a NaN ratio fails.
          if not ratio <= scale_rel:
            failures.append(
              f"{key}: max|got-ref|/max|ref| = {ratio:.6e} exceeds "
              f"scale_rel={scale_rel:g} "
              f"(max|got-ref|={max_abs:.6e}, max|ref|={scale:.6e})",
            )
        elif numeric and rtol is not None and atol is not None:
          n_bad, n_rows, max_abs = _rows_over(got, ref, rtol, atol)
          worst = max(worst, max_abs)
          if n_bad:
            failures.append(
              f"{key}: {n_bad}/{n_rows} rows exceed atol={atol:g}+rtol={rtol:g}",
            )
      compared += 1
  if failures:
    report = "\n".join(failures)
    msg = f"Max absolute difference: {worst:.6e}\n{report}"
    raise AssertionError(msg)
  return compared


def _oracle_pairs() -> list[tuple[str, str]]:
  """Read checkpoint and fixture ids from the dump at collection time."""
  data: np.lib.npyio.NpzFile | None = None
  for precision in ("f64", "f32"):
    try:
      data = load_score_dump(precision)
    except OracleAbsentError:
      continue
    break
  if data is None:
    return []
  try:
    checkpoints = names(data["checkpoint_ids"])
    fixtures = names(data["fixture_names"])
  finally:
    data.close()
  return [(checkpoint, fixture) for checkpoint in checkpoints for fixture in fixtures]


_PAIRS = _oracle_pairs()
_PAIR_IDS = [f"{checkpoint}-{fixture}" for checkpoint, fixture in _PAIRS]


def _skip_absent_oracle() -> None:
  if not _PAIRS:
    pytest.skip("laser_score oracle absent")


@pytest.mark.tier_1
@pytest.mark.parametrize("precision", ("f64", "f32"))
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_1_dtype_shape(
  oracle: object,
  precision: str,
  checkpoint: str,
  fixture: str,
) -> None:
  """Shapes and dtypes match, and integer and bool fields match exactly."""
  _skip_absent_oracle()
  data = open_dump(oracle, precision)
  try:
    assert _run(data, precision, checkpoint, fixture, numeric=False) > 0
  finally:
    data.close()


@pytest.mark.tier_2
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_2_f64(oracle: object, checkpoint: str, fixture: str) -> None:
  """f64 score outputs match at rtol=1e-8, atol=1e-11 under x64."""
  _skip_absent_oracle()
  data = open_dump(oracle, "f64")
  try:
    assert _run(data, "f64", checkpoint, fixture, numeric=True) > 0
  finally:
    data.close()


@pytest.mark.tier_3
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_3_f32(oracle: object, checkpoint: str, fixture: str) -> None:
  """f32 score outputs match scale-relatively: max|got-ref| <= 4e-4 * max|ref|."""
  _skip_absent_oracle()
  data = open_dump(oracle, "f32")
  try:
    assert _run(data, "f32", checkpoint, fixture, numeric=True) > 0
  finally:
    data.close()


def test_scale_relative_band_rejects_an_injected_defect() -> None:
  """NEGATIVE CONTROL for the option-3 band. Needs no oracle, so it always runs.

  A band that cannot fail is not a band. Run 3af3be02's 10x headroom only means
  something if a defect of the size the decision doc cares about is actually
  rejected, so this injects one. It also checks the clean case still passes, so
  the control is not vacuous in the other direction, and pins the two edge cases
  where a scale-relative rule could quietly forgive: NaN and a zero scale.
  """
  tol = _SCALE_REL["f32"]
  rng = np.random.default_rng(0)
  ref = rng.normal(size=(64, 9)).astype(np.float32)

  # Clean: a ~1e-6 relative perturbation, the order of f32 round-off. Passes.
  clean = (ref * np.float32(1.0 + 1e-6)).astype(np.float32)
  assert _scale_relative(clean, ref)[2] <= tol

  # A 1e-2 RELATIVE defect -- the size the decision doc requires separation
  # from -- must be rejected, and with real margin rather than barely.
  defect = (ref * np.float32(1.0 + 1e-2)).astype(np.float32)
  ratio = _scale_relative(defect, ref)[2]
  assert not ratio <= tol
  assert ratio / tol > 10.0, f"injected defect only {ratio / tol:.1f}x over the band"

  # Shared NaNs are not errors; a one-sided NaN is infinite error.
  with_nan = ref.copy()
  with_nan[0, 0] = np.nan
  assert _scale_relative(with_nan.copy(), with_nan)[2] == 0.0
  assert not _scale_relative(ref, with_nan)[2] <= tol

  # A zero scale admits an exact match and nothing else.
  zeros = np.zeros((4,), dtype=np.float32)
  assert _scale_relative(zeros, zeros)[2] == 0.0
  assert _scale_relative(zeros + np.float32(1e-30), zeros)[2] == float("inf")


@pytest.mark.tier_5
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_5_trace_budget(
  oracle: object,
  max_traces: int,
  checkpoint: str,
  fixture: str,
) -> None:
  """One decoder shape traces at most ``max_traces`` times."""
  import chex

  _skip_absent_oracle()
  data = open_dump(oracle, "f32")
  try:
    prefix = f"{checkpoint}__{fixture}__"
    present = [key for key in data.files if key.startswith(prefix)]
    assert len(present) > 0
  finally:
    data.close()
  model = LaserDecoder(key=jax.random.key(0))
  length = 4
  atoms = 2
  prot_k = 3
  lig_k = 2
  dtype = jnp.float32
  prot_s = jnp.zeros((length, 256), dtype=dtype)
  prot_v = jnp.zeros((length, 10, 3), dtype=dtype)
  lig_s = jnp.zeros((atoms, 256), dtype=dtype)
  pr_edges = jnp.zeros((length, prot_k, 128), dtype=dtype)
  pr_n = jnp.zeros((length, prot_k), dtype=jnp.int32)
  pr_m = jnp.ones((length, prot_k), dtype=jnp.bool_)
  lp_edges = jnp.zeros((length, lig_k, 128), dtype=dtype)
  lp_n = jnp.zeros((length, lig_k), dtype=jnp.int32)
  lp_m = jnp.ones((length, lig_k), dtype=jnp.bool_)
  order = jnp.arange(length, dtype=jnp.int32)
  sequence = jnp.zeros((length,), dtype=jnp.int32)
  chi = jnp.zeros((length, 4), dtype=dtype)
  chex.clear_trace_counter()

  @chex.assert_max_traces(n=max_traces)
  def kernel(
    layer: LaserDecoder,
    scalars: jax.Array,
    vectors: jax.Array,
    ligand: jax.Array,
  ) -> jax.Array:
    logits, _chi, _banked, _order_lp = layer(
      scalars,
      vectors,
      ligand,
      pr_edges,
      pr_n,
      pr_m,
      lp_edges,
      lp_n,
      lp_m,
      order,
      sequence,
      chi,
    )
    return logits

  compiled = eqx.filter_jit(kernel)
  compiled(model, prot_s, prot_v, lig_s)
  compiled(model, prot_s, prot_v, lig_s)
