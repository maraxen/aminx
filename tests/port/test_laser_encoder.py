"""laser_encoder wave.

One item per (checkpoint, fixture). Ids are read from the dump at collection
time; an absent oracle yields no pairs and the wave skips.

f64 uses rtol=1e-8, atol=1e-11. f32 uses rtol=1e-4, atol=1e-7 and reports every
key whose rows exceed that bound (debt #2299: the ligand path can miss a
same-precision f32 oracle on a minority of rows; the bound is not loosened).

Edge indices are compared exactly, separate from the embedding leaves, so a
kNN convention error fails before the stack is judged.
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
from port.reference.laser_encoder.algo import OracleAbsentError, load as load_encoder_dump

from aminx.families.laser_mpnn.featurize import featurize
from aminx.model.laser.encoders import LIGAND_ATOMS, LaserEncoder, encode_structure, ligand_atom_bucket
from aminx.model.laser.graphs import GraphStructure

pytestmark = [pytest.mark.port_wave("laser_encoder"), pytest.mark.parity_heavy]

_RTOL = {"f64": 1e-8, "f32": 1e-4}
# House pairing for the declared rtol. Not a measured deviation; the
# orchestrator replaces atol from the graded run. See targets/laser_encoder.toml.
_ATOL = {"f64": 1e-11, "f32": 1e-7}
_CHECKPOINTS = {
  "nothing_heldout": "model_weights/laser_weights_0p1A_nothing_heldout.pt",
  "noise_ligandmpnn_split": "model_weights/laser_weights_0p1A_noise_ligandmpnn_split.pt",
  "soluble_65000": "model_weights/soluble_weights_no_heldout_drop_clusters_optstep_65000.pt",
}
_LOAD_PREFIXES = (
  "ligand_encoder.",
  "ligand_encoder_output_gvp.",
  "backbone_frame_vec_input_layer.",
  "protein_encoder_layers.",
  "prot_prot_rbf_encoding.",
  "lig_prot_rbf_encoding.",
  "prot_prot_edge_input_layer.",
  "lig_prot_edge_input_layer.",
)
_EXACT = ("pr_pr_idx", "lig_pr_idx", "sequence_indices", "chain_mask")
_FLOATS = (
  "prot_scalars",
  "prot_vectors",
  "lig_scalars",
  "lig_vectors",
  "pr_pr_eattr",
  "lig_pr_eattr",
  "chi_angles",
)
_REPO = Path(__file__).resolve().parents[2]


def _laser_root() -> Path:
  return Path(os.environ.get("AMINX_LASER_ROOT", "~/repos/LASErMPNN")).expanduser()


def _fixture_path(name: str) -> Path:
  if name == "4jnj-1_prot":
    return _laser_root() / "example_pdbs" / "4jnj-1_prot.pdb"
  return _REPO / "tests" / "fixtures" / "laser" / f"{name}.pdb"


_BLOB_CACHE: dict[str, dict[str, Any]] = {}
_FEATURE_CACHE: dict[str, Any] = {}


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


def _load(module: LaserEncoder, state: dict[str, Any], dtype: np.dtype) -> LaserEncoder:
  for key, value in state.items():
    text = str(key)
    if not text.startswith(_LOAD_PREFIXES):
      continue
    array = np.asarray(value.detach().cpu().numpy())
    if np.issubdtype(array.dtype, np.floating):
      array = array.astype(dtype, copy=False)
    module = _set_path(module, text.split("."), jnp.asarray(array))
  return module


def _features(fixture: str) -> Any:
  cached = _FEATURE_CACHE.get(fixture)
  if cached is not None:
    return cached
  path = _fixture_path(fixture)
  if not path.is_file():
    pytest.skip(f"LASEr fixture absent: {path}")
  features = featurize(path)
  _FEATURE_CACHE[fixture] = features
  return features


def _predict(
  checkpoint: str,
  fixture: str,
  dtype: np.dtype,
) -> dict[str, np.ndarray]:
  blob = _blob(checkpoint)
  state = blob["model_state_dict"]
  if not isinstance(state, dict):
    msg = "model_state_dict is not a dict"
    raise TypeError(msg)
  model = LaserEncoder(key=jax.random.key(0))
  model = _load(model, state, dtype)
  features = _features(fixture)
  numpy_dtype = np.float64 if dtype == np.float64 else np.float32
  period = np.asarray(state["ligand_featurizer.atomic_number_idx_to_period_idx"])
  group = np.asarray(state["ligand_featurizer.atomic_number_idx_to_group_idx"])
  encoded = encode_structure(
    model,
    features.backbone_coords,
    features.ligand_coords,
    features.ligand_atomic_numbers,
    features.ligand_subbatch_indices,
    period,
    group,
    _graph_structure(blob),
  )
  return {
    "lig_scalars": encoded.lig_scalars,
    "lig_vectors": encoded.lig_vectors,
    "prot_scalars": encoded.prot_scalars,
    "prot_vectors": encoded.prot_vectors,
    "pr_pr_eattr": encoded.pr_pr_eattr,
    "lig_pr_eattr": encoded.lig_pr_eattr,
    "pr_pr_idx": encoded.pr_pr_idx,
    "lig_pr_idx": encoded.lig_pr_idx,
    "sequence_indices": np.asarray(features.sequence_indices),
    "chain_mask": np.asarray(features.chain_mask),
    "chi_angles": np.asarray(features.chi_angles, dtype=numpy_dtype),
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
  dtype = np.float64 if precision == "f64" else np.float32
  rtol = _RTOL[precision]
  atol = _ATOL[precision]
  prefix = f"{checkpoint}__{fixture}__"
  with x64_context(precision), jax.default_matmul_precision("highest"):
    predicted = _predict(checkpoint, fixture, dtype)
    for field in _EXACT + _FLOATS:
      key = prefix + field
      if key not in data.files:
        msg = f"laser_encoder dump has no {key}"
        raise AssertionError(msg)
      got = predicted[field]
      ref = np.asarray(data[key])
      label = f"laser_encoder {precision} {key}"
      if got.shape != ref.shape or got.dtype != ref.dtype:
        assert_shape_dtype(got, ref, label)
      if field in _EXACT:
        if not np.array_equal(got, ref):
          n_bad = int(np.sum(got != ref))
          failures.append(f"{key}: {n_bad} entries differ (exact edge/index field)")
          if np.issubdtype(np.asarray(got).dtype, np.number):
            delta = np.abs(got.astype(np.float64) - ref.astype(np.float64))
            worst = max(worst, float(np.max(delta)) if delta.size else 0.0)
          else:
            worst = max(worst, 1.0)
      elif numeric:
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
      data = load_encoder_dump(precision)
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
    pytest.skip("laser_encoder oracle absent")


@pytest.mark.tier_1
@pytest.mark.parametrize("precision", ("f64", "f32"))
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_1_dtype_shape(
  oracle: object,
  precision: str,
  checkpoint: str,
  fixture: str,
) -> None:
  """Shapes and dtypes match, and the host edge indices match exactly."""
  _skip_absent_oracle()
  data = open_dump(oracle, precision)
  try:
    assert _run(data, precision, checkpoint, fixture, numeric=False) > 0
  finally:
    data.close()


@pytest.mark.tier_2
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_2_f64(oracle: object, checkpoint: str, fixture: str) -> None:
  """f64 encoder outputs match at rtol=1e-8, atol=1e-11 under x64."""
  _skip_absent_oracle()
  data = open_dump(oracle, "f64")
  try:
    assert _run(data, "f64", checkpoint, fixture, numeric=True) > 0
  finally:
    data.close()


@pytest.mark.tier_3
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_3_f32(oracle: object, checkpoint: str, fixture: str) -> None:
  """f32 encoder outputs match at rtol=1e-4, atol=1e-7, reporting every key.

  Debt #2299: a same-precision f32 ligand path can exceed a tight bound on a
  minority of rows while f64 stays clean. This tier records those keys and the
  worst absolute delta. It does not widen the bound.
  """
  _skip_absent_oracle()
  data = open_dump(oracle, "f32")
  try:
    assert _run(data, "f32", checkpoint, fixture, numeric=True) > 0
  finally:
    data.close()


@pytest.mark.tier_5
@pytest.mark.parametrize(("checkpoint", "fixture"), _PAIRS, ids=_PAIR_IDS)
def test_tier_5_trace_budget(
  oracle: object,
  max_traces: int,
  checkpoint: str,
  fixture: str,
) -> None:
  """One ligand-atom bucket traces at most ``max_traces`` times."""
  import chex

  _skip_absent_oracle()
  data = open_dump(oracle, "f32")
  try:
    prefix = f"{checkpoint}__{fixture}__"
    assert any(key.startswith(prefix) for key in data.files)
  finally:
    data.close()
  assert LIGAND_ATOMS.bucket_boundaries is not None
  bucket = ligand_atom_bucket(3)
  assert bucket == ligand_atom_bucket(8)
  model = LaserEncoder(key=jax.random.key(0))
  length = 4
  width = 2
  atoms = bucket
  backbone = jnp.zeros((length, 5, 3), dtype=jnp.float32)
  lig_coords = jnp.zeros((atoms, 3), dtype=jnp.float32)
  lig_features = jnp.zeros((atoms, 26), dtype=jnp.float32)
  pr_n = jnp.zeros((length, width), dtype=jnp.int32)
  pr_m = jnp.ones((length, width), dtype=jnp.bool_)
  lp_n = jnp.zeros((length, width), dtype=jnp.int32)
  lp_m = jnp.ones((length, width), dtype=jnp.bool_)
  ll_n = jnp.zeros((atoms, width), dtype=jnp.int32)
  ll_m = jnp.ones((atoms, width), dtype=jnp.bool_)
  ll_d = jnp.zeros((atoms, width), dtype=jnp.float32)
  pr_d = jnp.zeros((length, width, 25), dtype=jnp.float32)
  lp_d = jnp.zeros((length, width, 5), dtype=jnp.float32)
  sink = jnp.arange(length, dtype=jnp.int32)
  slots = jnp.zeros((length,), dtype=jnp.int32)
  chex.clear_trace_counter()

  @chex.assert_max_traces(n=max_traces)
  def kernel(
    layer: LaserEncoder,
    bb: jax.Array,
    coords: jax.Array,
    feats: jax.Array,
  ) -> jax.Array:
    lig_s, _lig_v, _prot_s, _prot_v, _pr, _lp = layer(
      bb,
      coords,
      feats,
      pr_n,
      pr_m,
      lp_n,
      lp_m,
      ll_n,
      ll_m,
      ll_d,
      pr_d,
      lp_d,
      sink,
      slots,
      sink,
      slots,
    )
    return lig_s

  compiled = eqx.filter_jit(kernel)
  compiled(model, backbone, lig_coords, lig_features)
  compiled(model, backbone, lig_coords, lig_features)
