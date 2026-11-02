# ruff: noqa: S101, SLF001
"""ThermoMPNN and ThermoMPNN-D knob semantics tests.

Tests the semantic behavior and plumbing of all 6 ThermoMPNNOptions knobs:
1. `mode`: "single" vs "double" output structure.
2. `distance_threshold`: CA-CA distance thresholding for epistatic candidate pairs.
3. `ensemble`: averaging predictions across ensemble members vs single model.
4. `dtype`: float32 vs float16 execution and precision band.
5. `batch_size`: chunking of double-mutant pairs produces identical predictions.
6. `checkpoint_id`: architecture routing and distinct outputs between v1 and v2 models.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "tests" / "families") not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT / "tests" / "families"))

from test_thermo_mpnn_parity import (
  _UBQ_PDB,
  _build_synthetic_v1_state_dict,
  _build_synthetic_v2_double_state_dict,
  _build_synthetic_v2_single_state_dict,
  _parse_pdb_backbone,
)

from aminx.families.thermo_mpnn import (
  ThermoMPNNEnsemble,
  convert_thermo_mpnn_double_state_dict,
  convert_thermo_mpnn_state_dict,
)
from aminx.families.thermo_mpnn import driver as thermo_driver_module
from aminx.host.family_runner import _check_chunk
from aminx.families.thermo_mpnn.driver import ThermoMPNNDriver
from aminx.families.thermo_mpnn.model import ThermoMPNNDouble
from aminx.families.thermo_mpnn.stats import pearson_r
from aminx.run.options import ThermoMPNNOptions


def test_knob_semantics_mode() -> None:
  """Mode knob toggles between single-mutant table and double-mutant Siamese predictions."""
  driver = ThermoMPNNDriver()
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  sd_double = _build_synthetic_v2_double_state_dict()
  model = convert_thermo_mpnn_double_state_dict(sd_double)

  res_single = driver.score(
    model,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    options=ThermoMPNNOptions(mode="single"),
  )
  assert "ddg" in res_single
  assert "ddg_double" not in res_single
  assert "pair_indices" not in res_single
  assert res_single["ddg"].shape == (len(sequence), 21)

  res_double = driver.score(
    model,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    options=ThermoMPNNOptions(mode="double", distance_threshold=6.0),
  )
  assert "ddg" in res_double
  assert "ddg_double" in res_double
  assert "pair_indices" in res_double
  num_pairs = res_double["pair_indices"].shape[0]
  assert num_pairs > 0
  assert res_double["ddg_double"].shape == (num_pairs, 21, 21)


def test_knob_semantics_distance_threshold() -> None:
  """Tighter distance threshold selects a strict subset of candidate residue pairs."""
  driver = ThermoMPNNDriver()
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  sd_double = _build_synthetic_v2_double_state_dict()
  model = convert_thermo_mpnn_double_state_dict(sd_double)

  res_tight = driver.score(
    model,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    options=ThermoMPNNOptions(mode="double", distance_threshold=4.0),
  )
  res_loose = driver.score(
    model,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    options=ThermoMPNNOptions(mode="double", distance_threshold=8.0),
  )

  pairs_tight = set(map(tuple, np.asarray(res_tight["pair_indices"]).tolist()))
  pairs_loose = set(map(tuple, np.asarray(res_loose["pair_indices"]).tolist()))

  assert len(pairs_tight) > 0
  assert len(pairs_loose) > len(pairs_tight)
  assert pairs_tight.issubset(pairs_loose)

  ca_coords = np.asarray(coords)[:, 1, :]
  for i, j in pairs_tight:
    dist = float(np.linalg.norm(ca_coords[i] - ca_coords[j]))
    assert dist <= 4.0


def test_knob_semantics_ensemble() -> None:
  """Ensemble=True averages all member models; ensemble=False uses only member 0."""
  driver = ThermoMPNNDriver()
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)

  sd1 = _build_synthetic_v1_state_dict()
  rng2 = np.random.default_rng(202)
  sd2 = _build_synthetic_v1_state_dict()
  for k in sd2:
    if "weight" in k:
      sd2[k] = sd2[k] + rng2.standard_normal(sd2[k].shape).astype(np.float32) * 0.1

  m1 = convert_thermo_mpnn_state_dict(sd1)
  m2 = convert_thermo_mpnn_state_dict(sd2)
  ensemble = ThermoMPNNEnsemble(models=(m1, m2))

  pred1 = driver.score(m1, coords, mask, residue_index, chain_index, sequence)["ddg"]
  pred2 = driver.score(m2, coords, mask, residue_index, chain_index, sequence)["ddg"]
  assert not np.allclose(pred1, pred2, atol=1e-3)

  res_ens_true = driver.score(
    ensemble,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    options=ThermoMPNNOptions(ensemble=True),
  )
  expected_ens = (np.asarray(pred1) + np.asarray(pred2)) / 2.0
  np.testing.assert_allclose(res_ens_true["ddg"], expected_ens, atol=1e-5)

  res_ens_false = driver.score(
    ensemble,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    options=ThermoMPNNOptions(ensemble=False),
  )
  np.testing.assert_allclose(res_ens_false["ddg"], pred1, atol=1e-5)
  assert not np.allclose(res_ens_true["ddg"], res_ens_false["ddg"], atol=1e-3)


def test_knob_semantics_dtype() -> None:
  """Float16 execution runs stably and matches float32 within tolerance band."""
  driver = ThermoMPNNDriver()
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  sd = _build_synthetic_v1_state_dict()
  model = convert_thermo_mpnn_state_dict(sd)

  res_f32 = driver.score(
    model,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    options=ThermoMPNNOptions(dtype="float32"),
  )
  with pytest.warns(UserWarning, match="float16 is not validated"):
    res_f16 = driver.score(
      model,
      coords,
      mask,
      residue_index,
      chain_index,
      sequence,
      options=ThermoMPNNOptions(dtype="float16"),
    )

  f32_ddg = np.asarray(res_f32["ddg"])
  f16_ddg = np.asarray(res_f16["ddg"])

  max_abs_diff = float(np.max(np.abs(f32_ddg - f16_ddg)))
  assert max_abs_diff < 1.5e-2
  assert max_abs_diff > 0.0

  assert pearson_r(f32_ddg, f16_ddg) > 0.999


def test_knob_semantics_batch_size() -> None:
  """Batch_size chunks candidate pairs without altering double-mutant predictions."""
  driver = ThermoMPNNDriver()
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  sd_double = _build_synthetic_v2_double_state_dict()
  model = convert_thermo_mpnn_double_state_dict(sd_double)

  res_chunked = driver.score(
    model,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    options=ThermoMPNNOptions(mode="double", distance_threshold=6.0, batch_size=2),
  )
  res_unchunked = driver.score(
    model,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    options=ThermoMPNNOptions(mode="double", distance_threshold=6.0, batch_size=500),
  )

  num_pairs = res_chunked["pair_indices"].shape[0]
  assert num_pairs > 4
  np.testing.assert_array_equal(res_chunked["pair_indices"], res_unchunked["pair_indices"])
  np.testing.assert_allclose(res_chunked["ddg_double"], res_unchunked["ddg_double"], atol=1e-5)


@pytest.fixture
def routed_loaders(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
  """Swap the checkpoint loaders for synthetic-weight fakes that record ``(kind, id)`` per call."""
  calls: list[tuple[str, str]] = []

  def _single(checkpoint_id: str, *, local_path: object = None, key: object = None) -> eqx.Module:
    del local_path, key
    calls.append(("single", checkpoint_id))
    sd = (
      _build_synthetic_v1_state_dict()
      if "v1" in checkpoint_id
      else _build_synthetic_v2_single_state_dict()
    )
    return convert_thermo_mpnn_state_dict(sd)

  def _double(checkpoint_id: str, *, local_path: object = None, key: object = None) -> eqx.Module:
    del local_path, key
    calls.append(("double", checkpoint_id))
    return convert_thermo_mpnn_double_state_dict(_build_synthetic_v2_double_state_dict())

  monkeypatch.setattr(thermo_driver_module, "load_thermo_mpnn_checkpoint", _single)
  monkeypatch.setattr(thermo_driver_module, "load_thermo_mpnn_double_checkpoint", _double)
  return calls


def test_knob_semantics_checkpoint_id(routed_loaders: list[tuple[str, str]]) -> None:
  """checkpoint_id resolves option > spec.checkpoint_id > default and drives load, stages and schema."""
  driver = ThermoMPNNDriver()

  def _spec(spec_id: str | None, opts: ThermoMPNNOptions | None) -> SimpleNamespace:
    return SimpleNamespace(checkpoint_id=spec_id, thermompnn=opts, model_local_path=None)

  # Layer 3: nothing set anywhere -> v1 default, single-mutant outputs only.
  spec = _spec(None, None)
  model = driver.load(spec)
  assert routed_loaders[-1] == ("single", "thermo_mpnn_v1_default")
  assert not driver.stages(spec, "score:ddg", model)._double_mutant
  assert "ddg_double" not in driver.result_schema(spec, "score:ddg")

  # Layer 2: spec.checkpoint_id with a default-valued options block (used to be shadowed by v1).
  spec = _spec("thermo_mpnn_v2_ens1", ThermoMPNNOptions())
  model = driver.load(spec)
  assert routed_loaders[-1] == ("single", "thermo_mpnn_v2_ens1")
  assert not driver.stages(spec, "score:ddg", model)._double_mutant
  assert "ddg_double" not in driver.result_schema(spec, "score:ddg")

  # Layer 2 with a ThermoMPNN-D id: loads the double model and both stage and schema go double.
  spec = _spec("thermo_mpnn_d_ens1", None)
  model = driver.load(spec)
  assert routed_loaders[-1] == ("double", "thermo_mpnn_d_ens1")
  assert isinstance(model, ThermoMPNNDouble)
  assert driver.stages(spec, "score:ddg", model)._double_mutant
  schema = driver.result_schema(spec, "score:ddg")
  assert "ddg_double" in schema
  assert schema["ddg_double"].attrs["ddg_double_method"] == "epistatic"

  # Ensemble id routes to three double members.
  n_before = len(routed_loaders)
  spec = _spec("thermo_mpnn_d_ensemble", None)
  driver.load(spec)
  assert routed_loaders[n_before:] == [("double", cid) for cid in thermo_driver_module.V2_DOUBLE_ENSEMBLE_IDS]

  # Layer 1: explicit option beats spec.checkpoint_id in both directions.
  spec = _spec("thermo_mpnn_v2_ens1", ThermoMPNNOptions(checkpoint_id="thermo_mpnn_d_ens2"))
  driver.load(spec)
  assert routed_loaders[-1] == ("double", "thermo_mpnn_d_ens2")
  assert "ddg_double" in driver.result_schema(spec, "score:ddg")

  spec = _spec("thermo_mpnn_d_ens1", ThermoMPNNOptions(checkpoint_id="thermo_mpnn_v1_default"))
  model = driver.load(spec)
  assert routed_loaders[-1] == ("single", "thermo_mpnn_v1_default")
  assert not isinstance(model, ThermoMPNNDouble)
  assert "ddg_double" not in driver.result_schema(spec, "score:ddg")


def test_knob_semantics_mode_and_dtype_validation() -> None:
  """Unknown mode or dtype values are rejected instead of silently accepted."""
  driver = ThermoMPNNDriver()
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  model = convert_thermo_mpnn_state_dict(_build_synthetic_v1_state_dict())

  with pytest.raises(ValueError, match="mode"):
    driver.score(model, coords, mask, residue_index, chain_index, sequence, mode="doubel")
  with pytest.raises(ValueError, match="dtype"):
    driver.score(model, coords, mask, residue_index, chain_index, sequence, dtype="fp16")
  with pytest.raises(ValueError, match="mode"):
    ThermoMPNNOptions(mode="doubel")  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="dtype"):
    ThermoMPNNOptions(dtype="fp16")  # type: ignore[arg-type]


def test_knob_semantics_double_model_default_outputs() -> None:
  """A ThermoMPNNDouble model with default arguments returns ddg, ddg_double and pair_indices."""
  driver = ThermoMPNNDriver()
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  model = convert_thermo_mpnn_double_state_dict(_build_synthetic_v2_double_state_dict())

  res = driver.score(model, coords, mask, residue_index, chain_index, sequence)
  assert "ddg" in res
  assert "ddg_double" in res
  assert "pair_indices" in res
  num_pairs = res["pair_indices"].shape[0]
  assert num_pairs > 0
  assert res["ddg_double"].shape == (num_pairs, 21, 21)


def test_knob_semantics_no_pairs_within_threshold() -> None:
  """No residue pair within distance_threshold gives zero epistatic pairs, not adjacent pairs."""
  driver = ThermoMPNNDriver()
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  model = convert_thermo_mpnn_double_state_dict(_build_synthetic_v2_double_state_dict())

  res = driver.score(
    model,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    options=ThermoMPNNOptions(mode="double", distance_threshold=3.0),
  )
  assert np.asarray(res["pair_indices"]).shape == (0, 2)
  assert np.asarray(res["ddg_double"]).shape == (0, 21, 21)


def test_knob_semantics_additive_double_for_single_model() -> None:
  """Single-mutant model with mode='double' returns s_i + s_j and says so in the schema."""
  driver = ThermoMPNNDriver()
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  model = convert_thermo_mpnn_state_dict(_build_synthetic_v2_single_state_dict())

  res = driver.score(
    model,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    options=ThermoMPNNOptions(mode="double", distance_threshold=6.0),
  )
  pairs = np.asarray(res["pair_indices"])
  ddg = np.asarray(res["ddg"])
  assert pairs.shape[0] > 0
  expected = ddg[pairs[:, 0], :, None] + ddg[pairs[:, 1], None, :]
  np.testing.assert_allclose(np.asarray(res["ddg_double"]), expected, atol=1e-5)

  spec = SimpleNamespace(
    checkpoint_id="thermo_mpnn_v2_ens1",
    thermompnn=ThermoMPNNOptions(mode="double"),
    model_local_path=None,
  )
  schema = driver.result_schema(spec, "score:ddg")
  assert schema["ddg_double"].attrs["ddg_double_method"] == "additive"


def test_knob_semantics_float16_passes_result_schema() -> None:
  """float16 stage output goes through the family_runner schema check (float32 declared)."""
  driver = ThermoMPNNDriver()
  spec = SimpleNamespace(
    checkpoint_id="thermo_mpnn_v1_default",
    thermompnn=ThermoMPNNOptions(dtype="float16"),
    model_local_path=None,
  )
  model = convert_thermo_mpnn_state_dict(_build_synthetic_v1_state_dict())
  stages = driver.stages(spec, "score:ddg", model)
  batch = driver.collate_batch((driver.prepare_structure(_UBQ_PDB),), input_indices=(0,))
  with pytest.warns(UserWarning, match="float16 is not validated"):
    produced = dict(
      stages(
        batch,
        chunk_start=0,
        chunk_count=1,
        scalar_dropout=False,
        dropout_key=jax.random.PRNGKey(0),
      ),
    )
  schema = driver.result_schema(spec, "score:ddg")
  # Mirror family_runner._consume_batch: single-structure output gains the leading axis.
  leading = {
    name: np.asarray(value)[None, ...]
    if np.asarray(value).ndim == len(schema[name].dims)
    else np.asarray(value)
    for name, value in produced.items()
  }
  _check_chunk(leading, schema, n_structures=1)
  assert np.asarray(produced["ddg"]).dtype == np.float32


def test_pearson_r_zero_variance_is_nan() -> None:
  """A constant output has undefined correlation: NaN, never 1.0, so it cannot pass an r gate."""
  ramp = np.arange(12, dtype=np.float64)
  assert np.isnan(pearson_r(np.ones(12), ramp))
  assert np.isnan(pearson_r(ramp, np.full(12, 3.0)))
  assert pearson_r(ramp, 2.0 * ramp + 1.0) == pytest.approx(1.0)


def test_knob_semantics_double_model_forces_double_without_options() -> None:
  """Without options, a ThermoMPNNDouble model gives double output even when double_mutant=False."""
  driver = ThermoMPNNDriver()
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  model = convert_thermo_mpnn_double_state_dict(_build_synthetic_v2_double_state_dict())

  res = driver.score(
    model,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    double_mutant=False,
  )
  assert "ddg_double" in res
  assert res["ddg_double"].shape[1:] == (21, 21)


def test_checkpoint_double_predicate_is_shared_by_load_and_schema(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A local .pt id containing 'double' routes to the double loader and the schema agrees."""
  driver = ThermoMPNNDriver()
  routed: list[str] = []
  monkeypatch.setattr(
    thermo_driver_module,
    "load_thermo_mpnn_double_pt",
    lambda path, *, key=None: routed.append(f"double:{path.name}") or "double-model",
  )
  monkeypatch.setattr(
    thermo_driver_module,
    "load_thermo_mpnn_pt",
    lambda path, *, key=None: routed.append(f"single:{path.name}") or "single-model",
  )
  assert thermo_driver_module._is_double_checkpoint("thermo_mpnn_double_custom")
  assert driver.load_model("thermo_mpnn_double_custom", local_path="w.pt") == "double-model"
  assert driver.load_model("thermo_mpnn_v1_default", local_path="w.pt") == "single-model"
  assert routed == ["double:w.pt", "single:w.pt"]

  spec = SimpleNamespace(
    checkpoint_id="thermo_mpnn_double_custom",
    thermompnn=None,
    model_local_path=None,
  )
  assert "ddg_double" in driver.result_schema(spec, "score:ddg")


def test_knob_semantics_mixed_pair_counts_in_one_batch() -> None:
  """A batch whose items have P>0 and P=0 pairs pads the pair axis instead of failing."""
  driver = ThermoMPNNDriver()
  spec = SimpleNamespace(
    checkpoint_id="thermo_mpnn_d_ens1",
    thermompnn=ThermoMPNNOptions(mode="double", distance_threshold=6.0),
    model_local_path=None,
  )
  model = convert_thermo_mpnn_double_state_dict(_build_synthetic_v2_double_state_dict())
  item_close = driver.prepare_structure(_UBQ_PDB)
  item_far = dict(item_close)
  item_far["coords"] = np.asarray(item_close["coords"]) * 10.0  # no CA-CA pair within 6 A
  batch = driver.collate_batch((item_close, item_far), input_indices=(0, 1))
  stages = driver.stages(spec, "score:ddg", model)
  produced = dict(
    stages(
      batch,
      chunk_start=0,
      chunk_count=2,
      scalar_dropout=False,
      dropout_key=jax.random.PRNGKey(0),
    ),
  )
  pairs = np.asarray(produced["pair_indices"])
  doubles = np.asarray(produced["ddg_double"])
  n_close = int(np.sum(pairs[0, :, 0] >= 0))
  assert n_close > 0
  assert pairs.shape == (2, n_close, 2)
  assert doubles.shape == (2, n_close, 21, 21)
  assert np.all(pairs[1] == -1)
  assert np.all(np.isnan(doubles[1]))
  assert np.all(np.isfinite(doubles[0]))


def test_knob_semantics_zero_pairs_through_write_outputs(tmp_path: Path) -> None:
  """P=0 outputs survive write_outputs with the declared (0, 21, 21) and (0, 2) shapes."""
  driver = ThermoMPNNDriver()
  results = {
    "ddg": np.zeros((5, 21), dtype=np.float32),
    "ddg_double": np.zeros((0, 21, 21), dtype=np.float32),
    "pair_indices": np.zeros((0, 2), dtype=np.int32),
  }
  out = driver.write_outputs(tmp_path / "zero_pairs", results)
  npy = out / "ddg_double.npy"
  if npy.exists():  # zarr absent: the documented .npy fallback
    assert np.load(npy).shape == (0, 21, 21)
    assert np.load(out / "pair_indices.npy").shape == (0, 2)
  else:
    zarr = pytest.importorskip("zarr")
    group = zarr.open_group(str(out), mode="r")
    assert group["ddg_double"].shape == (0, 21, 21)
    assert group["pair_indices"].shape == (0, 2)


def _float_warnings(record: list[warnings.WarningMessage]) -> list[str]:
  """Messages of the float16 'not validated' warning among ``record``."""
  return [str(w.message) for w in record if "float16 is not validated" in str(w.message)]


def test_knob_semantics_float16_warns_once_per_score_call() -> None:
  """Selecting float16 warns on each score() call, with the sidecar and run id cited."""
  driver = ThermoMPNNDriver()
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  model = convert_thermo_mpnn_state_dict(_build_synthetic_v1_state_dict())
  with warnings.catch_warnings(record=True) as record:
    warnings.simplefilter("always")
    driver.score(
      model,
      coords,
      mask,
      residue_index,
      chain_index,
      sequence,
      options=ThermoMPNNOptions(dtype="float16"),
    )
  messages = _float_warnings(record)
  assert len(messages) == 1
  assert "thermo_mpnn_driver_f16_v2.bth.toml" in messages[0]
  assert "18c1d29c" in messages[0]


def test_knob_semantics_float32_and_float64_do_not_warn() -> None:
  """float32 (the default) and float64 selections emit no float16 warning."""
  driver = ThermoMPNNDriver()
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  model = convert_thermo_mpnn_state_dict(_build_synthetic_v1_state_dict())
  with warnings.catch_warnings(record=True) as record:
    warnings.simplefilter("always")
    driver.score(model, coords, mask, residue_index, chain_index, sequence)
    driver.score(
      model,
      coords,
      mask,
      residue_index,
      chain_index,
      sequence,
      options=ThermoMPNNOptions(dtype="float32"),
    )
  assert _float_warnings(record) == []
  with pytest.raises(ValueError, match="dtype"):
    driver.score(model, coords, mask, residue_index, chain_index, sequence, dtype="float64")
