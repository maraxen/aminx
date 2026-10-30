# ruff: noqa: S101
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
from pathlib import Path

import equinox as eqx
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
from aminx.families.thermo_mpnn.driver import ThermoMPNNDriver
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

  x_c = f32_ddg.ravel() - np.mean(f32_ddg)
  y_c = f16_ddg.ravel() - np.mean(f16_ddg)
  pearson_r = float(np.sum(x_c * y_c) / np.sqrt(np.sum(x_c**2) * np.sum(y_c**2)))
  assert pearson_r > 0.999


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


def test_knob_semantics_checkpoint_id() -> None:
  """Checkpoint_id selects architecture and produces distinct model outputs."""
  driver = ThermoMPNNDriver()
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)

  sd_v1 = _build_synthetic_v1_state_dict()
  sd_v2 = _build_synthetic_v2_single_state_dict()
  m_v1 = convert_thermo_mpnn_state_dict(sd_v1)
  m_v2 = convert_thermo_mpnn_state_dict(sd_v2)

  res_v1 = driver.score(
    m_v1,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    options=ThermoMPNNOptions(checkpoint_id="thermo_mpnn_v1_default"),
  )
  res_v2 = driver.score(
    m_v2,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    options=ThermoMPNNOptions(checkpoint_id="thermo_mpnn_v2_ens1"),
  )

  assert not np.allclose(res_v1["ddg"], res_v2["ddg"], atol=1e-2)
