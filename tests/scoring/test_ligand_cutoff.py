"""Unit tests for ligand cutoff scoring (Clause 1: ligand_cutoff_for_score)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.scoring.ligand_cutoff import (
  DEFAULT_LIGAND_CUTOFF_ANGSTROMS,
  compute_ligand_cutoff_mask,
  compute_loss_xy,
  compute_virtual_c_beta,
  evaluate_ligand_cutoff_score,
)


def test_virtual_c_beta_calculation() -> None:
  """Test that compute_virtual_c_beta reproduces the empirical empirical formula."""
  # Create known backbone geometry: N at origin, CA on x-axis, C in xy-plane
  n = jnp.array([0.0, 0.0, 0.0])
  ca = jnp.array([1.46, 0.0, 0.0])
  c = jnp.array([1.96, 1.42, 0.0])

  coords = jnp.stack([n, ca, c], axis=0)  # (3, 3)

  b = ca - n
  c_vec = c - ca
  a = jnp.cross(b, c_vec)
  expected_cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca

  computed_cb = compute_virtual_c_beta(coords[None, ...])[0]
  assert jnp.allclose(computed_cb, expected_cb, atol=1e-6)


def test_ligand_cutoff_mask_distance_threshold() -> None:
  """Test that residues are correctly masked by distance to ligand atoms."""
  # 3 residues
  ca1 = jnp.array([0.0, 0.0, 0.0])
  ca2 = jnp.array([6.0, 0.0, 0.0])
  ca3 = jnp.array([15.0, 0.0, 0.0])

  # Mock backbone coordinates with N, CA, C aligned so virtual C-beta is near CA
  coords = jnp.zeros((3, 4, 3))
  coords = coords.at[:, 1, :].set(jnp.stack([ca1, ca2, ca3], axis=0))
  # Place N and C with normal bond vectors
  coords = coords.at[:, 0, :].set(coords[:, 1, :] - jnp.array([1.45, 0.0, 0.0]))
  coords = coords.at[:, 2, :].set(coords[:, 1, :] + jnp.array([0.5, 1.4, 0.0]))

  # Place a ligand atom near residue 1 (dist ~ 2.0 A) and residue 2 (dist ~ 5.0 A), far from residue 3
  ligand_coords = jnp.array([[2.0, 0.0, 0.0], [99.0, 99.0, 99.0]])
  ligand_mask = jnp.array([1.0, 0.0])  # second atom is masked padding

  res_mask = jnp.array([1.0, 1.0, 1.0])

  # Cutoff 8.0 A: residue 1 and 2 should be within cutoff, residue 3 outside
  mask_xy, d_xy = compute_ligand_cutoff_mask(
    coords, ligand_coords, ligand_mask, res_mask, cutoff=8.0
  )
  assert mask_xy[0] == 1.0
  assert mask_xy[1] == 1.0
  assert mask_xy[2] == 0.0

  # Cutoff 4.0 A: only residue 1 should be within cutoff
  mask_xy_strict, _ = compute_ligand_cutoff_mask(
    coords, ligand_coords, ligand_mask, res_mask, cutoff=4.0
  )
  assert mask_xy_strict[0] == 1.0
  assert mask_xy_strict[1] == 0.0
  assert mask_xy_strict[2] == 0.0


def test_ligand_cutoff_mask_masked_residues_and_padding() -> None:
  """Test that invalid protein residues or empty ligand sets yield zero mask."""
  coords = jnp.zeros((2, 4, 3))
  ligand_coords = jnp.array([[0.0, 0.0, 0.0]])
  ligand_mask = jnp.array([1.0])

  # Protein residue 1 is padding (mask=0)
  res_mask = jnp.array([1.0, 0.0])
  mask_xy, _ = compute_ligand_cutoff_mask(coords, ligand_coords, ligand_mask, res_mask)
  assert mask_xy[0] == 1.0
  assert mask_xy[1] == 0.0

  # When all ligand atoms are masked out (ligand_mask all 0s)
  mask_xy_empty, _ = compute_ligand_cutoff_mask(
    coords, ligand_coords, jnp.array([0.0]), res_mask
  )
  assert jnp.all(mask_xy_empty == 0.0)

  # When ligand_coords is None
  mask_xy_none, _ = compute_ligand_cutoff_mask(coords, None, None, res_mask)
  assert jnp.all(mask_xy_none == 0.0)


def test_compute_loss_xy_arithmetic() -> None:
  """Test loss_xy computes exact weighted categorical cross-entropy."""
  # Sequence of 3 residues: tokens [0, 5, 10]
  seq = jnp.array([0, 5, 10])

  # Logits where true tokens have known values
  logits = jnp.zeros((3, 21))
  # Residue 0: confident prediction
  logits = logits.at[0, 0].set(10.0)
  # Residue 1: unconfident prediction
  logits = logits.at[1, 5].set(-5.0)

  log_probs = jax.nn.log_softmax(logits, axis=-1)
  ce_per_res = -jnp.array([log_probs[0, 0], log_probs[1, 5], log_probs[2, 10]])

  # Mask only residue 0 and residue 1
  mask_xy = jnp.array([1.0, 1.0, 0.0])
  expected_loss = (ce_per_res[0] + ce_per_res[1]) / 2.0

  loss_xy = compute_loss_xy(logits, seq, mask_xy)
  assert jnp.isclose(loss_xy, expected_loss, atol=1e-5)


def test_evaluate_ligand_cutoff_score_result_structure() -> None:
  """Test evaluate_ligand_cutoff_score returns complete LigandScoreResult."""
  coords = jnp.zeros((4, 4, 3))
  seq = jnp.array([1, 2, 3, 4])
  logits = jnp.ones((4, 21))
  res_mask = jnp.ones(4)
  ligand_coords = jnp.zeros((2, 3))
  ligand_mask = jnp.ones(2)

  result = evaluate_ligand_cutoff_score(
    coordinates=coords,
    sequence=seq,
    logits=logits,
    residue_mask=res_mask,
    ligand_coords=ligand_coords,
    ligand_mask=ligand_mask,
    cutoff=DEFAULT_LIGAND_CUTOFF_ANGSTROMS,
  )

  assert result.loss_xy >= 0.0
  assert 0.0 <= result.ligand_confidence <= 1.0
  assert result.mask_xy.shape == (4,)
  assert result.num_ligand_proximal_residues == 4
