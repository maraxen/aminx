"""Unit tests for membrane residue list parser and physics features (Clause 2: membrane_label_construction)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from aminx.io.membrane import (
  construct_membrane_labels,
  construct_membrane_physics_features,
  parse_residue_spec,
)


def test_parse_residue_spec_various_formats() -> None:
  """Test parsing space-separated, comma-separated, lists, and empty specs."""
  assert parse_residue_spec("A1 A2 A5") == {"A1", "A2", "A5"}
  assert parse_residue_spec("A1, A2, A5") == {"A1", "A2", "A5"}
  assert parse_residue_spec(["A1", "A2"]) == {"A1", "A2"}
  assert parse_residue_spec({0, 1, 3}) == {"0", "1", "3"}
  assert parse_residue_spec("") == set()
  assert parse_residue_spec(None) == set()


def test_construct_membrane_labels_formula_and_quirk() -> None:
  """Test per-residue membrane labeling reproduces 2*buried*(1-interface) + 1*interface*(1-buried)."""
  residues = ["A1", "A2", "A3", "A4"]

  # A1: buried only -> 2
  # A2: interface only -> 1
  # A3: neither -> 0
  # A4: both buried and interface -> 0 (upstream quirk)
  buried = "A1 A4"
  interface = "A2 A4"

  labels = construct_membrane_labels(
    encoded_residues=residues,
    buried_residues=buried,
    interface_residues=interface,
  )

  expected = jnp.array([2, 1, 0, 0], dtype=jnp.int32)
  assert jnp.array_equal(labels, expected)


def test_construct_membrane_labels_global_broadcast() -> None:
  """Test broadcasting global membrane label {0, 1, 2} across all residues."""
  for global_label in (0, 1, 2):
    labels = construct_membrane_labels(
      encoded_residues=5,
      global_label=global_label,
    )
    assert jnp.array_equal(labels, jnp.full((5,), global_label, dtype=jnp.int32))

  # Invalid global label raises ValueError
  with pytest.raises(ValueError, match="global_label must be in"):
    construct_membrane_labels(encoded_residues=5, global_label=3)


def test_construct_membrane_physics_features_one_hot() -> None:
  """Test construct_membrane_physics_features produces exact (L, 3) one-hot float32."""
  residues = ["A1", "A2", "A3"]
  features = construct_membrane_physics_features(
    encoded_residues=residues,
    buried_residues="A1",
    interface_residues="A2",
  )

  assert features.shape == (3, 3)
  assert features.dtype == jnp.float32

  # Residue 0 (A1): buried -> class 2 -> [0.0, 0.0, 1.0]
  assert jnp.array_equal(features[0], jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32))
  # Residue 1 (A2): interface -> class 1 -> [0.0, 1.0, 0.0]
  assert jnp.array_equal(features[1], jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32))
  # Residue 2 (A3): neither -> class 0 -> [1.0, 0.0, 0.0]
  assert jnp.array_equal(features[2], jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32))


def test_construct_membrane_physics_features_numeric_residue_indices() -> None:
  """Test integer sequence length or integer residue indices."""
  features = construct_membrane_physics_features(
    encoded_residues=4,
    buried_residues=[0, 1],
    interface_residues=[2],
  )
  assert features.shape == (4, 3)
  # 0, 1: buried -> class 2
  assert jnp.array_equal(features[0], jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32))
  assert jnp.array_equal(features[1], jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32))
  # 2: interface -> class 1
  assert jnp.array_equal(features[2], jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32))
  # 3: neither -> class 0
  assert jnp.array_equal(features[3], jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32))
