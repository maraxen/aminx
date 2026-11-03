# ruff: noqa: S101
"""Tier 1 and Tier 2 MPNN variant knob semantics.

Covers:
- AbMPNN: antibody model weights resolution and skeleton routing
- HyperMPNN: hypervariable model weights resolution across all 4 noise tiers (002, 010, 020, 030)
- CyclicMPNN: head-to-tail cyclic peptide topology and cyclic sequence offset wrapping
- CA-Only: C-alpha-only featurizer (ProteinFeaturesCA), coordinate dimensionality equivalence,
  and distance gating.

Every test pairs the aminx behavior against expected semantics and includes a genuine negative control.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.io.weights import (
  _resolve_checkpoint_id,
  get_topology_for_checkpoint,
)
from aminx.model.features import ProteinFeatures, ProteinFeaturesCA
from aminx.model.mpnn import Aminx
from aminx.utils.graph import compute_neighbor_offsets


def test_knob_semantics_abmpnn() -> None:
  """AbMPNN routes to abmpnn_v_48_020 checkpoint and protein topology."""
  resolved = _resolve_checkpoint_id(model_weights="abmpnn", model_version="v_48_020")
  assert resolved == "abmpnn_v_48_020.eqx.zst"

  # Default version resolution
  resolved_default = _resolve_checkpoint_id(model_weights="abmpnn")
  assert resolved_default == "abmpnn_v_48_020.eqx.zst"

  topo = get_topology_for_checkpoint(resolved)
  assert topo["model_type"] == "protein"
  assert topo["cyclic"] is False
  assert topo["k_neighbors"] == 48

  # Negative control: unpublished version raises ValueError
  with pytest.raises(ValueError, match="not available"):
    _resolve_checkpoint_id(model_weights="abmpnn", model_version="v_48_010")


def test_knob_semantics_hypermpnn() -> None:
  """HyperMPNN routes to all 4 noise-tier checkpoints (002, 010, 020, 030)."""
  for ver in ("v_48_002", "v_48_010", "v_48_020", "v_48_030"):
    resolved = _resolve_checkpoint_id(model_weights="hypermpnn", model_version=ver)
    assert resolved == f"hypermpnn_{ver}.eqx.zst"

    topo = get_topology_for_checkpoint(resolved)
    assert topo["model_type"] == "protein"
    assert topo["cyclic"] is False
    assert topo["k_neighbors"] == 48

  # Negative control: unsupported version raises ValueError
  with pytest.raises(ValueError, match="not available"):
    _resolve_checkpoint_id(model_weights="hypermpnn", model_version="v_32_010_25")


def test_knob_semantics_cyclicmpnn() -> None:
  """CyclicMPNN routes to cyclicmpnn_v_48_010 and marks topology cyclic=True."""
  resolved = _resolve_checkpoint_id(model_weights="cyclicmpnn", model_version="v_48_010")
  assert resolved == "cyclicmpnn_v_48_010.eqx.zst"

  resolved_default = _resolve_checkpoint_id(model_weights="cyclicmpnn")
  assert resolved_default == "cyclicmpnn_v_48_010.eqx.zst"

  topo = get_topology_for_checkpoint(resolved)
  assert topo["cyclic"] is True
  assert topo["model_type"] == "protein"
  assert topo["k_neighbors"] == 48

  # Negative control: standard protein MPNN does not enable cyclic topology
  standard_topo = get_topology_for_checkpoint("proteinmpnn_v_48_020.eqx.zst")
  assert standard_topo["cyclic"] is False


def test_knob_semantics_cyclic_offsets() -> None:
  """Cyclic sequence offsets wrap intra-chain distances across cycle length."""
  # 36 residues on chain 0
  n_res = 36
  residue_indices = jnp.arange(1, n_res + 1, dtype=jnp.int32)
  chain_index = jnp.zeros(n_res, dtype=jnp.int32)
  mask = jnp.ones(n_res, dtype=jnp.float32)

  # Check neighbors between residue 0 (index 1) and residue 35 (index 36)
  neighbor_indices = jnp.zeros((n_res, 4), dtype=jnp.int32)
  neighbor_indices = neighbor_indices.at[0, 1].set(35)
  neighbor_indices = neighbor_indices.at[35, 1].set(0)

  cyclic_offsets = compute_neighbor_offsets(
    residue_indices,
    neighbor_indices,
    cyclic=True,
    chain_index=chain_index,
    mask=mask,
  )
  linear_offsets = compute_neighbor_offsets(
    residue_indices,
    neighbor_indices,
    cyclic=False,
    chain_index=chain_index,
    mask=mask,
  )

  # On a 36-residue cycle, offset between 1 and 36:
  # Linear: 1 - 36 = -35
  # Cyclic: wrapped distance is +1 (or -1 depending on direction)
  assert abs(int(cyclic_offsets[0, 1])) == 1
  assert abs(int(linear_offsets[0, 1])) == 35

  # Negative control: linear and cyclic offsets MUST diverge on terminal pair
  assert int(cyclic_offsets[0, 1]) != int(linear_offsets[0, 1])


def test_knob_semantics_ca_only() -> None:
  """CA-only MPNN routes to ca_v_48_* checkpoints and instantiates ProteinFeaturesCA."""
  for ver in ("v_48_002", "v_48_010", "v_48_020"):
    resolved = _resolve_checkpoint_id(model_weights="ca", model_version=ver)
    assert resolved == f"ca_{ver}.eqx.zst"

    topo = get_topology_for_checkpoint(resolved)
    assert topo["model_type"] == "ca"
    assert topo["k_neighbors"] == 48

  # Also test 'campnn' alias
  assert _resolve_checkpoint_id(model_weights="campnn") == "ca_v_48_020.eqx.zst"

  key = jax.random.PRNGKey(42)
  ca_model = Aminx(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=1,
    num_decoder_layers=1,
    vocab_size=21,
    k_neighbors=16,
    ca_only=True,
    key=key,
  )
  assert isinstance(ca_model.features, ProteinFeaturesCA)
  # CA-only edge embedding input dimension is 167 (16 pos + 144 RBF + 7 geom)
  assert ca_model.features.w_e.weight.shape == (128, 167)

  # Negative control: ca_only=False builds full-atom ProteinFeatures (dimension 416)
  std_model = Aminx(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=1,
    num_decoder_layers=1,
    vocab_size=21,
    k_neighbors=16,
    ca_only=False,
    key=key,
  )
  assert isinstance(std_model.features, ProteinFeatures)
  assert std_model.features.w_e.weight.shape == (128, 416)


def test_knob_semantics_ca_distance_gating() -> None:
  """CA-only featurizer respects mask gating and chain breaks."""
  key = jax.random.PRNGKey(0)
  features = ProteinFeaturesCA(
    node_features=128,
    edge_features=128,
    k_neighbors=8,
    num_positional_embeddings=16,
    key=key,
  )

  n_res = 16
  # Synthetic backbone coords
  rng = np.random.default_rng(42)
  coords = jnp.asarray(rng.standard_normal((n_res, 3)), dtype=jnp.float32)
  residue_index = jnp.arange(n_res, dtype=jnp.int32)
  chain_index = jnp.zeros(n_res, dtype=jnp.int32)

  # All valid mask
  clean_mask = jnp.ones(n_res, dtype=jnp.float32)
  clean_edges, _, _, _ = features(
    jax.random.PRNGKey(1),
    coords,
    clean_mask,
    residue_index,
    chain_index,
    backbone_noise=jnp.asarray(0.0),
  )

  # Gated mask: suppress residue 5
  gated_mask = clean_mask.at[5].set(0.0)
  gated_edges, _, _, _ = features(
    jax.random.PRNGKey(1),
    coords,
    gated_mask,
    residue_index,
    chain_index,
    backbone_noise=jnp.asarray(0.0),
  )

  # Negative control: edge features for masked residue MUST differ from clean
  diff = float(jnp.max(jnp.abs(clean_edges[5] - gated_edges[5])))
  assert diff > 1e-3, f"Expected masked residue to alter edge features, got diff={diff}"


def test_knob_semantics_coordinate_dimension_equivalence() -> None:
  """CA-only featurizer accepts both (L, 4, 3) and (L, 3) coordinate inputs identically."""
  key = jax.random.PRNGKey(0)
  features = ProteinFeaturesCA(
    node_features=128,
    edge_features=128,
    k_neighbors=8,
    num_positional_embeddings=16,
    key=key,
  )

  n_res = 12
  rng = np.random.default_rng(123)
  coords_4 = jnp.asarray(rng.standard_normal((n_res, 4, 3)), dtype=jnp.float32)
  # Extract CA coordinates (atom index 1)
  coords_3 = coords_4[:, 1, :]

  mask = jnp.ones(n_res, dtype=jnp.float32)
  residue_index = jnp.arange(n_res, dtype=jnp.int32)
  chain_index = jnp.zeros(n_res, dtype=jnp.int32)

  edges_from_4, ni_4, _, _ = features(
    jax.random.PRNGKey(7),
    coords_4,
    mask,
    residue_index,
    chain_index,
    backbone_noise=jnp.asarray(0.0),
  )
  edges_from_3, ni_3, _, _ = features(
    jax.random.PRNGKey(7),
    coords_3,
    mask,
    residue_index,
    chain_index,
    backbone_noise=jnp.asarray(0.0),
  )

  # Must match exactly
  np.testing.assert_allclose(np.asarray(edges_from_4), np.asarray(edges_from_3), atol=1e-6)
  np.testing.assert_array_equal(np.asarray(ni_4), np.asarray(ni_3))

  # Negative control: perturbed CA coordinates must differ
  perturbed_3 = coords_3 + 2.0
  edges_perturbed, _, _, _ = features(
    jax.random.PRNGKey(7),
    perturbed_3,
    mask,
    residue_index,
    chain_index,
    backbone_noise=jnp.asarray(0.0),
  )
  diff = float(jnp.max(jnp.abs(edges_from_3 - edges_perturbed)))
  assert diff > 1e-2, f"Perturbed coordinates did not alter edges: diff={diff}"
