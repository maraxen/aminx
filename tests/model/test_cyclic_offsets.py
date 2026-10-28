"""Unit tests for head-to-tail cyclic relative positional offset wrapping (Debt #3075)."""

from __future__ import annotations

import pytest

import jax
import jax.numpy as jnp

from aminx.model.features import ProteinFeatures, ProteinFeaturesCA
from aminx.model.mpnn import Aminx
from aminx.utils.graph import compute_neighbor_offsets


def test_compute_neighbor_offsets_cyclic_false_bit_identical() -> None:
  """Verify cyclic=False produces bit-identical offsets to default linear behavior."""
  res_idx = jnp.arange(10, dtype=jnp.int32)
  neighbors = jnp.broadcast_to(res_idx[None, :], (10, 10))
  chain_idx = jnp.zeros((10,), dtype=jnp.int32)
  mask = jnp.ones((10,), dtype=jnp.float32)

  linear_default = compute_neighbor_offsets(res_idx, neighbors)
  linear_explicit = compute_neighbor_offsets(
    res_idx, neighbors, cyclic=False, chain_index=chain_idx, mask=mask,
  )
  assert jnp.array_equal(linear_default, linear_explicit)


def test_compute_neighbor_offsets_cyclic_wrapping_and_antisymmetry() -> None:
  """Verify shortest signed circular distance and antisymmetric tie-breaking."""
  n_res = 10
  res_idx = jnp.arange(n_res, dtype=jnp.int32)
  neighbors = jnp.broadcast_to(res_idx[None, :], (n_res, n_res))
  chain_idx = jnp.zeros((n_res,), dtype=jnp.int32)
  mask = jnp.ones((n_res,), dtype=jnp.float32)

  offsets = compute_neighbor_offsets(
    res_idx, neighbors, cyclic=True, chain_index=chain_idx, mask=mask,
  )
  assert int(offsets[0, 9]) == 1
  assert int(offsets[9, 0]) == -1
  assert int(offsets[1, 8]) == 3
  assert int(offsets[8, 1]) == -3
  assert jnp.array_equal(offsets, -offsets.T)


def test_compute_neighbor_offsets_cyclic_multichain_and_padding() -> None:
  """Verify per-chain span L_c ignores padded residues and non-contiguous numbering gaps."""
  res_idx = jnp.array([101, 102, 103, 104, 105, 106, 0, 0, 1, 2, 3, 4], dtype=jnp.int32)
  chain_idx = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32)
  mask = jnp.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=jnp.float32)
  n_total = res_idx.shape[0]
  neighbors = jnp.broadcast_to(jnp.arange(n_total, dtype=jnp.int32)[None, :], (n_total, n_total))

  offsets = compute_neighbor_offsets(
    res_idx, neighbors, cyclic=True, chain_index=chain_idx, mask=mask,
  )
  assert int(offsets[0, 5]) == 1
  assert int(offsets[5, 0]) == -1
  assert int(offsets[8, 11]) == 1
  assert int(offsets[11, 8]) == -1

  # 1D per-residue cyclic mask: chain 0 cyclic, chain 1 linear
  cyclic_1d = chain_idx == 0
  offsets_mixed = compute_neighbor_offsets(
    res_idx, neighbors, cyclic=cyclic_1d, chain_index=chain_idx, mask=mask,
  )
  assert int(offsets_mixed[0, 5]) == 1
  assert int(offsets_mixed[5, 0]) == -1
  assert int(offsets_mixed[8, 11]) == -3
  assert int(offsets_mixed[11, 8]) == 3


def test_aminx_cyclic_features_and_protein_features_threading() -> None:
  """Verify ProteinFeatures, ProteinFeaturesCA, and Aminx thread cyclic flag cleanly."""
  key = jax.random.PRNGKey(0)
  n_res = 12
  coords = jax.random.normal(key, (n_res, 4, 3)) * 3.8
  mask = jnp.ones((n_res,), dtype=jnp.float32)
  res_idx = jnp.arange(n_res, dtype=jnp.int32)
  chain_idx = jnp.zeros((n_res,), dtype=jnp.int32)

  feat_bb = ProteinFeatures(
    edge_features=32,
    node_features=32,
    k_neighbors=8,
    key=key,
  )
  e_lin, _, _, _ = feat_bb(key, coords, mask, res_idx, chain_idx, 0.0, cyclic=False)
  e_cyc, _, _, _ = feat_bb(key, coords, mask, res_idx, chain_idx, 0.0, cyclic=True)
  assert e_lin.shape == e_cyc.shape
  assert not jnp.allclose(e_lin, e_cyc)

  feat_ca = ProteinFeaturesCA(
    edge_features=32,
    node_features=32,
    k_neighbors=8,
    key=key,
  )
  e_ca_lin, _, _, _ = feat_ca(key, coords, mask, res_idx, chain_idx, 0.0, cyclic=False)
  e_ca_cyc, _, _, _ = feat_ca(key, coords, mask, res_idx, chain_idx, 0.0, cyclic=True)
  assert e_ca_lin.shape == e_ca_cyc.shape
  assert not jnp.allclose(e_ca_lin, e_ca_cyc)

  model = Aminx(
    node_features=32,
    edge_features=32,
    hidden_features=32,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=8,
    key=key,
  )
  v_lin, e_m_lin, _ = model(
    coords, mask, res_idx, chain_idx, prng_key=key, backbone_noise=0.0, cyclic=False,
  )
  v_cyc, e_m_cyc, _ = model(
    coords, mask, res_idx, chain_idx, prng_key=key, backbone_noise=0.0, cyclic=True,
  )
  assert v_lin.shape == (n_res, 32)
  assert not jnp.allclose(v_lin, v_cyc)
  assert not jnp.allclose(e_m_lin, e_m_cyc)


@pytest.mark.parametrize("length", [8, 10])
def test_cyclic_neighbor_offsets_even_length_antipodal(length: int):
  """Verify even-length cyclic chains (L=8, 10) wrap cleanly including antipodal L//2."""
  res_idx = jnp.arange(length, dtype=jnp.int32)
  chain_idx = jnp.zeros((length,), dtype=jnp.int32)
  e_idx = jnp.broadcast_to(res_idx[None, :], (length, length))
  cyclic_mask = jnp.ones((length,), dtype=jnp.int32)

  offsets = compute_neighbor_offsets(
    res_idx,
    e_idx,
    chain_index=chain_idx,
    cyclic=cyclic_mask,
  )
  assert offsets.shape == (length, length)
  assert int(offsets[0, length - 1]) == 1
  assert int(offsets[length - 1, 0]) == -1
  half = length // 2
  assert abs(int(offsets[0, half])) == half


@pytest.mark.parametrize(
  ("checkpoint_id", "expected_cyclic"),
  [("cyclicmpnn", True), ("cyclicmpnn_v_48_010.eqx.zst", True), ("antibmpnn", False)],
)
def test_load_model_sets_static_cyclic_flag_from_family(
  monkeypatch: pytest.MonkeyPatch, checkpoint_id: str, expected_cyclic: bool
) -> None:
  """Verify load_model builds the skeleton with cyclic set from the checkpoint family."""
  from aminx.io import weights as weights_mod

  captured: dict[str, object] = {}

  def fake_load_weights(*, checkpoint_id: str, local_path: str | None, skeleton: object) -> object:
    captured["skeleton"] = skeleton
    return skeleton

  monkeypatch.setattr(weights_mod, "load_weights", fake_load_weights)
  weights_mod.load_model(checkpoint_id)

  assert captured["skeleton"].cyclic is expected_cyclic
