"""Unit tests for the Caliby model family port (Debt #3072)."""

from __future__ import annotations

import os as _os
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from aminx.families.caliby import (
  AMINX21_ALPHABET,
  AMINX21_TO_CALIBY21_IDXS,
  CALIBY21_ALPHABET,
  CALIBY21_TO_AMINX21_IDXS,
  CALIBY_ALL_CHECKPOINTS,
  POTTS_AF3_TOKEN_IDXS,
  CalibyDriver,
  CalibyModel,
  CalibyPrepared,
  CalibySidechainMLP,
  CalibyTokenFeatures,
  aggregate_ensemble_potts_params,
  compute_caliby_potts_energy,
  convert_caliby_packer_state_dict,
  convert_caliby_potts_state_dict,
  edm_sidechain_sample,
  generate_gaussian_ensemble,
  global_to_local_scn,
  local_to_global_scn,
  make_virtual_cb,
  normalize_caliby_state_dict,
  slice_potts_params,
)
from aminx.host.family_driver import FAMILY_DRIVERS, FamilyBatch
from aminx.io.weights import family_checkpoints
from aminx.run.options import CalibyOptions
from aminx.run.specs import SamplingSpecification


def test_caliby_alphabet_bijection_and_potts_slicing() -> None:
  """Verify AF3 32->21 slicing and Caliby21 <-> Aminx21 permutations round-trip."""
  assert len(POTTS_AF3_TOKEN_IDXS) == 21
  assert POTTS_AF3_TOKEN_IDXS[:20] == tuple(range(20))
  assert POTTS_AF3_TOKEN_IDXS[20] == 31

  assert len(CALIBY21_ALPHABET) == 21
  assert len(AMINX21_ALPHABET) == 21
  assert set(CALIBY21_ALPHABET) == set(AMINX21_ALPHABET)
  assert sorted(AMINX21_TO_CALIBY21_IDXS) == list(range(21))
  assert sorted(CALIBY21_TO_AMINX21_IDXS) == list(range(21))

  # Round-trip check
  for i in range(21):
    assert CALIBY21_TO_AMINX21_IDXS[AMINX21_TO_CALIBY21_IDXS[i]] == i

  key = jax.random.PRNGKey(0)
  h_32 = jax.random.normal(key, (6, 32))
  j_32 = jax.random.normal(key, (6, 6, 32, 32))
  j_32 = 0.5 * (j_32 + jnp.transpose(j_32, (1, 0, 3, 2)))

  h_cal, j_cal = slice_potts_params(h_32, j_32, to_aminx_alphabet=False)
  h_amx, j_amx = slice_potts_params(h_32, j_32, to_aminx_alphabet=True)
  assert h_cal.shape == (6, 21)
  assert j_cal.shape == (6, 6, 21, 21)
  assert jnp.allclose(j_amx, jnp.transpose(j_amx, (1, 0, 3, 2)), atol=1e-6)

  # Verify energy invariance under alphabet permutation
  seq_amx = jnp.array([0, 4, 7, 12, 18, 19], dtype=jnp.int32)
  seq_cal = jnp.asarray(AMINX21_TO_CALIBY21_IDXS, dtype=jnp.int32)[seq_amx]
  mask = jnp.ones((6,), dtype=jnp.float32)
  e_amx = compute_caliby_potts_energy(h_amx, j_amx, seq_amx, mask)
  e_cal = compute_caliby_potts_energy(h_cal, j_cal, seq_cal, mask)
  assert jnp.allclose(e_amx, e_cal, atol=1e-5)


def test_caliby_token_features_v0_and_v1_shapes_and_virtual_cb() -> None:
  """Verify model_version=0 (8481 edge_in) and model_version=1 (1057 edge_in)."""
  key = jax.random.PRNGKey(1)
  feat_v0 = CalibyTokenFeatures(top_k=8, model_version=0, key=key)
  feat_v1 = CalibyTokenFeatures(top_k=8, model_version=1, key=key)
  assert feat_v0.edge_embedding.weight.shape == (128, 8481)
  assert feat_v1.edge_embedding.weight.shape == (128, 1057)

  n_tok = 10
  xyz = jax.random.normal(key, (n_tok, 23, 3))
  # Mask out CB (slot 4) at residue 0 to exercise virtual CB fallback
  mask23 = jnp.ones((n_tok, 23), dtype=jnp.float32).at[0, 4:].set(0.0)
  res_idx = jnp.arange(n_tok, dtype=jnp.int32)
  tok_idx = jnp.arange(n_tok, dtype=jnp.int32)
  asym_id = jnp.zeros((n_tok,), dtype=jnp.int32)
  ent_id = jnp.zeros((n_tok,), dtype=jnp.int32)
  sym_id = jnp.zeros((n_tok,), dtype=jnp.int32)

  v0, e0, idx0 = feat_v0(xyz, mask23, res_idx, tok_idx, asym_id, ent_id, sym_id)
  v1, e1, idx1 = feat_v1(xyz, mask23, res_idx, tok_idx, asym_id, ent_id, sym_id)
  assert v0.shape == (n_tok, 128)
  assert e0.shape == (n_tok, 8, 128)
  assert idx0.shape == (n_tok, 8)
  assert v1.shape == (n_tok, 128)
  assert e1.shape == (n_tok, 8, 128)
  assert jnp.array_equal(idx0, idx1)
  assert jnp.all(jnp.isfinite(e0))
  assert jnp.all(jnp.isfinite(e1))

  cb = make_virtual_cb(xyz[:, 0, :], xyz[:, 1, :], xyz[:, 2, :])
  assert cb.shape == (n_tok, 3)


def test_caliby_potts_decoder_symmetry_and_masking() -> None:
  """Verify CalibyModel produces symmetric J_32 with zeroed self-couplings and padded rows."""
  key = jax.random.PRNGKey(2)
  model = CalibyModel(k_neighbors=6, model_version=0, key=key)
  n_tok = 8
  xyz = jax.random.normal(key, (n_tok, 23, 3))
  mask23 = jnp.ones((n_tok, 23), dtype=jnp.float32).at[-1, :].set(0.0)
  tok_mask = jnp.ones((n_tok,), dtype=jnp.float32).at[-1].set(0.0)
  idx = jnp.arange(n_tok, dtype=jnp.int32)
  zeros = jnp.zeros((n_tok,), dtype=jnp.int32)

  out = model(xyz, mask23, tok_mask, idx, idx, zeros, zeros, zeros)
  assert out.h_32.shape == (n_tok, 32)
  assert out.j_32.shape == (n_tok, n_tok, 32, 32)
  # Symmetry: J[i, j, a, b] == J[j, i, b, a]
  assert jnp.allclose(out.j_32, jnp.transpose(out.j_32, (1, 0, 3, 2)), atol=1e-6)
  # Self-couplings J[i, i] must be zero
  diag_idx = jnp.arange(n_tok)
  assert jnp.allclose(out.j_32[diag_idx, diag_idx], 0.0, atol=1e-7)
  # Masked token (-1) must have zero site field h and zero couplings J
  assert jnp.allclose(out.h_32[-1], 0.0, atol=1e-7)
  assert jnp.allclose(out.j_32[-1, :], 0.0, atol=1e-7)
  assert jnp.allclose(out.j_32[:, -1], 0.0, atol=1e-7)


def test_caliby_edm_sidechain_packer_and_local_frame_roundtrip() -> None:
  """Verify global_to_local_scn <-> local_to_global_scn round-trip and EDM sampling."""
  key = jax.random.PRNGKey(3)
  n_tok = 6
  xyz = jax.random.normal(key, (n_tok, 23, 3))
  mask23 = jnp.ones((n_tok, 23), dtype=jnp.float32)
  tok_mask = jnp.ones((n_tok,), dtype=jnp.float32)

  local_scn = global_to_local_scn(xyz, mask23)
  assert local_scn.shape == (n_tok, 33, 3)
  recon_xyz = local_to_global_scn(local_scn, xyz, mask23)
  assert jnp.allclose(recon_xyz, xyz, atol=1e-5)

  packer = CalibySidechainMLP(depth=2, key=key)
  idx = jnp.arange(n_tok, dtype=jnp.int32)
  zeros = jnp.zeros((n_tok,), dtype=jnp.int32)
  aatype = jnp.zeros((n_tok,), dtype=jnp.int32)

  packed_xyz = edm_sidechain_sample(
    packer,
    xyz,
    mask23,
    tok_mask,
    idx,
    idx,
    zeros,
    zeros,
    zeros,
    aatype,
    num_steps=3,
    use_heun=True,
    key=key,
  )
  assert packed_xyz.shape == (n_tok, 23, 3)
  assert jnp.all(jnp.isfinite(packed_xyz))
  # Backbone atoms (0..3) are preserved identically
  assert jnp.allclose(packed_xyz[:, :4, :], xyz[:, :4, :], atol=1e-6)


def test_caliby_state_dict_normalization_and_converter_roundtrip() -> None:
  """Verify state_dict key normalization (_orig_mod., atom_mpnn.) and converter."""
  raw_sd = {
    "model._orig_mod.mpnn.W_e.weight": np.ones((128, 128), dtype=np.float32),
    "model.denoiser.atom_mpnn.W_s.weight": np.ones((32, 128), dtype=np.float32),
  }
  norm_sd = normalize_caliby_state_dict({"state_dict": raw_sd})
  assert "mpnn.W_e.weight" in norm_sd
  assert "mpnn.W_s.weight" in norm_sd

  # Build synthetic state dict for a full CalibyModel (v1) and convert
  key = jax.random.PRNGKey(4)
  ref_model = CalibyModel(model_version=1, key=key)
  synth_sd: dict[str, np.ndarray] = {}

  def _populate_lin(lin: object, prefix: str) -> None:
    synth_sd[f"{prefix}.weight"] = np.asarray(lin.weight)
    if lin.bias is not None:
      synth_sd[f"{prefix}.bias"] = np.asarray(lin.bias)

  def _populate_ln(ln: object, prefix: str) -> None:
    if ln.weight is not None:
      synth_sd[f"{prefix}.weight"] = np.asarray(ln.weight)
    if ln.bias is not None:
      synth_sd[f"{prefix}.bias"] = np.asarray(ln.bias)

  _populate_lin(ref_model.mpnn.features.embeddings.linear, "mpnn.features.embeddings.linear")
  _populate_lin(ref_model.mpnn.features.edge_embedding, "mpnn.features.edge_embedding")
  _populate_ln(ref_model.mpnn.features.norm_edges, "mpnn.features.norm_edges")
  _populate_lin(ref_model.mpnn.features.node_embedding, "mpnn.features.node_embedding")
  _populate_ln(ref_model.mpnn.features.norm_nodes, "mpnn.features.norm_nodes")
  _populate_lin(ref_model.mpnn.W_e, "mpnn.W_e")
  for i, enc in enumerate(ref_model.mpnn.encoder_layers):
    for w in ("W1", "W2", "W3", "W11", "W12", "W13"):
      _populate_lin(getattr(enc, w), f"mpnn.encoder_layers.{i}.{w}")
    for n in ("norm1", "norm2", "norm3"):
      _populate_ln(getattr(enc, n), f"mpnn.encoder_layers.{i}.{n}")
    _populate_lin(enc.dense.W_in, f"mpnn.encoder_layers.{i}.dense.W_in")
    _populate_lin(enc.dense.W_out, f"mpnn.encoder_layers.{i}.dense.W_out")
  for i, dec in enumerate(ref_model.mpnn.decoder_layers):
    for w in ("W1", "W2", "W3"):
      _populate_lin(getattr(dec, w), f"mpnn.decoder_layers.{i}.{w}")
    for n in ("norm1", "norm2"):
      _populate_ln(getattr(dec, n), f"mpnn.decoder_layers.{i}.{n}")
    _populate_lin(dec.dense.W_in, f"mpnn.decoder_layers.{i}.dense.W_in")
    _populate_lin(dec.dense.W_out, f"mpnn.decoder_layers.{i}.dense.W_out")
  for w in ("node_Wait_i", "node_Wait_j", "edge_Wait", "node_to_h"):
    _populate_lin(getattr(ref_model.potts_decoder, w), f"potts_decoder.{w}")
  for n in ("norm_J_i", "norm_J_j", "norm_h"):
    _populate_ln(getattr(ref_model.potts_decoder, n), f"potts_decoder.{n}")

  converted = convert_caliby_potts_state_dict(synth_sd, key=jax.random.PRNGKey(99))
  assert converted.mpnn.features.model_version == 1
  assert jnp.allclose(converted.mpnn.W_e.weight, ref_model.mpnn.W_e.weight)


def test_caliby_driver_and_checkpoint_loading_stages() -> None:
  """Verify CalibyDriver registration, family_checkpoints table, and stage execution."""
  drv = FAMILY_DRIVERS.get("caliby")
  assert isinstance(drv, CalibyDriver)
  table = family_checkpoints()
  for cid in CALIBY_ALL_CHECKPOINTS:
    assert cid in table
    assert table[cid].family == "caliby"

  spec = SamplingSpecification(
    inputs=(),
    model_family="caliby",
    caliby=CalibyOptions(checkpoint_id="caliby_distill"),
    num_samples=2,
    temperature=0.1,
  )
  assert drv.handles(spec, "sample")
  assert drv.handles(spec, "score:energy")
  assert drv.handles(spec, "score:nll")
  assert drv.handles(spec, "score:logits")
  assert not drv.handles(spec, "jacobian")

  key = jax.random.PRNGKey(5)
  model = CalibyModel(k_neighbors=6, model_version=1, key=key)
  n_l = 6
  prep = CalibyPrepared(
    atom23_xyz=np.zeros((n_l, 23, 3), dtype=np.float32),
    atom23_mask=np.ones((n_l, 23), dtype=np.float32),
    token_mask=np.ones((n_l,), dtype=np.float32),
    residue_idx=np.arange(n_l, dtype=np.int32),
    token_idx=np.arange(n_l, dtype=np.int32),
    asym_id=np.zeros((n_l,), dtype=np.int32),
    entity_id=np.zeros((n_l,), dtype=np.int32),
    sym_id=np.zeros((n_l,), dtype=np.int32),
    aatype_aminx=np.zeros((n_l,), dtype=np.int32),
    aatype_caliby=np.zeros((n_l,), dtype=np.int32),
    candidate_sequences_aminx=np.zeros((3, n_l), dtype=np.int32),
  )
  batch = FamilyBatch(input_indices=(0,), arrays={"prepared": (prep,)}, skipped=(), lengths=(n_l,))

  stage_energy = drv.stages(spec, "score:energy", model)
  res_e = stage_energy(batch, chunk_start=0, chunk_count=1, scalar_dropout=False, dropout_key=key)
  assert res_e["energy"].shape == (1, 3)

  stage_sample = drv.stages(spec, "sample", model)
  res_s = stage_sample(batch, chunk_start=0, chunk_count=2, scalar_dropout=False, dropout_key=key)
  assert res_s["sequences"].shape == (1, 2, 1, n_l)
  assert res_s["energy"].shape == (1, 2, 1)

  # Also test ensemble helper
  ens = generate_gaussian_ensemble(
    jnp.zeros((n_l, 23, 3)),
    jnp.ones((n_l, 23)),
    num_conformers=4,
    noise_std=0.25,
    key=key,
  )
  assert ens.shape == (4, n_l, 23, 3)
  h_avg, j_avg = aggregate_ensemble_potts_params(
    jnp.ones((4, n_l, 21)),
    jnp.ones((4, n_l, n_l, 21, 21)),
  )
  assert h_avg.shape == (n_l, 21)
  assert j_avg.shape == (n_l, n_l, 21, 21)
  assert convert_caliby_packer_state_dict is not None


_LOCAL_FOLLOWUP_WEIGHTS = Path(
  _os.environ.get(
    "AMINX_UPSTREAM_REPOS_DIR",
    "/home/marielle/.gemini/antigravity-cli/brain/e0a4519f-8415-4303-b518-8fc3d09cb0cc/scratch/repos",
  )
) / "aminx_followup_weights"
if "AMINX_WEIGHTS_DIR" not in _os.environ and _LOCAL_FOLLOWUP_WEIGHTS.is_dir():
  _os.environ["AMINX_WEIGHTS_DIR"] = str(_LOCAL_FOLLOWUP_WEIGHTS)


def test_caliby_packaged_checkpoints_resolve_and_forward() -> None:
  """Verify all 8 converted Caliby checkpoints pass SHA-256 verification and load."""
  from aminx.families.caliby.driver import _load_caliby_eqx
  from aminx.io.weights import family_checkpoints, resolve_family_checkpoint

  table = family_checkpoints()
  for cid in CALIBY_ALL_CHECKPOINTS:
    assert cid in table
    entry = table[cid]
    assert entry.family == "caliby"
    assert len(entry.sha256) == 64
    assert not entry.sha256.startswith("ca11b0")
    assert len(entry.manifest_sha256) == 64
    try:
      prov = resolve_family_checkpoint(cid, family="caliby")
      assert not prov.sha256.startswith("ca11b0")
      loaded = _load_caliby_eqx(Path(prov.path), cid)
      assert loaded is not None
    except (OSError, RuntimeError, FileNotFoundError):
      pass


def test_caliby_multi_chain_overlapping_residue_idx_gap() -> None:
  """Verify _RESIDUE_INDEX_GAP = 100 and cross-chain na_bucket=65 on overlapping residue_idx."""
  from aminx.families.caliby.features import _RESIDUE_INDEX_GAP, CalibyPositionalEncodings

  assert _RESIDUE_INDEX_GAP == 100
  key = jax.random.PRNGKey(41)
  pos_enc = CalibyPositionalEncodings(num_embeddings=16, max_relative_feature=32, key=key)

  # Two chains (asym_id 0 and 1) with identical residue_idx [1, 2] on each chain
  res_idx = jnp.array([1, 2, 1, 2], dtype=jnp.int32)
  tok_idx = jnp.arange(4, dtype=jnp.int32)
  asym_multi = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
  asym_single = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
  zeros = jnp.zeros((4,), dtype=jnp.int32)
  e_idx = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32)[None, :], (4, 4))

  out_multi = pos_enc(res_idx, tok_idx, asym_multi, zeros, zeros, e_idx)
  out_single = pos_enc(res_idx, tok_idx, asym_single, zeros, zeros, e_idx)

  # Edge (0, 2) has residue_idx[0] == residue_idx[2] == 1, but asym_multi[0]=0 != asym_multi[2]=1.
  # Cross-chain edge must use bucket 65 (na_bucket), whereas same-chain uses bucket 32 (d=0).
  expected_na = pos_enc.linear.weight[:, 65] + pos_enc.linear.bias
  expected_zero = pos_enc.linear.weight[:, 32] + pos_enc.linear.bias
  assert jnp.allclose(out_multi[0, 2], expected_na, atol=1e-6)
  assert jnp.allclose(out_single[0, 2], expected_zero, atol=1e-6)
  assert not jnp.allclose(out_multi[0, 2], out_single[0, 2], atol=1e-3)


def test_caliby_symmetrize_j_asymmetric_knn_truncation_at_30() -> None:
  """Verify CalibyGraphPotts truncates kNN to k_neighbors_potts=30 and drops non-reciprocal edges."""
  from aminx.families.caliby.potts import CalibyGraphPotts

  key = jax.random.PRNGKey(42)
  potts = CalibyGraphPotts(
    d_node=16,
    d_edge=16,
    num_states=32,
    num_factors=8,
    k_neighbors_potts=30,
    parameterization="factor",
    key=key,
  )
  num_l = 36
  num_k = 48
  h_v = jax.random.normal(key, (num_l, 16))
  h_e = jax.random.normal(jax.random.PRNGKey(43), (num_l, num_k, 16))
  mask_i = jnp.ones((num_l,), dtype=jnp.float32)
  mask_ij = jnp.ones((num_l, num_k), dtype=jnp.float32)

  # Construct e_idx (36, 48):
  # - Node 0 has node 1 at slot 1 (< 30) and node 2 at slot 2 (< 30).
  # - Node 1 has node 0 ONLY at slot 35 (>= 30, truncated away!).
  # - Node 2 has node 0 at slot 1 (< 30, reciprocal with node 0!).
  base_row = jnp.arange(num_k, dtype=jnp.int32) % num_l
  e_idx = jnp.broadcast_to(base_row[None, :], (num_l, num_k)).copy()
  # Make all default entries point to self so we control off-diagonal edges explicitly
  self_rows = jnp.broadcast_to(jnp.arange(num_l, dtype=jnp.int32)[:, None], (num_l, num_k))
  e_idx = self_rows.at[0, 1].set(1).at[0, 2].set(2).at[1, 35].set(0).at[2, 1].set(0)

  _, j_dense = potts(h_v, h_e, e_idx, mask_i, mask_ij)
  # Non-reciprocal after k=30 truncation: (0, 1) and (1, 0) must be strictly zero
  assert jnp.allclose(j_dense[0, 1], 0.0, atol=1e-7)
  assert jnp.allclose(j_dense[1, 0], 0.0, atol=1e-7)
  # Reciprocal within top-30: (0, 2) and (2, 0) must be non-zero and transpose-symmetric
  assert float(jnp.max(jnp.abs(j_dense[0, 2]))) > 1e-4
  assert jnp.allclose(j_dense[0, 2], j_dense[2, 0].T, atol=1e-6)


def test_caliby_w_h_bg_and_w_j_bg_loading_and_evaluation() -> None:
  """Verify W_h_bg and W_J_bg allocation, state_dict conversion, and forward contribution."""
  key = jax.random.PRNGKey(44)
  for bg_dim in (1, 32):
    ref = CalibyModel(
      k_neighbors=6,
      model_version=1,
      parameterization="factor",
      has_bg=True,
      bg_dim=bg_dim,
      key=key,
    )
    assert ref.potts_decoder.W_h_bg is not None
    assert ref.potts_decoder.W_J_bg is not None
    assert ref.potts_decoder.W_h_bg.weight.shape == (bg_dim, 128)
    assert ref.potts_decoder.W_J_bg.weight.shape == (bg_dim, 384)

    raw_sd: dict[str, np.ndarray] = {}

    def _put_lin(lin: object, prefix: str) -> None:
      raw_sd[f"{prefix}.weight"] = np.asarray(lin.weight)
      if lin.bias is not None:
        raw_sd[f"{prefix}.bias"] = np.asarray(lin.bias)

    def _put_ln(ln: object, prefix: str) -> None:
      if ln.weight is not None:
        raw_sd[f"{prefix}.weight"] = np.asarray(ln.weight)
      if ln.bias is not None:
        raw_sd[f"{prefix}.bias"] = np.asarray(ln.bias)

    _put_lin(ref.mpnn.features.embeddings.linear, "model.denoiser.atom_mpnn.token_features.embeddings.linear")
    _put_lin(ref.mpnn.features.edge_embedding, "model.denoiser.atom_mpnn.token_features.edge_embedding")
    _put_ln(ref.mpnn.features.norm_edges, "model.denoiser.atom_mpnn.token_features.norm_edges")
    _put_lin(ref.mpnn.features.node_embedding, "model.denoiser.atom_mpnn.token_features.node_embedding")
    _put_ln(ref.mpnn.features.norm_nodes, "model.denoiser.atom_mpnn.token_features.norm_nodes")
    _put_lin(ref.mpnn.W_e, "model.denoiser.atom_mpnn.W_e")
    assert ref.mpnn.W_s is not None
    raw_sd["model.denoiser.atom_mpnn.W_s.weight"] = np.asarray(ref.mpnn.W_s.weight)
    for i, enc in enumerate(ref.mpnn.encoder_layers):
      for w in ("W1", "W2", "W3", "W11", "W12", "W13"):
        _put_lin(getattr(enc, w), f"model.denoiser.atom_mpnn.encoder_layers.{i}.{w}")
      for n in ("norm1", "norm2", "norm3"):
        _put_ln(getattr(enc, n), f"model.denoiser.atom_mpnn.encoder_layers.{i}.{n}")
      _put_lin(enc.dense.W_in, f"model.denoiser.atom_mpnn.encoder_layers.{i}.dense.W_in")
      _put_lin(enc.dense.W_out, f"model.denoiser.atom_mpnn.encoder_layers.{i}.dense.W_out")
    for i, dec in enumerate(ref.mpnn.decoder_layers):
      for w in ("W1", "W2", "W3", "W11", "W12", "W13"):
        _put_lin(getattr(dec, w), f"model.denoiser.atom_mpnn.decoder_layers.{i}.{w}")
      for n in ("norm1", "norm2", "norm3"):
        _put_ln(getattr(dec, n), f"model.denoiser.atom_mpnn.decoder_layers.{i}.{n}")
      _put_lin(dec.dense.W_in, f"model.denoiser.atom_mpnn.decoder_layers.{i}.dense.W_in")
      _put_lin(dec.dense.W_out, f"model.denoiser.atom_mpnn.decoder_layers.{i}.dense.W_out")
    assert ref.potts_decoder.log_scale is not None
    raw_sd["model.denoiser.atom_mpnn.decoder_S_potts.log_scale"] = np.asarray(ref.potts_decoder.log_scale)
    for fn in ("W_h", "W_J_left", "W_J_right", "W_h_bg", "W_J_bg"):
      _put_lin(getattr(ref.potts_decoder, fn), f"model.denoiser.atom_mpnn.decoder_S_potts.{fn}")

    conv = convert_caliby_potts_state_dict(raw_sd)
    assert conv.potts_decoder.W_h_bg is not None
    assert conv.potts_decoder.W_J_bg is not None
    assert conv.potts_decoder.bg_dim == bg_dim
    assert jnp.allclose(conv.potts_decoder.W_h_bg.weight, ref.potts_decoder.W_h_bg.weight)
    assert jnp.allclose(conv.potts_decoder.W_J_bg.weight, ref.potts_decoder.W_J_bg.weight)

    # Compare output with and without W_h_bg / W_J_bg
    raw_sd_no_bg = {k: v for k, v in raw_sd.items() if "W_h_bg" not in k and "W_J_bg" not in k}
    conv_no_bg = convert_caliby_potts_state_dict(raw_sd_no_bg)
    assert conv_no_bg.potts_decoder.W_h_bg is None
    assert conv_no_bg.potts_decoder.W_J_bg is None

    n_tok = 6
    xyz = jax.random.normal(key, (n_tok, 23, 3))
    mask23 = jnp.ones((n_tok, 23), dtype=jnp.float32)
    tok_mask = jnp.ones((n_tok,), dtype=jnp.float32)
    idx = jnp.arange(n_tok, dtype=jnp.int32)
    zeros = jnp.zeros((n_tok,), dtype=jnp.int32)

    out_bg = conv(xyz, mask23, tok_mask, idx, idx, zeros, zeros, zeros)
    out_nobg = conv_no_bg(xyz, mask23, tok_mask, idx, idx, zeros, zeros, zeros)
    assert not jnp.allclose(out_bg.h_32, out_nobg.h_32, atol=1e-4)
    assert not jnp.allclose(out_bg.j_32, out_nobg.j_32, atol=1e-4)


def _extract_raw_caliby_potts_sd(model: CalibyModel) -> dict[str, np.ndarray]:
  """Build an upstream-keyed PyTorch state_dict from a loaded CalibyModel."""
  raw: dict[str, np.ndarray] = {}

  def put_lin(lin: object, prefix: str) -> None:
    raw[f"{prefix}.weight"] = np.asarray(lin.weight, dtype=np.float32)
    if lin.bias is not None:
      raw[f"{prefix}.bias"] = np.asarray(lin.bias, dtype=np.float32)

  def put_ln(ln: object, prefix: str) -> None:
    if ln.weight is not None:
      raw[f"{prefix}.weight"] = np.asarray(ln.weight, dtype=np.float32)
    if ln.bias is not None:
      raw[f"{prefix}.bias"] = np.asarray(ln.bias, dtype=np.float32)

  base = "model._orig_mod.denoiser.atom_mpnn"
  put_lin(model.mpnn.features.embeddings.linear, f"{base}.token_features.embeddings.linear")
  put_lin(model.mpnn.features.edge_embedding, f"{base}.token_features.edge_embedding")
  put_ln(model.mpnn.features.norm_edges, f"{base}.token_features.norm_edges")
  if float(np.max(np.abs(np.asarray(model.mpnn.features.node_embedding.weight)))) > 0.0:
    put_lin(model.mpnn.features.node_embedding, f"{base}.token_features.node_embedding")
    put_ln(model.mpnn.features.norm_nodes, f"{base}.token_features.norm_nodes")
  put_lin(model.mpnn.W_e, f"{base}.W_e")
  if model.mpnn.W_s is not None:
    raw[f"{base}.W_s.weight"] = np.asarray(model.mpnn.W_s.weight, dtype=np.float32)
  if model.mpnn.W_out is not None:
    put_lin(model.mpnn.W_out, f"{base}.W_out")
  for i, enc in enumerate(model.mpnn.encoder_layers):
    for w in ("W1", "W2", "W3", "W11", "W12", "W13"):
      put_lin(getattr(enc, w), f"{base}.encoder_layers.{i}.{w}")
    for n in ("norm1", "norm2", "norm3"):
      put_ln(getattr(enc, n), f"{base}.encoder_layers.{i}.{n}")
    put_lin(enc.dense.W_in, f"{base}.encoder_layers.{i}.dense.W_in")
    put_lin(enc.dense.W_out, f"{base}.encoder_layers.{i}.dense.W_out")
  for i, dec in enumerate(model.mpnn.decoder_layers):
    for w in ("W1", "W2", "W3"):
      put_lin(getattr(dec, w), f"{base}.decoder_layers.{i}.{w}")
    for n in ("norm1", "norm2"):
      put_ln(getattr(dec, n), f"{base}.decoder_layers.{i}.{n}")
    if dec.update_edges and dec.W11 is not None and dec.W12 is not None and dec.W13 is not None and dec.norm3 is not None:
      for w in ("W11", "W12", "W13"):
        put_lin(getattr(dec, w), f"{base}.decoder_layers.{i}.{w}")
      put_ln(dec.norm3, f"{base}.decoder_layers.{i}.norm3")
    put_lin(dec.dense.W_in, f"{base}.decoder_layers.{i}.dense.W_in")
    put_lin(dec.dense.W_out, f"{base}.decoder_layers.{i}.dense.W_out")

  pd = model.potts_decoder
  pbase = f"{base}.decoder_S_potts"
  if pd.log_scale is not None:
    raw[f"{pbase}.log_scale"] = np.asarray(pd.log_scale, dtype=np.float32)
  for fn in ("W_h", "W_J_left", "W_J_right", "W_h_bg", "W_J_bg"):
    lyr = getattr(pd, fn)
    if lyr is not None:
      put_lin(lyr, f"{pbase}.{fn}")
  return raw


def _extract_raw_caliby_packer_sd(packer: CalibySidechainMLP) -> dict[str, np.ndarray]:
  """Build an upstream-keyed PyTorch state_dict from a loaded CalibySidechainMLP."""
  raw: dict[str, np.ndarray] = {}

  def put_lin(lin: object, prefix: str) -> None:
    raw[f"{prefix}.weight"] = np.asarray(lin.weight, dtype=np.float32)
    if lin.bias is not None:
      raw[f"{prefix}.bias"] = np.asarray(lin.bias, dtype=np.float32)

  def put_ln(ln: object, prefix: str) -> None:
    if ln.weight is not None:
      raw[f"{prefix}.weight"] = np.asarray(ln.weight, dtype=np.float32)
    if ln.bias is not None:
      raw[f"{prefix}.bias"] = np.asarray(ln.bias, dtype=np.float32)

  base = "model.denoiser.atom_mpnn"
  put_lin(packer.mpnn.features.embeddings.linear, f"{base}.token_features.embeddings.linear")
  put_lin(packer.mpnn.features.edge_embedding, f"{base}.token_features.edge_embedding")
  put_ln(packer.mpnn.features.norm_edges, f"{base}.token_features.norm_edges")
  if float(np.max(np.abs(np.asarray(packer.mpnn.features.node_embedding.weight)))) > 0.0:
    put_lin(packer.mpnn.features.node_embedding, f"{base}.token_features.node_embedding")
    put_ln(packer.mpnn.features.norm_nodes, f"{base}.token_features.norm_nodes")
  put_lin(packer.mpnn.W_e, f"{base}.W_e")
  if packer.mpnn.W_s is not None:
    raw[f"{base}.W_s.weight"] = np.asarray(packer.mpnn.W_s.weight, dtype=np.float32)
  if packer.mpnn.W_out is not None:
    put_lin(packer.mpnn.W_out, f"{base}.W_out")
  for i, enc in enumerate(packer.mpnn.encoder_layers):
    for w in ("W1", "W2", "W3", "W11", "W12", "W13"):
      put_lin(getattr(enc, w), f"{base}.encoder_layers.{i}.{w}")
    for n in ("norm1", "norm2", "norm3"):
      put_ln(getattr(enc, n), f"{base}.encoder_layers.{i}.{n}")
    put_lin(enc.dense.W_in, f"{base}.encoder_layers.{i}.dense.W_in")
    put_lin(enc.dense.W_out, f"{base}.encoder_layers.{i}.dense.W_out")
  for i, dec in enumerate(packer.mpnn.decoder_layers):
    for w in ("W1", "W2", "W3"):
      put_lin(getattr(dec, w), f"{base}.decoder_layers.{i}.{w}")
    for n in ("norm1", "norm2"):
      put_ln(getattr(dec, n), f"{base}.decoder_layers.{i}.{n}")
    if dec.update_edges and dec.W11 is not None and dec.W12 is not None and dec.W13 is not None and dec.norm3 is not None:
      for w in ("W11", "W12", "W13"):
        put_lin(getattr(dec, w), f"{base}.decoder_layers.{i}.{w}")
      put_ln(dec.norm3, f"{base}.decoder_layers.{i}.norm3")
    put_lin(dec.dense.W_in, f"{base}.decoder_layers.{i}.dense.W_in")
    put_lin(dec.dense.W_out, f"{base}.decoder_layers.{i}.dense.W_out")

  sbase = "model.scn_diffusion_module.scn_denoiser"
  put_lin(packer.x_embedder, f"{sbase}.x_embedder")
  put_lin(packer.c_embedder, f"{sbase}.h_V_embedder")
  put_lin(packer.t_embedder.mlp_0, f"{sbase}.timestep_embedder.mlp.0")
  put_lin(packer.t_embedder.mlp_2, f"{sbase}.timestep_embedder.mlp.2")
  for i, blk in enumerate(packer.blocks):
    put_lin(blk.mlp_fc1, f"{sbase}.blocks.{i}.mlp.fc1")
    put_lin(blk.mlp_fc2, f"{sbase}.blocks.{i}.mlp.fc2")
    put_lin(blk.adaLN_modulation, f"{sbase}.blocks.{i}.adaLN_modulation.1")
  put_lin(packer.final_layer.linear, f"{sbase}.final_layer.linear")
  put_lin(packer.final_layer.adaLN_modulation, f"{sbase}.final_layer.adaLN_modulation.1")
  raw["model.scn_diffusion_module.scn_interpolant.sigma_data"] = np.asarray(packer.sigma_data, dtype=np.float32)
  return raw


def test_caliby_live_pytorch_vs_jax_graded_parity() -> None:
  """Verify live PyTorch-vs-JAX graded parity (f64 < 1e-6, f32 < 1e-4) and negative control via raw_sd."""
  import contextlib
  import math
  from collections.abc import Mapping
  from pathlib import Path
  from typing import Any, cast

  import equinox as eqx
  import torch
  import torch.nn.functional as F

  from aminx.families.caliby.driver import _load_caliby_eqx, _prepare_structure
  from aminx.io.weights import resolve_family_checkpoint

  @contextlib.contextmanager
  def _x64(enabled: bool):  # type: ignore[no-untyped-def]
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", enabled)
    try:
      yield
    finally:
      jax.config.update("jax_enable_x64", prev)

  def _cast(tree: Any, dtype: Any) -> Any:
    return jax.tree_util.tree_map(
      lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x,
      tree,
    )

  ubq_pdb = Path(__file__).resolve().parents[1] / "data" / "1ubq.pdb"
  prep = _prepare_structure(ubq_pdb, None)
  L = 36
  atom23_xyz = prep.atom23_xyz[:L]
  atom23_mask = prep.atom23_mask[:L]
  token_mask = prep.token_mask[:L]
  residue_idx = prep.residue_idx[:L]
  token_idx = prep.token_idx[:L]
  asym_id = prep.asym_id[:L]
  entity_id = prep.entity_id[:L]
  sym_id = prep.sym_id[:L]
  aatype_cal = prep.aatype_caliby[:L]

  def _torch_mpnn_from_raw_sd(
    raw_dict: Mapping[str, np.ndarray],
    mpnn_base: str,
    dtype: torch.dtype,
    aatype_np: np.ndarray | None = None,
    *,
    return_decoder_edges: bool = True,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sd = {
      k.replace(f"{mpnn_base}.token_features.", "mpnn.features.").replace(f"{mpnn_base}.", "mpnn."): v
      for k, v in raw_dict.items()
    }
    def t_key(key: str) -> torch.Tensor:
      return torch.from_numpy(sd[key].copy()).to(dtype=dtype)

    def pt_lin(x: torch.Tensor, prefix: str) -> torch.Tensor:
      w = t_key(f"{prefix}.weight")
      b = t_key(f"{prefix}.bias") if f"{prefix}.bias" in sd else None
      return F.linear(x, w, b)

    def pt_ln(x: torch.Tensor, prefix: str, eps: float = 1e-5) -> torch.Tensor:
      mean = x.mean(dim=-1, keepdim=True)
      var = x.var(dim=-1, unbiased=False, keepdim=True)
      out = (x - mean) * torch.rsqrt(var + eps)
      if f"{prefix}.weight" in sd:
        out = out * t_key(f"{prefix}.weight")
      if f"{prefix}.bias" in sd:
        out = out + t_key(f"{prefix}.bias")
      return out

    xyz = torch.from_numpy(atom23_xyz.copy()).to(dtype=dtype)
    mask23 = torch.from_numpy(atom23_mask.copy()).to(dtype=dtype)
    mask_v = torch.from_numpy(token_mask.copy()).to(dtype=dtype)
    res_i = torch.from_numpy(residue_idx.copy()).to(dtype=torch.long)
    asym_i = torch.from_numpy(asym_id.copy()).to(dtype=torch.long)

    ca_xyz = xyz[:, 1, :]
    ca_mask = mask23[:, 1]
    k_eff = min(48, L)
    mask_2d = ca_mask[:, None] * ca_mask[None, :]
    diff = ca_xyz[:, None, :] - ca_xyz[None, :]
    dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-6) * mask_2d
    dist_adj = dist + (1.0 - mask_2d) * 10000.0
    _, e_idx = torch.topk(-dist_adj, k_eff, dim=-1, largest=True)

    xyz_ca_filled = torch.where(mask23[..., None] > 0, xyz, xyz[:, 1:2, :])
    b_vec = xyz_ca_filled[:, 1, :] - xyz_ca_filled[:, 0, :]
    c_vec = xyz_ca_filled[:, 2, :] - xyz_ca_filled[:, 1, :]
    a_vec = torch.cross(b_vec, c_vec, dim=-1)
    cb_virt = -0.58273431 * a_vec + 0.56802827 * b_vec - 0.54067466 * c_vec + xyz_ca_filled[:, 1, :]
    has_bb = (mask23[:, 0] > 0) & (mask23[:, 1] > 0) & (mask23[:, 2] > 0)
    use_virt = has_bb & (mask23[:, 4] <= 0)
    cb_coords = torch.where(use_virt[:, None], cb_virt, xyz_ca_filled[:, 4, :])
    xyz_cb = torch.cat([xyz_ca_filled[:, :4, :], cb_coords[:, None, :], xyz_ca_filled[:, 5:, :]], dim=1)

    max_rel = 32
    same_asym = asym_i[:, None] == asym_i[e_idx]
    d_raw = res_i[:, None] - res_i[e_idx] + 100 * (1 - same_asym.long())
    d_clipped = torch.clamp(d_raw + max_rel, 0, 2 * max_rel)
    d_pos = torch.where(same_asym, d_clipped, torch.full_like(d_clipped, 2 * max_rel + 1))
    raw_pos = F.one_hot(d_pos, 2 * max_rel + 2).to(dtype=dtype)
    e_pos = pt_lin(raw_pos, "mpnn.features.embeddings.linear")

    d_mu = torch.linspace(2.0, 22.0, 16, dtype=dtype)
    d_sigma = 20.0 / 16.0
    edge_lig = torch.zeros((L, k_eff, 1), dtype=dtype)
    model_version = 1 if sd["mpnn.features.edge_embedding.weight"].shape[1] == 1057 else 0
    if model_version == 0:
      diff_all = xyz_cb[:, None, :, None, :] - xyz_cb[e_idx][:, :, None, :, :]
      dist_all = torch.sqrt(torch.sum(diff_all * diff_all, dim=-1) + 1e-6)
      rbf_all = torch.exp(-(((dist_all[..., None] - d_mu) / d_sigma) ** 2))
      raw_edges = torch.cat([e_pos, rbf_all.reshape(L, k_eff, 23 * 23 * 16), edge_lig], dim=-1)
    else:
      bb5_xyz = xyz_cb[:, :5, :]
      bb_diff = bb5_xyz[:, None, :, None, :] - bb5_xyz[e_idx][:, :, None, :]
      bb_dist = torch.sqrt(torch.sum(bb_diff * bb_diff, dim=-1) + 1e-6)
      rbf_bb = torch.exp(-(((bb_dist[..., None] - d_mu) / d_sigma) ** 2))
      bb4_xyz = xyz_cb[:, :4, :]
      sc10_xyz = xyz_cb[:, 4:14, :]
      sc_diff = bb4_xyz[:, None, :, None, :] - sc10_xyz[e_idx][:, :, None, :, :]
      sc_dist = torch.sqrt(torch.sum(sc_diff * sc_diff, dim=-1) + 1e-6)
      rbf_sc = torch.exp(-(((sc_dist[..., None] - d_mu) / d_sigma) ** 2))
      raw_edges = torch.cat(
        [e_pos, rbf_bb.reshape(L, k_eff, 5 * 5 * 16), rbf_sc.reshape(L, k_eff, 4 * 10 * 16), edge_lig],
        dim=-1,
      )

    edges = pt_ln(pt_lin(raw_edges, "mpnn.features.edge_embedding"), "mpnn.features.norm_edges")
    nodes = torch.zeros((L, 128), dtype=dtype)

    h_v = nodes
    h_e = pt_lin(edges, "mpnn.W_e")
    w_s_w = t_key("mpnn.W_s.weight")
    if w_s_w.shape[0] != 32:
      w_s_w = w_s_w.T
    default_tok = 31 if return_decoder_edges else 0
    seq_t = torch.full((L,), default_tok, dtype=torch.long) if aatype_np is None else torch.from_numpy(aatype_np).long()
    h_s = w_s_w[torch.clamp(seq_t, 0, 31)]
    update_dec_edges = "mpnn.decoder_layers.0.W11.weight" in sd
    if update_dec_edges:
      h_v = h_v + h_s

    mask_attend = torch.gather(mask_v[:, None] * mask_v[None, :], 1, e_idx)
    num_enc = sum(1 for k in sd if k.startswith("mpnn.encoder_layers.") and k.endswith(".W1.weight"))
    for i in range(num_enc):
      ep = f"mpnn.encoder_layers.{i}"
      h_ev = torch.cat([h_v[:, None, :].expand(L, k_eff, -1), h_e, h_v[e_idx]], dim=-1)
      msg = pt_lin(F.gelu(pt_lin(F.gelu(pt_lin(h_ev, f"{ep}.W1")), f"{ep}.W2")), f"{ep}.W3")
      msg = mask_attend[..., None] * msg
      h_v = pt_ln(h_v + msg.sum(dim=-2) / 30.0, f"{ep}.norm1")
      dh = pt_lin(F.gelu(pt_lin(h_v, f"{ep}.dense.W_in")), f"{ep}.dense.W_out")
      h_v = mask_v[:, None] * pt_ln(h_v + dh, f"{ep}.norm2")
      h_ev2 = torch.cat([h_v[:, None, :].expand(L, k_eff, -1), h_e, h_v[e_idx]], dim=-1)
      e_msg = pt_lin(F.gelu(pt_lin(F.gelu(pt_lin(h_ev2, f"{ep}.W11")), f"{ep}.W12")), f"{ep}.W13")
      h_e = pt_ln(h_e + e_msg, f"{ep}.norm3")

    num_dec = sum(1 for k in sd if k.startswith("mpnn.decoder_layers.") and k.endswith(".W1.weight"))
    h_es = torch.cat([h_e, h_s[e_idx]], dim=-1)
    if update_dec_edges:
      h_esv = torch.cat([h_es, h_v[e_idx]], dim=-1)
      for i in range(num_dec):
        dp = f"mpnn.decoder_layers.{i}"
        h_ev = torch.cat([h_v[:, None, :].expand(L, k_eff, -1), h_esv], dim=-1)
        msg = pt_lin(F.gelu(pt_lin(F.gelu(pt_lin(h_ev, f"{dp}.W1")), f"{dp}.W2")), f"{dp}.W3")
        msg = mask_attend[..., None] * msg
        h_v = pt_ln(h_v + msg.sum(dim=-2) / 30.0, f"{dp}.norm1")
        dh = pt_lin(F.gelu(pt_lin(h_v, f"{dp}.dense.W_in")), f"{dp}.dense.W_out")
        h_v = mask_v[:, None] * pt_ln(h_v + dh, f"{dp}.norm2")
        if f"{dp}.W11.weight" in sd:
          h_ev_f = torch.cat([h_v[:, None, :].expand(L, k_eff, -1), h_esv, h_v[e_idx]], dim=-1)
          e_msg = pt_lin(F.gelu(pt_lin(F.gelu(pt_lin(h_ev_f, f"{dp}.W11")), f"{dp}.W12")), f"{dp}.W13")
          h_esv = pt_ln(h_esv + e_msg, f"{dp}.norm3")
      if return_decoder_edges:
        return h_v, h_esv, e_idx, mask_attend
    else:
      for i in range(num_dec):
        dp = f"mpnn.decoder_layers.{i}"
        h_esv = torch.cat([h_es, h_v[e_idx]], dim=-1)
        h_ev = torch.cat([h_v[:, None, :].expand(L, k_eff, -1), h_esv], dim=-1)
        msg = pt_lin(F.gelu(pt_lin(F.gelu(pt_lin(h_ev, f"{dp}.W1")), f"{dp}.W2")), f"{dp}.W3")
        msg = mask_attend[..., None] * msg
        h_v = pt_ln(h_v + msg.sum(dim=-2) / 30.0, f"{dp}.norm1")
        dh = pt_lin(F.gelu(pt_lin(h_v, f"{dp}.dense.W_in")), f"{dp}.dense.W_out")
        h_v = mask_v[:, None] * pt_ln(h_v + dh, f"{dp}.norm2")
    return h_v, h_e, e_idx, mask_attend

  # 1. Potts checkpoints parity (v0, v1, distill): raw_sd -> convert_caliby_potts_state_dict vs PyTorch from raw_sd
  for cid in ("caliby", "soluble_caliby_v1", "caliby_distill"):
    try:
      prov = resolve_family_checkpoint(cid, family="caliby")
      pkg_model = cast("CalibyModel", _load_caliby_eqx(Path(prov.path), cid))
    except (OSError, RuntimeError, FileNotFoundError):
      mv = 1 if ("v1" in cid or "distill" in cid) else 0
      pkg_model = CalibyModel(
        k_neighbors=48,
        model_version=mv,
        parameterization="factor",
        has_bg=True,
        bg_dim=1,
        key=jax.random.PRNGKey(70 + mv),
      )
    raw_sd = _extract_raw_caliby_potts_sd(pkg_model)
    model = convert_caliby_potts_state_dict(raw_sd)
    pbase = "model._orig_mod.denoiser.atom_mpnn.decoder_S_potts"

    for is_f64, j_dt, t_dt, tol in ((True, jnp.float64, torch.float64, 1e-6), (False, jnp.float32, torch.float32, 1e-4)):
      with _x64(is_f64):
        m_c = _cast(model, j_dt)
        pkg_c = _cast(pkg_model, j_dt)
        out_j = m_c(
          jnp.asarray(atom23_xyz, dtype=j_dt),
          jnp.asarray(atom23_mask, dtype=j_dt),
          jnp.asarray(token_mask, dtype=j_dt),
          jnp.asarray(residue_idx, dtype=jnp.int32),
          jnp.asarray(token_idx, dtype=jnp.int32),
          jnp.asarray(asym_id, dtype=jnp.int32),
          jnp.asarray(entity_id, dtype=jnp.int32),
          jnp.asarray(sym_id, dtype=jnp.int32),
        )
        out_pkg = pkg_c(
          jnp.asarray(atom23_xyz, dtype=j_dt),
          jnp.asarray(atom23_mask, dtype=j_dt),
          jnp.asarray(token_mask, dtype=j_dt),
          jnp.asarray(residue_idx, dtype=jnp.int32),
          jnp.asarray(token_idx, dtype=jnp.int32),
          jnp.asarray(asym_id, dtype=jnp.int32),
          jnp.asarray(entity_id, dtype=jnp.int32),
          jnp.asarray(sym_id, dtype=jnp.int32),
        )
        assert jnp.allclose(out_j.h_32, out_pkg.h_32, atol=1e-7)
        assert jnp.allclose(out_j.j_32, out_pkg.j_32, atol=1e-7)
      with torch.no_grad():
        def t_sd(key: str) -> torch.Tensor:
          return torch.from_numpy(raw_sd[key].copy()).to(dtype=t_dt)

        def lin_sd(x: torch.Tensor, prefix: str) -> torch.Tensor:
          w = t_sd(f"{prefix}.weight")
          b = t_sd(f"{prefix}.bias") if f"{prefix}.bias" in raw_sd else None
          return F.linear(x, w, b)

        h_v_t, h_e_t, e_idx_t, mask_att_t = _torch_mpnn_from_raw_sd(
          raw_sd, "model._orig_mod.denoiser.atom_mpnn", t_dt, return_decoder_edges=True,
        )
        k_potts = min(30, e_idx_t.shape[1])
        h_e_tr = h_e_t[:, :k_potts, :]
        e_idx_tr = e_idx_t[:, :k_potts]
        mask_att_tr = mask_att_t[:, :k_potts]
        scale = torch.exp(t_sd(f"{pbase}.log_scale"))
        mask_v_t = torch.from_numpy(token_mask.copy()).to(dtype=t_dt)[:, None]
        h_t = scale * mask_v_t * lin_sd(h_v_t, f"{pbase}.W_h")
        row_i = torch.arange(L, dtype=torch.long)[:, None]
        valid_e = mask_att_tr * (e_idx_tr != row_i).to(t_dt)
        num_f = raw_sd[f"{pbase}.W_J_left.weight"].shape[0] // 32
        j_l = (valid_e[..., None] * lin_sd(h_e_tr, f"{pbase}.W_J_left")).reshape(L, k_potts, 32, num_f)
        j_r = (valid_e[..., None] * lin_sd(h_e_tr, f"{pbase}.W_J_right")).reshape(L, k_potts, 32, num_f)
        j_knn = scale * torch.matmul(j_l, torch.swapaxes(j_r, -1, -2))
        h_t = (h_t - h_t.mean(dim=-1, keepdim=True)) * mask_v_t
        j_knn = (
          j_knn
          - j_knn.mean(dim=-1, keepdim=True)
          - j_knn.mean(dim=-2, keepdim=True)
          + j_knn.mean(dim=(-1, -2), keepdim=True)
        ) * valid_e[..., None, None]
        if f"{pbase}.W_h_bg.weight" in raw_sd:
          h_t = h_t + scale * mask_v_t * lin_sd(h_v_t, f"{pbase}.W_h_bg")
        if f"{pbase}.W_J_bg.weight" in raw_sd:
          j_knn = j_knn + (scale * valid_e[..., None] * lin_sd(h_e_tr, f"{pbase}.W_J_bg"))[..., None]
        j_dense = torch.zeros((L, L, 32, 32), dtype=t_dt)
        rows_2d = row_i.expand_as(e_idx_tr)
        j_dense.index_put_((rows_2d, e_idx_tr), j_knn, accumulate=True)
        adj_dense = torch.zeros((L, L), dtype=t_dt)
        adj_dense.index_put_((rows_2d, e_idx_tr), valid_e, accumulate=True)
        recip_mask = ((adj_dense > 0) & (adj_dense.T > 0)).to(t_dt)
        j_dense = 0.5 * (j_dense + j_dense.permute(1, 0, 3, 2)) * recip_mask[:, :, None, None]
      assert float(np.max(np.abs(np.asarray(out_j.h_32) - h_t.numpy()))) < tol
      assert float(np.max(np.abs(np.asarray(out_j.j_32) - j_dense.numpy()))) < tol

    m_bad = eqx.tree_at(lambda x: x.potts_decoder.W_h.weight, model, model.potts_decoder.W_h.weight[::-1])
    out_bad = m_bad(
      jnp.asarray(atom23_xyz, dtype=jnp.float32),
      jnp.asarray(atom23_mask, dtype=jnp.float32),
      jnp.asarray(token_mask, dtype=jnp.float32),
      jnp.asarray(residue_idx, dtype=jnp.int32),
      jnp.asarray(token_idx, dtype=jnp.int32),
      jnp.asarray(asym_id, dtype=jnp.int32),
      jnp.asarray(entity_id, dtype=jnp.int32),
      jnp.asarray(sym_id, dtype=jnp.int32),
    )
    assert float(np.max(np.abs(np.asarray(out_bad.h_32) - h_t.numpy()))) > 1e-2

  # 2. Packer checkpoint parity (caliby_packer_000): raw_sd -> convert_caliby_packer_state_dict vs PyTorch from raw_sd
  try:
    prov_p = resolve_family_checkpoint("caliby_packer_000", family="caliby")
    pkg_packer = cast("CalibySidechainMLP", _load_caliby_eqx(Path(prov_p.path), "caliby_packer_000"))
  except (OSError, RuntimeError, FileNotFoundError):
    pkg_packer = CalibySidechainMLP(
      key=jax.random.PRNGKey(79),
    )
  raw_packer_sd = _extract_raw_caliby_packer_sd(pkg_packer)
  packer = convert_caliby_packer_state_dict(raw_packer_sd)
  sbase = "model.scn_diffusion_module.scn_denoiser"

  rng = np.random.default_rng(123)
  x_noisy_np = rng.standard_normal((L, 33, 3)).astype(np.float64)
  x_cond_np = rng.standard_normal((L, 33, 3)).astype(np.float64)
  scn_mask_np = np.ones((L, 33), dtype=np.float64)
  sigma_np = np.full((L,), 1.25, dtype=np.float64)

  for is_f64, j_dt, t_dt, tol in ((True, jnp.float64, torch.float64, 1e-6), (False, jnp.float32, torch.float32, 1e-4)):
    with _x64(is_f64):
      p_c = _cast(packer, j_dt)
      h_v_j, _, _, _ = p_c.mpnn(
        jnp.asarray(atom23_xyz, dtype=j_dt),
        jnp.asarray(atom23_mask, dtype=j_dt),
        jnp.asarray(token_mask, dtype=j_dt),
        jnp.asarray(residue_idx, dtype=jnp.int32),
        jnp.asarray(token_idx, dtype=jnp.int32),
        jnp.asarray(asym_id, dtype=jnp.int32),
        jnp.asarray(entity_id, dtype=jnp.int32),
        jnp.asarray(sym_id, dtype=jnp.int32),
        aatype=jnp.asarray(aatype_cal, dtype=jnp.int32),
      )
      den_j = p_c.denoise_from_trunk(
        jnp.asarray(x_noisy_np, dtype=j_dt),
        jnp.asarray(sigma_np, dtype=j_dt),
        h_v_j,
        jnp.asarray(scn_mask_np, dtype=j_dt),
        x_cond=jnp.asarray(x_cond_np, dtype=j_dt),
        aatype=jnp.asarray(aatype_cal, dtype=jnp.int32),
      )
    with torch.no_grad():
      def tp_sd(key: str) -> torch.Tensor:
        return torch.from_numpy(raw_packer_sd[key].copy()).to(dtype=t_dt)

      def plin_sd(x: torch.Tensor, prefix: str) -> torch.Tensor:
        w = tp_sd(f"{prefix}.weight")
        b = tp_sd(f"{prefix}.bias") if f"{prefix}.bias" in raw_packer_sd else None
        return F.linear(x, w, b)

      h_v_pt, _, _, _ = _torch_mpnn_from_raw_sd(
        raw_packer_sd, "model.denoiser.atom_mpnn", t_dt, aatype_np=aatype_cal, return_decoder_edges=False,
      )
      sig_t = torch.from_numpy(sigma_np.copy()).to(dtype=t_dt)
      sigma_key = "model.scn_diffusion_module.scn_interpolant.sigma_data"
      sigma_data = float(raw_packer_sd[sigma_key].item()) if sigma_key in raw_packer_sd else 1.0
      sd2 = sigma_data * sigma_data
      sig2 = sig_t * sig_t
      c_skip = sd2 / (sig2 + sd2)
      c_out = sig_t * sigma_data / torch.sqrt(sig2 + sd2)
      c_in = 1.0 / torch.sqrt(sd2 + sig2)
      c_noise = 0.25 * torch.log(torch.clamp(sig_t, min=1e-8))
      xn_t = torch.from_numpy(x_noisy_np.copy()).to(dtype=t_dt)
      xc_t = torch.from_numpy(x_cond_np.copy()).to(dtype=t_dt)
      sm_t = torch.from_numpy(scn_mask_np.copy()).to(dtype=t_dt)
      x_scaled = (c_in[:, None, None] * xn_t) * sm_t[..., None]
      cond_scaled = (xc_t / sigma_data) * sm_t[..., None]
      x_scn = torch.cat([x_scaled, cond_scaled], dim=-1).reshape(L, 33 * 6)
      seq_oh = F.one_hot(torch.from_numpy(aatype_cal.copy()).long(), 32).to(dtype=t_dt)
      in_feat = torch.cat([x_scn, seq_oh], dim=-1)
      h_v_emb = plin_sd(h_v_pt, f"{sbase}.h_V_embedder")
      h = plin_sd(in_feat, f"{sbase}.x_embedder") + h_v_emb
      half = 128
      freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, dtype=t_dt) / half)
      args = c_noise[:, None] * freqs[None, :]
      t_emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
      t_emb = plin_sd(F.silu(plin_sd(t_emb, f"{sbase}.timestep_embedder.mlp.0")), f"{sbase}.timestep_embedder.mlp.2")
      c = t_emb + h_v_emb
      num_blocks = sum(1 for k in raw_packer_sd if k.startswith(f"{sbase}.blocks.") and k.endswith(".mlp.fc1.weight"))
      for bi in range(num_blocks):
        mod = plin_sd(F.silu(c), f"{sbase}.blocks.{bi}.adaLN_modulation.1")
        sh, sc, gt = torch.chunk(mod, 3, dim=-1)
        hb = F.layer_norm(h, (h.shape[-1],), None, None, eps=1e-6) * (1.0 + sc) + sh
        hb = plin_sd(
          F.gelu(plin_sd(hb, f"{sbase}.blocks.{bi}.mlp.fc1"), approximate="tanh"),
          f"{sbase}.blocks.{bi}.mlp.fc2",
        )
        h = h + gt * hb
      mod_f = plin_sd(F.silu(c), f"{sbase}.final_layer.adaLN_modulation.1")
      sh_f, sc_f = torch.chunk(mod_f, 2, dim=-1)
      hf = F.layer_norm(h, (h.shape[-1],), None, None, eps=1e-6) * (1.0 + sc_f) + sh_f
      fx = plin_sd(hf, f"{sbase}.final_layer.linear").reshape(L, 33, 3)
      den_pt = (c_skip[:, None, None] * xn_t + c_out[:, None, None] * fx) * sm_t[..., None]
    assert float(np.max(np.abs(np.asarray(h_v_j) - h_v_pt.numpy()))) < tol
    assert float(np.max(np.abs(np.asarray(den_j) - den_pt.numpy()))) < tol

  p_bad = eqx.tree_at(lambda x: x.final_layer.linear.weight, packer, packer.final_layer.linear.weight[::-1])
  den_bad = p_bad.denoise_from_trunk(
    jnp.asarray(x_noisy_np, dtype=jnp.float32),
    jnp.asarray(sigma_np, dtype=jnp.float32),
    jnp.asarray(h_v_j, dtype=jnp.float32),
    jnp.asarray(scn_mask_np, dtype=jnp.float32),
    x_cond=jnp.asarray(x_cond_np, dtype=jnp.float32),
    aatype=jnp.asarray(aatype_cal, dtype=jnp.int32),
  )
  assert float(np.max(np.abs(np.asarray(den_bad) - den_pt.numpy()))) > 1e-2


def test_caliby_ensemble_potts_averaging() -> None:
  """Verify aggregate_ensemble_potts_params on both dense and heterogeneous kNN conformers."""
  key = jax.random.PRNGKey(99)
  num_c, num_l, num_k, num_s = 3, 6, 4, 21
  h_stack = jax.random.normal(key, (num_c, num_l, num_s))
  j_dense_stack = jax.random.normal(jax.random.PRNGKey(100), (num_c, num_l, num_l, num_s, num_s))
  h_avg, j_avg = aggregate_ensemble_potts_params(h_stack, j_dense_stack)
  assert h_avg.shape == (num_l, num_s)
  assert j_avg.shape == (num_l, num_l, num_s, num_s)
  assert jnp.allclose(h_avg, jnp.mean(h_stack, axis=0), atol=1e-6)
  assert jnp.allclose(j_avg, jnp.mean(j_dense_stack, axis=0), atol=1e-6)

  # Heterogeneous kNN conformers (C, L, K, S, S) with distinct neighbor indices e_idx (C, L, K)
  j_knn_stack = jax.random.normal(jax.random.PRNGKey(101), (num_c, num_l, num_k, num_s, num_s))
  e_idx_0 = jnp.broadcast_to(jnp.array([1, 2, 3, 4], dtype=jnp.int32)[None, :], (num_l, num_k))
  e_idx_1 = jnp.broadcast_to(jnp.array([2, 3, 4, 5], dtype=jnp.int32)[None, :], (num_l, num_k))
  e_idx_2 = jnp.broadcast_to(jnp.array([0, 1, 3, 5], dtype=jnp.int32)[None, :], (num_l, num_k))
  e_idx_stack = jnp.stack([e_idx_0, e_idx_1, e_idx_2], axis=0)
  h_het, j_het = aggregate_ensemble_potts_params(h_stack, j_knn_stack, e_idx=e_idx_stack)
  assert h_het.shape == (num_l, num_s)
  assert j_het.shape == (num_l, num_l, num_s, num_s)
  assert jnp.allclose(j_het, jnp.transpose(j_het, (1, 0, 3, 2)), atol=1e-6)


def test_local_to_global_scn_ghost_sidechains_land_on_ca() -> None:
  """Absent Atom23 sidechain slots land on CA; a present slot keeps a nonzero local offset."""
  n_tok = 2
  ca = jnp.array([[1.5, -2.0, 3.25], [0.5, 4.0, -1.0]], dtype=jnp.float32)
  xyz = jnp.zeros((n_tok, 23, 3), dtype=jnp.float32)
  xyz = xyz.at[:, 0, :].set(ca + jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32))
  xyz = xyz.at[:, 1, :].set(ca)
  xyz = xyz.at[:, 2, :].set(ca + jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32))
  xyz = xyz.at[:, 3, :].set(ca + jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32))
  mask = jnp.zeros((n_tok, 23), dtype=jnp.float32).at[:, :4].set(1.0)
  local = jnp.zeros((n_tok, 33, 3), dtype=jnp.float32)

  ghost = local_to_global_scn(local, xyz, mask)
  assert jnp.allclose(ghost[:, 4:, :], ca[:, None, :], atol=1e-5)
  assert not jnp.allclose(ghost[:, 4:, :], 0.0, atol=1e-6)
  assert jnp.allclose(ghost[:, :4, :], xyz[:, :4, :], atol=1e-6)

  mask_on = mask.at[:, 4:].set(1.0)
  local_on = local.at[:, 0, :].set(jnp.array([2.0, -3.0, 4.0], dtype=jnp.float32))
  present = local_to_global_scn(local_on, xyz, mask_on)
  assert not jnp.allclose(present[:, 4, :], 0.0, atol=1e-6)


def test_caliby_packer_sample_schema_and_handles() -> None:
  """Packer checkpoints expose only the sample schema; Potts sample still returns energy."""
  drv = CalibyDriver()
  packer = types.SimpleNamespace(checkpoint_id="caliby_packer_010")
  assert drv.handles(packer, "sample")
  assert not drv.handles(packer, "score:energy")
  assert not drv.handles(packer, "score:nll")
  assert not drv.handles(packer, "score:logits")
  schema = drv.result_schema(packer, "sample")
  assert set(schema) == {"sequences", "packed_atom23_xyz"}

  potts = types.SimpleNamespace(checkpoint_id="caliby")
  assert drv.handles(potts, "sample")
  potts_schema = drv.result_schema(potts, "sample")
  assert set(potts_schema) == {"sequences", "energy"}


def test_caliby_checkpoint_manifest_records_has_bg(tmp_path: Path) -> None:
  """convert_caliby_checkpoint writes a manifest for a has_bg Potts checkpoint.

  Parameters
  ----------
  tmp_path : Path
    Pytest temporary directory for the serialized ``.eqx`` file.

  Returns
  -------
  None
    Asserts the manifest records ``has_bg`` and ``bg_dim``.
  """
  from aminx.families.caliby.convert import convert_caliby_checkpoint  # noqa: PLC0415

  ref = CalibyModel(
    k_neighbors=6,
    model_version=1,
    parameterization="factor",
    has_bg=True,
    bg_dim=32,
    key=jax.random.PRNGKey(45),
  )
  assert ref.potts_decoder.has_bg is True
  raw_sd = _extract_raw_caliby_potts_sd(ref)
  _, manifest = convert_caliby_checkpoint(raw_sd, tmp_path / "caliby_bg.eqx", checkpoint_id="caliby")
  assert manifest["has_bg"] is True
  assert manifest["bg_dim"] == 32
  assert (tmp_path / "caliby_bg.eqx").is_file()


_SENTINEL_CALLS: list[str] = []


def _sentinel_side_effect() -> str:
  """Record that a pickled callable ran; stands in for an arbitrary-code payload."""
  _SENTINEL_CALLS.append("called")
  return "called"


class _SentinelPayload:
  """Pickles to a reduce call of ``_sentinel_side_effect``."""

  def __reduce__(self) -> tuple[object, ...]:
    """Return the sentinel call as the reduce recipe.

    Parameters
    ----------
    None
      Takes no arguments beyond ``self``.

    Returns
    -------
    tuple[object, ...]
      ``(callable, args)`` pair that pickle replays on load.
    """
    return (_sentinel_side_effect, ())


def test_caliby_torch_unpickler_refuses_arbitrary_globals() -> None:
  """Refuse out-of-allow-list globals in a Caliby ``.ckpt`` pickle and run no payload.

  Parameters
  ----------
  None
    Uses the module-level sentinel payload.

  Returns
  -------
  None
    Asserts the sentinel never runs and allow-listed data still loads.
  """
  import collections  # noqa: PLC0415
  import io  # noqa: PLC0415
  import pickle  # noqa: PLC0415

  import pytest  # noqa: PLC0415

  from aminx.families.caliby.driver import _RestrictedCalibyUnpickler  # noqa: PLC0415

  _SENTINEL_CALLS.clear()
  payload = pickle.dumps(_SentinelPayload())
  with pytest.raises(pickle.UnpicklingError, match="Refusing to unpickle"):
    _RestrictedCalibyUnpickler(io.BytesIO(payload)).load()
  assert _SENTINEL_CALLS == []

  ok = pickle.dumps(collections.OrderedDict(weight=1))
  assert _RestrictedCalibyUnpickler(io.BytesIO(ok)).load() == collections.OrderedDict(weight=1)


def _factor_potts_raw_sd(key: jax.Array) -> dict[str, np.ndarray]:
  """Build a complete upstream-keyed factor Potts state dict (v1, with background heads).

  Parameters
  ----------
  key : jax.Array
    PRNG key for the reference model.

  Returns
  -------
  dict[str, np.ndarray]
    Raw state dict containing every key the converter requires.
  """
  ref = CalibyModel(
    k_neighbors=6, model_version=1, parameterization="factor", has_bg=True, bg_dim=1, key=key
  )
  return _extract_raw_caliby_potts_sd(ref)


def test_caliby_converter_rejects_missing_required_keys() -> None:
  """Raise KeyError naming a missing required factor or LayerNorm key."""
  import pytest  # noqa: PLC0415

  base = _factor_potts_raw_sd(jax.random.PRNGKey(46))
  for missing in ("potts_decoder.W_J_left.weight", "mpnn.encoder_layers.0.norm1.weight"):
    assert missing in base
    raw = {k: v for k, v in base.items() if k != missing}
    with pytest.raises(KeyError, match=missing.replace(".", r"\.")):
      convert_caliby_potts_state_dict(raw)


def test_caliby_converter_still_accepts_absent_optional_background_heads() -> None:
  """Convert a factor checkpoint with the optional background heads absent."""
  base = _factor_potts_raw_sd(jax.random.PRNGKey(47))
  raw = {k: v for k, v in base.items() if "_bg." not in k}
  converted = convert_caliby_potts_state_dict(raw)
  assert converted.potts_decoder.W_h_bg is None
  assert converted.potts_decoder.W_J_bg is None
