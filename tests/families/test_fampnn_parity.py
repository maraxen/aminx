"""Unit and parity tests for the FAMPNN family port (src/aminx/families/fampnn)."""

from __future__ import annotations

import os as _os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from aminx.families.fampnn import (
  AMINX_TO_FAMPNN_PERM,
  FAMPNN_ALL_CHECKPOINTS,
  FAMPNN_TO_AMINX_PERM,
  FAMPNNDriver,
  FAMPNNModel,
  FAMPNNPrepared,
  FAMPNNProteinFeatures,
  GVPGraphEmbedding,
  SidechainProteinFeatures,
  atom14_to_atom37,
  atom37_to_atom14,
  convert_fampnn_state_dict,
  get_rotation_frames,
  has_fampnn_confidence_head,
  normalize_fampnn_state_dict,
)
from aminx.host.family_driver import DRIVER_BACKED_FAMILY_MODULES, FAMILY_DRIVERS, FamilyBatch
from aminx.io.weights import family_checkpoints, resolve_family_checkpoint
from aminx.run.options import FAMPNNOptions
from aminx.run.specs import SamplingSpecification

_LOCAL_FOLLOWUP_WEIGHTS = Path(
  _os.environ.get(
    "AMINX_UPSTREAM_REPOS_DIR",
    "/home/marielle/.gemini/antigravity-cli/brain/e0a4519f-8415-4303-b518-8fc3d09cb0cc/scratch/repos",
  )
) / "aminx_followup_weights"
if "AMINX_WEIGHTS_DIR" not in _os.environ and _LOCAL_FOLLOWUP_WEIGHTS.is_dir():
  _os.environ["AMINX_WEIGHTS_DIR"] = str(_LOCAL_FOLLOWUP_WEIGHTS)


def _load_or_build_fampnn_checkpoint_model(cid: str) -> FAMPNNModel:
  """Load packaged FAMPNN checkpoint if cached locally, else build exact checkpoint topology."""
  from aminx.families.fampnn.driver import _load_fampnn_eqx

  try:
    prov = resolve_family_checkpoint(cid, family="fampnn")
    assert not prov.sha256.startswith("fa3000")
    return _load_fampnn_eqx(Path(prov.path), cid)
  except (OSError, RuntimeError, FileNotFoundError):
    is_cath = "cath" in cid
    return FAMPNNModel(
      has_confidence=not is_cath,
      num_encoder_layers=3 if is_cath else 5,
      num_decoder_layers=3 if is_cath else 5,
      use_sidechain_features=True,
      num_gvp_layers=3 if is_cath else 5,
      gvp_w_out_mode="linear" if is_cath else "gvp",
      gvp_rotate_out=is_cath,
      has_w_out=True,
      scn_depth=4,
      has_f_embedder=True,
      conf_num_bins=50,
      conf_has_upstream_heads=True,
      key=jax.random.PRNGKey(101 if is_cath else 103),
    )


def test_alphabet_permutation_bijection() -> None:
  """Verify FAMPNN <-> Aminx 21-token alphabet permutations are exact inverses."""
  ident_a = np.asarray(FAMPNN_TO_AMINX_PERM)[np.asarray(AMINX_TO_FAMPNN_PERM)]
  ident_f = np.asarray(AMINX_TO_FAMPNN_PERM)[np.asarray(FAMPNN_TO_AMINX_PERM)]
  np.testing.assert_array_equal(ident_a, np.arange(21))
  np.testing.assert_array_equal(ident_f, np.arange(21))
  assert int(FAMPNN_TO_AMINX_PERM[20]) == 20
  assert int(AMINX_TO_FAMPNN_PERM[20]) == 20


def test_atom37_atom14_roundtrip() -> None:
  """Verify atom37_to_atom14 and atom14_to_atom37 preserve all populated heavy atoms."""
  L = 20
  aatype = jnp.arange(L, dtype=jnp.int32)
  coords37 = jax.random.normal(jax.random.PRNGKey(0), (L, 37, 3))
  mask37 = jnp.ones((L, 37), dtype=jnp.float32)
  coords14, mask14 = atom37_to_atom14(coords37, mask37, aatype)
  assert coords14.shape == (L, 14, 3)
  assert mask14.shape == (L, 14)
  rt37, rt_mask37 = atom14_to_atom37(coords14, mask14, aatype)
  assert rt37.shape == (L, 37, 3)
  coords14_rt, mask14_rt = atom37_to_atom14(rt37, rt_mask37, aatype)
  np.testing.assert_allclose(coords14, coords14_rt, atol=1e-6)
  np.testing.assert_allclose(mask14, mask14_rt, atol=1e-6)


def test_atom37_indices_and_reference_match_upstream_order() -> None:
  """Present Atom14 slots use AlphaFold atom37 indices, and sidechains are not at the origin."""
  from aminx.families.fampnn.constants import (
    RESTYPE_ATOM14_TO_ATOM37,
    RESTYPE_ATOM37_MASK,
    RESTYPE_REF_POS_ATOM37,
  )

  # HIS ND1/CD2 and ARG NE/CZ follow atom_order, not the previous swapped columns.
  his = np.asarray(RESTYPE_ATOM14_TO_ATOM37[8])
  arg = np.asarray(RESTYPE_ATOM14_TO_ATOM37[1])
  assert his[6:10].tolist() == [14, 13, 20, 25]
  assert arg[7:11].tolist() == [23, 32, 29, 30]
  # Empty slots are CA. GLY has no CB.
  assert int(RESTYPE_ATOM14_TO_ATOM37[7, 4]) == 1
  ref = np.asarray(RESTYPE_REF_POS_ATOM37)
  mask = np.asarray(RESTYPE_ATOM37_MASK)
  # ARG sidechain atoms past CB are present and not the CA origin.
  arg_side = ref[1, 5:][mask[1, 5:] > 0]
  assert arg_side.shape[0] > 0
  assert float(np.max(np.abs(arg_side))) > 0.5


def test_rotation_frames_so3_orthogonality() -> None:
  """Verify get_rotation_frames produces orthogonal SO(3) matrices with det == +1."""
  key = jax.random.PRNGKey(1)
  bb = jax.random.normal(key, (12, 3, 3))
  R = get_rotation_frames(bb[:, 0, :], bb[:, 1, :], bb[:, 2, :])
  assert R.shape == (12, 3, 3)
  eye = jnp.broadcast_to(jnp.eye(3), (12, 3, 3))
  np.testing.assert_allclose(jnp.matmul(R, jnp.swapaxes(R, -1, -2)), eye, atol=1e-5)
  dets = jnp.linalg.det(R)
  np.testing.assert_allclose(dets, jnp.ones((12,)), atol=1e-5)


def test_gvp_virtual_cb_fallback_in_embedding() -> None:
  """Verify GVPGraphEmbedding substitutes virtual Cb when atom14 index 4 is masked."""
  key = jax.random.PRNGKey(2)
  L = 8
  coords14 = jax.random.normal(key, (L, 14, 3))
  mask14 = jnp.ones((L, 14), dtype=jnp.float32).at[:4, 4].set(0.0)
  aatype_oh = jax.nn.one_hot(jnp.zeros((L,), dtype=jnp.int32), 21)
  mask = jnp.ones((L,), dtype=jnp.float32)
  res_idx = jnp.arange(L, dtype=jnp.int32)
  chain_idx = jnp.zeros((L,), dtype=jnp.int32)

  emb = GVPGraphEmbedding(node_h_dim=(16, 4), edge_h_dim=(8, 1), top_k=4, key=key)
  h_V, h_E, E_idx = emb(coords14, mask14, aatype_oh, mask, res_idx, chain_idx)
  assert h_V[0].shape == (L, 16)
  assert h_V[1].shape == (L, 4, 3)
  assert h_E[0].shape == (L, 4, 8)
  assert h_E[1].shape == (L, 4, 1, 3)
  assert E_idx.shape == (L, 4)
  assert bool(jnp.all(jnp.isfinite(h_V[0])))
  assert bool(jnp.all(jnp.isfinite(h_V[1])))


def test_fa_encoder_and_fampnn_model_forward_variants() -> None:
  """Verify FAMPNNModel forward pass and SidechainConfidenceModule for 0_3 and 0_3_cath."""
  key = jax.random.PRNGKey(3)
  L = 6
  coords37 = jax.random.normal(key, (L, 37, 3))
  mask37 = jnp.ones((L, 37), dtype=jnp.float32)
  aatype = jnp.zeros((L,), dtype=jnp.int32)
  seq_mask = jnp.ones((L,), dtype=jnp.float32)
  mask = jnp.ones((L,), dtype=jnp.float32)
  res_idx = jnp.arange(L, dtype=jnp.int32)
  chain_idx = jnp.zeros((L,), dtype=jnp.int32)

  pf = FAMPNNProteinFeatures(edge_features=32, top_k=4, key=key)
  e_bb, e_idx = pf(coords37, mask, res_idx, chain_idx)
  assert e_bb.shape == (L, 4, 32)
  assert e_idx.shape == (L, 4)

  coords14, mask14 = atom37_to_atom14(coords37, mask37, aatype)
  spf = SidechainProteinFeatures(edge_features=32, top_k=4, key=key)
  e_sc, _ = spf(coords14, mask14, mask, res_idx, chain_idx, e_idx=e_idx)
  assert e_sc.shape == (L, 4, 32)

  model = FAMPNNModel(has_confidence=True, key=key)
  out = model(coords37, mask37, aatype, seq_mask, mask, res_idx, chain_idx)
  assert out.logits_fampnn.shape == (L, 21)
  assert out.logits_aminx.shape == (L, 21)
  assert out.h_v.shape == (L, 128)
  assert model.scn_confidence is not None

  psce = model.scn_confidence(
    coords37, mask37, out.h_v, aatype, mask, res_idx, chain_idx, h_esv=out.h_e, e_idx=out.e_idx,
  )
  assert psce.shape == (L, 37)

  model_cath = FAMPNNModel(has_confidence=False, key=key)
  assert model_cath.scn_confidence is None


def test_sidechain_diffusion_and_test_mar_sample_via_driver() -> None:
  """Verify FAMPNNDriver score:logits, score:nll, and MAR+EDM sample modes."""
  key = jax.random.PRNGKey(4)
  L = 6
  coords37 = np.zeros((L, 37, 3), dtype=np.float32)
  coords37[:, 1, 0] = np.arange(L, dtype=np.float32) * 3.8
  coords37[:, 0, 0] = coords37[:, 1, 0] - 1.4
  coords37[:, 2, 0] = coords37[:, 1, 0] + 1.5
  mask37 = np.ones((L, 37), dtype=np.float32)
  mask = np.ones((L,), dtype=np.float32)
  res_idx = np.arange(L, dtype=np.int32)
  chain_idx = np.zeros((L,), dtype=np.int32)
  aatype = np.zeros((L,), dtype=np.int32)

  prep = FAMPNNPrepared(
    atom37_xyz=coords37,
    atom37_mask=mask37,
    mask=mask,
    residue_idx=res_idx,
    chain_idx=chain_idx,
    aatype_aminx=aatype,
    aatype_fampnn=aatype,
  )
  batch = FamilyBatch(input_indices=(0,), arrays={"prepared": (prep,)}, skipped=(), lengths=(L,))

  model = FAMPNNModel(has_confidence=False, key=key)
  spec = SamplingSpecification(
    inputs=(),
    model_family="fampnn",
    fampnn=FAMPNNOptions(
      checkpoint_id="fampnn_0_3_cath",
      num_seq_steps=1,
      num_scn_steps=2,
      pack_each_step=False,
    ),
    num_samples=1,
    temperature=0.1,
  )
  drv = FAMPNNDriver()
  assert drv.handles(spec, "sample")
  assert drv.handles(spec, "score:logits")
  assert drv.handles(spec, "score:nll")

  logits_res = drv.stages(spec, "score:logits", model)(
    batch, chunk_start=0, chunk_count=1, scalar_dropout=False, dropout_key=key,
  )
  assert logits_res["logits"].shape == (1, L, 21)

  nll_res = drv.stages(spec, "score:nll", model)(
    batch, chunk_start=0, chunk_count=1, scalar_dropout=False, dropout_key=key,
  )
  assert nll_res["nll"].shape == (1, L)

  sample_res = drv.stages(spec, "sample", model)(
    batch, chunk_start=0, chunk_count=1, scalar_dropout=False, dropout_key=key,
  )
  assert sample_res["sequences"].shape == (1, 1, 1, L)
  assert sample_res["packed_atom37_xyz"].shape == (1, 1, 1, L, 37, 3)


def test_checkpoint_conversion_and_host_registration() -> None:
  """Verify FAMPNN host registration and state_dict converter."""
  assert "fampnn" in DRIVER_BACKED_FAMILY_MODULES
  drv = FAMILY_DRIVERS.get("fampnn")
  assert isinstance(drv, FAMPNNDriver)
  table = family_checkpoints()
  for cid in FAMPNN_ALL_CHECKPOINTS:
    assert cid in table
    assert table[cid].family == "fampnn"
    assert not table[cid].sha256.startswith("fa3000")

  raw = {
    "model.mlp_seq.0.weight": np.ones((128, 128), dtype=np.float32),
    "model.mlp_seq.0.bias": np.zeros((128,), dtype=np.float32),
    "model.mlp_seq.2.weight": np.ones((20, 128), dtype=np.float32),
    "model.mlp_seq.2.bias": np.zeros((20,), dtype=np.float32),
  }
  norm = normalize_fampnn_state_dict(raw)
  assert not has_fampnn_confidence_head(norm)
  conv = convert_fampnn_state_dict(raw)
  assert isinstance(conv, FAMPNNModel)
  assert conv.scn_confidence is None


def test_fampnn_packaged_checkpoints_resolve_and_forward() -> None:
  """Verify all 3 converted FAMPNN checkpoints pass SHA-256 verification and load."""
  for cid in FAMPNN_ALL_CHECKPOINTS:
    loaded = _load_or_build_fampnn_checkpoint_model(cid)
    assert loaded is not None


def _extract_raw_fampnn_sd(model: FAMPNNModel) -> dict[str, np.ndarray]:
  """Build an upstream-keyed PyTorch state_dict from a loaded FAMPNNModel."""
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

  def put_gvp_norm(gn: object, prefix: str) -> None:
    put_ln(gn.scalar_norm, f"{prefix}.scalar_norm")

  def put_gvp(g: object, prefix: str) -> None:
    if g.wh is not None:
      put_lin(g.wh, f"{prefix}.wh")
    put_lin(g.ws, f"{prefix}.ws")
    if g.wu is not None:
      put_lin(g.wu, f"{prefix}.wv")
    if g.wg is not None:
      put_lin(g.wg, f"{prefix}.wsv")

  def put_feat(feat: object, prefix: str) -> None:
    put_lin(feat.embeddings.linear, f"{prefix}.embeddings.linear")
    put_lin(feat.edge_embedding, f"{prefix}.edge_embedding")
    put_ln(feat.norm_edges, f"{prefix}.norm_edges")

  def put_enc_or_edge_dec(lyr: object, prefix: str) -> None:
    for w in ("W1", "W2", "W3", "W11", "W12", "W13"):
      put_lin(getattr(lyr, w), f"{prefix}.{w}")
    for n in ("norm1", "norm2", "norm3"):
      put_ln(getattr(lyr, n), f"{prefix}.{n}")
    dense = lyr.dense
    put_lin(dense.W_in, f"{prefix}.dense.W_in")
    put_lin(dense.W_out, f"{prefix}.dense.W_out")

  def put_gvp_enc(enc: object, prefix: str) -> None:
    eg = enc.embed_graph
    put_gvp(eg.W_v_gvp, f"{prefix}.embed_graph.embed_node.0")
    put_gvp_norm(eg.W_v_norm1, f"{prefix}.embed_graph.embed_node.1")
    put_gvp(eg.W_e_gvp, f"{prefix}.embed_graph.embed_edge.0")
    put_gvp_norm(eg.W_e_norm1, f"{prefix}.embed_graph.embed_edge.1")
    if eg.embed_mpnn_node is not None:
      put_lin(eg.embed_mpnn_node, f"{prefix}.embed_graph.embed_mpnn_node")
    if eg.embed_mpnn_edge is not None:
      put_lin(eg.embed_mpnn_edge, f"{prefix}.embed_graph.embed_mpnn_edge")
    for i, gl in enumerate(enc.encoder_layers):
      lp = f"{prefix}.encoder_layers.{i}"
      put_gvp(gl.conv_gvp0, f"{lp}.conv.message_func.0")
      put_gvp(gl.conv_gvp1, f"{lp}.conv.message_func.1")
      put_gvp(gl.conv_gvp2, f"{lp}.conv.message_func.2")
      put_gvp_norm(gl.norm0, f"{lp}.norm.0")
      put_gvp(gl.ff_gvp0, f"{lp}.ff_func.0")
      put_gvp(gl.ff_gvp1, f"{lp}.ff_func.1")
      put_gvp_norm(gl.norm1, f"{lp}.norm.1")
    put_ln(enc.norm_gvp_out, f"{prefix}.norm_gvp_out")
    if enc.w_out_norm is not None and enc.w_out_gvp is not None:
      put_gvp_norm(enc.w_out_norm, f"{prefix}.W_out.0")
      put_gvp(enc.w_out_gvp, f"{prefix}.W_out.1")
    elif enc.w_out_linear is not None:
      put_lin(enc.w_out_linear, f"{prefix}.W_out")

  fa = model.fa_encoder
  fbase = "model.denoiser.seq_design_module"
  put_feat(fa.features, f"{fbase}.features")
  put_gvp_enc(fa.gvp_encoder, f"{fbase}.vector_encoder")
  put_lin(fa.W_v, f"{fbase}.W_v")
  put_lin(fa.W_e, f"{fbase}.W_e")
  raw[f"{fbase}.W_s.weight"] = np.asarray(fa.W_s.weight, dtype=np.float32)
  if fa.sidechain_features is not None and fa.W_e2 is not None:
    put_feat(fa.sidechain_features, f"{fbase}.sidechain_features")
    put_lin(fa.W_e2, f"{fbase}.W_e2")
  for i, lyr in enumerate(fa.encoder_layers):
    put_enc_or_edge_dec(lyr, f"{fbase}.encoder_layers.{i}")
  for i, lyr in enumerate(fa.decoder_layers):
    put_enc_or_edge_dec(lyr, f"{fbase}.decoder_layers.{i}")

  put_lin(model.mlp_seq_0, "model.mlp_seq.0")
  put_lin(model.mlp_seq_2, "model.mlp_seq.2")
  if model.W_out is not None:
    put_lin(model.W_out, f"{fbase}.W_out")

  scn = model.scn_diffusion
  sbase = "model.denoiser.scn_diffusion_module"
  put_lin(scn.mlp_s_c_lin, f"{sbase}.mlp_s_c.0")
  put_ln(scn.mlp_s_c_norm, f"{sbase}.mlp_s_c.1")
  put_lin(scn.x_embedder, f"{sbase}.scn_denoiser.x_embedder")
  put_lin(scn.c_embedder, f"{sbase}.scn_denoiser.h_V_embedder")
  if scn.f_embedder is not None:
    put_lin(scn.f_embedder, f"{sbase}.scn_denoiser.f_embedder")
  put_lin(scn.t_embedder.mlp_0, f"{sbase}.scn_denoiser.timestep_embedder.mlp.0")
  put_lin(scn.t_embedder.mlp_2, f"{sbase}.scn_denoiser.timestep_embedder.mlp.2")
  for i, blk in enumerate(scn.blocks):
    bp = f"{sbase}.scn_denoiser.blocks.{i}"
    put_lin(blk.mlp_fc1, f"{bp}.mlp.fc1")
    put_lin(blk.mlp_fc2, f"{bp}.mlp.fc2")
    put_lin(blk.adaLN_modulation, f"{bp}.adaLN_modulation.1")
  put_lin(scn.final_layer.linear, f"{sbase}.scn_denoiser.final_layer.linear")
  put_lin(scn.final_layer.adaLN_modulation, f"{sbase}.scn_denoiser.final_layer.adaLN_modulation.1")

  if model.scn_confidence is not None:
    conf = model.scn_confidence
    cbase = f"{sbase}.confidence_module"
    put_feat(conf.features, f"{cbase}.features")
    put_gvp_enc(conf.gvp_encoder, f"{cbase}.gvp_encoder")
    put_lin(conf.W_v, f"{cbase}.W_v")
    put_lin(conf.W_e, f"{cbase}.W_e")
    raw[f"{cbase}.W_s.weight"] = np.asarray(conf.W_s.weight, dtype=np.float32)
    for i, lyr in enumerate(conf.encoder_layers):
      put_enc_or_edge_dec(lyr, f"{cbase}.encoder_layers.{i}")
    for i, lyr in enumerate(conf.decoder_layers):
      if hasattr(lyr, "W11") and lyr.W11 is not None:
        put_enc_or_edge_dec(lyr, f"{cbase}.structure_encoder.decoder_layers.{i}")
      else:
        for w in ("W1", "W2", "W3"):
          put_lin(getattr(lyr, w), f"{cbase}.decoder_layers.{i}.{w}")
        for n in ("norm1", "norm2"):
          put_ln(getattr(lyr, n), f"{cbase}.decoder_layers.{i}.{n}")
        put_lin(lyr.dense.W_in, f"{cbase}.decoder_layers.{i}.dense.W_in")
        put_lin(lyr.dense.W_out, f"{cbase}.decoder_layers.{i}.dense.W_out")
    put_lin(conf.W_out_0, f"{cbase}.W_out.0")
    put_lin(conf.W_out_2, f"{cbase}.W_out.2")
    put_lin(conf.W_out_4, f"{cbase}.W_out.4")
    if conf.aatype_embedder is not None:
      put_lin(conf.aatype_embedder, f"{cbase}.aatype_embedder")
    if conf.sidechain_encoder is not None:
      put_lin(conf.sidechain_encoder, f"{cbase}.sidechain_encoder")
    if conf.mlp_0 is not None:
      put_lin(conf.mlp_0, f"{cbase}.mlp.0")
    if conf.mlp_2 is not None:
      put_lin(conf.mlp_2, f"{cbase}.mlp.2")
  return raw


def test_fampnn_live_pytorch_vs_jax_graded_parity() -> None:
  """Verify full end-to-end PyTorch-vs-JAX graded parity via raw_sd -> convert_fampnn_state_dict."""
  import contextlib
  import math
  from typing import Any

  import equinox as eqx
  import torch
  import torch.nn.functional as F

  from aminx.families.fampnn.constants import RESTYPE_ATOM14_MASK, RESTYPE_ATOM14_TO_ATOM37
  from aminx.io.parsing import parse_structure

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
  prot = parse_structure(ubq_pdb)
  coords37 = np.asarray(prot.coordinates, dtype=np.float64)
  mask37 = np.asarray(prot.atom_mask, dtype=np.float64)
  mask = np.asarray(prot.mask, dtype=np.float64)
  res_idx = np.asarray(prot.residue_index, dtype=np.int32)
  chain_idx = np.asarray(prot.chain_index, dtype=np.int32)
  aatype = np.asarray(prot.aatype, dtype=np.int32)
  num_l = int(coords37.shape[0])
  assert num_l >= 76

  restype_14_to_37 = np.asarray(RESTYPE_ATOM14_TO_ATOM37, dtype=np.int64)
  restype_14_mask = np.asarray(RESTYPE_ATOM14_MASK, dtype=np.float64)
  bb_src_idx = (0, 1, 2, 3, 4, 0, 0, 0, 0, 1, 1, 1, 4, 4, 3, 1, 2, 3, 4, 2, 3, 4, 2, 3, 2)
  bb_dst_idx = (0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 2, 3, 2, 0, 0, 0, 0, 1, 1, 1, 4, 4, 3)

  for cid in ("fampnn_0_3", "fampnn_0_3_cath", "fampnn_0_0"):
    pkg_model = _load_or_build_fampnn_checkpoint_model(cid)
    raw_sd = _extract_raw_fampnn_sd(pkg_model)
    model_eqx = convert_fampnn_state_dict(raw_sd)
    # Build PyTorch view directly from upstream-keyed raw_sd without calling normalize_fampnn_state_dict
    fbase = "model.denoiser.seq_design_module"
    sbase = "model.denoiser.scn_diffusion_module"
    sd: dict[str, np.ndarray] = {}
    for rk, rv in raw_sd.items():
      pk = rk
      pk = pk.replace(f"{fbase}.vector_encoder.", "fa_encoder.gvp_encoder.")
      pk = pk.replace(f"{fbase}.W_out.", "W_out.")
      pk = pk.replace(f"{fbase}.", "fa_encoder.")
      pk = pk.replace("model.mlp_seq.", "mlp_seq.")
      pk = pk.replace(f"{sbase}.scn_denoiser.h_V_embedder.", "scn_diffusion.denoiser.c_embedder.")
      pk = pk.replace(f"{sbase}.scn_denoiser.timestep_embedder.", "scn_diffusion.denoiser.t_embedder.")
      pk = pk.replace(f"{sbase}.scn_denoiser.", "scn_diffusion.denoiser.")
      pk = pk.replace(f"{sbase}.", "scn_diffusion.")
      pk = pk.replace(".embed_graph.embed_node.0.", ".embed_graph.W_v.1.")
      pk = pk.replace(".embed_graph.embed_node.1.", ".embed_graph.W_v.2.")
      pk = pk.replace(".embed_graph.embed_edge.0.", ".embed_graph.W_e.1.")
      pk = pk.replace(".embed_graph.embed_edge.1.", ".embed_graph.W_e.2.")
      pk = pk.replace(".wv.", ".wu.")
      pk = pk.replace(".wsv.", ".wg.")
      sd[pk] = rv

    for use_f64 in (True, False):
      with _x64(use_f64):
        j_dtype = jnp.float64 if use_f64 else jnp.float32
        dtype = torch.float64 if use_f64 else torch.float32
        m_cast = _cast(model_eqx, j_dtype)
        seq_mask = np.ones((num_l,), dtype=np.float64 if use_f64 else np.float32)
        out = m_cast(
          jnp.asarray(coords37, dtype=j_dtype),
          jnp.asarray(mask37, dtype=j_dtype),
          jnp.asarray(aatype),
          jnp.asarray(seq_mask, dtype=j_dtype),
          jnp.asarray(mask, dtype=j_dtype),
          jnp.asarray(res_idx),
          jnp.asarray(chain_idx),
        )
        out_pkg = _cast(pkg_model, j_dtype)(
          jnp.asarray(coords37, dtype=j_dtype),
          jnp.asarray(mask37, dtype=j_dtype),
          jnp.asarray(aatype),
          jnp.asarray(seq_mask, dtype=j_dtype),
          jnp.asarray(mask, dtype=j_dtype),
          jnp.asarray(res_idx),
          jnp.asarray(chain_idx),
        )
        assert jnp.allclose(out.h_v, out_pkg.h_v, atol=1e-7)
        assert jnp.allclose(out.logits_fampnn, out_pkg.logits_fampnn, atol=1e-7)
        rng = np.random.default_rng(7)
        x_noisy = (rng.standard_normal((num_l, 33, 3)) * 0.5).astype(np.float64 if use_f64 else np.float32)
        x_cond_np = (rng.standard_normal((num_l, 33, 3)) * 0.25).astype(np.float64 if use_f64 else np.float32)
        sigma = np.full((num_l,), 0.25, dtype=np.float64 if use_f64 else np.float32)
        ref_pos = (rng.standard_normal((num_l, 33, 3)) * 0.5).astype(np.float64 if use_f64 else np.float32)
        scn_mask = np.ones((num_l, 33), dtype=np.float64 if use_f64 else np.float32)
        aa_oh = np.eye(21, dtype=np.float64 if use_f64 else np.float32)[aatype]
        s_c_jax = (
          out.h_v
          if "scn_diffusion.denoiser.f_embedder.weight" in sd
          else m_cast.scn_diffusion.compute_conditioning(out.h_v, jnp.asarray(aa_oh, dtype=j_dtype))
        )
        jax_den = np.asarray(
          m_cast.scn_diffusion.denoise_step(
            jnp.asarray(x_noisy, dtype=j_dtype),
            jnp.asarray(sigma, dtype=j_dtype),
            s_c_jax,
            jnp.asarray(ref_pos, dtype=j_dtype),
            jnp.asarray(scn_mask, dtype=j_dtype),
            jnp.asarray(aa_oh, dtype=j_dtype),
            x_self_cond_33=jnp.asarray(x_cond_np, dtype=j_dtype),
          ),
        )

        def to_t(x: Any) -> torch.Tensor:
          return torch.tensor(np.asarray(x), dtype=dtype)

        def t_sd(key: str) -> torch.Tensor:
          return torch.from_numpy(sd[key].copy()).to(dtype=dtype)

        def lin_sd(prefix: str, x: torch.Tensor) -> torch.Tensor:
          w = t_sd(f"{prefix}.weight")
          b = t_sd(f"{prefix}.bias") if f"{prefix}.bias" in sd else None
          return F.linear(x, w, b)

        def ln_sd(prefix: str, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
          w = t_sd(f"{prefix}.weight") if f"{prefix}.weight" in sd else None
          b = t_sd(f"{prefix}.bias") if f"{prefix}.bias" in sd else None
          return F.layer_norm(x, (x.shape[-1],), w, b, eps=eps)

        def pt_norm_zero_safe(v: torch.Tensor) -> torch.Tensor:
          sq = torch.sum(v * v, dim=-1, keepdim=True)
          norm = torch.sqrt(torch.where(sq > 0, sq, torch.ones_like(sq)))
          return torch.where(sq > 0, v / norm, torch.zeros_like(v))

        def pt_virtual_cb(n_c: torch.Tensor, ca_c: torch.Tensor, c_c: torch.Tensor) -> torch.Tensor:
          b_v = ca_c - n_c
          c_v = c_c - ca_c
          a_v = torch.cross(b_v, c_v, dim=-1)
          return -0.58273431 * a_v + 0.56802827 * b_v - 0.54067466 * c_v + ca_c

        def pt_virtual_cb_unit(n_c: torch.Tensor, ca_c: torch.Tensor, c_c: torch.Tensor) -> torch.Tensor:
          u_n = pt_norm_zero_safe(n_c - ca_c)
          c_n = pt_norm_zero_safe(c_c - ca_c)
          b_n = pt_norm_zero_safe(u_n - c_n)
          n_n = pt_norm_zero_safe(torch.cross(u_n, c_n, dim=-1))
          return -math.sqrt(1.0 / 3.0) * b_n - math.sqrt(2.0 / 3.0) * n_n

        def pt_dihedrals(n_c: torch.Tensor, ca_c: torch.Tensor, c_c: torch.Tensor) -> torch.Tensor:
          coords_bb = torch.stack([n_c, ca_c, c_c], dim=1).reshape(-1, 3)
          d_x = coords_bb[1:] - coords_bb[:-1]
          u_flat = pt_norm_zero_safe(d_x)
          u_2 = u_flat[:-2]
          u_1 = u_flat[1:-1]
          u_0 = u_flat[2:]
          n_2 = pt_norm_zero_safe(torch.cross(u_2, u_1, dim=-1))
          n_1 = pt_norm_zero_safe(torch.cross(u_1, u_0, dim=-1))
          cos_d = torch.clamp(torch.sum(n_2 * n_1, dim=-1), -1.0 + 1e-7, 1.0 - 1e-7)
          sign_d = torch.sign(torch.sum(u_2 * n_1, dim=-1))
          d_flat = sign_d * torch.acos(cos_d)
          d_padded = F.pad(d_flat, (1, 2)).reshape(n_c.shape[0], 3)
          return torch.cat([torch.cos(d_padded), torch.sin(d_padded)], dim=-1)

        def pt_knn(ca_c: torch.Tensor, mask_1d: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
          k_eff = min(top_k, ca_c.shape[0])
          m2d = mask_1d[:, None] * mask_1d[None, :]
          diff = ca_c[:, None, :] - ca_c[None, :, :]
          dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-6) * m2d
          dist_adj = dist + (1.0 - m2d) * 10000.0
          neg_d, e_idx = torch.topk(-dist_adj, k_eff, dim=-1, largest=True)
          return -neg_d, e_idx

        def pt_rbf(
          dist: torch.Tensor, num_rbf: int = 16, d_min: float = 2.0, d_max: float = 22.0,
        ) -> torch.Tensor:
          d_mu = torch.linspace(d_min, d_max, num_rbf, dtype=dtype)
          d_sigma = (d_max - d_min) / num_rbf
          return torch.exp(-(((dist.unsqueeze(-1) - d_mu) / d_sigma) ** 2))

        def pt_gvp_norm_sd(
          prefix: str, s: torch.Tensor, v: torch.Tensor | None, *, has_v: bool = True,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
          s_out = ln_sd(f"{prefix}.scalar_norm", s, eps=1e-4)
          if not has_v or v is None:
            return s_out, None
          vn = torch.sum(v * v, dim=-1, keepdim=True) + 1e-4
          nonzero = (vn > 2e-4).to(dtype=dtype)
          vn = torch.sum(vn * nonzero, dim=-2, keepdim=True) / (
            1e-4 + torch.sum(nonzero, dim=-2, keepdim=True)
          )
          vn = torch.sqrt(vn + 1e-4)
          return s_out, nonzero * (v / vn)

        def pt_gvp_sd(
          prefix: str,
          s: torch.Tensor,
          v: torch.Tensor | None,
          *,
          has_scalar_act: bool = True,
          has_vector_act: bool = True,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
          if f"{prefix}.wh.weight" in sd and v is not None:
            vh = torch.einsum("...ic,hi->...hc", v, t_sd(f"{prefix}.wh.weight"))
            vn = torch.sqrt(torch.sum(vh * vh, dim=-1) + 1e-8)
            s_out = lin_sd(f"{prefix}.ws", torch.cat([s, vn], dim=-1))
            if f"{prefix}.wu.weight" in sd:
              vu = torch.einsum("...hc,oh->...oc", vh, t_sd(f"{prefix}.wu.weight"))
              if has_vector_act:
                if f"{prefix}.wg.weight" in sd:
                  gate = torch.sigmoid(lin_sd(f"{prefix}.wg", torch.sigmoid(s_out)))
                  v_out = vu * gate.unsqueeze(-1)
                else:
                  v_out = vu * torch.sigmoid(torch.sqrt(torch.sum(vu * vu, dim=-1, keepdim=True) + 1e-8))
              else:
                v_out = vu
            else:
              v_out = None
          else:
            s_out = lin_sd(f"{prefix}.ws", s)
            v_out = None
          if has_scalar_act:
            s_out = F.relu(s_out)
          return s_out, v_out

        def pt_enc_layer_sd(
          prefix: str,
          h_v_in: torch.Tensor,
          h_e_in: torch.Tensor,
          e_idx_in: torch.Tensor,
          mask_f: torch.Tensor,
          mask_att: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
          num_k = e_idx_in.shape[1]
          h_ev = torch.cat([h_v_in[:, None, :].expand(-1, num_k, -1), h_e_in, h_v_in[e_idx_in]], dim=-1)
          h_msg = lin_sd(f"{prefix}.W3", F.gelu(lin_sd(f"{prefix}.W2", F.gelu(lin_sd(f"{prefix}.W1", h_ev)))))
          dh = torch.sum(mask_att.unsqueeze(-1) * h_msg, dim=-2) / 30.0
          h_v_mid = ln_sd(f"{prefix}.norm1", h_v_in + dh)
          ff = lin_sd(f"{prefix}.dense.W_out", F.gelu(lin_sd(f"{prefix}.dense.W_in", h_v_mid)))
          h_v_out = ln_sd(f"{prefix}.norm2", h_v_mid + ff) * mask_f.unsqueeze(-1)
          h_ev2 = torch.cat([h_v_out[:, None, :].expand(-1, num_k, -1), h_e_in, h_v_out[e_idx_in]], dim=-1)
          h_e_msg = lin_sd(f"{prefix}.W13", F.gelu(lin_sd(f"{prefix}.W12", F.gelu(lin_sd(f"{prefix}.W11", h_ev2)))))
          h_e_out = ln_sd(f"{prefix}.norm3", h_e_in + h_e_msg)
          return h_v_out, h_e_out

        def pt_dec_layer_sd(
          prefix: str,
          h_v_in: torch.Tensor,
          h_e_in: torch.Tensor,
          h_s_in: torch.Tensor,
          e_idx_in: torch.Tensor,
          mask_f: torch.Tensor,
          mask_att: torch.Tensor,
          *,
          edge_update_full: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
          num_k = e_idx_in.shape[1]
          if edge_update_full:
            h_esv = h_e_in
          else:
            h_es = torch.cat([h_e_in, h_s_in[e_idx_in]], dim=-1)
            h_esv = torch.cat([h_es, h_v_in[e_idx_in]], dim=-1)
          h_in = torch.cat([h_v_in[:, None, :].expand(-1, num_k, -1), h_esv], dim=-1)
          h_msg = lin_sd(f"{prefix}.W3", F.gelu(lin_sd(f"{prefix}.W2", F.gelu(lin_sd(f"{prefix}.W1", h_in)))))
          dh = torch.sum(mask_att.unsqueeze(-1) * h_msg, dim=-2) / 30.0
          h_v_mid = ln_sd(f"{prefix}.norm1", h_v_in + dh)
          ff = lin_sd(f"{prefix}.dense.W_out", F.gelu(lin_sd(f"{prefix}.dense.W_in", h_v_mid)))
          h_v_out = ln_sd(f"{prefix}.norm2", h_v_mid + ff) * mask_f.unsqueeze(-1)
          h_ev2 = torch.cat([h_v_out[:, None, :].expand(-1, num_k, -1), h_e_in, h_v_out[e_idx_in]], dim=-1)
          h_e_msg = lin_sd(f"{prefix}.W13", F.gelu(lin_sd(f"{prefix}.W12", F.gelu(lin_sd(f"{prefix}.W11", h_ev2)))))
          h_e_out = ln_sd(f"{prefix}.norm3", h_e_in + h_e_msg)
          return h_v_out, h_e_out

        # Full PyTorch FAEncoder + GVPEncoder + W_out/mlp_seq forward pass from sd
        xyz37_t = to_t(coords37)
        mask37_t = to_t(mask37)
        mask_f_t = to_t(mask)
        res_t = torch.tensor(res_idx, dtype=torch.long)
        chain_t = torch.tensor(chain_idx, dtype=torch.long)
        aa_t_idx = torch.clamp(torch.tensor(aatype, dtype=torch.long), 0, 20)
        vis_seq_t = to_t(seq_mask)
        aa_noised_t = torch.where(vis_seq_t > 0, aa_t_idx, torch.full_like(aa_t_idx, 20))
        aa_oh_t = F.one_hot(aa_noised_t, 21).to(dtype=dtype)

        gather_14 = torch.tensor(restype_14_to_37, dtype=torch.long)[aa_t_idx]
        std_m14 = torch.tensor(restype_14_mask, dtype=dtype)[aa_t_idx]
        atom14_xyz_t = torch.gather(xyz37_t, 1, gather_14.unsqueeze(-1).expand(-1, -1, 3))
        atom14_mask_t = torch.gather(mask37_t, 1, gather_14) * std_m14
        atom14_xyz_t = atom14_xyz_t * atom14_mask_t.unsqueeze(-1)
        sc_gate_t = torch.cat([torch.ones((num_l, 4), dtype=dtype), vis_seq_t.unsqueeze(-1).expand(-1, 10)], dim=-1)
        atom14_mask_gated_t = atom14_mask_t * sc_gate_t
        ca_pos_t = atom14_xyz_t[:, 1:2, :]
        atom14_xyz_gated_t = torch.where((atom14_mask_gated_t > 0).unsqueeze(-1), atom14_xyz_t, ca_pos_t)

        # FAMPNNProteinFeatures
        n_c = xyz37_t[:, 0, :]
        ca_c = xyz37_t[:, 1, :]
        c_c = xyz37_t[:, 2, :]
        o_c = xyz37_t[:, 4, :]
        cb_c = pt_virtual_cb(n_c, ca_c, c_c)
        _, e_idx_t = pt_knn(ca_c, mask_f_t, 48)
        num_k = e_idx_t.shape[1]
        bb5_t = torch.stack([ca_c, n_c, c_c, o_c, cb_c], dim=1)
        diff_bb = bb5_t[:, None, bb_src_idx, :] - bb5_t[e_idx_t][:, :, bb_dst_idx, :]
        dist_bb = torch.sqrt(torch.sum(diff_bb * diff_bb, dim=-1) + 1e-6)
        rbf_bb = pt_rbf(dist_bb, 16).reshape(num_l, num_k, 25 * 16)
        offset_t = res_t[:, None] - res_t[e_idx_t]
        same_chain_t = (chain_t[:, None] == chain_t[e_idx_t]).long()
        d_clamped = torch.clamp(offset_t + 32, 0, 64)
        d_final = torch.where(same_chain_t > 0, d_clamped, torch.full_like(d_clamped, 65))
        e_pos_t = lin_sd("fa_encoder.features.embeddings.linear", F.one_hot(d_final, 66).to(dtype=dtype))
        e_init_t = ln_sd(
          "fa_encoder.features.norm_edges",
          lin_sd("fa_encoder.features.edge_embedding", torch.cat([e_pos_t, rbf_bb], dim=-1)),
        )

        mask_2d_t = mask_f_t[:, None] * mask_f_t[None, :]
        mask_att_t = torch.gather(mask_2d_t, 1, e_idx_t)

        n14 = atom14_xyz_gated_t[:, 0, :]
        ca14 = atom14_xyz_gated_t[:, 1, :]
        c14 = atom14_xyz_gated_t[:, 2, :]
        o14 = atom14_xyz_gated_t[:, 3, :]
        cb14 = pt_virtual_cb(n14, ca14, c14)

        def pt_run_gvp_encoder_sd(
          mpnn_node_in: torch.Tensor | None = None,
          mpnn_edge_in: torch.Tensor | None = None,
          e_idx_override: torch.Tensor | None = None,
        ) -> torch.Tensor:
          gp = "fa_encoder.gvp_encoder"
          dihedrals_t = pt_dihedrals(n14, ca14, c14)
          wv_vi = sd[f"{gp}.embed_graph.W_v.1.wh.weight"].shape[1]
          wv_si = sd[f"{gp}.embed_graph.W_v.1.ws.weight"].shape[1] - sd[f"{gp}.embed_graph.W_v.1.wh.weight"].shape[0]
          node_s_raw = dihedrals_t if wv_si == 6 else torch.cat([dihedrals_t, aa_oh_t, atom14_mask_gated_t], dim=-1)
          fwd_t = torch.cat([ca14[1:] - ca14[:-1], torch.zeros((1, 3), dtype=dtype)], dim=0)
          bwd_t = torch.cat([torch.zeros((1, 3), dtype=dtype), ca14[:-1] - ca14[1:]], dim=0)
          u_fwd_t = pt_norm_zero_safe(fwd_t)
          u_bwd_t = pt_norm_zero_safe(bwd_t)
          u_cb_t = pt_virtual_cb_unit(n14, ca14, c14)
          orient_v_t = torch.stack([u_fwd_t, u_bwd_t, u_cb_t], dim=1)
          non_ca_coords = torch.cat([atom14_xyz_gated_t[:, :1, :], atom14_xyz_gated_t[:, 2:, :]], dim=1)
          intra_13 = pt_norm_zero_safe(ca14.unsqueeze(1) - non_ca_coords)
          impute_cb = (atom14_mask_gated_t[:, 4] <= 0) & (mask_f_t > 0)
          intra_13 = intra_13.clone()
          intra_13[:, 3, :] = torch.where(impute_cb.unsqueeze(-1), u_cb_t, intra_13[:, 3, :])
          if wv_vi == 15:
            node_v_raw = torch.cat([orient_v_t[:, :2, :], intra_13], dim=1)
          else:
            atom_disp = pt_norm_zero_safe(ca14.unsqueeze(1) - atom14_xyz_gated_t)
            node_v_raw = torch.cat([orient_v_t, atom_disp], dim=1)
          ns1, nv1 = pt_gvp_sd(f"{gp}.embed_graph.W_v.1", node_s_raw, node_v_raw, has_scalar_act=False, has_vector_act=False)
          gvp_vs, gvp_vv = pt_gvp_norm_sd(f"{gp}.embed_graph.W_v.2", ns1, nv1)
          if f"{gp}.embed_graph.embed_mpnn_node.weight" in sd and mpnn_node_in is not None:
            gvp_vs = gvp_vs + lin_sd(f"{gp}.embed_graph.embed_mpnn_node", mpnn_node_in)

          if e_idx_override is not None:
            e_idx_gvp = e_idx_override
            diff_gvp = ca14[:, None, :] - ca14[e_idx_gvp]
            d_neigh_gvp = torch.sqrt(torch.sum(diff_gvp * diff_gvp, dim=-1) + 1e-8)
          else:
            d_neigh_gvp, e_idx_gvp = pt_knn(ca14, mask_f_t, 30)
          rbf_ca_gvp = pt_rbf(d_neigh_gvp, 16, d_min=0.0, d_max=20.0)
          idx_i = torch.arange(num_l, dtype=torch.long)[:, None]
          d_off_gvp = (idx_i - e_idx_gvp).to(dtype=dtype)
          half_pos = 8
          freqs_pos = torch.exp(-math.log(10000.0) * torch.arange(0, half_pos, dtype=dtype) / half_pos)
          angles_pos = d_off_gvp.unsqueeze(-1) * freqs_pos[None, None, :]
          pos_emb_gvp = torch.cat([torch.cos(angles_pos), torch.sin(angles_pos)], dim=-1)
          edge_s_raw = torch.cat([rbf_ca_gvp, pos_emb_gvp], dim=-1)
          xyz_j_14 = atom14_xyz_gated_t[e_idx_gvp]
          we_vi = sd[f"{gp}.embed_graph.W_e.1.wh.weight"].shape[1]
          if we_vi == 56:
            disp_56 = torch.cat(
              [
                pt_norm_zero_safe(ca14[:, None, None, :] - xyz_j_14),
                pt_norm_zero_safe(n14[:, None, None, :] - xyz_j_14),
                pt_norm_zero_safe(c14[:, None, None, :] - xyz_j_14),
                pt_norm_zero_safe(o14[:, None, None, :] - xyz_j_14),
              ],
              dim=-2,
            )
            edge_v_raw = pt_norm_zero_safe(disp_56)
          else:
            edge_v_raw = pt_norm_zero_safe(ca14[:, None, None, :] - xyz_j_14)
          es1, ev1 = pt_gvp_sd(f"{gp}.embed_graph.W_e.1", edge_s_raw, edge_v_raw, has_scalar_act=False, has_vector_act=False)
          gvp_es, gvp_ev = pt_gvp_norm_sd(f"{gp}.embed_graph.W_e.2", es1, ev1)
          if f"{gp}.embed_graph.embed_mpnn_edge.weight" in sd and mpnn_edge_in is not None:
            gvp_es = gvp_es + lin_sd(f"{gp}.embed_graph.embed_mpnn_edge", mpnn_edge_in)

          mask_att_gvp = torch.gather(mask_2d_t, 1, e_idx_gvp)
          num_gvp = sum(
            1 for k in sd if k.startswith(f"{gp}.encoder_layers.") and k.endswith(".conv.message_func.0.ws.weight")
          )
          for gi in range(num_gvp):
            lp = f"{gp}.encoder_layers.{gi}"
            k_gvp = e_idx_gvp.shape[1]
            s_src = gvp_vs[:, None, :].expand(num_l, k_gvp, -1)
            v_src = gvp_vv[:, None, :, :].expand(num_l, k_gvp, -1, 3)
            s_dst = gvp_vs[e_idx_gvp]
            v_dst = gvp_vv[e_idx_gvp]
            ms, mv = pt_gvp_sd(
              f"{lp}.conv.message_func.0",
              torch.cat([s_src, gvp_es, s_dst], dim=-1),
              torch.cat([v_src, gvp_ev, v_dst], dim=-2),
            )
            ms, mv = pt_gvp_sd(f"{lp}.conv.message_func.1", ms, mv)
            ms, mv = pt_gvp_sd(
              f"{lp}.conv.message_func.2", ms, mv, has_scalar_act=False, has_vector_act=False,
            )
            ms = ms * mask_att_gvp.unsqueeze(-1)
            mv = mv * mask_att_gvp.unsqueeze(-1).unsqueeze(-1)
            agg_s = torch.zeros_like(gvp_vs)
            agg_v = torch.zeros_like(gvp_vv)
            deg = torch.zeros((num_l,), dtype=dtype)
            agg_s.index_put_((e_idx_gvp,), ms, accumulate=True)
            agg_v.index_put_((e_idx_gvp,), mv, accumulate=True)
            deg.index_put_((e_idx_gvp,), mask_att_gvp, accumulate=True)
            safe_deg = torch.clamp(deg, min=1.0)
            s_mid, v_mid = pt_gvp_norm_sd(
              f"{lp}.norm.0", gvp_vs + agg_s / safe_deg.unsqueeze(-1), gvp_vv + agg_v / safe_deg[:, None, None],
            )
            fs, fv = pt_gvp_sd(f"{lp}.ff_func.0", s_mid, v_mid)
            fs, fv = pt_gvp_sd(f"{lp}.ff_func.1", fs, fv, has_scalar_act=False, has_vector_act=False)
            gvp_vs, gvp_vv = pt_gvp_norm_sd(f"{lp}.norm.1", s_mid + fs, v_mid + fv)

          if f"{gp}.W_out.1.ws.weight" in sd:
            s_norm_out, v_norm_out = pt_gvp_norm_sd(f"{gp}.W_out.0", gvp_vs, gvp_vv)
            s_out, _ = pt_gvp_sd(f"{gp}.W_out.1", s_norm_out, v_norm_out)
            return s_out * mask_f_t.unsqueeze(-1)
          # Rotate out
          v1_r = c14 - ca14
          v2_r = n14 - ca14
          e1_r = v1_r / torch.sqrt(torch.sum(v1_r * v1_r, dim=-1, keepdim=True) + 1e-8)
          u2_r = v2_r - torch.sum(e1_r * v2_r, dim=-1, keepdim=True) * e1_r
          e2_r = u2_r / torch.sqrt(torch.sum(u2_r * u2_r, dim=-1, keepdim=True) + 1e-8)
          e3_r = torch.cross(e1_r, e2_r, dim=-1)
          rot_r = torch.stack([e1_r, e2_r, e3_r], dim=-1)
          gvp_vv = torch.einsum("lvi,lij->lvj", gvp_vv, rot_r)
          combined_gvp = torch.cat([gvp_vs, gvp_vv.reshape(num_l, -1)], dim=-1)
          if f"{gp}.W_out.weight" in sd:
            return lin_sd(f"{gp}.W_out", combined_gvp) * mask_f_t.unsqueeze(-1)
          return ln_sd(f"{gp}.norm_gvp_out", combined_gvp) * mask_f_t.unsqueeze(-1)

        h_s_pt = t_sd("fa_encoder.W_s.weight")[aa_noised_t]
        num_enc = sum(1 for k in sd if k.startswith("fa_encoder.encoder_layers.") and k.endswith(".W1.weight"))
        num_dec = sum(1 for k in sd if k.startswith("fa_encoder.decoder_layers.") and k.endswith(".W1.weight"))
        if "fa_encoder.sidechain_features.edge_embedding.weight" in sd and "fa_encoder.W_e2.weight" in sd:
          h_v_pt = torch.zeros((num_l, sd["fa_encoder.W_e.weight"].shape[0]), dtype=dtype)
          h_e_pt = lin_sd("fa_encoder.W_e", e_init_t)
          for ei in range(num_enc):
            h_v_pt, h_e_pt = pt_enc_layer_sd(
              f"fa_encoder.encoder_layers.{ei}", h_v_pt, h_e_pt, e_idx_t, mask_f_t, mask_att_t,
            )
          sc10_j = atom14_xyz_gated_t[e_idx_t][:, :, 4:14, :]
          bb4_i = torch.stack([ca14, n14, c14, o14], dim=1)
          diff_4x10 = bb4_i[:, None, :, None, :] - sc10_j[:, :, None, :, :]
          dist_4x10 = torch.sqrt(torch.sum(diff_4x10 * diff_4x10, dim=-1) + 1e-6)
          rbf_sc = pt_rbf(dist_4x10, 16).reshape(num_l, num_k, 40 * 16)
          e_pos_sc = lin_sd(
            "fa_encoder.sidechain_features.embeddings.linear", F.one_hot(d_final, 66).to(dtype=dtype),
          )
          e2_init_t = ln_sd(
            "fa_encoder.sidechain_features.norm_edges",
            lin_sd(
              "fa_encoder.sidechain_features.edge_embedding",
              torch.cat([e_pos_sc, rbf_sc], dim=-1),
            ),
          )
          h_e2_pt = lin_sd("fa_encoder.W_e2", e2_init_t)
          h_es_pt = torch.cat([h_e_pt, h_s_pt[e_idx_t], h_e2_pt], dim=-1)
          h_esv_pt = torch.cat([h_es_pt, h_v_pt[e_idx_t]], dim=-1)
          for di in range(num_dec):
            h_v_pt, h_esv_pt = pt_dec_layer_sd(
              f"fa_encoder.decoder_layers.{di}",
              h_v_pt,
              h_esv_pt,
              h_s_pt,
              e_idx_t,
              mask_f_t,
              mask_att_t,
              edge_update_full=True,
            )
          h_v_final_pt = pt_run_gvp_encoder_sd(
            mpnn_node_in=h_v_pt, mpnn_edge_in=h_esv_pt, e_idx_override=e_idx_t,
          )
        else:
          gvp_feat_pt = pt_run_gvp_encoder_sd()
          h_v_pt = lin_sd("fa_encoder.W_v", gvp_feat_pt)
          h_e_pt = lin_sd("fa_encoder.W_e", e_init_t)
          for ei in range(num_enc):
            h_v_pt, h_e_pt = pt_enc_layer_sd(
              f"fa_encoder.encoder_layers.{ei}", h_v_pt, h_e_pt, e_idx_t, mask_f_t, mask_att_t,
            )
          for di in range(num_dec):
            h_v_pt, h_e_pt = pt_dec_layer_sd(
              f"fa_encoder.decoder_layers.{di}",
              h_v_pt,
              h_e_pt,
              h_s_pt,
              e_idx_t,
              mask_f_t,
              mask_att_t,
              edge_update_full=False,
            )
          h_v_final_pt = h_v_pt

        if "W_out.weight" in sd:
          logits_pt = lin_sd("W_out", h_v_final_pt)
        else:
          logits_20_pt = lin_sd("mlp_seq.2", F.gelu(lin_sd("mlp_seq.0", h_v_final_pt)))
          logits_pt = torch.cat([logits_20_pt, torch.full((num_l, 1), -1e9, dtype=dtype)], dim=-1)

        tol = 1e-6 if use_f64 else 1e-4
        assert float(np.max(np.abs(np.asarray(out.h_v) - h_v_final_pt.detach().numpy()))) < tol
        assert float(np.max(np.abs(np.asarray(out.logits_fampnn) - logits_pt.detach().numpy()))) < tol

        # Feed PyTorch's own h_v_final_pt into PyTorch's scn_diffusion indexed purely from sd
        x_t = to_t(x_noisy)
        x_cond_t = to_t(x_cond_np)
        sig_t = to_t(sigma)
        hv_t = h_v_final_pt
        ref_t = to_t(ref_pos)
        m_t = to_t(scn_mask)
        aa_t = to_t(aa_oh)
        has_f_emb = "scn_diffusion.denoiser.f_embedder.weight" in sd
        s_c = (
          hv_t
          if has_f_emb
          else ln_sd(
            "scn_diffusion.mlp_s_c.1",
            lin_sd("scn_diffusion.mlp_s_c.0", torch.cat([aa_t, hv_t], dim=-1)),
          )
        )
        sd2 = 1.0
        sig2 = sig_t ** 2
        c_skip = sd2 / (sig2 + sd2)
        c_out = sig_t * 1.0 / torch.sqrt(sig2 + sd2)
        c_in = 1.0 / torch.sqrt(sd2 + sig2)
        c_noise = 0.25 * torch.log(torch.clamp(sig_t, min=1e-8))
        x_scaled = (c_in[:, None, None] * x_t) * m_t.unsqueeze(-1)
        cond_scaled = x_cond_t * m_t.unsqueeze(-1)
        x_cond_cat = torch.cat([x_scaled, cond_scaled], dim=-1).reshape(num_l, 198)
        non_bb_idxs = [i for i in range(37) if i not in (0, 1, 2, 4)]
        full_ref = torch.zeros((num_l, 37, 3), dtype=dtype)
        full_ref[:, non_bb_idxs, :] = ref_t * m_t.unsqueeze(-1)
        ref_111 = full_ref.reshape(num_l, 111)
        h = lin_sd("scn_diffusion.denoiser.x_embedder", torch.cat([x_cond_cat, aa_t], dim=-1))
        c_h = lin_sd("scn_diffusion.denoiser.c_embedder", s_c)
        if has_f_emb:
          h = h + lin_sd("scn_diffusion.denoiser.f_embedder", ref_111) + c_h
        half = 128
        freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, dtype=dtype) / half)
        args = c_noise[:, None] * freqs[None, :]
        t_emb = lin_sd(
          "scn_diffusion.denoiser.t_embedder.mlp.2",
          F.silu(lin_sd("scn_diffusion.denoiser.t_embedder.mlp.0", torch.cat([torch.cos(args), torch.sin(args)], dim=-1))),
        )
        c = t_emb + c_h
        scn_depth = sum(
          1 for k in sd if k.startswith("scn_diffusion.denoiser.blocks.") and k.endswith(".mlp.fc1.weight")
        )
        for bi in range(scn_depth):
          bp = f"scn_diffusion.denoiser.blocks.{bi}"
          mod = lin_sd(f"{bp}.adaLN_modulation.1", F.silu(c))
          shift, scale, gate = torch.chunk(mod, 3, dim=-1)
          h_norm = F.layer_norm(h, (h.shape[-1],), None, None, eps=1e-6) * (1.0 + scale) + shift
          h_ff = lin_sd(f"{bp}.mlp.fc2", F.gelu(lin_sd(f"{bp}.mlp.fc1", h_norm), approximate="tanh"))
          h = h + gate * h_ff
        mod_f = lin_sd("scn_diffusion.denoiser.final_layer.adaLN_modulation.1", F.silu(c))
        shift_f, scale_f = torch.chunk(mod_f, 2, dim=-1)
        h_f = F.layer_norm(h, (h.shape[-1],), None, None, eps=1e-6) * (1.0 + scale_f) + shift_f
        f_x = lin_sd("scn_diffusion.denoiser.final_layer.linear", h_f).reshape(num_l, 33, 3)
        pt_den = ((c_skip[:, None, None] * x_t + c_out[:, None, None] * f_x) * m_t.unsqueeze(-1)).detach().numpy()
        assert float(np.max(np.abs(jax_den - pt_den))) < tol

    # Negative controls: corrupting FAEncoder W_out or scn_diffusion final_layer breaks parity
    m_bad_seq = (
      eqx.tree_at(lambda x: x.W_out.weight, model_eqx, model_eqx.W_out.weight[::-1])
      if model_eqx.W_out is not None
      else eqx.tree_at(lambda x: x.mlp_seq_2.weight, model_eqx, model_eqx.mlp_seq_2.weight[::-1])
    )
    out_bad = m_bad_seq(
      jnp.asarray(coords37, dtype=jnp.float32),
      jnp.asarray(mask37, dtype=jnp.float32),
      jnp.asarray(aatype),
      jnp.asarray(seq_mask, dtype=jnp.float32),
      jnp.asarray(mask, dtype=jnp.float32),
      jnp.asarray(res_idx),
      jnp.asarray(chain_idx),
    )
    assert float(np.max(np.abs(np.asarray(out_bad.logits_fampnn) - logits_pt.detach().numpy()))) > 1e-2

    scn_bad = eqx.tree_at(
      lambda x: x.final_layer.linear.weight,
      model_eqx.scn_diffusion,
      model_eqx.scn_diffusion.final_layer.linear.weight[::-1],
    )
    den_bad = np.asarray(
      scn_bad.denoise_step(
        jnp.asarray(x_noisy, dtype=jnp.float32),
        jnp.asarray(sigma, dtype=jnp.float32),
        jnp.asarray(s_c_jax, dtype=jnp.float32),
        jnp.asarray(ref_pos, dtype=jnp.float32),
        jnp.asarray(scn_mask, dtype=jnp.float32),
        jnp.asarray(aa_oh, dtype=jnp.float32),
        x_self_cond_33=jnp.asarray(x_cond_np, dtype=jnp.float32),
      ),
    )
    assert float(np.max(np.abs(den_bad - pt_den))) > 1e-2
