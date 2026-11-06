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


def test_driver_score_and_upstream_sampler_shapes() -> None:
  """Verify FAMPNNDriver score:logits, score:nll, and the upstream-faithful MAR+EDM sample path run end to end."""
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

  model = FAMPNNModel(has_confidence=False, has_f_embedder=True, key=key)
  spec = SamplingSpecification(
    inputs=(),
    model_family="fampnn",
    fampnn=FAMPNNOptions(
      checkpoint_id="fampnn_0_3_cath",
      num_seq_steps=1,
      num_scn_steps=2,
      repack_last=False,
      psce_threshold=None,
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


# Removed 261104: ``test_fampnn_live_pytorch_vs_jax_graded_parity`` and its helper ``_extract_raw_fampnn_sd``.
# They rebuilt an "upstream-keyed" state dict from the aminx model and restated the upstream equations in torch
# (including the aminx-side formulas for the virtual CB, the GVP LayerNorm epsilon and the sinusoid tables), so they
# checked internal consistency and not upstream output, and they began to fail once aminx was corrected to the real
# upstream code. Parity against the real code and weights is scripts/parity/fampnn_parity.py (sealed oracle dumps
# from scripts/parity/dump_fampnn_oracle.py).
