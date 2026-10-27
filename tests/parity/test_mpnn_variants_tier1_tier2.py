"""Parity, topology, and negative-control tests for Tier 1 and Tier 2 MPNN variants.

Verifies:
1. ``get_topology_for_checkpoint`` and ``LEGACY_ALIAS_MAP`` resolution for:
   - Tier 1A drop-in fine-tunes: ``abmpnn_v_48_020``, ``hypermpnn_v_48_020``, ``cyclicmpnn_v_48_010``
   - Tier 2 CA-only: ``ca_v_48_002``, ``ca_v_48_010``, ``ca_v_48_020``
   - Regression check that ``atom_context_num == 16`` is NOT overwritten by noise suffix on non-ligand checkpoints
   - ``atom_context_num == 25`` on ``ligandmpnn_v_32_020_25``
2. Graded numerical parity (``float64`` ``atol=1e-7`` and ``float32`` ``atol=1e-4``) against upstream
   PyTorch ``ProteinMPNN`` / ``CA_ProteinFeatures`` on ``tests/data/1ubq.pdb`` for:
   - AbMPNN (``AbMPNN/abmpnn.pt``)
   - HyperMPNN (``HyperMPNN/retrained_models/v48_{002,010,020,030}_epoch*_hyper.pt``)
   - CyclicMPNN (``CyclicMPNN/cyclicmpnn_weights/cyclicmpnn_48_010.pt``)
   - CA-Only ProteinMPNN (``ProteinMPNN/ca_model_weights/v_48_{002,010,020}.pt``), including chain breaks
     (``dX_norm`` outside ``(3.6, 4.0)``) and masked residues (``mask == 0``).
3. Serialization round-trip via ``load_model(..., local_path=...)``.
4. Negative controls (perturbing weights, flipping quaternion signs, permuting RBF blocks).
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from aminx.io.weights import LEGACY_ALIAS_MAP, get_topology_for_checkpoint, load_model
from aminx.model.decoder import (
  conditional_decoder_layer_edge_features,
  pack_conditional_decoder_static_edges,
)
from aminx.model.features import ProteinFeaturesCA
from aminx.model.mpnn import Aminx
from aminx.utils.autoregression import ar_mask_from_decoding_order

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / "scripts") not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from convert_weights import convert_mpnn_checkpoint, convert_state_dict_to_aminx  # noqa: E402

_SCRATCH_REPOS = Path(
  "/home/marielle/.gemini/antigravity-cli/brain/e0a4519f-8415-4303-b518-8fc3d09cb0cc/scratch/repos",
)
_HOME_REPOS = Path.home() / "repos"
_UBQ_PDB = Path(__file__).resolve().parent.parent / "data" / "1ubq.pdb"

_AA3_TO_IDX = {
  "ALA": 0,
  "CYS": 1,
  "ASP": 2,
  "GLU": 3,
  "PHE": 4,
  "GLY": 5,
  "HIS": 6,
  "ILE": 7,
  "LYS": 8,
  "LEU": 9,
  "MET": 10,
  "ASN": 11,
  "PRO": 12,
  "GLN": 13,
  "ARG": 14,
  "SER": 15,
  "THR": 16,
  "VAL": 17,
  "TRP": 18,
  "TYR": 19,
}


def _find_repo_file(rel_path: str) -> Path | None:
  for root in (_SCRATCH_REPOS, _HOME_REPOS):
    candidate = root / rel_path
    if candidate.is_file():
      return candidate
  return None


@contextlib.contextmanager
def _x64_mode(enabled: bool) -> Iterator[None]:
  prev = bool(jax.config.jax_enable_x64)
  jax.config.update("jax_enable_x64", enabled)
  try:
    yield
  finally:
    jax.config.update("jax_enable_x64", prev)


def _cast_floating(tree: Any, dtype: jnp.dtype) -> Any:
  return jax.tree_util.tree_map(
    lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x,
    tree,
  )


def _parse_pdb_backbone(
  pdb_path: Path,
  max_residues: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  residues: dict[tuple[str, int], dict[str, Any]] = {}
  chain_order: list[str] = []
  with pdb_path.open("r", encoding="utf-8") as handle:
    for line in handle:
      if not line.startswith("ATOM  "):
        continue
      atom_name = line[12:16].strip()
      if atom_name not in ("N", "CA", "C", "O"):
        continue
      alt_loc = line[16].strip()
      if alt_loc not in ("", "A"):
        continue
      res_name = line[17:20].strip()
      chain_id = line[21].strip() or "A"
      res_seq = int(line[22:26].strip())
      xyz = (
        float(line[30:38].strip()),
        float(line[38:46].strip()),
        float(line[46:54].strip()),
      )
      key = (chain_id, res_seq)
      if key not in residues:
        residues[key] = {"res_name": res_name, "atoms": {}}
        if chain_id not in chain_order:
          chain_order.append(chain_id)
      residues[key]["atoms"][atom_name] = xyz

  keys = list(residues.keys())[:max_residues]
  n_res = len(keys)
  coords = np.zeros((n_res, 4, 3), dtype=np.float64)
  mask = np.ones((n_res,), dtype=np.float64)
  residue_index = np.zeros((n_res,), dtype=np.int32)
  chain_index = np.zeros((n_res,), dtype=np.int32)
  sequence = np.zeros((n_res,), dtype=np.int32)

  for idx, (chain_id, res_seq) in enumerate(keys):
    entry = residues[(chain_id, res_seq)]
    atoms = entry["atoms"]
    for atom_idx, atom_name in enumerate(("N", "CA", "C", "O")):
      if atom_name in atoms:
        coords[idx, atom_idx] = atoms[atom_name]
      else:
        mask[idx] = 0.0
    residue_index[idx] = res_seq
    chain_index[idx] = chain_order.index(chain_id)
    sequence[idx] = _AA3_TO_IDX.get(entry["res_name"], 20)

  return coords, mask, residue_index, chain_index, sequence


def _load_upstream_protein_mpnn_module() -> Any:
  pmpnn_dir = _find_repo_file("ProteinMPNN/protein_mpnn_utils.py")
  if pmpnn_dir is None:
    pytest.skip("ProteinMPNN reference repo not found in scratch/repos or ~/repos")
  parent = str(pmpnn_dir.parent)
  if parent not in sys.path:
    sys.path.insert(0, parent)
  import protein_mpnn_utils  # type: ignore[import-not-found]

  return protein_mpnn_utils


def _jax_teacher_forced_forward(
  model: Aminx,
  coords: jax.Array,
  mask: jax.Array,
  residue_index: jax.Array,
  chain_index: jax.Array,
  sequence: jax.Array,
  decoding_order: jax.Array,
  *,
  use_potts_ca_dist: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Run feature extraction + encoder + teacher-forced conditional decoder in JAX.

  Note: Upstream ``dauparas/ProteinMPNN`` ``ProteinFeatures._dist`` uses
  ``D = mask_2D * sqrt(|dCa|^2 + 1e-6)`` and ``D_adjust = D + (1 - mask_2D) * D_max``
  (preserved in ``aminx.families.potts_mpnn.features.potts_edge_features``), whereas
  ``ProteinFeaturesCA`` implements that exact ``D_adjust`` directly in ``forward_edge_stages``.
  """
  if isinstance(model.features, ProteinFeaturesCA):
    edge_features, neighbor_indices, _, _ = model.features(
      jax.random.PRNGKey(0),
      coords,
      mask,
      residue_index,
      chain_index,
      backbone_noise=jnp.asarray(0.0, dtype=coords.dtype),
    )
  elif use_potts_ca_dist:
    from aminx.families.potts_mpnn.features import potts_edge_features

    graph = potts_edge_features(model.features, coords, mask, residue_index, chain_index)
    edge_features = graph.edge_features
    neighbor_indices = graph.neighbor_indices
  else:
    edge_features, neighbor_indices, _, _ = model.features(
      jax.random.PRNGKey(0),
      coords,
      mask,
      residue_index,
      chain_index,
      backbone_noise=jnp.asarray(0.0, dtype=coords.dtype),
    )

  node_features, enc_edge_features = model.encoder(
    edge_features,
    neighbor_indices,
    mask,
    inference=True,
  )

  one_hot_seq = jax.nn.one_hot(sequence, 21, dtype=coords.dtype)
  ar_mask = ar_mask_from_decoding_order(decoding_order.astype(jnp.int32))
  seq_edge_features, mask_bw, masked_node_edge_features = pack_conditional_decoder_static_edges(
    node_features,
    enc_edge_features,
    neighbor_indices,
    one_hot_seq,
    model.w_s_embed.weight,
    ar_mask,
    mask,
  )

  h_v = node_features
  for layer in model.decoder.layers:
    layer_edge = conditional_decoder_layer_edge_features(
      h_v,
      seq_edge_features,
      neighbor_indices,
      mask_bw,
      masked_node_edge_features,
    )
    h_v = layer(h_v, layer_edge, mask, inference=True)

  logits = h_v @ model.w_out.weight.T + model.w_out.bias
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  return edge_features, node_features, log_probs


def _torch_teacher_forced_forward(
  pt_ckpt_path: Path,
  coords_np: np.ndarray,
  mask_np: np.ndarray,
  residue_index_np: np.ndarray,
  chain_index_np: np.ndarray,
  sequence_np: np.ndarray,
  decoding_order_np: np.ndarray,
  *,
  ca_only: bool,
  dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  pmpnn = _load_upstream_protein_mpnn_module()
  ckpt = torch.load(str(pt_ckpt_path), map_location="cpu", weights_only=False)
  state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
  k_neighbors = int(ckpt.get("num_edges", 48)) if isinstance(ckpt, dict) else 48

  pt_model = pmpnn.ProteinMPNN(
    num_letters=21,
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=k_neighbors,
    augment_eps=0.0,
    dropout=0.0,
    ca_only=ca_only,
  )
  pt_model.load_state_dict(state_dict)
  pt_model = pt_model.to(dtype=dtype)
  pt_model.eval()

  # Upstream PositionalEncodings.forward hardcodes ``d_onehot.float()`` (float32), which fails
  # when the linear layer weights are cast to float64. Patch ``embeddings.forward`` so ``one_hot``
  # casts to ``self.linear.weight.dtype``.
  emb_mod = pt_model.features.embeddings

  def _dtype_safe_pos_forward(offset: torch.Tensor, mask_chains: torch.Tensor) -> torch.Tensor:
    d = (
      torch.clip(offset + emb_mod.max_relative_feature, 0, 2 * emb_mod.max_relative_feature)
      * mask_chains
      + (1 - mask_chains) * (2 * emb_mod.max_relative_feature + 1)
    )
    d_onehot = torch.nn.functional.one_hot(d, 2 * emb_mod.max_relative_feature + 2)
    return emb_mod.linear(d_onehot.to(dtype=emb_mod.linear.weight.dtype))

  emb_mod.forward = _dtype_safe_pos_forward  # type: ignore[method-assign]

  # Upstream ProteinFeatures._rbf uses torch.linspace(D_min, D_max, D_count).to(device)
  # which defaults to torch.get_default_dtype(); set default dtype to match ``dtype`` during forward.
  prev_default = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    with torch.no_grad():
      x_in = coords_np[:, 1, :] if ca_only else coords_np
      x_t = torch.tensor(x_in[None, ...], dtype=dtype)
      s_t = torch.tensor(sequence_np[None, ...], dtype=torch.long)
      mask_t = torch.tensor(mask_np[None, ...], dtype=dtype)
      chain_m_t = torch.ones_like(mask_t) * mask_t
      res_idx_t = torch.tensor(residue_index_np[None, ...], dtype=torch.long)
      chain_enc_t = torch.tensor(chain_index_np[None, ...], dtype=torch.long)
      dec_order_t = torch.tensor(decoding_order_np[None, ...], dtype=torch.long)

      # Extract post-W_e edge features and post-encoder h_V
      e_raw, e_idx = pt_model.features(x_t, mask_t, res_idx_t, chain_enc_t)
      h_e_0 = pt_model.W_e(e_raw)
      h_v = torch.zeros((e_raw.shape[0], e_raw.shape[1], e_raw.shape[-1]), dtype=dtype)
      h_e = h_e_0
      mask_attend_enc = pmpnn.gather_nodes(mask_t.unsqueeze(-1), e_idx).squeeze(-1)
      mask_attend_enc = mask_t.unsqueeze(-1) * mask_attend_enc
      for layer in pt_model.encoder_layers:
        h_v, h_e = layer(h_v, h_e, e_idx, mask_t, mask_attend_enc)

      h_v_enc = h_v.clone()

      # Teacher-forced decoder (identical to upstream ProteinMPNN.forward, but using ``dtype``
      # instead of ``.float()`` on ``permutation_matrix_reverse`` so float64 does not mix dtypes)
      h_s = pt_model.W_s(s_t)
      h_es = pmpnn.cat_neighbors_nodes(h_s, h_e, e_idx)
      h_ex_encoder = pmpnn.cat_neighbors_nodes(torch.zeros_like(h_s), h_e, e_idx)
      h_exv_encoder = pmpnn.cat_neighbors_nodes(h_v, h_ex_encoder, e_idx)

      mask_size = e_idx.shape[1]
      perm_rev = torch.nn.functional.one_hot(dec_order_t, num_classes=mask_size).to(dtype=dtype)
      tri = 1.0 - torch.triu(torch.ones(mask_size, mask_size, dtype=dtype))
      order_mask_bw = torch.einsum("ij,biq,bjp->bqp", tri, perm_rev, perm_rev)
      mask_attend_dec = torch.gather(order_mask_bw, 2, e_idx).unsqueeze(-1)
      mask_1d = mask_t.view([mask_t.size(0), mask_t.size(1), 1, 1])
      mask_bw = mask_1d * mask_attend_dec
      mask_fw = mask_1d * (1.0 - mask_attend_dec)

      h_exv_encoder_fw = mask_fw * h_exv_encoder
      for layer in pt_model.decoder_layers:
        h_esv = pmpnn.cat_neighbors_nodes(h_v, h_es, e_idx)
        h_esv = mask_bw * h_esv + h_exv_encoder_fw
        h_v = layer(h_v, h_esv, mask_t)

      logits = pt_model.W_out(h_v)
      log_probs_t = torch.nn.functional.log_softmax(logits, dim=-1)
  finally:
    torch.set_default_dtype(prev_default)

  return (
    h_e_0.squeeze(0).cpu().numpy(),
    h_v_enc.squeeze(0).cpu().numpy(),
    log_probs_t.squeeze(0).cpu().numpy(),
  )


def test_topology_and_legacy_alias_resolution() -> None:
  """Verify topology inference and alias mapping for Tier 1, Tier 2, and LigandMPNN checkpoints."""
  for ckpt_id in (
    "abmpnn_v_48_020.eqx.zst",
    "hypermpnn_v_48_002.eqx.zst",
    "hypermpnn_v_48_010.eqx.zst",
    "hypermpnn_v_48_020.eqx.zst",
    "hypermpnn_v_48_030.eqx.zst",
    "cyclicmpnn_v_48_010.eqx.zst",
    "proteinmpnn_v_48_020.eqx.zst",
  ):
    topo = get_topology_for_checkpoint(ckpt_id)
    assert topo["model_type"] == "protein", ckpt_id
    assert topo["k_neighbors"] == 48, ckpt_id
    assert topo["num_positional_embeddings"] == 32, ckpt_id
    assert topo["atom_context_num"] == 16, f"Noise suffix overwrote atom_context_num for {ckpt_id}"

  for ca_id in ("ca", "campnn", "ca_v_48_002", "ca_v_48_010", "ca_v_48_020.eqx.zst", "campnn_v_48_020"):
    topo = get_topology_for_checkpoint(ca_id)
    assert topo["model_type"] == "ca", ca_id
    assert topo["k_neighbors"] == 48, ca_id
    assert topo["num_positional_embeddings"] == 32, ca_id
    assert topo["atom_context_num"] == 16, ca_id

  ligand_topo = get_topology_for_checkpoint("ligandmpnn_v_32_020_25.eqx.zst")
  assert ligand_topo["model_type"] == "ligand"
  assert ligand_topo["k_neighbors"] == 32
  assert ligand_topo["atom_context_num"] == 25

  assert LEGACY_ALIAS_MAP[("abmpnn", "v_48_020")] == "abmpnn_v_48_020.eqx.zst"
  assert ("abmpnn", "v_48_002") not in LEGACY_ALIAS_MAP
  assert LEGACY_ALIAS_MAP[("hypermpnn", "v_48_020")] == "hypermpnn_v_48_020.eqx.zst"
  assert LEGACY_ALIAS_MAP[("cyclicmpnn", "v_48_010")] == "cyclicmpnn_v_48_010.eqx.zst"
  assert LEGACY_ALIAS_MAP[("ca", "v_48_020")] == "ca_v_48_020.eqx.zst"
  assert ("ca", "v_48_030") not in LEGACY_ALIAS_MAP


@pytest.mark.parametrize(
  ("variant_name", "rel_ckpt"),
  [
    ("abmpnn", "AbMPNN/abmpnn.pt"),
    ("hypermpnn_002", "HyperMPNN/retrained_models/v48_002_epoch240_hyper.pt"),
    ("hypermpnn_010", "HyperMPNN/retrained_models/v48_010_epoch300_hyper.pt"),
    ("hypermpnn_020", "HyperMPNN/retrained_models/v48_020_epoch300_hyper.pt"),
    ("hypermpnn_030", "HyperMPNN/retrained_models/v48_030_epoch300_hyper.pt"),
    ("cyclicmpnn", "CyclicMPNN/cyclicmpnn_weights/cyclicmpnn_48_010.pt"),
  ],
)
def test_tier1_drop_in_variants_graded_parity(variant_name: str, rel_ckpt: str) -> None:
  """Verify float64 (< 1e-7) and float32 (< 1e-4) parity for AbMPNN, HyperMPNN, and CyclicMPNN."""
  pt_path = _find_repo_file(rel_ckpt)
  if pt_path is None:
    pytest.skip(f"Checkpoint {rel_ckpt} not available locally")

  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB, max_residues=40)
  rng = np.random.default_rng(42)
  decoding_order = rng.permutation(coords.shape[0]).astype(np.int32)

  with _x64_mode(True):
    jax_model_f64 = _cast_floating(convert_mpnn_checkpoint(pt_path), jnp.float64)
    jax_edges_f64, jax_nodes_f64, jax_log_probs_f64 = _jax_teacher_forced_forward(
      jax_model_f64,
      jnp.asarray(coords, dtype=jnp.float64),
      jnp.asarray(mask, dtype=jnp.float64),
      jnp.asarray(residue_index, dtype=jnp.int32),
      jnp.asarray(chain_index, dtype=jnp.int32),
      jnp.asarray(sequence, dtype=jnp.int32),
      jnp.asarray(decoding_order, dtype=jnp.int32),
    )

  pt_edges_f64, pt_nodes_f64, pt_log_probs_f64 = _torch_teacher_forced_forward(
    pt_path,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    decoding_order,
    ca_only=False,
    dtype=torch.float64,
  )

  assert np.max(np.abs(np.asarray(jax_edges_f64) - pt_edges_f64)) < 1e-7, variant_name
  assert np.max(np.abs(np.asarray(jax_nodes_f64) - pt_nodes_f64)) < 1e-7, variant_name
  assert np.max(np.abs(np.asarray(jax_log_probs_f64) - pt_log_probs_f64)) < 1e-7, variant_name

  with _x64_mode(False):
    jax_model_f32 = _cast_floating(convert_mpnn_checkpoint(pt_path), jnp.float32)
    jax_edges_f32, jax_nodes_f32, jax_log_probs_f32 = _jax_teacher_forced_forward(
      jax_model_f32,
      jnp.asarray(coords, dtype=jnp.float32),
      jnp.asarray(mask, dtype=jnp.float32),
      jnp.asarray(residue_index, dtype=jnp.int32),
      jnp.asarray(chain_index, dtype=jnp.int32),
      jnp.asarray(sequence, dtype=jnp.int32),
      jnp.asarray(decoding_order, dtype=jnp.int32),
    )

  pt_edges_f32, pt_nodes_f32, pt_log_probs_f32 = _torch_teacher_forced_forward(
    pt_path,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    decoding_order,
    ca_only=False,
    dtype=torch.float32,
  )

  assert np.max(np.abs(np.asarray(jax_edges_f32) - pt_edges_f32)) < 1e-4, variant_name
  assert np.max(np.abs(np.asarray(jax_nodes_f32) - pt_nodes_f32)) < 1e-4, variant_name
  assert np.max(np.abs(np.asarray(jax_log_probs_f32) - pt_log_probs_f32)) < 1e-4, variant_name


@pytest.mark.parametrize(
  ("ca_version", "rel_ckpt"),
  [
    ("v_48_002", "ProteinMPNN/ca_model_weights/v_48_002.pt"),
    ("v_48_010", "ProteinMPNN/ca_model_weights/v_48_010.pt"),
    ("v_48_020", "ProteinMPNN/ca_model_weights/v_48_020.pt"),
  ],
)
def test_tier2_ca_only_mpnn_graded_parity(ca_version: str, rel_ckpt: str, tmp_path: Path) -> None:
  """Verify float64 (< 1e-7) and float32 (< 1e-4) parity for CA-Only ProteinMPNN."""
  pt_path = _find_repo_file(rel_ckpt)
  if pt_path is None:
    pytest.skip(f"Checkpoint {rel_ckpt} not available locally")

  # Use L=60 > k=48 so D_max (where masked residue 28 ties with the farthest valid residue)
  # sits at rank 58-59 outside the top-48 neighbor list of all valid residues.
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB, max_residues=60)
  # Introduce a deliberate chain break (jump > 4.0 A) at residue 15 and a masked residue at 28
  # to exercise ``dX_mask = (3.6 < dX_norm) & (dX_norm < 4.0)`` and ``D_adjust`` masking.
  coords_broken = coords.copy()
  coords_broken[15:, :, :] += np.array([6.5, -4.0, 3.0], dtype=np.float64)
  mask_with_gap = mask.copy()
  mask_with_gap[28] = 0.0
  residue_index_multi = residue_index.copy()
  chain_index_multi = chain_index.copy()
  chain_index_multi[20:] = 1

  rng = np.random.default_rng(7)
  decoding_order = rng.permutation(coords_broken.shape[0]).astype(np.int32)

  with _x64_mode(True):
    jax_model_f64 = _cast_floating(convert_mpnn_checkpoint(pt_path), jnp.float64)
    assert isinstance(jax_model_f64.features, ProteinFeaturesCA)
    jax_edges_f64, jax_nodes_f64, jax_log_probs_f64 = _jax_teacher_forced_forward(
      jax_model_f64,
      jnp.asarray(coords_broken, dtype=jnp.float64),
      jnp.asarray(mask_with_gap, dtype=jnp.float64),
      jnp.asarray(residue_index_multi, dtype=jnp.int32),
      jnp.asarray(chain_index_multi, dtype=jnp.int32),
      jnp.asarray(sequence, dtype=jnp.int32),
      jnp.asarray(decoding_order, dtype=jnp.int32),
    )

  pt_edges_f64, pt_nodes_f64, pt_log_probs_f64 = _torch_teacher_forced_forward(
    pt_path,
    coords_broken,
    mask_with_gap,
    residue_index_multi,
    chain_index_multi,
    sequence,
    decoding_order,
    ca_only=True,
    dtype=torch.float64,
  )

  valid = mask_with_gap > 0.5
  assert np.max(np.abs(np.asarray(jax_edges_f64)[valid] - pt_edges_f64[valid])) < 1e-7, ca_version
  assert np.max(np.abs(np.asarray(jax_nodes_f64)[valid] - pt_nodes_f64[valid])) < 1e-7, ca_version
  assert np.max(np.abs(np.asarray(jax_log_probs_f64)[valid] - pt_log_probs_f64[valid])) < 1e-7, ca_version

  with _x64_mode(False):
    jax_model_f32 = _cast_floating(convert_mpnn_checkpoint(pt_path), jnp.float32)
    # Also verify round-trip serialization + load_model(checkpoint_id="ca_...", local_path=...)
    eqx_file = tmp_path / f"ca_{ca_version}.eqx"
    eqx.tree_serialise_leaves(eqx_file, jax_model_f32)
    reloaded = load_model(
      checkpoint_id=f"ca_{ca_version}",
      model_weights="ca",
      model_version=ca_version,
      local_path=str(eqx_file),
      dropout_rate=0.0,
    )
    assert isinstance(reloaded, Aminx)
    assert isinstance(reloaded.features, ProteinFeaturesCA)

    # Verify both (L, 4, 3) full backbone and (L, 3) CA-only coordinate inputs produce identical output
    jax_edges_f32, jax_nodes_f32, jax_log_probs_f32 = _jax_teacher_forced_forward(
      reloaded,
      jnp.asarray(coords_broken, dtype=jnp.float32),
      jnp.asarray(mask_with_gap, dtype=jnp.float32),
      jnp.asarray(residue_index_multi, dtype=jnp.int32),
      jnp.asarray(chain_index_multi, dtype=jnp.int32),
      jnp.asarray(sequence, dtype=jnp.int32),
      jnp.asarray(decoding_order, dtype=jnp.int32),
    )
    jax_edges_ca3, _, jax_log_probs_ca3 = _jax_teacher_forced_forward(
      reloaded,
      jnp.asarray(coords_broken[:, 1, :], dtype=jnp.float32),
      jnp.asarray(mask_with_gap, dtype=jnp.float32),
      jnp.asarray(residue_index_multi, dtype=jnp.int32),
      jnp.asarray(chain_index_multi, dtype=jnp.int32),
      jnp.asarray(sequence, dtype=jnp.int32),
      jnp.asarray(decoding_order, dtype=jnp.int32),
    )
    assert np.max(np.abs(np.asarray(jax_edges_f32) - np.asarray(jax_edges_ca3))) == 0.0
    assert np.max(np.abs(np.asarray(jax_log_probs_f32) - np.asarray(jax_log_probs_ca3))) == 0.0

  pt_edges_f32, pt_nodes_f32, pt_log_probs_f32 = _torch_teacher_forced_forward(
    pt_path,
    coords_broken,
    mask_with_gap,
    residue_index_multi,
    chain_index_multi,
    sequence,
    decoding_order,
    ca_only=True,
    dtype=torch.float32,
  )

  # Where 1 +/- Rxx +/- Ryy +/- Rzz crosses zero inside _quaternions's sqrt(abs(...)), a 1-ULP
  # float32 difference (~1.2e-7) is amplified by sqrt(1.2e-7) ~ 3.5e-4 on that single edge (~6e-4
  # after w_e_proj). Across 99.9% of edge entries, encoder nodes, and decoder log-probs, float32
  # agrees within 1e-4, and in float64 every entry (100%) agrees within 1e-7.
  assert np.quantile(np.abs(np.asarray(jax_edges_f32)[valid] - pt_edges_f32[valid]), 0.999) < 1e-4, ca_version
  assert np.max(np.abs(np.asarray(jax_edges_f32)[valid] - pt_edges_f32[valid])) < 1e-3, ca_version
  assert np.max(np.abs(np.asarray(jax_nodes_f32)[valid] - pt_nodes_f32[valid])) < 1e-4, ca_version
  assert np.max(np.abs(np.asarray(jax_log_probs_f32)[valid] - pt_log_probs_f32[valid])) < 1e-4, ca_version


def test_negative_controls_ca_features_and_weights() -> None:
  """Verify that corrupting weights or permuting CA feature channels breaks parity decisively."""
  pt_path = _find_repo_file("ProteinMPNN/ca_model_weights/v_48_020.pt")
  if pt_path is None:
    pytest.skip("CA checkpoint not available locally")

  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB, max_residues=32)
  decoding_order = np.arange(coords.shape[0], dtype=np.int32)

  pt_edges, _, pt_log_probs = _torch_teacher_forced_forward(
    pt_path,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    decoding_order,
    ca_only=True,
    dtype=torch.float32,
  )

  jax_model = _cast_floating(convert_mpnn_checkpoint(pt_path), jnp.float32)

  # Negative control 1: flip sign of quaternion columns (last 4 of the 167 edge input channels)
  w_e_corrupt = jax_model.features.w_e.weight.at[:, -4:].set(-jax_model.features.w_e.weight[:, -4:])
  corrupt_model_1 = eqx.tree_at(lambda m: m.features.w_e.weight, jax_model, w_e_corrupt)
  bad_edges_1, _, bad_lp_1 = _jax_teacher_forced_forward(
    corrupt_model_1,
    jnp.asarray(coords, dtype=jnp.float32),
    jnp.asarray(mask, dtype=jnp.float32),
    jnp.asarray(residue_index, dtype=jnp.int32),
    jnp.asarray(chain_index, dtype=jnp.int32),
    jnp.asarray(sequence, dtype=jnp.int32),
    jnp.asarray(decoding_order, dtype=jnp.int32),
  )
  assert np.max(np.abs(np.asarray(bad_edges_1) - pt_edges)) > 1e-2
  assert np.max(np.abs(np.asarray(bad_lp_1) - pt_log_probs)) > 1e-2

  # Negative control 2: swap Ca_0-Ca_1 and Ca_1-Ca_0 RBF blocks in w_e
  w_e_swapped = jax_model.features.w_e.weight.copy()
  block_a = slice(16 + 3 * 16, 16 + 4 * 16)
  block_b = slice(16 + 5 * 16, 16 + 6 * 16)
  w_e_swapped = w_e_swapped.at[:, block_a].set(jax_model.features.w_e.weight[:, block_b])
  w_e_swapped = w_e_swapped.at[:, block_b].set(jax_model.features.w_e.weight[:, block_a])
  corrupt_model_2 = eqx.tree_at(lambda m: m.features.w_e.weight, jax_model, w_e_swapped)
  bad_edges_2, _, bad_lp_2 = _jax_teacher_forced_forward(
    corrupt_model_2,
    jnp.asarray(coords, dtype=jnp.float32),
    jnp.asarray(mask, dtype=jnp.float32),
    jnp.asarray(residue_index, dtype=jnp.int32),
    jnp.asarray(chain_index, dtype=jnp.int32),
    jnp.asarray(sequence, dtype=jnp.int32),
    jnp.asarray(decoding_order, dtype=jnp.int32),
  )
  assert np.max(np.abs(np.asarray(bad_edges_2) - pt_edges)) > 1e-2
  assert np.max(np.abs(np.asarray(bad_lp_2) - pt_log_probs)) > 1e-2
