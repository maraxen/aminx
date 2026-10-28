"""Unit tests for AntiBMPNN weight aliases, CDR region masking, log-odds biasing, and Barley descriptors."""

from __future__ import annotations

import os

from pathlib import Path
from typing import get_args

import jax
import jax.numpy as jnp
import pytest

from aminx.inference.logits import ArithmeticMeanLogits
from aminx.io.weights import (
  _BARE_VARIANT_DEFAULTS,
  LEGACY_ALIAS_MAP,
  get_topology_for_checkpoint,
)
from aminx.model import (
  CDRFrequencyLogitTransform,
  CDRPhysicochemicalDescriptors,
  CDRRegionMasks,
  build_cdr_region_mask,
  compute_cdr_physicochemical_descriptors,
  make_cdr_frequency_logit_transform,
)
from aminx.model.cdr_bias import ALPHABET_21, BARLEY_PD1_VOLUME, BARLEY_PD2_HYDRO
from aminx.model.versions import MODEL_VERSION, MODEL_WEIGHTS


def _encode_sequence(seq_str: str) -> jax.Array:
  """Encode a string of 1-letter amino-acid codes into Aminx 21-token indices."""
  return jnp.array(
    [ALPHABET_21.index(ch) if ch in ALPHABET_21 else 20 for ch in seq_str],
    dtype=jnp.int32,
  )


def test_antibmpnn_weights_registration_and_topology() -> None:
  """Verify antibmpnn and v_48_000 registration in versions.py and weights.py."""
  assert "antibmpnn" in get_args(MODEL_WEIGHTS)
  assert "v_48_000" in get_args(MODEL_VERSION)
  assert _BARE_VARIANT_DEFAULTS["antibmpnn"] == "antibmpnn_v_48_000.eqx.zst"
  assert ("antibmpnn", "v_48_002") not in LEGACY_ALIAS_MAP

  for ver in ("v_48_000", "v_48_010", "v_48_020", "v_48_030"):
    expected_file = f"antibmpnn_{ver}.eqx.zst"
    assert LEGACY_ALIAS_MAP[("antibmpnn", ver)] == expected_file
    topo = get_topology_for_checkpoint(expected_file)
    assert topo["model_type"] == "protein"
    assert topo["k_neighbors"] == 48
    assert topo["num_positional_embeddings"] == 32


def test_antibmpnn_v_48_002_is_refused_not_aliased() -> None:
  """Verify requesting antibmpnn v_48_002 raises instead of silently loading v_48_000."""
  from aminx.io.weights import load_model

  with pytest.raises(ValueError, match="v_48_002"):
    load_model(model_weights="antibmpnn", model_version="v_48_002")


def test_build_cdr_region_mask_and_test_cdr_region_mask_chothia_and_imgt() -> None:
  """Verify Chothia and IMGT CDR bounds on heavy and light chains."""
  res_idx = jnp.array([25, 26, 32, 52, 95, 103, 24, 34, 50, 88, 89, 97], dtype=jnp.int32)
  chain_idx = jnp.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=jnp.int32)
  mask = jnp.ones_like(res_idx, dtype=jnp.float32)

  chothia = build_cdr_region_mask(res_idx, chain_idx, scheme="chothia", mask=mask)
  assert isinstance(chothia, CDRRegionMasks)
  assert jnp.array_equal(
    chothia.h1,
    jnp.array([False, True, True, False, False, False, False, False, False, False, False, False]),
  )
  assert jnp.array_equal(
    chothia.h2,
    jnp.array([False, False, False, True, False, False, False, False, False, False, False, False]),
  )
  assert jnp.array_equal(
    chothia.h3,
    jnp.array([False, False, False, False, True, False, False, False, False, False, False, False]),
  )
  assert jnp.array_equal(
    chothia.l1,
    jnp.array([False, False, False, False, False, False, True, True, False, False, False, False]),
  )
  assert jnp.array_equal(
    chothia.l2,
    jnp.array([False, False, False, False, False, False, False, False, True, False, False, False]),
  )
  assert jnp.array_equal(
    chothia.l3,
    jnp.array([False, False, False, False, False, False, False, False, False, False, True, True]),
  )
  assert jnp.array_equal(chothia.heavy_cdr, chothia.h1 | chothia.h2 | chothia.h3)
  assert jnp.array_equal(chothia.light_cdr, chothia.l1 | chothia.l2 | chothia.l3)
  assert jnp.array_equal(chothia.any_cdr, chothia.heavy_cdr | chothia.light_cdr)

  partial_mask = mask.at[1].set(0.0).at[10].set(0.0)
  masked_chothia = build_cdr_region_mask(res_idx, chain_idx, scheme="chothia", mask=partial_mask)
  assert not bool(masked_chothia.h1[1])
  assert not bool(masked_chothia.l3[10])

  imgt_res = jnp.array([26, 27, 38, 56, 105, 117], dtype=jnp.int32)
  imgt_chain = jnp.zeros_like(imgt_res)
  imgt = build_cdr_region_mask(imgt_res, imgt_chain, scheme="imgt")
  assert jnp.array_equal(imgt.h1, jnp.array([False, True, True, False, False, False]))
  assert jnp.array_equal(imgt.h2, jnp.array([False, False, False, True, False, False]))
  assert jnp.array_equal(imgt.h3, jnp.array([False, False, False, False, True, True]))

  with pytest.raises(ValueError, match="Unknown CDR numbering scheme"):
    build_cdr_region_mask(res_idx, chain_idx, scheme="kabat_invalid")


def test_cdr_frequency_logit_transform_invariants() -> None:
  """Verify uniform no-op, framework zero-bias, and unknown token 'X' invariance."""
  n_res = 6
  cdr_mask = jnp.array([False, True, True, False, True, False])
  per_state = jax.random.normal(jax.random.PRNGKey(42), (2, n_res, 21))

  base_fn = ArithmeticMeanLogits(jnp.ones(1, dtype=jnp.float32))
  uniform_fn = make_cdr_frequency_logit_transform(cdr_mask, base_transform=base_fn)
  assert isinstance(uniform_fn, CDRFrequencyLogitTransform)
  expected_base = base_fn(per_state, None)
  actual_uniform = uniform_fn(per_state, None)
  assert jnp.allclose(actual_uniform, expected_base, atol=1e-7)

  target = jnp.full((20,), 0.02, dtype=jnp.float32).at[19].set(0.62)
  bg = jnp.full((20,), 1.0 / 20.0, dtype=jnp.float32)
  biased_fn = make_cdr_frequency_logit_transform(
    cdr_mask,
    target_frequencies=target,
    background_frequencies=bg,
    weight=1.5,
    base_transform=base_fn,
  )
  out = biased_fn(per_state, None)
  diff = out - expected_base

  assert jnp.all(diff[~cdr_mask] == 0.0)
  assert jnp.all(diff[:, 20] == 0.0)
  assert jnp.all(diff[cdr_mask, 19] > 0.0)
  assert jnp.all(diff[cdr_mask, 0] < 0.0)


def test_cdr_physicochemical_descriptors_barley_and_lehninger() -> None:
  """Verify Barley PC1/PC2 and Lehninger charge match AntiBMPNN README examples to 1e-4."""
  seq1 = _encode_sequence("ARSKSTYLSRDSSGYDY")
  seq2 = _encode_sequence("ARSKSTYLSYNSSGYDY")
  seq3 = _encode_sequence("ARSKSTYLSYDSSGYDY")

  assert len(BARLEY_PD1_VOLUME) == 21
  assert len(BARLEY_PD2_HYDRO) == 21

  batch_seqs = jnp.stack([seq1, seq2, seq3], axis=0)
  batch_desc = jax.jit(compute_cdr_physicochemical_descriptors)(batch_seqs)
  assert isinstance(batch_desc, CDRPhysicochemicalDescriptors)

  expected_pd1 = jnp.array([-0.4271, -0.3929, -0.4071], dtype=jnp.float32)
  expected_pd2 = jnp.array([0.3253, 0.2118, 0.2376], dtype=jnp.float32)
  expected_pd3 = jnp.array([0.9881, 0.9858, -0.0140], dtype=jnp.float32)

  assert jnp.allclose(batch_desc.pd1_volume, expected_pd1, atol=1e-4)
  assert jnp.allclose(batch_desc.pd2_hydro, expected_pd2, atol=1e-4)
  assert jnp.allclose(batch_desc.pd3_charge, expected_pd3, atol=1e-4)

  d1 = compute_cdr_physicochemical_descriptors(seq1)
  assert jnp.allclose(d1.pd1_volume, expected_pd1[0], atol=1e-4)
  assert jnp.allclose(d1.pd2_hydro, expected_pd2[0], atol=1e-4)
  assert jnp.allclose(d1.pd3_charge, expected_pd3[0], atol=1e-4)

  empty_mask = jnp.zeros_like(seq1, dtype=jnp.bool_)
  empty_desc = compute_cdr_physicochemical_descriptors(seq1, empty_mask)
  assert float(empty_desc.pd1_volume) == 0.0
  assert float(empty_desc.pd2_hydro) == 0.0
  assert float(empty_desc.pd3_charge) == 0.0


def test_variant_docs_and_notice_and_test_tier5_docs() -> None:
  """Verify Tier 1-5 documentation and NOTICE-weights.md include required entries."""
  repo_root = Path(__file__).resolve().parents[2]
  doc_text = (repo_root / "docs" / "models" / "mpnn_variants.md").read_text(encoding="utf-8")
  for term in (
    "Tier 1",
    "Tier 2",
    "Tier 3",
    "Tier 4",
    "Tier 5",
    "AntiBMPNN",
    "CyclicMPNN",
    "ESM-IF1",
    "ProteinDPO",
    "Frame2seq",
    "ProstT5",
  ):
    assert term in doc_text

  notice_text = (repo_root / "NOTICE-weights.md").read_text(encoding="utf-8")
  for term in ("AntiBMPNN", "Caliby", "FAMPNN", "10.5281/zenodo.13387792"):
    assert term in notice_text


def test_antibmpnn_packaged_checkpoints_load() -> None:
  """Verify all 4 converted AntiBMPNN .eqx.zst checkpoints load via load_model."""
  from aminx.io.weights import load_model

  for cid in ("antibmpnn_v_48_000", "antibmpnn_v_48_010", "antibmpnn_v_48_020", "antibmpnn_v_48_030"):
    model = load_model(checkpoint_id=cid)
    assert model.encoder.layers[0].edge_message_mlp.layers[0].weight.shape == (128, 384)


def _parse_pdb_backbone_ubq(
  pdb_path: Path,
  max_residues: int = 40,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Parse N, CA, C, O coordinates and sequence from a PDB file."""
  import numpy as np

  aa3_to_idx = {
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
  residues: dict[tuple[str, int], dict[str, object]] = {}
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
      atoms_dict = residues[key]["atoms"]
      assert isinstance(atoms_dict, dict)
      atoms_dict[atom_name] = xyz

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
    assert isinstance(atoms, dict)
    for atom_idx, atom_name in enumerate(("N", "CA", "C", "O")):
      if atom_name in atoms:
        coords[idx, atom_idx] = atoms[atom_name]
      else:
        mask[idx] = 0.0
    residue_index[idx] = res_seq
    chain_index[idx] = chain_order.index(chain_id)
    sequence[idx] = aa3_to_idx.get(str(entry["res_name"]), 20)

  return coords, mask, residue_index, chain_index, sequence  # type: ignore[return-value]


@pytest.mark.parametrize(
  ("cid", "pt_filename"),
  [
    ("antibmpnn_v_48_000", "antibmpnn_000.pt"),
    ("antibmpnn_v_48_010", "antibmpnn_010.pt"),
    ("antibmpnn_v_48_020", "antibmpnn_020.pt"),
    ("antibmpnn_v_48_030", "antibmpnn_030.pt"),
  ],
)
def test_antibmpnn_pytorch_vs_jax_graded_parity_and_negative_control(
  cid: str,
  pt_filename: str,
) -> None:
  """Verify float64 (< 1e-7) and float32 (< 1e-4) parity against PyTorch ProteinMPNN + negative control."""
  import contextlib
  import sys

  import equinox as eqx
  import numpy as np
  import torch
  import torch.nn as nn

  from aminx.families.potts_mpnn.features import potts_edge_features
  from aminx.io.weights import load_model
  from aminx.model.decoder import (
    conditional_decoder_layer_edge_features,
    pack_conditional_decoder_static_edges,
  )
  from aminx.utils.autoregression import ar_mask_from_decoding_order

  repo_root = Path(__file__).resolve().parents[2]
  if str(repo_root / "scripts") not in sys.path:
    sys.path.insert(0, str(repo_root / "scripts"))
  from convert_weights import convert_mpnn_checkpoint

  pt_path: Path | None = None
  env_upstream = os.environ.get("AMINX_UPSTREAM_REPOS_DIR")
  search_roots = (
    (Path(env_upstream),)
    if env_upstream
    else (
      Path.home() / "repos",
      Path("/home/marielle/.gemini/antigravity-cli/brain/e0a4519f-8415-4303-b518-8fc3d09cb0cc/scratch/repos"),
    )
  )
  for root in search_roots:
    cand_pt = root / "AntiBMPNN" / "antibmpnn_model_weights" / pt_filename
    if cand_pt.is_file():
      pt_path = cand_pt
      break

  def _gather_edges(edges: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)

  def _gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    return neighbor_features.view([*list(neighbor_idx.shape)[:3], -1])

  def _cat_neighbors_nodes(
    h_nodes: torch.Tensor,
    h_neighbors: torch.Tensor,
    e_idx: torch.Tensor,
  ) -> torch.Tensor:
    return torch.cat([h_neighbors, _gather_nodes(h_nodes, e_idx)], -1)

  class _PosWiseFF(nn.Module):
    def __init__(self, num_hidden: int, num_ff: int) -> None:
      super().__init__()
      self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
      self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
      self.act = nn.GELU()

    def forward(self, h_v: torch.Tensor) -> torch.Tensor:
      return self.W_out(self.act(self.W_in(h_v)))

  class _EncLayer(nn.Module):
    def __init__(self, num_hidden: int, num_in: int, scale: int = 30) -> None:
      super().__init__()
      self.scale = scale
      self.norm1 = nn.LayerNorm(num_hidden)
      self.norm2 = nn.LayerNorm(num_hidden)
      self.norm3 = nn.LayerNorm(num_hidden)
      self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
      self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
      self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
      self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
      self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
      self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
      self.act = nn.GELU()
      self.dense = _PosWiseFF(num_hidden, num_hidden * 4)

    def forward(
      self,
      h_v: torch.Tensor,
      h_e: torch.Tensor,
      e_idx: torch.Tensor,
      mask_v: torch.Tensor,
      mask_attend: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
      h_ev = _cat_neighbors_nodes(h_v, h_e, e_idx)
      h_v_exp = h_v.unsqueeze(-2).expand(-1, -1, h_ev.size(-2), -1)
      h_ev = torch.cat([h_v_exp, h_ev], -1)
      h_msg = self.W3(self.act(self.W2(self.act(self.W1(h_ev)))))
      h_msg = mask_attend.unsqueeze(-1) * h_msg
      dh = torch.sum(h_msg, -2) / self.scale
      h_v = self.norm1(h_v + dh)
      dh = self.dense(h_v)
      h_v = self.norm2(h_v + dh)
      h_v = mask_v.unsqueeze(-1) * h_v
      h_ev = _cat_neighbors_nodes(h_v, h_e, e_idx)
      h_v_exp = h_v.unsqueeze(-2).expand(-1, -1, h_ev.size(-2), -1)
      h_ev = torch.cat([h_v_exp, h_ev], -1)
      h_msg = self.W13(self.act(self.W12(self.act(self.W11(h_ev)))))
      h_e = self.norm3(h_e + h_msg)
      return h_v, h_e

  class _DecLayer(nn.Module):
    def __init__(self, num_hidden: int, num_in: int, scale: int = 30) -> None:
      super().__init__()
      self.scale = scale
      self.norm1 = nn.LayerNorm(num_hidden)
      self.norm2 = nn.LayerNorm(num_hidden)
      self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
      self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
      self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
      self.act = nn.GELU()
      self.dense = _PosWiseFF(num_hidden, num_hidden * 4)

    def forward(self, h_v: torch.Tensor, h_e: torch.Tensor, mask_v: torch.Tensor) -> torch.Tensor:
      h_v_exp = h_v.unsqueeze(-2).expand(-1, -1, h_e.size(-2), -1)
      h_ev = torch.cat([h_v_exp, h_e], -1)
      h_msg = self.W3(self.act(self.W2(self.act(self.W1(h_ev)))))
      dh = torch.sum(h_msg, -2) / self.scale
      h_v = self.norm1(h_v + dh)
      dh = self.dense(h_v)
      h_v = self.norm2(h_v + dh)
      return mask_v.unsqueeze(-1) * h_v

  class _PosEnc(nn.Module):
    def __init__(self, num_embeddings: int = 16, max_relative_feature: int = 32) -> None:
      super().__init__()
      self.max_relative_feature = max_relative_feature
      self.linear = nn.Linear(2 * max_relative_feature + 2, num_embeddings)

    def forward(self, offset: torch.Tensor, mask_chains: torch.Tensor) -> torch.Tensor:
      d = (
        torch.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        * mask_chains
        + (1 - mask_chains) * (2 * self.max_relative_feature + 1)
      )
      d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 2)
      return self.linear(d_onehot.to(dtype=self.linear.weight.dtype))

  class _ProtFeatures(nn.Module):
    def __init__(self, edge_features: int = 128, top_k: int = 48) -> None:
      super().__init__()
      self.top_k = top_k
      self.num_rbf = 16
      self.embeddings = _PosEnc(16, 32)
      self.edge_embedding = nn.Linear(16 + 16 * 25, edge_features, bias=False)
      self.norm_edges = nn.LayerNorm(edge_features)

    def _rbf(self, d: torch.Tensor) -> torch.Tensor:
      d_mu = torch.linspace(2.0, 22.0, self.num_rbf, dtype=d.dtype, device=d.device).view(1, 1, 1, -1)
      d_sigma = (22.0 - 2.0) / self.num_rbf
      return torch.exp(-(((d.unsqueeze(-1) - d_mu) / d_sigma) ** 2))

    def _get_rbf(self, a: torch.Tensor, b: torch.Tensor, e_idx: torch.Tensor) -> torch.Tensor:
      d_ab = torch.sqrt(torch.sum((a[:, :, None, :] - b[:, None, :, :]) ** 2, -1) + 1e-6)
      d_neighbors = _gather_edges(d_ab[:, :, :, None], e_idx)[:, :, :, 0]
      return self._rbf(d_neighbors)

    def forward(
      self,
      x: torch.Tensor,
      mask: torch.Tensor,
      residue_idx: torch.Tensor,
      chain_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
      b_vec = x[:, :, 1, :] - x[:, :, 0, :]
      c_vec = x[:, :, 2, :] - x[:, :, 1, :]
      a_vec = torch.cross(b_vec, c_vec, dim=-1)
      cb = -0.58273431 * a_vec + 0.56802827 * b_vec - 0.54067466 * c_vec + x[:, :, 1, :]
      ca, n, c, o = x[:, :, 1, :], x[:, :, 0, :], x[:, :, 2, :], x[:, :, 3, :]

      mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
      dx = ca.unsqueeze(1) - ca.unsqueeze(2)
      d = mask_2d * torch.sqrt(torch.sum(dx**2, 3) + 1e-6)
      d_max, _ = torch.max(d, -1, keepdim=True)
      d_adjust = d + (1.0 - mask_2d) * d_max
      d_neighbors, e_idx = torch.topk(d_adjust, min(self.top_k, x.shape[1]), dim=-1, largest=False)

      pairs = [
        (ca, ca), (n, n), (c, c), (o, o), (cb, cb),
        (ca, n), (ca, c), (ca, o), (ca, cb),
        (n, c), (n, o), (n, cb), (cb, c), (cb, o), (o, c),
        (n, ca), (c, ca), (o, ca), (cb, ca),
        (c, n), (o, n), (cb, n), (c, cb), (o, cb), (c, o),
      ]
      rbf_all = [self._rbf(d_neighbors)] + [self._get_rbf(pa, pb, e_idx) for pa, pb in pairs[1:]]
      rbf_cat = torch.cat(rbf_all, dim=-1)
      offset = residue_idx[:, :, None] - residue_idx[:, None, :]
      offset = _gather_edges(offset[:, :, :, None], e_idx)[:, :, :, 0]
      d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
      e_chains = _gather_edges(d_chains[:, :, :, None], e_idx)[:, :, :, 0]
      e_pos = self.embeddings(offset.long(), e_chains)
      e = self.norm_edges(self.edge_embedding(torch.cat((e_pos, rbf_cat), -1)))
      return e, e_idx

  class _RefProteinMPNN(nn.Module):
    def __init__(self, k_neighbors: int = 48) -> None:
      super().__init__()
      self.features = _ProtFeatures(128, top_k=k_neighbors)
      self.W_e = nn.Linear(128, 128, bias=True)
      self.W_s = nn.Embedding(21, 128)
      self.encoder_layers = nn.ModuleList([_EncLayer(128, 256) for _ in range(3)])
      self.decoder_layers = nn.ModuleList([_DecLayer(128, 384) for _ in range(3)])
      self.W_out = nn.Linear(128, 21, bias=True)

  def _state_dict_from_aminx(m: object) -> dict[str, torch.Tensor]:
    def _t(arr: object) -> torch.Tensor:
      return torch.from_numpy(np.asarray(arr).copy())

    sd: dict[str, torch.Tensor] = {
      "features.embeddings.linear.weight": _t(m.features.w_pos.weight),
      "features.embeddings.linear.bias": _t(m.features.w_pos.bias),
      "features.edge_embedding.weight": _t(m.features.w_e.weight),
      "features.norm_edges.weight": _t(m.features.norm_edges.weight),
      "features.norm_edges.bias": _t(m.features.norm_edges.bias),
      "W_e.weight": _t(m.features.w_e_proj.weight),
      "W_e.bias": _t(m.features.w_e_proj.bias),
      "W_s.weight": _t(m.w_s_embed.weight),
      "W_out.weight": _t(m.w_out.weight),
      "W_out.bias": _t(m.w_out.bias),
    }
    for idx, enc in enumerate(m.encoder.layers):
      p = f"encoder_layers.{idx}."
      for name, lin in zip(("W1", "W2", "W3"), enc.edge_message_mlp.layers, strict=True):
        sd[p + f"{name}.weight"] = _t(lin.weight)
        sd[p + f"{name}.bias"] = _t(lin.bias)
      for name, lin in zip(("W11", "W12", "W13"), enc.edge_update_mlp.layers, strict=True):
        sd[p + f"{name}.weight"] = _t(lin.weight)
        sd[p + f"{name}.bias"] = _t(lin.bias)
      for name, lin in zip(("dense.W_in", "dense.W_out"), enc.dense.layers, strict=True):
        sd[p + f"{name}.weight"] = _t(lin.weight)
        sd[p + f"{name}.bias"] = _t(lin.bias)
      for name, ln in (("norm1", enc.norm1), ("norm2", enc.norm2), ("norm3", enc.norm3)):
        sd[p + f"{name}.weight"] = _t(ln.weight)
        sd[p + f"{name}.bias"] = _t(ln.bias)
    for idx, dec in enumerate(m.decoder.layers):
      p = f"decoder_layers.{idx}."
      for name, lin in zip(("W1", "W2", "W3"), dec.message_mlp.layers, strict=True):
        sd[p + f"{name}.weight"] = _t(lin.weight)
        sd[p + f"{name}.bias"] = _t(lin.bias)
      for name, lin in zip(("dense.W_in", "dense.W_out"), dec.dense.layers, strict=True):
        sd[p + f"{name}.weight"] = _t(lin.weight)
        sd[p + f"{name}.bias"] = _t(lin.bias)
      for name, ln in (("norm1", dec.norm1), ("norm2", dec.norm2)):
        sd[p + f"{name}.weight"] = _t(ln.weight)
        sd[p + f"{name}.bias"] = _t(ln.bias)
    return sd

  ubq_pdb = repo_root / "tests" / "data" / "1ubq.pdb"
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone_ubq(ubq_pdb, max_residues=40)
  rng = np.random.default_rng(42)
  decoding_order = rng.permutation(coords.shape[0]).astype(np.int32)
  loaded_eqx = load_model(checkpoint_id=cid, dropout_rate=0.0)

  def _jax_forward(model: object, dtype: jnp.dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m_cast = jax.tree_util.tree_map(
      lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x,
      model,
    )
    c_j = jnp.asarray(coords, dtype=dtype)
    mask_j = jnp.asarray(mask, dtype=dtype)
    res_j = jnp.asarray(residue_index, dtype=jnp.int32)
    ch_j = jnp.asarray(chain_index, dtype=jnp.int32)
    seq_j = jnp.asarray(sequence, dtype=jnp.int32)
    ord_j = jnp.asarray(decoding_order, dtype=jnp.int32)

    graph = potts_edge_features(m_cast.features, c_j, mask_j, res_j, ch_j)
    edge_features = graph.edge_features
    neighbor_indices = graph.neighbor_indices
    node_features, enc_edge_features = m_cast.encoder(
      edge_features,
      neighbor_indices,
      mask_j,
      inference=True,
    )
    one_hot_seq = jax.nn.one_hot(seq_j, 21, dtype=dtype)
    ar_mask = ar_mask_from_decoding_order(ord_j)
    seq_edge_features, mask_bw, masked_node_edge_features = pack_conditional_decoder_static_edges(
      node_features,
      enc_edge_features,
      neighbor_indices,
      one_hot_seq,
      m_cast.w_s_embed.weight,
      ar_mask,
      mask_j,
    )
    h_v = node_features
    for layer in m_cast.decoder.layers:
      layer_edge = conditional_decoder_layer_edge_features(
        h_v,
        seq_edge_features,
        neighbor_indices,
        mask_bw,
        masked_node_edge_features,
      )
      h_v = layer(h_v, layer_edge, mask_j, inference=True)
    logits = h_v @ m_cast.w_out.weight.T + m_cast.w_out.bias
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return np.asarray(edge_features), np.asarray(node_features), np.asarray(log_probs)

  def _torch_forward(dtype: torch.dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pt_path is not None:
      ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
      state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    else:
      state_dict = _state_dict_from_aminx(loaded_eqx)

    pt_model = _RefProteinMPNN(k_neighbors=48)
    pt_model.load_state_dict(state_dict, strict=True)
    pt_model = pt_model.to(dtype=dtype)
    pt_model.eval()
    with torch.no_grad():
      x_t = torch.tensor(coords[None, ...], dtype=dtype)
      s_t = torch.tensor(sequence[None, ...], dtype=torch.long)
      mask_t = torch.tensor(mask[None, ...], dtype=dtype)
      res_idx_t = torch.tensor(residue_index[None, ...], dtype=torch.long)
      chain_enc_t = torch.tensor(chain_index[None, ...], dtype=torch.long)
      dec_order_t = torch.tensor(decoding_order[None, ...], dtype=torch.long)

      e_raw, e_idx = pt_model.features(x_t, mask_t, res_idx_t, chain_enc_t)
      h_e_0 = pt_model.W_e(e_raw)
      h_v = torch.zeros((e_raw.shape[0], e_raw.shape[1], e_raw.shape[-1]), dtype=dtype)
      h_e = h_e_0
      mask_attend_enc = _gather_nodes(mask_t.unsqueeze(-1), e_idx).squeeze(-1)
      mask_attend_enc = mask_t.unsqueeze(-1) * mask_attend_enc
      for layer in pt_model.encoder_layers:
        h_v, h_e = layer(h_v, h_e, e_idx, mask_t, mask_attend_enc)
      h_v_enc = h_v.clone()

      h_s = pt_model.W_s(s_t)
      h_es = _cat_neighbors_nodes(h_s, h_e, e_idx)
      h_ex_encoder = _cat_neighbors_nodes(torch.zeros_like(h_s), h_e, e_idx)
      h_exv_encoder = _cat_neighbors_nodes(h_v, h_ex_encoder, e_idx)

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
        h_esv = _cat_neighbors_nodes(h_v, h_es, e_idx)
        h_esv = mask_bw * h_esv + h_exv_encoder_fw
        h_v = layer(h_v, h_esv, mask_t)
      logits = pt_model.W_out(h_v)
      log_probs_t = torch.nn.functional.log_softmax(logits, dim=-1)
    return (
      h_e_0.squeeze(0).cpu().numpy(),
      h_v_enc.squeeze(0).cpu().numpy(),
      log_probs_t.squeeze(0).cpu().numpy(),
    )

  @contextlib.contextmanager
  def _x64(enabled: bool):  # type: ignore[no-untyped-def]
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", enabled)
    try:
      yield
    finally:
      jax.config.update("jax_enable_x64", prev)

  with _x64(True):
    jax_ref = convert_mpnn_checkpoint(pt_path) if pt_path is not None else loaded_eqx
    jax_edges_f64, jax_nodes_f64, jax_lp_f64 = _jax_forward(jax_ref, jnp.float64)
  pt_edges_f64, pt_nodes_f64, pt_lp_f64 = _torch_forward(torch.float64)
  assert np.max(np.abs(jax_edges_f64 - pt_edges_f64)) < 1e-7, cid
  assert np.max(np.abs(jax_nodes_f64 - pt_nodes_f64)) < 1e-7, cid
  assert np.max(np.abs(jax_lp_f64 - pt_lp_f64)) < 1e-7, cid

  with _x64(False):
    jax_edges_f32, jax_nodes_f32, jax_lp_f32 = _jax_forward(loaded_eqx, jnp.float32)
    pt_edges_f32, pt_nodes_f32, pt_lp_f32 = _torch_forward(torch.float32)
    np.testing.assert_allclose(jax_edges_f32, pt_edges_f32, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(jax_nodes_f32, pt_nodes_f32, atol=1e-4, rtol=1e-4)
    assert np.max(np.abs(jax_lp_f32 - pt_lp_f32)) < 1e-4, cid

    # Negative control: permuting w_out weight rows breaks parity by > 1e-1
    corrupt_model = eqx.tree_at(
      lambda m: m.w_out.weight,
      loaded_eqx,
      jnp.roll(loaded_eqx.w_out.weight, shift=3, axis=0),
    )
    _, _, corrupt_lp = _jax_forward(corrupt_model, jnp.float32)
    assert np.max(np.abs(corrupt_lp - pt_lp_f32)) > 1e-1, cid



def test_cdr_logit_transform_three_arg_and_masks_dataclass() -> None:
  """Verify 3-arg LogitTransform protocol, ArithmeticMeanLogits composition, and CDRRegionMasks descriptor support."""
  residue_idx = jnp.arange(1, 13, dtype=jnp.int32)
  chain_idx = jnp.zeros(12, dtype=jnp.int32)
  masks = build_cdr_region_mask(
    residue_idx,
    chain_idx,
    cdr_ranges={"H1": (1, 3), "H2": (5, 7), "H3": (9, 11)},
  )
  target = jnp.full((20,), 0.02, dtype=jnp.float32).at[19].set(0.62)
  transform = make_cdr_frequency_logit_transform(
    cdr_mask=masks,
    target_frequencies=target,
    weight=1.5,
    region_weights={"H1": 0.5, "H2": 1.0, "H3": 2.0},
  )
  assert isinstance(transform, CDRFrequencyLogitTransform)
  logits_1d = jnp.zeros((21,), dtype=jnp.float32)
  out_step_h3 = transform(logits_1d, jnp.int32(9), jnp.ones((12,), dtype=jnp.float32))
  out_step_fw = transform(logits_1d, jnp.int32(3), jnp.ones((12,), dtype=jnp.float32))
  assert not jnp.allclose(out_step_h3, logits_1d)
  assert jnp.allclose(out_step_fw, logits_1d, atol=1e-7)

  fused_fn = ArithmeticMeanLogits(jnp.ones(1, dtype=jnp.float32), transforms=transform)
  per_state = jnp.zeros((2, 12, 21), dtype=jnp.float32)
  fused_out = fused_fn(per_state, None)
  assert not jnp.allclose(fused_out[9], logits_1d)
  assert jnp.allclose(fused_out[3], logits_1d, atol=1e-7)

  seq_1d = jnp.array([19, 19, 19, 0, 2, 2, 2, 0, 14, 14, 14, 0], dtype=jnp.int32)
  desc_from_dataclass = compute_cdr_physicochemical_descriptors(seq_1d, masks)
  desc_from_array = compute_cdr_physicochemical_descriptors(seq_1d, masks.any_cdr)
  assert jnp.allclose(desc_from_dataclass.pd1_volume, desc_from_array.pd1_volume, atol=1e-6)
  assert jnp.allclose(desc_from_dataclass.pd2_hydro, desc_from_array.pd2_hydro, atol=1e-6)
  assert jnp.allclose(desc_from_dataclass.pd3_charge, desc_from_array.pd3_charge, atol=1e-6)
