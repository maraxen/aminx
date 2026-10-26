"""Parity, invariance, and negative-control tests for ThermoMPNN and ThermoMPNN-D.

Verifies:
1. Single-mutant ``ThermoMPNN`` v1 (``thermoMPNN_default.pt``) and v2
   (``ThermoMPNN-ens1.ckpt``) in ``float64`` (``atol=1e-7``) and ``float32``
   (``atol=1e-4``) against the PyTorch reference equations on real protein
   coordinates (``tests/data/1ubq.pdb``).
2. Wild-type self-mutation identity (``ddg[i, S[i]] == 0.0`` to machine precision).
3. Double-mutant ``ThermoMPNNDouble`` (``ThermoMPNN-D-ens1.ckpt``) in ``float64``
   (``atol=1e-7``) and ``float32`` (``atol=1e-4``) against the PyTorch
   ``TransferModelv2Siamese`` reference equations across both neighbor-connected
   pairs (``j in E_idx[i]``) and distant pairs (``j not in E_idx[i]``).
4. Order invariance: ``ThermoMPNNDouble`` on ``(p0, p1), (a0, a1)`` vs
   ``(p1, p0), (a1, a0)`` has ``max_abs_diff == 0.0``.
5. Negative controls:
   - reversing ``all_mpnn_hid`` layer order,
   - omitting ``wt_embed - mut_embed`` sequence-embedding subtraction,
   - zeroing edge features in ``ThermoMPNNDouble``,
   - permuting ``ddg_out`` / final MLP output rows
   all fail parity decisively.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch import nn

from aminx.families.thermo_mpnn import (
  THERMO_ALPHABET,
  ThermoMPNN,
  ThermoMPNNDouble,
  cast_floating,
  convert_thermo_mpnn_double_state_dict,
  convert_thermo_mpnn_state_dict,
  load_thermo_mpnn_checkpoint,
  load_thermo_mpnn_double_checkpoint,
  load_thermo_mpnn_double_pt,
  load_thermo_mpnn_pt,
  thermo_mpnn_trunk,
)
from aminx.families.thermo_mpnn.convert import _load_raw_checkpoint
from aminx.families.thermo_mpnn.model import _layer_norm, _linear, _siamese_mlp

_SCRATCH_REPOS = Path(
  "/home/marielle/.gemini/antigravity-cli/brain/e0a4519f-8415-4303-b518-8fc3d09cb0cc/scratch/repos",
)
_V1_CKPT = _SCRATCH_REPOS / "ThermoMPNN" / "models" / "thermoMPNN_default.pt"
_V1_VANILLA_MPNN = _SCRATCH_REPOS / "ThermoMPNN" / "vanilla_model_weights" / "v_48_020.pt"
_V2_SINGLE_CKPT = _SCRATCH_REPOS / "ThermoMPNN-D" / "model_weights" / "ThermoMPNN-ens1.ckpt"
_V2_DOUBLE_CKPT = _SCRATCH_REPOS / "ThermoMPNN-D" / "model_weights" / "ThermoMPNN-D-ens1.ckpt"
_V2_VANILLA_MPNN = _SCRATCH_REPOS / "ThermoMPNN-D" / "vanilla_model_weights" / "v_48_020.pt"
_UBQ_PDB = Path(__file__).resolve().parent.parent / "data" / "1ubq.pdb"

_AA3_TO_1 = {
  "ALA": "A",
  "CYS": "C",
  "ASP": "D",
  "GLU": "E",
  "PHE": "F",
  "GLY": "G",
  "HIS": "H",
  "ILE": "I",
  "LYS": "K",
  "LEU": "L",
  "MET": "M",
  "ASN": "N",
  "PRO": "P",
  "GLN": "Q",
  "ARG": "R",
  "SER": "S",
  "THR": "T",
  "VAL": "V",
  "TRP": "W",
  "TYR": "Y",
}


@contextlib.contextmanager
def _x64_mode(enabled: bool) -> Iterator[None]:
  """Temporarily set ``jax_enable_x64`` and restore on exit."""
  prev = bool(jax.config.jax_enable_x64)
  jax.config.update("jax_enable_x64", enabled)
  try:
    yield
  finally:
    jax.config.update("jax_enable_x64", prev)


def _parse_pdb_backbone(
  pdb_path: Path,
  max_residues: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Extract ``(coords, mask, residue_index, chain_index, sequence)`` from a PDB file."""
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
      if chain_id not in chain_order:
        chain_order.append(chain_id)
      key = (chain_id, res_seq)
      if key not in residues:
        residues[key] = {"res_name": res_name, "atoms": {}}
      residues[key]["atoms"][atom_name] = xyz

  keys = list(residues.keys())
  if max_residues is not None:
    keys = keys[:max_residues]
  num_res = len(keys)
  coords = np.zeros((num_res, 4, 3), dtype=np.float64)
  mask = np.zeros((num_res,), dtype=np.float64)
  residue_index = np.zeros((num_res,), dtype=np.int32)
  chain_index = np.zeros((num_res,), dtype=np.int32)
  sequence = np.zeros((num_res,), dtype=np.int32)

  for idx, (chain_id, res_seq) in enumerate(keys):
    entry = residues[(chain_id, res_seq)]
    atoms = entry["atoms"]
    if all(a in atoms for a in ("N", "CA", "C", "O")):
      coords[idx, 0] = atoms["N"]
      coords[idx, 1] = atoms["CA"]
      coords[idx, 2] = atoms["C"]
      coords[idx, 3] = atoms["O"]
      mask[idx] = 1.0
    residue_index[idx] = res_seq
    chain_index[idx] = chain_order.index(chain_id)
    aa1 = _AA3_TO_1.get(entry["res_name"], "X")
    sequence[idx] = THERMO_ALPHABET.index(aa1)

  return coords, mask, residue_index, chain_index, sequence


# ---------------------------------------------------------------------------
# Pure-PyTorch reference implementation matching upstream ThermoMPNN v1 & v2
# ---------------------------------------------------------------------------


def _torch_gather_edges(edges: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
  neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
  return torch.gather(edges, 2, neighbors)


def _torch_gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
  neighbors_flat = neighbor_idx.contiguous().view((neighbor_idx.shape[0], -1))
  nf = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
  neighbor_features = torch.gather(nodes, 1, nf)
  return neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])


def _torch_cat_neighbors_nodes(
  h_nodes: torch.Tensor,
  h_neighbors: torch.Tensor,
  e_idx: torch.Tensor,
) -> torch.Tensor:
  h_nodes_gathered = _torch_gather_nodes(h_nodes, e_idx)
  return torch.cat([h_neighbors, h_nodes_gathered], -1)


class _TorchPositionalEncodings(nn.Module):
  def __init__(self, num_embeddings: int, max_relative_feature: int = 32) -> None:
    super().__init__()
    self.num_embeddings = num_embeddings
    self.max_relative_feature = max_relative_feature
    self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

  def forward(self, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d = torch.clip(
      offset + self.max_relative_feature,
      0,
      2 * self.max_relative_feature,
    ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
    d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
    return self.linear(d_onehot.to(self.linear.weight.dtype))


class _TorchProteinFeatures(nn.Module):
  def __init__(
    self,
    edge_features: int = 128,
    node_features: int = 128,
    num_positional_embeddings: int = 16,
    num_rbf: int = 16,
    top_k: int = 48,
  ) -> None:
    super().__init__()
    self.edge_features = edge_features
    self.node_features = node_features
    self.top_k = top_k
    self.num_rbf = num_rbf
    self.embeddings = _TorchPositionalEncodings(num_positional_embeddings)
    edge_in = num_positional_embeddings + num_rbf * 25
    self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
    self.norm_edges = nn.LayerNorm(edge_features)

  def _dist(
    self,
    x: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    mask_2d = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
    dx = torch.unsqueeze(x, 1) - torch.unsqueeze(x, 2)
    d = mask_2d * torch.sqrt(torch.sum(dx**2, 3) + eps)
    d_max, _ = torch.max(d, -1, keepdim=True)
    d_adjust = d + (1.0 - mask_2d) * d_max
    k = min(self.top_k, x.shape[1])
    d_neighbors, e_idx = torch.topk(d_adjust, k, dim=-1, largest=False)
    return d_neighbors, e_idx

  def _rbf(self, d: torch.Tensor) -> torch.Tensor:
    d_min, d_max, d_count = 2.0, 22.0, self.num_rbf
    d_mu = torch.linspace(d_min, d_max, d_count, device=d.device, dtype=d.dtype)
    d_mu = d_mu.view([1, 1, 1, -1])
    d_sigma = (d_max - d_min) / d_count
    return torch.exp(-(((torch.unsqueeze(d, -1) - d_mu) / d_sigma) ** 2))

  def _get_rbf(self, a: torch.Tensor, b: torch.Tensor, e_idx: torch.Tensor) -> torch.Tensor:
    d_a_b = torch.sqrt(torch.sum((a[:, :, None, :] - b[:, None, :, :]) ** 2, -1) + 1e-6)
    d_neighbors = _torch_gather_edges(d_a_b[:, :, :, None], e_idx)[:, :, :, 0]
    return self._rbf(d_neighbors)

  def forward(
    self,
    x: torch.Tensor,
    mask: torch.Tensor,
    residue_idx: torch.Tensor,
    chain_labels: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    b = x[:, :, 1, :] - x[:, :, 0, :]
    c = x[:, :, 2, :] - x[:, :, 1, :]
    a = torch.cross(b, c, dim=-1)
    cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + x[:, :, 1, :]
    ca = x[:, :, 1, :]
    n = x[:, :, 0, :]
    c_atom = x[:, :, 2, :]
    o = x[:, :, 3, :]

    d_neighbors, e_idx = self._dist(ca, mask)
    rbf_all = [
      self._rbf(d_neighbors),
      self._get_rbf(n, n, e_idx),
      self._get_rbf(c_atom, c_atom, e_idx),
      self._get_rbf(o, o, e_idx),
      self._get_rbf(cb, cb, e_idx),
      self._get_rbf(ca, n, e_idx),
      self._get_rbf(ca, c_atom, e_idx),
      self._get_rbf(ca, o, e_idx),
      self._get_rbf(ca, cb, e_idx),
      self._get_rbf(n, c_atom, e_idx),
      self._get_rbf(n, o, e_idx),
      self._get_rbf(n, cb, e_idx),
      self._get_rbf(cb, c_atom, e_idx),
      self._get_rbf(cb, o, e_idx),
      self._get_rbf(o, c_atom, e_idx),
      self._get_rbf(n, ca, e_idx),
      self._get_rbf(c_atom, ca, e_idx),
      self._get_rbf(o, ca, e_idx),
      self._get_rbf(cb, ca, e_idx),
      self._get_rbf(c_atom, n, e_idx),
      self._get_rbf(o, n, e_idx),
      self._get_rbf(cb, n, e_idx),
      self._get_rbf(c_atom, cb, e_idx),
      self._get_rbf(o, cb, e_idx),
      self._get_rbf(c_atom, o, e_idx),
    ]
    rbf_cat = torch.cat(tuple(rbf_all), dim=-1)
    offset = residue_idx[:, :, None] - residue_idx[:, None, :]
    offset = _torch_gather_edges(offset[:, :, :, None], e_idx)[:, :, :, 0]
    d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
    e_chains = _torch_gather_edges(d_chains[:, :, :, None], e_idx)[:, :, :, 0]
    e_positional = self.embeddings(offset.long(), e_chains)
    e = torch.cat((e_positional, rbf_cat), -1)
    e = self.edge_embedding(e)
    e = self.norm_edges(e)
    return e, e_idx


class _TorchPositionWiseFeedForward(nn.Module):
  def __init__(self, num_hidden: int, num_ff: int) -> None:
    super().__init__()
    self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
    self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
    self.act = nn.GELU()

  def forward(self, h_v: torch.Tensor) -> torch.Tensor:
    return self.W_out(self.act(self.W_in(h_v)))


class _TorchEncLayer(nn.Module):
  def __init__(self, num_hidden: int, num_in: int, scale: float = 30.0) -> None:
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
    self.dense = _TorchPositionWiseFeedForward(num_hidden, num_hidden * 4)

  def forward(
    self,
    h_v: torch.Tensor,
    h_e: torch.Tensor,
    e_idx: torch.Tensor,
    mask_v: torch.Tensor,
    mask_attend: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    h_ev = _torch_cat_neighbors_nodes(h_v, h_e, e_idx)
    h_v_expand = h_v.unsqueeze(-2).expand(-1, -1, h_ev.size(-2), -1)
    h_ev = torch.cat([h_v_expand, h_ev], -1)
    h_message = self.W3(self.act(self.W2(self.act(self.W1(h_ev)))))
    h_message = mask_attend.unsqueeze(-1) * h_message
    dh = torch.sum(h_message, -2) / self.scale
    h_v = self.norm1(h_v + dh)
    dh = self.dense(h_v)
    h_v = self.norm2(h_v + dh)
    h_v = mask_v.unsqueeze(-1) * h_v

    h_ev = _torch_cat_neighbors_nodes(h_v, h_e, e_idx)
    h_v_expand = h_v.unsqueeze(-2).expand(-1, -1, h_ev.size(-2), -1)
    h_ev = torch.cat([h_v_expand, h_ev], -1)
    h_message = self.W13(self.act(self.W12(self.act(self.W11(h_ev)))))
    h_e = self.norm3(h_e + h_message)
    return h_v, h_e


class _TorchDecLayer(nn.Module):
  def __init__(self, num_hidden: int, num_in: int, scale: float = 30.0) -> None:
    super().__init__()
    self.scale = scale
    self.norm1 = nn.LayerNorm(num_hidden)
    self.norm2 = nn.LayerNorm(num_hidden)
    self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
    self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
    self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
    self.act = nn.GELU()
    self.dense = _TorchPositionWiseFeedForward(num_hidden, num_hidden * 4)

  def forward(self, h_v: torch.Tensor, h_e: torch.Tensor, mask_v: torch.Tensor) -> torch.Tensor:
    h_v_expand = h_v.unsqueeze(-2).expand(-1, -1, h_e.size(-2), -1)
    h_ev = torch.cat([h_v_expand, h_e], -1)
    h_message = self.W3(self.act(self.W2(self.act(self.W1(h_ev)))))
    dh = torch.sum(h_message, -2) / self.scale
    h_v = self.norm1(h_v + dh)
    dh = self.dense(h_v)
    h_v = self.norm2(h_v + dh)
    return mask_v.unsqueeze(-1) * h_v


class _TorchProteinMPNN(nn.Module):
  def __init__(
    self,
    hidden_dim: int = 128,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 3,
    vocab: int = 21,
    k_neighbors: int = 48,
    *,
    apply_chain_mask_to_order: bool = False,
  ) -> None:
    super().__init__()
    self.apply_chain_mask_to_order = apply_chain_mask_to_order
    self.features = _TorchProteinFeatures(hidden_dim, hidden_dim, top_k=k_neighbors)
    self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.W_s = nn.Embedding(vocab, hidden_dim)
    self.encoder_layers = nn.ModuleList(
      [_TorchEncLayer(hidden_dim, hidden_dim * 2) for _ in range(num_encoder_layers)],
    )
    self.decoder_layers = nn.ModuleList(
      [_TorchDecLayer(hidden_dim, hidden_dim * 3) for _ in range(num_decoder_layers)],
    )
    self.W_out = nn.Linear(hidden_dim, vocab, bias=True)

  def forward(
    self,
    x: torch.Tensor,
    s: torch.Tensor,
    mask: torch.Tensor,
    chain_m: torch.Tensor,
    residue_idx: torch.Tensor,
    chain_encoding_all: torch.Tensor,
  ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    e, e_idx = self.features(x, mask, residue_idx, chain_encoding_all)
    h_v = torch.zeros((e.shape[0], e.shape[1], e.shape[-1]), device=e.device, dtype=e.dtype)
    h_e = self.W_e(e)
    mask_attend = _torch_gather_nodes(mask.unsqueeze(-1), e_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    for enc_layer in self.encoder_layers:
      h_v, h_e = enc_layer(h_v, h_e, e_idx, mask, mask_attend)

    h_s = self.W_s(s)
    h_es = _torch_cat_neighbors_nodes(h_s, h_e, e_idx)
    h_ex_encoder = _torch_cat_neighbors_nodes(torch.zeros_like(h_s), h_e, e_idx)
    h_exv_encoder = _torch_cat_neighbors_nodes(h_v, h_ex_encoder, e_idx)

    eff_chain_m = chain_m * mask
    mask_size = e_idx.shape[1]
    order_mask_backward = torch.ones((x.shape[0], mask_size, mask_size), device=x.device, dtype=e.dtype)
    if self.apply_chain_mask_to_order:
      order_mask_backward = torch.einsum("bi,bij->bij", eff_chain_m, order_mask_backward)

    dec_mask_attend = torch.gather(order_mask_backward, 2, e_idx).unsqueeze(-1)
    mask_1d = mask.view([mask.size(0), mask.size(1), 1, 1])
    mask_bw = mask_1d * dec_mask_attend
    mask_fw = mask_1d * (1.0 - dec_mask_attend)
    h_exv_encoder_fw = mask_fw * h_exv_encoder

    all_hidden: list[torch.Tensor] = []
    for dec_layer in self.decoder_layers:
      h_esv = _torch_cat_neighbors_nodes(h_v, h_es, e_idx)
      h_esv = mask_bw * h_esv + h_exv_encoder_fw
      h_v = dec_layer(h_v, h_esv, mask)
      all_hidden.append(h_v)

    logits = self.W_out(h_v)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return list(reversed(all_hidden)), h_s, log_probs, h_e, e_idx


class _TorchLightAttention(nn.Module):
  def __init__(self, embeddings_dim: int = 384, kernel_size: int = 9) -> None:
    super().__init__()
    self.feature_convolution = nn.Conv1d(
      embeddings_dim,
      embeddings_dim,
      kernel_size,
      stride=1,
      padding=kernel_size // 2,
    )
    self.attention_convolution = nn.Conv1d(
      embeddings_dim,
      embeddings_dim,
      kernel_size,
      stride=1,
      padding=kernel_size // 2,
    )
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    o = self.feature_convolution(x)
    attention = self.attention_convolution(x)
    return torch.squeeze(o * self.softmax(attention), -1)


def _load_trunk_into_torch_mpnn(
  mpnn: _TorchProteinMPNN,
  norm_state: dict[str, np.ndarray],
  dtype: torch.dtype,
) -> None:
  mpnn_sd: dict[str, torch.Tensor] = {}
  for k, v in norm_state.items():
    if k.startswith("prot_mpnn."):
      mpnn_sd[k.removeprefix("prot_mpnn.")] = torch.as_tensor(v, dtype=dtype)
  mpnn.load_state_dict(mpnn_sd, strict=True)


def _run_torch_single_reference(
  norm_state: dict[str, np.ndarray],
  coords: np.ndarray,
  mask: np.ndarray,
  residue_index: np.ndarray,
  chain_index: np.ndarray,
  sequence: np.ndarray,
  *,
  dtype: torch.dtype,
) -> np.ndarray:
  """Execute upstream PyTorch single-mutant ThermoMPNN (v1 or v2) on ``coords``."""
  is_v1 = "both_out.1.weight" in norm_state
  mlp_prefix = "both_out" if is_v1 else "ddg_out"
  kernel_size = int(norm_state["light_attention.feature_convolution.weight"].shape[2])

  mpnn = _TorchProteinMPNN(apply_chain_mask_to_order=not is_v1).to(dtype)
  _load_trunk_into_torch_mpnn(mpnn, norm_state, dtype)
  mpnn.eval()

  la = _TorchLightAttention(embeddings_dim=384, kernel_size=kernel_size).to(dtype)
  la.load_state_dict(
    {
      "feature_convolution.weight": torch.as_tensor(
        norm_state["light_attention.feature_convolution.weight"],
        dtype=dtype,
      ),
      "feature_convolution.bias": torch.as_tensor(
        norm_state["light_attention.feature_convolution.bias"],
        dtype=dtype,
      ),
      "attention_convolution.weight": torch.as_tensor(
        norm_state["light_attention.attention_convolution.weight"],
        dtype=dtype,
      ),
      "attention_convolution.bias": torch.as_tensor(
        norm_state["light_attention.attention_convolution.bias"],
        dtype=dtype,
      ),
    },
    strict=True,
  )
  la.eval()

  l1 = nn.Linear(384, 64).to(dtype)
  l2 = nn.Linear(64, 32).to(dtype)
  l3 = nn.Linear(32, 21).to(dtype)
  for idx, layer in zip((1, 3, 5), (l1, l2, l3), strict=True):
    layer.weight.data.copy_(torch.as_tensor(norm_state[f"{mlp_prefix}.{idx}.weight"], dtype=dtype))
    layer.bias.data.copy_(torch.as_tensor(norm_state[f"{mlp_prefix}.{idx}.bias"], dtype=dtype))

  ddg_scale: nn.Linear | None = None
  if is_v1 and "ddg_out.weight" in norm_state:
    ddg_scale = nn.Linear(1, 1).to(dtype)
    ddg_scale.weight.data.copy_(torch.as_tensor(norm_state["ddg_out.weight"], dtype=dtype))
    ddg_scale.bias.data.copy_(torch.as_tensor(norm_state["ddg_out.bias"], dtype=dtype))

  with torch.no_grad():
    x_t = torch.as_tensor(coords[None, ...], dtype=dtype)
    s_t = torch.as_tensor(sequence[None, ...], dtype=torch.long)
    mask_t = torch.as_tensor(mask[None, ...], dtype=dtype)
    chain_m_t = mask_t.clone()
    res_idx_t = torch.as_tensor(residue_index[None, ...], dtype=torch.long)
    chain_idx_t = torch.as_tensor(chain_index[None, ...], dtype=torch.long)

    all_mpnn_hid, mpnn_embed, _, _, _ = mpnn(
      x_t,
      s_t,
      mask_t,
      chain_m_t,
      res_idx_t,
      chain_idx_t,
    )
    hid_cat = torch.cat(all_mpnn_hid[:2], dim=-1)
    embeds = torch.cat([hid_cat, mpnn_embed], dim=-1)[0]  # (L, 384)
    # Upstream passes [B, 384, 1] per mutation position into LightAttention
    la_out = la(embeds.unsqueeze(-1))  # (L, 384)
    h = torch.relu(la_out)
    h = torch.relu(l1(h))
    h = torch.relu(l2(h))
    raw_ddg = l3(h)  # (L, 21)
    if ddg_scale is not None:
      raw_ddg = ddg_scale(raw_ddg.unsqueeze(-1)).squeeze(-1)
    wt_ddg = torch.gather(raw_ddg, 1, s_t[0, :, None])
    ddg = raw_ddg - wt_ddg
    return ddg.cpu().numpy()


def _run_torch_double_reference(
  norm_state: dict[str, np.ndarray],
  coords: np.ndarray,
  mask: np.ndarray,
  residue_index: np.ndarray,
  chain_index: np.ndarray,
  sequence: np.ndarray,
  mut_positions: np.ndarray,
  mut_aas: np.ndarray,
  *,
  dtype: torch.dtype,
) -> np.ndarray:
  """Execute upstream PyTorch ``TransferModelv2Siamese`` on ``coords`` and mutations."""
  mpnn = _TorchProteinMPNN(apply_chain_mask_to_order=True).to(dtype)
  _load_trunk_into_torch_mpnn(mpnn, norm_state, dtype)
  mpnn.eval()

  proj_norm = nn.LayerNorm(512).to(dtype)
  proj_norm.weight.data.copy_(torch.as_tensor(norm_state["light_attention.0.weight"], dtype=dtype))
  proj_norm.bias.data.copy_(torch.as_tensor(norm_state["light_attention.0.bias"], dtype=dtype))
  proj_lin = nn.Linear(512, 128).to(dtype)
  proj_lin.weight.data.copy_(torch.as_tensor(norm_state["light_attention.1.weight"], dtype=dtype))
  proj_lin.bias.data.copy_(torch.as_tensor(norm_state["light_attention.1.bias"], dtype=dtype))

  d0 = nn.Linear(256, 128).to(dtype)
  d3 = nn.Linear(128, 128).to(dtype)
  d6 = nn.Linear(128, 1).to(dtype)
  for idx, layer in zip((0, 3, 6), (d0, d3, d6), strict=True):
    layer.weight.data.copy_(torch.as_tensor(norm_state[f"ddg_out.{idx}.weight"], dtype=dtype))
    layer.bias.data.copy_(torch.as_tensor(norm_state[f"ddg_out.{idx}.bias"], dtype=dtype))

  with torch.no_grad():
    x_t = torch.as_tensor(coords[None, ...], dtype=dtype)
    s_t = torch.as_tensor(sequence[None, ...], dtype=torch.long)
    mask_t = torch.as_tensor(mask[None, ...], dtype=dtype)
    chain_m_t = mask_t.clone()
    res_idx_t = torch.as_tensor(residue_index[None, ...], dtype=torch.long)
    chain_idx_t = torch.as_tensor(chain_index[None, ...], dtype=torch.long)
    pos_t = torch.as_tensor(mut_positions, dtype=torch.long)
    aas_t = torch.as_tensor(mut_aas, dtype=torch.long)

    all_mpnn_hid, wt_embed, _, mpnn_edges, e_idx = mpnn(
      x_t,
      s_t,
      mask_t,
      chain_m_t,
      res_idx_t,
      chain_idx_t,
    )
    struct_embed = torch.cat(all_mpnn_hid[:2], dim=-1)[0]  # (L, 256)
    wt_seq_embed = wt_embed[0]  # (L, 128)
    edges_l = mpnn_edges[0]  # (L, K, 128)
    e_idx_l = e_idx[0]  # (L, K)

    slot_list: list[torch.Tensor] = []
    for m in range(2):
      pos_m = pos_t[:, m]
      pos_other = pos_t[:, 1 - m]
      aa_m = aas_t[:, m]

      e_idx_m = e_idx_l[pos_m]  # (M, K)
      edges_m = edges_l[pos_m]  # (M, K, 128)
      matches = e_idx_m == pos_other[:, None]
      edge_vecs = torch.zeros(( pos_t.shape[0], 128), dtype=dtype)
      for row_idx in range(pos_t.shape[0]):
        hit = torch.nonzero(matches[row_idx], as_tuple=False)
        if hit.numel() > 0:
          k_idx = int(hit[0, 0].item())
          edge_vecs[row_idx] = edges_m[row_idx, k_idx]

      g_struct = struct_embed[pos_m]
      g_seq = wt_seq_embed[pos_m] - mpnn.W_s(aa_m)
      g_embed = torch.cat([g_struct, g_seq, edge_vecs], dim=-1)
      h_m = torch.relu(proj_lin(proj_norm(g_embed)))
      slot_list.append(h_m)

    embed_ab = torch.cat([slot_list[0], slot_list[1]], dim=-1)
    embed_ba = torch.cat([slot_list[1], slot_list[0]], dim=-1)
    ddg_a = d6(torch.relu(d3(torch.relu(d0(embed_ab))))).squeeze(-1)
    ddg_b = d6(torch.relu(d3(torch.relu(d0(embed_ba))))).squeeze(-1)
    return (0.5 * (ddg_a + ddg_b)).cpu().numpy()


# ---------------------------------------------------------------------------
# Deterministic synthetic state-dict builders (also used when checkpoints absent)
# ---------------------------------------------------------------------------


def _build_synthetic_mpnn_state_dict(rng: np.random.Generator) -> dict[str, np.ndarray]:
  def _u(*shape: int) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float32) * 0.08

  sd: dict[str, np.ndarray] = {
    "prot_mpnn.features.embeddings.linear.weight": _u(16, 66),
    "prot_mpnn.features.embeddings.linear.bias": _u(16),
    "prot_mpnn.features.edge_embedding.weight": _u(128, 416),
    "prot_mpnn.features.norm_edges.weight": np.ones((128,), dtype=np.float32) + _u(128) * 0.1,
    "prot_mpnn.features.norm_edges.bias": _u(128),
    "prot_mpnn.W_e.weight": _u(128, 128),
    "prot_mpnn.W_e.bias": _u(128),
    "prot_mpnn.W_s.weight": _u(21, 128),
    "prot_mpnn.W_out.weight": _u(21, 128),
    "prot_mpnn.W_out.bias": _u(21),
  }
  for idx in range(3):
    p = f"prot_mpnn.encoder_layers.{idx}."
    sd[f"{p}W1.weight"] = _u(128, 384)
    sd[f"{p}W1.bias"] = _u(128)
    sd[f"{p}W2.weight"] = _u(128, 128)
    sd[f"{p}W2.bias"] = _u(128)
    sd[f"{p}W3.weight"] = _u(128, 128)
    sd[f"{p}W3.bias"] = _u(128)
    sd[f"{p}W11.weight"] = _u(128, 384)
    sd[f"{p}W11.bias"] = _u(128)
    sd[f"{p}W12.weight"] = _u(128, 128)
    sd[f"{p}W12.bias"] = _u(128)
    sd[f"{p}W13.weight"] = _u(128, 128)
    sd[f"{p}W13.bias"] = _u(128)
    sd[f"{p}dense.W_in.weight"] = _u(512, 128)
    sd[f"{p}dense.W_in.bias"] = _u(512)
    sd[f"{p}dense.W_out.weight"] = _u(128, 512)
    sd[f"{p}dense.W_out.bias"] = _u(128)
    for norm_i in (1, 2, 3):
      sd[f"{p}norm{norm_i}.weight"] = np.ones((128,), dtype=np.float32) + _u(128) * 0.1
      sd[f"{p}norm{norm_i}.bias"] = _u(128)

  for idx in range(3):
    p = f"prot_mpnn.decoder_layers.{idx}."
    sd[f"{p}W1.weight"] = _u(128, 512)
    sd[f"{p}W1.bias"] = _u(128)
    sd[f"{p}W2.weight"] = _u(128, 128)
    sd[f"{p}W2.bias"] = _u(128)
    sd[f"{p}W3.weight"] = _u(128, 128)
    sd[f"{p}W3.bias"] = _u(128)
    sd[f"{p}dense.W_in.weight"] = _u(512, 128)
    sd[f"{p}dense.W_in.bias"] = _u(512)
    sd[f"{p}dense.W_out.weight"] = _u(128, 512)
    sd[f"{p}dense.W_out.bias"] = _u(128)
    for norm_i in (1, 2):
      sd[f"{p}norm{norm_i}.weight"] = np.ones((128,), dtype=np.float32) + _u(128) * 0.1
      sd[f"{p}norm{norm_i}.bias"] = _u(128)
  return sd


def _build_v1_state_dict() -> dict[str, np.ndarray]:
  if _V1_CKPT.is_file():
    sd = _load_raw_checkpoint(_V1_CKPT)
    if "prot_mpnn.W_s.weight" not in sd and _V1_VANILLA_MPNN.is_file():
      mpnn_sd = _load_raw_checkpoint(_V1_VANILLA_MPNN)
      for k, v in mpnn_sd.items():
        sd[f"prot_mpnn.{k}"] = v
    return sd
  rng = np.random.default_rng(101)
  sd = _build_synthetic_mpnn_state_dict(rng)
  sd["light_attention.feature_convolution.weight"] = rng.standard_normal((384, 384, 9)).astype(np.float32) * 0.05
  sd["light_attention.feature_convolution.bias"] = rng.standard_normal((384,)).astype(np.float32) * 0.05
  sd["light_attention.attention_convolution.weight"] = rng.standard_normal((384, 384, 9)).astype(np.float32) * 0.05
  sd["light_attention.attention_convolution.bias"] = rng.standard_normal((384,)).astype(np.float32) * 0.05
  sd["both_out.1.weight"] = rng.standard_normal((64, 384)).astype(np.float32) * 0.08
  sd["both_out.1.bias"] = rng.standard_normal((64,)).astype(np.float32) * 0.08
  sd["both_out.3.weight"] = rng.standard_normal((32, 64)).astype(np.float32) * 0.08
  sd["both_out.3.bias"] = rng.standard_normal((32,)).astype(np.float32) * 0.08
  sd["both_out.5.weight"] = rng.standard_normal((21, 32)).astype(np.float32) * 0.08
  sd["both_out.5.bias"] = rng.standard_normal((21,)).astype(np.float32) * 0.08
  sd["ddg_out.weight"] = np.array([[-1.35]], dtype=np.float32)
  sd["ddg_out.bias"] = np.array([0.12], dtype=np.float32)
  return sd


def _build_v2_single_state_dict() -> dict[str, np.ndarray]:
  if _V2_SINGLE_CKPT.is_file():
    sd = _load_raw_checkpoint(_V2_SINGLE_CKPT)
    if "prot_mpnn.W_s.weight" not in sd and _V2_VANILLA_MPNN.is_file():
      mpnn_sd = _load_raw_checkpoint(_V2_VANILLA_MPNN)
      for k, v in mpnn_sd.items():
        sd[f"prot_mpnn.{k}"] = v
    return sd
  rng = np.random.default_rng(202)
  sd = _build_synthetic_mpnn_state_dict(rng)
  sd["light_attention.feature_convolution.weight"] = rng.standard_normal((384, 384, 1)).astype(np.float32) * 0.05
  sd["light_attention.feature_convolution.bias"] = rng.standard_normal((384,)).astype(np.float32) * 0.05
  sd["light_attention.attention_convolution.weight"] = rng.standard_normal((384, 384, 1)).astype(np.float32) * 0.05
  sd["light_attention.attention_convolution.bias"] = rng.standard_normal((384,)).astype(np.float32) * 0.05
  sd["ddg_out.1.weight"] = rng.standard_normal((64, 384)).astype(np.float32) * 0.08
  sd["ddg_out.1.bias"] = rng.standard_normal((64,)).astype(np.float32) * 0.08
  sd["ddg_out.3.weight"] = rng.standard_normal((32, 64)).astype(np.float32) * 0.08
  sd["ddg_out.3.bias"] = rng.standard_normal((32,)).astype(np.float32) * 0.08
  sd["ddg_out.5.weight"] = rng.standard_normal((21, 32)).astype(np.float32) * 0.08
  sd["ddg_out.5.bias"] = rng.standard_normal((21,)).astype(np.float32) * 0.08
  return sd


def _build_v2_double_state_dict() -> dict[str, np.ndarray]:
  if _V2_DOUBLE_CKPT.is_file():
    sd = _load_raw_checkpoint(_V2_DOUBLE_CKPT)
    if "prot_mpnn.W_s.weight" not in sd and _V2_VANILLA_MPNN.is_file():
      mpnn_sd = _load_raw_checkpoint(_V2_VANILLA_MPNN)
      for k, v in mpnn_sd.items():
        sd[f"prot_mpnn.{k}"] = v
    return sd
  rng = np.random.default_rng(303)
  sd = _build_synthetic_mpnn_state_dict(rng)
  sd["light_attention.0.weight"] = np.ones((512,), dtype=np.float32) + rng.standard_normal((512,)).astype(np.float32) * 0.05
  sd["light_attention.0.bias"] = rng.standard_normal((512,)).astype(np.float32) * 0.05
  sd["light_attention.1.weight"] = rng.standard_normal((128, 512)).astype(np.float32) * 0.08
  sd["light_attention.1.bias"] = rng.standard_normal((128,)).astype(np.float32) * 0.08
  sd["ddg_out.0.weight"] = rng.standard_normal((128, 256)).astype(np.float32) * 0.08
  sd["ddg_out.0.bias"] = rng.standard_normal((128,)).astype(np.float32) * 0.08
  sd["ddg_out.3.weight"] = rng.standard_normal((128, 128)).astype(np.float32) * 0.08
  sd["ddg_out.3.bias"] = rng.standard_normal((128,)).astype(np.float32) * 0.08
  sd["ddg_out.6.weight"] = rng.standard_normal((1, 128)).astype(np.float32) * 0.08
  sd["ddg_out.6.bias"] = rng.standard_normal((1,)).astype(np.float32) * 0.08
  return sd


# ---------------------------------------------------------------------------
# 1. Single-mutant ThermoMPNN v1 & v2 Parity (float64 & float32)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["v1", "v2"])
def test_thermo_mpnn_single_parity_float64_and_float32(variant: str) -> None:
  """ThermoMPNN matches PyTorch reference within 1e-7 (float64) and 1e-4 (float32)."""
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  state_dict = _build_v1_state_dict() if variant == "v1" else _build_v2_single_state_dict()

  with _x64_mode(True):
    model = convert_thermo_mpnn_state_dict(state_dict)
    jax_f64 = np.asarray(
      model(
        jnp.asarray(coords, dtype=jnp.float64),
        jnp.asarray(mask, dtype=jnp.float64),
        jnp.asarray(residue_index, dtype=jnp.int32),
        jnp.asarray(chain_index, dtype=jnp.int32),
        jnp.asarray(sequence, dtype=jnp.int32),
      ),
    )
  torch_f64 = _run_torch_single_reference(
    state_dict,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    dtype=torch.float64,
  )
  max_diff_f64 = float(np.max(np.abs(jax_f64 - torch_f64)))
  assert max_diff_f64 < 1e-7, f"{variant} float64 max_abs_diff={max_diff_f64:.3e} >= 1e-7"

  with _x64_mode(False):
    model_f32 = convert_thermo_mpnn_state_dict(state_dict)
    jax_f32 = np.asarray(
      model_f32(
        jnp.asarray(coords, dtype=jnp.float32),
        jnp.asarray(mask, dtype=jnp.float32),
        jnp.asarray(residue_index, dtype=jnp.int32),
        jnp.asarray(chain_index, dtype=jnp.int32),
        jnp.asarray(sequence, dtype=jnp.int32),
      ),
    )
  torch_f32 = _run_torch_single_reference(
    state_dict,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    dtype=torch.float32,
  )
  max_diff_f32 = float(np.max(np.abs(jax_f32 - torch_f32)))
  assert max_diff_f32 < 1e-4, f"{variant} float32 max_abs_diff={max_diff_f32:.3e} >= 1e-4"


# ---------------------------------------------------------------------------
# 2. Wild-type self-mutation identity (ddg[i, S[i]] == 0.0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["v1", "v2"])
def test_thermo_mpnn_wildtype_self_mutation_zero(variant: str) -> None:
  """Self-mutation ddG[i, S[i]] is identically 0.0 to machine precision."""
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  state_dict = _build_v1_state_dict() if variant == "v1" else _build_v2_single_state_dict()

  with _x64_mode(True):
    model = convert_thermo_mpnn_state_dict(state_dict)
    ddg = np.asarray(
      model(
        jnp.asarray(coords, dtype=jnp.float64),
        jnp.asarray(mask, dtype=jnp.float64),
        jnp.asarray(residue_index, dtype=jnp.int32),
        jnp.asarray(chain_index, dtype=jnp.int32),
        jnp.asarray(sequence, dtype=jnp.int32),
      ),
    )
  wt_self = ddg[np.arange(sequence.shape[0]), sequence]
  assert float(np.max(np.abs(wt_self))) == 0.0


# ---------------------------------------------------------------------------
# 3. Double-mutant ThermoMPNN-D Parity (neighbor + distant pairs)
# ---------------------------------------------------------------------------


def _select_neighbor_and_distant_mutations(
  coords: np.ndarray,
  mask: np.ndarray,
  residue_index: np.ndarray,
  chain_index: np.ndarray,
  sequence: np.ndarray,
  model: ThermoMPNNDouble,
) -> tuple[np.ndarray, np.ndarray, int, int]:
  """Construct double mutations spanning both ``j in E_idx[i]`` and ``j not in E_idx[i]``."""
  trunk = thermo_mpnn_trunk(
    model.mpnn,
    jnp.asarray(coords, dtype=jnp.float32),
    jnp.asarray(mask, dtype=jnp.float32),
    jnp.asarray(residue_index, dtype=jnp.int32),
    jnp.asarray(chain_index, dtype=jnp.int32),
    jnp.asarray(sequence, dtype=jnp.int32),
    apply_chain_mask_to_order=True,
  )
  e_idx = np.asarray(trunk.neighbor_indices)
  num_res = e_idx.shape[0]

  neighbor_pairs: list[tuple[int, int]] = []
  distant_pairs: list[tuple[int, int]] = []
  for i in range(num_res):
    neighbors_i = set(int(x) for x in e_idx[i])
    for j in range(i + 1, num_res):
      if j in neighbors_i and len(neighbor_pairs) < 8:
        neighbor_pairs.append((i, j))
      elif j not in neighbors_i and len(distant_pairs) < 8:
        distant_pairs.append((i, j))
    if len(neighbor_pairs) >= 8 and len(distant_pairs) >= 8:
      break

  assert len(neighbor_pairs) > 0, "Expected at least one neighbor-connected pair"
  assert len(distant_pairs) > 0, "Expected at least one distant pair (L > 48)"

  all_pairs = neighbor_pairs + distant_pairs
  mut_positions = np.asarray(all_pairs, dtype=np.int32)
  mut_aas = np.stack(
    [
      (sequence[mut_positions[:, 0]] + 3) % 20,
      (sequence[mut_positions[:, 1]] + 7) % 20,
    ],
    axis=-1,
  ).astype(np.int32)
  return mut_positions, mut_aas, len(neighbor_pairs), len(distant_pairs)


def test_thermo_mpnn_double_parity_float64_and_float32() -> None:
  """ThermoMPNNDouble matches PyTorch TransferModelv2Siamese on neighbor and distant pairs."""
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  state_dict = _build_v2_double_state_dict()

  with _x64_mode(True):
    model = convert_thermo_mpnn_double_state_dict(state_dict)
    mut_positions, mut_aas, n_neigh, n_dist = _select_neighbor_and_distant_mutations(
      coords,
      mask,
      residue_index,
      chain_index,
      sequence,
      model,
    )
    assert n_neigh >= 4
    assert n_dist >= 4
    jax_f64 = np.asarray(
      model(
        jnp.asarray(coords, dtype=jnp.float64),
        jnp.asarray(mask, dtype=jnp.float64),
        jnp.asarray(residue_index, dtype=jnp.int32),
        jnp.asarray(chain_index, dtype=jnp.int32),
        jnp.asarray(sequence, dtype=jnp.int32),
        jnp.asarray(mut_positions, dtype=jnp.int32),
        jnp.asarray(mut_aas, dtype=jnp.int32),
      ),
    )

  torch_f64 = _run_torch_double_reference(
    state_dict,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    mut_positions,
    mut_aas,
    dtype=torch.float64,
  )
  max_diff_f64 = float(np.max(np.abs(jax_f64 - torch_f64)))
  assert max_diff_f64 < 1e-7, f"ThermoMPNN-D float64 max_abs_diff={max_diff_f64:.3e} >= 1e-7"

  with _x64_mode(False):
    model_f32 = convert_thermo_mpnn_double_state_dict(state_dict)
    jax_f32 = np.asarray(
      model_f32(
        jnp.asarray(coords, dtype=jnp.float32),
        jnp.asarray(mask, dtype=jnp.float32),
        jnp.asarray(residue_index, dtype=jnp.int32),
        jnp.asarray(chain_index, dtype=jnp.int32),
        jnp.asarray(sequence, dtype=jnp.int32),
        jnp.asarray(mut_positions, dtype=jnp.int32),
        jnp.asarray(mut_aas, dtype=jnp.int32),
      ),
    )
  torch_f32 = _run_torch_double_reference(
    state_dict,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    mut_positions,
    mut_aas,
    dtype=torch.float32,
  )
  max_diff_f32 = float(np.max(np.abs(jax_f32 - torch_f32)))
  assert max_diff_f32 < 1e-4, f"ThermoMPNN-D float32 max_abs_diff={max_diff_f32:.3e} >= 1e-4"


# ---------------------------------------------------------------------------
# 4. Double-mutant Siamese Order Invariance
# ---------------------------------------------------------------------------


def test_thermo_mpnn_double_order_invariance() -> None:
  """Swapping mutation slots ``(p0, p1), (a0, a1) <-> (p1, p0), (a1, a0)`` has 0.0 diff."""
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  state_dict = _build_v2_double_state_dict()

  with _x64_mode(True):
    model = convert_thermo_mpnn_double_state_dict(state_dict)
    mut_positions, mut_aas, _, _ = _select_neighbor_and_distant_mutations(
      coords,
      mask,
      residue_index,
      chain_index,
      sequence,
      model,
    )
    ddg_ab = np.asarray(
      model(
        jnp.asarray(coords, dtype=jnp.float64),
        jnp.asarray(mask, dtype=jnp.float64),
        jnp.asarray(residue_index, dtype=jnp.int32),
        jnp.asarray(chain_index, dtype=jnp.int32),
        jnp.asarray(sequence, dtype=jnp.int32),
        jnp.asarray(mut_positions, dtype=jnp.int32),
        jnp.asarray(mut_aas, dtype=jnp.int32),
      ),
    )
    ddg_ba = np.asarray(
      model(
        jnp.asarray(coords, dtype=jnp.float64),
        jnp.asarray(mask, dtype=jnp.float64),
        jnp.asarray(residue_index, dtype=jnp.int32),
        jnp.asarray(chain_index, dtype=jnp.int32),
        jnp.asarray(sequence, dtype=jnp.int32),
        jnp.asarray(mut_positions[:, ::-1], dtype=jnp.int32),
        jnp.asarray(mut_aas[:, ::-1], dtype=jnp.int32),
      ),
    )
  assert float(np.max(np.abs(ddg_ab - ddg_ba))) == 0.0


# ---------------------------------------------------------------------------
# 5. Negative Controls (must fail parity decisively)
# ---------------------------------------------------------------------------


def _corrupt_single_unreversed_hidden(
  model: ThermoMPNN,
  coords: jax.Array,
  mask: jax.Array,
  residue_index: jax.Array,
  chain_index: jax.Array,
  sequence: jax.Array,
) -> jax.Array:
  """Negative control A: use unreversed ``[dec_0, dec_1]`` instead of ``[dec_2, dec_1]``."""
  dtype = coords.dtype
  model_c = cast_floating(model, dtype)
  trunk = thermo_mpnn_trunk(
    model_c.mpnn,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    apply_chain_mask_to_order=model_c.apply_chain_mask_to_order,
  )
  unreversed = tuple(reversed(trunk.all_mpnn_hid))
  z = jnp.concatenate([*unreversed[: model_c.num_final_layers], trunk.mpnn_embed], axis=-1)
  h = model_c.light_attention(z)
  for layer in model_c.mlp_layers:
    h = jax.nn.relu(h)
    h = _linear(layer, h)
  raw_ddg = h
  if model_c.ddg_out is not None:
    raw_ddg = _linear(model_c.ddg_out, raw_ddg[..., None])[..., 0]
  wt_ddg = jnp.take_along_axis(raw_ddg, sequence[:, None].astype(jnp.int32), axis=-1)
  return raw_ddg - wt_ddg


def _corrupt_double_omit_seq_subtraction(
  model: ThermoMPNNDouble,
  coords: jax.Array,
  mask: jax.Array,
  residue_index: jax.Array,
  chain_index: jax.Array,
  sequence: jax.Array,
  mut_positions: jax.Array,
  mut_aas: jax.Array,
) -> jax.Array:
  """Negative control B: omit ``wt_embed - mut_embed`` subtraction (use ``wt_embed`` alone)."""
  dtype = coords.dtype
  model_c = cast_floating(model, dtype)
  trunk = thermo_mpnn_trunk(
    model_c.mpnn,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    apply_chain_mask_to_order=True,
  )
  struct_embed = jnp.concatenate(trunk.all_mpnn_hid[: model_c.num_final_layers], axis=-1)
  wt_embed = trunk.mpnn_embed
  mpnn_edges = trunk.mpnn_edges
  neighbor_indices = trunk.neighbor_indices

  slot_embeds: list[jax.Array] = []
  for m in range(2):
    pos_m = mut_positions[:, m].astype(jnp.int32)
    pos_other = mut_positions[:, 1 - m].astype(jnp.int32)
    e_idx_m = neighbor_indices[pos_m]
    edges_m = mpnn_edges[pos_m]
    matches = e_idx_m == pos_other[:, None]
    has_match = jnp.any(matches, axis=-1, keepdims=True)
    first_k = jnp.argmax(matches, axis=-1)
    selected_edge = jnp.take_along_axis(edges_m, first_k[:, None, None], axis=1)[:, 0, :]
    edge_vec = jnp.where(has_match, selected_edge, jnp.zeros_like(selected_edge))
    # Corrupted: omit - model_c.mpnn.w_s_embed.weight[aa_m]
    g_seq = wt_embed[pos_m]
    g_embed = jnp.concatenate([struct_embed[pos_m], g_seq, edge_vec], axis=-1)
    h_m = jax.nn.relu(_linear(model_c.proj_linear, _layer_norm(model_c.proj_norm, g_embed)))
    slot_embeds.append(h_m)

  embed_ab = jnp.concatenate([slot_embeds[0], slot_embeds[1]], axis=-1)
  embed_ba = jnp.concatenate([slot_embeds[1], slot_embeds[0]], axis=-1)
  ddg_a = _siamese_mlp(model_c.mlp_layers, embed_ab)[..., 0]
  ddg_b = _siamese_mlp(model_c.mlp_layers, embed_ba)[..., 0]
  return 0.5 * (ddg_a + ddg_b)


def _corrupt_double_zero_edges(
  model: ThermoMPNNDouble,
  coords: jax.Array,
  mask: jax.Array,
  residue_index: jax.Array,
  chain_index: jax.Array,
  sequence: jax.Array,
  mut_positions: jax.Array,
  mut_aas: jax.Array,
) -> jax.Array:
  """Negative control C: zero out ``mpnn_edges`` in ``ThermoMPNNDouble``."""
  dtype = coords.dtype
  model_c = cast_floating(model, dtype)
  trunk = thermo_mpnn_trunk(
    model_c.mpnn,
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    apply_chain_mask_to_order=True,
  )
  struct_embed = jnp.concatenate(trunk.all_mpnn_hid[: model_c.num_final_layers], axis=-1)
  wt_embed = trunk.mpnn_embed

  slot_embeds: list[jax.Array] = []
  for m in range(2):
    pos_m = mut_positions[:, m].astype(jnp.int32)
    aa_m = mut_aas[:, m].astype(jnp.int32)
    edge_vec = jnp.zeros((pos_m.shape[0], 128), dtype=dtype)
    g_seq = wt_embed[pos_m] - model_c.mpnn.w_s_embed.weight[aa_m]
    g_embed = jnp.concatenate([struct_embed[pos_m], g_seq, edge_vec], axis=-1)
    h_m = jax.nn.relu(_linear(model_c.proj_linear, _layer_norm(model_c.proj_norm, g_embed)))
    slot_embeds.append(h_m)

  embed_ab = jnp.concatenate([slot_embeds[0], slot_embeds[1]], axis=-1)
  embed_ba = jnp.concatenate([slot_embeds[1], slot_embeds[0]], axis=-1)
  ddg_a = _siamese_mlp(model_c.mlp_layers, embed_ab)[..., 0]
  ddg_b = _siamese_mlp(model_c.mlp_layers, embed_ba)[..., 0]
  return 0.5 * (ddg_a + ddg_b)


def test_negative_controls_fail_parity_decisively() -> None:
  """All four negative controls diverge strongly (> 1e-2) from the PyTorch reference."""
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  v1_sd = _build_v1_state_dict()
  v2_double_sd = _build_v2_double_state_dict()

  with _x64_mode(True):
    single_model = convert_thermo_mpnn_state_dict(v1_sd)
    double_model = convert_thermo_mpnn_double_state_dict(v2_double_sd)
    mut_positions, mut_aas, _, _ = _select_neighbor_and_distant_mutations(
      coords,
      mask,
      residue_index,
      chain_index,
      sequence,
      double_model,
    )

    coords_j = jnp.asarray(coords, dtype=jnp.float64)
    mask_j = jnp.asarray(mask, dtype=jnp.float64)
    res_j = jnp.asarray(residue_index, dtype=jnp.int32)
    chain_j = jnp.asarray(chain_index, dtype=jnp.int32)
    seq_j = jnp.asarray(sequence, dtype=jnp.int32)
    pos_j = jnp.asarray(mut_positions, dtype=jnp.int32)
    aas_j = jnp.asarray(mut_aas, dtype=jnp.int32)

    ref_single = np.asarray(single_model(coords_j, mask_j, res_j, chain_j, seq_j))
    ref_double = np.asarray(double_model(coords_j, mask_j, res_j, chain_j, seq_j, pos_j, aas_j))

    # (a) Reversing all_mpnn_hid order
    bad_hid = np.asarray(
      _corrupt_single_unreversed_hidden(single_model, coords_j, mask_j, res_j, chain_j, seq_j),
    )
    diff_hid = float(np.max(np.abs(bad_hid - ref_single)))
    assert diff_hid > 1e-2, f"Expected reversed hidden order to fail parity, got {diff_hid}"

    # (b) Omitting wt_embed - mut_embed subtraction
    bad_seq = np.asarray(
      _corrupt_double_omit_seq_subtraction(
        double_model,
        coords_j,
        mask_j,
        res_j,
        chain_j,
        seq_j,
        pos_j,
        aas_j,
      ),
    )
    diff_seq = float(np.max(np.abs(bad_seq - ref_double)))
    assert diff_seq > 1e-2, f"Expected omitted wt-mut subtraction to fail parity, got {diff_seq}"

    # (c) Zeroing edge features on neighbor-connected double mutations
    bad_edges = np.asarray(
      _corrupt_double_zero_edges(
        double_model,
        coords_j,
        mask_j,
        res_j,
        chain_j,
        seq_j,
        pos_j,
        aas_j,
      ),
    )
    diff_edges = float(np.max(np.abs(bad_edges - ref_double)))
    assert diff_edges > 1e-2, f"Expected zeroed edge features to fail parity, got {diff_edges}"

    # (d) Permuting ddg_out / final MLP output rows
    last_linear = single_model.mlp_layers[-1]
    permuted_last = eqx.tree_at(
      lambda mod: mod.weight,
      last_linear,
      jnp.roll(last_linear.weight, shift=1, axis=0),
    )
    permuted_model = eqx.tree_at(
      lambda mod: mod.mlp_layers[-1],
      single_model,
      permuted_last,
    )
    bad_perm = np.asarray(permuted_model(coords_j, mask_j, res_j, chain_j, seq_j))
    diff_perm = float(np.max(np.abs(bad_perm - ref_single)))
    assert diff_perm > 1e-2, f"Expected permuted output rows to fail parity, got {diff_perm}"


# ---------------------------------------------------------------------------
# 6. Converter validation and checkpoint loader tests
# ---------------------------------------------------------------------------


def test_converter_rejects_unconsumed_and_missing_keys() -> None:
  """Converters reject state dicts with extra unconsumed keys or missing required keys."""
  v2_sd = _build_v2_single_state_dict()
  extra_sd = dict(v2_sd)
  extra_sd["unexpected.layer.weight"] = np.zeros((4, 4), dtype=np.float32)
  with pytest.raises(ValueError, match="unconsumed keys"):
    convert_thermo_mpnn_state_dict(extra_sd)

  missing_sd = dict(v2_sd)
  missing_sd.pop("ddg_out.5.weight")
  with pytest.raises(KeyError, match="ddg_out.5.weight"):
    convert_thermo_mpnn_state_dict(missing_sd)


@pytest.mark.skipif(not _V1_CKPT.is_file(), reason="Upstream ThermoMPNN v1 checkpoint not present")
def test_load_thermo_mpnn_pt_real_checkpoints() -> None:
  """``load_thermo_mpnn_pt`` and ``load_thermo_mpnn_double_pt`` load upstream files."""
  m_v1 = load_thermo_mpnn_pt(_V1_CKPT)
  assert isinstance(m_v1, ThermoMPNN)
  assert m_v1.ddg_out is not None
  assert m_v1.light_attention.kernel_size == 9

  if _V2_SINGLE_CKPT.is_file():
    m_v2 = load_thermo_mpnn_pt(_V2_SINGLE_CKPT)
    assert isinstance(m_v2, ThermoMPNN)
    assert m_v2.ddg_out is None
    assert m_v2.light_attention.kernel_size == 1

  if _V2_DOUBLE_CKPT.is_file():
    m_d = load_thermo_mpnn_double_pt(_V2_DOUBLE_CKPT)
    assert isinstance(m_d, ThermoMPNNDouble)


def test_bucket_padding_invariance_and_eqx_checkpoint_roundtrip(tmp_path: Path) -> None:
  """Verify bucket padding invariance (L=76 vs padded L=128) and .eqx round-trip loading."""
  coords, mask, residue_index, chain_index, sequence = _parse_pdb_backbone(_UBQ_PDB)
  v1_sd = _build_v1_state_dict()
  v2_single_sd = _build_v2_single_state_dict()
  v2_double_sd = _build_v2_double_state_dict()

  with _x64_mode(True):
    v1_model = convert_thermo_mpnn_state_dict(v1_sd)
    v2_model = convert_thermo_mpnn_state_dict(v2_single_sd)
    double_model = convert_thermo_mpnn_double_state_dict(v2_double_sd)
    mut_positions, mut_aas, _, _ = _select_neighbor_and_distant_mutations(
      coords,
      mask,
      residue_index,
      chain_index,
      sequence,
      double_model,
    )

    # 1. Bucket padding invariance: L=76 unpadded vs padded to 128 with mask=0.0
    pad_len = 128 - coords.shape[0]
    coords_pad = np.pad(coords, ((0, pad_len), (0, 0), (0, 0)), constant_values=0.0)
    mask_pad = np.pad(mask, (0, pad_len), constant_values=0.0)
    res_pad = np.pad(residue_index, (0, pad_len), constant_values=999)
    chain_pad = np.pad(chain_index, (0, pad_len), constant_values=0)
    seq_pad = np.pad(sequence, (0, pad_len), constant_values=0)

    out_v2_unpadded = np.asarray(
      v2_model(
        jnp.asarray(coords, jnp.float64),
        jnp.asarray(mask, jnp.float64),
        jnp.asarray(residue_index, jnp.int32),
        jnp.asarray(chain_index, jnp.int32),
        jnp.asarray(sequence, jnp.int32),
      ),
    )
    out_v2_padded = np.asarray(
      v2_model(
        jnp.asarray(coords_pad, jnp.float64),
        jnp.asarray(mask_pad, jnp.float64),
        jnp.asarray(res_pad, jnp.int32),
        jnp.asarray(chain_pad, jnp.int32),
        jnp.asarray(seq_pad, jnp.int32),
      ),
    )[: coords.shape[0]]
    assert np.max(np.abs(out_v2_unpadded - out_v2_padded)) < 1e-6

    out_d_unpadded = np.asarray(
      double_model(
        jnp.asarray(coords, jnp.float64),
        jnp.asarray(mask, jnp.float64),
        jnp.asarray(residue_index, jnp.int32),
        jnp.asarray(chain_index, jnp.int32),
        jnp.asarray(sequence, jnp.int32),
        jnp.asarray(mut_positions, jnp.int32),
        jnp.asarray(mut_aas, jnp.int32),
      ),
    )
    out_d_padded = np.asarray(
      double_model(
        jnp.asarray(coords_pad, jnp.float64),
        jnp.asarray(mask_pad, jnp.float64),
        jnp.asarray(res_pad, jnp.int32),
        jnp.asarray(chain_pad, jnp.int32),
        jnp.asarray(seq_pad, jnp.int32),
        jnp.asarray(mut_positions, jnp.int32),
        jnp.asarray(mut_aas, jnp.int32),
      ),
    )
    assert np.max(np.abs(out_d_unpadded - out_d_padded)) < 1e-6

    # 2. Verify load_thermo_mpnn_checkpoint and load_thermo_mpnn_double_checkpoint preserve static flags
    v1_path = tmp_path / "thermo_mpnn_v1_default.eqx"
    v2_path = tmp_path / "thermo_mpnn_v2_ens1.eqx"
    d_path = tmp_path / "thermo_mpnn_d_ens1.eqx"
    eqx.tree_serialise_leaves(v1_path, v1_model)
    eqx.tree_serialise_leaves(v2_path, v2_model)
    eqx.tree_serialise_leaves(d_path, double_model)

    reloaded_v1 = load_thermo_mpnn_checkpoint("thermo_mpnn_v1_default", local_path=v1_path)
    reloaded_v2 = load_thermo_mpnn_checkpoint("thermo_mpnn_v2_ens1", local_path=v2_path)
    reloaded_d = load_thermo_mpnn_double_checkpoint("thermo_mpnn_d_ens1", local_path=d_path)
    assert reloaded_v1.apply_chain_mask_to_order is False
    assert reloaded_v1.ddg_out is not None
    assert reloaded_v2.apply_chain_mask_to_order is True
    assert reloaded_v2.ddg_out is None

    chain_mask_partial = np.ones_like(mask)
    chain_mask_partial[:10] = 0.0
    pred_v2_orig = np.asarray(
      v2_model(
        jnp.asarray(coords, jnp.float64),
        jnp.asarray(mask, jnp.float64),
        jnp.asarray(residue_index, jnp.int32),
        jnp.asarray(chain_index, jnp.int32),
        jnp.asarray(sequence, jnp.int32),
        chain_mask=jnp.asarray(chain_mask_partial, jnp.float64),
      ),
    )
    pred_v2_reloaded = np.asarray(
      reloaded_v2(
        jnp.asarray(coords, jnp.float64),
        jnp.asarray(mask, jnp.float64),
        jnp.asarray(residue_index, jnp.int32),
        jnp.asarray(chain_index, jnp.int32),
        jnp.asarray(sequence, jnp.int32),
        chain_mask=jnp.asarray(chain_mask_partial, jnp.float64),
      ),
    )
    assert np.max(np.abs(pred_v2_orig - pred_v2_reloaded)) == 0.0
    assert isinstance(reloaded_d, ThermoMPNNDouble)
