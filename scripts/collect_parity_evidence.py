"""Collect path-level parity evidence metrics and point samples for report generation."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

from aminx.io.parsing import parse_input
from aminx.model.ligand_mpnn import PrxteinLigandMPNN
from aminx.model.mpnn import Aminx
from aminx.model.packer import Packer as JAXPacker
from aminx.parity.evidence import (
  amino_acid_distribution,
  EvidenceMetricRecord,
  EvidencePointRecord,
  downsample_pair_points,
  max_abs_error,
  mean_abs_error,
  mean_kl_divergence,
  per_position_entropy,
  root_mean_square_error,
  safe_cosine_similarity,
  safe_pearson,
  safe_spearman,
  sequence_identity,
  token_agreement,
  write_metric_records_csv,
  write_metric_records_json,
  write_point_records_csv,
)
from aminx.parity.matrix import load_parity_matrix
from aminx.host.averaging import get_averaged_encodings
from aminx.run._exports import sample as run_sample
from aminx.run._exports import score as run_score
from aminx.run.specs import SamplingSpecification, ScoringSpecification
from aminx.sampling.conditional_logits import make_conditional_logits_fn
from aminx.utils.aa_convert import protein_sequence_to_string
from aminx.utils.data_structures import Protein


@dataclass(frozen=True, slots=True)
class BackboneCase:
  """One deterministic evidence case."""

  id: str
  kind: str
  atom37_coordinates: np.ndarray
  sequence: np.ndarray
  mask: np.ndarray
  residue_index: np.ndarray
  chain_index: np.ndarray
  seed: int


@dataclass(frozen=True, slots=True)
class CoreModels:
  """Reference and JAX core models."""

  torch: Any
  model_utils: Any
  pt_model: Any
  jax_model: Aminx
  checkpoint_id: str


@dataclass(frozen=True, slots=True)
class LigandModels:
  """Reference and JAX ligand models."""

  torch: Any
  pt_model: Any
  jax_model: PrxteinLigandMPNN
  checkpoint_id: str


@dataclass(frozen=True, slots=True)
class PackerModels:
  """Reference and JAX side-chain packer models."""

  torch: Any
  pt_model: Any
  jax_model: JAXPacker
  checkpoint_id: str


@dataclass(frozen=True, slots=True)
class CoreCaseInputs:
  """Prepared inputs for core model parity paths."""

  atom4_coordinates: np.ndarray
  atom37_coordinates: np.ndarray
  sequence: np.ndarray
  mask: np.ndarray
  chain_mask: np.ndarray
  residue_index: np.ndarray
  chain_index: np.ndarray
  randn: np.ndarray
  ar_mask: np.ndarray
  decoding_order: np.ndarray
  bias: np.ndarray


@dataclass(frozen=True, slots=True)
class EvidenceRuntimeConfig:
  """Runtime configuration for parity evidence collection."""

  atom_context_num: int
  keep_probability: float
  intrinsic_repeats: int
  intrinsic_seed_step: int
  macro_samples_per_case: int
  macro_include_no_ligand: bool
  macro_include_ligand: bool
  macro_include_sidechain_conditioned: bool


@dataclass(frozen=True, slots=True)
class SidechainMacroAcceptance:
  """Acceptance criteria for side-chain-conditioned macro parity."""

  identity_wasserstein_max: float
  entropy_wasserstein_max: float
  composition_js_max: float
  identity_ks_pvalue_min: float


@dataclass(frozen=True, slots=True)
class TiedMultistateComparisonLane:
  """One explicit tied/multistate comparison lane."""

  path_id: str
  input_context: Literal["ligand_context", "side_chain_conditioned"]
  condition: str
  comparison_api: Literal["sampling", "scoring"]
  reference_combiner: Literal["weighted_sum", "arithmetic_mean", "geometric_mean"]
  jax_multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"]
  token_comparison_enabled: bool
  is_primary: bool
  note: str


_TIED_PATH_IDS = {"tied-positions-and-multi-state", "ligand-tied-positions-and-multi-state"}


def _project_root() -> Path:
  return Path(__file__).resolve().parents[1]


def _resolve_reference_root(explicit: Path | None, project_root: Path) -> Path:
  candidates: list[Path] = []
  if explicit is not None:
    candidates.append(explicit)
  if reference_env := Path(os.environ["REFERENCE_PATH"]) if "REFERENCE_PATH" in os.environ else None:
    candidates.append(reference_env)
  candidates.extend(
    [
      project_root / "reference_ligandmpnn_clone",
      project_root.parent / "reference_ligandmpnn_clone",
    ],
  )
  for candidate in candidates:
    if candidate.exists():
      return candidate.resolve()
  searched = ", ".join(str(path) for path in candidates)
  msg = f"Could not find reference LigandMPNN checkout. Searched: {searched}"
  raise FileNotFoundError(msg)


def _prepend_reference_path(reference_root: Path) -> None:
  path_value = str(reference_root)
  if path_value not in sys.path:
    sys.path.insert(0, path_value)


def _stable_seed(identifier: str, offset: int = 0) -> int:
  return sum(ord(ch) for ch in identifier) + offset


def _load_convert_weight_functions() -> tuple[Any, Any]:
  """Load conversion helpers from scripts/convert_weights.py."""
  module_path = _project_root() / "scripts" / "convert_weights.py"
  spec = importlib.util.spec_from_file_location("parity_convert_weights", module_path)
  if spec is None or spec.loader is None:
    msg = f"Unable to load conversion helpers from {module_path}"
    raise RuntimeError(msg)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.convert_full_model, module.convert_packer_model


def _build_atom37_from_atom4(atom4: np.ndarray) -> np.ndarray:
  atom37 = np.zeros((atom4.shape[0], 37, 3), dtype=np.float32)
  atom37[:, 0, :] = atom4[:, 0, :]
  atom37[:, 1, :] = atom4[:, 1, :]
  atom37[:, 2, :] = atom4[:, 2, :]
  atom37[:, 4, :] = atom4[:, 3, :]
  return atom37


def _sanitize_case_arrays(
  atom37: np.ndarray,
  sequence: np.ndarray,
  mask: np.ndarray,
  residue_index: np.ndarray,
  chain_index: np.ndarray,
  *,
  max_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  atom37_clean = np.asarray(atom37, dtype=np.float32)
  sequence_clean = np.asarray(sequence, dtype=np.int64)
  mask_clean = np.asarray(mask, dtype=np.float32)
  residue_index_clean = np.asarray(residue_index, dtype=np.int64)
  chain_index_clean = np.asarray(chain_index, dtype=np.int64)

  atom37_clean = np.squeeze(atom37_clean)
  sequence_clean = np.squeeze(sequence_clean)
  mask_clean = np.squeeze(mask_clean)
  residue_index_clean = np.squeeze(residue_index_clean)
  chain_index_clean = np.squeeze(chain_index_clean)

  seq_len = int(min(
    atom37_clean.shape[0],
    sequence_clean.shape[0],
    mask_clean.shape[0],
    residue_index_clean.shape[0],
    chain_index_clean.shape[0],
  ))
  if max_length is not None:
    seq_len = min(seq_len, max_length)

  atom37_clean = atom37_clean[:seq_len]
  sequence_clean = sequence_clean[:seq_len]
  mask_clean = mask_clean[:seq_len]
  residue_index_clean = residue_index_clean[:seq_len]
  chain_index_clean = chain_index_clean[:seq_len]

  atom37_clean = np.nan_to_num(atom37_clean, nan=0.0, posinf=0.0, neginf=0.0)
  atom4 = atom37_clean[:, [0, 1, 2, 4], :]
  finite_mask = np.isfinite(atom4).all(axis=(1, 2)).astype(np.float32)
  mask_clean = mask_clean * finite_mask
  return atom37_clean, sequence_clean, mask_clean, residue_index_clean, chain_index_clean


def _load_case_corpus(project_root: Path, corpus_path: Path) -> tuple[list[BackboneCase], EvidenceRuntimeConfig]:
  payload = json.loads(corpus_path.read_text(encoding="utf-8"))
  real_backbones = payload.get("real_backbones", [])
  synthetic_backbones = payload.get("synthetic_backbones", [])
  ligand_context = payload.get("ligand_context", {})
  atom_context_num = int(ligand_context.get("atom_context_num", 16))
  keep_probability = float(ligand_context.get("mask_keep_probability", 0.8))
  intrinsic_payload = payload.get("intrinsic_noise", {})
  macro_payload = payload.get("macro_signals", {})
  runtime_config = EvidenceRuntimeConfig(
    atom_context_num=atom_context_num,
    keep_probability=keep_probability,
    intrinsic_repeats=max(1, int(intrinsic_payload.get("repeats", 5))),
    intrinsic_seed_step=max(1, int(intrinsic_payload.get("seed_step", 97))),
    macro_samples_per_case=max(2, int(macro_payload.get("samples_per_case", 32))),
    macro_include_no_ligand=bool(macro_payload.get("include_no_ligand", True)),
    macro_include_ligand=bool(macro_payload.get("include_ligand", True)),
    macro_include_sidechain_conditioned=bool(macro_payload.get("include_sidechain_conditioned", False)),
  )

  cases: list[BackboneCase] = []
  for item in real_backbones:
    if not isinstance(item, dict):
      continue
    rel_path = item.get("path")
    case_id = str(item.get("id"))
    if not isinstance(rel_path, str):
      continue
    target = project_root / rel_path
    if not target.exists():
      continue
    max_length = int(item["max_length"]) if "max_length" in item else None
    try:
      protein = next(parse_input(target))
      atom37, sequence, mask, residue_index, chain_index = _sanitize_case_arrays(
        np.asarray(protein.coordinates),
        np.asarray(protein.aatype),
        np.asarray(protein.mask),
        np.asarray(protein.residue_index),
        np.asarray(protein.chain_index),
        max_length=max_length,
      )
    except (ImportError, ModuleNotFoundError):
      atom37, sequence, mask, residue_index, chain_index = _parse_pdb_backbone_atom37(
        target,
        max_length=max_length,
      )
    if atom37.shape[0] < 8:
      continue
    cases.append(
      BackboneCase(
        id=case_id,
        kind="real_backbone",
        atom37_coordinates=atom37,
        sequence=sequence,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        seed=_stable_seed(case_id, offset=11),
      ),
    )

  for item in synthetic_backbones:
    if not isinstance(item, dict):
      continue
    case_id = str(item.get("id"))
    length = int(item.get("length", 64))
    seed = int(item.get("seed", _stable_seed(case_id)))
    rng = np.random.default_rng(seed)
    atom4 = rng.normal(size=(length, 4, 3)).astype(np.float32)
    atom37 = _build_atom37_from_atom4(atom4)
    sequence = rng.integers(0, 20, size=(length,), dtype=np.int64)
    mask = np.ones((length,), dtype=np.float32)
    residue_index = np.arange(length, dtype=np.int64)
    chain_index = np.zeros((length,), dtype=np.int64)
    cases.append(
      BackboneCase(
        id=case_id,
        kind="synthetic",
        atom37_coordinates=atom37,
        sequence=sequence,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        seed=seed,
      ),
    )

  return cases, runtime_config


def _parse_pdb_backbone_atom37(
  pdb_path: Path,
  *,
  max_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Parse a PDB file into minimal Atom37-compatible arrays without external parser backends."""
  residue_atoms: dict[tuple[str, str, str], dict[str, object]] = {}
  residue_order: list[tuple[str, str, str]] = []
  chain_to_index: dict[str, int] = {}
  residue_to_aa_index = {
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
  atom_order = ("N", "CA", "C", "O")

  for line in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines():
    if not line.startswith("ATOM"):
      continue
    atom_name = line[12:16].strip()
    if atom_name not in atom_order:
      continue
    altloc = line[16].strip()
    if altloc not in ("", "A"):
      continue
    residue_name = line[17:20].strip().upper()
    chain_id = line[21].strip() or "_"
    residue_seq = line[22:26].strip()
    insertion_code = line[26].strip()
    residue_key = (chain_id, residue_seq, insertion_code)
    if residue_key not in residue_atoms:
      residue_atoms[residue_key] = {"resname": residue_name}
      residue_order.append(residue_key)
      if chain_id not in chain_to_index:
        chain_to_index[chain_id] = len(chain_to_index)
    try:
      x = float(line[30:38])
      y = float(line[38:46])
      z = float(line[46:54])
    except ValueError:
      continue
    residue_atoms[residue_key][atom_name] = np.asarray([x, y, z], dtype=np.float32)

  if max_length is not None:
    residue_order = residue_order[:max_length]
  seq_len = len(residue_order)
  atom37 = np.zeros((seq_len, 37, 3), dtype=np.float32)
  sequence = np.full((seq_len,), 20, dtype=np.int64)
  mask = np.zeros((seq_len,), dtype=np.float32)
  residue_index = np.zeros((seq_len,), dtype=np.int64)
  chain_index = np.zeros((seq_len,), dtype=np.int64)
  atom37_indices = {"N": 0, "CA": 1, "C": 2, "O": 4}
  for idx, residue_key in enumerate(residue_order):
    chain_id, _, _ = residue_key
    atom_map = residue_atoms[residue_key]
    residue_name = str(atom_map.get("resname", "UNK"))
    sequence[idx] = residue_to_aa_index.get(residue_name, 20)
    residue_index[idx] = idx
    chain_index[idx] = chain_to_index.get(chain_id, 0)
    if all(name in atom_map for name in atom_order):
      mask[idx] = 1.0
    for atom_name, atom_position in atom_map.items():
      if atom_name in atom37_indices and isinstance(atom_position, np.ndarray):
        atom37[idx, atom37_indices[atom_name], :] = atom_position
  return atom37, sequence, mask, residue_index, chain_index


def _prepare_core_case_inputs(case: BackboneCase) -> CoreCaseInputs:
  seq_len = case.sequence.shape[0]
  atom4 = case.atom37_coordinates[:, [0, 1, 2, 4], :].astype(np.float32)
  rng = np.random.default_rng(case.seed)
  randn = rng.normal(size=(seq_len,)).astype(np.float32)
  chain_mask = case.mask.astype(np.float32)
  decoding_order = np.argsort((chain_mask + 1e-4) * np.abs(randn))
  ar_mask = np.zeros((seq_len, seq_len), dtype=np.int32)
  order_position = {int(token): idx for idx, token in enumerate(decoding_order)}
  for i in range(seq_len):
    for j in range(seq_len):
      if order_position[j] < order_position[i]:
        ar_mask[i, j] = 1
  bias = np.zeros((seq_len, 21), dtype=np.float32)
  bias[np.arange(seq_len), case.sequence] = 20.0
  return CoreCaseInputs(
    atom4_coordinates=atom4,
    atom37_coordinates=case.atom37_coordinates.astype(np.float32),
    sequence=case.sequence.astype(np.int64),
    mask=case.mask.astype(np.float32),
    chain_mask=chain_mask,
    residue_index=case.residue_index.astype(np.int64),
    chain_index=case.chain_index.astype(np.int64),
    randn=randn,
    ar_mask=ar_mask,
    decoding_order=decoding_order.astype(np.int64),
    bias=bias,
  )


def _load_core_models(reference_root: Path, project_root: Path) -> CoreModels:
  _prepend_reference_path(reference_root)
  model_utils = importlib.import_module("model_utils")
  import torch

  ref_checkpoint_path = reference_root / "model_params/proteinmpnn_v_48_020.pt"
  converted_checkpoint_path = project_root / "src/aminx/model_params/proteinmpnn_v_48_020.eqx.zst"
  checkpoint = torch.load(ref_checkpoint_path, map_location="cpu")
  pos_weight = checkpoint["model_state_dict"].get("features.embeddings.linear.weight")
  if pos_weight is None:
    pos_weight = checkpoint["model_state_dict"].get("features.positional_embeddings.linear.weight")
  num_positional_embeddings = int((pos_weight.shape[1] - 2) // 2) if pos_weight is not None else 16

  pt_model = model_utils.ProteinMPNN(
    num_letters=21,
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
  )
  pt_model.load_state_dict(checkpoint["model_state_dict"])
  pt_model.eval()

  jax_model = Aminx(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
    num_positional_embeddings=num_positional_embeddings,
    dropout_rate=0.0,
    key=jax.random.PRNGKey(0),
  )
  jax_model = eqx.tree_deserialise_leaves(converted_checkpoint_path, jax_model)
  return CoreModels(
    torch=torch,
    model_utils=model_utils,
    pt_model=pt_model,
    jax_model=jax_model,
    checkpoint_id="proteinmpnn_v_48_020",
  )


def _load_ligand_models(
  reference_root: Path,
  *,
  use_side_chain_context: bool = False,
) -> LigandModels:
  _prepend_reference_path(reference_root)
  model_utils = importlib.import_module("model_utils")
  import torch
  convert_full_model, _ = _load_convert_weight_functions()

  ref_checkpoint_path = reference_root / "model_params/ligandmpnn_v_32_020_25.pt"
  checkpoint = torch.load(ref_checkpoint_path, map_location="cpu")
  state_dict = checkpoint["model_state_dict"]
  state_dict_np = {name: value.detach().cpu().numpy() for name, value in state_dict.items()}

  pos_weight = state_dict.get("features.embeddings.linear.weight")
  num_positional_embeddings = int((pos_weight.shape[1] - 2) // 2) if pos_weight is not None else 16
  pt_model = model_utils.ProteinMPNN(
    num_letters=21,
    node_features=128,
    edge_features=128,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=32,
    atom_context_num=16,
    model_type="ligand_mpnn",
    ligand_mpnn_use_side_chain_context=use_side_chain_context,
    dropout=0.0,
  )
  pt_model.load_state_dict(state_dict)
  pt_model.eval()

  jax_model = PrxteinLigandMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=32,
    num_context_layers=2,
    num_positional_embeddings=num_positional_embeddings,
    dropout_rate=0.0,
    ligand_mpnn_use_side_chain_context=use_side_chain_context,
    key=jax.random.PRNGKey(0),
  )
  jax_model = convert_full_model(state_dict_np, jax_model)
  return LigandModels(
    torch=torch,
    pt_model=pt_model,
    jax_model=jax_model,
    checkpoint_id="ligandmpnn_v_32_020_25",
  )


def _load_packer_models(reference_root: Path) -> PackerModels:
  _prepend_reference_path(reference_root)
  sc_utils = importlib.import_module("sc_utils")
  import torch
  _, convert_packer_model = _load_convert_weight_functions()

  ref_checkpoint_path = reference_root / "model_params/ligandmpnn_sc_v_32_002_16.pt"
  checkpoint = torch.load(ref_checkpoint_path, map_location="cpu")
  pt_model = sc_utils.Packer(
    edge_features=128,
    node_features=128,
    num_positional_embeddings=16,
    num_chain_embeddings=16,
    num_rbf=16,
    top_k=30,
    augment_eps=0.0,
    atom37_order=False,
    device="cpu",
    atom_context_num=16,
    lower_bound=0.0,
    upper_bound=20.0,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dropout=0.0,
    num_mix=3,
  )
  pt_model.load_state_dict(checkpoint["model_state_dict"])
  pt_model.eval()
  state_dict_np = {name: value.detach().cpu().numpy() for name, value in pt_model.state_dict().items()}

  jax_model = JAXPacker(
    edge_features=128,
    node_features=128,
    num_positional_embeddings=16,
    num_rbf=16,
    top_k=30,
    atom37_order=False,
    atom_context_num=16,
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dropout=0.0,
    num_mix=3,
    key=jax.random.PRNGKey(0),
  )
  jax_model = convert_packer_model(state_dict_np, jax_model)
  return PackerModels(
    torch=torch,
    pt_model=pt_model,
    jax_model=jax_model,
    checkpoint_id="ligandmpnn_sc_v_32_002_16",
  )


def _core_feature_outputs(models: CoreModels, inputs: CoreCaseInputs) -> tuple[np.ndarray, np.ndarray]:
  with models.torch.no_grad():
    feature_dict = {
      "X": models.torch.from_numpy(inputs.atom4_coordinates[None]),
      "S": models.torch.from_numpy(inputs.sequence[None]),
      "mask": models.torch.from_numpy(inputs.mask[None]),
      "chain_mask": models.torch.from_numpy(inputs.chain_mask[None]),
      "R_idx": models.torch.from_numpy(inputs.residue_index[None]),
      "chain_labels": models.torch.from_numpy(inputs.chain_index[None]),
      "randn": models.torch.from_numpy(inputs.randn[None]),
      "batch_size": 1,
      "symmetry_residues": [[]],
      "symmetry_weights": [[]],
    }
    pt_edges, _ = models.pt_model.features(feature_dict)
    pt_projected = models.pt_model.W_e(pt_edges).numpy()[0]

  jax_edges, _, _, _ = models.jax_model.features(
    jax.random.PRNGKey(0),
    jnp.asarray(inputs.atom37_coordinates),
    jnp.asarray(inputs.mask),
    jnp.asarray(inputs.residue_index),
    jnp.asarray(inputs.chain_index),
    jnp.asarray(0.0, dtype=jnp.float32),
  )
  return pt_projected, np.asarray(jax_edges)


def _core_encoder_outputs(models: CoreModels, inputs: CoreCaseInputs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  with models.torch.no_grad():
    feature_dict = {
      "X": models.torch.from_numpy(inputs.atom4_coordinates[None]),
      "S": models.torch.from_numpy(inputs.sequence[None]),
      "mask": models.torch.from_numpy(inputs.mask[None]),
      "chain_mask": models.torch.from_numpy(inputs.chain_mask[None]),
      "R_idx": models.torch.from_numpy(inputs.residue_index[None]),
      "chain_labels": models.torch.from_numpy(inputs.chain_index[None]),
      "randn": models.torch.from_numpy(inputs.randn[None]),
      "batch_size": 1,
      "symmetry_residues": [[]],
      "symmetry_weights": [[]],
    }
    pt_nodes, pt_edges, _ = models.pt_model.encode(feature_dict)
  jax_edges, jax_idx, init_nodes, _ = models.jax_model.features(
    jax.random.PRNGKey(1),
    jnp.asarray(inputs.atom37_coordinates),
    jnp.asarray(inputs.mask),
    jnp.asarray(inputs.residue_index),
    jnp.asarray(inputs.chain_index),
    jnp.asarray(0.0, dtype=jnp.float32),
  )
  initial_node_features = None if init_nodes is None else jnp.asarray(init_nodes)
  jax_nodes, jax_edges_encoded = models.jax_model.encoder(
    jax_edges,
    jax_idx,
    jnp.asarray(inputs.mask),
    initial_node_features=initial_node_features,
    key=jax.random.PRNGKey(2),
  )
  return (
    pt_nodes.numpy()[0],
    np.asarray(jax_nodes),
    pt_edges.numpy()[0],
    np.asarray(jax_edges_encoded),
  )


def _core_unconditional_log_probs(models: CoreModels, inputs: CoreCaseInputs) -> tuple[np.ndarray, np.ndarray]:
  with models.torch.no_grad():
    feature_dict = {
      "X": models.torch.from_numpy(inputs.atom4_coordinates[None]),
      "S": models.torch.from_numpy(inputs.sequence[None]),
      "mask": models.torch.from_numpy(inputs.mask[None]),
      "chain_mask": models.torch.from_numpy(inputs.chain_mask[None]),
      "R_idx": models.torch.from_numpy(inputs.residue_index[None]),
      "chain_labels": models.torch.from_numpy(inputs.chain_index[None]),
      "randn": models.torch.from_numpy(inputs.randn[None]),
      "batch_size": 1,
      "symmetry_residues": [[]],
      "symmetry_weights": [[]],
    }
    pt_nodes, pt_edges, pt_idx = models.pt_model.encode(feature_dict)
    seq_embedding = models.torch.zeros_like(pt_nodes)
    enc_context = models.model_utils.cat_neighbors_nodes(seq_embedding, pt_edges, pt_idx)
    dec_context = models.model_utils.cat_neighbors_nodes(pt_nodes, enc_context, pt_idx)
    decoded_nodes = pt_nodes
    for layer in models.pt_model.decoder_layers:
      decoded_nodes = layer(decoded_nodes, dec_context, feature_dict["mask"])
    pt_log_probs = models.torch.log_softmax(models.pt_model.W_out(decoded_nodes), dim=-1).numpy()[0]

  _, logits_jax = models.jax_model(
    jnp.asarray(inputs.atom37_coordinates),
    jnp.asarray(inputs.mask),
    jnp.asarray(inputs.residue_index),
    jnp.asarray(inputs.chain_index),
    "unconditional",
  )
  jax_log_probs = np.asarray(jax.nn.log_softmax(logits_jax, axis=-1))
  return pt_log_probs, jax_log_probs


def _core_conditional_log_probs(models: CoreModels, inputs: CoreCaseInputs) -> tuple[np.ndarray, np.ndarray]:
  with models.torch.no_grad():
    feature_dict = {
      "X": models.torch.from_numpy(inputs.atom4_coordinates[None]),
      "S": models.torch.from_numpy(inputs.sequence[None]),
      "mask": models.torch.from_numpy(inputs.mask[None]),
      "chain_mask": models.torch.from_numpy(inputs.chain_mask[None]),
      "R_idx": models.torch.from_numpy(inputs.residue_index[None]),
      "chain_labels": models.torch.from_numpy(inputs.chain_index[None]),
      "randn": models.torch.from_numpy(inputs.randn[None]),
      "batch_size": 1,
      "symmetry_residues": [[]],
      "symmetry_weights": [[]],
    }
    pt_log_probs = models.pt_model.score(feature_dict, use_sequence=True)["log_probs"].numpy()[0]

  _, logits_jax = models.jax_model(
    jnp.asarray(inputs.atom37_coordinates),
    jnp.asarray(inputs.mask),
    jnp.asarray(inputs.residue_index),
    jnp.asarray(inputs.chain_index),
    "conditional",
    one_hot_sequence=jax.nn.one_hot(jnp.asarray(inputs.sequence), 21),
    ar_mask=jnp.asarray(inputs.ar_mask),
  )
  jax_log_probs = np.asarray(jax.nn.log_softmax(logits_jax, axis=-1))
  return pt_log_probs, jax_log_probs


def _core_autoregressive_outputs(
  models: CoreModels,
  inputs: CoreCaseInputs,
  *,
  sample_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  feature_dict = {
    "X": models.torch.from_numpy(inputs.atom4_coordinates[None]),
    "S": models.torch.from_numpy(inputs.sequence[None]),
    "mask": models.torch.from_numpy(inputs.mask[None]),
    "chain_mask": models.torch.from_numpy(inputs.chain_mask[None]),
    "R_idx": models.torch.from_numpy(inputs.residue_index[None]),
    "chain_labels": models.torch.from_numpy(inputs.chain_index[None]),
    "randn": models.torch.from_numpy(inputs.randn[None]),
    "batch_size": 1,
    "symmetry_residues": [[]],
    "symmetry_weights": [[]],
    "bias": models.torch.from_numpy(inputs.bias[None]),
    "temperature": 1.0,
  }
  with models.torch.no_grad():
    models.torch.manual_seed(sample_seed)
    pt_sample = models.pt_model.sample(feature_dict)
  jax_sequence, jax_logits = models.jax_model(
    jnp.asarray(inputs.atom37_coordinates),
    jnp.asarray(inputs.mask),
    jnp.asarray(inputs.residue_index),
    jnp.asarray(inputs.chain_index),
    "autoregressive",
    prng_key=jax.random.PRNGKey(sample_seed + 7),
    ar_mask=jnp.asarray(inputs.ar_mask),
    temperature=jnp.asarray(1.0, dtype=jnp.float32),
    bias=jnp.asarray(inputs.bias),
  )
  pt_tokens = pt_sample["S"].numpy()[0]
  jax_tokens = np.asarray(jnp.argmax(jax_sequence, axis=-1))
  pt_log_probs = pt_sample["log_probs"].numpy()[0]
  jax_log_probs = np.asarray(jax.nn.log_softmax(jax_logits, axis=-1))
  return pt_tokens, jax_tokens, pt_log_probs, jax_log_probs


def _build_tie_groups(seq_len: int) -> tuple[list[list[int]], list[list[float]], np.ndarray]:
  tie_groups = [[0, 1, 2], [max(3, seq_len // 2), min(max(3, seq_len // 2) + 1, seq_len - 1)]]
  tie_groups = [group for group in tie_groups if len(set(group)) > 1]
  tie_weights = [[1.0 for _ in group] for group in tie_groups]
  tie_group_map = np.arange(seq_len, dtype=np.int32)
  for group in tie_groups:
    anchor = group[0]
    for residue_idx in group[1:]:
      tie_group_map[residue_idx] = anchor
  _, compact_tie_group_map = np.unique(tie_group_map, return_inverse=True)
  return tie_groups, tie_weights, compact_tie_group_map.astype(np.int32, copy=False)


def _row_log_softmax(logits: np.ndarray) -> np.ndarray:
  shifted = logits - np.max(logits, axis=-1, keepdims=True)
  return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _combine_reference_tied_log_probs(
  reference_log_probs: np.ndarray,
  *,
  tie_groups: list[list[int]],
  tie_weights: list[list[float]],
  combiner: Literal["weighted_sum", "arithmetic_mean", "geometric_mean"],
  temperature: float = 1.0,
) -> np.ndarray:
  """Align reference tied-position log-probabilities to a selected combiner."""
  combined = np.asarray(reference_log_probs, dtype=np.float64).copy()
  for group, weights in zip(tie_groups, tie_weights, strict=True):
    group_indices = np.asarray(group, dtype=np.int32)
    if group_indices.size <= 1:
      continue
    group_weights = np.asarray(weights, dtype=np.float64)
    if group_weights.shape[0] != group_indices.shape[0]:
      msg = "Tie-group weights must align with tied residue indices."
      raise ValueError(msg)
    group_log_probs = np.asarray(reference_log_probs[group_indices], dtype=np.float64)

    if combiner == "weighted_sum":
      combined_logits = np.sum(group_log_probs * group_weights[:, None], axis=0, keepdims=True)
      combined_log_probs = _row_log_softmax(combined_logits)
    elif combiner == "arithmetic_mean":
      if np.any(group_weights <= 0.0):
        msg = "Arithmetic-mean combiner requires strictly positive symmetry weights."
        raise ValueError(msg)
      group_probs = np.exp(group_log_probs)
      group_probs = group_probs / np.sum(group_probs, axis=-1, keepdims=True)
      normalized_weights = group_weights / np.sum(group_weights)
      mean_probs = np.sum(group_probs * normalized_weights[:, None], axis=0, keepdims=True)
      combined_log_probs = np.log(np.clip(mean_probs, 1e-12, 1.0))
    elif combiner == "geometric_mean":
      if np.any(group_weights <= 0.0):
        msg = "Geometric-mean combiner requires strictly positive symmetry weights."
        raise ValueError(msg)
      normalized_weights = group_weights / np.sum(group_weights)
      weighted_mean = np.sum(group_log_probs * normalized_weights[:, None], axis=0, keepdims=True)
      scaled_logits = weighted_mean / max(float(temperature), 1e-6)
      combined_log_probs = _row_log_softmax(scaled_logits)
    else:
      msg = f"Unsupported tied-reference combiner: {combiner!r}"
      raise ValueError(msg)
    combined[group_indices] = combined_log_probs
  return combined.astype(np.float32)


def _combine_reference_tied_logits(
  reference_logits: np.ndarray,
  *,
  tie_groups: list[list[int]],
  tie_weights: list[list[float]],
  combiner: Literal["weighted_sum", "arithmetic_mean", "geometric_mean"],
  temperature: float = 1.0,
) -> np.ndarray:
  """Align reference tied-position logits to the JAX multi-state strategy semantics."""
  combined = np.asarray(reference_logits, dtype=np.float64).copy()
  for group, weights in zip(tie_groups, tie_weights, strict=True):
    group_indices = np.asarray(group, dtype=np.int32)
    if group_indices.size <= 1:
      continue
    group_weights = np.asarray(weights, dtype=np.float64)
    if group_weights.shape[0] != group_indices.shape[0]:
      msg = "Tie-group weights must align with tied residue indices."
      raise ValueError(msg)
    group_logits = np.asarray(reference_logits[group_indices], dtype=np.float64)

    if combiner == "weighted_sum":
      combined_logits = np.sum(group_logits * group_weights[:, None], axis=0, keepdims=True)
    elif combiner == "arithmetic_mean":
      if np.any(group_weights <= 0.0):
        msg = "Arithmetic-mean combiner requires strictly positive symmetry weights."
        raise ValueError(msg)
      normalized_weights = group_weights / np.sum(group_weights)
      max_logits = np.max(group_logits, axis=0, keepdims=True)
      mean_shifted = np.sum(
        np.exp(group_logits - max_logits) * normalized_weights[:, None],
        axis=0,
        keepdims=True,
      )
      combined_logits = np.log(np.clip(mean_shifted, 1e-12, None)) + max_logits
    elif combiner == "geometric_mean":
      if np.any(group_weights <= 0.0):
        msg = "Geometric-mean combiner requires strictly positive symmetry weights."
        raise ValueError(msg)
      normalized_weights = group_weights / np.sum(group_weights)
      combined_logits = (
        np.sum(group_logits * normalized_weights[:, None], axis=0, keepdims=True)
        / max(float(temperature), 1e-6)
      )
    else:
      msg = f"Unsupported tied-reference combiner: {combiner!r}"
      raise ValueError(msg)

    combined[group_indices] = combined_logits
  return combined.astype(np.float32)


def _core_tied_sampling_outputs(
  models: CoreModels,
  inputs: CoreCaseInputs,
  *,
  sample_seed: int,
  lane: TiedMultistateComparisonLane,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
  if lane.comparison_api != "sampling":
    msg = f"{lane.condition}: sampling tied-output helper requires comparison_api='sampling'."
    raise ValueError(msg)

  seq_len = int(inputs.sequence.shape[0])
  tie_groups, tie_weights, tie_group_map = _build_tie_groups(seq_len)

  feature_dict = {
    "X": models.torch.from_numpy(inputs.atom4_coordinates[None]),
    "S": models.torch.from_numpy(inputs.sequence[None]),
    "mask": models.torch.from_numpy(inputs.mask[None]),
    "chain_mask": models.torch.from_numpy(inputs.chain_mask[None]),
    "R_idx": models.torch.from_numpy(inputs.residue_index[None]),
    "chain_labels": models.torch.from_numpy(inputs.chain_index[None]),
    "randn": models.torch.from_numpy(inputs.randn[None]),
    "batch_size": 1,
    "symmetry_residues": tie_groups,
    "symmetry_weights": tie_weights,
    "bias": models.torch.from_numpy(inputs.bias[None]),
    "temperature": 1.0,
  }
  with models.torch.no_grad():
    models.torch.manual_seed(sample_seed)
    pt_sample = models.pt_model.sample(feature_dict)
  jax_sequence, jax_logits = models.jax_model(
    jnp.asarray(inputs.atom37_coordinates),
    jnp.asarray(inputs.mask),
    jnp.asarray(inputs.residue_index),
    jnp.asarray(inputs.chain_index),
    "autoregressive",
    prng_key=jax.random.PRNGKey(sample_seed + 9),
    ar_mask=jnp.asarray(inputs.ar_mask),
    temperature=jnp.asarray(1.0, dtype=jnp.float32),
    bias=jnp.asarray(inputs.bias),
    tie_group_map=jnp.asarray(tie_group_map),
    multi_state_strategy=lane.jax_multi_state_strategy,
  )
  pt_tokens = pt_sample["S"].numpy()[0]
  jax_tokens = np.asarray(jnp.argmax(jax_sequence, axis=-1))
  pt_log_probs = _combine_reference_tied_log_probs(
    pt_sample["log_probs"].numpy()[0],
    tie_groups=tie_groups,
    tie_weights=tie_weights,
    combiner=lane.reference_combiner,
    temperature=1.0,
  )
  jax_log_probs = np.asarray(jax.nn.log_softmax(jax_logits, axis=-1))
  return pt_tokens, jax_tokens, pt_log_probs, jax_log_probs, tie_groups


def _core_tied_scoring_outputs(
  models: CoreModels,
  inputs: CoreCaseInputs,
  *,
  lane: TiedMultistateComparisonLane,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
  if lane.comparison_api != "scoring":
    msg = f"{lane.condition}: scoring tied-output helper requires comparison_api='scoring'."
    raise ValueError(msg)

  seq_len = int(inputs.sequence.shape[0])
  tie_groups, tie_weights, tie_group_map = _build_tie_groups(seq_len)
  feature_dict = {
    "X": models.torch.from_numpy(inputs.atom4_coordinates[None]),
    "S": models.torch.from_numpy(inputs.sequence[None]),
    "mask": models.torch.from_numpy(inputs.mask[None]),
    "chain_mask": models.torch.from_numpy(inputs.chain_mask[None]),
    "R_idx": models.torch.from_numpy(inputs.residue_index[None]),
    "chain_labels": models.torch.from_numpy(inputs.chain_index[None]),
    "randn": models.torch.from_numpy(inputs.randn[None]),
    "batch_size": 1,
    "symmetry_residues": tie_groups,
    "symmetry_weights": tie_weights,
  }
  with models.torch.no_grad():
    pt_score = models.pt_model.score(feature_dict, use_sequence=True)

  pt_combined_logits = _combine_reference_tied_logits(
    pt_score["logits"].numpy()[0],
    tie_groups=tie_groups,
    tie_weights=tie_weights,
    combiner=lane.reference_combiner,
    temperature=1.0,
  )
  pt_log_probs = _row_log_softmax(pt_combined_logits)
  pt_tokens = np.asarray(np.argmax(pt_log_probs, axis=-1), dtype=np.int32)

  _, jax_logits = models.jax_model(
    jnp.asarray(inputs.atom37_coordinates),
    jnp.asarray(inputs.mask),
    jnp.asarray(inputs.residue_index),
    jnp.asarray(inputs.chain_index),
    "conditional",
    one_hot_sequence=jax.nn.one_hot(jnp.asarray(inputs.sequence), 21),
    ar_mask=jnp.asarray(inputs.ar_mask),
    tie_group_map=jnp.asarray(tie_group_map),
    multi_state_strategy=lane.jax_multi_state_strategy,
  )
  jax_log_probs = np.asarray(jax.nn.log_softmax(jax_logits, axis=-1))
  jax_tokens = np.asarray(jnp.argmax(jax_log_probs, axis=-1), dtype=np.int32)
  return pt_tokens, jax_tokens, pt_log_probs, jax_log_probs, tie_groups


def _build_ligand_inputs(
  case: BackboneCase,
  *,
  atom_context_num: int,
  keep_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  rng = np.random.default_rng(case.seed + 1000)
  seq_len = int(case.sequence.shape[0])
  y = rng.normal(size=(seq_len, atom_context_num, 3)).astype(np.float32)
  y_t = rng.integers(1, 30, size=(seq_len, atom_context_num), dtype=np.int64)
  y_m = (rng.random(size=(seq_len, atom_context_num)) < keep_probability).astype(np.float32)
  return y, y_t, y_m


def _ligand_feature_outputs(
  models: LigandModels,
  inputs: CoreCaseInputs,
  y: np.ndarray,
  y_t: np.ndarray,
  y_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  feature_dict = {
    "X": models.torch.from_numpy(inputs.atom4_coordinates[None]),
    "S": models.torch.from_numpy(inputs.sequence[None]),
    "mask": models.torch.from_numpy(inputs.mask[None]),
    "chain_mask": models.torch.from_numpy(inputs.chain_mask[None]),
    "R_idx": models.torch.from_numpy(inputs.residue_index[None]),
    "chain_labels": models.torch.from_numpy(inputs.chain_index[None]),
    "randn": models.torch.from_numpy(inputs.randn[None]),
    "batch_size": 1,
    "symmetry_residues": [[]],
    "symmetry_weights": [[]],
    "Y": models.torch.from_numpy(y[None]),
    "Y_t": models.torch.from_numpy(y_t[None]),
    "Y_m": models.torch.from_numpy(y_m[None]),
  }
  with models.torch.no_grad():
    _, edge_pt, _, y_nodes_pt, y_edges_pt, y_m_pt = models.pt_model.features(feature_dict)
    projected_edge_pt = models.pt_model.W_e(edge_pt).numpy()[0]
  _, edge_jax, _, y_nodes_jax, y_edges_jax, y_m_jax = models.jax_model.features(
    jax.random.PRNGKey(12),
    jnp.asarray(inputs.atom4_coordinates),
    jnp.asarray(inputs.mask),
    jnp.asarray(inputs.residue_index),
    jnp.asarray(inputs.chain_index),
    jnp.asarray(y),
    jnp.asarray(y_t),
    jnp.asarray(y_m),
  )
  reference = np.concatenate(
    [
      projected_edge_pt.ravel(),
      y_nodes_pt.numpy()[0].ravel(),
      y_edges_pt.numpy()[0].ravel(),
      y_m_pt.numpy()[0].ravel(),
    ],
  )
  observed = np.concatenate(
    [
      np.asarray(edge_jax).ravel(),
      np.asarray(y_nodes_jax).ravel(),
      np.asarray(y_edges_jax).ravel(),
      np.asarray(y_m_jax).ravel(),
    ],
  )
  return reference, observed


def _ligand_context_outputs(
  models: LigandModels,
  inputs: CoreCaseInputs,
  y: np.ndarray,
  y_t: np.ndarray,
  y_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  feature_dict = {
    "X": models.torch.from_numpy(inputs.atom4_coordinates[None]),
    "S": models.torch.from_numpy(inputs.sequence[None]),
    "mask": models.torch.from_numpy(inputs.mask[None]),
    "chain_mask": models.torch.from_numpy(inputs.chain_mask[None]),
    "R_idx": models.torch.from_numpy(inputs.residue_index[None]),
    "chain_labels": models.torch.from_numpy(inputs.chain_index[None]),
    "randn": models.torch.from_numpy(inputs.randn[None]),
    "batch_size": 1,
    "symmetry_residues": [[]],
    "symmetry_weights": [[]],
    "Y": models.torch.from_numpy(y[None]),
    "Y_t": models.torch.from_numpy(y_t[None]),
    "Y_m": models.torch.from_numpy(y_m[None]),
  }
  with models.torch.no_grad():
    h_v_pt, h_e_pt, _ = models.pt_model.encode(feature_dict)
    log_probs_pt = models.pt_model.score(feature_dict, use_sequence=True)["log_probs"].numpy()[0]

  h_v_jax, h_e_jax = _jax_ligand_context_state(
    models.jax_model,
    inputs,
    y,
    y_t,
    y_m,
    prng_key=jax.random.PRNGKey(14),
  )
  _, logits_jax = models.jax_model(
    jnp.asarray(inputs.atom4_coordinates),
    jnp.asarray(inputs.mask),
    jnp.asarray(inputs.residue_index),
    jnp.asarray(inputs.chain_index),
    jnp.asarray(y),
    jnp.asarray(y_t),
    jnp.asarray(y_m),
    "conditional",
    prng_key=jax.random.PRNGKey(13),
    ar_mask=jnp.asarray(inputs.ar_mask),
    one_hot_sequence=jax.nn.one_hot(jnp.asarray(inputs.sequence), 21),
    inference=True,
  )
  log_probs_jax = np.asarray(jax.nn.log_softmax(logits_jax, axis=-1))
  return (
    h_v_pt.numpy()[0],
    np.asarray(h_v_jax),
    h_e_pt.numpy()[0],
    np.asarray(h_e_jax),
    log_probs_pt,
    log_probs_jax,
  )


def _jax_ligand_context_state(
  model: PrxteinLigandMPNN,
  inputs: CoreCaseInputs,
  y: np.ndarray,
  y_t: np.ndarray,
  y_m: np.ndarray,
  *,
  prng_key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Run ligand context path in JAX and expose hidden-state tensors."""
  keys = jax.random.split(prng_key, 2)
  v, e, e_idx, y_nodes, y_edges, y_mask = model.features(
    keys[0],
    jnp.asarray(inputs.atom4_coordinates),
    jnp.asarray(inputs.mask),
    jnp.asarray(inputs.residue_index),
    jnp.asarray(inputs.chain_index),
    jnp.asarray(y),
    jnp.asarray(y_t),
    jnp.asarray(y_m),
  )

  h_v = jnp.zeros((e.shape[0], model.node_features_dim))
  h_e = e
  mask_2d = jnp.asarray(inputs.mask)[:, None] * jnp.asarray(inputs.mask)[None, :]
  mask_attend = jnp.take_along_axis(mask_2d, e_idx.astype(jnp.int32), axis=1)
  for layer in model.encoder.layers:
    h_v, h_e = layer(h_v, h_e, e_idx, jnp.asarray(inputs.mask), mask_attend, inference=True)

  h_v_context = jax.vmap(model.w_c)(h_v)
  h_e_context = jax.vmap(jax.vmap(model.w_v))(v)
  y_nodes = jax.vmap(jax.vmap(model.w_nodes_y))(y_nodes)
  y_edges = jax.vmap(jax.vmap(jax.vmap(model.w_edges_y)))(y_edges)
  y_mask_edges = y_mask[..., None] * y_mask[..., None, :]

  for layer_index in range(len(model.context_encoder)):
    y_nodes = jax.vmap(
      lambda node, edge, ligand_mask, edge_mask: model.y_context_encoder[layer_index](
        node,
        edge,
        ligand_mask,
        attention_mask=edge_mask,
        inference=True,
      ),
    )(y_nodes, y_edges, y_mask, y_mask_edges)
    h_e_context_cat = jnp.concatenate([h_e_context, y_nodes], axis=-1)
    h_v_context = model.context_encoder[layer_index](
      h_v_context,
      h_e_context_cat,
      jnp.asarray(inputs.mask),
      attention_mask=y_mask,
      inference=True,
    )

  h_v_context = jax.vmap(model.v_c)(h_v_context)
  h_v = h_v + jax.vmap(model.v_c_norm)(model.dropout(h_v_context, key=keys[1], inference=True))
  return jnp.asarray(h_v), jnp.asarray(h_e)


def _ligand_autoregressive_outputs(
  models: LigandModels,
  inputs: CoreCaseInputs,
  y: np.ndarray,
  y_t: np.ndarray,
  y_m: np.ndarray,
  *,
  sample_seed: int,
  xyz_37: np.ndarray | None = None,
  xyz_37_m: np.ndarray | None = None,
  chain_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  feature_dict = {
    "X": models.torch.from_numpy(inputs.atom4_coordinates[None]),
    "S": models.torch.from_numpy(inputs.sequence[None]),
    "mask": models.torch.from_numpy(inputs.mask[None]),
    "chain_mask": models.torch.from_numpy(inputs.chain_mask[None]),
    "R_idx": models.torch.from_numpy(inputs.residue_index[None]),
    "chain_labels": models.torch.from_numpy(inputs.chain_index[None]),
    "randn": models.torch.from_numpy(inputs.randn[None]),
    "batch_size": 1,
    "symmetry_residues": [[]],
    "symmetry_weights": [[]],
    "Y": models.torch.from_numpy(y[None]),
    "Y_t": models.torch.from_numpy(y_t[None]),
    "Y_m": models.torch.from_numpy(y_m[None]),
    "temperature": 1.0,
  }
  if xyz_37 is not None:
    feature_dict["xyz_37"] = models.torch.from_numpy(xyz_37[None])
  if xyz_37_m is not None:
    feature_dict["xyz_37_m"] = models.torch.from_numpy(xyz_37_m[None])
  if chain_mask is not None:
    feature_dict["chain_mask"] = models.torch.from_numpy(chain_mask[None])
  bias = np.zeros((inputs.sequence.shape[0], 21), dtype=np.float32)
  bias[np.arange(inputs.sequence.shape[0]), inputs.sequence] = 30.0
  feature_dict["bias"] = models.torch.from_numpy(bias[None])
  with models.torch.no_grad():
    models.torch.manual_seed(sample_seed)
    pt_sample = models.pt_model.sample(feature_dict)

  jax_sequence, logits_jax = models.jax_model(
    jnp.asarray(inputs.atom4_coordinates),
    jnp.asarray(inputs.mask),
    jnp.asarray(inputs.residue_index),
    jnp.asarray(inputs.chain_index),
    jnp.asarray(y),
    jnp.asarray(y_t),
    jnp.asarray(y_m),
    "autoregressive",
    prng_key=jax.random.PRNGKey(sample_seed + 15),
    ar_mask=jnp.asarray(inputs.ar_mask),
    temperature=1.0,
    bias=jnp.asarray(bias),
    inference=True,
    xyz_37=None if xyz_37 is None else jnp.asarray(xyz_37),
    xyz_37_m=None if xyz_37_m is None else jnp.asarray(xyz_37_m),
    chain_mask=None if chain_mask is None else jnp.asarray(chain_mask),
  )
  pt_tokens = pt_sample["S"].numpy()[0]
  jax_tokens = np.asarray(jnp.argmax(jax_sequence, axis=-1))
  pt_log_probs = pt_sample["log_probs"].numpy()[0]
  jax_log_probs = np.asarray(jax.nn.log_softmax(logits_jax, axis=-1))
  return pt_tokens, jax_tokens, pt_log_probs, jax_log_probs


def _ligand_tied_sampling_outputs(
  models: LigandModels,
  inputs: CoreCaseInputs,
  y: np.ndarray,
  y_t: np.ndarray,
  y_m: np.ndarray,
  *,
  sample_seed: int,
  lane: TiedMultistateComparisonLane,
  xyz_37: np.ndarray | None = None,
  xyz_37_m: np.ndarray | None = None,
  chain_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
  if lane.comparison_api != "sampling":
    msg = f"{lane.path_id}/{lane.condition}: expected sampling lane."
    raise ValueError(msg)

  seq_len = int(inputs.sequence.shape[0])
  tie_groups, tie_weights, tie_group_map = _build_tie_groups(seq_len)
  bias = np.zeros((inputs.sequence.shape[0], 21), dtype=np.float32)
  bias[np.arange(inputs.sequence.shape[0]), inputs.sequence] = 30.0

  feature_dict = {
    "X": models.torch.from_numpy(inputs.atom4_coordinates[None]),
    "S": models.torch.from_numpy(inputs.sequence[None]),
    "mask": models.torch.from_numpy(inputs.mask[None]),
    "chain_mask": models.torch.from_numpy(inputs.chain_mask[None]),
    "R_idx": models.torch.from_numpy(inputs.residue_index[None]),
    "chain_labels": models.torch.from_numpy(inputs.chain_index[None]),
    "randn": models.torch.from_numpy(inputs.randn[None]),
    "batch_size": 1,
    "symmetry_residues": tie_groups,
    "symmetry_weights": tie_weights,
    "Y": models.torch.from_numpy(y[None]),
    "Y_t": models.torch.from_numpy(y_t[None]),
    "Y_m": models.torch.from_numpy(y_m[None]),
    "temperature": 1.0,
    "bias": models.torch.from_numpy(bias[None]),
  }
  if xyz_37 is not None:
    feature_dict["xyz_37"] = models.torch.from_numpy(xyz_37[None])
  if xyz_37_m is not None:
    feature_dict["xyz_37_m"] = models.torch.from_numpy(xyz_37_m[None])
  if chain_mask is not None:
    feature_dict["chain_mask"] = models.torch.from_numpy(chain_mask[None])

  with models.torch.no_grad():
    models.torch.manual_seed(sample_seed)
    pt_sample = models.pt_model.sample(feature_dict)

  jax_sequence, logits_jax = models.jax_model(
    jnp.asarray(inputs.atom4_coordinates),
    jnp.asarray(inputs.mask),
    jnp.asarray(inputs.residue_index),
    jnp.asarray(inputs.chain_index),
    jnp.asarray(y),
    jnp.asarray(y_t),
    jnp.asarray(y_m),
    "autoregressive",
    prng_key=jax.random.PRNGKey(sample_seed + 21),
    ar_mask=jnp.asarray(inputs.ar_mask),
    temperature=1.0,
    bias=jnp.asarray(bias),
    inference=True,
    xyz_37=None if xyz_37 is None else jnp.asarray(xyz_37),
    xyz_37_m=None if xyz_37_m is None else jnp.asarray(xyz_37_m),
    chain_mask=None if chain_mask is None else jnp.asarray(chain_mask),
    tie_group_map=jnp.asarray(tie_group_map),
    multi_state_strategy=lane.jax_multi_state_strategy,
  )
  pt_tokens = pt_sample["S"].numpy()[0]
  jax_tokens = np.asarray(jnp.argmax(jax_sequence, axis=-1))
  pt_log_probs = _combine_reference_tied_log_probs(
    pt_sample["log_probs"].numpy()[0],
    tie_groups=tie_groups,
    tie_weights=tie_weights,
    combiner=lane.reference_combiner,
    temperature=1.0,
  )
  jax_log_probs = np.asarray(jax.nn.log_softmax(logits_jax, axis=-1))
  return pt_tokens, jax_tokens, pt_log_probs, jax_log_probs, tie_groups


def _ligand_tied_scoring_outputs(
  models: LigandModels,
  inputs: CoreCaseInputs,
  y: np.ndarray,
  y_t: np.ndarray,
  y_m: np.ndarray,
  *,
  lane: TiedMultistateComparisonLane,
  xyz_37: np.ndarray | None = None,
  xyz_37_m: np.ndarray | None = None,
  chain_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
  if lane.comparison_api != "scoring":
    msg = f"{lane.path_id}/{lane.condition}: expected scoring lane."
    raise ValueError(msg)

  seq_len = int(inputs.sequence.shape[0])
  tie_groups, tie_weights, tie_group_map = _build_tie_groups(seq_len)

  feature_dict = {
    "X": models.torch.from_numpy(inputs.atom4_coordinates[None]),
    "S": models.torch.from_numpy(inputs.sequence[None]),
    "mask": models.torch.from_numpy(inputs.mask[None]),
    "chain_mask": models.torch.from_numpy(inputs.chain_mask[None]),
    "R_idx": models.torch.from_numpy(inputs.residue_index[None]),
    "chain_labels": models.torch.from_numpy(inputs.chain_index[None]),
    "randn": models.torch.from_numpy(inputs.randn[None]),
    "batch_size": 1,
    "symmetry_residues": tie_groups,
    "symmetry_weights": tie_weights,
    "Y": models.torch.from_numpy(y[None]),
    "Y_t": models.torch.from_numpy(y_t[None]),
    "Y_m": models.torch.from_numpy(y_m[None]),
  }
  if xyz_37 is not None:
    feature_dict["xyz_37"] = models.torch.from_numpy(xyz_37[None])
  if xyz_37_m is not None:
    feature_dict["xyz_37_m"] = models.torch.from_numpy(xyz_37_m[None])
  if chain_mask is not None:
    feature_dict["chain_mask"] = models.torch.from_numpy(chain_mask[None])

  with models.torch.no_grad():
    pt_score = models.pt_model.score(feature_dict, use_sequence=True)

  pt_combined_logits = _combine_reference_tied_logits(
    pt_score["logits"].numpy()[0],
    tie_groups=tie_groups,
    tie_weights=tie_weights,
    combiner=lane.reference_combiner,
    temperature=1.0,
  )
  pt_log_probs = _row_log_softmax(pt_combined_logits)
  pt_tokens = np.asarray(np.argmax(pt_log_probs, axis=-1), dtype=np.int32)

  _, jax_logits = models.jax_model(
    jnp.asarray(inputs.atom4_coordinates),
    jnp.asarray(inputs.mask),
    jnp.asarray(inputs.residue_index),
    jnp.asarray(inputs.chain_index),
    jnp.asarray(y),
    jnp.asarray(y_t),
    jnp.asarray(y_m),
    "conditional",
    prng_key=jax.random.PRNGKey(int(np.sum(inputs.residue_index)) + 23),
    ar_mask=jnp.asarray(inputs.ar_mask),
    one_hot_sequence=jax.nn.one_hot(jnp.asarray(inputs.sequence), 21),
    inference=True,
    xyz_37=None if xyz_37 is None else jnp.asarray(xyz_37),
    xyz_37_m=None if xyz_37_m is None else jnp.asarray(xyz_37_m),
    chain_mask=None if chain_mask is None else jnp.asarray(chain_mask),
    tie_group_map=jnp.asarray(tie_group_map),
    multi_state_strategy=lane.jax_multi_state_strategy,
  )
  jax_log_probs = np.asarray(jax.nn.log_softmax(jax_logits, axis=-1))
  jax_tokens = np.asarray(jnp.argmax(jax_log_probs, axis=-1), dtype=np.int32)
  return pt_tokens, jax_tokens, pt_log_probs, jax_log_probs, tie_groups


def _packer_outputs(models: PackerModels, case: BackboneCase, atom_context_num: int) -> tuple[np.ndarray, np.ndarray]:
  seq_len = int(case.sequence.shape[0])
  rng = np.random.default_rng(case.seed + 2000)
  x_np = np.zeros((seq_len, 14, 3), dtype=np.float32)
  x_np[:, :4, :] = case.atom37_coordinates[:, [0, 1, 2, 4], :]
  y_np = rng.standard_normal((seq_len, atom_context_num, 3), dtype=np.float32)
  y_m_np = np.ones((seq_len, atom_context_num), dtype=np.float32)
  y_t_np = rng.integers(0, 119, (seq_len, atom_context_num), dtype=np.int64)
  x_m_np = np.ones((seq_len, 14), dtype=np.float32)

  feature_jax = {
    "S": jnp.asarray(case.sequence, dtype=jnp.int64),
    "X": jnp.asarray(x_np),
    "Y": jnp.asarray(y_np),
    "Y_m": jnp.asarray(y_m_np),
    "Y_t": jnp.asarray(y_t_np),
    "mask": jnp.asarray(case.mask, dtype=jnp.float32),
    "R_idx": jnp.asarray(case.residue_index, dtype=jnp.int64),
    "chain_labels": jnp.asarray(case.chain_index, dtype=jnp.int64),
    "X_m": jnp.asarray(x_m_np, dtype=jnp.float32),
  }
  feature_pt = {
    "S": models.torch.from_numpy(case.sequence[None]),
    "X": models.torch.from_numpy(x_np[None]),
    "Y": models.torch.from_numpy(y_np[None]),
    "Y_m": models.torch.from_numpy(y_m_np[None]),
    "Y_t": models.torch.from_numpy(y_t_np[None]),
    "mask": models.torch.from_numpy(case.mask[None]),
    "R_idx": models.torch.from_numpy(case.residue_index[None]),
    "chain_labels": models.torch.from_numpy(case.chain_index[None]),
    "X_m": models.torch.from_numpy(x_m_np[None]),
  }
  mean_jax, concentration_jax, mix_jax = models.jax_model(feature_jax)
  with models.torch.no_grad():
    h_v_pt, h_e_pt, e_idx_pt = models.pt_model.encode(feature_pt)
    feature_pt.update({"h_V": h_v_pt, "h_E": h_e_pt, "E_idx": e_idx_pt})
    mean_pt, concentration_pt, mix_pt = models.pt_model.decode(feature_pt)
  reference = np.concatenate(
    [
      mean_pt.detach().cpu().numpy()[0].ravel(),
      concentration_pt.detach().cpu().numpy()[0].ravel(),
      mix_pt.detach().cpu().numpy()[0].ravel(),
    ],
  )
  observed = np.concatenate(
    [
      np.asarray(mean_jax).ravel(),
      np.asarray(concentration_jax).ravel(),
      np.asarray(mix_jax).ravel(),
    ],
  )
  return reference, observed


def _append_scalar_metrics(
  metric_rows: list[EvidenceMetricRecord],
  *,
  path_id: str,
  tier: str,
  case: BackboneCase,
  checkpoint_id: str,
  correlation: float | None = None,
  correlation_threshold: float | None = None,
  cosine_similarity: float | None = None,
  spearman_correlation: float | None = None,
  mae: float | None = None,
  rmse: float | None = None,
  max_abs: float | None = None,
  kl: float | None = None,
  token_acc: float | None = None,
  token_threshold: float | None = None,
  allclose_pass: bool | None = None,
  metric_group: str | None = None,
  condition: str | None = None,
  note: str | None = None,
) -> None:
  def add(metric_name: str, metric_value: float, threshold: float | None = None, passed: bool | None = None) -> None:
    metric_rows.append(
      EvidenceMetricRecord(
        path_id=path_id,
        tier=tier,
        case_id=case.id,
        case_kind=case.kind,
        backbone_id=case.id if case.kind == "real_backbone" else "synthetic",
        seed=case.seed,
        sequence_length=int(case.sequence.shape[0]),
        checkpoint_id=checkpoint_id,
        metric_name=metric_name,
        metric_value=metric_value,
        threshold=threshold,
        passed=passed,
        metric_group=metric_group,
        condition=condition,
        note=note,
      ),
    )

  if correlation is not None:
    passed = correlation >= correlation_threshold if correlation_threshold is not None else None
    add("pearson_correlation", correlation, threshold=correlation_threshold, passed=passed)
  if cosine_similarity is not None:
    add("cosine_similarity", cosine_similarity)
  if spearman_correlation is not None:
    add("spearman_correlation", spearman_correlation)
  if mae is not None:
    add("mae", mae)
  if rmse is not None:
    add("rmse", rmse)
  if max_abs is not None:
    add("max_abs_error", max_abs)
  if kl is not None:
    add("mean_kl_divergence", kl)
  if token_acc is not None:
    passed = token_acc >= token_threshold if token_threshold is not None else None
    add("token_agreement", token_acc, threshold=token_threshold, passed=passed)
  if allclose_pass is not None:
    add("allclose_pass", float(allclose_pass), passed=allclose_pass)


def _append_point_records(
  point_rows: list[EvidencePointRecord],
  *,
  path_id: str,
  tier: str,
  case: BackboneCase,
  reference_values: np.ndarray,
  observed_values: np.ndarray,
  max_points: int,
  point_kind: str | None = None,
  condition: str | None = None,
) -> None:
  sampled_ref, sampled_obs = downsample_pair_points(
    reference_values,
    observed_values,
    max_points=max_points,
    seed=case.seed,
  )
  for ref_value, obs_value in zip(sampled_ref, sampled_obs, strict=False):
    point_rows.append(
      EvidencePointRecord(
        path_id=path_id,
        tier=tier,
        case_id=case.id,
        case_kind=case.kind,
        backbone_id=case.id if case.kind == "real_backbone" else "synthetic",
        seed=case.seed,
        sequence_length=int(case.sequence.shape[0]),
        reference_value=float(ref_value),
        observed_value=float(obs_value),
        point_kind=point_kind,
        condition=condition,
      ),
    )


def _append_intrinsic_baseline_metrics(
  metric_rows: list[EvidenceMetricRecord],
  *,
  path_id: str,
  tier: str,
  case: BackboneCase,
  checkpoint_id: str,
  reference_runs: list[np.ndarray],
  observed_runs: list[np.ndarray],
  condition: str | None = None,
  note: str | None = None,
) -> None:
  if len(reference_runs) < 2 or len(observed_runs) < 2:
    return
  if len(reference_runs) != len(observed_runs):
    msg = "Intrinsic baseline runs must have matching reference/observed replicate counts."
    raise ValueError(msg)

  ref_deltas = np.asarray(
    [mean_abs_error(reference_runs[0], run) for run in reference_runs[1:]],
    dtype=np.float64,
  )
  obs_deltas = np.asarray(
    [mean_abs_error(observed_runs[0], run) for run in observed_runs[1:]],
    dtype=np.float64,
  )
  pooled = np.concatenate([ref_deltas, obs_deltas], axis=0)
  cross_mae = mean_abs_error(reference_runs[0], observed_runs[0])
  pooled_mean = float(np.mean(pooled)) if pooled.size else 0.0
  ratio = cross_mae / max(pooled_mean, 1e-8)
  envelope_95 = float(np.quantile(pooled, 0.95)) if pooled.size else 0.0
  envelope_99 = float(np.quantile(pooled, 0.99)) if pooled.size else 0.0
  pass_95 = cross_mae <= envelope_95 if pooled.size else cross_mae <= 1e-8
  pass_99 = cross_mae <= envelope_99 if pooled.size else cross_mae <= 1e-8
  status_code = 0.0 if pass_95 else (1.0 if pass_99 else 2.0)

  def add(
    metric_name: str,
    metric_value: float,
    *,
    threshold: float | None = None,
    passed: bool | None = None,
  ) -> None:
    metric_rows.append(
      EvidenceMetricRecord(
        path_id=path_id,
        tier=tier,
        case_id=case.id,
        case_kind=case.kind,
        backbone_id=case.id if case.kind == "real_backbone" else "synthetic",
        seed=case.seed,
        sequence_length=int(case.sequence.shape[0]),
        checkpoint_id=checkpoint_id,
        metric_name=metric_name,
        metric_value=float(metric_value),
        threshold=threshold,
        passed=passed,
        metric_group="intrinsic",
        condition=condition,
        note=note,
      ),
    )

  add("intrinsic_ref_mae_mean", float(np.mean(ref_deltas)))
  add("intrinsic_ref_mae_std", float(np.std(ref_deltas)))
  add("intrinsic_jax_mae_mean", float(np.mean(obs_deltas)))
  add("intrinsic_jax_mae_std", float(np.std(obs_deltas)))
  add("intrinsic_pooled_mae_mean", pooled_mean)
  add("parity_cross_mae", cross_mae)
  add("parity_to_intrinsic_ratio", ratio)
  add("intrinsic_envelope95", envelope_95)
  add("intrinsic_envelope99", envelope_99)
  add("intrinsic_pass_95", float(pass_95), threshold=envelope_95, passed=pass_95)
  add("intrinsic_pass_99", float(pass_99), threshold=envelope_99, passed=pass_99)
  add("intrinsic_status_code", status_code)


def _append_macro_distribution_metrics(
  metric_rows: list[EvidenceMetricRecord],
  point_rows: list[EvidencePointRecord],
  *,
  path_id: str,
  tier: str,
  case: BackboneCase,
  checkpoint_id: str,
  condition: str,
  identity_reference: np.ndarray,
  identity_observed: np.ndarray,
  entropy_reference: np.ndarray,
  entropy_observed: np.ndarray,
  composition_reference: np.ndarray,
  composition_observed: np.ndarray,
  identity_wasserstein_max: float | None = None,
  entropy_wasserstein_max: float | None = None,
  composition_js_max: float | None = None,
  identity_ks_pvalue_min: float | None = None,
) -> dict[str, float]:
  ks_identity = ks_2samp(identity_reference, identity_observed)
  ks_entropy = ks_2samp(entropy_reference, entropy_observed)
  identity_wasserstein = float(wasserstein_distance(identity_reference, identity_observed))
  entropy_wasserstein = float(wasserstein_distance(entropy_reference, entropy_observed))
  composition_js = float(jensenshannon(composition_reference, composition_observed))
  identity_delta = float(np.mean(identity_observed) - np.mean(identity_reference))
  entropy_delta = float(np.mean(entropy_observed) - np.mean(entropy_reference))

  def add(
    metric_name: str,
    metric_value: float,
    *,
    threshold: float | None = None,
    passed: bool | None = None,
  ) -> None:
    metric_rows.append(
      EvidenceMetricRecord(
        path_id=path_id,
        tier=tier,
        case_id=case.id,
        case_kind=case.kind,
        backbone_id=case.id if case.kind == "real_backbone" else "synthetic",
        seed=case.seed,
        sequence_length=int(case.sequence.shape[0]),
        checkpoint_id=checkpoint_id,
        metric_name=metric_name,
        metric_value=metric_value,
        threshold=threshold,
        passed=passed,
        metric_group="macro",
        condition=condition,
      ),
    )

  add(
    "macro_identity_wasserstein",
    identity_wasserstein,
    threshold=identity_wasserstein_max,
    passed=(
      identity_wasserstein <= identity_wasserstein_max
      if identity_wasserstein_max is not None
      else None
    ),
  )
  add("macro_identity_ks_stat", float(ks_identity.statistic))
  add(
    "macro_identity_ks_pvalue",
    float(ks_identity.pvalue),
    threshold=identity_ks_pvalue_min,
    passed=(
      float(ks_identity.pvalue) >= identity_ks_pvalue_min
      if identity_ks_pvalue_min is not None
      else None
    ),
  )
  add(
    "macro_entropy_wasserstein",
    entropy_wasserstein,
    threshold=entropy_wasserstein_max,
    passed=(
      entropy_wasserstein <= entropy_wasserstein_max
      if entropy_wasserstein_max is not None
      else None
    ),
  )
  add("macro_entropy_ks_stat", float(ks_entropy.statistic))
  add("macro_entropy_ks_pvalue", float(ks_entropy.pvalue))
  add(
    "macro_composition_js_distance",
    composition_js,
    threshold=composition_js_max,
    passed=composition_js <= composition_js_max if composition_js_max is not None else None,
  )
  add("macro_identity_mean_delta", identity_delta)
  add("macro_entropy_mean_delta", entropy_delta)

  for ref_value, obs_value in zip(identity_reference, identity_observed, strict=False):
    point_rows.append(
      EvidencePointRecord(
        path_id=path_id,
        tier=tier,
        case_id=case.id,
        case_kind=case.kind,
        backbone_id=case.id if case.kind == "real_backbone" else "synthetic",
        seed=case.seed,
        sequence_length=int(case.sequence.shape[0]),
        reference_value=float(ref_value),
        observed_value=float(obs_value),
        point_kind="macro_sequence_identity",
        condition=condition,
      ),
    )
  for ref_value, obs_value in zip(entropy_reference, entropy_observed, strict=False):
    point_rows.append(
      EvidencePointRecord(
        path_id=path_id,
        tier=tier,
        case_id=case.id,
        case_kind=case.kind,
        backbone_id=case.id if case.kind == "real_backbone" else "synthetic",
        seed=case.seed,
        sequence_length=int(case.sequence.shape[0]),
        reference_value=float(ref_value),
        observed_value=float(obs_value),
        point_kind="macro_entropy",
        condition=condition,
      ),
    )

  return {
    "macro_identity_wasserstein": identity_wasserstein,
    "macro_entropy_wasserstein": entropy_wasserstein,
    "macro_composition_js_distance": composition_js,
    "macro_identity_ks_pvalue": float(ks_identity.pvalue),
  }


def _build_xyz37_mask(atom37_coordinates: np.ndarray, residue_mask: np.ndarray) -> np.ndarray:
  atom37 = np.asarray(atom37_coordinates, dtype=np.float32)
  atom_mask = (np.linalg.norm(atom37, axis=-1) > 1e-6).astype(np.float32)
  residue_mask_array = np.asarray(residue_mask, dtype=np.float32)
  atom_mask[:, :5] = np.maximum(atom_mask[:, :5], residue_mask_array[:, None])
  return atom_mask


def _has_sidechain_atoms(xyz_37_m: np.ndarray) -> bool:
  return bool(np.any(np.asarray(xyz_37_m[:, 5:], dtype=np.float32) > 0.5))


def _collect_no_ligand_macro_samples(
  case: BackboneCase,
  core_models: CoreModels,
  core_inputs: CoreCaseInputs,
  *,
  num_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  valid_mask = core_inputs.mask > 0.5
  identity_ref: list[float] = []
  identity_obs: list[float] = []
  entropy_ref: list[float] = []
  entropy_obs: list[float] = []
  sampled_tokens_ref: list[np.ndarray] = []
  sampled_tokens_obs: list[np.ndarray] = []

  for sample_index in range(num_samples):
    sample_seed = case.seed + 10_000 + sample_index
    pt_tokens, jax_tokens, pt_log_probs, jax_log_probs = _core_autoregressive_outputs(
      core_models,
      core_inputs,
      sample_seed=sample_seed,
    )
    sampled_tokens_ref.append(np.asarray(pt_tokens))
    sampled_tokens_obs.append(np.asarray(jax_tokens))
    identity_ref.append(sequence_identity(pt_tokens, core_inputs.sequence, valid_mask))
    identity_obs.append(sequence_identity(jax_tokens, core_inputs.sequence, valid_mask))
    entropy_ref.append(per_position_entropy(pt_log_probs, valid_mask))
    entropy_obs.append(per_position_entropy(jax_log_probs, valid_mask))

  token_ref = np.stack(sampled_tokens_ref, axis=0)
  token_obs = np.stack(sampled_tokens_obs, axis=0)
  mask_matrix = np.broadcast_to(valid_mask[None, :], token_ref.shape)
  composition_ref = amino_acid_distribution(token_ref, mask=mask_matrix)
  composition_obs = amino_acid_distribution(token_obs, mask=mask_matrix)
  return (
    np.asarray(identity_ref, dtype=np.float64),
    np.asarray(identity_obs, dtype=np.float64),
    np.asarray(entropy_ref, dtype=np.float64),
    np.asarray(entropy_obs, dtype=np.float64),
    composition_ref,
    composition_obs,
  )


def _collect_ligand_macro_samples(
  case: BackboneCase,
  ligand_models: LigandModels,
  core_inputs: CoreCaseInputs,
  y: np.ndarray,
  y_t: np.ndarray,
  y_m: np.ndarray,
  *,
  num_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  valid_mask = core_inputs.mask > 0.5
  identity_ref: list[float] = []
  identity_obs: list[float] = []
  entropy_ref: list[float] = []
  entropy_obs: list[float] = []
  sampled_tokens_ref: list[np.ndarray] = []
  sampled_tokens_obs: list[np.ndarray] = []

  for sample_index in range(num_samples):
    sample_seed = case.seed + 20_000 + sample_index
    pt_tokens, jax_tokens, pt_log_probs, jax_log_probs = _ligand_autoregressive_outputs(
      ligand_models,
      core_inputs,
      y,
      y_t,
      y_m,
      sample_seed=sample_seed,
    )
    sampled_tokens_ref.append(np.asarray(pt_tokens))
    sampled_tokens_obs.append(np.asarray(jax_tokens))
    identity_ref.append(sequence_identity(pt_tokens, core_inputs.sequence, valid_mask))
    identity_obs.append(sequence_identity(jax_tokens, core_inputs.sequence, valid_mask))
    entropy_ref.append(per_position_entropy(pt_log_probs, valid_mask))
    entropy_obs.append(per_position_entropy(jax_log_probs, valid_mask))

  token_ref = np.stack(sampled_tokens_ref, axis=0)
  token_obs = np.stack(sampled_tokens_obs, axis=0)
  mask_matrix = np.broadcast_to(valid_mask[None, :], token_ref.shape)
  composition_ref = amino_acid_distribution(token_ref, mask=mask_matrix)
  composition_obs = amino_acid_distribution(token_obs, mask=mask_matrix)
  return (
    np.asarray(identity_ref, dtype=np.float64),
    np.asarray(identity_obs, dtype=np.float64),
    np.asarray(entropy_ref, dtype=np.float64),
    np.asarray(entropy_obs, dtype=np.float64),
    composition_ref,
    composition_obs,
  )


def _collect_sidechain_conditioned_macro_samples(
  case: BackboneCase,
  ligand_models: LigandModels,
  core_inputs: CoreCaseInputs,
  y: np.ndarray,
  y_t: np.ndarray,
  y_m: np.ndarray,
  *,
  num_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  xyz_37 = np.asarray(core_inputs.atom37_coordinates, dtype=np.float32)
  xyz_37_m = _build_xyz37_mask(xyz_37, core_inputs.mask)
  if not _has_sidechain_atoms(xyz_37_m):
    msg = "No side-chain atoms available in case coordinates."
    raise ValueError(msg)

  valid_mask = core_inputs.mask > 0.5
  chain_mask = np.zeros_like(core_inputs.mask, dtype=np.float32)
  sidechain_y_m = np.zeros_like(y_m, dtype=np.float32)
  identity_ref: list[float] = []
  identity_obs: list[float] = []
  entropy_ref: list[float] = []
  entropy_obs: list[float] = []
  sampled_tokens_ref: list[np.ndarray] = []
  sampled_tokens_obs: list[np.ndarray] = []

  for sample_index in range(num_samples):
    sample_seed = case.seed + 30_000 + sample_index
    pt_tokens, jax_tokens, pt_log_probs, jax_log_probs = _ligand_autoregressive_outputs(
      ligand_models,
      core_inputs,
      y,
      y_t,
      sidechain_y_m,
      sample_seed=sample_seed,
      xyz_37=xyz_37,
      xyz_37_m=xyz_37_m,
      chain_mask=chain_mask,
    )
    sampled_tokens_ref.append(np.asarray(pt_tokens))
    sampled_tokens_obs.append(np.asarray(jax_tokens))
    identity_ref.append(sequence_identity(pt_tokens, core_inputs.sequence, valid_mask))
    identity_obs.append(sequence_identity(jax_tokens, core_inputs.sequence, valid_mask))
    entropy_ref.append(per_position_entropy(pt_log_probs, valid_mask))
    entropy_obs.append(per_position_entropy(jax_log_probs, valid_mask))

  token_ref = np.stack(sampled_tokens_ref, axis=0)
  token_obs = np.stack(sampled_tokens_obs, axis=0)
  mask_matrix = np.broadcast_to(valid_mask[None, :], token_ref.shape)
  composition_ref = amino_acid_distribution(token_ref, mask=mask_matrix)
  composition_obs = amino_acid_distribution(token_obs, mask=mask_matrix)
  return (
    np.asarray(identity_ref, dtype=np.float64),
    np.asarray(identity_obs, dtype=np.float64),
    np.asarray(entropy_ref, dtype=np.float64),
    np.asarray(entropy_obs, dtype=np.float64),
    composition_ref,
    composition_obs,
  )


def _evaluate_sidechain_conditioned_gate(
  *,
  requested: bool,
  macro_metrics: dict[str, float] | None,
  acceptance: SidechainMacroAcceptance,
  infeasible_reason: str | None = None,
) -> tuple[str, str]:
  if not requested:
    return ("warn", "Excluded by parity case corpus configuration.")
  if infeasible_reason is not None:
    return ("warn", infeasible_reason)
  if macro_metrics is None:
    return ("warn", "Requested but side-chain-conditioned metrics were not collected.")

  checks = [
    (
      "identity_wasserstein",
      macro_metrics["macro_identity_wasserstein"],
      acceptance.identity_wasserstein_max,
      "<=",
    ),
    (
      "entropy_wasserstein",
      macro_metrics["macro_entropy_wasserstein"],
      acceptance.entropy_wasserstein_max,
      "<=",
    ),
    (
      "composition_js_distance",
      macro_metrics["macro_composition_js_distance"],
      acceptance.composition_js_max,
      "<=",
    ),
    (
      "identity_ks_pvalue",
      macro_metrics["macro_identity_ks_pvalue"],
      acceptance.identity_ks_pvalue_min,
      ">=",
    ),
  ]
  failures: list[str] = []
  for metric_name, observed, threshold, comparator in checks:
    failed = observed > threshold if comparator == "<=" else observed < threshold
    if failed:
      failures.append(
        f"{metric_name}={observed:.4f} {comparator} {threshold:.4f} violated",
      )
  if failures:
    return ("fail", "Threshold violations: " + "; ".join(failures))
  return ("pass", "All side-chain-conditioned macro acceptance checks passed.")


def _sidechain_gate_status_code(status: str) -> float:
  status_map = {"pass": 0.0, "warn": 1.0, "fail": 2.0}
  return status_map.get(status, 1.0)


def _build_mock_protein(case: BackboneCase) -> Protein:
  atom4 = jnp.asarray(case.atom37_coordinates[:, [0, 1, 2, 4], :], dtype=jnp.float32)
  aatype = jnp.asarray(case.sequence, dtype=jnp.int8)
  return Protein(
    coordinates=atom4[None, ...],
    mask=jnp.asarray(case.mask[None, ...], dtype=jnp.float32),
    residue_index=jnp.asarray(case.residue_index[None, ...], dtype=jnp.int32),
    chain_index=jnp.asarray(case.chain_index[None, ...], dtype=jnp.int32),
    aatype=aatype[None, ...],
    one_hot_sequence=jax.nn.one_hot(aatype, 21)[None, ...],
    coulomb14scale=None,
    lj14scale=None,
  )


def _collect_fast_logit_helper_metrics(
  case: BackboneCase,
  core_inputs: CoreCaseInputs,
  core_models: CoreModels,
  *,
  point_sample_size: int,
) -> tuple[list[EvidenceMetricRecord], list[EvidencePointRecord]]:
  metric_rows: list[EvidenceMetricRecord] = []
  point_rows: list[EvidencePointRecord] = []
  model = eqx.nn.inference_mode(core_models.jax_model, value=True)
  # ``aminx.sampling.unconditional_logits`` (and its ``make_unconditional_logits_fn``) was deleted
  # in 0.2.0a1 with nothing superseding it, so only the conditional helper is compared here.
  conditional_helper = make_conditional_logits_fn(model)

  coordinates = jnp.asarray(case.atom37_coordinates)
  mask = jnp.asarray(case.mask)
  residue_index = jnp.asarray(case.residue_index)
  chain_index = jnp.asarray(case.chain_index)
  ar_mask = jnp.asarray(core_inputs.ar_mask)
  sequence_one_hot = jax.nn.one_hot(jnp.asarray(case.sequence), 21)
  key = jax.random.PRNGKey(case.seed + 3)

  helper_conditional = conditional_helper(
    key,
    coordinates,
    mask,
    residue_index,
    chain_index,
    sequence_one_hot,
    ar_mask,
    jnp.asarray(0.0, dtype=jnp.float32),
    None,
  )
  _, direct_conditional = model(
    coordinates,
    mask,
    residue_index,
    chain_index,
    "conditional",
    prng_key=key,
    ar_mask=ar_mask,
    one_hot_sequence=sequence_one_hot,
    backbone_noise=jnp.asarray(0.0, dtype=jnp.float32),
  )

  reference = np.asarray(direct_conditional).ravel()
  observed = np.asarray(helper_conditional).ravel()
  _append_scalar_metrics(
    metric_rows,
    path_id="logits-helper-branches",
    tier="parity_fast",
    case=case,
    checkpoint_id=core_models.checkpoint_id,
    correlation=safe_pearson(reference, observed),
    mae=mean_abs_error(reference, observed),
    rmse=root_mean_square_error(reference, observed),
    max_abs=max_abs_error(reference, observed),
    allclose_pass=bool(np.allclose(reference, observed, atol=1e-6, rtol=1e-6)),
    metric_group="fast",
  )
  _append_point_records(
    point_rows,
    path_id="logits-helper-branches",
    tier="parity_fast",
    case=case,
    reference_values=reference,
    observed_values=observed,
    max_points=point_sample_size,
    point_kind="fast_helper_logits",
  )
  return metric_rows, point_rows


def _collect_averaged_encoding_metrics(
  case: BackboneCase,
  core_models: CoreModels,
  *,
  point_sample_size: int,
) -> tuple[list[EvidenceMetricRecord], list[EvidencePointRecord]]:
  metric_rows: list[EvidenceMetricRecord] = []
  point_rows: list[EvidencePointRecord] = []
  model = eqx.nn.inference_mode(core_models.jax_model, value=True)

  atom4 = jnp.asarray(case.atom37_coordinates[:, [0, 1, 2, 4], :], dtype=jnp.float32)
  seq_len = atom4.shape[0]
  coordinates = jnp.stack([atom4, atom4 + 0.05], axis=0)
  sequence = jnp.asarray(case.sequence, dtype=jnp.int8)
  aatype = jnp.stack([sequence, sequence], axis=0)
  residue_index = jnp.stack(
    [
      jnp.asarray(case.residue_index, dtype=jnp.int32),
      jnp.asarray(case.residue_index, dtype=jnp.int32),
    ],
    axis=0,
  )
  chain_index = jnp.stack(
    [
      jnp.asarray(case.chain_index, dtype=jnp.int32),
      jnp.asarray(case.chain_index, dtype=jnp.int32),
    ],
    axis=0,
  )
  ensemble = Protein(
    coordinates=coordinates,
    mask=jnp.ones((2, seq_len), dtype=jnp.float32),
    residue_index=residue_index,
    chain_index=chain_index,
    aatype=aatype,
    one_hot_sequence=jax.nn.one_hot(aatype, 21),
    coulomb14scale=None,
    lj14scale=None,
  )
  first = get_averaged_encodings(
    ensemble,
    model,
    backbone_noise=(0.0, 0.2),
    noise_batch_size=2,
    random_seed=case.seed + 17,
    average_encoding_mode="inputs_and_noise",
  )
  second = get_averaged_encodings(
    ensemble,
    model,
    backbone_noise=(0.0, 0.2),
    noise_batch_size=2,
    random_seed=case.seed + 17,
    average_encoding_mode="inputs_and_noise",
  )
  reference = np.concatenate([np.asarray(first[0]).ravel(), np.asarray(first[1]).ravel()])
  observed = np.concatenate([np.asarray(second[0]).ravel(), np.asarray(second[1]).ravel()])
  _append_scalar_metrics(
    metric_rows,
    path_id="averaged-encoding",
    tier="parity_fast",
    case=case,
    checkpoint_id=core_models.checkpoint_id,
    correlation=safe_pearson(reference, observed),
    mae=mean_abs_error(reference, observed),
    rmse=root_mean_square_error(reference, observed),
    max_abs=max_abs_error(reference, observed),
    allclose_pass=bool(np.allclose(reference, observed, atol=1e-6, rtol=1e-6)),
    metric_group="fast",
  )
  _append_point_records(
    point_rows,
    path_id="averaged-encoding",
    tier="parity_fast",
    case=case,
    reference_values=reference,
    observed_values=observed,
    max_points=point_sample_size,
    point_kind="fast_averaged_encoding",
  )
  return metric_rows, point_rows


def _collect_end_to_end_api_metrics(
  case: BackboneCase,
  core_models: CoreModels,
  *,
  point_sample_size: int,
) -> tuple[list[EvidenceMetricRecord], list[EvidencePointRecord]]:
  metric_rows: list[EvidenceMetricRecord] = []
  point_rows: list[EvidencePointRecord] = []
  model = eqx.nn.inference_mode(core_models.jax_model, value=True)
  mock_protein = _build_mock_protein(case)

  sample_spec = SamplingSpecification(
    inputs=["synthetic.pdb"],
    num_samples=2,
    backbone_noise=[0.0, 0.2],
    temperature=[0.2, 0.7],
    compute_pseudo_perplexity=True,
    random_seed=case.seed + 23,
    batch_size=1,
  )
  with patch(
    "aminx.host.runner.prep_protein_stream_and_model",
    return_value=([mock_protein], model),
  ):
    sample_first = run_sample(sample_spec)
    sample_second = run_sample(sample_spec)

  sample_ref = np.asarray(sample_first["logits"])
  sample_obs = np.asarray(sample_second["logits"])
  sample_tokens_ref = np.asarray(sample_first["sequences"])
  sample_tokens_obs = np.asarray(sample_second["sequences"])
  _append_scalar_metrics(
    metric_rows,
    path_id="end-to-end-run-apis",
    tier="parity_fast",
    case=case,
    checkpoint_id=core_models.checkpoint_id,
    correlation=safe_pearson(sample_ref, sample_obs),
    mae=mean_abs_error(sample_ref, sample_obs),
    rmse=root_mean_square_error(sample_ref, sample_obs),
    max_abs=max_abs_error(sample_ref, sample_obs),
    token_acc=token_agreement(sample_tokens_ref, sample_tokens_obs),
    token_threshold=1.0,
    allclose_pass=bool(np.allclose(sample_ref, sample_obs, atol=1e-6, rtol=1e-6)),
    metric_group="fast",
    condition="sampling_api",
  )
  if "pseudo_perplexity" in sample_first:
    pseudo = np.asarray(sample_first["pseudo_perplexity"], dtype=np.float64)
    min_pseudo = float(np.min(pseudo))
    metric_rows.append(
      EvidenceMetricRecord(
        path_id="end-to-end-run-apis",
        tier="parity_fast",
        case_id=case.id,
        case_kind=case.kind,
        backbone_id=case.id if case.kind == "real_backbone" else "synthetic",
        seed=case.seed,
        sequence_length=int(case.sequence.shape[0]),
        checkpoint_id=core_models.checkpoint_id,
        metric_name="pseudo_perplexity_min",
        metric_value=min_pseudo,
        threshold=0.0,
        passed=min_pseudo > 0.0,
        metric_group="fast",
        condition="sampling_api",
      ),
    )
  _append_point_records(
    point_rows,
    path_id="end-to-end-run-apis",
    tier="parity_fast",
    case=case,
    reference_values=sample_ref,
    observed_values=sample_obs,
    max_points=point_sample_size,
    point_kind="fast_sampling_api",
    condition="sampling_api",
  )

  base_sequence = protein_sequence_to_string(np.asarray(case.sequence))
  mutated_tokens = np.asarray(case.sequence).copy()
  if mutated_tokens.size:
    mutated_tokens[0] = int((mutated_tokens[0] + 1) % 20)
  mutated_sequence = protein_sequence_to_string(mutated_tokens)
  score_spec = ScoringSpecification(
    inputs=["synthetic.pdb"],
    sequences_to_score=[base_sequence, mutated_sequence],
    backbone_noise=[0.0, 0.2],
    random_seed=case.seed + 29,
    batch_size=1,
  )
  with patch(
    "aminx.host.runner.prep_protein_stream_and_model",
    return_value=([mock_protein], model),
  ):
    score_first = run_score(score_spec)
    score_second = run_score(score_spec)

  score_ref = np.asarray(score_first["logits"])
  score_obs = np.asarray(score_second["logits"])
  _append_scalar_metrics(
    metric_rows,
    path_id="end-to-end-run-apis",
    tier="parity_fast",
    case=case,
    checkpoint_id=core_models.checkpoint_id,
    correlation=safe_pearson(score_ref, score_obs),
    mae=mean_abs_error(score_ref, score_obs),
    rmse=root_mean_square_error(score_ref, score_obs),
    max_abs=max_abs_error(score_ref, score_obs),
    allclose_pass=bool(np.allclose(score_ref, score_obs, atol=1e-6, rtol=1e-6)),
    metric_group="fast",
    condition="scoring_api",
  )
  _append_point_records(
    point_rows,
    path_id="end-to-end-run-apis",
    tier="parity_fast",
    case=case,
    reference_values=score_ref,
    observed_values=score_obs,
    max_points=point_sample_size,
    point_kind="fast_scoring_api",
    condition="scoring_api",
  )
  return metric_rows, point_rows


def _collect_for_case(
  case: BackboneCase,
  *,
  core_models: CoreModels,
  ligand_models: LigandModels,
  ligand_sidechain_models: LigandModels | None,
  packer_models: PackerModels,
  correlation_thresholds: dict[str, float],
  tied_lanes: tuple[TiedMultistateComparisonLane, ...],
  sidechain_acceptance: SidechainMacroAcceptance,
  sidechain_model_error: str | None,
  runtime_config: EvidenceRuntimeConfig,
  point_sample_size: int,
) -> tuple[list[EvidenceMetricRecord], list[EvidencePointRecord]]:
  metric_rows: list[EvidenceMetricRecord] = []
  point_rows: list[EvidencePointRecord] = []
  core_inputs = _prepare_core_case_inputs(case)
  valid_mask = core_inputs.mask > 0.5

  feature_ref, feature_obs = _core_feature_outputs(core_models, core_inputs)
  _append_scalar_metrics(
    metric_rows,
    path_id="protein-feature-extraction",
    tier="parity_heavy",
    case=case,
    checkpoint_id=core_models.checkpoint_id,
    correlation=safe_pearson(feature_ref, feature_obs, mask=valid_mask),
    cosine_similarity=safe_cosine_similarity(feature_ref, feature_obs, mask=valid_mask),
    spearman_correlation=safe_spearman(feature_ref, feature_obs, mask=valid_mask),
    mae=mean_abs_error(feature_ref, feature_obs, mask=valid_mask),
    rmse=root_mean_square_error(feature_ref, feature_obs, mask=valid_mask),
    max_abs=max_abs_error(feature_ref, feature_obs, mask=valid_mask),
    allclose_pass=bool(np.allclose(feature_ref, feature_obs, atol=2e-5, rtol=1e-5)),
  )
  _append_point_records(
    point_rows,
    path_id="protein-feature-extraction",
    tier="parity_heavy",
    case=case,
    reference_values=feature_ref,
    observed_values=feature_obs,
    max_points=point_sample_size,
  )

  enc_node_ref, enc_node_obs, enc_edge_ref, enc_edge_obs = _core_encoder_outputs(core_models, core_inputs)
  _append_scalar_metrics(
    metric_rows,
    path_id="protein-encoder",
    tier="parity_heavy",
    case=case,
    checkpoint_id=core_models.checkpoint_id,
    correlation=safe_pearson(enc_node_ref, enc_node_obs, mask=valid_mask),
    correlation_threshold=correlation_thresholds["protein-encoder"],
    cosine_similarity=safe_cosine_similarity(enc_node_ref, enc_node_obs, mask=valid_mask),
    spearman_correlation=safe_spearman(enc_node_ref, enc_node_obs, mask=valid_mask),
    mae=mean_abs_error(enc_node_ref, enc_node_obs, mask=valid_mask),
    rmse=root_mean_square_error(enc_node_ref, enc_node_obs, mask=valid_mask),
    max_abs=max_abs_error(enc_node_ref, enc_node_obs, mask=valid_mask),
  )
  _append_scalar_metrics(
    metric_rows,
    path_id="protein-encoder",
    tier="parity_heavy",
    case=case,
    checkpoint_id=core_models.checkpoint_id,
    correlation=safe_pearson(enc_edge_ref, enc_edge_obs, mask=valid_mask),
    correlation_threshold=correlation_thresholds["protein-encoder"],
    cosine_similarity=safe_cosine_similarity(enc_edge_ref, enc_edge_obs, mask=valid_mask),
    spearman_correlation=safe_spearman(enc_edge_ref, enc_edge_obs, mask=valid_mask),
    mae=mean_abs_error(enc_edge_ref, enc_edge_obs, mask=valid_mask),
    rmse=root_mean_square_error(enc_edge_ref, enc_edge_obs, mask=valid_mask),
    max_abs=max_abs_error(enc_edge_ref, enc_edge_obs, mask=valid_mask),
  )

  uncond_ref, uncond_obs = _core_unconditional_log_probs(core_models, core_inputs)
  _append_scalar_metrics(
    metric_rows,
    path_id="decoder-unconditional",
    tier="parity_heavy",
    case=case,
    checkpoint_id=core_models.checkpoint_id,
    correlation=safe_pearson(uncond_ref, uncond_obs, mask=valid_mask),
    correlation_threshold=correlation_thresholds["decoder-unconditional"],
    cosine_similarity=safe_cosine_similarity(uncond_ref, uncond_obs, mask=valid_mask),
    spearman_correlation=safe_spearman(uncond_ref, uncond_obs, mask=valid_mask),
    mae=mean_abs_error(uncond_ref, uncond_obs, mask=valid_mask),
    rmse=root_mean_square_error(uncond_ref, uncond_obs, mask=valid_mask),
    max_abs=max_abs_error(uncond_ref, uncond_obs, mask=valid_mask),
    kl=mean_kl_divergence(uncond_ref, uncond_obs, mask=valid_mask),
  )
  _append_point_records(
    point_rows,
    path_id="decoder-unconditional",
    tier="parity_heavy",
    case=case,
    reference_values=uncond_ref,
    observed_values=uncond_obs,
    max_points=point_sample_size,
  )

  cond_ref, cond_obs = _core_conditional_log_probs(core_models, core_inputs)
  _append_scalar_metrics(
    metric_rows,
    path_id="decoder-conditional-scoring",
    tier="parity_heavy",
    case=case,
    checkpoint_id=core_models.checkpoint_id,
    correlation=safe_pearson(cond_ref, cond_obs, mask=valid_mask),
    correlation_threshold=correlation_thresholds["decoder-conditional-scoring"],
    cosine_similarity=safe_cosine_similarity(cond_ref, cond_obs, mask=valid_mask),
    spearman_correlation=safe_spearman(cond_ref, cond_obs, mask=valid_mask),
    mae=mean_abs_error(cond_ref, cond_obs, mask=valid_mask),
    rmse=root_mean_square_error(cond_ref, cond_obs, mask=valid_mask),
    max_abs=max_abs_error(cond_ref, cond_obs, mask=valid_mask),
    kl=mean_kl_divergence(cond_ref, cond_obs, mask=valid_mask),
  )
  _append_point_records(
    point_rows,
    path_id="decoder-conditional-scoring",
    tier="parity_heavy",
    case=case,
    reference_values=cond_ref,
    observed_values=cond_obs,
    max_points=point_sample_size,
  )

  pt_tokens, jax_tokens, pt_log_probs, jax_log_probs = _core_autoregressive_outputs(
    core_models,
    core_inputs,
    sample_seed=case.seed,
  )
  _append_scalar_metrics(
    metric_rows,
    path_id="autoregressive-sampling",
    tier="parity_heavy",
    case=case,
    checkpoint_id=core_models.checkpoint_id,
    correlation=safe_pearson(pt_log_probs, jax_log_probs, mask=valid_mask),
    correlation_threshold=correlation_thresholds["autoregressive-sampling"],
    cosine_similarity=safe_cosine_similarity(pt_log_probs, jax_log_probs, mask=valid_mask),
    spearman_correlation=safe_spearman(pt_log_probs, jax_log_probs, mask=valid_mask),
    mae=mean_abs_error(pt_log_probs, jax_log_probs, mask=valid_mask),
    rmse=root_mean_square_error(pt_log_probs, jax_log_probs, mask=valid_mask),
    max_abs=max_abs_error(pt_log_probs, jax_log_probs, mask=valid_mask),
    token_acc=token_agreement(pt_tokens, jax_tokens, valid_mask),
    token_threshold=correlation_thresholds["autoregressive-sampling"],
    kl=mean_kl_divergence(pt_log_probs, jax_log_probs, mask=valid_mask),
  )

  core_tied_lanes = tuple(lane for lane in tied_lanes if lane.path_id == "tied-positions-and-multi-state")
  ligand_tied_lanes = tuple(
    lane for lane in tied_lanes if lane.path_id == "ligand-tied-positions-and-multi-state"
  )
  if not core_tied_lanes:
    msg = "No core tied/multistate lanes found for path 'tied-positions-and-multi-state'."
    raise RuntimeError(msg)
  primary_core_tied_lane = next(lane for lane in core_tied_lanes if lane.is_primary)
  tied_threshold = correlation_thresholds["tied-positions-and-multi-state"]
  for lane in core_tied_lanes:
    if lane.comparison_api == "sampling":
      tie_pt_tokens, tie_jax_tokens, tie_pt_log_probs, tie_jax_log_probs, tie_groups = (
        _core_tied_sampling_outputs(
          core_models,
          core_inputs,
          sample_seed=case.seed + 13,
          lane=lane,
        )
      )
    elif lane.comparison_api == "scoring":
      tie_pt_tokens, tie_jax_tokens, tie_pt_log_probs, tie_jax_log_probs, tie_groups = (
        _core_tied_scoring_outputs(
          core_models,
          core_inputs,
          lane=lane,
        )
      )
    else:
      msg = f"{lane.condition}: unsupported tied comparison_api {lane.comparison_api!r}."
      raise ValueError(msg)

    group_consistent_pt = float(
      all(np.all(tie_pt_tokens[group] == tie_pt_tokens[group[0]]) for group in tie_groups),
    )
    group_consistent_jax = float(
      all(np.all(tie_jax_tokens[group] == tie_jax_tokens[group[0]]) for group in tie_groups),
    )
    token_acc = (
      token_agreement(tie_pt_tokens, tie_jax_tokens, valid_mask)
      if lane.token_comparison_enabled
      else None
    )
    _append_scalar_metrics(
      metric_rows,
      path_id=lane.path_id,
      tier="parity_heavy",
      case=case,
      checkpoint_id=core_models.checkpoint_id,
      correlation=safe_pearson(tie_pt_log_probs, tie_jax_log_probs, mask=valid_mask),
      correlation_threshold=tied_threshold,
      cosine_similarity=safe_cosine_similarity(tie_pt_log_probs, tie_jax_log_probs, mask=valid_mask),
      spearman_correlation=safe_spearman(tie_pt_log_probs, tie_jax_log_probs, mask=valid_mask),
      mae=mean_abs_error(tie_pt_log_probs, tie_jax_log_probs, mask=valid_mask),
      rmse=root_mean_square_error(tie_pt_log_probs, tie_jax_log_probs, mask=valid_mask),
      max_abs=max_abs_error(tie_pt_log_probs, tie_jax_log_probs, mask=valid_mask),
      token_acc=token_acc,
      token_threshold=tied_threshold if lane.token_comparison_enabled else None,
      condition=lane.condition,
      note=lane.note,
    )
    group_consistency = min(group_consistent_pt, group_consistent_jax)
    metric_rows.append(
      EvidenceMetricRecord(
        path_id=lane.path_id,
        tier="parity_heavy",
        case_id=case.id,
        case_kind=case.kind,
        backbone_id=case.id if case.kind == "real_backbone" else "synthetic",
        seed=case.seed,
        sequence_length=int(case.sequence.shape[0]),
        checkpoint_id=core_models.checkpoint_id,
        metric_name="group_consistency",
        metric_value=group_consistency,
        threshold=1.0,
        passed=group_consistency >= 1.0,
        condition=lane.condition,
        note=lane.note,
      ),
    )

  y, y_t, y_m = _build_ligand_inputs(
    case,
    atom_context_num=runtime_config.atom_context_num,
    keep_probability=runtime_config.keep_probability,
  )
  ligand_feature_ref, ligand_feature_obs = _ligand_feature_outputs(ligand_models, core_inputs, y, y_t, y_m)
  _append_scalar_metrics(
    metric_rows,
    path_id="ligand-feature-extraction",
    tier="parity_heavy",
    case=case,
    checkpoint_id=ligand_models.checkpoint_id,
    correlation=safe_pearson(ligand_feature_ref, ligand_feature_obs, mask=valid_mask),
    cosine_similarity=safe_cosine_similarity(ligand_feature_ref, ligand_feature_obs, mask=valid_mask),
    spearman_correlation=safe_spearman(ligand_feature_ref, ligand_feature_obs, mask=valid_mask),
    mae=mean_abs_error(ligand_feature_ref, ligand_feature_obs, mask=valid_mask),
    rmse=root_mean_square_error(ligand_feature_ref, ligand_feature_obs, mask=valid_mask),
    max_abs=max_abs_error(ligand_feature_ref, ligand_feature_obs, mask=valid_mask),
    allclose_pass=bool(np.allclose(ligand_feature_ref, ligand_feature_obs, atol=1e-5, rtol=1e-5)),
  )
  _append_point_records(
    point_rows,
    path_id="ligand-feature-extraction",
    tier="parity_heavy",
    case=case,
    reference_values=ligand_feature_ref,
    observed_values=ligand_feature_obs,
    max_points=point_sample_size,
  )

  (
    ligand_hv_ref,
    ligand_hv_obs,
    ligand_he_ref,
    ligand_he_obs,
    ligand_cond_ref,
    ligand_cond_obs,
  ) = _ligand_context_outputs(
    ligand_models,
    core_inputs,
    y,
    y_t,
    y_m,
  )
  _append_scalar_metrics(
    metric_rows,
    path_id="ligand-conditioning-context",
    tier="parity_heavy",
    case=case,
    checkpoint_id=ligand_models.checkpoint_id,
    correlation=safe_pearson(ligand_hv_ref, ligand_hv_obs, mask=valid_mask),
    correlation_threshold=correlation_thresholds["ligand-conditioning-context"],
    cosine_similarity=safe_cosine_similarity(ligand_hv_ref, ligand_hv_obs, mask=valid_mask),
    spearman_correlation=safe_spearman(ligand_hv_ref, ligand_hv_obs, mask=valid_mask),
  )
  _append_scalar_metrics(
    metric_rows,
    path_id="ligand-conditioning-context",
    tier="parity_heavy",
    case=case,
    checkpoint_id=ligand_models.checkpoint_id,
    correlation=safe_pearson(ligand_he_ref, ligand_he_obs, mask=valid_mask),
    correlation_threshold=correlation_thresholds["ligand-conditioning-context"],
    cosine_similarity=safe_cosine_similarity(ligand_he_ref, ligand_he_obs, mask=valid_mask),
    spearman_correlation=safe_spearman(ligand_he_ref, ligand_he_obs, mask=valid_mask),
  )
  _append_scalar_metrics(
    metric_rows,
    path_id="ligand-conditioning-context",
    tier="parity_heavy",
    case=case,
    checkpoint_id=ligand_models.checkpoint_id,
    correlation=safe_pearson(ligand_cond_ref, ligand_cond_obs, mask=valid_mask),
    correlation_threshold=correlation_thresholds["ligand-conditioning-context"],
    cosine_similarity=safe_cosine_similarity(ligand_cond_ref, ligand_cond_obs, mask=valid_mask),
    spearman_correlation=safe_spearman(ligand_cond_ref, ligand_cond_obs, mask=valid_mask),
    mae=mean_abs_error(ligand_cond_ref, ligand_cond_obs, mask=valid_mask),
    rmse=root_mean_square_error(ligand_cond_ref, ligand_cond_obs, mask=valid_mask),
    kl=mean_kl_divergence(ligand_cond_ref, ligand_cond_obs, mask=valid_mask),
  )
  _append_point_records(
    point_rows,
    path_id="ligand-conditioning-context",
    tier="parity_heavy",
    case=case,
    reference_values=ligand_cond_ref,
    observed_values=ligand_cond_obs,
    max_points=point_sample_size,
  )

  lig_pt_tokens, lig_jax_tokens, lig_pt_log_probs, lig_jax_log_probs = _ligand_autoregressive_outputs(
    ligand_models,
    core_inputs,
    y,
    y_t,
    y_m,
    sample_seed=case.seed,
  )
  _append_scalar_metrics(
    metric_rows,
    path_id="ligand-autoregressive",
    tier="parity_heavy",
    case=case,
    checkpoint_id=ligand_models.checkpoint_id,
    correlation=safe_pearson(lig_pt_log_probs, lig_jax_log_probs, mask=valid_mask),
    correlation_threshold=correlation_thresholds["ligand-autoregressive"],
    cosine_similarity=safe_cosine_similarity(lig_pt_log_probs, lig_jax_log_probs, mask=valid_mask),
    spearman_correlation=safe_spearman(lig_pt_log_probs, lig_jax_log_probs, mask=valid_mask),
    mae=mean_abs_error(lig_pt_log_probs, lig_jax_log_probs, mask=valid_mask),
    rmse=root_mean_square_error(lig_pt_log_probs, lig_jax_log_probs, mask=valid_mask),
    max_abs=max_abs_error(lig_pt_log_probs, lig_jax_log_probs, mask=valid_mask),
    token_acc=token_agreement(lig_pt_tokens, lig_jax_tokens, valid_mask),
    token_threshold=correlation_thresholds["ligand-autoregressive"],
    kl=mean_kl_divergence(lig_pt_log_probs, lig_jax_log_probs, mask=valid_mask),
  )

  xyz_37 = np.asarray(core_inputs.atom37_coordinates, dtype=np.float32)
  xyz_37_m = _build_xyz37_mask(xyz_37, core_inputs.mask)
  has_sidechain_atoms = _has_sidechain_atoms(xyz_37_m)
  sidechain_chain_mask = np.zeros_like(core_inputs.mask, dtype=np.float32)
  sidechain_y_m = np.zeros_like(y_m, dtype=np.float32)
  for lane in ligand_tied_lanes:
    lane_threshold = correlation_thresholds[lane.path_id]
    current_models = ligand_models
    current_y_m = y_m
    current_xyz_37: np.ndarray | None = None
    current_xyz_37_m: np.ndarray | None = None
    current_chain_mask: np.ndarray | None = None
    if lane.input_context == "side_chain_conditioned":
      if ligand_sidechain_models is None:
        reason = (
          sidechain_model_error
          if sidechain_model_error is not None
          else "Requested side-chain-conditioned lane but side-chain model is unavailable."
        )
        metric_rows.append(
          EvidenceMetricRecord(
            path_id=lane.path_id,
            tier="parity_heavy",
            case_id=case.id,
            case_kind=case.kind,
            backbone_id=case.id if case.kind == "real_backbone" else "synthetic",
            seed=case.seed,
            sequence_length=int(case.sequence.shape[0]),
            checkpoint_id=ligand_models.checkpoint_id,
            metric_name="lane_infeasible",
            metric_value=1.0,
            metric_group="coverage",
            condition=lane.condition,
            note=reason,
          ),
        )
        continue
      if not has_sidechain_atoms:
        metric_rows.append(
          EvidenceMetricRecord(
            path_id=lane.path_id,
            tier="parity_heavy",
            case_id=case.id,
            case_kind=case.kind,
            backbone_id=case.id if case.kind == "real_backbone" else "synthetic",
            seed=case.seed,
            sequence_length=int(case.sequence.shape[0]),
            checkpoint_id=ligand_sidechain_models.checkpoint_id,
            metric_name="lane_infeasible",
            metric_value=1.0,
            metric_group="coverage",
            condition=lane.condition,
            note="Case has no side-chain atoms available.",
          ),
        )
        continue
      current_models = ligand_sidechain_models
      current_y_m = sidechain_y_m
      current_xyz_37 = xyz_37
      current_xyz_37_m = xyz_37_m
      current_chain_mask = sidechain_chain_mask

    if lane.comparison_api == "sampling":
      (
        lig_tie_pt_tokens,
        lig_tie_jax_tokens,
        lig_tie_pt_log_probs,
        lig_tie_jax_log_probs,
        lig_tie_groups,
      ) = _ligand_tied_sampling_outputs(
        current_models,
        core_inputs,
        y,
        y_t,
        current_y_m,
        sample_seed=case.seed + 23,
        lane=lane,
        xyz_37=current_xyz_37,
        xyz_37_m=current_xyz_37_m,
        chain_mask=current_chain_mask,
      )
    elif lane.comparison_api == "scoring":
      (
        lig_tie_pt_tokens,
        lig_tie_jax_tokens,
        lig_tie_pt_log_probs,
        lig_tie_jax_log_probs,
        lig_tie_groups,
      ) = _ligand_tied_scoring_outputs(
        current_models,
        core_inputs,
        y,
        y_t,
        current_y_m,
        lane=lane,
        xyz_37=current_xyz_37,
        xyz_37_m=current_xyz_37_m,
        chain_mask=current_chain_mask,
      )
    else:
      msg = f"{lane.path_id}/{lane.condition}: unsupported comparison_api {lane.comparison_api!r}."
      raise ValueError(msg)

    lig_group_consistent_pt = float(
      all(np.all(lig_tie_pt_tokens[group] == lig_tie_pt_tokens[group[0]]) for group in lig_tie_groups),
    )
    lig_group_consistent_jax = float(
      all(np.all(lig_tie_jax_tokens[group] == lig_tie_jax_tokens[group[0]]) for group in lig_tie_groups),
    )
    lig_token_acc = (
      token_agreement(lig_tie_pt_tokens, lig_tie_jax_tokens, core_inputs.mask > 0.5)
      if lane.token_comparison_enabled
      else None
    )
    _append_scalar_metrics(
      metric_rows,
      path_id=lane.path_id,
      tier="parity_heavy",
      case=case,
      checkpoint_id=current_models.checkpoint_id,
      correlation=safe_pearson(lig_tie_pt_log_probs, lig_tie_jax_log_probs, mask=valid_mask),
      correlation_threshold=lane_threshold,
      cosine_similarity=safe_cosine_similarity(lig_tie_pt_log_probs, lig_tie_jax_log_probs, mask=valid_mask),
      spearman_correlation=safe_spearman(lig_tie_pt_log_probs, lig_tie_jax_log_probs, mask=valid_mask),
      mae=mean_abs_error(lig_tie_pt_log_probs, lig_tie_jax_log_probs, mask=valid_mask),
      rmse=root_mean_square_error(lig_tie_pt_log_probs, lig_tie_jax_log_probs, mask=valid_mask),
      max_abs=max_abs_error(lig_tie_pt_log_probs, lig_tie_jax_log_probs, mask=valid_mask),
      token_acc=lig_token_acc,
      token_threshold=lane_threshold if lane.token_comparison_enabled else None,
      condition=lane.condition,
      note=lane.note,
    )
    lig_group_consistency = min(lig_group_consistent_pt, lig_group_consistent_jax)
    metric_rows.append(
      EvidenceMetricRecord(
        path_id=lane.path_id,
        tier="parity_heavy",
        case_id=case.id,
        case_kind=case.kind,
        backbone_id=case.id if case.kind == "real_backbone" else "synthetic",
        seed=case.seed,
        sequence_length=int(case.sequence.shape[0]),
        checkpoint_id=current_models.checkpoint_id,
        metric_name="group_consistency",
        metric_value=lig_group_consistency,
        threshold=1.0,
        passed=lig_group_consistency >= 1.0,
        condition=lane.condition,
        note=lane.note,
      ),
    )

  packer_ref, packer_obs = _packer_outputs(
    packer_models,
    case,
    runtime_config.atom_context_num,
  )
  _append_scalar_metrics(
    metric_rows,
    path_id="side-chain-packer",
    tier="parity_heavy",
    case=case,
    checkpoint_id=packer_models.checkpoint_id,
    correlation=safe_pearson(packer_ref, packer_obs),
    correlation_threshold=correlation_thresholds["side-chain-packer"],
    mae=mean_abs_error(packer_ref, packer_obs),
    rmse=root_mean_square_error(packer_ref, packer_obs),
    max_abs=max_abs_error(packer_ref, packer_obs),
    allclose_pass=bool(np.allclose(packer_ref, packer_obs, atol=1e-3, rtol=1e-4)),
  )
  _append_point_records(
    point_rows,
    path_id="side-chain-packer",
    tier="parity_heavy",
    case=case,
    reference_values=packer_ref,
    observed_values=packer_obs,
    max_points=point_sample_size,
  )

  primary_ligand_tied_lane = next((lane for lane in ligand_tied_lanes if lane.is_primary), None)

  if runtime_config.intrinsic_repeats > 1:
    if primary_core_tied_lane.comparison_api != "sampling":
      msg = (
        "Intrinsic tied baseline metrics require the primary tied lane to use the sampling API. "
        f"Got comparison_api={primary_core_tied_lane.comparison_api!r} for "
        f"{primary_core_tied_lane.condition!r}."
      )
      raise ValueError(msg)

    core_reference_runs: list[np.ndarray] = []
    core_observed_runs: list[np.ndarray] = []
    tie_reference_runs: list[np.ndarray] = []
    tie_observed_runs: list[np.ndarray] = []
    ligand_reference_runs: list[np.ndarray] = []
    ligand_observed_runs: list[np.ndarray] = []
    ligand_tied_reference_runs: list[np.ndarray] = []
    ligand_tied_observed_runs: list[np.ndarray] = []
    for repeat_index in range(runtime_config.intrinsic_repeats):
      repeat_seed = case.seed + repeat_index * runtime_config.intrinsic_seed_step
      _, _, pt_lp_rep, jax_lp_rep = _core_autoregressive_outputs(
        core_models,
        core_inputs,
        sample_seed=repeat_seed,
      )
      core_reference_runs.append(pt_lp_rep)
      core_observed_runs.append(jax_lp_rep)

      _, _, tie_pt_lp_rep, tie_jax_lp_rep, _ = _core_tied_sampling_outputs(
        core_models,
        core_inputs,
        sample_seed=repeat_seed + 13,
        lane=primary_core_tied_lane,
      )
      tie_reference_runs.append(tie_pt_lp_rep)
      tie_observed_runs.append(tie_jax_lp_rep)

      _, _, lig_pt_lp_rep, lig_jax_lp_rep = _ligand_autoregressive_outputs(
        ligand_models,
        core_inputs,
        y,
        y_t,
        y_m,
        sample_seed=repeat_seed,
      )
      ligand_reference_runs.append(lig_pt_lp_rep)
      ligand_observed_runs.append(lig_jax_lp_rep)

      if (
        primary_ligand_tied_lane is not None
        and primary_ligand_tied_lane.comparison_api == "sampling"
        and primary_ligand_tied_lane.input_context == "ligand_context"
      ):
        _, _, lig_tie_pt_lp_rep, lig_tie_jax_lp_rep, _ = _ligand_tied_sampling_outputs(
          ligand_models,
          core_inputs,
          y,
          y_t,
          y_m,
          sample_seed=repeat_seed + 27,
          lane=primary_ligand_tied_lane,
        )
        ligand_tied_reference_runs.append(lig_tie_pt_lp_rep)
        ligand_tied_observed_runs.append(lig_tie_jax_lp_rep)

    _append_intrinsic_baseline_metrics(
      metric_rows,
      path_id="autoregressive-sampling",
      tier="parity_heavy",
      case=case,
      checkpoint_id=core_models.checkpoint_id,
      reference_runs=core_reference_runs,
      observed_runs=core_observed_runs,
    )
    _append_intrinsic_baseline_metrics(
      metric_rows,
      path_id="tied-positions-and-multi-state",
      tier="parity_heavy",
      case=case,
      checkpoint_id=core_models.checkpoint_id,
      reference_runs=tie_reference_runs,
      observed_runs=tie_observed_runs,
      condition=primary_core_tied_lane.condition,
      note=primary_core_tied_lane.note,
    )
    _append_intrinsic_baseline_metrics(
      metric_rows,
      path_id="ligand-autoregressive",
      tier="parity_heavy",
      case=case,
      checkpoint_id=ligand_models.checkpoint_id,
      reference_runs=ligand_reference_runs,
      observed_runs=ligand_observed_runs,
    )
    if ligand_tied_reference_runs and ligand_tied_observed_runs and primary_ligand_tied_lane is not None:
      _append_intrinsic_baseline_metrics(
        metric_rows,
        path_id=primary_ligand_tied_lane.path_id,
        tier="parity_heavy",
        case=case,
        checkpoint_id=ligand_models.checkpoint_id,
        reference_runs=ligand_tied_reference_runs,
        observed_runs=ligand_tied_observed_runs,
        condition=primary_ligand_tied_lane.condition,
        note=primary_ligand_tied_lane.note,
      )

  if runtime_config.macro_include_no_ligand:
    (
      identity_ref,
      identity_obs,
      entropy_ref,
      entropy_obs,
      composition_ref,
      composition_obs,
    ) = _collect_no_ligand_macro_samples(
      case,
      core_models,
      core_inputs,
      num_samples=runtime_config.macro_samples_per_case,
    )
    _append_macro_distribution_metrics(
      metric_rows,
      point_rows,
      path_id="autoregressive-sampling",
      tier="parity_heavy",
      case=case,
      checkpoint_id=core_models.checkpoint_id,
      condition="no_ligand",
      identity_reference=identity_ref,
      identity_observed=identity_obs,
      entropy_reference=entropy_ref,
      entropy_observed=entropy_obs,
      composition_reference=composition_ref,
      composition_observed=composition_obs,
    )

  if runtime_config.macro_include_ligand:
    (
      ligand_identity_ref,
      ligand_identity_obs,
      ligand_entropy_ref,
      ligand_entropy_obs,
      ligand_composition_ref,
      ligand_composition_obs,
    ) = _collect_ligand_macro_samples(
      case,
      ligand_models,
      core_inputs,
      y,
      y_t,
      y_m,
      num_samples=runtime_config.macro_samples_per_case,
    )
    _append_macro_distribution_metrics(
      metric_rows,
      point_rows,
      path_id="ligand-autoregressive",
      tier="parity_heavy",
      case=case,
      checkpoint_id=ligand_models.checkpoint_id,
      condition="ligand_context",
      identity_reference=ligand_identity_ref,
      identity_observed=ligand_identity_obs,
      entropy_reference=ligand_entropy_ref,
      entropy_observed=ligand_entropy_obs,
      composition_reference=ligand_composition_ref,
      composition_observed=ligand_composition_obs,
    )

  sidechain_macro_metrics: dict[str, float] | None = None
  sidechain_infeasible_reason: str | None = None
  sidechain_checkpoint_id = (
    ligand_sidechain_models.checkpoint_id
    if ligand_sidechain_models is not None
    else ligand_models.checkpoint_id
  )
  if runtime_config.macro_include_sidechain_conditioned:
    if ligand_sidechain_models is None:
      sidechain_infeasible_reason = (
        sidechain_model_error
        if sidechain_model_error is not None
        else "Requested but side-chain-conditioned models could not be loaded."
      )
    else:
      try:
        (
          sidechain_identity_ref,
          sidechain_identity_obs,
          sidechain_entropy_ref,
          sidechain_entropy_obs,
          sidechain_composition_ref,
          sidechain_composition_obs,
        ) = _collect_sidechain_conditioned_macro_samples(
          case,
          ligand_sidechain_models,
          core_inputs,
          y,
          y_t,
          y_m,
          num_samples=runtime_config.macro_samples_per_case,
        )
      except ValueError as error:
        sidechain_infeasible_reason = (
          "Requested but side-chain-conditioned inputs were infeasible: "
          f"{error}"
        )
      except RuntimeError as error:
        sidechain_infeasible_reason = (
          "Requested but side-chain-conditioned collection failed: "
          f"{error}"
        )
      except Exception as error:  # noqa: BLE001
        sidechain_infeasible_reason = (
          "Requested but side-chain-conditioned collection failed: "
          f"{error}"
        )
      else:
        sidechain_macro_metrics = _append_macro_distribution_metrics(
          metric_rows,
          point_rows,
          path_id="ligand-autoregressive",
          tier="parity_heavy",
          case=case,
          checkpoint_id=sidechain_checkpoint_id,
          condition="side_chain_conditioned",
          identity_reference=sidechain_identity_ref,
          identity_observed=sidechain_identity_obs,
          entropy_reference=sidechain_entropy_ref,
          entropy_observed=sidechain_entropy_obs,
          composition_reference=sidechain_composition_ref,
          composition_observed=sidechain_composition_obs,
          identity_wasserstein_max=sidechain_acceptance.identity_wasserstein_max,
          entropy_wasserstein_max=sidechain_acceptance.entropy_wasserstein_max,
          composition_js_max=sidechain_acceptance.composition_js_max,
          identity_ks_pvalue_min=sidechain_acceptance.identity_ks_pvalue_min,
        )

  sidechain_status, sidechain_reason = _evaluate_sidechain_conditioned_gate(
    requested=runtime_config.macro_include_sidechain_conditioned,
    macro_metrics=sidechain_macro_metrics,
    acceptance=sidechain_acceptance,
    infeasible_reason=sidechain_infeasible_reason,
  )
  metric_rows.append(
    EvidenceMetricRecord(
      path_id="ligand-autoregressive",
      tier="parity_heavy",
      case_id=case.id,
      case_kind=case.kind,
      backbone_id=case.id if case.kind == "real_backbone" else "synthetic",
      seed=case.seed,
      sequence_length=int(case.sequence.shape[0]),
      checkpoint_id=sidechain_checkpoint_id,
      metric_name="macro_sidechain_conditioned_gate_status_code",
      metric_value=_sidechain_gate_status_code(sidechain_status),
      threshold=0.0,
      passed=sidechain_status == "pass",
      metric_group="macro",
      condition="side_chain_conditioned",
      note=sidechain_reason,
    ),
  )
  metric_rows.append(
    EvidenceMetricRecord(
      path_id="ligand-autoregressive",
      tier="parity_heavy",
      case_id=case.id,
      case_kind=case.kind,
      backbone_id=case.id if case.kind == "real_backbone" else "synthetic",
      seed=case.seed,
      sequence_length=int(case.sequence.shape[0]),
      checkpoint_id=sidechain_checkpoint_id,
      metric_name="macro_sidechain_conditioned_requested",
      metric_value=float(runtime_config.macro_include_sidechain_conditioned),
      metric_group="macro",
      condition="side_chain_conditioned",
      note=sidechain_reason,
    ),
  )
  sidechain_state_metric = (
    "macro_sidechain_conditioned_excluded"
    if not runtime_config.macro_include_sidechain_conditioned
    else "macro_sidechain_conditioned_infeasible"
    if sidechain_macro_metrics is None
    else "macro_sidechain_conditioned_evaluated"
  )
  metric_rows.append(
    EvidenceMetricRecord(
      path_id="ligand-autoregressive",
      tier="parity_heavy",
      case_id=case.id,
      case_kind=case.kind,
      backbone_id=case.id if case.kind == "real_backbone" else "synthetic",
      seed=case.seed,
      sequence_length=int(case.sequence.shape[0]),
      checkpoint_id=sidechain_checkpoint_id,
      metric_name=sidechain_state_metric,
      metric_value=1.0,
      passed=sidechain_status == "pass" if sidechain_macro_metrics is not None else None,
      metric_group="macro",
      condition="side_chain_conditioned",
      note=sidechain_reason,
    ),
  )
  return metric_rows, point_rows


def _extract_tied_multistate_lanes() -> tuple[TiedMultistateComparisonLane, ...]:
  matrix = load_parity_matrix()
  tied_paths = [path for path in matrix if path.id in _TIED_PATH_IDS]
  if not tied_paths:
    msg = "No tied/multistate paths were found in the parity matrix."
    raise RuntimeError(msg)

  valid_reference = {"weighted_sum", "arithmetic_mean", "geometric_mean"}
  valid_jax = {"arithmetic_mean", "geometric_mean", "product"}
  valid_api = {"sampling", "scoring"}
  equivalent_lane_pairs = {
    ("sampling", "weighted_sum", "product"),
    ("scoring", "weighted_sum", "product"),
    ("scoring", "arithmetic_mean", "arithmetic_mean"),
    ("scoring", "geometric_mean", "geometric_mean"),
  }
  lanes: list[TiedMultistateComparisonLane] = []
  for tied_path in tied_paths:
    lane_payloads = tied_path.acceptance.get("comparison_lanes")
    if not isinstance(lane_payloads, list) or not lane_payloads:
      msg = f"{tied_path.id}.acceptance.comparison_lanes must be a non-empty list."
      raise TypeError(msg)

    seen_conditions: set[str] = set()
    primary_count = 0
    for lane in lane_payloads:
      if not isinstance(lane, dict):
        msg = "Each tied multistate lane must be an object."
        raise TypeError(msg)
      condition = lane.get("condition")
      if not isinstance(condition, str) or not condition:
        msg = "Each tied multistate lane must define a non-empty condition string."
        raise TypeError(msg)
      if condition in seen_conditions:
        msg = f"{tied_path.id}: duplicate tied multistate condition {condition!r}."
        raise ValueError(msg)
      seen_conditions.add(condition)

      reference_combiner = lane.get("reference_combiner")
      if reference_combiner not in valid_reference:
        msg = f"{tied_path.id}/{condition}: unsupported reference_combiner {reference_combiner!r}."
        raise TypeError(msg)
      jax_strategy = lane.get("jax_multi_state_strategy")
      if jax_strategy not in valid_jax:
        msg = f"{tied_path.id}/{condition}: unsupported jax_multi_state_strategy {jax_strategy!r}."
        raise TypeError(msg)

      comparison_api = lane.get("comparison_api", "sampling")
      if comparison_api not in valid_api:
        msg = f"{tied_path.id}/{condition}: comparison_api must be one of {sorted(valid_api)}."
        raise TypeError(msg)

      token_comparison = lane.get("token_comparison")
      if token_comparison not in {"enabled", "disabled"}:
        msg = f"{tied_path.id}/{condition}: token_comparison must be 'enabled' or 'disabled'."
        raise TypeError(msg)
      if comparison_api != "sampling" and token_comparison == "enabled":
        msg = (
          f"{tied_path.id}/{condition}: token_comparison='enabled' is only valid for sampling "
          f"lanes, got comparison_api={comparison_api!r}."
        )
        raise ValueError(msg)
      is_primary = lane.get("is_primary")
      if not isinstance(is_primary, bool):
        msg = f"{tied_path.id}/{condition}: is_primary must be boolean."
        raise TypeError(msg)
      if is_primary:
        primary_count += 1

      lane_key = (comparison_api, reference_combiner, jax_strategy)
      if lane_key not in equivalent_lane_pairs:
        msg = (
          f"{tied_path.id}/{condition}: non-equivalent tied lane mapping {lane_key!r}. "
          "Allowed mappings are "
          f"{sorted(equivalent_lane_pairs)}."
        )
        raise ValueError(msg)

      input_context = lane.get("input_context", "ligand_context")
      if input_context not in {"ligand_context", "side_chain_conditioned"}:
        msg = (
          f"{tied_path.id}/{condition}: input_context must be one of "
          "['ligand_context', 'side_chain_conditioned']."
        )
        raise TypeError(msg)

      note = lane.get("note")
      lane_note = (
        str(note)
        if isinstance(note, str) and note
        else (
          "Reference and JAX tied outputs are compared with matching "
          f"{comparison_api} APIs using '{reference_combiner}' vs '{jax_strategy}'."
        )
      )
      lanes.append(
        TiedMultistateComparisonLane(
          path_id=tied_path.id,
          input_context=input_context,
          condition=condition,
          comparison_api=comparison_api,
          reference_combiner=reference_combiner,
          jax_multi_state_strategy=jax_strategy,
          token_comparison_enabled=token_comparison == "enabled",
          is_primary=is_primary,
          note=lane_note,
        ),
      )

    if primary_count != 1:
      msg = f"{tied_path.id}: exactly one tied comparison lane must set is_primary=true."
      raise ValueError(msg)
  return tuple(lanes)


def _extract_correlation_thresholds() -> dict[str, float]:
  matrix = load_parity_matrix()
  thresholds: dict[str, float] = {}
  for path in matrix:
    threshold = path.acceptance.get("correlation_min")
    if isinstance(threshold, (int, float)):
      thresholds[path.id] = float(threshold)
  return thresholds


def _extract_sidechain_macro_acceptance() -> SidechainMacroAcceptance:
  matrix = load_parity_matrix()
  ligand_autoregressive = next((path for path in matrix if path.id == "ligand-autoregressive"), None)
  if ligand_autoregressive is None:
    msg = "ligand-autoregressive path is missing from parity matrix."
    raise RuntimeError(msg)

  payload = ligand_autoregressive.acceptance.get("sidechain_conditioned_macro")
  if payload is None:
    payload = {}
  if not isinstance(payload, dict):
    msg = "ligand-autoregressive.sidechain_conditioned_macro must be a mapping."
    raise TypeError(msg)

  def _read_float(key: str, default: float) -> float:
    value = payload.get(key, default)
    if not isinstance(value, (int, float)):
      msg = f"sidechain_conditioned_macro.{key} must be numeric."
      raise TypeError(msg)
    return float(value)

  return SidechainMacroAcceptance(
    identity_wasserstein_max=_read_float("identity_wasserstein_max", 0.05),
    entropy_wasserstein_max=_read_float("entropy_wasserstein_max", 0.08),
    composition_js_max=_read_float("composition_js_max", 0.05),
    identity_ks_pvalue_min=_read_float("identity_ks_pvalue_min", 0.05),
  )


def _collect_fast_path_metrics(
  case: BackboneCase,
  *,
  core_models: CoreModels,
  point_sample_size: int,
) -> tuple[list[EvidenceMetricRecord], list[EvidencePointRecord]]:
  core_inputs = _prepare_core_case_inputs(case)
  all_metrics: list[EvidenceMetricRecord] = []
  all_points: list[EvidencePointRecord] = []

  helper_metrics, helper_points = _collect_fast_logit_helper_metrics(
    case,
    core_inputs,
    core_models,
    point_sample_size=point_sample_size,
  )
  all_metrics.extend(helper_metrics)
  all_points.extend(helper_points)

  avg_metrics, avg_points = _collect_averaged_encoding_metrics(
    case,
    core_models,
    point_sample_size=point_sample_size,
  )
  all_metrics.extend(avg_metrics)
  all_points.extend(avg_points)

  api_metrics, api_points = _collect_end_to_end_api_metrics(
    case,
    core_models,
    point_sample_size=point_sample_size,
  )
  all_metrics.extend(api_metrics)
  all_points.extend(api_points)
  return all_metrics, all_points


def _collect_checkpoint_family_audit_metrics(project_root: Path) -> list[EvidenceMetricRecord]:
  manifest_path = project_root / "tests/parity/parity_assets.json"
  if not manifest_path.exists():
    return []
  payload = json.loads(manifest_path.read_text(encoding="utf-8"))
  assets = payload.get("assets", [])
  if not isinstance(assets, list):
    return []

  expected_by_family: dict[str, int] = {}
  available_by_family: dict[str, int] = {}
  for asset in assets:
    if not isinstance(asset, dict):
      continue
    if asset.get("asset_kind") != "converted_checkpoint":
      continue
    required_for = asset.get("required_for", [])
    if not isinstance(required_for, list) or "parity_audit" not in required_for:
      continue
    family = str(asset.get("family", "unknown"))
    rel_path = asset.get("path")
    expected_by_family[family] = expected_by_family.get(family, 0) + 1
    if isinstance(rel_path, str) and (project_root / rel_path).exists():
      available_by_family[family] = available_by_family.get(family, 0) + 1

  rows: list[EvidenceMetricRecord] = []
  for family in sorted(expected_by_family):
    expected = expected_by_family[family]
    available = available_by_family.get(family, 0)
    ratio = float(available / expected) if expected else 0.0

    def add(metric_name: str, metric_value: float, *, threshold: float | None = None, passed: bool | None = None) -> None:
      rows.append(
        EvidenceMetricRecord(
          path_id="checkpoint-family-load",
          tier="parity_audit",
          case_id=family,
          case_kind="audit_family",
          backbone_id=family,
          seed=0,
          sequence_length=0,
          checkpoint_id="converted_checkpoints",
          metric_name=metric_name,
          metric_value=metric_value,
          threshold=threshold,
          passed=passed,
          metric_group="audit",
          condition=family,
        ),
      )

    add("family_expected_count", float(expected))
    add("family_available_count", float(available))
    add("family_available_ratio", ratio, threshold=1.0, passed=ratio >= 1.0)
  return rows


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--project-root", default=".", help="Repository root path.")
  parser.add_argument(
    "--reference-root",
    default=None,
    help="Path to LigandMPNN reference checkout. Defaults to REFERENCE_PATH or local clones.",
  )
  parser.add_argument(
    "--case-corpus",
    default="tests/parity/parity_case_corpus.json",
    help="Parity case corpus JSON path relative to project root.",
  )
  parser.add_argument("--output-dir", default="docs/parity/reports/evidence", help="Evidence artifact output directory.")
  parser.add_argument("--point-sample-size", type=int, default=4000, help="Maximum sampled pair points per path/case.")
  parser.add_argument("--max-cases", type=int, default=None, help="Optional maximum number of cases to process.")
  parser.add_argument(
    "--intrinsic-repeats",
    type=int,
    default=None,
    help="Override intrinsic repeat count from corpus config.",
  )
  parser.add_argument(
    "--macro-samples-per-case",
    type=int,
    default=None,
    help="Override macro sample count per case from corpus config.",
  )
  parser.add_argument(
    "--macro-sidechain-conditioned",
    choices=("inherit", "on", "off"),
    default="inherit",
    help="Override side-chain-conditioned macro lane inclusion.",
  )
  args = parser.parse_args()

  project_root = Path(args.project_root).resolve()
  reference_root = _resolve_reference_root(
    Path(args.reference_root).resolve() if args.reference_root else None,
    project_root,
  )
  corpus_path = (project_root / args.case_corpus).resolve()
  output_dir = (project_root / args.output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  cases, runtime_config = _load_case_corpus(project_root, corpus_path)
  if args.max_cases is not None:
    cases = cases[: max(1, args.max_cases)]
  if args.intrinsic_repeats is not None:
    runtime_config = replace(
      runtime_config,
      intrinsic_repeats=max(1, args.intrinsic_repeats),
    )
  if args.macro_samples_per_case is not None:
    runtime_config = replace(
      runtime_config,
      macro_samples_per_case=max(2, args.macro_samples_per_case),
    )
  if args.macro_sidechain_conditioned == "on":
    runtime_config = replace(runtime_config, macro_include_sidechain_conditioned=True)
  if args.macro_sidechain_conditioned == "off":
    runtime_config = replace(runtime_config, macro_include_sidechain_conditioned=False)
  if not cases:
    msg = "No usable cases found in parity corpus."
    raise RuntimeError(msg)

  correlation_thresholds = _extract_correlation_thresholds()
  tied_lanes = _extract_tied_multistate_lanes()
  sidechain_acceptance = _extract_sidechain_macro_acceptance()
  core_models = _load_core_models(reference_root, project_root)
  ligand_models = _load_ligand_models(reference_root, use_side_chain_context=False)
  ligand_sidechain_models: LigandModels | None = None
  sidechain_model_error: str | None = None
  needs_sidechain_lanes = any(
    lane.path_id == "ligand-tied-positions-and-multi-state"
    and lane.input_context == "side_chain_conditioned"
    for lane in tied_lanes
  )
  if runtime_config.macro_include_sidechain_conditioned or needs_sidechain_lanes:
    try:
      ligand_sidechain_models = _load_ligand_models(
        reference_root,
        use_side_chain_context=True,
      )
    except Exception as error:  # noqa: BLE001
      sidechain_model_error = str(error)
  packer_models = _load_packer_models(reference_root)

  all_metrics: list[EvidenceMetricRecord] = []
  all_points: list[EvidencePointRecord] = []
  for case in cases:
    case_metrics, case_points = _collect_for_case(
      case,
      core_models=core_models,
      ligand_models=ligand_models,
      ligand_sidechain_models=ligand_sidechain_models,
      packer_models=packer_models,
      correlation_thresholds=correlation_thresholds,
      tied_lanes=tied_lanes,
      sidechain_acceptance=sidechain_acceptance,
      sidechain_model_error=sidechain_model_error,
      runtime_config=runtime_config,
      point_sample_size=args.point_sample_size,
    )
    all_metrics.extend(case_metrics)
    all_points.extend(case_points)

  fast_case = replace(
    cases[0],
    id="parity-fast-contract",
    kind="synthetic",
    seed=cases[0].seed + 101,
  )
  fast_metrics, fast_points = _collect_fast_path_metrics(
    fast_case,
    core_models=core_models,
    point_sample_size=args.point_sample_size,
  )
  all_metrics.extend(fast_metrics)
  all_points.extend(fast_points)

  all_metrics.extend(_collect_checkpoint_family_audit_metrics(project_root))

  metrics_csv = output_dir / "evidence_metrics.csv"
  metrics_json = output_dir / "evidence_metrics.json"
  points_csv = output_dir / "evidence_points.csv"
  metadata_json = output_dir / "evidence_metadata.json"

  write_metric_records_csv(all_metrics, metrics_csv)
  write_metric_records_json(all_metrics, metrics_json)
  write_point_records_csv(all_points, points_csv)
  metadata = {
    "num_cases": len(cases),
    "case_ids": [case.id for case in cases],
    "num_metric_rows": len(all_metrics),
    "num_point_rows": len(all_points),
    "runtime_config": {
      "atom_context_num": runtime_config.atom_context_num,
      "mask_keep_probability": runtime_config.keep_probability,
      "intrinsic_repeats": runtime_config.intrinsic_repeats,
      "intrinsic_seed_step": runtime_config.intrinsic_seed_step,
      "macro_samples_per_case": runtime_config.macro_samples_per_case,
      "macro_include_no_ligand": runtime_config.macro_include_no_ligand,
      "macro_include_ligand": runtime_config.macro_include_ligand,
      "macro_include_sidechain_conditioned": runtime_config.macro_include_sidechain_conditioned,
      "tied_multistate_lanes": [
        {
          "path_id": lane.path_id,
          "input_context": lane.input_context,
          "condition": lane.condition,
          "comparison_api": lane.comparison_api,
          "reference_combiner": lane.reference_combiner,
          "jax_multi_state_strategy": lane.jax_multi_state_strategy,
          "token_comparison_enabled": lane.token_comparison_enabled,
          "is_primary": lane.is_primary,
        }
        for lane in tied_lanes
      ],
    },
    "overrides": {
      "max_cases": args.max_cases,
      "intrinsic_repeats": args.intrinsic_repeats,
      "macro_samples_per_case": args.macro_samples_per_case,
      "macro_sidechain_conditioned": args.macro_sidechain_conditioned,
    },
    "output_files": {
      "metrics_csv": str(metrics_csv.relative_to(project_root)),
      "metrics_json": str(metrics_json.relative_to(project_root)),
      "points_csv": str(points_csv.relative_to(project_root)),
    },
  }
  metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(f"Wrote {metrics_csv}")
  print(f"Wrote {metrics_json}")
  print(f"Wrote {points_csv}")
  print(f"Wrote {metadata_json}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
