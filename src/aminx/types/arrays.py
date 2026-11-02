"""Array type definitions for the Aminx project."""

# TODO(tech-debt): `.praxia/TECHNICAL_DEBT.md` §10 — tighten contracts (`Protocol`, aliases) at model/sampling boundaries.

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree

if TYPE_CHECKING:
  from aminx.model.ligand_mpnn import PrxteinLigandMPNN
  from aminx.model.mpnn import Aminx

ArrayLike = Union[Array, np.ndarray]

NodeFeatures = Float[ArrayLike, "num_atoms num_features"]  # Node features
Scalar = Int[ArrayLike, ""]
ScalarFloat = Float[ArrayLike, ""]
EdgeFeatures = Float[ArrayLike, "num_atoms num_neighbors num_features"]  # Edge features
Message = Float[ArrayLike, "num_atoms num_neighbors num_features"]  # Message passing features
AtomicCoordinate = Float[ArrayLike, "3"]  # Atomic coordinates (x, y, z)
NeighborIndices = Int[ArrayLike, "num_atoms num_neighbors"]  # Indices of neighboring nodes
BackboneCoordinates = Float[ArrayLike, "4 3"]  # Residue coordinates (x, y, z)
StructureAtomicCoordinates = Float[
  ArrayLike,
  "num_residues num_atoms 3",
]  # Atomic coordinates of the structure
AtomMask = Int[ArrayLike, "num_residues num_atoms"]  # Masks for atoms in the structure
AtomResidueIndex = Int[ArrayLike, "num_residues num_atoms"]  # Residue indices for atoms
AtomChainIndex = Int[ArrayLike, "num_residues num_atoms"]  # Chain indices for atoms
Parameters = Float[ArrayLike, "num_parameters"]  # Model parameters
ModelParameters = PyTree[str, "P"]
# Type union for migration: supports both legacy PyTree and new Equinox model
# Using Union with string annotation to avoid runtime import
Model = Union["Aminx", "PrxteinLigandMPNN", ModelParameters]
AlphaCarbonDistance = Float[
  ArrayLike,
  "num_atoms num_atoms",
]  # Distances between alpha carbon atoms
Distances = Float[ArrayLike, "num_atoms num_neighbors"]  # Distances between nodes
AtomIndexPair = Int[ArrayLike, "2"]  # Pairs of atom indices for edges
AttentionMask = Bool[ArrayLike, "num_atoms num_atoms"]  # Attention mask for nodes
Logits = Float[ArrayLike, "num_residues num_classes"]  # Logits for classification
DecodingOrder = Int[ArrayLike, "num_residues"]  # Order of residues for autoregressive decoding
ProteinSequence = Int[ArrayLike, "num_residues"]  # Sequence of residues
OneHotProteinSequence = Float[
  ArrayLike,
  "num_residues num_classes",
]  # One-hot encoded protein sequence
NodeEdgeFeatures = Float[
  ArrayLike,
  "num_atoms num_neighbors num_features",
]  # Combined node and edge features
SequenceEdgeFeatures = Float[
  ArrayLike,
  "num_residues num_neighbors num_features",
]  # Sequence edge features
AutoRegressiveMask = Bool[
  ArrayLike,
  "num_residues num_residues",
]  # Mask for autoregressive decoding
InputBias = Float[ArrayLike, "num_residues num_classes"]  # Input bias for classification
InputLengths = Int[ArrayLike, "num_sequences"]  # Lengths of input sequences
BFactors = Float[ArrayLike, "num_residues num_atom_types"]  # B-factors for residues
ResidueIndex = Int[ArrayLike, "num_residues"]  # Index of residues in the structure
ChainIndex = Int[ArrayLike, "num_residues"]  # Index of chains in the structure

DecodingOrderInputs = tuple[PRNGKeyArray, int]
DecodingOrderOutputs = tuple[DecodingOrder, PRNGKeyArray]
CEELoss = Float[ArrayLike, ""]  # Cross-entropy loss

AlphaCarbonMask = Int[ArrayLike, "num_residues"]
BackboneDihedrals = Float[ArrayLike, "num_residues 3"]  # Dihedral angles for backbone atoms
BackboneNoise = Float[ArrayLike, "n"]  # Noise added to backbone coordinates
BackboneAtomCoordinates = Float[ArrayLike, "num_residues 4 3"]  # Backbone atom coordinates
GroupMask = Bool[ArrayLike, "num_residues"]
LinkMask = Float[ArrayLike, "num_residues num_neighbors"]
TieGroupMap = Int[ArrayLike, "num_residues"]

Temperature = Float[ArrayLike, ""]  # Temperature for sampling
CategoricalJacobian = Float[ArrayLike, "num_residues num_classes num_residues num_classes"]
InterproteinMapping = Int[ArrayLike, "num_pairs max_length 2"]  # Mapping between protein pairs


class TrainingMetrics(dict):
  """Dictionary containing training metrics."""

  loss: float
  accuracy: float
  perplexity: float
  learning_rate: float


__all__ = [
  "AlphaCarbonDistance",
  "AlphaCarbonMask",
  "ArrayLike",
  "AtomChainIndex",
  "AtomIndexPair",
  "AtomMask",
  "AtomResidueIndex",
  "AtomicCoordinate",
  "AttentionMask",
  "AutoRegressiveMask",
  "BFactors",
  "BackboneAtomCoordinates",
  "BackboneCoordinates",
  "BackboneDihedrals",
  "BackboneNoise",
  "CEELoss",
  "CategoricalJacobian",
  "ChainIndex",
  "DecodingOrder",
  "DecodingOrderInputs",
  "DecodingOrderOutputs",
  "Distances",
  "EdgeFeatures",
  "GroupMask",
  "InputBias",
  "InputLengths",
  "InterproteinMapping",
  "LinkMask",
  "Logits",
  "Message",
  "Model",
  "ModelParameters",
  "NeighborIndices",
  "NodeEdgeFeatures",
  "NodeFeatures",
  "OneHotProteinSequence",
  "Parameters",
  "ProteinSequence",
  "ResidueIndex",
  "Scalar",
  "ScalarFloat",
  "SequenceEdgeFeatures",
  "StructureAtomicCoordinates",
  "Temperature",
  "TieGroupMap",
  "TrainingMetrics",
]
