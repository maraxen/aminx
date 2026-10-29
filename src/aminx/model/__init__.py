"""Public API for aminx.model.

Public contract (everything in __all__):
- Model classes: Aminx, PrxteinLigandMPNN, DiffusionAminx, Packer
- Encoder/decoder layers: Encoder, EncoderLayer, Decoder, DecoderLayer
- Feature extraction: ProteinFeatures
- Antibody CDR utilities: CDRRegionMasks, build_cdr_region_mask, CDRFrequencyLogitTransform,
  make_cdr_frequency_logit_transform, CDRPhysicochemicalDescriptors,
  compute_cdr_physicochemical_descriptors
- Capability introspection: ModelCapabilities, PRXTEIN_MPNN_CAPABILITIES, PRXTEIN_LIGAND_MPNN_CAPABILITIES

Internal: files in model/_inference/ are implementation details.
Importing from aminx.model._inference.* outside of model/ is unsupported.
"""

from __future__ import annotations

from .capabilities import (
  PRXTEIN_LIGAND_MPNN_CAPABILITIES,
  PRXTEIN_MPNN_CAPABILITIES,
  ModelCapabilities,
)
from .cdr_bias import (
  CDRFrequencyLogitTransform,
  CDRPhysicochemicalDescriptors,
  CDRRegionMasks,
  build_cdr_region_mask,
  compute_cdr_physicochemical_descriptors,
  make_cdr_frequency_logit_transform,
)
from .decoder import Decoder, DecoderLayer
from .diffusion_mpnn import DiffusionAminx
from .encoder import Encoder, EncoderLayer
from .features import ProteinFeatures
from .ligand_mpnn import PrxteinLigandMPNN
from .mpnn import Aminx
from .packer import Packer

__all__ = [
  "PRXTEIN_LIGAND_MPNN_CAPABILITIES",
  "PRXTEIN_MPNN_CAPABILITIES",
  "Aminx",
  "CDRFrequencyLogitTransform",
  "CDRPhysicochemicalDescriptors",
  "CDRRegionMasks",
  "Decoder",
  "DecoderLayer",
  "DiffusionAminx",
  "Encoder",
  "EncoderLayer",
  "ModelCapabilities",
  "Packer",
  "ProteinFeatures",
  "PrxteinLigandMPNN",
  "build_cdr_region_mask",
  "compute_cdr_physicochemical_descriptors",
  "make_cdr_frequency_logit_transform",
]
