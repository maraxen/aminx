"""Flat option dataclasses for driver-family knobs (PottsMPNN and LASErMPNN)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PottsMPNNOptions:
  """PottsMPNN inference knobs. ``None`` on the spec means family defaults."""

  optimization_mode: Literal["none", "potts", "potts_converge", "nodes"] = "potts"
  optimization_temperature: float = 0.0
  binding_energy_optimization: Literal["none", "both", "only"] = "none"
  binding_energy_json: str | None = None
  binding_energy_cutoff: float = 8.0
  mean_norm: bool = False
  filter_nan: bool = False
  mutant_fasta: str | None = None
  mutant_csv: str | None = None
  exclude_chains: str | None = None
  pssm_json: str | None = None
  pssm_threshold: float = 0.0
  pssm_multi: float = 0.0
  pssm_log_odds_flag: bool = False
  pssm_bias_flag: bool = False
  bias_by_res_json: str | None = None
  tied_beta: float | None = None
  tied_epistasis: bool = False
  skip_gaps: bool = False
  optimize_pdb: bool = False
  optimize_fasta: str | None = None
  write_pdb: bool = True
  emit_etab: bool = False
  emit_dense_hJ: bool = False  # noqa: N815 -- upstream knob name
  chain_design_mask_json: str | None = None


@dataclass(frozen=True)
class LaserOptions:
  """LASErMPNN inference knobs. ``None`` on the spec means family defaults."""

  fs_sequence_temp: float | None = None
  chi_temp: float | None = None
  seq_min_p: float = 0.0
  chi_min_p: float = 0.0
  disabled_residues: tuple[str, ...] = ("X", "C")
  disable_charged_fs: bool = False
  repack_only: bool = False
  fix_from_bfactor: bool = False
  ignore_ligand: bool = False
  use_water: bool = False
  noncanonical_aa_ligand: bool = False
  ala_budget: int = 4
  gly_budget: int = 0
  constrain_ala_gly_to_exposed_non_ss: bool = False
  budget_residue_selection: str | None = None
  ignore_chain_mask_zeros: bool = False
  tied_second_input: str | None = None
  tied_interpolation_lambda: float = 0.0
  selection_string: str | None = None
  n_decoding_orders: int = 10
  n_dropouts: int = 10
  proofread_dropout: bool = True
  repack_all: bool = False
  strict_load: bool = True
  output_fasta: bool = False
  output_fasta_only: bool = False
