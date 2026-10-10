"""Flat option dataclasses for driver-family knobs (PottsMPNN, ProtonPottsMPNN and LASErMPNN)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PottsMPNNOptions:
  """PottsMPNN inference knobs. ``None`` on the spec means family defaults."""

  optimization_mode: Literal["none", "potts", "potts_converge", "nodes"] = "potts"
  optimization_temperature: float = 0.0
  binding_energy_optimization: Literal["none", "both", "only"] = "none"
  binding_energy_json: str | None = None
  mean_norm: bool = False
  mutant_fasta: str | None = None
  mutant_csv: str | None = None
  exclude_chains: str | None = None
  pssm_json: str | None = None
  pssm_threshold: float = 0.0
  pssm_multi: float = 0.0
  pssm_log_odds_flag: bool = False
  pssm_bias_flag: bool = False
  bias_by_res_json: str | None = None
  tied_epistasis: bool = False
  skip_gaps: bool = False
  optimize_pdb: bool = False
  optimize_fasta: str | None = None
  emit_etab: bool = False
  emit_dense_hJ: bool = False  # noqa: N815 -- upstream knob name
  chain_design_mask_json: str | None = None


@dataclass(frozen=True)
class ProtonPottsOptions:
  """ProtonPottsMPNN inference knobs. ``None`` on the spec means family defaults.

  Residues are addressed as ``"<chain>:<number>"`` (``"A:42"``), or ``"<chain>:<number><insertion>"``
  where the file carries an insertion code. A token is a v6 protonation token (``"HIS-P"``), a
  one-letter standard residue, or ``"X"``.

  Attributes:
    protonation_labels_json: Path to a JSON object ``{"A:42": "HIS-P", ...}`` of pre-assigned
      protonation labels for the reference structure (aminx consumes labels; it does not assign
      them, spec §11a). ``None`` means no labels, so every residue is a standard residue.
    variants_json: Path to a JSON object ``{name: variant}``. A variant is either pins applied on
      top of the reference sequence, ``{"A:42": "HIS"}``, or a full per-residue token list whose
      length is the number of residues kept after featurisation. ``score:ddg`` requires it;
      ``score:energy`` scores the reference plus each variant.

  The ``design_*`` / pH-design fields configure ``ph_config.PHDesignConfig`` (``design_method``
  accepts ``block_descent`` or ``greedy_energy_block`` only; everything else is refused there).
  ``binder_chain=None`` means no design chain was given, which design runs reject. ``dep_map=()``
  means the family default map. ``explicit_centers`` requires ``center_types=()``.

  Nested sequences are stored as tuples (``dep_map`` as ``((type, (deprotonated, ...)), ...)``,
  ``explicit_centers`` as ``((residue_id, type), ...)``) so the options stay hashable: the spec holds
  them as a static field. ``options_from_json_value`` only converts the top-level list fields, so
  ``__post_init__`` normalises the nested lists that JSON decoding leaves behind.
  """

  protonation_labels_json: str | None = None
  variants_json: str | None = None
  binder_chain: str | None = None
  design_method: str = "block_descent"
  block_size: int = 3
  combined_lambda: float = 0.3
  temperature: float = 0.05
  samples_per_site: int = 2
  center_types: tuple[str, ...] = ("HIS-P", "ASP-P", "GLU-P")
  explicit_centers: tuple[tuple[int, str], ...] = ()
  dep_map: tuple[tuple[str, tuple[str, ...]], ...] = ()
  forbidden_tokens: tuple[str, ...] = ("HIS-A", "ASP-A", "GLU-A", "UNK")
  neighbour_k: int = 16
  max_mutations: int = 20
  infill_scope: str = "neighbourhood"
  repetitive_window_weight: float = 1.0
  repetitive_window_radius: int = 2
  repetitive_window_parents: tuple[str, ...] = ("ARG", "LYS", "HIS", "ASP", "GLU")
  block_max_rounds: int = 10
  cv_patience: int = 3
  cv_max: int = 50
  record_trajectory: bool = True

  def __post_init__(self) -> None:
    dep = self.dep_map.items() if isinstance(self.dep_map, Mapping) else self.dep_map
    object.__setattr__(self, "dep_map", tuple((str(k), tuple(str(t) for t in v)) for k, v in dep))
    object.__setattr__(
      self,
      "explicit_centers",
      tuple((int(r), str(t)) for r, t in self.explicit_centers),
    )


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
