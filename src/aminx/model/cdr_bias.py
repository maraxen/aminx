"""Antibody CDR sequence-numbering masks, frequency logit transforms, and Barley descriptors.

Implements the inference-time CDR region masking, target amino-acid frequency
log-odds logit biasing, and Barley/Lehninger physicochemical descriptor calculation
described in AntiBMPNN (Goudy et al., 2024; Zenodo DOI ``10.5281/zenodo.13387792``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from collections.abc import Callable

  from aminx.types.arrays import ChainIndex, Logits, ProteinMask, ResidueIndex, Sequence

  BatchLogitFn = Callable[..., jax.Array]

# Standard 21-token Aminx alphabet: "ACDEFGHIKLMNPQRSTVWYX"
# Indices 0..19 are the 20 standard amino acids in alphabetical one-letter order;
# index 20 ('X') is the unknown/padding token (always neutral / 0.0).
ALPHABET_21: str = "ACDEFGHIKLMNPQRSTVWYX"

# Barley et al. (2018) PC1 (volume / size) and PC2 (hydrophobicity / polarity)
# descriptor scales in Aminx 21-token alphabetical order (A..Y, X=0.0).
BARLEY_PD1_VOLUME: tuple[float, ...] = (
  -1.56,  # A (0)
  -0.76,  # C (1)
  -1.19,  # D (2)
  -0.37,  # E (3)
  1.10,   # F (4)
  -2.06,  # G (5)
  0.33,   # H (6)
  0.21,   # I (7)
  0.15,   # K (8)
  0.21,   # L (9)
  0.42,   # M (10)
  -0.95,  # N (11)
  -1.01,  # P (12)
  -0.12,  # Q (13)
  0.95,   # R (14)
  -1.32,  # S (15)
  -0.79,  # T (16)
  -0.38,  # V (17)
  2.19,   # W (18)
  1.29,   # Y (19)
  0.0,    # X (20)
)

BARLEY_PD2_HYDRO: tuple[float, ...] = (
  -0.96,  # A (0)
  -0.94,  # C (1)
  1.25,   # D (2)
  1.13,   # E (3)
  -1.52,  # F (4)
  -0.22,  # G (5)
  0.33,   # H (6)
  -1.47,  # I (7)
  1.17,   # K (8)
  -1.42,  # L (9)
  -1.01,  # M (10)
  0.81,   # N (11)
  -0.17,  # P (12)
  0.72,   # Q (13)
  0.88,   # R (14)
  0.85,   # S (15)
  0.28,   # T (16)
  -1.16,  # V (17)
  -1.27,  # W (18)
  -0.61,  # Y (19)
  0.0,    # X (20)
)

# Lehninger pKa table used by peptides.Peptide.charge(pKscale="Lehninger") in AntiBMPNN
LEHNINGER_PKA: dict[str, float] = {
  "N_term": 9.69,
  "C_term": 2.34,
  "K": 10.53,
  "R": 12.48,
  "H": 6.00,
  "D": 3.65,
  "E": 4.25,
  "C": 8.18,
  "Y": 10.07,
}

# Precomputed JAX lookup tables at module import time for JIT safety
_PD1_TABLE: jax.Array = jnp.array(BARLEY_PD1_VOLUME, dtype=jnp.float32)
_PD2_TABLE: jax.Array = jnp.array(BARLEY_PD2_HYDRO, dtype=jnp.float32)

# Chothia and IMGT inclusive residue-number bounds: (start, end)
_CDR_BOUNDS: dict[str, dict[str, tuple[int, int]]] = {
  "chothia": {
    "h1": (26, 32),
    "h2": (52, 56),
    "h3": (95, 102),
    "l1": (24, 34),
    "l2": (50, 56),
    "l3": (89, 97),
  },
  "imgt": {
    "h1": (27, 38),
    "h2": (56, 65),
    "h3": (105, 117),
    "l1": (27, 38),
    "l2": (56, 65),
    "l3": (105, 117),
  },
}


class CDRRegionMasks(NamedTuple):
  """Boolean masks ``(N,)`` identifying each antibody CDR loop and union region.

  Attributes
  ----------
  h1 : jax.Array
    Boolean mask of shape ``(N,)`` for CDR-H1 residues.
  h2 : jax.Array
    Boolean mask of shape ``(N,)`` for CDR-H2 residues.
  h3 : jax.Array
    Boolean mask of shape ``(N,)`` for CDR-H3 residues.
  l1 : jax.Array
    Boolean mask of shape ``(N,)`` for CDR-L1 residues.
  l2 : jax.Array
    Boolean mask of shape ``(N,)`` for CDR-L2 residues.
  l3 : jax.Array
    Boolean mask of shape ``(N,)`` for CDR-L3 residues.
  heavy_cdr : jax.Array
    Boolean mask ``h1 | h2 | h3`` of shape ``(N,)``.
  light_cdr : jax.Array
    Boolean mask ``l1 | l2 | l3`` of shape ``(N,)``.
  any_cdr : jax.Array
    Boolean mask ``heavy_cdr | light_cdr`` of shape ``(N,)``.
  """

  h1: jax.Array
  h2: jax.Array
  h3: jax.Array
  l1: jax.Array
  l2: jax.Array
  l3: jax.Array
  heavy_cdr: jax.Array
  light_cdr: jax.Array
  any_cdr: jax.Array


def build_cdr_region_mask(
  residue_index: ResidueIndex,
  chain_index: ChainIndex,
  *,
  heavy_chain_id: int | None = 0,
  light_chain_id: int | None = 1,
  scheme: str = "chothia",
  mask: ProteinMask | None = None,
) -> CDRRegionMasks:
  """Construct boolean masks for CDR loops under Chothia or IMGT numbering.

  Parameters
  ----------
  residue_index : ResidueIndex
    Per-residue sequence numbering of shape ``(N,)``.
  chain_index : ChainIndex
    Per-residue chain identifier of shape ``(N,)``.
  heavy_chain_id : int | None, default=0
    Chain index corresponding to the antibody heavy chain, or ``None`` if absent.
  light_chain_id : int | None, default=1
    Chain index corresponding to the antibody light chain, or ``None`` if absent.
  scheme : str, default="chothia"
    Numbering scheme name (``"chothia"`` or ``"imgt"``).
  mask : ProteinMask | None, optional
    Optional valid-residue mask of shape ``(N,)``. Invalid positions (``mask <= 0``)
    are excluded from all CDR masks.

  Returns
  -------
  CDRRegionMasks
    NamedTuple of boolean masks of shape ``(N,)`` for H1..H3, L1..L3, heavy_cdr,
    light_cdr, and any_cdr.

  Raises
  ------
  ValueError
    If ``scheme`` is not ``"chothia"`` or ``"imgt"``.
  """
  scheme_key = scheme.lower()
  if scheme_key not in _CDR_BOUNDS:
    msg = f"Unknown CDR numbering scheme {scheme!r}; expected one of {sorted(_CDR_BOUNDS)}."
    raise ValueError(msg)

  bounds = _CDR_BOUNDS[scheme_key]
  valid = jnp.ones_like(residue_index, dtype=jnp.bool_) if mask is None else (mask > 0)

  if heavy_chain_id is not None:
    is_heavy = valid & (chain_index == heavy_chain_id)
  else:
    is_heavy = jnp.zeros_like(valid, dtype=jnp.bool_)

  if light_chain_id is not None:
    is_light = valid & (chain_index == light_chain_id)
  else:
    is_light = jnp.zeros_like(valid, dtype=jnp.bool_)

  h1_lo, h1_hi = bounds["h1"]
  h2_lo, h2_hi = bounds["h2"]
  h3_lo, h3_hi = bounds["h3"]
  l1_lo, l1_hi = bounds["l1"]
  l2_lo, l2_hi = bounds["l2"]
  l3_lo, l3_hi = bounds["l3"]

  h1 = is_heavy & (residue_index >= h1_lo) & (residue_index <= h1_hi)
  h2 = is_heavy & (residue_index >= h2_lo) & (residue_index <= h2_hi)
  h3 = is_heavy & (residue_index >= h3_lo) & (residue_index <= h3_hi)

  l1 = is_light & (residue_index >= l1_lo) & (residue_index <= l1_hi)
  l2 = is_light & (residue_index >= l2_lo) & (residue_index <= l2_hi)
  l3 = is_light & (residue_index >= l3_lo) & (residue_index <= l3_hi)

  heavy_cdr = h1 | h2 | h3
  light_cdr = l1 | l2 | l3
  any_cdr = heavy_cdr | light_cdr

  return CDRRegionMasks(
    h1=h1,
    h2=h2,
    h3=h3,
    l1=l1,
    l2=l2,
    l3=l3,
    heavy_cdr=heavy_cdr,
    light_cdr=light_cdr,
    any_cdr=any_cdr,
  )


class CDRFrequencyLogitTransform(eqx.Module):
  """Composable ``BatchLogitFn`` applying CDR log-odds amino-acid frequency bias.

  Conforms to :class:`aminx.model._inference.logits.BatchLogitFn` so it can be
  passed directly to ``make_sampling_fn(..., logit_transform=...)`` or composed
  around an existing ``BatchLogitFn``.

  Attributes
  ----------
  cdr_bias : jax.Array
    Precomputed per-position, per-token log-odds bias of shape ``(N, 21)``.
  base_transform : BatchLogitFn | None
    Optional inner logit transform applied prior to adding ``cdr_bias``.
  """

  cdr_bias: jax.Array
  base_transform: BatchLogitFn | None = None

  def __call__(
    self,
    per_state: Logits,
    bias: Logits | None = None,
  ) -> Logits:
    """Apply inner logit transform (if any) followed by the CDR log-odds bias.

    Parameters
    ----------
    per_state : Logits
      Per-state logits of shape ``(K, N, 21)`` or ``(N, 21)``.
    bias : Logits | None, optional
      Optional explicit additive bias of shape ``(N, 21)``.

    Returns
    -------
    Logits
      Transformed logits with CDR frequency log-odds bias applied.
    """
    if self.base_transform is not None:
      out = self.base_transform(per_state, bias)
    else:
      out = per_state.mean(axis=0) if per_state.ndim == 3 else per_state
      if bias is not None:
        out = out + bias
    return out + self.cdr_bias


def make_cdr_frequency_logit_transform(
  cdr_mask: jax.Array,
  *,
  target_frequencies: jax.Array | None = None,
  background_frequencies: jax.Array | None = None,
  weight: float = 1.0,
  eps: float = 1e-6,
  base_transform: BatchLogitFn | None = None,
) -> CDRFrequencyLogitTransform:
  """Create a :class:`CDRFrequencyLogitTransform` for CDR amino-acid log-odds biasing.

  Computes the additive logit bias:
  ``delta_bias[i, a] = weight * cdr_mask[i] * log((p_target[a] + eps) / (p_bg[a] + eps))``
  for standard amino acids ``a in 0..19``, while leaving unknown token ``a == 20 ('X')``
  and non-CDR positions (``cdr_mask[i] == 0``) strictly unmodified (``0.0``).

  Parameters
  ----------
  cdr_mask : jax.Array
    Boolean or float mask of shape ``(N,)`` indicating CDR positions to bias.
  target_frequencies : jax.Array | None, optional
    Target amino-acid frequencies of shape ``(20,)``, ``(21,)``, ``(N, 20)``, or
    ``(N, 21)``. Defaults to uniform ``1 / 20`` across the 20 standard amino acids.
  background_frequencies : jax.Array | None, optional
    Background amino-acid frequencies of shape ``(20,)``, ``(21,)``, ``(N, 20)``, or
    ``(N, 21)``. Defaults to uniform ``1 / 20`` across the 20 standard amino acids.
  weight : float, default=1.0
    Scalar multiplier on the log-odds bias term.
  eps : float, default=1e-6
    Small positive constant preventing ``log(0)``.
  base_transform : BatchLogitFn | None, optional
    Optional inner ``BatchLogitFn`` (e.g. from ``make_logit_transform``) to run
    before adding the CDR log-odds bias.

  Returns
  -------
  CDRFrequencyLogitTransform
    Composable ``BatchLogitFn`` module holding the precomputed ``(N, 21)`` bias.

  Raises
  ------
  ValueError
    If ``target_frequencies`` or ``background_frequencies`` do not have last dimension
    ``20`` or ``21``.
  """
  mask_f = cdr_mask.astype(jnp.float32)
  n_res = mask_f.shape[0]

  def _extract_aa20(freq: jax.Array | None, name: str) -> jax.Array:
    if freq is None:
      return jnp.full((20,), 1.0 / 20.0, dtype=jnp.float32)
    arr = jnp.asarray(freq, dtype=jnp.float32)
    if arr.shape[-1] == 21:
      arr = arr[..., :20]
    elif arr.shape[-1] != 20:
      msg = f"{name} must have last dimension 20 or 21, got shape {arr.shape}."
      raise ValueError(msg)
    total = jnp.sum(arr, axis=-1, keepdims=True)
    return jnp.where(total > 0.0, arr / jnp.maximum(total, eps), 1.0 / 20.0)

  p_target = _extract_aa20(target_frequencies, "target_frequencies")
  p_bg = _extract_aa20(background_frequencies, "background_frequencies")

  log_odds_20 = jnp.asarray(weight, dtype=jnp.float32) * (
    jnp.log(p_target + eps) - jnp.log(p_bg + eps)
  )
  if log_odds_20.ndim == 1:
    log_odds_20 = jnp.broadcast_to(log_odds_20[None, :], (n_res, 20))
  elif log_odds_20.shape[0] != n_res:
    msg = (
      f"Per-position frequency array length {log_odds_20.shape[0]} does not match "
      f"cdr_mask length {n_res}."
    )
    raise ValueError(msg)

  biased_20 = mask_f[:, None] * log_odds_20
  zero_x = jnp.zeros((n_res, 1), dtype=jnp.float32)
  cdr_bias = jnp.concatenate([biased_20, zero_x], axis=-1)
  return CDRFrequencyLogitTransform(cdr_bias=cdr_bias, base_transform=base_transform)


class CDRPhysicochemicalDescriptors(NamedTuple):
  """Per-sequence physicochemical descriptors matching AntiBMPNN's evaluation pipeline.

  Attributes
  ----------
  pd1_volume : jax.Array
    Mean Barley PC1 (volume / size) descriptor per sequence.
  pd2_hydro : jax.Array
    Mean Barley PC2 (hydrophobicity / polarity) descriptor per sequence.
  pd3_charge : jax.Array
    Net Henderson-Hasselbalch charge at the target pH under the Lehninger pKa scale.
  """

  pd1_volume: jax.Array
  pd2_hydro: jax.Array
  pd3_charge: jax.Array


def compute_cdr_physicochemical_descriptors(
  sequences: Sequence,
  cdr_mask: jax.Array | None = None,
  *,
  ph: float = 7.4,
  include_termini: bool = True,
) -> CDRPhysicochemicalDescriptors:
  """Compute vectorized Barley PC1/PC2 and Lehninger net charge descriptors.

  Matches the ``peptides.Peptide(seq).descriptors()["PD1", "PD2"]`` and
  ``peptides.Peptide(seq).charge(pH=7.4, pKscale="Lehninger")`` evaluation
  metrics reported in ``AntiBMPNN/README.md``.

  Parameters
  ----------
  sequences : Sequence
    Integer token array of shape ``(N,)`` or ``(B, N)`` in Aminx 21-token encoding
    (``0..19`` standard amino acids, ``20`` unknown ``'X'``).
  cdr_mask : jax.Array | None, optional
    Optional boolean or float mask of shape ``(N,)`` or ``(B, N)`` selecting the
    residues over which to compute the descriptors. When ``None``, all standard
    amino-acid positions (``sequences < 20``) are included.
  ph : float, default=7.4
    Solution pH for the Henderson-Hasselbalch charge calculation.
  include_termini : bool, default=True
    Whether to include the N-terminal and C-terminal ionizable groups in ``pd3_charge``
    (matching ``peptides.Peptide.charge`` when a contiguous peptide/CDR segment is scored).

  Returns
  -------
  CDRPhysicochemicalDescriptors
    NamedTuple ``(pd1_volume, pd2_hydro, pd3_charge)`` of scalar arrays (when input
    is 1D ``(N,)``) or shape ``(B,)`` arrays (when input is 2D ``(B, N)``).
  """
  seq_clamped = jnp.clip(sequences, 0, 20)
  is_standard_aa = (sequences >= 0) & (sequences < 20)
  if cdr_mask is None:
    active = is_standard_aa.astype(jnp.float32)
  else:
    active = (cdr_mask > 0).astype(jnp.float32) * is_standard_aa.astype(jnp.float32)

  count = jnp.sum(active, axis=-1)
  safe_count = jnp.maximum(count, 1.0)

  pd1_vals = _PD1_TABLE[seq_clamped] * active
  pd2_vals = _PD2_TABLE[seq_clamped] * active

  pd1_mean = jnp.where(count > 0.0, jnp.sum(pd1_vals, axis=-1) / safe_count, 0.0)
  pd2_mean = jnp.where(count > 0.0, jnp.sum(pd2_vals, axis=-1) / safe_count, 0.0)

  # Henderson-Hasselbalch net charge under Lehninger pKa scale:
  # Positive groups: q_pos = +1 / (1 + 10^(pH - pKa))
  # Negative groups: q_neg = -1 / (1 + 10^(pKa - pH))
  ph_f = jnp.asarray(ph, dtype=jnp.float32)

  # Build per-token sidechain charge contribution table at pH
  q_k = 1.0 / (1.0 + jnp.power(10.0, ph_f - LEHNINGER_PKA["K"]))
  q_r = 1.0 / (1.0 + jnp.power(10.0, ph_f - LEHNINGER_PKA["R"]))
  q_h = 1.0 / (1.0 + jnp.power(10.0, ph_f - LEHNINGER_PKA["H"]))
  q_d = -1.0 / (1.0 + jnp.power(10.0, LEHNINGER_PKA["D"] - ph_f))
  q_e = -1.0 / (1.0 + jnp.power(10.0, LEHNINGER_PKA["E"] - ph_f))
  q_c = -1.0 / (1.0 + jnp.power(10.0, LEHNINGER_PKA["C"] - ph_f))
  q_y = -1.0 / (1.0 + jnp.power(10.0, LEHNINGER_PKA["Y"] - ph_f))

  # Alphabet order: A(0), C(1), D(2), E(3), F(4), G(5), H(6), I(7), K(8), L(9),
  # M(10), N(11), P(12), Q(13), R(14), S(15), T(16), V(17), W(18), Y(19), X(20)
  per_res_charge = (
    jnp.where(seq_clamped == 1, q_c, 0.0)
    + jnp.where(seq_clamped == 2, q_d, 0.0)
    + jnp.where(seq_clamped == 3, q_e, 0.0)
    + jnp.where(seq_clamped == 6, q_h, 0.0)
    + jnp.where(seq_clamped == 8, q_k, 0.0)
    + jnp.where(seq_clamped == 14, q_r, 0.0)
    + jnp.where(seq_clamped == 19, q_y, 0.0)
  )
  sidechain_charge = jnp.sum(per_res_charge * active, axis=-1)

  if include_termini:
    q_nterm = 1.0 / (1.0 + jnp.power(10.0, ph_f - LEHNINGER_PKA["N_term"]))
    q_cterm = -1.0 / (1.0 + jnp.power(10.0, LEHNINGER_PKA["C_term"] - ph_f))
    termini_charge = q_nterm + q_cterm
  else:
    termini_charge = jnp.asarray(0.0, dtype=jnp.float32)

  pd3_charge = jnp.where(count > 0.0, sidechain_charge + termini_charge, 0.0)

  return CDRPhysicochemicalDescriptors(
    pd1_volume=pd1_mean,
    pd2_hydro=pd2_mean,
    pd3_charge=pd3_charge,
  )


__all__ = [
  "BARLEY_PD1_VOLUME",
  "BARLEY_PD2_HYDRO",
  "LEHNINGER_PKA",
  "CDRFrequencyLogitTransform",
  "CDRPhysicochemicalDescriptors",
  "CDRRegionMasks",
  "build_cdr_region_mask",
  "compute_cdr_physicochemical_descriptors",
  "make_cdr_frequency_logit_transform",
]
