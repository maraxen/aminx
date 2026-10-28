"""Antibody CDR sequence-numbering masks, frequency logit transforms, and Barley descriptors.

Implements the inference-time CDR region masking, target amino-acid frequency
log-odds logit biasing, and Barley/Lehninger physicochemical descriptor calculation
described in AntiBMPNN (Zeyu Sun et al., 2024; <https://github.com/zeysun/AntiBMPNN>;
Zenodo DOI ``10.5281/zenodo.13387792``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from collections.abc import Callable, Mapping

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
  cdr_ranges: Mapping[str, tuple[int, int]] | None = None,
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
  cdr_ranges : Mapping[str, tuple[int, int]] | None, optional
    Optional custom inclusive ``(start, end)`` residue-number bounds keyed by loop
    name (``"h1".."l3"``, case-insensitive).

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

  bounds = dict(_CDR_BOUNDS[scheme_key])
  if cdr_ranges is not None:
    for k, v in cdr_ranges.items():
      bounds[k.lower()] = v
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


def _compute_region_scale(
  cdr_mask: CDRRegionMasks | jax.Array,
  region_masks: CDRRegionMasks | None = None,
  region_weights: Mapping[str, float] | None = None,
) -> jax.Array:
  """Compute per-position float32 scaling factor from CDR mask and optional loop weights."""
  if isinstance(cdr_mask, CDRRegionMasks):
    base_mask = cdr_mask.any_cdr.astype(jnp.float32)
    eff_regions = cdr_mask if region_masks is None else region_masks
  else:
    base_mask = jnp.asarray(cdr_mask, dtype=jnp.float32)
    eff_regions = region_masks

  if eff_regions is not None and region_weights:
    norm_weights = {k.lower(): float(v) for k, v in region_weights.items()}
    scale = jnp.zeros_like(base_mask, dtype=jnp.float32)
    for loop_name in ("h1", "h2", "h3", "l1", "l2", "l3", "heavy_cdr", "light_cdr", "any_cdr"):
      if loop_name in norm_weights:
        loop_m = getattr(eff_regions, loop_name).astype(jnp.float32)
        scale = scale + loop_m * norm_weights[loop_name]
    return base_mask * scale
  return base_mask


class CDRFrequencyLogitTransform(eqx.Module):
  """Composable logit transform applying CDR log-odds amino-acid frequency bias.

  Supports both:
  1. Standard ``BatchLogitFn`` / ``LogitTransform`` calls ``(per_state, bias=None)``
     or ``(logits, step, mask)`` where ``cdr_mask`` was pre-bound at construction
     time and ``logits`` has shape ``(S, L, V)``, ``(L, V)``, ``(S, V)`` (single-step
     AR at ``step``), or ``(V,)`` (single-step AR at ``step``).
  2. Direct calls ``(logits, cdr_mask, *, region_masks=None, region_weights=None)``
     where ``cdr_mask`` (a boolean/float mask of length ``L`` or ``CDRRegionMasks``)
     is passed as the second argument.

  Attributes
  ----------
  cdr_bias : jax.Array
    Precomputed per-position ``(N, 21)`` or per-token ``(21,)`` log-odds bias.
  log_odds_21 : jax.Array
    Unmasked log-odds vector of shape ``(21,)`` or ``(N, 21)`` with token 20 = 0.0.
  base_transform : BatchLogitFn | None
    Optional inner logit transform applied prior to adding ``cdr_bias``.
  has_bound_mask : bool
    Whether a per-residue ``cdr_mask`` was bound at construction time.
  """

  cdr_bias: jax.Array
  log_odds_21: jax.Array = eqx.field(default_factory=lambda: jnp.zeros((21,), dtype=jnp.float32))
  base_transform: BatchLogitFn | None = None
  has_bound_mask: bool = eqx.field(static=True, default=True)

  def __init__(
    self,
    cdr_bias: jax.Array,
    base_transform: BatchLogitFn | None = None,
    *,
    log_odds_21: jax.Array | None = None,
    has_bound_mask: bool = True,
  ) -> None:
    """Initialize CDR frequency log-odds logit transform.

    Parameters
    ----------
    cdr_bias : jax.Array
        Precomputed per-position ``(N, 21)`` or per-token ``(21,)`` log-odds bias.
    base_transform : BatchLogitFn | None, optional
        Optional inner logit transform applied prior to adding ``cdr_bias``.
    log_odds_21 : jax.Array | None, optional
        Unmasked log-odds vector of shape ``(21,)`` or ``(N, 21)``.
    has_bound_mask : bool, default=True
        Whether a per-residue ``cdr_mask`` was bound at construction time.

    Returns
    -------
    None
        Initializes instance attributes.
    """
    self.cdr_bias = cdr_bias
    self.log_odds_21 = cdr_bias if log_odds_21 is None else log_odds_21
    self.base_transform = base_transform
    self.has_bound_mask = has_bound_mask

  def __call__(  # noqa: PLR0911
    self,
    per_state: Logits,
    bias_or_step_or_mask: jax.Array | CDRRegionMasks | int | None = None,
    mask: jax.Array | None = None,
    *,
    region_masks: CDRRegionMasks | None = None,
    region_weights: Mapping[str, float] | None = None,
  ) -> Logits:
    """Apply inner logit transform (if any) followed by the CDR log-odds bias.

    Parameters
    ----------
    per_state : Logits
      Input logits array.
    bias_or_step_or_mask : jax.Array | CDRRegionMasks | int | None, optional
      Optional bias tensor, decoding step index, or CDR region mask.
    mask : jax.Array | None, optional
      Optional valid-residue mask for the 3-arg ``LogitTransform`` protocol.
    region_masks : CDRRegionMasks | None, optional
      Optional per-loop CDR boolean masks for loop-specific weighting.
    region_weights : Mapping[str, float] | None, optional
      Optional per-loop scaling factors keyed by loop name.

    Returns
    -------
    Logits
      Biased logits with the same shape as ``per_state``.
    """
    # Case 1: 3-arg LogitTransform protocol signature (logits, step, mask)
    if mask is not None or (
      bias_or_step_or_mask is not None
      and not isinstance(bias_or_step_or_mask, CDRRegionMasks)
      and jnp.asarray(bias_or_step_or_mask).ndim == 0
      and jnp.issubdtype(jnp.asarray(bias_or_step_or_mask).dtype, jnp.integer)
    ):
      step_idx = jnp.asarray(bias_or_step_or_mask, dtype=jnp.int32)
      out = self.base_transform(per_state, None) if self.base_transform is not None else per_state
      if self.cdr_bias.ndim == 2:
        if out.ndim >= 2 and out.shape[-2] == self.cdr_bias.shape[0]:
          return out + self.cdr_bias
        step_bias = self.cdr_bias[step_idx]
        return out + step_bias
      return out + self.cdr_bias

    # Case 2: Direct call (logits, cdr_mask, *, region_masks=..., region_weights=...)
    if isinstance(bias_or_step_or_mask, CDRRegionMasks) or (
      bias_or_step_or_mask is not None
      and (
        jnp.issubdtype(jnp.asarray(bias_or_step_or_mask).dtype, jnp.bool_)
        or jnp.asarray(bias_or_step_or_mask).shape[-1] != 21
        or not self.has_bound_mask
      )
    ):
      scale = _compute_region_scale(
        bias_or_step_or_mask,
        region_masks=region_masks,
        region_weights=region_weights,
      )
      out = self.base_transform(per_state, None) if self.base_transform is not None else per_state
      return out + scale[..., None] * self.log_odds_21

    # Case 3: Standard BatchLogitFn signature (per_state, bias=None)
    bias = bias_or_step_or_mask
    if self.base_transform is not None:
      out = self.base_transform(per_state, bias)
    else:
      if (
        per_state.ndim == 3
        and self.cdr_bias.ndim == 2
        and per_state.shape[1] == self.cdr_bias.shape[0]
      ):
        out = per_state.mean(axis=0)
      else:
        out = per_state
      if bias is not None:
        out = out + bias

    if region_masks is not None and region_weights:
      scale = _compute_region_scale(
        region_masks,
        region_masks=region_masks,
        region_weights=region_weights,
      )
      return out + scale[..., None] * self.log_odds_21

    if self.cdr_bias.ndim == 2 and out.ndim == 1:
      return out + self.cdr_bias[0]
    if self.cdr_bias.ndim == 2 and out.ndim == 2 and out.shape[0] != self.cdr_bias.shape[0]:
      return out + self.cdr_bias[0]
    return out + self.cdr_bias


def make_cdr_frequency_logit_transform(
  cdr_mask: CDRRegionMasks | jax.Array | None = None,
  *,
  target_frequencies: jax.Array | None = None,
  background_frequencies: jax.Array | None = None,
  weight: float = 1.0,
  eps: float = 1e-8,
  base_transform: BatchLogitFn | None = None,
  region_masks: CDRRegionMasks | None = None,
  region_weights: Mapping[str, float] | None = None,
) -> CDRFrequencyLogitTransform:
  """Create a :class:`CDRFrequencyLogitTransform` for CDR amino-acid log-odds biasing.

  Computes the additive logit bias:
  ``delta_bias[i, a] = weight * cdr_mask[i] * log(clip(p_target[a], eps, 1.0) / clip(p_bg[a], eps, 1.0))``
  for standard amino acids ``a in 0..19``, while leaving unknown token ``a == 20 ('X')``
  and non-CDR positions (``cdr_mask[i] == 0``) strictly unmodified (``0.0``).

  Parameters
  ----------
  cdr_mask : CDRRegionMasks | jax.Array | None, optional
    Boolean or float mask of shape ``(N,)`` (or :class:`CDRRegionMasks`) indicating
    CDR positions to bias. If ``None``, the mask can be supplied at call time.
  target_frequencies : jax.Array | None, optional
    Target amino-acid frequencies of shape ``(20,)``, ``(21,)``, ``(N, 20)``, or
    ``(N, 21)``. Defaults to uniform ``1 / 20`` across the 20 standard amino acids.
  background_frequencies : jax.Array | None, optional
    Background amino-acid frequencies of shape ``(20,)``, ``(21,)``, ``(N, 20)``, or
    ``(N, 21)``. Defaults to uniform ``1 / 20`` across the 20 standard amino acids.
  weight : float, default=1.0
    Scalar multiplier on the log-odds bias term.
  eps : float, default=1e-8
    Small positive constant preventing ``log(0)``.
  base_transform : BatchLogitFn | None, optional
    Optional inner ``BatchLogitFn`` (e.g. from ``make_logit_transform``) to run
    before adding the CDR log-odds bias.
  region_masks : CDRRegionMasks | None, optional
    Optional loop masks for per-loop weighting when ``region_weights`` is supplied.
  region_weights : Mapping[str, float] | None, optional
    Optional per-loop weight multipliers keyed by loop name (``"h1".."l3"``).

  Returns
  -------
  CDRFrequencyLogitTransform
    Composable ``BatchLogitFn`` module holding the precomputed bias.

  Raises
  ------
  ValueError
    If ``target_frequencies`` or ``background_frequencies`` do not have last dimension
    ``20`` or ``21``.
  """
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

  log_odds_20 = jnp.asarray(weight, dtype=jnp.float32) * jnp.log(
    jnp.clip(p_target, eps, 1.0) / jnp.clip(p_bg, eps, 1.0),
  )
  zero_x_shape = (*log_odds_20.shape[:-1], 1)
  log_odds_21 = jnp.concatenate(
    [log_odds_20, jnp.zeros(zero_x_shape, dtype=jnp.float32)],
    axis=-1,
  )

  if cdr_mask is None and region_masks is None:
    return CDRFrequencyLogitTransform(
      cdr_bias=log_odds_21,
      base_transform=base_transform,
      log_odds_21=log_odds_21,
      has_bound_mask=False,
    )

  eff_mask_source = cdr_mask if cdr_mask is not None else region_masks
  if eff_mask_source is None:
    msg = "cdr_mask or region_masks must be provided"
    raise ValueError(msg)
  mask_f = _compute_region_scale(
    eff_mask_source,
    region_masks=region_masks,
    region_weights=region_weights,
  )
  n_res = mask_f.shape[0]

  if log_odds_20.ndim == 1:
    log_odds_20_b = jnp.broadcast_to(log_odds_20[None, :], (n_res, 20))
  elif log_odds_20.shape[0] != n_res:
    msg = (
      f"Per-position frequency array length {log_odds_20.shape[0]} does not match "
      f"cdr_mask length {n_res}."
    )
    raise ValueError(msg)
  else:
    log_odds_20_b = log_odds_20

  biased_20 = mask_f[:, None] * log_odds_20_b
  zero_x = jnp.zeros((n_res, 1), dtype=jnp.float32)
  cdr_bias = jnp.concatenate([biased_20, zero_x], axis=-1)
  return CDRFrequencyLogitTransform(
    cdr_bias=cdr_bias,
    base_transform=base_transform,
    log_odds_21=log_odds_21,
    has_bound_mask=True,
  )


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
  cdr_mask: CDRRegionMasks | jax.Array | None = None,
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
  cdr_mask : CDRRegionMasks | jax.Array | None, optional
    Optional boolean or float mask of shape ``(N,)`` or ``(B, N)``, or a
    :class:`CDRRegionMasks` instance (automatically converted to ``cdr_mask.any_cdr``),
    selecting the residues over which to compute the descriptors. When ``None``, all
    standard amino-acid positions (``sequences < 20``) are included.
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
  if isinstance(cdr_mask, CDRRegionMasks):
    cdr_mask = cdr_mask.any_cdr

  seq_clamped = jnp.clip(sequences, 0, 20)
  is_standard_aa = (sequences >= 0) & (sequences < 20)
  if cdr_mask is None:
    active = is_standard_aa.astype(jnp.float32)
  else:
    active = (jnp.asarray(cdr_mask) > 0).astype(jnp.float32) * is_standard_aa.astype(jnp.float32)

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

