"""Post-processing virtual C-beta distance threshold and ligand-proximal scoring (Clause 1).

Implements the reference LigandMPNN ligand cutoff scoring protocol (clause
`ligand_cutoff_for_score`, data_utils.py:948, run.py:432-442):
- Virtual C-beta coordinate construction from backbone N, CA, C coordinates.
- Euclidean distance calculation from each protein residue's C-beta to nearest valid ligand atom.
- Ligand-proximal residue mask construction with distance cutoff (default 8.0 A).
- Auxiliary ligand-proximal confidence score (loss_XY) and confidence (exp(-loss_XY)).
"""

from __future__ import annotations

import math
import numbers
from typing import NamedTuple

import jax
import jax.numpy as jnp

DEFAULT_LIGAND_CUTOFF_ANGSTROMS: float = 8.0
SCORE_EPS: float = 1e-8


class LigandScoreResult(NamedTuple):
  """Container for ligand-proximal scoring metrics."""

  loss_xy: jax.Array
  ligand_confidence: jax.Array
  mask_xy: jax.Array
  closest_distances: jax.Array
  num_ligand_proximal_residues: jax.Array


@jax.jit
def compute_virtual_c_beta(
  coordinates: jax.Array,
) -> jax.Array:
  """Compute virtual C-beta coordinates from backbone atoms (N, CA, C).

  Matches the empirical formula from ProteinMPNN and LigandMPNN
  (data_utils.py:938-944):
      b = CA - N
      c = C - CA
      a = cross(b, c)
      CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

  Args:
    coordinates: Backbone coordinates where index 0 is N, index 1 is CA,
      index 2 is C. Shape (..., 3+, 3).

  Returns:
    Virtual C-beta coordinates with shape (..., 3).
  """
  n = coordinates[..., 0, :]
  ca = coordinates[..., 1, :]
  c = coordinates[..., 2, :]

  b = ca - n
  c_vec = c - ca
  a = jnp.cross(b, c_vec, axis=-1)

  return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca


def validate_ligand_cutoff(cutoff: object) -> None:
  """Raise ``ValueError`` unless ``cutoff`` is a finite, strictly positive distance in Angstroms.

  Only concrete numbers are checked. A traced cutoff (inside ``jax.jit``) is passed through,
  because its value is not known at trace time; callers validate before tracing.
  """
  if not isinstance(cutoff, numbers.Real):
    return
  value = float(cutoff)
  if not (math.isfinite(value) and value > 0.0):
    msg = f"ligand cutoff must be a finite positive distance in Angstroms, got {cutoff!r}"
    raise ValueError(msg)


@jax.jit
def compute_ligand_cutoff_mask(
  coordinates: jax.Array,
  ligand_coords: jax.Array | None,
  ligand_mask: jax.Array | None,
  residue_mask: jax.Array,
  cutoff: float = DEFAULT_LIGAND_CUTOFF_ANGSTROMS,
) -> tuple[jax.Array, jax.Array]:
  """Compute per-residue mask indicating residues within `cutoff` A of valid ligand atoms.

  Calculates the minimum distance from each protein residue's virtual C-beta
  to any valid ligand atom. Mask is 1.0 for valid residues whose nearest ligand
  atom distance is strictly less than `cutoff` and at least one valid ligand atom
  exists in the scene.

  Args:
    coordinates: (L, 3+, 3) backbone coordinates.
    ligand_coords: Ligand coordinates (M, 3) or (L, M, 3), or None.
    ligand_mask: Ligand validity mask (M,) or (L, M), or None.
    residue_mask: (L,) protein residue mask (1.0 for valid, 0.0 for padding).
    cutoff: Distance cutoff in Angstroms (default: 8.0 A).

  Returns:
    Tuple of:
      mask_xy: (L,) float32 array in {0.0, 1.0}.
      d_xy: (L,) float32 array of minimum distances from each residue's C-beta to valid ligand atoms.
  """
  l_len = coordinates.shape[0]
  cb = compute_virtual_c_beta(coordinates)  # (L, 3)
  res_mask = residue_mask.astype(jnp.float32)

  if ligand_coords is None or ligand_mask is None:
    empty_mask = jnp.zeros((l_len,), dtype=jnp.float32)
    inf_dist = jnp.full((l_len,), jnp.inf, dtype=jnp.float32)
    return empty_mask, inf_dist

  # Normalize shapes of ligand_coords and ligand_mask
  # ligand_coords: (M, 3) or (L, M, 3)
  if ligand_coords.ndim == 3 and ligand_coords.shape[0] == l_len:
    # Per-residue ligand context (L, M, 3)
    y_coords = ligand_coords
    y_mask = ligand_mask.astype(jnp.float32)  # (L, M)
    diff = cb[:, None, :] - y_coords  # (L, M, 3)
    sq_dist = jnp.sum(diff**2, axis=-1)  # (L, M)
    # Masked ligand atoms are excluded with +inf, not a finite sentinel: a finite value (the old
    # 1000 A^2, i.e. 31.6 A) would cap the nearest distance and mark residues proximal under any
    # cutoff above it, even when every valid atom is farther away.
    sq_dist_masked = jnp.where(y_mask > 0.0, sq_dist, jnp.inf)
    min_sq_dist = jnp.min(sq_dist_masked, axis=-1)  # (L,)
    has_valid_ligand = (jnp.sum(y_mask, axis=-1) > 0.0).astype(jnp.float32)
  else:
    # Flat ligand coordinates (M, 3)
    y_coords = ligand_coords.reshape(-1, 3)  # (M, 3)
    y_mask = ligand_mask.reshape(-1).astype(jnp.float32)  # (M,)
    diff = cb[:, None, :] - y_coords[None, :, :]  # (L, M, 3)
    sq_dist = jnp.sum(diff**2, axis=-1)  # (L, M)
    sq_dist_masked = jnp.where(y_mask[None, :] > 0.0, sq_dist, jnp.inf)
    min_sq_dist = jnp.min(sq_dist_masked, axis=-1)  # (L,)
    has_valid_ligand = (jnp.sum(y_mask) > 0.0).astype(jnp.float32)

  d_xy = jnp.sqrt(jnp.maximum(min_sq_dist, 0.0))  # (L,)
  is_within_cutoff = (d_xy < cutoff).astype(jnp.float32)
  mask_xy = is_within_cutoff * res_mask * has_valid_ligand

  return mask_xy, d_xy


@jax.jit
def compute_loss_xy(
  logits: jax.Array,
  sequence: jax.Array,
  mask_xy: jax.Array,
) -> jax.Array:
  """Compute auxiliary ligand-proximal negative log-likelihood (loss_XY).

  Matches reference get_score(S, log_probs, combined_mask) in data_utils.py:218-232:
      loss_per_residue = -(S_one_hot * log_probs).sum(-1)
      average_loss = sum(loss_per_residue * mask_xy) / (sum(mask_xy) + 1e-8)

  The two forms are told apart by DTYPE, never by shape: an integer token sequence whose length
  happens to equal the vocabulary size (L == 21) is still tokens. Rank/shape dispatch silently
  read such a sequence as one-hot.

  Args:
    logits: Model prediction logits, shape (..., L, 21). The leading axes may carry candidates
      and a noise-level axis; the sequence broadcasts against them.
    sequence: Integer sequence tokens (..., L) (any integer dtype), or a FLOATING one-hot
      (..., L, 21).
    mask_xy: Ligand-proximal residue mask, shape (..., L).

  Returns:
    Scalar or batch of ligand-proximal NLL loss_XY values, shaped like ``logits`` without its
    trailing vocabulary axis and its residue axis.

  Raises:
    ValueError: If ``sequence`` is floating but its last axis is not the vocabulary size.
  """
  vocab_size = logits.shape[-1]
  if jnp.issubdtype(sequence.dtype, jnp.integer):
    seq_one_hot = jax.nn.one_hot(sequence.astype(jnp.int32), vocab_size)
  else:
    if sequence.shape[-1] != vocab_size:
      msg = (
        f"compute_loss_xy: a floating sequence must be one-hot with last axis {vocab_size}, "
        f"got shape {sequence.shape}. Integer tokens must have an integer dtype."
      )
      raise ValueError(msg)
    seq_one_hot = sequence

  log_probs = jax.nn.log_softmax(logits, axis=-1)
  loss_per_residue = -jnp.sum(seq_one_hot * log_probs, axis=-1)  # (..., L)

  m_xy = mask_xy.astype(jnp.float32)
  numerator = jnp.sum(loss_per_residue * m_xy, axis=-1)
  denominator = jnp.sum(m_xy, axis=-1) + SCORE_EPS

  return numerator / denominator


def evaluate_ligand_cutoff_score(
  coordinates: jax.Array,
  sequence: jax.Array,
  logits: jax.Array,
  residue_mask: jax.Array,
  ligand_coords: jax.Array | None = None,
  ligand_mask: jax.Array | None = None,
  cutoff: float = DEFAULT_LIGAND_CUTOFF_ANGSTROMS,
) -> LigandScoreResult:
  """Evaluate ligand-proximal score and confidence metrics.

  Args:
    coordinates: (L, 3+, 3) backbone coordinates.
    sequence: (L,) integer tokens or (L, 21) one-hot sequence.
    logits: (L, 21) model predictions.
    residue_mask: (L,) residue mask.
    ligand_coords: Optional ligand coordinates.
    ligand_mask: Optional ligand atom mask.
    cutoff: Distance cutoff in Angstroms (default: 8.0 A).

  Returns:
    LigandScoreResult with loss_xy, ligand_confidence, mask_xy, closest_distances,
    and num_ligand_proximal_residues.

  Raises:
    ValueError: If ``cutoff`` is not a finite positive number.
  """
  validate_ligand_cutoff(cutoff)
  mask_xy, d_xy = compute_ligand_cutoff_mask(
    coordinates=coordinates,
    ligand_coords=ligand_coords,
    ligand_mask=ligand_mask,
    residue_mask=residue_mask,
    cutoff=cutoff,
  )
  loss_xy = compute_loss_xy(logits=logits, sequence=sequence, mask_xy=mask_xy)
  ligand_confidence = jnp.exp(-loss_xy)
  num_proximal = jnp.sum(mask_xy > 0.5).astype(jnp.int32)

  return LigandScoreResult(
    loss_xy=loss_xy,
    ligand_confidence=ligand_confidence,
    mask_xy=mask_xy,
    closest_distances=d_xy,
    num_ligand_proximal_residues=num_proximal,
  )


def evaluate_ligand_cutoff_candidates(
  coordinates: jax.Array,
  sequences: jax.Array,
  logits: jax.Array,
  residue_mask: jax.Array,
  ligand_coords: jax.Array | None = None,
  ligand_mask: jax.Array | None = None,
  cutoff: float = DEFAULT_LIGAND_CUTOFF_ANGSTROMS,
) -> LigandScoreResult:
  """Run :func:`evaluate_ligand_cutoff_score` for every candidate of one structure.

  This is the production path used by ``aminx.host.runner.score``. The candidate axis is the
  leading axis of ``sequences`` and ``logits``; the structure-level inputs are shared.

  Args:
    coordinates: (L, 3+, 3) backbone coordinates of the structure.
    sequences: (C, L) integer tokens, one row per candidate.
    logits: (C, ..., L, 21) logits per candidate. The ``...`` axes (a noise-level axis, when
      present) are carried through unchanged.
    residue_mask: (L,) residue mask.
    ligand_coords: Optional ligand coordinates for this structure.
    ligand_mask: Optional ligand atom mask for this structure.
    cutoff: Distance cutoff in Angstroms.

  Returns:
    LigandScoreResult whose fields carry a leading candidate axis: ``loss_xy`` and
    ``ligand_confidence`` are (C, ...); ``mask_xy`` and ``closest_distances`` are (C, L);
    ``num_ligand_proximal_residues`` is (C,). The mask depends only on the structure, so every
    row of ``mask_xy`` is identical.
  """
  validate_ligand_cutoff(cutoff)

  def _one(seq: jax.Array, lg: jax.Array) -> LigandScoreResult:
    return evaluate_ligand_cutoff_score(
      coordinates=coordinates,
      sequence=seq,
      logits=lg,
      residue_mask=residue_mask,
      ligand_coords=ligand_coords,
      ligand_mask=ligand_mask,
      cutoff=cutoff,
    )

  return jax.vmap(_one)(sequences, logits)
