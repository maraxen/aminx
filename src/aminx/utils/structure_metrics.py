"""Structural metrics and distance calculations for protein structures."""

from __future__ import annotations

import jax.numpy as jnp


def _extract_ca_coordinates(coordinates: jnp.ndarray) -> jnp.ndarray:
  if coordinates.ndim == 2:
    return coordinates
  if coordinates.ndim == 3:
    if coordinates.shape[1] < 2:
      msg = f"coordinates need a CA column (atom axis >= 2); got shape {coordinates.shape}"
      raise ValueError(msg)
    return coordinates[:, 1, :]
  msg = f"coordinates must be (L, 3) or (L, A, 3); got shape {coordinates.shape}"
  raise ValueError(msg)


def _extract_cb_coordinates(coordinates: jnp.ndarray) -> jnp.ndarray:
  if coordinates.ndim == 2:
    return coordinates
  if coordinates.ndim == 3:
    # MPNN atom order: N=0, CA=1, C=2, CB=3 (glycine may lack CB; caller supplies CA fallback)
    if coordinates.shape[1] < 4:
      # JAX would silently clamp the read to the last column and return a wrong atom (aminx #2152).
      msg = f"coordinates need a CB column (atom axis >= 4); got shape {coordinates.shape}"
      raise ValueError(msg)
    return coordinates[:, 3, :]
  msg = f"coordinates must be (L, 3) or (L, A, 3); got shape {coordinates.shape}"
  raise ValueError(msg)


def calculate_ca_distance_matrix(coordinates: jnp.ndarray) -> jnp.ndarray:
  """Pairwise Euclidean distances between C-alpha atoms."""
  ca = _extract_ca_coordinates(coordinates)
  diff = ca[:, None, :] - ca[None, :, :]
  return jnp.sqrt(jnp.sum(diff * diff, axis=-1))


def calculate_cb_distance_matrix(coordinates: jnp.ndarray) -> jnp.ndarray:
  """Pairwise Euclidean distances between C-beta atoms."""
  cb = _extract_cb_coordinates(coordinates)
  diff = cb[:, None, :] - cb[None, :, :]
  return jnp.sqrt(jnp.sum(diff * diff, axis=-1))


def calculate_closest_atom_distance_matrix(
  coordinates: jnp.ndarray,
  atom_mask: jnp.ndarray,
) -> jnp.ndarray:
  """Minimum inter-residue distance over all atom pairs."""
  if coordinates.ndim != 3:
    msg = f"closest-atom distances require (L, A, 3) coordinates; got {coordinates.shape}"
    raise ValueError(msg)
  length = coordinates.shape[0]
  diff = coordinates[:, None, :, None, :] - coordinates[None, :, None, :, :]
  dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
  pair_mask = atom_mask[:, None, :, None] * atom_mask[None, :, None, :]
  masked = jnp.where(pair_mask > 0, dist, jnp.inf)
  return jnp.min(masked, axis=(-2, -1))


def _kabsch_align(mobile: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
  mobile_centroid = mobile.mean(axis=0)
  target_centroid = target.mean(axis=0)
  mobile_centered = mobile - mobile_centroid
  target_centered = target - target_centroid
  cov = mobile_centered.T @ target_centered
  u, _, vt = jnp.linalg.svd(cov)
  d = jnp.eye(3)
  d = d.at[2, 2].set(jnp.sign(jnp.linalg.det(u @ vt)))
  rot = u @ d @ vt
  return mobile_centered @ rot.T + target_centroid


def calculate_rmsd(
  coordinates1: jnp.ndarray,
  coordinates2: jnp.ndarray,
  *,
  align: bool = True,
) -> jnp.ndarray:
  """Root-mean-square deviation between two coordinate sets (L, 3)."""
  if coordinates1.shape != coordinates2.shape:
    msg = f"coordinate shapes must match: {coordinates1.shape} vs {coordinates2.shape}"
    raise ValueError(msg)
  if align:
    coordinates1 = _kabsch_align(coordinates1, coordinates2)
  diff = coordinates1 - coordinates2
  return jnp.sqrt(jnp.mean(jnp.sum(diff * diff, axis=-1)))


def calculate_tm_score(
  coordinates1: jnp.ndarray,
  coordinates2: jnp.ndarray,
  sequence_length: int,
) -> jnp.ndarray:
  """Length-normalized TM-score approximation from aligned CA RMSD."""
  del sequence_length
  ca1 = _extract_ca_coordinates(coordinates1)
  ca2 = _extract_ca_coordinates(coordinates2)
  rmsd = calculate_rmsd(ca1, ca2, align=True)
  d0 = 1.24 * jnp.cbrt(float(ca1.shape[0]) - 15) - 1.8
  d0 = jnp.maximum(d0, 0.5)
  return 1.0 / (1.0 + (rmsd / d0) ** 2)


def calculate_cosine_similarity(
  features1: jnp.ndarray,
  features2: jnp.ndarray,
) -> jnp.ndarray:
  """Cosine similarity between two feature vectors."""
  f1 = features1.reshape(-1)
  f2 = features2.reshape(-1)
  denom = jnp.linalg.norm(f1) * jnp.linalg.norm(f2) + 1e-8
  return jnp.dot(f1, f2) / denom
