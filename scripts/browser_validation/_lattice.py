"""Deterministic cubic-lattice CA coordinate generator for genuine k-NN sort ties.

On a cubic grid, most points have several neighbors at IDENTICAL Euclidean
distance -- e.g. 6 axis-neighbors at exactly ``spacing``, 12 face-diagonal
neighbors at ``spacing * sqrt(2)``, 8 corner neighbors at ``spacing *
sqrt(3)`` -- so selecting the ``k`` nearest via ``aminx.model.features.top_k``
genuinely exercises its tie-break rule. Random coordinates give ties
probability zero, so they cannot test this at all.

Reused by T2 (``jax2onnx_spike.py``'s multi-key sort tie test, Phase 0) and T6
(``fixtures.py``'s ``tie_lattice`` fixture, Phase 1).
"""

from __future__ import annotations

import math

import jax.numpy as jnp


def cubic_lattice_ca(n: int, spacing: float = 1.0) -> jnp.ndarray:
  """Return ``(n, 3)`` CA coordinates on a ``spacing``-Angstrom cubic grid.

  Points are the first ``n`` integer lattice sites ``(i, j, k) * spacing``,
  enumerated in a fixed nested-loop order (``k`` fastest, then ``j``, then
  ``i``) over a cube of side ``ceil(n ** (1/3)) + 1`` -- always enough
  candidate sites for the requested ``n``. The enumeration is a closed-form
  Python computation with no RNG, so the returned coordinates are
  byte-identical across runs, processes, and platforms.

  Args:
    n: Number of CA coordinates to generate. Must be >= 1.
    spacing: Grid spacing in Angstrom between adjacent lattice points along
      one axis.

  Returns:
    ``(n, 3)`` float32 array of CA coordinates.

  Raises:
    ValueError: If ``n < 1``.
  """
  if n < 1:
    msg = f"n must be >= 1, got {n}"
    raise ValueError(msg)
  side = math.ceil(n ** (1.0 / 3.0)) + 1
  points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
  points = points[:n]
  return jnp.asarray(points, dtype=jnp.float32) * spacing


def lattice_neighbor_logits(coords: jnp.ndarray) -> jnp.ndarray:
  """Negated pairwise CA distance matrix, matching aminx's own distance convention.

  Mirrors ``aminx.utils.coordinates.compute_backbone_distance``'s
  ``sqrt(1e-6 + sum_of_squares)`` epsilon, so this is the same kind of array
  the real featurizer would sort over -- not an idealized zero-diagonal
  distance matrix. Negated because ``aminx.model.features.top_k`` selects the
  ``k`` LARGEST entries (nearest = smallest distance = largest negative
  distance).

  Args:
    coords: ``(n, 3)`` coordinates, e.g. from ``cubic_lattice_ca``.

  Returns:
    ``(n, n)`` float32 array ``-distance``.
  """
  diff = coords[:, None, :] - coords[None, :, :]
  distances = jnp.sqrt(1e-6 + jnp.sum(jnp.square(diff), axis=-1))
  return -distances
