"""Radial basis functions for distance encoding.

aminx.utils.radial_basis
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from aminx.types.arrays import AtomIndexPair, NeighborIndices, StructureAtomicCoordinates

AllAtomRBF = Float[Array, "L K R D"]
AtomPairRBF = Float[Array, "K R D"]

RADIAL_BASES = 16
RADIAL_BASE_MINIMUM, RADIAL_BASE_MAXIMUM = 2.0, 22.0
RBF_CENTERS = jnp.linspace(RADIAL_BASE_MINIMUM, RADIAL_BASE_MAXIMUM, RADIAL_BASES)


def rbf_centers(dtype: jnp.dtype) -> jax.Array:
  """RBF centres built in ``dtype`` at call time.

  ``RBF_CENTERS`` is materialised at import, so it is float32 whenever
  ``jax_enable_x64`` is switched on after import; a float64 forward then uses
  float32-rounded centres. Float32 values are identical to ``RBF_CENTERS``.
  """
  return jnp.linspace(RADIAL_BASE_MINIMUM, RADIAL_BASE_MAXIMUM, RADIAL_BASES, dtype=dtype)


RBF_SIGMA = (RADIAL_BASE_MAXIMUM - RADIAL_BASE_MINIMUM) / RADIAL_BASES

BACKBONE_PAIRS = jnp.array(
  [
    [1, 1],
    [0, 0],
    [2, 2],
    [3, 3],
    [4, 4],
    [1, 0],
    [1, 2],
    [1, 3],
    [1, 4],
    [0, 2],
    [0, 3],
    [0, 4],
    [4, 2],
    [4, 3],
    [3, 2],
    [0, 1],
    [2, 1],
    [3, 1],
    [4, 1],
    [2, 0],
    [3, 0],
    [4, 0],
    [2, 4],
    [3, 4],
    [2, 3],
  ],
)

DISTANCE_EPSILON = 1e-6


@jax.jit
def compute_radial_basis(
  backbone_coordinates: StructureAtomicCoordinates,
  neighbor_indices: NeighborIndices,
) -> AllAtomRBF:
  """Compute the radial basis functions for backbone coordinates."""

  def _rbf(pair: AtomIndexPair, neighbor_indices: NeighborIndices) -> AtomPairRBF:
    """Compute the radial basis function for a given pair of atoms."""
    atom1, atom2 = backbone_coordinates[:, pair[0], :], backbone_coordinates[:, pair[1], :]
    delta_coords = atom1[:, None, :] - atom2[None, :, :]
    distance_sq = jnp.sum(jnp.square(delta_coords), axis=-1)
    distance = jnp.sqrt(DISTANCE_EPSILON + distance_sq)
    neighbor_distances = jnp.take_along_axis(distance, neighbor_indices, axis=1)
    return jnp.exp(
      -(
        jnp.square(
          (neighbor_distances[..., None] - rbf_centers(neighbor_distances.dtype)) / RBF_SIGMA
        )
      ),
    )

  return (
    jax.vmap(lambda pair: _rbf(pair, neighbor_indices))(BACKBONE_PAIRS)
    .transpose((1, 2, 0, 3))
    .reshape(
      backbone_coordinates.shape[0],
      neighbor_indices.shape[1],
      -1,
    )
  )
