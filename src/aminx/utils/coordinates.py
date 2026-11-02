"""Utility functions for manipulating atomic coordinates.

aminx.utils.coordinates
"""

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from proxide.core.containers import atom_order

from aminx.types.arrays import (
  AlphaCarbonDistance,
  AtomicCoordinate,
  BackboneCoordinates,
  BackboneNoise,
  StructureAtomicCoordinates,
)

_MIN_BACKBONE_ATOMS = 4  # N, CA, C, O


@jax.jit
def apply_noise_to_coordinates(
  key: PRNGKeyArray,
  coordinates: StructureAtomicCoordinates,
  backbone_noise: BackboneNoise,
) -> tuple[StructureAtomicCoordinates, PRNGKeyArray]:
  """Add Gaussian noise to atomic coordinates.

  Args:
    coordinates: Atomic coordinates of the protein structure. (N, 37, 3)
    key: JAX random key for stochastic operations.
    backbone_noise: Standard deviation for Gaussian noise augmentation.

  Returns:
    Tuple of noisy coordinates and the updated JAX random key.

  Example:
    >>> key = jax.random.PRNGKey(0)
    >>> noisy_coords, new_key = apply_noise_to_coordinates(coords, key, 0.1)

  """
  key, coord_key = jax.random.split(key)

  def add_noise(coords: StructureAtomicCoordinates) -> StructureAtomicCoordinates:
    noise = jax.random.normal(coord_key, coords.shape, dtype=coords.dtype)
    scaled_noise = (backbone_noise * noise).astype(coords.dtype)
    return coords + scaled_noise

  def no_noise(coords: StructureAtomicCoordinates) -> StructureAtomicCoordinates:
    return coords

  noisy_coordinates = jax.lax.cond(
    backbone_noise > 0,
    add_noise,
    no_noise,
    coordinates,
  )
  return noisy_coordinates, key


@jax.jit
def extract_backbone_atoms(
  coordinates: StructureAtomicCoordinates,
  *,
  oxygen_index: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Extract N, CA, C, O coordinates in a layout-aware, backend-safe manner.

  Parameters
  ----------
  coordinates : StructureAtomicCoordinates
      Atomic coordinate array of shape ``(..., A, 3)`` where ``A >= 4``.
  oxygen_index : int | None, optional
      Explicit column index for Oxygen. If None (default), auto-resolved from shape
      and geometry.

  Returns
  -------
  tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
      Tuple of ``(nitrogen, alpha_carbon, carbon, oxygen)``, each shape ``(..., 3)``.
  """
  num_atoms = coordinates.shape[-2]
  if num_atoms < _MIN_BACKBONE_ATOMS:
    msg = (
      f"coordinates must have at least {_MIN_BACKBONE_ATOMS} atoms per residue (N, CA, C, O); "
      f"got atom axis of size {num_atoms} in shape {coordinates.shape}"
    )
    raise ValueError(msg)

  nitrogen = coordinates[..., 0, :]
  alpha_carbon = coordinates[..., 1, :]
  carbon = coordinates[..., 2, :]

  if oxygen_index is not None:
    oxygen = coordinates[..., oxygen_index, :]
  elif num_atoms == 4:
    # Compact layout [N, CA, C, O]
    oxygen = coordinates[..., 3, :]
  elif num_atoms in (14, 23):
    # Atom14 (OpenFold/FAMPNN) and Atom23 (Caliby): Oxygen is at index 3
    oxygen = coordinates[..., 3, :]
  elif num_atoms == 5:
    # Ambiguous 5-atom layout:
    # Option A: PDB order [N, CA, C, O, CB] -> O is at index 3
    # Option B: Atom37 first 5 [N, CA, C, CB, O] -> O is at index 4
    # Peptide carbonyl invariant: C=O (~1.23 A) vs C-CB (~2.45 A)
    cand3 = coordinates[..., 3, :]
    cand4 = coordinates[..., 4, :]
    d3_sq = jnp.sum((carbon - cand3) ** 2, axis=-1, keepdims=True)
    d4_sq = jnp.sum((carbon - cand4) ** 2, axis=-1, keepdims=True)
    oxygen = jnp.where(d3_sq <= d4_sq, cand3, cand4)
  elif num_atoms == 37:
    # Standard atom37 (AlphaFold / proxide): atom_order["O"] == 4
    oxygen = coordinates[..., atom_order["O"], :]
  else:
    # General fallback for layout with >= 5 atoms: disambiguate via carbonyl distance
    cand3 = coordinates[..., 3, :]
    cand4 = coordinates[..., 4, :]
    d3_sq = jnp.sum((carbon - cand3) ** 2, axis=-1, keepdims=True)
    d4_sq = jnp.sum((carbon - cand4) ** 2, axis=-1, keepdims=True)
    oxygen = jnp.where(d3_sq <= d4_sq, cand3, cand4)

  return nitrogen, alpha_carbon, carbon, oxygen


@jax.jit
def compute_backbone_coordinates(
  coordinates: StructureAtomicCoordinates,
  *,
  oxygen_index: int | None = None,
) -> BackboneCoordinates:
  """Compute backbone coordinates with per-residue C-beta handling.

  Parameters
  ----------
  coordinates : StructureAtomicCoordinates
      Atomic coordinates of the protein structure, shape ``(..., A, 3)`` where ``A >= 4``.
      Accepts 4-atom (N, CA, C, O), 5-atom (PDB or atom37), 14-atom, 23-atom, and 37-atom layouts.
  oxygen_index : int | None, optional
      Explicit index of the Oxygen atom along the atom axis. Default: None (auto-detect).

  Returns
  -------
  BackboneCoordinates
      Backbone coordinates with C-beta computed where necessary, shape ``(..., 5, 3)``
      in canonical PDB order: ``(N, CA, C, O, CB)``.
  """
  nitrogen, alpha_carbon, carbon, oxygen = extract_backbone_atoms(
    coordinates, oxygen_index=oxygen_index
  )

  alpha_to_nitrogen = alpha_carbon - nitrogen
  carbon_to_alpha = carbon - alpha_carbon
  beta_carbon = compute_c_beta(alpha_to_nitrogen, carbon_to_alpha, alpha_carbon)

  return jnp.stack(
    [nitrogen, alpha_carbon, carbon, oxygen, beta_carbon],
    axis=-2,
  )



@jax.jit
def compute_c_beta(
  alpha_to_nitrogen: AtomicCoordinate,
  carbon_to_alpha: AtomicCoordinate,
  alpha_carbon: AtomicCoordinate,
) -> AtomicCoordinate:
  """Compute C-beta coordinates.

  Uses a linear combination of the bond vectors to estimate C-beta.

  Coefficients are derived from empirical data and are used to ensure
  that the C-beta coordinates are consistent with the geometry of the protein backbone.

  Args:
    alpha_to_nitrogen: Bond vector from nitrogen to alpha carbon.
    carbon_to_alpha: Bond vector from alpha carbon to carbon.
    alpha_carbon: Coordinates of the alpha carbon atom.

  Returns:
    C-beta coordinates as an AtomicCoordinate.

  Example:
    >>> n_to_ca = jnp.array([1.0, 0.0, 0.0])
    >>> ca_to_c = jnp.array([0.0, 1.0, 0.0])
    >>> ca_coords = jnp.array([0 .0, 0.0, 0.0])
    >>> cb_coords = compute_c_beta(n_to_ca, ca_to_c, ca_coords)
    >>> cb_coords.shape
    (3,)

  """
  f1, f2, f3 = -0.58273431, 0.56802827, -0.54067466
  term1 = f1 * jnp.cross(alpha_to_nitrogen, carbon_to_alpha)
  term2 = f2 * alpha_to_nitrogen
  term3 = f3 * carbon_to_alpha
  return term1 + term2 + term3 + alpha_carbon


@jax.jit
def compute_backbone_distance(backbone_coordinates: BackboneCoordinates) -> AlphaCarbonDistance:
  """Compute pairwise distances between backbone atoms.

  Calculate the Euclidean distance between all pairs of backbone atom coordinates based on alpha
  carbon positions.

  Assumes backbone_coordinates is a 3D array of shape (N, 5, 3), where N is the number of atoms,
  5 is the number of backbone atoms (N, CA, C, O, N), and 3 is the spatial dimension (x, y, z).

  Args:
    backbone_coordinates: A 3D array of shape (N, 5, 3) representing the coordinates of backbone
    atoms.

  Returns:
    A 2D array of shape (N, N) containing the pairwise distances between backbone atoms.

  Example:
    >>> coords = jnp.zeros((10, 5, 3))  # Example coordinates
    >>> distances = compute_backbone_distance(coords)
    >>> distances.shape
    (10, 10)

  """
  alpha_coordinates = backbone_coordinates[:, atom_order["CA"], :]
  return jnp.sqrt(
    1e-6
    + jnp.sum(
      jnp.square(alpha_coordinates[:, None, :] - alpha_coordinates[None, :, :]),
      axis=-1,
    ),
  )


@jax.jit
def extend_coordinate(
  atom_a: AtomicCoordinate,
  atom_b: AtomicCoordinate,
  atom_c: AtomicCoordinate,
  bond_length: float,
  bond_angle: float,
  dihedral_angle: float,
) -> AtomicCoordinate:
  """Compute the position of a fourth atom (D) given three atoms (A, B, C) and internal coordinates.

  Given coordinates for atoms A, B, and C, and the desired bond length, bond angle, and dihedral
  angle, compute the coordinates of atom D such that:
    - |C-D| = bond_length
    - angle(B, C, D) = bond_angle
    - dihedral(A, B, C, D) = dihedral_angle

  Args:
    atom_a: Coordinates of atom A, shape (3,).
    atom_b: Coordinates of atom B, shape (3,).
    atom_c: Coordinates of atom C, shape (3,).
    bond_length: Desired bond length between C and D.
    bond_angle: Desired bond angle (in radians) at atom C.
    dihedral_angle: Desired dihedral angle (in radians) for atoms A-B-C-D.

  Returns:
    Coordinates of atom D, shape (3,).

  Example:
    >>> d = extend_coordinate(a, b, c, 1.5, 2.0, 3.14)
    >>> d.shape
    (3,)

  """

  def normalize(vec: AtomicCoordinate) -> AtomicCoordinate:
    return vec / jnp.linalg.norm(vec)

  bc = normalize(atom_b - atom_c)
  normal = normalize(jnp.cross(atom_b - atom_a, bc))
  term1 = bond_length * jnp.cos(bond_angle) * bc
  term2 = bond_length * jnp.sin(bond_angle) * jnp.cos(dihedral_angle) * jnp.cross(normal, bc)
  term3 = bond_length * jnp.sin(bond_angle) * jnp.sin(dihedral_angle) * -normal
  return atom_c + term1 + term2 + term3


@jax.jit
def compute_cb_precise(
  n_coord: AtomicCoordinate,
  ca_coord: AtomicCoordinate,
  c_coord: AtomicCoordinate,
) -> AtomicCoordinate:
  """Compute the C-beta atom position from backbone N, CA, and C coordinates.

  Does so precisely using trigonometric relationships based on the backbone geometry.

  Specifically, the position of the C-beta atom is determined by:

  - The bond length between the alpha carbon and the C-beta atom.
  - The bond angle between the nitrogen, alpha carbon, and C-beta atoms.
  - The dihedral angle involving the nitrogen, alpha carbon, and C-beta atoms.


  Unlike the compute_c_beta function, this function does not use a linear combination of bond
  vectors with approximate fixed coefficients. This is more accurate and flexible for different
  configurations of the protein backbone, but more computationally intensive.

  It is used in preparation of the atomic coordinates for the model input.
  It is not used in the model itself, but rather in the preprocessing of the input data
  to ensure that the C-beta atom is correctly placed based on the backbone structure.

  Uses standard geometry for C-beta placement:
    - N-CA-CB bond length: 1.522 Å
    - N-CA-CB bond angle: 1.927 radians
    - C-N-CA-CB dihedral angle: -2.143 radians

  Args:
    n_coord: Coordinates of the N atom, shape (3,).
    ca_coord: Coordinates of the CA atom, shape (3,).
    c_coord: Coordinates of the C atom, shape (3,).

  Returns:
    Coordinates of the C-beta atom, shape (3,).

  Example:
    >>> cb = compute_cb_precise(n, ca, c)
    >>> cb.shape
    (3,)

  """
  return extend_coordinate(
    c_coord,
    n_coord,
    ca_coord,
    bond_length=1.522,
    bond_angle=1.927,
    dihedral_angle=-2.143,
  )
