"""Continuous, off-lattice HP (hydrophobic-polar) contact energy.

Tier 1 of the HP-model coarse-init exploration (see
``scripts/validate/hp_lattice_sanity.py``, Tier 0 -- the discrete Lau & Dill
(1989) lattice-model reproduction this module is checked against for
*qualitative*, not numeric, consistency in ``tests/ebm/test_hp_energy.py``;
2D lattice and continuous 3D energies are not numerically comparable).

This is a smooth relaxation of the lattice model's own rule: every H-H
"topological neighbor" contact (non-sequence-adjacent, spatially close)
lowers the energy; every other contact type is inert. On a continuous CA
backbone there is no discrete lattice-adjacency notion, so "contact" becomes
a smooth, differentiable indicator in CA-CA distance -- a sigmoid rather
than a step function, so this composes with gradient-based samplers (the
existing Langevin loop in ``aminx.ebm.langevin``/``structure_prediction``)
without a non-differentiable kink.

Not yet wired into ``ProteinEBMModel``/``plan.py``'s dispatch axis -- the
synthetic two-domain windowing/consensus check (Tier 2, for chains beyond
the ~150-200 residue HP-sufficiency range per Lin & Zewail 2012) is
explicitly deferred, tracked as follow-on work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool

from aminx.utils.aa_convert import MPNN_ALPHABET

if TYPE_CHECKING:
  from aminx.ebm.contracts import AAType, Coords, Energy, ResidueMask

HPLabels = Bool[Array, "N"]
"""Per-residue H(True)/P(False) classification -- shape-identical to ``ResidueMask``, kept
as a distinct alias since the two are semantically different (hydrophobicity vs. validity)."""

# Same textbook hydrophobic-core set aminx.ebm.ddg_stability's
# identify_buried_hydrophobic_positions uses (Kyte-Doolittle "hydrophobic"
# tier) -- mirrored rather than imported, since that module's constant is a
# private implementation detail of a SASA-gated heuristic, not a standalone
# sequence->HP export.
_HYDROPHOBIC_LETTERS = frozenset("AVLIMFWY")

_HP_LOOKUP = jnp.array(
  [letter in _HYDROPHOBIC_LETTERS for letter in MPNN_ALPHABET],
  dtype=bool,
)
"""Per-alphabet-index H/P lookup, ``MPNN_ALPHABET`` order (mask token 20 -> False/polar)."""

# CA-CA contact cutoff: real ~8 Angstrom (a common coarse contact-map
# convention), scaled by aminx's coordinate_scaling=0.1 -- see
# aminx.ebm.ddg_stability's _COORDINATE_SCALING / _DEFAULT_UNFOLDED_STEP_LENGTH
# comment: 3.8 Angstrom real CA-CA bond == 0.38 scaled.
_DEFAULT_CUTOFF = 0.8

# Sigmoid steepness (1 / scaled-length-units); controls how sharply the
# contact indicator transitions from ~1 (in contact) to ~0 (not in contact)
# around _DEFAULT_CUTOFF.
_DEFAULT_SHARPNESS = 20.0

_DEFAULT_EPSILON = 1.0

__all__ = ["hp_contact_energy", "hp_labels_from_aatype"]


def hp_labels_from_aatype(aatype: AAType) -> HPLabels:
  """Classify each residue as hydrophobic (True) or polar (False).

  Uses ``aminx.utils.aa_convert.MPNN_ALPHABET``'s index ordering (the same
  alphabet ``AAType``'s ``[0, 20]`` indices are defined against) and the
  Kyte-Doolittle "hydrophobic" tier (mirrors
  ``aminx.ebm.ddg_stability._HYDROPHOBIC_LETTERS``). The mask token (index
  20, ``"X"``) classifies as polar; masked-out residues are excluded from
  the energy sum via ``ResidueMask`` regardless, so this choice is inert,
  not a physical claim.
  """
  return _HP_LOOKUP[aatype]


def hp_contact_energy(
  coords: Coords,
  hp_labels: HPLabels,
  mask: ResidueMask,
  *,
  cutoff: float = _DEFAULT_CUTOFF,
  sharpness: float = _DEFAULT_SHARPNESS,
  epsilon: float = _DEFAULT_EPSILON,
) -> Energy:
  """Continuous relaxation of the Lau & Dill (1989) H-H contact energy.

  ``E = -epsilon * sum_{i<j, |i-j|>1} mask_i * mask_j * hp_i * hp_j *
  sigmoid(sharpness * (cutoff - dist(i, j)))``

  The sigmoid replaces the lattice model's binary "topological neighbor"
  indicator; the ``|i - j| > 1`` exclusion mirrors the same
  sequence-adjacency exclusion (covalently bonded neighbors don't count as
  contacts). ``epsilon > 0`` lowers (more negative) energy for closer, more
  numerous H-H contacts -- the source paper writes this as ``epsilon < 0``
  directly on the contact count; here it's a positive scale factor on an
  explicit minus sign, to keep the public API's sign convention unambiguous
  (energy decreases as compact H-cores form).
  """
  diffs = coords[:, None, :] - coords[None, :, :]
  dist = jnp.linalg.norm(diffs, axis=-1)

  n = coords.shape[0]
  residue_index = jnp.arange(n)
  sequence_adjacent = jnp.abs(residue_index[:, None] - residue_index[None, :]) <= 1

  pair_mask = mask[:, None] & mask[None, :] & hp_labels[:, None] & hp_labels[None, :]
  pair_mask = pair_mask & ~sequence_adjacent

  contact = jax.nn.sigmoid(sharpness * (cutoff - dist))
  # Each pair counted twice (i, j) and (j, i); halve to match the lattice
  # model's per-pair (not per-ordered-pair) contact count.
  return -epsilon * 0.5 * jnp.sum(jnp.where(pair_mask, contact, 0.0))
