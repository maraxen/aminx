# ruff: noqa: S101
"""LASEr ``bb_noise``: a rigid per-residue shift, not per-atom jitter.

Spec 6.3 gives this knob a DIFFERENT rule from Potts ``noise`` and names
iid-per-atom as its own negative control, so the two have to be told apart
rather than merely "applied". Ported from ``utils/pdb_dataset.py:411-413``::

    if protein_training_noise > 0.0:
        noised = torch.round(self.backbone_coords, decimals=2) + (
            protein_training_noise * torch.randn((shape[0], 1, 3), ...))

The draw is ``(L, 1, 3)`` and broadcasts over the atom axis, so a residue's five
backbone atoms all move by the same vector. Rounding to 2 decimals happens
BEFORE the shift, so it is observable even when the shift is not.

Closes debt 2434, which recorded that aminx applied the generic per-atom rule --
the spec's negative control -- on this path.
"""

from __future__ import annotations

import numpy as np
import pytest

from aminx.families.laser_mpnn.driver import apply_backbone_noise, backbone_noise_level


class _Bundle:
  """Minimal stand-in for FeatureNoiseBundle."""

  def __init__(self, feature_type: str, levels: tuple[float, ...], *, enabled: bool = True) -> None:
    self.feature_type = feature_type
    self.noise_levels = levels
    self.enabled = enabled


class _Spec:
  def __init__(self, noise: list[_Bundle]) -> None:
    self.noise = noise


def _coords(length: int = 6) -> np.ndarray:
  """(L, 5, 3) with every atom distinct, and no value on a .005 rounding tie."""
  return (np.arange(length * 5 * 3, dtype=np.float64).reshape(length, 5, 3) * 0.013) + 0.001


def test_knob_semantics_laser_noise() -> None:
  """The shift is per RESIDUE, shared by all five backbone atoms."""
  coords = _coords()
  draws = np.arange(6 * 3, dtype=np.float64).reshape(6, 1, 3) * 0.1 - 0.3
  out = apply_backbone_noise(coords, 0.25, draws=draws)

  expected = np.round(coords, 2) + 0.25 * draws
  np.testing.assert_allclose(out, expected, rtol=0, atol=0)

  # The defining property, stated directly rather than only via `expected`:
  # within a residue, every atom moved by the SAME vector.
  #
  # RECOVERED BY SUBTRACTION, SO NOT EXACT. out[r,a] is rounded[r,a] + k, and
  # subtracting rounded[r,a] back off does not return k bit-for-bit once the
  # two operands differ in magnitude -- measured here at 6.9e-18 between two
  # atoms of one residue. The addition itself IS exact and identical per atom,
  # which `expected` above already pins at atol=0; this loop asserts the same
  # fact in the form a reader would state it, and so has to carry the rounding
  # that the restatement introduces.
  moved = out - np.round(coords, 2)
  scale = float(np.abs(np.round(coords, 2)).max())
  tol = 8.0 * np.finfo(np.float64).eps * max(scale, 1.0)
  for residue in range(coords.shape[0]):
    first = moved[residue, 0]
    for atom in range(1, coords.shape[1]):
      np.testing.assert_allclose(moved[residue, atom], first, rtol=0, atol=tol)

  # ...and different residues moved differently, or "shared" would be vacuous.
  # The gap between residues is 0.1*level = 0.025, four orders above `tol`.
  assert not np.allclose(moved[0, 0], moved[1, 0])
  assert float(np.abs(moved[0, 0] - moved[1, 0]).min()) > 100.0 * tol


def test_knob_semantics_laser_noise_rejects_per_atom_draws() -> None:
  """The negative control spec 6.3 names: iid-per-atom must NOT be accepted.

  A ``(L, 5, 3)`` draw is exactly the generic rule aminx used to apply here
  (debt 2434). Accepting it silently would reproduce that defect, so the shape
  is checked rather than broadcast-or-whatever.
  """
  coords = _coords()
  per_atom = np.zeros((6, 5, 3), dtype=np.float64)
  with pytest.raises(ValueError, match=r"draws must be \(L, 1, 3\)"):
    apply_backbone_noise(coords, 0.25, draws=per_atom)


def test_knob_semantics_laser_noise_rounds_before_shifting() -> None:
  """Rounding to 2 decimals is part of the rule, and precedes the shift.

  Asserted with a zero-magnitude draw so the rounding is the ONLY effect left:
  a port that shifted without rounding would return the input unchanged here,
  and one that rounded afterwards would round the shifted value instead.
  """
  coords = np.full((3, 5, 3), 1.23456, dtype=np.float64)
  out = apply_backbone_noise(coords, 1.0, draws=np.zeros((3, 1, 3)))
  np.testing.assert_allclose(out, np.full((3, 5, 3), 1.23), rtol=0, atol=0)
  assert not np.allclose(out, coords), "rounding must be observable"


def test_knob_semantics_laser_noise_is_off_at_zero() -> None:
  """At level 0 the coordinates are returned untouched -- NOT rounded.

  Upstream guards the whole block with ``if protein_training_noise > 0.0``, so
  the rounding does not happen either. A port that rounded unconditionally would
  perturb every zero-noise run, which is every production run.
  """
  coords = _coords()
  for level in (0.0, -1.0):
    out = apply_backbone_noise(coords, level, draws=np.zeros((6, 1, 3)))
    np.testing.assert_allclose(out, coords, rtol=0, atol=0)


def test_knob_semantics_laser_noise_level_from_spec() -> None:
  """``backbone_noise`` reaches the driver through the bundle list."""
  assert backbone_noise_level(_Spec([_Bundle("backbone", (0.25,))])) == 0.25
  assert backbone_noise_level(_Spec([])) == 0.0
  # A disabled bundle is not a level, and another feature type is not ours.
  assert backbone_noise_level(_Spec([_Bundle("backbone", (0.25,), enabled=False)])) == 0.0
  assert backbone_noise_level(_Spec([_Bundle("vdw", (0.25,))])) == 0.0


def test_knob_semantics_laser_noise_without_draws_or_key_is_a_noop() -> None:
  """No draws and no key must not silently draw unseeded noise."""
  coords = _coords()
  np.testing.assert_allclose(
    apply_backbone_noise(coords, 0.25), coords, rtol=0, atol=0,
  )
