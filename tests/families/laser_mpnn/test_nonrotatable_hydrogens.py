# ruff: noqa: S101
"""CPU checks for non-rotatable hydrogen placement. No oracle dumps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("prody", reason="needs the laser extra")

from aminx.families.laser_mpnn.featurize import (
  MAX_ATOMS,
  MAX_PROTONATED_ATOMS,
  build_rotamers,
)

_GOLDEN = Path(__file__).with_name("rotamer_heavy_golden.npz")
# Ideal protonated coordinates (not a textbook force field): polar N-H is about
# 0.86 A, aromatic C-H about 0.93 A, aliphatic C-H about 0.97 A.
_BOND_MIN_A = 0.85
_BOND_MAX_A = 1.10
# The golden reproduces to 0 ULP on the machine that wrote it, but numpy/scipy pick
# float kernels per CPU, so GitHub's runner differs in the last bits (aminx #165 CI) with
# identical numpy 2.4.6 / scipy 1.17.1. Measured on this golden: the f32 build sits at most
# 8.8e-6 A from the f64 build (p99 7.7e-6). The f32 band is >10x that scale; any real builder
# change (a shifted slot, a wrong frame) moves atoms by >= ~0.1 A. Slot occupancy (the NaN
# mask) is still compared exactly -- that is the pre-change invariant.
_GOLDEN_ATOL_A = {np.dtype(np.float32): 1e-4, np.dtype(np.float64): 1e-9}


def _load() -> np.lib.npyio.NpzFile:
  return np.load(_GOLDEN)


def test_default_matches_prechange_heavy_atoms() -> None:
  """``add_nonrotatable_hydrogens=False`` matches the pre-change builder."""
  data = _load()
  sequence = np.asarray(data["sequence"])
  for dtype, key in ((np.float32, "coords32"), (np.float64, "coords64")):
    got = build_rotamers(
      np.asarray(data["backbone"], dtype=dtype),
      np.asarray(data["chi"], dtype=dtype),
      sequence,
      dtype,
    )
    explicit = build_rotamers(
      np.asarray(data["backbone"], dtype=dtype),
      np.asarray(data["chi"], dtype=dtype),
      sequence,
      dtype,
      add_nonrotatable_hydrogens=False,
    )
    expected = np.asarray(data[key])
    assert got.shape == (sequence.shape[0], MAX_ATOMS, 3)
    assert got.dtype == expected.dtype
    atol = _GOLDEN_ATOL_A[expected.dtype]
    for built in (got, explicit):
      assert np.array_equal(np.isnan(built), np.isnan(expected))
      np.testing.assert_allclose(built, expected, rtol=0, atol=atol, equal_nan=True)


def test_hydrogens_keep_heavy_slots_and_bond_lengths() -> None:
  """Protonated builds keep placed heavy atoms and sit on X-H bonds."""
  data = _load()
  dtype = np.float64
  backbone = np.asarray(data["backbone"], dtype=dtype)
  chi = np.asarray(data["chi"], dtype=dtype)
  sequence = np.asarray(data["sequence"])
  heavy = build_rotamers(backbone, chi, sequence, dtype)
  prot = build_rotamers(
    backbone,
    chi,
    sequence,
    dtype,
    add_nonrotatable_hydrogens=True,
  )
  assert prot.shape == (sequence.shape[0], MAX_PROTONATED_ATOMS + 1, 3)
  assert prot.dtype == dtype
  present = np.isfinite(heavy)
  assert np.array_equal(prot[:, :MAX_ATOMS][present], heavy[present])
  new = np.zeros(prot.shape[:2], dtype=bool)
  new[:, :MAX_ATOMS] = ~np.isfinite(heavy[..., 0]) & np.isfinite(prot[:, :MAX_ATOMS, 0])
  new[:, MAX_ATOMS:] = np.isfinite(prot[:, MAX_ATOMS:, 0])
  assert int(new.sum()) > 0
  for row in range(sequence.shape[0]):
    parents = heavy[row][np.isfinite(heavy[row, :, 0])]
    placed = prot[row][new[row]]
    assert placed.shape[0] > 0
    delta = placed[:, None, :] - parents[None, :, :]
    nearest = np.linalg.norm(delta, axis=-1).min(axis=-1)
    assert np.all(nearest >= _BOND_MIN_A)
    assert np.all(nearest <= _BOND_MAX_A)
