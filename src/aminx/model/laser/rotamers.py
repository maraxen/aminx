"""Post-decode sidechain placement.

Ports ``RotamerBuilder.build_rotamers`` (``utils/build_rotamers.py``). Heavy
atoms come from :func:`aminx.families.laser_mpnn.featurize.build_rotamers`.
Non-rotatable hydrogens are Kabsch-aligned from the ideal protonated table
onto those heavy atoms, which is what makes the oracle frame ``(L, 24, 3)``.

The geometry helpers stay host NumPy. This runs after decode, once per
design, and the ideal-geometry SVD is the same host path featurize already
uses. Tracing it would put that SVD on the inference graph.

Math:
    For each hydrogen triad $(\\mathbf{a}, \\mathbf{b}, \\mathbf{c})$ and ideal
    hydrogen $\\mathbf{h}$, the rigid map $(R, \\mathbf{t})$ taking the ideal
    triad onto the built triad is applied as
    $\\mathbf{h}' = (\\mathbf{h} - \\bar{\\mathbf{m}}) R + \\bar{\\mathbf{f}}$.
    Atom slots a residue does not have stay NaN.

Pseudocode:
    heavy = build_rotamers(backbone, chi, sequence)   # (L, 14, 3), NaN-padded
    if not hydrogens: return heavy
    coords = pad heavy with NaN to (L, 24, 3)
    for each alignment triad:
        rows = residues that own a hydrogen on this triad
        R, mobile_com, fixed_com = kabsch(built triad, ideal triad)
        coords[rows, hydrogen] = apply(ideal hydrogen, R, mobile_com, fixed_com)
    return coords
"""

from __future__ import annotations

import functools

import numpy as np
from jaxtyping import Float, Int
from numpy.typing import NDArray

from aminx.families.laser_mpnn.featurize import (
  _ATOM_ORDER,
  _G,
  LASER_ALPHABET,
  LaserInputError,
  _apply,
  _as_float,
  _gather,
  _kabsch,
  _tables,
)
from aminx.families.laser_mpnn.featurize import (
  build_rotamers as _build_heavy,
)

# Working precision follows the caller. float32 is the production default;
# float64 is the parity opt-in and never a traced cast.
_FloatDtype = np.dtype[np.floating] | type[np.floating]

# Upstream pads heavy atoms (14) out to MAX_NUM_PROTONATED_RESIDUE_ATOMS + 1.
_HEAVY_SLOTS = 14
_PROTONATED_SLOTS = 24
# Index 23 is the NaN pad. Real hydrogens never land there.
_H_SENTINEL = 23

# Hydrogens appended after the heavy-atom order. X is alanine's list in the
# writer (upstream ``x_placeholder``) even though the builder's index row for
# X is glycine's: the coordinate slots and the PDB names are different tables.
_EXTRA_H: dict[str, tuple[str, ...]] = {
  "G": ("HA2", "HA3"),
  "A": ("HA", "HB1", "HB2", "HB3"),
  "S": ("HA", "HB2", "HB3"),
  "C": ("HA", "HB2", "HB3"),
  "T": ("HA", "HB", "HG21", "HG22", "HG23"),
  "P": ("HA", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3"),
  "V": ("HA", "HB", "HG11", "HG12", "HG13", "HG21", "HG22", "HG23"),
  "M": ("HA", "HB2", "HB3", "HG2", "HG3", "HE1", "HE2", "HE3"),
  "N": ("HA", "HB2", "HB3", "HD21", "HD22"),
  "I": ("HA", "HB", "HG21", "HG22", "HG23", "HD11", "HD12", "HD13", "HG12", "HG13"),
  "L": ("HA", "HB2", "HB3", "HG", "HD11", "HD12", "HD13", "HD21", "HD22", "HD23"),
  "D": ("HA", "HB2", "HB3"),
  "E": ("HA", "HB2", "HB3", "HG2", "HG3"),
  "K": ("HA", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3", "HE2", "HE3", "HZ1", "HZ2", "HZ3"),
  "Q": ("HA", "HB2", "HB3", "HG2", "HG3", "HE21", "HE22"),
  "H": ("HA", "HB2", "HB3", "HE1", "HD2"),
  "F": ("HA", "HB2", "HB3", "HD1", "HD2", "HE1", "HE2", "HZ"),
  "R": ("HA", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3", "HE", "HH21", "HH22", "HH12", "HH11"),
  "Y": ("HA", "HB2", "HB3", "HD1", "HD2", "HE1", "HE2"),
  "W": ("HA", "HB2", "HB3", "HD1", "HE1", "HE3", "HZ2", "HZ3", "HH2"),
  "X": ("HA", "HB1", "HB2", "HB3"),
}

_ALIGN: dict[str, tuple[tuple[tuple[str, str, str], tuple[str, ...]], ...]] = {
  "G": ((("N", "CA", "C"), ("HA2", "HA3")),),
  "A": ((("N", "CA", "CB"), ("HA", "HB1", "HB2", "HB3")),),
  "S": (
    (("N", "CA", "CB"), ("HA",)),
    (("CA", "CB", "OG"), ("HB2", "HB3")),
  ),
  "C": (
    (("N", "CA", "CB"), ("HA",)),
    (("CA", "CB", "SG"), ("HB2", "HB3")),
  ),
  "T": (
    (("N", "CA", "CB"), ("HA",)),
    (("CA", "CB", "CG2"), ("HB", "HG21", "HG22", "HG23")),
  ),
  "P": (
    (("N", "CA", "CB"), ("HA",)),
    (("CA", "CB", "CG"), ("HB2", "HB3")),
    (("CB", "CG", "CD"), ("HG2", "HG3")),
    (("CG", "CD", "N"), ("HD2", "HD3")),
  ),
  "V": (
    (("N", "CA", "CB"), ("HA",)),
    (("CG2", "CB", "CG1"), ("HB", "HG11", "HG12", "HG13", "HG21", "HG22", "HG23")),
  ),
  "M": (
    (("N", "CA", "CB"), ("HA",)),
    (("CA", "CB", "CG"), ("HB2", "HB3")),
    (("CB", "CG", "SD"), ("HG2", "HG3")),
    (("CG", "SD", "CE"), ("HE1", "HE2", "HE3")),
  ),
  "N": (
    (("N", "CA", "CB"), ("HA",)),
    (("CA", "CB", "CG"), ("HB2", "HB3")),
    (("OD1", "CG", "ND2"), ("HD21", "HD22")),
  ),
  "I": (
    (("N", "CA", "CB"), ("HA",)),
    (("CG1", "CB", "CG2"), ("HB", "HG21", "HG22", "HG23")),
    (("CB", "CG1", "CD1"), ("HG12", "HG13", "HD11", "HD12", "HD13")),
  ),
  "L": (
    (("N", "CA", "CB"), ("HA",)),
    (("CA", "CB", "CG"), ("HB2", "HB3")),
    (("CD1", "CG", "CD2"), ("HG", "HD11", "HD12", "HD13", "HD21", "HD22", "HD23")),
  ),
  "D": (
    (("N", "CA", "CB"), ("HA",)),
    (("CA", "CB", "CG"), ("HB2", "HB3")),
  ),
  "E": (
    (("N", "CA", "CB"), ("HA",)),
    (("CA", "CB", "CG"), ("HB2", "HB3")),
    (("CB", "CG", "CD"), ("HG2", "HG3")),
  ),
  "K": (
    (("N", "CA", "CB"), ("HA",)),
    (("CA", "CB", "CG"), ("HB2", "HB3")),
    (("CB", "CG", "CD"), ("HG2", "HG3")),
    (("CG", "CD", "CE"), ("HD2", "HD3")),
    (("CD", "CE", "NZ"), ("HE2", "HE3", "HZ1", "HZ2", "HZ3")),
  ),
  "Q": (
    (("N", "CA", "CB"), ("HA",)),
    (("CA", "CB", "CG"), ("HB2", "HB3")),
    (("CB", "CG", "CD"), ("HG2", "HG3")),
    (("OE1", "CD", "NE2"), ("HE21", "HE22")),
  ),
  "H": (
    (("N", "CA", "CB"), ("HA",)),
    (("CA", "CB", "CG"), ("HB2", "HB3")),
    (("CE1", "NE2", "CD2"), ("HE1", "HD2")),
  ),
  "F": (
    (("N", "CA", "CB"), ("HA",)),
    (("CA", "CB", "CG"), ("HB2", "HB3")),
    (("CE1", "CZ", "CE2"), ("HD1", "HD2", "HE1", "HE2", "HZ")),
  ),
  "R": (
    (("N", "CA", "CB"), ("HA",)),
    (("CA", "CB", "CG"), ("HB2", "HB3")),
    (("CB", "CG", "CD"), ("HG2", "HG3")),
    (("CG", "CD", "NE"), ("HD2", "HD3")),
    (("NE", "CZ", "NH2"), ("HE", "HH21", "HH22", "HH12", "HH11")),
  ),
  "Y": (
    (("N", "CA", "CB"), ("HA",)),
    (("CA", "CB", "CG"), ("HB2", "HB3")),
    (("CE1", "CZ", "CE2"), ("HE1", "HE2", "HD1", "HD2")),
  ),
  "W": (
    (("N", "CA", "CB"), ("HA",)),
    (("CA", "CB", "CG"), ("HB2", "HB3")),
    (("CD1", "CG", "CD2"), ("HD1", "HE1", "HE3", "HZ2", "HZ3", "HH2")),
  ),
}

_LONG = {
  "A": "ALA",
  "R": "ARG",
  "N": "ASN",
  "D": "ASP",
  "C": "CYS",
  "E": "GLU",
  "Q": "GLN",
  "G": "GLY",
  "H": "HIS",
  "I": "ILE",
  "L": "LEU",
  "K": "LYS",
  "M": "MET",
  "F": "PHE",
  "P": "PRO",
  "S": "SER",
  "T": "THR",
  "W": "TRP",
  "Y": "TYR",
  "V": "VAL",
  "X": "XAA",
}


def _extended_names(letter: str) -> tuple[str, ...]:
  """PDB atom order. X reuses alanine's names, matching upstream's walrus."""
  if letter == "X":
    return _ATOM_ORDER["A"] + _EXTRA_H["A"]
  return _ATOM_ORDER[letter] + _EXTRA_H[letter]


def _builder_names(letter: str) -> tuple[str, ...]:
  """Names whose indices the hydrogen alignment tables are built from.

  Sequence index 20 (X) is glycine's row in ``RotamerBuilder``, not alanine's.
  """
  if letter == "X":
    return _ATOM_ORDER["G"] + _EXTRA_H["G"]
  return _ATOM_ORDER[letter] + _EXTRA_H[letter]


@functools.cache
def _alignment_index() -> tuple[NDArray[np.int64], NDArray[np.int64]]:
  """Triad and hydrogen indices, shape ``(21, T, 3)`` and ``(21, T, H)``."""
  letters = LASER_ALPHABET[:20]
  n_triad = max(len(_ALIGN[letter]) for letter in letters)
  n_h = max(len(hydrogens) for letter in letters for _, hydrogens in _ALIGN[letter])
  triads = np.full((21, n_triad, 3), _H_SENTINEL, dtype=np.int64)
  hydrogens = np.full((21, n_triad, n_h), _H_SENTINEL, dtype=np.int64)
  for index, letter in enumerate(letters):
    order = _builder_names(letter)
    for triad_i, (triad, names) in enumerate(_ALIGN[letter]):
      triads[index, triad_i] = [order.index(name) for name in triad]
      hydrogens[index, triad_i, : len(names)] = [order.index(name) for name in names]
  # X copies glycine, the same cat() RotamerBuilder does for index 20.
  triads[20] = triads[_G]
  hydrogens[20] = hydrogens[_G]
  return triads, hydrogens


def _ideal_protonated(dtype: _FloatDtype) -> NDArray[np.floating]:
  """Ideal protonated coords, ``(21, 24, 3)``, last row glycine for X."""
  raw = _as_float(_tables(dtype)["ideal_prot"], dtype)
  pad = np.full((raw.shape[0], 1, 3), np.nan, dtype=dtype)
  padded = np.concatenate([raw, pad], axis=1)
  return np.concatenate([padded, padded[_G : _G + 1]], axis=0)


def _add_hydrogens(
  heavy: NDArray[np.floating],
  sequence: NDArray[np.integer],
  dtype: _FloatDtype,
) -> NDArray[np.floating]:
  """Align non-rotatable hydrogens onto ``heavy``. Absent slots stay NaN."""
  if heavy.shape[0] == 0:
    return np.zeros((0, _PROTONATED_SLOTS, 3), dtype=dtype)
  width = _PROTONATED_SLOTS - heavy.shape[1]
  pad = np.full((heavy.shape[0], width, 3), np.nan, dtype=dtype)
  output = np.concatenate([_as_float(heavy, dtype), pad], axis=1)
  ideal = _ideal_protonated(dtype)[sequence]
  triads, hydrogens = _alignment_index()
  triad_idx = triads[sequence]
  hydrogen_idx = hydrogens[sequence]
  # One pass per triad. The kept rows depend on which residues own that
  # hydrogen, which is the same Python loop as upstream ``add_nonrotatable_hydrogens``.
  for triad_i in range(triad_idx.shape[1]):
    current = hydrogen_idx[:, triad_i]
    resindex, slot = np.nonzero(current != _H_SENTINEL)
    if resindex.size == 0:
      continue
    picked = current[resindex, slot]
    fixed = _gather(output, triad_idx[:, triad_i], dtype)[resindex]
    mobile = _gather(ideal, triad_idx[:, triad_i], dtype)[resindex]
    ideal_h = _gather(ideal[resindex], picked[:, None], dtype)
    rotation, mobile_com, fixed_com = _kabsch(fixed, mobile, dtype)
    aligned = _apply(ideal_h, rotation, mobile_com, fixed_com, dtype)[:, 0]
    if aligned.size and bool(np.isnan(aligned).all()):
      msg = "Failed to align non-rotatable hydrogens"
      raise LaserInputError(msg)
    output[resindex, picked] = aligned
  return output


def build_rotamers(
  backbone: Float[np.ndarray, "L 5 3"],
  chi_angles: Float[np.ndarray, "L 4"],
  sequence: Int[np.ndarray, " L"],
  *,
  dtype: _FloatDtype = np.float32,
  add_nonrotatable_hydrogens: bool = True,
) -> NDArray[np.floating]:
  """Sidechain coordinates in dataset atom order.

  ``chi_angles`` is NaN where the residue has no chi at that index. With
  hydrogens the frame is ``(L, 24, 3)``; without, ``(L, 14, 3)``. Slots the
  residue does not have stay NaN.
  """
  heavy = _build_heavy(backbone, chi_angles, sequence, dtype=dtype)
  if not add_nonrotatable_hydrogens:
    return heavy
  return _add_hydrogens(heavy, np.asarray(sequence), dtype)


def residue_atom_names(sequence_index: int, n_slots: int) -> tuple[str, ...]:
  """Atom names for one residue, heavy (14) or protonated (24) frame."""
  letter = LASER_ALPHABET[int(sequence_index)]
  if n_slots == _PROTONATED_SLOTS:
    return _extended_names(letter)
  if n_slots == _HEAVY_SLOTS:
    return _ATOM_ORDER[letter]
  msg = f"coordinate frame must have 14 or 24 atom slots, got {n_slots}"
  raise LaserInputError(msg)


def residue_name(sequence_index: int) -> str:
  """Three-letter residue name for a LASEr sequence index."""
  return _LONG[LASER_ALPHABET[int(sequence_index)]]


def placed_atoms(
  coords: Float[np.ndarray, "L A 3"],
  sequence: Int[np.ndarray, " L"],
) -> list[tuple[int, str, str, NDArray[np.floating]]]:
  """Finite atoms as ``(residue, atom_name, resname, xyz)``.

  NaN slots are omitted. A zero-filled absent slot would be emitted, which is
  why the parity test checks the NaN mask before the values.
  """
  frame = np.asarray(coords)
  seq = np.asarray(sequence)
  placed: list[tuple[int, str, str, NDArray[np.floating]]] = []
  n_slots = int(frame.shape[1])
  for residue in range(int(frame.shape[0])):
    names = residue_atom_names(int(seq[residue]), n_slots)
    resname = residue_name(int(seq[residue]))
    for slot, name in enumerate(names):
      xyz = np.asarray(frame[residue, slot], dtype=frame.dtype)
      if np.isnan(xyz).any():
        continue
      placed.append((residue, name, resname, xyz))
  return placed
