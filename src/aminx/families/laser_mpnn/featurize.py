"""Host-numpy LASErMPNN featurizer.

Ports inference parsing in ``run_inference.py`` (``ProteinComplexData``) and the
post-``construct_graphs`` first-shell mask (``utils/pdb_dataset.py``). Geometry is
NumPy; nothing here is traced by JAX.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import prody as pr
from jaxtyping import Bool, Float, Int
from numpy.typing import NDArray
from prody.measure.measure import calcPhi, calcPsi
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

LASER_ALPHABET = "ARNDCEQGHILKMFPSTWYVX"
HEAVY_ATOM_CONTACT_A = 5.0
GLY_CONTACT_PAD_A = 0.3
LIG_PR_DISTANCE_CUTOFF = 20.0
LIG_PR_KNN_K = 48
MAX_ATOMS = 14
_X = LASER_ALPHABET.index("X")
_G = LASER_ALPHABET.index("G")
_Y = LASER_ALPHABET.index("Y")
_A = LASER_ALPHABET.index("A")
_P = LASER_ALPHABET.index("P")

_LONG_TO_SHORT = {
  "CYS": "C",
  "ASP": "D",
  "SER": "S",
  "GLN": "Q",
  "LYS": "K",
  "ILE": "I",
  "PRO": "P",
  "THR": "T",
  "PHE": "F",
  "ASN": "N",
  "GLY": "G",
  "HIS": "H",
  "LEU": "L",
  "ARG": "R",
  "TRP": "W",
  "ALA": "A",
  "VAL": "V",
  "GLU": "E",
  "TYR": "Y",
  "MET": "M",
  "XAA": "X",
}
_ATOM_ORDER: dict[str, tuple[str, ...]] = {
  "G": ("N", "CA", "C", "O"),
  "X": ("N", "CA", "C", "O"),
  "A": ("N", "CA", "C", "O", "CB"),
  "S": ("N", "CA", "C", "O", "CB", "OG", "HG"),
  "C": ("N", "CA", "C", "O", "CB", "SG", "HG"),
  "T": ("N", "CA", "C", "O", "CB", "OG1", "CG2", "HG1"),
  "P": ("N", "CA", "C", "O", "CB", "CG", "CD"),
  "V": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
  "M": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
  "N": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
  "I": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
  "L": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
  "D": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
  "E": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
  "K": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
  "Q": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
  "H": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "HD1", "HE2"),
  "F": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
  "R": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
  "Y": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "HH"),
  "W": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2"),
}
_CHI_ATOMS: dict[str, dict[int, tuple[str, ...]]] = {
  "C": {1: ("N", "CA", "CB", "SG"), 2: ("CA", "CB", "SG", "HG")},
  "D": {1: ("N", "CA", "CB", "CG"), 2: ("CA", "CB", "CG", "OD1")},
  "E": {1: ("N", "CA", "CB", "CG"), 2: ("CA", "CB", "CG", "CD"), 3: ("CB", "CG", "CD", "OE1")},
  "F": {1: ("N", "CA", "CB", "CG"), 2: ("CA", "CB", "CG", "CD1")},
  "H": {1: ("N", "CA", "CB", "CG"), 2: ("CA", "CB", "CG", "ND1")},
  "I": {1: ("N", "CA", "CB", "CG1"), 2: ("CA", "CB", "CG1", "CD1")},
  "K": {
    1: ("N", "CA", "CB", "CG"),
    2: ("CA", "CB", "CG", "CD"),
    3: ("CB", "CG", "CD", "CE"),
    4: ("CG", "CD", "CE", "NZ"),
  },
  "L": {1: ("N", "CA", "CB", "CG"), 2: ("CA", "CB", "CG", "CD1")},
  "M": {1: ("N", "CA", "CB", "CG"), 2: ("CA", "CB", "CG", "SD"), 3: ("CB", "CG", "SD", "CE")},
  "N": {1: ("N", "CA", "CB", "CG"), 2: ("CA", "CB", "CG", "OD1")},
  "P": {1: ("N", "CA", "CB", "CG"), 2: ("CA", "CB", "CG", "CD")},
  "Q": {1: ("N", "CA", "CB", "CG"), 2: ("CA", "CB", "CG", "CD"), 3: ("CB", "CG", "CD", "OE1")},
  "R": {
    1: ("N", "CA", "CB", "CG"),
    2: ("CA", "CB", "CG", "CD"),
    3: ("CB", "CG", "CD", "NE"),
    4: ("CG", "CD", "NE", "CZ"),
  },
  "S": {1: ("N", "CA", "CB", "OG"), 2: ("CA", "CB", "OG", "HG")},
  "T": {1: ("N", "CA", "CB", "OG1"), 2: ("CA", "CB", "OG1", "HG1")},
  "V": {1: ("N", "CA", "CB", "CG1")},
  "W": {1: ("N", "CA", "CB", "CG"), 2: ("CA", "CB", "CG", "CD1")},
  "Y": {1: ("N", "CA", "CB", "CG"), 2: ("CA", "CB", "CG", "CD1"), 3: ("CE1", "CZ", "OH", "HH")},
}
# Non-rotatable hydrogens appended after ``_ATOM_ORDER``, same lists upstream
# builds ``hydrogen_extended_dataset_atom_order`` from. ``X`` copies alanine.
_NONROTATABLE_H: dict[str, tuple[str, ...]] = {
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
# Heavy-atom triads used to place those hydrogens. Insertion order is the
# upstream triad order. Backbone amide hydrogens are not in this map.
_HYDROGEN_ALIGNMENT: dict[str, dict[tuple[str, str, str], tuple[str, ...]]] = {
  "G": {("N", "CA", "C"): ("HA2", "HA3")},
  "A": {("N", "CA", "CB"): ("HA", "HB1", "HB2", "HB3")},
  "S": {("N", "CA", "CB"): ("HA",), ("CA", "CB", "OG"): ("HB2", "HB3")},
  "C": {("N", "CA", "CB"): ("HA",), ("CA", "CB", "SG"): ("HB2", "HB3")},
  "T": {("N", "CA", "CB"): ("HA",), ("CA", "CB", "CG2"): ("HB", "HG21", "HG22", "HG23")},
  "P": {
    ("N", "CA", "CB"): ("HA",),
    ("CA", "CB", "CG"): ("HB2", "HB3"),
    ("CB", "CG", "CD"): ("HG2", "HG3"),
    ("CG", "CD", "N"): ("HD2", "HD3"),
  },
  "V": {
    ("N", "CA", "CB"): ("HA",),
    ("CG2", "CB", "CG1"): ("HB", "HG11", "HG12", "HG13", "HG21", "HG22", "HG23"),
  },
  "M": {
    ("N", "CA", "CB"): ("HA",),
    ("CA", "CB", "CG"): ("HB2", "HB3"),
    ("CB", "CG", "SD"): ("HG2", "HG3"),
    ("CG", "SD", "CE"): ("HE1", "HE2", "HE3"),
  },
  "N": {
    ("N", "CA", "CB"): ("HA",),
    ("CA", "CB", "CG"): ("HB2", "HB3"),
    ("OD1", "CG", "ND2"): ("HD21", "HD22"),
  },
  "I": {
    ("N", "CA", "CB"): ("HA",),
    ("CG1", "CB", "CG2"): ("HB", "HG21", "HG22", "HG23"),
    ("CB", "CG1", "CD1"): ("HG12", "HG13", "HD11", "HD12", "HD13"),
  },
  "L": {
    ("N", "CA", "CB"): ("HA",),
    ("CA", "CB", "CG"): ("HB2", "HB3"),
    ("CD1", "CG", "CD2"): ("HG", "HD11", "HD12", "HD13", "HD21", "HD22", "HD23"),
  },
  "D": {("N", "CA", "CB"): ("HA",), ("CA", "CB", "CG"): ("HB2", "HB3")},
  "E": {
    ("N", "CA", "CB"): ("HA",),
    ("CA", "CB", "CG"): ("HB2", "HB3"),
    ("CB", "CG", "CD"): ("HG2", "HG3"),
  },
  "K": {
    ("N", "CA", "CB"): ("HA",),
    ("CA", "CB", "CG"): ("HB2", "HB3"),
    ("CB", "CG", "CD"): ("HG2", "HG3"),
    ("CG", "CD", "CE"): ("HD2", "HD3"),
    ("CD", "CE", "NZ"): ("HE2", "HE3", "HZ1", "HZ2", "HZ3"),
  },
  "Q": {
    ("N", "CA", "CB"): ("HA",),
    ("CA", "CB", "CG"): ("HB2", "HB3"),
    ("CB", "CG", "CD"): ("HG2", "HG3"),
    ("OE1", "CD", "NE2"): ("HE21", "HE22"),
  },
  "H": {
    ("N", "CA", "CB"): ("HA",),
    ("CA", "CB", "CG"): ("HB2", "HB3"),
    ("CE1", "NE2", "CD2"): ("HE1", "HD2"),
  },
  "F": {
    ("N", "CA", "CB"): ("HA",),
    ("CA", "CB", "CG"): ("HB2", "HB3"),
    ("CE1", "CZ", "CE2"): ("HD1", "HD2", "HE1", "HE2", "HZ"),
  },
  "R": {
    ("N", "CA", "CB"): ("HA",),
    ("CA", "CB", "CG"): ("HB2", "HB3"),
    ("CB", "CG", "CD"): ("HG2", "HG3"),
    ("CG", "CD", "NE"): ("HD2", "HD3"),
    ("NE", "CZ", "NH2"): ("HE", "HH21", "HH22", "HH12", "HH11"),
  },
  "Y": {
    ("N", "CA", "CB"): ("HA",),
    ("CA", "CB", "CG"): ("HB2", "HB3"),
    ("CE1", "CZ", "CE2"): ("HE1", "HE2", "HD1", "HD2"),
  },
  "W": {
    ("N", "CA", "CB"): ("HA",),
    ("CA", "CB", "CG"): ("HB2", "HB3"),
    ("CD1", "CG", "CD2"): ("HD1", "HE1", "HE3", "HZ2", "HZ3", "HH2"),
  },
}


def _hydrogen_extended(letter: str) -> tuple[str, ...]:
  return _ATOM_ORDER[letter] + _NONROTATABLE_H[letter]


def _build_hydrogen_tables() -> tuple[NDArray[np.int64], NDArray[np.int64]]:
  """Triad and hydrogen indices, including the glycine row reused for ``X``."""
  n_triads = max(len(triads) for triads in _HYDROGEN_ALIGNMENT.values())
  n_h = max(len(names) for triads in _HYDROGEN_ALIGNMENT.values() for names in triads.values())
  sentinel = max(len(_hydrogen_extended(letter)) for letter in _NONROTATABLE_H)
  triad = np.full((20, n_triads, 3), sentinel, dtype=np.int64)
  align = np.full((20, n_triads, n_h), sentinel, dtype=np.int64)
  for aa_idx, letter in enumerate(LASER_ALPHABET[:20]):
    order = _hydrogen_extended(letter)
    for triad_idx, (atoms, hydros) in enumerate(_HYDROGEN_ALIGNMENT[letter].items()):
      triad[aa_idx, triad_idx] = np.asarray([order.index(name) for name in atoms], dtype=np.int64)
      align[aa_idx, triad_idx, : len(hydros)] = np.asarray(
        [order.index(name) for name in hydros],
        dtype=np.int64,
      )
  triad = np.concatenate([triad, triad[_G : _G + 1]], axis=0)
  align = np.concatenate([align, align[_G : _G + 1]], axis=0)
  return triad, align


_H_TRIAD, _H_ALIGN = _build_hydrogen_tables()
MAX_PROTONATED_ATOMS = max(len(_hydrogen_extended(letter)) for letter in _NONROTATABLE_H)
_SYMBOLS = [
  "H",
  "He",
  "Li",
  "Be",
  "B",
  "C",
  "N",
  "O",
  "F",
  "Ne",
  "Na",
  "Mg",
  "Al",
  "Si",
  "P",
  "S",
  "Cl",
  "Ar",
  "K",
  "Ca",
  "Sc",
  "Ti",
  "V",
  "Cr",
  "Mn",
  "Fe",
  "Co",
  "Ni",
  "Cu",
  "Zn",
  "Ga",
  "Ge",
  "As",
  "Se",
  "Br",
  "Kr",
  "Rb",
  "Sr",
  "Y",
  "Zr",
  "Nb",
  "Mo",
  "Tc",
  "Ru",
  "Rh",
  "Pd",
  "Ag",
  "Cd",
  "In",
  "Sn",
  "Sb",
  "Te",
  "I",
  "Xe",
  "Cs",
  "Ba",
  "La",
  "Ce",
  "Pr",
  "Nd",
  "Pm",
  "Sm",
  "Eu",
  "Gd",
  "Tb",
  "Dy",
  "Ho",
  "Er",
  "Tm",
  "Yb",
  "Lu",
  "Hf",
  "Ta",
  "W",
  "Re",
  "Os",
  "Ir",
  "Pt",
  "Au",
  "Hg",
  "Tl",
  "Pb",
  "Bi",
  "Po",
  "At",
  "Rn",
  "Fr",
  "Ra",
  "Ac",
  "Th",
  "Pa",
  "U",
  "Np",
  "Pu",
  "Am",
  "Cm",
  "Bk",
  "Cf",
  "Es",
  "Fm",
  "Md",
  "No",
  "Lr",
  "Rf",
  "Db",
  "Sg",
  "Bh",
  "Hs",
  "Mt",
  "Ds",
  "Rg",
  "Cn",
  "Nh",
  "Fl",
  "Mc",
  "Lv",
  "Ts",
  "Og",
]
_ATOM_Z = {symbol: index + 1 for index, symbol in enumerate(_SYMBOLS)}
_ATOM_Z["D"] = 1

_NT_CAP_INDEX = np.array([2, 1, 0], dtype=np.int64)
_CT_CAP_INDEX = np.array([0, 1, 2], dtype=np.int64)
_NT_ALIGN = np.array(
  [
    [2.542540, 2.501473, -0.877191],
    [1.247870, 2.778495, -1.047117],
    [0.190429, 1.770458, -1.245623],
  ],
  dtype=np.float64,
)
_NT_REST = np.array(
  [
    [2.974705, 1.360642, -0.916232],
    [3.463361, 3.667905, -0.600216],
    [0.906658, 3.713693, -0.951603],
    [3.176811, 4.196102, 0.274407],
    [3.402269, 4.350083, -1.426998],
    [4.492675, 3.336663, -0.482255],
  ],
  dtype=np.float64,
)
_CT_ALIGN = np.array(
  [
    [0.263855, -0.282649, 1.828291],
    [-0.864910, -1.324407, 1.977151],
    [-1.895221, -1.281854, 1.128452],
  ],
  dtype=np.float64,
)
_CT_REST = np.array(
  [
    [-2.958410, -2.273808, 1.041114],
    [-1.885658, -0.516907, 0.462938],
    [-3.754330, -1.878560, 0.403826],
    [-3.305397, -2.481747, 2.070469],
    [-2.607824, -3.192171, 0.625321],
  ],
  dtype=np.float64,
)
_CAP_NC = 1.33484
_NT_ANGLE = np.deg2rad(124.8122)
_CT_ANGLE = np.deg2rad(118.7812)
# NT types O C H H H H sit after the imputed carbon; index 3 of the concatenated cap is H0.
_PROLINE_CAP = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)

ARRAY_FIELDS = (
  "sequence_indices",
  "chi_angles",
  "chi_mask",
  "backbone_coords",
  "phi_psi_angles",
  "heavy_atom_coords",
  "chain_indices",
  "chain_mask",
  "resnum_indices",
  "batch_indices",
  "extra_atom_contact_mask",
  "first_shell_ligand_contact_mask",
  "sidechain_contact_number",
  "residue_burial_counts",
  "sampled_chain_mask",
  "msa_depth_weight",
  "sc_mediated_hbond_counts",
  "ligand_coords",
  "ligand_atomic_numbers",
  "ligand_batch_indices",
  "ligand_subbatch_indices",
  "row_to_resindex",
  "ss_index",
)
JSON_FIELDS = (
  "pdb_code",
  "ss_code",
  "ligand_elements",
)

_GEOMETRY = Path(__file__).with_name("ideal_geometry.npz")


class LaserInputError(ValueError):
  """Raised when a LASEr input PDB cannot be featurized."""


@dataclass(frozen=True, slots=True, eq=False)
class LaserFeatures:
  """One structure packed the way upstream inference builds a length-``L`` batch item."""

  sequence_indices: Int[np.ndarray, " L"]
  chi_angles: Float[np.ndarray, "L 4"]
  chi_mask: Bool[np.ndarray, "L 4"]
  backbone_coords: Float[np.ndarray, "L 5 3"]
  phi_psi_angles: Float[np.ndarray, "L 2"]
  heavy_atom_coords: Float[np.ndarray, "L atom 3"]
  chain_indices: Int[np.ndarray, " L"]
  chain_mask: Bool[np.ndarray, " L"]
  resnum_indices: Int[np.ndarray, " L"]
  batch_indices: Int[np.ndarray, " L"]
  extra_atom_contact_mask: Bool[np.ndarray, " L"]
  first_shell_ligand_contact_mask: Bool[np.ndarray, " L"]
  sidechain_contact_number: Int[np.ndarray, " L"]
  residue_burial_counts: Int[np.ndarray, " L"]
  sampled_chain_mask: Bool[np.ndarray, " L"]
  msa_depth_weight: Float[np.ndarray, " L"]
  sc_mediated_hbond_counts: Int[np.ndarray, " L"]
  ligand_coords: Float[np.ndarray, "A 3"]
  ligand_atomic_numbers: Int[np.ndarray, " A"]
  ligand_batch_indices: Int[np.ndarray, " A"]
  ligand_subbatch_indices: Int[np.ndarray, " A"]
  row_to_resindex: Int[np.ndarray, " L"]
  ss_index: Int[np.ndarray, " L"]
  pdb_code: str
  ss_code: tuple[str, ...]
  ligand_elements: tuple[str, ...]
  exposed_mask: Bool[np.ndarray, " L"]
  budget_residue_mask: Bool[np.ndarray, " L"]


# Working precision for the protein path. float32 stays the default so the f32 tiers
# keep today's casts; float64 is opt-in via featurize(..., dtype=np.float64).
_FloatDtype = np.dtype[np.floating] | type[np.floating]


def _as_f64(value: NDArray[np.generic]) -> NDArray[np.float64]:
  return np.asarray(value, dtype=np.float64)


def _as_float(value: NDArray[np.generic], dtype: _FloatDtype) -> NDArray[np.floating]:
  return np.asarray(value, dtype=dtype)


def _geometry() -> dict[str, NDArray[np.generic]]:
  with np.load(_GEOMETRY) as blob:
    return {key: blob[key] for key in blob.files}


def _tables(dtype: _FloatDtype = np.float32) -> dict[str, NDArray[np.generic]]:
  # Cache key is the dtype string so an f32 caller and an f64 caller each keep
  # their own tables and neither cast poisons the other.
  return _tables_cached(np.dtype(dtype).str)


@functools.cache
def _tables_cached(dtype_key: str) -> dict[str, NDArray[np.generic]]:
  dtype = np.dtype(dtype_key)
  raw = _geometry()
  ideal = _as_float(raw["ideal_aa"], dtype)
  lengths = _as_float(raw["bond_lengths"], dtype)
  angles = _as_float(raw["bond_angles"], dtype)
  alignment = np.array(raw["alignment"], dtype=np.int64, copy=True)
  alignment[alignment < 0] = MAX_ATOMS
  chi_index = np.full((20, 4, 4), MAX_ATOMS, dtype=np.int64)
  leftover = np.full((20, MAX_ATOMS), MAX_ATOMS, dtype=np.int64)
  for idx, letter in enumerate(LASER_ALPHABET[:20]):
    atoms = _CHI_ATOMS.get(letter)
    if atoms is None:
      continue
    order = _ATOM_ORDER[letter]
    placed: set[int] = set()
    for chi_num, names in atoms.items():
      idces = [order.index(name) for name in names]
      chi_index[idx, chi_num - 1] = idces
      placed.update(idces)
    rest = sorted(set(range(len(order))) - placed - {2, 3})
    leftover[idx, : len(rest)] = rest
  tyr_have = leftover[_Y] != MAX_ATOMS
  tyr_rest = sorted(
    set(leftover[_Y, tyr_have].tolist())
    | {_ATOM_ORDER["Y"].index(name) for name in ("CE1", "CZ", "OH", "HH")},
  )
  leftover[_Y] = MAX_ATOMS
  leftover[_Y, : len(tyr_rest)] = tyr_rest
  width = int((leftover != MAX_ATOMS).sum(axis=1).max())
  leftover = leftover[:, :width]
  gly = _G

  def _cat_row(block: NDArray[np.generic]) -> NDArray[np.generic]:
    return np.asarray(np.concatenate([block, block[gly : gly + 1]], axis=0))

  needs = ~(alignment == MAX_ATOMS).all(axis=-1)
  needs_x = np.concatenate([needs, np.array([False])])
  return {
    "ideal": _pad_atoms(_as_float(_cat_row(ideal), dtype), dtype),
    "lengths": _as_float(_cat_row(lengths), dtype),
    "angles": _as_float(_cat_row(angles), dtype),
    "chi_index": np.asarray(_cat_row(chi_index), dtype=np.int64),
    "alignment": np.asarray(_cat_row(alignment), dtype=np.int64),
    "leftover": np.asarray(_cat_row(leftover), dtype=np.int64),
    "needs": np.asarray(needs_x, dtype=bool),
    "ideal_prot": _as_float(raw["ideal_prot"], dtype),
    # One NaN slot past the longest protonated residue, then the glycine row for X.
    "ideal_prot_h": _cat_row(_pad_atoms(_as_float(raw["ideal_prot"], dtype), dtype)),
  }


def _pad_atoms(
  coords: NDArray[np.floating],
  dtype: _FloatDtype = np.float32,
) -> NDArray[np.floating]:
  nan = np.full((*coords.shape[:-2], 1, 3), np.nan, dtype=dtype)
  return np.concatenate([_as_float(coords, dtype), nan], axis=-2)


def _gather(
  coords: NDArray[np.floating],
  index: NDArray[np.integer],
  dtype: _FloatDtype = np.float32,
) -> NDArray[np.floating]:
  rows = np.arange(index.shape[0])[:, None]
  return np.asarray(coords[rows, index], dtype=dtype)


def _kabsch(
  fixed: NDArray[np.floating],
  mobile: NDArray[np.floating],
  dtype: _FloatDtype = np.float32,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
  fixed_w = _as_float(fixed, dtype)
  mobile_w = _as_float(mobile, dtype)
  fixed_com = fixed_w.mean(axis=1, keepdims=True)
  mobile_com = mobile_w.mean(axis=1, keepdims=True)
  centered_m = mobile_w - mobile_com
  centered_f = fixed_w - fixed_com
  if np.isnan(centered_m).any() or np.isnan(centered_f).any():
    msg = "NaNs in alignment matrices"
    raise LaserInputError(msg)
  covariance = np.swapaxes(centered_m, -1, -2) @ centered_f
  left, _, right = np.linalg.svd(covariance)
  rotation = left @ right
  neg = np.linalg.det(rotation) < 0.0
  if np.any(neg):
    flipped = np.array(right, copy=True)
    flipped[neg, -1] *= -1
    rotation = np.array(rotation, copy=True)
    rotation[neg] = left[neg] @ flipped[neg]
  return rotation, mobile_com, fixed_com


def _apply(
  coords: NDArray[np.floating],
  rotation: NDArray[np.floating],
  mobile_com: NDArray[np.floating],
  fixed_com: NDArray[np.floating],
  dtype: _FloatDtype = np.float32,
) -> NDArray[np.floating]:
  return (
    (_as_float(coords, dtype) - _as_float(mobile_com, dtype)) @ _as_float(rotation, dtype)
  ) + _as_float(fixed_com, dtype)


def _extend(
  prev: NDArray[np.floating],
  bond_lengths: NDArray[np.floating],
  bond_angles: NDArray[np.floating],
  dihedral: NDArray[np.floating],
  dtype: _FloatDtype = np.float32,
) -> NDArray[np.floating]:
  prev_w = _as_float(prev, dtype)
  if prev_w.shape[0] == 0:
    return np.zeros((0, 3), dtype=dtype)
  # Same 1e-6 magnitude as upstream build_rotamers; only the scalar's dtype follows.
  eps = np.dtype(dtype).type(1e-6)
  bc = prev_w[:, 1] - prev_w[:, 2]
  bc = bc / (np.linalg.norm(bc, axis=-1, keepdims=True) + eps)
  ba = np.cross(prev_w[:, 1] - prev_w[:, 0], bc)
  ba = ba / (np.linalg.norm(ba, axis=-1, keepdims=True) + eps)
  tangent = np.cross(ba, bc)
  length = _as_float(bond_lengths, dtype)
  angle = _as_float(bond_angles, dtype)
  dihedral_w = _as_float(dihedral, dtype)
  d1 = length * np.cos(angle)
  d2 = length * np.sin(angle) * np.cos(dihedral_w)
  d3 = -length * np.sin(angle) * np.sin(dihedral_w)
  return prev_w[:, 2] + bc * d1 + tangent * d2 + ba * d3


def idealize_backbone(
  backbone: NDArray[np.floating],
  phi_psi: NDArray[np.floating],
  dtype: _FloatDtype = np.float32,
) -> NDArray[np.floating]:
  """Ideal N, CA, CB, C, O frames. ``backbone`` is (L, 5, 3) as N, CA, CB, C, O."""
  count = backbone.shape[0]
  frames = np.array(
    np.broadcast_to(_tables(dtype)["ideal_prot"][_A, [0, 1, 2]], (count, 3, 3)),
    dtype=dtype,
    copy=True,
  )
  rotation, mobile_com, fixed_com = _kabsch(
    _as_float(backbone, dtype)[:, [0, 1, 3]],
    frames,
    dtype,
  )
  placed = _apply(frames, rotation, mobile_com, fixed_com, dtype)
  nitrogen, ca, carbon = placed[:, 0], placed[:, 1], placed[:, 2]
  bond_c = carbon - ca
  bond_n = ca - nitrogen
  normal = np.cross(bond_n, bond_c)
  cb = -0.58273431 * normal + 0.56802827 * bond_n - 0.54067466 * bond_c + ca
  psi = np.asarray(phi_psi[:, 1], dtype=dtype)
  dihedral = np.deg2rad(np.nan_to_num(psi + np.dtype(dtype).type(180.0)))[:, None]
  oxygen = _extend(
    np.stack([nitrogen, ca, carbon], axis=1),
    np.full((count, 1), 1.23, dtype=dtype),
    np.full((count, 1), np.deg2rad(120.8), dtype=dtype),
    dihedral,
    dtype,
  )
  return np.stack([nitrogen, ca, cb, carbon, oxygen], axis=1)


def _chi_angles(
  heavy: NDArray[np.floating],
  sequence: NDArray[np.integer],
  dtype: _FloatDtype = np.float32,
) -> NDArray[np.floating]:
  output = np.full((heavy.shape[0], 4), np.nan, dtype=dtype)
  keep = sequence != _X
  if not np.any(keep):
    return output
  coords = _pad_atoms(_as_float(heavy, dtype)[keep], dtype)
  seq = sequence[keep]
  index = _tables(dtype)["chi_index"][seq]
  flat = index.reshape(seq.shape[0], -1)
  gathered = _gather(coords, flat, dtype).reshape(seq.shape[0], 4, 4, 3)
  b0 = gathered[:, :, 0] - gathered[:, :, 1]
  b1 = gathered[:, :, 1] - gathered[:, :, 2]
  b2 = gathered[:, :, 2] - gathered[:, :, 3]
  n1 = np.cross(b0, b1)
  n2 = np.cross(b1, b2)
  m1 = np.cross(n1, b1 / np.linalg.norm(b1, axis=-1, keepdims=True))
  angle = np.rad2deg(np.arctan2(np.sum(m1 * n2, axis=-1), np.sum(n1 * n2, axis=-1)))
  output[keep] = angle.astype(dtype)
  return output


def _generate_ideal(
  sequence: NDArray[np.integer],
  chi: NDArray[np.floating],
  dtype: _FloatDtype = np.float32,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
  tables = _tables(dtype)
  ideal = np.array(tables["ideal"][sequence], dtype=dtype, copy=True)
  unadjusted = np.array(ideal, copy=True)
  chi_index = np.asarray(tables["chi_index"][sequence], dtype=np.int64)
  lengths = _as_float(tables["lengths"][sequence], dtype)
  angles = np.deg2rad(_as_float(tables["angles"][sequence], dtype))
  chi_rad = np.deg2rad(_as_float(chi, dtype))
  for chi_i in range(4):
    rows = np.nonzero(~np.isnan(chi[:, chi_i]))[0]
    if rows.size == 0:
      continue
    prev = _gather(ideal[rows], chi_index[rows, chi_i, :3], dtype)
    nxt = _extend(
      prev,
      lengths[rows, chi_i, None],
      angles[rows, chi_i, None],
      chi_rad[rows, chi_i, None],
      dtype,
    )
    ideal[rows, chi_index[rows, chi_i, 3]] = nxt
  return ideal, unadjusted


def _place_tyr(
  coords: NDArray[np.floating],
  sequence: NDArray[np.integer],
  chi3: NDArray[np.floating],
  dtype: _FloatDtype = np.float32,
) -> NDArray[np.floating]:
  tables = _tables(dtype)
  out = np.array(coords, dtype=dtype, copy=True)
  index = np.asarray(tables["chi_index"][sequence], dtype=np.int64)
  prev = _gather(out, index[:, 2, :3], dtype)
  nxt = _extend(
    prev,
    _as_float(tables["lengths"][sequence, 2, None], dtype),
    np.deg2rad(_as_float(tables["angles"][sequence, 2, None], dtype)),
    np.deg2rad(_as_float(chi3, dtype)[:, None]),
    dtype,
  )
  out[np.arange(out.shape[0]), index[:, 2, 3]] = nxt
  return out


def _add_nonrotatable_hydrogens(
  heavy: NDArray[np.floating],
  sequence: NDArray[np.integer],
) -> NDArray[np.floating]:
  """Place non-rotatable hydrogens on a heavy-atom rotamer build.

  Pads the atom axis to ``MAX_PROTONATED_ATOMS + 1`` with NaN, then Kabsch-aligns
  each ideal protonated triad onto the built heavy atoms. Alignment runs in
  float64. Backbone amide hydrogens are left absent.
  """
  width = MAX_PROTONATED_ATOMS + 1
  if heavy.shape[0] == 0:
    return np.empty((0, width, 3), dtype=heavy.dtype)
  output = np.full((heavy.shape[0], width, 3), np.nan, dtype=heavy.dtype)
  output[:, : heavy.shape[1]] = heavy
  output_f64 = np.asarray(output, dtype=np.float64)
  ideal = np.asarray(_tables(np.float64)["ideal_prot_h"][sequence], dtype=np.float64)
  triads = _H_TRIAD[sequence]
  hydrogens = _H_ALIGN[sequence]
  for idx in range(triads.shape[1]):
    curr_h = hydrogens[:, idx]
    resindex, _hydr_slot = np.nonzero(curr_h != MAX_PROTONATED_ATOMS)
    if resindex.size == 0:
      continue
    hydr_idx = curr_h[resindex, _hydr_slot]
    fixed = _gather(output_f64, triads[:, idx], np.float64)[resindex]
    mobile = _gather(ideal, triads[:, idx], np.float64)[resindex]
    count = resindex.shape[0]
    ideal_h = ideal[resindex][np.arange(count)[:, None], hydr_idx[:, None]]
    rotation, mobile_com, fixed_com = _kabsch(fixed, mobile, np.float64)
    aligned = _apply(ideal_h, rotation, mobile_com, fixed_com, np.float64)[:, 0]
    if aligned.size and not np.any(~np.isnan(aligned)):
      msg = "Failed to align non-rotatable hydrogens..."
      raise ValueError(msg)
    output[resindex, hydr_idx] = aligned
    output_f64[resindex, hydr_idx] = aligned
  return output


def build_rotamers(
  backbone: NDArray[np.floating],
  chi: NDArray[np.floating],
  sequence: NDArray[np.integer],
  dtype: _FloatDtype = np.float32,
  *,
  add_nonrotatable_hydrogens: bool = False,
) -> NDArray[np.floating]:
  """Full-atom coordinates in dataset atom order.

  The default keeps the heavy-atom axis (``MAX_ATOMS``).
  ``add_nonrotatable_hydrogens=True`` returns the protonated axis
  (``MAX_PROTONATED_ATOMS + 1``) with NaN padding and no backbone amide H.
  """
  tables = _tables(dtype)
  ideal, unadjusted = _generate_ideal(sequence, chi, dtype)
  needs = tables["needs"][sequence]
  if np.any(needs):
    align_index = tables["alignment"][sequence[needs]]
    leftover = tables["leftover"][sequence[needs]]
    fixed = _gather(ideal[needs], align_index, dtype)
    mobile = _gather(unadjusted[needs], align_index, dtype)
    rows, cols = np.nonzero(leftover != MAX_ATOMS)
    atom_index = leftover[rows, cols]
    leftover_coords = unadjusted[needs][rows, atom_index]
    rotation, mobile_com, fixed_com = _kabsch(fixed, mobile, dtype)
    moved = _apply(
      leftover_coords[:, None, :],
      rotation[rows],
      mobile_com[rows],
      fixed_com[rows],
      dtype,
    )[:, 0]
    subset = np.array(ideal[needs], copy=True)
    subset[rows, atom_index] = moved
    ideal[needs] = subset
  is_tyr = sequence == _Y
  if np.any(is_tyr):
    ideal[is_tyr] = _place_tyr(ideal[is_tyr], sequence[is_tyr], chi[is_tyr, 2], dtype)
  not_gly = (sequence != _G) & (sequence != _X)
  fixed_index = np.where(not_gly[:, None], np.array([0, 1, 2]), np.array([0, 1, 3]))
  mobile_index = np.where(not_gly[:, None], np.array([0, 1, 4]), np.array([0, 1, 2]))
  fixed = _gather(_as_float(backbone, dtype), fixed_index, dtype)
  mobile = _gather(ideal, mobile_index, dtype)
  rotation, mobile_com, fixed_com = _kabsch(fixed, mobile, dtype)
  present = ~np.isnan(ideal[:, :, 0])
  rows, cols = np.nonzero(present)
  moved = _apply(
    ideal[rows, cols][:, None, :],
    rotation[rows],
    mobile_com[rows],
    fixed_com[rows],
    dtype,
  )[:, 0]
  ideal[rows, cols] = moved
  ideal[:, :4] = _as_float(backbone, dtype)[:, [0, 1, 3, 4]]
  heavy = ideal[:, :MAX_ATOMS]
  if not add_nonrotatable_hydrogens:
    return heavy
  return _add_nonrotatable_hydrogens(heavy, sequence)


def _lig_prot_edges(
  ca: NDArray[np.floating],
  ligand: NDArray[np.floating],
  *,
  k: int,
  cutoff: float,
) -> NDArray[np.int64]:
  if ligand.shape[0] == 0 or ca.shape[0] == 0:
    return np.zeros((2, 0), dtype=np.int64)
  dist = np.linalg.norm(_as_f64(ca)[:, None, :] - _as_f64(ligand)[None, :, :], axis=-1)
  connected = np.nonzero((dist < cutoff).sum(axis=1) > 0)[0]
  kk = min(k, int(ligand.shape[0]))
  if connected.size == 0 or kk == 0:
    return np.zeros((2, 0), dtype=np.int64)
  nearest = np.argsort(dist[connected], axis=1)[:, :kk]
  return np.stack([nearest.reshape(-1).astype(np.int64), np.repeat(connected, kk).astype(np.int64)])


def _pairwise_ras(
  ligand: NDArray[np.floating],
  atoms: NDArray[np.floating],
  dtype: _FloatDtype = np.float32,
) -> NDArray[np.floating]:
  delta = _as_float(atoms, dtype)[:, None, :, :] - _as_float(ligand, dtype)[None, :, None, :]
  return np.linalg.norm(delta, axis=-1).astype(dtype)


def _contact_rows(
  ligand: NDArray[np.floating],
  atoms: NDArray[np.floating],
  hydrogen: NDArray[np.bool_],
  threshold: float,
  dtype: _FloatDtype = np.float32,
) -> NDArray[np.bool_]:
  if atoms.shape[0] == 0:
    return np.zeros((0,), dtype=bool)
  distances = _pairwise_ras(ligand, atoms, dtype)
  close = np.asarray(np.nan_to_num(distances, nan=np.inf) < np.dtype(dtype).type(threshold))
  per_atom = np.asarray(close.any(axis=-1), dtype=bool)
  return np.asarray(per_atom[:, ~hydrogen].any(axis=-1), dtype=bool)


def first_shell_contact_mask(
  fa_coords: NDArray[np.floating],
  sequence_indices: NDArray[np.integer],
  ligand_coords: NDArray[np.floating],
  ligand_atomic_numbers: NDArray[np.integer],
  *,
  lig_pr_knn_k: int = LIG_PR_KNN_K,
  lig_pr_distance_cutoff: float = LIG_PR_DISTANCE_CUTOFF,
  dtype: _FloatDtype = np.float32,
) -> NDArray[np.bool_]:
  """Post-``construct_graphs`` first shell.

  A residue is marked when it has a CA within ``lig_pr_distance_cutoff`` of a
  ligand atom and a heavy atom within 5 angstroms of a ligand heavy atom. Gly and X use
  CA only, with a 0.3 angstrom pad. No ligand atoms yields an all-False mask.
  ``ligand_atomic_numbers`` are elemental Z (hydrogen is 1).
  """
  length = int(fa_coords.shape[0])
  mask = np.zeros((length,), dtype=bool)
  if ligand_coords.shape[0] == 0 or length == 0:
    return mask
  z_index = np.asarray(ligand_atomic_numbers, dtype=np.int64) - 1
  edges = _lig_prot_edges(
    fa_coords[:, 1],
    ligand_coords,
    k=lig_pr_knn_k,
    cutoff=lig_pr_distance_cutoff,
  )
  if edges.shape[1] == 0:
    return mask
  protein = np.unique(edges[1])
  putative = _as_f64(fa_coords)[protein]
  seq = sequence_indices[protein]
  is_gly = (seq == _G) | (seq == _X)
  hydrogen = z_index == 0
  contact = np.zeros(protein.shape[0], dtype=bool)
  contact[~is_gly] = _contact_rows(
    ligand_coords,
    putative[~is_gly, 4:],
    hydrogen,
    HEAVY_ATOM_CONTACT_A,
    dtype,
  )
  contact[is_gly] = _contact_rows(
    ligand_coords,
    putative[is_gly, 1:2],
    hydrogen,
    HEAVY_ATOM_CONTACT_A + GLY_CONTACT_PAD_A,
    dtype,
  )
  mask[protein[contact]] = True
  return mask


def _hydrogen_positions(coord: NDArray[np.floating]) -> NDArray[np.float64]:
  vec_cn = coord[:, 1:, 0] - coord[:, :-1, 2]
  vec_cn = vec_cn / np.linalg.norm(vec_cn, axis=-1, keepdims=True)
  vec_can = coord[:, 1:, 0] - coord[:, 1:, 1]
  vec_can = vec_can / np.linalg.norm(vec_can, axis=-1, keepdims=True)
  vec_nh = vec_cn + vec_can
  vec_nh = vec_nh / np.linalg.norm(vec_nh, axis=-1, keepdims=True)
  return coord[:, 1:, 0] + 1.01 * vec_nh


def _hbond_map(coord: NDArray[np.floating]) -> NDArray[np.float64]:
  """pydssp 0.9.1 hydrogen-bond map for backbone ``(B, L, 4, 3)``."""
  _b, length, _atoms, _xyz = coord.shape
  if length < 2:
    return np.zeros((coord.shape[0], length, length), dtype=np.float64)
  hydrogen = _hydrogen_positions(coord)
  span = length - 1
  nmap = np.repeat(coord[:, 1:, 0][:, :, None, :], span, axis=2)
  hmap = np.repeat(hydrogen[:, :, None, :], span, axis=2)
  cmap = np.repeat(coord[:, :-1, 2][:, None, :, :], span, axis=1)
  omap = np.repeat(coord[:, :-1, 3][:, None, :, :], span, axis=1)
  d_on = np.linalg.norm(omap - nmap, axis=-1)
  d_ch = np.linalg.norm(cmap - hmap, axis=-1)
  d_oh = np.linalg.norm(omap - hmap, axis=-1)
  d_cn = np.linalg.norm(cmap - nmap, axis=-1)
  energy = np.pad(
    0.084 * (1.0 / d_on + 1.0 / d_ch - 1.0 / d_oh - 1.0 / d_cn) * 332.0,
    [(0, 0), (1, 0), (0, 1)],
  )
  local = ~np.eye(length, dtype=bool)
  local &= ~np.diag(np.ones(length - 1, dtype=bool), k=-1)
  if length > 2:
    local &= ~np.diag(np.ones(length - 2, dtype=bool), k=-2)
  margin = 1.0
  hbond = np.clip(-0.5 - margin - energy, -margin, margin)
  hbond = (np.sin(hbond / margin * np.pi / 2.0) + 1.0) / 2.0
  return hbond * local[None]


def _unfold(block: NDArray[np.floating], window: int, axis: int) -> NDArray[np.float64]:
  idx = np.arange(window)[:, None] + np.arange(block.shape[axis] - window + 1)[None, :]
  taken = np.moveaxis(np.take(block, idx, axis=axis), axis - 1, -1)
  return np.asarray(taken, dtype=np.float64)


def secondary_structure(backbone_n_ca_c_o: NDArray[np.floating]) -> NDArray[np.str_]:
  """DSSP 3-state letters (``-``, ``H``, ``E``) as pydssp assigns them."""
  length = backbone_n_ca_c_o.shape[0]
  letters = np.array(["-", "H", "E"])
  if length == 0:
    return np.zeros((0,), dtype=letters.dtype)
  if length < 6:
    return np.full((length,), "-", dtype=letters.dtype)
  coord = _as_f64(backbone_n_ca_c_o)[None]
  hbmap = _as_f64(np.swapaxes(_hbond_map(coord), -1, -2))
  turn3 = np.diagonal(hbmap, axis1=-2, axis2=-1, offset=3) > 0.0
  turn4 = np.diagonal(hbmap, axis1=-2, axis2=-1, offset=4) > 0.0
  turn5 = np.diagonal(hbmap, axis1=-2, axis2=-1, offset=5) > 0.0
  h3 = np.pad(turn3[:, :-1] * turn3[:, 1:], [(0, 0), (1, 3)])
  h4 = np.pad(turn4[:, :-1] * turn4[:, 1:], [(0, 0), (1, 4)])
  h5 = np.pad(turn5[:, :-1] * turn5[:, 1:], [(0, 0), (1, 5)])
  helix4 = h4 + np.roll(h4, 1, 1) + np.roll(h4, 2, 1) + np.roll(h4, 3, 1)
  h3 = h3 * ~np.roll(helix4, -1, 1) * ~helix4
  h5 = h5 * ~np.roll(helix4, -1, 1) * ~helix4
  helix3 = h3 + np.roll(h3, 1, 1) + np.roll(h3, 2, 1)
  helix5 = h5 + np.roll(h5, 1, 1) + np.roll(h5, 2, 1) + np.roll(h5, 3, 1) + np.roll(h5, 4, 1)
  helix = (helix3 + helix4 + helix5) > 0
  unfolded = _unfold(_unfold(hbmap, 3, -2), 3, -2) > 0
  rev = np.swapaxes(unfolded, 1, 2)
  parallel = np.pad(
    (unfolded[:, :, :, 0, 1] * rev[:, :, :, 1, 2]) + (rev[:, :, :, 0, 1] * unfolded[:, :, :, 1, 2]),
    [(0, 0), (1, 1), (1, 1)],
  )
  antiparallel = np.pad(
    (unfolded[:, :, :, 1, 1] * rev[:, :, :, 1, 1]) + (unfolded[:, :, :, 0, 2] * rev[:, :, :, 0, 2]),
    [(0, 0), (1, 1), (1, 1)],
  )
  strand = (parallel + antiparallel).sum(axis=-1) > 0
  loop = (~helix * ~strand)[0]
  onehot = np.stack([loop, helix[0], strand[0]], axis=-1)
  return letters[np.argmax(onehot, axis=-1)]


def _alpha_triangles(points: NDArray[np.floating], alpha: float) -> NDArray[np.int64]:
  tetra = Delaunay(points)
  tetrapos = np.take(points, tetra.simplices, axis=0)
  normsq = np.sum(tetrapos**2, axis=2)[:, :, None]
  ones = np.ones((tetrapos.shape[0], tetrapos.shape[1], 1))
  det_a = np.linalg.det(np.concatenate((tetrapos, ones), axis=2))
  dx = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [1, 2]], ones), axis=2))
  dy = -np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 2]], ones), axis=2))
  dz = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 1]], ones), axis=2))
  det_c = np.linalg.det(np.concatenate((normsq, tetrapos), axis=2))
  radius = np.sqrt(np.maximum(0, dx**2 + dy**2 + dz**2 - 4 * det_a * det_c)) / (
    2 * np.abs(det_a) + 1e-6
  )
  tetras = tetra.simplices[radius < alpha]
  comb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
  triangles = np.sort(tetras[:, comb].reshape(-1, 3), axis=1)
  unique, counts = np.unique(triangles, axis=0, return_counts=True)
  return unique[counts == 1]


def _inside_mesh(
  test_points: NDArray[np.floating],
  vertices: NDArray[np.floating],
  triangles: NDArray[np.integer],
) -> NDArray[np.bool_]:
  outside = np.max(np.abs(vertices), axis=0) + np.random.default_rng().uniform(25, 100, size=3)
  outside = Rotation.random().as_matrix() @ outside
  direction = test_points - outside
  lengths = np.linalg.norm(direction, axis=1)
  direction = direction / lengths[:, None]
  tri = vertices[triangles]
  edge1 = tri[:, 1] - tri[:, 0]
  edge2 = tri[:, 2] - tri[:, 0]
  h = np.cross(direction[:, None, :], edge2[None, :, :])
  det = np.sum(edge1[None, :, :] * h, axis=-1)
  mask = np.abs(det) > 1e-6
  inv = np.where(mask, 1.0 / det, 0.0)
  s = test_points[:, None, :] - tri[None, :, 0]
  u = inv * np.sum(s * h, axis=-1)
  q = np.cross(s, edge1[None, :, :])
  v = inv * np.sum(direction[:, None, :] * q, axis=-1)
  t = inv * np.sum(edge2[None, :, :] * q, axis=-1)
  valid = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-6)
  t = np.where(valid, t, np.inf)
  return (np.sum(t <= lengths[:, None], axis=1) % 2) == 1


def exposed_mask(
  ca_coords: NDArray[np.floating],
  test_atoms: NDArray[np.floating],
  *,
  alpha: float = 9.0,
  num_rays: int = 10,
) -> NDArray[np.bool_]:
  """True where the test atom is outside the CA alpha-hull (upstream ``~burial``)."""
  count = ca_coords.shape[0]
  if count <= 4:
    return np.ones((count,), dtype=bool)
  ca = _as_f64(ca_coords)
  probe = _as_f64(test_atoms)
  center = ca.mean(axis=0)
  ca = ca - center
  probe = probe - center
  triangles = _alpha_triangles(ca, alpha)
  buried = _inside_mesh(probe, ca, triangles)
  for _ in range(num_rays - 1):
    buried &= _inside_mesh(probe, ca, triangles)
  return ~buried


def _elements(residue: pr.Residue) -> list[str]:
  elements = [str(item) for item in residue.getElements()]
  if not any(elements):
    elements = [
      "".join(char for char in name if not char.isnumeric()) for name in residue.getNames()
    ]
  return elements


def _atomic_numbers(elements: list[str]) -> NDArray[np.int64]:
  numbers = []
  for element in elements:
    key = element.capitalize()
    if key not in _ATOM_Z:
      msg = f"unknown ligand element {element!r}"
      raise LaserInputError(msg)
    numbers.append(_ATOM_Z[key])
  return np.asarray(numbers, dtype=np.int64)


def _methyl_caps(
  phi_psi: NDArray[np.floating],
  coords: NDArray[np.floating],
  sequence: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
  angles = np.array(phi_psi, dtype=np.float64, copy=True)
  for slot in (0, 1):
    if np.isnan(angles[slot]):
      angles[slot] = float(np.random.default_rng().uniform(-180.0, 180.0))
  phi, psi = np.deg2rad(angles)
  xyz = _as_f64(coords)
  nt_frame = xyz[_NT_CAP_INDEX][None]
  imputed_c = _extend(nt_frame, np.array([[_CAP_NC]]), np.array([[_NT_ANGLE]]), np.array([[phi]]))
  fixed = np.concatenate([imputed_c, nt_frame[:, 2], nt_frame[:, 1]], axis=0)[None]
  rotation, mobile_com, fixed_com = _kabsch(fixed, _NT_ALIGN[None])
  nt_rest = _apply(_NT_REST[None], rotation, mobile_com, fixed_com)[0]
  nt = np.concatenate([imputed_c, nt_rest], axis=0)
  ct_frame = xyz[_CT_CAP_INDEX][None]
  imputed_n = _extend(ct_frame, np.array([[_CAP_NC]]), np.array([[_CT_ANGLE]]), np.array([[psi]]))
  fixed = np.concatenate([ct_frame[:, 1], ct_frame[:, 2], imputed_n], axis=0)[None]
  rotation, mobile_com, fixed_com = _kabsch(fixed, _CT_ALIGN[None])
  ct_rest = _apply(_CT_REST[None], rotation, mobile_com, fixed_com)[0]
  ct = np.concatenate([imputed_n, ct_rest], axis=0)
  cap = np.concatenate([nt, ct], axis=0)
  cap_z = np.array([6, 8, 6, 1, 1, 1, 1, 7, 6, 1, 1, 1, 1], dtype=np.int64)
  if sequence == _P:
    cap = cap[_PROLINE_CAP]
    cap_z = cap_z[_PROLINE_CAP]
  return cap, cap_z


def _is_amino(residue: pr.Residue) -> bool:
  if str(residue.getResname()) in _LONG_TO_SHORT and all(
    residue.select(f"name {atom}") is not None for atom in ("N", "CA", "C", "O")
  ):
    return True
  names = set(residue.getNames())
  return all(atom in names for atom in ("N", "CA", "C", "O"))


def _residue_coords(residue: pr.Residue, letter: str) -> NDArray[np.float64]:
  coords = np.full((MAX_ATOMS, 3), np.nan, dtype=np.float64)
  order = _ATOM_ORDER[letter]
  for name, xyz in zip(residue.getNames(), residue.getCoords(), strict=False):
    if name in order:
      coords[order.index(str(name))] = xyz
  return coords


def _all_gly(view: pr.HierView) -> dict[tuple[str, str, int, str], pr.Residue]:
  atoms = view.getAtoms().select("name N or name C or name CA or name O")
  if atoms is None:
    return {}
  copied = atoms.copy()
  residues = [res for chain in copied.getHierView() for res in chain if len(res) == 4]
  group = None
  for residue in residues:
    copied = residue.copy()
    copied.setResnames("GLY")
    copied.setFlags("hetatm", False)  # noqa: FBT003
    group = copied if group is None else group + copied
  if group is None:
    return {}
  # ProDy 2.6 drops the protein/aminoacid flags that calcPhi used to require.
  present = np.ones(group.numAtoms(), dtype=bool)
  group._flags["protein"] = present  # noqa: SLF001
  group._flags["aminoacid"] = present.copy()  # noqa: SLF001
  found: dict[tuple[str, str, int, str], pr.Residue] = {}
  for chain in group.getHierView():
    for residue in chain:
      key = (
        str(chain.getSegname()),
        str(chain.getChid()),
        int(residue.getResnum()),
        str(residue.getIcode()),
      )
      found[key] = residue
  return found


def _phi_psi(residue: pr.Residue, dtype: _FloatDtype = np.float32) -> NDArray[np.floating]:
  try:
    phi = float(calcPhi(residue))
  except ValueError:
    phi = float(np.nan)
  try:
    psi = float(calcPsi(residue))
  except ValueError:
    psi = float(np.nan)
  return np.asarray([phi, psi], dtype=dtype)


def _cap_names(numbers: NDArray[np.integer]) -> list[str]:
  counts: dict[str, int] = {}
  names: list[str] = []
  for number in numbers.tolist():
    element = _SYMBOLS[int(number) - 1]
    names.append(f"{element}{counts.get(element, 0)}")
    counts[element] = counts.get(element, 0) + 1
  return names


class _LigandSink:
  def __init__(self) -> None:
    self.xyz: list[NDArray[np.float64]] = []
    self.z: list[NDArray[np.int64]] = []
    self.sub: list[NDArray[np.int64]] = []
    self.elements: list[str] = []
    self._next = 0

  def push(self, coords: NDArray[np.floating], elements: list[str]) -> None:
    block = _as_f64(coords)
    self.xyz.append(block)
    self.z.append(_atomic_numbers(elements))
    self.sub.append(np.full((block.shape[0],), self._next, dtype=np.int64))
    self.elements.extend(elements)
    self._next += 1


def _require_binary_bfactors(view: pr.HierView) -> None:
  betas = np.concatenate([chain.getBetas() for chain in view.iterChains()])
  unique = np.unique(betas)
  allowed = np.isin(unique, np.array([0.0, 1.0])).all()
  if unique.size > 2 or not allowed:
    msg = f"fix_from_bfactor requires B-factors in {{0, 1}}, found {unique.tolist()}"
    raise LaserInputError(msg)


def _append_protein(
  residue: pr.Residue,
  letter: str,
  chain_index: int,
  coords: NDArray[np.float64],
  angles: NDArray[np.floating],
  sequence: list[int],
  heavy: list[NDArray[np.float64]],
  phi_psi: list[NDArray[np.floating]],
  chains: list[int],
  fixed: list[bool],
  resindex: list[int],
  crystal_bb: list[NDArray[np.float64]],
  exposure_probe: list[NDArray[np.float64]],
) -> None:
  sequence.append(LASER_ALPHABET.index(letter))
  heavy.append(coords)
  phi_psi.append(angles)
  chains.append(chain_index)
  betas = np.asarray(residue.getBetas(), dtype=np.float64)
  fixed.append(bool(np.isclose(float(np.max(betas)), 1.0)))
  resindex.append(int(residue.getResindex()))
  crystal_bb.append(np.stack([coords[0], coords[1], coords[2], coords[3]]))
  cb = residue.select("name CB")
  probe = cb.getCoords()[0] if cb is not None else coords[1]
  exposure_probe.append(np.asarray(probe, dtype=np.float64))


def _push_ncaa_ligand(
  residue: pr.Residue,
  coords: NDArray[np.float64],
  angles: NDArray[np.floating],
  sink: _LigandSink,
) -> None:
  cap, cap_z = _methyl_caps(angles, coords, _X)
  sink.push(residue.getCoords(), _elements(residue))
  names = _cap_names(cap_z)
  keep = np.array([name != "H0" for name in names])
  if "H" not in set(residue.getNames()):
    keep = np.ones(cap.shape[0], dtype=bool)
  kept = [element for element, flag in zip(_elements_from_z(cap_z), keep, strict=True) if flag]
  sink.push(cap[keep], kept)


def _ligand_arrays(
  sink: _LigandSink,
  *,
  ignore: bool,
  dtype: np.dtype[np.floating] | type[np.floating],
) -> tuple[
  NDArray[np.floating],
  NDArray[np.int64],
  NDArray[np.int64],
  NDArray[np.int64],
  tuple[str, ...],
]:
  if ignore or not sink.xyz:
    return (
      np.zeros((0, 3), dtype=dtype),
      np.zeros((0,), dtype=np.int64),
      np.zeros((0,), dtype=np.int64),
      np.zeros((0,), dtype=np.int64),
      (),
    )
  coords = np.concatenate(sink.xyz).astype(dtype)
  return (
    coords,
    np.concatenate(sink.z),
    np.zeros((coords.shape[0],), dtype=np.int64),
    np.concatenate(sink.sub),
    tuple(sink.elements),
  )


def featurize(
  path: Path | str,
  *,
  fix_from_bfactor: bool = False,
  use_water: bool = False,
  noncanonical_aa_ligand: bool = False,
  ignore_ligand: bool = False,
  lig_pr_knn_k: int = LIG_PR_KNN_K,
  lig_pr_distance_cutoff: float = LIG_PR_DISTANCE_CUTOFF,
  dtype: np.dtype[np.floating] | type[np.floating] = np.float32,
) -> LaserFeatures:
  """Featurize one PDB the way LASEr inference builds ``BatchData`` before the model."""
  pr.confProDy(verbosity="none")
  pdb_path = Path(path)
  if not pdb_path.is_file():
    msg = f"missing pdb: {pdb_path}"
    raise LaserInputError(msg)
  parsed = pr.parsePDB(str(pdb_path))
  if not isinstance(parsed, pr.AtomGroup):
    msg = f"ProDy parsePDB failed for {pdb_path}"
    raise LaserInputError(msg)
  view = parsed.getHierView()
  if fix_from_bfactor:
    _require_binary_bfactors(view)
  gly = _all_gly(view)
  sequence: list[int] = []
  heavy: list[NDArray[np.float64]] = []
  phi_psi: list[NDArray[np.floating]] = []
  chains: list[int] = []
  fixed: list[bool] = []
  resindex: list[int] = []
  crystal_bb: list[NDArray[np.float64]] = []
  exposure_probe: list[NDArray[np.float64]] = []
  sink = _LigandSink()
  for chain_index, chain in enumerate(view):
    for residue in chain:
      resname = str(residue.getResname())
      letter = _LONG_TO_SHORT.get(resname, "X")
      key = (
        str(chain.getSegname()),
        str(chain.getChid()),
        int(residue.getResnum()),
        str(residue.getIcode()),
      )
      if resname == "HOH":
        if use_water:
          sink.push(residue.getCoords(), _elements(residue))
        continue
      if not _is_amino(residue):
        sink.push(residue.getCoords(), _elements(residue))
        continue
      coords = _residue_coords(residue, letter)
      angles = _phi_psi(gly[key], dtype)
      if noncanonical_aa_ligand and letter == "X":
        _push_ncaa_ligand(residue, coords, angles, sink)
        continue
      if np.isnan(coords[:3]).any():
        continue
      _append_protein(
        residue,
        letter,
        chain_index,
        coords,
        angles,
        sequence,
        heavy,
        phi_psi,
        chains,
        fixed,
        resindex,
        crystal_bb,
        exposure_probe,
      )
  if not sequence:
    msg = f"no protein residues in {pdb_path}"
    raise LaserInputError(msg)
  return _assemble(
    pdb_path,
    sequence,
    heavy,
    phi_psi,
    chains,
    fixed,
    resindex,
    crystal_bb,
    exposure_probe,
    sink,
    fix_from_bfactor=fix_from_bfactor,
    ignore_ligand=ignore_ligand,
    lig_pr_knn_k=lig_pr_knn_k,
    lig_pr_distance_cutoff=lig_pr_distance_cutoff,
    dtype=dtype,
  )


def _assemble(
  pdb_path: Path,
  sequence: list[int],
  heavy: list[NDArray[np.float64]],
  phi_psi: list[NDArray[np.floating]],
  chains: list[int],
  fixed: list[bool],
  resindex: list[int],
  crystal_bb: list[NDArray[np.float64]],
  exposure_probe: list[NDArray[np.float64]],
  sink: _LigandSink,
  *,
  fix_from_bfactor: bool,
  ignore_ligand: bool,
  lig_pr_knn_k: int,
  lig_pr_distance_cutoff: float,
  dtype: np.dtype[np.floating] | type[np.floating],
) -> LaserFeatures:
  seq = np.asarray(sequence, dtype=np.int64)
  heavy_arr = np.stack(heavy).astype(dtype)
  angles_arr = np.stack(phi_psi).astype(dtype)
  gathered = heavy_arr[:, [0, 1, 4, 2, 3]]
  backbone = idealize_backbone(gathered, angles_arr, dtype)
  heavy_arr = np.array(heavy_arr, copy=True)
  heavy_arr[:, :5] = backbone[:, [0, 1, 3, 4, 2]]
  chi = _chi_angles(heavy_arr, seq, dtype)
  length = seq.shape[0]
  ligand_coords, ligand_z, ligand_batch, ligand_sub, elements = _ligand_arrays(
    sink,
    ignore=ignore_ligand,
    dtype=dtype,
  )
  fa = build_rotamers(backbone, chi, seq, dtype)
  shell = first_shell_contact_mask(
    fa,
    seq,
    ligand_coords,
    ligand_z,
    lig_pr_knn_k=lig_pr_knn_k,
    lig_pr_distance_cutoff=lig_pr_distance_cutoff,
    dtype=dtype,
  )
  crystal = np.stack(crystal_bb)
  ss = secondary_structure(crystal)
  ss_index = np.asarray([{"-": 0, "H": 1, "E": 2}[str(letter)] for letter in ss], dtype=np.int64)
  exposed = exposed_mask(crystal[:, 1], np.stack(exposure_probe))
  chain_mask = (
    np.asarray(fixed, dtype=bool) if fix_from_bfactor else np.zeros((length,), dtype=bool)
  )
  return LaserFeatures(
    sequence_indices=seq,
    chi_angles=chi,
    chi_mask=~np.isnan(chi),
    backbone_coords=backbone,
    phi_psi_angles=angles_arr,
    heavy_atom_coords=heavy_arr,
    chain_indices=np.asarray(chains, dtype=np.int64),
    chain_mask=chain_mask,
    resnum_indices=np.arange(length, dtype=np.int64),
    batch_indices=np.zeros((length,), dtype=np.int64),
    extra_atom_contact_mask=np.zeros((length,), dtype=bool),
    first_shell_ligand_contact_mask=shell,
    sidechain_contact_number=np.zeros((length,), dtype=np.int64),
    residue_burial_counts=np.zeros((length,), dtype=np.int64),
    sampled_chain_mask=np.zeros((length,), dtype=bool),
    msa_depth_weight=np.zeros((length,), dtype=dtype),
    sc_mediated_hbond_counts=np.zeros((length,), dtype=np.int64),
    ligand_coords=ligand_coords,
    ligand_atomic_numbers=ligand_z,
    ligand_batch_indices=ligand_batch,
    ligand_subbatch_indices=ligand_sub,
    row_to_resindex=np.asarray(resindex, dtype=np.int64),
    ss_index=ss_index,
    pdb_code=str(pdb_path),
    ss_code=tuple(str(letter) for letter in ss),
    ligand_elements=elements,
    exposed_mask=exposed,
    budget_residue_mask=np.isin(ss, np.array(["H", "E"])) & exposed,
  )


def _elements_from_z(numbers: NDArray[np.integer]) -> list[str]:
  return [_SYMBOLS[int(number) - 1] for number in numbers.tolist()]
