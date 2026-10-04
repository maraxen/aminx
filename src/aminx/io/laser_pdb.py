"""PDB text for LASEr full-atom coordinates.

This is the shared writer every option in
``outputs/_upstream_ref/DECISION_261004_laser-pdb-output-scope.md`` needs.
It emits the atoms already present in a heavy or protonated coordinate array,
in upstream name order, with the input residue identifiers, an optional
per-residue B-factor, and optional ligand atoms as HETATM.

Backbone amide hydrogen placement and titratable-hydrogen cleanup stay outside
this module. Both are still open in that decision (upstream
``impute_backbone_nh_coords`` and ``cleanup_titratable_hydrogens`` plus the
geometric H-bond detector). Writing them here would choose an option before
the decision is made.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aminx.families.laser_mpnn.featurize import (
  LASER_ALPHABET,
  MAX_ATOMS,
  MAX_PROTONATED_ATOMS,
  residue_atom_names,
)

# One-letter LASEr index to the PDB residue name. Same map the featurizer parses.
_RESIDUE_NAME = {
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

_PROTONATED_WIDTH = MAX_PROTONATED_ATOMS + 1
_PDB_LINE = (
  "%-6s%5d %-4s%1s%-3s%2s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f      %4s%2s%2s"
)


@dataclass(frozen=True, slots=True)
class LigandAtoms:
  """Ligand atoms appended after the protein as HETATM, in the given order."""

  coords: NDArray[np.floating]
  names: tuple[str, ...]
  elements: tuple[str, ...]
  resnames: tuple[str, ...]
  chain_ids: tuple[str, ...]
  resnums: tuple[int, ...]
  icodes: tuple[str, ...]
  segnames: tuple[str, ...] = ()
  occupancies: NDArray[np.floating] | None = None
  bfactors: NDArray[np.floating] | None = None


def write_laser_pdb(
  coords: NDArray[np.floating],
  sequence_indices: NDArray[np.integer],
  chain_ids: Sequence[str],
  resnums: Sequence[int],
  icodes: Sequence[str],
  *,
  bfactors: NDArray[np.floating] | None = None,
  segnames: Sequence[str] | None = None,
  ligand: LigandAtoms | None = None,
) -> str:
  """Return PDB text for one LASEr full-atom structure.

  ``coords`` is ``(L, A, 3)`` with ``A`` equal to ``MAX_ATOMS`` (heavy) or
  ``MAX_PROTONATED_ATOMS + 1`` (protonated). ``sequence_indices`` are LASEr
  alphabet indices. Absent atoms are the NaN slots. ``bfactors`` is one value
  per residue, written on every atom of that residue; the default is 0.

  Parameters
  ----------
  coords:
    Per-residue coordinates in heavy or protonated layout.
  sequence_indices:
    LASEr alphabet index of each residue.
  chain_ids:
    PDB chain id of each residue.
  resnums:
    PDB residue number of each residue.
  icodes:
    PDB insertion code of each residue. Blank is an empty string.
  bfactors:
    Optional per-residue temperature factor.
  segnames:
    Optional per-residue segment name. Blank when omitted.
  ligand:
    Optional ligand atoms written as HETATM after the protein.

  Returns
  -------
  str
    PDB records, one atom per line, ending with ``END``.
  """
  xyz = np.asarray(coords, dtype=np.float64)
  sequence = np.asarray(sequence_indices, dtype=np.int64)
  if xyz.ndim != 3 or xyz.shape[-1] != 3:
    msg = f"coords must be (L, A, 3), got {xyz.shape}"
    raise ValueError(msg)
  length = int(xyz.shape[0])
  width = int(xyz.shape[1])
  if sequence.shape != (length,):
    msg = f"sequence_indices must be ({length},), got {sequence.shape}"
    raise ValueError(msg)
  protonated = _protonated(width)
  _match(chain_ids, length, "chain_ids")
  _match(resnums, length, "resnums")
  _match(icodes, length, "icodes")
  segments = _texts(segnames, length)
  betas = _betas(bfactors, length)

  lines: list[str] = []
  serial = 1
  for row in range(length):
    letter = _letter(int(sequence[row]))
    names = residue_atom_names(letter, protonated=protonated)
    residue = _RESIDUE_NAME[letter]
    chain = chain_ids[row]
    number = _resnum(int(resnums[row]))
    icode = icodes[row]
    segment = segments[row]
    beta = float(betas[row])
    for slot, name in enumerate(names):
      point = np.asarray(xyz[row, slot], dtype=np.float64).reshape(3)
      if bool(np.isnan(point).any()):
        continue
      lines.append(
        _atom_line(
          record="ATOM",
          serial=serial,
          name=name,
          resname=residue,
          chain=chain,
          resnum=number,
          icode=icode,
          xyz=point,
          occupancy=1.0,
          bfactor=beta,
          segname=segment,
          element=name[0],
        )
      )
      serial += 1
  if ligand is not None:
    _append_ligand(lines, ligand, serial)
  lines.append("END")
  return "\n".join(lines) + "\n"


def _protonated(width: int) -> bool:
  if width == MAX_ATOMS:
    return False
  if width == _PROTONATED_WIDTH:
    return True
  msg = f"atom axis must be {MAX_ATOMS} or {_PROTONATED_WIDTH}, got {width}"
  raise ValueError(msg)


def _letter(index: int) -> str:
  if index < 0 or index >= len(LASER_ALPHABET):
    msg = f"sequence index {index} is outside LASER_ALPHABET"
    raise ValueError(msg)
  return LASER_ALPHABET[index]


def _match(values: Sequence[object], length: int, label: str) -> None:
  if len(values) != length:
    msg = f"{label} length {len(values)} != {length}"
    raise ValueError(msg)


def _texts(values: Sequence[str] | None, length: int) -> tuple[str, ...]:
  if values is None:
    return tuple("" for _ in range(length))
  _match(values, length, "segnames")
  return tuple(values)


def _betas(bfactors: NDArray[np.floating] | None, length: int) -> NDArray[np.float64]:
  if bfactors is None:
    return np.zeros((length,), dtype=np.float64)
  betas = np.asarray(bfactors, dtype=np.float64)
  if betas.shape != (length,):
    msg = f"bfactors must be ({length},), got {betas.shape}"
    raise ValueError(msg)
  return betas


def _resnum(number: int) -> int:
  if number < -999 or number > 9999:
    msg = f"residue number {number} does not fit in PDB columns 23-26"
    raise ValueError(msg)
  return number


def _atom_line(
  *,
  record: str,
  serial: int,
  name: str,
  resname: str,
  chain: str,
  resnum: int,
  icode: str,
  xyz: NDArray[np.float64],
  occupancy: float,
  bfactor: float,
  segname: str,
  element: str,
) -> str:
  if serial < 1 or serial > 99999:
    msg = "PDB serial numbers stop at 99999"
    raise ValueError(msg)
  atom = name[:4]
  if len(atom) < 4:
    atom = f" {atom}"
  atom = f"{atom:<4}"
  return _PDB_LINE % (
    record,
    serial,
    atom,
    " ",
    resname[:3],
    chain[:1],
    resnum,
    icode[:1],
    float(xyz[0]),
    float(xyz[1]),
    float(xyz[2]),
    float(occupancy),
    float(bfactor),
    segname[:4],
    element[:2],
    "",
  )


def _append_ligand(lines: list[str], ligand: LigandAtoms, serial: int) -> None:
  xyz = np.asarray(ligand.coords, dtype=np.float64)
  if xyz.ndim != 2 or xyz.shape[-1] != 3:
    msg = f"ligand coords must be (N, 3), got {xyz.shape}"
    raise ValueError(msg)
  count = int(xyz.shape[0])
  _match(ligand.names, count, "ligand names")
  _match(ligand.elements, count, "ligand elements")
  _match(ligand.resnames, count, "ligand resnames")
  _match(ligand.chain_ids, count, "ligand chain_ids")
  _match(ligand.resnums, count, "ligand resnums")
  _match(ligand.icodes, count, "ligand icodes")
  segments = _texts(ligand.segnames or None, count)
  occupancies = _per_atom(ligand.occupancies, count, 1.0, "ligand occupancies")
  betas = _per_atom(ligand.bfactors, count, 0.0, "ligand bfactors")
  for index in range(count):
    element = ligand.elements[index]
    if element == "":
      msg = f"ligand atom {index} is missing an element"
      raise ValueError(msg)
    lines.append(
      _atom_line(
        record="HETATM",
        serial=serial,
        name=ligand.names[index],
        resname=ligand.resnames[index],
        chain=ligand.chain_ids[index],
        resnum=_resnum(int(ligand.resnums[index])),
        icode=ligand.icodes[index],
        xyz=xyz[index],
        occupancy=float(occupancies[index]),
        bfactor=float(betas[index]),
        segname=segments[index],
        element=element,
      )
    )
    serial += 1


def _per_atom(
  values: NDArray[np.floating] | None,
  count: int,
  fill: float,
  label: str,
) -> NDArray[np.float64]:
  if values is None:
    return np.full((count,), fill, dtype=np.float64)
  array = np.asarray(values, dtype=np.float64)
  if array.shape != (count,):
    msg = f"{label} must be ({count},), got {array.shape}"
    raise ValueError(msg)
  return array
