"""Full-atom PDB reader that reproduces upstream FAMPNN structure loading without Biopython.

Ports ``fampnn.data.protein.read_pdb`` (``fampnn/data/protein.py:51-140``, run with
``max_conformers=1``) and the feature assembly of ``fampnn.data.data.load_feats_from_pdb`` /
``process_single_pdb`` (``fampnn/data/data.py:19-56, 582-617``). Behaviour reproduced:

* first model only; chains in order of first appearance; hetero residues are skipped unless their
  name is in upstream's ``ncaa_mapping`` (those become ``X``); unknown residue names become ``X``;
* atoms outside upstream's 37-name table (hydrogens, ``OXT`` is kept) are dropped; for an altloc atom
  the first conformer in the file is used (``DisorderedAtom`` list order);
* a residue with no recognised atom is skipped;
* ``residue_index = resseq + offset`` where ``offset`` counts insertion codes seen so far in the chain
  (it restarts at 0 for every chain; ``read_pdb`` increments it *before* the residue that carries the code);
* ``missing_atom_mask = (1 - atom_mask) * restype_atom37_mask[aatype]``: atoms the residue type has but
  the file lacks.

Known gaps (untested against upstream): mmCIF input; point-mutation residues (two residue names under
one residue number), which Biopython merges into a ``DisorderedResidue``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aminx.families.fampnn.constants import FAMPNN_ALPHABET, RESTYPE_ATOM37_MASK

# fampnn/data/residue_constants.py:535-574
ATOM_TYPES: tuple[str, ...] = (
  "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2", "ND1", "ND2",
  "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2",
  "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
)  # fmt: skip
_ATOM_ORDER = {name: i for i, name in enumerate(ATOM_TYPES)}

# residue_constants.py:857-885 (restype_1to3 inverted)
_THREE_TO_ONE = {
  "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
  "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
  "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}  # fmt: skip
# residue_constants.py:1302-1314: hetero residues that are kept (and typed X, because the lookup below
# uses restype_3to1, not this mapping's values)
_NCAA = frozenset({"MSE", "CME", "CSO", "TPO", "SEP", "PTR", "MLY", "HYP", "PCA", "ACE", "GTS", "GSH"})
_X = 20


@dataclass(frozen=True)
class FampnnStructure:
  """One structure as upstream FAMPNN's loader sees it (single conformer)."""

  atom37_xyz: np.ndarray  # (L, 37, 3) float64; zero where absent
  atom37_mask: np.ndarray  # (L, 37) 1 where the file had the atom
  missing_atom_mask: np.ndarray  # (L, 37) 1 where the residue type has the atom but the file does not
  aatype: np.ndarray  # (L,) FAMPNN order 0..20 (X = 20)
  residue_index: np.ndarray  # (L,) PDB residue number plus insertion-code offset
  chain_index: np.ndarray  # (L,) 0-based chain order of first appearance
  chain_id_mapping: dict[str, int]


def read_fampnn_pdb(path: str | Path) -> FampnnStructure:
  """Read ``path`` the way upstream ``read_pdb`` does.

  Raises
  ------
  ValueError
    If the file has no residue with a recognised atom.
  """
  chains: dict[str, list[dict]] = {}
  order: list[str] = []
  cur_key: tuple | None = None
  cur: dict | None = None
  offset: dict[str, int] = {}
  for line in Path(path).read_text().splitlines():
    rec = line[:6].strip()
    if rec == "ENDMDL":
      break
    if rec not in ("ATOM", "HETATM") or len(line) < 54:
      continue
    chain = line[21]
    resname = line[17:20].strip()
    key = (chain, rec == "HETATM", int(line[22:26]), line[26])
    if key != cur_key:
      cur_key = key
      if chain not in chains:
        chains[chain] = []
        order.append(chain)
        offset[chain] = 0
      cur = {"chain": chain, "het": rec == "HETATM", "resseq": key[2], "icode": line[26], "resname": resname, "atoms": {}}
      chains[chain].append(cur)
    name = line[12:16].strip()
    if name in _ATOM_ORDER and name not in cur["atoms"]:  # type: ignore[index]
      cur["atoms"][name] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))  # type: ignore[index]

  rows: list[dict] = []
  for cid in order:
    off = 0
    for res in chains[cid]:
      if res["icode"] != " ":
        off += 1
      if res["het"] and res["resname"] not in _NCAA:
        continue
      if not res["atoms"]:
        continue
      res["off"] = off
      rows.append(res)
  if not rows:
    msg = f"no residues with recognised atoms in {path}"
    raise ValueError(msg)

  num = len(rows)
  xyz = np.zeros((num, 37, 3))
  amask = np.zeros((num, 37))
  aatype = np.zeros((num,), dtype=np.int64)
  ridx = np.zeros((num,), dtype=np.int64)
  cidx = np.zeros((num,), dtype=np.int64)
  mapping = {cid: n for n, cid in enumerate(order)}
  for i, res in enumerate(rows):
    letter = _THREE_TO_ONE.get(res["resname"], "X")
    aatype[i] = FAMPNN_ALPHABET.index(letter) if letter != "X" else _X
    for name, pos in res["atoms"].items():
      xyz[i, _ATOM_ORDER[name]] = pos
      amask[i, _ATOM_ORDER[name]] = 1.0
    ridx[i] = res["resseq"] + res["off"]
    cidx[i] = mapping[res["chain"]]
  # residue_constants.restype_atom37_mask: all-zero row for X, so every atom of an X residue counts as ghost
  type_mask = np.asarray(RESTYPE_ATOM37_MASK, dtype=np.float64)[aatype]
  type_mask[aatype == _X] = 0.0
  missing = (1.0 - amask) * type_mask
  return FampnnStructure(xyz * amask[..., None], amask, missing, aatype, ridx, cidx, mapping)
