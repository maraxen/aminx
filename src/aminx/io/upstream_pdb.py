"""Upstream Dauparas PDB parser shared across MPNN families."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_ALPHA_1 = "ARNDCQEGHILKMFPSTWYV-"
_ALPHA_3 = [
  "ALA",
  "ARG",
  "ASN",
  "ASP",
  "CYS",
  "GLN",
  "GLU",
  "GLY",
  "HIS",
  "ILE",
  "LEU",
  "LYS",
  "MET",
  "PHE",
  "PRO",
  "SER",
  "THR",
  "TRP",
  "TYR",
  "VAL",
  "GAP",
]
_AA_3_N = {name: index for index, name in enumerate(_ALPHA_3)}
_AA_N_1 = dict(enumerate(_ALPHA_1))
_BACKBONE = ("N", "CA", "C", "O")
_CHAIN_ALPHABET = [
  *"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
  *[str(item) for item in range(300)],
]


class PottsInputError(ValueError):
  """Raised when an MPNN PDB input is not a ``.pdb`` file."""


def _parse_biounits(
  path: Path,
  atoms: Sequence[str],
  chain: str | None,
  *,
  skip_gaps: bool,
) -> tuple[np.ndarray, str] | tuple[str, str]:
  """Parse one chain. Returns ``('no_chain', 'no_chain')`` when the chain is absent."""
  xyz: dict[int, dict[str, dict[str, np.ndarray]]] = {}
  seq: dict[int, dict[str, str]] = {}
  min_resn = 10**6
  max_resn = -(10**6)
  with path.open("rb") as handle:
    for raw in handle:
      line = raw.decode("utf-8", "ignore").rstrip()
      if line[:6] == "HETATM" and line[17:20] == "MSE":
        line = line.replace("HETATM", "ATOM  ")
        line = line.replace("MSE", "MET")
      if line[:4] != "ATOM":
        continue
      if chain is not None and line[21:22] != chain:
        continue
      atom = line[12:16].strip()
      resname = line[17:20]
      resn_token = line[22:27].strip()
      if resn_token[-1].isalpha():
        icode, resn = resn_token[-1], int(resn_token[:-1]) - 1
      else:
        icode, resn = "", int(resn_token) - 1
      min_resn = min(min_resn, resn)
      max_resn = max(max_resn, resn)
      xyz.setdefault(resn, {}).setdefault(icode, {})
      seq.setdefault(resn, {}).setdefault(icode, resname)
      if atom not in xyz[resn][icode]:
        coords = [float(line[start : start + 8]) for start in (30, 38, 46)]
        xyz[resn][icode][atom] = np.asarray(coords, dtype=np.float64)

  if not xyz:
    return "no_chain", "no_chain"

  seq_idx: list[int] = []
  xyz_rows: list[np.ndarray] = []
  for resn in range(min_resn, max_resn + 1):
    if resn in seq:
      for icode in sorted(seq[resn]):
        seq_idx.append(_AA_3_N.get(seq[resn][icode], 20))
        for atom in atoms:
          coords = xyz[resn][icode].get(atom)
          xyz_rows.append(np.full(3, np.nan) if coords is None else coords)
    elif skip_gaps:
      continue
    else:
      seq_idx.append(20)
      xyz_rows.extend(np.full(3, np.nan) for _atom in atoms)
  coords_out = np.asarray(xyz_rows, dtype=np.float64).reshape(-1, len(atoms), 3)
  sequence = "".join(_AA_N_1.get(index, "-") for index in seq_idx)
  return coords_out, sequence


def parse_pdb_upstream(
  path: str | Path,
  chains: Sequence[str] | None = None,
  skip_gaps: bool = False,  # noqa: FBT001, FBT002
  *,
  ca_only: bool = False,
) -> list[dict[str, Any]]:
  """Port of ``parse_PDB`` / ``parse_PDB_biounits``.

  Absent chains are dropped. Chain order is ``chains`` when given, otherwise the upstream
  alphabet (A-Z, a-z, ``"0"`` .. ``"299"``). Non-``.pdb`` paths raise ``PottsInputError``.
  """
  pdb_path = Path(path)
  if pdb_path.suffix.lower() != ".pdb":
    raise PottsInputError("pottsmpnn_requires_pdb")
  chain_alphabet = list(chains) if chains else list(_CHAIN_ALPHABET)
  atoms = ("CA",) if ca_only else _BACKBONE
  parsed: dict[str, Any] = {}
  concat_seq = ""
  chain_order: list[str] = []
  for letter in chain_alphabet:
    xyz, seq = _parse_biounits(pdb_path, atoms, letter, skip_gaps=skip_gaps)
    if isinstance(xyz, str):
      continue
    concat_seq += seq
    parsed[f"seq_chain_{letter}"] = seq
    coord_map: dict[str, list[list[float]]] = {}
    if ca_only:
      coord_map[f"CA_chain_{letter}"] = xyz.reshape(-1, 3).tolist()
    else:
      for index, atom in enumerate(_BACKBONE):
        coord_map[f"{atom}_chain_{letter}"] = xyz[:, index, :].tolist()
    parsed[f"coords_chain_{letter}"] = coord_map
    chain_order.append(letter)
  text = pdb_path.as_posix()
  parsed["name"] = text[text.rfind("/") + 1 : -4]
  parsed["num_of_chains"] = len(chain_order)
  parsed["seq"] = concat_seq
  parsed["chain_order"] = chain_order
  return [parsed]
