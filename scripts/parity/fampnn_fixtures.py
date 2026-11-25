"""Fixture helpers shared by the FAMPNN oracle dump and parity vehicle (pure Python, no torch/jax).

``make_split_ab_pdb`` derives a two-chain structure from ``tests/data/1ubq.pdb`` by relabelling chain A
residues >= SPLIT_RESSEQ as chain B (coordinates and residue numbers untouched). It exercises the
chain-aware parts of the featurizer (chain embeddings, cross-chain edges, fixed-position chain
lookup) on real coordinates.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

SPLIT_RESSEQ = 40


def make_split_ab_pdb(src: Path, dst: Path, split: int = SPLIT_RESSEQ) -> str:
  """Write a two-chain copy of ``src`` to ``dst``; return the sha256 of the written file."""
  out: list[str] = []
  for line in src.read_text().splitlines():
    if line.startswith(("ATOM", "HETATM", "TER", "ANISOU")) and len(line) > 26:
      resseq = int(line[22:26])
      if line[21] == "A" and resseq >= split:
        line = line[:21] + "B" + line[22:]
    out.append(line)
  dst.parent.mkdir(parents=True, exist_ok=True)
  dst.write_text("\n".join(out) + "\n")
  return hashlib.sha256(dst.read_bytes()).hexdigest()


def _fmt_atom(line: str, *, record: str | None = None, name: str | None = None, resname: str | None = None,
              altloc: str | None = None, resseq: int | None = None, icode: str | None = None,
              occ: float | None = None, dx: float = 0.0) -> str:
  """Rewrite fixed-column fields of one PDB ATOM/HETATM line (the other columns are kept)."""
  chars = list(line.ljust(80))
  if record is not None:
    chars[0:6] = record.ljust(6)
  if name is not None:
    chars[12:16] = name.rjust(4) if len(name) == 4 else (" " + name).ljust(4)
  if altloc is not None:
    chars[16] = altloc
  if resname is not None:
    chars[17:20] = resname.ljust(3)
  if resseq is not None:
    chars[22:26] = str(resseq).rjust(4)
  if icode is not None:
    chars[26] = icode
  if occ is not None:
    chars[54:60] = f"{occ:6.2f}"
  if dx:
    chars[30:38] = f"{float(line[30:38]) + dx:8.3f}"
  return "".join(chars).rstrip()


def make_parser_stress_pdb(src: Path, dst: Path) -> str:
  """Derive a parser stress input from ``src`` (1ubq): altlocs, insertion codes, HETATM, missing and unknown residues.

  * residue 7: CB/CG1 get altlocs A (occupancy 0.40) then B (0.60, shifted +0.5 A in x): upstream keeps the first
    conformer in file order, which is not the higher-occupancy one;
  * residues 36 and 37 become 35A and 35B (insertion codes), so later residues are offset by 2;
  * residue 1 (MET) becomes HETATM MSE (a mapped non-canonical residue: kept, typed X), SD renamed SE;
  * residue 20 loses every atom after CB (missing sidechain atoms);
  * residue 50 is renamed ABC (unknown ATOM residue: typed X);
  * a HETATM GOL residue (unmapped het: skipped) is appended after the chain, before the waters.
  """
  out: list[str] = []
  gol_done = False
  for line in src.read_text().splitlines():
    rec = line[:6].strip()
    if rec in ("ATOM", "HETATM") and len(line) >= 54:
      resseq = int(line[22:26])
      name = line[12:16].strip()
      if rec == "HETATM" and not gol_done:
        for i, (nm, el) in enumerate((("C1", "C"), ("O1", "O"))):
          out.append(f"HETATM{9000 + i:5d} {nm:<3s}  GOL A 200      10.000  10.000  1{i}.000  1.00 20.00           {el}")
        gol_done = True
      if resseq == 7 and name in ("CB", "CG1"):
        out.append(_fmt_atom(line, altloc="A", occ=0.40))
        out.append(_fmt_atom(line, altloc="B", occ=0.60, dx=0.5))
        continue
      if resseq in (36, 37):
        line = _fmt_atom(line, resseq=35, icode="A" if resseq == 36 else "B")
      elif resseq == 1:
        line = _fmt_atom(line, record="HETATM", resname="MSE", name="SE" if name == "SD" else None)
      elif resseq == 20 and name not in ("N", "CA", "C", "O", "CB"):
        continue
      elif resseq == 50:
        line = _fmt_atom(line, resname="ABC")
    out.append(line)
  dst.parent.mkdir(parents=True, exist_ok=True)
  dst.write_text("\n".join(out) + "\n")
  return hashlib.sha256(dst.read_bytes()).hexdigest()
