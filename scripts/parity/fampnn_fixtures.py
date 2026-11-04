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
