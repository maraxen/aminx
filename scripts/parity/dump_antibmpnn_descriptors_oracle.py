"""Sealed descriptor oracle for AntiBMPNN's design post-processing (run on titanix, ``peptides`` env).

Upstream scores designed segments in ``helper_scripts/parsing_design_result.py``:

* ``CDRH3_Comb`` (line 5) slices ``seq[start-1:end]`` (lines 24-26), then
  ``peptides.Peptide(seq).physical_descriptors()`` -> PD1 (volume), PD2 (hydrophobicity)
  (lines 44-48) and ``peptide.charge(pH=7.4)`` (default ``pKscale="Lehninger"``, line 51).

This script (1) runs the REAL ``CDRH3_Comb`` on the ``.fa`` produced by the real runner
(``dump_antibmpnn_oracle.py``) and reads back its CSV, and (2) calls the same ``peptides`` API at
full precision on a set of sampled / native sequences and slices, since the CSV rounds to 4 decimals.
Writes ``antibmpnn_descriptors_oracle.npz`` + manifest with the sha256 of every input.

Usage (titanix, venv with pandas + peptides)::

    pepvenv/bin/python scripts/parity/dump_antibmpnn_descriptors_oracle.py \\
        --antibmpnn-root /home/solab/repos/AntiBMPNN --oracle-dir /home/solab/scratch/p4-antibmpnn/oracle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
START, END = 102, 106  # example_scripts.sh DESIGN_ONLY_POSITIONS first/last


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--antibmpnn-root", type=Path, required=True)
  parser.add_argument("--oracle-dir", type=Path, required=True)
  args = parser.parse_args()
  root = args.antibmpnn_root.resolve()
  out = args.oracle_dir.resolve()
  sys.path.insert(0, str(root / "helper_scripts"))
  import pandas as pd
  import parsing_design_result as pdr  # the REAL upstream helper
  import peptides

  assert Path(pdr.__file__).resolve() == root / "helper_scripts" / "parsing_design_result.py"
  oracle = np.load(out / "antibmpnn_oracle.npz")

  # (1) the real CDRH3_Comb on the real runner's .fa (copied so the upstream CSV lands in our dir)
  fa = next((out / "cli_run" / "seqs").glob("*.fa"))
  fa_copy = out / "cli_run" / "designs_for_parsing.fa"
  fa_copy.write_text(fa.read_text())
  pdr.CDRH3_Comb(str(fa_copy), START, END)
  csv = pd.read_csv(str(fa_copy) + "_Paresd_Result.csv")
  cols = ["combination", "PD1_volume", "PD2_hydro", "PD3_charge"]
  dump: dict[str, np.ndarray] = {
    "csv/combination": csv["combination"].astype(str).to_numpy(),
    "csv/PD1_volume": csv["PD1_volume"].to_numpy(dtype=np.float64),
    "csv/PD2_hydro": csv["PD2_hydro"].to_numpy(dtype=np.float64),
    "csv/PD3_charge": csv["PD3_charge"].to_numpy(dtype=np.float64),
  }
  assert set(cols) <= set(csv.columns)

  # (2) full-precision peptides calls on sampled sequences and slices (parsing_design_result.py:44-52)
  seqs: list[str] = ["".join(ALPHABET[i] for i in oracle["feat/cdr5/S"][0])]
  for cfg in ("cdr5", "knobs"):
    s = oracle[f"antibmpnn_000/f32/{cfg}/S_sample"][0]
    seqs.append("".join(ALPHABET[i] for i in s))
  seqs.extend(["DEKRHCY", "KKKKKRRRRR", "DDDEEE", "GSGSGS", "AVILMFWC", "HHHHHH"])
  slices = [(1, 121), (START, END), (26, 35), (50, 66)]
  rows = []
  for seq in seqs:
    for lo, hi in slices:
      part = seq[lo - 1 : hi]
      if not part:
        continue
      pep = peptides.Peptide(part)
      pd1, pd2 = pep.physical_descriptors()
      rows.append((part, float(pd1), float(pd2), float(pep.charge(pH=7.4))))
  dump["direct/seq"] = np.asarray([r[0] for r in rows])
  dump["direct/PD1"] = np.asarray([r[1] for r in rows], dtype=np.float64)
  dump["direct/PD2"] = np.asarray([r[2] for r in rows], dtype=np.float64)
  dump["direct/charge74"] = np.asarray([r[3] for r in rows], dtype=np.float64)
  # pH sweep and a pKa-sensitive probe (a wrong pKa scale must move these)
  probe = "DEKRHCYDEKRHCY"
  dump["probe/seq"] = np.asarray(probe)
  dump["probe/ph"] = np.asarray([5.0, 6.0, 7.0, 7.4, 8.0, 9.0, 10.0])
  dump["probe/charge"] = np.asarray([peptides.Peptide(probe).charge(pH=p) for p in dump["probe/ph"]], dtype=np.float64)
  dump["start_end"] = np.asarray([START, END])
  path = out / "antibmpnn_descriptors_oracle.npz"
  np.savez_compressed(path, **dump)
  manifest = {
    "dump": path.name,
    "dump_sha256": sha256_file(path),
    "peptides_version": peptides.__version__,
    "parsing_design_result_sha256": sha256_file(root / "helper_scripts" / "parsing_design_result.py"),
    "fa_sha256": sha256_file(fa),
    "antibmpnn_oracle_dump_sha256": sha256_file(out / "antibmpnn_oracle.npz"),
  }
  (out / "antibmpnn_descriptors_oracle.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
  main()
