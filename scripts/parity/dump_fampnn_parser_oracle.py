"""Upstream FAMPNN structure-loader output on the parser stress input (altlocs, insertion codes, HETATM, X residues).

Runs the real ``fampnn.data.data.load_feats_from_pdb`` + ``process_single_pdb`` (``~/repos/fampnn``, Biopython
``PDBParser`` underneath) on ``tests/data/fampnn_parser_stress.pdb`` and writes ``tests/data/fampnn_parser_oracle.json``.
``tests/families/test_fampnn_pdb_parser.py`` compares ``aminx.families.fampnn.pdb.read_fampnn_pdb`` against it with
no torch dependency. Nothing from aminx is imported here. Run on titanix with the oracle environment of
``launch_fampnn_oracle.sh`` (``PYTHONPATH=<P4_ROOT>/stubs:<P4_ROOT>/extra``).

usage: dump_fampnn_parser_oracle.py --fampnn-root ~/repos/fampnn --pdb tests/data/fampnn_parser_stress.pdb --out tests/data/fampnn_parser_oracle.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--fampnn-root", required=True)
  ap.add_argument("--pdb", required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  sys.path.insert(0, str(Path(args.fampnn_root).resolve()))
  assert "aminx" not in sys.modules
  from fampnn.data.data import load_feats_from_pdb, process_single_pdb

  data = load_feats_from_pdb(args.pdb)
  single = process_single_pdb(data)
  arrays = {
    "x": single["x"].numpy().astype(np.float64),
    "atom_mask": data["all_atom_mask"].numpy().astype(np.float64),
    "missing_atom_mask": data["missing_atom_mask"].numpy().astype(np.float64),
    "aatype": data["aatype"].numpy().astype(np.int64),
    "residue_index": data["residue_index"].numpy().astype(np.int64),
    "chain_index": data["chain_index"].numpy().astype(np.int64),
    "seq_mask": single["seq_mask"].numpy().astype(np.float64),
  }
  # plain JSON (not an LFS-tracked .npz) so the committed reference is readable in any checkout
  payload = {k: v.tolist() for k, v in arrays.items()}
  payload["chain_id_mapping"] = data["chain_id_mapping"]
  payload["pdb_sha256"] = hashlib.sha256(Path(args.pdb).read_bytes()).hexdigest()
  Path(args.out).write_text(json.dumps(payload, sort_keys=True))
  print({k: v.shape for k, v in arrays.items()})
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
