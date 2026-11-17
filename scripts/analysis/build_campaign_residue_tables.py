"""Build per-target residue tables with the parser a campaign actually ran.

Run this inside an environment pinned to the campaign's lock (for the tev_design AR campaign:
aminx 0.1.0a19 wheel + proxide 0.1.0a11). Each table records native tokens in MPNN order, residue
labels, chain lengths and the parser identity, so a downstream EDA never re-parses with a different
parser version (proxide 0.2.0 reorders multi-chain targets). Task 261106_campaign-eda-respec.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

AF2_ORDER = "ARNDCQEGHILKMFPSTWYVX"
MPNN_ORDER = "ACDEFGHIKLMNPQRSTVWYX"
AF2_TO_MPNN = np.array([MPNN_ORDER.index(c) for c in AF2_ORDER])

log = logging.getLogger("build_campaign_residue_tables")


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--manifest", type=Path, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--expect-aminx", required=True)
  ap.add_argument("--expect-proxide", required=True)
  args = ap.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

  from aminx.io.parsing.dispatch import parse_input  # noqa: PLC0415 - the pinned build's parser

  identity = {"aminx": md.version("aminx"), "proxide": md.version("proxide"), "python": sys.version.split()[0]}
  manifest = json.loads(args.manifest.read_text())
  inputs = sorted({r["sampling_spec"]["inputs"][0] for r in manifest["rows"]})
  tables = {}
  for path in inputs:
    proteins = list(parse_input(path))
    if len(proteins) != 1:
      raise ValueError(f"{path}: expected one model, got {len(proteins)}")
    p = proteins[0]
    aa = np.asarray(p.aatype).astype(int)
    resnum = np.asarray(p.residue_index).astype(int)
    chain = np.asarray(p.chain_index).astype(int)
    tables[path] = {
      "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
      "length": int(aa.shape[0]),
      "native_mpnn": AF2_TO_MPNN[aa].tolist(),
      "residue_label": [f"{AF2_ORDER[a]}{n}/c{c}" for a, n, c in zip(aa, resnum, chain, strict=True)],
      "chain_lengths": np.bincount(chain).tolist(),
    }
    log.info("%s: L=%d chains=%s", Path(path).name, aa.shape[0], tables[path]["chain_lengths"])
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps({"parser_identity": identity, "alphabet": MPNN_ORDER, "tables": tables}, indent=1))

  results = {
    "n_targets": len(tables),
    "aminx_version": identity["aminx"],
    "proxide_version": identity["proxide"],
    "parser_matches_campaign": identity["aminx"] == args.expect_aminx and identity["proxide"] == args.expect_proxide,
  }
  log.info("results: %s", json.dumps(results))
  if rp := os.environ.get("BTH_RESULTS_PATH"):
    Path(rp).write_text(json.dumps(results))
  return 0


if __name__ == "__main__":
  sys.exit(main())
