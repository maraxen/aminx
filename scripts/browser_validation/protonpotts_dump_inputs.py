"""Dump the ProtonPotts table-graph inputs for one PDB, produced by aminx's own Python path.

task_id 261009_protonpotts-onnx, backlog #5816. Reference for the browser port
``browser/protonpotts-scorer/protonpotts_inputs.mjs``. The Python path is the driver's own::

  loaded_residues(pdb) -> labels -> featurize_pdb(pdb, labels) -> kept_residues(pdb)
  -> coords = X[0,:,:4,:], present = 1, residue_idx = R_idx[0], chain_index = chain_labels[0]
  -> padded to the bucket by the potts_mpnn.featurize.pad recipe

Labels: ``--labels-json`` (a list, one entry per loaded residue, '' for none) or ``--auto-labels``, a deterministic rule
that exercises the protonation tokens: HIS alternates HIS-P / HIS-A, ASP alternates ASP-D / ASP-A, GLU is GLU-P.
Output JSON: ``{pdb, pdb_sha256, bucket, labels, loaded, kept, tokens, inputs: {name: {dtype, dims, data}}}``;
every array row-major flattened, bool as 0/1 with dtype "bool".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

logger = logging.getLogger("protonpotts_dump_inputs")

AUTO = {"HIS": ("HIS-P", "HIS-A"), "ASP": ("ASP-D", "ASP-A"), "GLU": ("GLU-P", "GLU-P")}


def auto_labels(loaded: list[tuple[str, int, str, str]]) -> list[str]:
  seen: dict[str, int] = {}
  out = []
  for _chain, _number, _icode, name in loaded:
    if name in AUTO:
      n = seen.get(name, 0)
      out.append(AUTO[name][n % 2])
      seen[name] = n + 1
    else:
      out.append("")
  return out


def dump(pdb: Path, bucket: int, labels: list[str] | None, *, auto: bool) -> dict[str, Any]:
  from aminx.families.protonpotts_mpnn.features import featurize_pdb, kept_residues, loaded_residues  # noqa: PLC0415

  loaded = loaded_residues(pdb)
  if auto:
    labels = auto_labels(loaded)
  feats = featurize_pdb(pdb, labels)
  kept = kept_residues(pdb)
  length = int(feats["S"].shape[1])
  n_pad = bucket - length
  if n_pad < 0:
    msg = f"bucket {bucket} < L_total {length}"
    raise ValueError(msg)

  def tail(row: np.ndarray, fill: float) -> np.ndarray:
    return np.concatenate([row, np.full((n_pad, *row.shape[1:]), fill, dtype=row.dtype)]) if n_pad else row

  arrays = {
    "coords": tail(np.asarray(feats["X"][0, :, :4, :], dtype=np.float32), 0.0),
    "present": tail(np.ones(length, dtype=np.float32), 0.0),
    "residue_idx": tail(np.asarray(feats["R_idx"][0], dtype=np.int32), -100),
    "chain_index": tail(np.asarray(feats["chain_labels"][0], dtype=np.int32), 0),
    "pad_valid": np.arange(bucket) < length,
  }
  inputs = {}
  for name, arr in arrays.items():
    is_bool = arr.dtype == np.bool_
    data = arr.astype(np.uint8) if is_bool else arr
    inputs[name] = {"dtype": "bool" if is_bool else str(arr.dtype), "dims": list(arr.shape),
                    "data": data.reshape(-1).tolist()}
  return {
    "pdb": pdb.name, "pdb_sha256": hashlib.sha256(pdb.read_bytes()).hexdigest(), "bucket": bucket,
    "labels": labels, "loaded": [list(r) for r in loaded], "kept": [list(r) for r in kept],
    "tokens": [int(t) for t in feats["S"][0]], "inputs": inputs,
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--pdb", type=Path, required=True)
  parser.add_argument("--bucket", type=int, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--labels-json", type=Path, default=None)
  parser.add_argument("--auto-labels", action="store_true")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  labels = json.loads(args.labels_json.read_text()) if args.labels_json else None
  record = dump(args.pdb, args.bucket, labels, auto=args.auto_labels)
  args.out.write_text(json.dumps(record))
  logger.info("wrote %s (L_total %d, bucket %d)", args.out, len(record["tokens"]), args.bucket)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
