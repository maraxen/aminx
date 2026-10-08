"""Dump the PottsMPNN ONNX graph inputs for one PDB, produced by aminx's own Python path.

task_id 261007_potts-onnx-export, backlog #5813 (X3 part A). Reference for the browser port
``browser/potts-sampler/potts_inputs.mjs``. The path is the driver's own:

  parse_pdb_upstream(pdb, skip_gaps=options.skip_gaps)[0]
  -> _featurize_one(parsed, PottsMPNNOptions(), name)
  -> prepare_sample(features, chains, options, spec, l_pad=bucket)

with spec from the optional controls (default: fixed_positions None, omit_aa (), bias None):
--fixed-positions 1,5,9, --omit-aa ACW, --bias-json bias.json. Output JSON:

  {"pdb", "pdb_sha256", "bucket", "l_total", "sequence", "chains",
   "controls": {"fixed_positions", "omit_aa", "bias"} (null when absent),
   "inputs": {name: {"dtype", "dims", "data"}}}

Every array is flattened row-major. Float NaN becomes null. bool arrays are written as 0/1
with dtype "bool". Infinities are rejected because JSON cannot carry them.

Run with the project environment, e.g.:
  uv run python scripts/browser_validation/potts_dump_inputs.py --pdb X.pdb --bucket 128 --out X_L128.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
  sys.path.insert(0, str(REPO))

logger = logging.getLogger("potts_dump_inputs")

INPUT_NAMES = (
  "coords",
  "present",
  "residue_idx",
  "chain_index",
  "pad_valid",
  "s_true",
  "chain_mask",
  "chain_m_pos",
  "tie_groups",
  "tied_beta",
  "omit",
  "bias",
  "bias_by_res",
  "pssm_coef",
  "pssm_bias",
  "pssm_log_odds_mask",
  "omit_aa_mask",
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _encode(arr: np.ndarray) -> dict[str, Any]:
  """One array as {dtype, dims, data}. NaN -> null, bool -> 0/1, inf is an error."""
  arr = np.asarray(arr)
  if arr.dtype == np.bool_:
    return {"dtype": "bool", "dims": list(arr.shape), "data": [int(v) for v in arr.ravel()]}
  if np.issubdtype(arr.dtype, np.floating):
    data: list[float | None] = []
    for value in arr.astype(np.float64).ravel().tolist():
      if math.isinf(value):
        msg = f"infinite value in {arr.dtype} array; JSON cannot represent it"
        raise ValueError(msg)
      data.append(None if math.isnan(value) else value)
    return {"dtype": str(arr.dtype), "dims": list(arr.shape), "data": data}
  return {"dtype": str(arr.dtype), "dims": list(arr.shape), "data": [int(v) for v in arr.ravel()]}


def _parse_fixed_positions(text: str | None) -> list[int] | None:
  """``"1,5,9"`` -> ``[1, 5, 9]``. ``None`` when the flag is absent."""
  if text is None:
    return None
  return [int(token) for token in text.split(",") if token.strip()]


def _load_bias(path: Path) -> list[Any]:
  """JSON list (length 21 or L) or list of lists (L x 21), as written to the dump's controls."""
  data = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(data, list):
    msg = f"{path}: bias JSON must be a list or a list of lists"
    raise SystemExit(msg)
  return data


def dump_inputs(
  pdb: Path,
  bucket: int,
  *,
  fixed_positions: list[int] | None = None,
  omit_aa: str | None = None,
  bias: list[Any] | None = None,
) -> dict[str, Any]:
  """Run the aminx Python path on one PDB and return the JSON-ready payload.

  ``fixed_positions`` are global 0-based rows, ``omit_aa`` a string of letters, and ``bias``
  a JSON-shaped list passed through ``prepare_sample`` as ``spec.bias`` (float32). The
  controls are recorded under ``"controls"`` exactly as given (``null`` when absent).
  """
  from aminx.families.potts_mpnn.driver import _chain_sequences, _featurize_one  # noqa: PLC0415
  from aminx.families.potts_mpnn.featurize import parse_pdb_upstream  # noqa: PLC0415
  from aminx.families.potts_mpnn.sample_host import prepare_sample  # noqa: PLC0415
  from aminx.run.options import PottsMPNNOptions  # noqa: PLC0415

  options = PottsMPNNOptions()
  fixed_arr = None if fixed_positions is None else np.asarray(fixed_positions, dtype=np.int32)
  bias_arr = None if bias is None else np.asarray(bias, dtype=np.float32)
  spec = SimpleNamespace(
    fixed_positions=fixed_arr,
    omit_aa=() if omit_aa is None else tuple(omit_aa),
    bias=bias_arr,
  )
  controls = {"fixed_positions": fixed_positions, "omit_aa": omit_aa, "bias": bias}
  parsed = parse_pdb_upstream(pdb, skip_gaps=options.skip_gaps)[0]
  features = _featurize_one(parsed, options, str(parsed["name"]))
  if int(features.L_total) > bucket:
    msg = f"{pdb.name}: L_total {features.L_total} does not fit bucket {bucket}"
    raise SystemExit(msg)
  chains = tuple((c.letter, c.sequence) for c in _chain_sequences(parsed, features))
  ready = prepare_sample(features, chains, options, spec, l_pad=bucket)

  inputs = {name: _encode(getattr(ready, name)) for name in INPUT_NAMES}
  return {
    "pdb": pdb.name,
    "pdb_sha256": _sha256(pdb),
    "bucket": bucket,
    "l_total": int(ready.l_total),
    "sequence": "".join(sequence for _letter, sequence in chains),
    "chains": [letter for letter, _sequence in chains],
    "controls": controls,
    "inputs": inputs,
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--pdb", type=Path, required=True)
  parser.add_argument("--bucket", type=int, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument(
    "--fixed-positions",
    default=None,
    help="comma-separated global 0-based rows, e.g. 1,5,9 (spec.fixed_positions)",
  )
  parser.add_argument("--omit-aa", default=None, help="letters to omit, e.g. ACW (spec.omit_aa)")
  parser.add_argument(
    "--bias-json",
    type=Path,
    default=None,
    help="JSON list (length 21, length L, or L x 21) for spec.bias",
  )
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

  payload = dump_inputs(
    args.pdb,
    args.bucket,
    fixed_positions=_parse_fixed_positions(args.fixed_positions),
    omit_aa=args.omit_aa,
    bias=None if args.bias_json is None else _load_bias(args.bias_json),
  )
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
  logger.info(
    "wrote %s (L_total=%d, bucket=%d, chains=%s)",
    args.out,
    payload["l_total"],
    args.bucket,
    "".join(payload["chains"]),
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
