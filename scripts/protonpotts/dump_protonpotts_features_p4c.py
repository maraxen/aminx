"""Dump upstream ProtonPottsMPNN (v6) features for the P4c cells: the behaviours P4b could not settle.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §39. Run in the
`aminx-oracles-protonpotts` environment; imports upstream `mpnn`, torch and numpy only, never aminx::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/protonpotts/dump_protonpotts_features_p4c.py --out <dir> \\
        --cell only_o=<1CQW.pdb> --cell gap=<1EL1.pdb> --cell many_gaps=<1BAH.pdb> \\
        --cell gap_and_drops=<2yc3.pdb> --cell icode=<1TPK.pdb> --cell many_drops=<1IFC.pdb>

WHY. The aminx featurizer matches the four P4b cells exactly on all six keys, but those cells cannot
discriminate four rules the featurizer had to choose (spec §39.1): whether O counts as a backbone atom
for dropping a residue, whether `R_idx` follows residue NUMBERING or merely counts residues, what
happens when a chain's first residue is dropped, and how insertion codes are numbered. The cells here
were chosen by a scan of the 137 available structures for exactly those features.

Unlike P4b, a cell may legitimately FAIL upstream (an insertion code may not featurize). The failure is
recorded as data, not hidden: a refused input is part of upstream's contract.

Thread pinning and the reproducibility child are shared with the P4b script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
  os.environ[_var] = "1"

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_protonpotts_features import (  # noqa: E402
  REQUIRED_KEYS,
  _content_hash,
  _featurize,
)

logger = logging.getLogger("dump_protonpotts_features_p4c")


def _parse_cells(raw: list[str]) -> dict[str, Path]:
  cells: dict[str, Path] = {}
  for item in raw:
    name, _, path = item.partition("=")
    if not name or not path:
      msg = f"--cell expects NAME=PATH, got {item!r}"
      raise SystemExit(msg)
    cells[name] = Path(path)
  return cells


def _run_cells(cells: dict[str, Path], out: Path) -> dict[str, dict]:
  out.mkdir(parents=True, exist_ok=True)
  results: dict[str, dict] = {}
  for name, pdb in cells.items():
    try:
      arrays = _featurize(pdb, labelled=False)
    except Exception as exc:  # noqa: BLE001 -- a refusal is data
      results[name] = {
        "ok": False,
        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        "traceback_tail": traceback.format_exc().splitlines()[-3:],
      }
      logger.info("cell %s raised upstream: %s", name, results[name]["error"])
      continue
    np.savez(out / f"{name}.npz", **arrays)
    results[name] = {
      "ok": True,
      "content_sha256": _content_hash(arrays),
      "n_residues": int(arrays["S"].shape[-1]),
      "keys": sorted(arrays),
    }
  return results


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cell", action="append", default=[], metavar="NAME=PATH")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--child-results-to", type=Path, default=None)
  parser.add_argument("--log-level", default="INFO")
  args = parser.parse_args()
  logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
  cells = _parse_cells(args.cell)

  results = _run_cells(cells, args.out / "protonpotts_v6_features_p4c")
  if args.child_results_to is not None:
    args.child_results_to.write_text(json.dumps(results), encoding="utf-8")
    return 0

  with tempfile.TemporaryDirectory() as tmp:
    child_json = Path(tmp) / "child.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--out", str(Path(tmp) / "c"),
               "--child-results-to", str(child_json)]
    for name, pdb in cells.items():
      command += ["--cell", f"{name}={pdb}"]
    subprocess.run(command, check=True, capture_output=True)
    child = json.loads(child_json.read_text(encoding="utf-8"))

  def _comparable(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "traceback_tail"}

  reproducible = all(_comparable(results[n]) == _comparable(child[n]) for n in cells)
  featurized = {n: r for n, r in results.items() if r["ok"]}
  flags = {
    "keys_complete": all(all(k in r["keys"] for k in REQUIRED_KEYS) for r in featurized.values()),
    "reproducible": reproducible,
    "n_cells": len(cells),
    "n_featurized": len(featurized),
    "n_raised": len(cells) - len(featurized),
  }
  files = {
    n: hashlib.sha256((args.out / "protonpotts_v6_features_p4c" / f"{n}.npz").read_bytes()).hexdigest()
    for n in featurized
  }
  (args.out / "features_p4c_manifest.json").write_text(
    json.dumps(
      {"cells": {n: str(p) for n, p in cells.items()}, "results": results, "file_sha256": files, "flags": flags},
      indent=2,
      sort_keys=True,
    ),
    encoding="utf-8",
  )
  results_path = os.environ.get("BTH_RESULTS_PATH")
  if results_path:
    Path(results_path).write_text(
      json.dumps({**flags, "raised_cells": sorted(n for n, r in results.items() if not r["ok"])}, indent=2, sort_keys=True),
      encoding="utf-8",
    )
  else:
    logger.warning("BTH_RESULTS_PATH unset: this run will record outcome 'unknown'")
  return 0


if __name__ == "__main__":
  sys.exit(main())
