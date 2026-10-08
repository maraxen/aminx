"""Dump upstream ProtonPottsMPNN (v6) Potts energies of fixed sequences, f32 and f64 (P4f).

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §42. Run in the `aminx-oracles-protonpotts`
environment; imports upstream `mpnn`, torch and numpy only, never aminx::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/protonpotts/dump_protonpotts_energy.py --encoder-dir <P4e out dir> \\
        --features-dir ~/projects/aminx-oracles-protonpotts --out <dir>

WHY. The sealed P4e dump holds upstream's merged table (`etab_postmerge`) per cell and precision, so the
aminx merge can be graded against it directly. It holds no ENERGY, and the energy is what `score:energy` and
`score:ddg` report. This applies upstream's own `PottsMPNN.calc_potts_eners` (a static method: no model, no
checkpoint) to that sealed table and E_idx for a fixed set of sequences per cell: the native `S`, eight
sequences drawn uniformly over all 30 tokens (so protonation tokens appear), and one sweep that cycles every
token through the chain (so every row and column of the table is indexed at least once).

Sequences are in UPSTREAM (foundry) token order; the aminx wave maps them with `upstream_to_aminx_index`.

The tables are not recomputed: their content hashes are checked against the sealed P4e manifest, so the
energies are of exactly the tables the encoder wave already graded. Threads pinned to 1 (spec §32).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
  os.environ[_var] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_protonpotts_features import _content_hash  # noqa: E402

logger = logging.getLogger("dump_protonpotts_energy")

SEED = 0
N_RANDOM = 8
V = 30
PRECISIONS = (("f32", torch.float32), ("f64", torch.float64))
CELLS = {
  "pkad_unlabelled": "features_v6/protonpotts_v6_features",
  "multichain": "features_v6/protonpotts_v6_features",
  "gap": "features_v6_p4c/protonpotts_v6_features_p4c",
  "only_o_1olr": "features_v6_p4d/protonpotts_v6_features_p4c",
}


def _sequences(native: np.ndarray, cell_index: int) -> np.ndarray:
  length = native.shape[0]
  rng = np.random.default_rng(SEED + cell_index)
  random_rows = rng.integers(0, V, size=(N_RANDOM, length))
  sweep = (np.arange(length) % V)[None, :]
  return np.concatenate([native[None, :], random_rows, sweep], axis=0).astype(np.int64)


def _energies(encoder_dir: Path, features_dir: Path, out: Path, manifest: dict) -> dict[str, dict]:
  from mpnn.model.pottsmpnn import PottsMPNN

  out.mkdir(parents=True, exist_ok=True)
  report: dict[str, dict] = {}
  for index, (cell, features_rel) in enumerate(CELLS.items()):
    with np.load(features_dir / features_rel / f"{cell}.npz") as z:
      native = z["S"][0]
    seqs = _sequences(native, index)
    entry: dict = {"n_sequences": int(seqs.shape[0]), "precisions": {}}
    for name, dtype in PRECISIONS:
      path = encoder_dir / "protonpotts_v6_encoder" / f"{cell}_{name}.npz"
      with np.load(path) as z:
        arrays = {key: z[key] for key in z.files}
      entry["precisions"][name] = {
        "table_content_sha256": _content_hash(arrays),
        "sealed_content_sha256": manifest["report"][cell]["precisions"][name]["content_sha256"],
      }
      etab = torch.from_numpy(arrays["etab_postmerge"]).to(dtype)[None]
      e_idx = torch.from_numpy(arrays["E_idx"])[None]
      energy = PottsMPNN.calc_potts_eners(etab, e_idx, torch.from_numpy(seqs))
      payload = {"seqs": seqs, "energy": energy.detach().cpu().numpy()}
      np.savez(out / f"{cell}_{name}.npz", **payload)
      entry["precisions"][name]["content_sha256"] = _content_hash(payload)
      entry["precisions"][name]["dtype"] = str(payload["energy"].dtype)
    report[cell] = entry
  return report


def _flatten(report: dict[str, dict]) -> dict[str, str]:
  return {f"{c}/{p}": v["content_sha256"] for c, e in report.items() for p, v in e["precisions"].items()}


def _child(args: argparse.Namespace) -> dict[str, str]:
  with tempfile.TemporaryDirectory() as tmp:
    result = Path(tmp) / "hashes.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--encoder-dir", str(args.encoder_dir),
               "--features-dir", str(args.features_dir), "--out", str(Path(tmp) / "child"),
               "--child-hashes-to", str(result)]
    subprocess.run(command, check=True, capture_output=True)
    return json.loads(result.read_text(encoding="utf-8"))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--encoder-dir", type=Path, required=True)
  parser.add_argument("--features-dir", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--child-hashes-to", type=Path, default=None)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  torch.set_num_threads(1)
  manifest = json.loads((args.encoder_dir / "encoder_manifest.json").read_text(encoding="utf-8"))

  directory = args.out / "protonpotts_v6_energy"
  report = _energies(args.encoder_dir, args.features_dir, directory, manifest)
  if args.child_hashes_to is not None:
    args.child_hashes_to.write_text(json.dumps(_flatten(report)), encoding="utf-8")
    return 0

  reproducible = _child(args) == _flatten(report)
  tables_match_sealed = all(
    p["table_content_sha256"] == p["sealed_content_sha256"]
    for e in report.values() for p in e["precisions"].values()
  )
  complete = True
  f64_ok = True
  precisions_differ = True
  floor: dict[str, float] = {}
  for cell, entry in report.items():
    a32 = np.load(directory / f"{cell}_f32.npz")
    a64 = np.load(directory / f"{cell}_f64.npz")
    complete &= a32["energy"].shape == (entry["n_sequences"],) and bool(np.isfinite(a32["energy"]).all())
    f64_ok &= a64["energy"].dtype == np.float64
    gap = float(np.abs(a64["energy"] - a32["energy"].astype(np.float64)).max())
    precisions_differ &= gap > 0.0
    floor[cell] = gap
  flags = {
    "tables_match_sealed": bool(tables_match_sealed),
    "complete": bool(complete),
    "f64_is_float64": bool(f64_ok),
    "precisions_differ": bool(precisions_differ),
    "reproducible": bool(reproducible),
    "n_cells": len(report),
  }
  (args.out / "energy_manifest.json").write_text(
    json.dumps(
      {"encoder_checkpoint_sha256": manifest["checkpoint_sha256"], "torch_version": torch.__version__,
       "seed": SEED, "n_random": N_RANDOM, "report": report, "f32_vs_f64_floor": floor, "flags": flags},
      indent=2, sort_keys=True),
    encoding="utf-8",
  )
  logger.info("flags %s", flags)
  results = os.environ.get("BTH_RESULTS_PATH")
  if results:
    Path(results).write_text(json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8")
  else:
    logger.warning("BTH_RESULTS_PATH unset: this run will record outcome 'unknown'")
  return 0


if __name__ == "__main__":
  sys.exit(main())
