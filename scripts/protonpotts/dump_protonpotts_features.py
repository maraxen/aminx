"""Dump the upstream ProtonPottsMPNN (v6) FEATURE DICT for several cells (P4b).

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §37. Run in the
`aminx-oracles-protonpotts` environment. Imports upstream `mpnn`, torch and numpy only, never aminx::

    bth run --project-slug aminx -- uv run --no-sync python \\
        scripts/protonpotts/dump_protonpotts_features.py \\
        --pkad-pdb <1BVC.pdb> --multichain-pdb <6m0j.pdb> --ligand-pdb <swe1_ligand.pdb> --out <dir>

WHY THIS EXISTS. The P4 oracle (`dump_protonpotts_oracles.py`) saved `S`, `X`, `X_m` and the model
outputs for ONE cell, so `protonpotts_features` could grade 3 keys of the feature dict. The
featurizer is the part of P5 that has to be exact on integers and masks and measured on
coordinates (decision 11e), and it needs every feature key, on cells that exercise chain
boundaries and a ligand-bearing structure. No checkpoint and no forward pass is involved: this
dumps the featurizer's output only, which is why it is cheap and why threads are still pinned
(`prepare_potts_input` can run torch ops, and §32 measured thread-dependence in this stack).

Cells: PKAD 1BVC unlabelled and labelled (the labelled one uses the same cycling labeller as P4,
so S changes and nothing else should), a two-chain structure, and a ligand-bearing structure.

Labels are not chemistry (§26.5). No number from these dumps is a protonation-state result.
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
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
  os.environ[_var] = "1"  # before torch: the dump must not depend on the thread partition (§32)

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_protonpotts_oracles import (  # noqa: E402
  PREFIX_LEN,
  V6_PROTONATION,
  _covered_indices,
  _make_labeller,
)

logger = logging.getLogger("dump_protonpotts_features")

EXTENDED_VOCAB = "v6"
SEED = 0
# The six keys the Potts inference path emits (probe, 261008). Y / Y_t / Y_m belong to the generic
# MPNN transform, not to this model's pipeline; claim 5 of the sidecar checks that they are absent.
REQUIRED_KEYS = ("X", "X_m", "S", "R_idx", "chain_labels", "residue_mask")
LIGAND_KEYS = ("Y", "Y_t", "Y_m")
STRUCTURE_KEYS = ("X", "X_m", "R_idx", "chain_labels", "residue_mask")
CELL_NAMES = ("pkad_unlabelled", "pkad_labelled", "multichain", "ligand")


def _featurize(pdb: Path, *, labelled: bool) -> dict[str, np.ndarray]:
  import mpnn.pipelines.potts_mpnn as pipelines
  import mpnn.potts_inference as inference

  labeller = _make_labeller() if labelled else None
  pipelines.get_protonation_state_transforms = (
    (lambda **_: [labeller]) if labelled else (lambda **_: [])
  )
  inference._PIPELINE_CACHE.clear()  # noqa: SLF001 -- trap 1 of the P4 dumper (§23.4)
  torch.manual_seed(SEED)
  torch.set_num_threads(1)
  batch = inference.prepare_potts_input(str(pdb), extended_vocab=EXTENDED_VOCAB)
  features = batch["network_input"]["input_features"]
  return {
    key: value.detach().cpu().numpy()
    for key, value in sorted(features.items())
    if isinstance(value, torch.Tensor)
  }


def _content_hash(arrays: dict[str, np.ndarray]) -> str:
  digest = hashlib.sha256()
  for key in sorted(arrays):
    array = np.ascontiguousarray(arrays[key])
    digest.update(f"{key}|{array.dtype}|{array.shape}|".encode())
    digest.update(array.tobytes())
  return digest.hexdigest()


def _dump_all(args: argparse.Namespace, out: Path) -> dict[str, dict[str, np.ndarray]]:
  cells = {
    "pkad_unlabelled": _featurize(args.pkad_pdb, labelled=False),
    "pkad_labelled": _featurize(args.pkad_pdb, labelled=True),
    "multichain": _featurize(args.multichain_pdb, labelled=False),
    "ligand": _featurize(args.ligand_pdb, labelled=False),
  }
  out.mkdir(parents=True, exist_ok=True)
  for name, arrays in cells.items():
    np.savez(out / f"{name}.npz", **arrays)
  return cells


def _child_hashes(args: argparse.Namespace) -> dict[str, str]:
  """Featurize again in a FRESH process, so reproducibility is across processes, not within one."""
  with tempfile.TemporaryDirectory() as tmp:
    result = Path(tmp) / "hashes.json"
    command = [
      sys.executable,
      str(Path(__file__).resolve()),
      "--pkad-pdb",
      str(args.pkad_pdb),
      "--multichain-pdb",
      str(args.multichain_pdb),
      "--ligand-pdb",
      str(args.ligand_pdb),
      "--out",
      str(Path(tmp) / "child"),
      "--child-hashes-to",
      str(result),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return json.loads(result.read_text(encoding="utf-8"))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pkad-pdb", type=Path, required=True)
  parser.add_argument("--multichain-pdb", type=Path, required=True)
  parser.add_argument("--ligand-pdb", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--child-hashes-to", type=Path, default=None)
  parser.add_argument("--log-level", default="INFO")
  args = parser.parse_args()
  logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

  cells = _dump_all(args, args.out / "protonpotts_v6_features")
  hashes = {name: _content_hash(arrays) for name, arrays in cells.items()}
  if args.child_hashes_to is not None:  # child mode: report hashes and stop
    args.child_hashes_to.write_text(json.dumps(hashes), encoding="utf-8")
    return 0

  reproducible = _child_hashes(args) == hashes

  plain, lab = cells["pkad_unlabelled"], cells["pkad_labelled"]
  covered = _covered_indices(lab["S"])
  extension_in_labelled = int((lab["S"] >= PREFIX_LEN).sum())
  flags = {
    "keys_complete": all(all(k in cell for k in REQUIRED_KEYS) for cell in cells.values()),
    "ligand_triple_absent": all(not any(k in cell for k in LIGAND_KEYS) for cell in cells.values()),
    "single_chain_control": int(np.unique(plain["chain_labels"]).size) == 1,
    "multichain_fired": int(np.unique(cells["multichain"]["chain_labels"]).size) >= 2,
    "coverage_full": len(covered) == len(V6_PROTONATION),
    "labels_reach_s": int((plain["S"] != lab["S"]).sum()) == extension_in_labelled > 0,
    "unlabelled_clean": int((plain["S"] >= PREFIX_LEN).sum()) == 0,
    "structure_invariant": all(np.array_equal(plain[k], lab[k]) for k in STRUCTURE_KEYS),
    "reproducible": reproducible,
    "n_cells": len(cells),
  }
  shapes = {
    name: {k: list(v.shape) for k, v in arrays.items()} for name, arrays in cells.items()
  }
  files = {
    name: hashlib.sha256((args.out / "protonpotts_v6_features" / f"{name}.npz").read_bytes()).hexdigest()
    for name in cells
  }
  manifest = args.out / "features_manifest.json"
  manifest.write_text(
    json.dumps(
      {
        "extended_vocab": EXTENDED_VOCAB,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "labels_are_chemistry": False,
        "cells": {
          "pkad_unlabelled": str(args.pkad_pdb),
          "pkad_labelled": str(args.pkad_pdb),
          "multichain": str(args.multichain_pdb),
          "ligand": str(args.ligand_pdb),
        },
        "content_sha256": hashes,
        "file_sha256": files,
        "shapes": shapes,
        "flags": flags,
      },
      indent=2,
      sort_keys=True,
    ),
    encoding="utf-8",
  )
  logger.info("wrote %s", manifest)

  results_path = os.environ.get("BTH_RESULTS_PATH")
  if results_path:
    Path(results_path).write_text(
      json.dumps(
        {
          **flags,
          "n_covered": len(covered),
          "n_s_changed": int((plain["S"] != lab["S"]).sum()),
          "n_chains_multichain": int(np.unique(cells["multichain"]["chain_labels"]).size),
          "n_residues_ligand_cell": int(cells["ligand"]["S"].shape[-1]),
          "content_sha256_pkad_unlabelled": hashes["pkad_unlabelled"],
          "labels_are_chemistry": False,
        },
        indent=2,
        sort_keys=True,
      ),
      encoding="utf-8",
    )
  else:
    logger.warning("BTH_RESULTS_PATH unset: this run will record outcome 'unknown'")
  return 0


if __name__ == "__main__":
  sys.exit(main())
