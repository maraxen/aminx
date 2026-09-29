"""Convert LASErMPNN ideal side-chain geometry tensors to ``ideal_geometry.npz``.

Reads the pinned upstream ``files/*.pt`` tensors that ``utils/constants.py``
loads (lines 147-152) and writes the arrays the aminx LASEr host featurizer
reads (``src/aminx/families/laser_mpnn/featurize.py``). ``alignment`` is the
raw ``rotamer_alignment.pt`` tensor; the ``-1`` remap upstream applies at
import (``constants.py:151-152``) is done by the featurizer.

Run in the torch oracle env (``~/projects/aminx-oracles`` on titanix):

  uv run --no-sync python scripts/recapture/laser_ideal_geometry_to_npz.py \
    --laser-root ~/repos/LASErMPNN --out src/aminx/families/laser_mpnn/ideal_geometry.npz

task_id 260929_potts-laser-xtrax-compose
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("laser_ideal_geometry_to_npz")

SOURCES = {
  "ideal_aa": "new_ideal_coords.pt",
  "bond_lengths": "new_ideal_bond_lengths.pt",
  "bond_angles": "new_ideal_bond_angles.pt",
  "alignment": "rotamer_alignment.pt",
  "ideal_prot": "ideal_aa_coords_prot.pt",
}


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
  """Write the npz and log per-source sha256 values."""
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--laser-root", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  import torch  # noqa: PLC0415 - oracle env only

  arrays: dict[str, np.ndarray] = {}
  for key, name in SOURCES.items():
    source = args.laser_root.expanduser() / "files" / name
    tensor = torch.load(source, map_location="cpu", weights_only=True)
    arrays[key] = tensor.detach().cpu().numpy()
    logger.info("%s <- %s sha256=%s %s %s", key, name, _sha256(source), arrays[key].shape, arrays[key].dtype)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(args.out, **arrays)
  logger.info("wrote %s sha256=%s", args.out, _sha256(args.out))


if __name__ == "__main__":
  main()
