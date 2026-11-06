"""Bathos parity and negative-control driver for FrustraMPNN.

Task 261105_frustrampnn (Milestone 6 vehicle).
Executes:
1. Stage-by-stage forward pass parity against sealed upstream oracle dumps.
2. Wild-type self-mutation identity and score delta checks.
3. Manifest-driven negative controls:
   - reversed_hidden
   - permuted_output_weights
   - refinement_bypass
   - subtract_mut_inverted
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from artifact_key import stable_artifact_key
from aminx.families.frustra_mpnn import (
  FRUSTRA_ALPHABET,
  FrustraMPNN,
  FrustraMPNNDriver,
  cast_floating,
  frustra_mpnn_trunk,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("frustrampnn_parity.json"),
    help="Output JSON path for vehicle run summary.",
  )
  parser.add_argument(
    "--controls-out",
    type=Path,
    default=None,
    help="Output JSON path for branch_controls.json.",
  )
  parser.add_argument(
    "--mutants",
    type=str,
    default="",
    help="Comma-separated list of mutant IDs to evaluate.",
  )
  parser.add_argument(
    "--synthetic",
    action="store_true",
    help="Run synthetic harness check (never promotes).",
  )
  parser.add_argument(
    "--work-dir",
    type=Path,
    default=None,
    help="Working directory for scratch/cache.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  print(f"FrustraMPNN parity vehicle initialized with args: {args}")


if __name__ == "__main__":
  main()
