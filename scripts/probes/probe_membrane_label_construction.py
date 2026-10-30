"""Pre-registered probe for Clause 2: membrane_label_construction."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import jax.numpy as jnp

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT / "src"))

from aminx.io.membrane import (
  construct_membrane_labels,
  construct_membrane_physics_features,
  parse_residue_spec,
)


def run_probe() -> dict[str, object]:
  """Execute probe verifying membrane residue parsing, label assignment, and one-hot physics features."""
  # 1. Parsing tests
  parsed_set = parse_residue_spec("A1, A2, A5")
  parser_passed = bool(parsed_set == {"A1", "A2", "A5"})

  # 2. Formula and quirk test:
  # Residues A1 (buried), A2 (interface), A3 (neither), A4 (both)
  residues = ["A1", "A2", "A3", "A4"]
  labels = construct_membrane_labels(
    encoded_residues=residues,
    buried_residues="A1 A4",
    interface_residues="A2 A4",
  )
  expected_labels = jnp.array([2, 1, 0, 0], dtype=jnp.int32)
  formula_passed = bool(jnp.array_equal(labels, expected_labels))
  quirk_both_zero = bool(int(labels[3]) == 0)

  # 3. Global broadcast test:
  global_0 = construct_membrane_labels(encoded_residues=4, global_label=0)
  global_1 = construct_membrane_labels(encoded_residues=4, global_label=1)
  global_2 = construct_membrane_labels(encoded_residues=4, global_label=2)
  global_passed = bool(
    jnp.all(global_0 == 0) and jnp.all(global_1 == 1) and jnp.all(global_2 == 2)
  )

  # 4. One-hot physics features test:
  features = construct_membrane_physics_features(
    encoded_residues=residues,
    buried_residues="A1 A4",
    interface_residues="A2 A4",
  )
  shape_valid = bool(features.shape == (4, 3))
  one_hot_valid = bool(
    jnp.array_equal(features[0], jnp.array([0.0, 0.0, 1.0]))
    and jnp.array_equal(features[1], jnp.array([0.0, 1.0, 0.0]))
    and jnp.array_equal(features[2], jnp.array([1.0, 0.0, 0.0]))
    and jnp.array_equal(features[3], jnp.array([1.0, 0.0, 0.0]))
  )

  probe_passed = bool(
    parser_passed and formula_passed and quirk_both_zero and global_passed and shape_valid and one_hot_valid
  )

  return {
    "probe_name": "membrane_label_construction",
    "clause": "membrane_label_construction",
    "status": "PASS" if probe_passed else "FAIL",
    "parser_passed": parser_passed,
    "formula_passed": formula_passed,
    "quirk_both_zero": quirk_both_zero,
    "global_passed": global_passed,
    "shape_valid": shape_valid,
    "one_hot_valid": one_hot_valid,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Probe for membrane_label_construction clause.")
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("outputs/probes/probe_membrane_label_construction.json"),
    help="Output JSON path.",
  )
  args = parser.parse_args()

  result = run_probe()

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  bth_results = os.environ.get("BTH_RESULTS_PATH")
  if bth_results:
    bth_path = Path(bth_results)
    bth_path.parent.mkdir(parents=True, exist_ok=True)
    bth_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  print(f"Probe membrane_label_construction: {result['status']}")


if __name__ == "__main__":
  main()
