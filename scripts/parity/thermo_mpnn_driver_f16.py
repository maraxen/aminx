"""Driver-path float16 gate for ThermoMPNN, graded by scripts/parity/thermo_mpnn_driver_f16.bth.toml.

Scores 1ubq through ``ThermoMPNNDriver.score`` in float32 and float16 for the upstream
checkpoints ``thermo_mpnn_v1_default`` (ddg) and ``thermo_mpnn_d_ens1`` (ddg and ddg_double),
and writes max-abs difference and Pearson r per compared array. The script emits no pass or
fail field: bathos grades the recorded numbers against the sidecar's [outcomes].

The float16 run reuses the float32 pair selection (``pair_indices``) so both dtypes compare
the same residue pairs. Requires the upstream weights; run it on titanix via ``bth run``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT / "src"))

from aminx.families.thermo_mpnn.driver import ThermoMPNNDriver  # noqa: E402
from aminx.families.thermo_mpnn.stats import pearson_r  # noqa: E402
from aminx.run.options import ThermoMPNNOptions  # noqa: E402

log = logging.getLogger("thermo_mpnn_driver_f16")

DEFAULT_PDB = _REPO_ROOT / "tests" / "data" / "1ubq.pdb"
V1_CHECKPOINT = "thermo_mpnn_v1_default"
DOUBLE_CHECKPOINT = "thermo_mpnn_d_ens1"


def _max_abs(reference: np.ndarray, candidate: np.ndarray) -> float:
  """Max absolute difference in float64; NaN when empty so an empty comparison cannot pass."""
  ref = np.asarray(reference, dtype=np.float64)
  cand = np.asarray(candidate, dtype=np.float64)
  if ref.size == 0 or ref.shape != cand.shape:
    return float("nan")
  return float(np.max(np.abs(ref - cand)))


def _score(driver: ThermoMPNNDriver, model: Any, inputs: dict[str, Any], **kwargs: Any) -> dict:  # noqa: ANN401
  """Call ``driver.score`` through the public options path."""
  return driver.score(
    model,
    inputs["coords"],
    inputs["mask"],
    inputs["residue_index"],
    inputs["chain_index"],
    inputs["sequence"],
    **kwargs,
  )


def run_driver_f16_suite(pdb_path: Path) -> dict[str, Any]:
  """Score 1ubq in float32 and float16 through the driver and return the graded numbers."""
  driver = ThermoMPNNDriver()
  prepared = driver.prepare_structure(pdb_path)
  inputs = {
    "coords": np.asarray(prepared["coords"], dtype=np.float32),
    "mask": np.asarray(prepared["mask"], dtype=np.float32),
    "residue_index": np.asarray(prepared["residue_index"], dtype=np.int32),
    "chain_index": np.asarray(prepared["chain_index"], dtype=np.int32),
    "sequence": np.asarray(prepared["sequence"], dtype=np.int32),
  }
  n_residues = int(inputs["sequence"].shape[0])

  v1_spec = SimpleNamespace(checkpoint_id=V1_CHECKPOINT, thermompnn=None, model_local_path=None)
  v1_model = driver.load(v1_spec)
  log.info("loaded %s", V1_CHECKPOINT)
  v1_f32 = _score(driver, v1_model, inputs, options=ThermoMPNNOptions(checkpoint_id=V1_CHECKPOINT, mode="single", dtype="float32"))
  v1_f16 = _score(driver, v1_model, inputs, options=ThermoMPNNOptions(checkpoint_id=V1_CHECKPOINT, mode="single", dtype="float16"))

  d_spec = SimpleNamespace(checkpoint_id=DOUBLE_CHECKPOINT, thermompnn=None, model_local_path=None)
  d_model = driver.load(d_spec)
  log.info("loaded %s", DOUBLE_CHECKPOINT)
  d_f32 = _score(driver, d_model, inputs, options=ThermoMPNNOptions(checkpoint_id=DOUBLE_CHECKPOINT, mode="double", dtype="float32"))
  pairs = np.asarray(d_f32["pair_indices"], dtype=np.int32)
  d_f16 = _score(
    driver,
    d_model,
    inputs,
    options=ThermoMPNNOptions(checkpoint_id=DOUBLE_CHECKPOINT, mode="double", dtype="float16"),
    pair_indices=pairs,
  )

  return {
    "pdb_id": pdb_path.stem,
    "weight_source": "upstream_checkpoint",
    "n_residues": n_residues,
    "v1_ddg_max_abs_diff": _max_abs(v1_f32["ddg"], v1_f16["ddg"]),
    "v1_ddg_pearson_r": pearson_r(v1_f16["ddg"], v1_f32["ddg"]),
    "d_ddg_max_abs_diff": _max_abs(d_f32["ddg"], d_f16["ddg"]),
    "d_ddg_pearson_r": pearson_r(d_f16["ddg"], d_f32["ddg"]),
    "d_n_pairs": int(pairs.shape[0]),
    "d_ddg_double_max_abs_diff": _max_abs(d_f32["ddg_double"], d_f16["ddg_double"]),
    "d_ddg_double_pearson_r": pearson_r(d_f16["ddg_double"], d_f32["ddg_double"]),
  }


def _json_safe(value: object) -> object:
  """Replace non-finite floats with None: a bare NaN token is not valid JSON."""
  if isinstance(value, float) and not math.isfinite(value):
    return None
  return value


def main() -> None:
  parser = argparse.ArgumentParser(description="Driver-path float16 gate for ThermoMPNN.")
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("outputs/parity/thermo_mpnn_driver_f16.json"),
    help="Output JSON path.",
  )
  parser.add_argument("--pdb", type=Path, default=DEFAULT_PDB, help="Input structure.")
  parser.add_argument("--log-level", default="INFO", help="Logging level.")
  args = parser.parse_args()
  logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

  result = {key: _json_safe(value) for key, value in run_driver_f16_suite(args.pdb).items()}
  payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(payload, encoding="utf-8")
  log.info("wrote %s", args.out)

  bth_results = os.environ.get("BTH_RESULTS_PATH")
  if bth_results:
    bth_path = Path(bth_results)
    bth_path.parent.mkdir(parents=True, exist_ok=True)
    bth_path.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
  main()
