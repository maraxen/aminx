"""Driver float16 gate v2, graded by scripts/parity/thermo_mpnn_driver_f16_v2.bth.toml.

Supersedes the fixed-bar gate in thermo_mpnn_driver_f16.py (which is left unchanged, so its
recorded run f12f3209 still measures what it measured). For each structure (1ubq, 1SMD, 1mbn) it
scores ThermoMPNNDriver.score in float32 and float16 and records, per compared array: the scale
S = max |float32 reference|, max-abs difference, RMS difference, and Pearson r. bathos grades the
tolerance K * eps16 * S from these fields; this script emits no pass or fail verdict.
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

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT / "src"))

from thermo_mpnn_driver_f16 import (  # noqa: E402
  DOUBLE_CHECKPOINT,
  V1_CHECKPOINT,
  _max_abs,
  _score,
)

from aminx.families.thermo_mpnn.driver import ThermoMPNNDriver  # noqa: E402
from aminx.families.thermo_mpnn.stats import pearson_r  # noqa: E402
from aminx.run.options import ThermoMPNNOptions  # noqa: E402

log = logging.getLogger("thermo_mpnn_driver_f16_v2")

FIXTURES: dict[str, str] = {
  "ubq": "1ubq.pdb",
  "ho1": "1SMD.pdb",
  "ho2": "1mbn.pdb",
}
EPS16 = 2.0**-10
K_LINEAR_BOUND = 68


def _rms(reference: np.ndarray, candidate: np.ndarray) -> float:
  """RMS of the elementwise difference in float64; NaN for empty or mismatched arrays."""
  ref = np.asarray(reference, dtype=np.float64)
  cand = np.asarray(candidate, dtype=np.float64)
  if ref.size == 0 or ref.shape != cand.shape:
    return float("nan")
  return float(np.sqrt(np.mean((ref - cand) ** 2)))


def _scale(reference: np.ndarray) -> float:
  """S = max |reference| in float64; NaN when empty."""
  ref = np.asarray(reference, dtype=np.float64)
  if ref.size == 0:
    return float("nan")
  return float(np.max(np.abs(ref)))


def _record(prefix: str, name: str, reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
  """Comparison fields for one quantity of one structure."""
  return {
    f"{prefix}_{name}_scale": _scale(reference),
    f"{prefix}_{name}_max_abs_diff": _max_abs(reference, candidate),
    f"{prefix}_{name}_rms_abs_diff": _rms(reference, candidate),
    f"{prefix}_{name}_pearson_r": pearson_r(candidate, reference),
  }


def _score_structure(
  driver: ThermoMPNNDriver,
  prefix: str,
  pdb_path: Path,
  v1_model: Any,  # noqa: ANN401
  d_model: Any,  # noqa: ANN401
) -> dict[str, Any]:
  """Score one structure in float32 and float16 and return its comparison fields."""
  prepared = driver.prepare_structure(pdb_path)
  inputs = {
    "coords": np.asarray(prepared["coords"], dtype=np.float32),
    "mask": np.asarray(prepared["mask"], dtype=np.float32),
    "residue_index": np.asarray(prepared["residue_index"], dtype=np.int32),
    "chain_index": np.asarray(prepared["chain_index"], dtype=np.int32),
    "sequence": np.asarray(prepared["sequence"], dtype=np.int32),
  }
  out: dict[str, Any] = {f"n_residues_{prefix}": int(inputs["sequence"].shape[0])}

  v1_f32 = _score(driver, v1_model, inputs, options=ThermoMPNNOptions(checkpoint_id=V1_CHECKPOINT, mode="single", dtype="float32"))
  v1_f16 = _score(driver, v1_model, inputs, options=ThermoMPNNOptions(checkpoint_id=V1_CHECKPOINT, mode="single", dtype="float16", allow_float16=True))
  out.update(_record(prefix, "v1_ddg", v1_f32["ddg"], v1_f16["ddg"]))

  d_f32 = _score(driver, d_model, inputs, options=ThermoMPNNOptions(checkpoint_id=DOUBLE_CHECKPOINT, mode="double", dtype="float32"))
  pairs = np.asarray(d_f32["pair_indices"], dtype=np.int32)
  d_f16 = _score(
    driver,
    d_model,
    inputs,
    options=ThermoMPNNOptions(checkpoint_id=DOUBLE_CHECKPOINT, mode="double", dtype="float16", allow_float16=True),
    pair_indices=pairs,
  )
  out[f"{prefix}_d_n_pairs"] = int(pairs.shape[0])
  out.update(_record(prefix, "d_ddg", d_f32["ddg"], d_f16["ddg"]))
  out.update(_record(prefix, "d_ddg_double", d_f32["ddg_double"], d_f16["ddg_double"]))
  return out


def run_v2_suite(fixture_dir: Path) -> dict[str, Any]:
  """Score the three structures and return the flat graded payload."""
  driver = ThermoMPNNDriver()
  v1_model = driver.load(SimpleNamespace(checkpoint_id=V1_CHECKPOINT, thermompnn=None, model_local_path=None))
  log.info("loaded %s", V1_CHECKPOINT)
  d_model = driver.load(SimpleNamespace(checkpoint_id=DOUBLE_CHECKPOINT, thermompnn=None, model_local_path=None))
  log.info("loaded %s", DOUBLE_CHECKPOINT)
  payload: dict[str, Any] = {"weight_source": "upstream_checkpoint"}
  for prefix, filename in FIXTURES.items():
    log.info("scoring %s (%s)", prefix, filename)
    payload.update(_score_structure(driver, prefix, fixture_dir / filename, v1_model, d_model))
  return payload


def _json_safe(value: object) -> object:
  """Replace non-finite floats with None: a bare NaN token is not valid JSON."""
  if isinstance(value, float) and not math.isfinite(value):
    return None
  return value


def main() -> None:
  parser = argparse.ArgumentParser(description="Driver float16 gate v2 for ThermoMPNN.")
  parser.add_argument("--out", type=Path, default=Path("outputs/parity/thermo_mpnn_driver_f16_v2.json"))
  parser.add_argument("--fixtures", type=Path, default=_REPO_ROOT / "tests" / "data", help="Directory with the fixtures.")
  parser.add_argument("--log-level", default="INFO")
  args = parser.parse_args()
  logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

  result = {key: _json_safe(value) for key, value in run_v2_suite(args.fixtures).items()}
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
