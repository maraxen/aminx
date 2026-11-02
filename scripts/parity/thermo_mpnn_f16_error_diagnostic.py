"""Diagnostic for the float16 ddg error on ho1 (1SMD), graded by thermo_mpnn_f16_error_diagnostic.bth.toml.

For thermo_mpnn_d_ens1 it scores 1ubq, 1SMD and 1mbn through ThermoMPNNDriver.score in float32 and
float16 (same inputs as thermo_mpnn_driver_f16_v2.py), and records per-entry error e = f16 - f32
summaries for the single-mutant ddg and the double-mutant ddg_double. It emits numbers only;
bathos grades the classes in the sidecar. Intermediate activations are not recorded.
The payload keys are checked against the sidecar's result_schema before anything is written.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tomllib
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

from thermo_mpnn_driver_f16 import DOUBLE_CHECKPOINT, _score  # noqa: E402
from thermo_mpnn_driver_f16_v2 import FIXTURES  # noqa: E402

from aminx.families.thermo_mpnn.driver import ThermoMPNNDriver  # noqa: E402
from aminx.families.thermo_mpnn.stats import pearson_r  # noqa: E402
from aminx.run.options import ThermoMPNNOptions  # noqa: E402

log = logging.getLogger("thermo_mpnn_f16_error_diagnostic")

SIDECAR = _HERE / "thermo_mpnn_f16_error_diagnostic.bth.toml"
NONFINITE_BOUND = 60000.0


def _rankdata(x: np.ndarray) -> np.ndarray:
  """Average ranks (ties share their mean rank)."""
  order = np.argsort(x, kind="mergesort")
  ranks = np.empty(x.shape[0], dtype=np.float64)
  ranks[order] = np.arange(1, x.shape[0] + 1, dtype=np.float64)
  _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
  sums = np.bincount(inverse, weights=ranks)
  return (sums / counts)[inverse]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
  """Spearman correlation of flattened arrays; NaN if either is constant or empty."""
  x = np.asarray(a, dtype=np.float64).ravel()
  y = np.asarray(b, dtype=np.float64).ravel()
  if x.size < 2 or x.size != y.size:
    return float("nan")
  return pearson_r(_rankdata(x), _rankdata(y))


def _top_share(sq: np.ndarray, frac: float) -> float:
  """Share of the total squared error held by the top ``frac`` of entries (NaN if total is 0)."""
  flat = np.sort(sq.ravel())[::-1]
  total = float(flat.sum())
  if flat.size == 0 or total <= 0.0:
    return float("nan")
  k = max(1, math.ceil(frac * flat.size))
  return float(flat[:k].sum() / total)


def _nonfinite_count(x: np.ndarray) -> int:
  return int(np.size(x) - np.count_nonzero(np.isfinite(x)))


def _max_abs_or_nan(x: np.ndarray) -> float:
  """max |x| in float64; NaN if any entry is non-finite (so the sidecar's NULL/nonfinite rules see it)."""
  arr = np.asarray(x, dtype=np.float64)
  if arr.size == 0 or not np.all(np.isfinite(arr)):
    return float("nan")
  return float(np.max(np.abs(arr)))


def _summarise_ddg(f32: np.ndarray, f16: np.ndarray, prefix: str) -> tuple[dict[str, Any], np.ndarray]:
  """Single-mutant summary; returns the fields and the error array."""
  f32 = np.asarray(f32, dtype=np.float64)
  f16 = np.asarray(f16, dtype=np.float64)
  err = f16 - f32
  abs_err = np.abs(err)
  sq = err**2
  flat_arg = int(np.nanargmax(abs_err)) if abs_err.size else 0
    
  i, a = divmod(flat_arg, f32.shape[1]) if abs_err.size else (0, 0)
  fields = {
    f"{prefix}_ddg_scale_f32": float(np.max(np.abs(f32))) if f32.size else float("nan"),
    f"{prefix}_ddg_max_abs_diff": float(np.max(abs_err)) if abs_err.size else float("nan"),
    f"{prefix}_ddg_rms_abs_diff": float(np.sqrt(np.mean(sq))) if sq.size else float("nan"),
    f"{prefix}_ddg_top1_share": _top_share(sq, 0.01),
    f"{prefix}_ddg_top5_share": _top_share(sq, 0.05),
    f"{prefix}_ddg_argmax_i": int(i),
    f"{prefix}_ddg_argmax_a": int(a),
    f"{prefix}_ddg_argmax_f32": float(f32.reshape(-1)[flat_arg]) if abs_err.size else float("nan"),
    f"{prefix}_ddg_nonfinite_count_f16": _nonfinite_count(f16),
    f"{prefix}_ddg_max_abs_f16": _max_abs_or_nan(f16),
    f"{prefix}_ddg_spearman_abserr_vs_abs_f32": _spearman(abs_err, np.abs(f32)),
    f"{prefix}_ddg_pearson_r": pearson_r(f16, f32),
  }
  return fields, err


def _summarise_dd(
  f32: np.ndarray,
  f16: np.ndarray,
  pairs: np.ndarray,
  prefix: str,
) -> tuple[dict[str, Any], np.ndarray]:
  """Double-mutant summary over (P, 21, 21); returns the fields and the error array."""
  f32 = np.asarray(f32, dtype=np.float64)
  f16 = np.asarray(f16, dtype=np.float64)
  err = f16 - f32
  abs_err = np.abs(err)
  sq = err**2
  if abs_err.size == 0:
    flat_arg, p, a, b = 0, 0, 0, 0
    pi = pj = -1
    arg_f32 = float("nan")
    max_abs = rms = float("nan")
    spear_sep = float("nan")
  else:
    flat_arg = int(np.nanargmax(abs_err))
    p, rem = divmod(flat_arg, 21 * 21)
    a, b = divmod(rem, 21)
    pi, pj = int(pairs[p, 0]), int(pairs[p, 1])
    arg_f32 = float(f32.reshape(-1)[flat_arg])
    max_abs = float(np.max(abs_err))
    rms = float(np.sqrt(np.mean(sq)))
    sep = np.abs(pairs[:, 0].astype(np.float64) - pairs[:, 1].astype(np.float64))
    sep_full = np.broadcast_to(sep[:, None, None], err.shape)
    spear_sep = _spearman(abs_err, sep_full)
  fields = {
    f"{prefix}_dd_max_abs_diff": max_abs,
    f"{prefix}_dd_rms_abs_diff": rms,
    f"{prefix}_dd_top1_share": _top_share(sq, 0.01),
    f"{prefix}_dd_top5_share": _top_share(sq, 0.05),
    f"{prefix}_dd_argmax_i": pi,
    f"{prefix}_dd_argmax_j": pj,
    f"{prefix}_dd_argmax_a": int(a),
    f"{prefix}_dd_argmax_b": int(b),
    f"{prefix}_dd_argmax_f32": arg_f32,
    f"{prefix}_dd_nonfinite_count_f16": _nonfinite_count(f16),
    f"{prefix}_dd_max_abs_f16": _max_abs_or_nan(f16),
    f"{prefix}_dd_spearman_abserr_vs_abs_f32": _spearman(abs_err, np.abs(f32)),
    f"{prefix}_dd_spearman_abserr_vs_sep": spear_sep,
    f"{prefix}_dd_pearson_r": pearson_r(f16, f32),
  }
  return fields, err


def _run_structure(
  driver: ThermoMPNNDriver,
  prefix: str,
  pdb_path: Path,
  d_model: Any,  # noqa: ANN401
  arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
  """Score one structure and return its summary fields (arrays are stored in ``arrays``)."""
  prepared = driver.prepare_structure(pdb_path)
  inputs = {
    "coords": np.asarray(prepared["coords"], dtype=np.float32),
    "mask": np.asarray(prepared["mask"], dtype=np.float32),
    "residue_index": np.asarray(prepared["residue_index"], dtype=np.int32),
    "chain_index": np.asarray(prepared["chain_index"], dtype=np.int32),
    "sequence": np.asarray(prepared["sequence"], dtype=np.int32),
  }
  n_res = int(inputs["sequence"].shape[0])
  d_f32 = _score(driver, d_model, inputs, options=ThermoMPNNOptions(checkpoint_id=DOUBLE_CHECKPOINT, mode="double", dtype="float32"))
  pairs = np.asarray(d_f32["pair_indices"], dtype=np.int32)
  d_f16 = _score(
    driver,
    d_model,
    inputs,
    options=ThermoMPNNOptions(checkpoint_id=DOUBLE_CHECKPOINT, mode="double", dtype="float16"),
    pair_indices=pairs,
  )
  out: dict[str, Any] = {f"{prefix}_n_residues": n_res, f"{prefix}_n_pairs": int(pairs.shape[0])}
  ddg_fields, ddg_err = _summarise_ddg(d_f32["ddg"], d_f16["ddg"], prefix)
  dd_fields, dd_err = _summarise_dd(d_f32["ddg_double"], d_f16["ddg_double"], pairs, prefix)
  out.update(ddg_fields)
  out.update(dd_fields)
  arrays[f"{prefix}_ddg_err"] = ddg_err.astype(np.float32)
  arrays[f"{prefix}_dd_err"] = dd_err.astype(np.float32)
  arrays[f"{prefix}_pairs"] = pairs
  return out


def _sidecar_schema_keys(path: Path) -> set[str]:
  return set(tomllib.loads(path.read_text(encoding="utf-8"))["result_schema"])


def run_diagnostic(fixture_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
  driver = ThermoMPNNDriver()
  d_model = driver.load(SimpleNamespace(checkpoint_id=DOUBLE_CHECKPOINT, thermompnn=None, model_local_path=None))
  log.info("loaded %s", DOUBLE_CHECKPOINT)
  payload: dict[str, Any] = {}
  arrays: dict[str, np.ndarray] = {}
  for prefix, filename in FIXTURES.items():
    log.info("scoring %s (%s)", prefix, filename)
    payload.update(_run_structure(driver, prefix, fixture_dir / filename, d_model, arrays))
  return payload, arrays


def _json_safe(value: object) -> object:
  if isinstance(value, float) and not math.isfinite(value):
    return None
  if isinstance(value, np.generic):
    return value.item()
  return value


def main() -> None:
  parser = argparse.ArgumentParser(description="float16 ddg error diagnostic for ho1 (1SMD) d_ens1.")
  parser.add_argument("--out", type=Path, default=Path("outputs/parity/thermo_mpnn_f16_error_diagnostic.json"))
  parser.add_argument("--npz", type=Path, default=Path("outputs/parity/thermo_mpnn_f16_error_diagnostic_pairs.npz"))
  parser.add_argument("--fixtures", type=Path, default=_REPO_ROOT / "tests" / "data")
  parser.add_argument("--sidecar", type=Path, default=SIDECAR)
  parser.add_argument("--log-level", default="INFO")
  args = parser.parse_args()
  logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

  payload, arrays = run_diagnostic(args.fixtures)
  expected = _sidecar_schema_keys(args.sidecar)
  got = set(payload)
  if got != expected:
    log.error("payload keys differ from the sidecar result_schema: missing=%s extra=%s",
              sorted(expected - got), sorted(got - expected))
    raise SystemExit(1)

  result = {key: _json_safe(value) for key, value in payload.items()}
  text = json.dumps(result, indent=2, sort_keys=True) + "\n"
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(text, encoding="utf-8")
  np.savez(args.npz, **arrays)
  log.info("wrote %s and %s", args.out, args.npz)

  bth_results = os.environ.get("BTH_RESULTS_PATH")
  if bth_results:
    bth_path = Path(bth_results)
    bth_path.parent.mkdir(parents=True, exist_ok=True)
    bth_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
  main()
