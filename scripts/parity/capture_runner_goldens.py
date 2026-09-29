"""Capture and compare in-memory MPNN runner goldens (T0.5a).

``--mode capture`` writes ``tests/golden/runner_v0/<case>.npz`` and
``manifest.json``. ``--mode compare`` reruns every case against those files.
``--negative-control`` (default on for compare) reruns one sample case at
``random_seed=1`` and records whether its arrays differ from the golden.

A graded mismatch writes the results JSON and exits 0. Unexpected exceptions
propagate. Bathos grades ``--mode compare`` via ``$BTH_RESULTS_PATH``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import jax
import jaxlib
from runner_golden_cases import (
  CASES,
  CHECKPOINT_IDS,
  FIXTURES,
  GOLDEN_DIR,
  NEGATIVE_CONTROL_SEED,
  REPO_ROOT,
  GoldenCase,
  arrays_byte_equal,
  compare_case,
  current_device_kind,
  load_case_npz,
  negative_control_case,
  run_case,
  save_case_npz,
  split_result,
)

log = logging.getLogger("capture_runner_goldens")

RESULTS_KEYS = ("n_cases", "n_mismatch", "mismatches", "negative_control_differs")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("capture", "compare"), required=True)
  parser.add_argument(
    "--negative-control",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="On compare, rerun one sample case at random_seed=1 and record whether it differs.",
  )
  parser.add_argument(
    "--out",
    type=Path,
    default=None,
    help="Results JSON path when $BTH_RESULTS_PATH is unset (compare only).",
  )
  parser.add_argument(
    "--golden-dir",
    type=Path,
    default=GOLDEN_DIR,
    help="Directory for manifest.json and per-case npz files.",
  )
  return parser.parse_args()


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  digest.update(path.read_bytes())
  return digest.hexdigest()


def _git_sha(repo: Path) -> str:
  completed = subprocess.run(
    ["git", "rev-parse", "HEAD"],  # noqa: S607
    cwd=repo,
    check=True,
    capture_output=True,
    text=True,
  )
  return completed.stdout.strip()


def _fixture_hashes() -> dict[str, str]:
  hashes: dict[str, str] = {}
  for fixture in FIXTURES:
    path = REPO_ROOT / fixture
    if not path.is_file():
      msg = f"fixture not found: {path}"
      raise FileNotFoundError(msg)
    hashes[fixture] = _sha256_file(path)
  return hashes


def _manifest(golden_dir: Path, fixture_hashes: dict[str, str]) -> dict[str, Any]:
  lock_path = REPO_ROOT / "uv.lock"
  return {
    "schema_version": 1,
    "cases": [case.to_manifest_entry() for case in CASES],
    "checkpoint_ids": list(CHECKPOINT_IDS),
    "fixtures": fixture_hashes,
    "capture_git_sha": _git_sha(REPO_ROOT),
    "jax_version": jax.__version__,
    "jaxlib_version": jaxlib.__version__,
    "uv_lock_sha256": _sha256_file(lock_path),
    "device_kind": current_device_kind(),
    "golden_dir": golden_dir.as_posix(),
    "negative_control_case_id": negative_control_case().case_id,
    "notes": (
      "In-memory rows only. Zarr rows are T0.5b. "
      "Every operation x checkpoint x fixture combination in cases is valid. "
      "1mbn supplies the HEM ligand for ligandmpnn; 1ubq is apo."
    ),
  }


def _capture(golden_dir: Path) -> None:
  fixture_hashes = _fixture_hashes()
  golden_dir.mkdir(parents=True, exist_ok=True)
  for case in CASES:
    log.info("capture %s", case.case_id)
    arrays, payload = split_result(run_case(case))
    save_case_npz(golden_dir / f"{case.case_id}.npz", arrays, payload)
    log.info("wrote %s (%d arrays)", case.case_id, len(arrays))
  manifest = _manifest(golden_dir, fixture_hashes)
  manifest_path = golden_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  log.info("wrote %s (%d cases)", manifest_path, len(CASES))


def _results_path(out: Path | None) -> Path:
  env = os.environ.get("BTH_RESULTS_PATH")
  if env:
    return Path(env)
  if out is not None:
    return out
  msg = "compare requires $BTH_RESULTS_PATH or --out"
  raise SystemExit(msg)


def _mismatch(case_id: str, detail: str) -> dict[str, str]:
  return {"case_id": case_id, "detail": detail}


def _compare_one(case: GoldenCase, golden_dir: Path) -> list[dict[str, str]]:
  npz_path = golden_dir / f"{case.case_id}.npz"
  if not npz_path.is_file():
    return [_mismatch(case.case_id, f"missing golden {npz_path}")]
  captured_arrays, captured_payload = load_case_npz(npz_path)
  messages = compare_case(captured_arrays, captured_payload, run_case(case))
  return [_mismatch(case.case_id, message) for message in messages]


def _negative_control_differs(golden_dir: Path) -> bool:
  """True when the seed=1 rerun is not byte-identical to the captured arrays."""
  case = negative_control_case()
  npz_path = golden_dir / f"{case.case_id}.npz"
  if not npz_path.is_file():
    log.error("negative control golden missing: %s", npz_path)
    return False
  captured_arrays, _payload = load_case_npz(npz_path)
  fresh_arrays, _fresh_payload = split_result(run_case(case, random_seed=NEGATIVE_CONTROL_SEED))
  if set(captured_arrays) != set(fresh_arrays):
    log.info("negative control %s differs (array keys)", case.case_id)
    return True
  for key, captured in captured_arrays.items():
    fresh = fresh_arrays[key]
    if not arrays_byte_equal(captured, fresh):
      log.info("negative control %s differs on %s", case.case_id, key)
      return True
  log.error(
    "negative control %s matched the golden at random_seed=%s",
    case.case_id,
    NEGATIVE_CONTROL_SEED,
  )
  return False


def _compare(golden_dir: Path, *, negative_control: bool) -> dict[str, Any]:
  mismatches: list[dict[str, str]] = []
  for case in CASES:
    log.info("compare %s", case.case_id)
    mismatches.extend(_compare_one(case, golden_dir))
  differed = _negative_control_differs(golden_dir) if negative_control else False
  mismatched_cases = {item["case_id"] for item in mismatches}
  results: dict[str, Any] = {
    "n_cases": len(CASES),
    "n_mismatch": len(mismatched_cases),
    "mismatches": mismatches,
    "negative_control_differs": differed,
  }
  unexpected = set(results) - set(RESULTS_KEYS)
  if unexpected:
    msg = f"results JSON has keys outside the sidecar schema: {sorted(unexpected)}"
    raise RuntimeError(msg)
  return results


def _write_results(results: dict[str, Any], path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
  log.info(
    "results n_cases=%s n_mismatch=%s negative_control_differs=%s -> %s",
    results["n_cases"],
    results["n_mismatch"],
    results["negative_control_differs"],
    path,
  )


def main() -> int:
  """Capture goldens or compare them. Graded compare failures exit 0."""
  logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
  args = _parse_args()
  golden_dir = Path(args.golden_dir)
  if args.mode == "capture":
    _capture(golden_dir)
    return 0
  results = _compare(golden_dir, negative_control=bool(args.negative_control))
  _write_results(results, _results_path(args.out))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
