#!/usr/bin/env python3
"""Do existing campaign done markers still verify under the installed xtrax's ``zarr_content_digest``? (aminx #2421)

Pre-registered in ``verify_done_markers.bth.toml``. READ-ONLY on the marker roots: it only reads stores. Its two controls run on a
private copy of one small store (a copy that must still match its marker, and a copy with one flipped byte that must NOT), so a
"verifies" result cannot be an instrument that always agrees.

Why it matters: the v2 -> v3 done-marker migration offers ``adopt-legacy``, which is only meaningful if a v2 marker's recorded digest
still equals what the current xtrax computes. If digest semantics had changed, every adopted marker would be rejected.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from xtrax.run import zarr_content_digest

logger = logging.getLogger("verify_done_markers")

MAX_STORE_BYTES = 400 * 1024 * 1024
COVERAGE_BAR = 0.95
SMALL_STORE_BYTES = 20 * 1024 * 1024


def _tree_bytes(path: Path) -> int:
  return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else path.stat().st_size


def _store_for(marker: Path) -> Path:
  return marker.with_name(marker.name[: -len(".done.json")])


def find_markers(roots: list[Path]) -> list[Path]:
  found: list[Path] = []
  for root in roots:
    found += sorted(m for m in root.rglob("*.done.json") if ".superseded." not in str(m) and not m.name.endswith(".v2.bak"))
  return found


def verify(markers: list[Path]) -> dict[str, Any]:
  tally = {"match": 0, "mismatch": 0, "store_missing": 0, "skipped_large": 0, "unreadable": 0}
  schemas: dict[str, int] = {}
  mismatches: list[dict[str, str]] = []
  smallest: tuple[int, Path, str] | None = None
  for marker in markers:
    try:
      payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
      tally["unreadable"] += 1
      continue
    schema = str(payload.get("schema_version"))
    schemas[schema] = schemas.get(schema, 0) + 1
    store = _store_for(marker)
    if not store.exists():
      tally["store_missing"] += 1
      continue
    nbytes = _tree_bytes(store)
    if nbytes > MAX_STORE_BYTES:
      tally["skipped_large"] += 1
      continue
    recorded = payload.get("content_digest_sha256")
    observed = zarr_content_digest(store)
    if observed == recorded:
      tally["match"] += 1
      if nbytes < SMALL_STORE_BYTES and (smallest is None or nbytes < smallest[0]):
        smallest = (nbytes, store, str(recorded))
    else:
      tally["mismatch"] += 1
      mismatches.append({"marker": str(marker), "recorded": str(recorded), "observed": observed})
  return {"tally": tally, "schemas": schemas, "mismatches": mismatches[:20], "control_source": smallest}


def run_controls(source: tuple[int, Path, str] | None) -> dict[str, Any]:
  """Controls on a private copy: unmodified must match the marker, one flipped byte must change the digest."""
  if source is None:
    return {"controls_ran": False, "unmodified_copy_matches": False, "negative_control_changed": False}
  _, store, recorded = source
  tmp = Path(tempfile.mkdtemp(prefix="verify_done_markers_"))
  try:
    copy = tmp / "copy"
    shutil.copytree(store, copy)
    unmodified_matches = zarr_content_digest(copy) == recorded
    chunk_files = [f for f in copy.rglob("*") if f.is_file() and f.name != "zarr.json" and f.stat().st_size > 0]
    if not chunk_files:
      return {"controls_ran": False, "unmodified_copy_matches": unmodified_matches, "negative_control_changed": False}
    target = max(chunk_files, key=lambda f: f.stat().st_size)
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0xFF
    target.write_bytes(bytes(data))
    # A flipped byte in a compressed chunk can make the decoder raise instead of yielding a different digest; either way the
    # store is rejected, which is what the negative control must show. Record which one happened.
    try:
      changed, mode = zarr_content_digest(copy) != recorded, "digest_differs"
    except Exception as exc:  # noqa: BLE001 - any decode failure on the corrupted copy means the store was rejected
      changed, mode = True, f"digest_raised:{type(exc).__name__}"
    return {
      "controls_ran": True,
      "unmodified_copy_matches": unmodified_matches,
      "negative_control_changed": changed,
      "negative_control_mode": mode,
    }
  finally:
    shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--root", action="append", required=True, help="Directory tree to search for *.done.json (repeatable)")
  parser.add_argument("--out", required=True, help="Result JSON path")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s")

  started = time.time()
  roots = [Path(r) for r in args.root]
  markers = find_markers(roots)
  logger.info("found %d done markers under %s", len(markers), [str(r) for r in roots])
  verified = verify(markers)
  controls = run_controls(verified["control_source"])
  tally = verified["tally"]
  n_checked = tally["match"] + tally["mismatch"]
  checked_fraction = n_checked / len(markers) if markers else 0.0
  result = {
    "xtrax_version": version("xtrax"),
    "markers_found": len(markers),
    "n_checked": n_checked,
    "n_match": tally["match"],
    "n_mismatch": tally["mismatch"],
    "n_store_missing": tally["store_missing"],
    "n_skipped_large": tally["skipped_large"],
    "n_unreadable": tally["unreadable"],
    "checked_fraction": round(checked_fraction, 4),
    "schemas": verified["schemas"],
    "mismatches": verified["mismatches"],
    **controls,
    "all_checked_match": tally["mismatch"] == 0 and n_checked > 0,
    "coverage_ok": checked_fraction >= COVERAGE_BAR,
    "controls_ok": bool(controls["controls_ran"] and controls["unmodified_copy_matches"] and controls["negative_control_changed"]),
    "seconds": round(time.time() - started, 1),
  }
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
  logger.info(
    "checked %d/%d markers: %d match, %d mismatch | controls_ok=%s",
    n_checked,
    len(markers),
    tally["match"],
    tally["mismatch"],
    result["controls_ok"],
  )


if __name__ == "__main__":
  main()
