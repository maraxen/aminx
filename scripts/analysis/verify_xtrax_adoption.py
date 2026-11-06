#!/usr/bin/env python3
"""Bathos verification script for aminx adoption of xtrax pack and eda.

Confirms:
1. Standard record_name ("unit.json") unit write, pack, readback, and validation.
2. Custom record_name unit write, pack, readback, and validation.
3. Negative controls: bit corruption, duplicate record collision (R-U1), corrupted export artifact.
4. Pack integrity: a unit packed beside a legacy .done.json keeps a verifiable record, and a
   member or record edited after commit is refused at pack.
5. EDA plan panel SVG rendering, Vega-Lite v5 spec generation, and manifest SHA-256 verification.

Run from the repo with `uv run --extra eda` (the plotting stack is the optional `eda` extra).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import numpy as np
import xtrax
import zarr
from xtrax.pack import (
  UnitPackRefused,
  is_packed,
  record_problem,
  unit_content_digest,
)
from xtrax.run.unit_index_adapters import UnitJsonAdapter
from xtrax.tiling import AxisDecision, AxisSpec, BatchPlan, Bucket, ChunkedMap

from aminx.host.pack_shim import (
  pack_aminx_unit,
  resolve_unit_adapter,
  validate_aminx_unit,
  write_aminx_unit,
)
from aminx.host.plan_eda import (
  export_and_verify_plan_artifacts,
  generate_aminx_plan_spec,
  render_aminx_plan_panels,
  verify_export_manifest_sha256,
)

T = TypeVar("T")
# Identity for the synthetic units below; write_aminx_unit refuses to invent one.
_DIGEST = "a" * 64

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("verify_xtrax_adoption")


def _make_sample_store(store_dir: Path) -> Path:
  store_dir.mkdir(parents=True, exist_ok=True)
  root = zarr.open_group(str(store_dir), mode="a")
  arr = root.create_array(name="data", shape=(8,), dtype="int32")
  arr[...] = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
  return store_dir


def _make_sample_plan() -> BatchPlan:
  spec_l = AxisSpec(name="L", cardinality=120, default_batch_size=10, bucket_boundaries=(64, 128, 256))
  dec_chunk = AxisDecision(
    spec=spec_l,
    batch_size=10,
    reasoning="chunk residues",
    strategy=ChunkedMap(batch_size=10),
  )
  spec_b = AxisSpec(name="B", cardinality=64, default_batch_size=64, bucket_boundaries=(32, 64, 128))
  dec_bucket = AxisDecision(
    spec=spec_b,
    batch_size=64,
    reasoning="bucket batch size",
    strategy=Bucket(boundaries=(32, 64, 128)),
  )
  return BatchPlan(decisions=[dec_chunk, dec_bucket])


def verify_pack_and_custom_records() -> tuple[bool, bool]:
  """Verifies standard and custom record_name pack and roundtrip."""
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)

    # 1. Standard record_name ("unit.json")
    unit_std = tmp_path / "unit_std"
    _make_sample_store(unit_std)
    rec_std = write_aminx_unit(unit_std, input_digest=_DIGEST, run_id="run-std-1")
    assert (unit_std / "unit.json").exists()
    digest_std = unit_content_digest(unit_std)
    assert rec_std["content_digest"] == digest_std
    pack_aminx_unit(unit_std)
    assert is_packed(unit_std)
    val_std = validate_aminx_unit(unit_std)
    assert val_std["content_digest"] == digest_std
    adapter_std = resolve_unit_adapter(unit_std)
    assert isinstance(adapter_std, UnitJsonAdapter)

    # 2. Custom record_name ("custom_rec.json")
    unit_cust = tmp_path / "unit_cust"
    _make_sample_store(unit_cust)
    rec_cust = write_aminx_unit(
      unit_cust, input_digest=_DIGEST, run_id="run-cust-1", record_name="custom_rec.json"
    )
    assert (unit_cust / "custom_rec.json").exists()
    assert not (unit_cust / "unit.json").exists()
    digest_cust = unit_content_digest(unit_cust, record_name="custom_rec.json")
    assert rec_cust["content_digest"] == digest_cust
    pack_aminx_unit(unit_cust, record_name="custom_rec.json")
    assert is_packed(unit_cust, record_name="custom_rec.json")
    assert not is_packed(unit_cust, record_name="unit.json")
    val_cust = validate_aminx_unit(unit_cust, record_name="custom_rec.json")
    assert val_cust["content_digest"] == digest_cust
    # No xtrax adapter reads a custom record name; resolution must refuse, not mis-report.
    try:
      resolve_unit_adapter(unit_cust, record_name="custom_rec.json")
    except ValueError:
      pass
    else:
      msg = "resolve_unit_adapter accepted a custom record name"
      raise AssertionError(msg)

    return True, True


def verify_negative_controls() -> bool:
  """Verifies refusal and failure controls: duplicate record, bit corruption, tampered export."""
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)

    # Control A: duplicate record write refusal (R-U1)
    unit_a = tmp_path / "unit_a"
    _make_sample_store(unit_a)
    write_aminx_unit(unit_a, input_digest=_DIGEST, run_id="run-a-1", record_name="dup.json")
    control_dup_raised = False
    try:
      write_aminx_unit(unit_a, input_digest=_DIGEST, run_id="run-a-2", record_name="dup.json")
    except UnitPackRefused:
      control_dup_raised = True
    assert control_dup_raised, "Expected UnitPackRefused on duplicate record write"

    # Control B: bit corruption in packed archive causes validate_aminx_unit to raise
    unit_b = tmp_path / "unit_b"
    _make_sample_store(unit_b)
    write_aminx_unit(unit_b, input_digest=_DIGEST, run_id="run-b-1")
    pack_aminx_unit(unit_b)
    tar_path = unit_b / "members.tar.zst"
    raw_tar = bytearray(tar_path.read_bytes())
    raw_tar[10] ^= 0xFF
    tar_path.write_bytes(raw_tar)
    control_corrupt_raised = False
    try:
      validate_aminx_unit(unit_b)
    except (ValueError, OSError, UnitPackRefused):
      control_corrupt_raised = True
    assert control_corrupt_raised, "Expected exception on corrupted members.tar.zst"

    # Control C: tampered export artifact causes verify_export_manifest_sha256 to raise
    export_c = tmp_path / "export_c"
    plan = _make_sample_plan()
    export_and_verify_plan_artifacts(plan, export_c, prefix="plan_c")
    tamper_file = export_c / "plan_c_axes.csv"
    tamper_file.write_text("corrupted,axes\n0,0\n", encoding="utf-8")
    control_tamper_raised = False
    try:
      verify_export_manifest_sha256(export_c)
    except ValueError:
      control_tamper_raised = True
    assert control_tamper_raised, "Expected ValueError on tampered export artifact"

    return True


def verify_pack_integrity_controls() -> tuple[bool, bool]:
  """Controls for the review-round-1 defects.

  Returns (legacy_quarantine_ok, tamper_refused_ok):
  - a unit packed beside a legacy .done.json keeps a record that verifies and indexes as reuse;
  - a member edited after commit, or a record edited after commit, is refused at pack.
  """
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)

    unit_q = tmp_path / "row_q.h5"
    _make_sample_store(unit_q)
    marker = unit_q.with_name(f"{unit_q.name}.done.json")
    marker.write_text(json.dumps({"schema_version": "campaign_done_marker_v3"}), encoding="utf-8")
    write_aminx_unit(unit_q, input_digest=_DIGEST)
    pack_aminx_unit(unit_q)
    record = json.loads((unit_q / "unit.json").read_text(encoding="utf-8"))
    verdict = UnitJsonAdapter().classify(tmp_path, unit_q.name, _DIGEST, verify=True)
    legacy_quarantine_ok = (
      record_problem(record) is None
      and verdict.__class__.__name__ == "IndexReuse"
      and not marker.exists()
      and unit_q.with_name(f"{unit_q.name}.done.json.packed").exists()
    )

    unit_t = tmp_path / "unit_t"
    _make_sample_store(unit_t)
    write_aminx_unit(unit_t, input_digest=_DIGEST)
    chunk = next(
      p for p in unit_t.rglob("*") if p.is_file() and p.name not in ("unit.json", "zarr.json")
    )
    chunk.write_bytes(b"tampered")
    member_refused = False
    try:
      pack_aminx_unit(unit_t)
    except UnitPackRefused:
      member_refused = True

    unit_r = tmp_path / "unit_r"
    _make_sample_store(unit_r)
    write_aminx_unit(unit_r, input_digest=_DIGEST)
    rec_path = unit_r / "unit.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    rec["meta"]["late_edit"] = True
    rec_path.write_text(json.dumps(rec), encoding="utf-8")
    record_refused = False
    try:
      pack_aminx_unit(unit_r)
    except UnitPackRefused:
      record_refused = True

    return legacy_quarantine_ok, member_refused and record_refused and not is_packed(unit_t)


def verify_eda_rendering_and_manifest() -> tuple[bool, bool]:
  """Verifies xtrax.eda plan panel SVG rendering, spec generation, and manifest sha256 verification."""
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)
    plan = _make_sample_plan()

    # 1. Panel rendering to SVG
    svg_bytes = render_aminx_plan_panels(plan, fmt="svg")
    assert isinstance(svg_bytes, bytes)
    assert len(svg_bytes) > 0
    assert b"<svg" in svg_bytes.lower()
    render_ok = True

    # 2. Spec generation
    spec = generate_aminx_plan_spec(plan)
    assert isinstance(spec, dict)
    assert "$schema" in spec

    # 3. Export and manifest verification
    manifest = export_and_verify_plan_artifacts(plan, tmp_path, prefix="plan_eda")
    assert manifest["schema_version"] == "2.0"
    assert "plan_eda.canonical.json" in manifest["artifacts"]
    assert "plan_eda_axes.csv" in manifest["artifacts"]
    assert "plan_eda.vl.json" in manifest["artifacts"]

    # Explicit direct manifest verify
    verified_manifest = verify_export_manifest_sha256(tmp_path)
    assert verified_manifest == manifest
    manifest_ok = True

    return render_ok, manifest_ok


def _guarded(check: Callable[[], T], failed: T) -> T:
  try:
    return check()
  except (AssertionError, OSError, ValueError, TypeError) as exc:
    log.exception("%s failed: %s", check.__name__, exc)
    return failed


def main() -> int:
  parser = argparse.ArgumentParser(description="Verify aminx xtrax adoption under Bathos")
  parser.add_argument(
    "--out",
    type=Path,
    default=Path(os.environ.get("BTH_RESULTS_PATH", "outputs/analysis/verify_xtrax_adoption.json")),
    help="Output JSON path for Bathos results",
  )
  args = parser.parse_args()

  log.info("Running aminx xtrax pack & eda adoption verification...")
  # Each check reports a boolean; a failed assertion is a False result, not a crash, so the
  # sidecar's outcomes always evaluate.
  pack_ok, custom_record_ok = _guarded(verify_pack_and_custom_records, (False, False))
  negative_ok = _guarded(verify_negative_controls, False)
  legacy_quarantine_ok, tamper_refused_ok = _guarded(verify_pack_integrity_controls, (False, False))
  eda_render_svg_ok, eda_manifest_sha256_ok = _guarded(
    verify_eda_rendering_and_manifest, (False, False)
  )
  eda_ok = eda_render_svg_ok and eda_manifest_sha256_ok
  controls_ok = negative_ok and legacy_quarantine_ok and tamper_refused_ok

  results = {
    "pack_ok": pack_ok,
    "eda_ok": eda_ok,
    "controls_ok": controls_ok,
    "custom_record_name_ok": custom_record_ok,
    "eda_render_svg_ok": eda_render_svg_ok,
    "eda_manifest_sha256_ok": eda_manifest_sha256_ok,
    "negative_controls_ok": negative_ok,
    "legacy_quarantine_ok": legacy_quarantine_ok,
    "tamper_refused_ok": tamper_refused_ok,
    "xtrax_version": getattr(xtrax, "__version__", "unknown"),
  }

  out_path = args.out
  out_path.parent.mkdir(parents=True, exist_ok=True)
  serialized = json.dumps(results, indent=2) + "\n"
  out_path.write_text(serialized, encoding="utf-8")

  bth_env_path = os.environ.get("BTH_RESULTS_PATH")
  if bth_env_path:
    p = Path(bth_env_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(serialized, encoding="utf-8")


  log.info("Wrote results to %s: %s", out_path, results)
  print(json.dumps(results, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
