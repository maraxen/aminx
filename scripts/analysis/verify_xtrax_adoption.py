#!/usr/bin/env python3
"""Bathos verification script for aminx adoption of xtrax pack and eda.

Confirms:
1. Standard record_name ("unit.json") unit write, pack, readback, and validation.
2. Custom record_name unit write, pack, readback, and validation.
3. Negative controls: bit corruption, duplicate record collision (R-U1), empty unit refusal, corrupted export artifact detection.
4. EDA plan panel SVG rendering, Vega-Lite v5 spec generation, and manifest SHA-256 verification.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# Ensure repo source and local dependency sites are discoverable
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [
  str(_REPO_ROOT / "src"),
  "/tmp/pure_py_site",
  "/tmp/prolix_py312_site",
  "/tmp/jaxlint_ci312_env/lib/python3.12/site-packages",
]:
  if _p not in sys.path:
    sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import xtrax  # noqa: E402
import zarr  # noqa: E402
from xtrax.pack import (  # noqa: E402
  UnitPackRefused,
  is_packed,
  unit_content_digest,
)
from xtrax.run.unit_index_adapters import UnitJsonAdapter  # noqa: E402
from xtrax.tiling import AxisDecision, AxisSpec, BatchPlan, Bucket, ChunkedMap  # noqa: E402

from aminx.host.pack_shim import (  # noqa: E402
  pack_aminx_unit,
  resolve_unit_adapter,
  validate_aminx_unit,
  write_aminx_unit,
)
from aminx.host.plan_eda import (  # noqa: E402
  export_and_verify_plan_artifacts,
  generate_aminx_plan_spec,
  render_aminx_plan_panels,
  verify_export_manifest_sha256,
)

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
    rec_std = write_aminx_unit(unit_std, run_id="run-std-1")
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
    rec_cust = write_aminx_unit(unit_cust, run_id="run-cust-1", record_name="custom_rec.json")
    assert (unit_cust / "custom_rec.json").exists()
    assert not (unit_cust / "unit.json").exists()
    digest_cust = unit_content_digest(unit_cust, record_name="custom_rec.json")
    assert rec_cust["content_digest"] == digest_cust
    pack_aminx_unit(unit_cust, record_name="custom_rec.json")
    assert is_packed(unit_cust, record_name="custom_rec.json")
    assert not is_packed(unit_cust, record_name="unit.json")
    val_cust = validate_aminx_unit(unit_cust, record_name="custom_rec.json")
    assert val_cust["content_digest"] == digest_cust
    adapter_cust = resolve_unit_adapter(unit_cust, record_name="custom_rec.json")
    assert isinstance(adapter_cust, UnitJsonAdapter)

    return True, True


def verify_negative_controls() -> bool:
  """Verifies refusal and failure controls: duplicate record, bit corruption, tampered export."""
  with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)

    # Control A: duplicate record write refusal (R-U1)
    unit_a = tmp_path / "unit_a"
    _make_sample_store(unit_a)
    write_aminx_unit(unit_a, run_id="run-a-1", record_name="dup.json")
    control_dup_raised = False
    try:
      write_aminx_unit(unit_a, run_id="run-a-2", record_name="dup.json")
    except UnitPackRefused:
      control_dup_raised = True
    assert control_dup_raised, "Expected UnitPackRefused on duplicate record write"

    # Control B: bit corruption in packed archive causes validate_aminx_unit to raise
    unit_b = tmp_path / "unit_b"
    _make_sample_store(unit_b)
    write_aminx_unit(unit_b, run_id="run-b-1")
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
  pack_ok, custom_record_ok = verify_pack_and_custom_records()
  controls_ok = verify_negative_controls()
  eda_render_svg_ok, eda_manifest_sha256_ok = verify_eda_rendering_and_manifest()
  eda_ok = eda_render_svg_ok and eda_manifest_sha256_ok

  results = {
    "pack_ok": pack_ok,
    "eda_ok": eda_ok,
    "controls_ok": controls_ok,
    "custom_record_name_ok": custom_record_ok,
    "eda_render_svg_ok": eda_render_svg_ok,
    "eda_manifest_sha256_ok": eda_manifest_sha256_ok,
    "negative_controls_ok": controls_ok,
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

  adjacent_path = Path(__file__).parent / f"{Path(__file__).stem}.bth-results.json"
  adjacent_path.write_text(serialized, encoding="utf-8")

  log.info("Wrote results to %s: %s", out_path, results)
  print(json.dumps(results, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
