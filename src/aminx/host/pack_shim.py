"""Non-invasive pack shim binding xtrax.pack to aminx units."""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xtrax.pack import (
  CodecUnavailable,
  LockUnavailable,
  UnitPackRefused,
  collect_loose_members,
  is_packed,
  list_unit_members,
  pack_unit,
  read_unit_member,
  unit_content_digest,
  write_unit_record,
)
from xtrax.run.unit_index_adapters import AminxDoneMarkerAdapter, UnitJsonAdapter

from aminx.host.campaign import (
  FAMILY_DRIVERS,
  _manifest_rows,
  _partial_output_path,
  _row_output_path,
  _select_row,
  compute_unit_input_hash,
  load_manifest,
  refuse_driver_family,
)
from aminx.host.runner import sample, score
from aminx.run.spec_json import _coerce_field_value, migrate_removed_spec_keys
from aminx.run.specs import SamplingSpecification, ScoringSpecification
from aminx.sampling.multistate_poe import sample_multistate_poe_campaign_row

logger = logging.getLogger(__name__)

UNIT_FAILED_STAMP = "unit.failed.json"

__all__ = [
  "UNIT_FAILED_STAMP",
  "CodecUnavailable",
  "LockUnavailable",
  "UnitPackRefused",
  "aminx_unit_check",
  "execute_packed_manifest",
  "is_packed",
  "list_unit_members",
  "pack_aminx_unit",
  "read_unit_member",
  "resolve_unit_adapter",
  "run_packed_manifest_row",
  "unit_content_digest",
  "validate_aminx_unit",
  "write_aminx_unit",
  "write_aminx_unit_failure",
]


def aminx_unit_check(
  unit_dir: Path | str,
  record: Mapping[str, Any] | None = None,
  content_digest: str | None = None,
) -> None:
  """Verification callback passed to pack_unit(..., check=aminx_unit_check).

  Validates that:
  1. The unit directory contains store files.
  2. The store has valid array datasets (excluding .zgroup and zarr.json metadata).
  3. No empty units are committed.

  Raises
  ------
  UnitPackRefused
      If validation fails (maps to R-U1 empty unit refusal).
  """
  del record, content_digest
  unit_path = Path(unit_dir)
  store_path = unit_path / "store.zarr" if (unit_path / "store.zarr").exists() else unit_path
  if not store_path.exists():
    msg = f"Unit store path missing in {unit_path}"
    raise UnitPackRefused("R-U1", msg)

  # Exclude metadata, failure stamps, and unit record; require actual array chunks
  files = [
    p
    for p in store_path.rglob("*")
    if p.is_file() and p.name not in ("unit.json", ".zgroup", "zarr.json", UNIT_FAILED_STAMP)
  ]
  if not files:
    msg = f"Unit has no array member files in {unit_path} (R-U1 empty unit refusal)"
    raise UnitPackRefused("R-U1", msg)


def write_aminx_unit(
  unit_dir: Path | str,
  *,
  sampling_spec_payload: Mapping[str, Any] | None = None,
  run_id: str | None = None,
  record_name: str = "unit.json",
  meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Computes unit identity and commits unit record via write_unit_record.

  Parameters
  ----------
  unit_dir : Path | str
      Directory containing the completed unit outputs.
  sampling_spec_payload : Mapping[str, Any] | None, default=None
      Sampling specification payload used for compute_unit_input_hash.
  run_id : str | None, default=None
      Unique execution run ID.
  record_name : str, default="unit.json"
      Name of the unit record file.
  meta : dict[str, Any] | None, default=None
      Additional metadata to record in unit record.

  Returns
  -------
  dict[str, Any]
      The committed unit record.
  """
  unit_path = Path(unit_dir)
  # Enforce non-empty store check before committing record (Directive 6 / R-U1)
  aminx_unit_check(unit_path, record={}, content_digest="")

  target_record_path = unit_path / record_name
  if target_record_path.exists():
    msg = f"{target_record_path}: unit already has a record"
    raise UnitPackRefused("R-U1", msg)

  if sampling_spec_payload is not None:
    input_hash, components = compute_unit_input_hash(sampling_spec_payload)
  else:
    input_hash = (meta or {}).get("input_digest", "0" * 64)
    components = (meta or {}).get("input_hash_components", {})

  unit_meta = meta.copy() if meta else {}
  if components:
    unit_meta["input_hash_components"] = components
  unit_meta["schema_bridge"] = "aminx_pack_shim_v1"

  record = write_unit_record(
    unit_path,
    input_digest=input_hash,
    run_id=run_id or "prod_run",
    input_payload=dict(sampling_spec_payload) if sampling_spec_payload is not None else None,
    meta=unit_meta,
  )

  # If custom record_name was requested, atomically relocate unit.json to target_record_path
  if record_name != "unit.json":
    default_record_path = unit_path / "unit.json"
    if default_record_path.exists():
      default_record_path.replace(target_record_path)

  return record


def pack_aminx_unit(
  unit_dir: Path | str,
  *,
  record_name: str = "unit.json",
  level: int = 3,
  lock: bool = True,
) -> dict[str, Any]:
  """Packs an aminx unit into members.tar.zst + members.idx.json.

  If a legacy sibling marker exists (e.g. `<unit_dir>.done.json`), it is
  quarantined and archived:
  1. Embedded in unit.json under meta['legacy_done_marker'].
  2. Atomically renamed to `<unit_dir>.done.json.packed`.

  Parameters
  ----------
  unit_dir : Path | str
      Unit directory containing loose files and unit.json.
  record_name : str, default="unit.json"
      Unit record name.
  level : int, default=3
      Zstandard compression level.
  lock : bool, default=True
      Whether to acquire POSIX file lock during packing.

  Returns
  -------
  dict[str, Any]
      Storage metadata from pack_unit.
  """
  unit_path = Path(unit_dir)
  if is_packed(unit_path, record_name=record_name):
    # Check if unexpected loose member files were regenerated in an already-packed unit
    loose = collect_loose_members(unit_path, record_name=record_name)
    if loose:
      msg = (
        f"{unit_path}: files named like pack artifacts present or regenerated loose files: {loose}"
      )
      raise UnitPackRefused("R-U3", msg)

  # Archive and quarantine legacy sibling done marker if present
  legacy_marker = unit_path.with_name(f"{unit_path.name}.done.json")
  if legacy_marker.exists():
    try:
      legacy_data = json.loads(legacy_marker.read_text(encoding="utf-8"))
      record_path = unit_path / record_name
      if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record.setdefault("meta", {})["legacy_done_marker"] = legacy_data
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
      legacy_marker.rename(unit_path.with_name(f"{unit_path.name}.done.json.packed"))
    except (OSError, ValueError) as exc:
      logger.warning("Failed to archive legacy done marker at %s: %s", legacy_marker, exc)

  packed_result = pack_unit(
    unit_path,
    record_name=record_name,
    check=aminx_unit_check,
    level=level,
    lock=lock,
  )

  # Verify member integrity post-pack
  unit_content_digest(unit_path, record_name=record_name)
  return packed_result


def validate_aminx_unit(
  unit_dir: Path | str,
  *,
  expected_input_digest: str | None = None,
  record_name: str = "unit.json",
) -> dict[str, Any]:
  """Validates unit record and member integrity on resume.

  Raises
  ------
  ValueError | TypeError
      If unit record is missing, corrupted, or unit_content_digest mismatches record['content_digest'].
  UnitPackRefused
      If expected_input_digest is provided and mismatches record['input_digest'].
  """
  unit_path = Path(unit_dir)
  record_path = unit_path / record_name
  if not record_path.exists():
    msg = f"Unit record missing at {record_path}"
    raise ValueError(msg)
  try:
    record = json.loads(record_path.read_text(encoding="utf-8"))
  except Exception as exc:
    msg = f"Corrupted unit record at {record_path}: {exc}"
    raise ValueError(msg) from exc

  if not isinstance(record, dict):
    msg = f"Unit record at {record_path} is not a valid JSON object"
    raise TypeError(msg)

  if expected_input_digest is not None and record.get("input_digest") != expected_input_digest:
    msg = (
      f"Input digest mismatch in {unit_path}: "
      f"expected {expected_input_digest}, got {record.get('input_digest')}"
    )
    raise UnitPackRefused("R-U2", msg)

  observed = unit_content_digest(unit_path, record_name=record_name)
  expected = record.get("content_digest")
  if observed != expected:
    msg = f"Unit content digest mismatch in {unit_path}: expected {expected}, observed {observed}"
    raise ValueError(msg)
  return record


def write_aminx_unit_failure(
  unit_dir: Path | str,
  *,
  exc: Exception | None = None,
  error: Exception | None = None,
  attempt_id: str | None = None,
  record_name: str = "unit.json",
  meta: dict[str, Any] | None = None,
) -> Path:
  """Writes UNIT_FAILED_STAMP upon worker failure (R-U4)."""
  del record_name
  unit_path = Path(unit_dir)
  unit_path.mkdir(parents=True, exist_ok=True)
  stamp_path = unit_path / UNIT_FAILED_STAMP
  err = exc if exc is not None else error
  err_msg = str(err) if err is not None else "Unknown error"
  err_type = err.__class__.__name__ if err is not None else "UnknownError"

  payload = {
    "schema": "xtrax.unit_failure/1",
    "attempt_id": attempt_id or uuid.uuid4().hex,
    "error_type": err_type,
    "error_message": err_msg,
    "failed_at_unix_s": time.time(),
    "meta": meta or {},
  }
  stamp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  return stamp_path


def run_packed_manifest_row(  # noqa: PLR0915
  manifest_path: Path | str,
  *,
  row_index: int | None = None,
  row_hash: str | None = None,
  pack: bool = True,
  record_name: str = "unit.json",
  **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
  """Executes a manifest row and commits via write_aminx_unit / pack_aminx_unit."""
  payload = (
    load_manifest(manifest_path)
    if isinstance(manifest_path, (str, Path))
    else {"rows": list(manifest_path)}
  )
  rows = _manifest_rows(manifest_path)
  row = _select_row(rows, row_index=row_index, row_hash=row_hash)
  sampling_spec_payload = row.get("sampling_spec")
  if not isinstance(sampling_spec_payload, dict):
    msg = "Manifest row must include a 'sampling_spec' object."
    raise TypeError(msg)

  row_hash_str = str(row["manifest_row_hash"])
  base_dir = Path(payload.get("output_dir", "campaign_runs"))
  campaign_id = str(payload.get("campaign_id", "default"))
  raw_output_path = sampling_spec_payload.get("output_h5_path")
  unit_dir = (
    Path(raw_output_path).resolve()
    if raw_output_path is not None
    else Path(_row_output_path(base_dir, campaign_id, row_hash_str)).resolve()
  )

  # Resume check: if already completed, validate and return
  input_hash, _ = compute_unit_input_hash(sampling_spec_payload)
  if (unit_dir / record_name).exists() or is_packed(unit_dir, record_name=record_name):
    try:
      record = validate_aminx_unit(
        unit_dir,
        expected_input_digest=input_hash,
        record_name=record_name,
      )
      return {
        "manifest_row_hash": row_hash_str,
        "status": "already_done",
        "record": record,
        "unit_dir": str(unit_dir),
      }
    except (OSError, ValueError, TypeError, UnitPackRefused) as exc:
      logger.warning(
        "Resume validation failed for %s: %s; recomputing",
        unit_dir,
        exc,
      )

  attempt_id = uuid.uuid4().hex
  partial_path = _partial_output_path(unit_dir, attempt_id)
  partial_path.parent.mkdir(parents=True, exist_ok=True)

  worker_payload = dict(sampling_spec_payload)
  worker_payload["output_path"] = str(partial_path)
  worker_payload["output_zarr_path"] = str(partial_path)
  worker_payload["output_h5_path"] = str(partial_path)
  worker_payload = migrate_removed_spec_keys(worker_payload)
  worker_payload = {
    key: _coerce_field_value(SamplingSpecification, key, value)
    for key, value in worker_payload.items()
  }
  sampling_spec = SamplingSpecification(**worker_payload)
  is_multi_backend_arm = "backend_arm_id" in row
  if not is_multi_backend_arm:
    refuse_driver_family(sampling_spec, surface="campaign")

  is_multistate_poe_row = (
    isinstance(sampling_spec.inputs, (list, tuple)) and len(sampling_spec.inputs) > 1
  )

  try:
    if sampling_spec.model_family == "thermompnn":
      score_spec = ScoringSpecification(
        inputs=sampling_spec.inputs,
        model_family="thermompnn",
        checkpoint_id=sampling_spec.checkpoint_id,
        thermompnn=sampling_spec.thermompnn,
        output_kind="ddg",
        random_seed=sampling_spec.random_seed,
        return_logits=False,
      )
      sample_result = score(score_spec)
      thermo_drv = FAMILY_DRIVERS.get("thermompnn")
      structs = sample_result.get("structures", {}) if isinstance(sample_result, Mapping) else {}
      first_struct = next(iter(structs.values()), {}) if isinstance(structs, Mapping) else {}
      thermo_arrays = first_struct.get("arrays", {}) if isinstance(first_struct, Mapping) else {}
      if thermo_drv is not None:
        thermo_drv.write_outputs(
          partial_path,
          thermo_arrays,
          structure_id=str(first_struct.get("structure_id", "structure_0")),
        )
    elif is_multistate_poe_row:
      sample_multistate_poe_campaign_row(sampling_spec)
    else:
      sample(sampling_spec)

    if not partial_path.exists():
      msg = f"Worker did not produce expected partial output store: {partial_path}"
      raise RuntimeError(msg)  # noqa: TRY301

    if unit_dir.exists():
      if unit_dir.is_dir():
        shutil.rmtree(unit_dir)
      else:
        unit_dir.unlink()
    partial_path.replace(unit_dir)

    record = write_aminx_unit(
      unit_dir,
      sampling_spec_payload=sampling_spec_payload,
      run_id=kwargs.get("run_id", "prod_run"),
      record_name=record_name,
    )
    if pack:
      pack_aminx_unit(unit_dir, record_name=record_name)

    return {
      "manifest_row_hash": row_hash_str,
      "status": "completed",
      "record": record,
      "unit_dir": str(unit_dir),
    }

  except Exception as exc:
    write_aminx_unit_failure(unit_dir, exc=exc, attempt_id=attempt_id)
    raise
  finally:
    if partial_path.exists():
      shutil.rmtree(partial_path)


def execute_packed_manifest(
  manifest_path: Path | str,
  *,
  row_hashes: Sequence[str] | None = None,
  continue_on_error: bool = False,
  pack: bool = True,
  **kwargs: Any,  # noqa: ANN401
) -> list[dict[str, Any]]:
  """Executes an entire manifest sequentially or with worker dispatch using packed units."""
  rows = _manifest_rows(manifest_path)
  requested_hashes = tuple(row_hashes or ())
  requested_hash_set = set(requested_hashes)

  results: list[dict[str, Any]] = []
  for idx, row in enumerate(rows):
    r_hash = str(row["manifest_row_hash"])
    if requested_hash_set and r_hash not in requested_hash_set:
      continue
    try:
      res = run_packed_manifest_row(
        manifest_path,
        row_index=idx,
        pack=pack,
        **kwargs,
      )
      results.append(res)
    except Exception as exc:
      results.append(
        {
          "manifest_row_hash": r_hash,
          "status": "error",
          "error": str(exc),
        },
      )
      if not continue_on_error:
        raise
  return results


def resolve_unit_adapter(
  path: Path | str,
  record_name: str = "unit.json",
) -> UnitJsonAdapter | AminxDoneMarkerAdapter:
  """Returns the appropriate adapter based on pack state and marker presence.

  Prioritizes is_packed / record_name / .done.json.packed returning UnitJsonAdapter(),
  falling back to loose .done.json returning AminxDoneMarkerAdapter().
  """
  unit_path = Path(path)
  if (
    is_packed(unit_path, record_name=record_name)
    or (unit_path / record_name).exists()
    or (unit_path.parent / f"{unit_path.name}.done.json.packed").exists()
  ):
    return UnitJsonAdapter()
  if unit_path.with_name(f"{unit_path.name}.done.json").exists():
    return AminxDoneMarkerAdapter()
  return UnitJsonAdapter()
