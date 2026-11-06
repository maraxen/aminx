"""Non-invasive pack shim binding xtrax.pack to aminx campaign units.

A ``--pack`` row runs under the same lock, resume and quarantine rules as
:func:`aminx.host.campaign.run_manifest_row`; only the completion stamp differs. The reference
runner stamps a sibling ``<row>.h5.done.json``. This runner commits an ``xtrax.unit/1`` record
(``unit.json``) inside the unit and, by default, packs the members into ``members.tar.zst``.

Resume order for a row (see :func:`_resolve_existing_unit`):

1. ``unit.json`` present: verify it (record digest, content digest, input digest). A verified unit
   is reused; a unit committed for other inputs is moved aside; any other verification failure is
   a hard error, as corruption is in the reference runner.
2. Otherwise a v3 ``.done.json`` present: validated exactly as the reference runner validates it,
   so a row finished by the legacy runner is reused, not recomputed.
3. Otherwise any existing output is refused, never deleted.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xtrax.pack import (
  UNIT_RECORD,
  CodecUnavailable,
  LockUnavailable,
  UnitPackRefused,
  collect_loose_members,
  is_packed,
  list_unit_members,
  pack_unit,
  read_unit_member,
  record_problem,
  unit_content_digest,
  write_unit_record,
)
from xtrax.run import fsync_directory, fsync_tree
from xtrax.run.unit_index_adapters import AminxDoneMarkerAdapter, UnitJsonAdapter

from aminx.host.campaign import (
  DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
  DEFAULT_LOCK_LEASE_SECONDS,
  FAMILY_DRIVERS,
  InputHashMismatchError,
  LegacyV2DoneMarkerError,
  StaleDoneMarkerSchemaError,
  _already_done_result,
  _campaign_lock_context,
  _collect_sink_arrays,
  _done_marker_path,
  _invalidate_stale_output,
  _lock_path,
  _manifest_rows,
  _partial_output_path,
  _quarantine_output,
  _read_done_marker,
  _row_output_path,
  _select_row,
  _validate_done_marker,
  compute_unit_input_hash,
  load_manifest,
  refuse_driver_family,
)
from aminx.host.runner import sample, score
from aminx.io.campaign_sink import UnifiedCampaignSink
from aminx.run.spec_json import _coerce_field_value, migrate_removed_spec_keys
from aminx.run.specs import SamplingSpecification, ScoringSpecification
from aminx.sampling.multistate_poe import sample_multistate_poe_campaign_row

if TYPE_CHECKING:
  from aminx.types.host_protocols import DistributedLockBackend

logger = logging.getLogger(__name__)

UNIT_FAILED_STAMP = "unit.failed.json"
# Prefix that keeps a pack-record digest from ever comparing equal to a zarr_content_digest:
# the two algorithms hash different things, so a mixed packed/unpacked rerun must read as a mismatch.
PACKED_GATE_DIGEST_PREFIX = "xtrax.pack:"


class InputDigestMismatchError(UnitPackRefused):
  """A committed unit was built from other inputs (R-U2).

  A distinct type, not just the rule id: xtrax also raises ``UnitPackRefused("R-U2", ...)`` for
  a sha256 mismatch inside a pack, which is corruption and must stay a hard error on resume.
  """

  def __init__(self, message: str) -> None:
    super().__init__("R-U2", message)


__all__ = [
  "PACKED_GATE_DIGEST_PREFIX",
  "UNIT_FAILED_STAMP",
  "CodecUnavailable",
  "InputDigestMismatchError",
  "LockUnavailable",
  "UnitPackRefused",
  "aminx_unit_check",
  "execute_packed_manifest",
  "failure_stamp_path",
  "is_packed",
  "list_unit_members",
  "pack_aminx_unit",
  "packed_unit_gate_digest",
  "read_unit_member",
  "resolve_unit_adapter",
  "run_packed_manifest_row",
  "unit_content_digest",
  "validate_aminx_unit",
  "write_aminx_unit",
  "write_aminx_unit_failure",
]


def _legacy_marker_path(unit_path: Path) -> Path:
  return unit_path.with_name(f"{unit_path.name}.done.json")


def _archived_marker_path(unit_path: Path) -> Path:
  return unit_path.with_name(f"{unit_path.name}.done.json.packed")


def failure_stamp_path(unit_dir: Path | str) -> Path:
  """Where a failed attempt is stamped: a sibling of the unit, never a file inside it.

  A file inside a committed unit would become a member, change its content digest and make the
  next resume reject a good unit.
  """
  unit_path = Path(unit_dir)
  return unit_path.with_name(f"{unit_path.name}.{UNIT_FAILED_STAMP}")


def _require_nonempty_store(unit_path: Path) -> None:
  """Refuse (R-U1) a unit with no array member files."""
  store_path = unit_path / "store.zarr" if (unit_path / "store.zarr").exists() else unit_path
  if not store_path.exists():
    msg = f"Unit store path missing in {unit_path}"
    raise UnitPackRefused("R-U1", msg)
  files = [
    p
    for p in store_path.rglob("*")
    if p.is_file() and p.name not in (UNIT_RECORD, ".zgroup", "zarr.json", UNIT_FAILED_STAMP)
  ]
  if not files:
    msg = f"Unit has no array member files in {unit_path} (R-U1 empty unit refusal)"
    raise UnitPackRefused("R-U1", msg)


def aminx_unit_check(
  unit_dir: str | os.PathLike[str],
  record: Mapping[str, Any] | None = None,
  content_digest: str | None = None,
) -> None:
  """Completion check passed to ``pack_unit(..., check=aminx_unit_check)``.

  ``check=`` replaces xtrax's default check, so this one keeps everything the default verifies
  and adds the aminx rule:

  1. the record is well formed and its ``record_digest`` matches its core;
  2. the members being packed hash to the record's ``content_digest``;
  3. the unit holds at least one array member file (not only zarr metadata).

  With ``record`` and ``content_digest`` omitted only rule 3 runs.

  Raises
  ------
  UnitPackRefused
      R-U1 when any rule fails.
  """
  unit_path = Path(unit_dir)
  if record is not None:
    problem = record_problem(dict(record))
    if problem is not None:
      msg = f"{unit_path}: {problem}"
      raise UnitPackRefused("R-U1", msg)
    if content_digest is not None and content_digest != record.get("content_digest"):
      msg = f"{unit_path}: members do not match content_digest"
      raise UnitPackRefused("R-U1", msg)
  _require_nonempty_store(unit_path)


def write_aminx_unit(
  unit_dir: Path | str,
  *,
  sampling_spec_payload: Mapping[str, Any] | None = None,
  input_digest: str | None = None,
  run_id: str | None = None,
  record_name: str = UNIT_RECORD,
  meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Computes unit identity and commits the unit record via ``write_unit_record``.

  Identity comes from ``sampling_spec_payload`` (``compute_unit_input_hash``) or, when there is
  no payload, from an explicit ``input_digest``. One of the two is required: a record without a
  real input digest would let resume accept a unit built from any inputs.

  A sibling legacy ``<unit>.done.json`` present at commit time is copied into
  ``meta['legacy_done_marker']`` here, before the record is digested. The record is never edited
  afterwards, because ``meta`` is part of the digest-covered core.

  Raises
  ------
  ValueError
      If neither ``sampling_spec_payload`` nor ``input_digest`` is given.
  UnitPackRefused
      R-U1 for an empty unit or an existing record; R-U3 from ``write_unit_record``.
  """
  unit_path = Path(unit_dir)
  _require_nonempty_store(unit_path)

  target_record_path = unit_path / record_name
  if target_record_path.exists():
    msg = f"{target_record_path}: unit already has a record"
    raise UnitPackRefused("R-U1", msg)

  unit_meta = dict(meta) if meta else {}
  if sampling_spec_payload is not None:
    input_hash, components = compute_unit_input_hash(sampling_spec_payload)
    unit_meta["input_hash_components"] = components
  elif input_digest is not None:
    input_hash = input_digest
  else:
    msg = "write_aminx_unit needs sampling_spec_payload or an explicit input_digest"
    raise ValueError(msg)
  unit_meta["schema_bridge"] = "aminx_pack_shim_v2"

  legacy_marker = _legacy_marker_path(unit_path)
  if legacy_marker.exists() and "legacy_done_marker" not in unit_meta:
    unit_meta["legacy_done_marker"] = json.loads(legacy_marker.read_text(encoding="utf-8"))

  record = write_unit_record(
    unit_path,
    input_digest=input_hash,
    run_id=run_id or "prod_run",
    input_payload=dict(sampling_spec_payload) if sampling_spec_payload is not None else None,
    meta=unit_meta,
  )

  # write_unit_record always creates unit.json; a custom name is a rename of that exact file.
  if record_name != UNIT_RECORD:
    (unit_path / UNIT_RECORD).replace(target_record_path)
    fsync_directory(unit_path)

  return record


def pack_aminx_unit(
  unit_dir: Path | str,
  *,
  record_name: str = UNIT_RECORD,
  level: int = 3,
  lock: bool = True,
) -> dict[str, Any]:
  """Packs a committed aminx unit into ``members.tar.zst`` + ``members.idx.json``.

  The record is never modified here. After a successful pack, a sibling legacy
  ``<unit>.done.json`` is renamed to ``<unit>.done.json.packed`` only when the record already
  archived exactly that marker (see :func:`write_aminx_unit`); a marker the record does not hold
  is left in place with a warning. A failed pack leaves both record and marker as they were.

  Returns
  -------
  dict[str, Any]
      The record returned by ``pack_unit``.
  """
  unit_path = Path(unit_dir)
  if is_packed(unit_path, record_name=record_name):
    loose = collect_loose_members(unit_path, record_name=record_name)
    if loose:
      msg = (
        f"{unit_path}: files named like pack artifacts present or regenerated loose files: {loose}"
      )
      raise UnitPackRefused("R-U3", msg)

  packed_record = pack_unit(
    unit_path,
    record_name=record_name,
    check=aminx_unit_check,
    level=level,
    lock=lock,
  )

  observed = unit_content_digest(unit_path, record_name=record_name)
  if observed != packed_record.get("content_digest"):
    msg = (
      f"{unit_path}: packed members digest {observed} does not match record "
      f"content_digest {packed_record.get('content_digest')}"
    )
    raise UnitPackRefused("R-U1", msg)

  legacy_marker = _legacy_marker_path(unit_path)
  if legacy_marker.exists():
    archived = (packed_record.get("meta") or {}).get("legacy_done_marker")
    current = json.loads(legacy_marker.read_text(encoding="utf-8"))
    if archived == current:
      legacy_marker.replace(_archived_marker_path(unit_path))
      fsync_directory(unit_path.parent)
    else:
      logger.warning(
        "Legacy marker %s is not archived in the unit record; leaving it in place",
        legacy_marker,
      )
  return packed_record


def validate_aminx_unit(
  unit_dir: Path | str,
  *,
  expected_input_digest: str | None = None,
  record_name: str = UNIT_RECORD,
) -> dict[str, Any]:
  """Validates a unit record and its members on resume.

  Raises
  ------
  ValueError | TypeError
      Record missing, unreadable, not well formed (``record_problem``), or members that do not
      hash to ``content_digest``.
  InputDigestMismatchError
      (a ``UnitPackRefused``, R-U2) when ``expected_input_digest`` differs from the record's
      ``input_digest``. Pack corruption surfaces as another ``UnitPackRefused`` or ``OSError``.
  """
  unit_path = Path(unit_dir)
  record_path = unit_path / record_name
  if not record_path.exists():
    msg = f"Unit record missing at {record_path}"
    raise ValueError(msg)
  try:
    record = json.loads(record_path.read_text(encoding="utf-8"))
  except ValueError as exc:
    msg = f"Corrupted unit record at {record_path}: {exc}"
    raise ValueError(msg) from exc

  if not isinstance(record, dict):
    msg = f"Unit record at {record_path} is not a valid JSON object"
    raise TypeError(msg)

  problem = record_problem(record)
  if problem is not None:
    msg = f"Unit record at {record_path} does not verify: {problem}"
    raise ValueError(msg)

  if expected_input_digest is not None and record.get("input_digest") != expected_input_digest:
    msg = (
      f"Input digest mismatch in {unit_path}: "
      f"expected {expected_input_digest}, got {record.get('input_digest')}"
    )
    raise InputDigestMismatchError(msg)

  observed = unit_content_digest(unit_path, record_name=record_name)
  expected = record.get("content_digest")
  if observed != expected:
    msg = f"Unit content digest mismatch in {unit_path}: expected {expected}, observed {observed}"
    raise ValueError(msg)
  return record


def packed_unit_gate_digest(unit_dir: Path | str) -> str | None:
  """Gate digest for a unit committed with ``unit.json``; None when there is no unit record.

  Used by ``campaign gates``. The value is namespaced with :data:`PACKED_GATE_DIGEST_PREFIX` so
  it can never equal a ``zarr_content_digest`` taken from a ``.done.json`` marker.

  Raises
  ------
  ValueError | TypeError
      When the unit record exists but does not verify.
  """
  unit_path = Path(unit_dir)
  if not (unit_path / UNIT_RECORD).exists():
    return None
  record = validate_aminx_unit(unit_path)
  return f"{PACKED_GATE_DIGEST_PREFIX}{record['content_digest']}"


def write_aminx_unit_failure(
  unit_dir: Path | str,
  *,
  exc: Exception | None = None,
  error: Exception | None = None,
  attempt_id: str | None = None,
  meta: dict[str, Any] | None = None,
) -> Path:
  """Writes the failure stamp for an attempt (R-U4) beside the unit, atomically."""
  stamp_path = failure_stamp_path(unit_dir)
  stamp_path.parent.mkdir(parents=True, exist_ok=True)
  err = exc if exc is not None else error
  payload = {
    "schema": "xtrax.unit_failure/1",
    "attempt_id": attempt_id or uuid.uuid4().hex,
    "error_type": err.__class__.__name__ if err is not None else "UnknownError",
    "error_message": str(err) if err is not None else "Unknown error",
    "failed_at_unix_s": time.time(),
    "meta": meta or {},
  }
  tmp_path = stamp_path.with_name(f"{stamp_path.name}.tmp.{uuid.uuid4().hex}")
  tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  tmp_path.replace(stamp_path)
  return stamp_path


def _packed_already_done(
  record: Mapping[str, Any],
  *,
  unit_dir: Path,
  record_name: str,
  manifest_hash: str,
  input_hash: str,
) -> dict[str, Any]:
  return {
    "status": "already_done",
    "output_h5_path": str(unit_dir),
    "unit_record_path": str(unit_dir / record_name),
    "manifest_row_hash": manifest_hash,
    "attempt_id": (record.get("meta") or {}).get("attempt_id"),
    "input_hash": input_hash,
    "packed": is_packed(unit_dir, record_name=record_name),
    "reuse": {
      "decision": "reused",
      "reason": "unit_record_verified",
      "input_hash": input_hash,
      "artifact_digest": f"{PACKED_GATE_DIGEST_PREFIX}{record.get('content_digest')}",
      "committed_at": record.get("committed_at"),
    },
  }


def _resolve_existing_unit(  # noqa: PLR0911
  *,
  unit_dir: Path,
  record_name: str,
  manifest_hash: str,
  input_hash: str,
  pack: bool,
  pack_lock: bool,
  mutate: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
  """Decide what to do with whatever already sits at ``unit_dir``.

  Returns ``(already_done_result, reuse)``: a result means the row is finished; otherwise
  ``reuse`` describes why it will be computed. With ``mutate=False`` (the unlocked fast path)
  nothing is moved, deleted or packed, and a case that would need a change returns ``(None, …)``
  so the locked pass decides.
  """
  reuse: dict[str, Any] = {
    "decision": "computed",
    "reason": "no_prior_artifact",
    "input_hash": input_hash,
  }
  done_marker_path = _done_marker_path(unit_dir)

  if (unit_dir / record_name).exists():
    try:
      record = validate_aminx_unit(
        unit_dir,
        expected_input_digest=input_hash,
        record_name=record_name,
      )
    except InputDigestMismatchError as exc:
      if not mutate:
        return None, reuse
      logger.warning("Recomputing manifest row %s: %s", manifest_hash, exc)
      return None, {
        "decision": "recomputed",
        "reason": "input_hash_mismatch",
        "input_hash": input_hash,
        "quarantined": _quarantine_output(
          marker_path=done_marker_path,
          output_h5_path=unit_dir,
          reason="input_hash_mismatch",
        ),
      }
    if pack and not is_packed(unit_dir, record_name=record_name):
      if not mutate:
        return None, reuse
      record = pack_aminx_unit(unit_dir, record_name=record_name, lock=pack_lock)
    return (
      _packed_already_done(
        record,
        unit_dir=unit_dir,
        record_name=record_name,
        manifest_hash=manifest_hash,
        input_hash=input_hash,
      ),
      reuse,
    )

  marker = _read_done_marker(done_marker_path)
  if marker is not None:
    try:
      _validate_done_marker(
        marker=marker,
        marker_path=done_marker_path,
        output_h5_path=unit_dir,
        manifest_row_hash=manifest_hash,
        input_hash=input_hash,
      )
    except LegacyV2DoneMarkerError as exc:
      if not mutate:
        return None, reuse
      logger.warning("Recomputing manifest row %s: %s", manifest_hash, exc)
      return None, {
        "decision": "recomputed",
        "reason": "legacy_v2_marker",
        "input_hash": input_hash,
        "quarantined": _quarantine_output(
          marker_path=done_marker_path,
          output_h5_path=unit_dir,
          reason="legacy_v2_marker",
        ),
      }
    except InputHashMismatchError as exc:
      if not mutate:
        return None, reuse
      logger.warning("Recomputing manifest row %s: %s", manifest_hash, exc)
      return None, {
        "decision": "recomputed",
        "reason": "input_hash_mismatch",
        "input_hash": input_hash,
        "stamped_input_hash": marker.get("input_hash"),
        "quarantined": _quarantine_output(
          marker_path=done_marker_path,
          output_h5_path=unit_dir,
          reason="input_hash_mismatch",
        ),
      }
    except StaleDoneMarkerSchemaError as exc:
      if not mutate:
        return None, reuse
      logger.warning("Discarding stale done marker for row %s, recomputing: %s", manifest_hash, exc)
      _invalidate_stale_output(marker_path=done_marker_path, output_h5_path=unit_dir)
      return None, {
        "decision": "recomputed",
        "reason": "stale_marker_schema",
        "input_hash": input_hash,
      }
    # A verified v3 row from the reference runner: reuse it as it is (left loose).
    return (
      _already_done_result(
        marker,
        output_h5_path=unit_dir,
        done_marker_path=done_marker_path,
        manifest_hash=manifest_hash,
        input_hash=input_hash,
      ),
      reuse,
    )

  if unit_dir.exists():
    msg = (
      f"Output already exists without a unit record or done marker for manifest row "
      f"{manifest_hash}: {unit_dir}"
    )
    raise RuntimeError(msg)
  return None, reuse


def _run_worker(
  sampling_spec: SamplingSpecification,
  partial_path: Path,
  *,
  is_multistate: bool,
) -> Any:  # noqa: ANN401
  """The reference runner's family dispatch, writing into ``partial_path``."""
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
    if thermo_drv is None:
      msg = "No family driver registered for 'thermompnn'"
      raise RuntimeError(msg)
    structs = sample_result.get("structures", {}) if isinstance(sample_result, Mapping) else {}
    first_struct = next(iter(structs.values()), {}) if isinstance(structs, Mapping) else {}
    thermo_arrays = first_struct.get("arrays", {}) if isinstance(first_struct, Mapping) else {}
    thermo_drv.write_outputs(
      partial_path,
      thermo_arrays,
      structure_id=str(first_struct.get("structure_id", "structure_0")),
    )
    return sample_result
  if is_multistate:
    return sample_multistate_poe_campaign_row(sampling_spec)
  return sample(sampling_spec)


def run_packed_manifest_row(  # noqa: PLR0915
  manifest_path: Path | str | Sequence[dict[str, Any]],
  *,
  row_index: int | None = None,
  row_hash: str | None = None,
  pack: bool = True,
  pack_lock: bool = True,
  record_name: str = UNIT_RECORD,
  run_id: str | None = None,
  lock_backend: str = "local_fs",
  distributed_lock_backend: DistributedLockBackend | None = None,
  lock_lease_seconds: int = DEFAULT_LOCK_LEASE_SECONDS,
  heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
  unified_sink: UnifiedCampaignSink | None = None,
  sink_root: str | Path | None = None,
) -> dict[str, Any]:
  """Executes one manifest row and commits it as an xtrax unit (packed unless ``pack=False``).

  Locking, heartbeat, quarantine and fsync behave as in
  :func:`aminx.host.campaign.run_manifest_row`; see the module docstring for resume order.
  """
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

  manifest_hash = str(row["manifest_row_hash"])
  raw_output_path = sampling_spec_payload.get("output_h5_path")
  if raw_output_path is not None:
    unit_dir = Path(raw_output_path).resolve()
  else:
    base_dir = Path(payload.get("output_dir", "campaign_runs"))
    campaign_id = str(payload.get("campaign_id", "default"))
    unit_dir = Path(_row_output_path(base_dir, campaign_id, manifest_hash))
  unit_dir.parent.mkdir(parents=True, exist_ok=True)

  input_hash, _ = compute_unit_input_hash(sampling_spec_payload)
  resolve_kwargs: dict[str, Any] = {
    "unit_dir": unit_dir,
    "record_name": record_name,
    "manifest_hash": manifest_hash,
    "input_hash": input_hash,
    "pack": pack,
    "pack_lock": pack_lock,
  }
  done, _ = _resolve_existing_unit(**resolve_kwargs, mutate=False)
  if done is not None:
    return done

  attempt_id = uuid.uuid4().hex
  partial_path = _partial_output_path(unit_dir, attempt_id)
  sink_arrays: dict[str, Any] | None = None
  sample_result: Any = None
  sampling_spec: SamplingSpecification | None = None
  with _campaign_lock_context(
    lock_backend=lock_backend,
    distributed_lock_backend=distributed_lock_backend,
    lock_key_path=_lock_path(unit_dir),
    owner_token=f"{manifest_hash}:{attempt_id}",
    manifest_row_hash=manifest_hash,
    attempt_id=attempt_id,
    lease_seconds=lock_lease_seconds,
    heartbeat_interval_seconds=heartbeat_interval_seconds,
  ) as heartbeat_errors:
    done, reuse = _resolve_existing_unit(**resolve_kwargs, mutate=True)
    if done is not None:
      return done

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
    if "backend_arm_id" not in row:
      refuse_driver_family(sampling_spec, surface="campaign")
    is_multistate = (
      isinstance(sampling_spec.inputs, (list, tuple)) and len(sampling_spec.inputs) > 1
    )

    record_committed = False
    try:
      sample_result = _run_worker(sampling_spec, partial_path, is_multistate=is_multistate)
      if heartbeat_errors:
        msg = (
          f"Lock heartbeat failed while executing manifest row {manifest_hash}: "
          f"{heartbeat_errors[0]}"
        )
        raise RuntimeError(msg) from heartbeat_errors[0]  # noqa: TRY301
      if not partial_path.exists():
        msg = f"Worker did not produce expected partial output store: {partial_path}"
        raise RuntimeError(msg)  # noqa: TRY301
      fsync_tree(partial_path)
      partial_path.replace(unit_dir)
      fsync_directory(unit_dir.parent)

      record = write_aminx_unit(
        unit_dir,
        sampling_spec_payload=sampling_spec_payload,
        run_id=run_id or "prod_run",
        record_name=record_name,
        meta={"attempt_id": attempt_id, "manifest_row_hash": manifest_hash},
      )
      record_committed = True
      if unified_sink is not None or sink_root is not None:
        # Read the arrays while the store is still a loose zarr; a pack hides it.
        sink_arrays = _collect_sink_arrays(unit_dir, sample_result)
      if pack:
        record = pack_aminx_unit(unit_dir, record_name=record_name, lock=pack_lock)
      reuse["artifact_digest"] = f"{PACKED_GATE_DIGEST_PREFIX}{record['content_digest']}"
    except Exception as exc:
      # After the record is committed the unit is valid (only the pack failed): the stamp still
      # goes beside it, and the next resume packs it.
      write_aminx_unit_failure(
        unit_dir,
        exc=exc,
        attempt_id=attempt_id,
        meta={"record_committed": record_committed},
      )
      raise
    finally:
      if partial_path.exists():
        _remove_partial(partial_path)
  failure_stamp_path(unit_dir).unlink(missing_ok=True)

  active_sink = unified_sink
  if active_sink is None and sink_root is not None:
    active_sink = UnifiedCampaignSink(
      sink_root,
      campaign_id=str(row.get("campaign_id") or payload.get("campaign_id") or "campaign"),
    )
  if active_sink is not None and sampling_spec is not None:
    arm_id = str(row.get("backend_arm_id") or sampling_spec.model_family or "default")
    active_sink.write_arm_results(
      arm_id=arm_id,
      model_family=str(sampling_spec.model_family or "proteinmpnn"),
      checkpoint_id=str(sampling_spec.checkpoint_id or ""),
      manifest_row_hash=manifest_hash,
      records=[{"arm_id": arm_id, "manifest_row_hash": manifest_hash, "status": "completed"}],
      arrays=sink_arrays or {},
    )

  result_payload = (
    dict(sample_result) if isinstance(sample_result, dict) else {"sample_result": sample_result}
  )
  result_payload.update(
    {
      "status": str(result_payload.get("status", "completed")),
      "output_h5_path": str(unit_dir),
      "unit_record_path": str(unit_dir / record_name),
      "manifest_row_hash": manifest_hash,
      "attempt_id": attempt_id,
      "lock_backend": lock_backend,
      "input_hash": input_hash,
      "packed": pack,
      "reuse": reuse,
    },
  )
  return result_payload


def _remove_partial(path: Path) -> None:
  """Remove a partial output (a directory store or a single file)."""
  if path.is_dir():
    shutil.rmtree(path)
  else:
    path.unlink(missing_ok=True)


def execute_packed_manifest(
  manifest_path: Path | str | Sequence[dict[str, Any]],
  *,
  row_hashes: Sequence[str] | None = None,
  continue_on_error: bool = False,
  pack: bool = True,
  **kwargs: Any,  # noqa: ANN401
) -> list[dict[str, Any]]:
  """Executes manifest rows sequentially through :func:`run_packed_manifest_row`.

  ``kwargs`` (lock backend, lease, heartbeat, sink, record name, run id) pass through unchanged.
  """
  rows = _manifest_rows(manifest_path)
  requested_hash_set = set(row_hashes or ())

  results: list[dict[str, Any]] = []
  for idx, row in enumerate(rows):
    r_hash = str(row["manifest_row_hash"])
    if requested_hash_set and r_hash not in requested_hash_set:
      continue
    try:
      results.append(run_packed_manifest_row(manifest_path, row_index=idx, pack=pack, **kwargs))
    except Exception as exc:
      results.append({"manifest_row_hash": r_hash, "status": "error", "error": str(exc)})
      if not continue_on_error:
        raise
  return results


def resolve_unit_adapter(
  path: Path | str,
  record_name: str = UNIT_RECORD,
) -> UnitJsonAdapter | AminxDoneMarkerAdapter:
  """Returns the xtrax index adapter that can read the unit at ``path``.

  ``UnitJsonAdapter`` for a unit with a ``unit.json`` record (loose or packed) or an archived
  ``.done.json.packed`` marker; ``AminxDoneMarkerAdapter`` for a loose legacy ``.done.json``.

  Raises
  ------
  ValueError
      For a custom ``record_name``: ``UnitJsonAdapter`` reads only ``unit.json``, so no xtrax
      adapter can index such a unit, and returning one would report it as missing.
  """
  if record_name != UNIT_RECORD:
    msg = (
      f"No xtrax index adapter reads a custom record name ({record_name!r}); "
      f"UnitJsonAdapter reads only {UNIT_RECORD!r}."
    )
    raise ValueError(msg)
  unit_path = Path(path)
  if (
    is_packed(unit_path, record_name=record_name)
    or (unit_path / record_name).exists()
    or _archived_marker_path(unit_path).exists()
  ):
    return UnitJsonAdapter()
  if _legacy_marker_path(unit_path).exists():
    return AminxDoneMarkerAdapter()
  return UnitJsonAdapter()
