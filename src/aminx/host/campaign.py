"""Planner and worker entrypoints for scheduler-agnostic campaign manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from xtrax.run import (
  canonical_json_bytes,
  fsync_directory,
  fsync_file,
  fsync_tree,
  zarr_content_digest,
)

# campaign manifest functions are implemented in this module (see build_manifest_row et al.)
from aminx.host.runner import sample
from aminx.host.spec_partition import campaign_sampling_spec_payload
from aminx.run.spec_json import _coerce_field_value
from aminx.run.specs import SamplingSpecification, pop_deprecated_spec_kwargs
from aminx.runtime import configure_multiprocessing
from aminx.sampling.multistate_poe import sample_multistate_poe_campaign_row

if TYPE_CHECKING:
  from aminx.types.host_protocols import DistributedLockBackend

logger = logging.getLogger(__name__)

LOCK_SCHEMA_VERSION = "campaign_lock_v1"
DONE_MARKER_SCHEMA_VERSION = "campaign_done_marker_v2"
MANIFEST_ROW_SCHEMA_VERSION = "campaign_manifest_row_v1"
MANIFEST_SCHEMA_VERSION = "campaign_manifest_v1"
DEFAULT_LOCK_LEASE_SECONDS = 1800
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 60
GateCellKey = tuple[str, str, str, tuple[str, ...], tuple[str, ...], bool, bool]
ScaleStageResult = dict[str, Any]


def build_manifest_row(
  spec: SamplingSpecification,
  *,
  campaign_id: str,
  job_index: int,
  chunk_id: int,
  sample_start: int,
  sample_count: int,
  fixed_policy: str,
  state_weight_profile: str,
  planner_version: str,
  dataset_fingerprint: str,
  environment_image: str,
  git_sha: str,
  config_hash: str,
  job_id: str,
  declared_state_weights: Sequence[float] | None = None,
) -> dict[str, Any]:
  """Build a deterministic manifest row with SHA256 hash.

  Computes a deterministic hash from schema_version, campaign_id, job_id,
  chunk_id, sample_start, sample_count, fixed_policy, state_weight_profile,
  model_family, ligand/sidechain_conditioning, multi_state_strategy,
  temperature, and backbone_noise. job_index is intentionally excluded from
  the hash payload: it is derivable from job_id and included in the row dict
  for caller convenience.

  ``declared_state_weights`` is the numeric vector a non-default ``state_weight_profile``
  LABEL resolved to (``None`` for the ``"equal"`` profile). It is hashed only when present, so
  every pre-existing row -- all of which are ``"equal"`` -- keeps the hash it always had, while
  two campaigns that reuse one label for different vectors can no longer collide on a hash (and
  therefore on an output path).

  Returns a dict ready for plan_campaign_manifest to extend with
  output_h5_path and sampling_spec.
  """
  temperature_list = list(spec.run_spec.sampling.temperature) if spec.run_spec.sampling.temperature else []
  backbone_noise_list = list(spec.run_spec.sampling.backbone_noise) if spec.run_spec.sampling.backbone_noise else []
  hash_payload = {
    "schema_version": MANIFEST_ROW_SCHEMA_VERSION,
    "campaign_id": campaign_id,
    "job_id": job_id,
    "chunk_id": int(chunk_id),
    "sample_start": int(sample_start),
    "sample_count": int(sample_count),
    "fixed_policy": fixed_policy,
    "state_weight_profile": state_weight_profile,
    "model_family": str(spec.model_family or ""),
    "ligand_conditioning": bool(spec.ligand_conditioning),
    "sidechain_conditioning": bool(spec.sidechain_conditioning),
    "multi_state_strategy": str(spec.multi_state_strategy or ""),
    "temperature": [str(float(t)) for t in temperature_list],
    "backbone_noise": [str(float(n)) for n in backbone_noise_list],
  }
  if declared_state_weights is not None:
    hash_payload["state_weights"] = [str(float(w)) for w in declared_state_weights]
  row_hash = hashlib.sha256(canonical_json_bytes(hash_payload)).hexdigest()
  return {
    "manifest_row_hash": row_hash,
    "job_id": job_id,
    "job_index": int(job_index),
    "chunk_id": int(chunk_id),
    "sample_start": int(sample_start),
    "sample_count": int(sample_count),
    "fixed_policy": fixed_policy,
    "state_weight_profile": state_weight_profile,
    "multi_state_strategy": str(spec.multi_state_strategy or ""),
    "temperature": temperature_list,
    "backbone_noise": backbone_noise_list,
    "ligand_conditioning": bool(spec.ligand_conditioning),
    "sidechain_conditioning": bool(spec.sidechain_conditioning),
    "checkpoint_id": spec.checkpoint_id,
    "planner_version": planner_version,
    "dataset_fingerprint": dataset_fingerprint,
    "environment_image": environment_image,
    "git_sha": git_sha,
    "config_hash": config_hash,
  }


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
  """Load and validate a campaign manifest JSON file.

  Validates that the file is valid UTF-8 JSON, the root is a dict,
  a 'rows' key is present and is a list, and each row is a dict.
  Raises FileNotFoundError, ValueError, or TypeError on invalid input.
  """
  path = Path(manifest_path)
  if not path.exists():
    msg = f"Manifest file not found: {path}"
    raise FileNotFoundError(msg)
  try:
    payload = json.loads(path.read_bytes().decode("utf-8"))
  except json.JSONDecodeError as exc:
    msg = f"Invalid JSON in manifest file {path}: {exc}"
    raise ValueError(msg) from exc
  except UnicodeDecodeError as exc:
    msg = f"Manifest file {path} is not valid UTF-8: {exc}"
    raise ValueError(msg) from exc
  if not isinstance(payload, dict):
    msg = f"Manifest payload must be a JSON object, got {type(payload).__name__}"
    raise TypeError(msg)
  if "rows" not in payload:
    msg = "Manifest payload must include 'rows' key"
    raise ValueError(msg)
  rows = payload["rows"]
  if not isinstance(rows, list):
    msg = f"Manifest 'rows' must be a list, got {type(rows).__name__}"
    raise TypeError(msg)
  for idx, row in enumerate(rows):
    if not isinstance(row, dict):
      msg = f"Manifest row {idx} must be a JSON object, got {type(row).__name__}"
      raise TypeError(msg)
  return payload


def validate_manifest_rows(rows: list[dict[str, Any]]) -> None:
  """Validate manifest rows for uniqueness and required fields.

  Enforces:
  - All rows have a non-empty manifest_row_hash
  - No duplicate manifest_row_hash (zero_lineage_collisions gate invariant)
  - All rows have non-empty checkpoint_id, fixed_policy, state_weight_profile
  - A row naming a real arm (fixed_policy != "none") actually CARRIES a mask in its
    sampling_spec -- see below

  **`required_fixed_policies` is GONE, deliberately.** It did this::

      policies_seen = {row["fixed_policy"] for row in rows}
      missing = set(required_fixed_policies) - policies_seen

  and `plan_campaign_manifest` called it with the very tuple its own row loop had just
  consumed. It compared the planner's output against the planner's input: it could not fail.
  It "validated" that a string label was present -- nothing about whether a residue was
  fixed. It was green for the entire life of the decorative-policy bug, across 882/882 beads
  that held nothing fixed. Fixing the comparison would not help; a gate whose two sides come
  from the same source is theatre wherever you point it.

  What replaces it checks a row against ITSELF: if you say you have an arm, the mask must be
  in your own sampling_spec. That is a claim the planner can actually get wrong.

  **This is not the real gate.** The final artifact is not this one: consumers patch the
  planned manifest afterwards and write it with a raw `json.dumps`, so nothing here survives
  to the worker as a guarantee. The binding check is the two-armed differential at sampling
  time (masked vs unmasked must DIFFER). Do not mistake this for proof.

  Raises ValueError on any violation.
  """
  hashes_seen: set[str] = set()
  for idx, row in enumerate(rows):
    if not isinstance(row, dict):  # pragma: no cover - guarded by load_manifest
      msg = f"Row {idx} is not a dict"
      raise ValueError(msg)
    row_hash = row.get("manifest_row_hash")
    if not row_hash:
      msg = f"Row {idx} missing or empty manifest_row_hash"
      raise ValueError(msg)
    row_hash_str = str(row_hash)
    if row_hash_str in hashes_seen:
      msg = f"Duplicate manifest_row_hash: {row_hash_str!r}"
      raise ValueError(msg)
    hashes_seen.add(row_hash_str)
  for row in rows:
    row_hash = str(row.get("manifest_row_hash", "unknown"))
    checkpoint_id = row.get("checkpoint_id")
    # Guard against None and against "None" (the string produced when spec.checkpoint_id=None
    # is stored via str() serialization in a caller that doesn't validate upstream).
    if (
      checkpoint_id is None
      or not str(checkpoint_id).strip()
      or str(checkpoint_id).strip() == "None"
    ):
      msg = f"Row {row_hash} has empty or missing checkpoint_id"
      raise ValueError(msg)
    fixed_policy = row.get("fixed_policy")
    if fixed_policy is None or not str(fixed_policy).strip():
      msg = f"Row {row_hash} has empty or missing fixed_policy"
      raise ValueError(msg)
    state_weight_profile = row.get("state_weight_profile")
    if state_weight_profile is None or not str(state_weight_profile).strip():
      msg = f"Row {row_hash} has empty or missing state_weight_profile"
      raise ValueError(msg)
    # A row that names an arm must CARRY that arm's mask. Checks the row against itself --
    # unlike the deleted `required_fixed_policies` arm, which checked the planner's output
    # against the planner's own input and therefore could never fail.
    if str(fixed_policy).strip() != _NO_ARM:
      spec_payload = row.get("sampling_spec") or {}
      arm_mask = spec_payload.get("fixed_mask")
      if arm_mask is None or not np.any(np.asarray(arm_mask)):
        msg = (
          f"Row {row_hash} declares fixed_policy={fixed_policy!r} but its sampling_spec "
          f"carries no fixed_mask with any set position. The worker reconstructs its spec "
          f"from sampling_spec alone, so this row would design every residue while claiming "
          f"otherwise -- the exact failure that voided an 882-bead library. Use "
          f"fixed_policy={_NO_ARM!r} if nothing should be fixed."
        )
        raise ValueError(msg)


def write_manifest(
  manifest_path: str | Path,
  rows: list[dict[str, Any]],
  *,
  metadata: dict[str, Any] | None = None,
) -> Path:
  """Write campaign manifest JSON atomically with indented formatting.

  Uses atomic write (tmp file → replace → fsync) to ensure durability.
  Writes indented JSON (indent=2, sort_keys=True) for human readability.
  Returns resolved absolute path to manifest file.
  """
  path = Path(manifest_path).resolve()
  path.parent.mkdir(parents=True, exist_ok=True)
  payload: dict[str, Any] = {"schema_version": MANIFEST_SCHEMA_VERSION}
  if metadata is not None:
    payload["metadata"] = metadata
  payload["rows"] = rows
  raw = json.dumps(
    payload,
    indent=2,
    sort_keys=True,
    ensure_ascii=False,
    allow_nan=False,
  ).encode("utf-8")
  tmp_path = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
  tmp_path.write_bytes(raw)
  fsync_file(tmp_path)
  tmp_path.replace(path)
  fsync_directory(path.parent)
  return path


@dataclass
class ManifestGateState:
  """Gate-relevant status collected from a campaign manifest."""

  manifest_path: Path
  total_rows: int
  completed_rows: int
  digest_by_hash: dict[str, str]
  cell_to_hashes: dict[GateCellKey, set[str]]
  metadata_issues: list[str]
  checkpoint_issues: list[str]
  runtime_issues: list[str]


def _normalize_inputs(inputs: Any) -> list[str]:  # noqa: ANN401
  if isinstance(inputs, str):
    return [inputs]
  try:
    normalized = [str(item) for item in inputs]
  except TypeError as exc:  # pragma: no cover - defensive
    msg = "Campaign planner requires string-like inputs."
    raise ValueError(msg) from exc
  if not normalized:
    msg = "Campaign planner requires at least one input structure path."
    raise ValueError(msg)
  return normalized


def _row_output_path(base_dir: Path, campaign_id: str, row_hash: str) -> str:
  return str((base_dir / campaign_id / f"{row_hash}.h5").resolve())




def _lock_path(output_h5_path: Path) -> Path:
  return output_h5_path.with_name(f"{output_h5_path.name}.lock")


def _done_marker_path(output_h5_path: Path) -> Path:
  return output_h5_path.with_name(f"{output_h5_path.name}.done.json")


def _partial_output_path(output_h5_path: Path, attempt_id: str) -> Path:
  return output_h5_path.with_name(f"{output_h5_path.name}.partial.{attempt_id}")


def _write_lock_file_exclusive(path: Path, payload: dict[str, Any]) -> None:
  lock_bytes = canonical_json_bytes(payload)
  fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
  try:
    os.write(fd, lock_bytes)
    os.fsync(fd)
  finally:
    os.close(fd)
  fsync_directory(path.parent)


def _write_lock_file_atomic(path: Path, payload: dict[str, Any]) -> None:
  tmp_path = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
  tmp_path.write_bytes(canonical_json_bytes(payload))
  fsync_file(tmp_path)
  tmp_path.replace(path)
  fsync_directory(path.parent)


def _read_lock_file(path: Path) -> tuple[dict[str, Any], bytes]:
  raw = path.read_bytes()
  parsed = json.loads(raw.decode("utf-8"))
  if not isinstance(parsed, dict):
    msg = f"Invalid lock payload at {path}: expected object."
    raise TypeError(msg)
  return parsed, raw


def _acquire_local_fs_lock(
  *,
  lock_path: Path,
  owner_token: str,
  manifest_row_hash: str,
  attempt_id: str,
  lease_seconds: int,
) -> None:
  now = time.time()
  payload: dict[str, Any] = {
    "schema_version": LOCK_SCHEMA_VERSION,
    "owner_token": owner_token,
    "manifest_row_hash": manifest_row_hash,
    "attempt_id": attempt_id,
    "created_at_unix_s": now,
    "heartbeat_at_unix_s": now,
    "lease_expires_at_unix_s": now + lease_seconds,
  }
  try:
    _write_lock_file_exclusive(lock_path, payload)
  except FileExistsError as exc:
    existing, existing_raw = _read_lock_file(lock_path)
    expires_at = float(existing.get("lease_expires_at_unix_s", 0.0))
    if expires_at >= now:
      msg = (
        f"Active lock already held for output {lock_path}: "
        f"owner={existing.get('owner_token')!r}, expires_at={expires_at}."
      )
      raise RuntimeError(msg) from exc

    # Stale-lock recovery with compare-and-swap precondition:
    # only steal if the lock bytes are unchanged from the observed stale state.
    tmp_payload_path = lock_path.with_name(f"{lock_path.name}.steal.{attempt_id}.tmp")
    tmp_payload_path.write_bytes(canonical_json_bytes(payload))
    fsync_file(tmp_payload_path)
    current_raw = lock_path.read_bytes()
    if current_raw != existing_raw:
      tmp_payload_path.unlink(missing_ok=True)
      msg = f"Lock at {lock_path} changed during stale-lock recovery; aborting steal."
      raise RuntimeError(msg) from exc
    tmp_payload_path.replace(lock_path)
    fsync_directory(lock_path.parent)
    observed, _ = _read_lock_file(lock_path)
    if observed.get("owner_token") != owner_token:
      msg = f"Failed to claim stale lock at {lock_path}: ownership did not transfer."
      raise RuntimeError(msg) from exc


def _heartbeat_local_fs_lock(
  *,
  lock_path: Path,
  owner_token: str,
  lease_seconds: int,
) -> None:
  lock_payload, _ = _read_lock_file(lock_path)
  if lock_payload.get("owner_token") != owner_token:
    msg = (
      f"Cannot refresh lock {lock_path}: owner mismatch "
      f"(expected {owner_token!r}, observed {lock_payload.get('owner_token')!r})."
    )
    raise RuntimeError(msg)
  now = time.time()
  lock_payload["heartbeat_at_unix_s"] = now
  lock_payload["lease_expires_at_unix_s"] = now + lease_seconds
  _write_lock_file_atomic(lock_path, lock_payload)


def _release_local_fs_lock(*, lock_path: Path, owner_token: str) -> None:
  if not lock_path.exists():
    return
  lock_payload, _ = _read_lock_file(lock_path)
  if lock_payload.get("owner_token") != owner_token:
    msg = (
      f"Cannot release lock {lock_path}: owner mismatch "
      f"(expected {owner_token!r}, observed {lock_payload.get('owner_token')!r})."
    )
    raise RuntimeError(msg)
  lock_path.unlink()
  fsync_directory(lock_path.parent)


@contextmanager
def _campaign_lock_context(  # noqa: PLR0915
  *,
  lock_backend: str,
  distributed_lock_backend: DistributedLockBackend | None,
  lock_key_path: Path,
  owner_token: str,
  manifest_row_hash: str,
  attempt_id: str,
  lease_seconds: int,
  heartbeat_interval_seconds: int,
) -> Generator[list[Exception]]:
  heartbeat_errors: list[Exception] = []
  stop_event = threading.Event()

  if lock_backend == "local_fs":
    _acquire_local_fs_lock(
      lock_path=lock_key_path,
      owner_token=owner_token,
      manifest_row_hash=manifest_row_hash,
      attempt_id=attempt_id,
      lease_seconds=lease_seconds,
    )

    def _heartbeat() -> None:
      while not stop_event.wait(heartbeat_interval_seconds):
        try:
          _heartbeat_local_fs_lock(
            lock_path=lock_key_path,
            owner_token=owner_token,
            lease_seconds=lease_seconds,
          )
        except Exception as exc:  # noqa: BLE001
          heartbeat_errors.append(exc)
          return

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
      yield heartbeat_errors
    finally:
      stop_event.set()
      heartbeat_thread.join(timeout=heartbeat_interval_seconds + 1)
      try:
        _release_local_fs_lock(lock_path=lock_key_path, owner_token=owner_token)
      except Exception:
        if not heartbeat_errors:
          raise
        logger.exception("Failed to release local lock after heartbeat failure.")
    return

  if lock_backend != "distributed":
    msg = f"Unknown lock backend: {lock_backend!r}."
    raise ValueError(msg)
  if distributed_lock_backend is None:
    msg = (
      "lock_backend='distributed' requires a DistributedLockBackend implementation "
      "when calling run_manifest_row from Python."
    )
    raise ValueError(msg)
  lock_key = str(lock_key_path.resolve())
  distributed_lock_backend.acquire(
    lock_key=lock_key,
    owner_token=owner_token,
    lease_seconds=lease_seconds,
  )

  def _distributed_heartbeat() -> None:
    while not stop_event.wait(heartbeat_interval_seconds):
      try:
        distributed_lock_backend.heartbeat(
          lock_key=lock_key,
          owner_token=owner_token,
          lease_seconds=lease_seconds,
        )
      except Exception as exc:  # noqa: BLE001
        heartbeat_errors.append(exc)
        return

  heartbeat_thread = threading.Thread(target=_distributed_heartbeat, daemon=True)
  heartbeat_thread.start()
  try:
    yield heartbeat_errors
  finally:
    stop_event.set()
    heartbeat_thread.join(timeout=heartbeat_interval_seconds + 1)
    try:
      distributed_lock_backend.release(lock_key=lock_key, owner_token=owner_token)
    except Exception:
      if not heartbeat_errors:
        raise
      logger.exception("Failed to release distributed lock after heartbeat failure.")


def _read_done_marker(path: Path) -> dict[str, Any] | None:
  if not path.exists():
    return None
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    msg = f"Done marker at {path} must be a JSON object."
    raise TypeError(msg)
  return payload


class StaleDoneMarkerSchemaError(ValueError):
  """A done marker predates DONE_MARKER_SCHEMA_VERSION (e.g. pre-Zarr-migration HDF5).

  Distinct from the other _validate_done_marker failures (hash/digest mismatch),
  which indicate real corruption and must stay hard errors: a schema-version
  mismatch means the row is safe to invalidate and recompute under the current
  storage format.
  """


def _invalidate_stale_output(*, marker_path: Path, output_h5_path: Path) -> None:
  """Discard a done marker and its output artifact written under a superseded schema."""
  if output_h5_path.is_dir():
    shutil.rmtree(output_h5_path)
  elif output_h5_path.exists():
    output_h5_path.unlink()
  marker_path.unlink(missing_ok=True)


def _validate_done_marker(
  *,
  marker: dict[str, Any],
  marker_path: Path,
  output_h5_path: Path,
  manifest_row_hash: str,
) -> None:
  if marker.get("schema_version") != DONE_MARKER_SCHEMA_VERSION:
    msg = (
      f"Done marker schema mismatch at {marker_path}: "
      f"expected {DONE_MARKER_SCHEMA_VERSION!r}, got {marker.get('schema_version')!r}."
    )
    raise StaleDoneMarkerSchemaError(msg)
  if marker.get("manifest_row_hash") != manifest_row_hash:
    msg = (
      f"Done marker manifest hash mismatch at {marker_path}: "
      f"expected {manifest_row_hash!r}, got {marker.get('manifest_row_hash')!r}."
    )
    raise ValueError(msg)
  if not output_h5_path.exists():
    msg = f"Done marker exists at {marker_path} but output store is missing: {output_h5_path}."
    raise ValueError(msg)
  observed_content_digest = zarr_content_digest(output_h5_path)
  expected_content_digest = marker.get("content_digest_sha256")
  if observed_content_digest != expected_content_digest:
    msg = (
      f"Done marker content digest mismatch at {marker_path}: "
      f"expected {expected_content_digest!r}, observed {observed_content_digest!r}."
    )
    raise ValueError(msg)


def _write_done_marker(
  *,
  marker_path: Path,
  output_h5_path: Path,
  manifest_row_hash: str,
  attempt_id: str,
  content_digest_sha256: str,
  lock_backend: str,
) -> None:
  marker_payload = {
    "schema_version": DONE_MARKER_SCHEMA_VERSION,
    "manifest_row_hash": manifest_row_hash,
    "attempt_id": attempt_id,
    "output_h5_path": str(output_h5_path.resolve()),
    "content_digest_sha256": content_digest_sha256,
    "lock_backend": lock_backend,
    "completed_at_unix_s": time.time(),
  }
  tmp_marker_path = marker_path.with_name(f"{marker_path.name}.tmp.{attempt_id}")
  tmp_marker_path.write_bytes(canonical_json_bytes(marker_payload))
  fsync_file(tmp_marker_path)
  tmp_marker_path.replace(marker_path)
  fsync_directory(marker_path.parent)


# The label for "nothing is fixed". A first-class arm, not an absence -- so the row's
# fixed_policy field stays non-empty and "design everything" is something you can name, grid
# on, and see in a manifest diff.
_NO_ARM = "none"


# `H38|D73|C143` -- an IDENTITY letter and a canonical (0-based) index, per position.
_ARM_RESIDUE_RE = re.compile(r"^([A-Z])(\d+)$")


def _parse_arm_declaration(label: str, text: str, *, length: int) -> tuple[np.ndarray, np.ndarray]:
  """Parse ``H38|D73|C143`` into a (mask, tokens) pair.

  **The letter is the point.** A bare position list would say *where* to freeze but not *what*
  to, and the two are one decision -- freezing without stating the identity locks every frozen
  position to token 0 (Alanine), which is why `_prepare_fixed_controls` refuses the pair. The
  letters supply `fixed_tokens` directly, so nothing has to infer them: no structure parsing
  at plan time, and no alphabet to get wrong.

  It also makes the deliberate overrides *visible*. TEV's 1LVB is a C151A mutant -- the
  crystal carries Ala where the catalysis needs Cys -- and `C143` says "hold Cys here" in the
  declaration itself, where a reader sees it, rather than in a comment somewhere else.

  And it is self-checking in a way a position list can never be. `CATALYTIC_PDB_IDS =
  [135,181,183]` shipped in a consumer for weeks as "the catalytic triad"; those indices hold
  **Ser/Ser/Pro**. Nothing about a bare `[127,173,175]` could have flagged that. `S127|S173|P175`
  would have looked wrong to anyone who knows what a catalytic triad is -- and a worker holding
  the structure can check the claim outright.

  Pipe-separated, never comma: `_parse_csv` splits on comma and would shatter this into
  separate bogus arms.

  Args:
    label: The arm's label, for error messages.
    text: The declaration, e.g. ``"H38|D73|C143"``.
    length: Mask length -- ``base_spec.max_length``, since the dataset op pads to it.

  Returns:
    ``(mask, tokens)``: a 1-D ``float32`` mask and a 1-D ``int32`` MPNN token array.

  Raises:
    ValueError: malformed entry, unknown residue letter, or an out-of-range index.

  """
  from aminx.utils.aa_convert import MPNN_ALPHABET  # noqa: PLC0415

  mask = np.zeros(length, dtype=np.float32)
  tokens = np.zeros(length, dtype=np.int32)
  for entry in (e.strip() for e in text.split("|") if e.strip()):
    match = _ARM_RESIDUE_RE.match(entry)
    if match is None:
      msg = (
        f"fixed_arms[{label!r}]: {entry!r} is not a RESIDUE+POSITION like 'H38'. Declare the "
        f"identity, not just the position -- freezing a position without saying what to hold "
        f"it at locks it to Alanine. Example: catalytic_triad=H38|D73|C143"
      )
      raise ValueError(msg)
    letter, index = match.group(1), int(match.group(2))
    if letter not in MPNN_ALPHABET:
      msg = f"fixed_arms[{label!r}]: {letter!r} in {entry!r} is not an amino acid letter."
      raise ValueError(msg)
    if not 0 <= index < length:
      msg = (
        f"fixed_arms[{label!r}]: position {index} in {entry!r} is outside [0, {length}). "
        f"Positions are 0-based CANONICAL indices (the reference state's own numbering), not "
        f"author/PDB numbering -- for TEV the catalytic triad His46/Asp81/Cys151 is "
        f"[38, 73, 143], i.e. author number minus 8."
      )
      raise ValueError(msg)
    mask[index] = 1.0
    tokens[index] = MPNN_ALPHABET.index(letter)
  if not np.any(mask):
    msg = f"fixed_arms[{label!r}]: {text!r} declares no positions."
    raise ValueError(msg)
  return mask, tokens


def _resolve_arm(
  label: str,
  source: Any,  # noqa: ANN401 -- str declaration | ArrayLike | PathLike | None
  *,
  length: int,
) -> tuple[Any, Any]:
  """Resolve an arm SOURCE into ``(mask, tokens)``. Tokens are None unless declared.

  Three shapes, in the order a caller is likely to reach for them:
    - ``"H38|D73|C143"`` -- the inline declaration. No file needed for the common case, which
      is the whole point: materialising a `.npy` to say "hold three residues" is friction.
    - a path to a ``.npy`` -- the escape hatch for large sets (an 84-residue active-site shell
      is not worth typing). Carries a mask only, so `fixed_tokens` must come from elsewhere.
    - an array -- library callers who already have one.

  Raises:
    ValueError: the mask is not 1-D. Per-state `(S, L)` masks are refused HERE, at the
      declaration, rather than at `multistate_poe.py`'s much later guard -- same verdict, but
      the error arrives while the caller still has the context to fix it.

  """
  if source is None:
    return None, None

  if isinstance(source, str) and not source.endswith(".npy"):
    return _parse_arm_declaration(label, source, length=length)

  mask = np.load(Path(source)) if isinstance(source, (str, Path)) else source
  mask = np.asarray(mask)
  if mask.ndim != 1:
    msg = (
      f"fixed_arms[{label!r}] has shape {mask.shape}; a fixed_mask must be 1-D (L,) in the "
      f"canonical reference frame. Per-state (S, L) masks are not supported: one sequence is "
      f"sampled, so a position is either designed or it is not (decode broadcasts the mask "
      f"over the group axis, and sampling/multistate_poe.py refuses a per-state mask "
      f"outright). Cross-state index shifts are what state_position_map resolves -- set that "
      f"instead."
    )
    raise ValueError(msg)
  return mask, None


# The one profile label aminx can resolve without being told what it means: "no weighting".
# It resolves to ``state_weights=None`` -- the exact value every campaign row carried before
# profiles resolved to anything -- so the default ``("equal",)`` changes no sampled output.
_EQUAL_PROFILE = "equal"

_WEIGHT_SPLIT_RE = re.compile(r"[|,]")


def _parse_weight_declaration(label: str, text: str) -> np.ndarray:
  """Parse ``0.7|0.3`` (or ``0.7,0.3``) into a 1-D float32 weight vector."""
  parts = [part.strip() for part in _WEIGHT_SPLIT_RE.split(text) if part.strip()]
  if not parts:
    msg = f"state_weight_profiles[{label!r}]: {text!r} declares no weights."
    raise ValueError(msg)
  try:
    return np.asarray([float(part) for part in parts], dtype=np.float32)
  except ValueError as exc:
    msg = (
      f"state_weight_profiles[{label!r}]: {text!r} is not a list of numbers like '0.7|0.3' "
      f"(one weight per state, pipe- or comma-separated) or a path to a 1-D .npy."
    )
    raise ValueError(msg) from exc


def _resolve_state_weight_profile(label: str, source: Any) -> np.ndarray | None:  # noqa: ANN401
  """Resolve one profile SOURCE into a weight vector, or None for the ``"equal"`` profile.

  Shapes, mirroring ``_resolve_arm``: ``None``/``"equal"`` (no weighting), an inline
  ``"0.7|0.3"``, a path to a 1-D ``.npy``, or an array/sequence of numbers.

  Raises:
    ValueError: not 1-D, empty, non-finite, negative, or summing to zero. The vector's length
      cannot be checked against the number of states here (the planner parses no structure);
      ``inference/logits.py`` refuses a mismatch when the row runs.

  """
  if source is None or (isinstance(source, str) and source.strip() == _EQUAL_PROFILE):
    return None
  if isinstance(source, str) and not source.endswith(".npy"):
    weights = _parse_weight_declaration(label, source)
  elif isinstance(source, (str, Path)):
    weights = np.asarray(np.load(Path(source)), dtype=np.float32)
  else:
    weights = np.asarray(source, dtype=np.float32)
  if weights.ndim != 1 or weights.size == 0:
    msg = (
      f"state_weight_profiles[{label!r}] has shape {weights.shape}; state weights must be a "
      f"non-empty 1-D vector with one entry per state."
    )
    raise ValueError(msg)
  if not np.all(np.isfinite(weights)) or np.any(weights < 0) or float(weights.sum()) <= 0.0:
    msg = (
      f"state_weight_profiles[{label!r}] = {weights.tolist()} must be finite, non-negative and "
      f"not all zero."
    )
    raise ValueError(msg)
  return weights


def _resolve_state_weight_profiles(
  profiles: Mapping[str, Any] | Sequence[str] | str,
) -> dict[str, np.ndarray | None]:
  """Resolve the ``state_weight_profiles`` declaration into ``{label: weights | None}``.

  **The declaration must carry its own referent** -- the same fix ``fixed_arms`` got for the
  same bug. ``state_weight_profiles`` used to be ``tuple[str, ...]`` of bare names: aminx
  cannot know what ``"weighted"`` means, so the name was hashed into the row, written to the
  row, checked for non-emptiness, and never reached ``SamplingSpecification.state_weights``.
  ``--state-weight-profiles equal,weighted`` produced two row-sets, differently labelled and
  identically weighted: duplicate work counted as an ablation.

  A ``Mapping`` ``label -> weights`` is the honest form. A bare sequence of names is still
  accepted for back-compat, but only ``"equal"`` is resolvable; any other bare name RAISES
  instead of becoming a label with no meaning.

  Raises:
    ValueError: empty declaration, empty label, an unresolvable bare name, ``"equal"`` mapped
      to weights, an invalid vector, or two labels that resolve to the same vector (a second
      row-set that cannot differ from the first is not a second profile).

  """
  if isinstance(profiles, str):
    profiles = (profiles,)
  if not profiles:
    msg = "state_weight_profiles is empty; pass ('equal',) for no weighting (the default)."
    raise ValueError(msg)

  declared: Mapping[str, Any]
  if isinstance(profiles, Mapping):
    declared = profiles
  else:
    declared = {}
    for name in profiles:
      if str(name).strip() != _EQUAL_PROFILE:
        msg = (
          f"state_weight_profiles: {name!r} is a bare name aminx cannot resolve -- it would be "
          f"written to the manifest as a label and never reach state_weights, so the row-set "
          f"would be weighted exactly like 'equal'. Declare the weights: "
          f"--state-weight-profile {name}=0.7|0.3 (CLI) or "
          f"state_weight_profiles={{{name!r}: [0.7, 0.3]}} (API)."
        )
        raise ValueError(msg)
      if str(name).strip() in declared:
        msg = f"state_weight_profiles: {str(name).strip()!r} listed twice; labels must be unique."
        raise ValueError(msg)
      declared[str(name).strip()] = None

  resolved: dict[str, np.ndarray | None] = {}
  for raw_label, source in declared.items():
    label = str(raw_label).strip()
    if not label:
      msg = "state_weight_profiles has an empty label."
      raise ValueError(msg)
    weights = _resolve_state_weight_profile(label, source)
    if label == _EQUAL_PROFILE and weights is not None:
      msg = (
        f"state_weight_profiles[{_EQUAL_PROFILE!r}] is reserved for 'no weighting' but was "
        f"given weights {weights.tolist()}. Use another label."
      )
      raise ValueError(msg)
    resolved[label] = weights

  seen: dict[tuple[float, ...] | None, str] = {}
  for label, weights in resolved.items():
    key = None if weights is None else tuple(float(w) for w in weights)
    if key in seen:
      msg = (
        f"state_weight_profiles: {label!r} and {seen[key]!r} resolve to the same weights; the "
        f"second row-set would duplicate the first and be counted as diversity."
      )
      raise ValueError(msg)
    seen[key] = label
  return resolved


def plan_campaign_manifest(
  *,
  base_spec: SamplingSpecification,
  campaign_id: str,
  designs_per_library_type: int,
  samples_chunk_size: int,
  output_root: str | Path,
  fixed_arms: Mapping[str, Any] | None = None,
  state_weight_profiles: Mapping[str, Any] | Sequence[str] = ("equal",),
  planner_version: str = "planner_v1",
  dataset_fingerprint: str = "unknown",
  environment_image: str = "unknown",
  git_sha: str = "unknown",
  config_hash: str = "unknown",
) -> list[dict[str, Any]]:
  """Plan campaign rows for all library/fixed-arm/profile combinations.

  Args:
    fixed_arms: Maps an arm LABEL to the **1-D canonical fixed_mask** it means, or to a path
      to a ``.npy`` holding one. ``None`` (the default) means fix nothing -- every residue
      designable -- which is the only default aminx can honestly ship, since it cannot know
      any particular protein's catalytic triad. Each arm becomes its own row-set.

      **The arm carries its own referent, and that is the whole point.** This replaces
      ``fixed_policies: tuple[str, ...]``, which took bare NAMES like ``"catalytic_triad"``
      that aminx had no way to resolve -- positions are protein-specific and live in the
      caller's own bundle. So the name was written into the manifest row as a label and
      never into the spec, the worker (which reads only ``row["sampling_spec"]``) never saw
      it, and **no residue was ever held fixed in any produced design**: 882/882 beads of a
      real library, silently violating a locked preregistration. A resolver callback is not
      an alternative -- callables cannot cross the manifest's JSON boundary, and
      ``spec_partition._assert_no_callable_knobs`` refuses them.

      **Masks are 1-D `(L,)` in the canonical reference frame, never per-state `(S, L)`.**
      ``types/bundles.py`` types ``fixed_mask`` as ``Float[Array, L]`` while
      ``state_position_map`` beside it is ``"S L"``; ``sampling/multistate_poe.py`` RAISES on
      a genuinely per-state mask; decode broadcasts the mask over the *group* axis because
      one sequence is sampled and a position is either designed or it is not. Cross-state
      index shifts are exactly what ``state_position_map`` exists to resolve.

    state_weight_profiles: Maps a profile LABEL to the weight vector it means -- ``0.7|0.3``,
      a sequence of numbers, or a path to a 1-D ``.npy`` -- or, for back-compat, a sequence of
      names in which only ``"equal"`` (``state_weights=None``, the default) is resolvable. Like
      ``fixed_arms``, the profile carries its own referent: the vector lands on each row's
      ``sampling_spec["state_weights"]``. ``base_spec.state_weights`` is kept for the
      ``"equal"`` profile (the pre-existing route), so a caller who set it directly is
      untouched. Two different profiles yield rows with different ``state_weights``.

  Raises:
    ValueError: a profile cannot be resolved (see ``_resolve_state_weight_profiles``);
      ``fixed_arms`` is an empty mapping (states "here are my arms", then supplies
      none -- a contradiction, and silently reading it as "design everything" is how the old
      zero-rows bug read), or a mask is not 1-D.

  """
  if designs_per_library_type <= 0:
    msg = "designs_per_library_type must be positive."
    raise ValueError(msg)
  if samples_chunk_size <= 0:
    msg = "samples_chunk_size must be positive."
    raise ValueError(msg)

  # `{}` is a contradiction, not a default. Distinct from `None` ("I did not ask for fixing"),
  # which yields the no-arm row-set below. The predecessor silently returned ZERO rows for an
  # empty policy set: the policy loop was outermost, so the body never ran, and
  # validate_manifest_rows then skipped every check because its own guard is
  # `if required_fixed_policies:`. An empty manifest is not a plan.
  if fixed_arms is not None and not fixed_arms:
    msg = (
      "fixed_arms is an empty mapping, which asks for arms and then supplies none. Pass "
      "fixed_arms=None to design every residue (the default), or supply at least one "
      "label -> 1-D canonical fixed_mask."
    )
    raise ValueError(msg)

  # The no-arm sentinel. `_NO_ARM` keeps the row's fixed_policy field non-empty so
  # validate_manifest_rows' existing emptiness check still means something, and keeps
  # "nothing fixed" a first-class, nameable arm rather than an absence.
  arms: Mapping[str, Any] = fixed_arms if fixed_arms is not None else {_NO_ARM: None}
  # Length comes from base_spec.max_length: the dataset op pads to it (prep.py:117), and
  # _prepare_fixed_controls sizes the mask against the PARSED structure. The planner parses
  # nothing, so this is the only length it can honestly use -- and the campaign CLIs expose
  # no --max-length, so it is the one every campaign actually runs.
  arm_length = base_spec.max_length or 512
  resolved_arms = {
    label: _resolve_arm(label, src, length=arm_length) for label, src in arms.items()
  }

  resolved_profiles = _resolve_state_weight_profiles(state_weight_profiles)

  output_root_path = Path(output_root)
  rows: list[dict[str, Any]] = []
  job_index = 0

  for fixed_policy, (arm_mask, arm_tokens) in resolved_arms.items():
    for state_weight_profile, profile_weights in resolved_profiles.items():
      for ligand_on in (False, True):
        for sidechain_on in (False, True):
          spec_variant = replace(
            base_spec,
            inputs=_normalize_inputs(base_spec.inputs),
            campaign_mode=True,
            return_logits=False,
            ligand_conditioning=ligand_on,
            sidechain_conditioning=sidechain_on,
            # THE LINE WHOSE ABSENCE WAS THE ENTIRE BUG.
            #
            # The arm's mask has to land on the SPEC, because `sampling_spec` is the only
            # thing the worker reconstructs from (run_manifest_row). Previously the loop
            # variable reached `build_manifest_row(..., fixed_policy=...)` -- the row's
            # LABEL -- and stopped there, so `campaign_sampling_spec_payload(spec_variant)`
            # below could not see it, and the model was never told to hold anything. Every
            # downstream check then inspected the label and was satisfied.
            #
            # `base_spec.fixed_mask` wins when no arm supplies one, so a caller who sets the
            # mask directly (the pre-existing route, and the one tev_design uses today via
            # its manifest patch) keeps working untouched.
            fixed_mask=arm_mask if arm_mask is not None else base_spec.fixed_mask,
            # The declaration's letters ARE the tokens -- H38 says "hold His here". Nothing
            # has to infer them, so there is no alphabet to get wrong and no structure to
            # parse at plan time. Falls back to the caller's own tokens for the .npy/array
            # shapes, which carry a mask only.
            fixed_tokens=arm_tokens if arm_tokens is not None else base_spec.fixed_tokens,
            # The profile's vector has to land on the SPEC for the same reason the arm's mask
            # does: `sampling_spec` is all the worker reads. The loop variable used to reach
            # only the row's LABEL (and its hash), so every profile sampled identically.
            # `base_spec.state_weights` wins for the "equal" profile, so a caller who sets it
            # directly keeps working untouched.
            state_weights=(
              profile_weights if profile_weights is not None else base_spec.state_weights
            ),
          )
          for chunk_index, sample_start in enumerate(
            range(0, designs_per_library_type, samples_chunk_size),
          ):
            sample_count = min(samples_chunk_size, designs_per_library_type - sample_start)
            row = build_manifest_row(
              spec_variant,
              campaign_id=campaign_id,
              job_index=job_index,
              chunk_id=chunk_index,
              sample_start=sample_start,
              sample_count=sample_count,
              fixed_policy=fixed_policy,
              state_weight_profile=state_weight_profile,
              planner_version=planner_version,
              dataset_fingerprint=dataset_fingerprint,
              environment_image=environment_image,
              git_sha=git_sha,
              config_hash=config_hash,
              job_id=f"{campaign_id}-job-{job_index}",
              declared_state_weights=profile_weights,
            )
            row["output_h5_path"] = _row_output_path(
              output_root_path,
              campaign_id=campaign_id,
              row_hash=row["manifest_row_hash"],
            )
            # Built from dataclasses.fields(), NOT hand-listed. The hand-written 23-key literal
            # this replaces drifted from an 88-field dataclass with nothing enforcing the sync,
            # which is the single root cause of all twelve silently-dropped knobs -- including
            # fixed_mask, i.e. no residue was ever actually held fixed in any produced design.
            # spec_partition raises at import if any field is classified nowhere.
            row["sampling_spec"] = campaign_sampling_spec_payload(
              spec_variant,
              campaign_owned={
                "grid_mode": True,
                # The planner owns chunking; the caller's samples_chunk_size is replaced, not
                # dropped -- this row covers exactly sample_count designs.
                "samples_chunk_size": sample_count,
                "job_id": row["job_id"],
                "chunk_id": row["chunk_id"],
                "sample_start": row["sample_start"],
                "sample_count": row["sample_count"],
                "output_h5_path": row["output_h5_path"],
              },
            )
            rows.append(row)
          job_index += 1

  validate_manifest_rows(rows)
  return rows


def write_campaign_manifest(
  *,
  base_spec: SamplingSpecification,
  campaign_id: str,
  manifest_path: str | Path,
  designs_per_library_type: int,
  samples_chunk_size: int,
  output_root: str | Path,
  fixed_arms: Mapping[str, Any] | None = None,
  state_weight_profiles: Mapping[str, Any] | Sequence[str] = ("equal",),
  planner_version: str = "planner_v1",
  dataset_fingerprint: str = "unknown",
  environment_image: str = "unknown",
  git_sha: str = "unknown",
  config_hash: str = "unknown",
) -> Path:
  """Plan rows and write campaign manifest JSON. See `plan_campaign_manifest` for fixed_arms."""
  rows = plan_campaign_manifest(
    base_spec=base_spec,
    campaign_id=campaign_id,
    designs_per_library_type=designs_per_library_type,
    samples_chunk_size=samples_chunk_size,
    output_root=output_root,
    fixed_arms=fixed_arms,
    state_weight_profiles=state_weight_profiles,
    planner_version=planner_version,
    dataset_fingerprint=dataset_fingerprint,
    environment_image=environment_image,
    git_sha=git_sha,
    config_hash=config_hash,
  )
  metadata = {
    "campaign_id": campaign_id,
    "planner_version": planner_version,
    "designs_per_library_type": designs_per_library_type,
    "samples_chunk_size": samples_chunk_size,
  }
  return write_manifest(manifest_path, rows, metadata=metadata)


def _select_row(
  rows: list[dict[str, Any]],
  *,
  row_index: int | None = None,
  row_hash: str | None = None,
) -> dict[str, Any]:
  if row_hash is not None:
    for row in rows:
      if row.get("manifest_row_hash") == row_hash:
        return row
    msg = f"Manifest row hash not found: {row_hash}"
    raise ValueError(msg)

  resolved_index = 0 if row_index is None else row_index
  if resolved_index < 0 or resolved_index >= len(rows):
    msg = f"Manifest row index out of range: {resolved_index}"
    raise IndexError(msg)
  return rows[resolved_index]


def run_manifest_row(  # noqa: PLR0915
  manifest_path: str | Path,
  *,
  row_index: int | None = None,
  row_hash: str | None = None,
  lock_backend: str = "local_fs",
  distributed_lock_backend: DistributedLockBackend | None = None,
  lock_lease_seconds: int = DEFAULT_LOCK_LEASE_SECONDS,
  heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> dict[str, Any]:
  """Execute one manifest row through the standard sampling runtime."""
  payload = load_manifest(manifest_path)
  rows = payload["rows"]
  if not isinstance(rows, list):  # pragma: no cover - guarded by load_manifest
    msg = "Manifest rows payload must be a list."
    raise TypeError(msg)

  row = _select_row(rows, row_index=row_index, row_hash=row_hash)
  sampling_spec_payload = row.get("sampling_spec")
  if not isinstance(sampling_spec_payload, dict):
    msg = "Manifest row must include a 'sampling_spec' object."
    raise TypeError(msg)

  raw_output_path = sampling_spec_payload.get("output_h5_path")
  if raw_output_path is None:
    msg = "Manifest row sampling_spec must include output_h5_path."
    raise ValueError(msg)
  output_h5_path = Path(raw_output_path).resolve()
  output_h5_path.parent.mkdir(parents=True, exist_ok=True)
  lock_path = _lock_path(output_h5_path)
  done_marker_path = _done_marker_path(output_h5_path)
  manifest_hash = str(row["manifest_row_hash"])

  existing_marker = _read_done_marker(done_marker_path)
  marker_is_stale = False
  if existing_marker is not None:
    try:
      _validate_done_marker(
        marker=existing_marker,
        marker_path=done_marker_path,
        output_h5_path=output_h5_path,
        manifest_row_hash=manifest_hash,
      )
    except StaleDoneMarkerSchemaError:
      marker_is_stale = True
    else:
      return {
        "status": "already_done",
        "output_h5_path": str(output_h5_path),
        "manifest_row_hash": manifest_hash,
        "done_marker_path": str(done_marker_path),
        "attempt_id": existing_marker.get("attempt_id"),
      }
  if not marker_is_stale and output_h5_path.exists():
    msg = (
      f"Output file already exists without done marker for manifest row {manifest_hash}: "
      f"{output_h5_path}"
    )
    raise RuntimeError(msg)

  attempt_id = uuid.uuid4().hex
  owner_token = f"{manifest_hash}:{attempt_id}"
  partial_path = _partial_output_path(output_h5_path, attempt_id)
  with _campaign_lock_context(
    lock_backend=lock_backend,
    distributed_lock_backend=distributed_lock_backend,
    lock_key_path=lock_path,
    owner_token=owner_token,
    manifest_row_hash=manifest_hash,
    attempt_id=attempt_id,
    lease_seconds=lock_lease_seconds,
    heartbeat_interval_seconds=heartbeat_interval_seconds,
  ) as heartbeat_errors:
    existing_marker = _read_done_marker(done_marker_path)
    if existing_marker is not None:
      try:
        _validate_done_marker(
          marker=existing_marker,
          marker_path=done_marker_path,
          output_h5_path=output_h5_path,
          manifest_row_hash=manifest_hash,
        )
      except StaleDoneMarkerSchemaError as exc:
        logger.warning(
          "Discarding stale done marker for manifest row %s, recomputing: %s",
          manifest_hash,
          exc,
        )
        _invalidate_stale_output(marker_path=done_marker_path, output_h5_path=output_h5_path)
      else:
        return {
          "status": "already_done",
          "output_h5_path": str(output_h5_path),
          "manifest_row_hash": manifest_hash,
          "done_marker_path": str(done_marker_path),
          "attempt_id": existing_marker.get("attempt_id"),
        }
    if output_h5_path.exists():
      msg = (
        f"Output file already exists without done marker for manifest row {manifest_hash}: "
        f"{output_h5_path}"
      )
      raise RuntimeError(msg)

    worker_payload = dict(sampling_spec_payload)
    worker_payload["output_h5_path"] = str(partial_path)
    pop_deprecated_spec_kwargs(worker_payload)
    # Coerce JSON scalars back to their spec types (lists -> ndarray for the array knobs,
    # list -> tuple for temperature) using spec_json's whitelist, which names exactly these
    # fields and exists for exactly this. Before the field-driven manifest write, array knobs
    # were dropped entirely and no worker ever saw them; now they arrive, and they arrive as
    # plain lists. Most consumers wrap defensively (_sample_batch does
    # jnp.asarray(spec...bias)), but relying on EVERY consumer to be careful is the posture
    # this audit exists to end -- restore the type at the boundary instead.
    #
    # Coercion is a VALUE transform, so strict unknown-key rejection is unaffected: an unknown
    # key still reaches the constructor and raises TypeError. That strictness is why this path
    # uses the plain constructor rather than run_specification_from_json_dict, which would
    # silently ignore unknown keys.
    worker_payload = {
      key: _coerce_field_value(SamplingSpecification, key, value)
      for key, value in worker_payload.items()
    }
    sampling_spec = SamplingSpecification(**worker_payload)
    # Genuine multi-state PoE rows (len(inputs) > 1) route to
    # sample_multistate_poe_campaign_row instead of sample() -- sample()'s real dispatcher
    # (_sample_batch) treats every --inputs path as an independent single-state structure and
    # never actually fuses states (praxia debt #572 follow-up, decisions/
    # 260713_no-real-multistate-sampling-path-exists.md). Single-structure ("spike-in") rows
    # are unaffected -- len(inputs) == 1 for those, so they keep going through sample() exactly
    # as before.
    is_multistate_poe_row = (
      isinstance(sampling_spec.inputs, (list, tuple)) and len(sampling_spec.inputs) > 1
    )
    try:
      sample_result = (
        sample_multistate_poe_campaign_row(sampling_spec)
        if is_multistate_poe_row
        else sample(sampling_spec)
      )
      if heartbeat_errors:
        msg = (
          f"Lock heartbeat failed while executing manifest row {manifest_hash}: "
          f"{heartbeat_errors[0]}"
        )
        raise RuntimeError(msg) from heartbeat_errors[0]
      if not partial_path.exists():
        msg = f"Worker did not produce expected partial output store: {partial_path}"
        raise RuntimeError(msg)
      fsync_tree(partial_path)
      content_digest_sha256 = zarr_content_digest(partial_path)
      partial_path.replace(output_h5_path)
      fsync_directory(output_h5_path.parent)
      _write_done_marker(
        marker_path=done_marker_path,
        output_h5_path=output_h5_path,
        manifest_row_hash=manifest_hash,
        attempt_id=attempt_id,
        content_digest_sha256=content_digest_sha256,
        lock_backend=lock_backend,
      )
    finally:
      if partial_path.exists():
        shutil.rmtree(partial_path)

  result_payload = (
    dict(sample_result) if isinstance(sample_result, dict) else {"sample_result": sample_result}
  )
  result_payload.update(
    {
      "status": str(result_payload.get("status", "completed")),
      "output_h5_path": str(output_h5_path),
      "manifest_row_hash": manifest_hash,
      "attempt_id": attempt_id,
      "done_marker_path": str(done_marker_path),
      "lock_backend": lock_backend,
    },
  )
  return result_payload


def _manifest_rows(manifest_path: str | Path) -> list[dict[str, Any]]:
  payload = load_manifest(manifest_path)
  rows = payload["rows"]
  if not isinstance(rows, list):  # pragma: no cover - guarded by load_manifest
    msg = "Manifest rows payload must be a list."
    raise TypeError(msg)
  normalized_rows: list[dict[str, Any]] = []
  for row in rows:
    if not isinstance(row, dict):  # pragma: no cover - guarded by load_manifest
      msg = "Manifest rows must be JSON objects."
      raise TypeError(msg)
    normalized_rows.append(row)
  return normalized_rows


def execute_manifest(
  manifest_path: str | Path,
  *,
  row_hashes: Sequence[str] | None = None,
  continue_on_error: bool = False,
  lock_backend: str = "local_fs",
  distributed_lock_backend: DistributedLockBackend | None = None,
  lock_lease_seconds: int = DEFAULT_LOCK_LEASE_SECONDS,
  heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> dict[str, Any]:
  """Execute all (or selected) manifest rows and return a structured summary."""
  rows = _manifest_rows(manifest_path)
  requested_hashes = tuple(row_hashes or ())
  requested_hash_set = set(requested_hashes)
  available_hashes = {str(row["manifest_row_hash"]) for row in rows}
  missing_hashes = sorted(requested_hash_set - available_hashes)
  if missing_hashes:
    msg = f"Requested manifest row hashes not found: {missing_hashes}"
    raise ValueError(msg)

  selected_rows = (
    [row for row in rows if str(row["manifest_row_hash"]) in requested_hash_set]
    if requested_hashes
    else rows
  )
  success_statuses = {"ok", "completed", "already_done"}
  row_results: list[dict[str, Any]] = []
  successful_rows = 0
  failed_rows = 0

  for row in selected_rows:
    row_hash = str(row["manifest_row_hash"])
    try:
      result = run_manifest_row(
        manifest_path,
        row_hash=row_hash,
        lock_backend=lock_backend,
        distributed_lock_backend=distributed_lock_backend,
        lock_lease_seconds=lock_lease_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
      )
      row_status = str(result.get("status", "completed"))
      row_results.append(
        {
          "manifest_row_hash": row_hash,
          "status": row_status,
          "output_h5_path": result.get("output_h5_path"),
        },
      )
      if row_status in success_statuses:
        successful_rows += 1
      else:
        failed_rows += 1
    except Exception as exc:
      failed_rows += 1
      row_results.append(
        {
          "manifest_row_hash": row_hash,
          "status": "error",
          "error": str(exc),
        },
      )
      if not continue_on_error:
        msg = f"Execution failed for manifest row {row_hash}: {exc}"
        raise RuntimeError(msg) from exc

  return {
    "schema_version": "campaign_execute_report_v1",
    "manifest_path": str(Path(manifest_path).resolve()),
    "selected_rows": len(selected_rows),
    "successful_rows": successful_rows,
    "failed_rows": failed_rows,
    "row_results": row_results,
  }


def plan_scale_ramp(
  *,
  base_spec: SamplingSpecification,
  campaign_id: str,
  manifest_dir: str | Path,
  output_root: str | Path,
  stage_designs_per_library_type: Sequence[int],
  samples_chunk_size: int,
  fixed_arms: Mapping[str, Any] | None = None,
  state_weight_profiles: Mapping[str, Any] | Sequence[str] = ("equal",),
  planner_version: str = "planner_v1",
  dataset_fingerprint: str = "unknown",
  environment_image: str = "unknown",
  git_sha: str = "unknown",
  config_hash: str = "unknown",
) -> dict[str, Any]:
  """Create staged manifests for pilot-to-scale rollout."""
  if not stage_designs_per_library_type:
    msg = "stage_designs_per_library_type must contain at least one stage."
    raise ValueError(msg)
  stage_sizes = tuple(int(size) for size in stage_designs_per_library_type)
  if any(size <= 0 for size in stage_sizes):
    msg = "All stage designs_per_library_type values must be positive."
    raise ValueError(msg)
  if any(current <= previous for previous, current in pairwise(stage_sizes)):
    msg = "Stage sizes must be strictly increasing for ramp planning."
    raise ValueError(msg)

  manifest_dir_path = Path(manifest_dir).resolve()
  output_root_path = Path(output_root).resolve()
  manifest_dir_path.mkdir(parents=True, exist_ok=True)
  output_root_path.mkdir(parents=True, exist_ok=True)

  stages: list[ScaleStageResult] = []
  for stage_index, designs_per_library_type in enumerate(stage_sizes):
    stage_campaign_id = f"{campaign_id}-dpl-{designs_per_library_type}"
    stage_manifest_path = manifest_dir_path / f"{stage_campaign_id}.manifest.json"
    stage_output_root = output_root_path / f"dpl_{designs_per_library_type}"
    write_campaign_manifest(
      base_spec=base_spec,
      campaign_id=stage_campaign_id,
      manifest_path=stage_manifest_path,
      designs_per_library_type=designs_per_library_type,
      samples_chunk_size=samples_chunk_size,
      output_root=stage_output_root,
      fixed_arms=fixed_arms,
      state_weight_profiles=state_weight_profiles,
      planner_version=planner_version,
      dataset_fingerprint=dataset_fingerprint,
      environment_image=environment_image,
      git_sha=git_sha,
      config_hash=config_hash,
    )
    stages.append(
      {
        "stage_index": stage_index,
        "designs_per_library_type": designs_per_library_type,
        "campaign_id": stage_campaign_id,
        "manifest_path": str(stage_manifest_path),
        "output_root": str(stage_output_root),
      },
    )

  return {
    "schema_version": "campaign_scale_ramp_plan_v1",
    "campaign_id": campaign_id,
    "samples_chunk_size": samples_chunk_size,
    "stages": stages,
    "rollback_triggers": [
      "integrity_failure",
      "determinism_drift",
      "resource_envelope_violation",
      "lock_integrity_anomaly",
    ],
  }


def evaluate_scale_ramp_reports(report_paths: Sequence[str | Path]) -> dict[str, Any]:
  """Evaluate staged gate reports and produce promote/rollback recommendation."""
  if not report_paths:
    msg = "At least one gate report path is required for rollout evaluation."
    raise ValueError(msg)

  stage_summaries: list[dict[str, Any]] = []
  rollback_stage_index: int | None = None
  rollback_reasons: set[str] = set()

  for stage_index, report_path in enumerate(report_paths):
    report_payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if not isinstance(report_payload, dict):
      msg = f"Gate report at {report_path} must be a JSON object."
      raise TypeError(msg)
    if report_payload.get("schema_version") != "campaign_gate_report_v1":
      msg = (
        f"Unsupported gate report schema at {report_path}: "
        f"{report_payload.get('schema_version')!r}."
      )
      raise ValueError(msg)
    gates = report_payload.get("gates")
    if not isinstance(gates, dict):
      msg = f"Gate report at {report_path} is missing 'gates' object."
      raise TypeError(msg)
    promote = bool(report_payload.get("promote"))
    integrity_failure = not (
      bool(gates.get("zero_crashes"))
      and bool(gates.get("zero_lineage_collisions"))
      and bool(gates.get("metadata_complete"))
      and bool(gates.get("checkpoint_preflight_pass"))
    )
    determinism_drift = not bool(gates.get("determinism_pass"))
    resource_violation = not bool(gates.get("resource_envelope_within_limits"))
    base_payload = report_payload.get("base")
    runtime_issues = (
      base_payload.get("runtime_issues")
      if isinstance(base_payload, dict) and isinstance(base_payload.get("runtime_issues"), list)
      else []
    )
    lock_anomaly = any("lock" in str(issue).lower() for issue in runtime_issues)

    if integrity_failure:
      rollback_reasons.add("integrity_failure")
    if determinism_drift:
      rollback_reasons.add("determinism_drift")
    if resource_violation:
      rollback_reasons.add("resource_envelope_violation")
    if lock_anomaly:
      rollback_reasons.add("lock_integrity_anomaly")

    if not promote and rollback_stage_index is None:
      rollback_stage_index = stage_index

    stage_summaries.append(
      {
        "stage_index": stage_index,
        "report_path": str(Path(report_path).resolve()),
        "manifest_path": report_payload.get("manifest_path"),
        "promote": promote,
        "gate_failures": sorted(
          reason
          for reason, failed in (
            ("integrity_failure", integrity_failure),
            ("determinism_drift", determinism_drift),
            ("resource_envelope_violation", resource_violation),
            ("lock_integrity_anomaly", lock_anomaly),
          )
          if failed
        ),
      },
    )

  promote = rollback_stage_index is None
  return {
    "schema_version": "campaign_scale_ramp_report_v1",
    "promote": promote,
    "rollback_stage_index": rollback_stage_index,
    "rollback_reasons": sorted(rollback_reasons),
    "stages": stage_summaries,
  }


def _row_cell_key(row: dict[str, Any]) -> GateCellKey:
  return (
    str(row["fixed_policy"]),
    str(row["state_weight_profile"]),
    str(row["multi_state_strategy"]),
    tuple(str(value) for value in row["temperature"]),
    tuple(str(value) for value in row["backbone_noise"]),
    bool(row["ligand_conditioning"]),
    bool(row["sidechain_conditioning"]),
  )


def _collect_manifest_gate_state(manifest_path: str | Path) -> ManifestGateState:
  payload = load_manifest(manifest_path)
  rows = payload["rows"]
  if not isinstance(rows, list):  # pragma: no cover - guarded by load_manifest
    msg = "Manifest rows payload must be a list."
    raise TypeError(msg)
  state = ManifestGateState(
    manifest_path=Path(manifest_path).resolve(),
    total_rows=len(rows),
    completed_rows=0,
    digest_by_hash={},
    cell_to_hashes={},
    metadata_issues=[],
    checkpoint_issues=[],
    runtime_issues=[],
  )
  for row in rows:
    if not isinstance(row, dict):  # pragma: no cover - guarded by load_manifest
      state.metadata_issues.append("Encountered non-object manifest row.")
      continue
    manifest_hash = str(row.get("manifest_row_hash", ""))
    if not manifest_hash:
      state.metadata_issues.append("Manifest row missing manifest_row_hash.")
      continue
    sampling_spec_payload = row.get("sampling_spec")
    if not isinstance(sampling_spec_payload, dict):
      state.metadata_issues.append(
        f"Manifest row {manifest_hash} is missing sampling_spec payload.",
      )
      continue
    output_value = sampling_spec_payload.get("output_h5_path")
    if output_value is None:
      state.metadata_issues.append(
        f"Manifest row {manifest_hash} is missing sampling_spec.output_h5_path.",
      )
      continue
    output_h5_path = Path(output_value).resolve()
    done_marker = _read_done_marker(_done_marker_path(output_h5_path))
    if done_marker is None:
      state.metadata_issues.append(
        f"Manifest row {manifest_hash} missing done marker for output {output_h5_path}.",
      )
      continue
    try:
      _validate_done_marker(
        marker=done_marker,
        marker_path=_done_marker_path(output_h5_path),
        output_h5_path=output_h5_path,
        manifest_row_hash=manifest_hash,
      )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
      state.metadata_issues.append(str(exc))
      continue

    state.completed_rows += 1
    state.digest_by_hash[manifest_hash] = str(done_marker["content_digest_sha256"])
    cell_key = _row_cell_key(row)
    state.cell_to_hashes.setdefault(cell_key, set()).add(manifest_hash)

    checkpoint_id = str(row.get("checkpoint_id", "")).strip()
    if not checkpoint_id:
      state.checkpoint_issues.append(f"Manifest row {manifest_hash} has empty checkpoint_id.")

    campaign_mode = bool(sampling_spec_payload.get("campaign_mode"))
    grid_mode = bool(sampling_spec_payload.get("grid_mode"))
    if not campaign_mode or not grid_mode:
      state.runtime_issues.append(
        f"Manifest row {manifest_hash} must run with campaign_mode=True and grid_mode=True.",
      )
    return_logits = bool(sampling_spec_payload.get("return_logits"))
    allow_logits = bool(sampling_spec_payload.get("allow_logits_in_campaign"))
    logits_budget = sampling_spec_payload.get("logits_memory_budget_mb")
    if return_logits and (not allow_logits or logits_budget is None):
      state.runtime_issues.append(
        f"Manifest row {manifest_hash} has invalid campaign logits configuration.",
      )
  return state


def evaluate_campaign_gates(
  manifest_path: str | Path,
  *,
  rerun_manifest_paths: tuple[str | Path, ...] = (),
  require_full_cell_rerun: bool = True,
) -> dict[str, Any]:
  """Evaluate campaign gating status for pilot promotion/rollback decisions."""
  base_state = _collect_manifest_gate_state(manifest_path)
  rerun_reports: list[dict[str, Any]] = []
  mismatched_hashes: set[str] = set()
  full_cell_rerun = False

  for rerun_manifest_path in rerun_manifest_paths:
    rerun_state = _collect_manifest_gate_state(rerun_manifest_path)
    shared_hashes = set(base_state.digest_by_hash).intersection(rerun_state.digest_by_hash)
    matched_hashes = {
      row_hash
      for row_hash in shared_hashes
      if base_state.digest_by_hash[row_hash] == rerun_state.digest_by_hash[row_hash]
    }
    mismatched = shared_hashes - matched_hashes
    mismatched_hashes.update(mismatched)
    if not full_cell_rerun:
      full_cell_rerun = any(
        row_hashes.issubset(matched_hashes) and bool(row_hashes)
        for row_hashes in base_state.cell_to_hashes.values()
      )
    rerun_reports.append(
      {
        "manifest_path": str(rerun_state.manifest_path),
        "total_rows": rerun_state.total_rows,
        "completed_rows": rerun_state.completed_rows,
        "shared_rows": len(shared_hashes),
        "matched_rows": len(matched_hashes),
        "mismatched_rows": len(mismatched),
        "metadata_issues": rerun_state.metadata_issues,
        "checkpoint_issues": rerun_state.checkpoint_issues,
        "runtime_issues": rerun_state.runtime_issues,
      },
    )

  rerun_supplied = bool(rerun_manifest_paths)
  if require_full_cell_rerun:
    determinism_pass = rerun_supplied and not mismatched_hashes and full_cell_rerun
  else:
    determinism_pass = (not mismatched_hashes) and (not rerun_supplied or full_cell_rerun)

  gates = {
    "zero_crashes": base_state.completed_rows == base_state.total_rows,
    "zero_lineage_collisions": True,  # enforced by load_manifest/validate_manifest_rows
    "metadata_complete": len(base_state.metadata_issues) == 0,
    "checkpoint_preflight_pass": len(base_state.checkpoint_issues) == 0,
    "determinism_pass": determinism_pass,
    "resource_envelope_within_limits": len(base_state.runtime_issues) == 0,
  }
  promote = all(gates.values())
  return {
    "schema_version": "campaign_gate_report_v1",
    "manifest_path": str(base_state.manifest_path),
    "promote": promote,
    "gates": gates,
    "base": {
      "total_rows": base_state.total_rows,
      "completed_rows": base_state.completed_rows,
      "metadata_issues": base_state.metadata_issues,
      "checkpoint_issues": base_state.checkpoint_issues,
      "runtime_issues": base_state.runtime_issues,
    },
    "reruns": {
      "required_full_cell_rerun": require_full_cell_rerun,
      "supplied": rerun_supplied,
      "full_cell_rerun": full_cell_rerun,
      "mismatched_row_hashes": sorted(mismatched_hashes),
      "reports": rerun_reports,
    },
  }


def _parse_csv(value: str) -> tuple[str, ...]:
  return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_fixed_arms(values: Sequence[str] | None) -> dict[str, str] | None:
  """Parse repeated ``label=declaration`` into a ``fixed_arms`` mapping.

  The value is passed through untouched -- ``_resolve_arm`` decides whether it is an inline
  declaration (``H38|D73|C143``) or a ``.npy`` path. This function only splits label from
  value, so there is exactly one place that knows the arm grammar.

  A REPEATED flag, deliberately not CSV. ``_parse_csv`` splits on comma, so any comma-bearing
  grammar (``triad=H38,D73,C143``) would be shattered into bogus separate cells -- and
  ``validate_manifest_rows`` would then compare the wreckage against itself and pass. Same
  idiom as ``cli._parse_tied_positions``: repeat the flag, split once on a non-comma
  separator.

  Inline is the common case and needs no file: requiring a caller to materialise a ``.npy``
  to say "hold three residues" is friction with nothing to show for it. A path stays supported
  for large sets (an 84-residue active-site shell is not worth typing).

  Returns None for no flags, so the caller gets the honest "fix nothing" default rather than
  an empty mapping (which `plan_campaign_manifest` rejects as a contradiction).

  Raises:
    ValueError: an entry has no ``=``, an empty label, or an empty path.

  """
  if not values:
    return None
  arms: dict[str, str] = {}
  for raw in values:
    label, sep, path = raw.partition("=")
    label, path = label.strip(), path.strip()
    if not sep or not label or not path:
      msg = (
        f"--fixed-arm expects LABEL=DECLARATION (e.g. --fixed-arm "
        f"catalytic_triad=H38|D73|C143), got {raw!r}. Repeat the flag for additional arms; "
        f"each becomes its own row-set. A .npy mask path also works for large sets."
      )
      raise ValueError(msg)
    if label in arms:
      msg = f"--fixed-arm {label!r} given twice; arm labels must be unique."
      raise ValueError(msg)
    arms[label] = path
  return arms


def parse_state_weight_profiles(
  names_csv: str | None,
  declarations: Sequence[str] | None,
) -> dict[str, Any]:
  """Combine ``--state-weight-profiles`` (names) and ``--state-weight-profile`` (LABEL=WEIGHTS).

  ``--state-weight-profile LABEL=0.7|0.3`` is the flag that carries its own referent (the same
  idiom as ``--fixed-arm``); it is repeatable and each label becomes its own row-set.
  ``--state-weight-profiles`` is the legacy CSV of bare names, in which only ``equal`` means
  anything -- any other name must ALSO be declared with weights, else this raises (a bare name
  was a label that never reached ``state_weights``).

  With neither flag the result is ``{"equal": None}``, i.e. today's default. With only
  declarations there is NO implicit ``equal`` baseline: add ``--state-weight-profiles equal``
  to keep one.

  Raises:
    ValueError: an entry has no ``=``, an empty label or weights, a duplicate label, or a bare
      non-``equal`` name with no declaration.

  """
  declared: dict[str, str] = {}
  for raw in declarations or ():
    label, sep, weights = raw.partition("=")
    label, weights = label.strip(), weights.strip()
    if not sep or not label or not weights:
      msg = (
        f"--state-weight-profile expects LABEL=WEIGHTS (e.g. --state-weight-profile "
        f"pocket_heavy=0.7|0.3; a 1-D .npy path also works), got {raw!r}. Repeat the flag for "
        f"more profiles; each becomes its own row-set."
      )
      raise ValueError(msg)
    if label in declared:
      msg = f"--state-weight-profile {label!r} given twice; profile labels must be unique."
      raise ValueError(msg)
    declared[label] = weights

  if names_csv is None and not declared:
    return {_EQUAL_PROFILE: None}
  profiles: dict[str, Any] = {}
  for name in _parse_csv(names_csv) if names_csv is not None else ():
    if name == _EQUAL_PROFILE:
      profiles[name] = None
    elif name in declared:
      profiles[name] = declared[name]
    else:
      msg = (
        f"--state-weight-profiles names {name!r}, which has no weights. A bare name is a label "
        f"aminx cannot resolve; declare it: --state-weight-profile {name}=0.7|0.3"
      )
      raise ValueError(msg)
  for label, weights in declared.items():
    profiles.setdefault(label, weights)
  return profiles


def _parse_int_csv(value: str) -> tuple[int, ...]:
  parsed: list[int] = []
  for item in value.split(","):
    stripped = item.strip()
    if not stripped:
      continue
    parsed.append(int(stripped))
  return tuple(parsed)


def _emit_json(payload: dict[str, Any], output_path: str | None = None) -> None:
  rendered = json.dumps(payload, sort_keys=True, indent=2)
  if output_path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered + "\n", encoding="utf-8")
  sys.stdout.write(rendered + "\n")


def _add_state_weight_arguments(parser: argparse.ArgumentParser) -> None:
  parser.add_argument(
    "--state-weight-profiles",
    default=None,
    help="Comma-separated profile NAMES; only 'equal' is resolvable by name. Default: equal "
    "(unless --state-weight-profile is given).",
  )
  parser.add_argument(
    "--state-weight-profile",
    action="append",
    default=None,
    metavar="LABEL=WEIGHTS",
    help="Profile LABEL=WEIGHTS, e.g. pocket_heavy=0.7|0.3 (one weight per state; a 1-D .npy "
    "path also works). Reaches each row's sampling_spec state_weights. Repeatable.",
  )


def _add_plan_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser("plan", help="Generate campaign manifest.")
  parser.add_argument("--inputs", required=True, help="Comma-separated input paths.")
  parser.add_argument("--campaign-id", required=True)
  parser.add_argument("--manifest-path", required=True)
  parser.add_argument("--output-root", required=True)
  parser.add_argument("--designs-per-library-type", type=int, required=True)
  parser.add_argument("--samples-chunk-size", type=int, required=True)
  # Repeatable LABEL=PATH, not a CSV of bare names. A bare name was unresolvable --
  # aminx cannot know any protein's catalytic triad -- so it was written to the row as a
  # label and never reached the model. Absent => fix nothing.
  parser.add_argument(
    "--fixed-arm", action="append", default=None, metavar="LABEL=PATH",
    help=(
      "Arm LABEL=DECLARATION, e.g. catalytic_triad=H38|D73|C143 (residue letter + 0-based "
      "canonical index). A .npy mask path also works. Repeatable."
    ),
  )
  _add_state_weight_arguments(parser)


def _add_worker_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser("worker", help="Execute one manifest row.")
  parser.add_argument("--manifest-path", required=True)
  parser.add_argument("--row-index", type=int, default=None)
  parser.add_argument("--row-hash", default=None)
  parser.add_argument("--lock-backend", default="local_fs")
  parser.add_argument("--lock-lease-seconds", type=int, default=DEFAULT_LOCK_LEASE_SECONDS)
  parser.add_argument(
    "--heartbeat-interval-seconds",
    type=int,
    default=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
  )


def _add_run_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser("run", help="Execute manifest rows sequentially.")
  parser.add_argument("--manifest-path", required=True)
  parser.add_argument(
    "--row-hash",
    action="append",
    default=[],
    help="Optional row hash filter; repeat to run specific rows only.",
  )
  parser.add_argument("--continue-on-error", action="store_true")
  parser.add_argument("--summary-path", default=None)
  parser.add_argument("--lock-backend", default="local_fs")
  parser.add_argument("--lock-lease-seconds", type=int, default=DEFAULT_LOCK_LEASE_SECONDS)
  parser.add_argument(
    "--heartbeat-interval-seconds",
    type=int,
    default=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
  )


def _add_gates_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser("gates", help="Evaluate pilot campaign gates.")
  parser.add_argument("--manifest-path", required=True)
  parser.add_argument(
    "--rerun-manifest-path",
    action="append",
    default=[],
    help="Optional rerun manifest path; may be passed multiple times.",
  )
  parser.add_argument("--report-path", default=None)
  parser.add_argument(
    "--allow-missing-rerun",
    action="store_true",
    help="Allow determinism gate without a full-cell rerun.",
  )


def _add_ramp_plan_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser("ramp-plan", help="Generate staged ramp manifests.")
  parser.add_argument("--inputs", required=True, help="Comma-separated input paths.")
  parser.add_argument("--campaign-id", required=True)
  parser.add_argument("--manifest-dir", required=True)
  parser.add_argument("--output-root", required=True)
  parser.add_argument("--stage-designs-per-library-type", required=True)
  parser.add_argument("--samples-chunk-size", type=int, required=True)
  # Repeatable LABEL=PATH, not a CSV of bare names. A bare name was unresolvable --
  # aminx cannot know any protein's catalytic triad -- so it was written to the row as a
  # label and never reached the model. Absent => fix nothing.
  parser.add_argument(
    "--fixed-arm", action="append", default=None, metavar="LABEL=PATH",
    help=(
      "Arm LABEL=DECLARATION, e.g. catalytic_triad=H38|D73|C143 (residue letter + 0-based "
      "canonical index). A .npy mask path also works. Repeatable."
    ),
  )
  _add_state_weight_arguments(parser)
  parser.add_argument("--plan-path", default=None)


def _add_ramp_eval_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser("ramp-evaluate", help="Evaluate staged gate reports.")
  parser.add_argument(
    "--report-path",
    action="append",
    required=True,
    help="Stage gate report path; pass in stage order.",
  )
  parser.add_argument("--summary-path", default=None)


def _handle_plan_command(args: argparse.Namespace) -> int:
  base_spec = SamplingSpecification(
    inputs=_parse_csv(args.inputs),
    return_logits=False,
  )
  write_campaign_manifest(
    base_spec=base_spec,
    campaign_id=args.campaign_id,
    manifest_path=args.manifest_path,
    designs_per_library_type=args.designs_per_library_type,
    samples_chunk_size=args.samples_chunk_size,
    output_root=args.output_root,
    fixed_arms=parse_fixed_arms(args.fixed_arm),
    state_weight_profiles=parse_state_weight_profiles(
      args.state_weight_profiles, args.state_weight_profile,
    ),
  )
  return 0


def _handle_worker_command(args: argparse.Namespace) -> int:
  run_manifest_row(
    args.manifest_path,
    row_index=args.row_index,
    row_hash=args.row_hash,
    lock_backend=args.lock_backend,
    lock_lease_seconds=args.lock_lease_seconds,
    heartbeat_interval_seconds=args.heartbeat_interval_seconds,
  )
  return 0


def _handle_run_command(args: argparse.Namespace) -> int:
  summary = execute_manifest(
    args.manifest_path,
    row_hashes=tuple(args.row_hash),
    continue_on_error=args.continue_on_error,
    lock_backend=args.lock_backend,
    lock_lease_seconds=args.lock_lease_seconds,
    heartbeat_interval_seconds=args.heartbeat_interval_seconds,
  )
  _emit_json(summary, args.summary_path)
  return 0 if summary["failed_rows"] == 0 else 1


def _handle_gates_command(args: argparse.Namespace) -> int:
  report = evaluate_campaign_gates(
    args.manifest_path,
    rerun_manifest_paths=tuple(args.rerun_manifest_path),
    require_full_cell_rerun=not args.allow_missing_rerun,
  )
  _emit_json(report, args.report_path)
  return 0 if report["promote"] else 2


def _handle_ramp_plan_command(args: argparse.Namespace) -> int:
  base_spec = SamplingSpecification(
    inputs=_parse_csv(args.inputs),
    return_logits=False,
  )
  plan_payload = plan_scale_ramp(
    base_spec=base_spec,
    campaign_id=args.campaign_id,
    manifest_dir=args.manifest_dir,
    output_root=args.output_root,
    stage_designs_per_library_type=_parse_int_csv(args.stage_designs_per_library_type),
    samples_chunk_size=args.samples_chunk_size,
    fixed_arms=parse_fixed_arms(args.fixed_arm),
    state_weight_profiles=parse_state_weight_profiles(
      args.state_weight_profiles, args.state_weight_profile,
    ),
  )
  _emit_json(plan_payload, args.plan_path)
  return 0


def _handle_ramp_eval_command(args: argparse.Namespace) -> int:
  summary = evaluate_scale_ramp_reports(tuple(args.report_path))
  _emit_json(summary, args.summary_path)
  return 0 if summary["promote"] else 2


def main(argv: list[str] | None = None) -> int:
  """CLI entrypoint for campaign planner and worker operations."""
  configure_multiprocessing()
  parser = argparse.ArgumentParser(description="Campaign planner/worker runner.")
  subparsers = parser.add_subparsers(dest="command", required=True)
  _add_plan_parser(subparsers)
  _add_worker_parser(subparsers)
  _add_run_parser(subparsers)
  _add_gates_parser(subparsers)
  _add_ramp_plan_parser(subparsers)
  _add_ramp_eval_parser(subparsers)

  args = parser.parse_args(argv)
  if args.command == "plan":
    return _handle_plan_command(args)
  if args.command == "worker":
    return _handle_worker_command(args)
  if args.command == "run":
    return _handle_run_command(args)
  if args.command == "gates":
    return _handle_gates_command(args)
  if args.command == "ramp-plan":
    return _handle_ramp_plan_command(args)
  if args.command == "ramp-evaluate":
    return _handle_ramp_eval_command(args)
  msg = f"Unsupported command: {args.command!r}"
  raise ValueError(msg)


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
