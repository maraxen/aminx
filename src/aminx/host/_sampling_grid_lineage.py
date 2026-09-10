import hashlib
import json
from typing import Any

import jax
import numpy as np

from aminx.run.specs import SamplingSpecification

# Bumped v1 -> v2 for task_id `260910_aminx-sink-provenance-schema`: new OPTIONAL attrs
# (logits_bias_semantics, prng_seed, aminx_version) added to sinks that key off this
# constant. Absent on a v1 store means "written before these fields existed", not
# "unknown" in any stronger sense -- aminx has no reader for these stores yet.
#
# This is the single source of truth for GRID_SCHEMA_VERSION. `host/streaming.py`
# previously carried an independent, coincidentally-identical duplicate definition of
# this same constant (a filed defect); it now imports this one instead of redefining it,
# which is what actually prevents the two from silently diverging again -- see
# `host/streaming.py`'s import of `GRID_SCHEMA_VERSION` from this module.
GRID_SCHEMA_VERSION = "grid_v2"


def _resolve_grid_lineage(spec: SamplingSpecification) -> dict[str, int | str] | None:
  if not spec.grid_mode:
    return None
  sample_count = int(spec.sample_count if spec.sample_count is not None else spec.run_spec.sampling.num_samples)
  if sample_count <= 0:
    msg = "sample_count must be positive when grid_mode=True."
    raise ValueError(msg)
  sample_start = int(spec.sample_start if spec.sample_start is not None else 0)
  if sample_start < 0:
    msg = "sample_start must be non-negative when grid_mode=True."
    raise ValueError(msg)
  chunk_id = int(spec.chunk_id if spec.chunk_id is not None else 0)
  if chunk_id < 0:
    msg = "chunk_id must be non-negative when grid_mode=True."
    raise ValueError(msg)
  job_id = spec.job_id or f"grid_{spec.run_spec.sampling.random_seed}"
  return {
    "job_id": job_id,
    "chunk_id": chunk_id,
    "sample_start": sample_start,
    "sample_count": sample_count,
  }


def _grid_sample_indices(lineage: dict[str, int | str]) -> np.ndarray:
  sample_start = int(lineage["sample_start"])
  sample_count = int(lineage["sample_count"])
  return np.arange(sample_start, sample_start + sample_count, dtype=np.int64)


def _grid_iteration_arrays(
  lineage: dict[str, int | str],
  *,
  chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  sample_start = int(lineage["sample_start"])
  sample_count = int(lineage["sample_count"])
  if chunk_size <= 0:
    msg = "samples_chunk_size must be positive when provided."
    raise ValueError(msg)
  iteration_ids: list[int] = []
  iteration_starts: list[int] = []
  iteration_counts: list[int] = []
  local_offset = 0
  while local_offset < sample_count:
    count = min(chunk_size, sample_count - local_offset)
    iteration_ids.append(len(iteration_ids))
    iteration_starts.append(sample_start + local_offset)
    iteration_counts.append(count)
    local_offset += count
  return (
    np.asarray(iteration_ids, dtype=np.int64),
    np.asarray(iteration_starts, dtype=np.int64),
    np.asarray(iteration_counts, dtype=np.int64),
  )


def _canonical_float_strings(values: Any) -> list[str]:  # noqa: ANN401
  return [format(float(value), ".17g") for value in values]


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
  ).encode("utf-8")


def _grid_manifest_row_hash(
  spec: SamplingSpecification,
  lineage: dict[str, int | str],
) -> str:
  payload = {
    "schema_version": GRID_SCHEMA_VERSION,
    "job_id": str(lineage["job_id"]),
    "chunk_id": int(lineage["chunk_id"]),
    "sample_start": int(lineage["sample_start"]),
    "sample_count": int(lineage["sample_count"]),
    "model_family": spec.model_family,
    "ligand_conditioning": bool(spec.ligand_conditioning),
    "sidechain_conditioning": bool(spec.sidechain_conditioning),
    "multi_state_strategy": spec.multi_state_strategy,
    "temperature": _canonical_float_strings(spec.run_spec.sampling.temperature),
    "backbone_noise": _canonical_float_strings(spec.run_spec.sampling.backbone_noise),
  }
  return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _grid_job_seed_hash(
  spec: SamplingSpecification,
  lineage: dict[str, int | str],
) -> str:
  payload = {
    "schema_version": GRID_SCHEMA_VERSION,
    "job_id": str(lineage["job_id"]),
    "model_family": spec.model_family,
    "ligand_conditioning": bool(spec.ligand_conditioning),
    "sidechain_conditioning": bool(spec.sidechain_conditioning),
    "multi_state_strategy": spec.multi_state_strategy,
    "temperature": _canonical_float_strings(spec.run_spec.sampling.temperature),
    "backbone_noise": _canonical_float_strings(spec.run_spec.sampling.backbone_noise),
  }
  return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _seed_words_from_manifest_hash(manifest_row_hash: str) -> tuple[int, int, int, int]:
  digest = bytes.fromhex(manifest_row_hash)
  words = [
    int.from_bytes(digest[offset : offset + 4], byteorder="big", signed=False)
    for offset in range(0, 16, 4)
  ]
  return (words[0], words[1], words[2], words[3])


def _base_sampling_key(
  spec: SamplingSpecification,
  *,
  grid_lineage: dict[str, int | str] | None,
) -> jax.Array:
  key = jax.random.key(spec.run_spec.sampling.random_seed)
  if grid_lineage is None:
    return key
  seed_hash = _grid_job_seed_hash(spec, grid_lineage)
  for seed_word in _seed_words_from_manifest_hash(seed_hash):
    key = jax.random.fold_in(key, seed_word)
  return key
