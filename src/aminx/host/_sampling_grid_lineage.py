import hashlib
import json
from typing import Any, Final

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
#
# THIS CONSTANT MUST NEVER FEED THE PRNG SEED DERIVATION. It did until this comment was
# written (audit finding, task_id `260910_aminx-sink-provenance-schema`, fixed same task):
# `_grid_job_seed_hash` hashed a payload keyed on `GRID_SCHEMA_VERSION`, and that hash folds
# directly into `_base_sampling_key`'s `jax.random.fold_in` chain -- so the v1->v2 bump in
# commit 7b1bc3b silently changed the sampled output of every grid-mode run (different
# tokens, different logits at later AR positions) with no behavioural-change warning. See
# `_SEED_HASH_SCHEMA_PIN` below, which is the value the seed hash actually uses now, and
# which must stay frozen forever regardless of how many more times this constant is bumped.
GRID_SCHEMA_VERSION = "grid_v2"

# FROZEN FOREVER. Feeds `_grid_job_seed_hash` -> `_seed_words_from_manifest_hash` ->
# `_base_sampling_key`'s `jax.random.fold_in` chain, i.e. it is load-bearing for every
# grid-mode run's sampled sequences and logits. Deliberately NOT `GRID_SCHEMA_VERSION`:
# that constant exists to label stores for readers and is expected to keep bumping over
# time, and coupling the two means every future schema bump silently reseeds every
# existing store's worth of designs. Changing this literal is equivalent to deliberately
# breaking reproducibility of every stored grid-mode run to date -- do not do it as part
# of a routine schema bump. If the seed-derivation payload genuinely must change, that is
# its own explicit, documented, reproducibility-breaking decision, not a side effect of
# relabelling stores.
_SEED_HASH_SCHEMA_PIN: Final[str] = "grid_v1"


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
  """Identify/dedupe a manifest row -- NOT part of the PRNG seed derivation.

  Investigated for task_id `260910_aminx-sink-provenance-schema`: this hash feeds only
  `root_attrs["manifest_row_hash"]` (store bookkeeping) and `campaign.py`'s done-marker
  matching (`_validate_done_marker`/`_write_done_marker`) -- it is never passed to
  `_base_sampling_key` or any other randomness-consuming call. It is therefore safe, and
  intentional, for it to keep tracking `GRID_SCHEMA_VERSION` (unlike `_grid_job_seed_hash`
  below, which must NOT).

  Consequence of that choice, so it isn't rediscovered as a surprise: bumping
  `GRID_SCHEMA_VERSION` changes every future `manifest_row_hash` for an otherwise-identical
  row. A campaign resumed after such a bump will regenerate its manifest with the new
  hash, `_validate_done_marker` will see the stored marker's old hash disagree, and it
  raises rather than silently reusing the old store -- so a resume across a schema bump
  forces genuinely-completed rows to be treated as not-done and rerun. That is a resume-
  cost/safety tradeoff (never a correctness bug: nothing mixes v1 and v2 semantics
  silently), and it is the intended effect of bumping a *schema* version -- it should be a
  breaking change against old resume state, not something that can quietly drift back in.
  """
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
  """PRNG-seed-feeding hash -- payload MUST use `_SEED_HASH_SCHEMA_PIN`, never `GRID_SCHEMA_VERSION`.

  See the module-level comments on both constants: this hash folds directly into
  `_base_sampling_key`, so its payload must never move for reasons unrelated to a
  deliberate, explicit reproducibility break.
  """
  payload = {
    "schema_version": _SEED_HASH_SCHEMA_PIN,
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
