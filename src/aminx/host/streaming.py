"""Zarr streaming output path for sampling."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from xtrax.run import SinkSpec, ZarrStagingSink

from aminx.host._sampling_grid_lineage import (
  GRID_SCHEMA_VERSION,
  _grid_iteration_arrays,
  _grid_manifest_row_hash,
  _grid_sample_indices,
  _resolve_grid_lineage,
)
from aminx.host._sampling_helper import (
  _canonical_structure_ids_for_spec,
  _structure_ids_for_batch,
  fixed_provenance_outputs,
)
from aminx.host.output_sinks import (
  streaming_tensor_sink_session,
  take_staging_sequences_logits,
)
from aminx.host.plan import (
  resolve_chunk_size,
  resolve_sample_start,
  resolve_target_samples,
)
from aminx.host.streaming_host import StreamingBatchHost
from aminx.io.sink_provenance import (
  logits_bias_semantics_outputs,
  prng_seed_attrs,
  resolve_aminx_version,
)

if TYPE_CHECKING:
  from grain.python import IterDataset

  from aminx.run.specs import SamplingSpecification


# Bumped v1 -> v2 alongside GRID_SCHEMA_VERSION for task_id
# `260910_aminx-sink-provenance-schema`: new OPTIONAL root attrs (logits_bias_semantics,
# prng_seed, aminx_version). Absent on a v1 store just means "written before these fields
# existed" -- aminx has no reader for these stores yet, so this makes no claim about
# reader behavior.
SAMPLING_SCHEMA_VERSION = "sampling_v2"
# GRID_SCHEMA_VERSION is re-exported (not redefined) from `_sampling_grid_lineage` --
# that module is now the single source of truth; see the comment on its own definition.


def _grid_lineage_attrs(grid_lineage: dict[str, int | str]) -> dict[str, Any]:
  """Common per-structure grid-lineage attrs, shared by both write branches below."""
  return {
    "job_id": str(grid_lineage["job_id"]),
    "chunk_id": int(grid_lineage["chunk_id"]),
    "sample_start": int(grid_lineage["sample_start"]),
    "sample_count": int(grid_lineage["sample_count"]),
  }


def _sample_streaming(
  spec: SamplingSpecification,
  protein_iterator: IterDataset,
  plan: Any,
  sample_batch_fn: Callable[..., tuple[Any, Any, Any | None]],
) -> dict[str, Any]:
  """Sample new sequences and stream results to a Zarr store.

  Non-campaign mode writes one design per structure directly. Campaign mode
  accumulates a structure's sample-chunks in memory *within one batch* (not
  across the whole campaign) and stages the concatenated result once per
  batch -- bounded by a single batch's dispatch size, not total campaign
  size. This trades the old HDF5 path's per-chunk incremental resize writes
  for simplicity; if a future campaign's per-batch memory footprint proves
  too large, xtrax.run.ZarrStagingSink would need an append-mode extension
  to restore true per-chunk incremental writes.
  """
  grid_lineage = _resolve_grid_lineage(spec)
  canonical_structure_ids = _canonical_structure_ids_for_spec(spec)
  resolved_structure_ids: list[str] = []
  total_num_samples = resolve_target_samples(spec, grid_lineage=grid_lineage)
  chunk_size = resolve_chunk_size(spec, total_num_samples, grid_lineage)
  sample_start = resolve_sample_start(grid_lineage)
  structure_batch_count_stream = StreamingBatchHost.structure_batch_count(protein_iterator)

  output_dir = Path(spec.run_spec.io.output_h5_path)
  sink = ZarrStagingSink(SinkSpec(output_dir=output_dir, format="zarr", flush_every=1))
  # Resolved HERE, at sink construction -- before any sampling compute below -- so a
  # PackageNotFoundError (resolve_aminx_version raises rather than swallowing it) fails
  # this run before any work is done, never after, so it cannot discard a completed run.
  resolved_aminx_version = resolve_aminx_version()

  root_attrs: dict[str, Any] = {
    "schema_version": GRID_SCHEMA_VERSION if spec.grid_mode else SAMPLING_SCHEMA_VERSION,
    "model_family": spec.model_family,
    "ligand_conditioning": int(spec.ligand_conditioning),
    "sidechain_conditioning": int(spec.sidechain_conditioning),
    "samples_chunk_size": chunk_size,
    "aminx_version": resolved_aminx_version,
    **prng_seed_attrs(spec.run_spec.sampling.random_seed),
  }
  root_arrays: dict[str, np.ndarray] = {}
  # Computed ONCE from `spec` (constant for the whole call: neither branch below ever
  # builds a per-structure or per-chunk bias) and reused for every group below that
  # stages a "logits" array. This is what makes AC1b's uniformity requirement hold by
  # construction here -- there is no code path left that could compute a second, diverging
  # value -- unlike `multistate_poe.py`'s per-(noise, temperature)-cell loop, which
  # recomputes per cell and therefore uses `assert_uniform_group_attr` explicitly.
  bias_attrs: dict[str, Any] = {}
  if spec.run_spec.sampling.return_logits:
    bias_arrays, bias_attrs = logits_bias_semantics_outputs(spec.run_spec.sampling.bias)
    root_arrays.update(bias_arrays)
    root_attrs.update(bias_attrs)
  if grid_lineage is not None:
    manifest_row_hash = _grid_manifest_row_hash(spec, grid_lineage)
    root_attrs.update(_grid_lineage_attrs(grid_lineage))
    root_attrs["manifest_row_hash"] = manifest_row_hash
    iteration_ids, iteration_starts, iteration_counts = _grid_iteration_arrays(
      grid_lineage,
      chunk_size=chunk_size,
    )
    root_arrays = {
      "sample_indices": _grid_sample_indices(grid_lineage),
      "grid_iteration_ids": iteration_ids,
      "grid_iteration_sample_start": iteration_starts,
      "grid_iteration_sample_count": iteration_counts,
    }
  sink.stage((), attrs=root_attrs, **root_arrays)

  structure_idx = 0
  with streaming_tensor_sink_session():
    for batch_idx, batched_ensemble in enumerate(protein_iterator):
      batch_size = batched_ensemble.coordinates.shape[0]
      batch_structure_ids = _structure_ids_for_batch(
        canonical_structure_ids,
        structure_offset=structure_idx,
        batch_size=batch_size,
      )
      if not spec.campaign_mode:
        key_chunk_start = int(grid_lineage["sample_start"]) if grid_lineage is not None else 0
        key_chunk_count = total_num_samples
        _, _, pseudo_perplexity = sample_batch_fn(
          spec,
          batched_ensemble,
          plan,
          canonical_structure_ids=canonical_structure_ids,
          batch_structure_ids=batch_structure_ids,
          chunk_sample_start=key_chunk_start,
          chunk_sample_count=key_chunk_count,
          batch_idx=batch_idx,
          structure_batch_count=structure_batch_count_stream,
          emit_structure_batch_io=True,
        )
        StreamingBatchHost.sink_barrier()
        sampled_sequences_np, sampled_logits_np = take_staging_sequences_logits(
          batch_idx,
          key_chunk_start,
          key_chunk_count,
        )
        for i in range(sampled_sequences_np.shape[0]):
          key = (f"structure_{structure_idx}",)
          arrays: dict[str, np.ndarray] = {"sequences": sampled_sequences_np[i]}
          if spec.run_spec.sampling.return_logits:
            arrays["logits"] = sampled_logits_np[i]
          if pseudo_perplexity is not None:
            arrays["pseudo_perplexity"] = pseudo_perplexity[i]
          attrs = {
            "structure_index": structure_idx,
            "structure_id": batch_structure_ids[i],
            "num_samples": int(sampled_sequences_np.shape[1]),
            "num_noise_levels": int(sampled_sequences_np.shape[2]),
            "num_temperatures": int(sampled_sequences_np.shape[3]),
            "sequence_length": int(sampled_sequences_np.shape[4]),
          }
          if grid_lineage is not None:
            attrs.update(_grid_lineage_attrs(grid_lineage))
          if spec.run_spec.sampling.return_logits:
            attrs.update(bias_attrs)
          sink.stage(key, attrs=attrs, **arrays)
          resolved_structure_ids.append(batch_structure_ids[i])
          structure_idx += 1
      else:
        structure_keys: list[tuple[str]] = []
        for i in range(batch_size):
          key = (f"structure_{structure_idx}",)
          attrs = {
            "structure_index": structure_idx,
            "structure_id": batch_structure_ids[i],
            "num_samples": total_num_samples,
            "num_noise_levels": len(spec.run_spec.sampling.backbone_noise),
            "num_temperatures": len(spec.run_spec.sampling.temperature),
            "sequence_length": int(batched_ensemble.coordinates.shape[1]),
          }
          if grid_lineage is not None:
            attrs.update(_grid_lineage_attrs(grid_lineage))
          sink.stage(key, attrs=attrs)
          structure_keys.append(key)
          resolved_structure_ids.append(batch_structure_ids[i])
          structure_idx += 1

        seq_parts: dict[tuple[str], list[np.ndarray]] = {k: [] for k in structure_keys}
        logits_parts: dict[tuple[str], list[np.ndarray]] = {k: [] for k in structure_keys}
        perplexity_parts: dict[tuple[str], list[np.ndarray]] = {k: [] for k in structure_keys}

        chunks = list(StreamingBatchHost.iter_chunks(total_num_samples, chunk_size))
        for chunk_idx, (chunk_start, chunk_count) in enumerate(chunks):
          chunk_sample_start = sample_start + chunk_start
          is_last_chunk = chunk_idx == len(chunks) - 1
          _, _, pseudo_perplexity = sample_batch_fn(
            spec,
            batched_ensemble,
            plan,
            canonical_structure_ids=canonical_structure_ids,
            batch_structure_ids=batch_structure_ids,
            chunk_sample_start=chunk_sample_start,
            chunk_sample_count=chunk_count,
            batch_idx=batch_idx,
            structure_batch_count=structure_batch_count_stream,
            emit_structure_batch_io=is_last_chunk,
          )
          StreamingBatchHost.sink_barrier()
          sampled_sequences_np, sampled_logits_np = take_staging_sequences_logits(
            batch_idx,
            chunk_sample_start,
            chunk_count,
          )
          for i, key in enumerate(structure_keys):
            seq_parts[key].append(sampled_sequences_np[i].astype(np.int32, copy=False))
            if spec.run_spec.sampling.return_logits:
              logits_parts[key].append(sampled_logits_np[i].astype(np.float32, copy=False))
            if pseudo_perplexity is not None:
              perplexity_parts[key].append(np.asarray(pseudo_perplexity[i], dtype=np.float32))

        for key in structure_keys:
          concat_arrays: dict[str, np.ndarray] = {"sequences": np.concatenate(seq_parts[key], axis=0)}
          # The evidence that anything was actually held fixed. Emitted beside the sequences
          # so a reader can CHECK them against it; without this the mask reached the model
          # and vanished, and 882/882 rows completed with valid digests and void science.
          fixed_arrays, fixed_attrs = fixed_provenance_outputs(
            spec, seq_len=int(batched_ensemble.coordinates.shape[1]),
          )
          concat_arrays.update(fixed_arrays)
          if logits_parts[key]:
            concat_arrays["logits"] = np.concatenate(logits_parts[key], axis=0)
            fixed_attrs.update(bias_attrs)
          if perplexity_parts[key]:
            concat_arrays["pseudo_perplexity"] = np.concatenate(perplexity_parts[key], axis=0)
          sink.stage(key, attrs=fixed_attrs, **concat_arrays)

  sink.drain()

  results = {
    "output_zarr_path": str(output_dir),
    "schema_version": GRID_SCHEMA_VERSION if spec.grid_mode else SAMPLING_SCHEMA_VERSION,
    "metadata": {
      "specification": spec,
      "skipped_inputs": getattr(protein_iterator, "skipped_frames", []),
      "structure_ids": resolved_structure_ids,
    },
  }
  if grid_lineage is not None:
    manifest_row_hash = _grid_manifest_row_hash(spec, grid_lineage)
    iteration_ids, iteration_starts, iteration_counts = _grid_iteration_arrays(
      grid_lineage,
      chunk_size=chunk_size,
    )
    results["metadata"]["lineage"] = {
      **grid_lineage,
      "manifest_row_hash": manifest_row_hash,
      "sample_indices": _grid_sample_indices(grid_lineage).tolist(),
      "grid_iteration_ids": iteration_ids.tolist(),
      "grid_iteration_sample_start": iteration_starts.tolist(),
      "grid_iteration_sample_count": iteration_counts.tolist(),
    }
  return results


def _sample_streaming_averaged(
  spec: SamplingSpecification,
  protein_iterator: IterDataset,
  model: Any,
  sample_batch_averaged_fn: Callable[..., tuple[Any, Any, Any | None]],
) -> dict[str, Any]:
  """Removed: averaged streaming must use ``_sample_streaming`` + encoding fusion."""
  del spec, protein_iterator, model, sample_batch_averaged_fn
  msg = (
    "_sample_streaming_averaged was removed. Use _sample_streaming with "
    "ArithmeticMeanEncodingFusion wired into the inference plan."
  )
  raise NotImplementedError(msg)
