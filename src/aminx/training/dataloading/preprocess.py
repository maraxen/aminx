"""Preprocessing pipeline for PQR files to array_record with physics features.

This module implements efficient parallel preprocessing with:
- Scatter-gather parallelization (workers write temporary shards)
- Resumable processing (JSONL metadata tracking)
- Progress monitoring
- Feature validation
- Index building

Based on the md_cath dataset preprocessing patterns.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
  from collections.abc import Sequence
  from pathlib import Path

import msgpack
import msgpack_numpy as m
import numpy as np
import tqdm
from array_record.python.array_record_module import (  # type: ignore[unresolved-import]
  ArrayRecordReader,
  ArrayRecordWriter,
)

from aminx.io.parsing import parse_structure

# Patch msgpack to handle numpy arrays
m.patch()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingSpecification:
  """Specifies parameters for the preprocessing pipeline."""

  input_dir: Path
  output_file: Path
  num_workers: int = 8
  force_field: str = "ff14SB"
  resume_from_checkpoint: bool = True
  compute_lj: bool = True
  compute_estat: bool = True
  compression: str = "zstd:9"
  validate_features: bool = True
  group_size: int = 1
  metadata_file: Path = field(init=False)
  index_file: Path = field(init=False)
  temp_dir: Path = field(init=False)
  estat_noise: float | Sequence[float] | None = None
  estat_noise_mode: Literal["direct", "thermal"] = "direct"
  vdw_noise: float | Sequence[float] | None = None
  vdw_noise_mode: Literal["direct", "thermal"] = "direct"

  def __post_init__(self) -> None:
    """Set dependent paths."""
    object.__setattr__(self, "metadata_file", self.output_file.with_suffix(".jsonl"))
    object.__setattr__(self, "index_file", self.output_file.with_suffix(".index"))
    object.__setattr__(self, "temp_dir", self.output_file.parent / f"tmp_{self.output_file.stem}")


def _worker_process_protein(
  args: tuple[Path, PreprocessingSpecification, Path],
) -> tuple[str, Path | None]:
  """Worker function to process a single protein structure.

  Args:
      args: Tuple containing (pqr_path, spec, temp_dir_path).

  Returns:
      A tuple of (protein_id, shard_path). shard_path is None if processing failed.

  """
  structure_path, spec, temp_dir_path = args
  protein_id = structure_path.stem

  try:
    # Use proxide-based parsing with physics computed via OutputSpec
    # compute_physics flag in dispatch handles setting spec.compute_vdw/estat
    # but we pass specific flags for clarity and dispatch handling
    protein = parse_structure(
      structure_path,
      compute_vdw=spec.compute_lj,
      compute_electrostatics=spec.compute_estat,
      force_field=spec.force_field,
    )

    # Physics features are computed by proxide separately
    # We need to concatenate them based on spec
    feats_list = []
    if spec.compute_lj:
      if protein.vdw_features is not None:
        feats_list.append(np.array(protein.vdw_features))
      else:
        feats_list.append(np.zeros((protein.coordinates.shape[0], 5), dtype=np.float32))

    if spec.compute_estat:
      if protein.electrostatic_features is not None:
        feats_list.append(np.array(protein.electrostatic_features))
      else:
        feats_list.append(np.zeros((protein.coordinates.shape[0], 5), dtype=np.float32))

    if feats_list:
      physics_features = np.concatenate(feats_list, axis=-1)
    else:
      # Default fallback if nothing requested but array expected?
      # Maintain 5 dim for backward compat if needed, or 0?
      # Logic below defaults to empty if physics_features is None?
      # Original code defaulted to (N, 5).
      physics_features = np.zeros((protein.coordinates.shape[0], 5), dtype=np.float32)

    record_data = {
      "protein_id": protein_id,
      "source_file": str(structure_path),
      "coordinates": np.array(protein.coordinates),
      # AF-ordered (ARNDCQEGHILKMFPSTWYVX), exactly as proxide parsed it -- NOT the model's
      # MPNN token space. Readers convert via aminx.utils.aa_convert.training_labels; do not
      # rewrite this key in place (it would silently invalidate existing array_record stores).
      "aatype": np.array(protein.aatype),
      "atom_mask": np.array(protein.full_atom_mask)
      if protein.full_atom_mask is not None
      else np.array([]),
      "residue_index": np.array(protein.residue_index),
      "chain_index": np.array(protein.chain_index),
      "mask": np.array(protein.mask),
      "physics_features": np.array(physics_features),
      "full_coordinates": np.array(protein.full_coordinates)
      if protein.full_coordinates is not None
      else np.array([]),
      "charges": np.array(protein.charges) if protein.charges is not None else np.array([]),
      "radii": np.array(protein.radii) if protein.radii is not None else np.array([]),
    }

    # Validate features if requested
    if spec.validate_features:
      for key, value in record_data.items():
        if isinstance(value, np.ndarray) and value.size > 0 and not np.all(np.isfinite(value)):
          logger.warning("Non-finite values in %s for %s", key, structure_path)
          return (protein_id, None)

    shard_path = temp_dir_path / f"shard-{uuid.uuid4().hex}.array_record"
    writer = ArrayRecordWriter(
      str(shard_path),
      f"{spec.compression},group_size:{spec.group_size}",
    )
    writer.write(msgpack.packb(record_data))
    writer.close()

  except Exception:
    logger.exception("Failed to process %s", structure_path)
    return (protein_id, None)

  else:
    return (protein_id, shard_path)


def _raise_no_features_error() -> None:
  """Raise ValueError if no features were requested."""
  msg = "No physics features requested in preprocessing"
  raise ValueError(msg)


def _load_checkpoint_metadata(metadata_file: Path) -> dict[str, Any]:
  """Load existing progress from the metadata JSONL file."""
  if not metadata_file.exists():
    return {"processed_files": set(), "failed_files": set(), "total_records": 0}

  processed = set()
  failed = set()
  count = 0

  with metadata_file.open("r") as f:
    for line in f:
      if not line.strip():
        continue
      try:
        entry = json.loads(line)
        p_id = entry["id"]
        if entry["status"] == "success":
          processed.add(p_id)
          count += 1
        else:
          failed.add(p_id)
      except (json.JSONDecodeError, KeyError):
        logger.warning("Skipping invalid metadata line: %s", line.strip())
        continue

  return {"processed_files": processed, "failed_files": failed, "total_records": count}


def _append_metadata(metadata_file: Path, protein_id: str, status: str) -> None:
  """Append a single result entry to the metadata file."""
  with metadata_file.open("a") as f:
    f.write(json.dumps({"id": protein_id, "status": status}) + "\n")


def _filter_files(
  pqr_files: list[Path],
  metadata_file: Path,
  *,
  resume: bool,
) -> tuple[list[Path], set[str], set[str]]:
  """Filter file list based on checkpoint metadata."""
  checkpoint_data = _load_checkpoint_metadata(metadata_file)
  processed_files = checkpoint_data["processed_files"]
  failed_files = checkpoint_data["failed_files"]
  files_to_process = pqr_files

  if resume and (processed_files or failed_files):
    logger.info(
      "Resuming from checkpoint: %d processed, %d failed",
      len(processed_files),
      len(failed_files),
    )
    files_to_process = [
      f for f in pqr_files if f.stem not in processed_files and f.stem not in failed_files
    ]
    logger.info("Remaining: %d files to process", len(files_to_process))

  return files_to_process, processed_files, failed_files


def _run_scatter_phase(
  files_to_process: list[Path],
  spec: PreprocessingSpecification,
) -> list[Path]:
  """Run the parallel processing 'scatter' phase.

  Returns:
      A list of paths to the generated shards.
      Logs failures directly to the metadata file.

  """
  logger.info("Starting scatter phase: parallel processing to temporary shards")
  spec.temp_dir.mkdir(parents=True, exist_ok=True)
  worker_args = [(pqr_file, spec, spec.temp_dir) for pqr_file in files_to_process]

  shard_paths = []

  if spec.num_workers == 0:
    logger.info("Running in single-process mode (num_workers=0)")
    for args in tqdm.tqdm(worker_args, desc="Processing proteins"):
      protein_id, shard_path = _worker_process_protein(args)
      if shard_path:
        shard_paths.append(shard_path)
      else:
        _append_metadata(spec.metadata_file, protein_id, "failed")
  else:
    with mp.Pool(processes=spec.num_workers) as pool:
      for protein_id, shard_path in tqdm.tqdm(
        pool.imap_unordered(_worker_process_protein, worker_args),
        total=len(worker_args),
        desc="Processing proteins",
      ):
        if shard_path:
          shard_paths.append(shard_path)
        else:
          _append_metadata(spec.metadata_file, protein_id, "failed")

  logger.info("Scatter phase complete: %d shards created", len(shard_paths))
  return shard_paths


def _cleanup_temp_files(shard_paths: list[Path], temp_dir: Path) -> None:
  """Remove temporary shard files and directory."""
  logger.info("Cleaning up temporary shards")
  try:
    for shard_path in shard_paths:
      shard_path.unlink(missing_ok=True)
    temp_dir.rmdir()
  except OSError as e:
    logger.warning("Could not clean up temp directory %s: %s", temp_dir, e)


def _log_summary_and_get_results(
  spec: PreprocessingSpecification,
  initial_processed: set[str],
  initial_failed: set[str],
) -> dict[str, Any]:
  """Reload metadata, log final summary, and return results dict."""
  final_stats = _load_checkpoint_metadata(spec.metadata_file)
  total_success = final_stats["total_records"]
  total_failed = len(final_stats["failed_files"])

  num_success_this_run = total_success - len(initial_processed)
  num_failed_this_run = total_failed - len(initial_failed)

  logger.info("=" * 60)
  logger.info("Preprocessing complete!")
  logger.info("  Successfully processed (this run): %d", num_success_this_run)
  logger.info("  Failed (this run): %d", num_failed_this_run)
  logger.info("  Total processed (all runs): %d", total_success)
  logger.info("  Total failed (all runs): %d", total_failed)
  logger.info("  Output file: %s", spec.output_file)
  logger.info("  Index file: %s", spec.index_file)
  logger.info("  Metadata file: %s", spec.metadata_file)
  logger.info("=" * 60)

  return {
    "output_file": spec.output_file,
    "index_file": spec.index_file,
    "metadata_file": spec.metadata_file,
    "success_count": total_success,
    "failure_count": total_failed,
  }


def run_preprocessing_pipeline(spec: PreprocessingSpecification) -> dict[str, Any]:
  """Execute the full preprocessing pipeline.

  Args:
      spec: PreprocessingSpecification object.

  Returns:
      A dictionary with execution statistics and paths.

  """
  logger.info("Starting Aminx preprocessing pipeline")
  logger.info("Input dir: %s", spec.input_dir)
  logger.info("Output file: %s", spec.output_file)

  # 1. Discover files
  pqr_files = sorted(spec.input_dir.glob("*.pqr"))
  if not pqr_files:
    msg = f"No PQR files found in {spec.input_dir}"
    raise FileNotFoundError(msg)

  # 2. Filter based on checkpoint
  files_to_process, initial_processed, initial_failed = _filter_files(
    pqr_files,
    spec.metadata_file,
    resume=spec.resume_from_checkpoint,
  )

  if not files_to_process:
    logger.info("All files already processed according to checkpoint. Exiting.")
    return _log_summary_and_get_results(spec, initial_processed, initial_failed)

  # 3. Run scatter phase (parallel processing)
  # Physics features are computed by proxide via OutputSpec
  shard_paths = _run_scatter_phase(files_to_process, spec)

  if not shard_paths:
    logger.error("No shards were successfully created. Exiting.")
    return _log_summary_and_get_results(spec, initial_processed, initial_failed)

  # 5. Run gather phase (merge shards into final array_record)
  logger.info("Starting gather phase: merging %d shards", len(shard_paths))
  writer = ArrayRecordWriter(str(spec.output_file), spec.compression)

  for shard_path in tqdm.tqdm(shard_paths, desc="Merging shards"):
    reader = ArrayRecordReader(str(shard_path))
    for record in reader:
      writer.write(record)
      # Log success to metadata
      record_data = msgpack.unpackb(record)
      _append_metadata(spec.metadata_file, record_data["protein_id"], "success")

  writer.close()

  # 6. Cleanup
  _cleanup_temp_files(shard_paths, spec.temp_dir)

  # 7. Final summary
  return _log_summary_and_get_results(spec, initial_processed, initial_failed)
