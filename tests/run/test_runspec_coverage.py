"""Tests for RunSpec sub-config coverage gate and anti-drift mechanism."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import pytest

from aminx.run._runspec_coverage import (
  MIGRATED_FIELDS,
  RunSpecCoverageError,
  _default_subconfig_classes,
  assert_runspec_coverage_is_exhaustive,
)


def test_runspec_coverage_passes_on_current_tree() -> None:
  """Gate 1: Default execution on the current codebase passes cleanly."""
  import aminx.run  # noqa: F401

  assert_runspec_coverage_is_exhaustive()


def test_unclassified_field_raises_coverage_error() -> None:
  """Gate 2: Adding a synthetic unclassified field raises RunSpecCoverageError."""
  @dataclass
  class SyntheticExtraConfig:
    dummy_field: int = 1
    unclassified_extra: str = "danger"

  subconfigs = dict(_default_subconfig_classes())
  subconfigs["synthetic"] = SyntheticExtraConfig

  with pytest.raises(RunSpecCoverageError, match="classified nowhere.*synthetic.dummy_field"):
    assert_runspec_coverage_is_exhaustive(subconfig_classes=subconfigs)


def test_stale_classification_raises_coverage_error() -> None:
  """Gate 3: Retaining a classification for a non-existent field raises RunSpecCoverageError."""
  stale_migrated = dict(MIGRATED_FIELDS)
  stale_migrated["sampling.non_existent_ghost_field"] = ("aminx.host.plan",)

  with pytest.raises(RunSpecCoverageError, match="no longer RunSpec sub-config fields"):
    assert_runspec_coverage_is_exhaustive(migrated=stale_migrated)


def test_vague_reason_raises_coverage_error() -> None:
  """Gate 4: A mirrored reason with < 4 words raises RunSpecCoverageError."""
  vague_mirrored = {"io.cache_path": "too short"}

  # Even if classified, vague reasons are rejected
  with pytest.raises(RunSpecCoverageError, match="needing a real reason"):
    assert_runspec_coverage_is_exhaustive(mirrored=vague_mirrored)


def test_derived_bucket_coupling_with_serializer() -> None:
  """Gate 5: Dropping a block in serialization dynamically exposes unclassified fields."""
  def mock_serializer(probe: Any) -> dict[str, Any]:
    # Deliberately drop the 'multistate' block from serialization output
    return {
      "version": 2,
      "io": {"sink_kind": "zarr", "output_dir": None, "manifest_path": None},
      "resource": {
        "n_devices": 1,
        "sample_batch_size": 1,
        "structure_batch_size": 1,
        "max_buffer_size": None,
      },
      "precision": {"compute": "fp32"},
    }

  with pytest.raises(RunSpecCoverageError, match="classified nowhere.*multistate.mode"):
    assert_runspec_coverage_is_exhaustive(to_dict_fn=mock_serializer)


def test_migrated_reader_modules_are_importable() -> None:
  """Ensure every module listed in MIGRATED_FIELDS actually exists and is importable."""
  distinct_modules = {mod for readers in MIGRATED_FIELDS.values() for mod in readers}
  assert distinct_modules, "MIGRATED_FIELDS must not be empty"

  for mod_name in sorted(distinct_modules):
    mod = importlib.import_module(mod_name)
    assert mod is not None, f"Could not import {mod_name}"
