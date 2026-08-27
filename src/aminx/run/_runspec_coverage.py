"""Exhaustive coverage and anti-drift gate for RunSpec sub-config fields.

Enforces that every sub-config field on RunSpec is explicitly accounted for:
- SERIALIZATION_ONLY: derived dynamically by executing run_spec_portable_to_dict on a probe spec.
- MIGRATED: canonically consumed by downstream host/kernel execution modules (must name valid reader modules).
- MIRRORED: deliberately duplicated with flat specification fields, with an explicit >=4 word reason.

Mirrors the anti-drift mechanism of src/aminx/host/spec_partition.py.
task_id: 260827_runspec-scaffolding-remediation-spec
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aminx.run.run_spec_portable_json import (
  _placeholder_run_spec,
  run_spec_portable_to_dict,
)
from aminx.run.spec import (
  IOConfig,
  MultistateConfig,
  PlannerTopology,
  PrecisionConfig,
  ResourceConfig,
  SamplingConfig,
)

if TYPE_CHECKING:
  from collections.abc import Callable, Mapping


class RunSpecCoverageError(RuntimeError):
  """A RunSpec sub-config field is unclassified or stale, risking silent drift."""


def _default_subconfig_classes() -> dict[str, type]:
  return {
    "io": IOConfig,
    "resource": ResourceConfig,
    "multistate": MultistateConfig,
    "precision": PrecisionConfig,
    "plan": PlannerTopology,
    "sampling": SamplingConfig,
  }


def _all_subconfig_field_names(
  subconfig_classes: dict[str, type] | None = None,
) -> set[str]:
  classes = subconfig_classes or _default_subconfig_classes()
  names: set[str] = set()
  for sub_name, cls in classes.items():
    for f in fields(cls):
      names.add(f"{sub_name}.{f.name}")
  return names


def _derive_serialization_only(
  to_dict_fn: Callable[[Any], dict[str, Any]] | None = None,
) -> set[str]:
  serializer = to_dict_fn or run_spec_portable_to_dict
  probe = _placeholder_run_spec(
    io=IOConfig(
      output_dir=Path("_probe_output"),
      sink_kind="zarr",
      manifest_path=Path("_probe_manifest.csv"),
      output_h5_path=None,
      cache_path=None,
    ),
    multistate=MultistateConfig(mode="single", n_states=1, combine_strategy="mean"),
    resource=ResourceConfig(
      n_devices=1,
      sample_batch_size=1,
      structure_batch_size=1,
      max_buffer_size=None,
    ),
    precision=PrecisionConfig(compute="fp32"),
  )
  dumped = serializer(probe)
  derived: set[str] = set()
  for block, value in dumped.items():
    if isinstance(value, dict):
      for field_name in value:
        derived.add(f"{block}.{field_name}")
  return derived


# Canonical migrated fields with named reader modules.
MIGRATED_FIELDS: Mapping[str, tuple[str, ...]] = {
  "io.output_h5_path": ("aminx.host.streaming", "aminx.host.runner", "aminx.sampling.multistate_poe"),
  "io.cache_path": ("aminx.host.prep",),
  "plan.use_unified_driver": ("aminx.host.kernel_dispatch",),
  "sampling.num_samples": ("aminx.host.kernel_dispatch", "aminx.host.plan"),
  "sampling.random_seed": ("aminx.host.kernel_dispatch",),
  "sampling.return_logits": ("aminx.host.kernel_dispatch",),
  "sampling.compute_pseudo_perplexity": ("aminx.host.kernel_dispatch",),
  "sampling.return_decoding_orders": ("aminx.host.kernel_dispatch",),
  "sampling.return_logit_fingerprint": ("aminx.host.kernel_dispatch",),
  "sampling.backbone_noise": ("aminx.host.kernel_dispatch",),
  "sampling.temperature": ("aminx.host.kernel_dispatch",),
  "sampling.bias": ("aminx.host.kernel_dispatch",),
  "sampling.fixed_mask": ("aminx.host.kernel_dispatch",),
  "sampling.fixed_positions": ("aminx.host.kernel_dispatch",),
  "sampling.fixed_tokens": ("aminx.host.kernel_dispatch",),
  "sampling.sampling_strategy": ("aminx.host.plan",),
  "sampling.multi_state_strategy": ("aminx.host.plan",),
  "sampling.multi_state_temperature": ("aminx.host.plan",),
  "sampling.multi_state_sharpness": ("aminx.host.plan",),
  "sampling.use_rolling_state": ("aminx.host.plan",),
  "sampling.decoding_order_fn": ("aminx.host.kernel_dispatch",),
  "sampling.tie_group_map": ("aminx.host.kernel_dispatch",),
  "sampling.state_position_map": ("aminx.host.kernel_dispatch",),
  "sampling.state_weights": ("aminx.host.kernel_dispatch",),
  "sampling.structure_mapping": ("aminx.host.kernel_dispatch",),
}

# Deliberately mirrored fields requiring explicit mechanism reasons (>=4 words).
MIRRORED_FIELDS: Mapping[str, str] = {}


def assert_runspec_coverage_is_exhaustive(
  subconfig_classes: dict[str, type] | None = None,
  to_dict_fn: Callable[[Any], dict[str, Any]] | None = None,
  migrated: Mapping[str, tuple[str, ...]] | None = None,
  mirrored: Mapping[str, str] | None = None,
) -> None:
  """Ensure every RunSpec sub-config field is classified. Raise RunSpecCoverageError otherwise."""
  actual = _all_subconfig_field_names(subconfig_classes)
  serialization_only = _derive_serialization_only(to_dict_fn)
  migrated_map = MIGRATED_FIELDS if migrated is None else migrated
  mirrored_map = MIRRORED_FIELDS if mirrored is None else mirrored

  classified = serialization_only | set(migrated_map) | set(mirrored_map)
  unclassified = actual - classified
  if unclassified:
    msg = (
      f"RunSpec sub-config field(s) classified nowhere: {sorted(unclassified)!r}. "
      f"Every field must be derived via serialization, listed in MIGRATED_FIELDS with reader "
      f"modules, or listed in MIRRORED_FIELDS with a substantial mechanism reason."
    )
    raise RunSpecCoverageError(msg)

  stale = (set(migrated_map) | set(mirrored_map)) - actual
  if stale:
    msg = f"Classified name(s) that are no longer RunSpec sub-config fields: {sorted(stale)!r}"
    raise RunSpecCoverageError(msg)

  vague = [name for name, reason in mirrored_map.items() if len(reason.split()) < 4]
  if vague:
    msg = f"MIRRORED_FIELDS entries needing a real reason (>=4 words): {vague!r}"
    raise RunSpecCoverageError(msg)


# Execute gate at import time to prevent silent drift.
assert_runspec_coverage_is_exhaustive()
