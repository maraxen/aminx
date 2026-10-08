"""JSON-safe round-trip for a **subset** of :class:`RunSpec` static fields.

Full Equinox / PyTree serialization is out of scope. Wire format versions:

- **v1** (legacy): ``multistate``, ``resource``, ``precision`` only; ``io`` is a
  placeholder on deserialize.
- **v2** (current): adds ``io`` (``sink_kind``, optional ``output_dir`` / ``manifest_path``
  as POSIX strings or null). Other sub-configs remain **placeholders** aligned with
  :func:`build_run_spec` defaults for a minimal single-GPU
  :class:`~aminx.run.specs.RunSpecification`-style inference stub.

:func:`run_spec_portable_to_dict` always emits **v2**. Note: :func:`run_spec_portable_to_dict`
raises :exc:`ValueError` when serializing specs with ``grid.grid_mode=True`` or
``ligand.model_family`` other than ``'proteinmpnn'`` (not representable in v2 wire
format until v3 is defined); use the full spec_json (campaign path) for such specs.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from .spec import (
  GridLineageConfig,
  IOConfig,
  LigandConfig,
  MultistateConfig,
  PlannerTopology,
  PrecisionConfig,
  ResourceConfig,
  RunSpec,
)

PORTABLE_RUN_SPEC_VERSION = 2
PORTABLE_RUN_SPEC_LEGACY_VERSION = 1
_SUPPORTED_PORTABLE_VERSIONS = frozenset(
  {PORTABLE_RUN_SPEC_VERSION, PORTABLE_RUN_SPEC_LEGACY_VERSION},
)

_PrecisionLabel = Literal["fp32", "fp16", "bf16"]


def _as_int(path: str, value: object) -> int:
  if type(value) is not int:
    msg = f"[portable_run_spec] {path}: expected int, got {type(value).__name__}"
    raise TypeError(msg)
  return value


def _optional_path_str(path: str, value: object) -> Path | None:
  if value is None:
    return None
  if type(value) is not str:
    msg = f"[portable_run_spec] {path}: expected str or null, got {type(value).__name__}"
    raise TypeError(msg)
  return Path(value)


def _path_to_json_str(p: Path | None) -> str | None:
  if p is None:
    return None
  return p.as_posix()


def _as_optional_positive_int(path: str, value: object) -> int | None:
  if value is None:
    return None
  if type(value) is not int:
    msg = f"[portable_run_spec] {path}: expected int or null, got {type(value).__name__}"
    raise TypeError(msg)
  if value <= 0:
    msg = f"[portable_run_spec] {path}: expected positive int or null, got {value}"
    raise ValueError(msg)
  return value


def _as_str(path: str, value: object) -> str:
  if not isinstance(value, str):
    msg = f"[portable_run_spec] {path}: expected str, got {type(value).__name__}"
    raise TypeError(msg)
  return value


def _as_mapping(path: str, value: object) -> Mapping[str, Any]:
  if not isinstance(value, Mapping):
    msg = f"[portable_run_spec] {path}: expected object, got {type(value).__name__}"
    raise TypeError(msg)
  return cast("Mapping[str, Any]", value)


def _placeholder_run_spec(
  *,
  multistate: MultistateConfig,
  resource: ResourceConfig,
  precision: PrecisionConfig,
  io: IOConfig | None = None,
) -> RunSpec:
  """Defaults mirror :func:`build_run_spec` for a bare ``RunSpecification``-like spec."""
  io_final = (
    io
    if io is not None
    else IOConfig(
      output_dir=None,
      sink_kind="none",
      manifest_path=None,
      output_h5_path=None,
      cache_path=None,
    )
  )
  ligand = LigandConfig(
    model_family="proteinmpnn",
    use_side_chain_context=None,
    ligand_conditioning=False,
    sidechain_conditioning=False,
    context_path=None,
  )
  grid = GridLineageConfig(
    grid_mode=False,
    campaign_mode=False,
    job_id=None,
    chunk_id=None,
    sample_start=None,
    sample_count=None,
  )
  plan = PlannerTopology(use_unified_driver=True)
  return RunSpec(
    seed=0,
    axes=[],
    carry_specs={},
    boundaries=None,
    io=io_final,
    resource=resource,
    multistate=multistate,
    ligand=ligand,
    grid=grid,
    precision=precision,
    plan=plan,
  )


def run_spec_portable_to_dict(run_spec: RunSpec) -> dict[str, Any]:
  """Export JSON-native fields from ``run_spec`` (static scalar subset only)."""
  # Guard: grid_mode runs are not representable in v2 portable format
  if run_spec.grid.grid_mode:
    msg = (
      "[portable_run_spec] grid_mode runs are not representable in the v2 portable "
      "wire format (v3 not yet defined); use the full spec_json (campaign) path instead."
    )
    raise ValueError(msg)

  # Guard: only proteinmpnn is representable in v2. ligandmpnn and the driver
  # families carry no family field on the wire and would round-trip as proteinmpnn.
  family = run_spec.ligand.model_family
  if family != "proteinmpnn":
    msg = (
      f"[portable_run_spec] {family} runs are not representable in the v2 portable "
      "wire format (v3 not yet defined); use the full spec_json (campaign) path instead."
    )
    raise ValueError(msg)

  return {
    "version": PORTABLE_RUN_SPEC_VERSION,
    "io": {
      "sink_kind": run_spec.io.sink_kind,
      "output_dir": _path_to_json_str(run_spec.io.output_dir),
      "manifest_path": _path_to_json_str(run_spec.io.manifest_path),
    },
    "multistate": {
      "mode": run_spec.multistate.mode,
      "n_states": run_spec.multistate.n_states,
      "combine_strategy": run_spec.multistate.combine_strategy,
    },
    "resource": {
      "n_devices": run_spec.resource.n_devices,
      "sample_batch_size": run_spec.resource.sample_batch_size,
      "structure_batch_size": run_spec.resource.structure_batch_size,
      "max_buffer_size": run_spec.resource.max_buffer_size,
    },
    "precision": {"compute": run_spec.precision.compute},
  }


def run_spec_portable_from_dict(data: Mapping[str, Any]) -> RunSpec:
  """Parse portable dict; unknown top-level keys are ignored."""
  raw_ver = data.get("version", None)
  if type(raw_ver) is not int or raw_ver not in _SUPPORTED_PORTABLE_VERSIONS:
    msg = f"[portable_run_spec] version: expected int in {_SUPPORTED_PORTABLE_VERSIONS!r}, got {raw_ver!r}"
    raise ValueError(msg)
  wire_ver = raw_ver

  for key in ("multistate", "resource", "precision"):
    if key not in data:
      msg = f"[portable_run_spec] missing required key {key!r}"
      raise ValueError(msg)

  ms = _as_mapping("multistate", data["multistate"])
  for sub in ("mode", "n_states", "combine_strategy"):
    if sub not in ms:
      msg = f"[portable_run_spec] multistate.{sub}: required key missing"
      raise ValueError(msg)
  multistate = cast(
    "MultistateConfig",
    MultistateConfig(
      mode=_as_str("multistate.mode", ms["mode"]),
      n_states=_as_int("multistate.n_states", ms["n_states"]),
      combine_strategy=_as_str("multistate.combine_strategy", ms["combine_strategy"]),
    ),
  )

  res = _as_mapping("resource", data["resource"])
  for sub in ("n_devices", "sample_batch_size", "structure_batch_size"):
    if sub not in res:
      msg = f"[portable_run_spec] resource.{sub}: required key missing"
      raise ValueError(msg)
  mbs_raw = res.get("max_buffer_size", None)
  resource = cast(
    "ResourceConfig",
    ResourceConfig(
      n_devices=_as_int("resource.n_devices", res["n_devices"]),
      sample_batch_size=_as_int("resource.sample_batch_size", res["sample_batch_size"]),
      structure_batch_size=_as_int("resource.structure_batch_size", res["structure_batch_size"]),
      max_buffer_size=_as_optional_positive_int("resource.max_buffer_size", mbs_raw),
    ),
  )

  prec = _as_mapping("precision", data["precision"])
  if "compute" not in prec:
    msg = "[portable_run_spec] precision.compute: required key missing"
    raise ValueError(msg)
  compute_raw = prec["compute"]
  compute = _as_str("precision.compute", compute_raw)
  if compute not in ("fp32", "fp16", "bf16"):
    msg = f"[portable_run_spec] precision.compute: invalid {compute!r}"
    raise ValueError(msg)
  precision = cast("PrecisionConfig", PrecisionConfig(compute=cast("_PrecisionLabel", compute)))

  io_cfg: IOConfig | None = None
  if wire_ver == PORTABLE_RUN_SPEC_VERSION:
    if "io" not in data:
      msg = "[portable_run_spec] missing required key 'io' (required for wire format v2)"
      raise ValueError(msg)
    iom = _as_mapping("io", data["io"])
    if "sink_kind" not in iom:
      msg = "[portable_run_spec] io.sink_kind: required key missing"
      raise ValueError(msg)
    io_cfg = cast(
      "IOConfig",
      IOConfig(
        output_dir=_optional_path_str("io.output_dir", iom.get("output_dir", None)),
        sink_kind=_as_str("io.sink_kind", iom["sink_kind"]),
        manifest_path=_optional_path_str("io.manifest_path", iom.get("manifest_path", None)),
        output_h5_path=None,
        cache_path=None,
      ),
    )

  return _placeholder_run_spec(
    multistate=multistate,
    resource=resource,
    precision=precision,
    io=io_cfg,
  )
