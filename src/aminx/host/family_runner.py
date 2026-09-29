"""Host loop for a registered FamilyDriver.

One batch times one sample chunk is in memory at a time. Pad rows are sliced
off on the host before they reach the sink or the in-memory result. This
module never calls ``prep_protein_stream_and_model`` and never applies
``eqx.nn.inference_mode``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
from xtrax.run import ZarrStagingSink, derive_sink_spec
from xtrax.tiling import AxisSpec

from aminx.host._sampling_helper import _canonical_structure_ids_for_spec
from aminx.host.family_driver import ALPHABET, FamilyBatch, FamilyDriver, SinkArraySpec
from aminx.host.plan import resolve_chunk_size, resolve_target_samples  # noqa: TID251
from aminx.tiling.dispatch import make_axis_dispatch_via_xtrax
from aminx.tiling.strategy import SafeMap

_PROOFREAD_PURPOSES = frozenset(
  {
    "score:proofread_unconditional",
    "score:proofread_conditional",
  },
)


def _output_kind(purpose: str) -> str:
  """``score:<output_kind>`` suffix. Non-score purposes carry an empty kind."""
  if purpose.startswith("score:"):
    return purpose.split(":", 1)[1]
  return ""


def _scalar_dropout(spec: Any, purpose: str) -> bool:  # noqa: ANN401
  """LASEr proofreading keeps scalar dropout on unless the option says otherwise.

  ``LaserOptions`` is T0.6. Until that field exists, the upstream default
  (dropout enabled) applies to the two proofreading purposes only.
  """
  if purpose not in _PROOFREAD_PURPOSES:
    return False
  laser = getattr(spec, "laser", None)
  if laser is None:
    return True
  return bool(getattr(laser, "proofread_dropout", True))


def _dropout_key(base_key: jax.Array, purpose: str, chunk_start: int) -> jax.Array:
  """Host key handed to the stage.

  Unconditional proofreading is ``fold_in(key, 0)``. Conditional proofreading
  receives the base key; the stage folds focus, dropout, and order indices.
  Other purposes fold the chunk start so chunks do not share a dropout mask.
  """
  if purpose == "score:proofread_unconditional":
    return jax.random.fold_in(base_key, 0)
  if purpose == "score:proofread_conditional":
    return base_key
  return jax.random.fold_in(base_key, int(chunk_start))


def _chunk_windows(spec: Any, purpose: str) -> list[tuple[int, int]]:  # noqa: ANN401
  """``(chunk_start, chunk_count)`` windows. Non-sample purposes are one window."""
  if purpose != "sample":
    return [(0, 1)]
  total = resolve_target_samples(spec)
  size = resolve_chunk_size(spec, total, None)
  windows: list[tuple[int, int]] = []
  start = 0
  while start < total:
    count = min(size, total - start)
    windows.append((start, count))
    start += count
  return windows


def _bind_declared_axes(axes: list[AxisSpec]) -> None:
  """Bind each declared axis through xtrax. Drivers must not vmap these."""
  for axis in axes:
    tile = axis.default_batch_size if axis.default_batch_size > 0 else 1
    make_axis_dispatch_via_xtrax(SafeMap(tile=tile), axis=axis.name)


def _structure_id(structure_ids: list[str], index: int) -> str:
  if 0 <= index < len(structure_ids):
    return structure_ids[index]
  return f"structure_{index}"


def _check_chunk(
  produced: Mapping[str, Any],
  schema: Mapping[str, SinkArraySpec],
  *,
  n_structures: int,
) -> None:
  expected = set(schema)
  got = set(produced)
  if got != expected:
    msg = f"result_schema mismatch: expected keys {sorted(expected)}, got {sorted(got)}"
    raise RuntimeError(msg)
  for name, array_spec in schema.items():
    batched = np.asarray(produced[name])
    if batched.shape[0] != n_structures:
      msg = (
        f"result_schema mismatch for {name!r}: leading axis {batched.shape[0]} "
        f"!= {n_structures} structures"
      )
      raise RuntimeError(msg)
    if batched.ndim != len(array_spec.dims) + 1:
      msg = (
        f"result_schema mismatch for {name!r}: ndim {batched.ndim} != "
        f"{len(array_spec.dims) + 1} (leading structure axis plus {array_spec.dims})"
      )
      raise RuntimeError(msg)
    if batched.dtype != np.dtype(array_spec.dtype):
      msg = f"result_schema mismatch for {name!r}: dtype {batched.dtype} != {array_spec.dtype}"
      raise RuntimeError(msg)


def _slice_l_total(array: np.ndarray, dims: tuple[str, ...], length: int) -> np.ndarray:
  """Drop pad rows on every axis whose schema dim is ``L_total``."""
  indexer = [slice(0, length) if dim == "L_total" else slice(None) for dim in dims]
  return np.asarray(array[tuple(indexer)])


def _refuse_driver_modes(spec: Any) -> None:  # noqa: ANN401
  """Grid, campaign, and multistate fusion are not driver semantics in v1."""
  if getattr(spec, "grid_mode", False):
    msg = "family drivers do not support grid_mode in v1"
    raise ValueError(msg)
  if getattr(spec, "campaign_mode", False):
    msg = "family drivers do not support campaign_mode in v1"
    raise ValueError(msg)
  if getattr(spec, "state_position_map", None) is not None:
    msg = "family drivers do not support state_position_map in v1"
    raise ValueError(msg)


def _chunk_schema(schema: Mapping[str, SinkArraySpec]) -> dict[str, SinkArraySpec]:
  """Per-chunk arrays. ``level=structure`` entries are staged once, later."""
  return {
    name: array_spec
    for name, array_spec in schema.items()
    if array_spec.attrs.get("level") != "structure"
  }


def run_family_driver(
  driver: FamilyDriver,
  spec: Any,  # noqa: ANN401
  purpose: str,
) -> dict[str, Any]:
  """Run ``driver`` for ``purpose`` and stage or concatenate per structure.

  When ``output_h5_path`` is set, arrays land in a ``ZarrStagingSink`` built
  with ``derive_sink_spec`` the same way ``host/streaming.py`` builds its sink.
  Otherwise chunks are concatenated within each structure and returned.
  """
  _refuse_driver_modes(spec)
  model = driver.load(spec)
  stages = driver.stages(spec, purpose, model)
  schema = _chunk_schema(dict(driver.result_schema(spec, purpose)))
  structure_ids = _canonical_structure_ids_for_spec(spec)
  windows = _chunk_windows(spec, purpose)
  output_kind = _output_kind(purpose)
  scalar_dropout = _scalar_dropout(spec, purpose)
  base_key = jax.random.PRNGKey(int(spec.random_seed))

  output_path = spec.run_spec.io.output_h5_path
  sink: ZarrStagingSink | None = None
  if output_path is not None:
    sink = ZarrStagingSink(
      derive_sink_spec(
        spec.run_spec,
        output_dir=Path(output_path),
        format="zarr",
        flush_every=1,
      ),
    )

  skipped: list[dict[str, int | str]] = []
  memory: dict[int, dict[str, list[np.ndarray]]] = {}
  seen_index = -1

  for batch in driver.batches(spec):
    _consume_batch(
      driver,
      spec,
      purpose,
      batch,
      stages=stages,
      schema=schema,
      structure_ids=structure_ids,
      windows=windows,
      scalar_dropout=scalar_dropout,
      base_key=base_key,
      sink=sink,
      skipped=skipped,
      memory=memory,
      seen_index=seen_index,
    )
    seen_index = _last_index(batch, seen_index)

  root_attrs: dict[str, Any] = {
    "schema_version": f"{driver.name}_v1",
    "model_family": str(spec.model_family),
    "purpose": purpose,
    "output_kind": output_kind,
    "alphabet": ALPHABET,
    "skipped_inputs": skipped,
  }
  if sink is not None:
    sink.stage((), attrs=root_attrs)
    sink.finalize()

  structures = _assemble_memory(memory, structure_ids) if sink is None else {}
  return {
    "schema_version": root_attrs["schema_version"],
    "model_family": spec.model_family,
    "purpose": purpose,
    "output_kind": output_kind,
    "skipped_inputs": skipped,
    "structures": structures,
  }


def _last_index(batch: FamilyBatch, seen_index: int) -> int:
  if not batch.input_indices:
    return seen_index
  return batch.input_indices[-1]


def _note_skipped(
  batch: FamilyBatch,
  structure_ids: list[str],
  skipped: list[dict[str, int | str]],
) -> None:
  for index, reason in batch.skipped:
    skipped.append(
      {
        "input_index": int(index),
        "structure_id": _structure_id(structure_ids, int(index)),
        "reason": str(reason),
      },
    )


def _require_increasing(batch: FamilyBatch, seen_index: int) -> None:
  previous = seen_index
  for index in batch.input_indices:
    if index <= previous:
      msg = (
        "FamilyBatch.input_indices must be strictly increasing across the "
        f"batch stream (saw {index} after {previous})"
      )
      raise ValueError(msg)
    previous = index
  if len(batch.lengths) != len(batch.input_indices):
    msg = "FamilyBatch.lengths must have one L_total per input_indices entry"
    raise ValueError(msg)


def _consume_batch(
  driver: FamilyDriver,
  spec: Any,  # noqa: ANN401
  purpose: str,
  batch: FamilyBatch,
  *,
  stages: Any,  # noqa: ANN401
  schema: Mapping[str, SinkArraySpec],
  structure_ids: list[str],
  windows: list[tuple[int, int]],
  scalar_dropout: bool,
  base_key: jax.Array,
  sink: ZarrStagingSink | None,
  skipped: list[dict[str, int | str]],
  memory: dict[int, dict[str, list[np.ndarray]]],
  seen_index: int,
) -> None:
  _bind_declared_axes(list(driver.axes(spec, purpose, batch)))
  _note_skipped(batch, structure_ids, skipped)
  _require_increasing(batch, seen_index)
  if not batch.input_indices:
    return
  n_structures = len(batch.input_indices)
  for chunk_start, chunk_count in windows:
    produced = stages(
      batch,
      chunk_start=chunk_start,
      chunk_count=chunk_count,
      scalar_dropout=scalar_dropout,
      dropout_key=_dropout_key(base_key, purpose, chunk_start),
    )
    _check_chunk(produced, schema, n_structures=n_structures)
    for local, index in enumerate(batch.input_indices):
      length = batch.lengths[local]
      sliced: dict[str, np.ndarray] = {}
      for name, array_spec in schema.items():
        batched = np.asarray(produced[name])
        sliced[name] = _slice_l_total(batched[local], array_spec.dims, length)
      if sink is not None:
        # ``**`` of a dict[str, ndarray] is the array payload. ty binds that splat
        # to ``attrs``; pass the dict through a helper it can see as ``**arrays``.
        _stage_chunk(sink, (f"structure_{index}", str(chunk_start)), sliced)
      else:
        bucket = memory.setdefault(index, {name: [] for name in sliced})
        for name, array in sliced.items():
          bucket[name].append(array)
  if sink is None:
    return
  for index in batch.input_indices:
    sink.stage(
      (f"structure_{index}",),
      attrs={
        "structure_index": int(index),
        "structure_id": _structure_id(structure_ids, index),
      },
    )


def _stage_chunk(
  sink: ZarrStagingSink,
  key: tuple[str, ...],
  arrays: dict[str, np.ndarray],
) -> None:
  # ty binds a ``dict[str, ndarray]`` splat to ``attrs`` rather than ``**arrays``.
  stage = cast("Any", sink.stage)
  stage(key, **arrays)


def _assemble_memory(
  memory: dict[int, dict[str, list[np.ndarray]]],
  structure_ids: list[str],
) -> dict[str, dict[str, Any]]:
  structures: dict[str, dict[str, Any]] = {}
  for index, parts in memory.items():
    arrays = {
      name: np.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]
      for name, pieces in parts.items()
    }
    structures[str(index)] = {
      "structure_id": _structure_id(structure_ids, index),
      "arrays": arrays,
    }
  return structures
