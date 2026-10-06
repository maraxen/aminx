"""Turn runner result dicts into JSON summaries plus an optional npz side file.

Scalars and arrays at or under :data:`DEFAULT_INLINE_CAP` are inlined as
JSON. Larger arrays are written once to ``<output_dir>/<run_id>.npz``.
Sampled tokens are decoded with aminx's alphabet and trimmed to the real
residue length when that length is known.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from aminx.agent.provenance import capture
from aminx.agent.requests import spec_to_json_dict
from aminx.run.spec_json import SpecJSONEncodeError
from aminx.utils.aa_convert import protein_sequence_to_string

if TYPE_CHECKING:
  from aminx.run.specs import RunSpecification

DEFAULT_INLINE_CAP = 10_000

_SKIP_ROUTE = frozenset({"metadata", "schema_version"})
_KINDS = frozenset({"sample", "score", "inspect", "jacobian"})
_SEQUENCE_NDIM = 5


def default_output_dir(environ: Mapping[str, str] | None = None) -> Path:
  """Return the directory for agent side files.

  Precedence matches the XDG cache rule used for CLI input caches:
  ``$AMINX_AGENT_OUTPUT_DIR``, else ``$XDG_CACHE_HOME/aminx/agent``, else
  ``~/.cache/aminx/agent``.

  Parameters
  ----------
  environ : mapping or None, optional
    Environment to read. ``None`` reads :data:`os.environ` and uses
    :meth:`Path.home` for the last resort.

  Returns
  -------
  Path
    Directory path. The directory is not created.
  """
  if environ is None:
    explicit = os.environ.get("AMINX_AGENT_OUTPUT_DIR")
    xdg = os.environ.get("XDG_CACHE_HOME")
    home = Path.home()
  else:
    explicit = environ.get("AMINX_AGENT_OUTPUT_DIR")
    xdg = environ.get("XDG_CACHE_HOME")
    home_text = environ.get("HOME")
    home = Path(home_text) if home_text else Path.home()
  if explicit:
    return Path(explicit)
  if xdg:
    return Path(xdg) / "aminx" / "agent"
  return home / ".cache" / "aminx" / "agent"


def new_run_id() -> str:
  """Return a new run id of a UTC timestamp plus 8 hex characters.

  Returns
  -------
  str
    ``YYYYMMDDTHHMMSSZ-`` followed by 8 lowercase hex characters.
  """
  stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
  return f"{stamp}-{secrets.token_hex(4)}"


def to_jsonable(value: object) -> object:
  """Convert ``value`` into a JSON-compatible object.

  Numpy and JAX scalars become Python scalars. Arrays become nested lists.
  Named tuples become dicts. Paths become strings. Specification dataclasses
  are encoded with :func:`aminx.agent.requests.spec_to_json_dict`.

  Parameters
  ----------
  value : object
    Value to convert.

  Returns
  -------
  object
    A value :func:`json.dumps` can encode.
  """
  named = _namedtuple_items(value)
  scalar = getattr(value, "item", None)
  if value is None or isinstance(value, (str, int, float, bool)):
    result: object = value
  elif isinstance(value, Path):
    result = str(value)
  elif isinstance(value, np.generic):
    result = value.item()
  elif named is not None:
    result = {name: to_jsonable(item) for name, item in named}
  elif _is_array(value):
    result = np.asarray(value).tolist()
  elif callable(scalar) and getattr(value, "shape", None) == ():
    result = to_jsonable(scalar())
  elif is_dataclass(value) and not isinstance(value, type):
    result = _dataclass_json(value)
  elif isinstance(value, Mapping):
    result = {str(key): to_jsonable(item) for key, item in value.items()}
  elif isinstance(value, (list, tuple)):
    result = [to_jsonable(item) for item in value]
  else:
    result = str(value)
  return result


def _dataclass_json(value: object) -> object:
  """Encode a dataclass, falling back to a field walk when it is not a spec."""
  try:
    return spec_to_json_dict(value)  # type: ignore[arg-type]
  except (SpecJSONEncodeError, TypeError):
    return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}


def shape_result(
  kind: str,
  results: Mapping[str, Any],
  *,
  spec: RunSpecification,
  lengths: Mapping[str, int] | None,
  output_dir: Path,
  run_id: str,
  inline_cap: int = DEFAULT_INLINE_CAP,
  provenance: Mapping[str, object] | None = None,
) -> dict[str, Any]:
  """Shape a runner result into a JSON document and an optional npz file.

  Parameters
  ----------
  kind : str
    ``sample``, ``score``, ``inspect``, or ``jacobian``.
  results : mapping
    Dict returned by the matching ``host.runner`` function.
  spec : RunSpecification
    Specification that produced ``results``.
  lengths : mapping or None
    Residue counts keyed by structure id. A missing id leaves sequences
    untrimmed and adds a warning.
  output_dir : Path
    Directory for the side file. Created when a side file is written.
  run_id : str
    File stem for ``<output_dir>/<run_id>.npz``.
  inline_cap : int, optional
    Maximum array size (number of elements) kept inline.
  provenance : mapping or None, optional
    Provenance object to record. ``None`` calls
    :func:`aminx.agent.provenance.capture`.

  Returns
  -------
  dict
    JSON-serializable summary. ``side_file`` is ``None`` when every array
    fit under ``inline_cap``.
  """
  if kind not in _KINDS:
    msg = f"unknown result kind {kind!r}"
    raise ValueError(msg)
  metadata = _metadata(results)
  structure_ids = _structure_ids(metadata)
  warnings: list[str] = []
  structures = _structures_for(
    kind,
    results,
    spec=spec,
    lengths=lengths,
    structure_ids=structure_ids,
    warnings=warnings,
  )
  store: dict[str, np.ndarray] = {}
  routed = _route_results(results, store, inline_cap, structure_ids)
  spec_dict = _spec_dict(spec)
  recorded = provenance if provenance is not None else capture()
  summary: dict[str, Any] = {
    "kind": kind,
    "run_id": run_id,
    "schema_version": to_jsonable(results.get("schema_version")),
    "structure_ids": structure_ids,
    "skipped_inputs": to_jsonable(metadata.get("skipped_inputs", [])),
    "spec": spec_dict,
    "spec_sha256": _sha256(spec_dict),
    "aminx_version": _aminx_version(),
    "provenance": to_jsonable(dict(recorded)),
    "warnings": warnings,
  }
  summary.update(routed)
  if structures is not None:
    summary["structures"] = structures
  fused = metadata.get("fused_structure_ids")
  if fused is not None:
    summary["fused_structure_ids"] = to_jsonable(fused)
  if "lineage" in metadata:
    summary["lineage"] = to_jsonable(metadata["lineage"])
  summary["side_file"] = _write_side_file(output_dir, run_id, store)
  summary["warnings"] = warnings
  return summary


def _structures_for(
  kind: str,
  results: Mapping[str, Any],
  *,
  spec: RunSpecification,
  lengths: Mapping[str, int] | None,
  structure_ids: Sequence[str],
  warnings: list[str],
) -> list[dict[str, object]] | None:
  """Build per-structure sample or score records, when the result has them."""
  if kind == "sample" and "sequences" in results and "output_zarr_path" not in results:
    return _sample_structures(
      results,
      lengths=lengths,
      structure_ids=structure_ids,
      warnings=warnings,
    )
  if kind == "score" and "scores" in results:
    return _score_structures(
      results["scores"],
      spec,
      structure_ids=structure_ids,
      warnings=warnings,
    )
  return None


def _sample_structures(
  results: Mapping[str, Any],
  *,
  lengths: Mapping[str, int] | None,
  structure_ids: Sequence[str],
  warnings: list[str],
) -> list[dict[str, object]]:
  """Decode sampled tokens into trimmed sequence records."""
  array = np.asarray(results["sequences"])
  if array.ndim != _SEQUENCE_NDIM:
    warnings.append(
      f"sample sequences have ndim {array.ndim}; expected 5 (B, N, noise, temperature, L)",
    )
    return []
  batch, n_samples, n_noise, n_temp, _length = (int(dim) for dim in array.shape)
  perplexity = _aligned_perplexity(results.get("pseudo_perplexity"), array.shape[:4], warnings)
  sample_indices = results.get("sample_indices")
  index_array = None if sample_indices is None else np.asarray(sample_indices)
  noted: set[str] = set()
  structures: list[dict[str, object]] = []
  for batch_index in range(batch):
    structure_id = _structure_id_at(structure_ids, batch_index)
    known = _known_length(structure_id, lengths, warnings, noted)
    samples: list[dict[str, object]] = []
    for sample_slot in range(n_samples):
      for noise_index in range(n_noise):
        for temperature_index in range(n_temp):
          text = protein_sequence_to_string(
            array[batch_index, sample_slot, noise_index, temperature_index],
          )
          sample: dict[str, object] = {
            "sample_index": _sample_index(index_array, sample_slot),
            "noise_index": noise_index,
            "temperature_index": temperature_index,
            "sequence": text if known is None else text[:known],
          }
          if perplexity is not None:
            sample["pseudo_perplexity"] = float(
              perplexity[batch_index, sample_slot, noise_index, temperature_index],
            )
          samples.append(sample)
    structures.append(
      {
        "structure_id": structure_id,
        "length": known,
        "trimmed": known is not None,
        "samples": samples,
      },
    )
  return structures


def _score_structures(
  scores: object,
  spec: RunSpecification,
  *,
  structure_ids: Sequence[str],
  warnings: list[str],
) -> list[dict[str, object]]:
  """Pair negative-log-likelihood rows with ``sequences_to_score``."""
  array = np.asarray(scores)
  if array.ndim == 1:
    array = array.reshape(1, -1)
  if array.ndim != 2:
    warnings.append(f"score array has ndim {array.ndim}; expected 2")
    return []
  sequences = [str(item) for item in getattr(spec, "sequences_to_score", ()) or ()]
  if len(sequences) < int(array.shape[1]):
    warnings.append(
      "sequences_to_score is shorter than the score array; missing sequences are empty",
    )
  rows: list[dict[str, object]] = []
  for row_index in range(int(array.shape[0])):
    scored: list[dict[str, object]] = []
    for column in range(int(array.shape[1])):
      sequence = sequences[column] if column < len(sequences) else ""
      scored.append({"sequence": sequence, "nll": float(array[row_index, column])})
    rows.append(
      {
        "structure_id": _structure_id_at(structure_ids, row_index),
        "scores": scored,
      },
    )
  return rows


def _aligned_perplexity(
  value: object,
  expected_shape: tuple[int, ...],
  warnings: list[str],
) -> np.ndarray | None:
  """Return pseudo-perplexity when its shape matches the sample grid."""
  if value is None:
    return None
  array = np.asarray(value)
  if tuple(int(dim) for dim in array.shape) != expected_shape:
    warnings.append(
      f"pseudo_perplexity shape {tuple(array.shape)} does not match sample grid {expected_shape}",
    )
    return None
  return array


def _sample_index(index_array: np.ndarray | None, slot: int) -> int:
  """Return the grid sample index when one was recorded, else the slot."""
  if index_array is not None and index_array.ndim == 1 and slot < int(index_array.shape[0]):
    return int(index_array[slot])
  return slot


def _known_length(
  structure_id: str,
  lengths: Mapping[str, int] | None,
  warnings: list[str],
  noted: set[str],
) -> int | None:
  """Return a residue count, warning once when the id is absent."""
  if lengths is not None and structure_id in lengths:
    return int(lengths[structure_id])
  if structure_id not in noted:
    noted.add(structure_id)
    warnings.append(f"structure {structure_id!r} has no length; sequences were not trimmed")
  return None


def _route_results(
  results: Mapping[str, Any],
  store: dict[str, np.ndarray],
  inline_cap: int,
  structure_ids: Sequence[str],
) -> dict[str, object]:
  """Place every top-level result value, parking large arrays in ``store``."""
  routed: dict[str, object] = {}
  for key, value in results.items():
    if not isinstance(key, str) or key in _SKIP_ROUTE:
      continue
    routed[key] = _route(value, key, store, inline_cap, structure_ids)
  return routed


def _route(
  value: object,
  logical: str,
  store: dict[str, np.ndarray],
  inline_cap: int,
  structure_ids: Sequence[str],
) -> object:
  """Route one value, using ``logical`` as the side-file key for arrays."""
  named = _namedtuple_items(value)
  if named is not None:
    return {
      name: _route(item, f"{logical}/{name}", store, inline_cap, structure_ids)
      for name, item in named
    }
  if _is_array(value):
    return _place_array(logical, value, store, inline_cap)
  if isinstance(value, Mapping):
    return {
      str(key): _route(item, f"{logical}/{key}", store, inline_cap, structure_ids)
      for key, item in value.items()
    }
  if isinstance(value, list):
    return [
      _route(
        item,
        f"{logical}/{_structure_id_at(structure_ids, index)}",
        store,
        inline_cap,
        structure_ids,
      )
      for index, item in enumerate(value)
    ]
  return to_jsonable(value)


def _place_array(
  logical_key: str,
  value: object,
  store: dict[str, np.ndarray],
  inline_cap: int,
) -> object:
  """Inline an array at or under the cap, otherwise park it in ``store``."""
  array = np.asarray(value)
  if int(array.size) <= inline_cap:
    return array.tolist()
  store[logical_key] = array
  return {"shape": [int(dim) for dim in array.shape], "dtype": str(array.dtype)}


def _write_side_file(
  output_dir: Path,
  run_id: str,
  store: Mapping[str, np.ndarray],
) -> dict[str, object] | None:
  """Write ``store`` to one compressed npz file, or return ``None`` if empty."""
  if not store:
    return None
  output_dir.mkdir(parents=True, exist_ok=True)
  path = output_dir / f"{run_id}.npz"
  safe = {key.replace("/", "__"): np.asarray(array) for key, array in store.items()}
  np.savez_compressed(path, **safe)
  manifest = {
    key: {
      "shape": [int(dim) for dim in np.asarray(array).shape],
      "dtype": str(np.asarray(array).dtype),
    }
    for key, array in store.items()
  }
  return {"path": str(path), "arrays": manifest}


def _metadata(results: Mapping[str, Any]) -> Mapping[str, Any]:
  """Return the result metadata mapping, or an empty mapping."""
  raw = results.get("metadata", {})
  if isinstance(raw, Mapping):
    return raw
  return {}


def _structure_ids(metadata: Mapping[str, Any]) -> list[str]:
  """Return metadata structure ids as strings."""
  raw = metadata.get("structure_ids", [])
  if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
    return [str(item) for item in raw]
  return []


def _structure_id_at(structure_ids: Sequence[str], index: int) -> str:
  """Return the id at ``index``, or ``structure_{index}`` past the end."""
  if 0 <= index < len(structure_ids):
    return structure_ids[index]
  return f"structure_{index}"


def _spec_dict(spec: RunSpecification) -> dict[str, Any]:
  """Encode ``spec``, replacing an encode failure with an error object."""
  try:
    payload = spec_to_json_dict(spec)
  except (SpecJSONEncodeError, TypeError) as exc:
    return {"unserializable": str(exc)}
  if isinstance(payload, dict):
    return payload
  return {"unserializable": "spec JSON was not an object"}


def _sha256(payload: Mapping[str, Any]) -> str:
  """Hash the canonical JSON form of ``payload``."""
  encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(encoded.encode()).hexdigest()


def _aminx_version() -> str:
  """Return the installed aminx version, or ``unknown``."""
  try:
    return importlib.metadata.version("aminx")
  except importlib.metadata.PackageNotFoundError:
    return "unknown"


def _namedtuple_items(value: object) -> list[tuple[str, object]] | None:
  """Return namedtuple pairs, or ``None`` when ``value`` is not one."""
  field_names = getattr(value, "_fields", None)
  if not isinstance(value, tuple) or not isinstance(field_names, tuple):
    return None
  if not field_names or not all(isinstance(name, str) for name in field_names):
    return None
  names = [name for name in field_names if isinstance(name, str)]
  if len(names) != len(value):
    return None
  return list(zip(names, value, strict=True))


def _is_array(value: object) -> bool:
  """Return whether ``value`` is a rank-1-or-higher numpy or JAX array."""
  if isinstance(value, (str, bytes, np.generic)):
    return False
  if isinstance(value, np.ndarray):
    return value.ndim >= 1
  shape = getattr(value, "shape", None)
  module = type(value).__module__
  if isinstance(shape, tuple) and module.startswith(("numpy", "jax", "jaxlib")):
    return len(shape) >= 1
  return False


__all__ = [
  "DEFAULT_INLINE_CAP",
  "default_output_dir",
  "new_run_id",
  "shape_result",
  "to_jsonable",
]
