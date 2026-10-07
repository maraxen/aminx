"""JSON serialization for run specifications (Typer / CLI friendly).

Round-trip uses a stable ``_spec_class`` discriminator and JSON-native values only.
Non-portable fields (callables, live streams, some custom objects) must be absent
or JSON export raises :class:`SpecJSONEncodeError`.

Pickle-based migration is intentionally out of scope; prefer this module for new tooling.
"""

from __future__ import annotations

import difflib
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from xtrax.config import classify_schema_version

from aminx.run.options import LaserOptions, PottsMPNNOptions
from aminx.run.specs import (
  InspectionSpecification,
  JacobianSpecification,
  RunSpecification,
  SamplingSpecification,
  ScoringSpecification,
)

_OPTIONS_FIELDS: dict[str, type[Any]] = {
  "potts_mpnn": PottsMPNNOptions,
  "laser": LaserOptions,
}

_SPEC_CLASS_BY_NAME: dict[str, type[RunSpecification]] = {
  "RunSpecification": RunSpecification,
  "ScoringSpecification": ScoringSpecification,
  "SamplingSpecification": SamplingSpecification,
  "JacobianSpecification": JacobianSpecification,
  "InspectionSpecification": InspectionSpecification,
}


class SpecJSONEncodeError(TypeError):
  """Raised when a specification field cannot be encoded to JSON."""


class SpecJSONDecodeError(ValueError):
  """Raised when JSON cannot be decoded into a known specification type."""


try:
  from aminx.training.specs import TrainingSpecification
except ImportError:
  pass
else:
  _SPEC_CLASS_BY_NAME["TrainingSpecification"] = TrainingSpecification


def _is_arraylike(value: object) -> bool:
  if value is None:
    return False
  mod = type(value).__module__
  if mod in ("numpy", "jax.numpy", "jax._src.numpy.lax_numpy"):
    return True
  return hasattr(value, "shape") and hasattr(value, "tolist")


def _to_json_value(value: object, *, field_name: str) -> object:
  if value is None or isinstance(value, bool | int | float | str):
    return value
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, Mapping):
    return {str(k): _to_json_value(v, field_name=f"{field_name}.{k}") for k, v in value.items()}
  if isinstance(value, tuple | list):
    return [_to_json_value(v, field_name=f"{field_name}[]") for v in value]
  if isinstance(value, set | frozenset):
    return [_to_json_value(v, field_name=f"{field_name}[]") for v in sorted(value, key=str)]
  if _is_arraylike(value):
    try:
      return np.asarray(value).tolist()
    except (TypeError, ValueError) as exc:
      msg = f"Field {field_name!r}: cannot convert array-like to JSON ({exc})"
      raise SpecJSONEncodeError(msg) from exc
  msg = f"Field {field_name!r}: unsupported type {type(value).__name__!r} for JSON encoding"
  raise SpecJSONEncodeError(msg)


def _ensure_jsonable_inputs(value: object, field_name: str) -> object:
  if isinstance(value, str):
    return value
  if isinstance(value, Sequence) and not isinstance(value, str | bytes):
    out: list[str] = []
    for i, item in enumerate(value):
      if isinstance(item, str):
        out.append(item)
      else:
        msg = f"Field {field_name!r}: item {i} must be str for JSON (got {type(item).__name__})"
        raise SpecJSONEncodeError(msg)
    return out
  msg = f"Field {field_name!r}: inputs must be str or list[str] for JSON encoding"
  raise SpecJSONEncodeError(msg)


_NON_JSON_ROOT_FIELDS = frozenset(
  {
    "run_spec",
    "decoding_order_fn",
    "foldcomp_database",
    "combine_fn",
    "encoding_fusion",
    "decoding_fusion",
  },
)

# Fields __post_init__ DERIVES from other, serialized fields. Skipped on encode and rebuilt on
# decode, so omitting them is lossless -- verified both directions:
#   - set backbone_noise=0.25            -> noise == [FeatureNoiseBundle(feature_type='backbone', ...)]
#   - set noise=[FeatureNoiseBundle(...)] -> backbone_noise == (0.3,)   (back-populated)
# `noise` is `list[FeatureNoiseBundle]`, a nested dataclass `_to_json_value` has no branch for,
# so without this every real spec would fail to encode: __post_init__ populates `noise` from
# `backbone_noise`, and every campaign sets `backbone_noise`.
#
# Distinct from _NON_JSON_ROOT_FIELDS, which must be None and raises otherwise. These are
# expected to be non-None and are dropped precisely because they carry no information the
# serialized fields do not already carry.
_DERIVED_FIELDS = frozenset({"noise"})

# Current spec JSON document version. Legacy documents omit the key and still load.
SPEC_JSON_SCHEMA_VERSION: Final[int] = 1


@dataclass(frozen=True)
class RemovedSpecKey:
  """One removed specification key and how a saved document should treat it."""

  policy: Literal["drop", "error"]
  note: str


# A key with policy "error" raises SpecJSONDecodeError with its note (this is what
# #2519/#2531 will use for knobs that used to do something). policy "drop" removes a key
# that is already ignored, so the document's meaning stays the same, and emits one
# DeprecationWarning naming the key.
_REMOVED_SPEC_KEYS: Mapping[str, RemovedSpecKey] = {
  "output_path": RemovedSpecKey(
    policy="drop",
    note="removed; streaming output uses output_h5_path",
  ),
  "score_batch_size": RemovedSpecKey(
    policy="drop",
    note="removed; already ignored",
  ),
  "average_logits": RemovedSpecKey(
    policy="drop",
    note="removed; already ignored",
  ),
  "combine_noise_batch_size": RemovedSpecKey(
    policy="drop",
    note="removed; already ignored",
  ),
  "gmm_min_iters": RemovedSpecKey(
    policy="drop",
    note="removed; already ignored",
  ),
  "average_encoding_mode": RemovedSpecKey(
    policy="drop",
    note="removed; encoding fusion is encoding_fusion",
  ),
}


def migrate_removed_spec_keys(payload: Mapping[str, Any]) -> dict[str, Any]:
  """Apply :data:`_REMOVED_SPEC_KEYS` and return a new mapping.

  ``drop`` deletes the key and emits one :class:`DeprecationWarning` naming it.
  ``error`` raises :class:`SpecJSONDecodeError` with that key's note.
  """
  migrated = dict(payload)
  for key, entry in _REMOVED_SPEC_KEYS.items():
    if key not in migrated:
      continue
    if entry.policy == "drop":
      migrated.pop(key)
      warnings.warn(
        f"Removed spec key {key!r} is deprecated and ignored. {entry.note}",
        DeprecationWarning,
        stacklevel=2,
      )
      continue
    if entry.policy == "error":
      raise SpecJSONDecodeError(entry.note)
    msg = f"Unknown removal policy {entry.policy!r} for spec key {key!r}"
    raise SpecJSONDecodeError(msg)
  return migrated


def _require_supported_schema_version(data: Mapping[str, Any]) -> None:
  """Accept the current version and legacy documents; reject anything else."""
  status = classify_schema_version(data, SPEC_JSON_SCHEMA_VERSION)
  if status.kind in {"ok", "missing"}:
    return
  if status.kind in {"newer_than_supported", "mismatched"}:
    msg = (
      f"Unsupported spec JSON schema_version: found {status.found!r}, "
      f"supported {status.current!r} "
      f"(document schema_version is {data.get('schema_version')!r})"
    )
    raise SpecJSONDecodeError(msg)
  msg = f"Unrecognized spec JSON schema version status {status.kind!r}"
  raise SpecJSONDecodeError(msg)


def _reject_unaccepted_keys(cls: type[Any], data: Mapping[str, Any]) -> None:
  """Reject unknown keys and non-null values for fields JSON cannot carry."""
  init_names = {f.name for f in fields(cls) if f.init}
  non_init_names = {f.name for f in fields(cls) if not f.init}
  non_null_carriers: list[str] = []
  unknown: list[str] = []
  for key, value in data.items():
    if key in {"_spec_class", "schema_version"} or key in _DERIVED_FIELDS:
      continue
    if key in _NON_JSON_ROOT_FIELDS or key in non_init_names:
      if value is not None:
        non_null_carriers.append(key)
      continue
    if key not in init_names:
      unknown.append(key)
  if non_null_carriers:
    names = ", ".join(repr(key) for key in sorted(non_null_carriers))
    msg = f"Spec JSON field(s) {names} must be null"
    raise SpecJSONDecodeError(msg)
  if not unknown:
    return
  candidates = sorted(init_names)
  parts: list[str] = []
  for key in sorted(unknown):
    matches = difflib.get_close_matches(key, candidates, n=1, cutoff=0.6)
    if matches:
      parts.append(f"{key!r} (did you mean {matches[0]!r}?)")
    else:
      parts.append(repr(key))
  msg = "Unknown spec JSON key(s): " + "; ".join(parts)
  raise SpecJSONDecodeError(msg)


def run_specification_to_json_dict(spec: RunSpecification) -> dict[str, Any]:
  """Return a JSON-serializable dict for ``spec`` (includes ``_spec_class`` and ``schema_version``)."""
  if not is_dataclass(spec):
    msg = "run_specification_to_json_dict expects a dataclass specification instance"
    raise TypeError(msg)
  cls = type(spec)
  name = cls.__name__
  if name not in _SPEC_CLASS_BY_NAME:
    msg = f"Unknown specification class {name!r}; register it in spec_json._SPEC_CLASS_BY_NAME"
    raise SpecJSONEncodeError(msg)
  payload: dict[str, Any] = {
    "_spec_class": name,
    "schema_version": SPEC_JSON_SCHEMA_VERSION,
  }
  for f in fields(spec):
    if not f.init:
      continue
    if f.name in _NON_JSON_ROOT_FIELDS:
      val = getattr(spec, f.name)
      # Skip non-None callable fields (they're reconstructed from configuration)
      if callable(val):
        continue
      if val is not None:
        msg = f"Field {f.name!r} must be None for JSON encoding (got {type(val).__name__})"
        raise SpecJSONEncodeError(msg)
      continue
    if f.name in _DERIVED_FIELDS:
      # __post_init__ rebuilds it from serialized fields; carrying it would be redundant, and
      # it is a nested dataclass _to_json_value cannot encode anyway.
      continue
    if f.name in _OPTIONS_FIELDS:
      payload[f.name] = _options_to_json_value(getattr(spec, f.name), field_name=f.name)
      continue
    raw = getattr(spec, f.name)
    if f.name == "inputs":
      payload[f.name] = _ensure_jsonable_inputs(raw, f.name)
    else:
      payload[f.name] = _to_json_value(raw, field_name=f.name)
  return payload


def run_specification_to_json(spec: RunSpecification, *, indent: int | None = 2) -> str:
  """Serialize ``spec`` to a JSON string."""
  return json.dumps(run_specification_to_json_dict(spec), indent=indent, sort_keys=True)


def _options_to_json_value(value: object, *, field_name: str) -> object:
  """Encode an options dataclass as ``{field: json value}``, or ``None``."""
  if value is None:
    return None
  if not is_dataclass(value):
    msg = f"Field {field_name!r}: expected an options dataclass, got {type(value).__name__}"
    raise SpecJSONEncodeError(msg)
  return {
    item.name: _to_json_value(getattr(value, item.name), field_name=f"{field_name}.{item.name}")
    for item in fields(value)
  }


def options_from_json_value(cls: type[Any], value: Any) -> Any:
  """Decode a JSON object into ``cls``. Unknown keys raise :class:`SpecJSONDecodeError`."""
  if value is None:
    return None
  if not isinstance(value, Mapping):
    msg = f"{cls.__name__} JSON must be an object or null, got {type(value).__name__}"
    raise SpecJSONDecodeError(msg)
  known = {item.name: item for item in fields(cls)}
  unknown = sorted(set(value) - set(known))
  if unknown:
    msg = f"Unknown {cls.__name__} field(s): {unknown}"
    raise SpecJSONDecodeError(msg)
  kwargs: dict[str, Any] = {}
  for name, raw in value.items():
    item = known[name]
    decoded = tuple(raw) if isinstance(raw, list) and isinstance(item.default, tuple) else raw
    kwargs[name] = decoded
  return cls(**kwargs)


def _coerce_field_value(_cls: type[Any], field_name: str, value: Any) -> Any:
  """Best-effort coercion from JSON into constructor-friendly values."""
  if field_name in _OPTIONS_FIELDS:
    return options_from_json_value(_OPTIONS_FIELDS[field_name], value)
  if value is None:
    return None
  if field_name.endswith("_path") or field_name in {
    "topology",
    "cache_path",
    "output_dir",
    "model_local_path",
    "checkpoint_registry_path",
    "preprocessed_index_path",
    "output_h5_path",
    "ligand_context_path",
    "checkpoint_dir",
    "resume_from_checkpoint",
    "validation_data",
    "validation_preprocessed_path",
    "validation_preprocessed_index_path",
  }:
    if isinstance(value, str):
      return Path(value)
  if field_name == "inputs":
    if isinstance(value, str):
      return value
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
      return value
  if field_name == "temperature" and isinstance(value, list):
    if all(isinstance(x, (int, float)) or x is None for x in value):
      return tuple(None if x is None else float(x) for x in value)
  if (
    field_name in {"backbone_noise", "estat_noise", "vdw_noise"}
    and isinstance(value, list)
    and value
    and all(isinstance(x, (int, float)) for x in value)
  ):
    return tuple(float(x) for x in value)
  if field_name == "omit_aa" and isinstance(value, list):
    return tuple(str(x) for x in value)
  if field_name == "omit_aa_per_position" and isinstance(value, Mapping):
    return {int(key): str(letters) for key, letters in value.items()}
  if field_name == "tied_positions" and isinstance(value, list):
    if not value:
      return []
    if all(isinstance(pair, (list, tuple)) and len(pair) == 2 for pair in value):
      return [tuple(int(a) for a in pair) for pair in value]  # type: ignore[return-value]
  if isinstance(value, list) and field_name in {
    "tie_group_map",
    "state_position_map",
    "structure_mapping",
    "bias",
    "fixed_positions",
    "fixed_mask",
    "fixed_tokens",
    "state_weights",
    "combine_weights",
  }:
    try:
      return np.asarray(value)
    except (TypeError, ValueError):
      return value
  return value


def run_specification_from_json_dict(data: Mapping[str, Any]) -> RunSpecification:
  """Deserialize a mapping produced by :func:`run_specification_to_json_dict`."""
  if "_spec_class" not in data:
    msg = "Missing required '_spec_class' key"
    raise SpecJSONDecodeError(msg)
  name = data["_spec_class"]
  if not isinstance(name, str) or name not in _SPEC_CLASS_BY_NAME:
    msg = f"Unknown or invalid _spec_class: {name!r}"
    raise SpecJSONDecodeError(msg)
  cls = _SPEC_CLASS_BY_NAME[name]
  _require_supported_schema_version(data)
  migrated = migrate_removed_spec_keys(data)
  _reject_unaccepted_keys(cls, migrated)
  kwargs: dict[str, Any] = {}
  for f in fields(cls):
    if not f.init:
      continue
    if f.name in _NON_JSON_ROOT_FIELDS or f.name in _DERIVED_FIELDS:
      continue
    if f.name not in migrated:
      if f.default is not MISSING:
        kwargs[f.name] = f.default
      elif f.default_factory is not MISSING:
        kwargs[f.name] = f.default_factory()  # type: ignore[misc]
      else:
        msg = f"Missing required field {f.name!r}"
        raise SpecJSONDecodeError(msg)
    else:
      kwargs[f.name] = _coerce_field_value(cls, f.name, migrated[f.name])
  # Legacy documents (no schema_version) recorded seed 0 while the falsy coalesce actually ran 42.
  # Decode that one value as 42 so a reload reproduces the original run. schema_version 1
  # keeps 0.
  recorded_seed = kwargs.get("random_seed")
  if (
    "schema_version" not in data
    and recorded_seed == 0
    and not isinstance(recorded_seed, bool)
  ):
    warnings.warn(
      "Pre-schema spec recorded random_seed 0 but ran with seed 42; "
      "decoding as 42 to reproduce the original run.",
      UserWarning,
      stacklevel=2,
    )
    kwargs["random_seed"] = 42
  return cls(**kwargs)  # type: ignore[arg-type]


def run_specification_from_json(text: str) -> RunSpecification:
  """Parse JSON text into a concrete specification instance."""
  data = json.loads(text)
  if not isinstance(data, dict):
    msg = "JSON root must be an object"
    raise SpecJSONDecodeError(msg)
  return run_specification_from_json_dict(data)
