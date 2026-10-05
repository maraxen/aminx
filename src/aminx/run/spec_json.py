"""JSON serialization for run specifications (Typer / CLI friendly).

Round-trip uses a stable ``_spec_class`` discriminator and JSON-native values only.
Non-portable fields (callables, live streams, some custom objects) must be absent
or JSON export raises :class:`SpecJSONEncodeError`.

Pickle-based migration is intentionally out of scope; prefer this module for new tooling.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aminx.run.specs import (
  InspectionSpecification,
  JacobianSpecification,
  RunSpecification,
  SamplingSpecification,
  ScoringSpecification,
)

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


def run_specification_to_json_dict(spec: RunSpecification) -> dict[str, Any]:
  """Return a JSON-serializable dict for ``spec`` (includes ``_spec_class``)."""
  if not is_dataclass(spec):
    msg = "run_specification_to_json_dict expects a dataclass specification instance"
    raise TypeError(msg)
  cls = type(spec)
  name = cls.__name__
  if name not in _SPEC_CLASS_BY_NAME:
    msg = f"Unknown specification class {name!r}; register it in spec_json._SPEC_CLASS_BY_NAME"
    raise SpecJSONEncodeError(msg)
  payload: dict[str, Any] = {"_spec_class": name}
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
    raw = getattr(spec, f.name)
    if f.name == "inputs":
      payload[f.name] = _ensure_jsonable_inputs(raw, f.name)
    else:
      payload[f.name] = _to_json_value(raw, field_name=f.name)
  return payload


def run_specification_to_json(spec: RunSpecification, *, indent: int | None = 2) -> str:
  """Serialize ``spec`` to a JSON string."""
  return json.dumps(run_specification_to_json_dict(spec), indent=indent, sort_keys=True)


def _coerce_field_value(_cls: type[Any], field_name: str, value: Any) -> Any:
  """Best-effort coercion from JSON into constructor-friendly values."""
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
  if (
    field_name in {"backbone_noise", "estat_noise", "vdw_noise", "temperature"}
    and isinstance(value, list)
    and value
    and all(isinstance(x, (int, float)) for x in value)
  ):
    return tuple(float(x) for x in value)
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
    "ar_mask",
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
  kwargs: dict[str, Any] = {}
  for f in fields(cls):
    if not f.init:
      continue
    if f.name in _NON_JSON_ROOT_FIELDS:
      continue
    if f.name not in data:
      if f.default is not MISSING:
        kwargs[f.name] = f.default
      elif f.default_factory is not MISSING:
        kwargs[f.name] = f.default_factory()  # type: ignore[misc]
      else:
        msg = f"Missing required field {f.name!r}"
        raise SpecJSONDecodeError(msg)
    else:
      kwargs[f.name] = _coerce_field_value(cls, f.name, data[f.name])
  return cls(**kwargs)  # type: ignore[arg-type]


def run_specification_from_json(text: str) -> RunSpecification:
  """Parse JSON text into a concrete specification instance."""
  data = json.loads(text)
  if not isinstance(data, dict):
    msg = "JSON root must be an object"
    raise SpecJSONDecodeError(msg)
  return run_specification_from_json_dict(data)
