"""Build run specifications from agent-supplied paths and options.

Options are checked against the dataclass fields before the specification is
constructed. Unknown keys, deprecated keys, and fields that cannot be
represented in JSON are reported together. Input paths must already exist on
disk; this module does not fetch them.
"""

from __future__ import annotations

import difflib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from aminx.host._sampling_helper import _canonical_structure_id
from aminx.io.parsing import parse_structure
from aminx.potts.spec import PottsRunSpec
from aminx.run.spec_json import (
  _REMOVED_SPEC_KEYS,
  SpecJSONDecodeError,
  SpecJSONEncodeError,
  run_specification_from_json,
  run_specification_from_json_dict,
  run_specification_to_json_dict,
)
from aminx.run.specs import (
  InspectionSpecification,
  JacobianSpecification,
  RunSpecification,
  SamplingSpecification,
  ScoringSpecification,
)

logger = logging.getLogger(__name__)

SPEC_KINDS: tuple[str, ...] = ("sample", "score", "inspect", "jacobian")

_SPEC_CLASS: dict[str, type[RunSpecification]] = {
  "sample": SamplingSpecification,
  "score": ScoringSpecification,
  "inspect": InspectionSpecification,
  "jacobian": JacobianSpecification,
}

# Dataclass fields that are not JSON and must not be supplied by an agent.
_UNSETTABLE: frozenset[str] = frozenset(
  {
    "run_spec",
    "decoding_order_fn",
    "foldcomp_database",
    "combine_fn",
    "encoding_fusion",
    "decoding_fusion",
    "conformational_states",
    "carry_specs",
    "dedup_specs",
    "noise",
  },
)


def build_spec(
  kind: str,
  inputs: Sequence[str],
  options: Mapping[str, Any] | None = None,
) -> RunSpecification:
  """Construct a run specification for ``kind``.

  Parameters
  ----------
  kind : str
    One of :data:`SPEC_KINDS`.
  inputs : sequence of str
    Local structure paths. Each must name an existing file.
  options : mapping or None, optional
    Dataclass fields of the specification class. ``None`` is an empty set
    of options.

  Returns
  -------
  RunSpecification
    The constructed specification. Subclass ``__post_init__`` validation
    still runs.

  Raises
  ------
  ValueError
    Unknown ``kind``, any option that is unknown, deprecated, or unsettable
    (one message listing every offending key), or a missing input path.
  """
  cls = _SPEC_CLASS.get(kind)
  if cls is None:
    msg = f"unknown spec kind {kind!r}; expected one of {', '.join(SPEC_KINDS)}"
    raise ValueError(msg)
  supplied = dict(options or {})
  problems = _option_problems(cls, supplied)
  if problems:
    msg = "invalid options: " + "; ".join(problems)
    raise ValueError(msg)
  resolved = _absolute_inputs(inputs)
  return cls(inputs=resolved, **supplied)


def structure_lengths(spec: RunSpecification) -> dict[str, int]:
  """Return residue counts keyed by the runner's structure id.

  Each existing file input is parsed with :func:`aminx.io.parsing.parse_structure`.
  When ``spec.chain_id`` is set, the count is the residues on those chains.
  A path that cannot be parsed is omitted. This function does not raise for
  parse failures.

  Parameters
  ----------
  spec : RunSpecification
    Specification whose ``inputs`` are local structure paths.

  Returns
  -------
  dict of str to int
    ``{structure_id: residue_count}`` for every input that parsed.
  """
  lengths: dict[str, int] = {}
  for index, item in enumerate(_input_items(spec)):
    if not isinstance(item, (str, Path)):
      continue
    path = Path(item)
    if not path.is_file():
      continue
    try:
      count = _residue_count(_parsed_structure(path), spec.chain_id)
    except Exception as exc:  # noqa: BLE001 -- lengths are best-effort
      logger.debug("structure length skipped for %s: %s", path, exc)
      continue
    lengths[_canonical_structure_id(item, index)] = count
  return lengths


def spec_to_json_dict(spec: RunSpecification | PottsRunSpec) -> dict[str, Any]:
  """Return a JSON object for ``spec``.

  Potts specifications are written with :meth:`PottsRunSpec.to_json` and
  tagged with ``kind`` ``"potts"``. Every other specification uses
  :func:`aminx.run.spec_json.run_specification_to_json_dict`.

  Parameters
  ----------
  spec : RunSpecification or PottsRunSpec
    Specification to encode.

  Returns
  -------
  dict
    JSON-ready mapping.
  """
  if isinstance(spec, PottsRunSpec):
    payload = json.loads(spec.to_json())
    if not isinstance(payload, dict):
      msg = "PottsRunSpec.to_json() did not return an object"
      raise SpecJSONEncodeError(msg)
    payload["kind"] = "potts"
    return payload
  return run_specification_to_json_dict(spec)


def spec_from_json(
  text_or_dict: str | Mapping[str, Any],
) -> RunSpecification | PottsRunSpec:
  """Decode a specification from JSON text or a mapping.

  Parameters
  ----------
  text_or_dict : str or mapping
    JSON text, or an object produced by :func:`spec_to_json_dict`.

  Returns
  -------
  RunSpecification or PottsRunSpec
    The decoded specification.

  Raises
  ------
  SpecJSONDecodeError
    When the document is not a specification object.
  """
  if isinstance(text_or_dict, str):
    data = json.loads(text_or_dict)
    if not isinstance(data, dict):
      msg = "JSON root must be an object"
      raise SpecJSONDecodeError(msg)
  else:
    data = text_or_dict
  if _is_potts_payload(data):
    payload = dict(data)
    payload.pop("kind", None)
    payload.pop("_spec_class", None)
    return PottsRunSpec.from_json(json.dumps(payload))
  if isinstance(text_or_dict, str):
    return run_specification_from_json(text_or_dict)
  return run_specification_from_json_dict(data)


def _option_problems(cls: type[RunSpecification], options: Mapping[str, Any]) -> list[str]:
  """Describe every option key that must not be passed to ``cls``."""
  init_names = {field.name for field in fields(cls) if field.init}
  problems: list[str] = []
  for key in sorted(options):
    if key in _REMOVED_SPEC_KEYS:
      problems.append(f"{key} is deprecated and is not accepted")
      continue
    if key in _UNSETTABLE:
      problems.append(f"{key} cannot be set through the agent")
      continue
    if key == "inputs":
      problems.append("inputs must be passed as the inputs argument, not in options")
      continue
    if key not in init_names:
      matches = difflib.get_close_matches(key, sorted(init_names), n=3, cutoff=0.6)
      if matches:
        suggestion = ", ".join(matches)
        problems.append(f"unknown option {key!r} (did you mean {suggestion}?)")
      else:
        problems.append(f"unknown option {key!r}")
  return problems


def _absolute_inputs(inputs: Sequence[str]) -> list[str]:
  """Resolve existing files to absolute path strings."""
  missing: list[str] = []
  resolved: list[str] = []
  for raw in inputs:
    path = Path(raw)
    if not path.is_file():
      missing.append(str(raw))
      continue
    resolved.append(str(path.resolve()))
  if missing:
    msg = "input paths do not exist: " + ", ".join(missing)
    raise ValueError(msg)
  return resolved


def check_unique_structure_ids(spec: RunSpecification) -> None:
  """Raise when two inputs map to the same runner structure id.

  The runner keys structures by file stem, so ``a/model.pdb`` and
  ``b/model.pdb`` would share lengths and side-file keys and one would
  silently overwrite or mis-trim the other.

  Parameters
  ----------
  spec : RunSpecification
    Specification whose ``inputs`` are checked.

  Raises
  ------
  ValueError
    Two or more inputs share a structure id. The message names them.
  """
  seen: dict[str, list[str]] = {}
  for index, item in enumerate(_input_items(spec)):
    seen.setdefault(_canonical_structure_id(item, index), []).append(str(item))
  clashes = {key: items for key, items in seen.items() if len(items) > 1}
  if clashes:
    detail = "; ".join(f"{key!r}: {', '.join(items)}" for key, items in sorted(clashes.items()))
    msg = f"inputs share a structure id (the file stem); rename or copy them apart: {detail}"
    raise ValueError(msg)


def _input_items(spec: RunSpecification) -> list[object]:
  """Return specification inputs as a list, matching the runner's wrapping."""
  raw = spec.inputs
  if isinstance(raw, (str, Path)) or hasattr(raw, "read"):
    return [raw]
  try:
    return list(raw)
  except TypeError:
    return [raw]


def _parsed_structure(path: Path) -> object:
  """Parse ``path`` into one structure object."""
  parsed = parse_structure(path)
  if hasattr(parsed, "aatype"):
    return parsed
  return next(iter(parsed))


def _chain_targets(chain_id: object) -> set[str] | None:
  """Normalize ``spec.chain_id`` to a set of chain letters."""
  if chain_id is None:
    return None
  if isinstance(chain_id, str):
    return {chain_id}
  if isinstance(chain_id, Sequence):
    return {str(item) for item in chain_id}
  return {str(chain_id)}


def _residue_count(protein: object, chain_id: object) -> int:
  """Count residues, honouring a chain filter when chain ids are present."""
  aatype = getattr(protein, "aatype", None)
  if aatype is None:
    msg = "parsed structure has no aatype"
    raise TypeError(msg)
  length = int(np.asarray(aatype).shape[0])
  targets = _chain_targets(chain_id)
  if targets is None:
    return length
  chain_ids = getattr(protein, "chain_ids", None)
  chain_index = getattr(protein, "chain_index", None)
  if chain_ids is None or chain_index is None:
    return length
  allowed = {index for index, cid in enumerate(chain_ids) if str(cid) in targets}
  if not allowed:
    return 0
  index = np.asarray(chain_index)
  if index.shape[0] != length:
    return length
  return int(np.isin(index, list(allowed)).sum())


def _is_potts_payload(data: Mapping[str, Any]) -> bool:
  """Return whether ``data`` is a Potts specification document."""
  if data.get("kind") == "potts" or data.get("_spec_class") == "PottsRunSpec":
    return True
  return "_spec_class" not in data and "trw_spec" in data and "weights_path" in data


__all__ = [
  "SPEC_KINDS",
  "build_spec",
  "check_unique_structure_ids",
  "spec_from_json",
  "spec_to_json_dict",
  "structure_lengths",
]
