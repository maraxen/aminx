"""MCP tools for sampling, scoring, inspection, and checkpoint lookup.

This module imports ``cisternal`` at import time and is loaded only by the
MCP server. Tool bodies are asynchronous. Model runs happen in a worker
thread, one at a time, so the server event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import importlib.resources
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import cisternal

from aminx.agent.provenance import capture
from aminx.agent.requests import (
  build_spec,
  spec_from_json,
  spec_to_json_dict,
  structure_lengths,
)
from aminx.agent.shaping import (
  DEFAULT_INLINE_CAP,
  default_output_dir,
  new_run_id,
  shape_result,
  to_jsonable,
)
from aminx.host import runner as host_runner
from aminx.io import weights as weights_io
from aminx.run.spec_json import SpecJSONDecodeError

REGISTRY = "aminx"
TOOL_NAMES: tuple[str, ...] = (
  "sample",
  "score",
  "inspect",
  "jacobian",
  "spec_emit",
  "spec_validate",
  "list_checkpoints",
)
DEFAULT_CHECKPOINT_ID = "proteinmpnn_v_48_020"
# Every checkpoint aminx can load. Wheels ship no weight files (pyproject excludes
# model_params/*.eqx.zst; they download from the Hub on first use), so a glob of the installed
# package finds nothing under uvx. tests/agent/test_mcp_surface.py keeps this in sync with the
# source tree.
KNOWN_CHECKPOINT_IDS: tuple[str, ...] = (
  "global_label_membrane_mpnn_v_48_020",
  "ligandmpnn_sc_v_32_002_16",
  "ligandmpnn_v_32_005_25",
  "ligandmpnn_v_32_010_25",
  "ligandmpnn_v_32_020_25",
  "ligandmpnn_v_32_030_25",
  "per_residue_label_membrane_mpnn_v_48_020",
  "proteinmpnn_v_48_002",
  "proteinmpnn_v_48_010",
  "proteinmpnn_v_48_020",
  "proteinmpnn_v_48_030",
  "solublempnn_v_48_002",
  "solublempnn_v_48_010",
  "solublempnn_v_48_020",
  "solublempnn_v_48_030",
)

_RUN_LOCK = threading.Lock()
_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
  "sample": host_runner.sample,
  "score": host_runner.score,
  "inspect": host_runner.inspect,
  "jacobian": host_runner.jacobian,
}


def _merge_options(
  typed: Mapping[str, Any],
  options: Mapping[str, Any] | None,
) -> dict[str, Any]:
  """Merge typed parameters into ``options``.

  ``None`` values are omitted so an unset optional parameter keeps the
  specification default. A key set both ways is an error.

  Parameters
  ----------
  typed : mapping
    Specification fields taken from typed tool parameters.
  options : mapping or None
    Extra specification fields. ``None`` means no extra fields.

  Returns
  -------
  dict
    Combined options for :func:`aminx.agent.requests.build_spec`.

  Raises
  ------
  ValueError
    ``typed`` and ``options`` both set the same key. The message names it.
  """
  merged = dict(options or {})
  for key, value in typed.items():
    if value is None:
      continue
    if key in merged:
      msg = f"{key} was given both as a parameter and inside options"
      raise ValueError(msg)
    merged[key] = value
  return merged


async def _run(
  kind: Literal["sample", "score", "inspect", "jacobian"],
  inputs: list[str],
  typed_options: Mapping[str, Any],
  options: dict[str, Any] | None,
  output_dir: str | None,
  inline_cap: int,
) -> dict[str, Any]:
  """Build a specification and shape one runner result.

  Parameters
  ----------
  kind : {"sample", "score", "inspect", "jacobian"}
    Which runner to call.
  inputs : list of str
    Local structure paths.
  typed_options : mapping
    Typed parameters, already renamed to specification fields.
  options : dict or None
    Extra specification fields.
  output_dir : str or None
    Side-file directory. ``None`` uses the default cache directory.
  inline_cap : int
    Maximum array size kept inline.

  Returns
  -------
  dict
    JSON summary from :func:`aminx.agent.shaping.shape_result`.

  Raises
  ------
  ValueError
    Invalid options, a duplicate key, or a missing input path.
  """
  merged = _merge_options(typed_options, options)
  spec = build_spec(kind, inputs, merged)

  def worker() -> dict[str, Any]:
    with _RUN_LOCK:
      lengths = structure_lengths(spec)
      results = _RUNNERS[kind](spec=spec)
      directory = Path(output_dir) if output_dir else default_output_dir()
      return shape_result(
        kind,
        results,
        spec=spec,
        lengths=lengths,
        output_dir=directory,
        run_id=new_run_id(),
        inline_cap=inline_cap,
        provenance=capture(),
      )

  return await asyncio.to_thread(worker)


@cisternal.tool(registry=REGISTRY)
async def sample(
  inputs: list[str],
  num_samples: int | None = None,
  temperature: float | list[float] | None = None,
  random_seed: int | None = None,
  checkpoint_id: str | None = None,
  chain_id: str | list[str] | None = None,
  options: dict[str, Any] | None = None,
  output_dir: str | None = None,
  inline_cap: int = DEFAULT_INLINE_CAP,
) -> dict[str, Any]:
  """Design amino-acid sequences for protein backbones. Inputs are local structure file paths. The first call per checkpoint and shape compiles JAX and can take minutes; later calls with the same shapes are fast. Large arrays come back as a side-file path.

  Parameters
  ----------
  inputs : list of str
    Local structure file paths.
  num_samples : int or None, optional
    Sequences to draw per structure. ``None`` keeps the specification default (1).
  temperature : float, list of float, or None, optional
    Sampling temperature or temperatures. ``None`` keeps the specification default (0.1).
  random_seed : int or None, optional
    Seed for the sampler. ``None`` keeps the specification default (42).
  checkpoint_id : str or None, optional
    Packaged checkpoint id. ``None`` keeps the specification default.
  chain_id : str, list of str, or None, optional
    Chains to design. ``None`` keeps the specification default.
  options : dict or None, optional
    Other sampling-specification fields. A key also passed as a typed
    parameter is an error.
  output_dir : str or None, optional
    Directory for the npz side file.
  inline_cap : int, optional
    Maximum number of array elements kept in the JSON result.

  Returns
  -------
  dict
    Shaped sample result, including decoded sequences, the spec, and provenance.
  """
  typed = {
    "num_samples": num_samples,
    "temperature": temperature,
    "random_seed": random_seed,
    "checkpoint_id": checkpoint_id,
    "chain_id": chain_id,
  }
  return await _run("sample", inputs, typed, options, output_dir, inline_cap)


@cisternal.tool(registry=REGISTRY)
async def score(
  inputs: list[str],
  sequences: list[str],
  random_seed: int | None = None,
  checkpoint_id: str | None = None,
  chain_id: str | list[str] | None = None,
  options: dict[str, Any] | None = None,
  output_dir: str | None = None,
  inline_cap: int = DEFAULT_INLINE_CAP,
) -> dict[str, Any]:
  """Score amino-acid sequences against protein backbones. Inputs are local structure file paths. The first call per checkpoint and shape compiles JAX and can take minutes; later calls with the same shapes are fast. Large arrays come back as a side-file path.

  Parameters
  ----------
  inputs : list of str
    Local structure file paths.
  sequences : list of str
    Sequences to score. Stored on the specification as ``sequences_to_score``.
  random_seed : int or None, optional
    Seed for any stochastic scoring path. ``None`` keeps the specification default (42).
  checkpoint_id : str or None, optional
    Packaged checkpoint id. ``None`` keeps the specification default.
  chain_id : str, list of str, or None, optional
    Chains to score. ``None`` keeps the specification default.
  options : dict or None, optional
    Other scoring-specification fields.
  output_dir : str or None, optional
    Directory for the npz side file.
  inline_cap : int, optional
    Maximum number of array elements kept in the JSON result.

  Returns
  -------
  dict
    Shaped score result, including per-sequence negative log-likelihoods.
  """
  typed = {
    "sequences_to_score": sequences,
    "random_seed": random_seed,
    "checkpoint_id": checkpoint_id,
    "chain_id": chain_id,
  }
  return await _run("score", inputs, typed, options, output_dir, inline_cap)


@cisternal.tool(registry=REGISTRY)
async def inspect(
  inputs: list[str],
  features: list[str] | None = None,
  checkpoint_id: str | None = None,
  chain_id: str | list[str] | None = None,
  options: dict[str, Any] | None = None,
  output_dir: str | None = None,
  inline_cap: int = DEFAULT_INLINE_CAP,
) -> dict[str, Any]:
  """Inspect model features for protein backbones. Inputs are local structure file paths. The first call per checkpoint and shape compiles JAX and can take minutes; later calls with the same shapes are fast. Large arrays come back as a side-file path.

  Parameters
  ----------
  inputs : list of str
    Local structure file paths.
  features : list of str or None, optional
    Inspection features. ``None`` keeps the specification default. Passed as
    ``inspection_features``.
  checkpoint_id : str or None, optional
    Packaged checkpoint id. ``None`` keeps the specification default.
  chain_id : str, list of str, or None, optional
    Chains to inspect. ``None`` keeps the specification default.
  options : dict or None, optional
    Other inspection-specification fields.
  output_dir : str or None, optional
    Directory for the npz side file.
  inline_cap : int, optional
    Maximum number of array elements kept in the JSON result.

  Returns
  -------
  dict
    Shaped inspection result. Large feature arrays are side-file paths.
  """
  typed: dict[str, Any] = {
    "checkpoint_id": checkpoint_id,
    "chain_id": chain_id,
  }
  if features is not None:
    typed["inspection_features"] = features
  return await _run("inspect", inputs, typed, options, output_dir, inline_cap)


@cisternal.tool(registry=REGISTRY)
async def jacobian(
  inputs: list[str],
  mode: Literal["categorical", "reverse"] | None = None,
  checkpoint_id: str | None = None,
  chain_id: str | list[str] | None = None,
  options: dict[str, Any] | None = None,
  output_dir: str | None = None,
  inline_cap: int = DEFAULT_INLINE_CAP,
) -> dict[str, Any]:
  """Compute a sequence Jacobian for protein backbones. Inputs are local structure file paths. The first call per checkpoint and shape compiles JAX and can take minutes; later calls with the same shapes are fast. Large arrays come back as a side-file path.

  Parameters
  ----------
  inputs : list of str
    Local structure file paths.
  mode : {"categorical", "reverse"} or None, optional
    Jacobian mode, passed as ``jacobian_mode``. ``None`` keeps the specification default
    (categorical).
  checkpoint_id : str or None, optional
    Packaged checkpoint id. ``None`` keeps the specification default.
  chain_id : str, list of str, or None, optional
    Chains to include. ``None`` keeps the specification default.
  options : dict or None, optional
    Other Jacobian-specification fields.
  output_dir : str or None, optional
    Directory for the npz side file.
  inline_cap : int, optional
    Maximum number of array elements kept in the JSON result.

  Returns
  -------
  dict
    Shaped Jacobian result. Gradient arrays are referenced from the side file
    when they exceed ``inline_cap``.
  """
  typed = {
    "jacobian_mode": mode,
    "checkpoint_id": checkpoint_id,
    "chain_id": chain_id,
  }
  return await _run("jacobian", inputs, typed, options, output_dir, inline_cap)


@cisternal.tool(registry=REGISTRY)
async def spec_emit(
  kind: Literal["sample", "score", "inspect", "jacobian"],
  inputs: list[str],
  options: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Emit a run specification as JSON without running the model. Inputs are local structure file paths.

  Parameters
  ----------
  kind : {"sample", "score", "inspect", "jacobian"}
    Specification class to build.
  inputs : list of str
    Local structure file paths.
  options : dict or None, optional
    Specification fields. Unknown keys are errors.

  Returns
  -------
  dict
    ``{"spec": ...}`` from :func:`aminx.agent.requests.spec_to_json_dict`.
  """
  spec = build_spec(kind, inputs, options)
  return {"spec": spec_to_json_dict(spec)}


@cisternal.tool(registry=REGISTRY)
async def spec_validate(spec: dict[str, Any]) -> dict[str, Any]:
  """Validate a run-specification JSON object without running the model.

  Parameters
  ----------
  spec : dict
    Object produced by :func:`spec_emit` or written by hand.

  Returns
  -------
  dict
    ``{"ok": True, "spec_class": ...}`` when the object decodes, or
    ``{"ok": False, "errors": [...]}`` when it does not. Validation failure
    is the normal answer, so decode errors are returned rather than raised.
  """
  try:
    decoded = spec_from_json(spec)
  except (SpecJSONDecodeError, ValueError, TypeError) as exc:
    return {"ok": False, "errors": [str(exc)]}
  return {"ok": True, "spec_class": type(decoded).__name__}


def _packaged_checkpoint_ids() -> set[str]:
  """Return checkpoint ids whose weight files are installed with the package.

  Returns
  -------
  set of str
    Filenames under ``aminx.model_params`` with ``.eqx.zst`` removed. Empty for a wheel
    install, whose weights come from the Hub.
  """
  try:
    root = importlib.resources.files("aminx.model_params")
    return {
      entry.name.removesuffix(".eqx.zst")
      for entry in root.iterdir()
      if entry.name.endswith(".eqx.zst")
    }
  except (ModuleNotFoundError, FileNotFoundError, NotADirectoryError):
    return set()


def _checkpoint_ids() -> list[str]:
  """Return every known and packaged checkpoint id, sorted.

  Returns
  -------
  list of str
    ``KNOWN_CHECKPOINT_IDS`` plus any extra packaged id.
  """
  return sorted(set(KNOWN_CHECKPOINT_IDS) | _packaged_checkpoint_ids())


def _json_topology(checkpoint_id: str) -> dict[str, Any]:
  """Return checkpoint topology as a JSON object.

  Parameters
  ----------
  checkpoint_id : str
    Bare checkpoint id.

  Returns
  -------
  dict
    Topology mapping with JSON scalars.
  """
  topology = to_jsonable(weights_io.get_topology_for_checkpoint(checkpoint_id))
  if not isinstance(topology, dict):
    msg = f"topology for {checkpoint_id} is not a JSON object"
    raise TypeError(msg)
  return topology


def _legacy_alias_map() -> dict[str, str]:
  """Return the legacy alias map with JSON object keys.

  Tuple keys ``(model_weights, model_version)`` become ``weights/version``.

  Returns
  -------
  dict
    Alias filename keyed by ``model_weights/model_version``.
  """
  return {
    f"{weights}/{version}": filename
    for (weights, version), filename in weights_io.LEGACY_ALIAS_MAP.items()
  }


@cisternal.tool(registry=REGISTRY)
async def list_checkpoints(with_sha256: bool = False) -> dict[str, Any]:  # noqa: FBT001, FBT002
  """List loadable checkpoints, the default id, and the legacy alias map. `packaged` says whether the weights are installed locally; otherwise they download from the Hub on first use. When with_sha256 is true this hashes every checkpoint file and may download missing weights.

  Parameters
  ----------
  with_sha256 : bool, optional
    When true, add ``sha256``, ``source``, and ``hub_revision`` from
    :func:`aminx.io.weights.weight_provenance`. Hashing reads every file
    and may download it from the hub.

  Returns
  -------
  dict
    ``checkpoints``, ``default_checkpoint``, and ``legacy_alias_map``.
  """
  packaged = _packaged_checkpoint_ids()
  entries: list[dict[str, Any]] = []
  for checkpoint_id in _checkpoint_ids():
    entry: dict[str, Any] = {
      "checkpoint_id": checkpoint_id,
      "packaged": checkpoint_id in packaged,
      "topology": _json_topology(checkpoint_id),
    }
    if with_sha256:
      provenance = weights_io.weight_provenance(checkpoint_id)
      entry["sha256"] = provenance.sha256
      entry["source"] = provenance.source
      entry["hub_revision"] = provenance.hub_revision
    entries.append(entry)
  return {
    "checkpoints": entries,
    "default_checkpoint": DEFAULT_CHECKPOINT_ID,
    "legacy_alias_map": _legacy_alias_map(),
  }
