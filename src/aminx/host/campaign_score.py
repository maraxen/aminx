"""Campaign row kind ``task="score_conditionals"``: teacher-forced, mask-conditioned logits (#3486).

A design-arm screen needs, for ONE fixed structure and context, the per-position conditional
logits of a few fixed sequences under K matched visibility masks (decoding orders), so that two
conditions scored under the same K masks can be differenced position by position. Before this row
kind the only route was ``runner.score(..., ar_mask=(L, L), return_logits=True)`` once per mask,
which bypassed the campaign layer, so the consumer re-implemented done markers, input hashes,
locks and resume. A ``score_conditionals`` row runs through :func:`aminx.host.campaign.run_manifest_row`
with the same lock, done-marker, input-hash and quarantine semantics as a sampling row.

Row contract
------------
A row is a JSON object with ``task = "score_conditionals"`` and a ``unit_spec`` payload (sampling rows
keep ``sampling_spec``; :func:`aminx.host.campaign._row_unit_payload` reads either). The payload
names everything that determines the output:

* ``inputs``: exactly one structure file; ``chain_id``: chain selection (a list keeps, e.g., a
  protein chain and its own peptide copy);
* ``checkpoint_id`` (weights hashed by the campaign input hash, revision included via the weights
  env) and the derived ``model_family``;
* ``sequences``: N fixed sequences as MPNN-alphabet LETTERS, all of the masks' length L. Letters,
  not token ids: an AF-order token array is a valid-looking MPNN token array and silently scores
  the wrong residues, so ids are never accepted here and the letter round trip is asserted;
* ``ar_masks_path``: a ``.npy`` of shape ``(K, L, L)``, ``ar_masks[k, i, j] == 1`` iff position
  ``i`` SEES position ``j``; zero diagonal; no all-zero mask; no all-zero row at a designable
  position. Two rows of a matched pair reference the same file, so "same K orders" is checkable
  by its sha256, which enters both the row hash and the unit input hash;
* ``designable_mask``: ``(L,)`` 0/1; ``fixed_position_context``: a free-text label of the rule
  the masks were built with (recorded and hashed, not interpreted);
* ``ligand_context_path`` (optional, LigandMPNN only): its bytes enter the input hash;
* ``random_seed``: recorded; the scores do not depend on it (pinned by a test);
* ``max_length``, ``logits_memory_budget_mb`` and the campaign flags.

Output: a Zarr store with ``logits`` ``(K, N, L, 21)`` float32 (trimmed to the L scored
residues), ``nll`` ``(K, N)``, and root attrs naming the schema, sequences, structure, checkpoint,
mask sha256 and mask semantics. Temperature is not an input: teacher-forced logits do not depend
on it, and a ``temperature`` key in the payload is refused.

task_id: 261118_aminx-campaign-teacher-forced-scoring
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCORE_CONDITIONALS_TASK = "score_conditionals"
SCORE_CONDITIONALS_SCHEMA_VERSION = "score_conditionals_v1"
SCORE_CONDITIONALS_ROW_SCHEMA_VERSION = "score_conditionals_row_v1"
MASK_SEMANTICS = "ar_masks[k, i, j] == 1 iff position i SEES position j; diagonal 0"
_MPNN_LETTERS = frozenset("ACDEFGHIKLMNPQRSTVWYX")
_REFUSED_PAYLOAD_KEYS = ("temperature", "sequence_tokens", "orders")


@dataclass(frozen=True)
class ScoreConditionalsUnit:
  """One unit of a teacher-forced conditional-logits screen: one structure, N sequences, K masks.

  Parameters
  ----------
  label : str
    Condition name (e.g. ``"P1p_E__ligand_on"``); part of the row hash and the gate cell.
  structure : str
    Path to the structure file.
  checkpoint_id : str
    Weights identifier (``proteinmpnn_v_48_020``, ``solublempnn_v_48_020``,
    ``ligandmpnn_v_32_020_25``, ...).
  sequences : Sequence[str]
    N sequences in MPNN-alphabet letters, each exactly L long.
  ar_masks_path : str
    ``.npy`` holding the ``(K, L, L)`` int mask stack.
  designable_mask : Sequence[int]
    ``(L,)`` 0/1 designable positions.
  logits_memory_budget_mb : int
    Upper bound on the unit's logits (``K * N * L * 21 * 4`` bytes) -- refused at plan time when
    exceeded, so an oversized unit never reaches a worker.
  chain_id : str | Sequence[str] | None
    Chain selection, passed to the loader as ``ScoringSpecification.chain_id``.
  ligand_context_path : str | None
    LigandMPNN ligand context (``.npz``); requires a ``ligandmpnn`` checkpoint.
  fixed_position_context : str
    Label of the mask-construction rule, recorded and hashed.
  random_seed : int
    Recorded; scores do not depend on it.
  max_length : int
    Loader padding length (the runner trims to its length rung).
  """

  label: str
  structure: str
  checkpoint_id: str
  sequences: Sequence[str]
  ar_masks_path: str
  designable_mask: Sequence[int]
  logits_memory_budget_mb: int
  chain_id: str | Sequence[str] | None = None
  ligand_context_path: str | None = None
  fixed_position_context: str = "explicit_masks"
  random_seed: int = 0
  max_length: int = 512
  extra: Mapping[str, Any] = field(default_factory=dict)


def _sha256(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def model_family_for_checkpoint(checkpoint_id: str) -> str:
  """``ligandmpnn`` for LigandMPNN weights, else ``proteinmpnn`` (ProteinMPNN, SolubleMPNN)."""
  return "ligandmpnn" if checkpoint_id.startswith("ligandmpnn") else "proteinmpnn"


def validate_mpnn_sequences(sequences: Sequence[str]) -> int:
  """Assert MPNN-alphabet letters of one common length; return that length.

  Round-trips each sequence through ``aminx``'s own MPNN tokeniser so an alphabet mismatch is
  caught here, not as quietly wrong logits.
  """
  from aminx.utils.aa_convert import (  # noqa: PLC0415
    protein_sequence_to_string,
    string_to_protein_sequence,
  )

  if not sequences:
    msg = "score_conditionals: at least one sequence is required."
    raise ValueError(msg)
  lengths = set()
  for seq in sequences:
    if not isinstance(seq, str):
      msg = (
        f"score_conditionals: sequences must be MPNN-alphabet letter strings, got {type(seq).__name__}. "
        "Token ids are refused: an AF-order id array scores the wrong residues without error."
      )
      raise TypeError(msg)
    bad = sorted(set(seq) - _MPNN_LETTERS)
    if bad:
      msg = f"score_conditionals: sequence has non-MPNN letters {bad}."
      raise ValueError(msg)
    if protein_sequence_to_string(string_to_protein_sequence(seq)) != seq:
      msg = f"score_conditionals: MPNN tokenise/detokenise round trip changed {seq!r}."
      raise ValueError(msg)
    lengths.add(len(seq))
  if len(lengths) != 1:
    msg = f"score_conditionals: all sequences must share one length, got {sorted(lengths)}."
    raise ValueError(msg)
  return lengths.pop()


def load_and_validate_masks(path: str | Path, designable_mask: Sequence[int], length: int) -> np.ndarray:
  """Load the ``(K, L, L)`` stack and apply ScoringSpecification's mask + designable checks."""
  from aminx.run.specs import _validated_designable_mask, _validated_scoring_ar_masks  # noqa: PLC0415

  masks = _validated_scoring_ar_masks(np.load(Path(path), allow_pickle=False))
  if masks.shape[-1] != length:
    msg = f"score_conditionals: masks cover {masks.shape[-1]} positions but the sequences have {length}."
    raise ValueError(msg)
  _validated_designable_mask(np.asarray(designable_mask), masks)
  return masks


def logits_bytes(k: int, n: int, length: int) -> int:
  """Bytes of the unit's float32 ``(K, N, L, 21)`` logits."""
  return int(k) * int(n) * int(length) * 21 * 4


def build_score_conditionals_row(
  unit: ScoreConditionalsUnit,
  *,
  campaign_id: str,
  job_index: int,
  output_root: str | Path,
  git_sha: str,
  git_provenance_source: str,
  planner_version: str = "planner_v1",
  dataset_fingerprint: str = "unknown",
  environment_image: str = "unknown",
  config_hash: str = "unknown",
) -> dict[str, Any]:
  """Validate one unit and build its manifest row (hash, gate fields, ``unit_spec`` payload)."""
  from aminx.host.campaign import _row_output_path  # noqa: PLC0415 -- campaign imports this module lazily
  from xtrax.run import canonical_json_bytes  # noqa: PLC0415

  length = validate_mpnn_sequences(unit.sequences)
  masks = load_and_validate_masks(unit.ar_masks_path, unit.designable_mask, length)
  k, n = int(masks.shape[0]), len(unit.sequences)
  family = model_family_for_checkpoint(unit.checkpoint_id)
  if unit.ligand_context_path is not None:
    if family != "ligandmpnn":
      msg = (
        f"score_conditionals: ligand_context_path is set but {unit.checkpoint_id!r} has no ligand "
        "channel; the ligand would be ignored."
      )
      raise ValueError(msg)
    if not Path(unit.ligand_context_path).is_file():
      msg = f"score_conditionals: ligand_context_path {unit.ligand_context_path!r} is not a file."
      raise FileNotFoundError(msg)
  if not Path(unit.structure).is_file():
    msg = f"score_conditionals: structure {unit.structure!r} is not a file."
    raise FileNotFoundError(msg)
  budget = int(unit.logits_memory_budget_mb)
  if budget <= 0 or logits_bytes(k, n, length) > budget * (1 << 20):
    msg = (
      f"score_conditionals: unit {unit.label!r} holds {logits_bytes(k, n, length) / (1 << 20):.1f} MiB of "
      f"logits (K={k}, N={n}, L={length}) but logits_memory_budget_mb={budget}. Split the masks "
      "across rows or raise the budget."
    )
    raise ValueError(msg)
  refused = [key for key in _REFUSED_PAYLOAD_KEYS if key in unit.extra]
  if refused:
    msg = f"score_conditionals: payload keys {refused} are not inputs of this row kind."
    raise ValueError(msg)

  masks_sha = _sha256(unit.ar_masks_path)
  ligand_sha = None if unit.ligand_context_path is None else _sha256(unit.ligand_context_path)
  chain = None if unit.chain_id is None else (
    [unit.chain_id] if isinstance(unit.chain_id, str) else [str(c) for c in unit.chain_id]
  )
  hash_payload = {
    "schema_version": SCORE_CONDITIONALS_ROW_SCHEMA_VERSION,
    "task": SCORE_CONDITIONALS_TASK,
    "campaign_id": campaign_id,
    "label": unit.label,
    "structure_sha256": _sha256(unit.structure),
    "chain_id": chain,
    "checkpoint_id": unit.checkpoint_id,
    "sequences": list(unit.sequences),
    "ar_masks_sha256": masks_sha,
    "designable_mask": [int(v) for v in unit.designable_mask],
    "ligand_context_sha256": ligand_sha,
    "fixed_position_context": unit.fixed_position_context,
    "random_seed": int(unit.random_seed),
    "max_length": int(unit.max_length),
  }
  row_hash = hashlib.sha256(canonical_json_bytes(hash_payload)).hexdigest()
  output_path = _row_output_path(Path(output_root), campaign_id=campaign_id, row_hash=row_hash)
  unit_spec = {
    "task": SCORE_CONDITIONALS_TASK,
    "inputs": [str(unit.structure)],
    "chain_id": chain,
    "checkpoint_id": unit.checkpoint_id,
    "model_family": family,
    "sequences": list(unit.sequences),
    "ar_masks_path": str(unit.ar_masks_path),
    "ar_masks_sha256": masks_sha,
    "designable_mask": [int(v) for v in unit.designable_mask],
    "fixed_position_context": unit.fixed_position_context,
    "ligand_context_path": None if unit.ligand_context_path is None else str(unit.ligand_context_path),
    "ligand_context_sha256": ligand_sha,
    "random_seed": int(unit.random_seed),
    "max_length": int(unit.max_length),
    "backbone_noise": [0.0],
    "campaign_mode": True,
    "grid_mode": True,
    "return_logits": True,
    "allow_logits_in_campaign": True,
    "logits_memory_budget_mb": budget,
    "output_path": output_path,
    "output_zarr_path": output_path,
    "output_h5_path": output_path,
  }
  return {
    "manifest_row_hash": row_hash,
    "task": SCORE_CONDITIONALS_TASK,
    "label": unit.label,
    "job_id": f"{campaign_id}-job-{job_index}",
    "job_index": int(job_index),
    "chunk_id": 0,
    "sample_start": 0,
    "sample_count": k * n,
    # Gate-cell fields shared with sampling rows. A score row designs nothing and fixes nothing,
    # so its arm is the no-arm sentinel; the condition is carried by ``label``.
    "fixed_policy": "none",
    "state_weight_profile": "equal",
    "multi_state_strategy": "arithmetic_mean",
    "temperature": [],
    "backbone_noise": [0.0],
    "ligand_conditioning": unit.ligand_context_path is not None,
    "sidechain_conditioning": False,
    "checkpoint_id": unit.checkpoint_id,
    "model_family": family,
    "n_masks": k,
    "n_sequences": n,
    "length": length,
    "planner_version": planner_version,
    "dataset_fingerprint": dataset_fingerprint,
    "environment_image": environment_image,
    "git_sha": git_sha,
    "git_provenance_source": git_provenance_source,
    "config_hash": config_hash,
    "output_h5_path": output_path,
    "unit_spec": unit_spec,
  }


def plan_score_conditionals_manifest(
  units: Sequence[ScoreConditionalsUnit],
  *,
  campaign_id: str,
  output_root: str | Path,
  git_sha: str | None = None,
  git_provenance_source: str | None = None,
  planner_version: str = "planner_v1",
  dataset_fingerprint: str = "unknown",
  environment_image: str = "unknown",
  config_hash: str = "unknown",
) -> list[dict[str, Any]]:
  """Plan one ``score_conditionals`` row per unit; validated like every other manifest."""
  from aminx.host.campaign import _resolve_git_provenance, validate_manifest_rows  # noqa: PLC0415

  if not units:
    msg = "plan_score_conditionals_manifest: no units; an empty manifest is not a plan."
    raise ValueError(msg)
  sha, source = _resolve_git_provenance(git_sha, git_provenance_source)
  rows = [
    build_score_conditionals_row(
      unit,
      campaign_id=campaign_id,
      job_index=index,
      output_root=output_root,
      git_sha=sha,
      git_provenance_source=source,
      planner_version=planner_version,
      dataset_fingerprint=dataset_fingerprint,
      environment_image=environment_image,
      config_hash=config_hash,
    )
    for index, unit in enumerate(units)
  ]
  validate_manifest_rows(rows)
  return rows


def task_input_files(payload: Mapping[str, Any]) -> dict[str, Any]:
  """Files whose BYTES determine a score_conditionals unit, for ``compute_unit_input_hash``."""
  from aminx.host.campaign import _hash_input_entry  # noqa: PLC0415

  files: dict[str, Any] = {"ar_masks": _hash_input_entry(str(payload["ar_masks_path"]))}
  if payload.get("ligand_context_path"):
    files["ligand_context"] = _hash_input_entry(str(payload["ligand_context_path"]))
  return files


def scoring_spec_for_unit(payload: Mapping[str, Any], masks: np.ndarray) -> Any:  # noqa: ANN401
  """The ScoringSpecification a worker runs for this unit."""
  from aminx.run.specs import ScoringSpecification  # noqa: PLC0415

  ligand = payload.get("ligand_context_path")
  return ScoringSpecification(
    inputs=list(payload["inputs"]),
    chain_id=payload.get("chain_id"),
    checkpoint_id=str(payload["checkpoint_id"]),
    model_family=str(payload["model_family"]),
    sequences_to_score=list(payload["sequences"]),
    ar_masks=masks,
    designable_mask=np.asarray(payload["designable_mask"], np.int8),
    ligand_context_path=ligand,
    ligand_conditioning=bool(ligand),
    return_logits=True,
    backbone_noise=0.0,
    random_seed=int(payload["random_seed"]),
    max_length=int(payload["max_length"]),
  )


def execute_score_conditionals_unit(payload: Mapping[str, Any], partial_path: Path) -> dict[str, Any]:
  """Score one unit and write its store at ``partial_path``; the caller commits it."""
  from aminx.host.runner import score  # noqa: PLC0415
  from aminx.io.sink_provenance import resolve_aminx_version  # noqa: PLC0415

  refused = [key for key in _REFUSED_PAYLOAD_KEYS if key in payload]
  if refused:
    msg = f"score_conditionals: payload keys {refused} are not inputs of this row kind."
    raise ValueError(msg)
  length = validate_mpnn_sequences(payload["sequences"])
  if _sha256(payload["ar_masks_path"]) != payload["ar_masks_sha256"]:
    msg = (
      f"score_conditionals: {payload['ar_masks_path']} no longer matches the sha256 the row was "
      "planned with; re-plan the manifest rather than score different masks under this row."
    )
    raise ValueError(msg)
  masks = load_and_validate_masks(payload["ar_masks_path"], payload["designable_mask"], length)
  if logits_bytes(masks.shape[0], len(payload["sequences"]), length) > int(payload["logits_memory_budget_mb"]) * (1 << 20):
    msg = "score_conditionals: unit exceeds logits_memory_budget_mb."
    raise ValueError(msg)
  result = score(scoring_spec_for_unit(payload, masks))
  logits = np.asarray(result["logits"])[0][..., :length, :].astype(np.float32)  # (K, N, L, 21)
  nll = np.asarray(result["scores"])[0].astype(np.float32)  # (K, N)
  structure_ids = result["metadata"].get("structure_ids") or ["structure_0"]
  write_score_conditionals_store(
    partial_path,
    logits=logits,
    nll=nll,
    attrs={
      "schema_version": SCORE_CONDITIONALS_SCHEMA_VERSION,
      "task": SCORE_CONDITIONALS_TASK,
      "structure_id": str(structure_ids[0]),
      "inputs": list(payload["inputs"]),
      "chain_id": payload.get("chain_id"),
      "checkpoint_id": payload["checkpoint_id"],
      "model_family": payload["model_family"],
      "sequences": list(payload["sequences"]),
      "ar_masks_sha256": payload["ar_masks_sha256"],
      "mask_semantics": MASK_SEMANTICS,
      "designable_mask": list(payload["designable_mask"]),
      "fixed_position_context": payload.get("fixed_position_context"),
      "ligand_context_sha256": payload.get("ligand_context_sha256"),
      "random_seed": int(payload["random_seed"]),
      "axes": {"logits": ["mask", "sequence", "position", "token"], "nll": ["mask", "sequence"]},
      "aminx_version": resolve_aminx_version() or "unknown",
    },
  )
  return {"status": "completed", "logits_shape": list(logits.shape), "nll": nll}


def write_score_conditionals_store(
  path: Path,
  *,
  logits: np.ndarray,
  nll: np.ndarray,
  attrs: Mapping[str, Any],
) -> None:
  """Write the unit store (Zarr v2 layout, as the other campaign stores)."""
  import zarr  # noqa: PLC0415

  try:
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
  except TypeError:  # zarr 2
    group = zarr.open_group(str(path), mode="w")
  group.attrs.update(dict(attrs))
  for name, data in (("logits", logits), ("nll", nll)):
    if hasattr(group, "create_array"):  # zarr 3
      group.create_array(name, data=data)
    else:
      group.create_dataset(name, data=data)


def read_score_conditionals_store(path: str | Path) -> dict[str, Any]:
  """Read a unit store back: ``{"logits": (K, N, L, 21), "nll": (K, N), "attrs": {...}}``."""
  import zarr  # noqa: PLC0415

  group = zarr.open_group(str(path), mode="r")
  return {
    "logits": np.asarray(group["logits"][...]),
    "nll": np.asarray(group["nll"][...]),
    "attrs": dict(group.attrs),
  }
