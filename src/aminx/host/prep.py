"""Core user interface for the Aminx package."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from io import StringIO

  from grain.python import IterDataset

  from aminx.run.specs import Specs
  from aminx.types.arrays import Model

import equinox as eqx
import jax
from proxide.ops.dataset import create_protein_dataset

from aminx.host.family_driver import FAMILY_DRIVERS
from aminx.io.weights import load_model
from aminx.run.resources import proxide_dataset_resource_kwargs


def _loader_inputs(inputs: Sequence[str | StringIO] | str | StringIO) -> Sequence[str | StringIO]:
  return (inputs,) if not isinstance(inputs, Sequence) else inputs  # type: ignore[invalid-return-type]


def _artifact_path_for_registry_entry(registry_path: Path, artifact_path: str) -> Path:
  path = Path(artifact_path)
  return path if path.is_absolute() else (registry_path.parent / path).resolve()


def _resolve_local_checkpoint_from_registry(spec: Specs) -> str | None:
  if spec.checkpoint_registry_path is None:
    return None
  if spec.checkpoint_id is None:
    msg = "checkpoint_id is required when checkpoint_registry_path is set"
    raise ValueError(msg)
  reg_path = spec.checkpoint_registry_path
  payload = json.loads(reg_path.read_text(encoding="utf-8"))
  entries = payload.get("entries", [])
  found = None
  for entry in entries:
    if (
      entry.get("model_family") == spec.model_family
      and entry.get("checkpoint_id") == spec.checkpoint_id
    ):
      found = entry
      break
  if found is None:
    msg = (
      f"No checkpoint registry entry found for model_family={spec.model_family!r} "
      f"checkpoint_id={spec.checkpoint_id!r}"
    )
    raise ValueError(msg)
  checkpoint_name = spec.checkpoint_id if isinstance(spec.checkpoint_id, str) else ""
  driver_family = spec.model_family in {"pottsmpnn", "protonpottsmpnn", "lasermpnn"} or checkpoint_name.startswith(
    (
      "pottsmpnn_",
      "protonpottsmpnn_",
      "lasermpnn_",
    ),
  )
  if driver_family and "sha256" not in found:
    msg = (
      f"Registry entry for model_family={spec.model_family!r} "
      f"checkpoint_id={spec.checkpoint_id!r} must include sha256"
    )
    raise ValueError(msg)
  artifact = _artifact_path_for_registry_entry(reg_path, str(found["artifact_path"]))
  if "sha256" in found:
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != found["sha256"]:
      msg = f"Registry checksum mismatch for {artifact}: expected {found['sha256']}, got {digest}"
      raise ValueError(msg)
  return str(artifact)


def prep_protein_stream_and_model(
  spec: Specs,
) -> tuple[IterDataset, Model]:
  """Prepare the protein data stream and model parameters.

  Args:
      spec: A RunSpecification object containing configuration options.

  Returns:
      A tuple containing the protein data iterator and model in inference mode.

  Checkpoint resolution for :func:`~aminx.io.weights.load_model`:

  - If ``spec.model_local_path`` is set, its path is passed as ``local_path`` and any
    ``checkpoint_registry_path`` entry is **not** read (explicit local artifact wins).
  - Else if ``spec.checkpoint_registry_path`` is set, the registry JSON must contain
    top-level ``"entries"``: a list of objects with ``model_family``,
    ``checkpoint_id``, ``artifact_path`` (relative paths resolve next to the registry
    file), and optional ``sha256``. ``pottsmpnn`` and ``lasermpnn`` entries (and any
    ``pottsmpnn_`` / ``lasermpnn_`` checkpoint id) require ``sha256``. The entry
    matching ``spec.model_family`` and ``spec.checkpoint_id`` is chosen; if ``sha256``
    is present it must match the artifact bytes or :class:`ValueError` is raised.
    Missing entry raises :class:`ValueError`. ``checkpoint_id`` must be set on the
    spec when using the registry.
  - Otherwise weights load from packaged resources via ``checkpoint_id`` /
    ``model_weights`` + ``model_version`` as in :func:`load_model`.

  """
  parse_kwargs = {
    "chain_id": spec.chain_id,
    "model": spec.model,
    "altloc": spec.altloc,
    "topology": spec.topology,
  }
  ram_mb, workers, buf = proxide_dataset_resource_kwargs(spec, context="inference")
  protein_iterator = create_protein_dataset(
    _loader_inputs(spec.inputs),
    batch_size=spec.batch_size,
    foldcomp_database=spec.foldcomp_database,
    parse_kwargs=parse_kwargs,
    pass_mode=spec.pass_mode,
    use_preprocessed=spec.use_preprocessed,
    preprocessed_index_path=spec.preprocessed_index_path,
    split=spec.split,
    use_electrostatics=spec.use_electrostatics,
    estat_noise=spec.estat_noise,
    estat_noise_mode=spec.estat_noise_mode,
    use_vdw=spec.use_vdw,
    vdw_noise=spec.vdw_noise,
    vdw_noise_mode=spec.vdw_noise_mode,
    max_length=spec.max_length,
    truncation_strategy=spec.truncation_strategy,
    ram_budget_mb=ram_mb,
    max_workers=workers,
    max_buffer_size=buf,
  )
  local_ckpt = None
  if spec.model_local_path is not None:
    local_ckpt = str(spec.model_local_path)
  elif reg_path := _resolve_local_checkpoint_from_registry(spec):
    local_ckpt = reg_path

  if spec.run_spec.io.cache_path is not None:
    jax.config.update("jax_compilation_cache_dir", str(spec.run_spec.io.cache_path))

  # These three MUST reach load_model, not just the dataset. They change how the model is
  # BUILT, not merely what it is fed:
  #   - use_side_chain_context -> weights.py's ligand skeleton (ligand models only; the
  #     protein branch ignores it, so forwarding is safe for every checkpoint family)
  #   - use_electrostatics / use_vdw -> physics_feature_dim (weights.py:206-207)
  # Until now prep.py forwarded electrostatics/vdw to the DATASET (:111, :114) and nothing to
  # load_model, so a caller asking for sidechain context got a model built with sidechains
  # off and no error -- occurrences #6 and #11 of the audit's declared-but-never-delivered
  # class. training/trainer.py:159-164 already forwards these; the inference path did not.
  #
  # bool(...) because the spec fields are `bool | None` (specs.py:241,254,255) while
  # load_model takes plain bools -- None means "not set", i.e. off.
  #
  # If a checkpoint has no physics weights and the caller asks for physics, deserialization
  # now fails loudly. That is the correct outcome: it is what "I asked for electrostatics"
  # should do when the checkpoint cannot provide them. Silently ignoring the request is the
  # bug being fixed.
  #
  # sidechain_conditioning IMPLIES use_side_chain_context. These are two different
  # switches for one capability: sidechain_conditioning makes _sampling_helper build
  # and pass the atom_37 sidechain context, while use_side_chain_context is what
  # actually BUILDS the model branch that consumes it. Requesting the former while
  # leaving the latter unset (None -> False) fed real sidechain context to a model
  # that structurally ignored it -- a silent no-op, the same declared-but-never-
  # delivered class as the bug the comment above describes. Confirmed empirically
  # 2026-07-28: with sidechain_conditioning=True and the ctx flag None, AR-decode
  # logits were BIT-IDENTICAL to sidechain_conditioning=False (max|diff|=0.0); with
  # the ctx flag set, the same context moved them (max|diff|=2.47). This silently
  # invalidated the SC arms of a whole tev_design campaign (its "SC conditioning is
  # inert" conclusion was measuring a no-op, not the feature).
  #
  # An explicit `ligand_mpnn_use_side_chain_context=False` alongside
  # sidechain_conditioning=True is contradictory, so it raises rather than silently
  # picking one -- never resolve a contradiction by guessing.
  if spec.sidechain_conditioning and spec.ligand_mpnn_use_side_chain_context is False:
    msg = (
      "sidechain_conditioning=True with ligand_mpnn_use_side_chain_context=False is "
      "contradictory: the sidechain context would be built and passed, but the model "
      "would be built without the branch that consumes it (a silent no-op). Set "
      "ligand_mpnn_use_side_chain_context=True (or leave it unset to have it implied), "
      "or set sidechain_conditioning=False."
    )
    raise ValueError(msg)

  model_conditioning = {
    "use_side_chain_context": bool(spec.ligand_mpnn_use_side_chain_context)
    or bool(spec.sidechain_conditioning),
    "use_electrostatics": bool(spec.use_electrostatics),
    "use_vdw": bool(spec.use_vdw),
  }

  driver = FAMILY_DRIVERS.get(spec.model_family)
  if driver is not None:
    # Fallback purposes: the registered driver loaded the composite module and
    # the MPNN path continues on its embedded core. Handled purposes never reach
    # this function.
    loaded = driver.load(spec)
    core = driver.mpnn_core(loaded)
    if core is None:
      msg = f"{driver.name} does not provide an MPNN core for fallback purposes"
      raise ValueError(msg)
    # Same inference-mode step as the load_model branches. Returned as Model so
    # the MPNN callers of this function keep their previous type.
    return protein_iterator, eqx.nn.inference_mode(core, value=True)  # type: ignore[return-value]
  if spec.checkpoint_id is not None:
    model = load_model(
      checkpoint_id=spec.checkpoint_id,
      model_weights=spec.model_weights,
      model_version=spec.model_version,
      local_path=local_ckpt,
      **model_conditioning,
    )
  else:
    model = load_model(
      model_version=spec.model_version,
      model_weights=spec.model_weights,
      local_path=local_ckpt,
      **model_conditioning,
    )

  # Set model to inference mode (disables dropout, etc.)
  model = eqx.nn.inference_mode(model, value=True)

  return protein_iterator, model
