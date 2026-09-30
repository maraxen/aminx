"""Convert an in-scope PottsMPNN torch checkpoint into an Equinox PottsMPNN.

Reuses ``scripts/convert_weights.py`` for the ProteinMPNN tensors. The only new
map is ``etab_out.weight`` / ``etab_out.bias``. A checkpoint without those keys
(the ``proteinmpnn_compatible_model_weights`` files) raises and is never written
into a ``pottsmpnn`` registry fragment.

Writes the converted artifact and a registry-entry JSON fragment whose
``sha256`` is the converted file, ``source_sha256`` is the torch file, and
``upstream_commit`` is the pinned PottsMPNN revision.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np

from aminx.families.potts_mpnn.model import PottsMPNN
from aminx.families.potts_mpnn.potts_head import PottsHead

UPSTREAM_COMMIT = "0cb0a58874e373d664114f1735deab77614e4ab5"
ETAB_WEIGHT = "etab_out.weight"
ETAB_BIAS = "etab_out.bias"
REGISTRY_REQUIRED = ("sha256", "source_sha256", "upstream_commit")

# Five T0.2 in-scope checkpoints. Ids match oracle_manifest / pottsmpnn_full keys.
IN_SCOPE_CHECKPOINTS: tuple[tuple[str, str], ...] = (
  ("vanilla_20", "vanilla_model_weights/pottsmpnn_20.pt"),
  ("vanilla_30", "vanilla_model_weights/pottsmpnn_30.pt"),
  ("soluble_20", "soluble_model_weights/sol_pottsmpnn_20.pt"),
  ("soluble_30", "soluble_model_weights/sol_pottsmpnn_30.pt"),
  ("ft", "ft_model_weights/potts_ft.pt"),
)

# Duplicate exclusion (§4.1). These ids must never appear under model_family pottsmpnn.
PROTEINMPNN_COMPATIBLE_IDS: tuple[str, ...] = (
  "proteinmpnn_compatible_pottsmpnn_20",
  "proteinmpnn_compatible_pottsmpnn_30",
  "proteinmpnn_compatible_pottsmpnn_msa_20",
  "proteinmpnn_compatible_sol_pottsmpnn_20",
  "proteinmpnn_compatible_sol_pottsmpnn_30",
  "proteinmpnn_compatible_sol_pottsmpnn_msa_20",
)


def _convert_weights():
  path = Path(__file__).resolve().parents[1] / "convert_weights.py"
  spec = importlib.util.spec_from_file_location("aminx_convert_weights", path)
  if spec is None or spec.loader is None:
    msg = f"cannot load {path}"
    raise ImportError(msg)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def _as_numpy_state(state: dict[str, Any]) -> dict[str, np.ndarray]:
  arrays: dict[str, np.ndarray] = {}
  for key, value in state.items():
    if hasattr(value, "detach"):
      value = value.detach().cpu().numpy()
    arrays[key] = np.asarray(value)
  return arrays


def require_etab_out(state: dict[str, Any]) -> None:
  """Raise if ``etab_out.*`` is missing.

  ProteinMPNN-only files load under Potts with a randomly initialised head.
  Conversion refuses them so they cannot be registered as ``pottsmpnn``.
  """
  missing = [key for key in (ETAB_WEIGHT, ETAB_BIAS) if key not in state]
  if missing:
    msg = (
      "missing etab_out.* "
      f"{missing}; a proteinmpnn_compatible checkpoint cannot be converted to PottsMPNN"
    )
    raise ValueError(msg)


def _reject_compatible_id(checkpoint_id: str, source: Path | None = None) -> None:
  text = checkpoint_id
  if source is not None:
    text = f"{text} {source.as_posix()}"
  if "proteinmpnn_compatible" in text:
    msg = f"checkpoint_id {checkpoint_id!r} is a proteinmpnn_compatible duplicate and is not pottsmpnn"
    raise ValueError(msg)


def validate_pottsmpnn_registry(entries: list[dict[str, Any]]) -> None:
  """Reject compatible ids and pottsmpnn rows that omit the required hashes."""
  registered = {checkpoint_id for checkpoint_id, _path in IN_SCOPE_CHECKPOINTS}
  for entry in entries:
    if entry.get("model_family") != "pottsmpnn":
      continue
    checkpoint_id = str(entry.get("checkpoint_id", ""))
    artifact = str(entry.get("artifact_path", ""))
    if "proteinmpnn_compatible" in checkpoint_id or "proteinmpnn_compatible" in artifact:
      msg = f"proteinmpnn_compatible id {checkpoint_id!r} must not be registered under pottsmpnn"
      raise ValueError(msg)
    if checkpoint_id in PROTEINMPNN_COMPATIBLE_IDS:
      msg = f"{checkpoint_id} is excluded from the pottsmpnn registry"
      raise ValueError(msg)
    missing = [key for key in REGISTRY_REQUIRED if not entry.get(key)]
    if missing:
      msg = f"pottsmpnn registry entry {checkpoint_id!r} requires {missing}"
      raise ValueError(msg)
    if checkpoint_id not in registered:
      msg = f"checkpoint_id {checkpoint_id!r} is outside the five in-scope pottsmpnn checkpoints"
      raise ValueError(msg)


def registry_fragment(
  *,
  checkpoint_id: str,
  artifact_path: str,
  artifact_bytes: bytes,
  source_bytes: bytes,
  upstream_commit: str = UPSTREAM_COMMIT,
) -> dict[str, str]:
  """One registry object. ``sha256`` is the converted artifact prep loads."""
  _reject_compatible_id(checkpoint_id)
  fragment = {
    "model_family": "pottsmpnn",
    "checkpoint_id": checkpoint_id,
    "artifact_path": artifact_path,
    "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "upstream_commit": upstream_commit,
  }
  validate_pottsmpnn_registry([fragment])
  return fragment


def convert_state_dict(state: dict[str, Any], *, key: jax.Array | None = None) -> PottsMPNN:
  """Build a ``PottsMPNN`` from a torch-like state dict. Raises without ``etab_out``."""
  require_etab_out(state)
  arrays = _as_numpy_state(state)
  weights = _convert_weights()
  if key is None:
    key = jax.random.PRNGKey(0)
  model = PottsMPNN(key=key)
  mpnn = weights.convert_full_model(arrays, model.mpnn)
  model = eqx.tree_at(lambda module: module.mpnn, model, mpnn)
  linear: PottsHead = model.potts_head
  converted = weights.convert_linear_layer(arrays[ETAB_WEIGHT], arrays[ETAB_BIAS], linear.linear)
  return eqx.tree_at(lambda module: module.potts_head.linear, model, converted)


def write_converted(
  state: dict[str, Any],
  *,
  checkpoint_id: str,
  source_path: Path,
  output_path: Path,
  registry_path: Path,
  upstream_commit: str = UPSTREAM_COMMIT,
) -> dict[str, str]:
  """Serialise the Equinox artifact and the registry JSON fragment."""
  _reject_compatible_id(checkpoint_id, source_path)
  model = convert_state_dict(state)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  eqx.tree_serialise_leaves(output_path, model)
  fragment = registry_fragment(
    checkpoint_id=checkpoint_id,
    artifact_path=output_path.name,
    artifact_bytes=output_path.read_bytes(),
    source_bytes=source_path.read_bytes(),
    upstream_commit=upstream_commit,
  )
  registry_path.write_text(json.dumps(fragment, indent=2) + "\n", encoding="utf-8")
  return fragment


def _load_torch_state(path: Path) -> dict[str, Any]:
  try:
    import torch
  except ImportError as exc:
    msg = "torch is required to read a PottsMPNN checkpoint"
    raise ImportError(msg) from exc
  blob = torch.load(path, map_location="cpu", weights_only=False)
  if isinstance(blob, dict) and "model_state_dict" in blob:
    return blob["model_state_dict"]
  if isinstance(blob, dict):
    return blob
  msg = f"{path} did not contain a state dict"
  raise TypeError(msg)


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint-id", required=True)
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--registry-fragment", type=Path, required=True)
  parser.add_argument("--upstream-commit", default=UPSTREAM_COMMIT)
  args = parser.parse_args(argv)
  state = _load_torch_state(args.input)
  fragment = write_converted(
    state,
    checkpoint_id=args.checkpoint_id,
    source_path=args.input,
    output_path=args.output,
    registry_path=args.registry_fragment,
    upstream_commit=args.upstream_commit,
  )
  print(json.dumps({"checkpoint_id": fragment["checkpoint_id"], "sha256": fragment["sha256"]}))


if __name__ == "__main__":
  main()
