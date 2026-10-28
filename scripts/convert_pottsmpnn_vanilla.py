"""Convert the vanilla PottsMPNN torch checkpoint to the aminx ``PottsMPNN`` tree.

The vanilla checkpoint is the only PottsMPNN file that carries ``etab_out``; the
``proteinmpnn_compatible`` file has no Potts head and must not be used here.

The name map is explicit (torch key -> aminx leaf path) and self-checking:

* every torch tensor maps to exactly one aminx leaf, and every torch key is listed;
* every aminx parameter leaf is hit exactly once (the dropout ``p`` and
  ``inference`` scalars are hyperparameters, not weights, and are excluded by name);
* the torch shape equals the skeleton leaf shape for every mapped pair.

Any violation raises and lists the offending names. The skeleton is built from the
real constructor (``PottsMPNN(key=...)``) so constructor arguments cannot be skipped.

Usage:
    uv run --no-sync python scripts/convert_pottsmpnn_vanilla.py \\
        --checkpoint .scratch_potts/pottsmpnn_20_vanilla.pt
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aminx.families.potts_mpnn import PottsMPNN
from aminx.io.weights import verify_pinned_file

logger = logging.getLogger(__name__)

# sha256 of vanilla_model_weights/pottsmpnn_20.pt (VENDOR_PIN.toml in KeatingLab/PottsMPNN).
VANILLA_PIN_SHA256 = "77e797fd30fb4da11151d0d6f0d55d13ea25c00c45048dcc4aa634dc3620aa5c"

N_ENCODER_LAYERS = 3
N_DECODER_LAYERS = 3

# Torch encoder layer EncLayer (potts_mpnn_utils.py:829-875) -> aminx EncoderLayer.
# W1/W2/W3 form the edge message MLP, W11/W12/W13 the edge update MLP, dense is the FFN.
_ENCODER_MLP_LAYERS = {
  "W1": "edge_message_mlp.layers[0]",
  "W2": "edge_message_mlp.layers[1]",
  "W3": "edge_message_mlp.layers[2]",
  "W11": "edge_update_mlp.layers[0]",
  "W12": "edge_update_mlp.layers[1]",
  "W13": "edge_update_mlp.layers[2]",
}
# Torch DecLayer (potts_mpnn_utils.py:877-917) -> aminx DecoderLayer; no edge update MLP.
_DECODER_MLP_LAYERS = {
  "W1": "message_mlp.layers[0]",
  "W2": "message_mlp.layers[1]",
  "W3": "message_mlp.layers[2]",
}
_FFN_LAYERS = {"dense.W_in": "dense.layers[0]", "dense.W_out": "dense.layers[1]"}

# Dropout hyperparameters appear as array leaves in the skeleton but carry no weights.
_HYPERPARAMETER_ATTRS = ("p", "inference")


def build_name_map(
  n_encoder_layers: int = N_ENCODER_LAYERS,
  n_decoder_layers: int = N_DECODER_LAYERS,
) -> dict[str, str]:
  """Return ``{torch_key: aminx_leaf_path}`` for the vanilla PottsMPNN.

  Aminx paths use ``jax.tree_util.keystr`` form, e.g. ``.mpnn.encoder.layers[0].norm1.weight``.
  """
  name_map: dict[str, str] = {}

  def linear(torch_prefix: str, aminx_prefix: str) -> None:
    name_map[f"{torch_prefix}.weight"] = f"{aminx_prefix}.weight"
    name_map[f"{torch_prefix}.bias"] = f"{aminx_prefix}.bias"

  def layer_norm(torch_prefix: str, aminx_prefix: str) -> None:
    linear(torch_prefix, aminx_prefix)

  # Featurisation: torch ProteinFeatures (potts_mpnn_utils.py:1126) -> aminx ProteinFeatures.
  linear("features.embeddings.linear", ".mpnn.features.w_pos")
  name_map["features.edge_embedding.weight"] = ".mpnn.features.w_e.weight"  # aminx: no bias
  layer_norm("features.norm_edges", ".mpnn.features.norm_edges")
  # Torch W_e (PottsMPNN.__init__ line 1244) feeds the encoder; aminx keeps it in features.
  linear("W_e", ".mpnn.features.w_e_proj")
  name_map["W_s.weight"] = ".mpnn.w_s_embed.weight"
  linear("W_out", ".mpnn.w_out")
  # The Potts head (PottsMPNN.__init__ line 1261; run_encoder line 1793).
  linear("etab_out", ".potts_head.linear")

  for layer in range(n_encoder_layers):
    torch_prefix = f"encoder_layers.{layer}"
    aminx_prefix = f".mpnn.encoder.layers[{layer}]"
    for torch_name, aminx_name in _ENCODER_MLP_LAYERS.items():
      linear(f"{torch_prefix}.{torch_name}", f"{aminx_prefix}.{aminx_name}")
    for torch_name, aminx_name in _FFN_LAYERS.items():
      linear(f"{torch_prefix}.{torch_name}", f"{aminx_prefix}.{aminx_name}")
    for norm in ("norm1", "norm2", "norm3"):
      layer_norm(f"{torch_prefix}.{norm}", f"{aminx_prefix}.{norm}")

  for layer in range(n_decoder_layers):
    torch_prefix = f"decoder_layers.{layer}"
    aminx_prefix = f".mpnn.decoder.layers[{layer}]"
    for torch_name, aminx_name in _DECODER_MLP_LAYERS.items():
      linear(f"{torch_prefix}.{torch_name}", f"{aminx_prefix}.{aminx_name}")
    for torch_name, aminx_name in _FFN_LAYERS.items():
      linear(f"{torch_prefix}.{torch_name}", f"{aminx_prefix}.{aminx_name}")
    for norm in ("norm1", "norm2"):
      layer_norm(f"{torch_prefix}.{norm}", f"{aminx_prefix}.{norm}")

  return name_map


def skeleton_parameter_shapes(model: PottsMPNN) -> dict[str, tuple[int, ...]]:
  """Shapes of the weight leaves of ``model``, keyed by keystr path.

  Dropout hyperparameters are excluded, and the exclusion is asserted to touch only dropout
  modules so a real weight can never be silently dropped from the check.
  """
  shapes: dict[str, tuple[int, ...]] = {}
  for path, leaf in jax.tree_util.tree_flatten_with_path(eqx.filter(model, eqx.is_array))[0]:
    key = jax.tree_util.keystr(path)
    if _is_hyperparameter(key):
      if "dropout" not in key:
        msg = f"non-dropout scalar {key} would be excluded as a hyperparameter"
        raise ValueError(msg)
      continue
    shapes[key] = tuple(leaf.shape)
  return shapes


def _is_hyperparameter(key: str) -> bool:
  return any(key.endswith(f".{attr}") for attr in _HYPERPARAMETER_ATTRS)


def check_name_map(
  torch_shapes: dict[str, tuple[int, ...]],
  skeleton_shapes: dict[str, tuple[int, ...]],
  name_map: dict[str, str],
) -> None:
  """Raise unless ``name_map`` is a complete, one-to-one, shape-preserving correspondence."""
  unmapped_torch = sorted(set(torch_shapes) - set(name_map))
  missing_torch = sorted(set(name_map) - set(torch_shapes))
  if unmapped_torch or missing_torch:
    msg = f"name map is not total over the checkpoint: unmapped torch keys {unmapped_torch}, map keys absent from the checkpoint {missing_torch}"
    raise ValueError(msg)

  targets = list(name_map.values())
  duplicated = sorted(path for path, count in Counter(targets).items() if count > 1)
  if duplicated:
    msg = f"aminx leaves targeted by more than one torch key: {duplicated}"
    raise ValueError(msg)

  unhit = sorted(set(skeleton_shapes) - set(targets))
  unknown = sorted(set(targets) - set(skeleton_shapes))
  if unhit or unknown:
    msg = f"skeleton leaves with no torch source {unhit}; map targets absent from the skeleton {unknown}"
    raise ValueError(msg)

  mismatches = [
    f"{torch_key} {tuple(torch_shapes[torch_key])} -> {aminx_path} {skeleton_shapes[aminx_path]}"
    for torch_key, aminx_path in name_map.items()
    if tuple(torch_shapes[torch_key]) != skeleton_shapes[aminx_path]
  ]
  if mismatches:
    msg = "shape mismatches: " + "; ".join(mismatches)
    raise ValueError(msg)


def convert_state(state: dict[str, np.ndarray], model: PottsMPNN) -> PottsMPNN:
  """Write a torch state dict into the aminx skeleton ``model`` after the self-check."""
  torch_shapes = {key: tuple(value.shape) for key, value in state.items()}
  skeleton_shapes = skeleton_parameter_shapes(model)
  name_map = build_name_map()
  check_name_map(torch_shapes, skeleton_shapes, name_map)

  new_values = {name_map[key]: jnp.asarray(value, dtype=jnp.float32) for key, value in state.items()}
  leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(model)
  new_leaves = []
  written = 0
  for path, leaf in leaves_with_path:
    key = jax.tree_util.keystr(path)
    if key in new_values:
      new_leaves.append(new_values[key])
      written += 1
    else:
      new_leaves.append(leaf)
  if written != len(new_values):
    msg = f"wrote {written} leaves but the map has {len(new_values)} targets"
    raise RuntimeError(msg)
  logger.info("wrote %d leaves from %d torch tensors", written, len(state))
  return jax.tree_util.tree_unflatten(treedef, new_leaves)


def load_vanilla_state(path: Path) -> dict[str, np.ndarray]:
  """Read a vanilla checkpoint after refusing it unless its sha256 matches the pin."""
  import torch  # converter-only dependency; the aminx library never imports torch

  verify_pinned_file(path, VANILLA_PIN_SHA256, what="PottsMPNN vanilla checkpoint")
  blob: Any = torch.load(path, map_location="cpu", weights_only=True)
  if isinstance(blob, dict) and "model_state_dict" in blob:
    blob = blob["model_state_dict"]
  return {key: value.detach().cpu().numpy() for key, value in blob.items()}


def main(argv: list[str] | None = None) -> None:
  """Convert the pinned checkpoint, print the coverage summary, and optionally serialise."""
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--out", type=Path, default=None, help="optional .eqx output")
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO)

  state = load_vanilla_state(args.checkpoint)
  model = PottsMPNN(key=jax.random.PRNGKey(0))
  converted = convert_state(state, model)
  print(f"converted {len(state)} torch tensors, 0 unmapped, shapes checked")
  if args.out is not None:
    eqx.tree_serialise_leaves(args.out, converted)
    print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
