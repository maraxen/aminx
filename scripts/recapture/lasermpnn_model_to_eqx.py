"""Map every LASErMPNN checkpoint key onto the Equinox module tree B2 will build.

B2 has not landed, so this script does not construct a model. It records, for each
upstream ``state_dict`` key, the shape, dtype, parameter-or-buffer kind, and the
planned destination path from spec §5.2:

- ``ligand_featurizer`` / ``ligand_encoder`` / ``ligand_encoder_output_gvp`` → ``LaserLigandEncoder``
- ``protein_encoder_layers`` and the edge/RBF/frame layers that feed them → ``LaserEncoder``
- ``protein_decoder_layers`` and the sequence embed/output → ``LaserDecoderLayer``
- ``chi_prediction_layers`` / ``chi_vector_update_layers`` → ``LaserChiHead``
- ``chi_offset_prediction_layers`` → ``LaserChiOffsetHead``
- ``rotamer_builder`` / ``hbond_network_detector`` → ``aminx.model.laser.rotamers``

``ligand_encoder_output_gvp.dropout.vdropout.dummy_param`` is mapped (it is an empty
``Parameter`` in ``_VDropout``), not dropped. The ligand-encoder pretraining file
``pretrained_ligand_encoder_weights.pt`` is not a ``LASErMPNN`` checkpoint (T0.2) and
is not part of this map.

Write the committed inventory with::

    uv run --frozen python scripts/recapture/lasermpnn_model_to_eqx.py \\
        --checkpoint /path/to/laser_weights_0p1A_nothing_heldout.pt \\
        --inventory tests/port/reference/laser_weights/key_inventory.toml
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

UPSTREAM_COMMIT = "e70f2c6d765416f7e29d51bfd6d4e08496438878"

# Three checkpoints that ``load_model_from_parameter_dict`` builds as LASErMPNN.
# Paths are relative to the pinned LASErMPNN checkout (VENDOR_PIN.toml).
LASER_CHECKPOINTS: tuple[tuple[str, str], ...] = (
  ("nothing_heldout", "model_weights/laser_weights_0p1A_nothing_heldout.pt"),
  ("noise_ligandmpnn_split", "model_weights/laser_weights_0p1A_noise_ligandmpnn_split.pt"),
  (
    "soluble_65000",
    "model_weights/soluble_weights_no_heldout_drop_clusters_optstep_65000.pt",
  ),
)

# (upstream prefix, destination prefix). Longest-prefix match is by list order
# (no prefix here is a prefix of another).
_DESTINATION_ROOTS: tuple[tuple[str, str], ...] = (
  ("hbond_network_detector.", "rotamers.hbond_network_detector."),
  ("rotamer_builder.", "rotamers.builder."),
  ("ligand_featurizer.", "laser_ligand_encoder.featurizer."),
  ("ligand_encoder_output_gvp.", "laser_ligand_encoder.output_gvp."),
  ("ligand_encoder.", "laser_ligand_encoder.encoder."),
  ("protein_encoder_layers.", "laser_encoder.layers."),
  ("protein_decoder_layers.", "laser_decoder.layers."),
  ("chi_prediction_layers.", "laser_chi_head.prediction."),
  ("chi_offset_prediction_layers.", "laser_chi_offset_head.layers."),
  ("chi_vector_update_layers.", "laser_chi_head.vector_update."),
  ("prot_prot_rbf_encoding.", "laser_encoder.prot_prot_rbf."),
  ("lig_prot_rbf_encoding.", "laser_encoder.lig_prot_rbf."),
  ("prot_prot_edge_input_layer.", "laser_encoder.prot_prot_edge_input."),
  ("lig_prot_edge_input_layer.", "laser_encoder.lig_prot_edge_input."),
  ("backbone_frame_vec_input_layer.", "laser_encoder.backbone_frame."),
  ("sequence_label_embedding.", "laser_decoder.sequence_label_embedding."),
  ("sequence_output_layer.", "laser_decoder.sequence_output."),
)

_BUFFER_ROOTS: tuple[str, ...] = (
  "hbond_network_detector.",
  "rotamer_builder.",
  "ligand_featurizer.",
)


def destination_for(key: str) -> str:
  """Planned Equinox path for one upstream state-dict key. Raises if unmapped."""
  for prefix, dest in _DESTINATION_ROOTS:
    if key.startswith(prefix):
      return dest + key[len(prefix) :]
  msg = f"unmapped LASEr state_dict key {key}"
  raise KeyError(msg)


def kind_for(key: str) -> str:
  """``buffer`` for persisted buffers, ``parameter`` otherwise (including ``dummy_param``)."""
  if key.endswith("D_mu"):
    return "buffer"
  if key.startswith(_BUFFER_ROOTS):
    return "buffer"
  return "parameter"


def _dtype_name(value: object) -> str:
  dtype = getattr(value, "dtype", None)
  text = str(dtype)
  if text.startswith("torch."):
    return text.removeprefix("torch.")
  return text


def _shape(value: object) -> list[int]:
  shape = getattr(value, "shape", None)
  if shape is None:
    msg = "state_dict value has no shape"
    raise TypeError(msg)
  return [int(dim) for dim in shape]


def inventory_rows(state: dict[str, Any]) -> list[dict[str, object]]:
  """One row per key. Every key has a destination (0 unmapped)."""
  rows: list[dict[str, object]] = []
  for key in sorted(state):
    value = state[key]
    rows.append(
      {
        "upstream": key,
        "shape": _shape(value),
        "dtype": _dtype_name(value),
        "kind": kind_for(key),
        "destination": destination_for(key),
      },
    )
  return rows


def unmapped_keys(state: dict[str, Any]) -> list[str]:
  """Keys for which ``destination_for`` raises."""
  missing: list[str] = []
  for key in state:
    try:
      destination_for(str(key))
    except KeyError:
      missing.append(str(key))
  return missing


def render_inventory(rows: list[dict[str, object]]) -> str:
  """TOML document. ``shape`` is a decimal list; strings are quoted."""
  lines = [
    f'upstream_commit = "{UPSTREAM_COMMIT}"',
    f"n_keys = {len(rows)}",
    'source_checkpoint = "model_weights/laser_weights_0p1A_nothing_heldout.pt"',
    "",
  ]
  for row in rows:
    shape = row["shape"]
    if not isinstance(shape, list):
      msg = "inventory shape must be a list"
      raise TypeError(msg)
    shape_text = "[" + ", ".join(str(int(dim)) for dim in shape) + "]"
    lines.extend(
      [
        "[[key]]",
        f'upstream = "{row["upstream"]}"',
        f"shape = {shape_text}",
        f'dtype = "{row["dtype"]}"',
        f'kind = "{row["kind"]}"',
        f'destination = "{row["destination"]}"',
        "",
      ],
    )
  return "\n".join(lines)


def load_inventory(path: Path) -> dict[str, dict[str, object]]:
  """Parse ``key_inventory.toml`` into ``upstream -> row``."""
  document = tomllib.loads(path.read_text())
  rows = document.get("key", [])
  if not isinstance(rows, list) or not rows:
    msg = f"{path} has no [[key]] rows"
    raise ValueError(msg)
  parsed: dict[str, dict[str, object]] = {}
  for row in rows:
    upstream = str(row["upstream"])
    if upstream in parsed:
      msg = f"duplicate inventory key {upstream}"
      raise ValueError(msg)
    parsed[upstream] = row
  return parsed


def load_state_dict(path: Path) -> dict[str, Any]:
  """Load ``model_state_dict`` from a LASEr checkpoint. Imports torch lazily."""
  try:
    import torch  # noqa: PLC0415
  except ImportError as exc:
    msg = "torch is required to read a LASErMPNN checkpoint"
    raise ImportError(msg) from exc
  blob = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(blob, dict) or "model_state_dict" not in blob:
    msg = f"{path} has no model_state_dict"
    raise TypeError(msg)
  state = blob["model_state_dict"]
  if not isinstance(state, dict):
    msg = f"{path} model_state_dict is not a dict"
    raise TypeError(msg)
  return state


def write_inventory(state: dict[str, Any], path: Path) -> int:
  """Write the inventory. Returns the number of keys."""
  rows = inventory_rows(state)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(render_inventory(rows), encoding="utf-8")
  return len(rows)


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--inventory", type=Path, required=True)
  args = parser.parse_args(argv)
  state = load_state_dict(args.checkpoint)
  missing = unmapped_keys(state)
  if missing:
    msg = f"{len(missing)} unmapped keys, first {missing[:5]}"
    raise SystemExit(msg)
  n_keys = write_inventory(state, args.inventory)
  print(f"wrote {n_keys} keys to {args.inventory}")


if __name__ == "__main__":
  main()
