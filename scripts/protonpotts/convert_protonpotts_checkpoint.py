"""Convert the ProtonPottsMPNN v6 checkpoint into an aminx ``PottsMPNN`` (V = 30) Equinox artifact.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §41. Two steps, because only the oracle
environment has torch and only an aminx environment has JAX::

    # oracle env: read the torch checkpoint, write plain numpy
    python scripts/protonpotts/convert_protonpotts_checkpoint.py \\
        --checkpoint <epoch-0125.ckpt> --export-state-npz <state.npz>
    # aminx env: renames + permutations + shape checks + conversion
    uv run --no-sync python scripts/protonpotts/convert_protonpotts_checkpoint.py \\
        --state-npz <state.npz> --out <model.eqx>

The permutations are in ``aminx.families.protonpotts_mpnn.convert`` and are unit tested there; whether
they are the RIGHT ones is settled only by the ``protonpotts_encoder`` comparison against upstream.

Lives in ``scripts/protonpotts/`` (unscoped) on purpose. ``scripts/recapture/pottsmpnn_model_to_eqx.py`` is
a globally scoped path that the shipped Potts driver also loads at runtime, so extending it would
invalidate every ledger row. This script reuses ``scripts/convert_weights.py`` (also unscoped).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("convert_protonpotts_checkpoint")

UPSTREAM_COMMIT = "09682abf"  # ProtonPottsMPNN, the oracle pin (spec §17)


def _read_checkpoint_state(path: Path) -> dict[str, np.ndarray]:
  import torch  # oracle env only

  blob = torch.load(path, map_location="cpu", weights_only=False)
  state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
  return {key: value.detach().cpu().numpy() for key, value in state.items()}


def _convert_weights():  # noqa: ANN202
  path = Path(__file__).resolve().parents[1] / "convert_weights.py"
  spec = importlib.util.spec_from_file_location("aminx_convert_weights", path)
  if spec is None or spec.loader is None:
    msg = f"cannot load {path}"
    raise ImportError(msg)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def convert_state(state: dict[str, np.ndarray]):  # noqa: ANN201
  """A foundry-layout numpy state dict -> an aminx ``PottsMPNN`` over the v6 alphabet."""
  import equinox as eqx
  import jax

  from aminx.families.protonpotts_mpnn.convert import to_aminx_layout
  from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6
  from aminx.families.potts_mpnn.model import PottsMPNN

  arrays = to_aminx_layout(state)
  weights = _convert_weights()
  model = PottsMPNN(key=jax.random.PRNGKey(0), alphabet=PROTONPOTTS_V6)
  mpnn = weights.convert_full_model(arrays, model.mpnn)
  model = eqx.tree_at(lambda m: m.mpnn, model, mpnn)
  head = weights.convert_linear_layer(
    arrays["etab_out.weight"], arrays["etab_out.bias"], model.potts_head.linear
  )
  return eqx.tree_at(lambda m: m.potts_head.linear, model, head)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--checkpoint", type=Path)
  source.add_argument("--state-npz", type=Path)
  parser.add_argument("--export-state-npz", type=Path, help="write the numpy state and stop")
  parser.add_argument("--out", type=Path)
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s")

  if args.checkpoint is not None:
    state = _read_checkpoint_state(args.checkpoint)
    source_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
  else:
    with np.load(args.state_npz) as loaded:
      state = {key: loaded[key] for key in loaded.files}
    source_sha = hashlib.sha256(args.state_npz.read_bytes()).hexdigest()
  logger.info("state dict: %d tensors", len(state))

  if args.export_state_npz is not None:
    args.export_state_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.export_state_npz, **state)
    args.export_state_npz.with_suffix(".json").write_text(
      json.dumps({"checkpoint_sha256": source_sha, "upstream_commit": UPSTREAM_COMMIT}, indent=2) + "\n",
      encoding="utf-8",
    )
    logger.info("wrote %s", args.export_state_npz)
    return 0
  if args.out is None:
    parser.error("--out is required unless --export-state-npz is given")

  import equinox as eqx

  model = convert_state(state)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  eqx.tree_serialise_leaves(args.out, model)
  manifest = {
    "source_sha256": source_sha,
    "artifact_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
    "upstream_commit": UPSTREAM_COMMIT,
    "alphabet": "protonpottsmpnn_v6_30",
  }
  args.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
  logger.info("wrote %s %s", args.out, manifest)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
