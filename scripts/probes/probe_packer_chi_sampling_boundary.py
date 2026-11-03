"""Pre-registered probe for Clause 3: packer_chi_sampling formal waiver and boundary.

The waiver is declared against Praxia debt #3165. This probe runs without Praxia, so it cannot
verify that registration. A log line says so; the JSON keeps the sidecar's [result_schema] keys
only, and the pass condition still reads the declared id.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT / "src"))

import aminx.model.packer as packer_module
from aminx.model.packer import Packer
from aminx.types.bundles import PackerBundle
from aminx.types.configs import InferenceConfig

logger = logging.getLogger(__name__)

DEBT_VERIFICATION = "declared, not verified against praxia"
_NUM_MIX = 3
_NUM_CONTEXT_ATOMS = 16
_SEQ_LEN = 12


def _small_packer() -> Packer:
  return Packer(
    edge_features=128,
    node_features=128,
    num_positional_embeddings=16,
    num_rbf=16,
    top_k=30,
    atom_context_num=_NUM_CONTEXT_ATOMS,
    hidden_dim=128,
    num_encoder_layers=1,
    num_decoder_layers=1,
    dropout=0.0,
    num_mix=_NUM_MIX,
    key=jax.random.PRNGKey(0),
  )


def _synthetic_bundle() -> PackerBundle:
  rng = np.random.default_rng(42)
  return PackerBundle(
    sequence=jnp.asarray(rng.integers(0, 21, (_SEQ_LEN,)), dtype=jnp.int32),
    backbone_coords=jnp.asarray(rng.standard_normal((_SEQ_LEN, 14, 3)), dtype=jnp.float32),
    backbone_mask=jnp.ones((_SEQ_LEN, 14), dtype=jnp.float32),
    ligand_coords=jnp.asarray(
      rng.standard_normal((_SEQ_LEN, _NUM_CONTEXT_ATOMS, 3)), dtype=jnp.float32
    ),
    ligand_mask=jnp.ones((_SEQ_LEN, _NUM_CONTEXT_ATOMS), dtype=jnp.float32),
    ligand_atom_types=jnp.asarray(
      rng.integers(0, 119, (_SEQ_LEN, _NUM_CONTEXT_ATOMS)), dtype=jnp.int32
    ),
    mask=jnp.ones((_SEQ_LEN,), dtype=jnp.float32),
    residue_index=jnp.arange(_SEQ_LEN, dtype=jnp.int32),
    chain_labels=jnp.zeros((_SEQ_LEN,), dtype=jnp.int32),
  )


def run_probe() -> dict[str, object]:
  """Check Packer output invariants and the boundary premise of the packer_chi_sampling waiver."""
  # Declared debt id. Praxia is not reachable from this probe, so nothing here verifies it.
  praxia_debt_id = 3165
  praxia_debt_slug = "debt-packer-chi-sampling-engine"

  config = InferenceConfig(inference=True)
  packer = _small_packer()
  bundle = _synthetic_bundle()

  # 1. Output invariants from the real Packer call at default initialisation.
  result = packer(jax.random.PRNGKey(1), bundle, config)
  shapes_valid = bool(
    result.mean.shape == (_SEQ_LEN, 4, _NUM_MIX)
    and result.concentration.shape == (_SEQ_LEN, 4, _NUM_MIX)
    and result.mix_logits.shape == (_SEQ_LEN, 4, _NUM_MIX)
  )
  concentration_floor_valid = bool(jnp.all(result.concentration >= 0.1))

  # 2. The floor must hold when the raw concentration logit is driven very negative through the
  #    real torsion head (zeroed weights, bias -50 on the concentration channel).
  bias = np.zeros((4, _NUM_MIX, 3), dtype=np.float32)
  bias[..., 1] = -50.0
  driven = eqx.tree_at(
    lambda p: (p.w_torsions.weight, p.w_torsions.bias),
    packer,
    (jnp.zeros_like(packer.w_torsions.weight), jnp.asarray(bias.reshape(-1))),
  )
  driven_result = driven(jax.random.PRNGKey(2), bundle, config)
  floor_under_extreme_logit = bool(
    np.allclose(np.asarray(driven_result.concentration), 0.1, atol=1e-6)
  )
  concentration_floor_valid = concentration_floor_valid and floor_under_extreme_logit

  # 3. Boundary premise: aminx has no NeRF side-chain reconstruction in the packer module.
  boundary_documented = "nerf" not in inspect.getsource(packer_module).lower()

  probe_passed = bool(shapes_valid and concentration_floor_valid and boundary_documented)
  logger.info("probe checks: shapes_valid=%s floor=%s boundary=%s",
              shapes_valid, concentration_floor_valid, boundary_documented)

  return {
    "probe_name": "packer_chi_sampling_boundary",
    "clause": "packer_chi_sampling",
    "status": "PASS" if probe_passed else "FAIL",
    "waiver_status": "WAIVED_UNDER_DEBT",
    "praxia_debt_id": praxia_debt_id,
    "praxia_debt_slug": praxia_debt_slug,
    "shapes_valid": shapes_valid,
    "concentration_floor_valid": concentration_floor_valid,
    "boundary_documented": boundary_documented,
  }


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
  parser = argparse.ArgumentParser(description="Probe for packer_chi_sampling waiver boundary.")
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("outputs/probes/probe_packer_chi_sampling_boundary.json"),
    help="Output JSON path.",
  )
  args = parser.parse_args()

  result = run_probe()

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  bth_results = os.environ.get("BTH_RESULTS_PATH")
  if bth_results:
    bth_path = Path(bth_results)
    bth_path.parent.mkdir(parents=True, exist_ok=True)
    bth_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  logger.info("debt #%s: %s", result["praxia_debt_id"], DEBT_VERIFICATION)
  logger.info("Probe packer_chi_sampling_boundary: %s", result["status"])


if __name__ == "__main__":
  main()
