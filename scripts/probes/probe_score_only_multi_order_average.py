"""Pre-registered probe for Clause 4: score_only_multi_order_average."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT / "src"))

from aminx.model.mpnn import Aminx
from aminx.scoring.multi_order import score_multi_order_average


def run_probe() -> dict[str, object]:
  """Execute probe verifying multi-order score averaging, probability normalization, and Jensen's inequality."""
  praxia_debt_id = 3166
  praxia_debt_slug = "debt-multi-order-scoring-average"

  seq_len = 6
  num_orders = 4
  key = jax.random.PRNGKey(999)

  model = Aminx(
    node_features=64,
    edge_features=64,
    hidden_features=64,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=6,
    key=jax.random.PRNGKey(123),
  )

  coords = jnp.zeros((seq_len, 4, 3))
  coords = coords.at[:, 1, 0].set(jnp.arange(seq_len) * 3.8)
  mask = jnp.ones(seq_len)
  residue_index = jnp.arange(seq_len)
  chain_index = jnp.zeros(seq_len, dtype=jnp.int32)
  sequence = jnp.arange(seq_len) % 20

  result = score_multi_order_average(
    model=model,
    prng_key=key,
    sequence=sequence,
    structure_coordinates=coords,
    mask=mask,
    residue_index=residue_index,
    chain_index=chain_index,
    num_orders=num_orders,
  )

  shapes_valid = bool(
    result.mean_probs.shape == (seq_len, 21)
    and result.std_probs.shape == (seq_len, 21)
    and result.sample_nlls.shape == (num_orders,)
    and result.decoding_orders.shape == (num_orders, seq_len)
  )

  probs_normalized = bool(
    jnp.allclose(jnp.sum(result.mean_probs, axis=-1), jnp.ones(seq_len), atol=1e-5)
  )

  jensen_satisfied = bool(
    float(result.ensemble_nll) <= float(result.mean_sample_nll) + 1e-6
  )

  probe_passed = bool(shapes_valid and probs_normalized and jensen_satisfied)

  return {
    "probe_name": "score_only_multi_order_average",
    "clause": "score_only_multi_order_average",
    "status": "PASS" if probe_passed else "FAIL",
    "praxia_debt_id": praxia_debt_id,
    "praxia_debt_slug": praxia_debt_slug,
    "num_orders": num_orders,
    "ensemble_nll": float(result.ensemble_nll),
    "mean_sample_nll": float(result.mean_sample_nll),
    "shapes_valid": shapes_valid,
    "probs_normalized": probs_normalized,
    "jensen_satisfied": jensen_satisfied,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Probe for score_only_multi_order_average clause.")
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("outputs/probes/probe_score_only_multi_order_average.json"),
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

  print(f"Probe score_only_multi_order_average: {result['status']}")


if __name__ == "__main__":
  main()
