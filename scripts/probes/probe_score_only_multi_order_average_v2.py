"""Pre-registered probe v2 for Clause 4: score_only_multi_order_average (library-only, debt #3214).

Strengthens v1 (probe_score_only_multi_order_average.py, left unchanged) after the Opus re-audit:

* The causal-mask flip is looped over ALL decoding steps (v1 checked only steps 3 and 5 of 6),
  plus the last step explicitly (it is the final iteration of the loop).
* The sensitivity control (flip the first-decoded token, later positions must move) is kept.
* ``fixed_first_order_valid`` is INFORMATIONAL: it is recorded but excluded from the pass/fail
  predicate. It checks the sampler utility ``random_design_order``, not
  ``score_multi_order_average`` (which draws orders with random_decoding_order and has no
  fixed_mask/chain_mask parameter).

Margin and known limitation: the causal tolerance is 1e-5; the sensitivity measured earlier (v1
run) was 1.43e-4. The ~14x gap between the tolerance and the smallest effect the control can see
means a leak smaller than ~1e-5 would go undetected. That is a known limitation of this probe at
the toy model scale, not a proof of strict causality.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
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
from aminx.utils.decoding_order import random_design_order

logger = logging.getLogger(__name__)


def _score(model: Aminx, key: jax.Array, sequence: jax.Array, num_orders: int, **structure):
  return score_multi_order_average(
    model=model,
    prng_key=key,
    sequence=sequence,
    num_orders=num_orders,
    **structure,
  )


def run_probe() -> dict[str, object]:
  """Check shapes, the causal-mask invariant, a sensitivity control and fixed-first ordering.

  Causal invariant (one order, so ``mean_probs`` is that order's per-position distribution):
  changing the identity of the token decoded at step ``t`` must not change the distribution at
  any position decoded at or before step ``t``. A mask that leaked future tokens fails it.
  Sensitivity control: changing the FIRST-decoded token must change some other position, so the
  invariant is not satisfied by a model that ignores the sequence. Neither check depends on an
  upstream dump; none was available, so the reference's chain-mask order
  (LigandMPNN model_utils.py:581-584) is checked as a property, not as a bit-for-bit match.
  """
  # Declared debt id. Praxia is not reachable from this probe, so nothing here verifies it.
  praxia_debt_id = 3214
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
  structure = {
    "structure_coordinates": coords,
    "mask": jnp.ones(seq_len),
    "residue_index": jnp.arange(seq_len),
    "chain_index": jnp.zeros(seq_len, dtype=jnp.int32),
  }
  sequence = jnp.arange(seq_len) % 20

  result = _score(model, key, sequence, num_orders, **structure)

  shapes_valid = bool(
    result.mean_probs.shape == (seq_len, 21)
    and result.std_probs.shape == (seq_len, 21)
    and result.sample_nlls.shape == (num_orders,)
    and result.decoding_orders.shape == (num_orders, seq_len)
  )

  # --- causal-mask invariant, single order ---
  base = _score(model, key, sequence, 1, **structure)
  order = [int(i) for i in base.decoding_orders[0]]

  def _flip(seq: jax.Array, pos: int) -> jax.Array:
    return seq.at[pos].set((seq[pos] + 7) % 20)

  causal_max_diff = 0.0
  for step in range(seq_len):
    flipped = _score(model, key, _flip(sequence, order[step]), 1, **structure)
    upto = jnp.asarray(order[: step + 1])
    diff = jnp.max(jnp.abs(flipped.mean_probs[upto] - base.mean_probs[upto]))
    causal_max_diff = max(causal_max_diff, float(diff))
  causal_invariant_holds = bool(causal_max_diff < 1e-5)

  # --- sensitivity control: the first-decoded token is visible to every later position ---
  first = order[0]
  flipped_first = _score(model, key, _flip(sequence, first), 1, **structure)
  later = jnp.asarray(order[1:])
  sensitivity_diff = float(
    jnp.max(jnp.abs(flipped_first.mean_probs[later] - base.mean_probs[later]))
  )
  sensitivity_control_passed = bool(sensitivity_diff > 1e-6)

  # --- INFORMATIONAL: fixed positions decode first (reference argsort((chain_mask + 1e-4) * |randn|)).
  # Checks the sampler utility random_design_order, not score_multi_order_average. ---
  fixed_positions = {1, 4}
  fixed_mask = jnp.zeros(seq_len).at[jnp.asarray(sorted(fixed_positions))].set(1.0)
  untied = jnp.arange(seq_len, dtype=jnp.int32)
  fixed_first_order_valid = True
  for seed in range(8):
    design_order = [
      int(i) for i in random_design_order(jax.random.PRNGKey(seed), untied, fixed_mask)
    ]
    fixed_first_order_valid = bool(
      fixed_first_order_valid
      and set(design_order[: len(fixed_positions)]) == fixed_positions
      and sorted(design_order) == list(range(seq_len))
    )

  # Informational, not gating: score_multi_order_average draws orders with random_decoding_order
  # and exposes no fixed_mask/chain_mask, so it cannot reproduce the reference's chain-mask-aware
  # order. This records that gap; it is the reason the debt (#3214) stays open.
  multi_order_has_fixed_param = bool(
    {"fixed_mask", "chain_mask"} & set(inspect.signature(score_multi_order_average).parameters)
  )

  # fixed_first_order_valid is informational (see module docstring): it is NOT in this predicate.
  probe_passed = bool(shapes_valid and causal_invariant_holds and sensitivity_control_passed)

  return {
    "probe_name": "score_only_multi_order_average_v2",
    "clause": "score_only_multi_order_average",
    "status": "PASS" if probe_passed else "FAIL",
    "praxia_debt_id": praxia_debt_id,
    "praxia_debt_slug": praxia_debt_slug,
    "num_orders": num_orders,
    "ensemble_nll": float(result.ensemble_nll),
    "mean_sample_nll": float(result.mean_sample_nll),
    "shapes_valid": shapes_valid,
    "causal_invariant_holds": causal_invariant_holds,
    "causal_max_diff": causal_max_diff,
    "sensitivity_control_passed": sensitivity_control_passed,
    "sensitivity_diff": sensitivity_diff,
    "fixed_first_order_valid": fixed_first_order_valid,
    "multi_order_has_fixed_param": multi_order_has_fixed_param,
  }


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
  parser = argparse.ArgumentParser(description="Probe for score_only_multi_order_average clause.")
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("outputs/probes/probe_score_only_multi_order_average_v2.json"),
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

  logger.info("debt #%s: declared, not verified against praxia", result["praxia_debt_id"])
  if not result["multi_order_has_fixed_param"]:
    logger.warning("score_multi_order_average has no fixed_mask/chain_mask parameter (debt #3214).")
  logger.info("Probe score_only_multi_order_average_v2: %s", result["status"])


if __name__ == "__main__":
  main()
