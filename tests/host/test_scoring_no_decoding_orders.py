"""``runner.score`` must not publish a decoding order it never used (debt #1717).

Every score() path is full-context teacher forcing: the mask is ``1 - I`` and there are no
waves, so no decoding order participates in the computation. The runner nevertheless drew a
random permutation per candidate and returned it as ``results["decoding_orders"]``, where a
consumer would reasonably read it as describing the scoring run. It is now ``None``.

Marked slow: it loads real weights and runs the full pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

STRUCTURE = Path(__file__).resolve().parents[1] / "data" / "1ubq.pdb"
UBIQUITIN_SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


@pytest.mark.slow
@pytest.mark.requires_weights
@pytest.mark.parametrize("average_node_features", [False, True])
def test_score_reports_no_decoding_order(average_node_features: bool) -> None:  # noqa: FBT001
  from aminx.host import runner  # noqa: PLC0415
  from aminx.run.specs import ScoringSpecification  # noqa: PLC0415

  if not STRUCTURE.is_file():
    pytest.skip(f"structure fixture missing: {STRUCTURE}")

  result = runner.score(
    ScoringSpecification(
      inputs=str(STRUCTURE),
      chain_id="A",
      checkpoint_id="proteinmpnn_v_48_020",
      sequences_to_score=[UBIQUITIN_SEQUENCE],
      backbone_noise=0.0,
      average_node_features=average_node_features,
      return_decoding_orders=True,
    ),
  )

  assert "decoding_orders" in result, "requested key must be present (as an explicit None)"
  assert result["decoding_orders"] is None
  assert result["scores"].shape == (1, 1)
