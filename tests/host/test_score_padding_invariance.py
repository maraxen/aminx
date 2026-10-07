"""Padding the structure must not change the score.

``ScoringSpecification.max_length`` defaults to 512, so a 76- or 214-residue chain is
routinely padded well beyond its real length. ``host/runner.py`` states the intended
invariant -- padded positions are masked out and contribute nothing -- but nothing asserted
it, and a measurement on 1LVB chain A suggested it was violated: mean scores of
1.588822 / 1.572196 / 1.562965 / 1.562179 at ``max_length`` 214 / 256 / 384 / 512, monotone,
spread 0.0266 nats.

**That measurement could not settle the question, and this test can.** It was taken while the
score path derived its causal mask from a freshly drawn decoding order, so changing
``max_length`` changed the permutation as well as the padding -- two moving parts, one
number. With the default mask now order-free (``full_context_ar_mask``), padding is the only
thing that varies, and the invariant is well posed.

The answer is that padding contributes **exactly nothing**: bit-identical scores across
76 / 96 / 128 / 192. So the earlier 0.0266 spread was the decoding order, in full. Note the
reasoning that made it look otherwise -- "monotone in pad length, therefore not order noise"
-- was itself wrong: a uniform permutation over 512 positions has different statistics from
one over 214, so an order effect *can* trend monotonically with length.

This gates real work. Runner length bucketing (``aminx.host.bucketing``, spec S8) *changes
the pad length* of every scored structure, so it is only safe if this test passes. If it fails,
the failure is a genuine masking leak and must be fixed before bucketing lands -- not
absorbed as noise.

Marked slow: loads real weights and runs the pipeline once per pad length.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

STRUCTURE = Path(__file__).resolve().parents[1] / "data" / "1ubq.pdb"

# 1UBQ chain A, 76 residues.
UBIQUITIN_SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
PAD_LENGTHS = (76, 96, 128, 192)

# Measured spread across the four pad lengths above on CPU: EXACTLY 0.0 -- bit-identical.
# On a CUDA backend (titanix, 260826_chain-selection-vendor-superset-audit finding FT1), the
# measured spread was 2.384e-07 nats -- real, reproducible, and consistent with float32
# non-associative summation reordering from the different padded array shapes (exactly the
# "a different XLA backend could reorder the masked sum enough to move the last ulp" case this
# comment already called out before it was ever observed). The bound below is set a further
# ~2 orders of magnitude above that measurement -- generous enough to absorb legitimate
# cross-backend float32 reduction-order noise, while remaining ~3 orders of magnitude below
# the 0.0160-nat effect size that gets claimed downstream. A real masking leak would clear
# this bound by a wide margin, not sit at its edge.
PADDING_TOLERANCE_NATS = 1e-5


@pytest.mark.slow
@pytest.mark.requires_weights
def test_score_does_not_depend_on_max_length() -> None:
  """Same structure, same sequence, same key -- only the pad length differs."""
  pytest.importorskip("aminx")
  # Local: must follow importorskip so collection stays cheap when weights are absent.
  from aminx.host import runner  # noqa: PLC0415
  from aminx.run.specs import ScoringSpecification  # noqa: PLC0415

  if not STRUCTURE.is_file():
    pytest.skip(f"structure fixture missing: {STRUCTURE}")

  scores: dict[int, float] = {}
  for max_length in PAD_LENGTHS:
    result = runner.score(
      ScoringSpecification(
        inputs=str(STRUCTURE),
        chain_id="A",
        checkpoint_id="proteinmpnn_v_48_020",
        sequences_to_score=[UBIQUITIN_SEQUENCE],
        backbone_noise=0.0,
        random_seed=42,
        max_length=max_length,
        return_logits=False,
      ),
    )
    scores[max_length] = float(np.asarray(result["scores"], dtype=np.float64).ravel()[0])

  spread = max(scores.values()) - min(scores.values())
  assert spread <= PADDING_TOLERANCE_NATS, (
    f"score varies with max_length by {spread:.3e} nats across {PAD_LENGTHS}: {scores}. "
    "With an order-free mask this cannot be decoding-order noise — it is a masking leak, "
    "and the length-bucketing work is blocked until it is fixed, since bucketing changes "
    "the pad length of every scored structure."
  )
