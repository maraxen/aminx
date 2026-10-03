# ruff: noqa: S101
"""Spec §5.4b: the score decoding order is three tiers, and the seed only shuffles within one.

``decoding_order`` (``driver.py:131-153``) is ``argsort(uniforms + tier)`` with
tier 0 for fixed residues, 1 for designable non-contact, and 2 for designable
contact. It had NO test. The name ``decoding_order`` appears in
``tests/sampling/`` several times, but every one of those is
``random_decoding_order`` / ``ar_mask_from_decoding_order`` in
``aminx.utils.autoregression`` -- a different function in a different module.

THE TIER SEPARATION IS STRUCTURAL, NOT STATISTICAL, and that is what makes it
worth pinning. ``jax.random.uniform`` draws from ``[0, 1)``, so ``uniforms +
tier`` can never carry a tier-1 row past a tier-2 row: the largest possible
tier-1 key is strictly below the smallest possible tier-2 key. Were the draw
ever changed to a closed interval, or the tier offsets dropped (one of the
spec's named mutants), the tiers would interleave for some seeds and not
others -- a bug that appears only on certain random_seed values and would be
near-impossible to reproduce from a report.

So the tier assertion is swept over many seeds rather than checked on one.

The "injected oracle order matches" half of this spec row needs an upstream
fixture and belongs in ``tests/port/``, which is a scoped path; this is the
analytic half.
"""

from __future__ import annotations

import numpy as np

from aminx.families.laser_mpnn.driver import decoding_order

# Rows 0 and 3 fixed; 2 and 5 are designable ligand contacts; 1 and 4 are
# designable and untouched. Interleaved deliberately, so a tier grouping that
# accidentally followed index order would not pass.
_FIXED = np.asarray([True, False, False, True, False, False])
_CONTACT = np.asarray([False, False, True, False, False, True])

_TIER0 = {0, 3}
_TIER1 = {1, 4}
_TIER2 = {2, 5}

_SEEDS = range(40)


def test_knob_semantics_laser_score_order() -> None:
  """Tiers never interleave, over 40 seeds."""
  for seed in _SEEDS:
    order = decoding_order(_FIXED, _CONTACT, seed=seed)
    assert order.shape == (6,)
    assert set(order.tolist()) == set(range(6)), order

    first, middle, last = set(order[:2]), set(order[2:4]), set(order[4:])
    assert first == _TIER0, f"seed {seed}: fixed rows must lead, got {order}"
    assert middle == _TIER1, f"seed {seed}: non-contact next, got {order}"
    assert last == _TIER2, f"seed {seed}: contacts last, got {order}"


def test_laser_score_order_seeds_differ() -> None:
  """The seed shuffles WITHIN a tier, and different seeds give different orders.

  Without this, the tier test above is satisfied by a deterministic sort that
  ignores the seed entirely -- which would make every score reproducible and
  wrong in the same way, and would silently drop the per-seed variation the
  proofread ensemble depends on.
  """
  seen = {decoding_order(_FIXED, _CONTACT, seed=seed).tobytes() for seed in _SEEDS}
  assert len(seen) > 1, "every seed produced the same order"

  # Specifically: the within-tier arrangement is what varies. Both tier-0
  # orderings must appear across these seeds, or the seed is not reaching the
  # fixed rows.
  tier0_orders = {tuple(decoding_order(_FIXED, _CONTACT, seed=seed)[:2]) for seed in _SEEDS}
  assert tier0_orders == {(0, 3), (3, 0)}, tier0_orders


def test_laser_score_order_is_deterministic_for_one_seed() -> None:
  """Same seed and same masks give the same permutation, every time.

  This is the property the docstring claims a parent process and a fresh
  subprocess rely on, and it is cheap to pin.
  """
  for seed in (0, 7, 31):
    first = decoding_order(_FIXED, _CONTACT, seed=seed)
    second = decoding_order(_FIXED, _CONTACT, seed=seed)
    np.testing.assert_array_equal(first, second)


def test_laser_score_order_tier2_is_empty_at_inference() -> None:
  """With no extra-atom contacts, every designable row is tier 1.

  ``extra_atom_contact_mask`` is zeros at inference (``driver.py:139``), so the
  production path only ever sees two tiers. Pinned because the three-tier
  fixture above is deliberately NOT the inference case, and a reader should not
  conclude that tier 2 is routinely populated.
  """
  no_contact = np.zeros(6, dtype=bool)
  for seed in (0, 5, 11):
    order = decoding_order(_FIXED, no_contact, seed=seed)
    assert set(order[:2]) == _TIER0, order
    # The remaining four are one tier, so they may appear in any arrangement.
    assert set(order[2:]) == _TIER1 | _TIER2, order
