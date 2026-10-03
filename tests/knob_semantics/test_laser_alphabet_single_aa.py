# ruff: noqa: S101
"""Spec §5.1a: one amino acid's column lands on that same amino acid's column.

``test_alphabet_boundary_is_a_bijection`` (``test_driver.py:124``) proves the
two INDEX TABLES are built correctly -- ``LASER_ALPHABET[laser] ==
ALPHABET[canonical]`` for every slot. It does not prove that
``canonical_logits`` applies them the right way round, and that is a different
claim: ``jnp.take(x, LASER_OF_CANONICAL, -1)`` and ``jnp.take(x,
CANONICAL_OF_LASER, -1)`` are BOTH bijections built from BOTH correct tables, so
a bijection test passes under either. Inverting the direction would relabel
every residue as a different amino acid and keep every existing assertion green.

MEASURED, because the obvious guesses about this permutation are wrong:

  ALPHABET      ACDEFGHIKLMNPQRSTVWYX   (ProteinMPNN order, NOT AlphaFold)
  LASER_ALPHABET ARNDCEQGHILKMFPSTWYVX

17 of the 21 letters sit at different indices, and the permutation is NOT an
involution -- the forward and inverse tables genuinely differ::

  fwd [0, 4, 3, 5, 13, 7, 8, 9, 11, 10, 12, 2, 14, 6, 1, 15, 16, 19, 17, 18, 20]
  inv [0, 14, 11, 2, 1, 3, 13, 5, 6, 7, 9, 8, 10, 4, 12, 15, 16, 18, 19, 17, 20]

That matters: had it been a transposition it would be self-inverse and NO test
could tell the directions apart. It is not, so this one can.

ON THE SPEC NAME. §5.1a lists the inbound conversions as "``sequences_to_score``,
``fixed_tokens``, ``bias``/``omit_aa*`` columns, ``disabled_residues``,
budget/charged letters". The ``bias``/``omit_aa*`` half has no implementation to
test, because ``LaserDriver._HANDLED`` (``driver.py:43-50``) is four SCORE
purposes -- ``score:nll``, ``score:logits`` and the two proofread purposes -- and
carries no sampling surface at all, so no inbound bias ever reaches a LASEr
decode. The single-amino-acid property is therefore pinned at the boundary that
DOES exist, which is the one those columns would have to pass through.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from aminx.families.laser_mpnn.driver import (
  ALPHABET,
  CANONICAL_OF_LASER,
  LASER_OF_CANONICAL,
  canonical_logits,
)
from aminx.families.laser_mpnn.featurize import LASER_ALPHABET

_MARK = 7.5


def test_knob_semantics_laser_bias_single_aa() -> None:
  """For every letter, a value on its LASEr column arrives on its canonical column."""
  moved: list[str] = []
  for letter in ALPHABET:
    laser_row = np.zeros((21,), dtype=np.float32)
    laser_row[LASER_ALPHABET.index(letter)] = _MARK

    out = np.asarray(canonical_logits(jnp.asarray(laser_row)))

    landed = int(np.argmax(out))
    assert landed == ALPHABET.index(letter), (
      f"{letter!r} was placed on LASEr column {LASER_ALPHABET.index(letter)} and "
      f"arrived on canonical column {landed}, not {ALPHABET.index(letter)}"
    )
    assert out[landed] == _MARK
    # Nothing else moved: exactly one non-zero, so the take is a permutation
    # and not a broadcast or a partial gather.
    assert int(np.count_nonzero(out)) == 1, out
    if LASER_ALPHABET.index(letter) != ALPHABET.index(letter):
      moved.append(letter)

  # 17 letters actually change index. If a future alphabet edit made the two
  # orders agree, every assertion above would hold trivially -- so the fixture
  # states how much work it is doing.
  assert len(moved) == 17, moved


def test_laser_alphabet_direction_control_the_inverse_table_fails() -> None:
  """CONTROL: taking with the INVERSE table gives a different, wrong answer.

  This is what the bijection test cannot see. Both tables are valid
  permutations derived from a correct letter mapping; only one of them is the
  right direction for ``canonical_logits``. Without this control, inverting
  ``:109`` would break nothing in the suite.
  """
  disagreements = 0
  for letter in ALPHABET:
    laser_row = np.zeros((21,), dtype=np.float32)
    laser_row[LASER_ALPHABET.index(letter)] = _MARK

    right = np.asarray(canonical_logits(jnp.asarray(laser_row)))
    wrong = np.asarray(jnp.take(jnp.asarray(laser_row), CANONICAL_OF_LASER, axis=-1))
    if int(np.argmax(right)) != int(np.argmax(wrong)):
      disagreements += 1

  # Not all 21 -- the four fixed points of the permutation land identically
  # either way, which is precisely why a spot check on one letter would be a
  # coin flip and this sweeps the whole alphabet.
  assert disagreements == 17, (
    f"the inverse table must disagree on the 17 moved letters, got {disagreements}"
  )


def test_laser_alphabet_tables_are_not_self_inverse() -> None:
  """The property that makes the direction testable at all.

  A self-inverse (involution) permutation has identical forward and inverse
  tables, and no test anywhere could distinguish the two directions. Recorded
  as an assertion so that if the alphabets are ever edited into an involution,
  this fails loudly rather than letting the control above quietly become
  vacuous.
  """
  forward = np.asarray(LASER_OF_CANONICAL)
  inverse = np.asarray(CANONICAL_OF_LASER)
  assert not np.array_equal(forward, inverse)
  # And they really are inverses of each other.
  np.testing.assert_array_equal(forward[inverse], np.arange(21))
  np.testing.assert_array_equal(inverse[forward], np.arange(21))
