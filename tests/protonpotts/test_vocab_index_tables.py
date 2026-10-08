"""Pin the two ProtonPottsMPNN extended vocabularies by INDEX, so an upstream reorder is loud.

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` §25.3.

THE DEFECT THIS GUARDS. Upstream ships two extended vocabularies and they are both live:

  v3/v4 -> 32 tokens = 21 standard + PROTONATION_TOKENS (11)   <- the INFERENCE DEFAULT
  v6    -> 30 tokens = 21 standard + V6_PROTONATION_TOKENS (9)

They share the same 21-token prefix, so with no protonation labels present the two produce a
bit-identical `S`. That shared prefix is exactly what makes a mismatch silent. Above index 20
they agree at NO index. Read index 27 across: v4 says `ASP-D`, v6 says `GLU-P`. Cross the two
and an aspartate silently becomes a protonated glutamate -- different residue, different
charge, shape-valid, dtype-valid, in range, and nothing raises.

WHAT THIS FILE DOES AND DOES NOT COVER.

  covered      the index -> token mapping of both extension regions, pinned as literals
  covered      the premise that the two extensions collide (runs with no upstream installed)
  covered      conformance of those literals to upstream, WHEN upstream is importable
  NOT covered  the 21-token prefix's letter content. Upstream's prefix is three-letter residue
               names (`ALA`, `ARG`, ...) while aminx's `POTTS_ALPHABET` is one-letter codes, so
               asserting equality between them needs a 1<->3 letter map this file deliberately
               does not assume. `tests/test_alphabet_conformance.py` pins aminx's 21-token side.
  NOT covered  anything about aminx's own ProtonPotts alphabet, which does not exist yet --
               §11c (30 tokens vs 21 + side channel) is still the user's open decision.

If a conformance test here fails, do NOT edit the literal to match. Upstream changing an
extension ordering is the finding, and every checkpoint trained under the old ordering is
affected by it.
"""

from __future__ import annotations

import pytest

# Pinned from ProtonPottsMPNN @ 09682ab,
# foundry/models/mpnn/src/mpnn/transforms/feature_aggregation/token_encodings.py:124-146.
# Index = 21 + position, since both extensions begin immediately after the 21-token prefix.
PREFIX_LEN = 21

V4_PROTONATION_TOKENS = (
  "HID",  # 21 -- delta tautomer (neutral, ND1 protonated)
  "HIE",  # 22 -- epsilon tautomer (neutral, NE2 protonated)
  "HIS-P",  # 23 -- doubly protonated (+1, imidazolium)
  "HIS-D",  # 24 -- deprotonated (-1, imidazolate)
  "HIS-A",  # 25 -- ambiguous
  "ASP-P",  # 26
  "ASP-D",  # 27
  "ASP-A",  # 28
  "GLU-P",  # 29
  "GLU-D",  # 30
  "GLU-A",  # 31
)

V6_PROTONATION_TOKENS = (
  "HIS-P",  # 21
  "HIS-S",  # 22 -- neutral, a SINGLE token: v6 predicts charge state, not the HID/HIE tautomer
  "HIS-A",  # 23
  "ASP-P",  # 24
  "ASP-D",  # 25
  "ASP-A",  # 26
  "GLU-P",  # 27
  "GLU-D",  # 28
  "GLU-A",  # 29
)

V4_SIZE = PREFIX_LEN + len(V4_PROTONATION_TOKENS)
V6_SIZE = PREFIX_LEN + len(V6_PROTONATION_TOKENS)


def _index_of(tokens: tuple[str, ...], token: str) -> int:
  return PREFIX_LEN + tokens.index(token)


# --------------------------------------------------------------------------------------------
# Premise. These need no upstream install, so they run everywhere and can never silently skip.
# --------------------------------------------------------------------------------------------


def test_the_two_vocabularies_have_the_declared_widths() -> None:
  assert V4_SIZE == 32
  assert V6_SIZE == 30


def test_v6_drops_the_tautomers_and_the_imidazolate() -> None:
  """The documented difference: v6 predicts CHARGE STATE, not tautomer."""
  dropped = set(V4_PROTONATION_TOKENS) - set(V6_PROTONATION_TOKENS)
  assert dropped == {"HID", "HIE", "HIS-D"}
  added = set(V6_PROTONATION_TOKENS) - set(V4_PROTONATION_TOKENS)
  assert added == {"HIS-S"}


def test_no_protonation_index_means_the_same_thing_in_both() -> None:
  """The collision, stated as the property rather than as a table.

  This is the premise of §25.3. If it ever becomes false -- if upstream aligns the two
  extensions -- that is a real change and this test should fail so someone reads it, not be
  quietly relaxed.
  """
  overlap = range(PREFIX_LEN, min(V4_SIZE, V6_SIZE))
  agreeing = [
    i
    for i in overlap
    if V4_PROTONATION_TOKENS[i - PREFIX_LEN] == V6_PROTONATION_TOKENS[i - PREFIX_LEN]
  ]
  assert agreeing == []


def test_the_worked_collision_at_index_27() -> None:
  """The specific swap quoted in §25.3 and in the commit message, pinned so it stays quotable."""
  assert V4_PROTONATION_TOKENS[27 - PREFIX_LEN] == "ASP-D"
  assert V6_PROTONATION_TOKENS[27 - PREFIX_LEN] == "GLU-P"


def test_v6_indices_match_what_the_spike_measured() -> None:
  """§23 injected HIS-S / ASP-D / GLU-P and observed S taking 22 / 25 / 27.

  Those were MEASURED on a real forward pass, so this ties the literal table to an observation
  rather than to a second reading of the same tuple.
  """
  assert _index_of(V6_PROTONATION_TOKENS, "HIS-S") == 22
  assert _index_of(V6_PROTONATION_TOKENS, "ASP-D") == 25
  assert _index_of(V6_PROTONATION_TOKENS, "GLU-P") == 27


# --------------------------------------------------------------------------------------------
# Conformance to upstream. Skips where the ProtonPottsMPNN env is absent (i.e. everywhere but
# the aminx-oracles-protonpotts env on titanix).
# --------------------------------------------------------------------------------------------


@pytest.fixture
def upstream_tokens():  # noqa: ANN201
  return pytest.importorskip(
    "mpnn.transforms.feature_aggregation.token_encodings",
    reason="ProtonPottsMPNN not importable; run in the aminx-oracles-protonpotts env",
  )


def test_pinned_v4_matches_upstream(upstream_tokens) -> None:  # noqa: ANN001
  assert upstream_tokens.PROTONATION_TOKENS == V4_PROTONATION_TOKENS


def test_pinned_v6_matches_upstream(upstream_tokens) -> None:  # noqa: ANN001
  assert upstream_tokens.POTTS_MPNN_V6_PROTONATION_TOKENS == V6_PROTONATION_TOKENS


def test_the_prefix_is_21_tokens_and_is_shared(upstream_tokens) -> None:  # noqa: ANN001
  """Both extensions start at 21, which is why an unlabelled cell cannot reveal a mismatch."""
  assert len(upstream_tokens.token_order) == PREFIX_LEN
  v4_full = upstream_tokens.token_order + upstream_tokens.PROTONATION_TOKENS
  v6_full = upstream_tokens.token_order + upstream_tokens.POTTS_MPNN_V6_PROTONATION_TOKENS
  assert v4_full[:PREFIX_LEN] == v6_full[:PREFIX_LEN]
  assert len(v4_full) == V4_SIZE
  assert len(v6_full) == V6_SIZE
