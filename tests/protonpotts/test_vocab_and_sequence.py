"""P3/P5: the v6 vocabulary and the ``S`` encoder.

The oracle-conformance test needs the P4 dump (``AMINX_PROTONPOTTS_ORACLE`` pointing at
``oracle_f32.npz``, sha256 ``bbb68283...``) and skips without it, like the upstream-conformance
tests beside it. Everything else runs with nothing installed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from aminx.families.potts_mpnn.alphabet import POTTS_MPNN
from aminx.families.protonpotts_mpnn.sequence import encode_sequence
from aminx.families.protonpotts_mpnn.vocab import (
  FOUNDRY_PREFIX_ORDER,
  PROTONPOTTS_V6,
  THREE_TO_ONE,
  V6_PROTONATION_TOKENS,
  canonical_letter,
  index_token,
  token_index,
  upstream_to_aminx_index,
)

ORACLE_SHA256 = "bbb6828335d28e7427b18d7bdc1c39138e8716022fb54c1a875e4fcd3d6a0d6e"


# --- vocabulary ----------------------------------------------------------------------


def test_v6_is_thirty_tokens_with_a_thirty_square_pair_table() -> None:
  assert PROTONPOTTS_V6.size == 30
  assert (PROTONPOTTS_V6.pair_side, PROTONPOTTS_V6.pair_dim) == (30, 900)
  assert PROTONPOTTS_V6.x_index == 20


def test_first_21_indices_are_the_shipped_potts_alphabet() -> None:
  """aminx order for 0-20 must stay identical to ``pottsmpnn_21``: the weights are converted into it."""
  assert PROTONPOTTS_V6.symbols[:21] == POTTS_MPNN.symbols


def test_extension_is_the_nine_v6_tokens_in_upstream_order() -> None:
  assert PROTONPOTTS_V6.symbols[21:] == V6_PROTONATION_TOKENS
  assert [token_index(t) for t in V6_PROTONATION_TOKENS] == list(range(21, 30))


@pytest.mark.parametrize("v4_only", ["HID", "HIE", "HIS-D"])
def test_v4_only_tokens_are_refused_not_resolved_to_x(v4_only: str) -> None:
  with pytest.raises(ValueError, match="not in the protonpottsmpnn_v6_30 vocabulary"):
    token_index(v4_only)


def test_index_token_round_trips_and_bounds() -> None:
  assert [token_index(index_token(i)) for i in range(30)] == list(range(30))
  for bad in (-1, 30):
    with pytest.raises(ValueError, match="outside"):
      index_token(bad)


@pytest.mark.parametrize(
  ("index", "letter"),
  [(token_index("HIS-P"), "H"), (token_index("HIS-S"), "H"), (token_index("ASP-D"), "D"),
   (token_index("GLU-A"), "E"), (token_index("W"), "W"), (token_index("X"), "X")],
)
def test_canonical_projection(index: int, letter: str) -> None:
  assert canonical_letter(index) == letter


def test_every_variant_projects_to_a_standard_letter() -> None:
  standard = set(PROTONPOTTS_V6.standard_symbols)
  assert {canonical_letter(i) for i in range(21, 30)} <= standard


def test_upstream_index_map_is_a_bijection_onto_the_aminx_indices() -> None:
  mapped = [upstream_to_aminx_index(i) for i in range(30)]
  assert sorted(mapped) == list(range(30))


def test_upstream_index_map_known_points() -> None:
  # foundry order is ARND...: ASP=3, GLU=6, HIS=8 (also read off the oracle, spec §23); UNK=20.
  assert index_token(upstream_to_aminx_index(3)) == "D"
  assert index_token(upstream_to_aminx_index(6)) == "E"
  assert index_token(upstream_to_aminx_index(8)) == "H"
  assert index_token(upstream_to_aminx_index(20)) == "X"
  assert index_token(upstream_to_aminx_index(27)) == "GLU-P"  # v4 says ASP-D here: the collision
  with pytest.raises(ValueError, match="outside"):
    upstream_to_aminx_index(30)


def test_three_letter_table_covers_exactly_the_foundry_standard_residues() -> None:
  assert set(THREE_TO_ONE) == set(FOUNDRY_PREFIX_ORDER[:-1])
  assert sorted(THREE_TO_ONE.values()) == sorted(PROTONPOTTS_V6.standard_symbols)


# --- the S encoder -------------------------------------------------------------------


def test_unlabelled_sequence_is_the_plain_potts_encoding() -> None:
  s = encode_sequence(["ALA", "HIS", "TYR", "UNK", "MSE"])
  assert s.tolist() == [token_index(x) for x in ("A", "H", "Y", "X", "X")]
  assert s.dtype == np.int64


def test_label_wins_over_residue_name_and_empty_label_falls_back() -> None:
  s = encode_sequence(["HIS", "HIS", "ASP", "GLU"], ["HIS-P", "", "ASP-D", "GLU-A"])
  assert s.tolist() == [
    token_index("HIS-P"),
    token_index("H"),
    token_index("ASP-D"),
    token_index("GLU-A"),
  ]


def test_none_labels_equals_all_empty_labels() -> None:
  names = ["HIS", "ASP", "GLY"]
  assert encode_sequence(names).tolist() == encode_sequence(names, ["", "", ""]).tolist()


def test_v4_label_is_refused() -> None:
  with pytest.raises(ValueError, match="v4 labels"):
    encode_sequence(["HIS"], ["HID"])


def test_label_for_the_wrong_parent_residue_is_refused() -> None:
  with pytest.raises(ValueError, match="belongs to D but the residue is HIS"):
    encode_sequence(["HIS"], ["ASP-P"])


def test_label_count_must_match_residue_count() -> None:
  with pytest.raises(ValueError, match="one entry per residue"):
    encode_sequence(["HIS", "ASP"], ["HIS-P"])


# --- the P4 oracle, exactly ----------------------------------------------------------


def _oracle() -> dict[str, np.ndarray]:
  raw = os.environ.get("AMINX_PROTONPOTTS_ORACLE")
  if not raw:
    pytest.skip("AMINX_PROTONPOTTS_ORACLE not set")
  assert raw is not None
  path = Path(raw)
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  assert digest == ORACLE_SHA256, f"not the sealed P4 oracle: {digest}"
  with np.load(path) as z:
    return {k: z[k] for k in ("S_unlabelled", "S_labelled")}


def test_oracle_labelled_s_is_reproduced_exactly() -> None:
  """Rebuild names and labels from the dump, encode them, and require the dumped labelled ``S``.

  The dump holds only index arrays, so names come from the foundry prefix order and a label is
  any extension index (>= 21) at a position whose unlabelled token is its parent residue. This
  exercises the extension-region mapping against truth rather than against my own table, and
  every residue the dump carries.
  """
  z = _oracle()
  unlabelled, labelled = z["S_unlabelled"][0], z["S_labelled"][0]
  names = [FOUNDRY_PREFIX_ORDER[int(i)] for i in unlabelled]
  labels = [
    V6_PROTONATION_TOKENS[int(lab) - len(FOUNDRY_PREFIX_ORDER)] if lab >= 21 else ""
    for lab in labelled
  ]
  got = encode_sequence(names, labels)
  want = np.array([upstream_to_aminx_index(int(i)) for i in labelled], dtype=np.int64)
  assert got.tolist() == want.tolist()
  assert (labelled >= 21).sum() > 0  # the cell is non-vacuous: labels actually change S
  assert (got != encode_sequence(names)).sum() == (labelled >= 21).sum()


# The dumper's labelling rule (scripts/protonpotts/dump_protonpotts_oracles.py:74), copied because
# that module imports the oracle env at top level. By ordinal within residue type: the k-th HIS
# gets CYCLE["HIS"][k % 3], and so on. This is the oracle's own ground truth for which TOKEN each
# dumped index means, which is what makes the within-parent order testable at all.
CYCLE = {
  "HIS": ("HIS-P", "HIS-S", "HIS-A"),
  "ASP": ("ASP-P", "ASP-D", "ASP-A"),
  "GLU": ("GLU-P", "GLU-D", "GLU-A"),
}
FOUNDRY_PARENT_INDEX = {"HIS": 8, "ASP": 3, "GLU": 6}


def test_extension_order_is_pinned_by_literal_not_by_its_own_table() -> None:
  """An independent copy, so swapping two tokens in ``vocab.py`` cannot also edit this.

  Added after a mutation check: swapping ASP-P and ASP-D in vocab.py passed every other test,
  because each compared the table with something derived from the table.
  """
  assert V6_PROTONATION_TOKENS == (
    "HIS-P", "HIS-S", "HIS-A", "ASP-P", "ASP-D", "ASP-A", "GLU-P", "GLU-D", "GLU-A",
  )  # fmt: skip


def test_oracle_extension_indices_match_the_dumpers_cycling_rule() -> None:
  """Each token's dumped upstream index, derived from the cycling rule, equals ``21 + position``."""
  z = _oracle()
  unlabelled, labelled = z["S_unlabelled"][0], z["S_labelled"][0]
  observed: dict[str, set[int]] = {}
  for parent, foundry_index in FOUNDRY_PARENT_INDEX.items():
    for k, pos in enumerate(np.flatnonzero(unlabelled == foundry_index)):
      observed.setdefault(CYCLE[parent][k % 3], set()).add(int(labelled[pos]))
  assert set(observed) == set(V6_PROTONATION_TOKENS)  # all nine reached: a non-vacuous cell
  assert all(len(v) == 1 for v in observed.values()), observed  # one index per token name
  got = {name: next(iter(v)) for name, v in observed.items()}
  assert got == {name: 21 + i for i, name in enumerate(V6_PROTONATION_TOKENS)}
