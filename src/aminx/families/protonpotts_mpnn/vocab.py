"""ProtonPottsMPNN v6 vocabulary: token names, aminx indices, and the upstream index maps.

Spec: ``.praxia/docs/specs/261007_protonpottsmpnn-support.md`` §5, §15.2, §21.4, §22.2, §25.

The v6 model speaks a 30-token vocabulary: the 20 standard residues, ``X``, and nine
protonation tokens. Two facts shape this module.

1. **v4 and v6 collide.** Upstream ships a 32-token (v3/v4, the inference default) and a
   30-token (v6) extension. They share the 21-token prefix and then disagree at EVERY index
   from 21 up (index 27 is ``ASP-D`` in v4 and ``GLU-P`` in v6). A path must therefore name
   its vocabulary; this module has exactly one, ``PROTONPOTTS_V6``, whose ``name`` says so,
   and a label from the v4 extension (``HID``, ``HIE``, ``HIS-D``) is REFUSED rather than
   resolved to ``X``.
2. **Token order is a conversion-time question, not a runtime one** (§15.2). The model's rows
   are permuted into aminx's order when the weights are converted (P6). So the index space
   here is aminx's own: the 20 standard letters in alphabetical one-letter order (the same as
   ``POTTS_MPNN``), ``X`` at 20, then the nine v6 tokens at 21-29 in upstream's order. The
   upstream-order tables below exist only so a dumped oracle ``S`` can be compared.

The pair table is ``30 x 30`` (measured: ``etab_out`` is ``(1, L, K, 30, 30)``, spec §2a).
"""

from __future__ import annotations

from aminx.families.potts_mpnn.alphabet import PottsAlphabet

STANDARD_LETTERS = "ACDEFGHIKLMNPQRSTVWY"

# Upstream v6 protonation tokens, in index order 21..29
# (ProtonPottsMPNN @ 09682ab, token_encodings.py:153-157). Pinned by
# tests/protonpotts/test_vocab_index_tables.py against upstream.
V6_PROTONATION_TOKENS: tuple[str, ...] = (
  "HIS-P",
  "HIS-S",
  "HIS-A",
  "ASP-P",
  "ASP-D",
  "ASP-A",
  "GLU-P",
  "GLU-D",
  "GLU-A",
)

THREE_TO_ONE: dict[str, str] = {
  "ALA": "A",
  "ARG": "R",
  "ASN": "N",
  "ASP": "D",
  "CYS": "C",
  "GLN": "Q",
  "GLU": "E",
  "GLY": "G",
  "HIS": "H",
  "ILE": "I",
  "LEU": "L",
  "LYS": "K",
  "MET": "M",
  "PHE": "F",
  "PRO": "P",
  "SER": "S",
  "THR": "T",
  "TRP": "W",
  "TYR": "Y",
  "VAL": "V",
}

# Upstream (foundry) index order for the 21-token prefix: atomworks STANDARD_AA + UNK
# (token_encodings.py:5). Verified 261008 against atomworks on titanix, and against the P4
# oracle (ASP, GLU, HIS sit at indices 3, 6, 8: spec §23).
FOUNDRY_PREFIX_ORDER: tuple[str, ...] = (
  "ALA",
  "ARG",
  "ASN",
  "ASP",
  "CYS",
  "GLN",
  "GLU",
  "GLY",
  "HIS",
  "ILE",
  "LEU",
  "LYS",
  "MET",
  "PHE",
  "PRO",
  "SER",
  "THR",
  "TRP",
  "TYR",
  "VAL",
  "UNK",
)

PROTONPOTTS_V6 = PottsAlphabet(
  name="protonpottsmpnn_v6_30",
  symbols=(*STANDARD_LETTERS, "X", *V6_PROTONATION_TOKENS),
  x_index=20,
  pair_side=30,
  # The pair table is indexed by the 30 model tokens directly. Unlike ``pottsmpnn_21`` there
  # is no separate "-" slot: the measured table is 30 x 30, not 22 x 22.
  etab_symbols=(*STANDARD_LETTERS, "X", *V6_PROTONATION_TOKENS),
  standard_symbols=tuple(STANDARD_LETTERS),
)

_INDEX = {symbol: i for i, symbol in enumerate(PROTONPOTTS_V6.symbols)}
_PARENT = {token: THREE_TO_ONE[token[:3]] for token in V6_PROTONATION_TOKENS}


def token_index(token: str) -> int:
  """aminx index of a one-letter residue, ``X``, or a v6 protonation token."""
  try:
    return _INDEX[token]
  except KeyError:
    msg = (
      f"{token!r} is not in the {PROTONPOTTS_V6.name} vocabulary. v4 extension tokens "
      "(HID, HIE, HIS-D, ...) are deliberately refused: upstream would silently resolve "
      "them to UNK, which hides a v4/v6 mix-up (spec §22.2)."
    )
    raise ValueError(msg) from None


def index_token(index: int) -> str:
  """The token name at an aminx index."""
  if not 0 <= index < PROTONPOTTS_V6.size:
    msg = f"index {index} is outside [0, {PROTONPOTTS_V6.size})"
    raise ValueError(msg)
  return PROTONPOTTS_V6.symbols[index]


def canonical_letter(index: int) -> str:
  """Project a model index to its canonical one-letter residue (``HIS-P`` -> ``H``).

  Protonation variants are tokens but not residues, so every place that assumes a 21-letter
  sequence (FASTA sinks, ``sequences_to_score``, DMS targets) goes through this.
  """
  token = index_token(index)
  return _PARENT.get(token, token)


def upstream_to_aminx_index(upstream_index: int) -> int:
  """aminx index for an upstream (foundry v6) index, for comparing a dumped oracle ``S``."""
  if 0 <= upstream_index < len(FOUNDRY_PREFIX_ORDER):
    name = FOUNDRY_PREFIX_ORDER[upstream_index]
    return PROTONPOTTS_V6.x_index if name == "UNK" else _INDEX[THREE_TO_ONE[name]]
  extension = upstream_index - len(FOUNDRY_PREFIX_ORDER)
  if 0 <= extension < len(V6_PROTONATION_TOKENS):
    return _INDEX[V6_PROTONATION_TOKENS[extension]]
  msg = f"upstream index {upstream_index} is outside the 30-token v6 vocabulary"
  raise ValueError(msg)
