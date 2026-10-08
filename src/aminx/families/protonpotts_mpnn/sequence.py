"""Build the ProtonPottsMPNN ``S`` tensor from residue names and pre-assigned protonation labels.

Port of upstream ``_build_protonation_aware_seq`` at inference
(``protonation_label_rate = 1.0``; ProtonPottsMPNN @ 09682ab,
``feature_aggregation/potts_mpnn.py:15-78``). aminx consumes PRE-LABELLED structures (spec §11a,
decided 261007): it does not assign protonation states, it reads them.

Same as upstream, a non-empty label wins over the residue name. Stricter than upstream in two
places, both deliberate and both only for input upstream would accept silently:

* a label outside the v6 vocabulary raises. Upstream resolves it to ``UNK``, so a v4 label
  (``HID``) fed to a v6 model silently becomes an unknown residue (spec §22.2);
* a label whose parent residue disagrees with the residue name raises (``ASP-P`` on a
  histidine). Upstream does not check, and the result would be a valid-looking token for the
  wrong amino acid.

For every input upstream accepts and means, the result is identical.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from aminx.families.protonpotts_mpnn.vocab import (
  _PARENT,
  PROTONPOTTS_V6,
  THREE_TO_ONE,
  V6_PROTONATION_TOKENS,
  token_index,
)


def encode_sequence(
  res_names: Sequence[str],
  protonation_labels: Sequence[str] | None = None,
) -> np.ndarray:
  """``S`` as an ``(L,)`` int64 array of aminx v6 indices.

  Args:
    res_names: Three-letter residue names, one per residue (``"HIS"``). A name that is not one
      of the 20 standard residues becomes ``X``, as upstream resolves it to ``UNK``.
    protonation_labels: One entry per residue, or ``None`` for no labels. An empty string means
      "no label for this residue". A non-empty entry must be one of the nine v6 tokens and
      must belong to the same parent residue as ``res_names[i]``.

  Raises:
    ValueError: lengths disagree, a label is outside the v6 vocabulary, or a label's parent
      residue does not match the residue name.
  """
  if protonation_labels is not None and len(protonation_labels) != len(res_names):
    msg = (
      f"protonation_labels has {len(protonation_labels)} entries for {len(res_names)} residues; "
      "pass one entry per residue, with '' for residues that carry no label."
    )
    raise ValueError(msg)

  out = np.empty(len(res_names), dtype=np.int64)
  for i, name in enumerate(res_names):
    label = protonation_labels[i] if protonation_labels is not None else ""
    if label:
      if label not in V6_PROTONATION_TOKENS:
        msg = (
          f"residue {i}: label {label!r} is not a {PROTONPOTTS_V6.name} protonation token "
          f"{V6_PROTONATION_TOKENS}. v4 labels (HID, HIE, HIS-D, ...) are refused, not "
          "resolved to X (spec §22.2)."
        )
        raise ValueError(msg)
      parent = THREE_TO_ONE.get(name)
      if parent is not None and _PARENT[label] != parent:
        msg = f"residue {i}: label {label!r} belongs to {_PARENT[label]} but the residue is {name}"
        raise ValueError(msg)
      out[i] = token_index(label)
    else:
      letter = THREE_TO_ONE.get(name)
      out[i] = token_index(letter) if letter is not None else PROTONPOTTS_V6.x_index
  return out
