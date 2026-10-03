# ruff: noqa: S101
"""Spec §9 error table: overlapping tied groups must raise, not double-decode.

Upstream ``potts_mpnn_utils.py:1606-1614`` builds its tied groups by iterating
the listing and decoding each group once. A position appearing in two groups is
therefore decoded twice, and the second pass silently overwrites the first --
there is no error, just a wrong sequence. The spec's §9 row makes that a
``ValueError`` in aminx.

``build_tie_groups_np`` (``sample_host.py:223-226``) does raise it, and
``tests/families/potts_mpnn/test_sample.py:317`` already pins that at the unit
level. What was missing, and is why this file exists, is whether a USER can
reach the guard at all -- see ``test_tied_overlap_is_unreachable_from_the_spec``.
"""

from __future__ import annotations

import numpy as np
import pytest

from aminx.families.potts_mpnn.featurize import tied_featurize_port
from aminx.families.potts_mpnn.sample_host import build_tie_groups_np


def test_tied_overlap_raises() -> None:
  """A position in two groups raises rather than being decoded twice.

  Both halves of the guard are checked, because they are different mistakes:
  a position shared BETWEEN groups (upstream's silent double-decode) and a
  position repeated WITHIN one group (a group that decodes itself twice).
  """
  with pytest.raises(ValueError, match="overlapping tied groups"):
    build_tie_groups_np(((0, 1), (1, 2)), 3, 3)

  with pytest.raises(ValueError, match="overlapping tied groups"):
    build_tie_groups_np(((0, 0),), 3, 3)

  # The control: the same shape without the overlap must build fine, so the
  # assertions above are about the overlap and not about the call failing for
  # some unrelated reason (a bad l_total, an empty group, a dtype).
  table = build_tie_groups_np(((0, 1), (2,)), 3, 3)
  assert table.shape[0] == 3
  assert set(table[0, :2].tolist()) == {0, 1}


def test_tied_overlap_is_unreachable_from_the_spec() -> None:
  """TRIPWIRE: ``tied_positions`` is inert on the Potts path, so the guard is dead.

  MEASURED 261003 by reading the call graph, not the docstrings:

    * ``features.tied_pos`` is populated at exactly one place,
      ``featurize.py:441``, from the ``tied_positions_dict`` argument. When that
      argument is ``None`` the branch at ``:438-439`` sets ``tied_pos = ()``.
    * Both of the driver's ``tied_featurize_port`` call sites --
      ``driver.py:573`` and ``driver.py:774`` -- pass no ``tied_positions_dict``.
      ``:774`` passes ``pssm_dict`` and ``bias_by_res_dict`` and stops; the
      docstring at ``:767-772`` says the per-chain conversion "does not exist
      yet".
    * So ``features.tied_pos`` is ALWAYS ``()`` from the public surface, which
      makes ``build_tie_groups_np(features.tied_pos, ...)``
      (``sample_host.py:325``) incapable of ever seeing an overlap, and
      ``tied=bool(features.tied_pos)`` (``:337``) permanently ``False`` -- the
      whole tied decode path is dead, not merely the error.
    * The generic ``RunSpecification.tied_positions`` has its own, separate
      dead end: its only consumer is ``utils/autoregression.resolve_tie_groups``
      (``:41``), which has ZERO callers anywhere in ``src/aminx/``.

  So the knob is inert in two independent ways, which is the same shape as the
  ``pssm_json``/``bias_by_res_json`` defect (debt 2443): the field parses, and
  reaches nothing.

  WHEN THIS IS FIXED, THIS TEST MUST FAIL, and that is the point of it. Pass
  ``tied_positions_dict`` through ``driver.py:774``, then DELETE this test and
  replace it with one that drives an overlapping ``tied_positions`` through
  ``sample()`` and asserts the ``ValueError`` surfaces to the caller. Do not
  relax the assertion below to keep it green.
  """
  parsed = {
    "seq_chain_A": "ACD",
    "coords_chain_A": {
      f"{atom}_chain_A": [[float(i), 0.0, float(j)] for i in range(3)]
      for j, atom in enumerate(("N", "CA", "C", "O"))
    },
    "name": "toy",
    "num_of_chains": 1,
    "seq": "ACD",
    "chain_order": ["A"],
  }
  # Overlapping groups, exactly the input the guard exists to reject...
  overlapping = {"toy": [{"A": [1, 2]}, {"A": [2, 3]}]}

  # ...and featurize DOES honour it when the dict is actually handed over.
  with_dict = tied_featurize_port([parsed], None, tied_positions_dict=overlapping)[0]
  assert with_dict.tied_pos, "featurize ignored a tied_positions_dict it was given"
  with pytest.raises(ValueError, match="overlapping tied groups"):
    build_tie_groups_np(with_dict.tied_pos, with_dict.L_total, with_dict.L_total)

  # The defect: the driver's call shape drops it, so tied_pos comes back empty
  # and the guard above can never fire in a real run.
  as_driver_calls_it = tied_featurize_port([parsed], None)[0]
  assert as_driver_calls_it.tied_pos == (), (
    "tied_pos is no longer empty without a tied_positions_dict -- the inertness "
    "this tripwire records has changed; re-read driver.py:774 and replace this "
    "test with the end-to-end one described in the docstring."
  )
  assert np.array_equal(
    build_tie_groups_np(
      as_driver_calls_it.tied_pos, as_driver_calls_it.L_total,
      as_driver_calls_it.L_total,
    ),
    np.arange(as_driver_calls_it.L_total, dtype=np.int32).reshape(-1, 1),
  ), "empty tied_pos no longer degenerates to one singleton group per row"
