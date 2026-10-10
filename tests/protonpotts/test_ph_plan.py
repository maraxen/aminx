"""Host-side pH-design planning (``ph_plan``) on a hand-built 12-residue graph.

Every expected value is derived by hand from the graph below and written as a literal. Nothing
is random. Vocabulary indices (v6): HIS-P=21, HIS-S=22, ASP-P=24, ASP-D=25, GLU-P=27, X=20,
HIS-A=23, ASP-A=26, GLU-A=29.

The graph has L=12 and K=6 (slot 0 = self). Position 11 is not a free binder position.

    p : row (e_idx[p])        p : row (e_idx[p])
    0 : [0,1,2,3,4,5]         6 : [6,7,8,9,10,11]
    1 : [1,0,2,6,7,8]         7 : [7,6,8,9,10,11]
    2 : [2,3,4,5,6,7]         8 : [8,7,9,10,11,6]
    3 : [3,2,4,5,6,7]         9 : [9,0,8,6,7,10]   (0 at slot 1: incoming-only for 0)
    4 : [4,3,5,6,7,8]        10 : [10,8,9,6,0,7]   (0 at slot 4: incoming-only, cut at k<=2)
    5 : [5,4,6,7,8,9]        11 : [11,10,9,8,7,6]
"""

from __future__ import annotations

import numpy as np
import pytest

from aminx.families.protonpotts_mpnn.ph_plan import (
  Pin,
  Plan,
  block_partners,
  block_table,
  finalize_plan,
  knn_rank,
  neighbour_mask,
  placement_scores,
  plan_centre_free,
  plan_from_center_types,
  plan_from_explicit_centers,
  valid_token_mask,
)

E_IDX = np.array(
  [
    [0, 1, 2, 3, 4, 5],
    [1, 0, 2, 6, 7, 8],
    [2, 3, 4, 5, 6, 7],
    [3, 2, 4, 5, 6, 7],
    [4, 3, 5, 6, 7, 8],
    [5, 4, 6, 7, 8, 9],
    [6, 7, 8, 9, 10, 11],
    [7, 6, 8, 9, 10, 11],
    [8, 7, 9, 10, 11, 6],
    [9, 0, 8, 6, 7, 10],
    [10, 8, 9, 6, 0, 7],
    [11, 10, 9, 8, 7, 6],
  ]
)
L = 12
RES_ID = np.arange(1, L + 1)  # residue number of position p is p + 1
BINDER = np.ones(L, dtype=bool)
BINDER[11] = False
DEP_MAP = {"HIS-P": ("HIS-S",), "ASP-P": ("ASP-D",), "GLU-P": ("GLU-D",)}


def _flat(mask: np.ndarray) -> list[int]:
  return np.flatnonzero(mask).tolist()


def _field() -> np.ndarray:
  """Native-sequence conditional energies, (L, 30). Zero everywhere unless set below.

  HIS-P score = field[:,21] - field[:,22]; ASP-P score = field[:,24] - field[:,25].
  Position 11 is the best HIS-P site but is not a binder position, so it must never be chosen.
  """
  field = np.zeros((L, 30))
  field[:, 21] = 1.0  # HIS-P default score 1.0
  field[2, 21] = -3.0  # HIS-P best at position 2 (score -3.0)
  field[7, 21] = -1.0  # HIS-P second at position 7 (score -1.0)
  field[11, 21] = -10.0  # not a binder position
  field[2, 24] = -2.0  # ASP-P best at position 2 (score -2.0): competes with HIS-P
  field[5, 24] = -1.5  # ASP-P next best at position 5 (score -1.5)
  return field


# --- valid_token_mask -------------------------------------------------------------------------

def test_valid_token_mask_forbidden_residues_and_unk_leave_26_true() -> None:
  mask = valid_token_mask(["HIS-A", "ASP-A", "GLU-A", "UNK"])
  assert mask.shape == (30,)
  assert int(mask.sum()) == 26
  for idx in (20, 23, 26, 29):  # X, HIS-A, ASP-A, GLU-A
    assert not mask[idx]
  assert mask[21]  # HIS-P is not forbidden


def test_valid_token_mask_ignores_names_missing_from_v6() -> None:
  # HIS-D is upstream's default forbidden name; v6 has no such token, so it is ignored.
  mask = valid_token_mask(["HIS-D"])
  assert int(mask.sum()) == 29
  assert not mask[20]  # only X is invalid


# --- neighbour_mask ---------------------------------------------------------------------------

def test_neighbour_mask_includes_outgoing_and_incoming_only_neighbours() -> None:
  # Outgoing 1..5 from row 0. Incoming: 1 (slot 1 of row 1), 9 (slot 1), 10 (slot 4).
  # 9 and 10 are not in row 0, so they are incoming-only neighbours.
  mask = neighbour_mask(E_IDX, 0, BINDER, neighbour_k=0)
  assert _flat(mask) == [1, 2, 3, 4, 5, 9, 10]


def test_neighbour_k_truncates_both_directions() -> None:
  # neighbour_k=2 -> slots 1..2. Position 10 has 0 only at slot 4, so it drops out.
  assert _flat(neighbour_mask(E_IDX, 0, BINDER, neighbour_k=2)) == [1, 2, 9]
  # neighbour_k=1 -> slot 1 only.
  assert _flat(neighbour_mask(E_IDX, 0, BINDER, neighbour_k=1)) == [1, 9]


def test_neighbour_mask_intersects_binder_mask() -> None:
  binder = BINDER.copy()
  binder[9] = False
  assert _flat(neighbour_mask(E_IDX, 0, binder, neighbour_k=0)) == [1, 2, 3, 4, 5, 10]


# --- block_partners / block_table -------------------------------------------------------------

def test_block_partners_can_be_shorter_than_block_size() -> None:
  # Row 4 partners (kNN order) are 3, 5, 6, 7, 8; only 3 is designable.
  assert block_partners(E_IDX[4, 1:], 4, {3, 4}, 3) == [4, 3]


def test_block_partners_size_one_is_the_position_itself() -> None:
  assert block_partners(E_IDX[2, 1:], 2, {2, 3, 4}, 1) == [2]


def test_block_table_pads_short_blocks_with_minus_one() -> None:
  blocks, valid = block_table(E_IDX, [2, 3, 4, 9, 10], 3)
  assert blocks.tolist() == [
    [2, 3, 4],
    [3, 2, 4],
    [4, 3, -1],
    [9, 10, -1],
    [10, 9, -1],
  ]
  assert valid.tolist() == [
    [True, True, True],
    [True, True, True],
    [True, True, False],
    [True, True, False],
    [True, True, False],
  ]


# --- knn_rank / finalize_plan -----------------------------------------------------------------

def _pin(position: int, ptype: str = "HIS-P") -> Pin:
  prot_idx, dep_idxs = {"HIS-P": (21, (22,)), "ASP-P": (24, (25,))}[ptype]
  return Pin(position, ptype, prot_idx, dep_idxs, int(RES_ID[position]))


def test_knn_rank_orders_by_centre_row_then_position() -> None:
  # Row 0 partners: 1,2,3,4,5 (ranks 0..4). 9 and 10 are unranked, so position order breaks the tie.
  assert knn_rank(E_IDX, [1, 2, 3, 4, 5, 9, 10], [_pin(0)]) == [1, 2, 3, 4, 5, 9, 10]


def test_explicit_plan_is_capped_by_max_mutations() -> None:
  plan = plan_from_explicit_centers(
    E_IDX, BINDER, RES_ID, [(1, "HIS-P")], DEP_MAP,
    infill_scope="neighbourhood", neighbour_k=0, max_mutations=6,
  )
  # Full designable set is (1,2,3,4,5,9,10). Capping at 6 drops 10, the later unranked one.
  assert plan.designable == (1, 2, 3, 4, 5, 9)


def test_finalize_plan_chain_scope_is_every_free_binder_position_minus_pins() -> None:
  plan = finalize_plan(
    E_IDX, BINDER, [_pin(0)],
    infill_scope="chain", neighbour_k=0, max_mutations=0,
  )
  assert plan == Plan(
    pins=(_pin(0),),
    designable=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    label="HIS-P",
  )


def test_finalize_plan_returns_none_when_nothing_is_designable() -> None:
  only_pin = np.zeros(L, dtype=bool)
  only_pin[0] = True
  assert finalize_plan(
    E_IDX, only_pin, [_pin(0)],
    infill_scope="chain", neighbour_k=0, max_mutations=0,
  ) is None


def test_finalize_plan_rejects_unknown_scope() -> None:
  with pytest.raises(ValueError, match="infill_scope"):
    finalize_plan(E_IDX, BINDER, [_pin(0)], infill_scope="ring", neighbour_k=0, max_mutations=0)


# --- placement_scores -------------------------------------------------------------------------

def test_placement_scores_are_prot_minus_min_dep_and_inf_outside_binder() -> None:
  scores = placement_scores(_field(), 21, [22], BINDER)
  assert scores[2] == -3.0
  assert scores[7] == -1.0
  assert scores[0] == 1.0  # any free residue type can be a centre
  assert np.isinf(scores[11])


# --- centre-type planning ---------------------------------------------------------------------

def test_two_centre_types_compete_and_second_takes_next_best_position() -> None:
  plan = plan_from_center_types(
    _field(), E_IDX, BINDER, RES_ID, ["HIS-P", "ASP-P"], DEP_MAP,
    infill_scope="neighbourhood", neighbour_k=0, max_mutations=0,
  )
  # HIS-P takes position 2 (score -3). ASP-P's best is also 2, which is taken, so it takes 5.
  assert plan.pins == (
    Pin(position=2, protonation_type="HIS-P", prot_idx=21, dep_idxs=(22,), res_id=3),
    Pin(position=5, protonation_type="ASP-P", prot_idx=24, dep_idxs=(25,), res_id=6),
  )
  # N(2) = {0,1,3,4,5,6,7}, N(5) = {0,2,3,4,6,7,8,9}; union minus pins {2,5}.
  assert plan.designable == (0, 1, 3, 4, 6, 7, 8, 9)
  assert plan.label == "ASP-P+HIS-P"


def test_centre_type_plan_is_capped_by_max_mutations() -> None:
  plan = plan_from_center_types(
    _field(), E_IDX, BINDER, RES_ID, ["HIS-P", "ASP-P"], DEP_MAP,
    infill_scope="neighbourhood", neighbour_k=0, max_mutations=4,
  )
  # Ranks: 3->0, 4->0, 6->1, 7->2, 8->3, 9->4; 0 and 1 are unranked and sort last.
  assert plan.designable == (3, 4, 6, 7)


def test_v3_dep_map_naming_hid_is_refused() -> None:
  with pytest.raises(ValueError, match="HID"):
    plan_from_center_types(
      _field(), E_IDX, BINDER, RES_ID, ["HIS-P"], {"HIS-P": ("HID",)},
      infill_scope="neighbourhood", neighbour_k=0, max_mutations=0,
    )


def test_unknown_centre_type_is_refused() -> None:
  with pytest.raises(ValueError, match="unknown centre protonation type"):
    plan_from_center_types(
      _field(), E_IDX, BINDER, RES_ID, ["HIS-S"], DEP_MAP,
      infill_scope="neighbourhood", neighbour_k=0, max_mutations=0,
    )


def test_centre_type_missing_from_dep_map_is_refused() -> None:
  with pytest.raises(ValueError, match="no contrast tokens"):
    plan_from_center_types(
      _field(), E_IDX, BINDER, RES_ID, ["GLU-P"], {"HIS-P": ("HIS-S",)},
      infill_scope="neighbourhood", neighbour_k=0, max_mutations=0,
    )


# --- explicit centres -------------------------------------------------------------------------

def test_explicit_centre_plan_pins_by_residue_number() -> None:
  plan = plan_from_explicit_centers(
    E_IDX, BINDER, RES_ID, [(1, "HIS-P")], DEP_MAP,
    infill_scope="neighbourhood", neighbour_k=0, max_mutations=0,
  )
  assert plan == Plan(
    pins=(Pin(position=0, protonation_type="HIS-P", prot_idx=21, dep_idxs=(22,), res_id=1),),
    designable=(1, 2, 3, 4, 5, 9, 10),
    label="HIS-P",
  )


def test_explicit_centre_outside_the_binder_is_refused() -> None:
  # Residue 12 is position 11, which is not a free binder position.
  with pytest.raises(ValueError, match="not a free binder position"):
    plan_from_explicit_centers(
      E_IDX, BINDER, RES_ID, [(12, "HIS-P")], DEP_MAP,
      infill_scope="neighbourhood", neighbour_k=0, max_mutations=0,
    )


# --- centre-free ------------------------------------------------------------------------------

def test_centre_free_plan_designs_every_free_binder_position() -> None:
  assert plan_centre_free(BINDER) == Plan(pins=(), designable=tuple(range(11)), label="")


def test_centre_free_plan_is_none_for_an_empty_binder() -> None:
  assert plan_centre_free(np.zeros(L, dtype=bool)) is None
