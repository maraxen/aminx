"""potts_order wave.

Spec §4.2 and §4.3 name three upstream comparisons that share one f64 dump,
``potts_order/order_f64.npz``. Orders and neighbour sets are integers, so the
comparison is ``array_equal`` / set equality. There is no tolerance.

``tests/port/mutant_registry.py`` resolves one ``AMINX_PORT_MUTANT`` name to a
context manager. It does not attach a mutant to a single test, and
``tests/port/mutants/`` only carries the port self-test. The negative controls
below are in-test monkeypatches.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from port.a1_compare import open_dump, x64_context

from aminx.families.potts_mpnn import decode as potts_decode
from aminx.families.potts_mpnn import sample_host
from aminx.families.potts_mpnn.features import potts_edge_features
from aminx.families.potts_mpnn.model import PottsMPNN

pytestmark = [pytest.mark.port_wave("potts_order"), pytest.mark.parity_heavy]

_NEIGHBOUR_CASES = ("L30", "L48", "L49", "partition_Lp20")


def _vector(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
  array = np.asarray(data[key])
  if array.ndim == 2 and array.shape[0] == 1:
    return array[0]
  return array


def _upstream_sets(e_idx: np.ndarray) -> list[set[int]]:
  return [{int(index) for index in row.tolist()} for row in e_idx]


def _valid_sets(e_idx: np.ndarray, pad_valid: np.ndarray) -> list[set[int]]:
  valid = np.asarray(potts_decode.neighbor_valid(jnp.asarray(e_idx), jnp.asarray(pad_valid)))
  sets: list[set[int]] = []
  for row, mask in zip(e_idx, valid, strict=True):
    sets.append({int(index) for index, ok in zip(row.tolist(), mask.tolist(), strict=True) if ok})
  return sets


def _encoder_indices(
  model: PottsMPNN,
  coords: np.ndarray,
  mask: np.ndarray,
  residue_idx: np.ndarray,
  chain_encoding: np.ndarray,
) -> np.ndarray:
  graph = potts_edge_features(
    model.mpnn.features,
    jnp.asarray(coords),
    jnp.asarray(mask),
    jnp.asarray(residue_idx),
    jnp.asarray(chain_encoding),
  )
  return np.asarray(graph.neighbor_indices)


def test_valid_neighbour_set_matches_upstream(oracle: object) -> None:
  """Encoder neighbour sets match upstream on real rows at the K=48 boundary.

  The four fixtures are L=30, L=48, L=49, and a binding partition with L_p<48.
  ``neighbor_valid`` drops an index that is out of range or on a pad row, so
  the set of indices it keeps is the valid neighbour set. On these unpadded
  graphs every real row is in range; a K or tie-break drift still changes the
  set, which a length check would not see.
  """
  data = open_dump(oracle, "f64")
  try:
    cases = [str(name) for name in np.asarray(data["neighbour_cases"]).tolist()]
    assert cases == list(_NEIGHBOUR_CASES)
    with x64_context("f64"):
      model = PottsMPNN(key=jax.random.PRNGKey(0))
      for name in cases:
        coords = np.asarray(data[f"{name}__coords"])
        mask = np.asarray(data[f"{name}__mask"])
        residue_idx = np.asarray(data[f"{name}__residue_idx"])
        chain_encoding = np.asarray(data[f"{name}__chain_encoding"])
        upstream = np.asarray(data[f"{name}__E_idx"])
        length = int(coords.shape[0])
        if name == "L30":
          assert length == 30
        elif name == "L48":
          assert length == 48
        elif name == "L49":
          assert length == 49
        else:
          assert length < 48
          assert int(np.asarray(data[f"{name}__L_p"])) == length
        pad_valid = np.ones((length,), dtype=bool)
        got = _encoder_indices(model, coords, mask, residue_idx, chain_encoding)
        assert _valid_sets(got, pad_valid) == _upstream_sets(upstream), name
  finally:
    data.close()


def _singleton_groups(length: int) -> np.ndarray:
  return sample_host.build_tie_groups_np((), length, length)


def _scheduled_order(
  chain_m: np.ndarray,
  chain_m_pos: np.ndarray,
  present: np.ndarray,
  randn: np.ndarray,
) -> np.ndarray:
  length = int(chain_m.shape[0])
  _groups, _size, _rank, order = potts_decode.schedule_groups(
    jnp.asarray(_singleton_groups(length)),
    jnp.ones((length,), dtype=bool),
    jnp.asarray(randn),
    jnp.asarray(chain_m),
    jnp.asarray(chain_m_pos),
    jnp.asarray(present),
    None,
  )
  return np.asarray(order)


def test_knob_semantics_order_generation(oracle: object, monkeypatch: pytest.MonkeyPatch) -> None:
  """AR and refine orders match upstream on one randn, and the two key mutants do not.

  The fixture has a fixed present row and a gap row. The AR key multiplies
  ``chain_M_pos`` and ``present``; the refine key is ``chain_mask`` alone.
  Dropping ``chain_M_pos`` promotes the fixed row from early to late, and
  swapping the keys moves the gap row, so each mutant disagrees with the
  oracle it was compared against. A fixture without those rows would leave
  both mutants equal to the oracle.
  """
  data = open_dump(oracle, "f64")
  try:
    randn = _vector(data, "knob_randn")
    chain_m = _vector(data, "knob_chain_M")
    chain_m_pos = _vector(data, "knob_chain_M_pos")
    present = _vector(data, "knob_mask")
    ar_oracle = _vector(data, "knob_ar_order").astype(np.int64)
    refine_oracle = _vector(data, "knob_refine_order").astype(np.int64)
    assert bool(np.any(present == 0))
    assert bool(np.any((chain_m_pos == 0) & (present == 1)))
    with x64_context("f64"):
      ar_order = _scheduled_order(chain_m, chain_m_pos, present, randn)
      refine_order = np.asarray(sample_host.fresh_refine_order(chain_m, randn))
      assert np.array_equal(ar_order, ar_oracle)
      assert np.array_equal(refine_order, refine_oracle)

      original = potts_decode.schedule_groups

      def drop_chain_m_pos(
        tie_groups: jax.Array,
        pad_valid: jax.Array,
        randn_arg: jax.Array,
        chain_mask: jax.Array,
        chain_m_pos_arg: jax.Array,
        present_arg: jax.Array,
        decoding_order: jax.Array | None,
      ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        del chain_m_pos_arg
        return original(
          tie_groups,
          pad_valid,
          randn_arg,
          chain_mask,
          jnp.ones_like(chain_mask),
          present_arg,
          decoding_order,
        )

      monkeypatch.setattr(potts_decode, "schedule_groups", drop_chain_m_pos)
      dropped = _scheduled_order(chain_m, chain_m_pos, present, randn)
      assert not np.array_equal(dropped, ar_oracle)
      monkeypatch.undo()

      def refine_key_as_ar(chain_mask: np.ndarray, randn_arg: np.ndarray) -> np.ndarray:
        key = (chain_mask.astype(np.float64) * chain_m_pos * present + 1.0e-4) * np.abs(
          randn_arg.astype(np.float64),
        )
        return np.argsort(key, kind="stable").astype(np.int32)

      def ar_key_as_refine(
        tie_groups: jax.Array,
        pad_valid: jax.Array,
        randn_arg: jax.Array,
        chain_mask: jax.Array,
        chain_m_pos_arg: jax.Array,
        present_arg: jax.Array,
        decoding_order: jax.Array | None,
      ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        del chain_m_pos_arg, present_arg
        ones = jnp.ones_like(chain_mask)
        return original(
          tie_groups,
          pad_valid,
          randn_arg,
          chain_mask,
          ones,
          ones,
          decoding_order,
        )

      monkeypatch.setattr(sample_host, "fresh_refine_order", refine_key_as_ar)
      monkeypatch.setattr(potts_decode, "schedule_groups", ar_key_as_refine)
      swapped_refine = np.asarray(sample_host.fresh_refine_order(chain_m, randn))
      swapped_ar = _scheduled_order(chain_m, chain_m_pos, present, randn)
      assert not np.array_equal(swapped_refine, refine_oracle)
      assert not np.array_equal(swapped_ar, ar_oracle)
  finally:
    data.close()


def _group_rank(order: np.ndarray, group: np.ndarray) -> list[int]:
  rank = np.empty(order.shape[0], dtype=np.int64)
  rank[order.astype(np.int64)] = np.arange(order.shape[0])
  return [int(rank[index]) for index in group.tolist()]


def test_tied_rank_flat_matches_upstream(oracle: object, monkeypatch: pytest.MonkeyPatch) -> None:
  """Grouped ``rank_flat`` matches the flattened oracle order; the raw rank does not.

  The dumped group is listed in an order that is not its raw-rank order, and a
  residue outside the group has a raw rank strictly between two members. Upstream
  emits the whole group in listing order when the first member is visited, so
  ``rank_flat`` and ``mask_attend`` move that non-member. Scheduling from the
  raw rank leaves the non-member where the key put it, and that comparison fails.
  Overlapping groups are rejected before any oracle is read.
  """
  with pytest.raises(ValueError, match="overlapping tied groups"):
    sample_host.build_tie_groups_np(((0, 1), (1, 2)), 4, 4)
  data = open_dump(oracle, "f64")
  try:
    group = np.asarray(data["tied_group"]).astype(np.int64)
    assert group.shape[0] >= 2
    randn_rows = np.asarray(data["tied_randn"])
    orders = np.asarray(data["tied_decoding_order"])
    backward = np.asarray(data["tied_order_mask_backward"])
    e_idx = np.asarray(data["tied_E_idx"])
    chain_m = _vector(data, "tied_chain_M")
    chain_m_pos = _vector(data, "tied_chain_M_pos")
    present = _vector(data, "tied_mask")
    assert randn_rows.shape[0] == 3
    length = int(chain_m.shape[0])
    tie_groups = sample_host.build_tie_groups_np((tuple(int(v) for v in group),), length, length)
    with x64_context("f64"):
      original = potts_decode.schedule_groups
      for seed in range(3):
        randn = randn_rows[seed]
        oracle_order = orders[seed].astype(np.int64)
        _groups, _size, rank_flat, raw_order = original(
          jnp.asarray(tie_groups),
          jnp.ones((length,), dtype=bool),
          jnp.asarray(randn),
          jnp.asarray(chain_m),
          jnp.asarray(chain_m_pos),
          jnp.asarray(present),
          None,
        )
        rank_flat_np = np.asarray(rank_flat)
        assert np.array_equal(rank_flat_np[:length], np.argsort(oracle_order))
        attend = rank_flat_np[e_idx] < rank_flat_np[:, None]
        gathered = backward[seed][np.arange(length)[:, None], e_idx]
        assert np.array_equal(attend, gathered > 0)
        raw = np.asarray(raw_order).astype(np.int64)
        member_ranks = _group_rank(raw, group)
        listed = [int(index) for index in group.tolist()]
        by_rank = sorted(listed, key=lambda index: int(np.argsort(raw)[index]))
        assert listed != by_rank
        lo, hi = min(member_ranks), max(member_ranks)
        members = set(listed)
        between = [
          index
          for index in range(length)
          if index not in members and lo < int(np.argsort(raw)[index]) < hi
        ]
        assert between

        def raw_rank_only(
          tie_groups_arg: jax.Array,
          pad_valid: jax.Array,
          randn_arg: jax.Array,
          chain_mask: jax.Array,
          chain_m_pos_arg: jax.Array,
          present_arg: jax.Array,
          decoding_order: jax.Array | None,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
          group_order, size, _rank_flat, order = original(
            tie_groups_arg,
            pad_valid,
            randn_arg,
            chain_mask,
            chain_m_pos_arg,
            present_arg,
            decoding_order,
          )
          return group_order, size, jnp.argsort(order).astype(jnp.int32), order

        monkeypatch.setattr(potts_decode, "schedule_groups", raw_rank_only)
        _g, _s, raw_rank, _order = potts_decode.schedule_groups(
          jnp.asarray(tie_groups),
          jnp.ones((length,), dtype=bool),
          jnp.asarray(randn),
          jnp.asarray(chain_m),
          jnp.asarray(chain_m_pos),
          jnp.asarray(present),
          None,
        )
        assert not np.array_equal(np.asarray(raw_rank)[:length], np.argsort(oracle_order))
        monkeypatch.undo()
  finally:
    data.close()
