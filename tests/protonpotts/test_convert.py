"""P6: the foundry -> aminx layout permutations, on synthetic arrays where every entry carries its identity.

These pin the PERMUTATION MATH only. That the permutations are the right ones for the real checkpoint is
settled by the ``protonpotts_encoder`` comparison against upstream (a wrong one is shape-valid and
silent), not by anything in this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from aminx.families.protonpotts_mpnn.convert import (
  EDGE_EMBEDDING_SHAPE,
  EXPECTED_SHAPES,
  RENAMES,
  V,
  pair_permutation,
  permute_edge_embedding,
  permute_etab,
  permute_token_rows,
  to_aminx_layout,
  token_permutation,
)
from aminx.families.protonpotts_mpnn.vocab import index_token, upstream_to_aminx_index
from aminx.utils.radial_basis import BACKBONE_PAIRS

ATOMS = ("N", "Ca", "C", "O", "Cb")


def test_pair_permutation_is_a_permutation_of_the_25_foundry_blocks() -> None:
  assert sorted(pair_permutation().tolist()) == list(range(25))


def test_pair_permutation_reads_the_foundry_row_major_block_for_each_aminx_pair() -> None:
  perm = pair_permutation()
  for slot, (i, j) in enumerate(np.asarray(BACKBONE_PAIRS).tolist()):
    assert perm[slot] == 5 * i + j


@pytest.mark.parametrize(
  ("slot", "pair"), [(0, "Ca-Ca"), (1, "N-N"), (5, "Ca-N"), (15, "N-Ca"), (24, "C-O")]
)
def test_known_legacy_slots_hold_the_named_pair(slot: int, pair: str) -> None:
  """aminx's slots are ProteinMPNN's legacy list (foundry weights.py legacy_order): spot-check by name."""
  first, second = pair.split("-")
  i, j = ATOMS.index(first), ATOMS.index(second)
  assert tuple(np.asarray(BACKBONE_PAIRS)[slot].tolist()) == (i, j)
  assert pair_permutation()[slot] == 5 * i + j


def test_edge_embedding_blocks_move_and_positional_columns_do_not() -> None:
  weight = np.zeros(EDGE_EMBEDDING_SHAPE, dtype=np.float32)
  weight[:, :16] = -1.0  # positional columns
  for block in range(25):  # foundry block b filled with b
    weight[:, 16 + block * 16 : 16 + (block + 1) * 16] = float(block)
  out = permute_edge_embedding(weight)
  assert (out[:, :16] == -1.0).all()
  for slot in range(25):
    assert (out[:, 16 + slot * 16 : 16 + (slot + 1) * 16] == pair_permutation()[slot]).all()
  assert out.shape == weight.shape


def test_edge_embedding_keeps_the_16_rbf_columns_in_order_within_a_block() -> None:
  weight = np.arange(EDGE_EMBEDDING_SHAPE[0] * EDGE_EMBEDDING_SHAPE[1], dtype=np.float64).reshape(
    EDGE_EMBEDDING_SHAPE
  )
  out = permute_edge_embedding(weight)
  slot = 7
  src = int(pair_permutation()[slot])
  assert np.array_equal(
    out[:, 16 + slot * 16 : 16 + (slot + 1) * 16], weight[:, 16 + src * 16 : 16 + (src + 1) * 16]
  )


def test_token_permutation_is_the_inverse_of_the_upstream_to_aminx_map() -> None:
  take = token_permutation()
  assert sorted(take.tolist()) == list(range(V))
  for upstream in range(V):
    assert take[upstream_to_aminx_index(upstream)] == upstream


def test_token_rows_land_at_their_aminx_index() -> None:
  rows = np.arange(V, dtype=np.float64)[:, None] * np.ones((1, 4))  # row u holds u
  out = permute_token_rows(rows)
  for upstream in range(V):
    assert (out[upstream_to_aminx_index(upstream)] == upstream).all()
  # foundry index 3 is ASP and aminx index 2 is D: the row for ASP moved from 3 to 2
  assert index_token(2) == "D"
  assert (out[2] == 3).all()


def test_token_rows_work_on_a_vector_and_reject_a_wrong_width() -> None:
  assert permute_token_rows(np.arange(V)).shape == (V,)
  with pytest.raises(ValueError, match="30 token rows"):
    permute_token_rows(np.zeros((21, 4)))


def test_etab_permutes_both_token_axes_of_the_pair_table() -> None:
  a, b = np.meshgrid(np.arange(V), np.arange(V), indexing="ij")
  flat = (a * 1000 + b).reshape(V * V, 1).astype(np.float64)  # entry (a, b) holds 1000a + b
  out = permute_etab(flat).reshape(V, V)
  for ua in range(V):
    for ub in range(V):
      assert out[upstream_to_aminx_index(ua), upstream_to_aminx_index(ub)] == ua * 1000 + ub


def test_etab_bias_is_permuted_like_the_table() -> None:
  a, b = np.meshgrid(np.arange(V), np.arange(V), indexing="ij")
  bias = (a * 1000 + b).reshape(-1).astype(np.float64)
  out = permute_etab(bias).reshape(V, V)
  assert out[upstream_to_aminx_index(3), upstream_to_aminx_index(6)] == 3 * 1000 + 6


def _state() -> dict[str, np.ndarray]:
  rng = np.random.default_rng(0)
  state = {key: rng.normal(size=shape).astype(np.float32) for key, shape in EXPECTED_SHAPES.items()}
  state["encoder_layers.0.W1.weight"] = rng.normal(size=(128, 384)).astype(np.float32)  # passes through
  return state


def test_to_aminx_layout_renames_permutes_and_passes_the_rest_through() -> None:
  state = _state()
  out = to_aminx_layout(state)
  for old, new in RENAMES.items():
    assert new in out
    assert old not in out
  assert np.array_equal(out["encoder_layers.0.W1.weight"], state["encoder_layers.0.W1.weight"])
  assert np.array_equal(out["W_s.weight"], permute_token_rows(state["W_s.weight"]))
  assert np.array_equal(
    out["features.edge_embedding.weight"],
    permute_edge_embedding(state["graph_featurization_module.edge_embedding.weight"]),
  )
  assert np.array_equal(out["etab_out.weight"], permute_etab(state["etab_out.weight"]))
  assert np.array_equal(out["features.embeddings.linear.weight"],
                        state["graph_featurization_module.positional_embedding.embed_positional_features.weight"])


@pytest.mark.parametrize("key", ["W_s.weight", "etab_out.weight", "W_out.bias"])
def test_a_wrong_shape_is_rejected_before_any_permutation(key: str) -> None:
  state = _state()
  state[key] = np.zeros((21, *state[key].shape[1:]), dtype=np.float32)  # the 21-wide Potts shape
  with pytest.raises(ValueError, match="expected"):
    to_aminx_layout(state)


def test_a_missing_key_is_rejected() -> None:
  state = _state()
  del state["W_s.weight"]
  with pytest.raises(KeyError, match="W_s.weight"):
    to_aminx_layout(state)
