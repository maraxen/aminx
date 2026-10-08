# ruff: noqa: S101
"""PottsAlphabet threading through PottsHead and PottsMPNN.

The leaf-invariance test is the checkpoint-compatibility guarantee: the default
alphabet must produce exactly the array leaves that an explicit ``POTTS_MPNN``
produces, so existing serialised weights still load.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aminx.families.potts_mpnn.alphabet import POTTS_MPNN, PottsAlphabet
from aminx.families.potts_mpnn.model import PottsMPNN
from aminx.families.potts_mpnn.potts_head import PottsHead


def _array_leaf_shapes(model: eqx.Module) -> list[tuple[int, ...]]:
  leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
  return [tuple(leaf.shape) for leaf in leaves]


def test_default_alphabet_is_shipped_potts_mpnn() -> None:
  """PottsMPNN() defaults to POTTS_MPNN and a 400-wide pair projection."""
  model = PottsMPNN(key=jax.random.PRNGKey(0))
  assert model.alphabet is POTTS_MPNN
  assert model.potts_head.pair_side == 20
  assert model.potts_head.linear.weight.shape == (400, 128)


def test_leaf_invariance_default_matches_explicit_alphabet() -> None:
  """Default and explicit POTTS_MPNN give identical array-leaf shapes."""
  key = jax.random.PRNGKey(0)
  default_model = PottsMPNN(key=key)
  explicit_model = PottsMPNN(key=key, alphabet=POTTS_MPNN)

  default_shapes = _array_leaf_shapes(default_model)
  explicit_shapes = _array_leaf_shapes(explicit_model)
  assert len(default_shapes) == len(explicit_shapes)
  assert default_shapes == explicit_shapes

  # The alphabet and pair_side are static metadata, never array leaves.
  leaves = jax.tree_util.tree_leaves(eqx.filter(default_model, eqx.is_array))
  for leaf in leaves:
    assert leaf is not POTTS_MPNN
    assert not isinstance(leaf, (PottsAlphabet, int, str))


def test_custom_alphabet_threads_through_potts_head() -> None:
  """A 3-token alphabet yields a 3x3 table with slot-0 off-diagonals zeroed."""
  toy = PottsAlphabet(
    name="toy3",
    symbols=("A", "C", "G"),
    x_index=2,
    pair_side=3,
    etab_symbols=("A", "C", "G"),
    standard_symbols=("A", "C"),
  )
  head = PottsHead(8, key=jax.random.key(0), alphabet=toy)
  assert head.pair_side == 3
  assert head.linear.weight.shape == (9, 8)

  edge_features = jax.random.normal(jax.random.key(1), (2, 2, 8))
  e_idx = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
  present = jnp.ones(2)
  pad_valid = jnp.ones(2, dtype=bool)
  out = head(edge_features, e_idx, present, pad_valid)

  assert out.shape == (2, 2, 3, 3)
  off_diagonal = ~np.eye(3, dtype=bool)
  slot0 = np.asarray(out[:, 0])
  assert np.all(slot0[:, off_diagonal] == 0.0)
