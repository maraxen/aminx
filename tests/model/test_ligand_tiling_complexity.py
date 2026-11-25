"""Ligand tiling loop complexity (debt #1981).

``map_chunks_axis0`` walks L in fixed-size slabs. Per-iteration work must not
grow with L (criterion A). Work is a jaxpr proxy, not wall time: a fused kernel
can change real cost without changing this trace.

The must-fire mutant closes over the whole ``y`` and re-embeds it in every slab,
the same closure the graded probe measured.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from aminx.model.ligand_features import ProteinFeaturesLigand
from aminx.model.ligand_tiling import map_chunks_axis0, map_chunks_axis0_multi
from tests.helpers.loop_scaling import (
  A_MUTANT_MIN,
  A_THRESHOLD,
  LoopRow,
  assert_work_independent,
  measure_rows,
  pick_outermost_while,
  work_ratios,
)

_SENTINEL = {"tiling": 0}


def _features() -> ProteinFeaturesLigand:
  return ProteinFeaturesLigand(
    node_features=128,
    edge_features=128,
    k_neighbors=8,
    ligand_l_chunk=-1,
    key=jax.random.PRNGKey(0),
  )


def _trace(extent: int, *, multi: bool, mutant: bool) -> tuple[Callable[..., Any], tuple[Any, ...]]:
  feat = _features()
  embed = feat._y_edges_coords_to_embed
  if multi:
    left = jnp.ones((extent, 2), dtype=jnp.float32)
    right = jnp.ones((extent, 5), dtype=jnp.float32)

    def synthetic(left_slab: jax.Array, right_slab: jax.Array) -> tuple[jax.Array, jax.Array]:
      return left_slab + jnp.float32(0.25), jnp.sin(right_slab)

    def traced_multi(left_arr: jax.Array, right_arr: jax.Array) -> tuple[jax.Array, jax.Array]:
      return map_chunks_axis0_multi(synthetic, 8, (left_arr, right_arr))

    return traced_multi, (left, right)

  y = jnp.ones((extent, 12, 3), dtype=jnp.float32)

  def clean(slab: jax.Array) -> jax.Array:
    return embed(slab)

  def bad(slab: jax.Array, full: jax.Array = y, embed_fn: Any = embed) -> jax.Array:
    _SENTINEL["tiling"] += 1
    return embed_fn(slab) + jnp.float32(0) * jnp.sum(embed_fn(full))

  fn = bad if mutant else clean

  def traced(y_arr: jax.Array) -> jax.Array:
    return map_chunks_axis0(y_arr, chunk_size=8, fn=fn)

  return traced, (y,)


def _rows(extents: tuple[int, ...], *, multi: bool, mutant: bool) -> list[LoopRow]:
  return measure_rows(lambda n: _trace(n, multi=multi, mutant=mutant), extents)


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("extents", [(32, 64, 128), (37, 74, 148)])
def test_tiling_work_is_independent_of_length(multi: bool, extents: tuple[int, ...]) -> None:
  """Clean slabs, including lengths that are not a multiple of the chunk size."""
  rows = _rows(extents, multi=multi, mutant=False)
  row = pick_outermost_while(rows)
  assert row.primitive == "while"
  assert_work_independent(rows)
  assert max(work_ratios(row)) < A_THRESHOLD


def test_tiling_full_input_closure_must_fire() -> None:
  """Re-embedding the whole ``y`` in every slab has to clear the A margin."""
  _SENTINEL["tiling"] = 0
  clean = _rows((32, 64, 128), multi=False, mutant=False)
  mutant = _rows((32, 64, 128), multi=False, mutant=True)
  assert _SENTINEL["tiling"] > 0
  clean_row = pick_outermost_while(clean)
  mutant_row = pick_outermost_while(mutant)
  clean_ratios = work_ratios(clean_row)
  mutant_ratios = work_ratios(mutant_row)
  assert max(clean_ratios) < A_THRESHOLD
  assert min(mutant_ratios) >= A_MUTANT_MIN
  assert min(mutant_ratios) > max(clean_ratios)
  with pytest.raises(AssertionError, match="work ratio"):
    assert_work_independent(mutant)
