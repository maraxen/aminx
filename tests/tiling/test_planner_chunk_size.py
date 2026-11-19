"""``plan_axis_strategy`` must never return a 0-size ChunkedMap tile.

xtrax's BatchPlanner falls back to ChunkedMap with the axis template's ``default_batch_size``,
which is 0 for the candidate axis. A 0-size tile estimates to 0 bytes, so it always "fits", and
then the dispatcher runs the whole axis at once: 448 (mask, sequence) pairs at ~165 MB each asked
a 24 GiB GPU for 21 GiB (bathos run 78f04b0b). The tile must be sized from the budget.

task_id: 261118_aminx-campaign-teacher-forced-scoring
"""

from __future__ import annotations

import pytest
from xtrax.tiling import ChunkedMap, Vmap

from aminx.tiling.axes import N_CANDIDATES
from aminx.tiling.memory_budget import MEMORY_BUDGET_ENV, resolve_memory_budget_bytes
from aminx.tiling.planner import plan_axis_strategy

_BUDGET = 4 * 1024**3
_PER_PAIR = 164_539_236  # the measured per-pair activation estimate at L=512


@pytest.fixture(autouse=True)
def _budget(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv(MEMORY_BUDGET_ENV, str(_BUDGET))


@pytest.mark.parametrize("cardinality", [30, 64, 448])
def test_over_budget_axis_gets_a_real_tile(cardinality: int) -> None:
  strategy = plan_axis_strategy(
    N_CANDIDATES, cardinality, None, activation_bytes_per_element=_PER_PAIR, size_unsized_tiles=True,
  )
  assert isinstance(strategy, ChunkedMap)
  assert strategy.batch_size >= 1
  # The same budget the planner resolves (an explicit env budget is used as given).
  assert strategy.batch_size * _PER_PAIR <= resolve_memory_budget_bytes(headroom=0.8)
  assert (strategy.batch_size + 1) * _PER_PAIR > resolve_memory_budget_bytes(headroom=0.8)


def test_default_keeps_the_legacy_dispatch() -> None:
  """Existing callers are unchanged: without the opt-in the 0-size fallback is returned as before."""
  strategy = plan_axis_strategy(N_CANDIDATES, 448, None, activation_bytes_per_element=_PER_PAIR)
  assert strategy == ChunkedMap(batch_size=0)


def test_axis_that_fits_stays_vmapped() -> None:
  assert isinstance(plan_axis_strategy(N_CANDIDATES, 7, None, activation_bytes_per_element=_PER_PAIR), Vmap)


def test_override_still_wins() -> None:
  strategy = plan_axis_strategy(N_CANDIDATES, 448, 5, activation_bytes_per_element=_PER_PAIR)
  assert strategy == ChunkedMap(batch_size=5)


def test_pair_tile_is_capped() -> None:
  """The score_pairs tile never exceeds _MAX_PAIR_TILE, whatever the planner allows (#3485)."""
  from aminx.host.runner import _MAX_PAIR_TILE, _pair_tile  # noqa: PLC0415

  assert _pair_tile(Vmap(), 7) is None
  assert _pair_tile(Vmap(), 448) == _MAX_PAIR_TILE
  assert _pair_tile(ChunkedMap(batch_size=186), 448) == _MAX_PAIR_TILE
  assert _pair_tile(ChunkedMap(batch_size=6), 448) == 6
  assert _pair_tile(ChunkedMap(batch_size=40), 20) is None
