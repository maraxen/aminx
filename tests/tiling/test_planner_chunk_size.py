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
from aminx.tiling.memory_budget import MEMORY_BUDGET_ENV
from aminx.tiling.planner import plan_axis_strategy

_BUDGET = 4 * 1024**3
_HEADROOM = 0.8
_PER_PAIR = 164_539_236  # the measured per-pair activation estimate at L=512


@pytest.fixture(autouse=True)
def _budget(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv(MEMORY_BUDGET_ENV, str(_BUDGET))


@pytest.mark.parametrize("cardinality", [30, 64, 448])
def test_over_budget_axis_gets_a_real_tile(cardinality: int) -> None:
  strategy = plan_axis_strategy(N_CANDIDATES, cardinality, None, activation_bytes_per_element=_PER_PAIR)
  assert isinstance(strategy, ChunkedMap)
  assert strategy.batch_size >= 1
  assert strategy.batch_size * _PER_PAIR <= _BUDGET * _HEADROOM


def test_axis_that_fits_stays_vmapped() -> None:
  assert isinstance(plan_axis_strategy(N_CANDIDATES, 7, None, activation_bytes_per_element=_PER_PAIR), Vmap)


def test_override_still_wins() -> None:
  strategy = plan_axis_strategy(N_CANDIDATES, 448, 5, activation_bytes_per_element=_PER_PAIR)
  assert strategy == ChunkedMap(batch_size=5)
