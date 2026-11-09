"""Regression tests for aminx #274: the sample-axis planner underestimated per-sample activation memory.

Measured on one production single-state MS-L row (TITAN RTX, L=256 padded, ProteinMPNN v_48,
``XLA_PYTHON_CLIENT_MEM_FRACTION=0.85``, ``PREALLOCATE=false``, nvidia-smi peak memory.used, tev_design
``.praxia/docs/research/261106_aminx-incremental-decode-batched-cond.md``):

    width  20 ->  2254 MiB      width  64 ->  8654 MiB
    width 128 -> 16846 MiB      width 256 -> RESOURCE_EXHAUSTED

i.e. ~145 MiB/sample between 20 and 64 and ~128 MiB/sample between 64 and 128. The planner's estimate was
``seq_len * (128 + 32*48 + 21) * 4 * 2.5`` = ~4.1 MiB/sample (15-35x low), so a full vmap was chosen at 20, 128 and 1024
samples alike and wide rows OOMed instead of being chunked. The budget itself was fine (it honours
``XLA_PYTHON_CLIENT_MEM_FRACTION``); the per-sample estimate was the defect.

These tests pin the safety property in MEASURED units, not in the planner's own units: a planned tile must never
exceed what the measured per-sample cost says fits the budget.
"""

import inspect

import pytest

from aminx.host import plan as plan_module
from aminx.host.plan import (
    AxisNames,
    _sampling_activation_bytes_per_element,
    decision_for,
    make_sampling_planner,
)
from aminx.run.specs import SamplingSpecification

MIB = 2**20
GIB = 2**30

SEQ_LEN = 256
# Highest measured per-sample cost (the 20 -> 64 slope); the 64 -> 128 slope was 128 MiB.
MEASURED_PER_SAMPLE_BYTES = 145.5 * MIB
MEASURED_PER_SAMPLE_BYTES_LOW = 128 * MIB
# aminx's own device budget at headroom 0.80 for XLA_PYTHON_CLIENT_MEM_FRACTION 0.30 and 0.85 (bytes_limit 7.04 / 19.94 GiB).
BUDGET_MEM_FRACTION_030 = 5.63 * GIB
BUDGET_MEM_FRACTION_085 = 15.95 * GIB


@pytest.fixture
def spec():
    return SamplingSpecification(inputs="test.pdb", model_family="proteinmpnn")


def _plan(spec, n_samples, budget_bytes, seq_len=SEQ_LEN):
    original = plan_module.resolve_memory_budget_bytes
    plan_module.resolve_memory_budget_bytes = lambda headroom=0.8: budget_bytes
    try:
        return make_sampling_planner(spec, n_samples_override=n_samples, seq_len=seq_len)
    finally:
        plan_module.resolve_memory_budget_bytes = original


def _tile(decision):
    strategy = decision.strategy
    tile = getattr(strategy, "tile", None)
    return tile if tile is not None else strategy.batch_size


def test_per_sample_estimate_is_calibrated_to_measurement():
    """The planner's per-sample byte estimate must cover the measured cost without being wildly conservative.

    Lower bound: the estimate must be at least the highest measured per-sample cost (never under-plan).
    Upper bound: within 1.3x of it, so a conservative estimate does not needlessly chunk narrow rows.
    """
    multiplier = inspect.signature(make_sampling_planner).parameters["activation_multiplier"].default
    per_sample = _sampling_activation_bytes_per_element(SEQ_LEN) * multiplier
    assert per_sample >= MEASURED_PER_SAMPLE_BYTES, (
        f"estimate {per_sample / MIB:.1f} MiB/sample is below the measured {MEASURED_PER_SAMPLE_BYTES / MIB:.1f}"
    )
    assert per_sample <= 1.3 * MEASURED_PER_SAMPLE_BYTES, (
        f"estimate {per_sample / MIB:.1f} MiB/sample is more than 1.3x the measured {MEASURED_PER_SAMPLE_BYTES / MIB:.1f}"
    )


def test_128_samples_within_mem_fraction_030_budget_is_chunked_not_full_vmap(spec):
    """The #274 repro: 128 samples (~16 GiB measured) cannot fit a 5.63 GiB budget, so must not be a full vmap."""
    decision = decision_for(_plan(spec, 128, BUDGET_MEM_FRACTION_030), AxisNames.N_SAMPLES)
    assert type(decision.strategy).__name__ == "ChunkedMap"
    tile = _tile(decision)
    assert 1 <= tile < 128
    assert tile * MEASURED_PER_SAMPLE_BYTES <= BUDGET_MEM_FRACTION_030, (
        f"tile {tile} x {MEASURED_PER_SAMPLE_BYTES / MIB:.1f} MiB exceeds the {BUDGET_MEM_FRACTION_030 / GIB:.2f} GiB budget"
    )


def test_20_sample_production_row_stays_a_full_vmap(spec):
    """The production row width (2.2 GiB measured) fits even the 0.30 budget and must not be slowed by chunking."""
    for budget in (BUDGET_MEM_FRACTION_030, BUDGET_MEM_FRACTION_085):
        decision = decision_for(_plan(spec, 20, budget), AxisNames.N_SAMPLES)
        assert type(decision.strategy).__name__ == "Vmap"


def test_mem_fraction_085_vmaps_64_but_chunks_128(spec):
    """At the 15.95 GiB budget: 64 samples (8.5 GiB measured) fit; 128 (16.5 GiB measured) do not."""
    assert type(decision_for(_plan(spec, 64, BUDGET_MEM_FRACTION_085), AxisNames.N_SAMPLES).strategy).__name__ == "Vmap"
    wide = decision_for(_plan(spec, 128, BUDGET_MEM_FRACTION_085), AxisNames.N_SAMPLES)
    assert type(wide.strategy).__name__ == "ChunkedMap"
    assert _tile(wide) * MEASURED_PER_SAMPLE_BYTES <= BUDGET_MEM_FRACTION_085


@pytest.mark.parametrize("n_samples", [256, 1024])
@pytest.mark.parametrize("budget", [BUDGET_MEM_FRACTION_030, BUDGET_MEM_FRACTION_085])
def test_very_wide_rows_never_plan_a_tile_beyond_the_measured_budget(spec, n_samples, budget):
    """The safety property, for every width that OOMed or would: tile * measured-per-sample <= budget."""
    decision = decision_for(_plan(spec, n_samples, budget), AxisNames.N_SAMPLES)
    assert type(decision.strategy).__name__ == "ChunkedMap"
    assert _tile(decision) * MEASURED_PER_SAMPLE_BYTES <= budget
