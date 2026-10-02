"""Regression test for the n_samples axis planner/runtime cardinality mismatch.

See .praxia/docs/specs/260706_samples-axis-planner-cardinality-mismatch.md and
Finding D of .praxia/docs/specs/260707_xtrax-migration-gap-audit-runspec-scaffolding.md.

samples_batch_size (a static SamplingSpecification default, 16) and the actual
per-call sample count (resolved via resolve_target_samples, independently from
samples_chunk_size/num_samples) used to feed the N_SAMPLES axis planner and the
real dispatched array size respectively -- with no cross-validation between
them. When the plan decided Vmap because the small samples_batch_size fit the
memory budget, that decision was silently applied to the real (possibly much
larger) array with no re-check. The fix: kernel_dispatch._sample_batch now
resolves the real per-call count first and passes it to make_sampling_planner
as n_samples_override, so the planner's decision is verified against the
array size that's actually dispatched.
"""

from aminx.host.plan import AxisNames, decision_for, make_sampling_planner
from aminx.run.specs import SamplingSpecification


def test_planner_cardinality_matches_override_not_samples_batch_size():
    """The N_SAMPLES axis is planned against n_samples_override, not samples_batch_size."""
    spec = SamplingSpecification(inputs="test.pdb", model_family="ligandmpnn")
    assert spec.samples_batch_size == 16

    plan = make_sampling_planner(spec, n_samples_override=500)
    decision = decision_for(plan, AxisNames.N_SAMPLES)

    assert decision.spec.cardinality == 500


def test_planner_without_override_falls_back_to_samples_batch_size():
    """Backward-compat: omitting the override preserves the pre-fix cardinality source."""
    spec = SamplingSpecification(inputs="test.pdb", model_family="ligandmpnn")

    plan = make_sampling_planner(spec)
    decision = decision_for(plan, AxisNames.N_SAMPLES)

    assert decision.spec.cardinality == spec.samples_batch_size


def test_planner_demotes_from_vmap_when_real_cardinality_exceeds_budget():
    """The core safety property: a large real sample count forces a memory-safe strategy.

    Reproduces the original empirical repro (260706 doc): with a small default
    samples_batch_size (16), the plan picked Vmap. Before the fix, that Vmap
    decision was then silently applied to a much larger real array. After the
    fix, the planner sees the real (large) count directly and, when it
    genuinely would not fit the memory budget, demotes to SafeMap instead of
    picking Vmap for a cardinality it was never actually verified against.
    """
    spec = SamplingSpecification(inputs="test.pdb", model_family="ligandmpnn")

    small_plan = make_sampling_planner(
        spec,
        n_samples_override=16,
        param_bytes=1e9,
        activation_multiplier=50.0,
    )
    small_decision = decision_for(small_plan, AxisNames.N_SAMPLES)
    assert type(small_decision.strategy).__name__ == "Vmap"

    large_plan = make_sampling_planner(
        spec,
        n_samples_override=50_000_000,
        param_bytes=1e9,
        activation_multiplier=50.0,
    )
    large_decision = decision_for(large_plan, AxisNames.N_SAMPLES)
    assert type(large_decision.strategy).__name__ == "ChunkedMap"
    assert large_decision.spec.cardinality == 50_000_000


def test_sample_batch_passes_real_target_num_samples_to_planner():
    """kernel_dispatch._sample_batch resolves target_num_samples before planning.

    Static check on call order/argument wiring: resolve_target_samples must
    run before make_sampling_planner, and its result must be threaded through
    as n_samples_override -- guards against a future edit silently reverting
    to the disconnected spec.samples_batch_size-only call.
    """
    import inspect

    from aminx.host import kernel_dispatch

    src = inspect.getsource(kernel_dispatch._sample_batch)
    resolve_idx = src.index("resolve_target_samples(")
    plan_idx = src.index("make_sampling_planner(")
    assert resolve_idx < plan_idx, (
        "resolve_target_samples must be called before make_sampling_planner "
        "so the real sample count is available to pass as n_samples_override"
    )
    assert "n_samples_override=target_num_samples" in src
