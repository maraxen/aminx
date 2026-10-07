"""Plan topology validation runs on the inference-plan construction path."""

from __future__ import annotations

import equinox as eqx
import pytest

from aminx.host.plan import PlanTopologyError, make_inference_plan
from aminx.run.spec import build_run_spec
from aminx.types.stages import UnconditionalDecodeStep


class _DummyModel(eqx.Module):
    """No weights. Enough for make_inference_plan's straight-through wiring."""

    decoder: object = None
    w_s_embed: object = None


class _PlanSpec:
    """Duck spec the existing host plan tests already use."""

    sampling_strategy = "temperature"
    use_rolling_state = False
    multi_state_strategy = "arithmetic_mean"
    multi_state_temperature = 1.0
    state_weights = None
    temperature = (1.0,)
    average_node_features = False


def _spec_with_run_spec(cls: type[_PlanSpec]) -> _PlanSpec:
    cls.run_spec = build_run_spec(cls)
    return cls()


def test_default_sample_plan_builds() -> None:
    plan = make_inference_plan(_DummyModel(), _spec_with_run_spec(_PlanSpec), purpose="sample")
    assert plan.stage_set is not None
    assert plan.decode_fn is not None


def test_default_score_plan_builds() -> None:
    plan = make_inference_plan(_DummyModel(), _spec_with_run_spec(_PlanSpec), purpose="score")
    assert plan.stage_set is not None
    assert plan.decode_fn is not None


class _STESpec(_PlanSpec):
    sampling_strategy = "straight_through"


def _ste_model() -> _DummyModel:
    return _DummyModel(decoder=object(), w_s_embed=type("Embed", (), {"weight": None})())


def test_ste_with_unconditional_decode_step_raises_on_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """STEDecode + UnconditionalDecodeStep fails in make_inference_plan, not a direct validator call.

    make_inference_plan imports ConditionalDecodeStep lazily, so replacing it on
    aminx.types.stages makes the STE wiring install an UnconditionalDecodeStep.
    """

    def _unconditional(decoder: object, w_s_embed: object) -> UnconditionalDecodeStep:
        del w_s_embed
        return UnconditionalDecodeStep(decoder=decoder)

    monkeypatch.setattr("aminx.types.stages.ConditionalDecodeStep", _unconditional)
    with pytest.raises(PlanTopologyError, match="STEDecode"):
        make_inference_plan(_ste_model(), _spec_with_run_spec(_STESpec), purpose="sample")


def test_ste_with_conditional_decode_step_builds() -> None:
    """Control for the test above: the production STE wiring (ConditionalDecodeStep) passes validation."""
    plan = make_inference_plan(_ste_model(), _spec_with_run_spec(_STESpec), purpose="sample")
    assert plan.stage_set.decode_step is not None
