"""T0.6 spec surface: output_kind, family options, omit-AA, UNSET temperature."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace

import pytest

from aminx.host.spec_partition import CAMPAIGN_OWNED_KEYS, campaign_sampling_spec_payload
from aminx.run.options import LaserOptions, PottsMPNNOptions
from aminx.run.spec import UNSET, mpnn_temperatures
from aminx.run.spec_json import (
    SpecJSONDecodeError,
    run_specification_from_json,
    run_specification_from_json_dict,
    run_specification_to_json,
    run_specification_to_json_dict,
)
from aminx.run.specs import SamplingSpecification, ScoringSpecification, build_run_spec


def test_unset_never_reaches_build_run_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []
    real = build_run_spec

    def spy(spec: object) -> object:
        seen.append(getattr(spec, "temperature", None))
        return real(spec)

    monkeypatch.setattr("aminx.run.specs.build_run_spec", spy)
    spec = SamplingSpecification(inputs="a.pdb")
    assert seen
    assert all(value is not UNSET for value in seen)
    assert spec.temperature == (0.1,)
    assert spec.run_spec.sampling.temperature == (0.1,)


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        ("proteinmpnn", (0.1,)),
        ("ligandmpnn", (0.1,)),
        ("membrane", (0.1,)),
        ("pottsmpnn", (0.1,)),
        ("lasermpnn", (None,)),
    ],
)
def test_temperature_family_defaults(family: str, expected: tuple[float | None, ...]) -> None:
    spec = SamplingSpecification(inputs="a.pdb", model_family=family)  # type: ignore[arg-type]
    assert spec.temperature == expected
    assert spec.run_spec.sampling.temperature == expected


def test_checkpoint_prefix_resolves_before_temperature_default() -> None:
    laser = SamplingSpecification(inputs="a.pdb", checkpoint_id="lasermpnn_custom")
    assert laser.model_family == "lasermpnn"
    assert laser.temperature == (None,)
    potts = SamplingSpecification(inputs="a.pdb", checkpoint_id="pottsmpnn_custom")
    assert potts.model_family == "pottsmpnn"
    assert potts.temperature == (0.1,)


def test_none_temperature_on_mpnn_raises() -> None:
    with pytest.raises(ValueError, match="lasermpnn"):
        SamplingSpecification(inputs="a.pdb", temperature=None)
    with pytest.raises(ValueError, match="lasermpnn"):
        SamplingSpecification(inputs="a.pdb", model_family="proteinmpnn", temperature=(None,))


def test_lasermpnn_zero_temperature_is_none() -> None:
    scalar = SamplingSpecification(inputs="a.pdb", model_family="lasermpnn", temperature=0.0)
    assert scalar.temperature == (None,)
    mixed = SamplingSpecification(
        inputs="a.pdb",
        model_family="lasermpnn",
        temperature=(0.0, 0.3),
    )
    assert mixed.temperature == (None, 0.3)


def test_replace_keeps_concrete_temperature() -> None:
    spec = SamplingSpecification(inputs="a.pdb", model_family="proteinmpnn", temperature=0.4)
    moved = replace(spec, model_family="lasermpnn")
    assert moved.temperature == (0.4,)
    assert moved.run_spec.sampling.temperature == (0.4,)


def test_replace_unset_reresolves_family_default() -> None:
    spec = SamplingSpecification(inputs="a.pdb", model_family="proteinmpnn")
    moved = replace(spec, model_family="lasermpnn", temperature=UNSET)
    assert moved.temperature == (None,)
    back = replace(moved, model_family="pottsmpnn", temperature=UNSET)
    assert back.temperature == (0.1,)


def test_mpnn_temperatures_rejects_none() -> None:
    spec = SamplingSpecification(inputs="a.pdb", model_family="lasermpnn")
    with pytest.raises(ValueError, match="argmax"):
        mpnn_temperatures(spec.run_spec)
    concrete = SamplingSpecification(inputs="a.pdb")
    assert mpnn_temperatures(concrete.run_spec) == (0.1,)


def test_unset_pickle_round_trip_is_singleton() -> None:
    assert pickle.loads(pickle.dumps(UNSET)) is UNSET


def test_null_temperature_json_round_trip() -> None:
    spec = SamplingSpecification(inputs="a.pdb", model_family="lasermpnn", temperature=None)
    payload = run_specification_to_json_dict(spec)
    assert payload["temperature"] == [None]
    text = run_specification_to_json(spec)
    assert json.loads(text)["temperature"] == [None]
    restored = run_specification_from_json(text)
    assert isinstance(restored, SamplingSpecification)
    assert restored.temperature == (None,)
    assert restored.model_family == "lasermpnn"


def test_options_mismatch_family_raises() -> None:
    with pytest.raises(ValueError, match="potts_mpnn"):
        SamplingSpecification(inputs="a.pdb", potts_mpnn=PottsMPNNOptions())
    with pytest.raises(ValueError, match="laser"):
        SamplingSpecification(inputs="a.pdb", model_family="pottsmpnn", laser=LaserOptions())


def test_options_thread_onto_run_spec() -> None:
    potts = PottsMPNNOptions(optimization_mode="nodes", skip_gaps=True)
    laser = LaserOptions(fs_sequence_temp=0.2, disabled_residues=("A",))
    potts_spec = SamplingSpecification(
        inputs="a.pdb",
        model_family="pottsmpnn",
        potts_mpnn=potts,
        omit_aa=("C", "W"),
        omit_aa_per_position={4: "Y"},
    )
    assert potts_spec.run_spec.potts_mpnn == potts
    assert potts_spec.run_spec.laser is None
    assert potts_spec.run_spec.sampling.omit_aa == ("C", "W")
    assert potts_spec.run_spec.sampling.omit_aa_per_position == {4: "Y"}
    laser_spec = SamplingSpecification(inputs="a.pdb", model_family="lasermpnn", laser=laser)
    assert laser_spec.run_spec.laser == laser
    assert laser_spec.run_spec.potts_mpnn is None


def test_spec_json_options_round_trip_and_unknown_key() -> None:
    potts = PottsMPNNOptions(
        optimization_mode="none",
        binding_energy_json="bind.json",
        emit_dense_hJ=True,
    )
    spec = SamplingSpecification(
        inputs="a.pdb",
        checkpoint_id="pottsmpnn_round",
        potts_mpnn=potts,
        omit_aa=("C",),
        omit_aa_per_position={3: "W"},
    )
    payload = run_specification_to_json_dict(spec)
    assert payload["potts_mpnn"]["optimization_mode"] == "none"
    assert payload["potts_mpnn"]["emit_dense_hJ"] is True
    assert payload["laser"] is None
    restored = run_specification_from_json_dict(payload)
    assert isinstance(restored, SamplingSpecification)
    assert restored.potts_mpnn == potts
    assert restored.omit_aa == ("C",)
    assert restored.omit_aa_per_position == {3: "W"}
    assert restored.checkpoint_id == "pottsmpnn_round"

    laser = LaserOptions(disabled_residues=("G",), chi_temp=0.5)
    laser_spec = SamplingSpecification(
        inputs="b.pdb",
        checkpoint_id="lasermpnn_round",
        laser=laser,
    )
    laser_restored = run_specification_from_json_dict(run_specification_to_json_dict(laser_spec))
    assert isinstance(laser_restored, SamplingSpecification)
    assert laser_restored.laser == laser
    assert laser_restored.model_family == "lasermpnn"

    bad = run_specification_to_json_dict(spec)
    bad["potts_mpnn"]["not_a_field"] = 1
    with pytest.raises(SpecJSONDecodeError, match="not_a_field"):
        run_specification_from_json_dict(bad)


def _campaign_owned(**overrides: object) -> dict[str, object]:
    owned: dict[str, object] = {
        "grid_mode": False,
        "samples_chunk_size": None,
        "job_id": None,
        "chunk_id": None,
        "sample_start": None,
        "sample_count": None,
        "output_h5_path": "out.zarr",
    }
    owned.update(overrides)
    assert set(owned) == CAMPAIGN_OWNED_KEYS
    return owned


def test_campaign_round_trip_potts_and_laser_ids() -> None:
    potts = SamplingSpecification(
        inputs="a.pdb",
        checkpoint_id="pottsmpnn_campaign",
        potts_mpnn=PottsMPNNOptions(mean_norm=True, pssm_multi=0.5),
        campaign_mode=True,
        return_logits=False,
        omit_aa=("C",),
    )
    payload = campaign_sampling_spec_payload(potts, campaign_owned=_campaign_owned())
    from aminx.run.spec_json import _coerce_field_value

    coerced = {
        key: _coerce_field_value(SamplingSpecification, key, value) for key, value in payload.items()
    }
    restored = SamplingSpecification(**coerced)
    assert restored.checkpoint_id == "pottsmpnn_campaign"
    assert restored.model_family == "pottsmpnn"
    assert restored.potts_mpnn == potts.potts_mpnn
    assert restored.omit_aa == ("C",)
    assert restored.temperature == (0.1,)

    laser = SamplingSpecification(
        inputs="b.pdb",
        checkpoint_id="lasermpnn_campaign",
        laser=LaserOptions(ala_budget=2, proofread_dropout=False),
        campaign_mode=True,
        return_logits=False,
    )
    laser_payload = campaign_sampling_spec_payload(laser, campaign_owned=_campaign_owned())
    assert laser_payload["temperature"] == [None]
    laser_coerced = {
        key: _coerce_field_value(SamplingSpecification, key, value)
        for key, value in laser_payload.items()
    }
    laser_restored = SamplingSpecification(**laser_coerced)
    assert laser_restored.checkpoint_id == "lasermpnn_campaign"
    assert laser_restored.laser == laser.laser
    assert laser_restored.temperature == (None,)


def test_output_kind_rules() -> None:
    with pytest.raises(ValueError, match="sequences"):
        ScoringSpecification(inputs="a.pdb", output_kind="nll")
    with pytest.raises(ValueError, match="output_kind"):
        ScoringSpecification(
            inputs="a.pdb",
            sequences_to_score=["ACDE"],
            output_kind="logits",
        )
    with pytest.raises(ValueError, match="output_kind"):
        ScoringSpecification(
            inputs="a.pdb",
            model_family="proteinmpnn",
            sequences_to_score=["ACDE"],
            output_kind="energy",
        )
    energy = ScoringSpecification(
        inputs="a.pdb",
        model_family="pottsmpnn",
        output_kind="energy",
    )
    assert energy.output_kind == "energy"
    assert energy.sequences_to_score == ()
    logits = ScoringSpecification(
        inputs="a.pdb",
        model_family="pottsmpnn",
        sequences_to_score=["ACDE"],
        output_kind="logits",
    )
    assert logits.output_kind == "logits"
    proof = ScoringSpecification(
        inputs="a.pdb",
        model_family="lasermpnn",
        output_kind="proofread_conditional",
    )
    assert proof.output_kind == "proofread_conditional"
    with pytest.raises(ValueError, match="output_kind"):
        ScoringSpecification(
            inputs="a.pdb",
            model_family="lasermpnn",
            output_kind="ddg",
        )
    with pytest.raises(ValueError, match="sequences"):
        ScoringSpecification(
            inputs="a.pdb",
            model_family="lasermpnn",
            output_kind="logits",
        )
