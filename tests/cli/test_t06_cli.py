"""T0.6 CLI flags: --output-kind, options JSON, unset sampling temperature."""

from __future__ import annotations

import inspect
import json

from typer.testing import CliRunner

from aminx.cli import _RunBase, _base_spec_kwargs, app, run_sample, spec_emit_sample
from aminx.run.options import PottsMPNNOptions

runner = CliRunner()


def test_sampling_temperature_flag_defaults_to_unset() -> None:
    assert inspect.signature(run_sample).parameters["temperature"].default is None
    assert inspect.signature(spec_emit_sample).parameters["temperature"].default is None


def test_base_spec_kwargs_parses_options_json() -> None:
    base = _RunBase(
        topology=None,
        model_weights="original",
        model_version="v_48_020",
        model_family="pottsmpnn",
        checkpoint_id="pottsmpnn_cli",
        model_local_path=None,
        checkpoint_registry_path=None,
        ligand_mpnn_use_side_chain_context=None,
        batch_size=None,
        backbone_noise="0.0",
        backbone_noise_mode="direct",
        estat_noise=None,
        estat_noise_mode="direct",
        vdw_noise=None,
        vdw_noise_mode="direct",
        use_electrostatics=False,
        use_vdw=False,
        random_seed=42,
        chain_id=None,
        model=None,
        altloc="first",
        cache_path=None,
        output_dir=None,
        max_buffer_size=None,
        overwrite_cache=False,
        max_length=512,
        length_bucketing=True,
        truncation_strategy="none",
        host_resource_allocation_strategy="auto",
        ram_budget_mb=None,
        max_workers=None,
        n_devices=None,
        use_preprocessed=False,
        preprocessed_index_path=None,
        split="inference",
        tied_position=[],
        pass_mode="intra",
        multi_state_temperature=1.0,
        input_type="auto",
        input_cache_dir=None,
        potts_options_json=json.dumps({"optimization_mode": "nodes", "mean_norm": True}),
        laser_options_json=None,
    )
    kwargs = _base_spec_kwargs(base)
    assert kwargs["potts_mpnn"] == PottsMPNNOptions(optimization_mode="nodes", mean_norm=True)
    assert kwargs["laser"] is None


def test_run_sample_omitted_temperature_uses_mpnn_default() -> None:
    result = runner.invoke(app, ["run", "sample", "--inputs", "t.pdb", "--emit-json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["temperature"] == [0.1]


def test_run_sample_explicit_temperature_and_laser_default() -> None:
    explicit = runner.invoke(
        app,
        ["run", "sample", "--inputs", "t.pdb", "--temperature", "0.25", "--emit-json"],
    )
    assert explicit.exit_code == 0, explicit.output
    assert json.loads(explicit.output)["temperature"] == [0.25]

    laser = runner.invoke(
        app,
        [
            "run",
            "--checkpoint-id",
            "lasermpnn_cli",
            "sample",
            "--inputs",
            "t.pdb",
            "--emit-json",
        ],
    )
    assert laser.exit_code == 0, laser.output
    assert json.loads(laser.output)["temperature"] == [None]


def test_run_score_output_kind_and_potts_options() -> None:
    ok = runner.invoke(
        app,
        [
            "run",
            "--checkpoint-id",
            "pottsmpnn_cli",
            "--potts-options-json",
            json.dumps({"optimization_mode": "nodes"}),
            "score",
            "--inputs",
            "t.pdb",
            "--output-kind",
            "energy",
            "--emit-json",
        ],
    )
    assert ok.exit_code == 0, ok.output
    payload = json.loads(ok.output)
    assert payload["output_kind"] == "energy"
    assert payload["potts_mpnn"]["optimization_mode"] == "nodes"
    assert payload["sequences_to_score"] == []

    unknown = runner.invoke(
        app,
        [
            "run",
            "--checkpoint-id",
            "pottsmpnn_cli",
            "--potts-options-json",
            json.dumps({"not_a_field": 1}),
            "score",
            "--inputs",
            "t.pdb",
            "--output-kind",
            "energy",
            "--emit-json",
        ],
    )
    assert unknown.exit_code != 0

    missing_sequences = runner.invoke(
        app,
        ["run", "score", "--inputs", "t.pdb", "--emit-json"],
    )
    assert missing_sequences.exit_code != 0
