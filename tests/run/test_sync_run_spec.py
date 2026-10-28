"""Tests for _sync_run_spec guard flag to prevent double-fire on construction (#1906)."""

from unittest.mock import patch

from aminx.run.specs import (
    InspectionSpecification,
    JacobianSpecification,
    SamplingSpecification,
    ScoringSpecification,
)


def test_sync_run_spec_fires_once_sampling():
    """SamplingSpecification fires build_run_spec exactly once during construction.

    Verifies that the _run_spec_synced guard flag prevents double-firing:
    the parent RunSpecification.__post_init__ calls _sync_run_spec(),
    and the child SamplingSpecification.__post_init__ resets the flag
    and calls _sync_run_spec() again, but the guard ensures it fires only once.
    """
    with patch("aminx.run.specs.build_run_spec") as mock_build:
        mock_build.return_value = None
        SamplingSpecification(inputs=["dummy.pdb"], num_samples=4)
    assert mock_build.call_count == 1, f"Expected 1 call, got {mock_build.call_count}"


def test_sync_run_spec_fires_once_scoring():
    """ScoringSpecification fires build_run_spec exactly once during construction."""
    with patch("aminx.run.specs.build_run_spec") as mock_build:
        mock_build.return_value = None
        ScoringSpecification(inputs=["dummy.pdb"], sequences_to_score=["ACDEF"])
    assert mock_build.call_count == 1, f"Expected 1 call, got {mock_build.call_count}"


def test_sync_run_spec_fires_once_jacobian():
    """JacobianSpecification fires build_run_spec exactly once during construction."""
    with patch("aminx.run.specs.build_run_spec") as mock_build:
        mock_build.return_value = None
        JacobianSpecification(inputs=["dummy.pdb"])
    assert mock_build.call_count == 1, f"Expected 1 call, got {mock_build.call_count}"


def test_sync_run_spec_fires_once_inspection():
    """InspectionSpecification fires build_run_spec exactly once during construction."""
    with patch("aminx.run.specs.build_run_spec") as mock_build:
        mock_build.return_value = None
        InspectionSpecification(inputs=["dummy.pdb"])
    assert mock_build.call_count == 1, f"Expected 1 call, got {mock_build.call_count}"


def test_run_spec_populated_after_construction():
    """run_spec field is populated after construction for all spec types."""
    with patch("aminx.run.specs.build_run_spec") as mock_build:
        mock_spec = object()
        mock_build.return_value = mock_spec

        # Test each spec type
        for spec_cls in [
            SamplingSpecification,
            ScoringSpecification,
            JacobianSpecification,
            InspectionSpecification,
        ]:
            if spec_cls == ScoringSpecification:
                spec = spec_cls(inputs=["dummy.pdb"], sequences_to_score=["ACDEF"])
            else:
                spec = spec_cls(inputs=["dummy.pdb"])

            assert hasattr(spec, "run_spec"), f"{spec_cls.__name__} missing run_spec"
            assert (
                spec.run_spec is mock_spec
            ), f"{spec_cls.__name__} run_spec not set correctly"


def test_output_path_aliases_sync_and_replace():
    """output_path, output_zarr_path, and output_h5_path stay synchronized across init and dataclasses.replace, and reject conflicts."""
    import dataclasses
    import pytest

    s1 = SamplingSpecification(inputs=["a.pdb"], output_zarr_path="/tmp/out1.zarr")
    assert str(s1.output_path) == "/tmp/out1.zarr"
    assert str(s1.output_zarr_path) == "/tmp/out1.zarr"
    assert str(s1.output_h5_path) == "/tmp/out1.zarr"
    assert str(s1.run_spec.io.output_path) == "/tmp/out1.zarr"

    # dataclasses.replace overriding one alias updates all three
    s2 = dataclasses.replace(s1, output_h5_path="/tmp/out2.zarr")
    assert str(s2.output_h5_path) == "/tmp/out2.zarr"
    assert str(s2.output_path) == "/tmp/out2.zarr"
    assert str(s2.output_zarr_path) == "/tmp/out2.zarr"
    assert str(s2.run_spec.io.output_path) == "/tmp/out2.zarr"

    # Conflicting explicit aliases at construction raise ValueError
    with pytest.raises(ValueError, match="Conflicting output paths"):
        SamplingSpecification(
            inputs=["a.pdb"],
            output_path="/tmp/a.zarr",
            output_zarr_path="/tmp/b.zarr",
        )


def test_deterministic_run_id_content_sensitivity_and_output_path_invariance():
    """_compute_deterministic_run_id hashes array values, family options, and StringIO contents, and ignores output_path."""
    import io
    import numpy as np
    from aminx.run.options import LaserOptions, PottsMPNNOptions

    # Output path aliases must not alter deterministic run_id
    a = SamplingSpecification(inputs=["a.pdb"], output_path="/tmp/one.zarr")
    b = SamplingSpecification(inputs=["a.pdb"], output_path="/tmp/two.zarr")
    assert a.run_spec.run_id == b.run_spec.run_id

    # Different numpy array values with identical shape/dtype must produce distinct run_ids
    arr1 = SamplingSpecification(inputs=["a.pdb"], fixed_mask=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    arr2 = SamplingSpecification(inputs=["a.pdb"], fixed_mask=np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert arr1.run_spec.run_id != arr2.run_spec.run_id

    # Different family option fields must produce distinct run_ids
    opt1 = SamplingSpecification(
        inputs=["a.pdb"], model_family="lasermpnn", laser=LaserOptions(ala_budget=1)
    )
    opt2 = SamplingSpecification(
        inputs=["a.pdb"], model_family="lasermpnn", laser=LaserOptions(ala_budget=2)
    )
    assert opt1.run_spec.run_id != opt2.run_spec.run_id

    potts1 = SamplingSpecification(
        inputs=["a.pdb"], model_family="pottsmpnn", potts_mpnn=PottsMPNNOptions(optimization_mode="potts")
    )
    potts2 = SamplingSpecification(
        inputs=["a.pdb"], model_family="pottsmpnn", potts_mpnn=PottsMPNNOptions(optimization_mode="nodes")
    )
    assert potts1.run_spec.run_id != potts2.run_spec.run_id

    # Different StringIO contents must produce distinct run_ids and preserve cursor position
    buf1 = io.StringIO("ATOM 1\n")
    buf2 = io.StringIO("ATOM 2\n")
    s_io1 = SamplingSpecification(inputs=[buf1])
    s_io2 = SamplingSpecification(inputs=[buf2])
    assert s_io1.run_spec.run_id != s_io2.run_spec.run_id
    assert buf1.tell() == 0
