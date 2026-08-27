"""Tests for portable JSON serialization and guards (RS-8)."""

from __future__ import annotations

import pytest

from aminx.run import InspectionSpecification
import aminx.run
from aminx.run.run_spec_portable_json import (
    run_spec_portable_from_dict,
    run_spec_portable_to_dict,
)


class TestPortableJsonRoundtrip:
    """Test basic portable JSON serialization roundtrip."""

    @pytest.fixture
    def baseline_spec(self) -> dict:
        """Minimal valid v2 portable JSON for a plain proteinmpnn non-grid spec."""
        return {
            "version": 2,
            "io": {
                "sink_kind": "none",
                "output_dir": None,
                "manifest_path": None,
            },
            "multistate": {
                "mode": "single",
                "n_states": 1,
                "combine_strategy": "arithmetic_mean",
            },
            "resource": {
                "n_devices": 1,
                "sample_batch_size": 32,
                "structure_batch_size": 32,
                "max_buffer_size": None,
            },
            "precision": {"compute": "fp32"},
        }

    def test_portable_to_dict_roundtrips_plain_spec(
        self, baseline_spec: dict
    ) -> None:
        """Plain proteinmpnn/non-grid spec should roundtrip through portable JSON."""
        # Deserialize baseline to RunSpec
        baseline = run_spec_portable_from_dict(baseline_spec)

        # Serialize back to dict
        d = run_spec_portable_to_dict(baseline)

        # Verify version
        assert d["version"] == 2

        # Verify roundtrip: deserialize again and should succeed
        roundtripped = run_spec_portable_from_dict(d)
        assert roundtripped is not None
        assert roundtripped.multistate.mode == baseline_spec["multistate"]["mode"]
        assert roundtripped.precision.compute == baseline_spec["precision"]["compute"]


class TestPortableJsonGuards:
    """Test RS-8 guards that prevent silent data loss."""

    @pytest.fixture
    def baseline_spec(self) -> dict:
        """Minimal valid v2 portable JSON for a plain proteinmpnn non-grid spec."""
        return {
            "version": 2,
            "io": {
                "sink_kind": "none",
                "output_dir": None,
                "manifest_path": None,
            },
            "multistate": {
                "mode": "single",
                "n_states": 1,
                "combine_strategy": "arithmetic_mean",
            },
            "resource": {
                "n_devices": 1,
                "sample_batch_size": 32,
                "structure_batch_size": 32,
                "max_buffer_size": None,
            },
            "precision": {"compute": "fp32"},
        }

    def test_portable_from_dict_raises_on_grid_block(
        self, baseline_spec: dict
    ) -> None:
        """from_dict should raise ValueError on a payload carrying a 'grid' block.

        `RunSpec.grid`/`.ligand` no longer exist (deleted as dead scaffolding --
        see .praxia/docs/specs/260827_runspec-scaffolding-remediation-...md WS-A),
        so `to_dict` can no longer detect a grid/ligandmpnn run at all: the guard
        moves to the only place the information still exists, the wire payload
        itself. This test exercises a path production can actually reach --
        the previous `to_dict`-mutation tests could not, since `to_dict`'s one
        production caller is always fed by `from_dict`'s own (grid/ligand-free)
        output (see .praxia/docs/specs/260611_runspec-unification.md RS-8).
        """
        payload = dict(baseline_spec)
        payload["grid"] = {"grid_mode": True}

        with pytest.raises(ValueError, match="v3"):
            run_spec_portable_from_dict(payload)

    def test_portable_from_dict_raises_on_ligand_block(
        self, baseline_spec: dict
    ) -> None:
        """from_dict should raise ValueError on a payload carrying a 'ligand' block."""
        payload = dict(baseline_spec)
        payload["ligand"] = {"model_family": "ligandmpnn"}

        with pytest.raises(ValueError, match="v3"):
            run_spec_portable_from_dict(payload)

    def test_portable_from_dict_raises_on_arbitrary_unknown_key(
        self, baseline_spec: dict
    ) -> None:
        """from_dict must reject ANY unknown top-level key, not just grid/ligand.

        This is what actually closes the silent-data-loss gap: a v2 payload
        carrying an unrecognized block used to be silently ignored (see the
        module docstring's prior "unknown top-level keys are ignored" contract).
        """
        payload = dict(baseline_spec)
        payload["some_future_v3_block"] = {"anything": True}

        with pytest.raises(ValueError, match="unknown top-level key"):
            run_spec_portable_from_dict(payload)


class TestInspectionSpecificationExport:
    """Test that InspectionSpecification is properly exported."""

    def test_inspection_specification_importable(self) -> None:
        """InspectionSpecification should be importable from aminx.run."""
        # Already imported at module top, but explicit check
        assert InspectionSpecification is not None

    def test_inspection_specification_in_all(self) -> None:
        """InspectionSpecification should be in aminx.run.__all__."""
        assert "InspectionSpecification" in aminx.run.__all__
