"""Tests for RunSpec sub-configs and build_run_spec."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aminx.run.spec import (
    PlannerTopology,
    RunSpec,
    build_run_spec,
    topology_hash,
)


class TestPlannerTopology:
    """Tests for PlannerTopology config class."""

    def test_planner_topology_default(self) -> None:
        """PlannerTopology should instantiate with default use_unified_driver=True."""
        pt = PlannerTopology(use_unified_driver=True)
        assert pt.use_unified_driver is True

    def test_planner_topology_false(self) -> None:
        """PlannerTopology should instantiate with use_unified_driver=False."""
        pt = PlannerTopology(use_unified_driver=False)
        assert pt.use_unified_driver is False

    def test_planner_topology_is_eqx_module(self) -> None:
        """PlannerTopology should be an Equinox Module."""
        import equinox as eqx

        pt = PlannerTopology(use_unified_driver=True)
        assert isinstance(pt, eqx.Module)

    def test_planner_topology_static_field(self) -> None:
        """use_unified_driver should be marked as static."""
        import equinox as eqx

        pt = PlannerTopology(use_unified_driver=True)
        # Access field info to verify static marking
        assert hasattr(pt, "use_unified_driver")
        # Static fields don't participate in pytree leaves
        leaves, treedef = eqx.tree_flatten_one_level(pt)
        assert len(leaves) == 0  # All fields are static, no leaves


class TestTopologyHash:
    """Tests for topology_hash function."""

    def test_topology_hash_returns_16_char_hex(self) -> None:
        """topology_hash should return a 16-character hex string."""
        pt = PlannerTopology(use_unified_driver=True)
        h = topology_hash(pt)
        assert isinstance(h, str)
        assert len(h) == 16
        # Should be valid hex
        int(h, 16)

    def test_topology_hash_determinism(self) -> None:
        """topology_hash should be deterministic for same input."""
        pt1 = PlannerTopology(use_unified_driver=True)
        pt2 = PlannerTopology(use_unified_driver=True)
        assert topology_hash(pt1) == topology_hash(pt2)

    def test_topology_hash_true_vs_false(self) -> None:
        """topology_hash should differ for True vs False."""
        pt_true = PlannerTopology(use_unified_driver=True)
        pt_false = PlannerTopology(use_unified_driver=False)
        h_true = topology_hash(pt_true)
        h_false = topology_hash(pt_false)
        assert h_true != h_false

    def test_topology_hash_multiple_calls_consistent(self) -> None:
        """topology_hash should return same result on multiple calls."""
        pt = PlannerTopology(use_unified_driver=False)
        hashes = [topology_hash(pt) for _ in range(5)]
        assert len(set(hashes)) == 1  # All hashes identical

    def test_topology_hash_value_stable(self) -> None:
        """topology_hash value is stable and backward-compatible.

        Regression test: verify reflective derivation produces the same hash as the
        old hand-maintained implementation for the current single-field topology.
        """
        # Compute expected value using the original hand-maintained logic
        expected = hashlib.sha256(
            json.dumps({"use_unified_driver": True}, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Actual reflective implementation should produce identical hash
        actual = topology_hash(PlannerTopology(use_unified_driver=True))
        assert actual == expected

    def test_topology_hash_reflective_picks_up_new_fields(self) -> None:
        """topology_hash picks up new fields via reflection (anti-regression).

        This test proves that the implementation is truly reflective. If it were
        still hand-maintained, adding a new field would NOT affect the hash.
        With reflection, all dataclass fields are automatically included.
        """
        from typing import cast

        import equinox as eqx

        # Define a probe topology with two fields to demonstrate reflectivity
        class _ProbeTopology(eqx.Module):
            a: bool = eqx.field(static=True)
            b: int = eqx.field(static=True)

        # Hash should differ when second field changes
        # Cast to PlannerTopology for type checking (they're both eqx.Module with dataclass fields)
        h1 = topology_hash(cast(PlannerTopology, _ProbeTopology(a=True, b=1)))
        h2 = topology_hash(cast(PlannerTopology, _ProbeTopology(a=True, b=2)))
        assert h1 != h2, "Second field 'b' must affect hash (proves reflection)"


class TestBuildRunSpecPlan:
    """Tests for build_run_spec populating plan field."""

    @pytest.fixture
    def minimal_spec(self) -> object:
        """Minimal object for build_run_spec."""

        class MinimalSpec:
            pass

        return MinimalSpec()

    def test_build_run_spec_has_plan_field(self, minimal_spec: object) -> None:
        """build_run_spec should add plan field to RunSpec."""
        rs = build_run_spec(minimal_spec)
        assert hasattr(rs, "plan")

    def test_plan_is_planner_topology(self, minimal_spec: object) -> None:
        """RunSpec.plan should be a PlannerTopology instance."""
        rs = build_run_spec(minimal_spec)
        assert isinstance(rs.plan, PlannerTopology)

    def test_plan_default_use_unified_driver_true(
        self, minimal_spec: object
    ) -> None:
        """plan.use_unified_driver should default to True."""
        rs = build_run_spec(minimal_spec)
        assert rs.plan.use_unified_driver is True

    def test_plan_respects_spec_attribute(self) -> None:
        """plan.use_unified_driver should use spec.use_unified_driver if present."""

        class SpecWithDriver:
            use_unified_driver = False

        spec = SpecWithDriver()
        rs = build_run_spec(spec)
        assert rs.plan.use_unified_driver is False

    def test_plan_coerces_to_bool(self) -> None:
        """plan.use_unified_driver should coerce non-bool to bool."""

        class SpecWithTruthy:
            use_unified_driver = 1

        spec = SpecWithTruthy()
        rs = build_run_spec(spec)
        assert rs.plan.use_unified_driver is True
        assert isinstance(rs.plan.use_unified_driver, bool)


class TestRunIdDeterminism:
    """Tests for deterministic RunSpec.run_id population (backlog-5560)."""

    def test_run_id_is_deterministic(self) -> None:
        from aminx.run.specs import SamplingSpecification
        from xtrax.run.sink import derive_sink_spec

        spec1 = SamplingSpecification(inputs="dummy.pdb", random_seed=42)
        spec2 = SamplingSpecification(inputs="dummy.pdb", random_seed=42)
        rs1 = build_run_spec(spec1)
        rs2 = build_run_spec(spec2)

        assert rs1.run_id is not None
        assert rs1.run_id == rs2.run_id
        assert rs1.run_id.startswith("run-")

        # derive_sink_spec uses rs.run_id instead of generating a fresh UUID
        sink_spec1 = derive_sink_spec(rs1, output_dir=Path("/tmp/out"))
        sink_spec2 = derive_sink_spec(rs2, output_dir=Path("/tmp/out"))
        assert sink_spec1.run_id == rs1.run_id
        assert sink_spec1.run_id == sink_spec2.run_id

    def test_run_id_differs_for_different_specs(self) -> None:
        from aminx.run.specs import SamplingSpecification

        spec1 = SamplingSpecification(inputs="dummy.pdb", random_seed=42)
        spec2 = SamplingSpecification(inputs="dummy.pdb", random_seed=43)
        rs1 = build_run_spec(spec1)
        rs2 = build_run_spec(spec2)

        assert rs1.run_id != rs2.run_id

    def test_explicit_run_id_respected(self) -> None:
        from aminx.run.specs import SamplingSpecification

        spec = SamplingSpecification(inputs="dummy.pdb", run_id="custom-run-123")
        rs = build_run_spec(spec)
        assert rs.run_id == "custom-run-123"


class TestOutputPaths:
    """Tests for output_path canonical attribute and aliases."""

    def test_output_path_canonical(self) -> None:
        from aminx.run.specs import SamplingSpecification

        spec = SamplingSpecification(inputs="dummy.pdb", output_path="/tmp/test.zarr")
        assert spec.output_path == Path("/tmp/test.zarr")
        assert spec.output_zarr_path == Path("/tmp/test.zarr")
        assert spec.output_h5_path == Path("/tmp/test.zarr")

        rs = build_run_spec(spec)
        assert rs.io.output_path == Path("/tmp/test.zarr")
        assert rs.io.output_zarr_path == Path("/tmp/test.zarr")
        assert rs.io.output_h5_path == Path("/tmp/test.zarr")

    def test_output_zarr_path_sets_canonical(self) -> None:
        from aminx.run.specs import SamplingSpecification

        spec = SamplingSpecification(inputs="dummy.pdb", output_zarr_path="/tmp/test2.zarr")
        assert spec.output_path == Path("/tmp/test2.zarr")
        assert spec.output_zarr_path == Path("/tmp/test2.zarr")
        assert spec.output_h5_path == Path("/tmp/test2.zarr")

    def test_output_h5_path_deprecated_sets_canonical(self) -> None:
        from aminx.run.specs import SamplingSpecification

        spec = SamplingSpecification(inputs="dummy.pdb", output_h5_path="/tmp/test3.h5")
        assert spec.output_path == Path("/tmp/test3.h5")
        assert spec.output_zarr_path == Path("/tmp/test3.h5")
        assert spec.output_h5_path == Path("/tmp/test3.h5")

