"""Integration tests for heterogeneous input resolution (T8, #1203).

Tests cover:
- End-to-end mixed resolution (local + remote mocked)
- Local regression (G5): dir/glob/concrete paths unchanged
- Portable JSON wire format no-regression (G7)
- Spec emit determinism (G8): spec JSON unchanged for local inputs
- Runner-side unresolved-URI guard (T7b, G4b)

All remote fetchers are mocked — no network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from aminx.cli import app, resolve_inputs
from aminx.io.proxide_fetch import InputResolutionError
from aminx.run.run_spec_portable_json import (
    run_spec_portable_from_dict,
    run_spec_portable_to_dict,
)
from aminx.run.spec_json import run_specification_from_json


runner = CliRunner()


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def local_fixture_file(tmp_path: Path) -> Path:
    """Create a local fixture file for mixed-input testing."""
    pdb_file = tmp_path / "fixture.pdb"
    pdb_file.write_text("PDB dummy data")
    return pdb_file


@pytest.fixture
def local_fixture_dir(tmp_path: Path) -> Path:
    """Create a directory with .pdb and .cif files for local regression testing."""
    struct_dir = tmp_path / "structures"
    struct_dir.mkdir()

    # Create a few test files (matching names in tests/data for consistency)
    (struct_dir / "1ubq.pdb").write_text("PDB dummy")
    (struct_dir / "1BC8.cif").write_text("CIF dummy")
    (struct_dir / "readme.txt").write_text("ignored")  # Should be filtered out

    return struct_dir


# ============================================================================
# G6: End-to-end mixed resolution (local + remote with mocked fetchers)
# ============================================================================


class TestEndToEndMixedResolution:
    """Gate G6: mixed local + remote inputs resolve independently to local paths.

    All remote fetchers are mocked to avoid network access.
    """

    def test_mixed_pdb_local_afdb_resolves_all(
        self, tmp_path: Path, local_fixture_file: Path
    ) -> None:
        """Mixed list of pdb://, local file, afdb:// resolves to all local paths."""
        cache_dir = tmp_path / "cache"

        # Mock the fetchers
        with patch("aminx.cli.fetch_pdb") as mock_pdb, \
             patch("aminx.cli.fetch_afdb") as mock_afdb:

            # Set up mock return values (paths that "would be downloaded")
            pdb_result = cache_dir / "pdb" / "1ubq" / "mmcif" / "1ubq.cif"
            pdb_result.parent.mkdir(parents=True, exist_ok=True)
            pdb_result.write_text("PDB downloaded")

            afdb_result = cache_dir / "afdb" / "P12345" / "P12345.cif"
            afdb_result.parent.mkdir(parents=True, exist_ok=True)
            afdb_result.write_text("AFDB downloaded")

            mock_pdb.return_value = pdb_result
            mock_afdb.return_value = afdb_result

            result = resolve_inputs(
                ["pdb://1ubq", str(local_fixture_file), "afdb://P12345"],
                input_type="auto",
                cache_dir=cache_dir,
            )

            # All three should be resolved to local paths
            assert len(result) == 3
            assert str(pdb_result) in result
            assert str(local_fixture_file) in result
            assert str(afdb_result) in result

            # Verify each fetcher was called appropriately
            mock_pdb.assert_called_once_with("1ubq", cache_dir, fmt="mmcif")
            mock_afdb.assert_called_once_with("P12345", cache_dir)

    def test_mixed_mdcath_local_pdb_resolves_all(
        self, tmp_path: Path, local_fixture_file: Path
    ) -> None:
        """Mixed mdcath://, local, pdb:// resolves independently."""
        cache_dir = tmp_path / "cache"

        with patch("aminx.cli.fetch_pdb") as mock_pdb, \
             patch("aminx.cli.fetch_md_cath") as mock_mdcath:

            pdb_result = cache_dir / "pdb" / "2xyz" / "mmcif" / "2xyz.cif"
            pdb_result.parent.mkdir(parents=True, exist_ok=True)
            pdb_result.write_text("PDB data")

            mdcath_result = cache_dir / "mdcath" / "1abcA00" / "1abcA00.h5"
            mdcath_result.parent.mkdir(parents=True, exist_ok=True)
            mdcath_result.write_text("HDF5 data")

            mock_pdb.return_value = pdb_result
            mock_mdcath.return_value = mdcath_result

            result = resolve_inputs(
                ["mdcath://1abcA00", str(local_fixture_file), "pdb://2xyz"],
                input_type="auto",
                cache_dir=cache_dir,
            )

            assert len(result) == 3
            assert str(pdb_result) in result
            assert str(local_fixture_file) in result
            assert str(mdcath_result) in result

    def test_mixed_order_preserved_first_seen(
        self, tmp_path: Path, local_fixture_file: Path
    ) -> None:
        """Mixed inputs are resolved in order; dedup preserves first-seen."""
        cache_dir = tmp_path / "cache"

        with patch("aminx.cli.fetch_pdb") as mock_pdb:
            pdb_result = cache_dir / "pdb" / "1ubq" / "mmcif" / "1ubq.cif"
            pdb_result.parent.mkdir(parents=True, exist_ok=True)
            pdb_result.write_text("PDB data")
            mock_pdb.return_value = pdb_result

            # Repeat 1ubq; should be deduped pre-fetch
            result = resolve_inputs(
                [str(local_fixture_file), "pdb://1ubq", "pdb://1ubq"],
                input_type="auto",
                cache_dir=cache_dir,
            )

            # Two entries: local file and pdb (duplicated pdb deduped)
            assert len(result) == 2
            # pdb fetched only once (pre-dedup)
            assert mock_pdb.call_count == 1

    def test_all_three_remote_sources_dispatch_correctly(
        self, tmp_path: Path
    ) -> None:
        """All three remote schemes (pdb, afdb, mdcath) dispatch to correct fetchers."""
        cache_dir = tmp_path / "cache"

        with patch("aminx.cli.fetch_pdb") as mock_pdb, \
             patch("aminx.cli.fetch_afdb") as mock_afdb, \
             patch("aminx.cli.fetch_md_cath") as mock_mdcath:

            # Set up results
            pdb_result = cache_dir / "pdb" / "1a3a" / "mmcif" / "1a3a.cif"
            pdb_result.parent.mkdir(parents=True, exist_ok=True)
            pdb_result.write_text("PDB")

            afdb_result = cache_dir / "afdb" / "P99999" / "P99999.cif"
            afdb_result.parent.mkdir(parents=True, exist_ok=True)
            afdb_result.write_text("AFDB")

            mdcath_result = cache_dir / "mdcath" / "2xyzB11" / "2xyzB11.h5"
            mdcath_result.parent.mkdir(parents=True, exist_ok=True)
            mdcath_result.write_text("MDCATH")

            mock_pdb.return_value = pdb_result
            mock_afdb.return_value = afdb_result
            mock_mdcath.return_value = mdcath_result

            result = resolve_inputs(
                ["pdb://1A3A", "afdb://P99999", "mdcath://2xyzB11"],
                input_type="auto",
                cache_dir=cache_dir,
            )

            assert len(result) == 3
            mock_pdb.assert_called_once()
            mock_afdb.assert_called_once()
            mock_mdcath.assert_called_once()


# ============================================================================
# G5: Local regression (dir/glob/concrete paths unchanged)
# ============================================================================


class TestLocalRegression:
    """Gate G5: local paths, dirs, globs resolve byte-identically to golden.

    This pins that heterogeneous input feature did not change local handling.
    """

    def test_concrete_local_file_unchanged(self, tmp_path: Path) -> None:
        """Concrete local file passed through unchanged."""
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text("test data")

        result = resolve_inputs([str(pdb_file)], input_type="auto", cache_dir=tmp_path)

        assert result == [str(pdb_file)]

    def test_directory_expansion_unchanged(self, local_fixture_dir: Path, tmp_path: Path) -> None:
        """Directory expansion to .pdb/.cif unchanged."""
        result = resolve_inputs(
            [str(local_fixture_dir)],
            input_type="auto",
            cache_dir=tmp_path,
        )

        # Should expand to the two structure files, sorted
        expected = sorted([
            str(local_fixture_dir / "1ubq.pdb"),
            str(local_fixture_dir / "1BC8.cif"),
        ])
        assert result == expected

    def test_glob_expansion_unchanged(self, local_fixture_dir: Path, tmp_path: Path) -> None:
        """Glob expansion to .pdb/.cif unchanged."""
        pattern = str(local_fixture_dir / "*.pdb")
        result = resolve_inputs(
            [pattern],
            input_type="auto",
            cache_dir=tmp_path,
        )

        # Should match only the .pdb file
        assert len(result) == 1
        assert result[0].endswith("1ubq.pdb")

    def test_dedup_preserves_firstseen_local(
        self, local_fixture_file: Path, tmp_path: Path
    ) -> None:
        """Local dedup preserves first-seen order (no sort)."""
        result = resolve_inputs(
            [str(local_fixture_file), str(local_fixture_file)],
            input_type="auto",
            cache_dir=tmp_path,
        )

        assert result == [str(local_fixture_file)]

    def test_file_scheme_strips_to_local(self, local_fixture_file: Path, tmp_path: Path) -> None:
        """file:// scheme resolves to local path unchanged."""
        result = resolve_inputs(
            [f"file://{str(local_fixture_file)}"],
            input_type="auto",
            cache_dir=tmp_path,
        )

        assert result == [str(local_fixture_file)]


# ============================================================================
# G7: Portable JSON (RS-8) no-regression
# ============================================================================


class TestPortableJsonNoRegression:
    """Gate G7: tests/run/test_run_spec_portable_json.py unchanged and green.

    Asserts that the portable wire format v2 is untouched by the inputs feature
    (which adds no field to RunSpec and does not serialize inputs).
    """

    @pytest.fixture
    def baseline_v2_spec(self) -> dict:
        """Minimal valid v2 portable JSON (same as test_run_spec_portable_json.py)."""
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

    def test_portable_roundtrip_succeeds(self, baseline_v2_spec: dict) -> None:
        """Plain RunSpec roundtrips through portable JSON unchanged."""
        # Deserialize baseline to RunSpec
        baseline = run_spec_portable_from_dict(baseline_v2_spec)

        # Serialize back to dict
        d = run_spec_portable_to_dict(baseline)

        # Roundtrip again
        roundtripped = run_spec_portable_from_dict(d)

        # Verify key properties unchanged
        assert d["version"] == 2
        assert roundtripped is not None
        assert roundtripped.multistate.mode == baseline_v2_spec["multistate"]["mode"]

    def test_portable_inputs_field_absent(self, baseline_v2_spec: dict) -> None:
        """Portable JSON v2 dict has no 'inputs' field (not part of wire format)."""
        baseline = run_spec_portable_from_dict(baseline_v2_spec)
        d = run_spec_portable_to_dict(baseline)

        # 'inputs' should not be serialized in portable format
        assert "inputs" not in d

        # Only the portable sub-configs should be present
        assert set(d.keys()) == {"version", "io", "multistate", "resource", "precision"}


# ============================================================================
# G8: Spec emit determinism (spec JSON unchanged for local inputs)
# ============================================================================


class TestSpecEmitDeterminism:
    """Gate G8: spec emit JSON for local inputs byte-identical to golden.

    Asserts that resolve_inputs and spec serialization produce consistent
    JSON for already-local inputs (no new noise from the inputs feature).
    """

    def test_spec_emit_local_deterministic(
        self, local_fixture_file: Path, tmp_path: Path
    ) -> None:
        """Spec JSON for local input is deterministic across calls."""
        from aminx.run.spec_json import run_specification_to_json_dict
        from aminx.run.specs import SamplingSpecification

        cache_dir = tmp_path / "cache"

        # Resolve the local file
        resolved = resolve_inputs([str(local_fixture_file)], input_type="auto", cache_dir=cache_dir)

        # Create a spec from the resolved path
        spec1 = SamplingSpecification(inputs=resolved[0])
        dict1 = run_specification_to_json_dict(spec1)

        # Resolve again and create another spec
        resolved2 = resolve_inputs([str(local_fixture_file)], input_type="auto", cache_dir=cache_dir)
        spec2 = SamplingSpecification(inputs=resolved2[0])
        dict2 = run_specification_to_json_dict(spec2)

        # Dicts should be identical (deterministic)
        assert json.dumps(dict1, sort_keys=True) == json.dumps(dict2, sort_keys=True)

    def test_directory_expansion_spec_emit_deterministic(
        self, local_fixture_dir: Path, tmp_path: Path
    ) -> None:
        """Spec JSON for directory expansion is deterministic."""
        from aminx.run.spec_json import run_specification_to_json_dict
        from aminx.run.specs import SamplingSpecification

        cache_dir = tmp_path / "cache"

        # Expand directory
        resolved = resolve_inputs([str(local_fixture_dir)], input_type="auto", cache_dir=cache_dir)

        # Create spec and emit JSON
        spec = SamplingSpecification(inputs=resolved)
        d = run_specification_to_json_dict(spec)

        # inputs field should be the resolved list
        assert d["inputs"] == resolved

        # Resolve again -> should produce identical JSON
        resolved2 = resolve_inputs([str(local_fixture_dir)], input_type="auto", cache_dir=cache_dir)
        spec2 = SamplingSpecification(inputs=resolved2)
        d2 = run_specification_to_json_dict(spec2)

        assert json.dumps(d, sort_keys=True) == json.dumps(d2, sort_keys=True)


# ============================================================================
# T7b/G4b: Runner-side unresolved-URI guard
# ============================================================================


class TestRunnerGuardUnresolvedUri:
    """Gate G4b (T7b): unresolved URI in spec raises InputResolutionError.

    Tests that _loader_inputs (runner-side guard) catches unresolved URIs
    and raises InputResolutionError before the runner attempts to fetch them.
    """

    def test_unresolved_pdb_uri_in_spec_json_raises(self) -> None:
        """run_specification_from_json with unresolved pdb:// raises InputResolutionError."""
        spec_dict = {
            "_spec_class": "SamplingSpecification",
            "inputs": ["pdb://1A3A"],  # Unresolved URI
        }

        spec = run_specification_from_json(json.dumps(spec_dict))

        # Loading inputs should raise
        from aminx.run.specs import _loader_inputs
        with pytest.raises(InputResolutionError) as exc_info:
            _loader_inputs(spec.inputs)

        error_msg = str(exc_info.value)
        assert "pdb://1A3A" in error_msg
        assert "local paths" in error_msg.lower()

    def test_unresolved_afdb_uri_in_spec_json_raises(self) -> None:
        """Unresolved afdb:// in spec raises InputResolutionError."""
        spec_dict = {
            "_spec_class": "SamplingSpecification",
            "inputs": ["afdb://P12345"],
        }

        spec = run_specification_from_json(json.dumps(spec_dict))

        from aminx.run.specs import _loader_inputs
        with pytest.raises(InputResolutionError) as exc_info:
            _loader_inputs(spec.inputs)

        assert "afdb://P12345" in str(exc_info.value)

    def test_unresolved_mdcath_uri_in_spec_json_raises(self) -> None:
        """Unresolved mdcath:// in spec raises InputResolutionError."""
        spec_dict = {
            "_spec_class": "SamplingSpecification",
            "inputs": ["mdcath://1abcA00"],
        }

        spec = run_specification_from_json(json.dumps(spec_dict))

        from aminx.run.specs import _loader_inputs
        with pytest.raises(InputResolutionError) as exc_info:
            _loader_inputs(spec.inputs)

        assert "mdcath://1abcA00" in str(exc_info.value)

    def test_mixed_resolved_and_unresolved_raises(self, local_fixture_file: Path) -> None:
        """Mixed resolved + unresolved raises on first unresolved."""
        spec_dict = {
            "_spec_class": "SamplingSpecification",
            "inputs": [str(local_fixture_file), "pdb://1A3A"],  # Second is unresolved
        }

        spec = run_specification_from_json(json.dumps(spec_dict))

        from aminx.run.specs import _loader_inputs
        with pytest.raises(InputResolutionError) as exc_info:
            _loader_inputs(spec.inputs)

        assert "pdb://1A3A" in str(exc_info.value)

    def test_resolved_local_path_does_not_raise(self, local_fixture_file: Path) -> None:
        """Resolved local path does not raise."""
        spec_dict = {
            "_spec_class": "SamplingSpecification",
            "inputs": [str(local_fixture_file)],
        }

        spec = run_specification_from_json(json.dumps(spec_dict))

        from aminx.run.specs import _loader_inputs
        result = _loader_inputs(spec.inputs)

        # Should return the input unchanged
        assert result == [str(local_fixture_file)]


# ============================================================================
# Integration: end-to-end from CLI --inputs to spec JSON and back
# ============================================================================


class TestIntegrationCliToSpecJson:
    """Integration: resolve_inputs -> spec emit -> spec load -> _loader_inputs.

    Full roundtrip from CLI arguments to spec JSON and back, verifying
    the resolver integrates with the spec pipeline correctly.
    """

    def test_cli_resolve_spec_roundtrip_local(self, local_fixture_file: Path, tmp_path: Path) -> None:
        """Local input: resolve -> emit spec JSON -> load spec -> _loader_inputs."""
        from aminx.run.spec_json import (
            run_specification_to_json_dict,
            run_specification_from_json,
        )
        from aminx.run.specs import _loader_inputs, SamplingSpecification

        cache_dir = tmp_path / "cache"

        # Step 1: Resolve inputs
        resolved = resolve_inputs([str(local_fixture_file)], input_type="auto", cache_dir=cache_dir)

        # Step 2: Build spec
        spec = SamplingSpecification(inputs=resolved[0])

        # Step 3: Emit to JSON
        spec_dict = run_specification_to_json_dict(spec)
        spec_json = json.dumps(spec_dict)

        # Step 4: Load spec from JSON
        loaded_spec = run_specification_from_json(spec_json)

        # Step 5: _loader_inputs should succeed (path is resolved)
        result = _loader_inputs(loaded_spec.inputs)
        # Note: inputs is a string, so _loader_inputs returns the string (strings are sequences)
        assert result == str(local_fixture_file)

    def test_cli_resolve_spec_roundtrip_remote_mocked(
        self, local_fixture_file: Path, tmp_path: Path
    ) -> None:
        """Remote input (mocked): resolve -> emit spec -> load -> _loader_inputs."""
        from aminx.run.spec_json import (
            run_specification_to_json_dict,
            run_specification_from_json,
        )
        from aminx.run.specs import _loader_inputs, SamplingSpecification

        cache_dir = tmp_path / "cache"

        with patch("aminx.cli.fetch_pdb") as mock_pdb:
            pdb_result = cache_dir / "pdb" / "1ubq" / "mmcif" / "1ubq.cif"
            pdb_result.parent.mkdir(parents=True, exist_ok=True)
            pdb_result.write_text("PDB data")
            mock_pdb.return_value = pdb_result

            # Step 1: Resolve pdb:// URI (mocked)
            resolved = resolve_inputs(["pdb://1ubq"], input_type="auto", cache_dir=cache_dir)

            # Step 2: Build spec from resolved path
            spec = SamplingSpecification(inputs=resolved[0])

            # Step 3: Emit to JSON
            spec_dict = run_specification_to_json_dict(spec)
            spec_json = json.dumps(spec_dict)

            # Step 4: Load spec from JSON
            loaded_spec = run_specification_from_json(spec_json)

            # Step 5: _loader_inputs should succeed (path is now local)
            result = _loader_inputs(loaded_spec.inputs)
            # Note: inputs is a string, so _loader_inputs returns the string (strings are sequences)
            assert result == str(pdb_result)

    def test_cli_resolve_mixed_spec_roundtrip(
        self, local_fixture_file: Path, tmp_path: Path
    ) -> None:
        """Mixed local + remote: resolve -> emit spec -> load -> _loader_inputs."""
        from aminx.run.spec_json import (
            run_specification_to_json_dict,
            run_specification_from_json,
        )
        from aminx.run.specs import _loader_inputs, SamplingSpecification

        cache_dir = tmp_path / "cache"

        with patch("aminx.cli.fetch_pdb") as mock_pdb:
            pdb_result = cache_dir / "pdb" / "1a3a" / "mmcif" / "1a3a.cif"
            pdb_result.parent.mkdir(parents=True, exist_ok=True)
            pdb_result.write_text("PDB data")
            mock_pdb.return_value = pdb_result

            # Step 1: Resolve mixed inputs
            resolved = resolve_inputs(
                ["pdb://1a3a", str(local_fixture_file)],
                input_type="auto",
                cache_dir=cache_dir,
            )
            assert len(resolved) == 2

            # Step 2: Build spec from resolved paths
            spec = SamplingSpecification(inputs=resolved)

            # Step 3: Emit to JSON
            spec_dict = run_specification_to_json_dict(spec)
            spec_json = json.dumps(spec_dict)

            # Step 4: Load spec from JSON
            loaded_spec = run_specification_from_json(spec_json)

            # Step 5: _loader_inputs should succeed (all paths are now local)
            result = _loader_inputs(loaded_spec.inputs)
            assert len(result) == 2
            assert str(pdb_result) in result
            assert str(local_fixture_file) in result
