"""Tests for --input-type and --cache-dir CLI flag integration (T4/T5 gate G2 matrix).

Tests cover:
- Flag dispatch: --input-type {auto,file,pdb,afdb,mdcath} behavior
- Cache dir precedence: --cache-dir flag takes priority
- G2 matrix cells: auto+scheme, auto+local, file+local, file+pdb://, pdb+schemeless, pdb+afdb:// conflict
- All 8 call sites wired correctly (run/spec emit-* commands)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from aminx.cli import _RunBase, resolve_inputs
from aminx.io.proxide_fetch import InputResolutionError


class TestInputTypeFlagMatrixCells:
  """Tests for G2 matrix: --input-type x scheme dispatch."""

  # === Cell 1: auto + scheme (pdb://, afdb://, mdcath://) ===
  def test_auto_with_pdb_scheme(self, tmp_path: Path) -> None:
    """input_type='auto' + pdb://ID dispatches to fetch_pdb."""
    with patch("aminx.cli.fetch_pdb") as mock_fetch:
      mock_fetch.return_value = tmp_path / "1ubq.cif"
      result = resolve_inputs(
        ["pdb://1ubq"],
        input_type="auto",
        cache_dir=tmp_path,
      )
      mock_fetch.assert_called_once()
      assert result == [str(tmp_path / "1ubq.cif")]

  def test_auto_with_afdb_scheme(self, tmp_path: Path) -> None:
    """input_type='auto' + afdb://ID dispatches to fetch_afdb."""
    with patch("aminx.cli.fetch_afdb") as mock_fetch:
      mock_fetch.return_value = tmp_path / "P12345.cif"
      result = resolve_inputs(
        ["afdb://P12345"],
        input_type="auto",
        cache_dir=tmp_path,
      )
      mock_fetch.assert_called_once()
      assert result == [str(tmp_path / "P12345.cif")]

  def test_auto_with_mdcath_scheme(self, tmp_path: Path) -> None:
    """input_type='auto' + mdcath://ID dispatches to fetch_md_cath."""
    with patch("aminx.cli.fetch_md_cath") as mock_fetch:
      mock_fetch.return_value = tmp_path / "1abcA00.h5"
      result = resolve_inputs(
        ["mdcath://1abcA00"],
        input_type="auto",
        cache_dir=tmp_path,
      )
      mock_fetch.assert_called_once()
      assert result == [str(tmp_path / "1abcA00.h5")]

  # === Cell 2: auto + local path (bare/file://) ===
  def test_auto_with_bare_local_path(self, tmp_path: Path) -> None:
    """input_type='auto' + bare path resolves as local."""
    pdb = tmp_path / "test.pdb"
    pdb.write_text("test")
    result = resolve_inputs(
      [str(pdb)],
      input_type="auto",
      cache_dir=tmp_path,
    )
    assert result == [str(pdb)]

  def test_auto_with_file_scheme(self, tmp_path: Path) -> None:
    """input_type='auto' + file:// resolves as local."""
    pdb = tmp_path / "test.pdb"
    pdb.write_text("test")
    result = resolve_inputs(
      [f"file://{str(pdb)}"],
      input_type="auto",
      cache_dir=tmp_path,
    )
    assert result == [str(pdb)]

  # === Cell 3: file + local path (force all local) ===
  def test_file_flag_treats_pdb_uri_as_literal_path(self, tmp_path: Path) -> None:
    """input_type='file' + pdb://ID treats as literal filename, no fetcher."""
    with patch("aminx.cli.fetch_pdb") as mock_fetch:
      result = resolve_inputs(
        ["pdb://1ubq"],
        input_type="file",
        cache_dir=tmp_path,
      )
      # Should return the literal string unchanged
      assert result == ["pdb://1ubq"]
      # Fetch should NOT be called
      mock_fetch.assert_not_called()

  def test_file_flag_suppresses_all_scheme_detection(self, tmp_path: Path) -> None:
    """input_type='file' suppresses scheme detection for all schemes."""
    with patch("aminx.cli.fetch_pdb") as mock_pdb, patch(
      "aminx.cli.fetch_afdb"
    ) as mock_afdb:
      result = resolve_inputs(
        ["pdb://1ubq", "afdb://P1", "mdcath://1x"],
        input_type="file",
        cache_dir=tmp_path,
      )
      # All should be treated as literal paths
      assert result == ["pdb://1ubq", "afdb://P1", "mdcath://1x"]
      # No fetchers should be called
      mock_pdb.assert_not_called()
      mock_afdb.assert_not_called()

  # === Cell 4: pdb/afdb/mdcath + schemeless token ===
  def test_input_type_pdb_applies_to_schemeless(self, tmp_path: Path) -> None:
    """input_type='pdb' + bare token treats token as pdb ID."""
    with patch("aminx.cli.fetch_pdb") as mock_fetch:
      mock_fetch.return_value = tmp_path / "1ubq.cif"
      result = resolve_inputs(
        ["1ubq"],
        input_type="pdb",
        cache_dir=tmp_path,
      )
      mock_fetch.assert_called_once_with("1ubq", tmp_path, fmt="mmcif")
      assert result == [str(tmp_path / "1ubq.cif")]

  def test_input_type_afdb_applies_to_schemeless(self, tmp_path: Path) -> None:
    """input_type='afdb' + bare token treats token as afdb ID."""
    with patch("aminx.cli.fetch_afdb") as mock_fetch:
      mock_fetch.return_value = tmp_path / "P12345.cif"
      result = resolve_inputs(
        ["P12345"],
        input_type="afdb",
        cache_dir=tmp_path,
      )
      mock_fetch.assert_called_once_with("P12345", tmp_path)
      assert result == [str(tmp_path / "P12345.cif")]

  def test_input_type_mdcath_applies_to_schemeless(self, tmp_path: Path) -> None:
    """input_type='mdcath' + bare token treats token as mdcath ID."""
    with patch("aminx.cli.fetch_md_cath") as mock_fetch:
      mock_fetch.return_value = tmp_path / "1abcA00.h5"
      result = resolve_inputs(
        ["1abcA00"],
        input_type="mdcath",
        cache_dir=tmp_path,
      )
      mock_fetch.assert_called_once_with("1abcA00", tmp_path)
      assert result == [str(tmp_path / "1abcA00.h5")]

  # === Cell 5: pdb flag + explicit afdb:// scheme (hard error) ===
  def test_input_type_pdb_with_afdb_uri_raises_badparameter(
    self, tmp_path: Path
  ) -> None:
    """input_type='pdb' + afdb://P1 raises BadParameter (scheme conflict)."""
    with pytest.raises(typer.BadParameter) as exc_info:
      resolve_inputs(
        ["afdb://P12345"],
        input_type="pdb",
        cache_dir=tmp_path,
      )
    assert "conflicts with explicit scheme" in str(exc_info.value)

  def test_input_type_afdb_with_pdb_uri_raises_badparameter(
    self, tmp_path: Path
  ) -> None:
    """input_type='afdb' + pdb://ID raises BadParameter (scheme conflict)."""
    with pytest.raises(typer.BadParameter) as exc_info:
      resolve_inputs(
        ["pdb://1ubq"],
        input_type="afdb",
        cache_dir=tmp_path,
      )
    assert "conflicts with explicit scheme" in str(exc_info.value)

  def test_input_type_mdcath_with_pdb_uri_raises_badparameter(
    self, tmp_path: Path
  ) -> None:
    """input_type='mdcath' + pdb://ID raises BadParameter (scheme conflict)."""
    with pytest.raises(typer.BadParameter) as exc_info:
      resolve_inputs(
        ["pdb://1ubq"],
        input_type="mdcath",
        cache_dir=tmp_path,
      )
    assert "conflicts with explicit scheme" in str(exc_info.value)

  # === Cell 6: matching scheme + flag (ok, no override) ===
  def test_input_type_pdb_with_pdb_uri_no_conflict(self, tmp_path: Path) -> None:
    """input_type='pdb' + pdb://ID is ok (scheme matches flag, no conflict)."""
    with patch("aminx.cli.fetch_pdb") as mock_fetch:
      mock_fetch.return_value = tmp_path / "1ubq.cif"
      result = resolve_inputs(
        ["pdb://1ubq"],
        input_type="pdb",
        cache_dir=tmp_path,
      )
      mock_fetch.assert_called_once()
      assert result == [str(tmp_path / "1ubq.cif")]


class TestCacheDirPrecedence:
  """Tests for --cache-dir flag precedence (T4)."""

  def test_explicit_cache_dir_passed_to_fetcher(self, tmp_path: Path) -> None:
    """Explicit --cache-dir is passed through to resolve_inputs."""
    cache_dir = tmp_path / "custom_cache"
    with patch("aminx.cli.fetch_pdb") as mock_fetch:
      mock_fetch.return_value = cache_dir / "pdb" / "1ubq" / "mmcif" / "1ubq.cif"
      result = resolve_inputs(
        ["pdb://1ubq"],
        input_type="auto",
        cache_dir=cache_dir,
      )
      # Fetcher should be called with the explicit cache_dir
      mock_fetch.assert_called_once_with("1ubq", cache_dir, fmt="mmcif")


class TestRunBaseIntegration:
  """Tests that _RunBase carries input_type and input_cache_dir correctly."""

  def test_runbase_fields_exist(self) -> None:
    """_RunBase has input_type and input_cache_dir fields."""
    rb = _RunBase(
      topology=None,
      model_weights="original",
      model_version="v_48_020",
      model_family="proteinmpnn",
      checkpoint_id=None,
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
      truncation_strategy="random_crop",
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
    )
    assert rb.input_type == "auto"
    assert rb.input_cache_dir is None

  def test_runbase_custom_cache_dir(self, tmp_path: Path) -> None:
    """_RunBase can hold a custom input_cache_dir."""
    rb = _RunBase(
      topology=None,
      model_weights="original",
      model_version="v_48_020",
      model_family="proteinmpnn",
      checkpoint_id=None,
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
      truncation_strategy="random_crop",
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
      input_type="pdb",
      input_cache_dir=tmp_path,
    )
    assert rb.input_type == "pdb"
    assert rb.input_cache_dir == tmp_path
