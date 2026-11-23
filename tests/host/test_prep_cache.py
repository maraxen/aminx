"""Tests for JAX compilation cache configuration in prep.py."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import pytest

from xtrax.jax_cache import JaxCacheConfigError, JaxCacheDirError

from aminx.host.prep import prep_protein_stream_and_model
from aminx.run.specs import SamplingSpecification


pytestmark = pytest.mark.usefixtures("jax_cache_isolated")


def _run_prep_mocked(spec: SamplingSpecification) -> None:
  """Run prep with the dataset and model loader mocked; re-raise xtrax cache errors."""
  with patch("aminx.host.prep.create_protein_dataset") as mock_dataset, \
       patch("aminx.host.prep.load_model") as mock_load_model:
    mock_dataset.return_value = MagicMock()
    mock_load_model.return_value = MagicMock()
    try:
      prep_protein_stream_and_model(spec)
    except (JaxCacheDirError, JaxCacheConfigError):
      raise
    except Exception:
      # Mock-induced failures after the cache is configured are expected and ignored.
      pass


class TestPrepCacheConfig:
  """Test suite for JAX compilation cache configuration."""

  def test_cache_path_set_updates_jax_config(self, tmp_path: Path) -> None:
    """When cache_path is set on spec, jax_compilation_cache_dir should be updated."""
    cache_dir = tmp_path / "jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    spec = SamplingSpecification(
      inputs="dummy.pdb",
      cache_path=cache_dir,
    )

    _run_prep_mocked(spec)

    # xtrax namespaces the directory as <dir>/jaxlib-<ver>
    expected = cache_dir / f"jaxlib-{importlib.metadata.version('jaxlib')}"
    assert jax.config.jax_compilation_cache_dir == str(expected)
    assert expected.is_dir()

  def test_cache_path_none_does_not_update_jax_config(
    self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    """When cache_path is None, prep leaves the cache to the entry point.

    A distinct preset flag plus an xtrax env directory make this fail if prep called
    ``jax_cache.configure()`` anyway.
    """
    jax.config.update("jax_compilation_cache_dir", str(tmp_path / "preset"))
    monkeypatch.setenv("XTRAX_JAX_CACHE_DIR", str(tmp_path / "env_cache"))
    original_cache_dir = jax.config.jax_compilation_cache_dir

    spec = SamplingSpecification(
      inputs="dummy.pdb",
      cache_path=None,
    )

    _run_prep_mocked(spec)

    # Verify that jax.config was not changed
    cache_config = jax.config.jax_compilation_cache_dir
    assert cache_config == original_cache_dir

  def test_cache_path_converts_string_to_path(self, tmp_path: Path) -> None:
    """When cache_path is a string, it should be converted to Path and used."""
    cache_dir = tmp_path / "jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    spec = SamplingSpecification(
      inputs="dummy.pdb",
      cache_path=str(cache_dir),
    )

    # Verify that cache_path was converted to Path in __post_init__
    assert isinstance(spec.cache_path, Path)

    _run_prep_mocked(spec)

    # Verify jax.config was set to the namespaced directory under the converted Path
    cache_config = jax.config.jax_compilation_cache_dir
    assert cache_config == str(cache_dir / f"jaxlib-{importlib.metadata.version('jaxlib')}")


class TestPrepCacheFailLoud:
  """xtrax cache errors reach the caller instead of being swallowed (debt #3527)."""

  def test_unwritable_cache_dir_raises(self, tmp_path: Path) -> None:
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    spec = SamplingSpecification(inputs="dummy.pdb", cache_path=blocker)

    with pytest.raises(JaxCacheDirError):
      _run_prep_mocked(spec)

  def test_file_like_cache_path_is_not_used_as_cache_dir(
    self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    """A cache_path with a suffix is an output-file hint, never a compilation cache directory."""
    jax.config.update("jax_compilation_cache_dir", str(tmp_path / "preset"))
    store = tmp_path / "out" / "store.h5"
    spec = SamplingSpecification(inputs="dummy.pdb", cache_path=store)

    _run_prep_mocked(spec)

    assert jax.config.jax_compilation_cache_dir == str(tmp_path / "preset")
    assert not store.exists()
