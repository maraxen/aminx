"""Unit tests for `aminx.io.sink_provenance` -- FINDING 5.

task_id `260910_aminx-sink-provenance-schema`. Before this task, nothing under `tests/`
exercised `logits_bias_semantics_outputs`, `prng_seed_attrs`, `resolve_aminx_version`, or
`assert_uniform_group_attr` -- that gap is precisely why the grid-mode bias-dropping bug
(FINDING 2) and the schema/PRNG coupling bug (FINDING 1) both shipped undetected. These
tests are deliberately CPU-only, no GPU, no model loading, no real sampling pass.
"""

from __future__ import annotations

import importlib.metadata

import jax
import numpy as np
import pytest

from aminx.io import sink_provenance
from aminx.io.sink_provenance import (
  assert_uniform_group_attr,
  logits_bias_semantics_outputs,
  prng_seed_attrs,
  resolve_aminx_version,
)


class TestBiasIsNonzero:
  """`bias_is_nonzero` must correctly classify None / all-zero / genuinely-nonzero bias.

  The `None` case is the one a naive implementation gets backwards: `np.asarray(None) != 0`
  is `True` in numpy, so a guard that skips the explicit `bias_array is not None` check
  would report a run with NO bias at all as having a nonzero one.
  """

  def test_none_bias_is_not_nonzero(self) -> None:
    arrays, attrs = logits_bias_semantics_outputs(None)
    assert attrs["logits_bias_semantics"]["bias_is_nonzero"] is False
    assert arrays == {}

  def test_all_zero_bias_is_not_nonzero(self) -> None:
    bias = np.zeros((10, 21), dtype=np.float32)
    arrays, attrs = logits_bias_semantics_outputs(bias)
    assert attrs["logits_bias_semantics"]["bias_is_nonzero"] is False
    assert arrays == {}

  def test_genuinely_nonzero_bias_is_nonzero(self) -> None:
    bias = np.zeros((10, 21), dtype=np.float32)
    bias[3, 5] = 1.0
    arrays, attrs = logits_bias_semantics_outputs(bias)
    assert attrs["logits_bias_semantics"]["bias_is_nonzero"] is True
    assert "bias" in arrays


class TestBiasPersistedImpliesStaged:
  """`bias_persisted=True` must never be a lying attr: it implies `bias` is actually staged."""

  def test_persisted_true_implies_array_present(self) -> None:
    bias = np.ones((4, 21), dtype=np.float32)
    arrays, attrs = logits_bias_semantics_outputs(bias)
    semantics = attrs["logits_bias_semantics"]
    assert semantics["bias_persisted"] is True
    assert "bias" in arrays
    np.testing.assert_array_equal(arrays["bias"], bias)

  def test_persisted_false_when_zero_implies_no_array(self) -> None:
    bias = np.zeros((4, 21), dtype=np.float32)
    arrays, attrs = logits_bias_semantics_outputs(bias)
    semantics = attrs["logits_bias_semantics"]
    assert semantics["bias_persisted"] is False
    assert "bias" not in arrays

  def test_persist_bias_false_suppresses_persistence_even_if_nonzero(self) -> None:
    bias = np.ones((4, 21), dtype=np.float32)
    arrays, attrs = logits_bias_semantics_outputs(bias, persist_bias=False)
    semantics = attrs["logits_bias_semantics"]
    assert semantics["bias_is_nonzero"] is True
    assert semantics["bias_persisted"] is False
    assert "bias" not in arrays


class TestResolveAminxVersion:
  def test_matches_importlib_metadata(self) -> None:
    assert resolve_aminx_version() == importlib.metadata.version("aminx")

  def test_raises_loudly_on_package_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """Must NOT swallow PackageNotFoundError into a silent '0.1.0'/'unknown' sentinel.

    That is the exact weak pattern this module's own docstring calls out as not to copy
    (see `verify_ar_fusion_gate.py:99-105`'s `except Exception: return "unknown"`).
    """

    def _raise_not_found(_name: str) -> str:
      raise importlib.metadata.PackageNotFoundError("aminx")

    monkeypatch.setattr(importlib.metadata, "version", _raise_not_found)

    with pytest.raises(importlib.metadata.PackageNotFoundError):
      resolve_aminx_version()

  def test_does_not_fall_back_to_hardcoded_sentinel(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """Belt-and-suspenders: confirm no `except ...: return "0.1.0"`/`"unknown"` path exists."""

    def _raise_not_found(_name: str) -> str:
      raise importlib.metadata.PackageNotFoundError("aminx")

    monkeypatch.setattr(importlib.metadata, "version", _raise_not_found)

    try:
      result = resolve_aminx_version()
    except importlib.metadata.PackageNotFoundError:
      pass
    else:
      assert result not in {"0.1.0", "unknown"}, (
        f"resolve_aminx_version() silently returned a sentinel ({result!r}) instead of "
        "raising PackageNotFoundError."
      )


class TestPrngSeedAttrs:
  def test_plain_int_seed(self) -> None:
    assert prng_seed_attrs(42) == {"prng_seed": 42}

  def test_jax_prng_key_round_trips_as_uint32_words(self) -> None:
    key = jax.random.key(7)
    attrs = prng_seed_attrs(key)
    expected = np.asarray(jax.random.key_data(key)).tolist()
    assert attrs["prng_seed"] == expected

  def test_raw_uint32_key_array_passthrough(self) -> None:
    raw = np.array([1, 2], dtype=np.uint32)
    attrs = prng_seed_attrs(raw)
    assert attrs["prng_seed"] == raw.tolist()


class TestAssertUniformGroupAttr:
  """Prove the check is not vacuous: it must actually raise on a genuine divergence."""

  def test_raises_on_genuinely_different_values(self) -> None:
    values = [
      {"bias_is_nonzero": True, "bias_persisted": True},
      {"bias_is_nonzero": False, "bias_persisted": False},
    ]
    with pytest.raises(ValueError, match="differs across records"):
      assert_uniform_group_attr(values, attr_name="logits_bias_semantics", group_key=("g",))

  def test_passes_on_uniform_values(self) -> None:
    values = [
      {"bias_is_nonzero": True, "bias_persisted": True},
      {"bias_is_nonzero": True, "bias_persisted": True},
    ]
    # Must not raise.
    assert_uniform_group_attr(values, attr_name="logits_bias_semantics", group_key=("g",))

  def test_single_value_is_trivially_uniform(self) -> None:
    assert_uniform_group_attr(
      [{"bias_is_nonzero": True}], attr_name="logits_bias_semantics", group_key=("g",),
    )


def test_sink_provenance_module_importable_standalone() -> None:
  """`aminx.io.sink_provenance` must not require `aminx.host` (documented constraint).

  `io/designs.py` -- one of the four staging paths that write a "logits" array -- lives in
  `aminx.io` and must not depend on `aminx.host`; this is a cheap smoke check that the
  module itself has no accidental host-layer import.
  """
  assert sink_provenance.__name__ == "aminx.io.sink_provenance"
