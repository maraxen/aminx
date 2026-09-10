"""Tests for aminx.io.designs.DesignZarrWriter."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np
import pytest
import zarr

from aminx.io import designs as designs_module
from aminx.io.designs import DesignMetadata, DesignPayload, DesignZarrWriter
from aminx.io.sink_provenance import logits_bias_semantics_outputs

N_CANONICAL = 4
N_STATES = 2


def _payload(seq_value: int = 1) -> DesignPayload:
  metadata: DesignMetadata = {
    "pool_type": "BackboneOnly",
    "state_mapping": [0, 1],
    "weight_strategy": "uniform",
    "combination_algorithm": "none",
    "structure_ids": ["struct_0"],
    "parent_structure_idx": 0,
  }
  return {
    "sequence": np.full(N_CANONICAL, seq_value, dtype=np.uint8),
    "logits": np.zeros((N_CANONICAL, 21), dtype=np.float32),
    "scores": np.array([0.5], dtype=np.float32),
    "state_weights": np.ones(N_STATES, dtype=np.float32) / N_STATES,
    "metadata": metadata,
  }


def _writer(tmp_path: Path) -> DesignZarrWriter:
  return DesignZarrWriter.from_multistate_shapes(
    str(tmp_path / "designs.zarr"),
    n_canonical=N_CANONICAL,
    n_states=N_STATES,
  )


def test_write_and_close_persists_to_zarr(tmp_path: Path) -> None:
  writer = _writer(tmp_path)
  writer.write((0,), _payload(seq_value=7))
  writer.close()

  root = zarr.open_group(str(tmp_path / "designs.zarr"), mode="r")
  group = root["0"]
  assert np.array_equal(group["sequence"][:], np.full(N_CANONICAL, 7, dtype=np.uint8))
  assert group["logits"].dtype == np.float16
  assert group["scores"][:] == pytest.approx([0.5])
  assert group.attrs["pool_type"] == "BackboneOnly"
  assert group.attrs["parent_structure_idx"] == 0


def test_context_manager_drains_on_exit(tmp_path: Path) -> None:
  with _writer(tmp_path) as writer:
    writer.write((0,), _payload())

  root = zarr.open_group(str(tmp_path / "designs.zarr"), mode="r")
  assert "0" in root


def test_multiple_designs_get_distinct_keys(tmp_path: Path) -> None:
  writer = _writer(tmp_path)
  writer.write((0,), _payload(seq_value=1))
  writer.write((1,), _payload(seq_value=2))
  writer.close()

  root = zarr.open_group(str(tmp_path / "designs.zarr"), mode="r")
  assert np.array_equal(root["0"]["sequence"][:], np.full(N_CANONICAL, 1, dtype=np.uint8))
  assert np.array_equal(root["1"]["sequence"][:], np.full(N_CANONICAL, 2, dtype=np.uint8))


def test_wrong_sequence_shape_raises(tmp_path: Path) -> None:
  writer = _writer(tmp_path)
  bad = _payload()
  bad["sequence"] = np.zeros(N_CANONICAL + 1, dtype=np.uint8)
  with pytest.raises(AssertionError, match="sequence shape"):
    writer.write((0,), bad)


def test_out_of_range_logits_raise(tmp_path: Path) -> None:
  writer = _writer(tmp_path)
  bad = _payload()
  bad["logits"] = np.full((N_CANONICAL, 21), 1e5, dtype=np.float32)
  with pytest.raises(AssertionError, match="float16-safe range"):
    writer.write((0,), bad)


def test_nested_structure_key_becomes_zarr_group_path(tmp_path: Path) -> None:
  writer = _writer(tmp_path)
  writer.write(("structure_3", "sample_0"), _payload())
  writer.close()

  root = zarr.open_group(str(tmp_path / "designs.zarr"), mode="r")
  assert "sequence" in root["structure_3"]["sample_0"]


class TestAminxVersionResolutionFindingE:
  """FINDING E (code-review round, PR #154): construction must fail ACTIONABLY on a

  bare-source/vendored aminx import, and support an explicit opt-in override instead of
  a silent `except Exception: return "unknown"` fallback.
  """

  def test_construction_raises_actionable_error_without_distribution_metadata(
    self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """Simulates a bare-source/vendored aminx import: no `.dist-info` to resolve.

    Patches the underlying `importlib.metadata.version` (NOT `resolve_aminx_version`
    itself) so this exercises the REAL `resolve_aminx_version` -- including its
    actionable-message wrapping -- as actually called from `DesignZarrWriter.__init__`,
    rather than substituting a stand-in that bypasses the code under test.
    """

    def _raise_not_found(_name: str) -> str:
      raise importlib.metadata.PackageNotFoundError("aminx")

    monkeypatch.setattr(importlib.metadata, "version", _raise_not_found)

    with pytest.raises(importlib.metadata.PackageNotFoundError) as exc_info:
      DesignZarrWriter.from_multistate_shapes(
        str(tmp_path / "designs.zarr"), n_canonical=N_CANONICAL, n_states=N_STATES,
      )

    message = str(exc_info.value)
    assert "bare-source" in message or "vendored" in message, (
      f"DesignZarrWriter construction raised a non-actionable message: {message!r}"
    )

  def test_explicit_aminx_version_override_bypasses_resolution(
    self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """The explicit opt-in: passing `aminx_version=` must skip `resolve_aminx_version()`

    entirely (construction must not raise even though resolution would), and the writer
    must stamp exactly the caller-supplied value, unmodified.
    """

    def _fail_if_called() -> str:
      msg = "resolve_aminx_version() must not be called when aminx_version= is supplied"
      raise AssertionError(msg)

    monkeypatch.setattr(designs_module, "resolve_aminx_version", _fail_if_called)

    writer = DesignZarrWriter.from_multistate_shapes(
      str(tmp_path / "designs.zarr"),
      n_canonical=N_CANONICAL,
      n_states=N_STATES,
      aminx_version="0.1.0a999-vendored-override",
    )
    writer.write((0,), _payload())
    writer.close()

    root = zarr.open_group(str(tmp_path / "designs.zarr"), mode="r")
    assert root["0"].attrs["aminx_version"] == "0.1.0a999-vendored-override"


class TestLogitsBiasSemanticsContract:
  """Code-review round 2 on PR #154: `write()` must refuse a bias_persisted claim it
  cannot possibly satisfy.
  """

  def test_write_refuses_bias_persisted_claim(self, tmp_path: Path) -> None:
    """`DesignZarrWriter` stages no `bias` array and has no root group, so
    `bias_persisted: True` is unconditionally a lie here.

    Uses the REAL `logits_bias_semantics_outputs` with a nonzero bias -- i.e. exactly what
    a caller doing the obvious thing produces, since `persist_bias` defaults to True. That
    is what makes this a real trap rather than a hypothetical one.
    """
    _, attrs = logits_bias_semantics_outputs(np.ones((4, 21), dtype=np.float32))
    semantics = attrs["logits_bias_semantics"]
    assert semantics["bias_persisted"] is True, "precondition: the default must claim True"

    writer = DesignZarrWriter.from_multistate_shapes(
      str(tmp_path / "designs.zarr"),
      n_canonical=N_CANONICAL,
      n_states=N_STATES,
      aminx_version="0.1.0a999-test",
    )
    with pytest.raises(ValueError, match="bias_persisted"):
      writer.write((0,), _payload(), logits_bias_semantics=semantics)

  def test_write_accepts_non_persisting_semantics(self, tmp_path: Path) -> None:
    """The documented escape hatch: `persist_bias=False` describes the semantics without
    claiming an array exists, and must be staged verbatim.
    """
    _, attrs = logits_bias_semantics_outputs(
      np.ones((4, 21), dtype=np.float32), persist_bias=False,
    )
    semantics = attrs["logits_bias_semantics"]
    assert semantics["bias_persisted"] is False
    assert semantics["bias_array_group"] is None

    writer = DesignZarrWriter.from_multistate_shapes(
      str(tmp_path / "designs.zarr"),
      n_canonical=N_CANONICAL,
      n_states=N_STATES,
      aminx_version="0.1.0a999-test",
    )
    writer.write((0,), _payload(), logits_bias_semantics=semantics)
    writer.close()

    root = zarr.open_group(str(tmp_path / "designs.zarr"), mode="r")
    stored = root["0"].attrs["logits_bias_semantics"]
    assert stored["bias_persisted"] is False
    # `None` must survive the zarr attrs JSON round-trip as None, not "None"/absent --
    # the reviewer flagged this as unverified.
    assert stored["bias_array_group"] is None
    # And it must still carry the real information: the bias WAS nonzero.
    assert stored["bias_is_nonzero"] is True
