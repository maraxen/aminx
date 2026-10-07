"""Campaign rows record git provenance without hashing it."""

from __future__ import annotations

from pathlib import Path

import pytest

from aminx.host.campaign import plan_campaign_manifest
from aminx.run.specs import SamplingSpecification

_SHA = "a" * 40


def _fake_capture(*_args: object, **_kwargs: object) -> dict[str, object]:
  """Return a fixed provenance record."""
  return {
    "sha": _SHA,
    "dirty": False,
    "branch": "test",
    "provenance_source": "builtin",
  }


def _spec() -> SamplingSpecification:
  """Return the small sampling spec used by the campaign manifest tests."""
  return SamplingSpecification(
    inputs=["/tmp/test.pdb"],
    checkpoint_id="ckpt_001",
    model_family="ligandmpnn",
    temperature=[1.0],
    backbone_noise=[0.0],
  )


def _plan(tmp_path: Path, **kwargs: object) -> list[dict[str, object]]:
  """Plan one chunk of a campaign."""
  return plan_campaign_manifest(
    base_spec=_spec(),
    campaign_id="prov",
    designs_per_library_type=1,
    samples_chunk_size=1,
    output_root=tmp_path,
    fixed_arms=None,
    state_weight_profiles=("equal",),
    **kwargs,  # type: ignore[arg-type]
  )


def test_capture_fills_git_sha_and_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """A missing git_sha is taken from capture(), including its source."""
  monkeypatch.setattr("aminx.agent.provenance.capture", _fake_capture)
  rows = _plan(tmp_path)
  assert rows
  for row in rows:
    assert row["git_sha"] == _SHA
    assert row["git_provenance_source"] == "builtin"


def test_explicit_git_sha_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """A caller-supplied git_sha is kept and marked as an argument."""
  monkeypatch.setattr("aminx.agent.provenance.capture", _fake_capture)
  rows = _plan(tmp_path, git_sha="deadbeef")
  assert rows
  for row in rows:
    assert row["git_sha"] == "deadbeef"
    assert row["git_provenance_source"] == "argument"


def test_manifest_row_hash_ignores_git_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Captured and explicit shas produce identical manifest_row_hash values."""
  monkeypatch.setattr("aminx.agent.provenance.capture", _fake_capture)
  captured = _plan(tmp_path)
  explicit = _plan(tmp_path / "explicit", git_sha="unknown")
  assert [row["manifest_row_hash"] for row in captured] == [row["manifest_row_hash"] for row in explicit]
  assert captured[0]["git_sha"] != explicit[0]["git_sha"]
  assert captured[0]["git_provenance_source"] != explicit[0]["git_provenance_source"]
