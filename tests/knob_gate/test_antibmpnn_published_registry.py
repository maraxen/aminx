# ruff: noqa: S101
"""The four published AntiBMPNN weight blobs recorded by antibmpnn_parity_v3 resolve in the registry."""

from __future__ import annotations

import pytest

from knob_gate import _coverage

_PUBLISHED = (
  "daef630d7ef35e862450cdc81ccdcd7ab94ecc99b36c006fe336dcd5483b2bae",
  "d9c4f2343b18e854f2a4565ab7250fc9e2538b981e5538d55af1c4bc63c16414",
  "90b2554dc20e550acca8f4e6d864264638c2fad3798e2b55d8a489a144fa0e58",
  "5c155208f707cee0d6b7733f897c503cb379c6b8a31a4bc9ccf3e7f01aa04abf",
)


@pytest.mark.parametrize("sha", _PUBLISHED)
def test_published_antibmpnn_blob_is_registered(sha: str) -> None:
  assert _coverage.default_registry_sha256(sha) == sha


def test_unregistered_blob_is_not_found() -> None:
  assert _coverage.default_registry_sha256("0" * 64) is None
