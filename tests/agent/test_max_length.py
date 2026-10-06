"""S7-08: MCP tools fit ``max_length`` to the parsed inputs instead of padding to 512."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("cisternal")
pytest.importorskip("fastmcp")

from aminx.agent.tools import MAX_LENGTH_BUCKET, fitted_max_length


def _spec(inputs: list[str], *, max_length: int = 512, pass_mode: str = "intra") -> SimpleNamespace:
  """Stand-in carrying the three fields the fitter reads."""
  return SimpleNamespace(inputs=inputs, max_length=max_length, pass_mode=pass_mode)


def test_single_structure_rounds_up_to_bucket() -> None:
  """76 residues pad to 128, not the 512 default."""
  assert fitted_max_length(_spec(["a.pdb"]), {"a": 76}, explicit=False) == 128


def test_exact_bucket_is_kept() -> None:
  """A length already on a bucket boundary is not rounded further."""
  assert fitted_max_length(_spec(["a.pdb"]), {"a": 128}, explicit=False) == 128


def test_intra_uses_longest_structure() -> None:
  """Independent structures share one padded shape sized by the longest."""
  spec = _spec(["a.pdb", "b.pdb"])
  assert fitted_max_length(spec, {"a": 40, "b": 130}, explicit=False) == 192


def test_inter_uses_summed_length() -> None:
  """``inter`` joins inputs, so the fit covers their sum and truncates nothing."""
  spec = _spec(["a.pdb", "b.pdb"], pass_mode="inter")
  assert fitted_max_length(spec, {"a": 100, "b": 100}, explicit=False) == 256


def test_never_below_measured_residues() -> None:
  """Every fitted value covers the residues it measured."""
  for count in range(1, 400, 7):
    fitted = fitted_max_length(_spec(["a.pdb"]), {"a": count}, explicit=False)
    assert fitted is not None
    assert fitted >= count
    assert fitted % MAX_LENGTH_BUCKET == 0


def test_explicit_max_length_is_kept() -> None:
  """A caller-set ``max_length`` is never replaced."""
  assert fitted_max_length(_spec(["a.pdb"], max_length=512), {"a": 76}, explicit=True) is None


def test_unparsed_input_keeps_spec_value() -> None:
  """If any input has no measured length the spec is left alone."""
  spec = _spec(["a.pdb", "b.pdb"])
  assert fitted_max_length(spec, {"a": 76}, explicit=False) is None
  assert fitted_max_length(spec, {}, explicit=False) is None


def test_never_grows_past_spec_value() -> None:
  """A structure longer than the default keeps the default (existing truncation behaviour)."""
  assert fitted_max_length(_spec(["a.pdb"]), {"a": 600}, explicit=False) is None
