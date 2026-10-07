"""Seed 0 is a real seed. None defaults to 42 once, at specification construction."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from aminx.run.spec import build_run_spec
from aminx.run.spec_json import (
  run_specification_from_json_dict,
  run_specification_to_json_dict,
)
from aminx.run.specs import SamplingSpecification


def test_build_run_spec_keeps_seed_zero_and_defaults_none() -> None:
  zero = SamplingSpecification(inputs="test.pdb", random_seed=0)
  assert zero.random_seed == 0
  assert zero.run_spec.sampling.random_seed == 0
  assert build_run_spec(zero).sampling.random_seed == 0

  defaulted = SamplingSpecification(inputs="test.pdb", random_seed=None)
  assert defaulted.random_seed == 42
  assert defaulted.run_spec.sampling.random_seed == 42


def test_random_seed_rejects_non_int() -> None:
  with pytest.raises(TypeError, match="random_seed"):
    SamplingSpecification(inputs="test.pdb", random_seed=1.5)  # type: ignore[arg-type]


def test_legacy_json_seed_zero_decodes_as_forty_two() -> None:
  encoded = run_specification_to_json_dict(SamplingSpecification(inputs="test.pdb", random_seed=0))
  encoded.pop("schema_version")
  encoded["random_seed"] = 0
  with pytest.warns(UserWarning, match="seed 42") as caught:
    decoded = run_specification_from_json_dict(encoded)
  assert len(caught) == 1
  message = str(caught[0].message)
  assert "seed 0" in message
  assert "decoding as 42" in message
  assert decoded.random_seed == 42
  assert decoded.run_spec.sampling.random_seed == 42


def test_schema_version_1_json_keeps_seed_zero() -> None:
  encoded = run_specification_to_json_dict(SamplingSpecification(inputs="test.pdb", random_seed=0))
  assert encoded["schema_version"] == 1
  assert encoded["random_seed"] == 0
  with warnings.catch_warnings():
    warnings.simplefilter("error", UserWarning)
    decoded = run_specification_from_json_dict(encoded)
  assert decoded.random_seed == 0
  assert decoded.run_spec.sampling.random_seed == 0


def test_legacy_json_nonzero_seed_is_unchanged() -> None:
  encoded = run_specification_to_json_dict(SamplingSpecification(inputs="test.pdb", random_seed=7))
  encoded.pop("schema_version")
  with warnings.catch_warnings():
    warnings.simplefilter("error", UserWarning)
    decoded = run_specification_from_json_dict(encoded)
  assert decoded.random_seed == 7


def test_src_has_no_or_42_seed_coalesce() -> None:
  """Score, inspect, and jacobian share this field; the scan covers sites those tests do not run."""
  root = Path(__file__).resolve().parents[2]
  hits: list[str] = []
  for folder in ("src", "scripts"):
    base = root / folder
    if not base.is_dir():
      continue
    for path in base.rglob("*.py"):
      if "or 42" in path.read_text(encoding="utf-8"):
        hits.append(str(path.relative_to(root)))
  assert hits == []


def test_random_seed_accepts_numpy_integer() -> None:
  """A derived numpy integer seed is accepted and normalised to a Python int."""
  import numpy as np  # noqa: PLC0415

  spec = SamplingSpecification(inputs="test.pdb", random_seed=np.int64(0))
  assert spec.random_seed == 0
  assert type(spec.random_seed) is int
