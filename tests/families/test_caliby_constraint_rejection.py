# ruff: noqa: S101
"""Caliby refuses constraint fields it never reads instead of silently ignoring them."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aminx.families.caliby.driver import _UNSUPPORTED_CONSTRAINT_FIELDS, CalibyDriver

_SET_VALUES = {
  "fixed_positions": (1, 2),
  "fixed_tokens": {1: "A"},
  "fixed_mask": [True, False],
  "bias": {"A": 1.0},
  "omit_aa": ("C",),
  "omit_aa_per_position": {1: "C"},
}


def test_rejected_fields_cover_the_table() -> None:
  assert set(_SET_VALUES) == set(_UNSUPPORTED_CONSTRAINT_FIELDS)


@pytest.mark.parametrize("field", _UNSUPPORTED_CONSTRAINT_FIELDS)
def test_set_constraint_raises_naming_the_option(field: str) -> None:
  spec = SimpleNamespace(checkpoint_id="caliby", **{field: _SET_VALUES[field]})
  with pytest.raises(ValueError, match=field):
    CalibyDriver().load(spec)


@pytest.mark.parametrize("field", _UNSUPPORTED_CONSTRAINT_FIELDS)
def test_empty_constraint_is_not_rejected(field: str) -> None:
  from aminx.families.caliby.driver import _reject_unsupported_constraints  # noqa: PLC0415

  _reject_unsupported_constraints(SimpleNamespace(**{field: ()}))
  _reject_unsupported_constraints(SimpleNamespace(**{field: None}))
