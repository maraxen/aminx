"""Contract tests for the Caliby driver's checkpoint-id resolution that need no checkpoint weights.

These pin which checkpoint id a run resolves when the id is set on the spec, on the Caliby
options, on both, or on neither, and that conflicting values are refused. They mirror
``test_fampnn_driver_contract.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import aminx.families.caliby.driver as driver_mod
from aminx.families.caliby.driver import CalibyDriver, _is_packer_checkpoint
from aminx.run.options import CalibyOptions


def _spec(**fields: Any) -> SimpleNamespace:  # noqa: ANN401
  base: dict[str, Any] = {
    "checkpoint_id": None,
    "model_local_path": None,
    "checkpoint_registry_path": None,
    "caliby": None,
  }
  base.update(fields)
  return SimpleNamespace(**base)


@pytest.fixture
def resolved(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
  """Stub the artifact resolver and the loader; record the checkpoint id each one was given."""
  calls: dict[str, Any] = {}

  def fake_resolve(spec: Any, family: str, checkpoint_id: str | None = None) -> str:  # noqa: ANN401, ARG001
    calls["resolved_id"] = checkpoint_id
    return "/nowhere/caliby.eqx.zst"

  def fake_load(path: Any, checkpoint_id: str) -> str:  # noqa: ANN401
    calls["loaded_id"] = checkpoint_id
    return "MODEL"

  monkeypatch.setattr(driver_mod, "_resolve_driver_checkpoint", fake_resolve)
  monkeypatch.setattr(driver_mod, "_load_caliby_eqx", fake_load)
  return calls


def test_default_options_and_unset_spec_resolve_the_potts_default(resolved: dict[str, Any]) -> None:
  CalibyDriver().load(_spec(caliby=CalibyOptions()))

  assert resolved["resolved_id"] == "caliby"
  assert resolved["loaded_id"] == "caliby"


def test_options_checkpoint_id_is_used_when_spec_is_unset(resolved: dict[str, Any]) -> None:
  CalibyDriver().load(_spec(caliby=CalibyOptions(checkpoint_id="caliby_distill")))

  assert resolved["resolved_id"] == "caliby_distill"
  assert resolved["loaded_id"] == "caliby_distill"


def test_spec_checkpoint_id_is_used_when_options_are_unset(resolved: dict[str, Any]) -> None:
  CalibyDriver().load(_spec(checkpoint_id="soluble_caliby", caliby=CalibyOptions(checkpoint_id=None)))

  assert resolved["resolved_id"] == "soluble_caliby"
  assert resolved["loaded_id"] == "soluble_caliby"


def test_spec_checkpoint_id_is_used_when_options_are_default(resolved: dict[str, Any]) -> None:
  """A spec id that differs from the Caliby options default is not a conflict."""
  CalibyDriver().load(_spec(checkpoint_id="caliby_distill", caliby=CalibyOptions()))

  assert resolved["loaded_id"] == "caliby_distill"


def test_equal_spec_and_options_checkpoint_ids_are_accepted(resolved: dict[str, Any]) -> None:
  CalibyDriver().load(
    _spec(checkpoint_id="caliby_distill", caliby=CalibyOptions(checkpoint_id="caliby_distill")),
  )

  assert resolved["loaded_id"] == "caliby_distill"


def test_conflicting_spec_and_options_checkpoint_ids_raise(resolved: dict[str, Any]) -> None:
  with pytest.raises(ValueError, match="conflicting Caliby checkpoint ids"):
    CalibyDriver().load(
      _spec(checkpoint_id="caliby", caliby=CalibyOptions(checkpoint_id="caliby_distill")),
    )

  assert "loaded_id" not in resolved


def test_packer_selection_follows_the_resolved_id() -> None:
  assert _is_packer_checkpoint(_spec(caliby=CalibyOptions(checkpoint_id="caliby_packer_010")))
  assert _is_packer_checkpoint(_spec(checkpoint_id="caliby_packer_030", caliby=CalibyOptions()))
  assert not _is_packer_checkpoint(_spec(caliby=CalibyOptions()))
