"""Contract tests for the FAMPNN driver that need no checkpoint weights.

These pin the driver's own decisions: which checkpoint id a run resolves, which spec fields it
refuses rather than dropping, and that the logits it returns belong to the sequence it returns.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import aminx.families.fampnn.driver as driver_mod
from aminx.families.fampnn.driver import FAMPNNDriver
from aminx.run.options import FAMPNNOptions


def _spec(**fields: Any) -> SimpleNamespace:  # noqa: ANN401
  base: dict[str, Any] = {
    "checkpoint_id": None,
    "model_local_path": None,
    "checkpoint_registry_path": None,
    "fampnn": None,
    "fixed_positions": None,
    "fixed_mask": None,
    "fixed_tokens": None,
  }
  base.update(fields)
  return SimpleNamespace(**base)


@pytest.fixture
def resolved(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
  """Stub the artifact resolver and the loader; record the checkpoint id each one was given."""
  calls: dict[str, Any] = {}

  def fake_resolve(spec: Any, family: str, checkpoint_id: str | None = None) -> str:  # noqa: ANN401, ARG001
    calls["resolved_id"] = checkpoint_id
    return "/nowhere/fampnn.eqx.zst"

  def fake_load(path: Any, checkpoint_id: str) -> str:  # noqa: ANN401
    calls["loaded_id"] = checkpoint_id
    return "MODEL"

  monkeypatch.setattr(driver_mod, "_resolve_driver_checkpoint", fake_resolve)
  monkeypatch.setattr(driver_mod, "_load_fampnn_eqx", fake_load)
  return calls


def test_default_options_and_unset_spec_resolve_the_default_checkpoint(resolved: dict[str, Any]) -> None:
  FAMPNNDriver().load(_spec(fampnn=FAMPNNOptions()))

  assert resolved["resolved_id"] == "fampnn_0_3"
  assert resolved["loaded_id"] == "fampnn_0_3"


def test_options_checkpoint_id_is_used_when_spec_is_unset(resolved: dict[str, Any]) -> None:
  FAMPNNDriver().load(_spec(fampnn=FAMPNNOptions(checkpoint_id="fampnn_0_0")))

  assert resolved["resolved_id"] == "fampnn_0_0"
  assert resolved["loaded_id"] == "fampnn_0_0"


def test_spec_checkpoint_id_is_used_when_options_are_unset(resolved: dict[str, Any]) -> None:
  FAMPNNDriver().load(_spec(checkpoint_id="fampnn_0_3_cath", fampnn=FAMPNNOptions(checkpoint_id=None)))

  assert resolved["resolved_id"] == "fampnn_0_3_cath"
  assert resolved["loaded_id"] == "fampnn_0_3_cath"


def test_equal_spec_and_options_checkpoint_ids_are_accepted(resolved: dict[str, Any]) -> None:
  FAMPNNDriver().load(
    _spec(checkpoint_id="fampnn_0_0", fampnn=FAMPNNOptions(checkpoint_id="fampnn_0_0")),
  )

  assert resolved["loaded_id"] == "fampnn_0_0"


def test_conflicting_spec_and_options_checkpoint_ids_raise(resolved: dict[str, Any]) -> None:
  with pytest.raises(ValueError, match="conflicting FAMPNN checkpoint ids"):
    FAMPNNDriver().load(
      _spec(checkpoint_id="fampnn_0_0", fampnn=FAMPNNOptions(checkpoint_id="fampnn_0_3_cath")),
    )

  assert "loaded_id" not in resolved
