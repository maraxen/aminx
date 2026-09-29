# ruff: noqa: S101
"""B1.5 gate: the LASEr key inventory covers every checkpoint key (0 unmapped).

The three LASErMPNN checkpoints in ``VENDOR_PIN.toml`` share one state-dict key
set (T0.2). ``pretrained_ligand_encoder_weights.pt`` is the ligand-encoder
pretraining artifact and is not a ``LASErMPNN`` checkpoint, so it is not part of
this gate. Torch and the weight files are optional: the test skips when either
is absent (the orchestrator runs it on titanix).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_INVENTORY = _REPO / "tests" / "port" / "reference" / "laser_weights" / "key_inventory.toml"
_CONVERTER = _REPO / "scripts" / "recapture" / "lasermpnn_model_to_eqx.py"
# Env-overridable: the weights live on titanix, where the laptop path does not exist.
# A hardcoded path made this gate skip silently on the box that has the checkpoints.
_LASER_ROOT = Path(os.environ.get("AMINX_LASER_ROOT", "~/repos/LASErMPNN")).expanduser()


def _converter() -> object:
  spec = importlib.util.spec_from_file_location("lasermpnn_model_to_eqx", _CONVERTER)
  if spec is None or spec.loader is None:
    msg = f"cannot load {_CONVERTER}"
    raise ImportError(msg)
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def test_inventory_file_lists_a_destination_for_every_row() -> None:
  """The committed inventory is non-empty and every row has a destination."""
  converter = _converter()
  rows = converter.load_inventory(_INVENTORY)
  assert rows, "key inventory is empty"
  for upstream, row in rows.items():
    assert row["destination"], upstream
    assert row["dtype"], upstream
    assert row["kind"] in {"parameter", "buffer"}, upstream
    assert converter.destination_for(upstream) == row["destination"]


def test_pinned_laser_checkpoints_have_zero_unmapped_keys() -> None:
  """Every key of each LASErMPNN checkpoint is in the inventory, with shape and dtype."""
  torch = pytest.importorskip("torch")
  converter = _converter()
  inventory = converter.load_inventory(_INVENTORY)
  if not _LASER_ROOT.is_dir():
    pytest.skip(f"LASErMPNN checkout is not at {_LASER_ROOT}")
  pin = _LASER_ROOT / "VENDOR_PIN.toml"
  if not pin.is_file():
    pytest.skip(f"missing {pin}")
  seen = 0
  for checkpoint_id, relative in converter.LASER_CHECKPOINTS:
    path = _LASER_ROOT / relative
    if not path.is_file():
      pytest.skip(f"missing weights {path}")
    state = converter.load_state_dict(path)
    missing = sorted(set(state) - set(inventory))
    assert missing == [], (checkpoint_id, missing[:8], len(missing))
    extra = sorted(set(inventory) - set(state))
    assert extra == [], (checkpoint_id, extra[:8], len(extra))
    for key, value in state.items():
      row = inventory[key]
      shape = tuple(int(dim) for dim in row["shape"])
      assert tuple(value.shape) == shape, (checkpoint_id, key)
      assert str(value.dtype).removeprefix("torch.") == row["dtype"], (checkpoint_id, key)
    seen += 1
  assert seen == len(converter.LASER_CHECKPOINTS)
  del torch
