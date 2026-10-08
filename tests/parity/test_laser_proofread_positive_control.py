"""Positive-control wiring for laser_proofread_parity.

The four negative controls must keep the flags they had before this arm
existed. ``scalar_both_off`` is the one configuration in which both sides
leave scalar dropout off.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from scripts.parity.laser_proofread_parity import (
  MUTANTS,
  POSITIVE_ARM,
  ArmPlan,
  _oracle_command,
  _replay_paths,
  _touch_scalar_dropout,
  arm_plan,
  scalar_both_off_plan,
  split_mutants,
)


class _Drop(torch.nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.scalar = torch.nn.Dropout(0.1)
    self.idle = torch.nn.Dropout(0.0)


def test_positive_control_docstring_states_the_instrument_floor() -> None:
  text = scalar_both_off_plan.__doc__ or ""
  assert "must report near-zero" in text
  assert "non-zero result means instrument floor rather than a port defect" in text


def test_negative_controls_keep_their_flags() -> None:
  assert arm_plan("reduction_swap") == ArmPlan(
    scalar=True,
    vector=False,
    ddof=1,
    std_over="reps",
    scalar_eval=False,
  )
  assert arm_plan("ddof_0") == ArmPlan(
    scalar=True,
    vector=False,
    ddof=0,
    std_over="orders",
    scalar_eval=False,
  )
  # scalar_off disables aminx only. Upstream stays on the train-mode capture.
  assert arm_plan("scalar_off") == ArmPlan(
    scalar=False,
    vector=False,
    ddof=1,
    std_over="orders",
    scalar_eval=False,
  )
  assert arm_plan("vector_on") == ArmPlan(
    scalar=True,
    vector=True,
    ddof=1,
    std_over="orders",
    scalar_eval=False,
  )
  assert arm_plan("clean") == ArmPlan(
    scalar=True,
    vector=False,
    ddof=1,
    std_over="orders",
    scalar_eval=False,
  )


def test_positive_control_disables_scalar_dropout_on_both_sides() -> None:
  plan = arm_plan(POSITIVE_ARM)
  assert plan == scalar_both_off_plan()
  assert plan.scalar is False
  assert plan.vector is False
  assert plan.ddof == 1
  assert plan.std_over == "orders"
  assert plan.scalar_eval is True


def test_positive_arm_is_not_a_listed_negative() -> None:
  negatives, run_positive = split_mutants(",".join(MUTANTS))
  assert negatives == list(MUTANTS)
  assert run_positive is False
  combined, requested = split_mutants(",".join((*MUTANTS, POSITIVE_ARM)))
  assert combined == list(MUTANTS)
  assert requested is True
  with pytest.raises(SystemExit, match="unknown mutant"):
    split_mutants("not_a_control")


def test_replay_paths_leave_the_train_capture_in_place(tmp_path: Path) -> None:
  assert _replay_paths(tmp_path, scalar_eval=False) == (
    tmp_path / "masks.npz",
    tmp_path / "masks.json",
    tmp_path / "draws.npz",
    tmp_path / "draws.json",
  )
  assert _replay_paths(tmp_path, scalar_eval=True)[2:] == (
    tmp_path / "draws_eval.npz",
    tmp_path / "draws_eval.json",
  )


def test_eval_flag_does_not_train_dropout_and_train_flag_still_does() -> None:
  model = _Drop()
  model.eval()
  assert _touch_scalar_dropout(model, scalar_eval=True) == []
  assert model.scalar.training is False
  assert model.idle.training is False
  touched = _touch_scalar_dropout(model, scalar_eval=False)
  assert touched == [model.scalar]
  assert model.scalar.training is True
  assert model.idle.training is False
  model.train()
  with pytest.raises(RuntimeError, match="stay in eval"):
    _touch_scalar_dropout(model, scalar_eval=True)


def test_train_oracle_command_omits_scalar_eval(tmp_path: Path) -> None:
  args = argparse.Namespace(laser_root=tmp_path, oracle_python="python")
  train = _oracle_command(
    args,
    script="laser_proofread_parity.py",
    checkpoint=tmp_path / "weights.pt",
    job_path=tmp_path / "job.json",
    upstream_path=tmp_path / "upstream.json",
    work=tmp_path,
    scalar_eval=False,
  )
  positive = _oracle_command(
    args,
    script="laser_proofread_parity.py",
    checkpoint=tmp_path / "weights.pt",
    job_path=tmp_path / "job.json",
    upstream_path=tmp_path / "upstream_eval.json",
    work=tmp_path,
    scalar_eval=True,
  )
  assert "--scalar-eval" not in train
  assert train[train.index("--upstream") + 1].endswith("upstream.json")
  assert positive[-1] == "--scalar-eval"
  assert positive[positive.index("--upstream") + 1].endswith("upstream_eval.json")
