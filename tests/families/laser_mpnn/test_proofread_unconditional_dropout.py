# ruff: noqa: S101
"""Unconditional proofread honours ``LaserOptions.proofread_dropout``.

Debt #2424 dropped the per-structure dropout key and called
``unconditional_logits`` outside the plan, so two seeds could not disagree.
(a) is that regression: it fails on the base commit.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest

pytest.importorskip("prody", reason="needs the laser extra")

from aminx.families.laser_mpnn.driver import LASErMPNN, LaserDriver
from aminx.host.family_driver import FAMILY_DRIVERS
from aminx.host.runner import score
from aminx.run.options import LaserOptions
from aminx.run.specs import ScoringSpecification
from tests.families.laser_mpnn.test_driver import _write_pdb


@pytest.fixture
def registered() -> Iterator[LaserDriver]:
  driver = LaserDriver()
  FAMILY_DRIVERS.register("lasermpnn")(driver)
  yield driver
  FAMILY_DRIVERS.discard("lasermpnn")


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
  path = tmp_path / "tiny.eqx"
  eqx.tree_serialise_leaves(path, LASErMPNN(key=jax.random.PRNGKey(0)))
  return path


def _unconditional(model_path: Path, pdb: Path, *, dropout: bool, seed: int) -> np.ndarray:
  """``proofread_mean`` for one tiny complex, one unconditional forward."""
  result = score(
    ScoringSpecification(
      inputs=str(pdb),
      model_family="lasermpnn",
      checkpoint_id="lasermpnn_test",
      model_local_path=model_path,
      output_kind="proofread_unconditional",
      sequences_to_score=["AA"],
      random_seed=seed,
      laser=LaserOptions(proofread_dropout=dropout),
    ),
  )
  return np.asarray(result["structures"]["0"]["arrays"]["proofread_mean"])


def test_unconditional_proofread_dropout_follows_the_seed(
  registered: LaserDriver,
  tmp_path: Path,
  model_path: Path,
) -> None:
  """(a) dropout on differs across seeds, (b) off does not, (c) one seed repeats.

  The structure and the option are fixed, so the only seed-dependent state in
  the unconditional forward is the dropout key. An all-ones keep would make
  (a) identical, which is the #2424 defect.
  """
  del registered
  pdb = tmp_path / "complex.pdb"
  _write_pdb(pdb)
  first = _unconditional(model_path, pdb, dropout=True, seed=1)
  second = _unconditional(model_path, pdb, dropout=True, seed=2)
  assert first.shape == second.shape
  assert first.shape[-1] == 21
  assert not np.allclose(first, second)
  off_a = _unconditional(model_path, pdb, dropout=False, seed=1)
  off_b = _unconditional(model_path, pdb, dropout=False, seed=2)
  np.testing.assert_array_equal(off_a, off_b)
  # Same seed twice is the same answer: the keeps are keyed, not ambient.
  np.testing.assert_array_equal(first, _unconditional(model_path, pdb, dropout=True, seed=1))
