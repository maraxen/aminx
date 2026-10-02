# ruff: noqa: S101
"""LaserDriver seam: registration, alphabet boundary, and nll versus logits."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("prody", reason="needs the laser extra")

from aminx.families.laser_mpnn.driver import (
  ALPHABET,
  CANONICAL_OF_LASER,
  LASER_ALPHABET,
  LASER_OF_CANONICAL,
  LASErMPNN,
  LaserDriver,
)
from aminx.host.family_driver import FAMILY_DRIVERS
from aminx.host.runner import score
from aminx.run.options import LaserOptions
from aminx.run.specs import ScoringSpecification


def _atom(
  serial: int,
  name: str,
  resname: str,
  chain: str,
  resseq: int,
  x: float,
  y: float,
  z: float,
  *,
  element: str,
  het: bool = False,
) -> str:
  record = "HETATM" if het else "ATOM  "
  return (
    f"{record}{serial:5d} {name:>4} {resname:>3} {chain}"
    f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {element:>2}"
  )


def _write_pdb(path: Path) -> str:
  """Two alanines plus one ligand carbon so the ligand kNN is non-empty."""
  lines = [
    _atom(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, element="N"),
    _atom(2, "CA", "ALA", "A", 1, 1.5, 0.0, 0.0, element="C"),
    _atom(3, "C", "ALA", "A", 1, 2.5, 1.0, 0.0, element="C"),
    _atom(4, "O", "ALA", "A", 1, 2.2, 2.2, 0.0, element="O"),
    _atom(5, "CB", "ALA", "A", 1, 1.5, -1.4, 0.5, element="C"),
    _atom(6, "N", "ALA", "A", 2, 3.8, 1.2, 0.0, element="N"),
    _atom(7, "CA", "ALA", "A", 2, 5.0, 2.0, 0.0, element="C"),
    _atom(8, "C", "ALA", "A", 2, 6.2, 1.2, 0.4, element="C"),
    _atom(9, "O", "ALA", "A", 2, 6.4, 0.0, 0.2, element="O"),
    _atom(10, "CB", "ALA", "A", 2, 5.2, 3.2, 0.8, element="C"),
    _atom(11, "C1", "LIG", "B", 1, 8.0, 2.0, 1.0, element="C", het=True),
  ]
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return "AA"


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


def test_registration_and_handles(registered: LaserDriver, tmp_path: Path) -> None:
  import aminx.families.laser_mpnn as laser_mpnn

  driver = FAMILY_DRIVERS.get("lasermpnn")
  assert driver is registered
  assert driver is not None
  assert driver.name == "lasermpnn"
  assert driver.options_type is LaserOptions
  assert driver.mpnn_fallback_purposes == frozenset()
  assert laser_mpnn.LaserDriver is LaserDriver
  assert driver.handles(None, "score:nll")
  assert driver.handles(None, "score:logits")
  assert driver.handles(None, "score:proofread_unconditional")
  assert driver.handles(None, "score:proofread_conditional")
  for purpose in (
    "sample",
    "score:energy",
    "score:ddg",
    "jacobian",
    "inspect",
  ):
    assert not driver.handles(None, purpose)
  model = LASErMPNN(key=jax.random.PRNGKey(1))
  assert driver.mpnn_core(model) is None
  weights = tmp_path / "roundtrip.eqx"
  eqx.tree_serialise_leaves(weights, model)
  loaded = driver.load(
    ScoringSpecification(
      inputs="unused.pdb",
      model_family="lasermpnn",
      checkpoint_id="lasermpnn_test",
      model_local_path=weights,
      output_kind="nll",
      sequences_to_score=["A"],
    ),
  )
  assert isinstance(loaded, LASErMPNN)


def test_alphabet_boundary_is_a_bijection() -> None:
  laser_of = [int(index) for index in np.asarray(LASER_OF_CANONICAL)]
  canonical_of = [int(index) for index in np.asarray(CANONICAL_OF_LASER)]
  assert len(laser_of) == 21
  assert sorted(laser_of) == list(range(21))
  assert [canonical_of[index] for index in laser_of] == list(range(21))
  for canonical, laser in enumerate(laser_of):
    assert LASER_ALPHABET[laser] == ALPHABET[canonical]
    assert ALPHABET[canonical_of[laser]] == LASER_ALPHABET[laser]
  # E and Q are the letters whose LASEr slots are not the AlphaFold slots.
  assert LASER_ALPHABET.index("E") != ALPHABET.index("E")


def test_nll_matches_gathered_log_softmax(registered: LaserDriver, tmp_path: Path, model_path: Path) -> None:
  del registered
  pdb = tmp_path / "complex.pdb"
  native = _write_pdb(pdb)
  mutant = "AG"
  spec_kwargs = {
    "inputs": str(pdb),
    "model_family": "lasermpnn",
    "checkpoint_id": "lasermpnn_test",
    "model_local_path": model_path,
    "sequences_to_score": [native, mutant],
    "random_seed": 3,
  }
  nll = score(ScoringSpecification(output_kind="nll", **spec_kwargs))
  logits = score(ScoringSpecification(output_kind="logits", **spec_kwargs))
  nll_rows = np.asarray(nll["structures"]["0"]["arrays"]["nll"])
  logit_rows = np.asarray(logits["structures"]["0"]["arrays"]["logits"])
  assert nll_rows.shape[0] == 2
  assert logit_rows.shape == (2, nll_rows.shape[1], 21)
  log_prob = np.asarray(jax.nn.log_softmax(jnp.asarray(logit_rows), axis=-1))
  for row, sequence in enumerate((native, mutant)):
    index = np.asarray([ALPHABET.index(letter) for letter in sequence], dtype=np.int32)
    gathered = -log_prob[row, np.arange(index.shape[0]), index]
    np.testing.assert_allclose(nll_rows[row], gathered, rtol=0, atol=0)


def _proofread(model_path: Path, pdb: Path, *, dropout: bool, seed: int = 3) -> np.ndarray:
  """``proofread_mean`` for one tiny complex. Two orders, two reps."""
  result = score(
    ScoringSpecification(
      inputs=str(pdb),
      model_family="lasermpnn",
      checkpoint_id="lasermpnn_test",
      model_local_path=model_path,
      output_kind="proofread_conditional",
      sequences_to_score=["AA"],
      random_seed=seed,
      laser=LaserOptions(
        n_decoding_orders=2,
        n_dropouts=2,
        proofread_dropout=dropout,
      ),
    ),
  )
  return np.asarray(result["structures"]["0"]["arrays"]["proofread_mean"])


def test_proofread_conditional_runs_without_an_injected_mask_catalog(
  registered: LaserDriver,
  tmp_path: Path,
  model_path: Path,
) -> None:
  """Production proofreading injects no masks and must still run.

  Regression for debt #2422. The parity vehicle replays a recorded catalog, so
  a mask miss there is a broken replay and raises; production records nothing,
  so every call is a miss and the same raise took the whole purpose down. No
  test drove this path, which is why a raise on the default options survived.
  """
  del registered
  pdb = tmp_path / "complex.pdb"
  _write_pdb(pdb)
  mean = _proofread(model_path, pdb, dropout=True)
  assert mean.ndim >= 2
  assert np.isfinite(mean).all()


def test_proofread_dropout_actually_perturbs_the_ensemble(
  registered: LaserDriver,
  tmp_path: Path,
  model_path: Path,
) -> None:
  """Dropout on and off must disagree at one seed.

  The seed fixes the decoding orders and the inverse-CDF draws, so the only
  thing left to differ is the keep-masks. An all-ones keep -- the shape a
  missing mask used to take -- would make these two runs identical, which is
  the silent failure a bare does-it-crash check cannot see.
  """
  del registered
  pdb = tmp_path / "complex.pdb"
  _write_pdb(pdb)
  live = _proofread(model_path, pdb, dropout=True)
  off = _proofread(model_path, pdb, dropout=False)
  assert live.shape == off.shape
  assert not np.allclose(live, off)
  # Same seed twice is the same answer: the draws are keyed, not ambient.
  np.testing.assert_array_equal(live, _proofread(model_path, pdb, dropout=True))


def test_proofread_unconditional_matches_its_declared_schema(
  registered: LaserDriver,
  tmp_path: Path,
  model_path: Path,
) -> None:
  """The cheaper proofread purpose carried the same stray candidate axis.

  One forward pass, no dropout ensemble, so this fails on the result-schema
  mismatch alone rather than on anything to do with mask replay.
  """
  del registered
  pdb = tmp_path / "complex.pdb"
  _write_pdb(pdb)
  result = score(
    ScoringSpecification(
      inputs=str(pdb),
      model_family="lasermpnn",
      checkpoint_id="lasermpnn_test",
      model_local_path=model_path,
      output_kind="proofread_unconditional",
      sequences_to_score=["AA"],
      random_seed=3,
    ),
  )
  arrays = result["structures"]["0"]["arrays"]
  mean = np.asarray(arrays["proofread_mean"])
  ids = np.asarray(arrays["residue_ids"])
  assert mean.ndim == 2
  assert mean.shape[-1] == 21
  assert ids.shape == (mean.shape[0],)
  assert np.isfinite(mean).all()
  # Softmax rows, so each focus row sums to one.
  np.testing.assert_allclose(mean.sum(axis=-1), np.ones(mean.shape[0]), atol=1e-5)
