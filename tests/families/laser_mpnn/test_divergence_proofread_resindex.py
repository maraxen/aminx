# ruff: noqa: S101
"""Spec §6.5b: proofread reports ProDy ``resindex``, not the batch row.

``run_proofreading.py:69-79,120`` uses a residue's ProDy ``resindex`` directly
as an index into the batch. ``resindex`` counts EVERY residue of the parsed
``AtomGroup`` -- ligands, waters, non-amino and dropped residues included --
while the batch row counts only the protein rows that were featurized. Upstream
assumes the two are equal. They are equal on ``4jnj-1_prot.pdb`` and on every
parity fixture, which is why the assumption survives; they stop being equal the
moment any non-row residue appears BEFORE the focus residue in file order, and
upstream then fixes or selects the wrong residue, or raises ``IndexError``.

aminx emits ``row_to_resindex`` from B0 and maps through it
(``featurize.py:1304``, consumed at ``driver.py:377`` and ``:422``).

THE FIXTURE IS A PAIR THAT DIFFERS ONLY IN RECORD ORDER. The ligand's
coordinates are identical in both files; only its position in the file moves.
MEASURED with a throwaway probe before this test was written:

    ligand last   row_to_resindex = [0, 1]   contacts = [False, True]
    ligand first  row_to_resindex = [1, 2]   contacts = [False, True]

So the geometry, the contact set and therefore the focus rows are unchanged,
and the only thing that moves is ``resindex``. That isolation is the whole
point: a fixture that moved the ligand in space would also move the contacts,
and a difference in the reported ids could then be explained by either.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("prody", reason="needs the laser extra")

import equinox as eqx
import jax

from aminx.families.laser_mpnn.driver import LASErMPNN, LaserDriver
from aminx.host.family_driver import FAMILY_DRIVERS
from aminx.host.runner import score
from aminx.run.specs import ScoringSpecification


def _atom(
  serial: int, name: str, resname: str, chain: str, resseq: int,
  x: float, y: float, z: float, element: str = "C", *, het: bool = False,
) -> str:
  record = "HETATM" if het else "ATOM  "
  return (
    f"{record}{serial:5d} {name:>4} {resname:>3} {chain}"
    f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.0:6.2f}          {element:>2}"
  )


_PROTEIN = [
  _atom(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
  _atom(2, "CA", "ALA", "A", 1, 1.5, 0.0, 0.0, "C"),
  _atom(3, "C", "ALA", "A", 1, 2.5, 1.0, 0.0, "C"),
  _atom(4, "O", "ALA", "A", 1, 2.2, 2.2, 0.0, "O"),
  _atom(5, "CB", "ALA", "A", 1, 1.5, -1.4, 0.5, "C"),
  _atom(6, "N", "ALA", "A", 2, 3.8, 1.2, 0.0, "N"),
  _atom(7, "CA", "ALA", "A", 2, 5.0, 2.0, 0.0, "C"),
  _atom(8, "C", "ALA", "A", 2, 6.2, 1.2, 0.4, "C"),
  _atom(9, "O", "ALA", "A", 2, 6.4, 0.0, 0.2, "O"),
  _atom(10, "CB", "ALA", "A", 2, 5.2, 3.2, 0.8, "C"),
]
# Same coordinates in both fixtures. Only its line position changes.
_LIGAND = [_atom(11, "C1", "LIG", "B", 1, 8.0, 2.0, 1.0, "C", het=True)]


@pytest.fixture
def registered():  # noqa: ANN201
  driver = LaserDriver()
  FAMILY_DRIVERS.register("lasermpnn")(driver)
  yield driver
  FAMILY_DRIVERS.discard("lasermpnn")


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
  path = tmp_path / "tiny.eqx"
  eqx.tree_serialise_leaves(path, LASErMPNN(key=jax.random.PRNGKey(0)))
  return path


def _ids(path: Path, lines: list[str], model_path: Path) -> np.ndarray:
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  result = score(
    ScoringSpecification(
      inputs=str(path),
      model_family="lasermpnn",
      checkpoint_id="lasermpnn_test",
      model_local_path=model_path,
      output_kind="proofread_unconditional",
      sequences_to_score=["AA"],
      random_seed=3,
    ),
  )
  return np.asarray(result["structures"]["0"]["arrays"]["residue_ids"])


def test_divergence_proofread_resindex_identity(
  registered, tmp_path: Path, model_path: Path,  # noqa: ANN001
) -> None:
  """Moving a HETATM earlier shifts the reported ids, because they are resindex."""
  del registered
  identity_holds = _ids(tmp_path / "last.pdb", [*_PROTEIN, *_LIGAND], model_path)
  identity_fails = _ids(tmp_path / "first.pdb", [*_LIGAND, *_PROTEIN], model_path)

  # Same contact set, so the same number of focus rows in both.
  assert identity_holds.shape == identity_fails.shape, (
    f"the fixtures must select the same rows: {identity_holds} vs {identity_fails}"
  )

  # With the ligand last, resindex == row and the ids are the plain rows. This
  # is the case every parity fixture exercises, and why the bug hides.
  np.testing.assert_array_equal(identity_holds, np.asarray([1], dtype=identity_holds.dtype))

  # With the ligand first, every protein resindex is one higher. Reporting the
  # ROW would give [1] again; reporting resindex gives [2]. That difference is
  # the entire divergence.
  np.testing.assert_array_equal(identity_fails, identity_holds + 1)


def test_proofread_resindex_control_rows_would_be_identical(
  registered, tmp_path: Path,  # noqa: ANN001
) -> None:
  """CONTROL: the two fixtures are indistinguishable by row index.

  The focus ROWS are the same in both files -- only ``resindex`` moves. So an
  implementation that reported rows would return the SAME array twice and the
  test above would fail on its last assertion. Stating it here means the test
  is pinned to the distinction rather than to two literals that happen to
  differ.
  """
  del registered
  from aminx.families.laser_mpnn.featurize import featurize  # noqa: PLC0415

  last = tmp_path / "c_last.pdb"
  last.write_text("\n".join([*_PROTEIN, *_LIGAND]) + "\n", encoding="utf-8")
  first = tmp_path / "c_first.pdb"
  first.write_text("\n".join([*_LIGAND, *_PROTEIN]) + "\n", encoding="utf-8")

  a = featurize(last)
  b = featurize(first)

  # Identical geometry => identical contacts => identical focus rows.
  np.testing.assert_array_equal(
    np.asarray(a.first_shell_ligand_contact_mask),
    np.asarray(b.first_shell_ligand_contact_mask),
  )
  assert int(a.sequence_indices.shape[0]) == int(b.sequence_indices.shape[0])

  # And the ONLY thing that differs is the resindex map.
  assert np.asarray(a.row_to_resindex).tolist() == [0, 1]
  assert np.asarray(b.row_to_resindex).tolist() == [1, 2]
