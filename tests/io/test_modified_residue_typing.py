"""Selenomethionine (MSE) is tokenised X, as the reference LigandMPNN parser does.

proxide >= 0.2.0 types MSE as MET by default, so aminx's token for an MSE residue became M
while the reference parser's is X (LigandMPNN Heavy Parity, ``test_invariant_6_tied_positions``
on 3HTN: 9 of 425 residues, red on main since the proxide 0.2.0 bump, #268). aminx now asks
proxide for ``modified_residue_typing="unknown"`` on every parse path it owns.

This file pins that without the reference clone: a 3-residue structure with one MSE, parsed
through aminx's own entry point, and the spec both aminx parse paths build.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from proxide import OutputSpec

from aminx.io.parsing import parse_structure
from aminx.io.parsing.dispatch import MODIFIED_RESIDUE_TYPING, _output_spec
from aminx.utils.aa_convert import af_to_mpnn

_ATOMS_BY_RESIDUE = {
  "ALA": ["N", "CA", "C", "O", "CB"],
  "MSE": ["N", "CA", "C", "O", "CB", "CG", "SE", "CE"],
  "GLY": ["N", "CA", "C", "O"],
}
_ELEMENT = {"N": "N", "CA": "C", "C": "C", "O": "O", "CB": "C", "CG": "C", "SE": "SE", "CE": "C"}
_X_TOKEN = 20  # MPNN alphabet ACDEFGHIKLMNPQRSTVWYX: index 20 is X
_MET_MPNN_TOKEN = 10  # ...and index 10 is M


def _write_pdb(path: Path) -> Path:
  """ALA, MSE (HETATM, as deposited), GLY along a straight line; coordinates are arbitrary."""
  lines: list[str] = []
  serial = 1
  for res_seq, (res_name, atoms) in enumerate(_ATOMS_BY_RESIDUE.items(), start=1):
    record = "HETATM" if res_name == "MSE" else "ATOM  "
    for k, atom in enumerate(atoms):
      x = 3.8 * res_seq + 0.7 * k
      y = 0.4 * k
      lines.append(
        f"{record}{serial:5d} {atom:<4s} {res_name:>3s} A{res_seq:4d}    "
        f"{x:8.3f}{y:8.3f}{0.0:8.3f}  1.00 20.00          {_ELEMENT[atom]:>2s}\n"
      )
      serial += 1
  path.write_text("".join(lines) + "END\n")
  return path


def test_aminx_asks_proxide_for_unknown_typing_by_default():
  assert MODIFIED_RESIDUE_TYPING == "unknown"
  assert _output_spec(48, {}).modified_residue_typing == "unknown"


def test_mse_is_x_through_aminx_parse_structure(tmp_path):
  protein = parse_structure(_write_pdb(tmp_path / "mse.pdb"))
  mpnn_tokens = np.asarray(af_to_mpnn(np.asarray(protein.aatype)))
  assert mpnn_tokens.shape[0] == 3
  assert int(mpnn_tokens[1]) == _X_TOKEN, f"MSE should be X, got token {int(mpnn_tokens[1])}"
  # the neighbours are ordinary residues, so the option moved only the modified one
  assert int(mpnn_tokens[0]) == 0  # A
  assert int(mpnn_tokens[2]) == 5  # G


def test_negative_control_default_proxide_typing_is_met(tmp_path):
  """Without the option the same residue is M, so the X above is the option's doing."""
  from proxide import parse_structure as proxide_parse_structure

  protein = proxide_parse_structure(
    str(_write_pdb(tmp_path / "mse.pdb")), OutputSpec(compute_rbf=True, rbf_num_neighbors=3)
  )
  mpnn_tokens = np.asarray(af_to_mpnn(np.asarray(protein.aatype)))
  assert int(mpnn_tokens[1]) == _MET_MPNN_TOKEN


@pytest.mark.parametrize("add_hydrogens", [False, True])
def test_unknown_typing_survives_hydrogen_addition(tmp_path, add_hydrogens):
  """proxide's topology rebuild after adding hydrogens used to re-type MSE back to MET."""
  protein = parse_structure(_write_pdb(tmp_path / "mse.pdb"), add_hydrogens=add_hydrogens)
  mpnn_tokens = np.asarray(af_to_mpnn(np.asarray(protein.aatype)))
  assert int(mpnn_tokens[1]) == _X_TOKEN


# --- the runner path (host/prep.py -> proxide create_protein_dataset), which the tests above do
# --- not reach: proxide 0.2.0's load_rust drops keywords it does not read, so without a guard the
# --- runner would keep tokenising MSE as M on an old proxide and say nothing.


def _runner_spec():
  from types import SimpleNamespace

  return SimpleNamespace(chain_id=None, model=None, altloc=None, topology=None)


def test_runner_parse_kwargs_request_unknown_typing():
  from aminx.host.prep import _proxide_parse_kwargs

  kwargs = _proxide_parse_kwargs(_runner_spec())
  assert kwargs["modified_residue_typing"] == MODIFIED_RESIDUE_TYPING == "unknown"


def test_runner_dataset_path_tokenises_mse_as_x(tmp_path):
  """The real create_protein_dataset call the runner makes, with the kwargs it builds."""
  from proxide.ops.dataset import create_protein_dataset

  from aminx.host.prep import _proxide_parse_kwargs

  dataset = create_protein_dataset(
    [str(_write_pdb(tmp_path / "mse.pdb"))],
    batch_size=1,
    parse_kwargs=_proxide_parse_kwargs(_runner_spec()),
    max_length=8,
    use_electrostatics=False,
    use_vdw=False,
  )
  batch = next(iter(dataset))
  mpnn_tokens = np.asarray(af_to_mpnn(np.asarray(batch.aatype)))[0]
  assert int(mpnn_tokens[1]) == _X_TOKEN, f"MSE should be X on the runner path, got {int(mpnn_tokens[1])}"


class _OutputSpecWithoutTheOption:
  """Stands in for proxide 0.2.0's OutputSpec, which has no modified_residue_typing."""


def test_guard_refuses_a_proxide_that_cannot_honour_the_option(monkeypatch):
  import aminx.io.parsing.dispatch as dispatch
  from aminx.host.prep import _proxide_parse_kwargs

  monkeypatch.setattr(dispatch, "OutputSpec", _OutputSpecWithoutTheOption)
  with pytest.raises(RuntimeError, match="modified_residue_typing"):
    dispatch.require_modified_residue_typing_support()
  with pytest.raises(RuntimeError, match="modified_residue_typing"):
    dispatch._output_spec(48, {})  # noqa: SLF001
  with pytest.raises(RuntimeError, match="modified_residue_typing"):
    _proxide_parse_kwargs(_runner_spec())
