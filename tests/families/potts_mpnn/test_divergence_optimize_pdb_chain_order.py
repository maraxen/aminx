# ruff: noqa: S101, SLF001
"""Spec §6.5b: ``optimize_pdb`` seeds in A0 row order, not alphabet order.

``sample_seqs.py:146-155`` builds the ``optimize_pdb`` seed by concatenating
chain sequences in listing order, while ``potts_mpnn_utils.py:268-290,:327``
featurizes in a DIFFERENT order -- sorted designed chains, then sorted fixed
chains. Whenever those two disagree, upstream seeds the refiner with a sequence
whose residues do not line up with the rows they are supposed to label: chain
B's residues are scored against chain A's coordinates. It does not raise,
because the total length is the same either way. That is what makes it worth a
test rather than an assertion -- the failure is silent and the length check
cannot see it.

aminx concatenates in A0 order (``driver.py:609`` orders by
``features.letter_list``, which is ``all_chains = masked_chains +
visible_chains``, ``featurize.py:352``).

THE FIXTURE HAS TO BE ONE WHERE THE TWO ORDERS DIVERGE. If the designed chain
is also the alphabetically first one, A0 order IS alphabet order and the whole
file passes under either implementation. So chain A is FIXED and chain B is
DESIGNED, which puts B first in A0 and second in the alphabet; the control at
the bottom flips the mask and shows the orders coinciding again, which is what
proves the main fixture is doing work.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aminx.families.potts_mpnn.driver import _chain_sequences, _featurize_one
from aminx.families.potts_mpnn.featurize import parse_pdb_upstream
from aminx.run.options import PottsMPNNOptions

_THREE = {"A": "ALA", "W": "TRP"}

_CHAIN_A = "AAAA"
_CHAIN_B = "WWWW"

# The two candidate orders, spelled out so the assertions read as a comparison
# between named alternatives rather than against a literal.
_A0_ORDER = _CHAIN_B + _CHAIN_A  # designed first: what aminx must produce
_ALPHABET_ORDER = _CHAIN_A + _CHAIN_B  # what upstream's listing order gives


def _atom(serial: int, name: str, resname: str, chain: str, resseq: int, x: float) -> str:
  return (
    f"ATOM  {serial:5d} {name:>4} {resname:>3} {chain}"
    f"{resseq:4d}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.00:6.2f}{0.0:6.2f}           C"
  )


@pytest.fixture
def pdb(tmp_path: Path) -> Path:
  lines: list[str] = []
  serial = 1
  for letter, sequence in (("A", _CHAIN_A), ("B", _CHAIN_B)):
    for index, amino in enumerate(sequence, start=1):
      for offset, atom in enumerate(("N", "CA", "C", "O")):
        lines.append(
          _atom(serial, atom, _THREE[amino], letter, index, float(index * 4 + offset)),
        )
        serial += 1
  lines.append("END")
  path = tmp_path / "two_chain.pdb"
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return path


def _mask(tmp_path: Path, name: str, designed: list[str], fixed: list[str]) -> str:
  path = tmp_path / f"mask_{''.join(designed)}.json"
  path.write_text(json.dumps({name: [designed, fixed]}), encoding="utf-8")
  return str(path)


def _native(pdb: Path, mask_path: str) -> tuple[tuple[str, ...], str]:
  """``(chain letters in A0 order, the concatenated optimize_pdb seed)``."""
  parsed = parse_pdb_upstream(pdb, skip_gaps=False)[0]
  name = str(parsed["name"])
  options = PottsMPNNOptions(chain_design_mask_json=mask_path)
  features = _featurize_one(parsed, options, name)
  chains = _chain_sequences(parsed, features)
  # This join is exactly sample_host.py:298's `native`.
  return tuple(chain.letter for chain in chains), "".join(
    chain.sequence for chain in chains
  )


def test_divergence_optimize_pdb_chain_order(pdb: Path, tmp_path: Path) -> None:
  """Designed chain B leads, although A is alphabetically first."""
  parsed = parse_pdb_upstream(pdb, skip_gaps=False)[0]
  name = str(parsed["name"])
  letters, native = _native(pdb, _mask(tmp_path, name, ["B"], ["A"]))

  assert letters == ("B", "A"), letters
  assert native == _A0_ORDER
  # The thing the length check cannot catch: both candidates are 8 residues.
  assert len(native) == len(_ALPHABET_ORDER)
  assert native != _ALPHABET_ORDER


def test_optimize_pdb_chain_order_control_orders_coincide(pdb: Path, tmp_path: Path) -> None:
  """CONTROL: flip the mask and A0 order IS alphabet order.

  Without this the main test is satisfied by an implementation that always
  reverses, or that sorts descending. It also demonstrates why the fixture
  above had to designate the SECOND chain: with A designed, the two orders
  agree and nothing is being tested.
  """
  parsed = parse_pdb_upstream(pdb, skip_gaps=False)[0]
  name = str(parsed["name"])
  letters, native = _native(pdb, _mask(tmp_path, name, ["A"], ["B"]))

  assert letters == ("A", "B"), letters
  assert native == _ALPHABET_ORDER


def test_optimize_pdb_chain_order_matches_the_rows_it_labels(pdb: Path, tmp_path: Path) -> None:
  """The seed lines up with the featurized rows, position by position.

  The ordering assertions above compare strings. This one checks the property
  those strings exist for: residue i of the seed is the residue featurized at
  row i. A port that got the chain order wrong would still produce an 8-mer,
  and only this comparison says it is the RIGHT 8-mer.
  """
  parsed = parse_pdb_upstream(pdb, skip_gaps=False)[0]
  name = str(parsed["name"])
  options = PottsMPNNOptions(chain_design_mask_json=_mask(tmp_path, name, ["B"], ["A"]))
  features = _featurize_one(parsed, options, name)
  _letters, native = _native(pdb, _mask(tmp_path, name, ["B"], ["A"]))

  # chain_m is 1 on designed rows and 0 on fixed ones, so it independently
  # reports where A0 put chain B -- without going through letter_list.
  chain_m = np.asarray(features.chain_m)
  designed_rows = np.flatnonzero(chain_m > 0)
  assert designed_rows.tolist() == [0, 1, 2, 3], (
    f"A0 must place the designed chain first; chain_m says {chain_m}"
  )
  seeded = np.asarray(list(native))
  assert set(seeded[designed_rows]) == {"W"}, seeded
  assert set(np.delete(seeded, designed_rows)) == {"A"}, seeded
