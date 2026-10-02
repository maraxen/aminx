# ruff: noqa: S101
"""PottsMPNN knob semantics at the driver level, driven through PottsMPNNOptions.

``test_potts_knobs.py`` covers knobs reachable at the function that implements
them. Those establish the semantics but not the plumbing: several
``PottsMPNNOptions`` fields turn out to be read nowhere outside their own
dataclass (aminx debt 2435, guarded by ``tests/lint/test_options_plumbing.py``), a
gap a model-level test cannot see. The tests here go through ``score()`` with a
real ``ScoringSpecification`` carrying real options, so a knob that is declared and
wired to nothing fails them.

The model is a random-init ``PottsMPNN`` serialised to ``tmp_path`` -- no
checkpoint download -- so this level is cheap enough to be the default for any knob
consumed in host code rather than in a pure function.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest

from aminx.families.potts_mpnn.driver import PottsMPNNDriver
from aminx.families.potts_mpnn.model import PottsMPNN
from aminx.host.family_driver import FAMILY_DRIVERS
from aminx.host.runner import sample, score
from aminx.run.options import PottsMPNNOptions
from aminx.run.specs import SamplingSpecification, ScoringSpecification

_THREE = {
  "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
  "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
  "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
  "W": "TRP", "Y": "TYR",
}


def _atom(
  serial: int, atom: str, resname: str, chain: str, resseq: int,
  x: float, y: float, z: float,
) -> str:
  """One fixed-width ATOM record.

  Column 16 is altLoc, so resName starts at 17 and chainID lands at 21. Getting
  this wrong does not raise -- the parser simply matches no chain and reports
  ``no_chain`` -- so the widths are spelled out rather than eyeballed.
  """
  return (
    f"{'ATOM':<6.6}{serial:5d} {atom:>4.4}{' ':1.1}{resname:>3.3} {chain:1.1}"
    f"{resseq:4d}{' ':1.1}   {x:8.3f}{y:8.3f}{z:8.3f}"
  )


def _write_pdb(path: Path, chains: dict[str, str]) -> None:
  lines: list[str] = []
  serial = 1
  for letter, sequence in chains.items():
    for index, amino in enumerate(sequence, start=1):
      for offset, atom in enumerate(("N", "CA", "C", "O")):
        lines.append(
          _atom(
            serial, atom, _THREE[amino], letter, index,
            x=float(index * 3 + offset), y=float(ord(letter)), z=float(offset),
          ),
        )
        serial += 1
  lines.append("END")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def registered() -> Iterator[PottsMPNNDriver]:
  driver = PottsMPNNDriver()
  FAMILY_DRIVERS.register("pottsmpnn")(driver)
  yield driver
  FAMILY_DRIVERS.discard("pottsmpnn")


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
  path = tmp_path / "tiny.eqx"
  eqx.tree_serialise_leaves(path, PottsMPNN(key=jax.random.PRNGKey(0)))
  return path


def _ddg(pdb: Path, weights: Path, options: PottsMPNNOptions) -> np.ndarray:
  spec = ScoringSpecification(
    inputs=str(pdb),
    model_family="pottsmpnn",
    checkpoint_id="pottsmpnn_vanilla_20",
    model_local_path=weights,
    output_kind="ddg",
    potts_mpnn=options,
  )
  return np.asarray(
    score(spec)["structures"]["0"]["arrays"]["ddg"], dtype=np.float64,
  )


def test_knob_semantics_mean_norm(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``mean_norm`` subtracts the mean of the ddG vector, so the result is centred.

  ``driver.py:208-211``: ``delta = block[1:] - block[0]`` and then, only when the
  knob is set, ``delta = delta - mean(delta)``. Two mutants are needed for the knob
  to be distinguishable at all -- with one, centring sends the single element to
  exactly 0 and any implementation that merely zeroed the output would pass.

  Driven through ``PottsMPNNOptions``, so this fails if the field stops being
  plumbed, which a test calling ``_score_one`` directly would not catch.
  """
  del registered  # the fixture's effect is the family registration, not a value
  pdb = tmp_path / "complex.pdb"
  _write_pdb(pdb, {"A": "AAA", "B": "CCC"})
  binding = tmp_path / "binding.json"
  binding.write_text('{"complex": [["A"], ["B"]]}\n', encoding="utf-8")
  table = tmp_path / "mutants.csv"
  table.write_text(
    "pdb,chain,mut_type,ddG_expt\n"
    "complex,A:B,A2C:C2D,0.5\n"
    "complex,A:B,A1D:C1E,1.5\n",
    encoding="utf-8",
  )
  shared = {"binding_energy_json": str(binding), "mutant_csv": str(table)}

  raw = _ddg(pdb, model_path, PottsMPNNOptions(**shared, mean_norm=False))
  centred = _ddg(pdb, model_path, PottsMPNNOptions(**shared, mean_norm=True))

  assert raw.shape == centred.shape == (2,), (raw.shape, centred.shape)
  np.testing.assert_allclose(centred, raw - raw.mean(), rtol=1e-6, atol=1e-6)
  assert abs(float(centred.mean())) < 1e-6, "the centred vector must have zero mean"
  assert abs(float(raw.mean())) > 1e-6, (
    "the uncentred mean is ~0, so this fixture cannot distinguish mean_norm from a "
    "no-op; pick mutants whose raw ddG mean is nonzero"
  )
  assert not np.allclose(raw, centred), "mean_norm must not be inert"


def _energies(pdb: Path, weights: Path, sequences: list[str]) -> np.ndarray:
  spec = ScoringSpecification(
    inputs=str(pdb),
    model_family="pottsmpnn",
    checkpoint_id="pottsmpnn_vanilla_20",
    model_local_path=weights,
    output_kind="energy",
    sequences_to_score=sequences,
    potts_mpnn=PottsMPNNOptions(),
  )
  return np.asarray(
    score(spec)["structures"]["0"]["arrays"]["energy"], dtype=np.float64,
  )


def test_knob_semantics_output_kind(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``output_kind`` picks absolute energy or the difference from the reference.

  Spec 6.3: upstream's ``ddG`` flag maps here, True being ``ddg`` and False
  ``energy``. The difference is not merely which key appears in the output --
  ``driver.py:206-208`` returns the raw block for ``energy`` and
  ``block[1:] - block[0]`` for ``ddg`` -- so this asserts the arithmetic relation
  between the two modes rather than that both merely run.
  """
  del registered
  pdb = tmp_path / "one.pdb"
  _write_pdb(pdb, {"A": "AAA"})
  table = tmp_path / "mutants.csv"
  table.write_text("pdb,chain,mut_type,ddG_expt\none,A,A1D,0.5\n", encoding="utf-8")

  ddg = _ddg(pdb, model_path, PottsMPNNOptions(mutant_csv=str(table)))
  assert ddg.shape == (1,)

  # mut_type positions are ZERO-based (driver.py:518, matching upstream
  # run_utils.py:737), so A1D edits the SECOND residue: AAA -> ADA.
  #
  # The energy path prepends the NATIVE reference to the scored sequences
  # (include_reference_in_output, driver.py:206-207), so asking for the mutant
  # alone returns exactly the (reference, mutant) pair this comparison needs.
  # Passing ["AAA", "ADA"] returns three values, not two.
  native, mutant = _energies(pdb, model_path, ["ADA"])
  np.testing.assert_allclose(ddg[0], mutant - native, rtol=1e-5, atol=1e-5)
  assert abs(float(mutant - native)) > 1e-6, (
    "the mutation must change the energy, or this comparison holds trivially for "
    "any implementation that returns zeros"
  )


def test_knob_semantics_exclude_chains(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``exclude_chains`` drops a chain from the deep mutational scan entirely.

  ``driver.py:534-541``: with no mutant table the driver enumerates every single
  mutant, and a chain named here contributes none. The separator is any of comma,
  colon or whitespace (``:645``).

  Counting is the observable. Each 2-residue chain yields 2 x 19 = 38 single
  mutants, so two chains give 76 and excluding one gives 38.
  """
  del registered
  pdb = tmp_path / "two.pdb"
  _write_pdb(pdb, {"A": "AA", "B": "CC"})

  both = _ddg(pdb, model_path, PottsMPNNOptions())
  only_a = _ddg(pdb, model_path, PottsMPNNOptions(exclude_chains="B"))

  assert both.shape == (76,), both.shape
  assert only_a.shape == (38,), only_a.shape
  assert both.shape != only_a.shape, "exclude_chains must not be inert"

  # The separator set is part of the contract, so exercise one of the others.
  colon = _ddg(pdb, model_path, PottsMPNNOptions(exclude_chains=":B:"))
  assert colon.shape == only_a.shape, (
    "colon and whitespace are documented separators alongside comma (driver.py:645)"
  )


def test_knob_semantics_mutant_csv(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``mutant_csv`` supplies the ddG candidate set from a table of substitutions.

  ``driver.py:424-425`` routes to ``_from_csv`` (``:494-531``), which is three
  behaviours, not one, so all three are asserted:

  * the ``pdb`` column selects rows (``:508``), so a shared table holding many
    structures contributes only its own rows;
  * ``chain`` and ``mut_type`` are colon-separated and zipped (``:515``), so one row
    carrying two chains is ONE candidate with both edits, not two candidates;
  * the wild-type letter is checked against the structure (``:519-524``), which is
    the only assertion here that proves the parser reads the PDB at all rather than
    taking the table's word for the sequence.

  Positions are 0-based into the gap-filled chain (spec "Index space",
  ``run_utils.py:737``), so ``A1D`` edits the SECOND residue. The table reads like
  1-based notation, which is why it is restated here.
  """
  del registered
  pdb = tmp_path / "one.pdb"
  _write_pdb(pdb, {"A": "AAA", "B": "CCC"})
  table = tmp_path / "mutants.csv"
  table.write_text(
    "pdb,chain,mut_type,ddG_expt\n"
    "one,A,A1D,0.5\n"
    "one,A:B,A2C:C0E,1.5\n"
    "other,A,A1D,2.5\n",
    encoding="utf-8",
  )

  ddg = _ddg(pdb, model_path, PottsMPNNOptions(mutant_csv=str(table)))

  # Two rows, not three: the `other` row belongs to a different structure. Not
  # four either: the two-chain row is a single double mutant.
  assert ddg.shape == (2,), (
    "expected one candidate per matching row; the `other` row must be filtered by "
    "pdb name and the A:B row must stay a single double mutant"
  )

  native, single, double = _energies(pdb, model_path, ["ADACCC", "AACECC"])
  np.testing.assert_allclose(ddg[0], single - native, rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(ddg[1], double - native, rtol=1e-5, atol=1e-5)
  assert abs(float(double - single)) > 1e-6, (
    "the two rows must land on different sequences, or this comparison cannot "
    "distinguish them from each other"
  )

  # The wild-type letter is validated against the structure, so a table that
  # disagrees with the PDB is rejected rather than silently applied.
  wrong = tmp_path / "wrong.csv"
  wrong.write_text(
    "pdb,chain,mut_type,ddG_expt\none,A,W1D,0.5\n", encoding="utf-8",
  )
  with pytest.raises(ValueError, match="does not match chain A"):
    _ddg(pdb, model_path, PottsMPNNOptions(mutant_csv=str(wrong)))


def test_knob_semantics_mutant_fasta(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``mutant_fasta`` gives the candidates as whole sequences, and outranks the CSV.

  ``driver.py:422-425`` checks the FASTA field FIRST, so when both are set the CSV
  is never opened. The spec writes the resolution as "mutant_fasta / mutant_csv
  else single-mutant DMS" (tables at :511-512), so that order is the contract and
  not an accident of the ``if`` chain -- which is why it is asserted here rather
  than left to a reader of the source.

  ``_from_fasta`` (``:446-491``) names chains in the header, so a header listing
  one chain replaces only that chain and the rest stay wild-type. Omitting the
  chain list means the record must spell out every chain (``:478-480``).
  """
  del registered
  pdb = tmp_path / "one.pdb"
  _write_pdb(pdb, {"A": "AAA", "B": "CCC"})
  records = tmp_path / "mutants.fasta"
  records.write_text(
    ">one|A|0.5\nADA\n>other|A|1.5\nDDD\n", encoding="utf-8",
  )

  ddg = _ddg(pdb, model_path, PottsMPNNOptions(mutant_fasta=str(records)))
  assert ddg.shape == (1,), "the `other` record belongs to a different structure"

  # The header named chain A only, so chain B must still be the structure's own
  # CCC -- a parser that dropped the unnamed chain would give a different energy.
  native, mutant = _energies(pdb, model_path, ["ADACCC"])
  np.testing.assert_allclose(ddg[0], mutant - native, rtol=1e-5, atol=1e-5)
  assert abs(float(mutant - native)) > 1e-6, (
    "the record must change the energy, or this holds trivially for an "
    "implementation that returns zeros"
  )

  # Precedence: both fields set, and the CSV's two rows do not appear.
  table = tmp_path / "mutants.csv"
  table.write_text(
    "pdb,chain,mut_type,ddG_expt\none,A,A1D,0.5\none,B,C0E,1.5\n",
    encoding="utf-8",
  )
  both = _ddg(
    pdb, model_path,
    PottsMPNNOptions(mutant_fasta=str(records), mutant_csv=str(table)),
  )
  assert both.shape == (1,), (
    "mutant_fasta outranks mutant_csv (driver.py:422-425, spec :511-512); a shape "
    "of (2,) would mean the CSV won and (3,) that both were read"
  )

  # With no chain list in the header the record must cover every chain.
  partial = tmp_path / "partial.fasta"
  partial.write_text(">one\nADA\n", encoding="utf-8")
  with pytest.raises(ValueError, match="must list every chain"):
    _ddg(pdb, model_path, PottsMPNNOptions(mutant_fasta=str(partial)))


def test_knob_semantics_binding_energy_json(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``binding_energy_json`` turns the ddG into a BINDING ddG over partitions.

  ``driver.py:632-640`` (``_binding_partitions``) reads a mapping from structure
  name to a list of partitions, each a list of chain letters, and
  ``_partition_graphs`` then subtracts the partitions' own energies from the
  complex. Spec :512 -- "binding: axis ``partition``".

  "Setting the knob changed the number" would pass for any implementation that
  perturbed the result at all, so the load-bearing assertion here is structural
  and independent of the random weights: a SINGLE partition covering every chain
  is the complex, so it must cancel to exactly zero. Measured on this fixture it
  is 0.0, not approximately -- the subtraction is of identical graphs.

  The third assertion is the name keying: a payload for some other structure is
  not this structure's, so it is ignored and the result is the plain complex ddG,
  the same ``payload.get(name)`` selection the mutant table uses.
  """
  del registered
  pdb = tmp_path / "complex.pdb"
  _write_pdb(pdb, {"A": "AAA", "B": "CCC"})
  table = tmp_path / "mutants.csv"
  table.write_text(
    "pdb,chain,mut_type,ddG_expt\ncomplex,A,A1D,0.5\ncomplex,B,C1E,1.5\n",
    encoding="utf-8",
  )

  def _with(partitions: str | None) -> np.ndarray:
    kwargs = {}
    if partitions is not None:
      path = tmp_path / f"binding_{len(partitions)}_{partitions.count('[')}.json"
      path.write_text(partitions, encoding="utf-8")
      kwargs["binding_energy_json"] = str(path)
    return _ddg(pdb, model_path, PottsMPNNOptions(mutant_csv=str(table), **kwargs))

  plain = _with(None)
  split = _with('{"complex": [["A"], ["B"]]}')
  whole = _with('{"complex": [["A", "B"]]}')
  mismatch = _with('{"some_other_structure": [["A"], ["B"]]}')

  # A partition per chain is a real binding decomposition, so it must not agree
  # with the complex ddG -- otherwise the knob is inert.
  assert not np.allclose(plain, split), (
    "binding_energy_json must change the ddG; identical values mean the "
    "partitions were never subtracted"
  )

  # One partition that IS the complex subtracts the complex from itself.
  np.testing.assert_allclose(whole, 0.0, atol=1e-6)

  # Keyed by structure name, so another structure's entry is not ours.
  np.testing.assert_allclose(mismatch, plain, rtol=1e-6, atol=1e-6)
  assert not np.allclose(mismatch, whole), (
    "the fixture must have non-zero complex ddG, or the name-keying assertion "
    "would hold trivially against the zero case"
  )


def test_knob_semantics_binding_energy_optimization(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``binding_energy_optimization`` is refused by the sample path, by design.

  ``sample_host.py:757-775``: ``none`` returns a zero interface so refine uses
  the complex energy, while ``both`` and ``only`` raise -- "a zero table would
  leave ``both`` identical to ``none`` and would update no positions under
  ``only``". Refusing is the implemented contract, so that is what gets asserted;
  a test demanding a binding-refined sample would be asserting a feature that
  deliberately does not exist.

  THE KNOB IS INERT IN THE SCORE PATH. Measured 261002 on this fixture:
  ``none``, ``both`` and ``only`` give bit-identical ``score:ddg`` output, with
  and without a binding_energy_json. Binding in scoring is selected by
  binding_energy_json alone (:512); this field belongs to design. That is why
  this test drives ``sample()`` and not ``score()``.

  ``test_review_findings.py:96-111`` already asserts the same three outcomes, but
  it calls ``sample_binding_tables`` directly with a hand-built Options, so it
  would still pass if the driver never forwarded the field. This one goes through
  a real ``SamplingSpecification``, which is the part that can regress.
  """
  del registered
  pdb = tmp_path / "toy.pdb"
  _write_pdb(pdb, {"A": "AC"})

  def _sample(mode: str) -> dict:
    spec = SamplingSpecification(
      inputs=str(pdb),
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      num_samples=1,
      samples_chunk_size=1,
      return_logits=False,
      potts_mpnn=PottsMPNNOptions(
        optimization_mode="none", binding_energy_optimization=mode,
      ),
    )
    return sample(spec)["structures"]["0"]["arrays"]

  arrays = _sample("none")
  assert arrays["sequence"].shape == (1, 2), (
    "the default must still sample, or the refusals below prove nothing"
  )

  for mode in ("both", "only"):
    with pytest.raises(ValueError, match="needs partition etabs"):
      _sample(mode)
