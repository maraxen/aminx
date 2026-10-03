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

import json
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


_OPT_NATIVE = "ACDEFG"


def _refined(
  pdb: Path, weights: Path, options: PottsMPNNOptions, num_samples: int = 1,
) -> dict:
  spec = SamplingSpecification(
    inputs=str(pdb),
    model_family="pottsmpnn",
    checkpoint_id="pottsmpnn_vanilla_20",
    model_local_path=weights,
    num_samples=num_samples,
    samples_chunk_size=1,
    return_logits=False,
    potts_mpnn=options,
  )
  return sample(spec)["structures"]["0"]["arrays"]


def test_knob_semantics_optimize_pdb(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``optimize_pdb`` refines FROM the structure's own sequence instead of decoding.

  ``sample_host.py:299-301``: ``optimizing`` is
  ``(optimize_pdb or optimize_fasta) and optimization_mode != "none"``, so the
  mode VETOES the seed -- with ``none`` the sampler decodes autoregressively and
  emits ``sequence``/``sample_energy``/``sample_rank``, and with a refine mode it
  emits ``refined_sequence`` alone. Both halves of that ``and`` are exercised,
  because a knob that was ignored and a knob that could not be switched off look
  identical from one call.

  ``:308-312`` seeds the refinement from the native sequence, and
  ``force_optimize_num_samples`` (``:93-104``) pins ``num_samples`` to 1 because
  refining one seed many times is the same work repeated.

  ``test_sample.py:307`` already covers ``force_optimize_num_samples`` and
  ``sample_schema``, but it never calls ``sample()``, so it would pass if the
  sample path ignored the seed entirely. This runs it end to end.
  """
  del registered
  pdb = tmp_path / "toy.pdb"
  _write_pdb(pdb, {"A": _OPT_NATIVE})

  vetoed = _refined(
    pdb, model_path,
    PottsMPNNOptions(optimize_pdb=True, optimization_mode="none"),
  )
  assert "refined_sequence" not in vetoed, (
    "optimization_mode='none' must veto the seed (the `and` at :299-301)"
  )
  assert vetoed["sequence"].shape == (1, len(_OPT_NATIVE))

  refined = _refined(
    pdb, model_path,
    PottsMPNNOptions(optimize_pdb=True, optimization_mode="potts"),
  )
  assert "refined_sequence" in refined, "a refine mode must take the seeded path"
  assert "sequence" not in refined, (
    "the refine path replaces the decode outputs rather than adding to them"
  )

  # num_samples is forced to 1 END TO END, not just by the helper that does it.
  many = _refined(
    pdb, model_path,
    PottsMPNNOptions(optimize_pdb=True, optimization_mode="potts"),
    num_samples=4,
  )
  assert many["refined_sequence"].shape == (1, len(_OPT_NATIVE)), (
    "optimize_pdb forces num_samples=1 (:93-104); a shape of (4, L) means the "
    "forcing never reached the sample path"
  )
  np.testing.assert_array_equal(many["refined_sequence"], refined["refined_sequence"])


def test_knob_semantics_optimize_fasta(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``optimize_fasta`` supplies the refine seed, and outranks ``optimize_pdb``.

  ``sample_host.py:302-312`` is an ``if``/``elif``: with both fields set the
  FASTA is read and the native is not. The seed is observable because a different
  seed refines to a different answer -- measured on this fixture, the native
  seeds to ``[9 5 16 3 2 2]`` and the FASTA's ``WWWWWW`` to ``[4 4 3 2 0 0]``.
  That difference is what makes the precedence assertion meaningful; without it,
  "both set equals fasta-only" would hold trivially for an implementation that
  ignored both.

  ``load_optimize_fasta`` (``:173-184``) selects entries whose header starts with
  the structure name, so another structure's record is not this one's, and a
  sequence of the wrong length is rejected against ``L_total`` rather than
  silently padded.
  """
  del registered
  pdb = tmp_path / "toy.pdb"
  _write_pdb(pdb, {"A": _OPT_NATIVE})
  fasta = tmp_path / "opt.fasta"
  # `other` must differ from `toy`, or the name filter would be untested.
  fasta.write_text(">other\nYYYYYY\n>toy\nWWWWWW\n", encoding="utf-8")

  from_native = _refined(
    pdb, model_path,
    PottsMPNNOptions(optimize_pdb=True, optimization_mode="potts"),
  )["refined_sequence"]
  from_fasta = _refined(
    pdb, model_path,
    PottsMPNNOptions(optimize_fasta=str(fasta), optimization_mode="potts"),
  )["refined_sequence"]

  assert from_fasta.shape == (1, len(_OPT_NATIVE)), (
    "only the `toy` record is this structure's; (2, L) would mean `other` was "
    "loaded too"
  )
  assert not np.array_equal(from_native, from_fasta), (
    "a different seed must refine to a different answer, or the seed is inert "
    "and the precedence check below proves nothing"
  )

  both = _refined(
    pdb, model_path,
    PottsMPNNOptions(
      optimize_pdb=True, optimize_fasta=str(fasta), optimization_mode="potts",
    ),
  )["refined_sequence"]
  np.testing.assert_array_equal(both, from_fasta)
  assert not np.array_equal(both, from_native), (
    "optimize_fasta outranks optimize_pdb (the if/elif at :302-312)"
  )

  short = tmp_path / "short.fasta"
  short.write_text(">toy\nWWW\n", encoding="utf-8")
  with pytest.raises(ValueError, match="!= L_total"):
    _refined(
      pdb, model_path,
      PottsMPNNOptions(optimize_fasta=str(short), optimization_mode="potts"),
    )


def test_knob_semantics_emit_etab(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``emit_etab`` adds the Potts energy table and its neighbour index to the output.

  Two sites must agree or the output is malformed: ``sample_host.py:394-400``
  declares ``potts_etab`` / ``potts_E_idx`` in the schema, and ``:483-486``
  produces them. Asserting presence alone would pass for a schema that promised
  arrays the sampler never wrote, so the shapes are checked against the declared
  dims.

  ``K`` is 6 for a 6-residue chain and 48 for a 60-residue one, so both regimes
  are exercised; with only the short fixture the cap would be invisible.

  THE 48 DOES NOT COME FROM ``min(48, ready.l_total)`` AT ``:484``, although it
  reads that way. ``forward`` already carries at most 48 neighbour columns from
  the encoder and a slice past the end clamps, so that ``min`` is redundant:
  measured 261002, replacing it with ``ready.l_total`` -- or with 999 -- changes
  no output and this test still passes. Narrowing it to 2 IS caught, so the
  assertion does pin the slice; it just does not pin the line it appears to.
  """
  del registered

  def _arrays(chains: dict[str, str], *, emit: bool) -> dict:
    pdb = tmp_path / f"etab_{len(chains)}_{sum(map(len, chains.values()))}.pdb"
    _write_pdb(pdb, chains)
    spec = SamplingSpecification(
      inputs=str(pdb),
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      num_samples=1,
      samples_chunk_size=1,
      return_logits=False,
      potts_mpnn=PottsMPNNOptions(optimization_mode="none", emit_etab=emit),
    )
    return sample(spec)["structures"]["0"]["arrays"]

  off = _arrays({"A": "ACDEFG"}, emit=False)
  assert "potts_etab" not in off, "the default must not pay for the table"
  assert "potts_E_idx" not in off

  short = _arrays({"A": "ACDEFG"}, emit=True)
  assert short["potts_etab"].shape == (6, 6, 20, 20), short["potts_etab"].shape
  assert short["potts_E_idx"].shape == (6, 6)
  assert short["potts_etab"].dtype == np.float32
  assert short["potts_E_idx"].dtype == np.int32

  # K is capped at 48, which only a structure longer than 48 can show.
  long_ = _arrays({"A": "AC" * 30}, emit=True)
  assert long_["potts_etab"].shape == (60, 48, 20, 20), long_["potts_etab"].shape
  assert long_["potts_E_idx"].shape == (60, 48)


def test_knob_semantics_chain_design_mask_json(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``chain_design_mask_json`` names designed/fixed chains, which sets A0 row order.

  ``driver.py:618-628`` (``_chain_dict``) feeds ``tied_featurize_port``'s chain
  dict, and A0 row order is "sorted designed chains then sorted fixed chains"
  (spec :511, ``tied_featurize :327``). So marking B designed and A fixed SWAPS
  the order, and the same physical assignment must then be written B-first.

  That gives an exact identity rather than a "something changed" check: with
  A='AAA' and B='CC', assigning A=DDD and B=EE is written ``DDDEE`` unmasked and
  ``EEDDD`` under the mask, and the two must score identically. Measured on this
  fixture both are -8.244517 while ``DDDEE`` UNDER the mask is -8.570608, so the
  mask is not a no-op either.

  THE CHAINS MUST DIFFER IN LENGTH. ``_write_pdb`` places chain L at
  ``y=ord(L)``, so two equal-length chains sit 1A apart with identical x and z
  and swapping their sequences is nearly an exact symmetry -- with 'AAA'/'CCC'
  all three energies above agreed to 1e-6 and the test could not have failed.
  """
  del registered
  pdb = tmp_path / "complex.pdb"
  _write_pdb(pdb, {"A": "AAA", "B": "CC"})
  mask = tmp_path / "mask.json"
  mask.write_text('{"complex": [["B"], ["A"]]}\n', encoding="utf-8")
  other = tmp_path / "other.json"
  other.write_text('{"not_this_one": [["B"], ["A"]]}\n', encoding="utf-8")

  def _energy(seq: str, mask_path: str | None) -> float:
    kwargs = {"chain_design_mask_json": mask_path} if mask_path else {}
    spec = ScoringSpecification(
      inputs=str(pdb),
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      output_kind="energy",
      sequences_to_score=[seq],
      potts_mpnn=PottsMPNNOptions(**kwargs),
    )
    values = np.asarray(
      score(spec)["structures"]["0"]["arrays"]["energy"], dtype=np.float64,
    )
    # index 0 is the prepended native reference (include_reference_in_output).
    return float(values[1])

  # The fixture has to respond to BOTH chains, or a swap could not be seen.
  native = _energy("AAACC", None)
  assert not np.isclose(native, _energy("DDDCC", None)), "chain A must matter"
  assert not np.isclose(native, _energy("AAAEE", None)), "chain B must matter"

  plain = _energy("DDDEE", None)
  swapped = _energy("EEDDD", str(mask))
  unswapped = _energy("DDDEE", str(mask))

  np.testing.assert_allclose(swapped, plain, rtol=1e-6, atol=1e-6)
  assert not np.isclose(plain, unswapped, rtol=1e-6, atol=1e-6), (
    "the mask must reinterpret the row order; equal values mean it was ignored"
  )

  # Keyed by structure name, like the mutant table and the binding partitions.
  np.testing.assert_allclose(_energy("DDDEE", str(other)), plain,
                             rtol=1e-6, atol=1e-6)


_PSSM_SETTINGS: tuple[tuple[str, dict], ...] = (
  ("pssm_threshold below the fill", {"pssm_threshold": -1.0, "pssm_log_odds_flag": True}),
  ("pssm_threshold above the fill", {"pssm_threshold": 20000.0, "pssm_log_odds_flag": True}),
  ("pssm_log_odds_flag alone", {"pssm_log_odds_flag": True}),
  ("pssm_bias_flag alone", {"pssm_bias_flag": True}),
  ("pssm_multi alone", {"pssm_multi": 1.0}),
  ("pssm_multi with bias_flag", {"pssm_multi": 1.0, "pssm_bias_flag": True}),
  (
    "all four at once",
    {
      "pssm_multi": 1.0, "pssm_bias_flag": True,
      "pssm_log_odds_flag": True, "pssm_threshold": 20000.0,
    },
  ),
)


def test_pssm_knobs_do_nothing_without_a_pssm_file(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """With no ``pssm_json`` supplied, every pssm knob is inert -- and that is CORRECT.

  THIS TEST CHANGED MEANING ON 261003 AND WAS RENAMED RATHER THAN DELETED. It
  was written as a defect tripwire, when ``pssm_dict`` was never passed to
  ``tied_featurize_port`` and the knobs could not work even WITH a file (debts
  2440, 2443). That is fixed -- ``pssm_json`` now reaches the featurizer and has
  its own ``test_knob_semantics_pssm_json`` above.

  What survives is narrower and is NOT a defect: with no pssm file,
  ``pssm_coef`` and ``pssm_bias`` are zero-filled (``featurize.py:403-404``), so
  scaling or gating them changes nothing. A caller who sets these knobs and
  supplies no PSSM gets default behaviour, which is the only sensible outcome.
  Pinning it stops a future change from making the knobs act on an absent PSSM.

  Still not a ``test_knob_semantics_*`` name: it asserts an absence, and the
  gate should not read that as the knobs being covered.

  ``pssm_threshold``, ``pssm_multi``, ``pssm_log_odds_flag`` and
  ``pssm_bias_flag`` are all read off ``options`` (``sample_host.py:295``,
  ``:332``), so the plumbing lint correctly counts them as wired. They still
  cannot do anything, because their only input is unreachable: ``pssm_json`` is
  inert, and no driver call site passes ``pssm_dict`` to ``tied_featurize_port``
  (``driver.py:572``, ``:727``, ``:765`` pass batch and chain_dict only). Without
  it ``featurize.py:403-405`` fills ``pssm_coef`` and ``pssm_bias`` with zeros and
  ``pssm_log_odds`` with a uniform 10000.0 -- so the log-odds mask is a constant
  array, and multiplying every amino acid by the same constant leaves the
  distribution alone.

  THE CONTROL IS THE POINT. A test asserting "nothing changed" passes trivially
  when the fixture is dead, and this fixture's output is nearly constant, so the
  null result is worthless without it. ``omit_aa`` drives the same
  per-amino-acid masking the log-odds mask feeds (``decode.py:107``), and
  forbidding the residue the sampler keeps choosing must visibly move the output
  before any "unchanged" claim below is allowed to mean anything.
  """
  del registered
  pdb = tmp_path / "toy.pdb"
  _write_pdb(pdb, {"A": "ACDEFG"})

  def _sequence(omit: tuple[str, ...] = (), **options: object) -> np.ndarray:
    spec = SamplingSpecification(
      inputs=str(pdb),
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      num_samples=4,
      samples_chunk_size=2,
      return_logits=False,
      omit_aa=omit,
      potts_mpnn=PottsMPNNOptions(optimization_mode="none", **options),
    )
    return np.asarray(sample(spec)["structures"]["0"]["arrays"]["sequence"])

  default = _sequence()
  assert not np.array_equal(default, _sequence(omit=("S",))), (
    "control failed: forbidding an amino acid did not change the sampled "
    "sequence, so this fixture cannot detect per-residue masking and the "
    "unchanged-output assertions below would pass for any reason at all"
  )

  for label, options in _PSSM_SETTINGS:
    np.testing.assert_array_equal(
      _sequence(**options), default,
      err_msg=(
        f"{label}: a pssm knob changed the output. If pssm_json has now been "
        f"plumbed through tied_featurize_port, that is the fix for debt 2440 -- "
        f"delete this test and write real test_knob_semantics_* tests for "
        f"pssm_threshold, pssm_multi, pssm_log_odds_flag and pssm_bias_flag."
      ),
    )


def test_knob_semantics_checkpoint_id_and_model_local_path(
  registered: PottsMPNNDriver, tmp_path: Path,
) -> None:
  """An upstream weights path maps onto the PAIR (checkpoint_id, model_local_path).

  Nine alias rows carry this one rule -- every upstream entry point spells the
  argument differently (``model_weights``, ``model_weights_path``, ``weights``,
  ``check_path``) and all of them resolve here. Covering it means showing what
  each half of the pair is worth, not merely that a run with both set succeeds.

  Three measured facts, all on this fixture:

  * ``model_local_path`` selects WHICH weights load. Two random-init files give
    completely different energies, so the field is not decorative.
  * ``model_local_path`` WINS OUTRIGHT. A checkpoint_id that names nothing in the
    registry, paired with a valid local path, produces byte-identical output to
    the real id -- the id is not validated when a local path is given.
  * Neither half alone suffices: with no local path and no registry entry
    carrying a sha256, the driver refuses rather than guessing or downloading.
    That refusal is also why this test cannot touch the network.

  THE SECOND FACT IS A PROVENANCE HAZARD, recorded here because the gate depends
  on the opposite assumption: a ledger row's weights key is meant to identify the
  weights a run used (trap 3, which disqualified two runs for carrying an
  absolute path). Since checkpoint_id is accepted unchecked whenever
  model_local_path is set, that key can name one checkpoint while the numbers
  come from another file entirely. Nothing in this sprint relies on it -- the
  wave's keys were verified against the registry sha256 -- but a future reader
  should not infer that checkpoint_id alone is evidence of what ran.
  """
  del registered
  pdb = tmp_path / "toy.pdb"
  _write_pdb(pdb, {"A": "ACDEFG"})
  first = tmp_path / "seed0.eqx"
  second = tmp_path / "seed1.eqx"
  eqx.tree_serialise_leaves(first, PottsMPNN(key=jax.random.PRNGKey(0)))
  eqx.tree_serialise_leaves(second, PottsMPNN(key=jax.random.PRNGKey(1)))

  def _energy(**kwargs: object) -> np.ndarray:
    spec = ScoringSpecification(
      inputs=str(pdb),
      model_family="pottsmpnn",
      output_kind="energy",
      sequences_to_score=["ADDEFG"],
      potts_mpnn=PottsMPNNOptions(),
      **kwargs,
    )
    return np.asarray(
      score(spec)["structures"]["0"]["arrays"]["energy"], dtype=np.float64,
    )

  from_first = _energy(checkpoint_id="pottsmpnn_vanilla_20", model_local_path=first)
  from_second = _energy(checkpoint_id="pottsmpnn_vanilla_20", model_local_path=second)
  assert not np.allclose(from_first, from_second), (
    "two different weight files must give different energies, or "
    "model_local_path is not selecting anything"
  )

  # Same local file, a checkpoint_id that names nothing: the file is what loads.
  bogus = _energy(checkpoint_id="not_a_real_checkpoint", model_local_path=first)
  np.testing.assert_allclose(bogus, from_first, rtol=1e-6, atol=1e-6)

  # Neither half alone. The message names both routes, so a caller missing one
  # is told which to supply rather than being sent to the network.
  with pytest.raises(ValueError, match="model_local_path or a checkpoint registry"):
    _energy(checkpoint_id="pottsmpnn_vanilla_20")


def test_knob_semantics_inputs(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``inputs`` takes one structure or many, and the result is keyed by position.

  Eleven alias rows map onto this field, in two sub-rules that meet here: a
  SINGLE structure (``input_pdb_code``, ``pdb_file``, ``backbone``,
  ``input_pdb``) and a SET of them (an input directory, or the manual input-list
  file). ``driver._inputs`` wraps a str or Path into a one-element list and
  otherwise takes the sequence as given, so both spellings land on the same code
  path -- which is the claim the rows make and the reason one test covers them.

  Four things are asserted, because the cheap version of this test -- "two inputs
  give two entries" -- would pass for an implementation that ran the first
  structure twice:

  * a bare string and a one-element list are equivalent;
  * two structures give two entries, keyed "0" and "1";
  * the keys follow the ORDER GIVEN -- reversing the list swaps the results, so
    the mapping is positional and not, say, sorted by name;
  * each entry equals that structure scored ALONE, which is what rules out the
    first-structure-twice implementation.

  The fixtures differ in sequence (ACDEFG against WWWWWW) and share a backbone,
  so the reference energies differ sharply (-4.775694 against 8.999121) while the
  scored sequence is the same in both -- the contrast is in the structure, which
  is what is being keyed.
  """
  del registered
  alpha = tmp_path / "alpha.pdb"
  beta = tmp_path / "beta.pdb"
  _write_pdb(alpha, {"A": "ACDEFG"})
  _write_pdb(beta, {"A": "WWWWWW"})

  def _structures(inputs: object) -> dict:
    spec = ScoringSpecification(
      inputs=inputs,
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      output_kind="energy",
      sequences_to_score=["ADDEFG"],
      potts_mpnn=PottsMPNNOptions(),
    )
    return score(spec)["structures"]

  def _energy(entry: dict) -> np.ndarray:
    return np.asarray(entry["arrays"]["energy"], dtype=np.float64)

  bare = _structures(str(alpha))
  listed = _structures([str(alpha)])
  assert sorted(bare) == ["0"], sorted(bare)
  assert sorted(listed) == ["0"], sorted(listed)
  np.testing.assert_allclose(_energy(bare["0"]), _energy(listed["0"]))

  alone_alpha = _energy(bare["0"])
  alone_beta = _energy(_structures(str(beta))["0"])
  assert not np.allclose(alone_alpha, alone_beta), (
    "the two fixtures must score differently, or nothing below can tell which "
    "structure landed in which slot"
  )

  both = _structures([str(alpha), str(beta)])
  assert sorted(both) == ["0", "1"], sorted(both)
  np.testing.assert_allclose(_energy(both["0"]), alone_alpha)
  np.testing.assert_allclose(_energy(both["1"]), alone_beta)
  assert both["0"]["structure_id"] == "alpha"
  assert both["1"]["structure_id"] == "beta"

  # Positional, not sorted: reversing the list reverses the slots.
  reversed_ = _structures([str(beta), str(alpha)])
  np.testing.assert_allclose(_energy(reversed_["0"]), alone_beta)
  np.testing.assert_allclose(_energy(reversed_["1"]), alone_alpha)
  assert reversed_["0"]["structure_id"] == "beta"


def test_knob_semantics_output_dir(tmp_path: Path) -> None:
  """``output_dir`` overrides every inferred output root.

  Nine alias rows map an upstream output argument here and all nine say only
  "where outputs are written", so a test that set the field and read it back
  would assert nothing. The substance is the OVERRIDE, documented at
  ``specs.py:217-222`` and implemented as a four-step chain in
  ``spec.py:305-316``:

      explicit output_dir  ->  output_h5_path.parent  ->  cache_path  ->  None

  with ``cache_path`` contributing its PARENT when it looks like a file (it has a
  suffix) and itself when it looks like a directory. Every step is exercised,
  including both cache_path shapes, because the one that matters in practice --
  explicit beats everything -- is only meaningful if the things it beats would
  otherwise have won. So each lower rung is first shown to win on its own, then
  shown to lose.

  No model and no weights: this resolves at spec construction, so it is cheap.
  """
  explicit = tmp_path / "explicit"
  streamed = tmp_path / "streamed" / "out.zarr"
  cache_file = tmp_path / "cached" / "store.h5"
  cache_dir = tmp_path / "cachedir"

  def _resolved(**kwargs: object) -> Path | None:
    spec = SamplingSpecification(
      inputs="a.pdb",
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      num_samples=1,
      return_logits=False,
      potts_mpnn=PottsMPNNOptions(),
      **kwargs,
    )
    return spec.run_spec.io.output_dir

  # Each rung wins when it is the only one set.
  assert _resolved() is None
  assert _resolved(cache_path=str(cache_dir)) == cache_dir
  assert _resolved(cache_path=str(cache_file)) == cache_file.parent
  assert _resolved(output_h5_path=str(streamed)) == streamed.parent

  # The streaming parent outranks cache_path.
  assert _resolved(
    output_h5_path=str(streamed), cache_path=str(cache_file),
  ) == streamed.parent

  # And an explicit output_dir outranks both, which is the documented override.
  assert _resolved(output_dir=str(explicit)) == explicit
  assert _resolved(
    output_dir=str(explicit),
    output_h5_path=str(streamed),
    cache_path=str(cache_file),
  ) == explicit


def test_knob_semantics_batch_size(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``batch_size`` partitions the work and must not change the answer.

  All five alias rows say so in their own notes -- "output-invariance test; maps
  to batch_size" -- so the test the rows ask for is an invariance, not a
  behaviour.

  An invariance assertion is trivially satisfied by an output that never varies,
  so the work is set up to vary first: two structures and five scored sequences,
  twelve energies in all, and the test refuses to interpret the invariance unless
  those twelve are not all the same number.

  THE SAMPLING HALF IS THE STRONGER CLAIM. ``samples_chunk_size`` partitions six
  samples into chunks of 1, 2, 3 and 6, and the sampled sequences come back
  bit-identical every way round. That only holds if each sample's randomness is
  keyed by its own index rather than by its position within a chunk -- an
  implementation that drew per chunk would give four different answers here, and
  would look perfectly reasonable in a single-chunk test.
  """
  del registered
  alpha = tmp_path / "a.pdb"
  beta = tmp_path / "b.pdb"
  _write_pdb(alpha, {"A": "ACDEFG"})
  _write_pdb(beta, {"A": "WWWWWW"})
  sequences = ["ADDEFG", "AWDEFG", "ACDEFW", "WWDEFG", "ACWEFG"]

  def _scores(batch_size: int) -> np.ndarray:
    spec = ScoringSpecification(
      inputs=[str(alpha), str(beta)],
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      output_kind="energy",
      sequences_to_score=sequences,
      batch_size=batch_size,
      potts_mpnn=PottsMPNNOptions(),
    )
    out = score(spec)["structures"]
    return np.concatenate([
      np.asarray(out[key]["arrays"]["energy"], dtype=np.float64)
      for key in sorted(out)
    ])

  baseline = _scores(1)
  assert baseline.size == 12, baseline.size
  assert np.unique(baseline.round(6)).size > 1, (
    "the scored energies must vary, or invariance across batch_size holds "
    "trivially for any implementation that returns a constant"
  )
  for batch_size in (2, 3, 32):
    np.testing.assert_array_equal(_scores(batch_size), baseline)

  def _sampled(chunk: int) -> np.ndarray:
    spec = SamplingSpecification(
      inputs=str(alpha),
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      num_samples=6,
      samples_chunk_size=chunk,
      return_logits=False,
      potts_mpnn=PottsMPNNOptions(optimization_mode="none"),
    )
    return np.asarray(sample(spec)["structures"]["0"]["arrays"]["sequence"])

  one_at_a_time = _sampled(1)
  assert one_at_a_time.shape == (6, 6), one_at_a_time.shape
  for chunk in (2, 3, 6):
    np.testing.assert_array_equal(_sampled(chunk), one_at_a_time)


def test_knob_semantics_num_samples(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``num_samples`` is how many designs come back per input structure.

  The opposite shape to batch_size: this one MUST change the output, and the
  change is the cardinality rather than the values. Every per-sample array grows
  together -- ``sequence`` gains a row and ``sample_energy`` an entry -- so an
  implementation that resized one and not the other is caught.

  ``samples_chunk_size`` is held at 2 throughout, so the counts cannot be an
  artefact of the chunking: 1 is below it, 3 straddles it unevenly and 6 is a
  multiple, and all three come back exact.
  """
  del registered
  pdb = tmp_path / "toy.pdb"
  _write_pdb(pdb, {"A": "ACDEFG"})

  for requested in (1, 3, 6):
    spec = SamplingSpecification(
      inputs=str(pdb),
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      num_samples=requested,
      samples_chunk_size=2,
      return_logits=False,
      potts_mpnn=PottsMPNNOptions(optimization_mode="none"),
    )
    arrays = sample(spec)["structures"]["0"]["arrays"]
    assert arrays["sequence"].shape == (requested, 6), arrays["sequence"].shape
    assert arrays["sample_energy"].shape == (requested,), (
      arrays["sample_energy"].shape
    )
    assert arrays["sample_rank"].shape == (requested,), arrays["sample_rank"].shape


def _write_pdb_jittered(
  path: Path, sequence: str, jitter: float, seed: int,
) -> None:
  """``_write_pdb`` for one chain, with each atom displaced by ``jitter``.

  Needed because ``_write_pdb`` lays atoms out purely by residue index, so two
  structures with different SEQUENCES have identical coordinates. A control for
  a coordinate knob has to move the coordinates.
  """
  rng = np.random.default_rng(seed)
  lines: list[str] = []
  serial = 1
  for index, amino in enumerate(sequence, start=1):
    for offset, atom in enumerate(("N", "CA", "C", "O")):
      shift = rng.normal(scale=jitter, size=3) if jitter else np.zeros(3)
      lines.append(
        _atom(
          serial, atom, _THREE[amino], "A", index,
          x=float(index * 3 + offset) + float(shift[0]),
          y=float(ord("A")) + float(shift[1]),
          z=float(offset) + float(shift[2]),
        ),
      )
      serial += 1
  lines.append("END")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_backbone_noise_is_ignored_on_the_potts_path(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``backbone_noise`` reaches the spec and then nothing (debt 2442).

  THIS TEST ASSERTS A DEFECT, like the pssm one, and is named so the knob gate
  does not credit it as coverage. It exists so the day the defect is fixed, it
  fails and says what to do.

  Spec 6.3: Potts ``noise`` is eval ``augment_eps``, applied to all backbone
  atoms whenever ``augment_eps > 0`` INCLUDING at eval
  (``potts_mpnn_utils.py:1170-1171``), as iid Gaussian per atom. aminx builds the
  bundle correctly -- ``backbone_noise=1.0`` yields
  ``FeatureNoiseBundle(feature_type='backbone', noise_levels=(1.0,),
  mode='direct', enabled=True)`` -- and then consumes it nowhere on this path.
  ``apply_noise_to_coordinates`` (``utils/coordinates.py:42-46``) implements the
  rule, but its only callers live under ``model/``; nothing under ``families/``
  calls it.

  THE CONTROL IS WHAT MAKES THE NULL MEAN ANYTHING, and here it is unusually
  sharp: displacing the same coordinates ON DISK changes the energy at a sigma of
  0.02, which is a five-hundredth of the largest value the knob is given below.
  So the path is not merely sensitive to geometry, it is sensitive far below the
  scale at which the knob does nothing.

  Debt 2434 records the LASEr half of this. There the generic per-atom rule would
  have been WRONG anyway -- spec 6.3 calls iid-per-atom LASEr's negative control,
  since LASEr's own rule is a rigid per-residue translation plus 2-decimal
  rounding. On the Potts side the generic rule is the CORRECT one, and it still
  is not applied.
  """
  del registered
  clean = tmp_path / "clean.pdb"
  _write_pdb_jittered(clean, "ACDEFG", 0.0, 0)

  def _energy(path: Path, **kwargs: object) -> np.ndarray:
    spec = ScoringSpecification(
      inputs=str(path),
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      output_kind="energy",
      sequences_to_score=["ADDEFG"],
      potts_mpnn=PottsMPNNOptions(),
      **kwargs,
    )
    return np.asarray(
      score(spec)["structures"]["0"]["arrays"]["energy"], dtype=np.float64,
    )

  baseline = _energy(clean)

  # Control: the smallest displacement tested already moves the energy.
  jittered = tmp_path / "jittered.pdb"
  _write_pdb_jittered(jittered, "ACDEFG", 0.02, 7)
  assert not np.allclose(baseline, _energy(jittered)), (
    "control failed: displacing the coordinates by sigma=0.02 did not change "
    "the energy, so this fixture cannot detect a coordinate perturbation and "
    "the assertions below would hold for any reason at all"
  )

  # The bundle is built, so the spec half of the plumbing works.
  spec = ScoringSpecification(
    inputs=str(clean),
    model_family="pottsmpnn",
    checkpoint_id="pottsmpnn_vanilla_20",
    model_local_path=model_path,
    output_kind="energy",
    sequences_to_score=["ADDEFG"],
    backbone_noise=1.0,
    potts_mpnn=PottsMPNNOptions(),
  )
  assert len(spec.noise) == 1, spec.noise
  assert spec.noise[0].feature_type == "backbone"
  assert spec.noise[0].noise_levels == (1.0,)
  assert spec.noise[0].enabled is True

  # And it changes nothing, at any magnitude, including 500x the control's.
  for level in (0.02, 0.5, 1.0, 2.0, 10.0):
    np.testing.assert_array_equal(
      _energy(clean, backbone_noise=level), baseline,
      err_msg=(
        f"backbone_noise={level} changed the energy. If the Potts path now "
        f"applies augment_eps, that is the fix for debt 2442 -- delete this "
        f"test and write test_knob_semantics_noise asserting iid Gaussian "
        f"N/CA/C/O per atom against an oracle with injected randn (spec 6.3)."
      ),
    )


def test_knob_semantics_random_seed(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``random_seed`` fixes the sampling, and different seeds genuinely diverge.

  Spec :786 derives the decoding order from it --
  ``u = jax.random.uniform(key_order, (L,))`` -- and :879 maps upstream
  ``fix_decoding_order`` and ``decoding_order_offset`` onto it as a "seeded-order
  test", noting that the order source is the DRIVER and not ``decoding_order_fn``.
  :1173 states the same requirement from the golden side: ``random_seed=1`` must
  differ.

  Both halves are asserted, because each alone is satisfiable by a wrong
  implementation:

  * reproducibility alone holds for an implementation that ignores the seed
    entirely and samples deterministically;
  * divergence alone holds for one that reseeds from entropy every call, which
    would also make the goldens unreproducible.

  A SIXTEEN-RESIDUE CHAIN, NOT THE USUAL SIX. On the short fixture this model is
  peaked enough that every seed can agree, and the test would then be asserting
  that two constants are equal. The guard is explicit rather than implicit: the
  four samples drawn in a single run must not be identical to each other, which
  is what establishes the sampler is stochastic here at all.
  """
  del registered
  pdb = tmp_path / "toy.pdb"
  _write_pdb(pdb, {"A": "ACDEFGHIKLMNPQRS"})

  def _sampled(**kwargs: object) -> np.ndarray:
    spec = SamplingSpecification(
      inputs=str(pdb),
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      num_samples=4,
      samples_chunk_size=2,
      return_logits=False,
      potts_mpnn=PottsMPNNOptions(optimization_mode="none"),
      **kwargs,
    )
    return np.asarray(sample(spec)["structures"]["0"]["arrays"]["sequence"])

  zero = _sampled(random_seed=0)
  assert zero.shape == (4, 16), zero.shape

  # The sampler must actually be stochastic on this fixture, or "different seeds
  # give different answers" is a statement about nothing.
  assert len({tuple(row) for row in zero.tolist()}) > 1, (
    "all four samples in one run are identical, so this fixture cannot show "
    "that a seed changed anything"
  )

  np.testing.assert_array_equal(_sampled(random_seed=0), zero)

  for other in (1, 7):
    assert not np.array_equal(_sampled(random_seed=other), zero), (
      f"random_seed={other} produced the same samples as random_seed=0; the "
      f"seed is not reaching the decoding order (spec :786)"
    )


_ALPHA = "ACDEFGHIKLMNPQRSTVWYX"


def _letters(row: np.ndarray) -> str:
  return "".join(_ALPHA[int(i)] for i in row)


def test_knob_semantics_bias(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``bias`` has three accepted shapes, and the two 1-D ones can collide.

  ``_split_bias`` (``sample_host.py:241-267``) accepts ``(21,)`` as a global
  per-amino-acid bias, ``(L,)`` as per-position, and ``(L, 21)`` as per-position
  per-amino-acid.

  THE COLLISION IS THE POINT. Its docstring says a 1-D vector whose length is the
  chain is per-position "including when that length is also 21" -- so on a
  21-residue chain a length-21 vector must NOT be read as the global bias. The
  branch ORDER is what implements that, and swapping the two ``if``s would be
  invisible on every chain whose length is not 21. Both sides are therefore
  exercised on fixtures of length 6 and length 21.

  A one-hot bias separates the readings sharply. Read globally, that amino acid
  is favoured at EVERY position; read per-position, every amino acid at ONE
  position is shifted by the same amount, which changes nothing about which wins
  there. Measured: on L=6 a one-hot on G turns SSSSSS into GGGGGG; on L=21 the
  identical vector leaves the sequence untouched.
  """
  del registered
  short = tmp_path / "short.pdb"
  long_ = tmp_path / "long.pdb"
  _write_pdb(short, {"A": "ACDEFG"})
  _write_pdb(long_, {"A": "ACDEFGHIKLMNPQRSTVWYA"})  # L == 21, == len(alphabet)

  def _sampled(pdb: Path, **kwargs: object) -> np.ndarray:
    spec = SamplingSpecification(
      inputs=str(pdb),
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      num_samples=2,
      samples_chunk_size=1,
      return_logits=False,
      random_seed=0,
      potts_mpnn=PottsMPNNOptions(optimization_mode="none"),
      **kwargs,
    )
    return np.asarray(sample(spec)["structures"]["0"]["arrays"]["sequence"])

  gly = _ALPHA.index("G")
  one_hot = np.zeros(21, dtype=np.float32)
  one_hot[gly] = 50.0

  # (21,) on a chain that is NOT 21 long: global, so G wins everywhere.
  unbiased = _sampled(short)
  assert "G" not in _letters(unbiased[0]), (
    f"the unbiased sample is {_letters(unbiased[0])}; it must not already be G "
    f"or the global-bias assertion proves nothing"
  )
  assert _letters(_sampled(short, bias=one_hot)[0]) == "GGGGGG"

  # (L, 21) is per-position per-amino-acid and reaches the same place here.
  grid = np.zeros((6, 21), dtype=np.float32)
  grid[:, gly] = 50.0
  assert _letters(_sampled(short, bias=grid)[0]) == "GGGGGG"

  # The collision: on L == 21 the SAME vector is per-position, so biasing
  # position 5 equally across all amino acids changes nothing.
  base_long = _sampled(long_)
  assert base_long.shape == (2, 21), base_long.shape
  np.testing.assert_array_equal(_sampled(long_, bias=one_hot), base_long)
  # ...and the reading really is the per-position one, not "bias does nothing on
  # long chains": the same vector reshaped to (21, 21) with a G column IS
  # per-position per-amino-acid, and that does bias towards G.
  long_grid = np.zeros((21, 21), dtype=np.float32)
  long_grid[:, gly] = 50.0
  assert _letters(_sampled(long_, bias=long_grid)[0]) == "G" * 21

  # An unusable shape names all three accepted ones rather than guessing.
  with pytest.raises(ValueError, match=r"bias shape \(7,\) is not"):
    _sampled(short, bias=np.zeros(7, dtype=np.float32))


def test_knob_semantics_fixed_positions(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``fixed_positions`` holds those positions at their native residue.

  ``sample_host.py:282-285`` zeroes ``chain_m_pos`` at each listed index, so the
  sampler does not design there. Indices are 0-based and out-of-range entries are
  skipped rather than raising.

  Measured on native ACDEFG: unconstrained the sampler returns SSSSSS, and fixing
  [0, 1, 2] returns ACDSSS -- the first three positions are exactly the native
  residues and the rest still move. Asserting both halves matters: an
  implementation that froze the WHOLE sequence would also keep the native prefix.
  """
  del registered
  pdb = tmp_path / "toy.pdb"
  _write_pdb(pdb, {"A": "ACDEFG"})

  def _sampled(**kwargs: object) -> str:
    spec = SamplingSpecification(
      inputs=str(pdb),
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      num_samples=2,
      samples_chunk_size=1,
      return_logits=False,
      random_seed=0,
      potts_mpnn=PottsMPNNOptions(optimization_mode="none"),
      **kwargs,
    )
    return _letters(
      np.asarray(sample(spec)["structures"]["0"]["arrays"]["sequence"])[0],
    )

  free = _sampled()
  assert free[:3] != "ACD", (
    f"the free sample is {free}; it must not already start with the native ACD "
    f"or fixing those positions would be indistinguishable from not fixing them"
  )

  fixed = _sampled(fixed_positions=[0, 1, 2])
  assert fixed[:3] == "ACD", fixed
  assert fixed[3:] == free[3:], (
    f"unfixed positions must be unaffected: {fixed} vs {free}"
  )

  # An empty list fixes nothing, so it must match the free sample exactly.
  assert _sampled(fixed_positions=[]) == free


def test_knob_semantics_sequences_to_score(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``sequences_to_score`` names the sequences to evaluate on this backbone.

  Three things, each of which a careless implementation gets wrong differently:

  * the NATIVE reference is prepended, so N sequences give N+1 energies
    (``include_reference_in_output``, ``driver.py:206-207``) and the reference is
    the same value whatever is scored alongside it;
  * every entry must have length ``L_total`` in A0 chain order, and the error
    names both the expected length and the chain order (spec :511) rather than
    failing somewhere downstream on a shape;
  * residues outside the etab alphabet are rejected by name.

  Measured on native ACDEFG: one sequence gives 2 energies, three give 4, and the
  first entry is the native in both -- which is what pins "prepended reference"
  rather than "first requested sequence".
  """
  del registered
  pdb = tmp_path / "toy.pdb"
  _write_pdb(pdb, {"A": "ACDEFG"})

  def _energies(sequences: list[str]) -> np.ndarray:
    spec = ScoringSpecification(
      inputs=str(pdb),
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      output_kind="energy",
      sequences_to_score=sequences,
      potts_mpnn=PottsMPNNOptions(),
    )
    return np.asarray(
      score(spec)["structures"]["0"]["arrays"]["energy"], dtype=np.float64,
    )

  one = _energies(["ADDEFG"])
  three = _energies(["ADDEFG", "AWDEFG", "ACDEFW"])
  assert one.shape == (2,), one.shape
  assert three.shape == (4,), three.shape

  # The reference is prepended, so it does not depend on what else was asked
  # for, and the first requested sequence keeps its slot behind it.
  np.testing.assert_allclose(three[0], one[0], rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(three[1], one[1], rtol=1e-6, atol=1e-6)
  assert len(np.unique(three.round(6))) == 4, (
    f"the four energies must differ, or ordering cannot be checked: {three}"
  )

  with pytest.raises(ValueError, match=r"has length 5; expected 6 in chain order"):
    _energies(["ADDEF"])
  with pytest.raises(ValueError, match="residue outside"):
    _energies(["ADDEFZ"])


def test_tied_positions_is_ignored_by_the_potts_driver(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``spec.tied_positions`` is validated and then discarded (debt 2443).

  ANOTHER TEST THAT ASSERTS A DEFECT, named so the gate does not credit it.

  ``tied_featurize_port`` takes a ``tied_positions_dict`` (``featurize.py:335``)
  and all three call sites in the Potts driver (``:572``, ``:727``, ``:765``)
  pass only ``batch`` and ``chain_dict``, so that argument is always ``None``.
  The same omission is why ``pssm_json`` and ``bias_by_res_json`` are inert --
  one root cause, three symptoms that each look like an isolated bug.

  THE CONTROL IS A BIAS THAT MAKES THE TIED POSITIONS DISAGREE. Tying is only
  observable if the positions would otherwise differ, so a (6, 21) bias drives
  position 0 to W and position 3 to K; untied they come back W..K.., and if the
  tie were honoured they would have to agree. Every accepted form of the field
  is tried, because "the tuple form is unsupported" and "the field is discarded"
  look identical from one of them.

  The VALIDATION still fires -- ``tied_positions='auto'`` without
  ``pass_mode='inter'`` raises -- which is the detail that makes this field look
  wired from the outside.
  """
  del registered
  pdb = tmp_path / "toy.pdb"
  _write_pdb(pdb, {"A": "ACDEFG"})
  grid = np.zeros((6, 21), dtype=np.float32)
  grid[0, _ALPHA.index("W")] = 40.0
  grid[3, _ALPHA.index("K")] = 40.0

  def _sampled(**kwargs: object) -> str:
    spec = SamplingSpecification(
      inputs=str(pdb),
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      num_samples=2,
      samples_chunk_size=1,
      return_logits=False,
      random_seed=0,
      bias=grid,
      potts_mpnn=PottsMPNNOptions(optimization_mode="none"),
      **kwargs,
    )
    return _letters(
      np.asarray(sample(spec)["structures"]["0"]["arrays"]["sequence"])[0],
    )

  untied = _sampled()
  assert untied[0] != untied[3], (
    f"control failed: the bias must make positions 0 and 3 disagree, else a tie "
    f"would be undetectable. Got {untied}"
  )

  for label, kwargs in (
    ("list of tuples", {"tied_positions": [(0, 3)]}),
    ("tuple of tuples", {"tied_positions": ((0, 3),)}),
    ("auto", {"tied_positions": "auto", "pass_mode": "inter"}),
    ("direct", {"tied_positions": "direct", "pass_mode": "inter"}),
  ):
    got = _sampled(**kwargs)
    assert got == untied, (
      f"tied_positions ({label}) changed the sample to {got}. If the driver now "
      f"passes a tied_positions_dict to tied_featurize_port, that is the fix for "
      f"debt 2443 -- delete this test and write test_knob_semantics_tied_positions "
      f"asserting that tied positions take the same residue."
    )

  # Validated, then discarded: the check still fires.
  with pytest.raises(ValueError, match="pass_mode must be 'inter'"):
    _sampled(tied_positions="auto")


def test_fixed_mask_and_tied_beta_are_still_inert(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """Two fields that still reach nothing (debt 2435).

  WAS THREE. ``bias_by_res_json`` was here until 261003 and is now plumbed --
  the driver passes ``bias_by_res_dict`` to ``tied_featurize_port`` (debt 2443)
  and it has a real knob test above. This test fired when that landed, which is
  what it was for, and the field was removed rather than the assertion relaxed.

  A THIRD TEST THAT ASSERTS DEFECTS, named so the gate does not credit it.

  * ``spec.fixed_mask`` -- the Potts featurize builds a LOCAL called
    ``fixed_mask`` from ``fixed_position_dict`` (``featurize.py:388-392``), which
    the driver never passes (debt 2443). The spec field of the same name is read
    nowhere on this path, so it is a fourth symptom of that one omission, not a
    separate bug.
  * ``options.bias_by_res_json`` -- allowlisted inert (debt 2435), and inert for
    the same reason: ``bias_by_res_dict`` is another of the five arguments the
    driver never passes.
  * ``options.tied_beta`` -- allowlisted inert for a different reason:
    ``options.tied_beta`` appears nowhere in ``src/``. The name belongs to a
    per-position array built by ``featurize._tied_groups`` and read off
    ``features``/``padded``/``ready``, which is exactly what hid it from the
    old token-based plumbing lint.

  THE CONTROL AND THE BIAS_BY_RES CASE ARE THE SAME EXPERIMENT RUN TWICE, which
  is what makes this sharp rather than merely negative. A per-residue bias
  favouring W is supplied two ways: as ``spec.bias``, where it moves the sample
  from SSSSSS to WSSSSS, and as ``options.bias_by_res_json``, where it does
  nothing. Same intent, same magnitude, different route, opposite outcome.

  ``fixed_mask`` all-zeros is the sharpest of the three on its own terms: if
  honoured it would mean nothing is designable, so the sample would have to stay
  at the native ACDEFG. It comes back SSSSSS.
  """
  del registered
  pdb = tmp_path / "toy.pdb"
  _write_pdb(pdb, {"A": "ACDEFG"})

  w_column = [
    [50.0 if i == _ALPHA.index("W") else 0.0 for i in range(21)] for _ in range(6)
  ]
  bias_json = tmp_path / "bias_by_res.json"
  bias_json.write_text(f'{{"toy": {{"A": {w_column}}}}}', encoding="utf-8")

  def _sampled(options: dict | None = None, **kwargs: object) -> str:
    spec = SamplingSpecification(
      inputs=str(pdb),
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      num_samples=2,
      samples_chunk_size=1,
      return_logits=False,
      random_seed=0,
      potts_mpnn=PottsMPNNOptions(optimization_mode="none", **(options or {})),
      **kwargs,
    )
    return _letters(
      np.asarray(sample(spec)["structures"]["0"]["arrays"]["sequence"])[0],
    )

  native = "ACDEFG"
  baseline = _sampled()
  assert baseline != native, (
    f"the sampler must redesign the native, or 'fixed_mask does nothing' and "
    f"'fixed_mask fixes everything' look the same. Got {baseline}"
  )

  # CONTROL: the identical per-residue W bias, by the route that works.
  grid = np.zeros((6, 21), dtype=np.float32)
  grid[0, _ALPHA.index("W")] = 40.0
  assert _sampled(bias=grid) != baseline, (
    "control failed: a per-residue bias did not move the sample, so nothing "
    "below can be read as a knob failing"
  )

  reason = (
    "If this now differs, the driver has started passing the dicts to "
    "tied_featurize_port (debt 2443) or the field has been plumbed (2435) -- "
    "delete this case and write a real test_knob_semantics_* test for it."
  )

  # fixed_mask: all-zeros would mean nothing is designable.
  for mask in ([0.0] * 6, [1.0] * 6, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]):
    got = _sampled(fixed_mask=np.asarray(mask, dtype=np.float32))
    assert got == baseline, f"fixed_mask={mask} changed the sample to {got}. {reason}"

  for value in (0.0, 5.0, 100.0):
    got = _sampled(options={"tied_beta": value})
    assert got == baseline, f"tied_beta={value} changed the sample to {got}. {reason}"


def _json_file(path: Path, payload: dict) -> str:
  """Upstream reads these as JSONL -- one object per line."""
  path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
  return str(path)


def test_knob_semantics_bias_by_res_json(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``bias_by_res_json`` adds a per-residue, per-amino-acid bias.

  ``bias_by_res_dict[name][chain]`` is an ``(L, 21)`` table that
  ``featurize.py:406-409`` folds into the per-residue bias. It was INERT until
  261003 -- the driver never passed ``bias_by_res_dict`` to
  ``tied_featurize_port`` at all (debt 2443) -- so this test is new coverage of
  a field that previously parsed and reached nothing.

  THE SAME EXPERIMENT BY TWO ROUTES is what makes the assertion sharp rather
  than merely positive: an identical G-favouring per-residue bias is supplied
  as ``spec.bias`` and as ``bias_by_res_json``, and both must now move the
  sample. Before the fix the first worked and the second did nothing, which is
  exactly the asymmetry the old tripwire recorded.
  """
  del registered
  pdb = tmp_path / "toy.pdb"
  _write_pdb(pdb, {"A": "ACDEFG"})
  gly = _ALPHA.index("G")
  table = [[50.0 if i == gly else 0.0 for i in range(21)] for _ in range(6)]

  def _sampled(**options: object) -> str:
    spec = SamplingSpecification(
      inputs=str(pdb),
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      num_samples=2,
      samples_chunk_size=1,
      return_logits=False,
      random_seed=0,
      potts_mpnn=PottsMPNNOptions(optimization_mode="none", **options),
    )
    return _letters(
      np.asarray(sample(spec)["structures"]["0"]["arrays"]["sequence"])[0],
    )

  baseline = _sampled()
  assert "G" not in baseline, f"fixture must not already be G: {baseline}"

  path = _json_file(tmp_path / "bias.json", {"toy": {"A": table}})
  assert _sampled(bias_by_res_json=path) == "GGGGGG"

  # A payload for another structure RAISES, and that is upstream-faithful.
  # potts_mpnn_utils.py:436-437 does `bias_by_res_dict[b['name']][letter]` with
  # no membership check, so a file that does not cover this structure is an
  # error rather than a no-op. That is the OPPOSITE of mutant_csv and
  # chain_design_mask_json, which filter on the name and silently contribute
  # nothing -- a difference worth pinning, since "keyed by structure name"
  # describes both and predicts the wrong behaviour for one of them.
  other = _json_file(tmp_path / "other.json", {"not_toy": {"A": table}})
  with pytest.raises(KeyError, match="toy"):
    _sampled(bias_by_res_json=other)


def test_knob_semantics_pssm_json(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``pssm_json`` supplies pssm_coef / pssm_bias / pssm_log_odds per chain.

  Also inert until 261003 for the same reason (debt 2443): ``pssm_dict`` was
  never passed. With it passed, the coefficient-and-bias half demonstrably
  reaches the sampler.

  NOT ALL OF IT IS REACHABLE YET. ``pssm_log_odds`` feeds a mask that this
  sampling path does not consume -- measured, with log-odds of +10 for G and
  -10 elsewhere and thresholds of -20, 0 and 20 straddling them, all three give
  the identical sequence. So ``pssm_threshold`` and ``pssm_log_odds_flag``
  remain uncovered and keep their tripwire; this test deliberately claims only
  the half that works, rather than asserting a pass the implementation does not
  earn.
  """
  del registered
  pdb = tmp_path / "toy.pdb"
  _write_pdb(pdb, {"A": "ACDEFG"})
  gly = _ALPHA.index("G")
  payload = {
    "toy": {
      "A": {
        "pssm_coef": [1.0] * 6,
        "pssm_bias": [[1.0 if i == gly else 0.0 for i in range(21)] for _ in range(6)],
        "pssm_log_odds": [[10.0 if i == gly else -10.0 for i in range(21)] for _ in range(6)],
      },
    },
  }

  def _sampled(**options: object) -> str:
    spec = SamplingSpecification(
      inputs=str(pdb),
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      num_samples=2,
      samples_chunk_size=1,
      return_logits=False,
      random_seed=0,
      potts_mpnn=PottsMPNNOptions(optimization_mode="none", **options),
    )
    return _letters(
      np.asarray(sample(spec)["structures"]["0"]["arrays"]["sequence"])[0],
    )

  baseline = _sampled()
  assert "G" not in baseline, f"fixture must not already be G: {baseline}"

  path = _json_file(tmp_path / "pssm.json", payload)
  assert _sampled(pssm_json=path, pssm_bias_flag=True, pssm_multi=1.0) == "GGGGGG"

  # Without the flag that consumes it, the same file changes nothing -- so the
  # effect above is the BIAS being applied, not merely the file being read.
  assert _sampled(pssm_json=path) == baseline


def test_pssm_log_odds_half_is_still_unreachable(
  registered: PottsMPNNDriver, model_path: Path, tmp_path: Path,
) -> None:
  """``pssm_threshold`` still cannot change a sample (debt 2440, part).

  Asserts a defect, so it is not a ``test_knob_semantics_*`` name. Narrower than
  the original tripwire: ``pssm_json`` and ``pssm_bias_flag`` now work, and only
  the log-odds mask does not.

  The thresholds STRADDLE the log-odds values (+10 for G, -10 elsewhere), so
  -20 would keep every amino acid, 0 only G, and 20 none. All three return the
  same sequence, which is what makes this a reach failure rather than a badly
  chosen fixture -- the first attempt used 0 and 5, which select the same mask
  and would have proved nothing.
  """
  del registered
  pdb = tmp_path / "toy.pdb"
  _write_pdb(pdb, {"A": "ACDEFG"})
  gly = _ALPHA.index("G")
  path = _json_file(
    tmp_path / "pssm.json",
    {
      "toy": {
        "A": {
          "pssm_coef": [1.0] * 6,
          "pssm_bias": [[0.0] * 21 for _ in range(6)],
          "pssm_log_odds": [
            [10.0 if i == gly else -10.0 for i in range(21)] for _ in range(6)
          ],
        },
      },
    },
  )

  def _sampled(threshold: float) -> str:
    spec = SamplingSpecification(
      inputs=str(pdb),
      model_family="pottsmpnn",
      checkpoint_id="pottsmpnn_vanilla_20",
      model_local_path=model_path,
      num_samples=2,
      samples_chunk_size=1,
      return_logits=False,
      random_seed=0,
      potts_mpnn=PottsMPNNOptions(
        optimization_mode="none",
        pssm_json=path,
        pssm_log_odds_flag=True,
        pssm_threshold=threshold,
      ),
    )
    return _letters(
      np.asarray(sample(spec)["structures"]["0"]["arrays"]["sequence"])[0],
    )

  outcomes = {_sampled(t) for t in (-20.0, 0.0, 20.0)}
  assert len(outcomes) == 1, (
    f"pssm_threshold changed the sample: {outcomes}. If the log-odds mask now "
    f"reaches this path, that is the rest of debt 2440 -- delete this test and "
    f"write test_knob_semantics_pssm_threshold."
  )
