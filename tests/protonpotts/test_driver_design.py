"""ProtonPottsDriver pH design (``sample``) and selectivity (``score:selectivity``), end to end on a toy PDB.

Fixtures and the random-model helper are copied from ``test_driver.py``. The models are random, so these tests
check the driver's contract (shapes, dtypes, pinned tokens, ordering, errors, determinism, the selectivity
recomputation), not the energies' agreement with upstream. That is settled by the ``protonpotts_energy`` wave.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.model import PottsMPNN
from aminx.families.protonpotts_mpnn import ProtonPottsDriver
from aminx.families.protonpotts_mpnn.energy import protonpotts_table
from aminx.families.protonpotts_mpnn.ph_config import DEFAULT_DEP_MAP
from aminx.families.protonpotts_mpnn.ph_potentials import candidate_energies_at
from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6, token_index
from aminx.host.family_driver import FAMILY_DRIVERS
from aminx.host.runner import sample, score
from aminx.run.options import ProtonPottsOptions
from aminx.run.specs import SamplingSpecification, ScoringSpecification

RESIDUES = ["ALA", "HIS", "ASP", "GLU", "HIS", "ALA", "GLY", "LYS", "SER", "THR"]
HIS_P = token_index("HIS-P")


def _atom(
  serial: int,
  name: str,
  resname: str,
  chain: str,
  number: int,
  xyz: tuple[float, float, float],
) -> str:
  padded = f" {name:<3}"
  return (
    f"ATOM  {serial:>5} {padded} {resname:>3} {chain}{number:>4}    "
    f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}{1.0:>6.2f}{20.0:>6.2f}          {name[0]:>2}"
  )


def _write_pdb(path: Path, chains: list[tuple[str, list[str], float]]) -> Path:
  """Backbone-only residues. Each chain is ``(letter, residue names, x offset)``, numbered from 1."""
  rng = np.random.default_rng(0)
  lines: list[str] = []
  for chain, names, offset in chains:
    for i, resname in enumerate(names, start=1):
      centre = np.array([3.8 * i + offset, 2.0 * np.sin(i), 2.0 * np.cos(i)])
      for k, atom in enumerate(("N", "CA", "C", "O")):
        xyz = centre + rng.normal(scale=0.4, size=3) + np.array([0.6 * k, 0.0, 0.0])
        lines.append(_atom(len(lines) + 1, atom, resname, chain, i, (xyz[0], xyz[1], xyz[2])))
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return path


@pytest.fixture
def pdb(tmp_path: Path) -> Path:
  return _write_pdb(tmp_path / "toy.pdb", [("A", RESIDUES, 0.0)])


@pytest.fixture
def two_chain_pdb(tmp_path: Path) -> Path:
  """Chain A has ONE residue (fewer free binder positions than two centre types, so no plan); chain B is ten."""
  return _write_pdb(tmp_path / "two_chain.pdb", [("A", ["HIS"], 0.0), ("B", RESIDUES, 50.0)])


@pytest.fixture
def registered() -> Iterator[ProtonPottsDriver]:
  driver = ProtonPottsDriver()
  FAMILY_DRIVERS.register("protonpottsmpnn")(driver)
  yield driver
  FAMILY_DRIVERS.discard("protonpottsmpnn")


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
  path = tmp_path / "tiny.eqx"
  eqx.tree_serialise_leaves(path, PottsMPNN(key=jax.random.PRNGKey(0), alphabet=PROTONPOTTS_V6))
  return path


def _json(tmp_path: Path, name: str, payload: object) -> str:
  path = tmp_path / name
  path.write_text(json.dumps(payload), encoding="utf-8")
  return str(path)


def _design_options(**overrides: Any) -> ProtonPottsOptions:  # noqa: ANN401
  values: dict[str, Any] = {
    "binder_chain": "A",
    "center_types": ("HIS-P",),
    "block_size": 2,
    "samples_per_site": 2,
    "neighbour_k": 0,
    "max_mutations": 0,
    "block_max_rounds": 3,
    "temperature": 0.05,
  }
  values.update(overrides)
  return ProtonPottsOptions(**values)


def _sample_spec(
  inputs: object,
  model_path: Path,
  options: ProtonPottsOptions,
  seed: int = 42,
) -> SamplingSpecification:
  return SamplingSpecification(
    inputs=inputs,  # type: ignore[arg-type]
    model_family="protonpottsmpnn",
    model_local_path=model_path,
    random_seed=seed,
    protonpotts=options,
  )


def _designs(pdb: Path, model_path: Path, options: ProtonPottsOptions, seed: int = 42) -> dict[str, Any]:
  spec = _sample_spec(str(pdb), model_path, options, seed)
  return sample(spec)["structures"]["0"]["arrays"]


# --- sample ------------------------------------------------------------------------------------


def test_sample_returns_schema_shaped_designs(
  registered: ProtonPottsDriver, pdb: Path, model_path: Path
) -> None:
  arrays = _designs(pdb, model_path, _design_options())

  assert set(arrays) == set(registered.result_schema(None, "sample"))
  n = arrays["sequence"].shape[0]
  assert n == 2  # samples_per_site, one plan
  assert arrays["sequence"].shape == (n, len(RESIDUES))
  assert arrays["sequence"].dtype == np.int32
  assert arrays["final_potts_energy"].shape == (n,)
  assert arrays["final_potts_energy"].dtype == np.float32
  assert arrays["selective_energy"].shape == (n,)
  assert arrays["selective_energy"].dtype == np.float32
  assert arrays["design_sample"].tolist() in ([0, 1], [1, 0])
  assert arrays["center_positions"].shape == (n, 1)
  assert arrays["center_types"].shape == (n, 1)
  assert np.isfinite(arrays["final_potts_energy"]).all()
  assert np.isfinite(arrays["selective_energy"]).all()

  schema = registered.result_schema(None, "sample")
  assert str(schema["sequence"].attrs["vocabulary"]).split(",") == list(PROTONPOTTS_V6.symbols)
  assert str(schema["center_types"].attrs["vocabulary"]).split(",") == list(PROTONPOTTS_V6.symbols)


def test_sample_pins_the_centre_and_sorts_by_energy(
  registered: ProtonPottsDriver, pdb: Path, model_path: Path
) -> None:
  del registered
  arrays = _designs(pdb, model_path, _design_options())

  positions = arrays["center_positions"][:, 0]
  assert (positions == positions[0]).all()  # one plan: every design pins the same centre
  assert 0 <= int(positions[0]) < len(RESIDUES)
  # Placement has no residue-type restriction (spec §45.2): the best-scoring free position becomes the centre,
  # whatever residue it is, so only the protonated token at that position is guaranteed (checked below).
  for row, position in zip(arrays["sequence"], positions, strict=True):
    assert row[position] == HIS_P
  assert (arrays["center_types"] == HIS_P).all()
  assert not (arrays["sequence"] == token_index("X")).any()  # UNK is never sampled
  energies = arrays["final_potts_energy"]
  assert np.all(np.diff(energies) >= 0.0)  # sorted ascending


def test_temperature_zero_is_deterministic(
  registered: ProtonPottsDriver, pdb: Path, model_path: Path
) -> None:
  del registered
  cold = _design_options(temperature=0.0)
  first = _designs(pdb, model_path, cold, seed=1)
  again = _designs(pdb, model_path, cold, seed=2)  # no randomness at temperature 0, whatever the seed
  assert np.array_equal(first["sequence"], again["sequence"])
  assert np.array_equal(first["final_potts_energy"], again["final_potts_energy"])


def test_same_seed_repeats_and_different_seeds_differ(
  registered: ProtonPottsDriver, pdb: Path, model_path: Path
) -> None:
  del registered
  options = _design_options(temperature=0.05)
  base = _designs(pdb, model_path, options, seed=7)
  again = _designs(pdb, model_path, options, seed=7)
  assert np.array_equal(base["sequence"], again["sequence"])
  others = [_designs(pdb, model_path, options, seed=s)["sequence"] for s in range(8)]
  assert any(not np.array_equal(base["sequence"], other) for other in others)


def test_num_samples_above_one_is_refused(
  registered: ProtonPottsDriver, pdb: Path, model_path: Path
) -> None:
  del registered
  spec = SamplingSpecification(
    inputs=str(pdb),
    model_family="protonpottsmpnn",
    model_local_path=model_path,
    num_samples=2,
    protonpotts=_design_options(),
  )
  with pytest.raises(ValueError, match="num_samples=1"):
    sample(spec)


@pytest.mark.parametrize("binder", [None, "Z"])
def test_missing_or_unknown_binder_chain_names_the_chains(
  registered: ProtonPottsDriver, pdb: Path, model_path: Path, binder: str | None
) -> None:
  del registered
  with pytest.raises(ValueError, match=r"Chains present: \['A'\]"):
    sample(_sample_spec(str(pdb), model_path, _design_options(binder_chain=binder)))


def test_structure_without_a_plan_is_reported_as_skipped(
  registered: ProtonPottsDriver, pdb: Path, two_chain_pdb: Path, model_path: Path
) -> None:
  del registered
  # Chain A has one residue, so two centre types have no placement plan there (upstream returns []).
  options = _design_options(center_types=("HIS-P", "ASP-P"))
  out = sample(_sample_spec([str(pdb), str(two_chain_pdb)], model_path, options))

  skipped = out["skipped_inputs"]
  assert [entry["input_index"] for entry in skipped] == [1]
  assert "no pH design plan" in str(skipped[0]["reason"])
  assert set(out["structures"]) == {"0"}  # the first structure has a plan and is designed
  assert out["structures"]["0"]["arrays"]["sequence"].shape[1] == len(RESIDUES)


# --- score:selectivity -------------------------------------------------------------------------


def _selectivity_spec(
  pdb: Path,
  model_path: Path,
  options: ProtonPottsOptions,
  variants: str | None = None,
) -> ScoringSpecification:
  kwargs: dict[str, Any] = {}
  if variants is not None:
    kwargs["variants_json"] = variants
  return ScoringSpecification(
    inputs=str(pdb),
    model_family="protonpottsmpnn",
    model_local_path=model_path,
    output_kind="selectivity",  # type: ignore[arg-type]
    protonpotts=ProtonPottsOptions(
      binder_chain=options.binder_chain,
      explicit_centers=options.explicit_centers,
      center_types=(),
      **kwargs,
    ),
  )


def test_selectivity_matches_an_independent_recomputation(
  registered: ProtonPottsDriver, pdb: Path, model_path: Path, tmp_path: Path
) -> None:
  variants = _json(tmp_path, "v.json", {"second_his": {"A:5": "HIS-P"}})
  options = ProtonPottsOptions(binder_chain="A", explicit_centers=((2, "HIS-P"),))
  spec = _selectivity_spec(pdb, model_path, options, variants)
  arrays = score(spec)["structures"]["0"]["arrays"]

  assert arrays["candidate_ids"].tolist() == [0, 1]  # reference first, then the variant
  assert arrays["selectivity"].shape == (2,)
  assert arrays["selectivity_per_centre"].shape == (2, 1)
  assert arrays["candidate_tokens"].shape == (2, len(RESIDUES))
  assert arrays["candidate_tokens"].dtype == np.int32
  assert arrays["selectivity"].dtype == np.float32
  vocabulary = registered.result_schema(spec, "score:selectivity")["candidate_tokens"].attrs["vocabulary"]
  assert str(vocabulary).split(",") == list(PROTONPOTTS_V6.symbols)
  # The centre is forced to its protonated token in every evaluated candidate.
  assert (arrays["candidate_tokens"][:, 1] == HIS_P).all()
  assert arrays["candidate_tokens"][1, 4] == HIS_P

  model = registered.load(spec)
  graph = next(registered.batches(spec)).arrays["prepared"][0].graph
  table, e_idx = protonpotts_table(
    model,
    jnp.asarray(graph.coords),
    jnp.asarray(graph.present),
    jnp.asarray(graph.residue_idx),
    jnp.asarray(graph.chain_index),
    jnp.asarray(graph.pad_valid),
  )
  dep = dict(DEFAULT_DEP_MAP)["HIS-P"]
  for c, seq in enumerate(arrays["candidate_tokens"]):
    row = np.asarray(candidate_energies_at(table, e_idx, jnp.asarray(seq), jnp.asarray([1])))[0]
    expected = float(row[HIS_P] - np.mean([row[token_index(d)] for d in dep]))
    assert arrays["selectivity_per_centre"][c, 0] == pytest.approx(expected, rel=1e-4, abs=1e-5)
    assert arrays["selectivity"][c] == pytest.approx(expected, rel=1e-4, abs=1e-5)


def test_selectivity_total_is_the_row_sum_of_the_centres(
  registered: ProtonPottsDriver, pdb: Path, model_path: Path
) -> None:
  del registered
  options = ProtonPottsOptions(binder_chain="A", explicit_centers=((2, "HIS-P"), (5, "HIS-P")))
  arrays = score(_selectivity_spec(pdb, model_path, options))["structures"]["0"]["arrays"]
  assert arrays["selectivity_per_centre"].shape == (1, 2)
  assert arrays["selectivity"] == pytest.approx(
    arrays["selectivity_per_centre"].sum(axis=1), rel=1e-5, abs=1e-5
  )


def test_selectivity_without_explicit_centres_raises(
  registered: ProtonPottsDriver, pdb: Path, model_path: Path
) -> None:
  del registered
  spec = ScoringSpecification(
    inputs=str(pdb),
    model_family="protonpottsmpnn",
    model_local_path=model_path,
    output_kind="selectivity",  # type: ignore[arg-type]
    protonpotts=ProtonPottsOptions(binder_chain="A"),
  )
  with pytest.raises(ValueError, match="explicit_centers"):
    score(spec)


# --- spec gating -------------------------------------------------------------------------------


def test_selectivity_is_gated_to_protonpottsmpnn() -> None:
  ScoringSpecification(inputs="x.pdb", model_family="protonpottsmpnn", output_kind="selectivity")  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="not supported for model_family"):
    ScoringSpecification(inputs="x.pdb", model_family="pottsmpnn", output_kind="selectivity")  # type: ignore[arg-type]
