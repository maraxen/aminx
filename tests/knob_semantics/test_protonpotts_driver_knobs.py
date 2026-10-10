# ruff: noqa: S101
"""ProtonPottsMPNN knob semantics at the driver level, driven through ``ProtonPottsOptions``.

Covers the input and structure knobs: protonation labels, variants, binder chain, design method, centre
selection, the selectivity contrast, forbidden tokens, infill scope, trajectory recording and the sampling seed.
Each test drives the public ``sample`` / ``score`` entry points with a real spec and checks that an observable
output moves when the knob moves and stays put when it does not.

The models are random-init ``PottsMPNN`` serialised to ``tmp_path`` (no checkpoint download), so the energies are
not upstream numbers. Only the knob's effect on the output is under test.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np
import pytest

from aminx.families.potts_mpnn.model import PottsMPNN
from aminx.families.protonpotts_mpnn import ProtonPottsDriver
from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6, token_index
from aminx.host.family_driver import FAMILY_DRIVERS
from aminx.host.runner import sample, score
from aminx.run.options import ProtonPottsOptions
from aminx.run.specs import SamplingSpecification, ScoringSpecification

RESIDUES = ["ALA", "HIS", "ASP", "GLU", "HIS", "ALA", "GLY", "LYS", "SER", "THR"]
HIS_P = token_index("HIS-P")
ASP_P = token_index("ASP-P")
AROMATIC = tuple(token_index(t) for t in ("F", "W", "Y"))


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
def toy_pdb(tmp_path: Path) -> Path:
  return _write_pdb(tmp_path / "toy.pdb", [("A", RESIDUES, 0.0)])


@pytest.fixture
def toy_two_chain_pdb(tmp_path: Path) -> Path:
  """Chain A has ONE residue (fewer free binder positions than two centre types, so no plan); chain B is ten."""
  return _write_pdb(tmp_path / "two_chain.pdb", [("A", ["HIS"], 0.0), ("B", RESIDUES, 50.0)])


@pytest.fixture
def registered() -> Iterator[ProtonPottsDriver]:
  driver = ProtonPottsDriver()
  FAMILY_DRIVERS.register("protonpottsmpnn")(driver)
  yield driver
  FAMILY_DRIVERS.discard("protonpottsmpnn")


@pytest.fixture
def toy_model_path(tmp_path: Path) -> Path:
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


def _score_spec(pdb: Path, model_path: Path, kind: str, options: ProtonPottsOptions) -> ScoringSpecification:
  return ScoringSpecification(
    inputs=str(pdb),
    model_family="protonpottsmpnn",
    model_local_path=model_path,
    output_kind=kind,  # type: ignore[arg-type]
    protonpotts=options,
  )


def _score_arrays(pdb: Path, model_path: Path, kind: str, options: ProtonPottsOptions) -> dict[str, Any]:
  return score(_score_spec(pdb, model_path, kind, options))["structures"]["0"]["arrays"]


def _native(pdb: Path, model_path: Path) -> np.ndarray:
  """The reference tokens with no labels and no pins: the sequence every design starts from."""
  return _score_arrays(pdb, model_path, "energy", ProtonPottsOptions())["candidate_tokens"][0]


def _contains(sequences: np.ndarray, tokens: tuple[int, ...]) -> bool:
  return bool(np.isin(sequences, list(tokens)).any())


# --- score:energy inputs -----------------------------------------------------------------------


def test_knob_semantics_protonation_labels_json(
  registered: ProtonPottsDriver, toy_pdb: Path, toy_model_path: Path, tmp_path: Path
) -> None:
  del registered
  bare = _score_arrays(toy_pdb, toy_model_path, "energy", ProtonPottsOptions())
  again = _score_arrays(toy_pdb, toy_model_path, "energy", ProtonPottsOptions())
  assert np.array_equal(bare["energy"], again["energy"])  # unlabelled, the reference is reproducible
  assert bare["candidate_tokens"][0, 1] == token_index("H")

  labels = _json(tmp_path, "labels.json", {"A:2": "HIS-P"})
  labelled = _score_arrays(toy_pdb, toy_model_path, "energy", ProtonPottsOptions(protonation_labels_json=labels))
  assert labelled["candidate_tokens"][0, 1] == HIS_P  # the label replaces the standard histidine
  assert int((labelled["candidate_tokens"][0] != bare["candidate_tokens"][0]).sum()) == 1
  assert abs(float(labelled["energy"][0]) - float(bare["energy"][0])) > 0.0  # the label reaches the energy

  unknown = _json(tmp_path, "unknown.json", {"A:99": "HIS-P"})
  with pytest.raises(ValueError, match="not in the structure"):
    score(_score_spec(toy_pdb, toy_model_path, "energy", ProtonPottsOptions(protonation_labels_json=unknown)))


def test_knob_semantics_variants_json(
  registered: ProtonPottsDriver, toy_pdb: Path, toy_model_path: Path, tmp_path: Path
) -> None:
  del registered
  variants = _json(
    tmp_path,
    "variants.json",
    {"his": {"A:2": "HIS-P"}, "two_pins": {"A:2": "HIS-P", "A:5": "HIS-P"}},
  )
  reference_only = _score_arrays(toy_pdb, toy_model_path, "energy", ProtonPottsOptions())
  scored = _score_arrays(toy_pdb, toy_model_path, "energy", ProtonPottsOptions(variants_json=variants))
  again = _score_arrays(toy_pdb, toy_model_path, "energy", ProtonPottsOptions(variants_json=variants))

  assert reference_only["energy"].shape == (1,)  # no variants: the reference alone
  assert scored["candidate_ids"].tolist() == [0, 1, 2]  # reference, then one row per variant
  assert scored["energy"].shape == (3,)
  assert np.array_equal(scored["energy"], again["energy"])
  assert scored["energy"][0] == pytest.approx(reference_only["energy"][0], rel=1e-5)  # adding variants keeps it
  assert abs(float(scored["energy"][1]) - float(scored["energy"][2])) > 0.0  # distinct pins, distinct energies

  with pytest.raises(ValueError, match="needs variants"):
    score(_score_spec(toy_pdb, toy_model_path, "ddg", ProtonPottsOptions()))
  ddg = _score_arrays(toy_pdb, toy_model_path, "ddg", ProtonPottsOptions(variants_json=variants))
  assert ddg["ddg"].shape == (2,)


# --- sample: binder, method, centres ----------------------------------------------------------


def test_knob_semantics_binder_chain(
  registered: ProtonPottsDriver, toy_two_chain_pdb: Path, toy_model_path: Path
) -> None:
  del registered
  native = _native(toy_two_chain_pdb, toy_model_path)
  on_a = sample(_sample_spec(str(toy_two_chain_pdb), toy_model_path, _design_options(binder_chain="A")))
  assert "0" not in on_a.get("structures", {})  # chain A has no designable binder position: no plan
  assert [entry["input_index"] for entry in on_a["skipped_inputs"]] == [0]

  on_b = sample(_sample_spec(str(toy_two_chain_pdb), toy_model_path, _design_options(binder_chain="B")))
  arrays = on_b["structures"]["0"]["arrays"]
  assert arrays["sequence"].shape == (2, native.shape[0])
  assert (arrays["center_positions"] >= 1).all()  # the centre lands on chain B, after chain A's residue
  assert (arrays["sequence"][:, 0] == native[0]).all()  # the non-binder chain is never redesigned

  with pytest.raises(ValueError, match=r"Chains present: \['A', 'B'\]"):
    sample(_sample_spec(str(toy_two_chain_pdb), toy_model_path, _design_options(binder_chain="Z")))


def test_knob_semantics_design_method(
  registered: ProtonPottsDriver, toy_pdb: Path, toy_model_path: Path
) -> None:
  del registered
  block = _designs(toy_pdb, toy_model_path, _design_options(design_method="block_descent"))
  greedy = _designs(
    toy_pdb,
    toy_model_path,
    _design_options(design_method="greedy_energy_block", center_types=(), infill_scope="chain"),
  )
  again = _designs(toy_pdb, toy_model_path, _design_options(design_method="block_descent"))

  assert (block["center_positions"] >= 0).all()  # block descent pins its HIS-P centre
  assert (greedy["center_positions"] == -1).all()  # centre-free greedy: no pins, the -1 sentinel
  assert (greedy["center_types"] == -1).all()
  assert greedy["sequence"].shape == block["sequence"].shape
  assert np.array_equal(block["sequence"], again["sequence"])
  assert not np.array_equal(greedy["center_positions"], block["center_positions"])

  # the host-loop methods pin a centre like block descent and design the same neighbourhood
  mcmc = _designs(toy_pdb, toy_model_path, _design_options(design_method="converged_mcmc", cv_patience=1, cv_max=2))
  assert (mcmc["center_positions"] >= 0).all()
  assert mcmc["sequence"].shape == block["sequence"].shape
  assert not np.array_equal(mcmc["sequence"], block["sequence"])  # a different optimiser reaches different designs

  with pytest.raises(ValueError, match="requires backend='mpnn'"):  # upstream's rule for autoregressive and mpnn_sample
    _designs(toy_pdb, toy_model_path, _design_options(design_method="autoregressive"))
  with pytest.raises(ValueError, match="2616"):  # random placement is not ported (debt #2616)
    _designs(toy_pdb, toy_model_path, _design_options(placement_by="random"))


def test_knob_semantics_center_types(
  registered: ProtonPottsDriver, toy_pdb: Path, toy_model_path: Path
) -> None:
  del registered
  his = _designs(toy_pdb, toy_model_path, _design_options(center_types=("HIS-P",)))
  again = _designs(toy_pdb, toy_model_path, _design_options(center_types=("HIS-P",)))
  asp = _designs(toy_pdb, toy_model_path, _design_options(center_types=("ASP-P",)))

  assert (his["center_types"] == HIS_P).all()  # the protonated centre type reported is the one requested
  assert (asp["center_types"] == ASP_P).all()
  assert np.array_equal(his["center_types"], again["center_types"])
  assert not np.array_equal(his["center_types"], asp["center_types"])
  for row, position in zip(asp["sequence"], asp["center_positions"][:, 0], strict=True):
    assert row[position] == ASP_P  # the pinned position carries the requested token in every design


def test_knob_semantics_explicit_centers(
  registered: ProtonPottsDriver, toy_pdb: Path, toy_model_path: Path
) -> None:
  del registered
  at_two = _designs(
    toy_pdb,
    toy_model_path,
    _design_options(center_types=(), explicit_centers=((2, "HIS-P"),)),
  )
  at_five = _designs(
    toy_pdb,
    toy_model_path,
    _design_options(center_types=(), explicit_centers=((5, "HIS-P"),)),
  )
  assert (at_two["center_positions"] == 1).all()  # residue 2 is the second kept residue, index 1
  assert (at_two["center_types"] == HIS_P).all()
  assert (at_five["center_positions"] == 4).all()  # residue 5 is index 4: the pin follows the named residue
  assert not np.array_equal(at_two["center_positions"], at_five["center_positions"])

  # The docstring requires center_types=(); the default non-empty center_types is refused alongside it.
  with pytest.raises(ValueError, match="set either explicit_centers or center_types"):
    _designs(toy_pdb, toy_model_path, _design_options(explicit_centers=((2, "HIS-P"),)))


# --- score:selectivity contrast and sampling knobs --------------------------------------------


def test_knob_semantics_dep_map(
  registered: ProtonPottsDriver, toy_pdb: Path, toy_model_path: Path
) -> None:
  del registered

  def selectivity(**kwargs: Any) -> float:  # noqa: ANN401
    options = ProtonPottsOptions(
      binder_chain="A",
      explicit_centers=((2, "HIS-P"),),
      center_types=(),
      **kwargs,
    )
    return float(_score_arrays(toy_pdb, toy_model_path, "selectivity", options)["selectivity"][0])

  default = selectivity()
  assert selectivity() == pytest.approx(default, rel=1e-6)  # same options, same contrast
  assert selectivity(dep_map=(("HIS-P", ("HIS-S",)),)) == pytest.approx(default, rel=1e-6)  # the family default
  swapped = selectivity(dep_map=(("HIS-P", ("ASP-D",)),))
  assert abs(swapped - default) > 1e-6  # a different deprotonated contrast token changes the gap


def test_knob_semantics_forbidden_tokens(
  registered: ProtonPottsDriver, toy_pdb: Path, toy_model_path: Path
) -> None:
  del registered
  seeds = range(8)
  # Temperature 1.0 so the free sampler actually reaches the aromatic tokens; the native sequence has none.
  free = [
    _designs(
      toy_pdb,
      toy_model_path,
      _design_options(temperature=1.0, forbidden_tokens=()),
      seed=s,
    )["sequence"]
    for s in seeds
  ]
  banned = [
    _designs(
      toy_pdb,
      toy_model_path,
      _design_options(temperature=1.0, forbidden_tokens=("F", "W", "Y")),
      seed=s,
    )["sequence"]
    for s in seeds
  ]
  assert any(_contains(seq, AROMATIC) for seq in free)  # without the restriction an aromatic is sampled
  assert not any(_contains(seq, AROMATIC) for seq in banned)  # with it, none ever appears


def test_knob_semantics_infill_scope(
  registered: ProtonPottsDriver, toy_pdb: Path, toy_model_path: Path
) -> None:
  del registered
  native = _native(toy_pdb, toy_model_path)

  def changed_positions(scope: str) -> set[int]:
    # The pinned centre differs from native in both arms, so it is in both sets; the comparison is on the rest.
    positions: set[int] = set()
    for seed in range(6):
      options = _design_options(
        temperature=1.0,
        neighbour_k=2,  # a small kNN, so "neighbourhood" is a strict subset of the chain
        max_mutations=0,
        infill_scope=scope,
      )
      for row in _designs(toy_pdb, toy_model_path, options, seed=seed)["sequence"]:
        positions.update(int(p) for p in np.flatnonzero(row != native))
    return positions

  neighbourhood = changed_positions("neighbourhood")
  chain = changed_positions("chain")
  assert len(chain) > len(neighbourhood)  # the whole chain is free, so more positions can move


def test_knob_semantics_record_trajectory(
  registered: ProtonPottsDriver, toy_pdb: Path, toy_model_path: Path
) -> None:
  # NOTE: weak discrimination -- record_trajectory is copied into PHDesignConfig and never read again. ph_descent.py
  # states "Trajectory recording (_record) is not ported; the result carries no trajectory", so no output can
  # change. This test pins that: both settings give identical arrays and the schema has no trajectory array.
  on = _designs(toy_pdb, toy_model_path, _design_options(record_trajectory=True))
  off = _designs(toy_pdb, toy_model_path, _design_options(record_trajectory=False))
  schema = set(registered.result_schema(None, "sample"))
  assert set(on) == set(off) == schema
  assert not any("traj" in name for name in on)
  for name in on:
    assert np.array_equal(on[name], off[name])


def test_knob_semantics_random_seed(
  registered: ProtonPottsDriver, toy_pdb: Path, toy_model_path: Path
) -> None:
  del registered
  options = _design_options(temperature=1.0)
  first = _designs(toy_pdb, toy_model_path, options, seed=7)
  again = _designs(toy_pdb, toy_model_path, options, seed=7)
  assert np.array_equal(first["sequence"], again["sequence"])  # the same seed repeats exactly
  assert np.array_equal(first["final_potts_energy"], again["final_potts_energy"])
  others = [_designs(toy_pdb, toy_model_path, options, seed=s)["sequence"] for s in range(1, 9)]
  assert any(not np.array_equal(first["sequence"], other) for other in others)  # different seeds differ

  cold = _design_options(temperature=0.0)
  a = _designs(toy_pdb, toy_model_path, cold, seed=1)
  b = _designs(toy_pdb, toy_model_path, cold, seed=2)
  assert np.array_equal(a["sequence"], b["sequence"])  # temperature 0 uses no randomness, whatever the seed
