"""ProtonPottsDriver: registration, refusals, input handling, and a synthetic end-to-end score (spec §43).

Whether the energies match upstream is settled by the ``protonpotts_energy`` wave. These tests check that the
driver feeds that already-graded computation correctly: the right rows, the right tokens, and the refusals.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.model import PottsMPNN
from aminx.families.protonpotts_mpnn import ProtonPottsDriver
from aminx.families.protonpotts_mpnn.energy import protonpotts_energies
from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6, token_index
from aminx.host.family_driver import FAMILY_DRIVERS
from aminx.host.runner import sample, score
from aminx.run.options import ProtonPottsOptions
from aminx.run.specs import SamplingSpecification, ScoringSpecification

RESIDUES = ["ALA", "HIS", "ASP", "GLU", "HIS", "ALA", "GLY", "LYS", "SER", "THR"]


def _atom(serial: int, name: str, resname: str, number: int, xyz: tuple[float, float, float]) -> str:
  padded = f" {name:<3}"
  return (
    f"ATOM  {serial:>5} {padded} {resname:>3} A{number:>4}    "
    f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}{1.0:>6.2f}{20.0:>6.2f}          {name[0]:>2}"
  )


@pytest.fixture
def pdb(tmp_path: Path) -> Path:
  rng = np.random.default_rng(0)
  lines: list[str] = []
  for i, resname in enumerate(RESIDUES, start=1):
    centre = np.array([3.8 * i, 2.0 * np.sin(i), 2.0 * np.cos(i)])
    for k, atom in enumerate(("N", "CA", "C", "O")):
      xyz = centre + rng.normal(scale=0.4, size=3) + np.array([0.6 * k, 0.0, 0.0])
      lines.append(_atom(len(lines) + 1, atom, resname, i, (xyz[0], xyz[1], xyz[2])))
  path = tmp_path / "toy.pdb"
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return path


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


def _spec(pdb: Path, weights: Path, kind: str, **options: object) -> ScoringSpecification:
  return ScoringSpecification(
    inputs=str(pdb),
    model_family="protonpottsmpnn",
    model_local_path=weights,
    output_kind=kind,  # type: ignore[arg-type]
    protonpotts=ProtonPottsOptions(**options),  # type: ignore[arg-type]
  )


def test_registration_handles_and_refusals(registered: ProtonPottsDriver, model_path: Path) -> None:
  driver = FAMILY_DRIVERS.get("protonpottsmpnn")
  assert driver is registered
  assert driver.handles(None, "score:energy")
  assert driver.handles(None, "score:ddg")
  assert driver.handles(None, "sample")
  assert driver.handles(None, "score:selectivity")
  assert not driver.handles(None, "score:nll")
  model = registered.load(_spec(Path("x"), model_path, "energy"))
  assert model.alphabet == PROTONPOTTS_V6
  with pytest.raises(ValueError, match="21-token"):
    registered.mpnn_core(model)


def test_spec_gating() -> None:
  with pytest.raises(ValueError, match="not supported for model_family"):
    ScoringSpecification(inputs="x.pdb", model_family="protonpottsmpnn", output_kind="nll")
  with pytest.raises(ValueError, match="protonpotts options require"):
    spec = ScoringSpecification(
      inputs="x.pdb",
      model_family="pottsmpnn",
      output_kind="energy",
      protonpotts=ProtonPottsOptions(),
    )
    _ = spec.run_spec


def test_sample_needs_one_call_per_structure(
  registered: ProtonPottsDriver, pdb: Path, model_path: Path
) -> None:
  del registered
  spec = SamplingSpecification(
    inputs=str(pdb),
    model_family="protonpottsmpnn",
    model_local_path=model_path,
    num_samples=2,
    protonpotts=ProtonPottsOptions(binder_chain="A"),
  )
  with pytest.raises(ValueError, match="num_samples=1"):
    sample(spec)


def test_energy_rows_tokens_and_value(
  registered: ProtonPottsDriver, pdb: Path, model_path: Path, tmp_path: Path
) -> None:
  variants = _json(
    tmp_path,
    "variants.json",
    {
      "his_protonated": {"A:2": "HIS-P"},
      "two_pins": {"A:2": "HIS-P", "A:5": "HIS-P"},
      "full": ["A", "H", "D", "E", "H", "A", "G", "K", "S", "T"],
    },
  )
  spec = _spec(pdb, model_path, "energy", variants_json=variants)
  arrays = score(spec)["structures"]["0"]["arrays"]

  assert arrays["energy"].shape == (4,)
  assert arrays["candidate_ids"].tolist() == [0, 1, 2, 3]
  tokens = arrays["candidate_tokens"]
  assert tokens.shape == (4, len(RESIDUES))
  hp = token_index("HIS-P")
  assert tokens[0, 1] == token_index("H")  # reference: no labels, a standard histidine
  assert tokens[1, 1] == hp
  assert tokens[2, 1] == hp
  assert tokens[2, 4] == hp
  assert (tokens[1] != tokens[0]).sum() == 1
  assert np.array_equal(tokens[3], tokens[0])  # the full list spells the reference
  assert arrays["energy"][3] == pytest.approx(arrays["energy"][0], rel=1e-5)

  model = registered.load(spec)
  graph = next(registered.batches(spec)).arrays["prepared"][0].graph
  manual = protonpotts_energies(
    model,
    jnp.asarray(graph.coords),
    jnp.asarray(graph.present),
    jnp.asarray(graph.residue_idx),
    jnp.asarray(graph.chain_index),
    jnp.asarray(graph.pad_valid),
    jnp.asarray(graph.sequences),
  )
  assert arrays["energy"] == pytest.approx(np.asarray(manual), rel=1e-5)
  assert np.isfinite(arrays["energy"]).all()
  assert float(np.abs(arrays["energy"][1] - arrays["energy"][0])) > 0.0  # the pin reaches the energy


def test_ddg_is_variant_minus_reference(pdb: Path, model_path: Path, registered: ProtonPottsDriver, tmp_path: Path) -> None:
  del registered
  variants = _json(tmp_path, "v.json", {"a": {"A:2": "HIS-P"}, "b": {"A:3": "ASP-P"}})
  energy = score(_spec(pdb, model_path, "energy", variants_json=variants))["structures"]["0"]["arrays"]
  ddg = score(_spec(pdb, model_path, "ddg", variants_json=variants))["structures"]["0"]["arrays"]
  assert ddg["ddg"].shape == (2,)
  assert ddg["ddg"] == pytest.approx(energy["energy"][1:] - energy["energy"][0], rel=1e-4, abs=1e-5)
  assert ddg["mutant_tokens"].shape == (2, len(RESIDUES))
  assert np.isnan(ddg["ddg_expt"]).all()


def test_variant_names_are_carried_with_the_ids(
  registered: ProtonPottsDriver, pdb: Path, model_path: Path, tmp_path: Path
) -> None:
  # id k is variant_names[k]: energy and selectivity rows start with the reference, ddg rows do not (debt #2618, item 5).
  variants = _json(tmp_path, "v.json", {"a": {"A:2": "HIS-P"}, "b": {"A:3": "ASP-P"}})
  energy_spec = _spec(pdb, model_path, "energy", variants_json=variants)
  ddg_spec = _spec(pdb, model_path, "ddg", variants_json=variants)
  energy_names = json.loads(registered.result_schema(energy_spec, "score:energy")["candidate_ids"].attrs["variant_names"])
  ddg_names = json.loads(registered.result_schema(ddg_spec, "score:ddg")["mutant_ids"].attrs["variant_names"])
  assert energy_names == ["reference", "a", "b"]
  assert ddg_names == ["a", "b"]
  arrays = score(energy_spec)["structures"]["0"]["arrays"]
  assert len(energy_names) == arrays["candidate_ids"].shape[0]  # one name per scored row
  assert len(ddg_names) == score(ddg_spec)["structures"]["0"]["arrays"]["mutant_ids"].shape[0]
  # no spec (or no variants) leaves the ids unannotated or reference-only
  assert registered.result_schema(None, "score:energy")["candidate_ids"].attrs == {}
  bare = registered.result_schema(_spec(pdb, model_path, "energy"), "score:energy")["candidate_ids"].attrs
  assert json.loads(bare["variant_names"]) == ["reference"]


def test_labels_set_the_reference(registered: ProtonPottsDriver, pdb: Path, model_path: Path, tmp_path: Path) -> None:
  del registered
  labels = _json(tmp_path, "labels.json", {"A:2": "HIS-P", "A:3": "ASP-D"})
  arrays = score(_spec(pdb, model_path, "energy", protonation_labels_json=labels))["structures"]["0"]["arrays"]
  assert arrays["candidate_tokens"].shape == (1, len(RESIDUES))
  assert arrays["candidate_tokens"][0, 1] == token_index("HIS-P")
  assert arrays["candidate_tokens"][0, 2] == token_index("ASP-D")


@pytest.mark.parametrize(
  ("kind", "options", "message"),
  [
    ("ddg", {}, "needs variants"),
    ("energy", {"variants_json": {"v": {"A:99": "HIS"}}}, "not in the structure"),
    ("energy", {"variants_json": {"v": {"A:2": "HID"}}}, "v4 extension tokens"),
    ("energy", {"variants_json": {"v": {"A:3": "HIS-P"}}}, "cannot apply"),
    ("energy", {"variants_json": {"v": ["A", "H"]}}, "tokens for"),
    ("energy", {"variants_json": {"v": {"two": "A"}}}, "not '<chain>:<number>"),
    ("energy", {"protonation_labels_json": {"A:2": "ASP-P"}}, "belongs to"),
  ],
)
def test_bad_inputs_raise(
  registered: ProtonPottsDriver, pdb: Path, model_path: Path, tmp_path: Path, kind: str, options: dict, message: str
) -> None:
  del registered
  resolved = {k: _json(tmp_path, f"{k}.json", v) for k, v in options.items()}
  with pytest.raises(ValueError, match=message):
    score(_spec(pdb, model_path, kind, **resolved))


def test_schema_stamps_the_vocabulary(registered: ProtonPottsDriver) -> None:
  schema = registered.result_schema(None, "score:energy")
  vocabulary = schema["candidate_tokens"].attrs["vocabulary"]
  assert str(vocabulary).split(",") == list(PROTONPOTTS_V6.symbols)
  assert "HIS-P" in str(vocabulary).split(",")
  assert "sequence" in registered.result_schema(None, "sample")
  with pytest.raises(ValueError, match="does not support"):
    registered.result_schema(None, "score:nll")
