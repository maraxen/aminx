# ruff: noqa: S101
"""PottsMPNNDriver: registration, energy, ddG, and binding partitions."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.driver import PottsMPNNDriver, absolute_energies
from aminx.families.potts_mpnn.model import PottsMPNN
from aminx.host.family_driver import FAMILY_DRIVERS
from aminx.host.runner import sample, score
from aminx.run.options import PottsMPNNOptions
from aminx.run.specs import SamplingSpecification, ScoringSpecification

_THREE = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


def _atom(
    serial: int,
    atom: str,
    resname: str,
    chain: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
) -> str:
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
                        serial,
                        atom,
                        _THREE[amino],
                        letter,
                        index,
                        x=float(index * 3 + offset),
                        y=float(ord(letter)),
                        z=float(offset),
                    ),
                )
                serial += 1
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


def _energy_spec(pdb: Path, weights: Path, sequences: list[str], **kwargs: object) -> ScoringSpecification:
    options = kwargs.pop("potts_mpnn", PottsMPNNOptions())
    return ScoringSpecification(
        inputs=str(pdb),
        model_family="pottsmpnn",
        checkpoint_id="pottsmpnn_vanilla_20",
        model_local_path=weights,
        output_kind="energy",
        sequences_to_score=sequences,
        potts_mpnn=options,  # type: ignore[arg-type]
    )


def test_registration_and_handles(registered: PottsMPNNDriver) -> None:
    import aminx.families.potts_mpnn as potts_mpnn

    driver = FAMILY_DRIVERS.get("pottsmpnn")
    assert driver is registered
    assert driver is not None
    assert driver.name == "pottsmpnn"
    assert potts_mpnn.PottsMPNNDriver is PottsMPNNDriver
    assert driver.handles(None, "score:energy")
    assert driver.handles(None, "score:ddg")
    assert not driver.handles(None, "sample")
    assert not driver.handles(None, "score:nll")
    assert not driver.handles(None, "score:logits")
    assert "score:nll" in driver.mpnn_fallback_purposes
    assert "sample" not in driver.mpnn_fallback_purposes


def test_sample_raises_unsupported_purpose(registered: PottsMPNNDriver) -> None:
    del registered
    spec = SamplingSpecification(
        inputs="a.pdb",
        model_family="pottsmpnn",
        checkpoint_id="pottsmpnn_vanilla_20",
        num_samples=1,
        return_logits=False,
    )
    with pytest.raises(ValueError, match="does not support sample"):
        sample(spec)


def test_synthetic_energy_matches_head(
    registered: PottsMPNNDriver,
    model_path: Path,
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "toy.pdb"
    _write_pdb(pdb, {"A": "AAA", "B": "CCC"})
    mutant = "AAACCD"
    spec = _energy_spec(pdb, model_path, [mutant])
    result = score(spec)
    arrays = result["structures"]["0"]["arrays"]
    assert arrays["energy"].shape == (2,)
    assert arrays["energy"].dtype == np.float32
    assert np.array_equal(arrays["candidate_ids"], np.asarray([0, 1], dtype=np.int32))

    model = registered.load(spec)
    prepared = next(registered.batches(spec)).arrays["prepared"][0]
    graph = prepared.graph
    manual = np.asarray(
        absolute_energies(
            model,
            jnp.asarray(graph.coords),
            jnp.asarray(graph.present),
            jnp.asarray(graph.residue_idx),
            jnp.asarray(graph.chain_index),
            jnp.asarray(graph.pad_valid),
            jnp.asarray(graph.sequences),
        ),
        dtype=np.float32,
    )
    assert arrays["energy"] == pytest.approx(manual)


def test_short_sequence_names_length_and_chain_order(
    registered: PottsMPNNDriver,
    model_path: Path,
    tmp_path: Path,
) -> None:
    del registered
    pdb = tmp_path / "toy.pdb"
    _write_pdb(pdb, {"A": "AAA", "B": "CCC"})
    spec = _energy_spec(pdb, model_path, ["A"])
    with pytest.raises(ValueError, match=r"expected 6.*\['A', 'B'\]"):
        score(spec)


def test_partition_ddg_is_bound_minus_unbound(
    registered: PottsMPNNDriver,
    model_path: Path,
    tmp_path: Path,
) -> None:
    del registered
    full = tmp_path / "complex.pdb"
    chain_a = tmp_path / "chainA.pdb"
    chain_b = tmp_path / "chainB.pdb"
    _write_pdb(full, {"A": "AAA", "B": "CCC"})
    _write_pdb(chain_a, {"A": "AAA"})
    _write_pdb(chain_b, {"B": "CCC"})
    binding = tmp_path / "binding.json"
    binding.write_text('{"complex": [["A"], ["B"]]}\n', encoding="utf-8")
    table = tmp_path / "mutants.csv"
    table.write_text(
        "pdb,chain,mut_type,ddG_expt\ncomplex,A:B,A2C:C2D,0.5\n",
        encoding="utf-8",
    )
    options = PottsMPNNOptions(
        binding_energy_json=str(binding),
        mutant_csv=str(table),
        mean_norm=False,
    )
    mutant = "AACCCD"
    ddg_spec = ScoringSpecification(
        inputs=str(full),
        model_family="pottsmpnn",
        checkpoint_id="pottsmpnn_vanilla_20",
        model_local_path=model_path,
        output_kind="ddg",
        potts_mpnn=options,
    )
    ddg = score(ddg_spec)["structures"]["0"]["arrays"]["ddg"]
    assert ddg.shape == (1,)

    def _pair(pdb: Path, sequence: str) -> np.ndarray:
        scored = score(_energy_spec(pdb, model_path, [sequence]))
        return np.asarray(scored["structures"]["0"]["arrays"]["energy"], dtype=np.float64)

    complex_energy = _pair(full, mutant)
    unbound_a = _pair(chain_a, "AAC")
    unbound_b = _pair(chain_b, "CCD")
    expected = (complex_energy[1] - complex_energy[0]) - (
        (unbound_a[1] - unbound_a[0]) + (unbound_b[1] - unbound_b[0])
    )
    assert float(ddg[0]) == pytest.approx(float(expected), rel=1e-5, abs=1e-4)


def test_registry_sha256_required(model_path: Path, tmp_path: Path, registered: PottsMPNNDriver) -> None:
    pdb = tmp_path / "toy.pdb"
    _write_pdb(pdb, {"A": "AA"})
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "model_family": "pottsmpnn",
                        "checkpoint_id": "pottsmpnn_vanilla_20",
                        "artifact_path": model_path.name,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    spec = ScoringSpecification(
        inputs=str(pdb),
        model_family="pottsmpnn",
        checkpoint_id="pottsmpnn_vanilla_20",
        checkpoint_registry_path=registry,
        output_kind="energy",
        sequences_to_score=["AA"],
    )
    with pytest.raises(ValueError, match="sha256"):
        registered.load(spec)
