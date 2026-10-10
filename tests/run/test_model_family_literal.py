"""model_family Literal, checkpoint-prefix derivation, portable JSON v2 refusal."""

from __future__ import annotations

from typing import Literal, get_args

import pytest

from aminx.run.run_spec_portable_json import run_spec_portable_to_dict
from aminx.run.specs import RunSpecification, SamplingSpecification


def _literal_members() -> set[str]:
    # ``get_type_hints(RunSpecification)`` evaluates every annotation, including
    # names that are not imported into the module (EncodingFusionFn). Read this
    # field's annotation on its own.
    raw = RunSpecification.__annotations__["model_family"]
    hint = eval(raw, {"Literal": Literal}) if isinstance(raw, str) else raw  # noqa: S307
    members: set[str] = set()
    for arg in get_args(hint):
        if arg is type(None):
            continue
        if get_args(arg):
            members.update(str(item) for item in get_args(arg))
        else:
            members.add(str(arg))
    return members


def test_model_family_literal_includes_driver_families() -> None:
    assert _literal_members() == {
        "proteinmpnn",
        "ligandmpnn",
        "pottsmpnn",
        "protonpottsmpnn",
        "lasermpnn",
    }


def test_prefix_derivation_precedes_model_type() -> None:
    potts = SamplingSpecification(inputs="x.pdb", checkpoint_id="pottsmpnn_vanilla_20")
    laser = SamplingSpecification(
        inputs="x.pdb",
        checkpoint_id="lasermpnn_0p1A_nothing_heldout",
    )
    proton = SamplingSpecification(inputs="x.pdb", checkpoint_id="protonpottsmpnn_v6_30")
    assert potts.model_family == "pottsmpnn"
    assert proton.model_family == "protonpottsmpnn"
    assert laser.model_family == "lasermpnn"


def test_explicit_family_is_never_overridden() -> None:
    explicit_protein = SamplingSpecification(
        inputs="x.pdb",
        model_family="proteinmpnn",
        checkpoint_id="pottsmpnn_vanilla_20",
    )
    explicit_ligand = SamplingSpecification(
        inputs="x.pdb",
        model_family="ligandmpnn",
        checkpoint_id="lasermpnn_0p1A_nothing_heldout",
    )
    explicit_potts = SamplingSpecification(
        inputs="x.pdb",
        model_family="pottsmpnn",
        checkpoint_id="ligandmpnn_v_32_020",
    )
    assert explicit_protein.model_family == "proteinmpnn"
    assert explicit_ligand.model_family == "ligandmpnn"
    assert explicit_potts.model_family == "pottsmpnn"


def test_ligand_and_protein_derivation_unchanged() -> None:
    ligand = SamplingSpecification(inputs="x.pdb", checkpoint_id="ligandmpnn_v_32_020")
    protein = SamplingSpecification(inputs="x.pdb", checkpoint_id="proteinmpnn_v_48_020")
    assert ligand.model_family == "ligandmpnn"
    assert protein.model_family == "proteinmpnn"


def test_portable_json_v2_refuses_driver_families() -> None:
    protein = SamplingSpecification(inputs="x.pdb", model_family="proteinmpnn")
    payload = run_spec_portable_to_dict(protein.run_spec)
    assert payload["version"] == 2

    for family, checkpoint in (
        ("pottsmpnn", "pottsmpnn_vanilla_20"),
        ("protonpottsmpnn", "protonpottsmpnn_v6_30"),
        ("lasermpnn", "lasermpnn_0p1A_nothing_heldout"),
        ("ligandmpnn", "ligandmpnn_v_32_020"),
    ):
        spec = SamplingSpecification(
            inputs="x.pdb",
            model_family=family,  # type: ignore[arg-type]
            checkpoint_id=checkpoint,
        )
        with pytest.raises(ValueError, match=family):
            run_spec_portable_to_dict(spec.run_spec)
