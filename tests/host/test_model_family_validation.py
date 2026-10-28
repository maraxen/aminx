"""model_family validation (#2484) and the laser optional-extra gate (#2483).

Unknown family names must fail loudly at spec construction, runner resolution, prep and
CLI parsing, and every check reads one table in ``aminx.host.family_driver``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Literal, get_args

import pytest
import typer
from typer.testing import CliRunner

from aminx.cli import _validate_model_family_option, app
from aminx.host import runner as _runner
from aminx.host.family_driver import (
    DRIVER_BACKED_FAMILY_MODULES,
    MODEL_FAMILIES,
    STOCK_MODEL_FAMILIES,
    ModelFamily,
    validate_model_family,
)
from aminx.model.versions import MODEL_WEIGHTS
from aminx.run.specs import RunSpecification, SamplingSpecification

_UNKNOWN = "foompnn"


def _literal_members() -> set[str]:
    raw = RunSpecification.__annotations__["model_family"]
    hint = eval(raw, {"Literal": Literal, "ModelFamily": ModelFamily}) if isinstance(raw, str) else raw  # noqa: S307
    members: set[str] = set()
    for arg in get_args(hint):
        if arg is type(None):
            continue
        if get_args(arg):
            members.update(str(item) for item in get_args(arg))
        else:
            members.add(str(arg))
    return members


def test_single_source_of_truth() -> None:
    assert MODEL_FAMILIES == (*STOCK_MODEL_FAMILIES, *DRIVER_BACKED_FAMILY_MODULES)
    assert _runner._DRIVER_BACKED_FAMILIES is DRIVER_BACKED_FAMILY_MODULES  # noqa: SLF001
    assert _literal_members() == set(MODEL_FAMILIES)


def test_default_family_and_valid_names_resolve() -> None:
    for name in MODEL_FAMILIES:
        assert validate_model_family(name) == name
    spec = SamplingSpecification(inputs="x.pdb", model_family="proteinmpnn")
    assert spec.model_family == "proteinmpnn"
    # A stock family is not driver-backed: resolution returns None and does not raise.
    assert _runner._family_driver_for(SimpleNamespace(model_family="proteinmpnn")) is None  # noqa: SLF001


@pytest.mark.parametrize("typo", ["foompnn", "lasermpn"])
def test_pure_typos_rejected(typo: str) -> None:
    with pytest.raises(ValueError, match=typo):
        validate_model_family(typo)


def test_temperature_and_weight_family_names_validate() -> None:
    # Families named by the temperature-default rule (spec.py _as_temperature_tuple) and
    # the MODEL_WEIGHTS registry's family-shaped name must all pass the guard.
    temperature_families = {"proteinmpnn", "ligandmpnn", "membrane", "pottsmpnn", "lasermpnn"}
    weight_families = set(get_args(MODEL_WEIGHTS)) & set(MODEL_FAMILIES)
    assert weight_families == {"membrane"}
    for name in temperature_families | weight_families:
        assert validate_model_family(name) == name
    # Weight-variant names are not families and stay rejected.
    for name in ("soluble", "sc", "original", "ligand"):
        with pytest.raises(ValueError):
            validate_model_family(name)


def test_unknown_family_rejected_by_spec_listing_valid_names() -> None:
    with pytest.raises(ValueError, match=_UNKNOWN) as excinfo:
        SamplingSpecification(inputs="x.pdb", model_family=_UNKNOWN)
    message = str(excinfo.value)
    for name in MODEL_FAMILIES:
        assert name in message


def test_unknown_family_rejected_by_runner_resolution() -> None:
    with pytest.raises(ValueError, match=_UNKNOWN):
        _runner._family_driver_for(SimpleNamespace(model_family=_UNKNOWN))  # noqa: SLF001


def test_unknown_family_rejected_by_cli_option_parsing() -> None:
    with pytest.raises(typer.BadParameter) as excinfo:
        _validate_model_family_option(_UNKNOWN)
    assert _UNKNOWN in str(excinfo.value)
    assert "lasermpnn" in str(excinfo.value)
    assert _validate_model_family_option(None) is None
    assert _validate_model_family_option("pottsmpnn") == "pottsmpnn"


def test_unknown_family_rejected_by_cli_invocation() -> None:
    result = CliRunner().invoke(
        app, ["run", "--model-family", _UNKNOWN, "sample", "--inputs", "x.pdb"]
    )
    assert result.exit_code != 0
    assert _UNKNOWN in result.output
    assert "lasermpnn" in result.output


def test_missing_prody_names_the_laser_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes ``import prody`` raise ModuleNotFoundError.
    monkeypatch.setitem(sys.modules, "prody", None)
    with pytest.raises(ImportError) as excinfo:
        _runner._require_family_extra("lasermpnn")  # noqa: SLF001
    message = str(excinfo.value)
    assert "--extra laser" in message
    assert "aminx[laser]" in message
    assert "prody" in message


def test_thermompnn_is_refused_until_it_has_a_driver() -> None:
    """ThermoMPNN has model and convert code but no FamilyDriver. Requesting it must fail at spec
    validation with a message that says why, not fall through to a stock or silent path."""
    with pytest.raises(ValueError, match="thermompnn has no driver yet"):
        validate_model_family("thermompnn")
    with pytest.raises(ValueError, match="thermompnn has no driver yet"):
        SamplingSpecification(inputs="x.pdb", model_family="thermompnn")
    assert "thermompnn" not in MODEL_FAMILIES


def test_missing_prody_fails_at_family_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "prody", None)
    with pytest.raises(ImportError, match="--extra laser"):
        _runner._family_driver_for(SimpleNamespace(model_family="lasermpnn"))  # noqa: SLF001


def test_prody_importable_gives_no_error() -> None:
    pytest.importorskip("prody")
    assert _runner._require_family_extra("lasermpnn") is None  # noqa: SLF001
    assert _runner._require_family_extra("proteinmpnn") is None  # noqa: SLF001
