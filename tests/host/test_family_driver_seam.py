"""FamilyDriver seam: dispatch, Zarr channel, fallback core, unsupported purpose."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import zarr

from aminx.host.family_driver import FAMILY_DRIVERS, FamilyBatch, SinkArraySpec
from aminx.host.prep import prep_protein_stream_and_model
from aminx.host.runner import jacobian, sample, score
from aminx.run.specs import JacobianSpecification, SamplingSpecification, ScoringSpecification


class _Core(eqx.Module):
    tag: str = eqx.field(static=True)
    weight: jax.Array


class _Wrapped(eqx.Module):
    mpnn: _Core
    flag: bool = eqx.field(static=True)


class _FakeSampleDriver:
    """Test-only driver. Registered under pottsmpnn for the duration of a test."""

    name = "pottsmpnn"
    options_type = type(None)
    mpnn_fallback_purposes: frozenset[str] = frozenset()

    def __init__(self, lengths: Mapping[str, int]) -> None:
        self._lengths = dict(lengths)
        self.loaded = False

    def handles(self, spec: Any, purpose: str) -> bool:
        del spec
        return purpose == "sample"

    def load(self, spec: Any) -> eqx.Module:
        del spec
        self.loaded = True
        return _Core(tag="family", weight=jnp.zeros((1,)))

    def mpnn_core(self, model: eqx.Module) -> eqx.Module | None:
        del model
        return None

    def batches(self, spec: Any) -> Iterator[FamilyBatch]:
        raw = spec.inputs
        items = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        indices: list[int] = []
        lengths: list[int] = []
        skipped: list[tuple[int, str]] = []
        for index, item in enumerate(items):
            key = str(item)
            if key in self._lengths:
                indices.append(index)
                lengths.append(self._lengths[key])
            else:
                skipped.append((index, "unparsable"))
        yield FamilyBatch(
            input_indices=tuple(indices),
            arrays={},
            skipped=tuple(skipped),
            lengths=tuple(lengths),
        )

    def axes(self, spec: Any, purpose: str, batch: FamilyBatch) -> list[Any]:
        del spec, purpose, batch
        return []

    def stages(self, spec: Any, purpose: str, model: eqx.Module) -> Any:
        del spec, purpose, model

        def _run(
            batch: FamilyBatch,
            *,
            chunk_start: int,
            chunk_count: int,
            scalar_dropout: bool,
            dropout_key: jax.Array,
        ) -> dict[str, jax.Array]:
            del scalar_dropout, dropout_key
            n = len(batch.input_indices)
            pad = max(batch.lengths) if batch.lengths else 0
            seq = np.full((n, chunk_count, pad), -1, dtype=np.int32)
            for local, length in enumerate(batch.lengths):
                for sample_offset in range(chunk_count):
                    sample_index = chunk_start + sample_offset
                    for residue in range(length):
                        seq[local, sample_offset, residue] = length * 100 + sample_index * 10 + residue
            return {"sequence": jnp.asarray(seq)}

        return _run

    def result_schema(self, spec: Any, purpose: str) -> Mapping[str, SinkArraySpec]:
        del spec, purpose
        return {"sequence": SinkArraySpec(dims=("N", "L_total"), dtype="int32", attrs={})}


class _FallbackDriver:
    name = "pottsmpnn"
    options_type = type(None)
    mpnn_fallback_purposes = frozenset({"score:nll", "jacobian", "inspect", "score:logits"})

    def __init__(self, wrapped: _Wrapped) -> None:
        self._wrapped = wrapped
        self.load_calls = 0

    def handles(self, spec: Any, purpose: str) -> bool:
        del spec, purpose
        return False

    def load(self, spec: Any) -> _Wrapped:
        del spec
        self.load_calls += 1
        return self._wrapped

    def mpnn_core(self, model: _Wrapped) -> _Core:
        return model.mpnn

    def batches(self, spec: Any) -> Iterator[FamilyBatch]:
        del spec
        return iter(())

    def axes(self, spec: Any, purpose: str, batch: FamilyBatch) -> list[Any]:
        del spec, purpose, batch
        return []

    def stages(self, spec: Any, purpose: str, model: eqx.Module) -> Any:
        del spec, purpose, model
        raise AssertionError("fallback purposes must not build stages")

    def result_schema(self, spec: Any, purpose: str) -> Mapping[str, SinkArraySpec]:
        del spec, purpose
        return {}


def _register(driver: object) -> None:
    FAMILY_DRIVERS.register("pottsmpnn")(driver)  # type: ignore[arg-type]


def _unregister() -> None:
    FAMILY_DRIVERS.discard("pottsmpnn")


def _sample_spec(inputs: list[str], output: str) -> SamplingSpecification:
    return SamplingSpecification(
        inputs=inputs,
        model_family="pottsmpnn",
        checkpoint_id="pottsmpnn_vanilla_20",
        num_samples=2,
        samples_chunk_size=1,
        output_h5_path=output,
        return_logits=False,
    )


def _expected(length: int, sample_index: int) -> np.ndarray:
    return np.asarray(
        [[length * 100 + sample_index * 10 + residue for residue in range(length)]],
        dtype=np.int32,
    )


def test_sample_dispatch_zarr_ragged_duplicate_stems_and_skips(tmp_path) -> None:
    driver = _FakeSampleDriver({"a/model.pdb": 4, "b/model.pdb": 7})
    _register(driver)
    try:
        joint = tmp_path / "joint.zarr"
        solo = tmp_path / "solo.zarr"
        sample(_sample_spec(
            ["a/model.pdb", "not-a-pdb.txt", "b/model.pdb"],
            str(joint),
        ))
        sample(_sample_spec(["a/model.pdb"], str(solo)))
        assert driver.loaded

        root = zarr.open_group(str(joint), mode="r")
        assert root.attrs["schema_version"] == "pottsmpnn_v1"
        assert root.attrs["model_family"] == "pottsmpnn"
        assert root.attrs["purpose"] == "sample"
        assert root.attrs["output_kind"] == ""
        assert root.attrs["alphabet"] == "ACDEFGHIKLMNPQRSTVWYX"
        skipped = list(root.attrs["skipped_inputs"])
        assert skipped == [{
            "input_index": 1,
            "structure_id": "not-a-pdb",
            "reason": "unparsable",
        }]
        assert set(root.group_keys()) >= {"structure_0", "structure_2"}
        assert "structure_1" not in set(root.group_keys())

        for group_name, length, index in (("structure_0", 4, 0), ("structure_2", 7, 2)):
            group = root[group_name]
            assert group.attrs["structure_id"] == "model"
            assert int(group.attrs["structure_index"]) == index
            for chunk, sample_index in (("0", 0), ("1", 1)):
                seq = np.asarray(group[chunk]["sequence"])
                assert seq.shape == (1, length)
                assert np.array_equal(seq, _expected(length, sample_index))

        solo_root = zarr.open_group(str(solo), mode="r")
        solo_seq = np.asarray(solo_root["structure_0"]["0"]["sequence"])
        joint_seq = np.asarray(root["structure_0"]["0"]["sequence"])
        assert np.array_equal(solo_seq, joint_seq)
    finally:
        _unregister()


def test_in_memory_chunks_concatenate_within_structure(tmp_path) -> None:
    del tmp_path
    driver = _FakeSampleDriver({"a/model.pdb": 4})
    _register(driver)
    try:
        result = sample(SamplingSpecification(
            inputs=["a/model.pdb"],
            model_family="pottsmpnn",
            checkpoint_id="pottsmpnn_vanilla_20",
            num_samples=2,
            samples_chunk_size=1,
            return_logits=False,
        ))
        seq = result["structures"]["0"]["arrays"]["sequence"]
        assert seq.shape == (2, 4)
        assert np.array_equal(seq[0], _expected(4, 0)[0])
        assert np.array_equal(seq[1], _expected(4, 1)[0])
        assert result["skipped_inputs"] == []
    finally:
        _unregister()


def test_unsupported_purpose_raises() -> None:
    _register(_FakeSampleDriver({}))
    try:
        spec = JacobianSpecification(
            inputs="a/model.pdb",
            model_family="pottsmpnn",
            checkpoint_id="pottsmpnn_vanilla_20",
        )
        with pytest.raises(ValueError, match="does not support jacobian"):
            jacobian(spec)
    finally:
        _unregister()


def test_fallback_purpose_loads_mpnn_core(monkeypatch: pytest.MonkeyPatch) -> None:
    core = _Core(tag="mpnn-core", weight=jnp.zeros((2,)))
    wrapped = _Wrapped(mpnn=core, flag=True)
    driver = _FallbackDriver(wrapped)
    _register(driver)

    def _no_dataset(*args: object, **kwargs: object) -> Iterator[object]:
        del args, kwargs
        return iter(())

    def _stop(model: eqx.Module) -> eqx.Module:
        raise RuntimeError(f"score-fn:{model.tag}")

    monkeypatch.setattr("aminx.host.prep.create_protein_dataset", _no_dataset)
    monkeypatch.setattr("aminx.host.prep.load_model", _stop)
    # ``aminx.scoring.score`` on the package is the function, not the module.
    monkeypatch.setattr(sys.modules["aminx.scoring.score"], "make_score_fn", _stop)
    try:
        spec = ScoringSpecification(
            inputs="x.pdb",
            sequences_to_score=["AC"],
            model_family="pottsmpnn",
            checkpoint_id="pottsmpnn_vanilla_20",
            return_logits=False,
        )
        with pytest.raises(RuntimeError, match="score-fn:mpnn-core"):
            score(spec)
        assert driver.load_calls == 1
        _iterator, model = prep_protein_stream_and_model(spec)
        del _iterator
        assert model.tag == "mpnn-core"
    finally:
        _unregister()


def _purge_family(key: str, module: str) -> None:
    """Drop a registered driver and its package so the lazy import must run again."""
    FAMILY_DRIVERS.discard(key)
    for name in [n for n in sys.modules if n == module or n.startswith(f"{module}.")]:
        del sys.modules[name]


@pytest.mark.parametrize(
    ("family", "module"),
    [
        ("pottsmpnn", "aminx.families.potts_mpnn"),
        ("lasermpnn", "aminx.families.laser_mpnn"),
    ],
)
def test_family_driver_for_lazy_imports_every_driver_backed_family(
    family: str, module: str
) -> None:
    """Dispatch must resolve a driver WITHOUT the caller importing the family first.

    The observable that would have caught debt #2403: with the family package absent
    from sys.modules, _family_driver_for must still return a driver. Only pottsmpnn
    was imported here, so lasermpnn resolved to None -- and because every call site
    reads ``if (d := _family_driver_for(spec)) is not None:``, that None skipped the
    block's own guards and fell through to the stock ProteinMPNN path, returning
    ProteinMPNN numbers for a LASEr request with no error.

    This test must NOT import the family module itself. A test that imports the
    module under test cannot catch a missing import in the caller, which is exactly
    why the existing seam tests were blind to this.
    """
    from aminx.host.runner import _family_driver_for  # noqa: PLC0415

    _purge_family(family, module)
    assert FAMILY_DRIVERS.get(family) is None, "precondition: driver must start unregistered"

    driver = _family_driver_for(SimpleNamespace(model_family=family))

    assert driver is not None, f"{family} did not resolve a driver; dispatch would fall through"
    assert driver.name == family


def test_family_driver_for_raises_rather_than_falling_back(monkeypatch: Any) -> None:
    """A driver-backed family with no driver must RAISE, never return None.

    Returning None is indistinguishable at the call site from "this spec is a stock
    MPNN spec", so the request is served by the wrong model instead of failing.
    """
    from aminx.host import runner as _runner  # noqa: PLC0415

    monkeypatch.setitem(
        _runner._DRIVER_BACKED_FAMILIES,  # noqa: SLF001
        "lasermpnn",
        "aminx.host.family_driver",  # imports fine, registers no lasermpnn driver
    )
    _purge_family("lasermpnn", "aminx.families.laser_mpnn")

    with pytest.raises(RuntimeError, match="requires a registered FamilyDriver"):
        _runner._family_driver_for(SimpleNamespace(model_family="lasermpnn"))  # noqa: SLF001


def test_stock_families_still_return_none_without_raising() -> None:
    """proteinmpnn/ligandmpnn have no driver BY DESIGN and must keep falling through."""
    from aminx.host.runner import _family_driver_for  # noqa: PLC0415

    for family in ("proteinmpnn", "ligandmpnn", None):
        assert _family_driver_for(SimpleNamespace(model_family=family)) is None
