"""The shared torch checkpoint loader refuses globals outside its allow-list.

A checkpoint pickle that names a callable through ``__reduce__`` must not run it. The loader is
shared by the Caliby and FAMPNN drivers.
"""

from __future__ import annotations

import collections
import io
import pickle
from pathlib import Path
from typing import Any

import pytest

from aminx.io.torch_checkpoint import RestrictedTorchUnpickler, load_restricted_torch_checkpoint

_SENTINEL_CALLS: list[str] = []


def _record_sentinel(tag: str) -> str:
    """Harmless callable that a hostile ``__reduce__`` would run; records each call."""
    _SENTINEL_CALLS.append(tag)
    return tag


class _SentinelPayload:
    """Pickles as a call to ``_record_sentinel``."""

    def __reduce__(self) -> tuple[Any, ...]:
        return (_record_sentinel, ("fired",))


@pytest.fixture(autouse=True)
def _clear_sentinel() -> None:
    _SENTINEL_CALLS.clear()


def test_unpickler_refuses_a_reduce_payload_without_calling_it() -> None:
    with pytest.raises(pickle.UnpicklingError, match="Refusing to unpickle"):
        RestrictedTorchUnpickler(io.BytesIO(pickle.dumps(_SentinelPayload()))).load()

    assert _SENTINEL_CALLS == []


def test_unpickler_still_resolves_plain_containers() -> None:
    payload = pickle.dumps(collections.OrderedDict(weight=1))

    assert RestrictedTorchUnpickler(io.BytesIO(payload)).load() == collections.OrderedDict(weight=1)


def test_checkpoint_loader_refuses_a_reduce_payload_inside_a_torch_file(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "hostile.pt"
    torch.save({"state_dict": _SentinelPayload()}, path)

    with pytest.raises(pickle.UnpicklingError, match="Refusing to unpickle"):
        load_restricted_torch_checkpoint(path)

    assert _SENTINEL_CALLS == []


def test_checkpoint_loader_reads_a_plain_state_dict(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "plain.pt"
    torch.save({"state_dict": {"w": torch.zeros(2)}}, path)

    loaded = load_restricted_torch_checkpoint(path)

    assert set(loaded["state_dict"]) == {"w"}
