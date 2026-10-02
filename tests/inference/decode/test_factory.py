"""Tests for make_decode_fn factory (Task 11).

Task 11.1: Factory tests
  - Happy path: (ConditionalMode(), Vmap()) → ConditionalDecode with VmapIterator
  - (UnconditionalMode(), SafeMap(tile=4)) → UnconditionalDecode with SafeMapIterator(tile=4)
  - (AutoregressiveMode(), Vmap()) → AutoregressiveDecode with JaxScanIterator + CarryShape
  - (STEMode(inner_mode=ConditionalMode(), iterations=50), Vmap()) → STEDecode(inner=ConditionalDecode(...), iterations=50)
  - Reject path with mode-name context (Risk D-12): make_decode_fn(model, ConditionalMode(), Scan(...))
    raises DispatchRejected with message matching "ConditionalMode" AND "heterogeneous"
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

from aminx.inference.decode.conditional import ConditionalDecode
from aminx.inference.decode.factory import make_decode_fn
from aminx.inference.decode.mode import (
    AutoregressiveMode,
    ConditionalMode,
    STEMode,
    UnconditionalMode,
)
from aminx.inference.decode.ste import STEDecode
from aminx.inference.decode.unconditional import UnconditionalDecode
from aminx.tiling.dispatch import DispatchRejected
from aminx.tiling.strategy import SafeMap, Scan, Vmap


class DummyModel:
    """Minimal model for factory testing."""
    pass


class TestMakeDecodeFnConditional:
    """Happy-path tests for ConditionalMode."""

    def test_conditional_with_vmap(self):
        """(ConditionalMode(), Vmap()) → ConditionalDecode with VmapIterator."""
        model = DummyModel()
        mode = ConditionalMode()
        strategy = Vmap()

        result = make_decode_fn(model, mode, strategy)

        assert isinstance(result, ConditionalDecode)
        assert type(result.state_iterator).__name__ == "VmapIterator"
        assert result.model is model

    def test_conditional_with_safemap(self):
        """(ConditionalMode(), SafeMap(tile=4)) → ConditionalDecode with SafeMapIterator(tile=4)."""
        model = DummyModel()
        mode = ConditionalMode()
        strategy = SafeMap(tile=4)

        result = make_decode_fn(model, mode, strategy)

        assert isinstance(result, ConditionalDecode)
        assert type(result.state_iterator).__name__ == "ChunkedMapIterator"
        assert result.state_iterator.tile == 4
        assert result.model is model


class TestMakeDecodeFnUnconditional:
    """Happy-path tests for UnconditionalMode."""

    def test_unconditional_with_vmap(self):
        """(UnconditionalMode(), Vmap()) → UnconditionalDecode with VmapIterator."""
        model = DummyModel()
        mode = UnconditionalMode()
        strategy = Vmap()

        result = make_decode_fn(model, mode, strategy)

        assert isinstance(result, UnconditionalDecode)
        assert type(result.state_iterator).__name__ == "VmapIterator"
        assert result.model is model

    def test_unconditional_with_safemap(self):
        """(UnconditionalMode(), SafeMap(tile=2)) → UnconditionalDecode with SafeMapIterator(tile=2)."""
        model = DummyModel()
        mode = UnconditionalMode()
        strategy = SafeMap(tile=2)

        result = make_decode_fn(model, mode, strategy)

        assert isinstance(result, UnconditionalDecode)
        assert type(result.state_iterator).__name__ == "ChunkedMapIterator"
        assert result.state_iterator.tile == 2


class TestMakeDecodeFnAutoregressive:
    """Happy-path tests for AutoregressiveMode."""

    def test_autoregressive_with_vmap(self):
        """(AutoregressiveMode(), Vmap()) → AutoregressiveDecode with JaxScanIterator + CarryShape."""
        model = DummyModel()
        mode = AutoregressiveMode()
        strategy = Vmap()

        result = make_decode_fn(model, mode, strategy)

        from aminx.inference.decode.autoregressive import AutoregressiveDecode

        assert isinstance(result, AutoregressiveDecode)
        assert type(result.state_iterator).__name__ == "VmapIterator"
        assert type(result.wave_iterator).__name__ == "JaxScanIterator"
        # Check wave_carry metadata
        assert result.wave_carry.name == "sequence"
        assert result.wave_carry.shape == (1024,)  # Default L for test
        assert result.wave_carry.dtype == jnp.int32
        assert result.model is model

    def test_autoregressive_with_safemap(self):
        """AutoregressiveMode with SafeMap(tile=4)."""
        model = DummyModel()
        mode = AutoregressiveMode()
        strategy = SafeMap(tile=4)

        result = make_decode_fn(model, mode, strategy)

        from aminx.inference.decode.autoregressive import AutoregressiveDecode

        assert isinstance(result, AutoregressiveDecode)
        assert type(result.state_iterator).__name__ == "ChunkedMapIterator"
        assert result.state_iterator.tile == 4
        assert type(result.wave_iterator).__name__ == "JaxScanIterator"


class TestMakeDecodeFnSTE:
    """Happy-path tests for STEMode."""

    def test_ste_with_conditional_inner_vmap(self):
        """(STEMode(inner_mode=ConditionalMode(), iterations=50), Vmap()) → STEDecode(inner=ConditionalDecode(...), iterations=50)."""
        model = DummyModel()
        mode = STEMode(inner_mode=ConditionalMode(), iterations=50)
        strategy = Vmap()

        result = make_decode_fn(model, mode, strategy)

        assert isinstance(result, STEDecode)
        assert result.iterations == 50
        assert isinstance(result.inner, ConditionalDecode)
        assert type(result.inner.state_iterator).__name__ == "VmapIterator"

    def test_ste_with_conditional_inner_safemap(self):
        """STEMode with SafeMap inner strategy."""
        model = DummyModel()
        mode = STEMode(inner_mode=ConditionalMode(), iterations=100)
        strategy = SafeMap(tile=3)

        result = make_decode_fn(model, mode, strategy)

        assert isinstance(result, STEDecode)
        assert result.iterations == 100
        assert isinstance(result.inner, ConditionalDecode)
        assert type(result.inner.state_iterator).__name__ == "ChunkedMapIterator"
        assert result.inner.state_iterator.tile == 3

    def test_ste_default_iterations(self):
        """STEMode with default iterations (100)."""
        model = DummyModel()
        mode = STEMode()  # iterations defaults to 100
        strategy = Vmap()

        result = make_decode_fn(model, mode, strategy)

        assert isinstance(result, STEDecode)
        assert result.iterations == 100


class TestMakeDecodeFnRejection:
    """Rejection tests (Risk D-12): mode-name context wrapping."""

    def test_conditional_mode_reject_scan(self):
        """ConditionalMode with Scan raises DispatchRejected with mode-name + heterogeneous in message."""
        model = DummyModel()
        mode = ConditionalMode()
        # Scan with a simple init/transition (the exact structure doesn't matter for rejection)
        strategy = Scan(init=jnp.zeros(()), transition=lambda c, x: (c, x))

        with pytest.raises(DispatchRejected) as exc_info:
            make_decode_fn(model, mode, strategy)

        # Message must contain both mode name and heterogeneous reference
        message = str(exc_info.value).lower()
        assert "conditionalmode" in message, f"Message missing 'ConditionalMode': {exc_info.value}"
        assert "heterogeneous" in message, f"Message missing 'heterogeneous': {exc_info.value}"

    def test_unconditional_mode_reject_scan(self):
        """UnconditionalMode with Scan raises DispatchRejected with mode-name + heterogeneous."""
        model = DummyModel()
        mode = UnconditionalMode()
        strategy = Scan(init=jnp.zeros((2, 3)), transition=lambda c, x: (c, x))

        with pytest.raises(DispatchRejected) as exc_info:
            make_decode_fn(model, mode, strategy)

        message = str(exc_info.value).lower()
        assert "unconditionalmode" in message, f"Message missing 'UnconditionalMode': {exc_info.value}"
        assert "heterogeneous" in message, f"Message missing 'heterogeneous': {exc_info.value}"

    def test_autoregressive_mode_reject_scan(self):
        """AutoregressiveMode with Scan raises DispatchRejected with mode-name + heterogeneous."""
        model = DummyModel()
        mode = AutoregressiveMode()
        strategy = Scan(init=jnp.zeros(()), transition=lambda c, x: (c, x))

        with pytest.raises(DispatchRejected) as exc_info:
            make_decode_fn(model, mode, strategy)

        message = str(exc_info.value).lower()
        assert "autoregressivemode" in message, f"Message missing 'AutoregressiveMode': {exc_info.value}"
        assert "heterogeneous" in message, f"Message missing 'heterogeneous': {exc_info.value}"

    def test_ste_mode_reject_scan(self):
        """STEMode with Scan raises DispatchRejected with STEMode name + heterogeneous."""
        model = DummyModel()
        mode = STEMode()
        strategy = Scan(init=jnp.zeros(()), transition=lambda c, x: (c, x))

        with pytest.raises(DispatchRejected) as exc_info:
            make_decode_fn(model, mode, strategy)

        message = str(exc_info.value).lower()
        # STE wraps the inner Conditional error, so we should see either STEMode or ConditionalMode
        assert ("stemode" in message or "conditionalmode" in message), f"Message missing mode name: {exc_info.value}"
        assert "heterogeneous" in message, f"Message missing 'heterogeneous': {exc_info.value}"
