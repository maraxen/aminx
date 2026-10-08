"""The converter refuses checkpoint/skeleton SHAPE mismatches instead of writing them in silently.

Companion to tests/model/test_bias_slots_match_reference.py (bias presence, #162/#163). A width or
vocabulary mismatch (e.g. a ProtonPottsMPNN etab_out (900, 128) loaded into a 400-row skeleton, or a
30-token W_out into a 21-token skeleton) used to be written in by eqx.tree_at and fail much later.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest

_CONVERT = Path(__file__).resolve().parents[2] / "scripts" / "convert_weights.py"


@pytest.fixture(scope="module")
def convert_weights():
    spec = importlib.util.spec_from_file_location("convert_weights_shapes_under_test", _CONVERT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _linear(in_features: int, out_features: int, *, use_bias: bool = True) -> eqx.nn.Linear:
    return eqx.nn.Linear(in_features, out_features, use_bias=use_bias, key=jax.random.PRNGKey(0))


def test_matching_shapes_convert_and_keep_the_weight(convert_weights) -> None:
    w = np.arange(400 * 128, dtype=np.float32).reshape(400, 128)
    b = np.arange(400, dtype=np.float32)
    out = convert_weights.convert_linear_layer(w, b, _linear(128, 400))
    np.testing.assert_array_equal(out.weight, w)
    np.testing.assert_array_equal(out.bias, b)


def test_wide_weight_into_narrow_skeleton_is_refused(convert_weights) -> None:
    """etab_out (900, 128) into a Linear(128 -> 400) skeleton."""
    w = np.zeros((900, 128), np.float32)
    with pytest.raises(ValueError, match=r"weight shape mismatch.*\(900, 128\).*\(400, 128\)"):
        convert_weights.convert_linear_layer(w, None, _linear(128, 400, use_bias=False))


def test_token_vocab_mismatch_is_refused(convert_weights) -> None:
    """A 30-token W_out (30, 128) into a 21-token Linear(128 -> 21) skeleton."""
    w = np.zeros((30, 128), np.float32)
    with pytest.raises(ValueError, match=r"weight shape mismatch.*\(30, 128\).*\(21, 128\)"):
        convert_weights.convert_linear_layer(w, None, _linear(128, 21, use_bias=False))


def test_bias_shape_mismatch_is_refused(convert_weights) -> None:
    w = np.zeros((3, 4), np.float32)
    b = np.zeros(2, np.float32)
    with pytest.raises(ValueError, match=r"bias shape mismatch.*\(2,\).*\(3,\)"):
        convert_weights.convert_linear_layer(w, b, _linear(4, 3))


def test_bias_presence_error_still_raises_checkpoint_side(convert_weights) -> None:
    """Presence check (#162) still fires when the checkpoint has a bias and the skeleton has no slot."""
    w = np.zeros((3, 4), np.float32)
    b = np.zeros(3, np.float32)
    with pytest.raises(ValueError, match="no slot for"):
        convert_weights.convert_linear_layer(w, b, _linear(4, 3, use_bias=False))


def test_bias_presence_error_still_raises_skeleton_side(convert_weights) -> None:
    """Presence check (#163) still fires when the skeleton has a bias and the checkpoint has none."""
    w = np.zeros((3, 4), np.float32)
    with pytest.raises(ValueError, match="random init"):
        convert_weights.convert_linear_layer(w, None, _linear(4, 3, use_bias=True))
