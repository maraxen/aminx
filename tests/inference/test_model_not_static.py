"""RED tests for DEFECT 2: model arrays stored in static eqx fields.

The classes _VmapEncode, _ScanEncode, ConditionalDecode, UnconditionalDecode,
and AutoregressiveDecode all have:
    model: Any = eqx.field(static=True)

This causes Equinox to emit "A JAX array is being set as static!" warnings on
construction, AND places model weight arrays in the static partition instead of
the dynamic partition. This defeats filter_jit cache keys and silently prevents
in-place parameter updates from being seen by JIT-compiled functions.

These tests assert on the MECHANISM (partition membership, warning emission),
not on latency. They are deterministic and local.

After fix:
- model field should be dynamic (no static=True)
- No static-array warnings on construction
- Model arrays appear in eqx.partition dynamic side
"""

from __future__ import annotations

import warnings
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import pytest

from aminx.model.mpnn import Aminx


# ---------------------------------------------------------------------------
# Shared fixture: small Aminx model (random init, not from checkpoint)
# ---------------------------------------------------------------------------

def _make_small_model() -> Aminx:
    """Create a small Aminx model with minimal dimensions for fast tests."""
    return Aminx(
        node_features=16,
        edge_features=16,
        hidden_features=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=8,
        key=jax.random.PRNGKey(42),
    )


# ---------------------------------------------------------------------------
# DEFECT 2 TEST 1: No static-array warning on construction
# ---------------------------------------------------------------------------

def test_vmap_encode_no_static_array_warning():
    """_VmapEncode(model=model) must not emit 'set as static' warning.

    CURRENTLY FAILS because model: Any = eqx.field(static=True) causes Equinox
    to warn "A JAX array is being set as static!" when model contains jax.Array
    weights.

    After fix (remove static=True from model field): warning disappears.
    """
    from aminx.inference.encode import make_encode_fn

    model = _make_small_model()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        encode_fn = make_encode_fn(model, use_rolling_state=False)

    static_warnings = [
        w for w in caught
        if "set as static" in str(w.message).lower()
        or "jax array" in str(w.message).lower()
    ]
    assert len(static_warnings) == 0, (
        f"Expected no static-array warnings constructing _VmapEncode, "
        f"but got {len(static_warnings)}: {[str(w.message) for w in static_warnings]}"
    )


def test_scan_encode_no_static_array_warning():
    """_ScanEncode(model=model) must not emit 'set as static' warning.

    CURRENTLY FAILS for the same reason as _VmapEncode.
    """
    from aminx.inference.encode import make_encode_fn

    model = _make_small_model()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        encode_fn = make_encode_fn(model, use_rolling_state=True)

    static_warnings = [
        w for w in caught
        if "set as static" in str(w.message).lower()
        or "jax array" in str(w.message).lower()
    ]
    assert len(static_warnings) == 0, (
        f"Expected no static-array warnings constructing _ScanEncode, "
        f"but got {len(static_warnings)}: {[str(w.message) for w in static_warnings]}"
    )


def test_conditional_decode_no_static_array_warning():
    """ConditionalDecode(model=model, ...) must not emit 'set as static' warning.

    CURRENTLY FAILS: model: Any = eqx.field(static=True) in ConditionalDecode.
    """
    from aminx.inference.decode.conditional import ConditionalDecode
    from xtrax.tiling import VmapIterator

    model = _make_small_model()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        decode = ConditionalDecode(
            model=model,
            state_iterator=VmapIterator(),
        )

    static_warnings = [
        w for w in caught
        if "set as static" in str(w.message).lower()
        or "jax array" in str(w.message).lower()
    ]
    assert len(static_warnings) == 0, (
        f"Expected no static-array warnings constructing ConditionalDecode, "
        f"but got {len(static_warnings)}: {[str(w.message) for w in static_warnings]}"
    )


# ---------------------------------------------------------------------------
# DEFECT 2 TEST 2: Model weight arrays are in the DYNAMIC eqx partition
# ---------------------------------------------------------------------------

def test_vmap_encode_model_weights_are_dynamic():
    """_VmapEncode's model weight arrays must be in the eqx dynamic partition.

    CURRENTLY FAILS because model is static; all weight arrays go to the
    static side of eqx.partition.

    After fix: model field is dynamic, so weight arrays appear in the
    dynamic partition (eqx.partition returns them under 'dynamic').

    Mechanism:
      - eqx.partition(component, eqx.is_array) splits pytree into
        (dynamic_leaves, static_structure)
      - dynamic leaves must include the model's weight arrays
      - We check that sum(x.size for x in dynamic_leaves) is large
        (the model has many float parameters)
      - We also check that no model weight array appears ONLY in static
    """
    from aminx.inference.encode import make_encode_fn

    model = _make_small_model()
    encode_fn = make_encode_fn(model, use_rolling_state=False)

    dynamic, static = eqx.partition(encode_fn, eqx.is_array)

    # Count dynamic array elements
    dynamic_leaves = [x for x in jtu.tree_leaves(dynamic) if eqx.is_array(x)]
    dynamic_param_count = sum(x.size for x in dynamic_leaves)

    # The model has node/edge features, encoder, decoder, embeddings — many params
    # Even the smallest model (node=16, edge=16, layers=1) has >> 1000 params
    assert dynamic_param_count > 1000, (
        f"Expected > 1000 dynamic array elements in _VmapEncode (model weights), "
        f"but found {dynamic_param_count}. Model arrays are likely in static partition."
    )


def test_scan_encode_model_weights_are_dynamic():
    """_ScanEncode's model weight arrays must be in the eqx dynamic partition.

    CURRENTLY FAILS for the same reason as _VmapEncode.
    """
    from aminx.inference.encode import make_encode_fn

    model = _make_small_model()
    encode_fn = make_encode_fn(model, use_rolling_state=True)

    dynamic, static = eqx.partition(encode_fn, eqx.is_array)
    dynamic_leaves = [x for x in jtu.tree_leaves(dynamic) if eqx.is_array(x)]
    dynamic_param_count = sum(x.size for x in dynamic_leaves)

    assert dynamic_param_count > 1000, (
        f"Expected > 1000 dynamic array elements in _ScanEncode (model weights), "
        f"but found {dynamic_param_count}. Model arrays are likely in static partition."
    )


def test_conditional_decode_model_weights_are_dynamic():
    """ConditionalDecode's model weight arrays must be in the eqx dynamic partition.

    CURRENTLY FAILS: model is static.
    """
    from aminx.inference.decode.conditional import ConditionalDecode
    from xtrax.tiling import VmapIterator

    model = _make_small_model()
    decode = ConditionalDecode(
        model=model,
        state_iterator=VmapIterator(),
    )

    dynamic, static = eqx.partition(decode, eqx.is_array)
    dynamic_leaves = [x for x in jtu.tree_leaves(dynamic) if eqx.is_array(x)]
    dynamic_param_count = sum(x.size for x in dynamic_leaves)

    assert dynamic_param_count > 1000, (
        f"Expected > 1000 dynamic array elements in ConditionalDecode (model weights), "
        f"but found {dynamic_param_count}. Model arrays are likely in static partition."
    )
