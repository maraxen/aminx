"""Tests for AutoregressiveConfig.inference_only wiring (use_while_loop passthrough).

Verifies two things:
1. Correctness parity: AutoregressiveDecode(use_while_loop=True) produces
   byte-identical output to use_while_loop=False (default) for the same
   inputs -- while_loop and scan implement the same algorithm, just a
   different XLA loop construct.
2. End-to-end wiring: make_decode_fn(..., autoregressive_config=...) and
   sample_autoregressive.kernel(..., inference_only=...) correctly propagate
   the flag down to AutoregressiveDecode.use_while_loop (previously
   hardcoded to False with no way to override -- see aminx debt tracking
   this session for the fix).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.decode.autoregressive import AutoregressiveDecode
from aminx.inference.decode.factory import make_decode_fn
from aminx.inference.decode.mode import AutoregressiveConfig, AutoregressiveMode
from aminx.inference.encode import make_encode_fn
from aminx.inference.logits import make_stage_set
from aminx.inference.sample_autoregressive import kernel as sample_autoregressive_kernel
from aminx.model import Aminx
from xtrax.tiling import CarryShape
from aminx.tiling.iterator import JaxScanIterator, VmapIterator
from aminx.tiling.strategy import Vmap
from aminx.types.bundles import InferenceBundle
from aminx.types.configs import InferenceConfig


def _build_fixture(num_residues: int = 8, seed: int = 7) -> tuple[Aminx, InferenceBundle, InferenceConfig]:
    rng = np.random.default_rng(seed)
    jax_key = jax.random.PRNGKey(seed)

    model = Aminx(
        node_features=64,
        edge_features=64,
        hidden_features=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        k_neighbors=5,
        dropout_rate=0.0,
        key=jax_key,
    )
    model = eqx.tree_inference(model, value=True)

    coordinates = jnp.array(rng.normal(size=(num_residues, 4, 3)).astype(np.float32))
    mask = jnp.ones((num_residues,), dtype=jnp.float32)
    residue_index = jnp.arange(num_residues, dtype=jnp.int32)
    chain_index = jnp.zeros((num_residues,), dtype=jnp.int32)

    bundle, config = build_inference_bundle(
        coords=coordinates,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        state_weights=jnp.ones(1),
        sequence=None,
        mode="sample_autoregressive",
    )
    return model, bundle, config


def test_use_while_loop_parity_with_scan():
    """AutoregressiveDecode(use_while_loop=True/False) produce identical output."""
    model, bundle, config = _build_fixture()
    L = int(bundle.geometry.residue_index.shape[0])
    wave_carry = CarryShape(name="sequence", shape=(L,), dtype=jnp.int32)
    stage_set = make_stage_set(strategy="arithmetic_mean", state_weights=bundle.conditioning.state_weights)

    def run(use_while_loop: bool):
        key = jax.random.PRNGKey(123)
        ar_decode = AutoregressiveDecode(
            model=model,
            state_iterator=VmapIterator(),
            wave_iterator=JaxScanIterator(),
            wave_carry=wave_carry,
            use_while_loop=use_while_loop,
        )
        k_enc, k_dec = jax.random.split(key)
        encode_fn = make_encode_fn(model, use_rolling_state=False)
        enc = encode_fn(bundle, k_enc, config)
        return ar_decode(key=k_dec, enc=enc, bundle=bundle, config=config, stage_set=stage_set)

    result_scan = run(use_while_loop=False)
    result_while = run(use_while_loop=True)

    np.testing.assert_array_equal(np.asarray(result_scan.sequence), np.asarray(result_while.sequence))
    np.testing.assert_allclose(
        np.asarray(result_scan.logits), np.asarray(result_while.logits), rtol=1e-5, atol=1e-5,
    )


def test_make_decode_fn_wires_autoregressive_config():
    """make_decode_fn(..., autoregressive_config=...) sets use_while_loop correctly."""
    model, _, _ = _build_fixture()

    decode_default = make_decode_fn(model=model, mode=AutoregressiveMode(), strategy=Vmap())
    assert decode_default.use_while_loop is False, "default (no config passed) must stay scan (safe for training)"

    decode_while = make_decode_fn(
        model=model, mode=AutoregressiveMode(), strategy=Vmap(),
        autoregressive_config=AutoregressiveConfig(inference_only=True),
    )
    assert decode_while.use_while_loop is True

    decode_scan_explicit = make_decode_fn(
        model=model, mode=AutoregressiveMode(), strategy=Vmap(),
        autoregressive_config=AutoregressiveConfig(inference_only=False),
    )
    assert decode_scan_explicit.use_while_loop is False


def test_sample_autoregressive_kernel_inference_only_parity():
    """sample_autoregressive.kernel(..., inference_only=True/False) produce identical output."""
    model, bundle, config = _build_fixture()
    stage_set = make_stage_set(strategy="arithmetic_mean", state_weights=bundle.conditioning.state_weights)

    result_scan = sample_autoregressive_kernel(
        model, jax.random.PRNGKey(99), bundle, config, stage_set, inference_only=False,
    )
    result_while = sample_autoregressive_kernel(
        model, jax.random.PRNGKey(99), bundle, config, stage_set, inference_only=True,
    )

    np.testing.assert_array_equal(np.asarray(result_scan.sequence), np.asarray(result_while.sequence))
    np.testing.assert_allclose(
        np.asarray(result_scan.logits), np.asarray(result_while.logits), rtol=1e-5, atol=1e-5,
    )
