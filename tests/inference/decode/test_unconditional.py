"""Tests for UnconditionalDecode mode class (Task 8).

UnconditionalDecode wraps unconditional scoring with state-axis iteration via
a MapIterator (Vmap, SafeMap). Tests verify numerical parity with
reference implementation over fixture sizes S ∈ {1, 4, 8} and
two iterator strategies (Vmap, SafeMap).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.decode.unconditional import UnconditionalDecode
from aminx.inference.logits import make_stage_set
from aminx.model import Aminx
from aminx.tiling.iterator import VmapIterator
from xtrax.tiling import ChunkedMapIterator
from aminx.types.configs import InferenceConfig
from aminx.types.bundles import (
    ConditioningBundle,
    EncoderOutput,
    GeometryBundle,
    InferenceBundle,
    WaveScheduleBundle,
    LigandBundle,
)
from aminx.types.stages import UnconditionalDecodeStep


def _build_synthetic_fixture(
    num_states: int = 1,
    num_residues: int = 8,
    seed: int = 42,
) -> tuple[Aminx, EncoderOutput, "InferenceBundle", InferenceConfig]:  # type: ignore
    """Build deterministic fixture with S=num_states, L=num_residues."""
    rng = np.random.default_rng(seed)
    jax_key = jax.random.PRNGKey(seed)

    # Build model
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

    # Build encoder output with S states
    L = num_residues
    K = 5  # k_neighbors

    node_features = jax.random.normal(jax.random.PRNGKey(seed), (num_states, L, 64))
    edge_features = jax.random.normal(jax.random.PRNGKey(seed + 1), (num_states, L, K, 64))
    neighbor_indices = jnp.zeros((num_states, L, K), dtype=jnp.int32)
    mask = jnp.ones((num_states, L), dtype=jnp.float32)

    enc = EncoderOutput(
        node_features=node_features,
        edge_features=edge_features,
        neighbor_indices=neighbor_indices,
        mask=mask,
    )

    # Build conditioning bundle (unconditional only uses bias)
    bias = jax.random.normal(jax.random.PRNGKey(seed + 2), (L, 21))
    cond = ConditioningBundle(
        fixed_mask=jnp.zeros((L,)),
        fixed_tokens=jnp.zeros((L,), dtype=jnp.int32),
        bias=bias,
        tie_group_map=jnp.zeros((num_states, L), dtype=jnp.int32),
        state_position_map=jnp.broadcast_to(jnp.arange(L)[None, :], (num_states, L)),
        state_weights=jnp.ones((num_states,)) / num_states,
        sequence_oh=jnp.zeros((L, 21)),  # zeros for unconditional
        ar_mask=jnp.ones((num_states, L, L)),  # full 1s for non-AR
        temperature=jnp.array(1.0),
    )

    config = InferenceConfig(inference=True)

    # Create geometry bundle for InferenceBundle
    coords = jax.random.normal(jax.random.PRNGKey(seed + 3), (num_states, L, 4, 3))
    residue_index_stack = jnp.broadcast_to(
        jnp.arange(L)[None, :],
        (num_states, L)
    )
    geometry = GeometryBundle(
        coords=coords,
        mask=mask,
        residue_index=residue_index_stack,
        chain_index=jnp.zeros((num_states, L), dtype=jnp.int32),
        n_states=num_states,
        n_canonical=20,
        n_flat=num_states * L,
    )

    # Create ligand bundle (empty for unconditional)
    ligand = LigandBundle(
        ligand_coords=jnp.zeros((num_states, L, 5, 3)),
        ligand_atom_types=jnp.zeros((num_states, L, 5), dtype=jnp.int32),
        ligand_mask=jnp.zeros((num_states, L, 5)),
    )

    # Create wave schedule bundle (empty, not used by unconditional)
    wave = WaveScheduleBundle.empty(L)

    # Create InferenceBundle
    bundle = InferenceBundle(
        geometry=geometry,
        conditioning=cond,
        ligand=ligand,
        wave=wave,
    )

    return model, enc, bundle, config


def _reference_decode_unconditional(
    model,
    key,
    enc,
    bundle,
    config,
    stage_set,
    num_states,
):
    """Reference implementation (reimplemented from driver._decode_unconditional to avoid assertion bug)."""
    cond = bundle.conditioning

    def decode_one(node_features, edge_features, neighbor_indices, mask):
        # Use fallback path: call model.decoder directly
        return model.decoder(
            node_features,
            edge_features,
            neighbor_indices,
            mask,
            key=key,
        )

    # Decode over states: (S, L, H)
    decoded = jax.vmap(decode_one, in_axes=(0, 0, 0, 0))(
        enc.node_features, enc.edge_features, enc.neighbor_indices, enc.mask,
    )

    # Project to logits: (S, L, V) -> (S, L, 21)
    logits_stack = jax.vmap(jax.vmap(model.w_out, in_axes=0), in_axes=0)(decoded)

    # Fuse across states
    return stage_set.logit_transform(logits_stack, bias=cond.bias)


@pytest.mark.parametrize("num_states", [1, 4, 8])
@pytest.mark.parametrize("iterator_type", ["vmap", "safemap"])
def test_unconditional_parity(num_states: int, iterator_type: str):
    """Test UnconditionalDecode matches reference implementation."""
    model, enc, bundle, config = _build_synthetic_fixture(num_states=num_states)
    jax_key = jax.random.PRNGKey(42)

    # Build stage_set without decode_step, with proper state_weights
    # (Using None for decode_step tests the fallback path model.decoder())
    state_weights = jnp.ones(num_states) / num_states
    stage_set = make_stage_set(state_weights=state_weights)

    # Reference: reimplement unconditional decode logic
    ref_logits = _reference_decode_unconditional(
        model=model,
        key=jax_key,
        enc=enc,
        bundle=bundle,
        config=config,
        stage_set=stage_set,
        num_states=num_states,
    )

    # Implementation: call UnconditionalDecode
    if iterator_type == "vmap":
        iterator = VmapIterator()
    else:
        iterator = ChunkedMapIterator(tile=2)

    unconditional_decode = UnconditionalDecode(
        model=model,
        state_iterator=iterator,
    )

    impl_logits = unconditional_decode(
        key=jax_key,
        enc=enc,
        bundle=bundle,
        config=config,
        stage_set=stage_set,
    )

    # Verify numerical parity (tolerance ≤1e-6)
    np.testing.assert_allclose(ref_logits, impl_logits, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
