"""Host compilation of omit_aa into the upstream -1e8 design bias."""

from __future__ import annotations

import jax.numpy as jnp

from aminx.host.omit_aa_bias import OMIT_AA_BIAS, compile_omit_aa_bias
from aminx.utils.aa_convert import MPNN_ALPHABET


def test_inactive_omit_returns_the_same_bias_object() -> None:
    bias = jnp.zeros((3, 21), dtype=jnp.float32)
    assert compile_omit_aa_bias(
        bias,
        fixed_mask_row=None,
        omit_aa=(),
        omit_aa_per_position=None,
        seq_len=3,
    ) is bias
    assert compile_omit_aa_bias(
        None,
        fixed_mask_row=None,
        omit_aa=(),
        omit_aa_per_position={},
        seq_len=3,
    ) is None


def test_omit_aa_applies_penalty_only_at_designed_positions() -> None:
    fixed = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    compiled = compile_omit_aa_bias(
        None,
        fixed_mask_row=fixed,
        omit_aa=("C",),
        omit_aa_per_position={2: "W"},
        seq_len=3,
    )
    c_index = MPNN_ALPHABET.index("C")
    w_index = MPNN_ALPHABET.index("W")
    assert float(compiled[0, c_index]) == 0.0
    assert float(compiled[1, c_index]) == OMIT_AA_BIAS
    assert float(compiled[2, c_index]) == OMIT_AA_BIAS
    assert float(compiled[2, w_index]) == OMIT_AA_BIAS
    assert float(compiled[1, w_index]) == 0.0


def test_omit_aa_adds_to_existing_bias() -> None:
    bias = jnp.ones((2, 21), dtype=jnp.float32)
    compiled = compile_omit_aa_bias(
        bias,
        fixed_mask_row=jnp.zeros((2,), dtype=jnp.float32),
        omit_aa=("A",),
        omit_aa_per_position=None,
        seq_len=2,
    )
    # float32 absorbs the 1.0 next to -1e8; compare in the bias dtype.
    assert float(compiled[0, 0]) == float(jnp.float32(1.0) + jnp.float32(OMIT_AA_BIAS))
    assert float(compiled[1, 1]) == 1.0
