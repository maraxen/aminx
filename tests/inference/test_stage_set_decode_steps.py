"""COMP-3: Tests for DecodeStepFn + SampleStepFn slots in StageSet.

RED phase — all tests must fail until:
  - StageSet gains decode_step and sample_step fields
  - ConditionalDecodeStep and UnconditionalDecodeStep implementations exist
  - Three inline decode_one closures are replaced with stage_set.decode_step
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 1. StageSet field contract
# ---------------------------------------------------------------------------

def test_stage_set_has_decode_step_field():
    """StageSet.decode_step exists and defaults to None."""
    from aminx.types.stages import StageSet
    ss = StageSet()
    assert hasattr(ss, "decode_step")
    assert ss.decode_step is None


def test_stage_set_has_sample_step_field():
    """StageSet.sample_step exists and defaults to None."""
    from aminx.types.stages import StageSet
    ss = StageSet()
    assert hasattr(ss, "sample_step")
    assert ss.sample_step is None


def test_stage_set_decode_step_accepts_callable():
    """StageSet can be constructed with a non-None decode_step."""
    from aminx.types.stages import StageSet

    def dummy_decode(*args, **kwargs):
        raise NotImplementedError

    ss = StageSet(decode_step=dummy_decode)
    assert ss.decode_step is dummy_decode


# ---------------------------------------------------------------------------
# 2. ConditionalDecodeStep implementation
# ---------------------------------------------------------------------------

def test_conditional_decode_step_is_eqx_module():
    """ConditionalDecodeStep is an equinox Module (JAX pytree)."""
    import equinox as eqx
    from aminx.types.stages import ConditionalDecodeStep
    assert issubclass(ConditionalDecodeStep, eqx.Module)


# ---------------------------------------------------------------------------
# 3. UnconditionalDecodeStep implementation
# ---------------------------------------------------------------------------

def test_unconditional_decode_step_is_eqx_module():
    """UnconditionalDecodeStep is an equinox Module (JAX pytree)."""
    import equinox as eqx
    from aminx.types.stages import UnconditionalDecodeStep
    assert issubclass(UnconditionalDecodeStep, eqx.Module)


# ---------------------------------------------------------------------------
# 4. StageSet with decode_step is a valid JAX pytree
# ---------------------------------------------------------------------------

def test_stage_set_with_decode_step_is_pytree():
    """StageSet with decode_step=ConditionalDecodeStep survives jax.tree.leaves."""
    import jax
    from aminx.types.stages import ConditionalDecodeStep, StageSet

    # ConditionalDecodeStep wraps a model decoder — use a dummy eqx.Module
    import equinox as eqx

    class DummyDecoder(eqx.Module):
        pass

    class DummyEmbed(eqx.Module):
        pass

    cd = ConditionalDecodeStep(decoder=DummyDecoder(), w_s_embed=DummyEmbed())
    ss = StageSet(decode_step=cd)
    leaves = jax.tree.leaves(ss)
    assert len(leaves) >= 0  # valid pytree, no exception


# ---------------------------------------------------------------------------
# 5. Kernel integration: stage_set.decode_step is accepted by kernels
#    (import-level smoke; full wiring tested in COMP-5 integration tests)
# ---------------------------------------------------------------------------

def test_score_conditional_kernel_signature_accepts_stage_set():
    """score_conditional.kernel accepts a StageSet argument (signature check)."""
    import inspect
    from aminx.inference.score_conditional import kernel

    sig = inspect.signature(kernel)
    assert "stage_set" in sig.parameters


def test_sample_autoregressive_kernel_signature_accepts_stage_set():
    """sample_autoregressive.kernel accepts a StageSet argument (signature check)."""
    import inspect
    from aminx.inference.sample_autoregressive import kernel

    sig = inspect.signature(kernel)
    assert "stage_set" in sig.parameters


# ---------------------------------------------------------------------------
# 6. DecodeStep __call__ delegation tests
# ---------------------------------------------------------------------------

def test_unconditional_decode_step_call_delegates_to_decoder():
    """UnconditionalDecodeStep.__call__ delegates to decoder and returns output."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from aminx.types.stages import UnconditionalDecodeStep

    class DummyDecoder(eqx.Module):
        def __call__(self, nf, ef, nei, mask, *, key, inference):
            # Passthrough: return node features as-is
            return nf

    step = UnconditionalDecodeStep(decoder=DummyDecoder())
    key = jax.random.PRNGKey(0)
    nf = jnp.ones((4, 8))
    ef = jnp.ones((4, 10, 8))
    nei = jnp.zeros((4, 10), dtype=jnp.int32)
    mask = jnp.ones(4)

    result = step(nf, ef, nei, mask, key=key, inference=True)

    # Verify result shape matches input
    assert result.shape == (4, 8)
    assert jnp.allclose(result, nf)


def test_conditional_decode_step_call_delegates_to_decoder():
    """ConditionalDecodeStep.__call__ delegates to decoder.call_conditional."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from aminx.types.stages import ConditionalDecodeStep

    class DummyDecoder(eqx.Module):
        def call_conditional(self, nf, ef, nei, mask, ar_mask, seq_oh, w_s_embed, *, key, inference):
            # Passthrough: return node features
            return nf

    class DummyEmbed(eqx.Module):
        weight = jnp.ones((21, 16))

    step = ConditionalDecodeStep(decoder=DummyDecoder(), w_s_embed=DummyEmbed())
    key = jax.random.PRNGKey(0)
    nf = jnp.ones((4, 8))
    ef = jnp.ones((4, 10, 8))
    nei = jnp.zeros((4, 10), dtype=jnp.int32)
    mask = jnp.ones(4)
    ar_mask = jnp.ones(4)
    seq_oh = jnp.zeros((4, 21))

    result = step(nf, ef, nei, mask, ar_mask, seq_oh, key=key, inference=True)

    # Verify result matches
    assert result.shape == (4, 8)
    assert jnp.allclose(result, nf)
