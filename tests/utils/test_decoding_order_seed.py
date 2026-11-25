"""``decoding_order_seed``: a serializable control that makes rows share one decoding order.

By default each sample's decoding order is drawn from its own sample key, and that key folds in the row's
manifest hash. So two rows (a single-state row and a fused row, or fused rows at different ``k``) always decode in
different orders. That matters even when every position but one is fixed: an autoregressive MPNN's logits at a
position depend on the order of the *other* positions, because each neighbour's hidden state is conditioned on its
own predecessors only. Rows can therefore never be paired on order, and an order effect reads as an operator effect.

``decoding_order_seed`` draws the order from the seed alone. These tests pin: the default key is unchanged
(bit for bit), the seed makes the order independent of the sample key, different seeds give different orders, and the
spec field is validated and reaches ``run_spec.sampling``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.inference.bundle_builder import (
    DECODING_ORDER_KEY_SALT,
    build_inference_bundle,
    decoding_order_key,
    with_decoding_order,
)
from aminx.run.specs import SamplingSpecification
from aminx.utils.autoregression import decoding_order_from_wave

LENGTH = 12
# Eleven fixed positions and one free probe: the one-free-probe design. The order among the fixed positions is
# still random, which is the point.
PROBE = 5


def _bundle():
    fixed = jnp.ones((LENGTH,), dtype=jnp.float32).at[PROBE].set(0.0)
    bundle, _config = build_inference_bundle(
        coords=jnp.zeros((LENGTH, 4, 3), jnp.float32),
        mask=jnp.ones((LENGTH,), jnp.float32),
        residue_index=jnp.arange(LENGTH, dtype=jnp.int32),
        chain_index=jnp.zeros((LENGTH,), jnp.int32),
        fixed_mask=fixed,
        mode="sample_ar",
        inference=True,
    )
    return bundle


def _order(bundle, key):
    out = with_decoding_order(bundle, key)
    return np.asarray(decoding_order_from_wave(out.wave, out.conditioning.tie_group_map[0]))


def test_default_key_is_unchanged():
    """No seed: the key is exactly the old derivation, so every existing result is bit-identical."""
    for s in (0, 1, 42):
        sample_key = jax.random.PRNGKey(s)
        expected = jax.random.fold_in(sample_key, DECODING_ORDER_KEY_SALT)
        got = decoding_order_key(sample_key)
        assert np.array_equal(np.asarray(got), np.asarray(expected))
        assert np.array_equal(np.asarray(decoding_order_key(sample_key, None)), np.asarray(expected))


def test_seed_makes_key_independent_of_sample_key():
    a = decoding_order_key(jax.random.PRNGKey(1), 7)
    b = decoding_order_key(jax.random.PRNGKey(2), 7)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_different_seeds_give_different_keys_and_zero_is_a_real_seed():
    keys = {s: np.asarray(decoding_order_key(jax.random.PRNGKey(0), s)).tobytes() for s in (0, 1, 2, 3)}
    assert len(set(keys.values())) == 4
    # Seed 0 must not be treated as "unset": with the sample key PRNGKey(5) the unseeded key is
    # derived from PRNGKey(5), while seed 0 derives from PRNGKey(0). An `if not order_seed`
    # regression would make these equal.
    sample_key = jax.random.PRNGKey(5)
    assert (
      np.asarray(decoding_order_key(sample_key, 0)).tobytes()
      != np.asarray(decoding_order_key(sample_key, None)).tobytes()
    )


def test_seed_pins_the_realised_order_and_mask_across_sample_keys():
    bundle = _bundle()
    orders = {tuple(_order(bundle, decoding_order_key(jax.random.PRNGKey(k), 3))) for k in range(8)}
    assert len(orders) == 1
    masks = {
        np.asarray(with_decoding_order(bundle, decoding_order_key(jax.random.PRNGKey(k), 3)).conditioning.ar_mask).tobytes()
        for k in range(8)
    }
    assert len(masks) == 1


def test_without_a_seed_the_order_varies_with_the_sample_key():
    """Control: the fixture really does vary in order, so the pinned test above can fail."""
    bundle = _bundle()
    orders = {tuple(_order(bundle, decoding_order_key(jax.random.PRNGKey(k)))) for k in range(8)}
    assert len(orders) > 1


def test_different_seeds_realise_different_orders():
    bundle = _bundle()
    orders = {tuple(_order(bundle, decoding_order_key(jax.random.PRNGKey(0), s))) for s in range(8)}
    assert len(orders) > 1


def test_probe_is_decoded_last_whatever_the_seed():
    """Fixed-first still holds under a seed; the seed only fixes the order among the fixed positions."""
    bundle = _bundle()
    for s in range(6):
        order = _order(bundle, decoding_order_key(jax.random.PRNGKey(0), s))
        assert int(order[-1]) == PROBE


def _spec(**kwargs):
    return SamplingSpecification(inputs=["state.pdb"], **kwargs)


def test_spec_default_is_none_and_reaches_run_spec():
    assert _spec().decoding_order_seed is None
    assert _spec().run_spec.sampling.decoding_order_seed is None
    assert _spec(decoding_order_seed=0).run_spec.sampling.decoding_order_seed == 0
    assert _spec(decoding_order_seed=11).run_spec.sampling.decoding_order_seed == 11


def test_spec_normalises_numpy_integers():
    spec = _spec(decoding_order_seed=np.int64(5))
    assert spec.decoding_order_seed == 5
    assert type(spec.decoding_order_seed) is int


@pytest.mark.parametrize("bad", [True, 1.5, "3"])
def test_spec_rejects_non_integer_seed(bad):
    with pytest.raises(TypeError, match="decoding_order_seed"):
        _spec(decoding_order_seed=bad)


@pytest.mark.parametrize("bad", [-1, 2**32, 2**63])
def test_spec_rejects_out_of_range_seed(bad):
    with pytest.raises(ValueError, match="decoding_order_seed"):
        _spec(decoding_order_seed=bad)


def test_spec_accepts_the_range_endpoints():
    assert _spec(decoding_order_seed=0).decoding_order_seed == 0
    assert _spec(decoding_order_seed=2**32 - 1).decoding_order_seed == 2**32 - 1
