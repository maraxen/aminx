"""The generic autoregressive sampler never emits the unknown token X (debt #2660).

Upstream LigandMPNN samples only the first 20 letters, so a designed position is never X.
A test that only checks the sampled sequence is weak: the synthetic model gives X almost no
probability by chance, so an unmasked sampler would pass it too. These tests force X to win
the logits, so an unmasked draw emits X, and check that the masked sampler does not.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from xtrax.tiling import CarryShape, JaxScanIterator, VmapIterator

from aminx.inference.decode.autoregressive import AutoregressiveDecode
from aminx.inference.encode import make_encode_fn
from aminx.inference.logits import make_stage_set
from aminx.inference.sample_autoregressive import SampleResult
from aminx.model import Aminx
from aminx.types.bundles import InferenceBundle
from aminx.types.configs import InferenceConfig
from aminx.types.stages import StageSet
from aminx.utils.aa_convert import MPNN_ALPHABET
from aminx.utils.sampling_support import X_INDEX, exclude_x_logits
from tests.inference.decode.test_autoregressive import _build_synthetic_fixture

#: Large enough that X dominates every draw when it is not excluded.
_BOOST = 60.0


LogitTransform = Callable[..., jax.Array]


def _boosted_stage_set(stage_set: StageSet, token: int) -> StageSet:
  """Return a copy of ``stage_set`` whose logit transforms add ``_BOOST`` to ``token``."""

  def wrap(fn: LogitTransform) -> LogitTransform:
    def boosted(logits: jax.Array, bias: jax.Array | None = None) -> jax.Array:
      base = fn(logits, bias=bias)
      return base.at[..., token].add(_BOOST)

    return boosted

  return dataclasses.replace(
    stage_set,
    logit_transform=wrap(stage_set.logit_transform),
    ar_logit_transform=(
      None if stage_set.ar_logit_transform is None else wrap(stage_set.ar_logit_transform)
    ),
  )


def _run_ar_decode_with(
  model: Aminx,
  bundle: InferenceBundle,
  config: InferenceConfig,
  stage_set: StageSet,
) -> SampleResult:
  """Encode then run the generic AR decode with a caller-supplied stage set."""
  L = int(bundle.geometry.residue_index.shape[1])
  ar_decode = AutoregressiveDecode(
    model=model,
    state_iterator=VmapIterator(),
    wave_iterator=JaxScanIterator(),
    wave_carry=CarryShape(name="sequence", shape=(L,), dtype=jnp.int32),
  )
  k_enc, k_dec = jax.random.split(jax.random.PRNGKey(7))
  enc = make_encode_fn(model, use_rolling_state=False)(bundle, k_enc, config)
  return ar_decode(key=k_dec, enc=enc, bundle=bundle, config=config, stage_set=stage_set)


def test_exclude_x_logits_sets_only_the_x_column_to_negative_infinity() -> None:
  logits = jnp.arange(3 * 21, dtype=jnp.float32).reshape(3, 21)
  masked = exclude_x_logits(logits)
  assert X_INDEX == MPNN_ALPHABET.index("X") == 20
  assert bool(jnp.all(jnp.isneginf(masked[:, X_INDEX])))
  keep = np.arange(21) != X_INDEX
  np.testing.assert_array_equal(np.asarray(masked[:, keep]), np.asarray(logits[:, keep]))


def test_masked_draw_never_selects_x_even_when_x_dominates() -> None:
  """Positive control for the mask: X is the largest logit, and the masked draw still avoids it."""
  row = jnp.zeros((21,)).at[X_INDEX].set(_BOOST)
  draws = jax.vmap(lambda k: jax.random.categorical(k, exclude_x_logits(row)))(
    jax.random.split(jax.random.PRNGKey(0), 4096),
  )
  assert not bool(jnp.any(draws == X_INDEX))


def test_unmasked_draw_does_select_x_when_x_dominates() -> None:
  """Negative control: without the mask the same logits select X, so the test above can fail."""
  row = jnp.zeros((21,)).at[X_INDEX].set(_BOOST)
  draws = jax.vmap(lambda k: jax.random.categorical(k, row))(
    jax.random.split(jax.random.PRNGKey(0), 4096),
  )
  assert bool(jnp.mean(draws == X_INDEX) > 0.99)


def test_potts_mh_sweep_never_emits_x_under_a_forced_x_field() -> None:
  """The Potts Zanella/MH sweep (``dlmc_sweep``) never moves an active site to X.

  ``logits = h + pair terms``, so +60 on the X column of ``h`` makes X the favoured state
  everywhere. The chain is started from an all-X sequence, which also exercises the entry
  guard that moves active X sites to alanine before the first sweep.
  """
  from aminx.potts.sampling import dlmc_sweep

  n, q = 12, 21
  h = jnp.zeros((n, q)).at[:, X_INDEX].set(_BOOST)
  j = jnp.zeros((n, n, q, q))
  w = jnp.zeros((n, n))
  mask = jnp.ones((n,))
  seq = jnp.full((n,), X_INDEX, dtype=jnp.int32)
  out = dlmc_sweep(jax.random.PRNGKey(3), seq, h, j, w, mask, n_steps=20)
  assert not np.any(np.asarray(out) == X_INDEX), np.asarray(out)


def test_potts_mh_sweep_still_moves_to_a_boosted_standard_letter() -> None:
  """Positive control: the sweep does move sites under a field, so the X test above is not vacuous."""
  from aminx.potts.sampling import dlmc_sweep

  n, q = 12, 21
  h = jnp.zeros((n, q)).at[:, 5].set(_BOOST)  # 5 = 'E' in the MPNN alphabet
  j = jnp.zeros((n, n, q, q))
  w = jnp.zeros((n, n))
  mask = jnp.ones((n,))
  seq = jnp.zeros((n,), dtype=jnp.int32)
  out = dlmc_sweep(jax.random.PRNGKey(3), seq, h, j, w, mask, n_steps=20)
  assert np.all(np.asarray(out) == 5), np.asarray(out)


def test_potts_mh_sweep_keeps_the_sequence_dtype() -> None:
  """The X entry guard does not promote an int8 sequence to int32."""
  from aminx.potts.sampling import dlmc_sweep

  n, q = 6, 21
  h = jnp.zeros((n, q)).at[:, X_INDEX].set(_BOOST)
  seq = jnp.full((n,), X_INDEX, dtype=jnp.int8)
  out = dlmc_sweep(
    jax.random.PRNGKey(0), seq, h, jnp.zeros((n, n, q, q)), jnp.zeros((n, n)), jnp.ones((n,)),
  )
  assert out.dtype == jnp.int8
  assert not np.any(np.asarray(out) == X_INDEX)


def test_potts_sweeps_reject_an_explicit_x_index_outside_the_alphabet() -> None:
  """A 20-letter model has no column 20, and JAX would drop that write silently."""
  from aminx.potts.sampling import dlmc_sweep, gibbs_sweep

  n, q = 4, 20
  args = (
    jax.random.PRNGKey(0),
    jnp.zeros((n,), dtype=jnp.int32),
    jnp.zeros((n, q)),
    jnp.zeros((n, n, q, q)),
    jnp.zeros((n, n)),
    jnp.ones((n,)),
  )
  with pytest.raises(ValueError, match="x_index=20"):
    dlmc_sweep(*args, x_index=X_INDEX)
  with pytest.raises(ValueError, match="x_index=20"):
    gibbs_sweep(*args, x_index=X_INDEX)
  # The default ("auto") excludes nothing on a non-MPNN alphabet, and None opts out.
  assert dlmc_sweep(*args).shape == (n,)
  assert gibbs_sweep(*args, x_index=None)[0].shape == (n,)


def test_potts_auto_x_index_resolves_to_mpnn_x_only_for_q21() -> None:
  from aminx.potts.sampling import _resolve_x_index

  assert _resolve_x_index("auto", 21) == X_INDEX
  assert _resolve_x_index("auto", 20) is None
  assert _resolve_x_index("auto", 3) is None
  assert _resolve_x_index(None, 21) is None
  assert _resolve_x_index(3, 21) == 3


def test_potts_gibbs_sweep_honours_a_custom_x_index() -> None:
  """With X at column 3 and +60 on it, the Gibbs sweep never picks column 3."""
  from aminx.potts.sampling import gibbs_sweep

  n, q, x_col = 12, 21, 3
  h = jnp.zeros((n, q)).at[:, x_col].set(_BOOST)
  out, _ = gibbs_sweep(
    jax.random.PRNGKey(5),
    jnp.zeros((n,), dtype=jnp.int32),
    h,
    jnp.zeros((n, n, q, q)),
    jnp.zeros((n, n)),
    jnp.ones((n,)),
    x_index=x_col,
  )
  assert not np.any(np.asarray(out) == x_col), np.asarray(out)


def test_generic_autoregressive_sampler_never_emits_x_under_a_forced_x_bias() -> None:
  model, bundle, config = _build_synthetic_fixture(num_states=1, num_residues=8, seed=42)
  base = make_stage_set(strategy="arithmetic_mean", state_weights=bundle.conditioning.state_weights)

  # Control: boosting an ordinary letter (Ala, index 0) makes the sampler emit it, so the
  # boost reaches the draw and an all-Ala result cannot be an artefact of the fixture.
  boosted_ala = _run_ar_decode_with(model, bundle, config, _boosted_stage_set(base, 0))
  assert np.all(np.asarray(boosted_ala.sequence) == 0), np.asarray(boosted_ala.sequence)

  # The claim: with X forced to dominate, no designed position is X.
  boosted_x = _run_ar_decode_with(model, bundle, config, _boosted_stage_set(base, X_INDEX))
  assert not np.any(np.asarray(boosted_x.sequence) == X_INDEX), np.asarray(boosted_x.sequence)
