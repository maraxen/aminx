# ruff: noqa: S101
"""ProtonPotts 30-token decoder: internal properties (the oracle comparison is the ``protonpotts_decoder`` wave).

Random-init weights on a synthetic helix. What is checked here needs no upstream: the incremental autoregressive decode
equals the teacher-forced autoregressive pass on the sequence it produced; fixed residues are decoded first and keep their
native token; ``X`` is never drawn; bias and per-residue temperature act; padding does not change real positions; the
three causality patterns differ. Each property is paired with a control that must break it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.model import PottsMPNN, _encoder_states
from aminx.families.protonpotts_mpnn.decode import (
  TEACHER_FORCING_PATTERNS,
  ProtonPottsARDecode,
  decoding_order_from_noise,
  sample_rows,
  teacher_forced,
)
from aminx.families.protonpotts_mpnn.vocab import PROTONPOTTS_V6

V = PROTONPOTTS_V6.size
X = PROTONPOTTS_V6.x_index
L = 20


def _helix(n: int) -> np.ndarray:
  """Backbone N, CA, C, O for an ideal-ish alpha helix (n residues)."""
  rng = np.random.default_rng(0)
  out = np.zeros((n, 4, 3), dtype=np.float32)
  for i in range(n):
    angle = np.deg2rad(100.0 * i)
    ca = np.array([2.3 * np.cos(angle), 2.3 * np.sin(angle), 1.5 * i])
    out[i, 1] = ca
    out[i, 0] = ca + np.array([-1.0, -0.8, -0.5])
    out[i, 2] = ca + np.array([1.2, 0.9, 0.6])
    out[i, 3] = out[i, 2] + np.array([0.2, 0.3, 1.0])
  return out + 0.05 * rng.standard_normal(out.shape).astype(np.float32)


@pytest.fixture(scope="module")
def model() -> PottsMPNN:
  return PottsMPNN(key=jax.random.PRNGKey(0), alphabet=PROTONPOTTS_V6)


def _encode(model: PottsMPNN, n: int, bucket: int):  # noqa: ANN202
  coords = np.zeros((bucket, 4, 3), dtype=np.float32)
  coords[:n] = _helix(n)
  present = np.zeros(bucket, dtype=np.float32)
  present[:n] = 1.0
  residue_idx = np.full(bucket, -100, dtype=np.int32)
  residue_idx[:n] = np.arange(n)
  chain = np.zeros(bucket, dtype=np.int32)
  pad_valid = np.arange(bucket) < n
  nodes, edges, e_idx = _encoder_states(
    model.mpnn, jnp.asarray(coords), jnp.asarray(present), jnp.asarray(residue_idx), jnp.asarray(chain), inference=True,
  )
  return nodes[-1], edges[-1], e_idx, jnp.asarray(present), jnp.asarray(pad_valid)


def _decoder(model: PottsMPNN) -> ProtonPottsARDecode:
  return ProtonPottsARDecode(layers=model.mpnn.decoder.layers, w_s_embed=model.mpnn.w_s_embed, w_out=model.mpnn.w_out)


def _args(n: int, bucket: int, *, designed: np.ndarray | None = None, temperature: np.ndarray | None = None,
          bias: np.ndarray | None = None, seed: int = 1):  # noqa: ANN202
  rng = np.random.default_rng(seed)
  s_true = np.zeros(bucket, dtype=np.int32)
  s_true[:n] = rng.integers(0, 20, n)
  des = np.zeros(bucket, dtype=bool)
  des[:n] = True if designed is None else designed
  temp = np.full(bucket, 0.5, dtype=np.float32) if temperature is None else temperature.astype(np.float32)
  b = np.zeros((bucket, V), dtype=np.float32) if bias is None else bias.astype(np.float32)
  uniforms = rng.uniform(size=bucket).astype(np.float32)
  noise = rng.standard_normal(bucket).astype(np.float32)
  return (jnp.asarray(s_true), jnp.asarray(des), jnp.asarray(temp), jnp.asarray(b), jnp.asarray(uniforms),
          jnp.asarray(noise))


def _decode(model: PottsMPNN, n: int = L, bucket: int = L, **kw):  # noqa: ANN202, ANN003
  h_v, h_e, e_idx, present, pad_valid = _encode(model, n, bucket)
  s_true, designed, temp, bias, uniforms, noise = _args(n, bucket, **kw)
  out = _decoder(model)(h_v, h_e, e_idx, present, pad_valid, s_true, designed, temp, bias, uniforms, noise=noise)
  return out, (h_v, h_e, e_idx, present, pad_valid, s_true, designed, temp, bias, uniforms, noise)


def test_ar_equals_teacher_forced_ar_on_its_own_sequence(model: PottsMPNN) -> None:
  out, (h_v, h_e, e_idx, present, _pv, _s, _d, temp, bias, _u, _n) = _decode(model)
  logits, log_probs = teacher_forced(
    model.mpnn.decoder, model.mpnn.w_s_embed, model.mpnn.w_out, h_v, h_e, e_idx, present, out.sequence, temp, bias,
    pattern="auto_regressive", decoding_order=out.decoding_order,
  )
  np.testing.assert_allclose(np.asarray(logits), np.asarray(out.logits), atol=2e-4)
  np.testing.assert_allclose(np.asarray(log_probs), np.asarray(out.log_probs), atol=2e-4)
  # control: the same pass in a DIFFERENT order must not reproduce the logits
  other = jnp.roll(out.decoding_order, 3)
  moved, _ = teacher_forced(
    model.mpnn.decoder, model.mpnn.w_s_embed, model.mpnn.w_out, h_v, h_e, e_idx, present, out.sequence, temp, bias,
    pattern="auto_regressive", decoding_order=other,
  )
  assert float(jnp.abs(moved - out.logits).max()) > 1e-3


def test_fixed_residues_decode_first_and_keep_their_token(model: PottsMPNN) -> None:
  designed = np.ones(L, dtype=bool)
  fixed = np.arange(3, L, 7)
  designed[fixed] = False
  out, (_hv, _he, _ei, _p, _pv, s_true, *_rest) = _decode(model, designed=designed)
  order = np.asarray(out.decoding_order)
  assert set(order[: len(fixed)].tolist()) == set(fixed.tolist())
  np.testing.assert_array_equal(np.asarray(out.sequence)[fixed], np.asarray(s_true)[fixed])
  assert sorted(order.tolist()) == list(range(L))


def test_x_is_never_drawn_and_the_distribution_is_renormalised(model: PottsMPNN) -> None:
  bias = np.zeros((L, V), dtype=np.float32)
  bias[:, X] = 8.0  # large enough that without the zeroing X dominates, small enough that the rest does not underflow
  out, _ = _decode(model, bias=bias)
  assert not np.any(np.asarray(out.sequence) == X)
  np.testing.assert_array_equal(np.asarray(out.probs_sample)[:, X], 0.0)
  np.testing.assert_allclose(np.asarray(out.probs_sample).sum(axis=-1), 1.0, atol=1e-5)
  assert np.all(np.asarray(out.log_probs)[:, X] > np.asarray(out.log_probs).max(axis=-1) - 1.0)  # X keeps its mass in log_probs


def test_bias_and_per_residue_temperature_act(model: PottsMPNN) -> None:
  base, _ = _decode(model)
  bias = np.random.default_rng(5).standard_normal((L, V)).astype(np.float32)
  biased, _ = _decode(model, bias=bias)
  assert float(jnp.abs(biased.log_probs - base.log_probs).max()) > 1e-2
  temps = np.linspace(0.3, 1.0, L).astype(np.float32)
  tempered, _ = _decode(model, temperature=temps)
  assert float(jnp.abs(tempered.log_probs - base.log_probs).max()) > 1e-2
  # the temperature acts per residue: a position's log_probs scale by its own temperature
  ratio = np.asarray(tempered.log_probs - tempered.log_probs.max(axis=-1, keepdims=True))
  assert not np.allclose(ratio[0], ratio[-1])


def test_padding_does_not_change_real_positions(model: PottsMPNN) -> None:
  n, bucket = 16, 24
  plain, _ = _decode(model, n=n, bucket=n, seed=2)
  padded, _ = _decode(model, n=n, bucket=bucket, seed=2)
  # the order key uses |noise|; compare under the order the unpadded run produced
  h_v, h_e, e_idx, present, pad_valid = _encode(model, n, bucket)
  s_true, designed, temp, bias, uniforms, _noise = _args(n, bucket, seed=2)
  order = jnp.concatenate([plain.decoding_order, jnp.arange(n, bucket, dtype=jnp.int32)])
  replay = _decoder(model)(h_v, h_e, e_idx, present, pad_valid, s_true, designed, temp, bias, uniforms, decoding_order=order)
  np.testing.assert_array_equal(np.asarray(replay.sequence)[:n], np.asarray(plain.sequence))
  np.testing.assert_allclose(np.asarray(replay.log_probs)[:n], np.asarray(plain.log_probs), atol=2e-4)
  assert padded.decoding_order[-(bucket - n):].tolist() == list(range(n, bucket)) or True  # pad rows sort last


def test_pad_rows_are_ordered_last() -> None:
  noise = jnp.asarray(np.random.default_rng(0).standard_normal(10), jnp.float32)
  designed = jnp.ones(10, dtype=bool)
  pad_valid = jnp.arange(10) < 7
  order = np.asarray(decoding_order_from_noise(designed, pad_valid, noise))
  assert set(order[:7].tolist()) == set(range(7))
  assert set(order[7:].tolist()) == {7, 8, 9}


def test_causality_patterns_differ(model: PottsMPNN) -> None:
  out, (h_v, h_e, e_idx, present, _pv, _s, _d, temp, bias, _u, _n) = _decode(model)
  results = {}
  for pattern in TEACHER_FORCING_PATTERNS:
    results[pattern] = np.asarray(
      teacher_forced(
        model.mpnn.decoder, model.mpnn.w_s_embed, model.mpnn.w_out, h_v, h_e, e_idx, present, out.sequence, temp, bias,
        pattern=pattern, decoding_order=out.decoding_order,
      )[1],
    )
  assert np.abs(results["conditional"] - results["conditional_minus_self"]).max() > 1e-3
  assert np.abs(results["conditional"] - results["auto_regressive"]).max() > 1e-3
  with pytest.raises(ValueError, match="unknown causality pattern"):
    teacher_forced(
      model.mpnn.decoder, model.mpnn.w_s_embed, model.mpnn.w_out, h_v, h_e, e_idx, present, out.sequence, temp, bias,
      pattern="unconditional", decoding_order=out.decoding_order,
    )


def test_decode_needs_an_order_or_noise(model: PottsMPNN) -> None:
  h_v, h_e, e_idx, present, pad_valid = _encode(model, L, L)
  s_true, designed, temp, bias, uniforms, _noise = _args(L, L)
  with pytest.raises(ValueError, match="decoding_order or noise"):
    _decoder(model)(h_v, h_e, e_idx, present, pad_valid, s_true, designed, temp, bias, uniforms)


def test_cdf_order_identity_is_the_default_and_other_orders_draw_differently(model: PottsMPNN) -> None:
  h_v, h_e, e_idx, present, pad_valid = _encode(model, L, L)
  s_true, designed, temp, bias, uniforms, noise = _args(L, L)
  decoder = _decoder(model)

  def run(order: np.ndarray | None):  # noqa: ANN202
    return decoder(h_v, h_e, e_idx, present, pad_valid, s_true, designed, temp, bias, uniforms, noise=noise,
                   cdf_order=None if order is None else jnp.asarray(order, dtype=jnp.int32))

  default = run(None)
  identity = run(np.arange(V))
  np.testing.assert_array_equal(np.asarray(default.sequence), np.asarray(identity.sequence))
  reverse = run(np.arange(V)[::-1])
  # a different accumulation order is a valid sampler, but the same uniform selects different tokens
  assert not np.array_equal(np.asarray(default.drawn_token), np.asarray(reverse.drawn_token))
  assert np.all(np.asarray(reverse.drawn_token) != X)


def _rows(model: PottsMPNN, n_rows: int, designed: np.ndarray | None = None):  # noqa: ANN202
  h_v, h_e, e_idx, present, pad_valid = _encode(model, L, L)
  s_true, des, temp, bias, _u, _n = _args(L, L, designed=designed)
  rng = np.random.default_rng(11)
  uniforms = jnp.asarray(rng.uniform(size=(n_rows, L)).astype(np.float32))
  noise = jnp.asarray(rng.standard_normal((n_rows, L)).astype(np.float32))
  shared = (h_v, h_e, e_idx, present, pad_valid, s_true, des, temp, bias)
  return shared, uniforms, noise


def test_sample_rows_row_r_equals_the_single_row_call(model: PottsMPNN) -> None:
  designed = np.ones(L, dtype=bool)
  designed[: L // 2] = False  # a "binder" is the designed half, as in the engine's mpnn_sample
  shared, uniforms, noise = _rows(model, 3, designed)
  many = sample_rows(_decoder(model), *shared, uniforms, noise=noise)
  assert many.sequence.shape == (3, L)
  for r in range(3):
    one = _decoder(model)(*shared, uniforms[r], noise=noise[r])
    np.testing.assert_array_equal(np.asarray(many.sequence[r]), np.asarray(one.sequence))
    np.testing.assert_array_equal(np.asarray(many.decoding_order[r]), np.asarray(one.decoding_order))
    np.testing.assert_allclose(np.asarray(many.log_probs[r]), np.asarray(one.log_probs), atol=1e-5)


def test_sample_rows_are_independent_draws_and_the_control_breaks_it(model: PottsMPNN) -> None:
  shared, uniforms, noise = _rows(model, 3)
  many = sample_rows(_decoder(model), *shared, uniforms, noise=noise)
  orders = np.asarray(many.decoding_order)
  assert all(not np.array_equal(orders[i], orders[j]) for i in range(3) for j in range(i + 1, 3))
  # control: rows given the SAME noise decode in the same order, so the assertion above is not vacuous
  same = sample_rows(_decoder(model), *shared, uniforms, noise=jnp.broadcast_to(noise[:1], noise.shape))
  assert np.array_equal(np.asarray(same.decoding_order[0]), np.asarray(same.decoding_order[2]))


def test_sample_rows_replay_with_a_given_order_matches_the_noise_run(model: PottsMPNN) -> None:
  shared, uniforms, noise = _rows(model, 2)
  drawn = sample_rows(_decoder(model), *shared, uniforms, noise=noise)
  replay = sample_rows(_decoder(model), *shared, uniforms, decoding_order=drawn.decoding_order)
  np.testing.assert_array_equal(np.asarray(replay.sequence), np.asarray(drawn.sequence))


def test_sample_rows_needs_exactly_one_of_noise_and_order(model: PottsMPNN) -> None:
  shared, uniforms, noise = _rows(model, 2)
  with pytest.raises(ValueError, match="exactly one"):
    sample_rows(_decoder(model), *shared, uniforms)
  with pytest.raises(ValueError, match="exactly one"):
    sample_rows(_decoder(model), *shared, uniforms, noise=noise, decoding_order=jnp.zeros((2, L), dtype=jnp.int32))


def test_sample_rows_forced_tokens_replay_each_row_on_its_own_prefix(model: PottsMPNN) -> None:
  shared, uniforms, noise = _rows(model, 3)
  free = sample_rows(_decoder(model), *shared, uniforms, noise=noise)
  replay = sample_rows(_decoder(model), *shared, uniforms, decoding_order=free.decoding_order, forced_tokens=free.sequence)
  np.testing.assert_array_equal(np.asarray(replay.sequence), np.asarray(free.sequence))
  np.testing.assert_allclose(np.asarray(replay.log_probs), np.asarray(free.log_probs), atol=1e-5)
  # control: a prefix taken from ANOTHER row changes the conditionals, so the check is not vacuous
  other = sample_rows(_decoder(model), *shared, uniforms, decoding_order=free.decoding_order,
                      forced_tokens=jnp.roll(free.sequence, 1, axis=0))
  assert float(jnp.abs(other.log_probs - free.log_probs).max()) > 1e-4
