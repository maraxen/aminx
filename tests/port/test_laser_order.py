"""laser_order wave.

Spec §5.4a / the vehicles table: ``decoding_order`` matches upstream
``_masked_sort_for_decoding_order`` on an injected per-tier uniform stream.
The comparison is exact. ``decoding_order`` draws one uniform per row, so the
test scatters the oracle's tier-0/1/2 stream back onto rows — the mapping §5.4a
assigns to aminx — and injects that vector through ``jax.random.uniform``.

The negative control is an in-test monkeypatch. ``mutant_registry`` only
resolves a process-wide ``AMINX_PORT_MUTANT`` and does not bind one to this test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from port.a1_compare import open_dump, x64_context

from aminx.families.laser_mpnn import driver as laser_driver

pytestmark = [pytest.mark.port_wave("laser_order"), pytest.mark.parity_heavy]


def _scatter(stream: np.ndarray, chain_mask: np.ndarray, contact: np.ndarray) -> np.ndarray:
  """Tier-0, then tier-1, then tier-2, each already in ascending row index."""
  fixed = np.asarray(chain_mask, dtype=bool)
  designable = ~fixed
  touched = np.asarray(contact, dtype=bool) & designable
  plain = designable & ~touched
  out = np.zeros(fixed.shape[0], dtype=np.float64)
  offset = 0
  for mask in (fixed, plain, touched):
    index = np.flatnonzero(mask)
    count = int(index.size)
    out[index] = stream[offset : offset + count]
    offset += count
  if offset != int(stream.shape[0]):
    msg = f"tier stream length {stream.shape[0]} != consumed {offset}"
    raise AssertionError(msg)
  return out


def _install_stream(monkeypatch: pytest.MonkeyPatch, row_uniforms: np.ndarray) -> None:
  def fake_uniform(
    key: jax.Array,
    shape: tuple[int, ...],
    dtype: jnp.dtype | None = None,
    **kwargs: object,
  ) -> jax.Array:
    del key, kwargs
    if tuple(int(dim) for dim in shape) != (int(row_uniforms.shape[0]),):
      msg = f"decoding_order asked for shape {shape}"
      raise AssertionError(msg)
    out_dtype = jnp.float64 if dtype is None else dtype
    return jnp.asarray(row_uniforms, dtype=out_dtype)

  monkeypatch.setattr(jax.random, "uniform", fake_uniform)


def _assert_case(
  monkeypatch: pytest.MonkeyPatch,
  chain_mask: np.ndarray,
  contact: np.ndarray,
  stream: np.ndarray,
  oracle_order: np.ndarray,
) -> None:
  row_uniforms = _scatter(stream, chain_mask, contact)
  _install_stream(monkeypatch, row_uniforms)
  got = np.asarray(
    laser_driver.decoding_order(chain_mask, contact, seed=0),
    dtype=np.int64,
  )
  assert np.array_equal(got, oracle_order.astype(np.int64))

  def drop_tier(cond: object, left: object, right: object) -> jax.Array:
    """Drop the tier offset. ``where`` is only used to build that offset."""
    del cond, right
    return laser_driver.jnp.zeros_like(laser_driver.jnp.asarray(left))

  # Tier offsets are what separate the tiers. Zeroing jnp.where drops them.
  monkeypatch.setattr(laser_driver.jnp, "where", drop_tier)
  dropped = np.asarray(
    laser_driver.decoding_order(chain_mask, contact, seed=0),
    dtype=np.int64,
  )
  assert not np.array_equal(dropped, oracle_order.astype(np.int64))
  monkeypatch.undo()


def test_knob_semantics_order_generation(oracle: object, monkeypatch: pytest.MonkeyPatch) -> None:
  """Both LASEr order fixtures match, and dropping the tier offset does not.

  (a) is an inference-featurized batch: ``extra_atom_contact_mask`` is the
  featurizer's zeros, so tier 2 is empty, and the chain mask mixes fixed and
  designable rows. (b) sets that contact mask by hand on designable rows, with
  fixed and non-contact rows still present, because inference cannot produce
  tier 2. The injected uniforms put fixed rows near 0.9 and designable rows
  near 0.05, so the offset is the only reason fixed rows come first. Without
  it the comparison fails; a one-tier mask would have left it equal.
  """
  data = open_dump(oracle, "f64")
  try:
    infer_chain = np.asarray(data["infer_chain_mask"]).astype(bool)
    infer_contact = np.asarray(data["infer_contact"]).astype(bool)
    hand_chain = np.asarray(data["hand_chain_mask"]).astype(bool)
    hand_contact = np.asarray(data["hand_contact"]).astype(bool)
    assert not bool(infer_contact.any())
    assert bool(infer_chain.any()) and bool((~infer_chain).any())
    assert bool(hand_chain.any())
    assert bool(hand_contact.any())
    assert bool((~hand_chain & ~hand_contact).any())
    with x64_context("f64"):
      _assert_case(
        monkeypatch,
        infer_chain,
        infer_contact,
        np.asarray(data["infer_uniforms"], dtype=np.float64),
        np.asarray(data["infer_order"]),
      )
      _assert_case(
        monkeypatch,
        hand_chain,
        hand_contact,
        np.asarray(data["hand_uniforms"], dtype=np.float64),
        np.asarray(data["hand_order"]),
      )
  finally:
    data.close()
