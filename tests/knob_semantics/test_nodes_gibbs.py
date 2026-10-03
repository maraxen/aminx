# ruff: noqa: S101
"""Spec §4.4: the Gibbs sweep builds its own forward context, never the AR one.

``optimization_mode="nodes"`` reuses the decoder row kernel, but it must NOT
reuse the autoregressive ``h_EXV_fw``. Upstream recomputes it from a mask that
is the SELF SLOT ONLY (``sample_seqs.py:346``, ``run_utils.py:187-191``), and
``refine.py:386`` does the same. The spec states the obligation as a negative
control -- "AR-order ``h_EXV_fw`` must fail" -- and that control did not exist:
``test_refine_modes_and_converge_stop`` (``test_sample.py:253``) checks that
``nodes`` refines and that ``mask_bw[:, 0]`` is zero, but never compares against
the AR context it is forbidden to inherit.

THE TWO MASKS DIVERGE IN TWO INDEPENDENT WAYS, and a test that caught only one
would miss the other:

1. SUPPORT. Gibbs ``mask_fw`` is the self slot alone, whatever the sequence
   position. The AR ``mask_fw`` is ``1 - (rank[nbr] < rank[self])``, so for any
   order that is not strictly N-to-C it puts weight on non-self slots -- the
   not-yet-decoded neighbours whose identities the Gibbs sweep must not see.
2. THE MULTIPLICAND. Gibbs uses ``m = present·chain_M_pos``
   (``refine.py:78``); AR uses ``m = present`` alone, which
   ``forward_context``'s own docstring calls out (``decode.py:195``). A FIXED
   position is therefore silent in the Gibbs context and live in the AR one.

Support is read off the returned ``h_EXV_fw`` rather than off an intermediate:
with ``h_v`` and ``h_e`` all ones, every slot of ``h_EXV`` is nonzero, so the
support of ``h_EXV_fw`` is exactly the support of the mask that scaled it. That
tests the array the row kernel actually consumes.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from aminx.families.potts_mpnn.decode import forward_context
from aminx.families.potts_mpnn.refine import nodes_attention

_L = 3
_K = 3
_HIDDEN = 4

# Slot 0 is the self edge, by the convention both masks are written against.
_E_IDX = jnp.asarray(np.asarray([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int32))

# Not N-to-C: residue 1 is decoded first, then 2, then 0. Under a strictly
# ascending rank every AR forward slot would be a later neighbour and the
# support comparison below could pass for the wrong reason.
_RANK = jnp.asarray(np.asarray([2, 0, 1], dtype=np.int32))

_ONES_V = jnp.ones((_L, _HIDDEN))
_ONES_E = jnp.ones((_L, _K, _HIDDEN))


def _support(h_exv_fw: jnp.ndarray) -> np.ndarray:
  """Which (row, slot) cells survived the forward mask."""
  return np.asarray(jnp.sum(jnp.abs(h_exv_fw), axis=-1) > 0)


def test_knob_semantics_nodes_gibbs() -> None:
  """Gibbs forward context is the self slot, and the AR context is not."""
  present = jnp.ones(_L)
  chain_m_pos = jnp.ones(_L)

  mask_bw, mask_fw = nodes_attention(present, chain_m_pos, _K)
  gibbs_support = np.asarray(np.asarray(mask_fw) > 0)

  # 1. Gibbs: the self slot, every row, nothing else.
  expected = np.zeros((_L, _K), dtype=bool)
  expected[:, 0] = True
  assert np.array_equal(gibbs_support, expected), mask_fw

  # The two masks partition m exactly -- no slot is both attended and forward,
  # and none is neither, which is what makes "self slot only" a real split
  # rather than a forward mask that happens to be small.
  np.testing.assert_allclose(
    np.asarray(mask_bw + mask_fw),
    np.broadcast_to(np.asarray(present * chain_m_pos)[:, None], (_L, _K)),
    rtol=0, atol=0,
  )

  # 2. NEGATIVE CONTROL: the AR forward context is a different array, and the
  # difference is exactly the non-self neighbours the Gibbs sweep must not read.
  _ar_bw, ar_h_exv_fw = forward_context(_ONES_V, _ONES_E, _E_IDX, _RANK, present)
  ar_support = _support(ar_h_exv_fw)
  assert ar_support[:, 1:].any(), (
    "the AR forward context must reach non-self slots, or this fixture cannot "
    f"distinguish it from the Gibbs one: {ar_support}"
  )
  assert not np.array_equal(ar_support, gibbs_support)


def test_nodes_gibbs_masks_fixed_positions_where_the_ar_context_does_not() -> None:
  """``m = present·chain_M_pos`` for Gibbs, ``m = present`` for AR.

  The second, independent divergence. A fixed position (``chain_M_pos = 0``)
  contributes nothing to the Gibbs forward context and still contributes to the
  AR one. Swapping in the AR context would therefore leak a fixed residue's
  node features into the sweep even if the slot pattern happened to agree.
  """
  present = jnp.ones(_L)
  chain_m_pos = jnp.asarray(np.asarray([1.0, 0.0, 1.0]))

  _bw, mask_fw = nodes_attention(present, chain_m_pos, _K)
  # Row 1 is fixed, so its whole forward row is dead -- self slot included.
  assert not np.asarray(np.asarray(mask_fw)[1] > 0).any()
  assert np.asarray(mask_fw)[0, 0] > 0
  assert np.asarray(mask_fw)[2, 0] > 0

  # The AR context never sees chain_m_pos at all: it takes `present` only, so
  # row 1 stays live. That asymmetry is the thing being pinned.
  _ar_bw, ar_h_exv_fw = forward_context(_ONES_V, _ONES_E, _E_IDX, _RANK, present)
  assert _support(ar_h_exv_fw)[1].any(), (
    "forward_context must ignore chain_m_pos (decode.py:195); if this fails the "
    "two masks have been unified and the Gibbs divergence is gone"
  )


def test_nodes_gibbs_forward_mask_is_independent_of_decoding_order() -> None:
  """CONTROL: ``nodes_attention`` takes no order, so no order can change it.

  Stated as a test because the tempting refactor is to give the Gibbs sweep the
  decoding order "for consistency". Then the two contexts would agree for a
  N-to-C order and silently diverge for every other one -- a bug that only the
  non-trivial fixtures above would ever surface.
  """
  present = jnp.ones(_L)
  chain_m_pos = jnp.ones(_L)
  first = nodes_attention(present, chain_m_pos, _K)

  # Every AR order gives a different AR forward context...
  supports = set()
  for rank in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
    _bw, h_exv_fw = forward_context(
      _ONES_V, _ONES_E, _E_IDX,
      jnp.asarray(np.asarray(rank, dtype=np.int32)),
      present,
    )
    supports.add(_support(h_exv_fw).tobytes())
  assert len(supports) > 1, "the AR context must depend on the order, or the control is vacuous"

  # ...while the Gibbs one is a function of (present, chain_M_pos, K) alone.
  second = nodes_attention(present, chain_m_pos, _K)
  np.testing.assert_array_equal(np.asarray(first[1]), np.asarray(second[1]))
