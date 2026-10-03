# ruff: noqa: S101
"""Spec §6.5b: the optimize path draws a fresh refine order instead of inheriting one.

``sample_seqs.py:122,178-180,325-330`` reads refine decoding orders from
``out_dir/out_name_decoding_order.json`` -- a file left behind by a PRIOR run,
with no config key naming it and no flag to disable it. So two invocations with
byte-identical arguments refine differently depending on whether something ran
in that directory earlier, and the user is given nothing that says so.

aminx never reads it (``sample_host.py:619-625`` passes
``stored_orders_present=False``), which equals upstream exactly whenever the
file is absent -- the ordinary case -- and diverges only when a prior run left
one behind.

THE GAP THIS FILLS. ``upstream_refine_order`` already had two tests
(``test_sample.py:322,331``), but both pass ``stored_orders_present=True``:
one covers the ``num_samples > 1`` key-mismatch quirk and one the chain-suffix
miss. Neither reaches the ``False`` branch, which is the one the optimize path
actually takes and the one this divergence row is about.
"""

from __future__ import annotations

import numpy as np

from aminx.families.potts_mpnn.sample_host import fresh_refine_order, upstream_refine_order

# A length-4 all-designable chain. The randn magnitudes are deliberately NOT
# ascending, so the fresh key does not coincidentally sort to N-to-C -- if it
# did, every assertion below would hold for an implementation that ignored the
# flag entirely.
_CHAIN = np.asarray([1.0, 1.0, 1.0, 1.0])
_RANDN = np.asarray([0.4, -0.2, 0.9, 0.1])

# What the optimize path at sample_host.py:619 passes: N-to-C, because there is
# no decoded order to inherit on a path that never decoded.
_AR_ORDER = np.arange(4, dtype=np.int32)


def test_divergence_refine_order_json() -> None:
  """``stored_orders_present=False`` draws fresh, and is not the N-to-C sweep."""
  order = upstream_refine_order(
    _AR_ORDER,
    _CHAIN,
    _RANDN,
    num_samples=1,
    chain_suffix="",
    stored_orders_present=False,
  )
  assert np.array_equal(order, fresh_refine_order(_CHAIN, _RANDN))
  # The whole point: these arguments would otherwise return _AR_ORDER verbatim.
  assert not np.array_equal(order, _AR_ORDER)


def test_refine_order_json_control_stored_order_is_honoured() -> None:
  """CONTROL: identical arguments, ``True``, return the stored order unchanged.

  Without this, ``test_divergence_refine_order_json`` is satisfied by a
  function that always draws fresh -- which would throw away the decoded order
  on the ordinary sample-then-refine path (``sample_host.py:564-570``), where
  inheriting it is correct and upstream-matching behaviour.
  """
  order = upstream_refine_order(
    _AR_ORDER,
    _CHAIN,
    _RANDN,
    num_samples=1,
    chain_suffix="",
    stored_orders_present=True,
  )
  assert np.array_equal(order, _AR_ORDER)


def test_refine_order_json_ignores_the_order_it_was_handed() -> None:
  """The ``False`` branch does not read ``ar_order`` at all.

  Pinned because the obvious rewrite -- falling back to ``ar_order`` when no
  stored file is present -- reads identically at the one call site that passes
  ``arange``, and would silently become N-to-C.
  """
  reversed_order = np.asarray([3, 2, 1, 0], dtype=np.int32)
  from_ascending = upstream_refine_order(
    _AR_ORDER, _CHAIN, _RANDN,
    num_samples=1, chain_suffix="", stored_orders_present=False,
  )
  from_reversed = upstream_refine_order(
    reversed_order, _CHAIN, _RANDN,
    num_samples=1, chain_suffix="", stored_orders_present=False,
  )
  assert np.array_equal(from_ascending, from_reversed)


def test_refine_order_json_false_outranks_every_stored_branch() -> None:
  """``False`` short-circuits before both quirks the existing tests cover.

  ``num_samples > 1`` (the ``_i``/``str(i)`` key mismatch) and a non-empty
  chain suffix each change what the ``True`` branch returns. Neither may
  resurrect a stored order once the file is known to be absent.
  """
  fresh = fresh_refine_order(_CHAIN, _RANDN)
  for num_samples in (1, 2):
    for chain_suffix in ("", "_A"):
      order = upstream_refine_order(
        _AR_ORDER, _CHAIN, _RANDN,
        num_samples=num_samples,
        chain_suffix=chain_suffix,
        stored_orders_present=False,
      )
      assert np.array_equal(order, fresh), f"{num_samples=} {chain_suffix=}"
