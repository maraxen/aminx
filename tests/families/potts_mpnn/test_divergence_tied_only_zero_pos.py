# ruff: noqa: S101
"""Spec §6.5b: an all-non-interface tied group raises, where upstream divides by zero.

``run_utils.py:401`` ends the tied refine non-epistasis path with
``predicted_E /= num_pos``, where ``num_pos`` counts the group's interface
residues. Under ``binding_energy_optimization="only"`` a group every member of
which is non-interface makes ``num_pos`` exactly ``0``, and upstream dies with
``ZeroDivisionError`` from inside a sweep -- a traceback that names the
arithmetic and not the group that caused it.

``check_tied_only_groups`` (``refine.py:84``) raises a ``ValueError`` naming the
offending group instead. That is the whole divergence, and it had no test:
grepping the name returns its definition, its import, and its one call site at
``sample_host.py:661``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.potts_mpnn.refine import check_tied_only_groups

# Two groups of two, -1 padded: rows 0,1 and rows 2,3.
_GROUPS = jnp.asarray(np.asarray([[0, 1], [2, 3]], dtype=np.int32))


def test_divergence_tied_only_zero_pos() -> None:
  """A group with no interface residue is named, not divided by."""
  # Rows 0,1 are interface; rows 2,3 are not -- so the SECOND group is empty of
  # interface residues and is the one that must be named.
  inter = jnp.asarray(np.asarray([True, True, False, False]))
  with pytest.raises(ValueError, match=r"tied group \[2, 3\] has no interface residue"):
    check_tied_only_groups(
      _GROUPS, inter, tied_epistasis=False, binding="only",
    )


def test_tied_only_zero_pos_guard_is_narrow() -> None:
  """CONTROLS: three ways the guard must stay silent.

  Without these, ``test_divergence_tied_only_zero_pos`` is satisfied by a
  function that raises unconditionally -- which would break every ordinary
  tied refine run rather than the one degenerate case the spec describes.
  """
  none_interface = jnp.asarray(np.asarray([False, False, False, False]))
  all_interface = jnp.asarray(np.asarray([True, True, True, True]))

  # 1. Not the "only" mode: upstream never reaches the division, so neither do we.
  check_tied_only_groups(
    _GROUPS, none_interface, tied_epistasis=False, binding="none",
  )

  # 2. Epistasis path: a different formula upstream, with no /= num_pos.
  check_tied_only_groups(
    _GROUPS, none_interface, tied_epistasis=True, binding="only",
  )

  # 3. Every group has an interface residue, so num_pos is never 0.
  check_tied_only_groups(
    _GROUPS, all_interface, tied_epistasis=False, binding="only",
  )


def test_tied_only_zero_pos_ignores_padding() -> None:
  """``-1`` padding is not a member, and an all-padding row is not an empty group.

  ``build_tie_groups_np`` pads short groups to a power-of-two width with ``-1``.
  Treating those as members would index ``inter_mask`` with ``-1`` -- which in
  numpy silently reads the LAST element rather than raising, so a padded group
  would inherit the interface status of the final residue.
  """
  padded = jnp.asarray(np.asarray([[0, -1], [2, 3]], dtype=np.int32))
  # Row 0's only real member is residue 0, which IS interface, so no raise --
  # even though residue 1 (the value -1 would wrap to, if read as an index into
  # a length-2 mask) is not.
  check_tied_only_groups(
    jnp.asarray(np.asarray([[0, -1]], dtype=np.int32)),
    jnp.asarray(np.asarray([True, False])),
    tied_epistasis=False,
    binding="only",
  )
  # And the second group of `padded` still raises on its own merits.
  with pytest.raises(ValueError, match=r"tied group \[2, 3\]"):
    check_tied_only_groups(
      padded,
      jnp.asarray(np.asarray([True, False, False, False])),
      tied_epistasis=False,
      binding="only",
    )
