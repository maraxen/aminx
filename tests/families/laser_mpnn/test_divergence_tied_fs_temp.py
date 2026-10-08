# ruff: noqa: S101, SLF001
"""Spec §6.5b: tied rejects the knobs it cannot honour, where upstream NameErrors.

``utils/model.py:485`` references an undefined ``batch`` on the tied path, so
upstream's ``run_inference_tied.py:607-616`` dies with ``NameError: name 'batch'
is not defined`` the moment ``fs_sequence_temp`` is set -- a traceback that
names a local variable and gives the caller nothing to act on.

``_refuse_tied_knobs`` (``model/laser/tied.py:82``) raises a ``ValueError``
naming the FLAG instead. Four more knobs share the property of not being
``tied_sample`` parameters at all, and are refused the same way rather than
being accepted and ignored -- which is the failure mode the
``options_plumbing_allowlist`` exists to track elsewhere.

Tested through the private name deliberately: it is the whole guard, its one
caller passes five keyword arguments straight through, and a test routed
through ``tied_sample`` would need a model and would still only reach this
function.

Every call below spells its five arguments out. A ``**defaults`` dict would be
shorter and would collapse the values to one union type, which costs the
keyword-level type checking this file otherwise gets for free (17 ty errors,
measured).
"""

from __future__ import annotations

import pytest

from aminx.model.laser.tied import _refuse_tied_knobs


def test_divergence_tied_fs_temp() -> None:
  """``fs_sequence_temp`` is refused by name, not by NameError."""
  with pytest.raises(ValueError, match="fs_sequence_temp") as caught:
    _refuse_tied_knobs(
      seq_min_p=0.0,
      chi_min_p=0.0,
      fs_sequence_temp=0.5,
      disable_charged_fs=False,
      ignore_chain_mask_zeros=False,
    )
  # The point of the divergence is that the message is actionable, so the flag
  # name has to be IN it -- not merely the reason the call failed.
  assert "tied_second_input rejects" in str(caught.value)


def test_fs_sequence_temp_zero_is_a_value_not_an_unset() -> None:
  """``0.0`` still raises; only ``None`` means unset.

  The guard tests ``is not None`` rather than truthiness. If that slipped to
  ``if fs_sequence_temp:`` then ``--fs_sequence_temp 0`` would sail through and
  reach upstream's NameError by the back door -- the same ``'0'``-is-truthy
  class of bug the temperature-zero divergence is about.
  """
  with pytest.raises(ValueError, match="fs_sequence_temp"):
    _refuse_tied_knobs(
      seq_min_p=0.0,
      chi_min_p=0.0,
      fs_sequence_temp=0.0,
      disable_charged_fs=False,
      ignore_chain_mask_zeros=False,
    )


def test_seq_min_p_is_refused_by_name() -> None:
  """Not a ``tied_sample`` parameter, so setting it is an error, not a no-op."""
  with pytest.raises(ValueError, match="seq_min_p"):
    _refuse_tied_knobs(
      seq_min_p=0.1,
      chi_min_p=0.0,
      fs_sequence_temp=None,
      disable_charged_fs=False,
      ignore_chain_mask_zeros=False,
    )


def test_chi_min_p_is_refused_by_name() -> None:
  """The χ sibling of ``seq_min_p``, refused for the same reason."""
  with pytest.raises(ValueError, match="chi_min_p"):
    _refuse_tied_knobs(
      seq_min_p=0.0,
      chi_min_p=0.1,
      fs_sequence_temp=None,
      disable_charged_fs=False,
      ignore_chain_mask_zeros=False,
    )


def test_disable_charged_fs_is_refused_by_name() -> None:
  """A bool knob: ``True`` is the set state, ``False`` the default."""
  with pytest.raises(ValueError, match="disable_charged_fs"):
    _refuse_tied_knobs(
      seq_min_p=0.0,
      chi_min_p=0.0,
      fs_sequence_temp=None,
      disable_charged_fs=True,
      ignore_chain_mask_zeros=False,
    )


def test_ignore_chain_mask_zeros_is_refused_by_name() -> None:
  """The fifth and last knob the tied path cannot honour."""
  with pytest.raises(ValueError, match="ignore_chain_mask_zeros"):
    _refuse_tied_knobs(
      seq_min_p=0.0,
      chi_min_p=0.0,
      fs_sequence_temp=None,
      disable_charged_fs=False,
      ignore_chain_mask_zeros=True,
    )


def test_tied_knob_refusal_is_silent_at_defaults() -> None:
  """CONTROL: the guard does not simply always raise.

  Without this, every assertion above is satisfied by ``raise ValueError(...)``
  on the first line, which would make the tied path unusable rather than
  strict. These are the values a caller who set none of these would send.
  """
  _refuse_tied_knobs(
    seq_min_p=0.0,
    chi_min_p=0.0,
    fs_sequence_temp=None,
    disable_charged_fs=False,
    ignore_chain_mask_zeros=False,
  )


def test_all_flagged_knobs_are_reported_at_once() -> None:
  """Every offending flag is listed, so a caller fixes them in one pass.

  Reporting only the first would make five knobs take five round trips.
  """
  with pytest.raises(ValueError) as caught:
    _refuse_tied_knobs(
      seq_min_p=0.1,
      chi_min_p=0.2,
      fs_sequence_temp=0.3,
      disable_charged_fs=True,
      ignore_chain_mask_zeros=True,
    )
  message = str(caught.value)
  for knob in (
    "seq_min_p",
    "chi_min_p",
    "fs_sequence_temp",
    "disable_charged_fs",
    "ignore_chain_mask_zeros",
  ):
    assert knob in message, f"{knob} was set but not named in {message!r}"
