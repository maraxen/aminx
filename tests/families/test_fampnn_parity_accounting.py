"""Verdict accounting of the FAMPNN parity vehicle: non-finite and shape-mismatched values never pass or trip.

Pure-numpy checks of ``Ledger``; no weights or oracle dumps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PARITY = Path(__file__).resolve().parents[2] / "scripts" / "parity"
sys.path.insert(0, str(_PARITY))

from fampnn_parity import Ledger, _n_kind, _n_over  # noqa: E402


def test_nonfinite_is_its_own_kind_and_not_over_band() -> None:
  led = Ledger({})
  led.cont("f64", "enc", "k", np.array([1.0, np.nan]), np.array([1.0, 1.0]))
  led.cont("f64", "enc", "k2", np.array([np.inf]), np.array([np.inf]))
  assert _n_kind(led.rows, "f64", "nonfinite") == 2
  assert _n_over(led.rows, "f64") == 0  # counted separately, not as a finite over-band diff
  assert not [r for r in led.rows if r["kind"] == "cont"]  # so a mutant maximum over cont rows never sees it


def test_shape_mismatch_is_counted_for_cont_and_disc() -> None:
  led = Ledger({})
  led.cont("f32", "enc", "a", np.zeros((2, 3)), np.zeros((3, 2)))
  led.disc("f32", "parse", "b", np.zeros(4, dtype=int), np.zeros(5, dtype=int))
  assert _n_kind(led.rows, "f32", "shape") == 2
  assert _n_kind(led.rows, "f64", "shape") == 0
  assert not [r for r in led.rows if r["kind"] == "disc"]  # a shape mismatch is not a discrete flip


def test_finite_over_band_and_discrete_flip_still_count() -> None:
  led = Ledger({})
  dev = led.cont("f64", "enc", "k", np.array([1.0]), np.array([1.5]))
  n = led.disc("f64", "parse", "d", np.array([1, 2]), np.array([1, 3]))
  assert dev == 0.5
  assert n == 1
  assert _n_over(led.rows, "f64") == 1
