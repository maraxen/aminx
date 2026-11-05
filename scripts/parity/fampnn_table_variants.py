"""Swap aminx's two host-side sinusoidal tables for upstream's own torch float32 values (parity diagnostics).

aminx forms its positional table (``features.positional_embedding_table``) and its timestep-frequency table
(``features.timestep_frequency_table``) on the host as correctly rounded float32 values. Upstream's values come
from torch's float32 ``cos``/``sin``/``exp`` kernels, which differ from correctly rounded by one ulp on a few
percent of entries and cannot be reproduced without torch. ``install`` replaces both providers with arrays captured
from the real upstream functions (``dump_fampnn_oracle_f64tables.py`` writes them to ``diag_tables.npz``), so the
"identity" arm of the parity vehicle isolates everything except those kernel roundings.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from aminx.families.fampnn import gvp, sidechain_diffusion


@contextmanager
def installed(variant: str, tabs: dict[str, np.ndarray]) -> Iterator[None]:
  """Context manager: patch the table providers for ``variant`` in {"torch_f32", "f64"}, restore on exit.

  Parameters
  ----------
  variant : str
    ``torch_f32`` uses upstream's torch float32 tables; ``f64`` uses float64 frequencies and angles.
  tabs : dict[str, np.ndarray]
    Arrays of ``diag_tables.npz`` (``pos_f32_torch``, ``pos_f64``, ``ts_freq_f32_torch``).

  Yields
  ------
  None
    Control returns to the caller with the providers patched.
  """
  half_ts = 128
  if variant == "torch_f32":
    pos, ts = tabs["pos_f32_torch"], tabs["ts_freq_f32_torch"]
  elif variant == "f64":
    pos = tabs["pos_f64"]
    ts = np.exp(-math.log(10000.0) * np.arange(0, half_ts, dtype=np.float64) / half_ts)
  else:
    msg = f"unknown table variant {variant}"
    raise KeyError(msg)
  centre = (pos.shape[0] + 1) // 2 - 1

  def pos_table(num_res: int, num_embeddings: int = 16) -> np.ndarray:
    """Rows for offsets ``-(num_res-1)..(num_res-1)`` of the captured upstream table."""
    if num_embeddings != 16 or num_res - 1 > centre:  # noqa: PLR2004
      msg = f"captured table covers 16 embeddings and |offset| <= {centre}"
      raise ValueError(msg)
    return np.asarray(pos[centre - (num_res - 1) : centre + num_res])

  def ts_table(half: int, max_period: float = 10000.0) -> Any:  # noqa: ANN401, ARG001
    """First ``half`` captured timestep frequencies."""
    return np.asarray(ts)[:half]

  old = (gvp.positional_embedding_table, sidechain_diffusion.timestep_frequency_table)
  gvp.positional_embedding_table = pos_table
  sidechain_diffusion.timestep_frequency_table = ts_table
  try:
    yield
  finally:
    gvp.positional_embedding_table, sidechain_diffusion.timestep_frequency_table = old
