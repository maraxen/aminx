"""Case iteration shared by the implemented A1 port waves.

``knn_boundary_tie`` fixtures are dropped. Etabs are compared on edges whose
endpoints are both present.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import jax
import numpy as np
import pytest

from aminx.families.potts_mpnn.featurize import knn_boundary_tie


def names(values: np.ndarray) -> list[str]:
  return [str(item) for item in values.tolist()]


def open_dump(oracle: Any, precision: str) -> np.lib.npyio.NpzFile:
  """Load a sealed dump, skipping when the oracle directory is absent."""
  try:
    return oracle.load(precision)  # type: ignore[no-any-return]
  except oracle.OracleAbsentError as exc:
    pytest.skip(str(exc))


def x64_context(precision: str) -> contextlib.AbstractContextManager[None]:
  """Enable x64 for f64 tiers. f32 keeps the process default."""
  if precision != "f64":
    return contextlib.nullcontext()
  enable = getattr(jax.experimental, "enable_x64", None)
  if enable is None:
    enable = jax.enable_x64
  return enable()


def iter_cases(data: np.lib.npyio.NpzFile, *, require_edges: bool) -> Iterator[tuple[Any, ...]]:
  """Yield checkpoint, fixture, prefix, mask, e_idx, and the kept-edge mask."""
  for checkpoint in names(data["checkpoint_ids"]):
    for fixture in names(data["fixture_names"]):
      prefix = f"{checkpoint}__{fixture}__"
      mask = data[prefix + "mask"]
      e_idx = data[prefix + "E_idx"]
      present = np.asarray(mask[0])
      if knn_boundary_tie(present, int(present.shape[0])):
        continue
      if require_edges:
        neighbours = np.asarray(e_idx[0])
        keep = (present > 0)[:, None] & (present[neighbours] > 0)
        if not bool(np.any(keep)):
          continue
      else:
        keep = np.ones((0,), dtype=bool)
      yield checkpoint, fixture, prefix, mask, e_idx, keep


def assert_shape_dtype(got: np.ndarray, ref: np.ndarray, label: str) -> None:
  assert got.shape == ref.shape, f"{label}: {got.shape} vs {ref.shape}"
  assert got.dtype == ref.dtype, f"{label}: {got.dtype} vs {ref.dtype}"


def assert_close(
  got: np.ndarray,
  ref: np.ndarray,
  *,
  rtol: float,
  atol: float,
  label: str,
) -> None:
  assert_shape_dtype(got, ref, label)
  np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol, err_msg=label)


def assert_energy_f32(got: np.ndarray, ref: np.ndarray, scale: np.ndarray, label: str) -> None:
  """r15: ``|err| <= 1e-5 * sum|terms|`` with ``sum|terms| = potts_energy(|etab|)``."""
  assert_shape_dtype(got, ref, label)
  error = np.abs(got.astype(np.float64) - ref.astype(np.float64))
  bound = 1e-5 * np.abs(scale.astype(np.float64))
  worst = int(np.argmax(error - bound))
  assert np.all(error <= bound), (
    f"{label}: |got-ref| {error.flat[worst]:.3e} > {bound.flat[worst]:.3e} (rtol*sum|terms|)"
  )
