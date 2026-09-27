"""Export-time length bucketing (D-C).

Padding a variable-length structure up to a small, fixed set of bucket sizes bounds
recompilation for a downstream export target (ONNX/IREE) the same way
``aminx.tiling.bucketing`` bounds it for inference: a handful of static shapes instead of
one per distinct length. This module is deliberately separate from both
``aminx.tiling.bucketing`` (inference-time batch grouping) and ``aminx.tiling.buckets``
(the ``LENGTH_BUCKETS`` inference-padding ladder, V7) -- the export ladder is its own
pinned set (``EXPORT_BUCKETS``), and the two must not be conflated (ODQ-B1).
"""

from __future__ import annotations

import numpy as np
from xtrax.tiling import select_bucket as _xtrax_select_bucket

from aminx.tiling.errors import TilingError

#: Pinned export bucket ladder (D-C). Distinct from ``tiling/bucketing.py``'s
#: ``BucketingConfig`` default and from ``tiling/buckets.py``'s ``LENGTH_BUCKETS`` --
#: neither is changed by this module (ODQ-B1).
EXPORT_BUCKETS: tuple[int, ...] = (128, 256, 512, 1024)


class ExportLengthError(TilingError):
  """Base for length-refusal errors raised by ``select_export_bucket``."""


class LengthBelowNeighborsError(ExportLengthError):
  """Raised when a structure's real length is below the checkpoint's ``k_neighbors``.

  A structure shorter than ``k_neighbors`` cannot fill its own neighbor slots (every
  residue would need more neighbors than there are other residues), so it is refused
  rather than silently clamped.
  """


class LengthAboveMaxBucketError(ExportLengthError):
  """Raised when a structure's real length exceeds the largest export bucket."""


def select_export_bucket(
  n_real: int,
  k_neighbors: int,
  buckets: tuple[int, ...] = EXPORT_BUCKETS,
) -> int:
  """Select the smallest export bucket that fits ``n_real``, or refuse.

  Args:
    n_real: Real (unpadded) residue count of the structure to export.
    k_neighbors: The checkpoint's ``k_neighbors`` (V10).
    buckets: Sorted, ascending export bucket ceilings. Defaults to ``EXPORT_BUCKETS``.

  Returns:
    The smallest bucket in ``buckets`` that is ``>= n_real``.

  Raises:
    LengthBelowNeighborsError: If ``n_real < k_neighbors``.
    LengthAboveMaxBucketError: If ``n_real > buckets[-1]``.
  """
  if n_real < k_neighbors:
    msg = (
      f"n_real={n_real} is below k_neighbors={k_neighbors}; a structure this short "
      "cannot fill its own neighbor slots."
    )
    raise LengthBelowNeighborsError(msg)
  if n_real > buckets[-1]:
    msg = f"n_real={n_real} exceeds the largest export bucket {buckets[-1]}."
    raise LengthAboveMaxBucketError(msg)
  # Selection itself is delegated to xtrax.tiling.select_bucket (D-C); the two guards
  # above give this module's own named refusals instead of a bare ValueError, and are
  # never triggered by the call below (n_real is already known to be in [k_neighbors,
  # buckets[-1]] here).
  return _xtrax_select_bucket(n_real, boundaries=buckets)


def pad_inputs(
  x4: np.ndarray,
  mask: np.ndarray,
  residue_index: np.ndarray,
  chain_index: np.ndarray,
  bucket: int,
) -> dict[str, np.ndarray]:
  """Host-side (NumPy) padding of export inputs up to ``bucket``.

  Padding convention (from ``jax2onnx_spike.py:117-119``): coordinates pad with ``0``,
  mask pads with ``0`` (padded rows are invalid), and ``residue_index``/``chain_index``
  pad by repeating the last real value.

  Args:
    x4: ``(L, 4, 3)`` real-length backbone coordinates (N, CA, C, O order; D-B).
    mask: ``(L,)`` real-length residue mask.
    residue_index: ``(L,)`` real-length residue indices.
    chain_index: ``(L,)`` real-length chain indices.
    bucket: Target padded length. Must be ``>= L``.

  Returns:
    A dict of ``{"coords", "mask", "residue_index", "chain_index"}``, each padded to
    ``bucket`` on axis 0. ``mask`` is ``float32``; ``residue_index``/``chain_index`` are
    ``int32``.

  Raises:
    ValueError: If ``bucket`` is smaller than the real length.
  """
  x4 = np.asarray(x4)
  mask = np.asarray(mask)
  residue_index = np.asarray(residue_index)
  chain_index = np.asarray(chain_index)

  n_real = x4.shape[0]
  if bucket < n_real:
    msg = f"bucket {bucket} is smaller than n_real {n_real}."
    raise ValueError(msg)
  n_pad = bucket - n_real

  coords_p = np.pad(x4, [(0, n_pad), (0, 0), (0, 0)], constant_values=0.0).astype(np.float32)
  mask_p = np.pad(mask, (0, n_pad), constant_values=0.0).astype(np.float32)

  last_residue = int(residue_index[-1]) if n_real > 0 else 0
  last_chain = int(chain_index[-1]) if n_real > 0 else 0
  residue_index_p = np.pad(residue_index, (0, n_pad), constant_values=last_residue).astype(
    np.int32,
  )
  chain_index_p = np.pad(chain_index, (0, n_pad), constant_values=last_chain).astype(np.int32)

  return {
    "coords": coords_p,
    "mask": mask_p,
    "residue_index": residue_index_p,
    "chain_index": chain_index_p,
  }
