"""Host-side residue span and xtrax bucket-rung selection.

Rung selection is lazy: importing this module does not import ``xtrax.export``
or the ONNX stack. Callers that only need :func:`batch_span` stay off that path.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def batch_span(mask: ArrayLike) -> int:
  """Return the residue span of a host mask.

  Parameters
  ----------
  mask : array_like
    Boolean or numeric mask of shape ``(L,)`` or ``(batch, L)``. Nonzero
    entries are valid residues. Extra leading axes are flattened into the
    batch, matching :class:`aminx.host.runner._PaddingCheck`.

  Returns
  -------
  int
    Max over rows of (index of the last valid residue + 1). ``0`` when every
    row is masked out or the length axis is empty.

  """
  mask_np = np.asarray(mask)
  if mask_np.ndim == 0 or mask_np.size == 0 or mask_np.shape[-1] == 0:
    return 0
  padded = int(mask_np.shape[-1])
  rows = mask_np.reshape(-1, padded) if mask_np.ndim != 1 else mask_np.reshape(1, padded)
  valid = rows > 0
  if valid.size == 0 or not bool(valid.any()):
    return 0
  # span per row = index of the last valid residue + 1 (0 for an all-masked row)
  last_from_end = np.argmax(valid[:, ::-1], axis=-1)
  span = np.where(valid.any(axis=-1), padded - last_from_end, 0)
  return int(span.max())


def bucket_ladder() -> tuple[int, ...]:
  """Return xtrax's export bucket ladder.

  Returns
  -------
  tuple[int, ...]
    ``tuple(xtrax.export.rings.BUCKET_LADDER)``. The rings import is local so
    importing this module does not load the export stack.

  """
  from xtrax.export.rings import BUCKET_LADDER  # noqa: PLC0415

  return tuple(BUCKET_LADDER)


def rung_for(
  span: int,
  padded_len: int | None,
  *,
  enabled: bool,
  pass_mode: str = "intra",  # noqa: S107
) -> int | None:
  """Pick the bucket rung for one batch, or ``None`` when the batch should not be trimmed.

  Parameters
  ----------
  span : int
    Residue span from :func:`batch_span` (last valid index + 1).
  padded_len : int or None
    Loader padded length. ``None`` means no trim.
  enabled : bool
    ``RunSpecification.length_bucketing``. ``False`` restores fixed padding.
  pass_mode : str, optional
    ``"inter"`` is not bucketed. Default is ``"intra"``.

  Returns
  -------
  int or None
    ``None`` when bucketing is disabled, ``padded_len`` is ``None``,
    ``pass_mode`` is ``"inter"``, ``span`` is not positive, or ``span``
    exceeds the last ladder rung. Otherwise
    ``min(select_bucket(span, boundaries=ladder), padded_len)``.

  """
  if not enabled or padded_len is None or pass_mode == "inter" or span <= 0:  # noqa: S105
    return None
  ladder = bucket_ladder()
  if span > ladder[-1]:
    return None
  from xtrax.tiling import select_bucket  # noqa: PLC0415

  return min(select_bucket(span, boundaries=ladder), padded_len)
