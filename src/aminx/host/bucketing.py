"""Host-side residue span and xtrax bucket-rung selection.

Rung selection is lazy: importing this module does not import ``xtrax.export``
or the ONNX stack. Callers that only need :func:`batch_span` stay off that path.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

# Every residue-axis array the sample trim slices, in one place (S8 risk: a missed
# field silently misaligns). The integer is the residue axis.
SAMPLE_RESIDUE_AXES: tuple[tuple[str, int], ...] = (
  ("coords", 1),
  ("mask", 1),
  ("residue_index", 1),
  ("chain_index", 1),
  ("structure_mapping", 1),
  ("fixed_mask", 1),
  ("fixed_tokens", 1),
  ("tie_group_map", 1),
  ("state_position_map", -1),  # (B, S, L): residue axis is last
  ("Y", 1),
  ("Y_t", 1),
  ("Y_m", 1),
  ("atom_37", 1),
  ("atom_37_mask", 1),
  ("chain_mask", 1),
  ("bias", 0),
)


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


def residue_axis(name: str) -> int:
  """Return the residue axis of a sample field listed in :data:`SAMPLE_RESIDUE_AXES`.

  Parameters
  ----------
  name : str
    Field name (``coords``, ``bias``, ``state_position_map``, ligand ``Y``, ...).

  Returns
  -------
  int
    Axis index passed to :func:`trim_residue_axis`.

  Raises
  ------
  KeyError
    If ``name`` is not in the residue-axis list.

  """
  for field, axis in SAMPLE_RESIDUE_AXES:
    if field == name:
      return axis
  msg = f"{name} is not a residue-axis sample field"
  raise KeyError(msg)


def _is_tracer(value: object) -> bool:
  """True when ``value`` is a JAX tracer (host span cannot be read)."""
  try:
    import jax  # noqa: PLC0415
  except ImportError:
    return False
  return isinstance(value, jax.core.Tracer)


def _residue_length(arr: ArrayLike, *, last_axis: bool) -> int:
  shape = np.shape(arr)
  if len(shape) == 0:
    return -1
  return int(shape[-1] if last_axis else shape[0])


def sample_rung(
  mask: ArrayLike | None,
  seq_len: int,
  *,
  max_length: int | None,
  enabled: bool,
  pass_mode: str = "intra",  # noqa: S107
  bias: ArrayLike | None = None,
  tie_group_map: ArrayLike | None = None,
  state_position_map: ArrayLike | None = None,
  structure_mapping: ArrayLike | None = None,
) -> int | None:
  """Choose the sample rung for one batch, or ``None`` when the batch must not be trimmed.

  Skip guards run only after a rung was selected. Each one logs a single INFO line
  naming the reason and forces ``None``, so the padded-length validation path
  (including ``BiasLengthError``) stays the one that runs.

  Parameters
  ----------
  mask : array_like or None
    Batch mask, shape ``(L,)`` or ``(batch, L)``. A JAX tracer is not trimmed.
  seq_len : int
    Padded residue length (``coordinates.shape[1]``).
  max_length : int or None
    ``RunSpecification.max_length``. ``None`` means no trim.
  enabled : bool
    ``RunSpecification.length_bucketing``.
  pass_mode : str, optional
    ``"inter"`` is not bucketed.
  bias, tie_group_map, state_position_map, structure_mapping : array_like or None
    Spec arrays checked by the skip guards. Read only when a rung would otherwise
    be returned.

  Returns
  -------
  int or None
    The rung, or ``None`` when there is nothing to trim or a skip guard fired.

  """
  # Opt-out, missing max_length, and inter return before the mask is read.
  if max_length is None or not enabled or pass_mode == "inter":  # noqa: S105
    return None
  if mask is None or _is_tracer(mask):
    return None
  rung = rung_for(
    batch_span(mask),
    seq_len,
    enabled=enabled,
    pass_mode=pass_mode,
  )
  if rung is None or rung >= seq_len:
    return None

  blocked = False
  if bias is not None and _residue_length(bias, last_axis=False) != seq_len:
    logger.info(
      "length bucketing skipped: user bias leading axis %s != padded length %s",
      _residue_length(bias, last_axis=False),
      seq_len,
    )
    blocked = True
  if tie_group_map is not None:
    tie = np.asarray(tie_group_map)
    tie_len = _residue_length(tie, last_axis=False)
    if tie_len != seq_len:
      logger.info(
        "length bucketing skipped: tie_group_map length %s != padded length %s",
        tie_len,
        seq_len,
      )
      blocked = True
    if tie.size and int(np.max(tie)) >= rung:
      logger.info(
        "length bucketing skipped: tie_group_map id %s >= rung %s",
        int(np.max(tie)),
        rung,
      )
      blocked = True
  if state_position_map is not None:
    spm = np.asarray(state_position_map)
    spm_len = _residue_length(spm, last_axis=True)
    if spm_len != seq_len:
      logger.info(
        "length bucketing skipped: state_position_map last axis %s != padded length %s",
        spm_len,
        seq_len,
      )
      blocked = True
    if spm.size and bool(np.any(spm >= rung)):
      logger.info(
        "length bucketing skipped: state_position_map value >= rung %s",
        rung,
      )
      blocked = True
  if structure_mapping is not None:
    mapping_len = _residue_length(structure_mapping, last_axis=True)
    if mapping_len != seq_len:
      logger.info(
        "length bucketing skipped: structure_mapping last axis %s != padded length %s",
        mapping_len,
        seq_len,
      )
      blocked = True
  if blocked:
    return None
  return rung


def trim_residue_axis(arr: Any, rung: int, axis: int) -> Any:  # noqa: ANN401
  """Slice ``arr`` to ``rung`` along ``axis``. ``None`` passes through.

  Parameters
  ----------
  arr : array_like or None
    Array whose residue axis is trimmed. ``None`` is returned unchanged.
  rung : int
    Exclusive end index on ``axis``.
  axis : int
    Residue axis. Negative axes count from the end.

  Returns
  -------
  array_like or None
    A view (or ``None``). The object type is unchanged.

  """
  if arr is None:
    return None
  axis_norm = axis if axis >= 0 else arr.ndim + axis
  slices: list[slice] = [slice(None)] * arr.ndim
  slices[axis_norm] = slice(0, rung)
  return arr[tuple(slices)]


def repad_residue_axis(arr: Any, padded_len: int, axis: int) -> Any:  # noqa: ANN401
  """Pad ``arr`` with 0 along ``axis`` out to ``padded_len``. ``None`` passes through.

  Parameters
  ----------
  arr : array_like or None
    Array to re-pad. ``None`` is returned unchanged.
  padded_len : int
    Length of ``axis`` after padding.
  axis : int
    Residue axis. Negative axes count from the end.

  Returns
  -------
  array_like or None
    ``arr`` when that axis is already ``padded_len``, otherwise a 0-padded copy.
    NumPy arrays stay NumPy; other arrays are padded with ``jax.numpy.pad``.

  """
  if arr is None:
    return None
  axis_norm = axis if axis >= 0 else arr.ndim + axis
  current = int(arr.shape[axis_norm])
  if current == padded_len:
    return arr
  if current > padded_len:
    msg = f"cannot re-pad axis {axis} from {current} to {padded_len}"
    raise ValueError(msg)
  pad_width = [(0, 0)] * arr.ndim
  pad_width[axis_norm] = (0, padded_len - current)
  if isinstance(arr, np.ndarray):
    return np.pad(arr, pad_width, mode="constant", constant_values=0)
  import jax.numpy as jnp  # noqa: PLC0415

  return jnp.pad(arr, pad_width, constant_values=0)
