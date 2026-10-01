"""Caller-attributed deprecation warnings for aminx's own tiling types (aminx debt #2371).

aminx still constructs its own ``SafeMap``/``SafeMapIterator`` internally (``tiling/planner.py``,
``sampling/multistate_poe.py``, ``tiling/dispatch.py``); migrating those to xtrax types is stage 2 of
the debt. Until then a plain ``DeprecationWarning`` in ``__post_init__`` would fire from aminx's own
code on every sample call, at a ``stacklevel`` the user cannot act on. So: warn only when the first
frame *outside* the construction machinery belongs to code that is not part of aminx.
"""

from __future__ import annotations

import sys
import warnings
from types import FrameType

# Frames that are only the mechanics of building the object: this package's two defining modules, the
# dataclass-generated ``__init__`` (compiled from a string), and equinox's module metaclass / __init__.
_MACHINERY_MODULES = frozenset({"aminx.tiling.strategy", "aminx.tiling.iterator", "dataclasses"})


def _is_machinery(frame: FrameType) -> bool:
  module = frame.f_globals.get("__name__", "")
  return (
    module in _MACHINERY_MODULES
    or module.startswith("equinox")
    or frame.f_code.co_filename == "<string>"
  )


def _is_aminx(module: str) -> bool:
  return module == "aminx" or module.startswith("aminx.")


def warn_deprecated(old: str, replacement: str) -> None:
  """Warn that ``old`` is deprecated, attributed to the user code that constructed it.

  Silent when the constructing frame is aminx's own code (see module docstring).
  """
  frame: FrameType | None = sys._getframe(1)  # noqa: SLF001 -- the caller of this helper
  stacklevel = 2
  while frame is not None and _is_machinery(frame):
    frame = frame.f_back
    stacklevel += 1
  if frame is not None and _is_aminx(frame.f_globals.get("__name__", "")):
    return
  warnings.warn(
    f"{old} is deprecated (aminx debt #2371) and will be removed in a future release; use {replacement}.",
    DeprecationWarning,
    stacklevel=stacklevel,
  )
