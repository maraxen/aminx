"""Compile-cache provenance for campaign telemetry (debt #3510).

Per-unit compile counts come from ``xtrax.profiling.count_backend_compiles``. This module only
reports which persistent compilation cache, if any, was in effect, for done markers.

task_id: 261119_aminx-3510-compile-shape-classes
"""

from __future__ import annotations

from typing import Any


def jax_compilation_cache_setting() -> dict[str, Any]:
  """The persistent compilation cache setting in effect, as JAX reports it.

  The directory is read from the live ``jax.config`` flag, which ``xtrax.jax_cache.configure``
  sets at each entry point. ``source`` is xtrax's record of the layer that chose it when that
  record still matches the live flag; otherwise the flag was changed outside xtrax and the
  source says so.

  Returns
  -------
  dict[str, Any]
    ``{"enabled", "dir", "source"}``; ``dir`` is ``None`` when no cache directory is set.

  """
  import jax  # noqa: PLC0415  -- keep importing this module free of jax
  from xtrax import jax_cache  # noqa: PLC0415

  live = getattr(jax.config, "jax_compilation_cache_dir", None)
  live_dir = str(live) if live else None

  applied = jax_cache.current()
  applied_dir = str(applied.dir) if applied is not None and applied.dir else None
  if applied is not None and applied_dir == live_dir:
    source = applied.source
  else:
    source = "jax.config (not set by xtrax jax_cache)"
  return {"enabled": bool(live_dir), "dir": live_dir, "source": source}
