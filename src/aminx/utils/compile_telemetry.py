"""Compile-cache provenance for campaign telemetry (debt #3510).

Per-unit compile counts come from ``xtrax.profiling.count_backend_compiles``. This module only
reports which persistent compilation cache, if any, was in effect, for done markers.

task_id: 261119_aminx-3510-compile-shape-classes
"""

from __future__ import annotations

from typing import Any


def jax_compilation_cache_setting() -> dict[str, Any]:
  """The persistent compilation cache setting in effect, as JAX reports it.

  aminx does not configure the cache. TODO(xtrax debt #3511): once
  ``xtrax.jax_cache.configure`` ships, call it at the runner/campaign entry points and record the
  ``CacheSetting(dir, source, enabled)`` it returns here instead of reading ``jax.config``.

  Returns
  -------
  dict[str, Any]
    ``{"enabled", "dir", "source"}``; ``dir`` is ``None`` when no cache directory is set.

  """
  import jax  # noqa: PLC0415  -- keep importing this module free of jax

  cache_dir = getattr(jax.config, "jax_compilation_cache_dir", None)
  return {
    "enabled": bool(cache_dir),
    "dir": str(cache_dir) if cache_dir else None,
    "source": "jax.config (not configured by aminx; pending xtrax debt #3511)",
  }
