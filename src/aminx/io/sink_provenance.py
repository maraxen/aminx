"""Shared provenance/schema helpers for every Zarr sink writer.

task_id `260910_aminx-sink-provenance-schema`. Three independent concerns, all additive
(new, OPTIONAL attrs -- absent means unknown; aminx has no reader for these stores yet,
so nothing here makes a claim about reader behavior):

1. **Logits bias semantics** (``logits_bias_semantics``): which of the two logit arrays a
   sink calls ``"logits"`` -- bias-free or bias-applied -- and whether the run actually
   had a nonzero bias to apply at all. See ``logits_bias_semantics_outputs``.
2. **PRNG seed** (``prng_seed``): the run-level seed, for exact reproducibility.
3. **Executing wheel version** (``aminx_version``): resolved via installed-distribution
   metadata, never via ``aminx.__version__`` (see ``resolve_aminx_version``).

Kept in ``aminx.io`` (not ``aminx.host``) because ``io/designs.py`` -- one of the four
staging paths that write a ``logits`` array -- lives here and must not depend on
``aminx.host``; the reverse dependency (``host``/``sampling`` importing from ``io``)
already exists elsewhere in the codebase (e.g. ``host/prep.py`` imports
``io/weights.py``).
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

import numpy as np

LOGITS_BIAS_SEMANTICS_SCHEMA = "logits_bias_semantics_v1"

# Hash-free provenance marker (audit finding A, task_id `260910_aminx-sink-provenance-schema`,
# code-review round on PR #154). AC4/D6's actual intent -- letting a reader distinguish a
# store that carries `logits_bias_semantics`/`prng_seed`/`aminx_version` from one that
# predates them -- was originally implemented by bumping `GRID_SCHEMA_VERSION`/
# `SAMPLING_SCHEMA_VERSION`. That was reverted: those two constants are NOT purely
# reader-facing labels -- `GRID_SCHEMA_VERSION` feeds `_grid_manifest_row_hash`, which feeds
# a campaign manifest row's output path and done-marker matching
# (`_sampling_grid_lineage.py`'s own revert comment has the full chain), so bumping them
# forces a full, silent recompute of any resumed campaign. This constant is the alternative:
# it participates in NO hash -- not `_grid_manifest_row_hash`, not `_grid_job_seed_hash`,
# not any sink's output path -- so staging it costs nothing beyond the one root attr.
# Present ⇒ this store was written by code new enough to also stage the three new fields
# above (still individually optional per their own docstrings). Absent ⇒ unknown, exactly
# like every other attr in this module.
SINK_PROVENANCE_VERSION = "sink_provenance_v1"

# Structural invariant of aminx.inference.decode.autoregressive's per-wave sampling
# closure (see autoregressive.py:353-397, specifically the `stored_logits`/
# `sampling_logits` split at 363-383): every AR sampling sink stages `stored_logits`
# (fused with a ZERO bias) under the name "logits", while `sampling_logits` (fused with
# `cond.bias`, the array actually passed to `jax.random.categorical`) is never itself
# staged. This pair of constants documents that fact so a future edit which changes
# which array is staged or sampled is drift-detectable by grep on this comment and the
# line numbers it cites -- if autoregressive.py:363-397 moves or changes which array
# feeds which output, update both together.
STORED_LOGITS_BIAS_APPLIED = False
SAMPLING_LOGITS_BIAS_APPLIED = True


def logits_bias_semantics_outputs(
  bias: Any | None,
  *,
  persist_bias: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
  """Build the ``(arrays, attrs)`` pair recording bias semantics for one AR sampling run.

  Returns ``(arrays, attrs)`` to merge into a root-level ``sink.stage`` call, matching the
  existing ``fixed_provenance_outputs`` convention (``host/_sampling_helper.py``).

  Args:
    bias: The RAW, pre-jit, per-position bias the caller passed to
      ``build_inference_bundle`` -- e.g. ``spec.run_spec.sampling.bias`` -- NOT
      ``ConditioningBundle.bias`` (``cond.bias``). ``cond.bias`` is always a concrete,
      possibly-all-zero array by the time it reaches the decode kernel
      (``bundle_builder.py:323`` synthesizes ``jnp.zeros((seq_len, 21))`` when the caller
      passed ``None``), so deriving ``bias_is_nonzero`` from it would always read
      "could be nonzero" and never actually detect an unbiased run. Passing the raw,
      possibly-``None`` value here is what makes ``bias_is_nonzero`` a real drift
      detector.
    persist_bias: When true (the adopted design decision) and the bias is actually
      nonzero, stage the raw bias array itself (shape ``(L, 21)``) under the array name
      ``"bias"``, once per run, so a reader can reconstruct ``sampling_logits`` from
      ``stored_logits`` without re-deriving the bias from elsewhere. Skipped when the
      bias is all-zero (nothing to reconstruct) or ``persist_bias=False``.

  Returns:
    ``(arrays, attrs)`` -- ``arrays`` contains ``{"bias": ...}`` only when a nonzero bias
    is being persisted, else empty. ``attrs`` always contains the ``logits_bias_semantics``
    group-level attr.
  """
  bias_array = None if bias is None else np.asarray(bias, dtype=np.float32)
  bias_is_nonzero = bool(bias_array is not None and np.any(bias_array != 0))
  bias_persisted = bool(bias_is_nonzero and persist_bias)

  arrays: dict[str, np.ndarray] = {}
  if bias_persisted:
    arrays["bias"] = bias_array

  attrs = {
    "logits_bias_semantics": {
      "schema": LOGITS_BIAS_SEMANTICS_SCHEMA,
      "stored_logits_bias_applied": STORED_LOGITS_BIAS_APPLIED,
      "sampling_logits_bias_applied": SAMPLING_LOGITS_BIAS_APPLIED,
      "bias_is_nonzero": bias_is_nonzero,
      "bias_persisted": bias_persisted,
    },
  }
  return arrays, attrs


def _freeze(value: Any) -> Any:
  """Make a JSON-safe attrs value hashable/comparable for the uniformity check below."""
  if isinstance(value, dict):
    return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
  if isinstance(value, (list, tuple)):
    return tuple(_freeze(v) for v in value)
  return value


def assert_uniform_group_attr(
  values: list[Any],
  *,
  attr_name: str,
  group_key: Any,
) -> None:
  """Raise if a group-level attr would silently vary across records staged under one key.

  AC1b: a single Zarr group's ``.attrs`` cannot represent per-record variation. When a
  caller stages several sample/noise/temperature cells' worth of data under ONE group
  key (either via one `sink.stage` call built from a python-level loop over cells, or via
  several `sink.stage` calls that share a key -- whose attrs merge with later-overwrites-
  earlier semantics per ``ZarrStagingSink.stage``), this must be called BEFORE staging so
  a genuine divergence raises loudly instead of resolving via silent last-write-wins.

  Args:
    values: One value per cell/record that would have contributed to the same group's
      attrs, e.g. one ``logits_bias_semantics`` dict computed per (noise, temperature)
      cell before they are all staged together under one fused key.
    attr_name: Name of the attr being checked, for the error message.
    group_key: The staging key the values would be merged under, for the error message.

  Raises:
    ValueError: If more than one distinct value is present.
  """
  distinct = {_freeze(v) for v in values}
  if len(distinct) > 1:
    msg = (
      f"{attr_name!r} differs across records staged under group key {group_key!r}: "
      f"{sorted(str(v) for v in distinct)}. A single Zarr group's attrs cannot represent "
      f"per-record variation -- this would silently resolve via last-write-wins."
    )
    raise ValueError(msg)


def resolve_aminx_version() -> str:
  """Resolve the executing aminx wheel's version via installed-distribution metadata.

  Deliberately does NOT use ``aminx.__version__``: that attribute falls back to a
  hardcoded ``"0.1.0"`` string on ``importlib.metadata.PackageNotFoundError`` and is
  therefore useless on bare-source imports / vendored copies (a known trap -- see
  tev_design's CLAUDE.md aminx section).

  Deliberately does NOT swallow arbitrary exceptions into an ``"unknown"`` sentinel --
  that is the weak pattern explicitly called out as not to copy. If distribution
  metadata genuinely cannot be resolved, this raises, and the CALLER is responsible for
  invoking it at a point where a raise is safe (run entry / sink construction), not while
  assembling result metadata after compute has already finished -- see callers of this
  function for the pattern.

  Raises:
    importlib.metadata.PackageNotFoundError: with an ACTIONABLE message (audit finding E,
      task_id `260910_aminx-sink-provenance-schema`, code-review round on PR #154) -- the
      bare ``importlib.metadata`` exception says only "No package metadata was found for
      aminx", which does not explain why a writer needs this at all or what to do about
      it. This is the exact failure mode a bare-source checkout or a vendored/``sys.path``-
      prepended copy hits (documented live pattern -- see tev_design's CLAUDE.md aminx
      section on the ``importlib.metadata`` vs. ``aminx.__version__`` trap): such an import
      has no ``.dist-info``, so distribution metadata genuinely cannot be resolved, and this
      now fails BEFORE the caller's expensive compute (sink construction), not silently
      after it, and not with a hardcoded ``"unknown"``/``"0.1.0"`` sentinel either. Re-raises
      the SAME exception type (via ``raise ... from e``) so callers catching
      ``PackageNotFoundError`` specifically still see it.
  """
  try:
    return importlib.metadata.version("aminx")
  except importlib.metadata.PackageNotFoundError as e:
    msg = (
      "Could not resolve the aminx wheel version via installed-distribution metadata "
      "(importlib.metadata.version('aminx')). This writer stamps 'aminx_version' as "
      "provenance on every design/sampling store so a later reader can tell which "
      "checkpoint/wheel produced it -- see tev_design's CLAUDE.md for why silently "
      "falling back to aminx.__version__'s hardcoded '0.1.0' sentinel is unacceptable "
      "here. The usual cause is that the CURRENTLY IMPORTED 'aminx' module has no "
      "associated distribution metadata -- a bare-source checkout, or a vendored copy "
      "reached by prepending it to sys.path, neither of which produces a .dist-info "
      "directory. An editable install (`pip install -e .` / `uv pip install -e .`) DOES "
      "produce .dist-info and resolves normally; that is the fix in the common case. "
      "If you are deliberately running from such a copy and understand you are giving up "
      "wheel-version provenance, pass an explicit override where the call site supports "
      "one (e.g. DesignZarrWriter(..., aminx_version=<value>)) rather than working around "
      "this function."
    )
    raise importlib.metadata.PackageNotFoundError(msg) from e


def prng_seed_attrs(seed: Any) -> dict[str, Any]:
  """Build the ``prng_seed`` root-attrs entry for one run.

  Every seed source in this codebase (``spec.run_spec.sampling.random_seed``) is a plain
  Python ``int`` (see ``run/specs.py``: ``random_seed: int = 42``), consumed via
  ``jax.random.key(...)``/``jax.random.PRNGKey(...)`` -- never handed to a sink as a key
  array. The int branch below is therefore what every current call site hits; the key
  array branch is defensive, in case a future caller persists a derived key directly
  instead of the originating seed int. A jax PRNG key is recorded as a list of its
  underlying uint32 words (via ``jax.random.key_data``) so it round-trips without
  depending on a specific jax key implementation being importable at read time.
  """
  if hasattr(seed, "shape") and hasattr(seed, "dtype"):
    import jax  # noqa: PLC0415

    if jax.dtypes.issubdtype(seed.dtype, jax.dtypes.prng_key):
      data = jax.random.key_data(seed)
    else:
      data = seed  # already a raw uint32 key array
    return {"prng_seed": np.asarray(data).tolist()}
  return {"prng_seed": int(seed)}
