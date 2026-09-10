"""Efficient storage of generated designs using Zarr."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from xtrax.run import SinkSpec, ZarrStagingSink, new_run_id

from aminx.io.sink_provenance import SINK_PROVENANCE_VERSION, resolve_aminx_version


def _to_numpy_uint8(x: jnp.ndarray | np.ndarray) -> np.ndarray:
  """Host-side snapshot to uint8; skips ``device_get`` when ``x`` is already NumPy."""
  if isinstance(x, np.ndarray):
    return np.asarray(x, dtype=np.uint8)
  return np.asarray(jax.device_get(x), dtype=np.uint8)


def _to_numpy_float32(x: jnp.ndarray | np.ndarray) -> np.ndarray:
  """Host-side snapshot to float32; skips ``device_get`` when ``x`` is already NumPy."""
  if isinstance(x, np.ndarray):
    return np.asarray(x, dtype=np.float32)
  return np.asarray(jax.device_get(x), dtype=np.float32)


class DesignMetadata(TypedDict):
  """Metadata for a single design."""

  pool_type: Literal["BackboneOnly", "BackboneLigand", "BackboneSidechain", "FullContext"]
  state_mapping: list[int]
  weight_strategy: str
  combination_algorithm: str
  structure_ids: list[str]
  parent_structure_idx: int


class DesignPayload(TypedDict):
  """Serialized design payload."""

  sequence: Any  # jnp.ndarray (uint8), shape (n_canonical,)
  logits: Any  # jnp.ndarray (float32 input, cast to float16 for storage), shape (n_canonical, 21)
  scores: Any  # jnp.ndarray (float32)
  state_weights: Any  # jnp.ndarray (float32)
  metadata: DesignMetadata


class DesignZarrWriter:
  """Writer for storing designs in a chunked Zarr store, one nested group per design.

  Note on logit precision:
    Logits are stored as float16 to halve storage and bandwidth. float16 preserves
    amino acid rank-ordering and softmax probabilities to <1% relative error for
    logit values in [-20, 20]. When reading, upcast to float32 before computing
    softmax or logsumexp to avoid float16 accumulation error:
        logits = group["logits"][...].astype(np.float32)
        probs = scipy.special.softmax(logits, axis=-1)
  """

  def __init__(
    self,
    path: str,
    n_canonical: int = 214,
    n_states: int = 9,
    flush_every: int = 1,
    run_id: str | None = None,
    *,
    aminx_version: str | None = None,
  ):
    """Initialize the writer.

    Args:
      path: Path to the output Zarr store (a directory).
      n_canonical: Number of canonical residues (for shape validation).
      n_states: Number of states (for shape validation).
      flush_every: Stage calls to buffer before an automatic drain to disk.
      run_id: Provenance join key stamped on the store, linking everything
        written here to the run that produced it. Defaults to a fresh
        ``xtrax.run.new_run_id()``.

        Pass it explicitly to **reopen an existing store**:
        ``ZarrStagingSink`` raises if the store on disk already carries a
        different ``run_id``, so a defaulted writer can only ever create a
        new store or reopen one it happens to match. There is no run context
        at this layer to derive it from -- unlike the sampling/streaming/runner
        sinks, which have a ``RunSpec`` in scope and go through
        ``xtrax.run.derive_sink_spec``.
      aminx_version: OPTIONAL explicit override for the ``aminx_version`` provenance attr,
        bypassing ``resolve_aminx_version()`` entirely (audit finding E, task_id
        `260910_aminx-sink-provenance-schema`). Construction otherwise HARD-FAILS on a
        bare-source checkout or a vendored/``sys.path``-prepended aminx import -- neither
        produces the ``.dist-info`` metadata ``resolve_aminx_version()`` requires (see that
        function's docstring), a usage pattern tev_design's CLAUDE.md documents as live.
        This is an explicit, caller-must-opt-in escape hatch, never a silent default: pass
        it only when you have independently verified which aminx you are actually running
        (e.g. via ``aminx.__file__``) and accept vouching for that string yourself, since it
        is stamped verbatim with no further validation.

    """
    self.path = path
    self.n_canonical = n_canonical
    self.n_states = n_states
    self.run_id = run_id or new_run_id()
    self._sink = ZarrStagingSink(
      SinkSpec(run_id=self.run_id, output_dir=Path(path), format="zarr", flush_every=flush_every),
    )
    # Resolved HERE, at construction -- before any `write()` call, i.e. before any
    # caller-side compute this writer will ever be handed the result of. A
    # PackageNotFoundError therefore fails a run before it starts, not after a design has
    # already been computed and is only now being staged. Skipped entirely when the caller
    # passed an explicit `aminx_version` override (see the Args docstring above) -- that is
    # the one supported way to bypass resolution, never a bare `except: "unknown"`.
    self._aminx_version = aminx_version if aminx_version is not None else resolve_aminx_version()

  @classmethod
  def from_multistate_shapes(
    cls,
    path: str,
    *,
    n_canonical: int,
    n_states: int,
    flush_every: int = 1,
    run_id: str | None = None,
    aminx_version: str | None = None,
  ) -> Self:
    """Writer sized like :class:`aminx.bundles.ProteinBundle` static axes.

    ``n_canonical`` and ``n_states`` match the stack payload's ``n_canonical`` /
    ``n_states`` (roadmap §3.2) so Zarr groups align with multistate campaigns.

    ``run_id`` is forwarded to :meth:`__init__`; see there for why a caller
    reopening an existing store has to supply it.

    ``aminx_version`` forwards to :meth:`__init__`'s same-named explicit opt-in override
    (audit finding E, task_id `260910_aminx-sink-provenance-schema`) -- see that
    docstring.
    """
    return cls(
      path,
      n_canonical=n_canonical,
      n_states=n_states,
      flush_every=flush_every,
      run_id=run_id,
      aminx_version=aminx_version,
    )

  def write(
    self,
    key: tuple[int, ...],
    payload: DesignPayload,
    *,
    logits_bias_semantics: dict[str, Any] | None = None,
  ) -> None:
    """Stage a design payload under ``key`` for drain into the Zarr store.

    Args:
      key: Design address, e.g. ``(structure_idx, sample_idx, noise_idx, temp_idx)``.
        Becomes the nested Zarr group path for this design.
      payload: Sequence/logits/scores/state_weights arrays plus JSON-safe metadata.
      logits_bias_semantics: OPTIONAL ``logits_bias_semantics`` attrs value (see
        ``aminx.io.sink_provenance.logits_bias_semantics_outputs``), for callers whose
        ``payload["logits"]`` actually went through the stored-vs-sampling AR bias split
        that schema describes. Left unset (absent, per the schema's own optionality) when
        a caller has no such split to report -- e.g. this writer's current caller,
        ``inference/optimize_ste.py``, produces ``logits`` via straight-through-estimator
        optimization against a hard per-position ``fixed_bias`` constraint, which is a
        different mechanism from ``cond.bias``/the AR decode's stored/sampling split, so
        fabricating a value here would misdescribe that path's actual provenance.

        Must NOT claim ``bias_persisted: True`` -- see Raises.

    Raises:
      ValueError: If ``logits_bias_semantics["bias_persisted"]`` is true. This writer
        stages only sequence/logits/scores/state_weights and has no root group, so no
        ``"bias"`` array can exist anywhere in the store for that claim to refer to.
        Build the dict with ``logits_bias_semantics_outputs(bias, persist_bias=False)``
        to describe the semantics without claiming persistence.
    """
    seq = _to_numpy_uint8(payload["sequence"])
    assert seq.shape == (self.n_canonical,), f"sequence shape {seq.shape} != {(self.n_canonical,)}"

    logits_f32 = _to_numpy_float32(payload["logits"])
    assert logits_f32.shape == (self.n_canonical, 21), (
      f"logits shape {logits_f32.shape} != {(self.n_canonical, 21)}"
    )
    assert np.isfinite(logits_f32).all() and np.abs(logits_f32).max() < 1e4, (
      f"Logit values out of float16-safe range: max={np.abs(logits_f32).max():.1f}"
    )
    logits = logits_f32.astype(np.float16)

    scores = _to_numpy_float32(payload["scores"]).flatten()
    assert scores.shape == (1,), f"scores shape {scores.shape} != (1,)"

    weights = _to_numpy_float32(payload["state_weights"])
    assert weights.shape == (self.n_states,), f"weights shape {weights.shape} != {(self.n_states,)}"

    if logits_bias_semantics is not None and logits_bias_semantics.get("bias_persisted"):
      # This writer stages exactly four arrays (below) and has NO root group -- every design
      # lives in its own nested `key` group. So there is nowhere a `"bias"` array could
      # have been staged, and `bias_persisted: True` here is unconditionally false: the
      # exact lying-attr failure mode this schema exists to prevent (code-review round 2 on
      # PR #154). Fails closed rather than trusting the caller, because the natural way to
      # build this dict -- `logits_bias_semantics_outputs(some_nonzero_bias)` -- returns
      # `bias_persisted=True` and `bias_array_group="/"` by DEFAULT, so a caller doing the
      # obvious thing would silently write the lie.
      msg = (
        "logits_bias_semantics['bias_persisted'] is True, but DesignZarrWriter cannot "
        "stage a 'bias' array: it writes only sequence/logits/scores/state_weights, and "
        "has no root group for `bias_array_group='/'` to refer to. Staging this would "
        "record a bias array that does not exist anywhere in the store. Pass "
        "logits_bias_semantics_outputs(bias, persist_bias=False) to describe the bias "
        "semantics without claiming persistence, or persist the bias through a writer "
        "that stages a root group (host/streaming.py, sampling/multistate_poe.py)."
      )
      raise ValueError(msg)

    attrs: dict[str, Any] = dict(payload["metadata"])
    attrs["aminx_version"] = self._aminx_version
    # Hash-free marker (audit finding A) -- present ⇒ this group was written by code new
    # enough to also stamp `aminx_version` (and `logits_bias_semantics`, when the caller
    # supplies one). Staged per-group here rather than at a root group because this writer
    # has no root-level stage call; it participates in no hash either way.
    attrs["sink_provenance_version"] = SINK_PROVENANCE_VERSION
    if logits_bias_semantics is not None:
      attrs["logits_bias_semantics"] = logits_bias_semantics

    self._sink.stage(
      key,
      sequence=seq,
      logits=logits,
      scores=scores,
      state_weights=weights,
      attrs=attrs,
    )

  def close(self) -> None:
    """Drain all pending designs to disk."""
    self._sink.drain()

  def __enter__(self):
    """Context manager entry."""
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit: always drain pending writes."""
    self.close()
    return False
