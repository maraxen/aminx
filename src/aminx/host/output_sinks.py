"""Default :class:`~aminx.protocols.DesignSink` builders and host sessions (Phase 5).

Registers ``noop`` and ``streaming_tensor_staging`` under
:data:`~aminx.registry.OUTPUT_SINKS`. Import this module (side-effect) before
using ``OUTPUT_SINKS.get`` in host code — :mod:`aminx.run.sampling` does so.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

import numpy as np

from aminx.registry import OUTPUT_SINKS


class NoopDesignSink:
  """Default host sink: ignore tensor payloads (tests may replace registry entries)."""

  def on_sampling_sequences_logits(
    self,
    batch_idx: object,
    batch_count: object,
    chunk_start: object,
    chunk_count: object,
    sequences_host: object,
    logits_host: object,
  ) -> None:
    del batch_idx, batch_count, chunk_start, chunk_count, sequences_host, logits_host

  def on_scoring_scores_logits(
    self,
    batch_idx: object,
    batch_count: object,
    scores_host: object,
    logits_host: object,
  ) -> None:
    del batch_idx, batch_count, scores_host, logits_host


class StreamingTensorStagingSink:
  """Stages sampling ``io_callback`` tensor payloads for drain after :func:`sink_barrier`."""

  __slots__ = ("_pending",)

  def __init__(self) -> None:
    self._pending: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}

  def on_sampling_sequences_logits(
    self,
    batch_idx: object,
    batch_count: object,
    chunk_start: object,
    chunk_count: object,
    sequences_host: object,
    logits_host: object,
  ) -> None:
    del batch_count
    key = (
      int(np.asarray(batch_idx)),
      int(np.asarray(chunk_start)),
      int(np.asarray(chunk_count)),
    )
    self._pending[key] = (np.asarray(sequences_host), np.asarray(logits_host))

  def on_scoring_scores_logits(
    self,
    batch_idx: object,
    batch_count: object,
    scores_host: object,
    logits_host: object,
  ) -> None:
    del batch_idx, batch_count, scores_host, logits_host

  def take_sequences_logits(
    self,
    batch_idx: int,
    chunk_start: int,
    chunk_count: int,
  ) -> tuple[np.ndarray, np.ndarray]:
    key = (batch_idx, chunk_start, chunk_count)
    try:
      return self._pending.pop(key)
    except KeyError as e:
      pending = list(self._pending.keys())
      msg = f"Streaming tensor sink missing entry for {key=}; pending={pending}"
      raise RuntimeError(msg) from e


_streaming_tensor_sink_ctx: ContextVar[StreamingTensorStagingSink | None] = ContextVar(
  "_streaming_tensor_sink_ctx",
  default=None,
)

# io_callback host threads may not inherit ContextVar: unordered callbacks dispatch
# via a background thread pool, not necessarily the thread that entered the session.
# Confirmed empirically for streaming_tensor_sink_session: single-structure batches
# happened to land the callback on the calling thread (ContextVar visible, bug masked);
# multi-structure batches (e.g. a PoE bead's multi-input fusion) landed it on a
# different thread, where the ContextVar reads back None and the write silently
# no-ops -- surfaced as "Streaming tensor sink missing entry" on read-back. campaign
# run/worker never run more than one row's session concurrently in-process
# (run_manifest_row executes rows sequentially), so a plain module global is safe
# here: there is exactly one active sink at a time.
_active_streaming_tensor_sink_io: StreamingTensorStagingSink | None = None


@contextmanager
def streaming_tensor_sink_session() -> Iterator[StreamingTensorStagingSink]:
  """Activate a fresh staging sink for one streaming sampling session."""
  global _active_streaming_tensor_sink_io  # noqa: PLW0603
  factory = OUTPUT_SINKS.get("streaming_tensor_staging")
  sink_any = factory()
  if not isinstance(sink_any, StreamingTensorStagingSink):
    msg = "streaming_tensor_staging factory must return StreamingTensorStagingSink"
    raise TypeError(msg)
  token: Token[StreamingTensorStagingSink | None] = _streaming_tensor_sink_ctx.set(sink_any)
  _active_streaming_tensor_sink_io = sink_any
  try:
    yield sink_any
  finally:
    _active_streaming_tensor_sink_io = None
    _streaming_tensor_sink_ctx.reset(token)


def take_staging_sequences_logits(
  batch_idx: int,
  chunk_start: int,
  chunk_count: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Drain tensor ``io_callback`` payloads for this streaming batch/chunk key."""
  sink = active_sampling_staging_sink()
  if sink is None:
    msg = "Streaming tensor sink is not active."
    raise RuntimeError(msg)
  return sink.take_sequences_logits(batch_idx, chunk_start, chunk_count)


def active_sampling_staging_sink() -> StreamingTensorStagingSink | None:
  """Return the active staging sink, if any (for advanced callers/tests)."""
  sink = _streaming_tensor_sink_ctx.get()
  if sink is not None:
    return sink
  return _active_streaming_tensor_sink_io


def _register_default_output_sinks() -> None:
  OUTPUT_SINKS.register("noop")(lambda: NoopDesignSink())
  OUTPUT_SINKS.register("streaming_tensor_staging")(lambda: StreamingTensorStagingSink())


_register_default_output_sinks()
