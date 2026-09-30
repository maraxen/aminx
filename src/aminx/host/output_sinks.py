"""Default :class:`~aminx.protocols.DesignSink` builders and host sessions (Phase 5).

Registers ``noop`` and ``streaming_tensor_staging`` under
:data:`~aminx.registry.OUTPUT_SINKS`. Import this module (side-effect) before
using ``OUTPUT_SINKS.get`` in host code — :mod:`aminx.run.sampling` does so.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.experimental
import numpy as np
from numpy.typing import NDArray

from aminx.registry import OUTPUT_SINKS

if TYPE_CHECKING:
  from aminx.run.options import LaserOptions
  from aminx.types.bundles import EncoderOutput
  from aminx.types.protocols import DesignSink


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

_scoring_tensor_sink_ctx: ContextVar[DesignSink | None] = ContextVar(
  "_scoring_tensor_sink_ctx",
  default=None,
)

# io_callback host threads may not inherit ContextVar (see _jacobian_sink_ctx's
# identical fallback below, and jax.experimental.io_callback's own docs: unordered
# callbacks dispatch via a background thread pool, not necessarily the thread that
# entered the session). Confirmed empirically for streaming_tensor_sink_session:
# single-structure batches happened to land the callback on the calling thread
# (ContextVar visible, bug masked); multi-structure batches (e.g. a PoE bead's
# multi-input fusion) landed it on a different thread, where the ContextVar reads
# back None and the write silently no-ops -- surfaced as "Streaming tensor sink
# missing entry" on read-back. campaign run/worker never run more than one row's
# session concurrently in-process (run_manifest_row executes rows sequentially),
# so a plain module global is safe here: there is exactly one active sink at a time.
_active_streaming_tensor_sink_io: StreamingTensorStagingSink | None = None
_active_scoring_sink_io: DesignSink | None = None


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


def active_scoring_sink() -> DesignSink | None:
  """Return the active scoring sink, if any."""
  sink = _scoring_tensor_sink_ctx.get()
  if sink is not None:
    return sink
  return _active_scoring_sink_io


@contextmanager
def scoring_tensor_sink_session(sink: DesignSink) -> Iterator[DesignSink]:
  """Optional host session for scoring tensor staging (streaming HDF5 follow-ups)."""
  global _active_scoring_sink_io  # noqa: PLW0603
  token: Token[DesignSink | None] = _scoring_tensor_sink_ctx.set(sink)
  _active_scoring_sink_io = sink
  try:
    yield sink
  finally:
    _active_scoring_sink_io = None
    _scoring_tensor_sink_ctx.reset(token)


class EncoderIntermediateStagingSink:
  """Stages encoder intermediate io_callback payloads.

  Keyed by (batch_idx, structure_idx, noise_idx).
  """

  __slots__ = ("_pending",)

  def __init__(self) -> None:
    self._pending: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}

  def on_encoder_intermediate(
    self,
    batch_idx: object,
    structure_idx: object,
    noise_idx: object,
    node_features: object,
    edge_features: object,
  ) -> None:
    key = (
      int(np.asarray(batch_idx)),
      int(np.asarray(structure_idx)),
      int(np.asarray(noise_idx)),
    )
    self._pending[key] = (np.asarray(node_features), np.asarray(edge_features))

  def take_intermediates(
    self,
    batch_idx: int,
    structure_idx: int,
    noise_idx: int,
  ) -> tuple[np.ndarray, np.ndarray]:
    key = (batch_idx, structure_idx, noise_idx)
    try:
      return self._pending.pop(key)
    except KeyError as e:
      pending = list(self._pending.keys())
      msg = f"Encoder intermediate sink missing entry for {key=}; pending={pending}"
      raise RuntimeError(msg) from e


_encoder_intermediate_sink_ctx: ContextVar[EncoderIntermediateStagingSink | None] = ContextVar(
  "_encoder_intermediate_sink_ctx",
  default=None,
)

# See _active_streaming_tensor_sink_io's comment above: same io_callback
# cross-thread ContextVar gap applies here.
_active_encoder_intermediate_sink_io: EncoderIntermediateStagingSink | None = None


@contextmanager
def encoder_sink_session() -> Iterator[EncoderIntermediateStagingSink]:
  """Activate a fresh staging sink for encoder intermediate captures."""
  global _active_encoder_intermediate_sink_io  # noqa: PLW0603
  sink = EncoderIntermediateStagingSink()
  token: Token[EncoderIntermediateStagingSink | None] = _encoder_intermediate_sink_ctx.set(sink)
  _active_encoder_intermediate_sink_io = sink
  try:
    yield sink
  finally:
    _active_encoder_intermediate_sink_io = None
    _encoder_intermediate_sink_ctx.reset(token)


def take_encoder_intermediates(
  batch_idx: int,
  structure_idx: int,
  noise_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Drain encoder intermediate payload for this (batch, structure, noise) key."""
  sink = active_encoder_staging_sink()
  if sink is None:
    msg = "Encoder intermediate sink is not active."
    raise RuntimeError(msg)
  return sink.take_intermediates(batch_idx, structure_idx, noise_idx)


def active_encoder_staging_sink() -> EncoderIntermediateStagingSink | None:
  """Return the active encoder intermediate sink, if any."""
  sink = _encoder_intermediate_sink_ctx.get()
  if sink is not None:
    return sink
  return _active_encoder_intermediate_sink_io


def _dispatch_encoder_intermediate_io(
  batch_idx: object,
  structure_idx: object,
  noise_idx: object,
  node_features: object,
  edge_features: object,
) -> None:
  """io_callback target: routes encoder intermediate to active staging sink."""
  sink = active_encoder_staging_sink()
  if sink is None:
    return
  sink.on_encoder_intermediate(batch_idx, structure_idx, noise_idx, node_features, edge_features)


class IoCallbackEncoderSink(eqx.Module):
  """Fires jax.experimental.io_callback per noise-level encoding into the active encoder sink."""

  def __call__(
    self,
    enc: EncoderOutput,
    batch_idx: object,
    structure_idx: object,
    noise_idx: object,
  ) -> None:
    jax.experimental.io_callback(
      _dispatch_encoder_intermediate_io,
      None,
      batch_idx,
      structure_idx,
      noise_idx,
      enc.node_features,
      enc.edge_features,
      ordered=False,
    )


class JacobianAccumulationSink:
  """Accumulates Jacobian tensors on the host (runner drain or ``io_callback`` staging)."""

  __slots__ = ("_pending",)

  def __init__(self) -> None:
    self._pending: list[np.ndarray] = []

  def on_jacobian(self, _structure_idx: object, jacobian_host: object) -> None:
    self._pending.append(np.asarray(jacobian_host))

  def take_all(self) -> list[np.ndarray]:
    return list(self._pending)


_jacobian_sink_ctx: ContextVar[JacobianAccumulationSink | None] = ContextVar(
  "_jacobian_sink_ctx",
  default=None,
)

# io_callback host threads may not inherit ContextVar; session sets this fallback.
_active_jacobian_sink_io: JacobianAccumulationSink | None = None


@contextmanager
def jacobian_sink_session() -> Iterator[JacobianAccumulationSink]:
  """Activate accumulation sink for jacobian ``io_callback`` staging."""
  global _active_jacobian_sink_io  # noqa: PLW0603
  sink = JacobianAccumulationSink()
  token: Token[JacobianAccumulationSink | None] = _jacobian_sink_ctx.set(sink)
  _active_jacobian_sink_io = sink
  try:
    yield sink
  finally:
    _active_jacobian_sink_io = None
    _jacobian_sink_ctx.reset(token)


def active_jacobian_sink() -> JacobianAccumulationSink | None:
  """Return the active jacobian accumulation sink, if any."""
  sink = _jacobian_sink_ctx.get()
  if sink is not None:
    return sink
  return _active_jacobian_sink_io


def _dispatch_jacobian_io(structure_idx: object, jacobian_host: object) -> None:
  """io_callback target: append Jacobian host array to active accumulation sink."""
  sink = active_jacobian_sink()
  if sink is None:
    return
  sink.on_jacobian(structure_idx, jacobian_host)


@jax.jit
def stage_jacobian_io(structure_idx: jax.Array, jacobian: jax.Array) -> jax.Array:
  """Stage a Jacobian to the host sink inside a JIT boundary."""
  jax.experimental.io_callback(
    _dispatch_jacobian_io,
    None,
    structure_idx,
    jacobian,
    ordered=False,
  )
  return jacobian


def laser_design_targets(options: LaserOptions) -> tuple[bool, bool]:
  """Which design files to write.

  ``output_fasta_only`` suppresses the PDB. Either fasta flag turns the FASTA
  on. Defaults stay PDB-only, matching ``run_batch_inference.py``.
  """
  write_fasta = options.output_fasta or options.output_fasta_only
  write_pdb = not options.output_fasta_only
  return write_pdb, write_fasta


def _pdb_atom_name(name: str) -> str:
  """Columns 13-16. A 4-character name fills them; shorter names lead with a space."""
  if len(name) >= 4:
    return name[:4]
  return f" {name:<3}"[:4]


def _element_symbol(name: str) -> str:
  """Columns 77-78. Amino-acid names encode the element in the first character."""
  return name[0]


def format_pdb_atom(
  serial: int,
  name: str,
  resname: str,
  chain_id: str,
  resnum: int,
  xyz: NDArray[np.floating],
  *,
  occupancy: float = 1.0,
  bfactor: float = 0.0,
  alt_loc: str = " ",
  icode: str = " ",
) -> str:
  """One ATOM record. Columns follow the PDB fixed-width spec so a parser can round-trip."""
  x, y, z = (float(value) for value in xyz)
  # serial and resnum wrap the way the format's columns do; designs stay inside them.
  serial_field = serial % 100000
  resnum_field = resnum % 10000
  return (
    f"ATOM  {serial_field:5d} {_pdb_atom_name(name)}{alt_loc[:1]:1s}{resname:>3s} "
    f"{chain_id[:1]:1s}{resnum_field:4d}{icode[:1]:1s}   "
    f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{bfactor:6.2f}"
    f"          {_element_symbol(name):>2s}"
  )


def write_design_pdb(
  path: Path,
  coords: NDArray[np.floating],
  sequence: NDArray[np.integer],
  *,
  chain_ids: Sequence[str] | None = None,
  resnums: Sequence[int] | None = None,
  bfactors: NDArray[np.floating] | None = None,
) -> None:
  """Write finite sidechain atoms. NaN slots are left out, not written as zeros."""
  # Lazy: rotamers imports the ProDy featurizer, and this module is imported
  # at package init to register the tensor sinks. PDB writing must not make
  # `import aminx` require the laser extra.
  from aminx.model.laser.rotamers import placed_atoms  # noqa: PLC0415

  length = int(np.asarray(sequence).shape[0])
  chains = ("A",) * length if chain_ids is None else tuple(chain_ids)
  numbers = tuple(range(1, length + 1)) if resnums is None else tuple(resnums)
  betas = None if bfactors is None else np.asarray(bfactors, dtype=np.float64)
  lines: list[str] = []
  serial = 1
  for residue, name, resname, xyz in placed_atoms(coords, sequence):
    beta = 0.0 if betas is None else float(betas[residue])
    lines.append(
      format_pdb_atom(
        serial,
        name,
        resname,
        chains[residue],
        numbers[residue],
        xyz,
        bfactor=beta,
      ),
    )
    serial += 1
  lines.append("END")
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_design_fasta(
  path: Path,
  sequence: NDArray[np.integer],
  *,
  header: str,
  append: bool = True,
) -> None:
  """Append one FASTA record. ``append`` keeps the batch writer's designs.fasta."""
  from aminx.families.laser_mpnn.featurize import LASER_ALPHABET  # noqa: PLC0415

  letters = "".join(LASER_ALPHABET[int(index)] for index in np.asarray(sequence).tolist())
  path.parent.mkdir(parents=True, exist_ok=True)
  mode = "a" if append else "w"
  with path.open(mode, encoding="utf-8") as handle:
    handle.write(f">{header}\n{letters}\n")


def emit_laser_design(
  options: LaserOptions,
  output_dir: Path,
  *,
  stem: str,
  coords: NDArray[np.floating],
  sequence: NDArray[np.integer],
  fasta_header: str,
  chain_ids: Sequence[str] | None = None,
  resnums: Sequence[int] | None = None,
  bfactors: NDArray[np.floating] | None = None,
) -> tuple[Path | None, Path | None]:
  """Write PDB and/or FASTA according to the existing LaserOptions flags."""
  write_pdb, write_fasta = laser_design_targets(options)
  pdb_path: Path | None = None
  fasta_path: Path | None = None
  if write_pdb:
    pdb_path = output_dir / f"{stem}.pdb"
    write_design_pdb(
      pdb_path,
      coords,
      sequence,
      chain_ids=chain_ids,
      resnums=resnums,
      bfactors=bfactors,
    )
  if write_fasta:
    fasta_path = output_dir / "designs.fasta"
    write_design_fasta(fasta_path, sequence, header=fasta_header, append=True)
  return pdb_path, fasta_path


def _register_default_output_sinks() -> None:
  OUTPUT_SINKS.register("noop")(lambda: NoopDesignSink())  # noqa: PLW0108
  OUTPUT_SINKS.register("streaming_tensor_staging")(lambda: StreamingTensorStagingSink())  # noqa: PLW0108


_register_default_output_sinks()
