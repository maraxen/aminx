"""PottsMPNN family driver for ``score:energy``, ``score:ddg``, and ``sample``.

``sample`` is ``MPNNEncode → PottsARDecode → PottsSampleEnergy → PottsRefine``
(refine only when ``optimization_mode`` is not ``none``). ``optimize_pdb`` /
``optimize_fasta`` with a refine mode skip autoregressive decode and ranking.
Registration runs when this module is imported.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from xtrax.tiling import AxisSpec, BatchPlanner

from aminx.families.potts_mpnn.etab import (
  ETAB_ALPHABET,
  merge_pair,
  pad_etab_energy,
  potts_energy,
)
from aminx.families.potts_mpnn.featurize import (
  PottsFeatures,
  PottsInputError,
  knn_boundary_tie,
  pad,
  parse_pdb_upstream,
  tied_featurize_port,
)
from aminx.families.potts_mpnn.model import PottsMPNN, _encoder_states, cast_floating
from aminx.families.potts_mpnn.sample_host import (
  SampleStages,
  force_optimize_num_samples,
  prepare_sample,
  sample_axes,
  sample_schema,
)
from aminx.host.family_driver import FAMILY_DRIVERS, FamilyBatch, FamilyStages, SinkArraySpec
from aminx.host.prep import _resolve_local_checkpoint_from_registry
from aminx.run.options import PottsMPNNOptions
from aminx.tiling.buckets import LENGTH_BUCKETS
from aminx.types.boundaries import AxisBoundary

log = logging.getLogger(__name__)

_HANDLED = frozenset({"score:energy", "score:ddg", "sample"})
_CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
_ETAB_INDEX = {letter: index for index, letter in enumerate(ETAB_ALPHABET)}
_PARTITION_BUCKETS = (8, 16, 32, 48, 64, 96, 128, 256, 512, 1024)


class _ChainSeq(NamedTuple):
  """One A0 chain, gaps still ``-``."""

  letter: str
  sequence: str


class _Candidate(NamedTuple):
  sequence: str
  chains: frozenset[str]
  expt: float


class _Graph(NamedTuple):
  """Device inputs for one featurized structure or partition."""

  coords: np.ndarray
  present: np.ndarray
  residue_idx: np.ndarray
  chain_index: np.ndarray
  pad_valid: np.ndarray
  sequences: np.ndarray


class _Prepared(NamedTuple):
  """One structure, its candidates, and any binding partitions."""

  graph: _Graph
  include_reference_in_output: bool
  expt: np.ndarray
  partitions: tuple[_Graph, ...]
  chain_order: tuple[str, ...]


class _SubtractPartitions:
  """``E_bind = E_complex - Σ_p E_p`` over stacked absolute energies."""

  def __call__(self, stacked: jax.Array) -> jax.Array:
    return stacked[0] - stacked[1:].sum(axis=0)


_BINDING_FUSE = AxisBoundary(fuse=_SubtractPartitions())


def absolute_energies(
  model: PottsMPNN,
  coords: jax.Array,
  present: jax.Array,
  residue_idx: jax.Array,
  chain_index: jax.Array,
  pad_valid: jax.Array,
  sequences: jax.Array,
) -> jax.Array:
  """Absolute Potts energies of ``sequences`` on one encoded structure.

  ``MPNNEncode → PottsHead → etab_energy → potts_energy``. Sequence rows are
  in the etab alphabet. The result follows ``coords.dtype``.
  """
  dtype = coords.dtype
  scored = cast_floating(model, dtype)
  present_f = present.astype(dtype)
  valid = pad_valid.astype(jnp.bool_)
  _nodes, edges, neighbor_indices = _encoder_states(
    scored.mpnn,
    coords.astype(dtype),
    present_f,
    residue_idx.astype(jnp.int32),
    chain_index.astype(jnp.int32),
    inference=True,
  )
  raw = scored.potts_head(edges[-1], neighbor_indices, present_f, valid)
  forward = merge_pair(
    raw,
    neighbor_indices,
    valid,
    denom=2,
    exclude_self=False,
  )
  table = merge_pair(
    forward,
    neighbor_indices,
    valid,
    denom=4,
    exclude_self=True,
  )
  return potts_energy(
    pad_etab_energy(table),
    neighbor_indices,
    valid,
    sequences.astype(jnp.int32),
  )


_JIT_ENERGIES = eqx.filter_jit(absolute_energies)


class _ScoreStages:
  """Host loop over a batch. Mutant rows are one broadcast, not a ``vmap``."""

  def __init__(self, model: PottsMPNN, *, mean_norm: bool) -> None:
    self._model = model
    self._mean_norm = mean_norm

  def __call__(
    self,
    batch: FamilyBatch,
    *,
    chunk_start: int,
    chunk_count: int,
    scalar_dropout: bool,
    dropout_key: jax.Array,
  ) -> Mapping[str, jax.Array]:
    del chunk_start, chunk_count, scalar_dropout, dropout_key
    prepared = cast("tuple[_Prepared, ...]", batch.arrays["prepared"])
    rows: list[jax.Array] = []
    ids: list[jax.Array] = []
    expt: list[jax.Array] = []
    for item in prepared:
      values = _score_one(self._model, item, mean_norm=self._mean_norm)
      rows.append(values)
      ids.append(jnp.arange(values.shape[0], dtype=jnp.int32))
      if item.include_reference_in_output:
        expt.append(jnp.full(values.shape, jnp.nan, dtype=jnp.float32))
      else:
        expt.append(jnp.asarray(item.expt, dtype=jnp.float32))
    stacked = jnp.stack(rows)
    id_stack = jnp.stack(ids)
    if prepared[0].include_reference_in_output:
      return {"energy": stacked, "candidate_ids": id_stack}
    return {
      "ddg": stacked,
      "mutant_ids": id_stack,
      "ddg_expt": jnp.stack(expt),
    }


def _score_one(model: PottsMPNN, item: _Prepared, *, mean_norm: bool) -> jax.Array:
  """Energies or ddG for one structure, fusing partitions when they exist."""
  block = _JIT_ENERGIES(model, *_device_graph(item.graph))
  if item.partitions:
    parts = [_JIT_ENERGIES(model, *_device_graph(part)) for part in item.partitions]
    stacked = jnp.stack([block, *parts], axis=0)
    block = _BINDING_FUSE.fuse(stacked) if _BINDING_FUSE.fuse is not None else block
  if item.include_reference_in_output:
    return block.astype(jnp.float32)
  delta = (block[1:] - block[0]).astype(jnp.float32)
  if mean_norm:
    delta = delta - jnp.mean(delta)
  return delta


def _device_graph(
  graph: _Graph,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  return (
    jnp.asarray(graph.coords),
    jnp.asarray(graph.present),
    jnp.asarray(graph.residue_idx),
    jnp.asarray(graph.chain_index),
    jnp.asarray(graph.pad_valid),
    jnp.asarray(graph.sequences),
  )


class PottsMPNNDriver:
  """Score and sample PottsMPNN through the family-driver seam."""

  name = "pottsmpnn"
  options_type = PottsMPNNOptions
  mpnn_fallback_purposes = frozenset({"jacobian", "inspect", "score:nll", "score:logits"})

  def handles(self, spec: Any, purpose: str) -> bool:  # noqa: ANN401
    """True for ``score:energy``, ``score:ddg``, and ``sample``."""
    del spec
    return purpose in _HANDLED

  def load(self, spec: Any) -> PottsMPNN:  # noqa: ANN401
    """Load a converted PottsMPNN. Does not enter inference mode.

    Optimize paths force ``num_samples`` to 1 before the runner plans chunks.

    ``model_local_path`` wins. A ``.pt`` path is converted on the fly
    (torch is imported only there). Otherwise the registry artifact is
    loaded, and ``pottsmpnn`` entries must carry ``sha256``.
    """
    force_optimize_num_samples(spec)
    local = getattr(spec, "model_local_path", None)
    if local is not None:
      path = Path(str(local))
      if path.suffix.lower() == ".pt":
        return _load_torch_checkpoint(path)
      return _load_eqx(path)
    artifact = _resolve_local_checkpoint_from_registry(spec)
    if artifact is None:
      msg = "PottsMPNNDriver requires model_local_path or a checkpoint registry entry with sha256"
      raise ValueError(msg)
    return _load_eqx(Path(artifact))

  def mpnn_core(self, model: eqx.Module) -> Any:  # noqa: ANN401
    """Embedded stock ProteinMPNN used by fallback purposes."""
    return cast("PottsMPNN", model).mpnn

  def batches(self, spec: Any) -> Iterator[FamilyBatch]:  # noqa: ANN401
    """One structure per batch, featurized by the A0 host port."""
    force_optimize_num_samples(spec)
    if _is_sample(spec):
      yield from _sample_batches(spec)
      return
    pending: list[tuple[int, str]] = []
    for index, item in enumerate(_inputs(spec)):
      try:
        prepared = _prepare_from_spec(item, spec)
      except PottsInputError as exc:
        pending.append((index, str(exc)))
        continue
      yield FamilyBatch(
        input_indices=(index,),
        arrays={"prepared": (prepared,)},
        skipped=tuple(pending),
        lengths=(int(prepared.graph.coords.shape[0]),),
      )
      pending = []
    if pending:
      yield FamilyBatch(
        input_indices=(),
        arrays={"prepared": ()},
        skipped=tuple(pending),
        lengths=(),
      )

  def axes(self, spec: Any, purpose: str, batch: FamilyBatch) -> list[AxisSpec]:  # noqa: ANN401
    """Structure, mutant, partition, or sample/temperature axes."""
    if purpose == "sample":
      return sample_axes(spec, batch)
    del spec
    prepared = cast("tuple[_Prepared, ...]", batch.arrays.get("prepared", ()))
    n_mut = int(prepared[0].graph.sequences.shape[0]) if prepared else 1
    n_part = len(prepared[0].partitions) if prepared else 0
    specs = [
      AxisSpec(
        name="structures",
        cardinality=max(len(batch.input_indices), 1),
        default_batch_size=1,
        heterogeneous=True,
        bucket_boundaries=LENGTH_BUCKETS,
      ),
      AxisSpec(
        name="mutants",
        cardinality=max(n_mut, 1),
        default_batch_size=max(n_mut, 1),
      ),
    ]
    if purpose == "score:ddg" and n_part > 0:
      specs.append(
        AxisSpec(
          name="partitions",
          cardinality=n_part,
          default_batch_size=1,
          heterogeneous=True,
          bucket_boundaries=_PARTITION_BUCKETS,
        ),
      )
    plan = BatchPlanner().plan(specs)
    mutant = next(decision for decision in plan.decisions if decision.spec.name == "mutants")
    if mutant.batch_size < 1:
      msg = "BatchPlanner returned a non-positive mutant batch size"
      raise ValueError(msg)
    return specs

  def stages(self, spec: Any, purpose: str, model: eqx.Module) -> FamilyStages:  # noqa: ANN401
    """Per-batch energy, ddG, or sample stage."""
    if purpose == "sample":
      return SampleStages(cast("PottsMPNN", model), spec)
    options = _options(spec)
    return _ScoreStages(cast("PottsMPNN", model), mean_norm=bool(options.mean_norm))

  def result_schema(self, spec: Any, purpose: str) -> Mapping[str, SinkArraySpec]:  # noqa: ANN401
    """§4.5 arrays. Per-position dims are ``L_total`` on the sample path."""
    if purpose == "sample":
      return sample_schema(spec)
    del spec
    if purpose == "score:energy":
      return {
        "energy": SinkArraySpec(dims=("N_cand",), dtype="float32", attrs={}),
        "candidate_ids": SinkArraySpec(dims=("N_cand",), dtype="int32", attrs={}),
      }
    if purpose == "score:ddg":
      return {
        "ddg": SinkArraySpec(dims=("N_mut",), dtype="float32", attrs={}),
        "mutant_ids": SinkArraySpec(dims=("N_mut",), dtype="int32", attrs={}),
        "ddg_expt": SinkArraySpec(dims=("N_mut",), dtype="float32", attrs={}),
      }
    msg = f"pottsmpnn does not support {purpose} in v1"
    raise ValueError(msg)


def _options(spec: Any) -> PottsMPNNOptions:  # noqa: ANN401
  options = getattr(spec, "potts_mpnn", None)
  if options is None:
    return PottsMPNNOptions()
  return cast("PottsMPNNOptions", options)


def _inputs(spec: Any) -> list[Any]:  # noqa: ANN401
  raw = spec.inputs
  if isinstance(raw, (str, Path)):
    return [raw]
  return list(raw)


def _pack(
  parsed: dict[str, Any],
  features: PottsFeatures,
  chains: tuple[_ChainSeq, ...],
  candidates: tuple[_Candidate, ...],
  options: PottsMPNNOptions,
  purpose: str,
  name: str,
) -> _Prepared:
  reference = "".join(chain.sequence for chain in chains)
  if purpose == "score:energy":
    rows = (reference, *[item.sequence for item in candidates])
    graph = _graph(features, rows)
    return _Prepared(
      graph=graph,
      include_reference_in_output=True,
      expt=np.full((len(rows),), np.nan, dtype=np.float32),
      partitions=(),
      chain_order=tuple(chain.letter for chain in chains),
    )
  if not candidates:
    msg = f"score:ddg produced no mutants for {name}"
    raise ValueError(msg)
  rows = (reference, *[item.sequence for item in candidates])
  touched = set().union(*(item.chains for item in candidates))
  partitions = _partition_graphs(parsed, chains, options, name, rows, touched)
  expt = np.asarray([item.expt for item in candidates], dtype=np.float32)
  return _Prepared(
    graph=_graph(features, rows),
    include_reference_in_output=False,
    expt=expt,
    partitions=partitions,
    chain_order=tuple(chain.letter for chain in chains),
  )


def _candidates(
  name: str,
  chains: tuple[_ChainSeq, ...],
  options: PottsMPNNOptions,
  purpose: str,
  sequences: Sequence[str] | None,
) -> tuple[_Candidate, ...]:
  """Resolve mutant rows. Energy with explicit sequences does not open DMS."""
  if sequences is None and purpose == "score:energy":
    # Filled by the caller from the spec when sequences_to_score is set.
    sequences = ()
  if purpose == "score:energy" and sequences:
    return tuple(_sequence_candidate(seq, chains, index) for index, seq in enumerate(sequences))
  if options.mutant_fasta:
    return _from_fasta(Path(options.mutant_fasta), name, chains)
  if options.mutant_csv:
    return _from_csv(Path(options.mutant_csv), name, chains)
  if purpose == "score:energy" and not sequences:
    return _dms(chains, options)
  if purpose == "score:ddg":
    return _dms(chains, options)
  return ()


def _sequence_candidate(sequence: str, chains: tuple[_ChainSeq, ...], index: int) -> _Candidate:
  expected = sum(len(chain.sequence) for chain in chains)
  if len(sequence) != expected:
    order = [chain.letter for chain in chains]
    msg = (
      f"sequences_to_score entry {index} has length {len(sequence)}; "
      f"expected {expected} in chain order {order}"
    )
    raise ValueError(msg)
  try:
    _encode(sequence)
  except KeyError as exc:
    msg = f"sequences_to_score entry {index} has a residue outside {ETAB_ALPHABET}"
    raise ValueError(msg) from exc
  return _Candidate(sequence, frozenset(chain.letter for chain in chains), float("nan"))


def _from_fasta(
  path: Path,
  name: str,
  chains: tuple[_ChainSeq, ...],
) -> tuple[_Candidate, ...]:
  text = path.read_text(encoding="utf-8").splitlines()
  found: list[_Candidate] = []
  by_letter = {chain.letter: chain.sequence for chain in chains}
  order = [chain.letter for chain in chains]
  for raw_header, seq in zip(text[::2], text[1::2], strict=False):
    header = raw_header.strip()
    if not header.startswith(">"):
      continue
    body = header[1:]
    pdb = body.split("|", 1)[0].strip()
    if pdb != name:
      continue
    parts = body.split("|")
    mut_chains: list[str] | None = None
    expt = float("nan")
    if len(parts) == 2:
      parsed = _float_or_none(parts[1])
      if parsed is None:
        mut_chains = parts[1].split(":")
      else:
        expt = parsed
    elif len(parts) >= 3:
      mut_chains = parts[1].split(":")
      parsed = _float_or_none(parts[2])
      expt = float("nan") if parsed is None else parsed
    pieces = seq.strip().split(":")
    letters = mut_chains or order
    if mut_chains is None and len(pieces) != len(order):
      msg = f"fasta mutant for {name} must list every chain when chains are omitted"
      raise ValueError(msg)
    replacement = dict(zip(letters, pieces, strict=False))
    built: list[str] = []
    for letter in order:
      chunk = replacement.get(letter, by_letter[letter])
      if len(chunk) != len(by_letter[letter]):
        msg = f"fasta chain {letter} length {len(chunk)} != {len(by_letter[letter])} for {name}"
        raise ValueError(msg)
      built.append(chunk)
    touched = frozenset(letters)
    found.append(_Candidate("".join(built), touched, expt))
  return tuple(found)


def _from_csv(
  path: Path,
  name: str,
  chains: tuple[_ChainSeq, ...],
) -> tuple[_Candidate, ...]:
  by_letter = {chain.letter: chain.sequence for chain in chains}
  order = [chain.letter for chain in chains]
  found: list[_Candidate] = []
  with path.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      if row.get("pdb") != name:
        continue
      chain_field = row.get("chain", "")
      mut_field = row.get("mut_type", "")
      expt = _float_or_none(row.get("ddG_expt", ""))
      edited = dict(by_letter)
      touched: set[str] = set()
      for chain, mut_type in zip(chain_field.split(":"), mut_field.split(":"), strict=False):
        if not chain or not mut_type:
          continue
        current = edited[chain]
        wild, pos_token, mut = mut_type[0], mut_type[1:-1], mut_type[-1]
        pos = int(pos_token)
        if current[pos] != wild:
          msg = (
            f"mutation {mut_type} does not match chain {chain} of {name} "
            f"at position {pos} ({current[pos]})"
          )
          raise ValueError(msg)
        if mut not in _CANONICAL:
          msg = f"mutation {mut_type} is outside the 20 canonical amino acids"
          raise ValueError(msg)
        edited[chain] = current[:pos] + mut + current[pos + 1 :]
        touched.add(chain)
      sequence = "".join(edited[letter] for letter in order)
      found.append(_Candidate(sequence, frozenset(touched), float("nan") if expt is None else expt))
  return tuple(found)


def _dms(chains: tuple[_ChainSeq, ...], options: PottsMPNNOptions) -> tuple[_Candidate, ...]:
  excluded = _excluded(options)
  by_letter = {chain.letter: chain.sequence for chain in chains}
  order = [chain.letter for chain in chains]
  found: list[_Candidate] = []
  for chain in chains:
    if chain.letter in excluded:
      continue
    for pos, wild in enumerate(chain.sequence):
      if wild == "-":
        continue
      for mut in _CANONICAL:
        if mut == wild:
          continue
        edited = dict(by_letter)
        edited[chain.letter] = chain.sequence[:pos] + mut + chain.sequence[pos + 1 :]
        sequence = "".join(edited[letter] for letter in order)
        found.append(_Candidate(sequence, frozenset({chain.letter}), float("nan")))
  return tuple(found)


def _partition_graphs(
  parsed: dict[str, Any],
  chains: tuple[_ChainSeq, ...],
  options: PottsMPNNOptions,
  name: str,
  rows: tuple[str, ...],
  touched: set[str],
) -> tuple[_Graph, ...]:
  """Re-featurize each flagged partition and score those chains only."""
  spec_parts = _binding_partitions(options, name)
  if not spec_parts:
    return ()
  graphs: list[_Graph] = []
  for partition in spec_parts:
    if not set(partition).intersection(touched):
      continue
    designed = sorted(partition)
    part_features = tied_featurize_port([parsed], {name: (designed, [])})[0]
    part_rows = tuple(_project(row, chains, designed) for row in rows)
    graphs.append(_graph(part_features, part_rows))
  return tuple(graphs)


def _project(full: str, chains: tuple[_ChainSeq, ...], keep: Sequence[str]) -> str:
  """Slice an A0-order sequence down to ``keep`` (partition chain order)."""
  cursor = 0
  pieces: dict[str, str] = {}
  for chain in chains:
    pieces[chain.letter] = full[cursor : cursor + len(chain.sequence)]
    cursor += len(chain.sequence)
  return "".join(pieces[letter] for letter in keep)


def _graph(features: PottsFeatures, rows: tuple[str, ...]) -> _Graph:
  padded, pad_valid = pad(features, features.L_total)
  encoded = np.stack([_encode(row) for row in rows]).astype(np.int32)
  if encoded.shape[1] != features.L_total:
    msg = f"encoded length {encoded.shape[1]} != L_total {features.L_total}"
    raise ValueError(msg)
  return _Graph(
    coords=np.asarray(padded.x, dtype=np.float32),
    present=np.asarray(padded.present, dtype=np.float32),
    residue_idx=np.asarray(padded.residue_idx, dtype=np.int32),
    chain_index=np.asarray(padded.chain_encoding, dtype=np.int32),
    pad_valid=np.asarray(pad_valid, dtype=np.bool_),
    sequences=encoded,
  )


def _encode(sequence: str) -> np.ndarray:
  return np.asarray([_ETAB_INDEX[letter] for letter in sequence], dtype=np.int32)


def _chain_sequences(parsed: dict[str, Any], features: PottsFeatures) -> tuple[_ChainSeq, ...]:
  return tuple(
    _ChainSeq(letter, str(parsed[f"seq_chain_{letter}"])) for letter in features.letter_list
  )


def _letters(parsed: dict[str, Any]) -> list[str]:
  return [str(letter) for letter in parsed["chain_order"]]


def _chain_dict(
  options: PottsMPNNOptions,
  name: str,
) -> dict[str, tuple[list[str], list[str]]] | None:
  if not options.chain_design_mask_json:
    return None
  payload = json.loads(Path(options.chain_design_mask_json).read_text(encoding="utf-8"))
  entry = payload.get(name)
  if not entry:
    return None
  designed, fixed = entry
  return {name: ([str(letter) for letter in designed], [str(letter) for letter in fixed])}


def _binding_partitions(options: PottsMPNNOptions, name: str) -> list[list[str]] | None:
  if not options.binding_energy_json:
    return None
  payload = json.loads(Path(options.binding_energy_json).read_text(encoding="utf-8"))
  entry = payload.get(name)
  if not entry:
    return None
  return [[str(letter) for letter in partition] for partition in entry]


def _excluded(options: PottsMPNNOptions) -> set[str]:
  if not options.exclude_chains:
    return set()
  return {part for part in re.split(r"[,:\s]+", options.exclude_chains) if part}


def _float_or_none(text: str | None) -> float | None:
  if text is None:
    return None
  token = text.strip()
  if not token or token.lower() == "nan":
    return None
  try:
    return float(token)
  except ValueError:
    return None


def _load_eqx(path: Path) -> PottsMPNN:
  skeleton = PottsMPNN(key=jax.random.PRNGKey(0))
  loaded = eqx.tree_deserialise_leaves(path, skeleton)
  return cast("PottsMPNN", loaded)


def _converter_module() -> Any:  # noqa: ANN401
  path = Path(__file__).resolve().parents[4] / "scripts" / "recapture" / "pottsmpnn_model_to_eqx.py"
  spec = importlib.util.spec_from_file_location("pottsmpnn_model_to_eqx", path)
  if spec is None or spec.loader is None:
    msg = f"cannot load {path}"
    raise ImportError(msg)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _load_torch_checkpoint(path: Path) -> PottsMPNN:
  """Build an Equinox PottsMPNN from a torch checkpoint. Torch stays lazy."""
  module = _converter_module()
  state = module._load_torch_state(path)  # noqa: SLF001 — lazy torch import lives here
  model = module.convert_state_dict(state)
  return cast("PottsMPNN", model)


def _spec_sequences(spec: Any) -> tuple[str, ...]:  # noqa: ANN401
  raw = getattr(spec, "sequences_to_score", ())
  return tuple(str(item) for item in raw)


def _is_sample(spec: Any) -> bool:  # noqa: ANN401
  return not hasattr(spec, "output_kind")


def _sample_batches(spec: Any) -> Iterator[FamilyBatch]:  # noqa: ANN401
  """One PDB per batch for purpose ``sample``."""
  options = _options(spec)
  pending: list[tuple[int, str]] = []
  for index, item in enumerate(_inputs(spec)):
    if not isinstance(item, (str, Path)):
      pending.append((index, "pottsmpnn_requires_pdb"))
      continue
    try:
      path = Path(item)
      parsed_list = parse_pdb_upstream(path, skip_gaps=options.skip_gaps)
      parsed = parsed_list[0]
      name = str(parsed["name"])
      features = tied_featurize_port([parsed], _chain_dict(options, name))[0]
      if knn_boundary_tie(features.present, features.L_total):
        log.warning("knn_boundary_tie for %s (L_total=%s)", name, features.L_total)
      chains = _chain_sequences(parsed, features)
      ready = prepare_sample(
        features,
        tuple((chain.letter, chain.sequence) for chain in chains),
        options,
        spec,
      )
    except PottsInputError as exc:
      pending.append((index, str(exc)))
      continue
    yield FamilyBatch(
      input_indices=(index,),
      arrays={"ready": (ready,)},
      skipped=tuple(pending),
      lengths=(ready.l_total,),
    )
    pending = []
  if pending:
    yield FamilyBatch(
      input_indices=(),
      arrays={"ready": ()},
      skipped=tuple(pending),
      lengths=(),
    )


def _prepare_from_spec(item: Any, spec: Any) -> _Prepared:  # noqa: ANN401
  options = _options(spec)
  purpose = f"score:{spec.output_kind}"
  if not isinstance(item, (str, Path)):
    raise PottsInputError("pottsmpnn_requires_pdb")
  path = Path(item)
  parsed_list = parse_pdb_upstream(path, skip_gaps=options.skip_gaps)
  parsed = parsed_list[0]
  name = str(parsed["name"])
  features = tied_featurize_port([parsed], _chain_dict(options, name))[0]
  if knn_boundary_tie(features.present, features.L_total):
    log.warning("knn_boundary_tie for %s (L_total=%s)", name, features.L_total)
  chains = _chain_sequences(parsed, features)
  explicit = _spec_sequences(spec) if purpose == "score:energy" else ()
  if explicit:
    candidates = tuple(
      _sequence_candidate(seq, chains, index) for index, seq in enumerate(explicit)
    )
  else:
    candidates = _candidates(name, chains, options, purpose, ())
  return _pack(parsed, features, chains, candidates, options, purpose, name)


FAMILY_DRIVERS.register("pottsmpnn")(PottsMPNNDriver())
