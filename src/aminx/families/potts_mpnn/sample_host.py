"""Host side of Potts sampling: encode, decode, energy, refine, rank.

The device graph is ``MPNNEncode → PottsARDecode → PottsSampleEnergy →
PottsRefine`` when ``optimization_mode`` is not ``none``. ``optimize_pdb`` /
``optimize_fasta`` with a refine mode skip the autoregressive decode and the
ranking. ``num_samples`` is forced to 1, with a warning when a larger value
was requested.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int
from xtrax.tiling import AxisSpec, BatchPlanner

from aminx.families.potts_mpnn.decode import PottsARDecode, floor_temperature
from aminx.families.potts_mpnn.etab import (
  ETAB_ALPHABET,
  merge_pair,
  model_to_etab,
  pad_etab_energy,
  potts_energy,
)
from aminx.families.potts_mpnn.featurize import PottsFeatures, pad
from aminx.families.potts_mpnn.model import PottsMPNN, _encoder_states, cast_floating
from aminx.families.potts_mpnn.refine import BindingTables, PottsRefine, check_tied_only_groups
from aminx.host.family_driver import FamilyBatch, SinkArraySpec
from aminx.host.omit_aa_bias import omit_letter_indices
from aminx.run.options import PottsMPNNOptions
from aminx.tiling.buckets import LENGTH_BUCKETS

log = logging.getLogger(__name__)

_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
_ETAB_INDEX = {letter: index for index, letter in enumerate(ETAB_ALPHABET)}


class PottsSampleEnergy(eqx.Module):
  """``sample_energy = potts_energy(etab_energy, S_AR)`` on the pre-refine sequence."""

  def __call__(
    self,
    etab: Float[Array, "L K A A"],
    e_idx: Int[Array, "L K"],
    pad_valid: Bool[Array, " L"],
    sequence: Int[Array, " L"],
  ) -> Float[Array, ""]:
    """Energy of one model-alphabet sequence. ``X`` maps to etab slot 21."""
    return potts_energy(etab, e_idx, pad_valid, model_to_etab(sequence))


def rank_sample_energies(energy: np.ndarray) -> np.ndarray:
  """Stable argsort. Rank 0 is the lowest pre-refine energy (Python ``sorted``)."""
  return np.argsort(np.asarray(energy), kind="stable").astype(np.int32)


def fresh_refine_order(chain_mask: np.ndarray, randn: np.ndarray) -> np.ndarray:
  """``argsort((chain_mask + 1e-4)·|randn|)`` with no ``chain_M_pos`` and no ``present``."""
  key = (chain_mask.astype(np.float64) + 1.0e-4) * np.abs(randn.astype(np.float64))
  return np.argsort(key, kind="stable").astype(np.int32)


def upstream_refine_order(
  ar_order: np.ndarray,
  chain_mask: np.ndarray,
  randn: np.ndarray,
  *,
  num_samples: int,
  chain_suffix: str,
  stored_orders_present: bool,
) -> np.ndarray:
  """Refine-order keying quirk (``sample_seqs.py:325-343``).

  Stored orders are missed when ``num_samples > 1`` because the lookup key is
  ``_i`` and the write key is ``str(i)``, so the sweep is N-to-C. A chain-suffix
  miss draws a fresh order. ``optimize_pdb`` / ``optimize_fasta`` never read a
  stored file (``stored_orders_present=False``).
  """
  if not stored_orders_present or chain_suffix:
    return fresh_refine_order(chain_mask, randn)
  if num_samples == 1:
    return np.asarray(ar_order, dtype=np.int32)
  return np.arange(chain_mask.shape[0], dtype=np.int32)


def force_optimize_num_samples(spec: Any) -> None:  # noqa: ANN401
  """Force ``num_samples`` to 1 when an optimize path is set. Warn if it was larger."""
  options = _options(spec)
  if not (options.optimize_pdb or options.optimize_fasta):
    return
  requested = int(getattr(spec, "num_samples", 1) or 1)
  if requested > 1:
    log.warning(
      "num_samples=%s requested with optimize_pdb/optimize_fasta; forcing num_samples=1",
      requested,
    )
  object.__setattr__(spec, "num_samples", 1)
  run_spec = getattr(spec, "run_spec", None)
  if run_spec is None:
    return
  # num_samples is a static field, so it is not a pytree leaf and eqx.tree_at cannot
  # reach it ("Operation undefined, 4 is not a leaf of the pytree"). Rebuild the
  # dataclass instead, which is how the rest of the codebase edits static spec fields.
  sampling = getattr(run_spec, "sampling", None)
  if sampling is None:
    return
  updated = eqx.tree_at(
    lambda item: item.sampling,
    run_spec,
    dataclasses.replace(sampling, num_samples=1),
    is_leaf=lambda node: node is sampling,
  )
  object.__setattr__(spec, "run_spec", updated)


def seq_to_ints(sequence: str) -> np.ndarray:
  """Upstream ``etab_utils.seq_to_ints`` (``-`` = 20, ``X`` = 21)."""
  try:
    return np.asarray([_ETAB_INDEX[letter] for letter in sequence], dtype=np.int32)
  except KeyError as exc:
    msg = f"sequence has a residue outside {ETAB_ALPHABET}"
    raise ValueError(msg) from exc


def parse_fasta_records(text: str) -> tuple[tuple[str, str], ...]:
  """``(header, sequence)`` records. Wrapped sequence lines are joined.

  Blank lines are skipped. A non-header line before the first ``>`` is an
  error, so a leading comment cannot shift later records off the header/sequence
  pairs the way ``lines[::2]`` / ``lines[1::2]`` does.
  """
  records: list[tuple[str, str]] = []
  header: str | None = None
  chunks: list[str] = []

  def flush() -> None:
    nonlocal header, chunks
    if header is None:
      return
    if not chunks:
      msg = f"FASTA header {header!r} has no sequence"
      raise ValueError(msg)
    records.append((header, "".join(chunks)))
    header = None
    chunks = []

  for line_no, raw in enumerate(text.splitlines(), start=1):
    line = raw.strip()
    if not line:
      continue
    if line.startswith(">"):
      flush()
      header = line[1:].strip()
      chunks = []
      continue
    if header is None:
      msg = f"FASTA line {line_no} is not a header: {line!r}"
      raise ValueError(msg)
    chunks.append(line)
  flush()
  return tuple(records)


def load_optimize_fasta(path: Path, pdb_name: str) -> tuple[str, ...]:
  """FASTA entries whose header ``startswith(pdb_name)``, file order, ``:`` stripped."""
  if not path.is_file():
    msg = f"optimize_fasta does not exist: {path}"
    raise ValueError(msg)
  found: list[str] = []
  for header, seq in parse_fasta_records(path.read_text(encoding="utf-8")):
    if header.startswith(pdb_name):
      found.append(seq.strip().replace(":", ""))
  if not found:
    msg = f"optimize_fasta has no entry starting with {pdb_name}"
    raise ValueError(msg)
  return tuple(found)


class _Ready(NamedTuple):
  """One structure, padded, ready for the sample stage."""

  coords: np.ndarray
  present: np.ndarray
  residue_idx: np.ndarray
  chain_index: np.ndarray
  pad_valid: np.ndarray
  s_true: np.ndarray
  chain_mask: np.ndarray
  chain_m_pos: np.ndarray
  tie_groups: np.ndarray
  tied_beta: np.ndarray
  omit: np.ndarray
  bias: np.ndarray
  bias_by_res: np.ndarray
  pssm_coef: np.ndarray
  pssm_bias: np.ndarray
  pssm_log_odds_mask: np.ndarray
  omit_aa_mask: np.ndarray
  loaded: np.ndarray
  name: str
  l_total: int
  tied: bool


def build_tie_groups_np(
  tied_pos: tuple[tuple[int, ...], ...],
  l_total: int,
  l_pad: int,
) -> np.ndarray:
  """Listing-order groups, then singletons for every real row, then ``-1`` pad."""
  seen: set[int] = set()
  groups: list[tuple[int, ...]] = []
  for group in tied_pos:
    overlap = [index for index in group if index in seen]
    if overlap or len(set(group)) != len(group):
      msg = f"overlapping tied groups: {group}"
      raise ValueError(msg)
    if any(index < 0 or index >= l_total for index in group):
      msg = f"tied group {group} is outside 0..{l_total - 1}"
      raise ValueError(msg)
    seen.update(group)
    groups.append(tuple(int(index) for index in group))
  groups.extend((index,) for index in range(l_total) if index not in seen)
  largest = max((len(group) for group in groups), default=1)
  m_max = 1 if largest <= 1 else 1 << (largest - 1).bit_length()
  table = np.full((l_pad, m_max), -1, dtype=np.int32)
  for row, group in enumerate(groups):
    table[row, : len(group)] = group
  return table


def _split_bias(
  bias_spec: Any,  # noqa: ANN401
  length: int,
  bias_by_res: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Split ``spec.bias`` into a global amino-acid vector and a per-residue table.

  A 1-D vector whose length is the chain is per-position, including when that
  length is also 21. Rank-2 ``(L, 21)`` is per-position too. Flattening it and
  keeping the first 21 entries would publish position 0 as a global bias.
  """
  alphabet = len(_ALPHABET)
  global_bias = np.zeros((alphabet,), dtype=np.float32)
  by_res = np.array(bias_by_res, dtype=np.float32, copy=True)
  if bias_spec is None:
    return global_bias, by_res
  arr = np.asarray(bias_spec, dtype=np.float32)
  if arr.ndim == 1 and arr.shape[0] == length:
    by_res[:length] += arr[:, None]
    return global_bias, by_res
  if arr.ndim == 1 and arr.shape[0] == alphabet:
    return np.asarray(arr, dtype=np.float32), by_res
  if arr.shape == (length, alphabet):
    by_res[:length] += arr
    return global_bias, by_res
  msg = f"bias shape {arr.shape} is not ({alphabet},), ({length},), or ({length}, {alphabet})"
  raise ValueError(msg)


def prepare_sample(
  features: PottsFeatures,
  parsed_chains: tuple[tuple[str, str], ...],
  options: PottsMPNNOptions,
  spec: Any,  # noqa: ANN401
) -> _Ready:
  """Pad one featurized structure and resolve optimize-path sequences."""
  l_pad = int(features.L_total)
  padded, pad_valid = pad(features, l_pad)
  # ``pad`` returns the caller's arrays when nothing is appended. Copy before
  # writing fixed positions so the featurized structure stays unchanged.
  chain_m_pos = np.asarray(padded.chain_m_pos, dtype=np.float32).copy()
  fixed = getattr(spec, "fixed_positions", None)
  if fixed is not None:
    for index in np.asarray(fixed, dtype=np.int32).reshape(-1):
      if 0 <= int(index) < chain_m_pos.shape[0]:
        chain_m_pos[int(index)] = 0.0
  omit = np.zeros((len(_ALPHABET),), dtype=np.float32)
  for index in omit_letter_indices(tuple(getattr(spec, "omit_aa", ()) or ())):
    omit[index] = 1.0
  bias, bias_by_res = _split_bias(
    getattr(spec, "bias", None),
    features.L_total,
    np.asarray(padded.bias_by_res, dtype=np.float32),
  )
  threshold = float(options.pssm_threshold)
  log_odds = np.asarray(padded.pssm_log_odds, dtype=np.float32)
  native = "".join(sequence for _letter, sequence in parsed_chains)
  loaded_rows: list[np.ndarray] = []
  optimizing = bool(options.optimize_pdb or options.optimize_fasta) and (
    options.optimization_mode != "none"
  )
  if optimizing and options.optimize_fasta:
    for sequence in load_optimize_fasta(Path(options.optimize_fasta), features.name):
      if len(sequence) != features.L_total:
        msg = f"optimize_fasta sequence length {len(sequence)} != L_total {features.L_total}"
        raise ValueError(msg)
      loaded_rows.append(seq_to_ints(sequence))
  elif optimizing and options.optimize_pdb:
    if len(native) != features.L_total:
      msg = f"optimize_pdb sequence length {len(native)} != L_total {features.L_total}"
      raise ValueError(msg)
    loaded_rows.append(seq_to_ints(native))
  loaded = (
    np.stack(loaded_rows).astype(np.int32) if loaded_rows else np.zeros((0, l_pad), dtype=np.int32)
  )
  return _Ready(
    coords=np.asarray(padded.x, dtype=np.float32),
    present=np.asarray(padded.present, dtype=np.float32),
    residue_idx=np.asarray(padded.residue_idx, dtype=np.int32),
    chain_index=np.asarray(padded.chain_encoding, dtype=np.int32),
    pad_valid=np.asarray(pad_valid, dtype=np.bool_),
    s_true=np.asarray(padded.s, dtype=np.int32),
    chain_mask=np.asarray(padded.chain_m, dtype=np.float32),
    chain_m_pos=chain_m_pos,
    tie_groups=build_tie_groups_np(features.tied_pos, features.L_total, l_pad),
    tied_beta=np.asarray(padded.tied_beta, dtype=np.float32),
    omit=omit,
    bias=bias,
    bias_by_res=bias_by_res,
    pssm_coef=np.asarray(padded.pssm_coef, dtype=np.float32),
    pssm_bias=np.asarray(padded.pssm_bias, dtype=np.float32),
    pssm_log_odds_mask=(log_odds > threshold).astype(np.float32),
    omit_aa_mask=np.asarray(padded.omit_aa_mask, dtype=np.float32),
    loaded=loaded,
    name=features.name,
    l_total=features.L_total,
    tied=bool(features.tied_pos),
  )


def sample_axes(spec: Any, batch: FamilyBatch) -> list[AxisSpec]:  # noqa: ANN401
  """``samples`` and ``temperatures``, or loaded sequences on the optimize path."""
  options = _options(spec)
  ready = cast("tuple[_Ready, ...]", batch.arrays.get("ready", ()))
  optimizing = bool(options.optimize_pdb or options.optimize_fasta) and (
    options.optimization_mode != "none"
  )
  if optimizing and ready:
    n_samples = max(int(ready[0].loaded.shape[0]), 1)
    n_temp = 1
  else:
    n_samples = max(int(getattr(spec, "num_samples", 1) or 1), 1)
    temps = spec.run_spec.sampling.temperature
    n_temp = max(len(temps), 1)
  specs = [
    AxisSpec(
      name="structures",
      cardinality=max(len(batch.input_indices), 1),
      default_batch_size=1,
      heterogeneous=True,
      bucket_boundaries=LENGTH_BUCKETS,
    ),
    AxisSpec(name="samples", cardinality=n_samples, default_batch_size=n_samples),
  ]
  if not optimizing:
    specs.append(
      AxisSpec(name="temperatures", cardinality=n_temp, default_batch_size=n_temp),
    )
  BatchPlanner().plan(specs)
  return specs


def sample_schema(spec: Any) -> dict[str, SinkArraySpec]:  # noqa: ANN401
  """§4.5 sample arrays. ``sample_rank`` is structure-level."""
  options = _options(spec)
  optimizing = bool(options.optimize_pdb or options.optimize_fasta) and (
    options.optimization_mode != "none"
  )
  sequence = SinkArraySpec(dims=("N", "L_total"), dtype="int32", attrs={})
  if optimizing:
    schema: dict[str, SinkArraySpec] = {"refined_sequence": sequence}
  else:
    schema = {
      "sequence": sequence,
      "sample_energy": SinkArraySpec(dims=("N",), dtype="float32", attrs={}),
      "sample_rank": SinkArraySpec(
        dims=("N",),
        dtype="int32",
        attrs={"level": "structure"},
      ),
    }
    if options.optimization_mode != "none":
      schema["refined_sequence"] = sequence
  if options.emit_etab:
    schema["potts_etab"] = SinkArraySpec(
      dims=("L_total", "K", "20", "20"),
      dtype="float32",
      attrs={"etab_convention": "forward_denom2"},
    )
    schema["potts_E_idx"] = SinkArraySpec(dims=("L_total", "K"), dtype="int32", attrs={})
  return schema


class SampleStages:
  """Per-chunk sample body. One structure is encoded once per call."""

  def __init__(self, model: PottsMPNN, spec: Any) -> None:  # noqa: ANN401
    self._model = model
    self._spec = spec
    self._options = _options(spec)

  def __call__(
    self,
    batch: FamilyBatch,
    *,
    chunk_start: int,
    chunk_count: int,
    scalar_dropout: bool,
    dropout_key: jax.Array,
  ) -> Mapping[str, jax.Array]:
    del scalar_dropout
    rows = [
      self._one(ready, chunk_start, chunk_count, dropout_key) for ready in batch.arrays["ready"]
    ]
    keys = rows[0].keys()
    return {key: jnp.stack([row[key] for row in rows]) for key in keys}

  def _one(
    self,
    ready: _Ready,
    chunk_start: int,
    chunk_count: int,
    dropout_key: jax.Array,
  ) -> dict[str, jax.Array]:
    options = self._options
    encoded = _encode(
      self._model,
      jnp.asarray(ready.coords),
      jnp.asarray(ready.present),
      jnp.asarray(ready.residue_idx),
      jnp.asarray(ready.chain_index),
      jnp.asarray(ready.pad_valid),
    )
    h_v, h_e, e_idx, forward, energy_table = encoded
    optimizing = bool(options.optimize_pdb or options.optimize_fasta) and (
      options.optimization_mode != "none"
    )
    tables = sample_binding_tables(
      options,
      h_v.shape[0],
      energy_table.shape[-1],
      energy_table.dtype,
    )
    if optimizing:
      refined = _refine_loaded(
        self._model,
        ready,
        h_v,
        h_e,
        e_idx,
        forward,
        tables,
        options,
        dropout_key,
      )
      produced: dict[str, jax.Array] = {"refined_sequence": refined}
    else:
      produced = _decode_chunk(
        self._model,
        ready,
        h_v,
        h_e,
        e_idx,
        forward,
        energy_table,
        tables,
        options,
        self._spec,
        chunk_start,
        chunk_count,
        dropout_key,
      )
    if options.emit_etab:
      k_eff = min(48, ready.l_total)
      produced["potts_etab"] = forward[:, :k_eff].astype(jnp.float32)
      produced["potts_E_idx"] = e_idx[:, :k_eff]
    return produced


def _decode_chunk(
  model: PottsMPNN,
  ready: _Ready,
  h_v: Array,
  h_e: Array,
  e_idx: Array,
  forward: Array,
  energy_table: Array,
  tables: BindingTables,
  options: PottsMPNNOptions,
  spec: Any,  # noqa: ANN401
  chunk_start: int,
  chunk_count: int,
  dropout_key: jax.Array,
) -> dict[str, jax.Array]:
  temperatures = tuple(
    floor_temperature(float(value))
    for value in spec.run_spec.sampling.temperature
    if value is not None
  )
  if not temperatures:
    temperatures = (0.1,)
  opt_temperature = floor_temperature(float(options.optimization_temperature))
  sequences: list[Array] = []
  energies: list[Array] = []
  refined: list[Array] = []
  decode = PottsARDecode(
    layers=model.mpnn.decoder.layers,
    w_s_embed=model.mpnn.w_s_embed,
    w_out=model.mpnn.w_out,
  )
  refiner = PottsRefine(
    layers=model.mpnn.decoder.layers,
    w_s_embed=model.mpnn.w_s_embed,
    w_out=model.mpnn.w_out,
  )
  scorer = PottsSampleEnergy()
  num_samples = int(getattr(spec, "num_samples", 1) or 1)
  for offset in range(chunk_count):
    for temp_index, temperature in enumerate(temperatures):
      key = jax.random.fold_in(dropout_key, (chunk_start + offset) * len(temperatures) + temp_index)
      randn_key, uniform_key, refine_key = jax.random.split(key, 3)
      length = h_v.shape[0]
      randn = jax.random.normal(randn_key, (length,), dtype=h_v.dtype)
      uniforms = jax.random.uniform(uniform_key, (length,), dtype=h_v.dtype)
      decoded = decode(
        h_v,
        h_e,
        e_idx,
        jnp.asarray(ready.present),
        jnp.asarray(ready.pad_valid),
        jnp.asarray(ready.s_true),
        jnp.asarray(ready.chain_mask),
        jnp.asarray(ready.chain_m_pos),
        jnp.asarray(ready.tie_groups),
        jnp.asarray(ready.tied_beta),
        randn,
        uniforms,
        jnp.asarray(ready.omit),
        jnp.asarray(ready.bias),
        jnp.asarray(ready.bias_by_res),
        jnp.asarray(ready.pssm_coef),
        jnp.asarray(ready.pssm_bias),
        jnp.asarray(ready.pssm_log_odds_mask),
        jnp.asarray(ready.omit_aa_mask),
        temperature=temperature,
        pssm_multi=float(options.pssm_multi),
        pssm_bias_flag=bool(options.pssm_bias_flag),
        pssm_log_odds_flag=bool(options.pssm_log_odds_flag),
      )
      sequences.append(decoded.sequence)
      energies.append(scorer(energy_table, e_idx, jnp.asarray(ready.pad_valid), decoded.sequence))
      if options.optimization_mode != "none":
        refine_randn = jax.random.normal(refine_key, (length,), dtype=h_v.dtype)
        order = upstream_refine_order(
          np.asarray(decoded.decoding_order),
          ready.chain_mask,
          np.asarray(refine_randn),
          num_samples=num_samples,
          chain_suffix="",
          stored_orders_present=True,
        )
        refined.append(
          _run_refine(
            refiner,
            decoded.sequence,
            ready,
            h_v,
            h_e,
            e_idx,
            forward,
            tables,
            options,
            order,
            _refine_uniforms(refine_key, length, options.optimization_mode, h_v.dtype),
            opt_temperature,
          ),
        )
  produced = {
    "sequence": jnp.stack(sequences),
    "sample_energy": jnp.stack(energies).astype(jnp.float32),
  }
  if refined:
    produced["refined_sequence"] = jnp.stack(refined)
  return produced


def _refine_loaded(
  model: PottsMPNN,
  ready: _Ready,
  h_v: Array,
  h_e: Array,
  e_idx: Array,
  forward: Array,
  tables: BindingTables,
  options: PottsMPNNOptions,
  dropout_key: jax.Array,
) -> Array:
  refiner = PottsRefine(
    layers=model.mpnn.decoder.layers,
    w_s_embed=model.mpnn.w_s_embed,
    w_out=model.mpnn.w_out,
  )
  opt_temperature = floor_temperature(float(options.optimization_temperature))
  rows: list[Array] = []
  length = h_v.shape[0]
  for index in range(ready.loaded.shape[0]):
    key = jax.random.fold_in(dropout_key, index)
    randn = jax.random.normal(key, (length,), dtype=h_v.dtype)
    order = upstream_refine_order(
      np.arange(length, dtype=np.int32),
      ready.chain_mask,
      np.asarray(randn),
      num_samples=1,
      chain_suffix="",
      stored_orders_present=False,
    )
    rows.append(
      _run_refine(
        refiner,
        jnp.asarray(ready.loaded[index]),
        ready,
        h_v,
        h_e,
        e_idx,
        forward,
        tables,
        options,
        order,
        _refine_uniforms(key, length, options.optimization_mode, h_v.dtype),
        opt_temperature,
      ),
    )
  return jnp.stack(rows)


def _run_refine(
  refiner: PottsRefine,
  sequence: Array,
  ready: _Ready,
  h_v: Array,
  h_e: Array,
  e_idx: Array,
  forward: Array,
  tables: BindingTables,
  options: PottsMPNNOptions,
  order: np.ndarray,
  uniforms: Array,
  temperature: float,
) -> Array:
  if ready.tied and options.binding_energy_optimization == "only" and not options.tied_epistasis:
    check_tied_only_groups(
      jnp.asarray(ready.tie_groups),
      tables.inter_mask,
      tied_epistasis=False,
      binding="only",
    )
  max_iters = _refine_sweep_cap(options.optimization_mode)
  uniforms = _require_refine_uniforms(uniforms, options.optimization_mode)
  result = refiner(
    options.optimization_mode,
    sequence.astype(jnp.int32),
    pad_etab_energy(forward),
    e_idx,
    jnp.asarray(ready.pad_valid),
    jnp.asarray(ready.present),
    jnp.asarray(ready.chain_mask),
    jnp.asarray(ready.chain_m_pos),
    jnp.asarray(order, dtype=jnp.int32),
    uniforms,
    jnp.asarray(ready.omit),
    jnp.asarray(ready.bias),
    jnp.asarray(ready.bias_by_res),
    jnp.asarray(ready.pssm_coef),
    jnp.asarray(ready.pssm_bias),
    jnp.asarray(ready.pssm_log_odds_mask),
    jnp.asarray(ready.omit_aa_mask),
    jnp.asarray(ready.tie_groups),
    jnp.asarray(ready.tied_beta),
    h_v,
    h_e,
    tables,
    temperature=temperature,
    pssm_multi=float(options.pssm_multi),
    pssm_bias_flag=bool(options.pssm_bias_flag),
    pssm_log_odds_flag=bool(options.pssm_log_odds_flag),
    binding=options.binding_energy_optimization,
    tied=ready.tied,
    tied_epistasis=bool(options.tied_epistasis),
    max_iters=max_iters,
  )
  return result.sequence


def _encode(
  model: PottsMPNN,
  coords: Array,
  present: Array,
  residue_idx: Array,
  chain_index: Array,
  pad_valid: Array,
) -> tuple[Array, Array, Array, Array, Array]:
  dtype = coords.dtype
  scored = cast_floating(model, dtype)
  present_f = present.astype(dtype)
  valid = pad_valid.astype(jnp.bool_)
  nodes, edges, neighbor_indices = _encoder_states(
    scored.mpnn,
    coords.astype(dtype),
    present_f,
    residue_idx.astype(jnp.int32),
    chain_index.astype(jnp.int32),
    inference=True,
  )
  raw = scored.potts_head(edges[-1], neighbor_indices, present_f, valid)
  forward = merge_pair(raw, neighbor_indices, valid, denom=2, exclude_self=False)
  table = merge_pair(forward, neighbor_indices, valid, denom=4, exclude_self=True)
  return nodes[-1], edges[-1], neighbor_indices, forward, pad_etab_energy(table)


def _refine_sweep_cap(mode: str) -> int:
  """``potts_converge`` is capped at 1000 sweeps; every other mode is one sweep."""
  return 1000 if mode == "potts_converge" else 1


def _require_refine_uniforms(uniforms: Array, mode: str) -> Array:
  """Reject a uniform table shorter than the sweep cap.

  Zero-padding the tail makes ``categorical_draw`` see ``u=0``, which selects
  the first positive bin for every later sweep.
  """
  max_iters = _refine_sweep_cap(mode)
  if int(uniforms.shape[0]) < max_iters:
    msg = (
      f"refine uniforms have {int(uniforms.shape[0])} rows but "
      f"optimization_mode={mode!r} runs up to {max_iters} sweeps"
    )
    raise ValueError(msg)
  return uniforms


def _refine_uniforms(key: jax.Array, length: int, mode: str, dtype: jnp.dtype) -> Array:
  """One uniform row per sweep. Converge indexes ``uniforms[iteration, cursor]``."""
  rows = _refine_sweep_cap(mode) if mode == "potts_converge" else 8
  return jax.random.uniform(key, (rows, length), dtype=dtype)


def sample_binding_tables(
  options: PottsMPNNOptions,
  length: int,
  alphabet: int,
  dtype: jnp.dtype,
) -> BindingTables:
  """Binding tables for one sample.

  ``none`` keeps a zero interface so refine uses the complex energy. ``both``
  and ``only`` need partition etabs; a zero table would leave ``both`` identical
  to ``none`` and would update no positions under ``only``.
  """
  if options.binding_energy_optimization == "none":
    return _dummy_binding(length, alphabet, dtype)
  msg = (
    "binding_energy_optimization="
    f"{options.binding_energy_optimization!r} needs partition etabs and an "
    "interface mask; the sample path will not refine against a zero binding table"
  )
  raise ValueError(msg)


def _dummy_binding(length: int, alphabet: int, dtype: jnp.dtype) -> BindingTables:
  return BindingTables(
    etab=jnp.zeros((1, 1, 1, alphabet, alphabet), dtype=dtype),
    e_idx=jnp.zeros((1, 1, 1), dtype=jnp.int32),
    pad_valid=jnp.ones((1, 1), dtype=jnp.bool_),
    complex_index=jnp.zeros((1, 1), dtype=jnp.int32),
    partition_of=jnp.zeros((length,), dtype=jnp.int32),
    local_of=jnp.zeros((length,), dtype=jnp.int32),
    inter_mask=jnp.zeros((length,), dtype=jnp.bool_),
  )


def _options(spec: Any) -> PottsMPNNOptions:  # noqa: ANN401
  options = getattr(spec, "potts_mpnn", None)
  if options is None:
    return PottsMPNNOptions()
  return cast("PottsMPNNOptions", options)
