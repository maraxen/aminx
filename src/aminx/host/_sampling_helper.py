from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from aminx.run.specs import SamplingSpecification
from aminx.utils.data_structures import Protein
from aminx.utils.decoding_order import DecodingOrderFn, random_decoding_order

LIGAND_PLACEHOLDER_ATOMS = 1
LIGAND_CONTEXT_KEYS = ("Y", "Y_t", "Y_m")
AMINO_ACID_VOCAB_SIZE = 21
RANK_WITH_TEMPERATURE = 4


_DEFAULT_DECODING_ORDER_FN = cast("DecodingOrderFn", random_decoding_order)


def _canonical_structure_id(input_item: Any, index: int) -> str:
  if isinstance(input_item, Path):
    source = str(input_item)
  elif isinstance(input_item, str):
    source = input_item
  else:
    source_name = getattr(input_item, "name", None)
    source = str(source_name) if source_name is not None else ""
  stem = Path(source).stem if source else ""
  if stem:
    return stem
  return f"structure_{index}"


def _canonical_structure_ids_for_spec(spec: SamplingSpecification) -> list[str]:
  raw_inputs = spec.inputs
  if isinstance(raw_inputs, (str, Path)) or hasattr(raw_inputs, "read"):
    input_items = [raw_inputs]
  else:
    try:
      input_items = list(raw_inputs)
    except TypeError:
      input_items = [raw_inputs]
  return [_canonical_structure_id(item, idx) for idx, item in enumerate(input_items)]


def _structure_ids_for_batch(
  canonical_ids: list[str],
  *,
  structure_offset: int,
  batch_size: int,
) -> list[str]:
  structure_ids: list[str] = []
  for local_idx in range(batch_size):
    global_idx = structure_offset + local_idx
    if global_idx < len(canonical_ids):
      structure_ids.append(canonical_ids[global_idx])
    else:
      structure_ids.append(f"structure_{global_idx}")
  return structure_ids


def _split_ligand_payload_key(payload_key: str) -> tuple[str, str] | None:
  for tensor_name in LIGAND_CONTEXT_KEYS:
    for separator in ("::", "/"):
      suffix = f"{separator}{tensor_name}"
      if payload_key.endswith(suffix):
        structure_id = payload_key[: -len(suffix)]
        return (structure_id, tensor_name)
  return None


def _normalize_keyed_ligand_array(
  value: Any,  # noqa: ANN401
  *,
  expected_ndim: int,
  key: str,
) -> np.ndarray:
  arr = np.asarray(value)
  if arr.ndim != expected_ndim:
    msg = f"Ligand context key '{key}' must have rank {expected_ndim}, got rank {arr.ndim}"
    raise ValueError(msg)
  if expected_ndim == 3:
    if arr.shape[2] != 3:
      msg = f"Ligand context key '{key}' must have shape (N, atoms, 3), got shape {arr.shape}"
      raise ValueError(msg)
  return arr


def _normalize_ligand_tensor(
  value: Any,  # noqa: ANN401
  *,
  batch_size: int,
  seq_len: int,
  target_rank: int,
  dtype: jnp.dtype,
  name: str,
) -> jax.Array:
  arr = jnp.asarray(value, dtype=dtype)
  if target_rank == 4:
    if arr.ndim == 3:
      arr = jnp.broadcast_to(arr[None, ...], (batch_size, *arr.shape))
    elif arr.ndim == 4 and arr.shape[0] == 1 and batch_size > 1:
      arr = jnp.broadcast_to(arr, (batch_size, *arr.shape[1:]))
  elif target_rank == 3:
    if arr.ndim == 2:
      arr = jnp.broadcast_to(arr[None, ...], (batch_size, *arr.shape))
    elif arr.ndim == 3 and arr.shape[0] == 1 and batch_size > 1:
      arr = jnp.broadcast_to(arr, (batch_size, *arr.shape[1:]))
  if arr.ndim != target_rank:
    msg = f"{name} must have rank {target_rank}, got rank {arr.ndim}"
    raise ValueError(msg)
  if arr.shape[0] != batch_size:
    msg = f"{name} batch mismatch: expected {batch_size}, got {arr.shape[0]}"
    raise ValueError(msg)
  if arr.shape[1] != seq_len:
    msg = f"{name} sequence length mismatch: expected {seq_len}, got {arr.shape[1]}"
    raise ValueError(msg)
  return arr


def _load_ligand_context_file(
  ligand_context_path: Path,
  *,
  canonical_structure_ids: Sequence[str] | None,
  batch_structure_ids: Sequence[str] | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  with np.load(ligand_context_path, allow_pickle=False) as npz_data:
    canonical_ids = (
      list(canonical_structure_ids)
      if canonical_structure_ids is not None
      else (list(batch_structure_ids) if batch_structure_ids is not None else None)
    )
    selected_ids = (
      list(batch_structure_ids)
      if batch_structure_ids is not None
      else (list(canonical_ids) if canonical_ids is not None else None)
    )
    keyed_payload: dict[str, dict[str, np.ndarray]] = {}
    parsed_keys: set[str] = set()
    for file_key in npz_data.files:
      split = _split_ligand_payload_key(file_key)
      if split is None:
        continue
      structure_id, tensor_name = split
      keyed_payload.setdefault(structure_id, {})[tensor_name] = np.asarray(npz_data[file_key])
      parsed_keys.add(file_key)

    if keyed_payload:
      canonical_ids = canonical_ids if canonical_ids is not None else sorted(keyed_payload)
      selected_ids = selected_ids if selected_ids is not None else list(canonical_ids)
      payload_ids = set(keyed_payload)
      canonical_id_set = set(canonical_ids)
      missing_ids = sorted(canonical_id_set - payload_ids)
      extra_ids = sorted(payload_ids - canonical_id_set)
      if missing_ids or extra_ids:
        msg = (
          "Ligand context keys must exactly match canonical structure IDs. "
          f"missing keys: {missing_ids}; extra keys: {extra_ids}."
        )
        raise ValueError(msg)
      unexpected_keys = sorted(set(npz_data.files) - parsed_keys)
      if unexpected_keys:
        msg = (
          f"Ligand context file '{ligand_context_path}' has unsupported keys: "
          f"{unexpected_keys}. Expected only <structure_id>::Y, <structure_id>::Y_t, "
          "<structure_id>::Y_m entries."
        )
        raise ValueError(msg)
      missing_tensor_keys = [
        f"{structure_id}:{tensor_name}"
        for structure_id in canonical_ids
        for tensor_name in LIGAND_CONTEXT_KEYS
        if tensor_name not in keyed_payload[structure_id]
      ]
      if missing_tensor_keys:
        msg = (
          f"Ligand context file '{ligand_context_path}' is missing keyed tensors: "
          f"{missing_tensor_keys}"
        )
        raise ValueError(msg)
      y_items = [
        _normalize_keyed_ligand_array(
          keyed_payload[structure_id]["Y"],
          expected_ndim=3,
          key=f"{structure_id}::Y",
        )
        for structure_id in selected_ids
      ]
      y_t_items = [
        _normalize_keyed_ligand_array(
          keyed_payload[structure_id]["Y_t"],
          expected_ndim=2,
          key=f"{structure_id}::Y_t",
        )
        for structure_id in selected_ids
      ]
      y_m_items = [
        _normalize_keyed_ligand_array(
          keyed_payload[structure_id]["Y_m"],
          expected_ndim=2,
          key=f"{structure_id}::Y_m",
        )
        for structure_id in selected_ids
      ]
      return (
        jnp.asarray(np.stack(y_items, axis=0)),
        jnp.asarray(np.stack(y_t_items, axis=0)),
        jnp.asarray(np.stack(y_m_items, axis=0)),
      )

    missing = [key for key in (*LIGAND_CONTEXT_KEYS, "structure_ids") if key not in npz_data]
    if missing:
      msg = (
        f"Ligand context file '{ligand_context_path}' is missing required keys: "
        f"{', '.join(missing)}"
      )
      raise ValueError(msg)

    payload_ids = [str(item) for item in np.asarray(npz_data["structure_ids"]).tolist()]
    canonical_ids = canonical_ids if canonical_ids is not None else payload_ids
    selected_ids = selected_ids if selected_ids is not None else canonical_ids
    payload_id_set = set(payload_ids)
    canonical_id_set = set(canonical_ids)
    missing_ids = sorted(canonical_id_set - payload_id_set)
    extra_ids = sorted(payload_id_set - canonical_id_set)
    if missing_ids or extra_ids:
      msg = (
        "Ligand context keys must exactly match canonical structure IDs. "
        f"missing keys: {missing_ids}; extra keys: {extra_ids}."
      )
      raise ValueError(msg)

    id_to_index = {structure_id: idx for idx, structure_id in enumerate(payload_ids)}
    gather_indices = np.asarray(
      [id_to_index[structure_id] for structure_id in selected_ids],
      dtype=np.int32,
    )
    return (
      jnp.asarray(np.asarray(npz_data["Y"])[gather_indices]),
      jnp.asarray(np.asarray(npz_data["Y_t"])[gather_indices]),
      jnp.asarray(np.asarray(npz_data["Y_m"])[gather_indices]),
    )


def _prepare_ligand_context(
  spec: SamplingSpecification,
  batched_ensemble: Protein,
  batch_size: int,
  seq_len: int,
  canonical_structure_ids: Sequence[str] | None = None,
  batch_structure_ids: Sequence[str] | None = None,
) -> dict[str, jax.Array | None]:
  if spec.model_family != "ligandmpnn":
    return {
      "Y": None,
      "Y_t": None,
      "Y_m": None,
      "atom_37": None,
      "atom_37_mask": None,
      "chain_mask": None,
    }

  Y = getattr(batched_ensemble, "Y", None)
  Y_t = getattr(batched_ensemble, "Y_t", None)
  Y_m = getattr(batched_ensemble, "Y_m", None)

  if spec.ligand_context_path is not None:
    file_Y, file_Y_t, file_Y_m = _load_ligand_context_file(
      spec.ligand_context_path,
      canonical_structure_ids=canonical_structure_ids,
      batch_structure_ids=batch_structure_ids,
    )
    Y = file_Y
    Y_t = file_Y_t
    Y_m = file_Y_m

  if Y is None or Y_t is None or Y_m is None:
    if spec.ligand_conditioning:
      msg = (
        "ligand_conditioning=True requires ligand context tensors (Y, Y_t, Y_m) "
        "on the batch or via ligand_context_path."
      )
      raise ValueError(msg)
    Y = jnp.zeros((batch_size, seq_len, LIGAND_PLACEHOLDER_ATOMS, 3), dtype=jnp.float32)
    Y_t = jnp.zeros((batch_size, seq_len, LIGAND_PLACEHOLDER_ATOMS), dtype=jnp.int32)
    Y_m = jnp.zeros((batch_size, seq_len, LIGAND_PLACEHOLDER_ATOMS), dtype=jnp.float32)

  Y = _normalize_ligand_tensor(
    Y,
    batch_size=batch_size,
    seq_len=seq_len,
    target_rank=4,
    dtype=jnp.float32,
    name="Y",
  )
  Y_t = _normalize_ligand_tensor(
    Y_t,
    batch_size=batch_size,
    seq_len=seq_len,
    target_rank=3,
    dtype=jnp.int32,
    name="Y_t",
  )
  Y_m = _normalize_ligand_tensor(
    Y_m,
    batch_size=batch_size,
    seq_len=seq_len,
    target_rank=3,
    dtype=jnp.float32,
    name="Y_m",
  )

  if not spec.sidechain_conditioning:
    return {
      "Y": Y,
      "Y_t": Y_t,
      "Y_m": Y_m,
      "atom_37": None,
      "atom_37_mask": None,
      "chain_mask": None,
    }

  atom_37 = jnp.asarray(batched_ensemble.coordinates, dtype=jnp.float32)
  if atom_37.ndim != 4 or atom_37.shape[0] != batch_size or atom_37.shape[1] != seq_len:
    msg = "coordinates must have shape (batch, residues, atoms, xyz) for sidechain conditioning."
    raise ValueError(msg)

  atom_37_mask_source = (
    batched_ensemble.atom_mask
    if batched_ensemble.atom_mask is not None
    else batched_ensemble.full_atom_mask
  )
  if atom_37_mask_source is None:
    msg = "sidechain_conditioning=True requires atom_mask or full_atom_mask."
    raise ValueError(msg)
  atom_37_mask = jnp.asarray(atom_37_mask_source, dtype=jnp.float32)
  if atom_37_mask.ndim != 3 or atom_37_mask.shape[0] != batch_size or atom_37_mask.shape[1] != seq_len:
    msg = "atom mask must have shape (batch, residues, atoms) for sidechain conditioning."
    raise ValueError(msg)

  # Compute chain_mask: 1=designable, 0=fixed.
  # Priority order: chain_mask_fixed > fixed_mask complement > all-ones fallback.
  # Convention: fixed_mask 1=fixed, chain_mask 1=designable → complement.
  if getattr(spec, "chain_mask_fixed", None) is not None:
    chain_mask = _broadcast_per_structure(
      spec.chain_mask_fixed,
      batch_size=batch_size,
      expected_len=seq_len,
      dtype=jnp.float32,
      name="chain_mask_fixed",
    )
    if chain_mask.dtype != jnp.float32:
      msg = f"chain_mask_fixed must be float32 after broadcast, got {chain_mask.dtype}"
      raise TypeError(msg)
  elif spec.fixed_mask is not None:
    fixed_mask_np = _broadcast_per_structure(
      spec.fixed_mask,
      batch_size=batch_size,
      expected_len=seq_len,
      dtype=jnp.float32,
      name="fixed_mask",
    )
    chain_mask = 1.0 - fixed_mask_np
    if chain_mask.dtype != jnp.float32:
      msg = f"chain_mask must be float32 after complement, got {chain_mask.dtype}"
      raise TypeError(msg)
  else:
    chain_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)

  return {
    "Y": Y,
    "Y_t": Y_t,
    "Y_m": Y_m,
    "atom_37": atom_37,
    "atom_37_mask": atom_37_mask,
    "chain_mask": chain_mask,
  }


def _noop_sampling_structure_batch_io(batch_idx: object, batch_count: object) -> None:
  """Host structure-batch boundary hook for sampling (Phase **5g** PR3a).

  ``jax.experimental.io_callback`` invokes this on the host with ``ordered=False``;
  completion order is **not** guaranteed to match program order. Only lightweight
  scalar markers cross here — sequences / logits stay on device until list concat or
  HDF5 / ArrayRecord materialization (see ``TODO_io_callback.txt``).

  Emitted **once per protein-iterator batch** when ``emit_structure_batch_io=True``
  (campaign paths chunk ``_sample_batch`` multiple times per ensemble — only the
  **last** chunk sets ``emit_structure_batch_io=True``).

  ``batch_count`` is ``-1`` when the protein iterator does not expose ``__len__``.

  Default implementation is a no-op so production pays minimal overhead; tests may
  monkeypatch this symbol **before** the first traced ``_sample_batch`` call.
  """
  del batch_idx, batch_count


def _noop_sampling_chunk_io(chunk_idx: object, chunk_count: object) -> None:
  """Host chunk-boundary hook for sampling (Phase 5g PR1).

  ``jax.experimental.io_callback`` invokes this on the host with ``ordered=False``.
  Lightweight marker to signal a chunk is done.
  """
  del chunk_idx, chunk_count


def _dispatch_sampling_tensor_batch_io(
  batch_idx: object,
  batch_count: object,
  chunk_start: object,
  chunk_count: object,
  sequences_host: object,
  logits_host: object,
) -> None:
  """Host tensor-batch boundary hook (Phase 5g PR4).

  Drains device tensors to the active :func:`streaming_tensor_sink_session`.
  """
  from aminx.host.output_sinks import active_sampling_staging_sink

  sink = active_sampling_staging_sink()
  if sink is not None:
    sink.on_sampling_sequences_logits(
      batch_idx,
      batch_count,
      chunk_start,
      chunk_count,
      sequences_host,
      logits_host,
    )


def _broadcast_per_structure(
  value: Any,  # noqa: ANN401
  *,
  batch_size: int,
  expected_len: int,
  dtype: jnp.dtype,
  name: str,
) -> jax.Array:
  arr = jnp.asarray(value, dtype=dtype)
  if arr.ndim == 1:
    if arr.shape[0] != expected_len:
      msg = f"{name} length mismatch: expected {expected_len}, got {arr.shape[0]}"
      raise ValueError(msg)
    return jnp.broadcast_to(arr[None, :], (batch_size, expected_len))
  if arr.ndim == 2:
    if arr.shape[1] != expected_len:
      msg = f"{name} length mismatch: expected {expected_len}, got {arr.shape[1]}"
      raise ValueError(msg)
    if arr.shape[0] == 1 and batch_size > 1:
      return jnp.broadcast_to(arr, (batch_size, expected_len))
    if arr.shape[0] != batch_size:
      msg = f"{name} batch mismatch: expected {batch_size}, got {arr.shape[0]}"
      raise ValueError(msg)
    return arr
  msg = f"{name} must have rank 1 or 2, got rank {arr.ndim}"
  raise ValueError(msg)


def _prepare_fixed_controls(
  spec: SamplingSpecification,
  *,
  batched_ensemble: Protein,
) -> tuple[jax.Array, jax.Array]:
  batch_size, seq_len = batched_ensemble.coordinates.shape[:2]

  fixed_mask_np = np.zeros((batch_size, seq_len), dtype=np.float32)
  fixed_tokens_np = np.zeros((batch_size, seq_len), dtype=np.int32)

  if spec.fixed_mask is not None:
    fixed_mask_np = np.asarray(spec.fixed_mask, dtype=np.float32)
    if fixed_mask_np.ndim == 1:
      fixed_mask_np = np.broadcast_to(
          fixed_mask_np[None, :], (batch_size, seq_len),
      ).copy()
    if fixed_mask_np.shape != (batch_size, seq_len):
      msg = f"fixed_mask must have shape ({batch_size}, {seq_len}), got {fixed_mask_np.shape}"
      raise ValueError(msg)

  if spec.fixed_positions is not None:
    fixed_pos = np.asarray(spec.fixed_positions, dtype=np.float32)
    fixed_pos_mask = _broadcast_per_structure(
      fixed_pos,
      batch_size=batch_size,
      expected_len=seq_len,
      dtype=jnp.float32,
      name="fixed_positions",
    )
    # Union: combine fixed_positions with fixed_mask (if both are set)
    fixed_mask_np = np.maximum(fixed_mask_np, fixed_pos_mask)

  if spec.fixed_tokens is not None:
    fixed_tok = np.asarray(spec.fixed_tokens, dtype=np.int32)
    fixed_tokens_np = _broadcast_per_structure(
      fixed_tok,
      batch_size=batch_size,
      expected_len=seq_len,
      dtype=jnp.int32,
      name="fixed_tokens",
    )

    invalid_tokens = (fixed_tokens_np < 0) | (fixed_tokens_np >= AMINO_ACID_VOCAB_SIZE)
    if np.any(invalid_tokens & (fixed_mask_np > 0)):
      msg = f"fixed_tokens must be in [0, {AMINO_ACID_VOCAB_SIZE - 1}] at masked positions."
      raise ValueError(msg)

  if spec.fixed_mask is not None:
    fm_broadcast = _broadcast_per_structure(
      np.asarray(spec.fixed_mask, dtype=np.float32),
      batch_size=batch_size,
      expected_len=seq_len,
      dtype=jnp.float32,
      name="fixed_mask",
    )
    fixed_mask_np = jnp.maximum(jnp.asarray(fixed_mask_np), fm_broadcast)

  return jnp.asarray(fixed_mask_np), jnp.asarray(fixed_tokens_np)
