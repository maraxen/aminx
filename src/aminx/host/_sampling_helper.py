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


def ligand_conditioning_mode(ligand_conditioning: bool | None) -> str:
  """Name the tri-state ``SamplingSpecification.ligand_conditioning`` for provenance.

  ``"auto"`` (None: use whatever is present), ``"required"`` (True), ``"ablated"`` (False).
  The legacy ``ligand_conditioning`` store attribute is ``int(bool(...))`` and so cannot tell
  ``auto`` from ``ablated``; this string can.
  """
  if ligand_conditioning is None:
    return "auto"
  return "required" if ligand_conditioning else "ablated"


def _prepare_ligand_context(
  spec: SamplingSpecification,
  batched_ensemble: Protein,
  batch_size: int,
  seq_len: int,
  canonical_structure_ids: Sequence[str] | None = None,
  batch_structure_ids: Sequence[str] | None = None,
  *,
  ablate_ligand: bool | None = None,
) -> dict[str, jax.Array | None]:
  """Build the ligand/side-chain tensors a LigandMPNN bundle consumes.

  ``ablate_ligand`` overrides the spec's tri-state ``ligand_conditioning``: ``None`` (default)
  derives it (``ligand_conditioning`` is False -> ablate). ``runner.score`` passes ``False``
  because ``ScoringSpecification.ligand_conditioning`` is a plain bool with "False == not
  required" semantics (a ``ligand_context_path`` alone requests the ligand there), and score
  already never consults the batch's tensors unless the ligand was requested.
  """
  if spec.model_family != "ligandmpnn":
    return {
      "Y": None,
      "Y_t": None,
      "Y_m": None,
      "atom_37": None,
      "atom_37_mask": None,
      "chain_mask": None,
    }

  # ligand_conditioning is a genuine tri-state (#114): None = use whatever is present (legacy),
  # True = require real tensors, False = ABLATE. The False arm used to be indistinguishable
  # from "unset" -- real tensors on the batch or from ligand_context_path were injected
  # regardless, so a caller (or a campaign's ligand_on=False row) asking for a no-ligand arm
  # silently got full ligand conditioning. Under ablation the declared ligand_context_path is
  # deliberately NOT read: a campaign hands every row the same base_spec, including its
  # ligand_context_path, and the False row must still mean "no ligand".
  if ablate_ligand is None:
    ablate_ligand = spec.ligand_conditioning is not None and not spec.ligand_conditioning
  Y = None if ablate_ligand else getattr(batched_ensemble, "Y", None)
  Y_t = None if ablate_ligand else getattr(batched_ensemble, "Y_t", None)
  Y_m = None if ablate_ligand else getattr(batched_ensemble, "Y_m", None)

  if spec.ligand_context_path is not None and not ablate_ligand:
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
  if (
    atom_37_mask.ndim != 3
    or atom_37_mask.shape[0] != batch_size
    or atom_37_mask.shape[1] != seq_len
  ):
    msg = "atom mask must have shape (batch, residues, atoms) for sidechain conditioning."
    raise ValueError(msg)

  # Compute chain_mask: 1=designable, 0=fixed.
  # Default is all-ones (all residues designable) unless overridden by fixed_mask.
  # Convention: fixed_mask 1=fixed, chain_mask 1=designable → complement.
  declared_fixed_mask = _resolve_mapped_by_field(
    spec.run_spec.sampling.fixed_mask,
    batch_structure_ids=batch_structure_ids,
    field_name="fixed_mask",
  )
  if declared_fixed_mask is not None:
    fixed_mask_np = _broadcast_per_structure(
      declared_fixed_mask,
      batch_size=batch_size,
      expected_len=seq_len,
      dtype=jnp.float32,
      name="fixed_mask",
    )
    chain_mask = 1.0 - fixed_mask_np
    assert chain_mask.dtype == jnp.float32
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


def fixed_provenance_outputs(
  spec: SamplingSpecification,
  *,
  seq_len: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
  """The evidence that a residue was actually held fixed. Emit this beside the sequences.

  Returns ``(arrays, attrs)`` to merge into the row's staged output.

  **Why this exists.** ``fixed_mask`` reached the model and stopped there: it appeared nowhere
  in ``io/`` or any sink, so after a run there was NO WAY to prove which positions had been
  frozen. A campaign produced 882/882 rows, every done-marker valid, every content digest
  intact -- and not one residue held fixed anywhere. The completion record was perfect and the
  science was void, because completion and validity were the only two things nobody had
  written down separately.

  **Why not the done marker.** ``_write_done_marker`` carries the row hash, attempt id and
  content digest -- integrity, not science. Worse, its digest is computed over the zarr tree
  *before* the marker is written, so anything added there sits outside the very hash that
  attests to it. This belongs inside the digested artifact, beside the data it describes.

  **Why 1-D and not the broadcast (batch, L).** This records the DECLARATION -- "these
  positions were requested frozen, at these identities" -- which is what a reader needs to
  check the sequences against. ``_prepare_fixed_controls``' per-structure broadcast is an
  implementation detail of one call.

  ``n_fixed`` is an attr, not a derived read, so a reader can reject an unfixed row without
  loading the mask -- and so "0 positions were fixed" is a value someone can SEE in the
  metadata rather than a silence they have to notice.

  Args:
    spec: The row's spec, post-reconstruction.
    seq_len: The parsed structure's padded length, used to emit an honest all-zero mask when
      nothing is fixed -- an absent key and "nothing was fixed" must not look alike.

  """
  sampling = spec.run_spec.sampling
  declared = sampling.fixed_mask
  mask = (
    np.zeros(seq_len, dtype=np.uint8)
    if declared is None
    else np.asarray(declared).reshape(-1)[:seq_len].astype(np.uint8)
  )
  arrays: dict[str, np.ndarray] = {"fixed_mask": mask}
  attrs: dict[str, Any] = {"n_fixed": int(np.count_nonzero(mask))}

  # Tokens only mean something where the mask selects, but emit the whole array: a reader
  # comparing sequences against it needs the same indexing, and slicing here would invent a
  # second convention to get wrong.
  if sampling.fixed_tokens is not None:
    arrays["fixed_tokens"] = (
      np.asarray(sampling.fixed_tokens)
      .reshape(-1)[:seq_len]
      .astype(
        np.int32,
      )
    )
    # Name the alphabet AS DATA, not in a key name. The consumer-side bug this audit found was
    # exactly a rename stripping an `_af` suffix and taking the convention with it: a rename
    # cannot strip a value. fixed_tokens are MPNN by contract (decode substitutes them
    # directly against MPNN-sampled tokens).
    attrs["fixed_tokens_alphabet"] = "MPNN"
  return arrays, attrs


def _token_name(token: int, alphabet: str) -> str:
  """Human-readable amino acid for an error message (never used for control flow).

  ``alphabet`` is REQUIRED and deliberately not defaulted. This helper is reachable from two
  contexts in different alphabets -- native ``aatype`` (AF) and ``fixed_tokens`` (MPNN) --
  and a default would silently pick one of them for both. The previous version defaulted to
  MPNN and was called with AF ``aatype``, so a divergence at a His position reported "K".
  An error message that misnames the residue is worse than no message: it sends the reader
  hunting for a mutation that isn't there.

  Args:
    token: The integer token.
    alphabet: The alphabet ``token`` is indexed in -- ``AF_ALPHABET`` or ``MPNN_ALPHABET``.

  """
  return alphabet[token] if 0 <= token < len(alphabet) else str(token)


def resolve_native_tokens(
  batched_ensemble: Protein,
  fixed_mask: Any,  # noqa: ANN401 -- ArrayLike; matches _broadcast_per_structure's convention
  *,
  allow_heterogeneous: bool = False,
) -> np.ndarray:
  """Resolve ``fixed_tokens`` from each structure's own native residues.

  The override for "freeze these positions at whatever is already there", as opposed to passing
  ``fixed_tokens`` explicitly. Returns an array shaped to ``batched_ensemble.aatype``.

  Deliberately a function, not a ``SamplingSpecification`` field: a new spec field would be
  silently dropped by ``host/campaign.py``'s hand-written manifest literal and become another
  declared-but-never-delivered knob -- the exact bug class this guard exists to close.

  **Divergence is refused, not resolved.** With multiple structures, "native" is only well
  defined if they agree at every frozen position. For a product-of-experts bead the design is ONE
  sequence, so a position whose native is Cys in one state and Ala in another (a real case: a
  C151A catalytic mutant crystallised precisely because it was inactive) has no correct answer
  this function can pick. Picking one silently is how a dead enzyme ships. Pass explicit
  ``fixed_tokens`` to state the intent, or ``allow_heterogeneous=True`` to accept structure 0's
  residue knowingly.

  Args:
    batched_ensemble: Batched structures; ``aatype`` supplies the native residues.
    fixed_mask: Which positions are frozen. 1 = fixed.
    allow_heterogeneous: Accept structure 0's residue where natives disagree.

  Returns:
    ``fixed_tokens`` in the **MPNN alphabet** (``ACDEFGHIKLMNPQRSTVWYX``), suitable for
    ``SamplingSpecification(fixed_tokens=...)``. The conversion from ``aatype``'s AF ordering
    happens here; callers must not convert again. Naming the alphabet is not decoration --
    every bug this function has had, and every one found in the surrounding audit, traces to
    a contract that said "integer labels" and left the ordering to be guessed.

  Raises:
    ValueError: Natives disagree at a frozen position and ``allow_heterogeneous`` is False.

  """
  from aminx.utils.aa_convert import MPNN_ALPHABET, af_to_mpnn  # noqa: PLC0415

  # Convert ONCE, here, so the whole function below is MPNN and there is no second
  # convention to keep straight. `aatype` is AF (proxide is AF-native); `fixed_tokens` is
  # substituted directly against model-sampled tokens at
  # inference/decode/autoregressive.py:346, which are MPNN. Returning the raw array -- what
  # this function did until now -- froze every position to a *different* residue than the
  # caller's own structure: AF His(8) reads as MPNN Lys. A "fixed catalytic triad" would
  # have come back H->K, D->E, C->F. Shipped in 026403d, the commit whose stated purpose was
  # preventing a silently-wrong fixed identity, and hidden by test fixtures hand-written in
  # the alphabet their author assumed.
  aatype = np.asarray(af_to_mpnn(jnp.asarray(batched_ensemble.aatype)), dtype=np.int32)
  if aatype.ndim == 1:
    aatype = aatype[None, :]
  mask = np.asarray(fixed_mask, dtype=np.float32)
  if mask.ndim == 1:
    mask = np.broadcast_to(mask[None, :], aatype.shape)

  frozen = np.where(np.any(mask > 0, axis=0))[0]
  if len(frozen) and aatype.shape[0] > 1 and not allow_heterogeneous:
    divergent = [int(p) for p in frozen if len(np.unique(aatype[:, p])) > 1]
    if divergent:
      detail = "; ".join(
        f"position {p}: "
        + ", ".join(
          f"structure {s}={_token_name(int(aatype[s, p]), MPNN_ALPHABET)}"
          for s in range(aatype.shape[0])
        )
        for p in divergent[:4]
      )
      msg = (
        f"Cannot resolve native fixed_tokens: structures disagree at {len(divergent)} frozen "
        f"position(s) -- {detail}. The design is a single sequence, so there is no native to "
        f"pick. Either pass fixed_tokens explicitly to declare the intended identity, or set "
        f"allow_heterogeneous=True to take structure 0's residue deliberately."
      )
      raise ValueError(msg)

  return aatype.copy()  # MPNN -- converted at the top of this function


def _resolve_mapped_by_field(
  value: object,
  *,
  batch_structure_ids: Sequence[str] | None,
  field_name: str,
) -> object:
  """Resolve a possibly-``MappedBy`` sampling field for the CURRENT batch, or pass it through.

  Every ``fixed_mask``/``fixed_tokens``-reading call site is expected to supply
  ``batch_structure_ids`` -- MappedBy support is a feature of the field, not of one code path,
  so ``sample()``'s independent-structures batching (``kernel_dispatch.py``) and the
  multistate/PoE combined-bead path (``multistate_poe.py``, keyed by ``spec.inputs`` since one
  state == one input structure there) both resolve it the same way. Raises only if a caller
  genuinely has no per-batch structure identity to resolve against, which should not happen at
  any current call site -- this is a defensive bug check, not an expected/normal outcome.
  """
  from aminx.run.batch_mapping import MappedBy, resolve_mapped_by  # noqa: PLC0415

  if not isinstance(value, MappedBy):
    return value
  if batch_structure_ids is None:
    msg = (
      f"{field_name} is a MappedBy(by={value.by!r}), but this call site has no per-batch "
      f"structure identity to resolve it against. Every fixed_mask/fixed_tokens call site is "
      f"expected to pass batch_structure_ids; this is a bug in the caller, not an expected "
      f"restriction on MappedBy -- see _resolve_mapped_by_field's docstring."
    )
    raise ValueError(msg)
  resolved = resolve_mapped_by(
    value,
    structure_ids=list(batch_structure_ids),
    field_name=field_name,
  )
  return np.stack([np.asarray(v) for v in resolved], axis=0)


def _prepare_fixed_controls(
  spec: SamplingSpecification,
  *,
  batched_ensemble: Protein,
  batch_structure_ids: Sequence[str] | None = None,
) -> tuple[jax.Array, jax.Array]:
  from aminx.utils.aa_convert import MPNN_ALPHABET  # noqa: PLC0415

  batch_size, seq_len = batched_ensemble.coordinates.shape[:2]

  fixed_mask_np = np.zeros((batch_size, seq_len), dtype=np.float32)
  fixed_tokens_np = np.zeros((batch_size, seq_len), dtype=np.int32)

  declared_fixed_mask = _resolve_mapped_by_field(
    spec.run_spec.sampling.fixed_mask,
    batch_structure_ids=batch_structure_ids,
    field_name="fixed_mask",
  )
  declared_fixed_tokens = _resolve_mapped_by_field(
    spec.run_spec.sampling.fixed_tokens,
    batch_structure_ids=batch_structure_ids,
    field_name="fixed_tokens",
  )

  if declared_fixed_mask is not None:
    fixed_mask_np = np.asarray(declared_fixed_mask, dtype=np.float32)
    if fixed_mask_np.ndim == 1:
      fixed_mask_np = np.broadcast_to(
        fixed_mask_np[None, :],
        (batch_size, seq_len),
      ).copy()
    if fixed_mask_np.shape != (batch_size, seq_len):
      msg = f"fixed_mask must have shape ({batch_size}, {seq_len}), got {fixed_mask_np.shape}"
      raise ValueError(msg)

  if spec.run_spec.sampling.fixed_positions is not None:
    fixed_pos = np.asarray(spec.run_spec.sampling.fixed_positions, dtype=np.float32)
    fixed_pos_mask = _broadcast_per_structure(
      fixed_pos,
      batch_size=batch_size,
      expected_len=seq_len,
      dtype=jnp.float32,
      name="fixed_positions",
    )
    # Union: combine fixed_positions with fixed_mask (if both are set)
    fixed_mask_np = np.maximum(fixed_mask_np, fixed_pos_mask)

  if declared_fixed_tokens is not None:
    fixed_tok = np.asarray(declared_fixed_tokens, dtype=np.int32)
    fixed_tokens_np = _broadcast_per_structure(
      fixed_tok,
      batch_size=batch_size,
      expected_len=seq_len,
      dtype=jnp.int32,
      name="fixed_tokens",
    )

  # fixed_mask selects WHICH positions are frozen; fixed_tokens says TO WHAT. They are one
  # decision, not two. Freezing without saying to what silently locks every frozen position to
  # token 0 -- Alanine (index 0 in BOTH alphabets, so this one is unambiguous) -- because
  # decode does `final_token = where(is_group_fixed, group_fixed_token, sampled)`
  # (inference/decode/autoregressive.py:346; an earlier version of this comment cited
  # decode/autoregressive.py:340, a path that does not exist).
  # A caller "fixing the catalytic triad" would get a dead Ala/Ala/Ala enzyme, with valid-shaped
  # output and no error. Refuse the ambiguity instead of resolving it to the most destructive
  # possible default.
  if np.any(np.asarray(fixed_mask_np) > 0) and declared_fixed_tokens is None:
    n_fixed = int(np.count_nonzero(np.asarray(fixed_mask_np)[0]))
    msg = (
      f"fixed_mask/fixed_positions freeze {n_fixed} position(s) but fixed_tokens is None, which "
      f"would silently lock every frozen position to token 0 = "
      f"'{_token_name(0, MPNN_ALPHABET)}' (Alanine) -- "
      f"e.g. a 'fixed catalytic triad' would come back as Ala/Ala/Ala, a dead enzyme, with no "
      f"error. Freezing a position and choosing its identity are one decision. Either:\n"
      f"  - pass fixed_tokens explicitly (a 1-D array locks the same identity across all "
      f"states -- the usual intent, e.g. a catalytic triad held at H/D/C everywhere); or\n"
      f"  - call resolve_native_tokens(batched_ensemble, fixed_mask) to freeze each position at "
      f"its own native residue."
    )
    raise ValueError(msg)

  # Validity is checked AFTER the guard and OUTSIDE the `fixed_tokens is not None` branch it
  # used to live in -- where it could never fire for the all-zeros default, which is exactly the
  # case that needed catching. Token 0 is a legal amino acid, not a sentinel, so range-checking
  # alone never would have caught it either.
  invalid_tokens = (fixed_tokens_np < 0) | (fixed_tokens_np >= AMINO_ACID_VOCAB_SIZE)
  if np.any(np.asarray(invalid_tokens) & (np.asarray(fixed_mask_np) > 0)):
    msg = f"fixed_tokens must be in [0, {AMINO_ACID_VOCAB_SIZE - 1}] at masked positions."
    raise ValueError(msg)

  if declared_fixed_mask is not None:
    fm_broadcast = _broadcast_per_structure(
      np.asarray(declared_fixed_mask, dtype=np.float32),
      batch_size=batch_size,
      expected_len=seq_len,
      dtype=jnp.float32,
      name="fixed_mask",
    )
    fixed_mask_np = jnp.maximum(jnp.asarray(fixed_mask_np), fm_broadcast)

  return jnp.asarray(fixed_mask_np), jnp.asarray(fixed_tokens_np)
