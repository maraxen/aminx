"""aminx Domain Plugin for xtrax Campaign EDA.

Implements the AminxDomainPlugin conforming to the xtrax.eda.domains protocol.
Extracts LibraryCells from aminx campaign stores (sequences, logits, scores with
derived NLL fallback and perplexity, MPNN 21-character alphabet normalization,
target native recovery, and designable masking) and computes domain-specific
library metrics.
"""

from __future__ import annotations

import importlib.metadata
import io
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# Standard biomolecular alphabets (21 characters)
MPNN_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWYX"
AF2_ALPHABET: str = "ARNDCQEGHILKMFPSTWYVX"
AF2_TO_MPNN: np.ndarray = np.array([MPNN_ALPHABET.index(c) for c in AF2_ALPHABET], dtype=np.int64)
MPNN_TO_AF2: np.ndarray = np.array([AF2_ALPHABET.index(c) for c in MPNN_ALPHABET], dtype=np.int64)

# Try importing standard domain protocol and dataclasses from xtrax.eda.campaign
try:
  from xtrax.eda.campaign.plugins import DomainPlugin as XtraxDomainPlugin
  from xtrax.eda.campaign.types import (
    DomainMetricResult as XtraxDomainMetricResult,
  )
  from xtrax.eda.campaign.types import (
    LibraryCell as XtraxLibraryCell,
  )
  from xtrax.eda.campaign.types import (
    ResidueTable as XtraxResidueTable,
  )
except ImportError:
  XtraxResidueTable = None
  XtraxLibraryCell = None
  XtraxDomainMetricResult = None
  XtraxDomainPlugin = None

try:
  from xtrax.pack import is_packed as _xtrax_is_packed
  from xtrax.pack import read_unit_member as _xtrax_read_unit_member
except ImportError:
  _xtrax_is_packed = None
  _xtrax_read_unit_member = None

try:
  import zarr as _zarr
except ImportError:
  _zarr = None

try:
  from aminx.io.parsing.dispatch import parse_input as _parse_input
except ImportError:
  _parse_input = None

_SINK_KEYS: tuple[str, ...] = (
  "sequences",
  "logits",
  "scores",
  "fixed_mask",
  "fixed_tokens",
  "native",
)


@dataclass(frozen=True, slots=True)
class ResidueTable:
  """Canonical structural residue mapping."""

  native_tokens: str  # Normalized single-letter amino acid string
  chain_ids: tuple[str, ...]  # Chain identifier per residue
  res_numbers: tuple[int, ...]  # PDB author residue number per residue
  insertion_codes: tuple[str, ...]  # PDB insertion code per residue
  real_mask: np.ndarray  # Boolean array (True for real target residues)
  parser_name: str  # Name of producing parser (e.g. "proxide")
  parser_version: str  # Version string of producing parser


@dataclass(frozen=True, slots=True)
class LibraryCell:
  """Standardized extracted data cell for a single experimental unit."""

  knobs: Mapping[str, object]  # Merged row and sink knob dictionary
  seed: int  # Stochastic replicate seed
  target: str  # Target identifier
  tokens: np.ndarray  # Integer token array (N, L)
  sequences: tuple[str, ...]  # Decoded sequence strings (N,)
  logits: np.ndarray | None  # Unnormalized model logits (N, L, alphabet_size) or None
  scores: np.ndarray | None  # Model-reported scores (N,) or None
  is_score_derived: bool  # True if scores were derived host-side from logits
  residues: ResidueTable  # Target residue mapping table
  mask: np.ndarray  # Boolean designable mask (L,)


@dataclass(frozen=True, slots=True)
class DomainMetricResult:
  """Named scalar metrics computed for a cell with strict type and finiteness contracts."""

  metrics: Mapping[str, float]  # Scalar metrics (e.g. {"diversity": 0.45, "rec": 0.62})
  flags: Mapping[str, bool]  # Boolean diagnostic flags (e.g. {"degenerate": False})
  tracks: Mapping[str, np.ndarray]  # Per-residue 1D arrays (e.g. {"entropy": H_l})

  def validate_schema(self) -> None:
    """Validate that all metric values are strictly numeric and finite.

    Enforces string keys, finite float values for metrics, boolean flags,
    and read-only 1D arrays for tracks.
    """
    for k, v in self.metrics.items():
      if not isinstance(k, str):
        raise TypeError(f"Domain metric key {k!r} must be str, got {type(k).__name__}")
      if not isinstance(v, (int, float, np.floating, np.integer)) or not math.isfinite(
        float(v),
      ):
        raise TypeError(
          f"Domain metric {k!r} has invalid value {v!r} (type {type(v).__name__}); "
          "metrics must be strictly finite floats or ints.",
        )
    for k, v in self.flags.items():
      if not isinstance(k, str):
        raise TypeError(f"Domain flag key {k!r} must be str, got {type(k).__name__}")
      if not isinstance(v, (bool, np.bool_)):
        raise TypeError(f"Domain flag {k!r} must be bool, got {type(v).__name__}")
    for k, v in self.tracks.items():
      if not isinstance(k, str):
        raise TypeError(f"Domain track key {k!r} must be str, got {type(k).__name__}")
      if not isinstance(v, np.ndarray):
        raise TypeError(f"Domain track {k!r} must be np.ndarray, got {type(v).__name__}")
      if v.flags.writeable:
        v.flags.writeable = False


@runtime_checkable
class DomainPlugin(Protocol):
  """Protocol for consumer domain analytical plugins."""

  @property
  def domain_name(self) -> str:
    """Unique identifier of the domain (e.g. 'aminx', 'demistify')."""
    ...

  @property
  def version(self) -> str:
    """Semantic version of the plugin implementation."""
    ...

  def extract_cell(
    self,
    store_path: Path | object,
    manifest_row: object,
    sink_schema: object = None,
  ) -> Sequence[LibraryCell]:
    """Extract one or more LibraryCells from a committed unit store."""
    ...

  def compute_metrics(self, cell: LibraryCell) -> DomainMetricResult:
    """Compute domain-specific scalar and per-residue metrics."""
    ...


def _get_row_attr(row: object, key: str, default: object = None) -> object:
  """Safely extract field from either dict or dataclass row representation."""
  if isinstance(row, Mapping):
    return row.get(key, default)
  return getattr(row, key, default)


def _decode_tokens_to_seqs(tokens: np.ndarray) -> tuple[str, ...]:
  """Decode (N, L) integer tokens array to sequence strings using MPNN alphabet."""
  alpha = MPNN_ALPHABET
  res: list[str] = [
    "".join(alpha[t] if 0 <= t < 21 else "X" for t in tokens[i]) for i in range(tokens.shape[0])
  ]
  return tuple(res)


def _encode_seqs_to_tokens(seqs: Sequence[str | bytes], max_len: int | None = None) -> np.ndarray:
  """Encode sequence strings to (N, L) integer token array."""
  n_samples = len(seqs)
  if n_samples == 0:
    return np.zeros((0, 0), dtype=np.int64)
  norm_seqs = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in seqs]
  l_max = max(len(s) for s in norm_seqs) if max_len is None else max_len
  arr = np.full((n_samples, l_max), 20, dtype=np.int64)  # 20 is 'X'
  alpha_map = {c: i for i, c in enumerate(MPNN_ALPHABET)}
  for i, s in enumerate(norm_seqs):
    for j, c in enumerate(s[:l_max]):
      arr[i, j] = alpha_map.get(c.upper(), 20)
  return arr


def _compute_derived_nll(tokens: np.ndarray, logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
  """Compute per-sample derived NLL: 1/|M| * sum_{l in M} (LSE(logits_l) - logits_{l, s_l})."""
  n_samples = tokens.shape[0]
  mask_sum = int(np.sum(mask))
  if mask_sum == 0:
    return np.zeros(n_samples, dtype=np.float64)

  # LogSumExp over the last axis
  max_logits = np.max(logits, axis=-1, keepdims=True)
  exp_shifted = np.exp(logits - max_logits)
  sum_exp = np.sum(exp_shifted, axis=-1)
  lse = np.log(np.clip(sum_exp, 1e-300, None)) + max_logits[..., 0]

  # Gather logit at sampled token: tokens in [0, 20]
  safe_tokens = np.clip(tokens, 0, logits.shape[-1] - 1)
  tok_logits = np.take_along_axis(logits, safe_tokens[..., None].astype(np.int64), axis=-1)[
    ...,
    0,
  ]

  nll_pos = lse - tok_logits  # shape (N, L)
  masked_nll = nll_pos * mask[None, :]
  return np.sum(masked_nll, axis=-1) / mask_sum


def _compute_positional_entropy(
  tokens: np.ndarray,
  logits: np.ndarray | None,
  n_samples: int,
  seq_len: int,
) -> np.ndarray:
  """Compute vectorized positional Shannon entropy track (L,) in bits."""
  if seq_len == 0:
    return np.zeros(0, dtype=np.float64)

  if logits is not None:
    flat = logits.astype(np.float64)
    max_l = np.max(flat, axis=-1, keepdims=True)
    exp_l = np.exp(flat - max_l)
    p = exp_l / np.clip(np.sum(exp_l, axis=-1, keepdims=True), 1e-300, None)
    mean_p = np.mean(p, axis=0)  # (L, 21)
    p_safe = np.where(mean_p > 0, mean_p, 1.0)
    log2_p = np.where(mean_p > 0, np.log2(p_safe), 0.0)
    return -np.sum(mean_p * log2_p, axis=-1)

  if n_samples > 0:
    valid_tokens = np.clip(tokens, 0, 20)
    # Vectorized one-hot over (N, L, 21)
    one_hot = (np.arange(21)[None, None, :] == valid_tokens[:, :, None]).astype(np.float64)
    mean_p = np.mean(one_hot, axis=0)  # (L, 21)
    p_safe = np.where(mean_p > 0, mean_p, 1.0)
    log2_p = np.where(mean_p > 0, np.log2(p_safe), 0.0)
    return -np.sum(mean_p * log2_p, axis=-1)

  return np.zeros(seq_len, dtype=np.float64)


def _slice_sink_arrays(
  seqs_raw: np.ndarray,
  logits_raw: np.ndarray | None,
  scores_raw: np.ndarray | None,
  base_knobs: Mapping[str, object],
) -> list[tuple[dict[str, object], np.ndarray, np.ndarray | None, np.ndarray | None]]:
  """Slice multi-dimensional sink arrays into per-condition 2D slices."""
  slices: list[tuple[dict[str, object], np.ndarray, np.ndarray | None, np.ndarray | None]] = []

  if seqs_raw.ndim == 1:
    slices.append(
      (
        dict(base_knobs),
        seqs_raw[None, :],
        logits_raw[None, :] if logits_raw is not None else None,
        scores_raw,
      ),
    )
  elif seqs_raw.ndim == 2:
    slices.append((dict(base_knobs), seqs_raw, logits_raw, scores_raw))
  elif seqs_raw.ndim == 3:
    n_temps = seqs_raw.shape[1]
    row_temps = base_knobs.get("temperature", [float(t) for t in range(n_temps)])
    if not isinstance(row_temps, (list, tuple)) or len(row_temps) != n_temps:
      row_temps = [float(t) for t in range(n_temps)]
    for ti in range(n_temps):
      knob_slice = dict(base_knobs)
      knob_slice["temperature"] = row_temps[ti]
      seq_slice = seqs_raw[:, ti, :]
      logit_slice = (
        logits_raw[:, ti, ...] if logits_raw is not None and logits_raw.ndim >= 3 else None
      )
      score_slice = (
        scores_raw[:, ti] if scores_raw is not None and scores_raw.ndim >= 2 else scores_raw
      )
      slices.append((knob_slice, seq_slice, logit_slice, score_slice))
  elif seqs_raw.ndim == 4:
    n_grids = seqs_raw.shape[1]
    n_temps = seqs_raw.shape[2]
    row_temps = base_knobs.get("temperature", [float(t) for t in range(n_temps)])
    if not isinstance(row_temps, (list, tuple)) or len(row_temps) != n_temps:
      row_temps = [float(t) for t in range(n_temps)]
    for gi in range(n_grids):
      for ti in range(n_temps):
        knob_slice = dict(base_knobs)
        knob_slice["grid_idx"] = gi
        knob_slice["temperature"] = row_temps[ti]
        seq_slice = seqs_raw[:, gi, ti, :]
        logit_slice = (
          logits_raw[:, gi, ti, ...] if logits_raw is not None and logits_raw.ndim >= 4 else None
        )
        score_slice = (
          scores_raw[:, gi, ti] if scores_raw is not None and scores_raw.ndim >= 3 else scores_raw
        )
        slices.append((knob_slice, seq_slice, logit_slice, score_slice))
  else:
    slices.append((dict(base_knobs), seqs_raw.reshape(-1, seqs_raw.shape[-1]), None, None))

  return slices


class AminxDomainPlugin:
  """Domain plugin extracting and computing generative protein sequence metrics for aminx."""

  @property
  def domain_name(self) -> str:
    return "aminx"

  @property
  def version(self) -> str:
    try:
      return importlib.metadata.version("aminx")
    except importlib.metadata.PackageNotFoundError:
      return "0.4.0"

  def _load_packed_sink(self, path: Path, record_name: str) -> dict[str, np.ndarray]:
    """Load sink arrays from a packed unit directory."""
    out: dict[str, np.ndarray] = {}
    if _xtrax_read_unit_member is None:
      return out
    for key in _SINK_KEYS:
      for ext in (".npy", ""):
        member_name = f"{key}{ext}" if ext else key
        try:
          raw = _xtrax_read_unit_member(path, member_name, record_name=record_name)
          out[key] = np.load(io.BytesIO(raw))
          break
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
          logger.debug("Packed member %s not loaded from %s: %s", member_name, path, exc)
    return out

  def _load_zarr_sink(self, path: Path, manifest_row: object) -> dict[str, np.ndarray]:
    """Load sink arrays from a Zarr group directory."""
    out: dict[str, np.ndarray] = {}
    if _zarr is None:
      return out
    try:
      group = _zarr.open_group(str(path), mode="r")
      row_hash = str(_get_row_attr(manifest_row, "manifest_row_hash", ""))
      target_group = group
      if "sequences" not in group:
        if row_hash and row_hash in group:
          target_group = group[row_hash]
        else:
          keys = list(group.group_keys())
          if len(keys) == 1:
            target_group = group[keys[0]]

      for key in _SINK_KEYS:
        if key in target_group:
          out[key] = np.asarray(target_group[key])

      if hasattr(target_group, "attrs") and "residue_table" in target_group.attrs:
        out["__residue_table_attr__"] = target_group.attrs["residue_table"]
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
      logger.debug("Failed reading zarr group at %s: %s", path, exc)
    return out

  def _load_sink_data(
    self,
    store_path: Path | object,
    manifest_row: object,
  ) -> Mapping[str, np.ndarray]:
    """Load sink arrays from loose directory, packed archive, zarr group, or mapping."""
    if isinstance(store_path, Mapping):
      return {k: np.asarray(v) for k, v in store_path.items()}

    path = Path(store_path)
    if path.is_file() and path.name.endswith(".npy"):
      return {"sequences": np.load(path)}

    out: dict[str, np.ndarray] = {}
    record_name = "done.json" if (path / "done.json").exists() else "unit.json"

    # 1. Packed archive (.tar.zst) support
    is_packed_unit = False
    if _xtrax_is_packed is not None:
      try:
        is_packed_unit = _xtrax_is_packed(path, record_name=record_name)
      except (OSError, ValueError, KeyError, RuntimeError):
        is_packed_unit = (path / "members.tar.zst").exists() or (path / "members.tar").exists()
    else:
      is_packed_unit = (path / "members.tar.zst").exists() or (path / "members.tar").exists()

    if is_packed_unit:
      out.update(self._load_packed_sink(path, record_name))

    # 2. Loose .npy files support
    if "sequences" not in out and path.is_dir():
      for key in _SINK_KEYS:
        npy_path = path / f"{key}.npy"
        if npy_path.is_file():
          try:
            out[key] = np.load(npy_path)
          except (OSError, ValueError) as exc:
            logger.debug("Failed reading loose file %s: %s", npy_path, exc)

    # 3. data.json fallback
    if "sequences" not in out and path.is_dir():
      json_path = path / "data.json"
      if json_path.is_file():
        try:
          with json_path.open(encoding="utf-8") as f:
            data = json.load(f)
          for k, v in data.items():
            out[k] = np.asarray(v)
        except (OSError, ValueError, TypeError) as exc:
          logger.debug("Failed reading data.json from %s: %s", path, exc)

    # 4. Zarr group directory fallback
    if "sequences" not in out and path.is_dir():
      out.update(self._load_zarr_sink(path, manifest_row))

    return out

  def _resolve_residue_table(
    self,
    sink_data: Mapping[str, np.ndarray],
    manifest_row: object,
    l_target: int,
  ) -> ResidueTable:
    """Resolve or construct the canonical ResidueTable."""
    # 1. Embedded residue table attribute
    if "__residue_table_attr__" in sink_data:
      attr_val = sink_data["__residue_table_attr__"]
      if isinstance(attr_val, str):
        attr_val = json.loads(attr_val)
      if isinstance(attr_val, Mapping):
        return ResidueTable(
          native_tokens=str(attr_val.get("native_tokens", "X" * l_target)),
          chain_ids=tuple(str(c) for c in attr_val.get("chain_ids", ("A",) * l_target)),
          res_numbers=tuple(int(r) for r in attr_val.get("res_numbers", range(1, l_target + 1))),
          insertion_codes=tuple(str(i) for i in attr_val.get("insertion_codes", ("") * l_target)),
          real_mask=np.asarray(
            attr_val.get("real_mask", np.ones(l_target, dtype=bool)),
            dtype=bool,
          ),
          parser_name=str(attr_val.get("parser_name", "proxide")),
          parser_version=str(attr_val.get("parser_version", "1.0.0")),
        )

    # 2. Target input parsing fallback if target_input exists on disk
    target_input = str(_get_row_attr(manifest_row, "target_input", ""))
    if target_input and Path(target_input).exists() and _parse_input is not None:
      try:
        proteins = list(_parse_input(target_input))
        if proteins:
          p = proteins[0]
          aa = np.asarray(p.aatype).astype(int)
          resnum = np.asarray(p.residue_index).astype(int)
          chain = np.asarray(p.chain_index).astype(int)
          mpnn_tokens = AF2_TO_MPNN[aa]
          native_str = "".join(MPNN_ALPHABET[tok] for tok in mpnn_tokens)
          try:
            proxide_ver = importlib.metadata.version("proxide")
          except importlib.metadata.PackageNotFoundError:
            proxide_ver = "unknown"
          return ResidueTable(
            native_tokens=native_str,
            chain_ids=tuple(str(c) for c in chain),
            res_numbers=tuple(int(r) for r in resnum),
            insertion_codes=tuple("" for _ in range(len(aa))),
            real_mask=np.ones(len(aa), dtype=bool),
            parser_name="proxide",
            parser_version=proxide_ver,
          )
      except (ImportError, ValueError, OSError) as e:
        logger.debug("Could not parse target_input %s: %s", target_input, e)

    # 3. Target native tokens from sink
    if "native" in sink_data:
      nat = np.asarray(sink_data["native"]).ravel()
      native_str = "".join(MPNN_ALPHABET[int(t)] if 0 <= int(t) < 21 else "X" for t in nat)
      return ResidueTable(
        native_tokens=native_str,
        chain_ids=tuple("A" for _ in range(len(native_str))),
        res_numbers=tuple(range(1, len(native_str) + 1)),
        insertion_codes=tuple("" for _ in range(len(native_str))),
        real_mask=np.ones(len(native_str), dtype=bool),
        parser_name="sink_native",
        parser_version="1.0.0",
      )

    # 4. Synthesize from fixed_tokens or length
    if "fixed_tokens" in sink_data:
      ft = np.asarray(sink_data["fixed_tokens"]).ravel()
      native_str = "".join(MPNN_ALPHABET[int(t)] if 0 <= int(t) < 21 else "X" for t in ft)
    else:
      native_str = "X" * l_target

    return ResidueTable(
      native_tokens=native_str,
      chain_ids=tuple("A" for _ in range(len(native_str))),
      res_numbers=tuple(range(1, len(native_str) + 1)),
      insertion_codes=tuple("" for _ in range(len(native_str))),
      real_mask=np.ones(len(native_str), dtype=bool),
      parser_name="synthesized",
      parser_version="1.0.0",
    )

  def extract_cell(
    self,
    store_path: Path | object,
    manifest_row: object,
    sink_schema: object = None,
  ) -> Sequence[LibraryCell]:
    """Extract one or more LibraryCells from a committed unit store."""
    del sink_schema  # conforms to protocol signature
    sink = self._load_sink_data(store_path, manifest_row)
    if "sequences" not in sink:
      msg = f"Store at {store_path} missing required 'sequences' array"
      raise ValueError(msg)

    seqs_raw = sink["sequences"]
    logits_raw = sink.get("logits")
    scores_raw = sink.get("scores")
    fixed_mask_raw = sink.get("fixed_mask")

    base_knobs: dict[str, object] = {}
    for k in ("arm_id", "arm_backend", "checkpoint_id", "fixed_policy", "state_weight_profile"):
      val = _get_row_attr(manifest_row, k)
      if val is not None:
        base_knobs[k] = val
    params = _get_row_attr(manifest_row, "parameters")
    if isinstance(params, Mapping):
      base_knobs.update(params)

    seed = int(_get_row_attr(manifest_row, "seed", 0))  # type: ignore[arg-type]
    target_input = str(_get_row_attr(manifest_row, "target_input", ""))
    target = Path(target_input).stem if target_input else "target"

    slices = _slice_sink_arrays(seqs_raw, logits_raw, scores_raw, base_knobs)
    cells: list[LibraryCell] = []

    for knob_dict, s_arr, l_arr, sc_arr in slices:
      cell = self._build_single_cell(
        knob_dict=knob_dict,
        s_arr=s_arr,
        l_arr=l_arr,
        sc_arr=sc_arr,
        fixed_mask_raw=fixed_mask_raw,
        sink=sink,
        manifest_row=manifest_row,
        seed=seed,
        target=target,
      )
      cells.append(cell)

    return cells

  def _build_single_cell(
    self,
    knob_dict: Mapping[str, object],
    s_arr: np.ndarray,
    l_arr: np.ndarray | None,
    sc_arr: np.ndarray | None,
    fixed_mask_raw: np.ndarray | None,
    sink: Mapping[str, np.ndarray],
    manifest_row: object,
    seed: int,
    target: str,
  ) -> LibraryCell:
    """Construct an immutable LibraryCell from a 2D slice."""
    n_samples, seq_len = s_arr.shape

    if np.issubdtype(s_arr.dtype, np.integer):
      tokens = np.array(s_arr, dtype=np.int64, copy=True)
      sequences = _decode_tokens_to_seqs(tokens)
    else:
      sequences = tuple(
        x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in s_arr.ravel()
      )[:n_samples]
      tokens = _encode_seqs_to_tokens(sequences, max_len=seq_len)

    res_table = self._resolve_residue_table(sink, manifest_row, seq_len)

    real_m = np.zeros(seq_len, dtype=bool)
    l_real = min(len(res_table.native_tokens), seq_len)
    real_m[:l_real] = res_table.real_mask[:l_real] if len(res_table.real_mask) >= l_real else True

    if fixed_mask_raw is not None:
      fm = np.asarray(fixed_mask_raw, dtype=bool).ravel()
      fm_padded = np.zeros(seq_len, dtype=bool)
      fm_padded[: min(len(fm), seq_len)] = fm[: min(len(fm), seq_len)]
      designable = real_m & ~fm_padded
    else:
      designable = real_m

    logits: np.ndarray | None = None
    if l_arr is not None:
      logits = np.array(l_arr, dtype=np.float32, copy=True)
      if logits.ndim == 2:
        logits = logits[None, ...]

    scores: np.ndarray | None = None
    is_derived = False
    if sc_arr is not None:
      scores = np.array(sc_arr, dtype=np.float64, copy=True).ravel()
    elif logits is not None:
      scores = _compute_derived_nll(tokens, logits, designable)
      is_derived = True

    tokens.flags.writeable = False
    designable.flags.writeable = False
    if logits is not None:
      logits.flags.writeable = False
    if scores is not None:
      scores.flags.writeable = False
    if hasattr(res_table, "real_mask") and isinstance(res_table.real_mask, np.ndarray):
      res_table.real_mask.flags.writeable = False

    return LibraryCell(
      knobs=knob_dict,
      seed=seed,
      target=target,
      tokens=tokens,
      sequences=sequences,
      logits=logits,
      scores=scores,
      is_score_derived=is_derived,
      residues=res_table,
      mask=designable,
    )

  def compute_metrics(self, cell: LibraryCell) -> DomainMetricResult:
    """Compute domain-specific scalar and per-residue metrics for a LibraryCell."""
    tokens = cell.tokens
    mask = cell.mask
    n_samples, seq_len = tokens.shape
    mask_sum = int(np.sum(mask))

    flags: dict[str, bool] = {}
    tracks: dict[str, np.ndarray] = {}

    # 1. Diversity and Pairwise Identity
    mean_diversity, mean_pid = self._calc_diversity(tokens, mask, n_samples, mask_sum, flags)

    # 2. Positional Shannon Entropy
    h_track = _compute_positional_entropy(tokens, cell.logits, n_samples, seq_len)
    h_track.flags.writeable = False
    tracks["positional_entropy"] = h_track
    mean_pos_entropy = float(np.mean(h_track[mask])) if mask_sum > 0 else 0.0

    # 3. Composition Shannon Entropy
    comp_entropy = self._calc_composition_entropy(tokens, mask, n_samples, mask_sum, flags)

    # 4. Target Native Recovery & Margin vs Shuffled Native
    mean_rec, mean_shuf_rec, margin = self._calc_recovery(
      tokens,
      mask,
      cell,
      n_samples,
      mask_sum,
      seq_len,
      flags,
    )

    # 5. Scores, NLL & Perplexity
    score_nll, perplexity = self._calc_scores(cell)
    flags["is_score_derived"] = bool(cell.is_score_derived)
    flags["score_derived"] = bool(cell.is_score_derived)

    # 6. Duplicate Rate
    unique_count = len(set(cell.sequences))
    duplicate_rate = float(1.0 - unique_count / n_samples) if n_samples > 0 else 0.0

    metrics: dict[str, float] = {
      "diversity": mean_diversity,
      "pairwise_identity": mean_pid,
      "mean_pairwise_identity": mean_pid,
      "positional_entropy": mean_pos_entropy,
      "mean_positional_entropy": mean_pos_entropy,
      "composition_entropy": comp_entropy,
      "recovery": mean_rec,
      "native_recovery": mean_rec,
      "recovery_shuffled": mean_shuf_rec,
      "recovery_shuffled_native": mean_shuf_rec,
      "recovery_margin": margin,
      "nll": score_nll,
      "score_nll": score_nll,
      "mean_nll": score_nll,
      "perplexity": perplexity,
      "duplicate_rate": duplicate_rate,
    }

    result = DomainMetricResult(
      metrics=metrics,
      flags=flags,
      tracks=tracks,
    )
    result.validate_schema()
    return result

  def _calc_diversity(
    self,
    tokens: np.ndarray,
    mask: np.ndarray,
    n_samples: int,
    mask_sum: int,
    flags: dict[str, bool],
  ) -> tuple[float, float]:
    """Calculate mean pairwise identity and diversity."""
    if n_samples <= 1:
      flags["single_sample"] = True
      return 0.0, (1.0 if n_samples == 1 else 0.0)
    if mask_sum == 0:
      flags["zero_designable_positions"] = True
      return 0.0, 0.0

    match = (tokens[:, None, :] == tokens[None, :, :]) & mask[None, None, :]
    pid_mat = np.sum(match, axis=-1) / mask_sum
    triu_idx = np.triu_indices(n_samples, k=1)
    mean_pid = float(np.mean(pid_mat[triu_idx]))
    mean_diversity = float(1.0 - mean_pid)
    return mean_diversity, mean_pid

  def _calc_composition_entropy(
    self,
    tokens: np.ndarray,
    mask: np.ndarray,
    n_samples: int,
    mask_sum: int,
    flags: dict[str, bool],
  ) -> float:
    """Calculate composition Shannon entropy in bits."""
    if n_samples > 0 and mask_sum > 0:
      des_tokens = tokens[:, mask].ravel()
      comp_counts = np.bincount(des_tokens, minlength=21)[:21]
      nonzero_comp = comp_counts > 0
      if np.any(nonzero_comp):
        p_comp = comp_counts[nonzero_comp].astype(np.float64) / des_tokens.size
        comp_entropy = float(-np.sum(p_comp * np.log2(p_comp)))
      else:
        comp_entropy = 0.0
    else:
      comp_entropy = 0.0

    flags["library_collapse"] = bool(comp_entropy < 3.5)
    return comp_entropy

  def _calc_recovery(
    self,
    tokens: np.ndarray,
    mask: np.ndarray,
    cell: LibraryCell,
    n_samples: int,
    mask_sum: int,
    seq_len: int,
    flags: dict[str, bool],
  ) -> tuple[float, float, float]:
    """Calculate native recovery and shuffled native margin."""
    native_tokens_str = cell.residues.native_tokens
    native_toks = np.array(
      [MPNN_ALPHABET.index(c) if c in MPNN_ALPHABET else 20 for c in native_tokens_str],
      dtype=tokens.dtype,
    )
    l_nat = min(len(native_toks), seq_len)
    ref = np.full(seq_len, 20, dtype=tokens.dtype)
    ref[:l_nat] = native_toks[:l_nat]

    if mask_sum > 0 and n_samples > 0:
      hit = (tokens == ref[None, :]) & mask[None, :]
      rec_per_sample = np.sum(hit, axis=-1) / mask_sum
      mean_rec = float(np.mean(rec_per_sample))

      # Deterministic seed for shuffled native baseline
      rng = np.random.default_rng(42)
      des_native_idx = np.where(mask[:l_nat])[0]
      shuffled_ref = ref.copy()
      if len(des_native_idx) > 1:
        shuffled_ref[des_native_idx] = shuffled_ref[rng.permutation(des_native_idx)]
      hit_shuf = (tokens == shuffled_ref[None, :]) & mask[None, :]
      rec_shuf_per_sample = np.sum(hit_shuf, axis=-1) / mask_sum
      mean_shuf_rec = float(np.mean(rec_shuf_per_sample))

      margin = mean_rec - mean_shuf_rec
    else:
      mean_rec = 0.0
      mean_shuf_rec = 0.0
      margin = 0.0

    is_degenerate = bool(margin < 0.10)
    flags["degenerate"] = is_degenerate
    flags["degenerate_arm"] = is_degenerate
    return mean_rec, mean_shuf_rec, margin

  def _calc_scores(self, cell: LibraryCell) -> tuple[float, float]:
    """Calculate NLL score and perplexity."""
    if cell.scores is not None and len(cell.scores) > 0:
      score_nll = float(np.mean(cell.scores))
      perplexity = float(np.exp(min(score_nll, 50.0)))
    else:
      score_nll = 0.0
      perplexity = 1.0
    return score_nll, perplexity
