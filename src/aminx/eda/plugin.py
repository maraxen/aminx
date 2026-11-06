"""aminx Domain Plugin for xtrax Campaign EDA.

Implements the AminxDomainPlugin conforming to the xtrax.eda.domains protocol.
Extracts LibraryCells from aminx campaign stores (sequences, logits, scores with
derived NLL fallback and perplexity, MPNN 21-character alphabet normalization,
target native recovery, and designable masking) and computes domain-specific
library metrics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# Standard biomolecular alphabets (21 characters)
MPNN_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWYX"
AF2_ALPHABET: str = "ARNDCQEGHILKMFPSTWYVX"
AF2_TO_MPNN: np.ndarray = np.array([MPNN_ALPHABET.index(c) for c in AF2_ALPHABET], dtype=np.int64)
MPNN_TO_AF2: np.ndarray = np.array([AF2_ALPHABET.index(c) for c in MPNN_ALPHABET], dtype=np.int64)

# Try importing standard domain protocol and dataclasses from xtrax.eda.campaign
try:
    from xtrax.eda.campaign.schema import (
        DomainMetricResult as XtraxDomainMetricResult,
        LibraryCell as XtraxLibraryCell,
        ResidueTable as XtraxResidueTable,
    )
    from xtrax.eda.campaign.plugins import DomainPlugin as XtraxDomainPlugin
except ImportError:
    XtraxResidueTable = None
    XtraxLibraryCell = None
    XtraxDomainMetricResult = None
    XtraxDomainPlugin = None


@dataclass(frozen=True, slots=True)
class ResidueTable:
    """Canonical structural residue mapping."""

    native_tokens: str                # Normalized single-letter amino acid string
    chain_ids: tuple[str, ...]        # Chain identifier per residue
    res_numbers: tuple[int, ...]      # PDB author residue number per residue
    insertion_codes: tuple[str, ...]  # PDB insertion code per residue
    real_mask: np.ndarray             # Boolean array (True for real target residues)
    parser_name: str                  # Name of producing parser (e.g. "proxide")
    parser_version: str               # Version string of producing parser


@dataclass(frozen=True, slots=True)
class LibraryCell:
    """Standardized extracted data cell for a single experimental unit."""

    knobs: Mapping[str, Any]          # Merged row and sink knob dictionary
    seed: int                         # Stochastic replicate seed
    target: str                       # Target identifier
    tokens: np.ndarray                # Integer token array (N, L)
    sequences: tuple[str, ...]        # Decoded sequence strings (N,)
    logits: np.ndarray | None         # Unnormalized model logits (N, L, alphabet_size) or None
    scores: np.ndarray | None         # Model-reported scores (N,) or None
    is_score_derived: bool            # True if scores were derived host-side from logits
    residues: ResidueTable            # Target residue mapping table
    mask: np.ndarray                  # Boolean designable mask (L,)


@dataclass(frozen=True, slots=True)
class DomainMetricResult:
    """Named scalar metrics computed for a cell with strict type and finiteness contracts."""

    metrics: Mapping[str, float]      # Scalar metrics (e.g. {"diversity": 0.45, "rec": 0.62})
    flags: Mapping[str, bool]         # Boolean diagnostic flags (e.g. {"degenerate": False})
    tracks: Mapping[str, np.ndarray]  # Per-residue 1D arrays (e.g. {"entropy": H_l})

    def validate_schema(self) -> None:
        """Validate that all metric values are strictly numeric and finite.

        Enforces that every value in `metrics` satisfies `isinstance(v, (int, float))` and
        `math.isfinite(v)` (non-NaN and non-Inf). Violations trigger immediate unit quarantine,
        guaranteeing that downstream PyArrow Parquet table conversion never fails with ArrowInvalid.
        """
        for k, v in self.metrics.items():
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                raise TypeError(
                    f"Domain metric {k!r} has invalid value {v!r} (type {type(v).__name__}); "
                    "metrics must be strictly finite floats or ints."
                )


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
        store_path: Path | Any,
        manifest_row: Any,
        sink_schema: Any = None,
    ) -> Sequence[LibraryCell]:
        """Extract one or more LibraryCells from a committed unit store."""
        ...

    def compute_metrics(self, cell: LibraryCell) -> DomainMetricResult:
        """Compute domain-specific scalar and per-residue metrics."""
        ...


def _get_row_attr(row: Any, key: str, default: Any = None) -> Any:
    """Safely extract field from either dict or dataclass row representation."""
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _decode_tokens_to_seqs(tokens: np.ndarray) -> tuple[str, ...]:
    """Decode (N, L) integer tokens array to sequence strings using MPNN alphabet."""
    n_samples, seq_len = tokens.shape
    alpha = MPNN_ALPHABET
    res: list[str] = []
    for i in range(n_samples):
        row = tokens[i]
        chars = [alpha[t] if 0 <= t < 21 else "X" for t in row]
        res.append("".join(chars))
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
    tok_logits = np.take_along_axis(logits, safe_tokens[..., None].astype(np.int64), axis=-1)[..., 0]

    nll_pos = lse - tok_logits  # shape (N, L)
    # Average over designable positions only
    masked_nll = nll_pos * mask[None, :]
    return np.sum(masked_nll, axis=-1) / mask_sum


class AminxDomainPlugin:
    """Domain plugin extracting and computing generative protein sequence metrics for aminx."""

    @property
    def domain_name(self) -> str:
        return "aminx"

    @property
    def version(self) -> str:
        try:
            import importlib.metadata as md
            return md.version("aminx")
        except Exception:
            return "0.4.0"

    def _load_sink_data(self, store_path: Path | Any, manifest_row: Any) -> Mapping[str, np.ndarray]:
        """Load sink arrays from path, group, or mapping."""
        if isinstance(store_path, Mapping):
            return {k: np.asarray(v) for k, v in store_path.items()}

        import zarr

        path = Path(store_path)
        if path.is_file() and path.name.endswith(".npy"):
            # Single numpy file
            return {"sequences": np.load(path)}

        # Open zarr group
        group = zarr.open_group(str(path), mode="r")
        # Check if arrays are at root or in subgroup
        row_hash = str(_get_row_attr(manifest_row, "manifest_row_hash", ""))
        target_group = group
        if "sequences" not in group:
            if row_hash and row_hash in group:
                target_group = group[row_hash]
            else:
                keys = list(group.group_keys())
                if len(keys) == 1:
                    target_group = group[keys[0]]

        out: dict[str, np.ndarray] = {}
        for key in ("sequences", "logits", "scores", "fixed_mask", "fixed_tokens", "native"):
            if key in target_group:
                out[key] = np.asarray(target_group[key])

        # Also store residue_table attribute if embedded
        if hasattr(target_group, "attrs"):
            if "residue_table" in target_group.attrs:
                out["__residue_table_attr__"] = target_group.attrs["residue_table"]

        return out

    def _resolve_residue_table(
        self,
        sink_data: Mapping[str, np.ndarray],
        manifest_row: Any,
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
                    real_mask=np.asarray(attr_val.get("real_mask", np.ones(l_target, dtype=bool)), dtype=bool),
                    parser_name=str(attr_val.get("parser_name", "proxide")),
                    parser_version=str(attr_val.get("parser_version", "1.0.0")),
                )

        # 2. Target input parsing fallback if target_input exists on disk
        target_input = _get_row_attr(manifest_row, "target_input", "")
        if target_input and Path(target_input).exists():
            try:
                from aminx.io.parsing.dispatch import parse_input
                proteins = list(parse_input(target_input))
                if proteins:
                    p = proteins[0]
                    aa = np.asarray(p.aatype).astype(int)
                    resnum = np.asarray(p.residue_index).astype(int)
                    chain = np.asarray(p.chain_index).astype(int)
                    mpnn_tokens = AF2_TO_MPNN[aa]
                    native_str = "".join(MPNN_ALPHABET[tok] for tok in mpnn_tokens)
                    try:
                        import importlib.metadata as md
                        proxide_ver = md.version("proxide")
                    except Exception:
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
            except Exception as e:
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
        store_path: Path | Any,
        manifest_row: Any,
        sink_schema: Any = None,
    ) -> Sequence[LibraryCell]:
        """Extract one or more LibraryCells from a committed unit store."""
        sink = self._load_sink_data(store_path, manifest_row)
        if "sequences" not in sink:
            msg = f"Store at {store_path} missing required 'sequences' array"
            raise ValueError(msg)

        seqs_raw = sink["sequences"]
        logits_raw = sink.get("logits")
        scores_raw = sink.get("scores")
        fixed_mask_raw = sink.get("fixed_mask")

        # Base knobs from manifest row
        base_knobs: dict[str, Any] = {}
        for k in ("arm_id", "arm_backend", "checkpoint_id", "fixed_policy", "state_weight_profile"):
            val = _get_row_attr(manifest_row, k)
            if val is not None:
                base_knobs[k] = val
        params = _get_row_attr(manifest_row, "parameters")
        if isinstance(params, Mapping):
            base_knobs.update(params)

        seed = int(_get_row_attr(manifest_row, "seed", 0))
        target_input = str(_get_row_attr(manifest_row, "target_input", ""))
        target = Path(target_input).stem if target_input else "target"

        # Determine dimensions and slicing
        # Possible shapes for sequences:
        # 1D: (L,) -> (1, L)
        # 2D: (N, L)
        # 3D: (N, T, L)
        # 4D: (N, G, T, L)
        slices: list[tuple[dict[str, Any], np.ndarray, np.ndarray | None, np.ndarray | None]] = []

        if seqs_raw.ndim == 1:
            slices.append((dict(base_knobs), seqs_raw[None, :], logits_raw[None, :] if logits_raw is not None else None, scores_raw))
        elif seqs_raw.ndim == 2:
            slices.append((dict(base_knobs), seqs_raw, logits_raw, scores_raw))
        elif seqs_raw.ndim == 3:
            # (sample, temperature, residue)
            n_temps = seqs_raw.shape[1]
            row_temps = base_knobs.get("temperature", [float(t) for t in range(n_temps)])
            if not isinstance(row_temps, (list, tuple)) or len(row_temps) != n_temps:
                row_temps = [float(t) for t in range(n_temps)]
            for ti in range(n_temps):
                knob_slice = dict(base_knobs)
                knob_slice["temperature"] = row_temps[ti]
                seq_slice = seqs_raw[:, ti, :]
                logit_slice = logits_raw[:, ti, ...] if logits_raw is not None and logits_raw.ndim >= 3 else None
                score_slice = scores_raw[:, ti] if scores_raw is not None and scores_raw.ndim >= 2 else scores_raw
                slices.append((knob_slice, seq_slice, logit_slice, score_slice))
        elif seqs_raw.ndim == 4:
            # (sample, grid, temperature, residue)
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
                    logit_slice = logits_raw[:, gi, ti, ...] if logits_raw is not None and logits_raw.ndim >= 4 else None
                    score_slice = scores_raw[:, gi, ti] if scores_raw is not None and scores_raw.ndim >= 3 else scores_raw
                    slices.append((knob_slice, seq_slice, logit_slice, score_slice))
        else:
            slices.append((dict(base_knobs), seqs_raw.reshape(-1, seqs_raw.shape[-1]), None, None))

        cells: list[LibraryCell] = []
        for knob_dict, s_arr, l_arr, sc_arr in slices:
            n_samples, seq_len = s_arr.shape

            # Convert sequences to integer tokens and decoded strings
            if np.issubdtype(s_arr.dtype, np.integer):
                tokens = np.array(s_arr, dtype=np.int64, copy=True)
                sequences = _decode_tokens_to_seqs(tokens)
            else:
                # String or byte array
                sequences = tuple(
                    x.decode("utf-8") if isinstance(x, bytes) else str(x)
                    for x in s_arr.ravel()
                )[:n_samples]
                tokens = _encode_seqs_to_tokens(sequences, max_len=seq_len)

            # Build ResidueTable
            res_table = self._resolve_residue_table(sink, manifest_row, seq_len)

            # Construct designable mask: real_mask & ~fixed_mask
            real_m = np.zeros(seq_len, dtype=bool)
            l_real = min(len(res_table.native_tokens), seq_len)
            real_m[:l_real] = res_table.real_mask[:l_real] if len(res_table.real_mask) >= l_real else True

            if fixed_mask_raw is not None:
                fm = np.asarray(fixed_mask_raw, dtype=bool).ravel()
                fm_padded = np.zeros(seq_len, dtype=bool)
                fm_padded[:min(len(fm), seq_len)] = fm[:min(len(fm), seq_len)]
                designable = real_m & ~fm_padded
            else:
                designable = real_m

            # Format logits
            logits: np.ndarray | None = None
            if l_arr is not None:
                logits = np.array(l_arr, dtype=np.float32, copy=True)
                if logits.ndim == 2:
                    logits = logits[None, ...]

            # Scores and derived fallback
            scores: np.ndarray | None = None
            is_derived = False
            if sc_arr is not None:
                scores = np.array(sc_arr, dtype=np.float64, copy=True).ravel()
                is_derived = False
            elif logits is not None:
                scores = _compute_derived_nll(tokens, logits, designable)
                is_derived = True

            # Zero-copy immutability: set writeable=False (MAJ-01)
            tokens.flags.writeable = False
            designable.flags.writeable = False
            if logits is not None:
                logits.flags.writeable = False
            if scores is not None:
                scores.flags.writeable = False
            if hasattr(res_table, "real_mask") and isinstance(res_table.real_mask, np.ndarray):
                res_table.real_mask.flags.writeable = False

            cell = LibraryCell(
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
            cells.append(cell)

        return cells

    def compute_metrics(self, cell: LibraryCell) -> DomainMetricResult:
        """Compute domain-specific scalar and per-residue metrics for a LibraryCell."""
        tokens = cell.tokens
        mask = cell.mask
        n_samples, seq_len = tokens.shape
        mask_sum = int(np.sum(mask))

        flags: dict[str, bool] = {}
        tracks: dict[str, np.ndarray] = {}

        # 1. Pairwise Identity & Hamming Diversity (BLK-04 guards)
        if n_samples <= 1:
            mean_diversity = 0.0
            mean_pid = 1.0 if n_samples == 1 else 0.0
            flags["single_sample"] = True
        elif mask_sum == 0:
            mean_diversity = 0.0
            mean_pid = 0.0
            flags["zero_designable_positions"] = True
        else:
            # Broadcast pairwise match across designable residues
            match = (tokens[:, None, :] == tokens[None, :, :]) & mask[None, None, :]
            pid_mat = np.sum(match, axis=-1) / mask_sum  # (N, N)
            triu_idx = np.triu_indices(n_samples, k=1)
            mean_pid = float(np.mean(pid_mat[triu_idx]))
            mean_diversity = float(1.0 - mean_pid)

        # 2. Positional Shannon Entropy (safe positive-support, bits, MAJ-07)
        h_track = np.zeros(seq_len, dtype=np.float64)
        if cell.logits is not None:
            flat = cell.logits.astype(np.float64)
            max_l = np.max(flat, axis=-1, keepdims=True)
            exp_l = np.exp(flat - max_l)
            p = exp_l / np.clip(np.sum(exp_l, axis=-1, keepdims=True), 1e-300, None)
            mean_p = np.mean(p, axis=0)  # (L, 21)
            for pos in range(seq_len):
                p_c = mean_p[pos]
                nonzero = p_c > 0
                if np.any(nonzero):
                    p_pos = p_c[nonzero]
                    h_track[pos] = float(-np.sum(p_pos * np.log2(p_pos)))
        else:
            for pos in range(seq_len):
                if n_samples > 0:
                    counts = np.bincount(tokens[:, pos], minlength=21)[:21]
                    nonzero = counts > 0
                    if np.any(nonzero):
                        p_c = counts[nonzero].astype(np.float64) / n_samples
                        h_track[pos] = float(-np.sum(p_c * np.log2(p_c)))

        tracks["positional_entropy"] = h_track
        mean_pos_entropy = float(np.mean(h_track[mask])) if mask_sum > 0 else 0.0

        # 3. Composition Shannon Entropy
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

        # 4. Target Native Recovery & Margin vs Shuffled Native
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

            # Composition-preserving shuffled native baseline
            rng = np.random.default_rng(cell.seed)
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

        # 5. Scores & Perplexity
        if cell.scores is not None and len(cell.scores) > 0:
            score_nll = float(np.mean(cell.scores))
            perplexity = float(np.exp(min(score_nll, 50.0)))
        else:
            score_nll = 0.0
            perplexity = 1.0
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
            "recovery_shuffled_native": mean_shuf_rec,
            "recovery_margin": margin,
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
