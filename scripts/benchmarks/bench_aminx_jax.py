#!/usr/bin/env python3
"""GPU benchmark adapter for aminx (JAX+Equinox) inference.

Measures cold-compile time, warm latency, and GPU memory usage for each
(seq_len, batch_size, precision, task) cell. Produces JSON output conforming to
the benchmark suite specification.

Usage:
    uv run python scripts/benchmarks/bench_aminx_jax.py --dry-run
    uv run python scripts/benchmarks/bench_aminx_jax.py --smoke
    uv run python scripts/benchmarks/bench_aminx_jax.py \
        --task score_conditional \
        --seq-lens 76 500 \
        --batch-sizes 1 4 16 \
        --precision fp32 \
        --hardware A100 \
        --n-warmup 10 \
        --n-timed 20 \
        --pdb-dir tests/data \
        --output-json results.json

Exit codes:
    0: SUCCESS
    1: FAILURE (missing fixtures, model load error, or benchmark error)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

if TYPE_CHECKING:
    from jaxtyping import PRNGKeyArray

# Suppress JAX warnings
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Element list matching LigandMPNN data_utils.py — index 0 = unknown, 1-indexed thereafter
_ELEMENT_LIST: list[str] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
]
_ELEMENT_LIST_UPPER: list[str] = [e.upper() for e in _ELEMENT_LIST]
_ELEMENT_TO_IDX: dict[str, int] = {e: i + 1 for i, e in enumerate(_ELEMENT_LIST_UPPER)}

# Canonical PDB map: nominal seq_len -> pdb filename
_PDB_MAP: dict[int, str] = {
    76: "1ubq.pdb",
    94: "1BC8.cif",
    150: "1mbn.pdb",
    300: "3pgk.pdb",
    500: "1SMD.pdb",
}

# Default PDB directory relative to the aminx package root (this file is scripts/benchmarks/)
_DEFAULT_PDB_DIR = Path(__file__).parents[2] / "tests" / "data"

# Standard 20-letter amino acid alphabet (index 0-19), unknown=20
_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_IDX = {aa: i for i, aa in enumerate(_AA_ALPHABET)}


# ============================================================================
# Configuration & Utilities
# ============================================================================


def _set_jax_defaults():
    """Set JAX configuration before importing models."""
    # Blackwell workaround: set XLA flags before any compilation
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_shard_autotuning=false")


def _get_cuda_version() -> str | None:
    """Try to get CUDA version, return None if unavailable."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "unknown"  # Can't easily extract full version; caller can query nvidia-smi
    except Exception:
        pass
    return None


def _query_gpu_memory_gb() -> float:
    """Query current GPU memory usage via nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            mb = int(result.stdout.strip().split("\n")[0].strip())
            return mb / 1024.0
    except Exception:
        pass
    return 0.0


class _BenchmarkSpec:
    """Minimal spec for make_inference_plan."""

    sampling_strategy = "temperature"
    use_rolling_state = False
    multi_state_strategy = "arithmetic_mean"
    multi_state_temperature = 1.0
    state_weights = None
    temperature = [1.0]
    average_node_features = False  # No encoding fusion for Wave 1


# make_inference_plan reads decode-relevant fields from spec.run_spec.sampling (260716, EPIC
# #1541 P4); build_run_spec() handles any duck spec via getattr(spec, "field", default).
def _attach_run_spec() -> None:
    from aminx.run.spec import build_run_spec

    _BenchmarkSpec.run_spec = build_run_spec(_BenchmarkSpec)


_attach_run_spec()


def _make_benchmark_spec_with_temperatures(temperature_list: list[float] | None = None) -> _BenchmarkSpec:
    """Create a benchmark spec with custom temperature list.

    Parameters
    ----------
    temperature_list : list[float] | None
        If provided, override the default temperature [1.0].

    Returns
    -------
    _BenchmarkSpec
        Spec with temperature set to temperature_list or [1.0] if None.
    """
    spec = _BenchmarkSpec()
    if temperature_list is not None:
        spec.temperature = temperature_list
    return spec


# ============================================================================
# PDB Loading
# ============================================================================


def load_pdb_as_arrays(pdb_path: str) -> dict[str, Any]:
    """Load a PDB file and return coordinate arrays using biopython.

    Parameters
    ----------
    pdb_path : str
        Path to PDB file.

    Returns
    -------
    dict with keys:
        coords:         float32 (L, 4, 3)   - N, CA, C, O in angstroms
        mask:           float32 (L,)         - 1.0 where residue is present
        sequence:       int32   (L,)         - amino acid indices 0-19 (unk=20)
        residue_index:  int32   (L,)         - per-residue sequence number
        chain_index:    int32   (L,)         - per-chain integer index (0-based)
        actual_len:     int                  - L
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    coords_list = []
    mask_list = []
    seq_list = []
    residue_index_list = []
    chain_index_list = []

    # Sort chains alphabetically and assign 0-based integer indices
    model_obj = next(iter(structure))
    chain_ids = sorted(chain.id for chain in model_obj)
    chain_to_idx = {cid: i for i, cid in enumerate(chain_ids)}

    atom_order = {"N": 0, "CA": 1, "C": 2, "O": 3}

    for chain in model_obj:
        chain_idx = chain_to_idx[chain.id]
        for residue in chain:
            # Skip HETATM residues (waters, ligands)
            if residue.get_id()[0] != " ":
                continue

            res_coord = np.zeros((4, 3), dtype=np.float32)
            atom_found = [False, False, False, False]

            for atom_name, atom_idx in atom_order.items():
                if atom_name in residue:
                    res_coord[atom_idx] = residue[atom_name].get_vector().get_array()
                    atom_found[atom_idx] = True

            # Mark present if at least CA exists
            res_mask = 1.0 if atom_found[1] else 0.0

            # Encode residue name: 3-letter -> 1-letter -> integer index
            resname = residue.get_resname().strip()
            try:
                from Bio.Data.IUPACData import protein_letters_3to1
                one_letter = protein_letters_3to1.get(resname.capitalize(), "X")
                aa_idx = _AA_TO_IDX.get(one_letter, 20)
            except Exception:
                aa_idx = 20

            res_idx = residue.get_id()[1]

            coords_list.append(res_coord)
            mask_list.append(res_mask)
            seq_list.append(aa_idx)
            residue_index_list.append(res_idx)
            chain_index_list.append(chain_idx)

    L = len(coords_list)
    return {
        "coords": np.array(coords_list, dtype=np.float32),              # (L, 4, 3)
        "mask": np.array(mask_list, dtype=np.float32),                   # (L,)
        "sequence": np.array(seq_list, dtype=np.int32),                  # (L,)
        "residue_index": np.array(residue_index_list, dtype=np.int32),  # (L,)
        "chain_index": np.array(chain_index_list, dtype=np.int32),      # (L,)
        "actual_len": L,
    }


def _load_pdb_fixture(pdb_dir: Path, seq_len: int) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int
]:
    """Load PDB fixture for a given nominal seq_len.

    Returns
    -------
    tuple
        (coords, mask, sequence, residue_index, chain_index, actual_len)
    """
    if seq_len not in _PDB_MAP:
        raise FileNotFoundError(
            f"No PDB fixture for seq_len={seq_len}. "
            f"Available: {list(_PDB_MAP.keys())}"
        )

    pdb_file = pdb_dir / _PDB_MAP[seq_len]
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB fixture not found: {pdb_file}")

    data = load_pdb_as_arrays(str(pdb_file))

    return (
        jnp.asarray(data["coords"], dtype=jnp.float32),
        jnp.asarray(data["mask"], dtype=jnp.float32),
        jnp.asarray(data["sequence"], dtype=jnp.int32),
        jnp.asarray(data["residue_index"], dtype=jnp.int32),
        jnp.asarray(data["chain_index"], dtype=jnp.int32),
        data["actual_len"],
    )


def _load_ligand_fixture(
    pdb_dir: Path, seq_len: int
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
    np.ndarray, np.ndarray,
    int,
] | None:
    """Load protein + ligand arrays from a CIF file.

    Returns protein backbone arrays (same as _load_pdb_fixture) plus raw ligand atom
    coordinates and element symbols. Returns None if the fixture is not a CIF file.
    Only handles 1BC8.cif — protein = chain C, ligand = chains A+B (non-water).
    """
    filename = _PDB_MAP.get(seq_len)
    if filename is None or not filename.endswith(".cif"):
        return None

    cif_path = pdb_dir / filename
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF fixture not found: {cif_path}")

    protein_atoms: list[tuple[str, int, float, float, float]] = []
    ligand_xyz: list[tuple[float, float, float]] = []
    ligand_elements: list[str] = []

    lines = cif_path.read_text().splitlines()
    in_loop = False
    col_names: list[str] = []
    col_map: dict[str, int] = {}

    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            col_names = []
            col_map = {}
            in_loop = True
            continue
        if in_loop and stripped.startswith("_atom_site."):
            col_names.append(stripped)
            col_map[stripped] = len(col_names) - 1
            continue
        if in_loop and col_names and not stripped.startswith("_") and stripped and not stripped.startswith("#"):
            parts = stripped.split()
            if len(parts) < len(col_names):
                continue
            group = parts[col_map.get("_atom_site.group_PDB", 0)]
            if group not in ("ATOM", "HETATM"):
                continue
            element = parts[col_map["_atom_site.type_symbol"]].upper()
            resname = parts[col_map["_atom_site.label_comp_id"]]
            auth_chain = parts[col_map["_atom_site.auth_asym_id"]]
            atom_name = parts[col_map["_atom_site.label_atom_id"]].strip('"')
            try:
                auth_seq = int(parts[col_map["_atom_site.auth_seq_id"]])
                x = float(parts[col_map["_atom_site.Cartn_x"]])
                y = float(parts[col_map["_atom_site.Cartn_y"]])
                z = float(parts[col_map["_atom_site.Cartn_z"]])
            except (ValueError, KeyError):
                continue

            if auth_chain == "C" and resname not in ("HOH",):
                protein_atoms.append((atom_name, auth_seq, x, y, z))
            elif auth_chain in ("A", "B") and resname not in ("HOH",):
                ligand_xyz.append((x, y, z))
                ligand_elements.append(element)

        elif in_loop and col_names and (stripped.startswith("_") or stripped == "loop_" or stripped == "#"):
            in_loop = False

    if not protein_atoms:
        raise ValueError(f"No protein atoms found in chain C of {cif_path}")

    from collections import defaultdict
    residue_atoms: dict[int, dict[str, tuple[float, float, float]]] = defaultdict(dict)
    for atom_name, res_seq, x, y, z in protein_atoms:
        if atom_name in ("N", "CA", "C", "O"):
            residue_atoms[res_seq][atom_name] = (x, y, z)

    sorted_resids = sorted(residue_atoms.keys())
    BACKBONE = ["N", "CA", "C", "O"]
    coords_list = []
    mask_list = []
    for res in sorted_resids:
        row = []
        present = True
        for atom in BACKBONE:
            if atom in residue_atoms[res]:
                row.append(residue_atoms[res][atom])
            else:
                row.append((0.0, 0.0, 0.0))
                present = False
        coords_list.append(row)
        mask_list.append(1.0 if present else 0.0)

    actual_len = len(sorted_resids)
    coords = jnp.array(coords_list, dtype=jnp.float32)
    mask = jnp.array(mask_list, dtype=jnp.float32)
    sequence = jnp.zeros(actual_len, dtype=jnp.int32)
    residue_index = jnp.array(sorted_resids, dtype=jnp.int32)
    chain_index = jnp.zeros(actual_len, dtype=jnp.int32)

    lig_xyz_arr = np.array(ligand_xyz, dtype=np.float32)
    lig_elem_arr = np.array(ligand_elements)

    return coords, mask, sequence, residue_index, chain_index, lig_xyz_arr, lig_elem_arr, actual_len


def _compute_ligand_nn(
    coords: jnp.ndarray,
    ligand_xyz: np.ndarray,
    ligand_elements: np.ndarray,
    num_neighbors: int = 16,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute per-residue nearest-neighbor ligand features (host-side).

    For each protein residue CA, finds the `num_neighbors` nearest ligand atoms.
    Returns arrays shaped (L, A, 3), (L, A), (L, A) ready for build_inference_bundle.
    """
    from scipy.spatial import cKDTree

    ca_coords = np.array(coords[:, 1, :])
    L = ca_coords.shape[0]
    A = num_neighbors

    tree = cKDTree(ligand_xyz)
    n_query = min(A, len(ligand_xyz))
    dists, idxs = tree.query(ca_coords, k=n_query)

    nn_coords = np.zeros((L, A, 3), dtype=np.float32)
    nn_types = np.zeros((L, A), dtype=np.int32)
    nn_mask = np.zeros((L, A), dtype=np.float32)

    for i in range(L):
        for j in range(n_query):
            idx = idxs[i, j] if n_query > 1 else idxs[i]
            nn_coords[i, j] = ligand_xyz[idx]
            elem = ligand_elements[idx].upper()
            nn_types[i, j] = _ELEMENT_TO_IDX.get(elem, 0)
            nn_mask[i, j] = 1.0

    return (
        jnp.array(nn_coords),
        jnp.array(nn_types),
        jnp.array(nn_mask),
    )


# ============================================================================
# Model & Plan Setup
# ============================================================================


_DEFAULT_CHECKPOINT_ID = "proteinmpnn_v_48_020"


def load_model(
    checkpoint_path: Path | None = None,
    checkpoint_id: str | None = None,
) -> Any:
    """Load pre-trained model via io.weights.load_model.

    checkpoint_id controls both topology (ligand vs protein architecture) and
    which bundled weights to load. Defaults to _DEFAULT_CHECKPOINT_ID when omitted.
    Pass checkpoint_path to override the weight bytes with a local .eqx file while
    still using checkpoint_id to determine the architecture.
    """
    from aminx.io.weights import load_model as _load

    effective_id = checkpoint_id or _DEFAULT_CHECKPOINT_ID
    key = random.PRNGKey(42)
    local_path = str(checkpoint_path) if checkpoint_path is not None else None
    model = _load(
        checkpoint_id=effective_id,
        local_path=local_path,
        key=key,
    )
    if local_path:
        logger.info(f"Loaded checkpoint: {local_path} (id={effective_id})")
    else:
        logger.info(f"Loaded bundled checkpoint: {effective_id}")
    return model


def create_inference_plan(model: Any, task: str, spec: _BenchmarkSpec | None = None) -> Any:
    """Create InferencePlan from model, configured for the given task.

    Parameters
    ----------
    task : str
        "score_conditional" uses ConditionalMode (default make_inference_plan).
        "ar_sample" builds InferencePlan with AutoregressiveMode.
    spec : _BenchmarkSpec | None
        If provided, use this spec; otherwise create default _BenchmarkSpec.
    """
    from aminx.host.plan import make_inference_plan

    if spec is None:
        spec = _BenchmarkSpec()

    if task == "ar_sample":
        from aminx.host.plan import InferenceComponents, InferencePlan
        from aminx.inference.decode.factory import make_decode_fn
        from aminx.inference.decode.mode import AutoregressiveConfig, AutoregressiveMode
        from aminx.inference.encode import make_encode_fn
        from aminx.inference.logits import make_stage_set
        from aminx.tiling.strategy import Vmap

        stage_set = make_stage_set(
            spec.multi_state_strategy,
            spec.multi_state_temperature,
            spec.state_weights,
        )
        encode_fn = make_encode_fn(model, use_rolling_state=spec.use_rolling_state)
        # inference_only moved from AutoregressiveMode to AutoregressiveConfig; passing it
        # to the mode raised TypeError. Dropping it instead of relocating it would have
        # silently reverted the wave axis to lax.scan (the config default is False), which
        # is not what a benchmark should measure -- AutoregressiveConfig's own docstring
        # says to set it True for inference/benchmarking.
        decode_fn = make_decode_fn(
            model,
            mode=AutoregressiveMode(),
            strategy=Vmap(),
            autoregressive_config=AutoregressiveConfig(inference_only=True),
        )
        components = InferenceComponents(encode_fn=encode_fn, stage_set=stage_set)
        return InferencePlan(model=model, components=components, decode_fn=decode_fn)
    else:
        # score_conditional uses ConditionalMode (make_inference_plan default)
        return make_inference_plan(model, spec)


# ============================================================================
# Timing
# ============================================================================


def measure_cold_compile_score(
    plan: Any,
    bundle: Any,
    key: "PRNGKeyArray",
    config: Any,
    fixture_name: str,
) -> tuple[float, str]:
    """Measure cold XLA compilation time for score_conditional (full encode+decode)."""
    jax.clear_caches()

    t0 = time.perf_counter()
    result = plan.score(bundle, key, config)
    jax.block_until_ready(result)
    compile_time_cold_s = time.perf_counter() - t0

    note = "JAX: XLA compilation; cold run via plan.score() (encode+decode end-to-end)"
    return compile_time_cold_s, note


def measure_cold_compile_sample(
    plan: Any,
    bundle: Any,
    key: "PRNGKeyArray",
    config: Any,
    fixture_name: str,
) -> tuple[float, str]:
    """Measure cold XLA compilation time for ar_sample."""
    jax.clear_caches()

    t0 = time.perf_counter()
    result = plan.sample(bundle, jax.random.fold_in(key, 0), config)
    jax.block_until_ready(result)
    compile_time_cold_s = time.perf_counter() - t0

    note = "JAX: cold run via plan.sample(); key derived with fold_in"
    return compile_time_cold_s, note


def measure_warm_latency_score(
    plan: Any,
    bundle: Any,
    key: "PRNGKeyArray",
    config: Any,
    n_warmup: int,
    n_timed: int,
) -> tuple[float, float, list[float]]:
    """Measure warm latency for score_conditional (full encode+decode per call)."""
    for _ in range(n_warmup):
        result = plan.score(bundle, key, config)
        jax.block_until_ready(result)

    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        result = plan.score(bundle, key, config)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    return float(np.median(times_arr)), float(np.percentile(times_arr, 95)), times


def measure_warm_latency_sample(
    plan: Any,
    bundle: Any,
    key: "PRNGKeyArray",
    config: Any,
    n_warmup: int,
    n_timed: int,
) -> tuple[float, float, list[float]]:
    """Measure warm latency for ar_sample (plan.sample end-to-end).

    Uses fold_in(key, i) per iteration — distinct subkey each call,
    no key array allocation per step.
    """
    for i in range(n_warmup):
        result = plan.sample(bundle, jax.random.fold_in(key, i), config)
        jax.block_until_ready(result)

    times = []
    for i in range(n_timed):
        subkey = jax.random.fold_in(key, n_warmup + i)
        t0 = time.perf_counter()
        result = plan.sample(bundle, subkey, config)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    return float(np.median(times_arr)), float(np.percentile(times_arr, 95)), times


# ============================================================================
# Benchmark Single Cell
# ============================================================================


def benchmark_cell(
    model: Any,
    plan: Any,
    pdb_dir: Path,
    seq_len: int,
    batch_size: int,
    precision: str,
    task: str,
    ligand_enabled: bool = False,
    n_warmup: int = 10,
    n_timed: int = 20,
) -> dict[str, Any] | None:
    """Benchmark a single (seq_len, batch_size, precision, task) cell.

    Parameters
    ----------
    model : Any
        The model instance.
    plan : Any
        Pre-built inference plan (shared across all cells for a given task).
    pdb_dir : Path
        Path to PDB fixture directory.
    seq_len : int
        Sequence length to benchmark.
    batch_size : int
        Batch size (number of states) to benchmark.
    precision : str
        Precision to benchmark.
    task : str
        Task type ('score_conditional' or 'ar_sample').
    ligand_enabled : bool
        Whether to load ligand data.
    n_warmup : int
        Number of warmup iterations.
    n_timed : int
        Number of timed iterations.

    Returns
    -------
    dict[str, Any] | None
        Benchmark result dict or None if cell was skipped.
    """
    from aminx.inference.bundle_builder import build_inference_bundle
    from aminx.tiling.bucketing import BucketingConfig
    _BUCKET_CFG = BucketingConfig()

    try:
        if seq_len not in _PDB_MAP:
            logger.info(
                f"  Skipping L={seq_len} (no PDB fixture; have L={list(_PDB_MAP.keys())})"
            )
            return None

        is_cif = _PDB_MAP.get(seq_len, "").endswith(".cif")

        ligand_coords_jax = None
        ligand_atom_types_jax = None
        ligand_mask_jax = None

        if is_cif:
            if not ligand_enabled:
                logger.info(f"  Skipping L={seq_len} (CIF fixture requires --ligand)")
                return None
            lig_result = _load_ligand_fixture(pdb_dir, seq_len)
            if lig_result is None:
                return None
            coords, mask, sequence, residue_index, chain_index, lig_xyz, lig_elems, actual_len = lig_result
            ligand_coords_jax, ligand_atom_types_jax, ligand_mask_jax = _compute_ligand_nn(
                coords, lig_xyz, lig_elems, num_neighbors=16
            )
        else:
            coords, mask, sequence, residue_index, chain_index, actual_len = _load_pdb_fixture(
                pdb_dir, seq_len
            )
            if ligand_enabled:
                lig_result = _load_ligand_fixture(pdb_dir, seq_len)
                if lig_result is not None:
                    _, _, _, _, _, lig_xyz, lig_elems, _ = lig_result
                    ligand_coords_jax, ligand_atom_types_jax, ligand_mask_jax = _compute_ligand_nn(
                        coords, lig_xyz, lig_elems, num_neighbors=16
                    )

        if seq_len != actual_len:
            logger.info(
                f"  Note: nominal L={seq_len}, loaded L={actual_len} from {_PDB_MAP[seq_len]}"
            )

        # For batch_size > 1: stack geometry along the state axis.
        if batch_size > 1:
            coords = jnp.stack([coords] * batch_size, axis=0)
            mask = jnp.stack([mask] * batch_size, axis=0)
            residue_index = jnp.stack([residue_index] * batch_size, axis=0)
            chain_index = jnp.stack([chain_index] * batch_size, axis=0)
        sequence_stacked = sequence

        if task == "score_conditional":
            bundle, config = build_inference_bundle(
                coords=coords,
                mask=mask,
                residue_index=residue_index,
                chain_index=chain_index,
                sequence=sequence_stacked,    # native sequence from PDB
                ligand_coords=ligand_coords_jax,
                ligand_atom_types=ligand_atom_types_jax,
                ligand_mask=ligand_mask_jax,
                temperature=1.0,
                mode="score_conditional",
                inference=True,
                bucket_config=_BUCKET_CFG,
            )
        else:  # ar_sample
            bundle, config = build_inference_bundle(
                coords=coords,
                mask=mask,
                residue_index=residue_index,
                chain_index=chain_index,
                sequence=None,               # no fixed sequence for AR sampling
                ligand_coords=ligand_coords_jax,
                ligand_atom_types=ligand_atom_types_jax,
                ligand_mask=ligand_mask_jax,
                temperature=1.0,
                mode="sample",
                inference=True,
                bucket_config=_BUCKET_CFG,
            )

        key = random.PRNGKey(42)

        logger.info(
            f"  Computing cold compile time "
            f"(seq_len={actual_len}, batch_size={batch_size}, task={task})..."
        )
        if task == "score_conditional":
            compile_time_s, compile_note = measure_cold_compile_score(
                plan, bundle, key, config, f"L{actual_len}B{batch_size}"
            )
        else:
            compile_time_s, compile_note = measure_cold_compile_sample(
                plan, bundle, key, config, f"L{actual_len}B{batch_size}"
            )

        logger.info("  Computing warm latency...")
        if task == "score_conditional":
            median_s, p95_s, times = measure_warm_latency_score(
                plan, bundle, key, config, n_warmup, n_timed
            )
        else:
            median_s, p95_s, times = measure_warm_latency_sample(
                plan, bundle, key, config, n_warmup, n_timed
            )

        latency_median_ms = median_s * 1000.0
        latency_p95_ms = p95_s * 1000.0
        latency_per_residue_us = (median_s * 1e6) / (actual_len * batch_size)
        throughput_seq_per_s = batch_size / median_s
        peak_gpu_memory_gb = _query_gpu_memory_gb()

        result = {
            "schema_version": "1",
            "model": "aminx_jax",
            "hardware": "unknown",  # Set by caller
            "seq_len": actual_len,
            "batch_size": batch_size,
            "task": task,
            "precision": precision,
            "ligand_conditioning": ligand_enabled,
            "axis_strategy": "Vmap",
            "average_encoding_mode": "inputs_and_noise",
            "compile_time_cold_s": float(compile_time_s),
            "compile_time_warm_s": 0.0,
            "compile_time_note": compile_note,
            "latency_median_ms": float(latency_median_ms),
            "latency_p95_ms": float(latency_p95_ms),
            "latency_per_residue_us": float(latency_per_residue_us),
            "throughput_seq_per_s": float(throughput_seq_per_s),
            "peak_gpu_memory_gb": float(peak_gpu_memory_gb),
            "n_warmup": n_warmup,
            "n_timed": n_timed,
            "jax_version": jax.__version__,
            "torch_version": None,
            "cuda_version": _get_cuda_version(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"  ✓ seq_len={actual_len}, batch_size={batch_size}, task={task}: "
            f"cold={compile_time_s:.3f}s, warm_median={latency_median_ms:.2f}ms"
        )

        return result

    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        return None


# ============================================================================
# Main
# ============================================================================


def main():
    """Run benchmark suite."""
    parser = argparse.ArgumentParser(
        description="GPU benchmark for aminx JAX adapter",
    )
    parser.add_argument(
        "--task",
        choices=["score_conditional", "ar_sample"],
        default="score_conditional",
        help="Task to benchmark (default: score_conditional)",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[76],
        help="Sequence lengths to benchmark (default: [76])",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Batch sizes to benchmark (default: [1])",
    )
    parser.add_argument(
        "--precision",
        type=str,
        nargs="+",
        default=["fp32"],
        choices=["bf16", "fp32"],
        help="Precisions to benchmark (default: [fp32])",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="unknown",
        help="Hardware identifier (default: unknown)",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=10,
        help="Warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--n-timed",
        type=int,
        default=20,
        help="Timed iterations (default: 20)",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=_DEFAULT_PDB_DIR,
        help="Directory containing PDB fixture files 1ubq.pdb and 1SMD.pdb (default: tests/data)",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=None,
        help="[DEPRECATED] Use --pdb-dir instead.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON file (default: stdout)",
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=None,
        help="Path to model checkpoint (.eqx)",
    )
    parser.add_argument(
        "--ligand",
        action="store_true",
        default=False,
        help="Enable ligand conditioning (requires 1BC8.cif fixture and ligand checkpoint).",
    )
    parser.add_argument(
        "--ligand-checkpoint",
        type=str,
        default="ligandmpnn_v_32_010_25",
        help="Checkpoint ID to use for ligand-conditioned runs (default: ligandmpnn_v_32_010_25).",
    )
    parser.add_argument(
        "--temperatures",
        type=str,
        default=None,
        help="Comma-separated temperature values (default: use spec temperature [1.0])",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without running benchmarks",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run minimal benchmark: seq_lens=[76], batch_sizes=[1], n_warmup=1, n_timed=3",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Handle deprecated --fixture-dir
    if args.fixture_dir is not None:
        warnings.warn(
            "--fixture-dir is deprecated; use --pdb-dir instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        if args.pdb_dir == _DEFAULT_PDB_DIR:
            args.pdb_dir = args.fixture_dir

    if args.smoke:
        args.seq_lens = [76]
        args.batch_sizes = [1]
        args.n_warmup = 1
        args.n_timed = 3
        logger.info("Smoke test mode: minimal iterations")

    # Parse --temperatures flag if provided
    temperatures_override = None
    if args.temperatures is not None:
        try:
            temperatures_override = [float(t.strip()) for t in args.temperatures.split(",")]
            logger.info(f"Temperature override: {temperatures_override}")
        except ValueError as e:
            logger.error(f"Failed to parse --temperatures '{args.temperatures}': {e}")
            return 1

    # Resolve checkpoint path (optional — None means use bundled .eqx.zst via io.weights)
    checkpoint_path = args.reference_path

    if args.dry_run:
        config = {
            "task": args.task,
            "seq_lens": args.seq_lens,
            "batch_sizes": args.batch_sizes,
            "precisions": args.precision,
            "hardware": args.hardware,
            "pdb_dir": str(args.pdb_dir),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "n_warmup": args.n_warmup,
            "n_timed": args.n_timed,
            "jax_version": jax.__version__,
            "cuda_available": jax.devices()[0].platform == "gpu" if jax.devices() else False,
        }
        logger.info("DRY RUN - Configuration:")
        logger.info(json.dumps(config, indent=2))
        return 0

    if not args.pdb_dir.exists():
        logger.error(f"PDB directory not found: {args.pdb_dir}")
        return 1

    _set_jax_defaults()

    # Fail loud if a GPU run is silently executing on CPU (e.g. missing CUDA
    # plugin). A SLURM GPU allocation does NOT guarantee jax uses the GPU.
    _backend = jax.default_backend()
    logger.info(f"JAX backend: {_backend}  devices: {jax.devices()}")
    if args.hardware.lower() != "cpu" and _backend != "gpu":
        logger.error(
            f"--hardware {args.hardware} requested but JAX backend is "
            f"'{_backend}', not 'gpu'. Refusing to report CPU timings as "
            f"{args.hardware}. Install the CUDA plugin (uv sync --extra cuda "
            f"with cuda=['jax[cuda]']) or pass --hardware CPU."
        )
        return 1

    if args.ligand:
        logger.info(f"Ligand mode: loading checkpoint {args.ligand_checkpoint}")
        try:
            model = load_model(checkpoint_id=args.ligand_checkpoint)
            logger.info("Ligand checkpoint loaded")
        except Exception as e:
            logger.error(f"Failed to load ligand checkpoint: {e}")
            return 1
    else:
        logger.info(f"Loading model {'with checkpoint' if checkpoint_path else 'with random init'}...")
        try:
            model = load_model(checkpoint_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return 1

    all_results = []
    total_cells = len(args.seq_lens) * len(args.batch_sizes) * len(args.precision)
    current_cell = 0

    # Build the inference plan once per task (before the cell loop).
    logger.info(f"Building inference plan for task={args.task}...")
    spec = _make_benchmark_spec_with_temperatures(temperatures_override)
    plan = create_inference_plan(model, args.task, spec)

    for seq_len in args.seq_lens:
        for batch_size in args.batch_sizes:
            for precision in args.precision:
                current_cell += 1
                logger.info(
                    f"Benchmarking cell {current_cell}/{total_cells}: "
                    f"seq_len={seq_len}, batch_size={batch_size}, "
                    f"precision={precision}, task={args.task}"
                )

                result = benchmark_cell(
                    model,
                    plan,
                    args.pdb_dir,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    precision=precision,
                    task=args.task,
                    ligand_enabled=args.ligand,
                    n_warmup=args.n_warmup,
                    n_timed=args.n_timed,
                )

                if result is not None:
                    result["hardware"] = args.hardware
                    all_results.append(result)
                else:
                    logger.warning(f"Skipped cell {current_cell}")

    if not all_results:
        logger.error("No benchmarks completed successfully")
        return 1

    output = {
        "schema_version": "1",
        "results": all_results,
        "total_cells": len(all_results),
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results written to {args.output_json}")
    else:
        print(json.dumps(output, indent=2))

    logger.info(f"Benchmark complete: {len(all_results)}/{total_cells} cells successful")
    return 0


if __name__ == "__main__":
    sys.exit(main())
