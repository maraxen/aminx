"""FamilyDriver implementation for ThermoMPNN (v1) and ThermoMPNN-D (v2).

Supports ``score:ddg`` across single-checkpoint v1 (``thermo_mpnn_v1_default``)
and 3-member v2 ensembles (``thermo_mpnn_v2_ensemble`` for single-mutant SSM and
``thermo_mpnn_d_ensemble`` for epistatic double-mutant ΔΔG).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from xtrax.tiling import AxisSpec

from aminx.families.thermo_mpnn.convert import (
  load_thermo_mpnn_checkpoint,
  load_thermo_mpnn_double_checkpoint,
  load_thermo_mpnn_double_pt,
  load_thermo_mpnn_pt,
)
from aminx.families.thermo_mpnn.model import (
  THERMO_ALPHABET,
  ThermoMPNN,
  ThermoMPNNDouble,
  cast_floating,
)
from aminx.host.bucketing import bucket_ladder
from aminx.host.family_driver import (
  FAMILY_DRIVERS,
  FamilyBatch,
  FamilyStages,
  SinkArraySpec,
)
from aminx.run.options import ThermoMPNNOptions

V2_SINGLE_ENSEMBLE_IDS: tuple[str, str, str] = (
  "thermo_mpnn_v2_ens1",
  "thermo_mpnn_v2_ens2",
  "thermo_mpnn_v2_ens3",
)

V2_DOUBLE_ENSEMBLE_IDS: tuple[str, str, str] = (
  "thermo_mpnn_d_ens1",
  "thermo_mpnn_d_ens2",
  "thermo_mpnn_d_ens3",
)

DEFAULT_CHECKPOINT_ID: str = "thermo_mpnn_v1_default"
_VALID_SCORE_MODES: frozenset[str] = frozenset(
  {"score:ddg", "score:ddg_double", "single", "double"}
)
_VALID_DTYPES: frozenset[str] = frozenset({"float32", "float16"})

log = logging.getLogger(__name__)

_AA3_TO_1: dict[str, str] = {
  "ALA": "A",
  "CYS": "C",
  "ASP": "D",
  "GLU": "E",
  "PHE": "F",
  "GLY": "G",
  "HIS": "H",
  "ILE": "I",
  "LYS": "K",
  "LEU": "L",
  "MET": "M",
  "ASN": "N",
  "PRO": "P",
  "GLN": "Q",
  "ARG": "R",
  "SER": "S",
  "THR": "T",
  "VAL": "V",
  "TRP": "W",
  "TYR": "Y",
}


def _options_for(spec: Any) -> ThermoMPNNOptions:  # noqa: ANN401
  """Return ``spec.thermompnn`` or the family defaults."""
  return getattr(spec, "thermompnn", None) or ThermoMPNNOptions()


def _resolve_checkpoint_id(spec: Any, opts: ThermoMPNNOptions) -> str:  # noqa: ANN401
  """Resolve the checkpoint id: explicit option, then ``spec.checkpoint_id``, then the default."""
  if opts.checkpoint_id:
    return str(opts.checkpoint_id)
  spec_id = getattr(spec, "checkpoint_id", None)
  if spec_id:
    return str(spec_id)
  return DEFAULT_CHECKPOINT_ID


def _is_double_checkpoint(checkpoint_id: str) -> bool:
  """Return True when ``checkpoint_id`` names a ThermoMPNN-D (double-mutant) checkpoint."""
  return checkpoint_id in V2_DOUBLE_ENSEMBLE_IDS or "_d_" in checkpoint_id or "double" in checkpoint_id


def _calc_dtype(name: str) -> Any:  # noqa: ANN401
  """Map an options dtype name to the JAX compute dtype."""
  return jnp.float16 if name == "float16" else jnp.float32


def _ensure_dtype(model: eqx.Module, dtype: Any) -> eqx.Module:  # noqa: ANN401
  """Return ``model`` with floating leaves in ``dtype``, skipping the cast when already there."""
  target = jnp.dtype(dtype)
  floats = [
    leaf
    for leaf in jax.tree.leaves(model)
    if eqx.is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.floating)
  ]
  if all(leaf.dtype == target for leaf in floats):
    return model
  return cast_floating(model, target)


class ThermoMPNNEnsemble(eqx.Module):
  """Equinox module wrapper around a tuple of ThermoMPNN or ThermoMPNNDouble ensemble members."""

  models: tuple[ThermoMPNN | ThermoMPNNDouble, ...]

  def __init__(self, models: Sequence[ThermoMPNN | ThermoMPNNDouble]) -> None:
    """Initialize ``ThermoMPNNEnsemble`` with ``models``."""
    self.models = tuple(models)


@eqx.filter_jit
def _score_single_ddg(
  model: ThermoMPNN,
  coords: Float[Array, "L 4 3"],
  mask: Float[Array, " L"],
  residue_index: Int[Array, " L"],
  chain_index: Int[Array, " L"],
  sequence: Int[Array, " L"],
) -> Float[Array, "L 21"]:
  """Compute single-mutant ΔΔG matrix ``(L, 21)`` for a single ThermoMPNN model.

  Parameters
  ----------
  model : ThermoMPNN
    Single-mutant ThermoMPNN model instance.
  coords : Float[Array, "L 4 3"]
    Backbone ``(N, CA, C, O)`` coordinates.
  mask : Float[Array, " L"]
    Residue validity mask.
  residue_index : Int[Array, " L"]
    Residue index array.
  chain_index : Int[Array, " L"]
    Chain index array.
  sequence : Int[Array, " L"]
    Wild-type integer sequence in MPNN 21-letter alphabet.

  Returns
  -------
  Float[Array, "L 21"]
    Predicted ΔΔG matrix ``(L, 21)`` with wild-type self-mutations zeroed.
  """
  ddg_raw = model(coords, mask, residue_index, chain_index, sequence)
  wt_ddg = jnp.take_along_axis(ddg_raw, sequence[:, None], axis=-1)
  return (ddg_raw - wt_ddg) * mask[:, None]


@eqx.filter_jit
def _score_double_ddg(
  model: ThermoMPNNDouble,
  coords: Float[Array, "L 4 3"],
  mask: Float[Array, " L"],
  residue_index: Int[Array, " L"],
  chain_index: Int[Array, " L"],
  sequence: Int[Array, " L"],
  pair_indices: Int[Array, "P 2"],
  *,
  batch_size: int = 10000,
) -> tuple[Float[Array, "L 21"], Float[Array, "P 21 21"], Int[Array, "P 2"]]:
  """Compute single and double-mutant ΔΔG tensors for a ThermoMPNNDouble model.

  Parameters
  ----------
  model : ThermoMPNNDouble
    Double-mutant ThermoMPNN-D model instance.
  coords : Float[Array, "L 4 3"]
    Backbone ``(N, CA, C, O)`` coordinates.
  mask : Float[Array, " L"]
    Residue validity mask.
  residue_index : Int[Array, " L"]
    Residue index array.
  chain_index : Int[Array, " L"]
    Chain index array.
  sequence : Int[Array, " L"]
    Wild-type integer sequence in MPNN 21-letter alphabet.
  pair_indices : Int[Array, "P 2"]
    Residue index pairs ``(P, 2)`` to evaluate.
  batch_size : int, default=10000
    Pair batch chunk size for double-mutant evaluation.

  Returns
  -------
  tuple[Float[Array, "L 21"], Float[Array, "P 21 21"], Int[Array, "P 2"]]
    Tuple ``(ddg_single, ddg_double, pair_indices)``.
  """
  num_l = int(coords.shape[0])
  num_p = int(pair_indices.shape[0])
  aa_range = jnp.arange(21, dtype=jnp.int32)
  pos_i = jnp.repeat(jnp.arange(num_l, dtype=jnp.int32), 21)
  wt_i = jnp.repeat(sequence.astype(jnp.int32), 21)
  mut_i = jnp.tile(aa_range, num_l)
  single_pos = jnp.stack([pos_i, pos_i], axis=-1)
  single_aas = jnp.stack([mut_i, wt_i], axis=-1)
  ddg_single = model(
    coords,
    mask,
    residue_index,
    chain_index,
    sequence,
    single_pos,
    single_aas,
  ).reshape(num_l, 21)
  wt_single = jnp.take_along_axis(ddg_single, sequence[:, None].astype(jnp.int32), axis=-1)
  ddg_single = (ddg_single - wt_single) * mask[:, None]

  if num_p == 0:
    return ddg_single, jnp.zeros((0, 21, 21), dtype=ddg_single.dtype), pair_indices

  if batch_size <= 0 or num_p <= batch_size:
    p_rep = jnp.repeat(pair_indices.astype(jnp.int32), 21 * 21, axis=0)
    aa_0 = jnp.tile(jnp.repeat(aa_range, 21), num_p)
    aa_1 = jnp.tile(aa_range, num_p * 21)
    double_aas = jnp.stack([aa_0, aa_1], axis=-1)
    ddg_double = model(
      coords,
      mask,
      residue_index,
      chain_index,
      sequence,
      p_rep,
      double_aas,
    ).reshape(num_p, 21, 21)
  else:
    ddg_double_chunks = []
    for start_idx in range(0, num_p, batch_size):
      chunk_p = pair_indices[start_idx : start_idx + batch_size]
      c_len = int(chunk_p.shape[0])
      p_rep = jnp.repeat(chunk_p.astype(jnp.int32), 21 * 21, axis=0)
      aa_0 = jnp.tile(jnp.repeat(aa_range, 21), c_len)
      aa_1 = jnp.tile(aa_range, c_len * 21)
      double_aas = jnp.stack([aa_0, aa_1], axis=-1)
      chunk_ddg = model(
        coords,
        mask,
        residue_index,
        chain_index,
        sequence,
        p_rep,
        double_aas,
      ).reshape(c_len, 21, 21)
      ddg_double_chunks.append(chunk_ddg)
    ddg_double = jnp.concatenate(ddg_double_chunks, axis=0)

  return ddg_single, ddg_double, pair_indices


class _ThermoScoreStages:
  """Per-batch ``score:ddg`` stage callable for :class:`ThermoMPNNDriver`."""

  def __init__(
    self,
    driver: ThermoMPNNDriver,
    model: eqx.Module,
    *,
    double_mutant: bool = False,
    opts: ThermoMPNNOptions | None = None,
  ) -> None:
    """Initialize the ThermoMPNN score stage wrapper.

    Parameters
    ----------
    driver : ThermoMPNNDriver
        Parent ThermoMPNN family driver.
    model : eqx.Module
        Loaded ThermoMPNN model or ensemble.
    double_mutant : bool, default=False
        Whether to compute double-mutant epistatic ddG predictions.
    opts : ThermoMPNNOptions | None, optional
        ThermoMPNN configuration options.

    Returns
    -------
    None
        Initializes stage state.
    """
    self._driver = driver
    self._model = model
    self._double_mutant = double_mutant
    self._opts = opts or ThermoMPNNOptions()

  def __call__(
    self,
    batch: FamilyBatch,
    *,
    chunk_start: int,
    chunk_count: int,
    scalar_dropout: bool,
    dropout_key: jax.Array,
  ) -> Mapping[str, jax.Array]:
    """Execute ``score:ddg`` across a collated :class:`FamilyBatch`.

    Parameters
    ----------
    batch : FamilyBatch
        Collated structure batch.
    chunk_start : int
        Starting index of the current chunk.
    chunk_count : int
        Number of items in the current chunk.
    scalar_dropout : bool
        Whether scalar dropout is enabled.
    dropout_key : jax.Array
        PRNG key for dropout.

    Returns
    -------
    Mapping[str, jax.Array]
        Dictionary of ``ddg`` (and optional ``ddg_double``) arrays.
    """
    del chunk_start, chunk_count, scalar_dropout, dropout_key
    target_dtype = jnp.float16 if self._opts.dtype == "float16" else jnp.float32
    prepared = cast("tuple[Mapping[str, Any], ...]", batch.arrays.get("prepared", ()))
    if not prepared:
      coords = jnp.asarray(batch.arrays["coords"], dtype=target_dtype)
      mask = jnp.asarray(batch.arrays["mask"], dtype=target_dtype)
      res_idx = jnp.asarray(batch.arrays["residue_index"], dtype=jnp.int32)
      chain_idx = jnp.asarray(batch.arrays["chain_index"], dtype=jnp.int32)
      seq = jnp.asarray(batch.arrays["sequence"], dtype=jnp.int32)
      return self._driver.score(
        self._model,
        coords,
        mask,
        res_idx,
        chain_idx,
        seq,
        mode=self._opts.mode,
        double_mutant=self._double_mutant,
        distance_threshold=self._opts.distance_threshold,
        batch_size=self._opts.batch_size,
        dtype=self._opts.dtype,
        ensemble=self._opts.ensemble,
      )

    ddg_rows: list[jax.Array] = []
    double_rows: list[jax.Array] = []
    pair_rows: list[jax.Array] = []
    for item in prepared:
      res = self._driver.score(
        self._model,
        jnp.asarray(item["coords"], dtype=target_dtype),
        jnp.asarray(item["mask"], dtype=target_dtype),
        jnp.asarray(item["residue_index"], dtype=jnp.int32),
        jnp.asarray(item["chain_index"], dtype=jnp.int32),
        jnp.asarray(item["sequence"], dtype=jnp.int32),
        mode=self._opts.mode,
        double_mutant=self._double_mutant,
        distance_threshold=self._opts.distance_threshold,
        batch_size=self._opts.batch_size,
        dtype=self._opts.dtype,
        ensemble=self._opts.ensemble,
      )
      ddg_rows.append(res["ddg"])
      if "ddg_double" in res:
        double_rows.append(res["ddg_double"])
        pair_rows.append(res["pair_indices"])
    out: dict[str, jax.Array] = {
      "ddg": ddg_rows[0] if len(ddg_rows) == 1 else jnp.stack(ddg_rows, axis=0),
    }
    if double_rows:
      out["ddg_double"] = (
        double_rows[0] if len(double_rows) == 1 else jnp.stack(double_rows, axis=0)
      )
      out["pair_indices"] = pair_rows[0] if len(pair_rows) == 1 else jnp.stack(pair_rows, axis=0)
    return out


class ThermoMPNNDriver:
  """Host ``FamilyDriver`` for ThermoMPNN (v1) and ThermoMPNN-D (v2).

  Attributes
  ----------
  family : str
    Registered family identifier (``"thermompnn"``).
  capabilities : tuple[str, ...]
    Supported capabilities (``("score:ddg",)``).
  """

  name: str = "thermompnn"
  family: str = "thermompnn"
  capabilities: tuple[str, ...] = ("score:ddg",)
  options_type = ThermoMPNNOptions
  mpnn_fallback_purposes: frozenset[str] = frozenset()

  def handles(self, spec: Any, purpose: str) -> bool:  # noqa: ANN401
    """Return True if purpose is supported by ThermoMPNNDriver."""
    del spec
    return purpose in self.capabilities

  def prepare_structure(  # noqa: PLR0915
    self,
    pdb_path: str | Path | Mapping[str, Any],
  ) -> dict[str, Any]:
    """Parse a structure path or feature mapping into ThermoMPNN backbone arrays.

    Parameters
    ----------
    pdb_path : str | Path | Mapping[str, Any]
      Path to a PDB/mmCIF file or pre-parsed dictionary of structure arrays.

    Returns
    -------
    dict[str, Any]
      Dictionary with ``coords`` ``(L, 4, 3)``, ``mask`` ``(L,)``,
      ``residue_index`` ``(L,)``, ``chain_index`` ``(L,)``, ``sequence`` ``(L,)``,
      ``structure_id``, and ``length``.
    """
    if isinstance(pdb_path, Mapping):
      coords_arr = np.asarray(pdb_path["coords"], dtype=np.float32)
      if coords_arr.ndim == 3 and coords_arr.shape[1] == 37:
        coords_arr = coords_arr[:, (0, 1, 2, 4), :]
      num_l = int(coords_arr.shape[0])
      return {
        "coords": coords_arr,
        "mask": np.asarray(
          pdb_path.get("mask", np.ones((num_l,), dtype=np.float32)), dtype=np.float32
        ),
        "residue_index": np.asarray(
          pdb_path.get("residue_index", np.arange(num_l, dtype=np.int32)),
          dtype=np.int32,
        ),
        "chain_index": np.asarray(
          pdb_path.get("chain_index", np.zeros((num_l,), dtype=np.int32)),
          dtype=np.int32,
        ),
        "sequence": np.clip(
          np.asarray(pdb_path.get("sequence", np.zeros((num_l,), dtype=np.int32)), dtype=np.int32),
          0,
          20,
        ),
        "structure_id": str(pdb_path.get("structure_id", "structure_0")),
        "length": num_l,
      }

    path = Path(pdb_path)
    try:
      from aminx.io.parsing import parse_structure  # noqa: PLC0415

      prot = parse_structure(path)
      coords37 = np.asarray(prot.coordinates, dtype=np.float32)
      if coords37.ndim == 4:
        coords37 = coords37[0]
      coords4 = coords37[:, (0, 1, 2, 4), :] if coords37.shape[1] >= 5 else coords37[:, :4, :]
      mask_arr = np.asarray(prot.mask, dtype=np.float32).reshape(-1)
      res_idx = np.asarray(prot.residue_index, dtype=np.int32).reshape(-1)
      chain_idx = np.asarray(prot.chain_index, dtype=np.int32).reshape(-1)
      seq_arr = np.clip(np.asarray(prot.aatype, dtype=np.int32).reshape(-1), 0, 20)
      return {
        "coords": coords4,
        "mask": mask_arr,
        "residue_index": res_idx,
        "chain_index": chain_idx,
        "sequence": seq_arr,
        "structure_id": path.stem,
        "length": int(coords4.shape[0]),
      }
    except Exception:  # noqa: BLE001
      residues: dict[tuple[str, int], dict[str, Any]] = {}
      chain_order: list[str] = []
      if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
          for line in handle:
            if not line.startswith("ATOM  "):
              continue
            atom_name = line[12:16].strip()
            if atom_name not in ("N", "CA", "C", "O"):
              continue
            alt_loc = line[16].strip()
            if alt_loc not in ("", "A"):
              continue
            res_name = line[17:20].strip()
            chain_id = line[21].strip() or "A"
            res_seq = int(line[22:26].strip())
            xyz = (
              float(line[30:38].strip()),
              float(line[38:46].strip()),
              float(line[46:54].strip()),
            )
            if chain_id not in chain_order:
              chain_order.append(chain_id)
            key = (chain_id, res_seq)
            if key not in residues:
              residues[key] = {"res_name": res_name, "atoms": {}}
            residues[key]["atoms"][atom_name] = xyz
      keys = list(residues.keys())
      num_res = len(keys)
      if num_res == 0:
        msg = f"No valid protein backbone residues found in {path}"
        raise ValueError(msg)  # noqa: B904
      coords4 = np.zeros((num_res, 4, 3), dtype=np.float32)
      mask_arr = np.zeros((num_res,), dtype=np.float32)
      res_idx = np.zeros((num_res,), dtype=np.int32)
      chain_idx = np.zeros((num_res,), dtype=np.int32)
      seq_arr = np.zeros((num_res,), dtype=np.int32)
      for idx, (chain_id, res_seq) in enumerate(keys):
        entry = residues[(chain_id, res_seq)]
        atoms = entry["atoms"]
        if all(a in atoms for a in ("N", "CA", "C", "O")):
          coords4[idx, 0] = atoms["N"]
          coords4[idx, 1] = atoms["CA"]
          coords4[idx, 2] = atoms["C"]
          coords4[idx, 3] = atoms["O"]
          mask_arr[idx] = 1.0
        res_idx[idx] = res_seq
        chain_idx[idx] = chain_order.index(chain_id)
        aa1 = _AA3_TO_1.get(entry["res_name"], "X")
        seq_arr[idx] = THERMO_ALPHABET.index(aa1)
      return {
        "coords": coords4,
        "mask": mask_arr,
        "residue_index": res_idx,
        "chain_index": chain_idx,
        "sequence": seq_arr,
        "structure_id": path.stem,
        "length": num_res,
      }

  def collate_batch(
    self,
    items: Sequence[Mapping[str, Any]],
    *,
    input_indices: tuple[int, ...] | None = None,
    skipped: tuple[tuple[int, str], ...] = (),
  ) -> FamilyBatch:
    """Collate prepared structure dictionaries into a :class:`FamilyBatch`.

    Parameters
    ----------
    items : Sequence[Mapping[str, Any]]
      Sequence of prepared structure dictionaries from :meth:`prepare_structure`.
    input_indices : tuple[int, ...] | None, optional
      Indices of input structures in the batch.
    skipped : tuple[tuple[int, str], ...], default=()
      Skipped ``(index, reason)`` entries.

    Returns
    -------
    FamilyBatch
      Host batch ready for :meth:`stages`.
    """
    prepared = tuple(dict(it) for it in items)
    indices = input_indices if input_indices is not None else tuple(range(len(prepared)))
    lengths = tuple(int(np.asarray(it["coords"]).shape[0]) for it in prepared)
    arrays: dict[str, Any] = {"prepared": prepared}
    if len(prepared) == 1:
      arrays.update(prepared[0])
    return FamilyBatch(
      input_indices=indices,
      arrays=arrays,
      skipped=skipped,
      lengths=lengths,
    )

  def load_model(
    self,
    checkpoint_id: str = "thermo_mpnn_v1_default",
    *,
    local_path: str | Path | None = None,
    key: jax.Array | None = None,
  ) -> ThermoMPNN | ThermoMPNNDouble | tuple[ThermoMPNN | ThermoMPNNDouble, ...]:
    """Load a single ThermoMPNN/ThermoMPNN-D model or 3-member v2 ensemble.

    Parameters
    ----------
    checkpoint_id : str, default="thermo_mpnn_v1_default"
      Checkpoint or ensemble identifier.
    local_path : str | Path | None, optional
      Optional local path override (``.eqx``, ``.eqx.zst``, ``.pt``, ``.ckpt``, ``.npz``).
    key : jax.Array | None, optional
      Optional PRNG key for skeleton construction.

    Returns
    -------
    ThermoMPNN | ThermoMPNNDouble | tuple[ThermoMPNN | ThermoMPNNDouble, ...]
      Loaded model or 3-model ensemble tuple.
    """
    if local_path is not None:
      path = Path(local_path)
      if path.suffix.lower() in {".pt", ".ckpt", ".npz"}:
        if _is_double_checkpoint(checkpoint_id):
          return load_thermo_mpnn_double_pt(path, key=key)
        return load_thermo_mpnn_pt(path, key=key)

    if checkpoint_id == "thermo_mpnn_v2_ensemble":
      return tuple(
        load_thermo_mpnn_checkpoint(cid, local_path=local_path, key=key)
        for cid in V2_SINGLE_ENSEMBLE_IDS
      )
    if checkpoint_id == "thermo_mpnn_d_ensemble":
      return tuple(
        load_thermo_mpnn_double_checkpoint(cid, local_path=local_path, key=key)
        for cid in V2_DOUBLE_ENSEMBLE_IDS
      )
    if _is_double_checkpoint(checkpoint_id):
      return load_thermo_mpnn_double_checkpoint(
        checkpoint_id,
        local_path=local_path,
        key=key,
      )
    return load_thermo_mpnn_checkpoint(
      checkpoint_id,
      local_path=local_path,
      key=key,
    )

  def load(self, spec: Any) -> eqx.Module:  # noqa: ANN401
    """Load the ThermoMPNN model or ensemble for ``spec`` as an ``eqx.Module``.

    Parameters
    ----------
    spec : Any
        Run specification providing ``checkpoint_id`` and optional ``model_local_path``.

    Returns
    -------
    eqx.Module
        Loaded :class:`ThermoMPNN`, :class:`ThermoMPNNDouble`, or :class:`ThermoMPNNEnsemble`.
    """
    opts = _options_for(spec)
    checkpoint_id = _resolve_checkpoint_id(spec, opts)
    local_path = getattr(spec, "model_local_path", None)
    loaded = self.load_model(checkpoint_id, local_path=local_path)
    target = _calc_dtype(opts.dtype)
    if isinstance(loaded, tuple):
      return ThermoMPNNEnsemble(tuple(_ensure_dtype(m, target) for m in loaded))
    return _ensure_dtype(loaded, target)

  def mpnn_core(self, model: eqx.Module) -> Any:  # noqa: ANN401
    """Return the embedded ProteinMPNN trunk, or None.

    Parameters
    ----------
    model : eqx.Module
        Loaded ThermoMPNN model or ensemble.

    Returns
    -------
    Any
        Embedded ``ProteinMPNN`` module or ``None``.
    """
    if isinstance(model, ThermoMPNNEnsemble) and model.models:
      return model.models[0].mpnn
    return getattr(model, "mpnn", None)

  def batches(self, spec: Any) -> Iterator[FamilyBatch]:  # noqa: ANN401
    """Yield one :class:`FamilyBatch` per input structure in ``spec.inputs``.

    Parameters
    ----------
    spec : Any
        Run specification containing ``inputs``.

    Returns
    -------
    Iterator[FamilyBatch]
        Iterator of collated single-structure batches.
    """
    raw_inputs = getattr(spec, "inputs", ())
    inputs_list = [raw_inputs] if isinstance(raw_inputs, (str, Path, Mapping)) else list(raw_inputs)
    pending: list[tuple[int, str]] = []
    for index, item in enumerate(inputs_list):
      try:
        prepared = self.prepare_structure(item)
      except Exception as exc:  # noqa: BLE001
        pending.append((index, str(exc)))
        continue
      yield self.collate_batch(
        (prepared,),
        input_indices=(index,),
        skipped=tuple(pending),
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
    """Return named xtrax axes for ``batch``.

    Parameters
    ----------
    spec : Any
        Run specification.
    purpose : str
        Execution purpose string.
    batch : FamilyBatch
        Collated structure batch.

    Returns
    -------
    list[AxisSpec]
        Named axis specifications for ``structures`` and ``mutants``.
    """
    del spec, purpose
    return [
      AxisSpec(
        name="structures",
        cardinality=max(len(batch.input_indices), 1),
        default_batch_size=1,
        heterogeneous=True,
        bucket_boundaries=bucket_ladder(),
      ),
      AxisSpec(
        name="mutants",
        cardinality=21,
        default_batch_size=21,
      ),
    ]

  def stages(self, spec: Any, purpose: str, model: eqx.Module) -> FamilyStages:  # noqa: ANN401
    """Return the ``score:ddg`` stage callable for ``model``.

    Parameters
    ----------
    spec : Any
        Run specification.
    purpose : str
        Execution purpose string (must be ``"score:ddg"``).
    model : eqx.Module
        Loaded ThermoMPNN model or ensemble.

    Returns
    -------
    FamilyStages
        Callable stage object computing per-batch ddG predictions.
    """
    if purpose != "score:ddg":
      msg = f"Unsupported ThermoMPNN purpose: {purpose!r}; expected 'score:ddg'."
      raise ValueError(msg)
    opts = _options_for(spec)
    ckpt_id = _resolve_checkpoint_id(spec, opts)
    double_mutant = (
      opts.mode == "double"
      or _is_double_checkpoint(ckpt_id)
      or isinstance(model, ThermoMPNNDouble)
      or (
        isinstance(model, ThermoMPNNEnsemble)
        and bool(model.models)
        and isinstance(model.models[0], ThermoMPNNDouble)
      )
    )
    # Cast the weights once here, not on every score() call.
    model = _ensure_dtype(model, _calc_dtype(opts.dtype))
    return _ThermoScoreStages(self, model, double_mutant=double_mutant, opts=opts)

  def result_schema(self, spec: Any, purpose: str) -> Mapping[str, SinkArraySpec]:  # noqa: ANN401
    """Return sink array schema for ``score:ddg``.

    Parameters
    ----------
    spec : Any
        Run specification.
    purpose : str
        Execution purpose string (must be ``"score:ddg"``).

    Returns
    -------
    Mapping[str, SinkArraySpec]
        Mapping of output array names to :class:`SinkArraySpec`.
    """
    if purpose != "score:ddg":
      msg = f"Unsupported ThermoMPNN purpose: {purpose!r}; expected 'score:ddg'."
      raise ValueError(msg)
    opts = _options_for(spec)
    ckpt_id = _resolve_checkpoint_id(spec, opts)
    schema: dict[str, SinkArraySpec] = {
      "ddg": SinkArraySpec(
        dims=("L_total", "A"),
        dtype="float32",
        attrs={"alphabet": THERMO_ALPHABET},
      ),
    }
    if opts.mode == "double" or _is_double_checkpoint(ckpt_id):
      # "epistatic": ThermoMPNN-D pair head. "additive": s_i + s_j from single-mutant
      # predictions of a non-D checkpoint scored with mode="double".
      method = "epistatic" if _is_double_checkpoint(ckpt_id) else "additive"
      schema["ddg_double"] = SinkArraySpec(
        dims=("P", "A", "A"),
        dtype="float32",
        attrs={"alphabet": THERMO_ALPHABET, "ddg_double_method": method},
      )
      schema["pair_indices"] = SinkArraySpec(
        dims=("P", "pair"),
        dtype="int32",
        attrs={},
      )
    return schema

  def write_outputs(
    self,
    output_dir: str | Path,
    results: Mapping[str, Any],
    *,
    structure_id: str = "structure_0",
  ) -> Path:
    """Persist ``score:ddg`` predictions to a Zarr store at ``output_dir``.

    Parameters
    ----------
    output_dir : str | Path
      Target output directory (Zarr store root).
    results : Mapping[str, Any]
      Output dictionary from :meth:`score` or :meth:`stages`.
    structure_id : str, default="structure_0"
      Structure identifier recorded in store attributes.

    Returns
    -------
    Path
      Resolved path to ``output_dir``.
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    try:
      import zarr  # noqa: PLC0415

      try:
        grp = zarr.open_group(str(out_path), mode="w", zarr_format=2)
      except TypeError:
        grp = zarr.open_group(str(out_path), mode="w")
      grp.attrs["family"] = self.family
      grp.attrs["structure_id"] = structure_id
      grp.attrs["alphabet"] = THERMO_ALPHABET
      for name, val in results.items():
        arr = np.asarray(val)
        if hasattr(grp, "create_dataset"):
          try:
            grp.create_dataset(name, data=arr, overwrite=True)
          except TypeError:
            grp[name] = arr
        else:
          grp[name] = arr
    except Exception:  # noqa: BLE001
      (out_path / ".zgroup").write_text(
        json.dumps({"zarr_format": 2}, indent=2) + "\n",
        encoding="utf-8",
      )
      (out_path / ".zattrs").write_text(
        json.dumps(
          {"family": self.family, "structure_id": structure_id, "alphabet": THERMO_ALPHABET},
          indent=2,
          sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
      )
      for name, val in results.items():
        np.save(out_path / f"{name}.npy", np.asarray(val))
    return out_path

  def sample(
    self,
    model: object,
    features: object,
    *,
    key: jax.Array | None = None,
    options: object | None = None,
  ) -> Mapping[str, Any]:
    """Refuse ``sample`` on ``ThermoMPNNDriver`` (ThermoMPNN is ``score:ddg`` only).

    Parameters
    ----------
    model : object
      Loaded ThermoMPNN model or ensemble.
    features : object
      Prepared structure features.
    key : jax.Array | None, optional
      Optional PRNG key.
    options : object | None, optional
      Optional sampling configuration.

    Returns
    -------
    Mapping[str, Any]
      Never returns; always raises ``NotImplementedError``.
    """
    del model, features, key, options
    msg = "Family 'thermompnn' does not support 'sample'; supported capabilities: ('score:ddg',)."
    raise NotImplementedError(msg)

  def score(  # noqa: PLR0915
    self,
    model_or_ensemble: (
      ThermoMPNN | ThermoMPNNDouble | ThermoMPNNEnsemble | Sequence[ThermoMPNN | ThermoMPNNDouble]
    ),
    coords: Float[Array, "L 4 3"],
    mask: Float[Array, " L"],
    residue_index: Int[Array, " L"],
    chain_index: Int[Array, " L"],
    sequence: Int[Array, " L"],
    *,
    mode: str = "score:ddg",
    double_mutant: bool = False,
    pair_indices: Int[Array, "P 2"] | None = None,
    distance_threshold: float = 5.0,
    batch_size: int = 10000,
    dtype: str = "float32",
    ensemble: bool = True,
    options: ThermoMPNNOptions | None = None,
  ) -> dict[str, jax.Array]:
    """Compute ΔΔG predictions for a single model or a 3-model ensemble.

    Parameters
    ----------
    model_or_ensemble : ThermoMPNN | ThermoMPNNDouble | ThermoMPNNEnsemble | Sequence[...]
      Single model, ``ThermoMPNNEnsemble``, or sequence of ensemble member models.
    coords : Float[Array, "L 4 3"]
      Backbone ``(N, CA, C, O)`` coordinates.
    mask : Float[Array, " L"]
      Residue validity mask.
    residue_index : Int[Array, " L"]
      Residue index array.
    chain_index : Int[Array, " L"]
      Chain index array.
    sequence : Int[Array, " L"]
      Wild-type integer sequence ``(L,)``.
    mode : str, default="score:ddg"
      Scoring capability mode (must be ``"score:ddg"``, ``"single"``, or ``"double"``).
    double_mutant : bool, default=False
      Whether to evaluate double-mutant ΔΔG. A ``ThermoMPNNDouble`` model always gets
      double-mutant output when ``options`` is None (as on ``origin/main``), even if this
      is False. When ``options`` is given, ``options.mode`` decides and this argument is
      ignored, so ``options.mode="single"`` on a Double model returns single-mutant output.
    pair_indices : Int[Array, "P 2"] | None, optional
      Optional explicit ``(P, 2)`` residue pairs for double-mutant scoring.
    distance_threshold : float, default=5.0
      C-alpha distance cutoff (Angstroms) for epistatic residue pairs in double mode.
    batch_size : int, default=10000
      Pair batch chunk size for ThermoMPNN-D double-mutant scoring.
    dtype : str, default="float32"
      Floating-point precision for trunk and head evaluation (``"float32"`` or ``"float16"``).
    ensemble : bool, default=True
      Whether to average across all ensemble members when using an ensemble.
    options : ThermoMPNNOptions | None, optional
      Optional ThermoMPNNOptions instance overriding individual knobs.

    Returns
    -------
    dict[str, jax.Array]
      ``"ddg"`` ``(L, 21)`` float32. When double-mutant scoring is on, also
      ``"ddg_double"`` ``(P, 21, 21)`` float32 and ``"pair_indices"`` ``(P, 2)`` int32.
      For a ``ThermoMPNNDouble`` model ``ddg_double`` is epistatic; for a single-mutant
      model it is the additive approximation ``s_i + s_j`` (see the schema attribute
      ``ddg_double_method``). ``P`` is 0 when no residue pair is within
      ``distance_threshold``; no pairs are substituted.
    """
    if mode not in _VALID_SCORE_MODES:
      msg = f"Unsupported ThermoMPNN score mode: {mode!r}; expected one of {sorted(_VALID_SCORE_MODES)}."
      raise ValueError(msg)
    opts = options
    if opts is not None:
      double_mutant = opts.mode == "double"
      distance_threshold = opts.distance_threshold
      batch_size = opts.batch_size
      dtype = opts.dtype
      ensemble = opts.ensemble
    if dtype not in _VALID_DTYPES:
      msg = f"Unsupported ThermoMPNN dtype: {dtype!r}; expected 'float32' or 'float16'."
      raise ValueError(msg)

    if mode in ("double", "score:ddg_double"):
      double_mutant = True

    if isinstance(model_or_ensemble, ThermoMPNNEnsemble):
      models: Sequence[Any] = model_or_ensemble.models
    elif isinstance(model_or_ensemble, Sequence):
      models = model_or_ensemble
    else:
      models = (model_or_ensemble,)
    if not models:
      msg = "ThermoMPNNDriver.score requires at least one model."
      raise ValueError(msg)

    if opts is None and isinstance(models[0], ThermoMPNNDouble):
      double_mutant = True

    if not ensemble and len(models) > 1:
      models = (models[0],)

    calc_dtype = _calc_dtype(dtype)
    coords = jnp.asarray(coords, dtype=calc_dtype)
    mask = jnp.asarray(mask, dtype=calc_dtype)
    models = tuple(_ensure_dtype(m, calc_dtype) for m in models)

    inv_n = 1.0 / float(len(models))
    if double_mutant:
      num_l = int(coords.shape[0])
      if pair_indices is None:
        ca = coords[:, 1, :]
        diff = ca[:, None, :] - ca[None, :, :]
        dist_matrix = jnp.sqrt(jnp.maximum(jnp.sum(diff**2, axis=-1), 1e-12))
        i_idx, j_idx = jnp.triu_indices(num_l, k=1)
        pair_dists = dist_matrix[i_idx, j_idx]
        valid_pair = (mask[i_idx] > 0) & (mask[j_idx] > 0) & (pair_dists <= distance_threshold)
        p_idx = jnp.stack([i_idx, j_idx], axis=-1)[valid_pair]
        if p_idx.shape[0] == 0:
          log.warning(
            "ThermoMPNN double-mutant scoring: no residue pair within distance_threshold=%.2f A; "
            "returning zero epistatic pairs.",
            distance_threshold,
          )
          p_idx = jnp.zeros((0, 2), dtype=jnp.int32)
      else:
        p_idx = jnp.asarray(pair_indices, dtype=jnp.int32)

      if isinstance(models[0], ThermoMPNNDouble):
        first_s, first_d, _ = _score_double_ddg(
          models[0],
          coords,
          mask,
          residue_index,
          chain_index,
          sequence,
          p_idx,
          batch_size=batch_size,
        )
        sum_single = first_s
        sum_double = first_d
        for m in models[1:]:
          s_ddg, d_ddg, _ = _score_double_ddg(
            m,
            coords,
            mask,
            residue_index,
            chain_index,
            sequence,
            p_idx,
            batch_size=batch_size,
          )
          sum_single = sum_single + s_ddg
          sum_double = sum_double + d_ddg
        ddg_single = sum_single * inv_n
        ddg_double = sum_double * inv_n
      else:
        sum_single = _score_single_ddg(
          models[0],
          coords,
          mask,
          residue_index,
          chain_index,
          sequence,
        )
        for m in models[1:]:
          sum_single = sum_single + _score_single_ddg(
            m,
            coords,
            mask,
            residue_index,
            chain_index,
            sequence,
          )
        s_avg = sum_single * inv_n
        ddg_single = s_avg
        # Additive approximation s_i + s_j (not epistatic); see ddg_double_method in the schema.
        ddg_double = s_avg[p_idx[:, 0], :, None] + s_avg[p_idx[:, 1], None, :]
      return {
        "ddg": ddg_single.astype(jnp.float32),
        "ddg_double": ddg_double.astype(jnp.float32),
        "pair_indices": p_idx.astype(jnp.int32),
      }

    if isinstance(models[0], ThermoMPNNDouble):
      first_s, _, _ = _score_double_ddg(
        models[0],
        coords,
        mask,
        residue_index,
        chain_index,
        sequence,
        jnp.zeros((0, 2), dtype=jnp.int32),
        batch_size=batch_size,
      )
      sum_ddg = first_s
      for m in models[1:]:
        s_ddg, _, _ = _score_double_ddg(
          m,
          coords,
          mask,
          residue_index,
          chain_index,
          sequence,
          jnp.zeros((0, 2), dtype=jnp.int32),
          batch_size=batch_size,
        )
        sum_ddg = sum_ddg + s_ddg
    else:
      sum_ddg = _score_single_ddg(
        models[0],
        coords,
        mask,
        residue_index,
        chain_index,
        sequence,
      )
      for m in models[1:]:
        sum_ddg = sum_ddg + _score_single_ddg(
          m,
          coords,
          mask,
          residue_index,
          chain_index,
          sequence,
        )
    return {"ddg": (sum_ddg * inv_n).astype(jnp.float32)}


def get_driver(*, family: str = "thermompnn") -> ThermoMPNNDriver:
  """Return a ``ThermoMPNNDriver`` instance for host dispatch.

  Parameters
  ----------
  family : str, default="thermompnn"
    Target family name.

  Returns
  -------
  ThermoMPNNDriver
    Instantiated driver for the ``"thermompnn"`` family.
  """
  del family
  return ThermoMPNNDriver()


FAMILY_DRIVERS.register("thermompnn")(ThermoMPNNDriver())
