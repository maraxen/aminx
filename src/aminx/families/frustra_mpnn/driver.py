"""Host FamilyDriver implementation for FrustraMPNN.

Supports:
- ``score:frustration`` (and ``score``): predicts (L, 21) local energetic frustration indices.
- Intermediate stage decomposition callables: ``features``, ``encoder``, ``decoder``, ``scores``.
- Sequence sampling via embedded ProteinMPNN trunk.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from xtrax.tiling import AxisSpec

from aminx.families.frustra_mpnn.convert import (
  load_frustra_mpnn_checkpoint,
)
from aminx.families.frustra_mpnn.model import (
  FRUSTRA_ALPHABET,
  FrustraMPNN,
  FrustraStageOutputs,
  cast_floating,
)
from aminx.families.frustra_mpnn.options import (
  DEFAULT_CHECKPOINT_ID,
  FrustraMPNNOptions,
)
from aminx.host.bucketing import bucket_ladder
from aminx.host.family_driver import (
  FAMILY_DRIVERS,
  FamilyBatch,
  FamilyStages,
  SinkArraySpec,
)

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


def _options_for(spec: Any) -> FrustraMPNNOptions:  # noqa: ANN401
  """Extract or construct FrustraMPNNOptions from spec."""
  opts = getattr(spec, "options", None)
  if isinstance(opts, FrustraMPNNOptions):
    return opts
  if isinstance(opts, Mapping):
    return FrustraMPNNOptions(**opts)
  subtract_mut = getattr(spec, "subtract_mut", False)
  dtype = getattr(spec, "dtype", "float32")
  return FrustraMPNNOptions(
    checkpoint_id=getattr(spec, "checkpoint_id", DEFAULT_CHECKPOINT_ID),
    subtract_mut=subtract_mut,
    dtype=dtype,
  )


class _FrustraScoreStages:
  """FamilyStages callable evaluating FrustraMPNN frustration index scoring."""

  def __init__(
    self,
    driver: FrustraMPNNDriver,
    model: FrustraMPNN,
    purpose: str = "score:frustration",
    opts: FrustraMPNNOptions | None = None,
  ) -> None:
    self._driver = driver
    self._model = model
    self._purpose = purpose
    self._opts = opts or FrustraMPNNOptions()

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
    target_dtype = jnp.float64 if self._opts.dtype == "float64" else jnp.float32
    prepared = cast("tuple[Mapping[str, Any], ...]", batch.arrays.get("prepared", ()))

    if not prepared:
      coords = jnp.asarray(batch.arrays["coords"], dtype=target_dtype)
      mask = jnp.asarray(batch.arrays["mask"], dtype=target_dtype)
      res_idx = jnp.asarray(batch.arrays["residue_index"], dtype=jnp.int32)
      chain_idx = jnp.asarray(batch.arrays["chain_index"], dtype=jnp.int32)
      seq = jnp.asarray(batch.arrays["sequence"], dtype=jnp.int32)
      chain_mask = (
        jnp.asarray(batch.arrays["chain_mask"], dtype=target_dtype)
        if "chain_mask" in batch.arrays
        else None
      )
      stages_out = self._model.forward_stages(
        coords,
        mask,
        res_idx,
        chain_idx,
        seq,
        chain_mask=chain_mask,
        inference=True,
      )
      return self._format_output(stages_out)

    rows: list[Mapping[str, jax.Array]] = []
    for item in prepared:
      c = jnp.asarray(item["coords"], dtype=target_dtype)
      m = jnp.asarray(item["mask"], dtype=target_dtype)
      r = jnp.asarray(item["residue_index"], dtype=jnp.int32)
      ci = jnp.asarray(item["chain_index"], dtype=jnp.int32)
      s = jnp.asarray(item["sequence"], dtype=jnp.int32)
      cm = jnp.asarray(item["chain_mask"], dtype=target_dtype) if "chain_mask" in item else None
      st = self._model.forward_stages(c, m, r, ci, s, chain_mask=cm, inference=True)
      rows.append(self._format_output(st))

    if len(rows) == 1:
      return rows[0]
    return {k: jnp.stack([r[k] for r in rows], axis=0) for k in rows[0]}

  def _format_output(self, stages: FrustraStageOutputs) -> dict[str, jax.Array]:
    scores = stages.frustration_scores
    if self._opts.positions is not None:
      if isinstance(self._opts.positions, str):
        pos_set = {int(p.strip()) for p in self._opts.positions.split(",") if p.strip()}
      else:
        pos_set = {int(p) for p in self._opts.positions}
      mask = jnp.array([i in pos_set for i in range(scores.shape[0])], dtype=bool)[:, None]
      scores = jnp.where(mask, scores, jnp.nan)

    if self._purpose in ("score:frustration", "score", "scores", "frustration"):
      return {"frustration_scores": scores}
    if self._purpose == "features":
      return {"features": stages.features}
    if self._purpose == "encoder":
      return {"encoder_out": stages.encoder_out}
    if self._purpose == "decoder":
      return {"decoder_hidden": stages.decoder_hidden}
    return {"frustration_scores": scores}


class FrustraMPNNDriver:
  """Host FamilyDriver for FrustraMPNN local energetic frustration scoring."""

  name: str = "frustrampnn"
  family: str = "frustrampnn"
  capabilities: tuple[str, ...] = (
    "score:frustration",
    "score",
    "scores",
    "features",
    "encoder",
    "decoder",
  )
  options_type = FrustraMPNNOptions
  mpnn_fallback_purposes: frozenset[str] = frozenset({"sample"})

  def handles(self, spec: Any, purpose: str) -> bool:  # noqa: ANN401
    """Return True if purpose is supported by FrustraMPNNDriver."""
    del spec
    return purpose in self.capabilities

  def prepare_structure(  # noqa: PLR0915
    self,
    pdb_path: str | Path | Mapping[str, Any],
    chains: str | Sequence[str] | None = None,
  ) -> dict[str, Any]:
    """Parse structure path or feature dictionary into standard arrays."""
    allowed_chains: set[str] | None = None
    if chains is not None:
      if isinstance(chains, str):
        allowed_chains = {c.strip() for c in chains.split(",") if c.strip()}
      else:
        allowed_chains = {str(c).strip() for c in chains if str(c).strip()}

    if isinstance(pdb_path, Mapping):
      coords_arr = np.asarray(pdb_path["coords"], dtype=np.float32)
      if coords_arr.ndim == 3 and coords_arr.shape[1] == 37:
        coords_arr = coords_arr[:, (0, 1, 2, 4), :]
      num_l = int(coords_arr.shape[0])
      return {
        "coords": coords_arr,
        "mask": np.asarray(
          pdb_path.get("mask", np.ones((num_l,), dtype=np.float32)),
          dtype=np.float32,
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
    if allowed_chains is None:
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
        from aminx.utils.aa_convert import training_labels  # noqa: PLC0415

        seq_arr = np.asarray(training_labels(prot.aatype), dtype=np.int32).reshape(-1)
        return {
          "coords": coords4,
          "mask": mask_arr,
          "residue_index": res_idx,
          "chain_index": chain_idx,
          "sequence": seq_arr,
          "structure_id": path.stem,
          "length": int(coords4.shape[0]),
        }
      except Exception:  # noqa: BLE001, S110
        pass

    if not path.is_file():
      msg = f"Structure path not found: {path}"
      raise FileNotFoundError(msg)

    residues: dict[tuple[str, int], dict[str, Any]] = {}
    chain_order: list[str] = []
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
        if allowed_chains is not None and chain_id not in allowed_chains:
          continue
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
      raise ValueError(msg)
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
      seq_arr[idx] = FRUSTRA_ALPHABET.index(aa1)
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
    """Collate prepared structure dictionaries into FamilyBatch."""
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

  def _resolve_paths(
    self,
    checkpoint_id: str | None,
    local_path: str | Path | None,
  ) -> tuple[Path, Path | None]:
    """Resolve filesystem paths for head checkpoint and trunk weights."""
    ckpt_path: Path | None = None
    if local_path is not None:
      p = Path(local_path)
      if p.is_file():
        ckpt_path = p

    roots: list[Path] = []
    raw = os.environ.get("AMINX_PARITY_REPOS", "")
    if raw.strip():
      for entry in raw.split(os.pathsep):
        stripped = entry.strip()
        if stripped:
          roots.append(Path(stripped).expanduser())
    default_root = Path.home() / "repos"
    if default_root not in roots:
      roots.append(default_root)

    if ckpt_path is None:
      name = checkpoint_id or DEFAULT_CHECKPOINT_ID
      candidates = [root / f"FrustraMPNN/model_weights/{name}.ckpt" for root in roots]
      candidates.append(Path(f"outputs/FrustraMPNN/model_weights/{name}.ckpt"))
      candidates.append(Path(f"tests/data/fixtures/{name}.ckpt"))
      for c in candidates:
        if c.is_file():
          ckpt_path = c
          break

    if ckpt_path is None:
      msg = f"Unable to locate FrustraMPNN checkpoint {checkpoint_id!r}"
      raise FileNotFoundError(msg)

    # Locate trunk v_48_020.pt
    trunk_candidates = [
      Path("outputs/FrustraMPNN/weights/vanilla_model_weights/v_48_020.pt"),
      Path("tests/data/fixtures/v_48_020.pt"),
    ]
    for root in roots:
      trunk_candidates.extend(
        [
          root / "FrustraMPNN/weights/vanilla_model_weights/v_48_020.pt",
          root / "ProteinMPNN/ca_model_weights/v_48_020.pt",
          root / "ThermoMPNN/vanilla_model_weights/v_48_020.pt",
        ],
      )

    trunk_path: Path | None = None
    for t in trunk_candidates:
      if t.is_file():
        trunk_path = t
        break

    return ckpt_path, trunk_path

  def load_model(
    self,
    checkpoint_id: str | None = DEFAULT_CHECKPOINT_ID,
    *,
    local_path: str | Path | None = None,
    trunk_path: str | Path | None = None,
    subtract_mut: bool = False,
    dtype: str = "float32",
    key: jax.Array | None = None,
  ) -> FrustraMPNN:
    """Load and instantiate FrustraMPNN Equinox PyTree."""
    del key
    ckpt_file, resolved_trunk = self._resolve_paths(checkpoint_id, local_path)
    use_trunk = Path(trunk_path) if trunk_path is not None else resolved_trunk
    return load_frustra_mpnn_checkpoint(
      ckpt_file,
      trunk_checkpoint_path=use_trunk,
      subtract_mut=subtract_mut,
      dtype=dtype,
    )

  def load(self, spec: Any) -> eqx.Module:  # noqa: ANN401
    """Load model for spec as an eqx.Module."""
    opts = _options_for(spec)
    local_path = getattr(spec, "model_local_path", None)
    return self.load_model(
      opts.checkpoint_id,
      local_path=local_path,
      subtract_mut=opts.subtract_mut,
      dtype=opts.dtype,
    )

  def mpnn_core(self, model: eqx.Module) -> Any:  # noqa: ANN401
    """Return embedded ProteinMPNN trunk."""
    return getattr(model, "mpnn", None)

  def batches(self, spec: Any) -> Iterator[FamilyBatch]:  # noqa: ANN401
    """Yield FamilyBatch instances from spec inputs."""
    opts = _options_for(spec)
    chains = getattr(opts, "chains", None)
    raw_inputs = getattr(spec, "inputs", ())
    inputs_list = [raw_inputs] if isinstance(raw_inputs, (str, Path, Mapping)) else list(raw_inputs)
    pending: list[tuple[int, str]] = []
    for index, item in enumerate(inputs_list):
      try:
        prepared = self.prepare_structure(item, chains=chains)
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
    """Return named xtrax axes for batch."""
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
    """Return stage callable for model and purpose."""
    if purpose not in self.capabilities:
      msg = f"Unsupported FrustraMPNN purpose: {purpose!r}; valid: {self.capabilities}"
      raise ValueError(msg)
    opts = _options_for(spec)
    target_dtype = jnp.float64 if opts.dtype == "float64" else jnp.float32
    typed_model = cast_floating(cast("FrustraMPNN", model), target_dtype)
    return _FrustraScoreStages(self, typed_model, purpose=purpose, opts=opts)

  def result_schema(self, spec: Any, purpose: str) -> Mapping[str, SinkArraySpec]:  # noqa: ANN401
    """Return sink array schema for purpose."""
    del spec
    if purpose in ("score:frustration", "score", "scores", "frustration"):
      return {
        "frustration_scores": SinkArraySpec(
          dims=("L_total", "A"),
          dtype="float32",
          attrs={"alphabet": FRUSTRA_ALPHABET},
        ),
      }
    if purpose == "features":
      return {
        "features": SinkArraySpec(dims=("L_total", "K", "E"), dtype="float32", attrs={}),
      }
    if purpose == "encoder":
      return {
        "encoder_out": SinkArraySpec(dims=("L_total", "D"), dtype="float32", attrs={}),
      }
    if purpose == "decoder":
      return {
        "decoder_hidden": SinkArraySpec(dims=("N_dec", "L_total", "D"), dtype="float32", attrs={}),
      }
    msg = f"Unsupported FrustraMPNN purpose: {purpose!r}"
    raise ValueError(msg)


# Register singleton instance in FAMILY_DRIVERS
FAMILY_DRIVERS.register("frustrampnn")(FrustraMPNNDriver())
