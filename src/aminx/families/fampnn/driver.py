"""FAMPNN family driver for ``sample``, ``score:nll``, and ``score:logits``.

``sample`` runs :func:`aminx.families.fampnn.sampler.mar_sample_upstream`, the port of upstream
``SeqDenoiser.sample`` (linear-K masked-autoregressive design with 50-step EDM sidechain packing,
psce re-masking, fixed positions and a final repack), across the 3 released checkpoints
(``fampnn_0_0``, ``fampnn_0_3``, ``fampnn_0_3_cath``). ``score:*`` masks the whole sequence and all
sidechains and reads the sequence head (upstream ``SeqDenoiser.score`` on a fully masked input).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from xtrax.tiling import AxisSpec

from aminx.families.fampnn.constants import (
  AMINX_TO_FAMPNN_ARR,
  FAMPNN_TO_AMINX_ARR,
  NON_BB_IDXS_ARR,
)
from aminx.families.fampnn.convert import convert_fampnn_state_dict
from aminx.families.fampnn.model import FAMPNNModel
from aminx.families.fampnn.pdb import read_fampnn_pdb
from aminx.families.fampnn.sampler import (
  draw_sampler_noise,
  mar_sample_upstream,
  override_masks,
  timestep_schedule,
)
from aminx.host.bucketing import bucket_ladder
from aminx.host.family_driver import FAMILY_DRIVERS, FamilyBatch, FamilyStages, SinkArraySpec
from aminx.host.prep import _resolve_driver_checkpoint
from aminx.io.torch_checkpoint import load_restricted_torch_checkpoint
from aminx.io.upstream_pdb import PottsInputError
from aminx.run.options import FAMPNNOptions

_HANDLED: frozenset[str] = frozenset({
  "sample",
  "score:nll",
  "score:logits",
})


class FAMPNNPrepared(NamedTuple):
  """Prepared single-structure inputs for FAMPNN scoring and sampling.

  Attributes
  ----------
  atom37_xyz : np.ndarray
    Atom37 coordinates ``(L, 37, 3)``: every recognised atom in the file (side chains included), zero elsewhere.
  atom37_mask : np.ndarray
    Atom37 presence mask ``(L, 37)`` (1 where the file had the atom).
  mask : np.ndarray
    Valid residue mask ``(L,)`` (upstream ``seq_mask``: all ones).
  residue_idx : np.ndarray
    PDB residue numbers plus insertion-code offsets ``(L,)``.
  chain_idx : np.ndarray
    Chain order of first appearance ``(L,)``.
  aatype_aminx : np.ndarray
    Input sequence in Aminx 21-token order ``(L,)``.
  aatype_fampnn : np.ndarray
    Input sequence in FAMPNN 21-token order ``(L,)``.
  missing_atom_mask : np.ndarray | None
    ``(L, 37)``: atoms the residue type has that the file lacks (upstream ``missing_atom_mask``).
  chain_id_mapping : dict[str, int] | None
    Chain letter to ``chain_idx`` (needed to parse ``fixed_pos_seq`` / ``fixed_pos_scn``).
  """

  atom37_xyz: np.ndarray
  atom37_mask: np.ndarray
  mask: np.ndarray
  residue_idx: np.ndarray
  chain_idx: np.ndarray
  aatype_aminx: np.ndarray
  aatype_fampnn: np.ndarray
  missing_atom_mask: np.ndarray | None = None
  chain_id_mapping: dict[str, int] | None = None


def _options(spec: Any) -> FAMPNNOptions:  # noqa: ANN401
  opt = getattr(spec, "fampnn", None)
  if opt is None:
    return FAMPNNOptions()
  return cast("FAMPNNOptions", opt)


_DEFAULT_CHECKPOINT_ID = "fampnn_0_3"

# Spec fields that constrain decoding but that the FAMPNN stages never read. Setting one would
# otherwise produce unconstrained output with no error, so ``load`` refuses the run instead.
_UNSUPPORTED_CONSTRAINT_FIELDS: tuple[str, ...] = (
  "fixed_positions",
  "fixed_tokens",
  "fixed_mask",
  "bias",
  "omit_aa",
  "omit_aa_per_position",
)


def _reject_unsupported_constraints(spec: Any) -> None:  # noqa: ANN401
  """Raise ``ValueError`` if ``spec`` sets a constraint field the FAMPNN driver would drop.

  Raises
  ------
  ValueError
    If any field in ``_UNSUPPORTED_CONSTRAINT_FIELDS`` is set to a non-empty value.
  """
  present = [name for name in _UNSUPPORTED_CONSTRAINT_FIELDS if _is_set(getattr(spec, name, None))]
  if present:
    msg = (
      f"fampnn does not apply {', '.join(present)}; the run would ignore them and return "
      "unconstrained sequences. Unset these fields for fampnn."
    )
    raise ValueError(msg)


def _is_set(value: Any) -> bool:  # noqa: ANN401
  return value is not None and not (isinstance(value, (tuple, list, Mapping)) and len(value) == 0)


def _checkpoint_id(spec: Any) -> str:  # noqa: ANN401
  """Effective FAMPNN checkpoint id: ``spec.checkpoint_id``, else ``spec.fampnn.checkpoint_id``.

  The two fields name the same thing, so a run that sets both must set them equal. A run that sets
  neither gets ``fampnn_0_3``.

  Raises
  ------
  ValueError
    If both are set and differ.
  """
  spec_id = getattr(spec, "checkpoint_id", None)
  options_id = _options(spec).checkpoint_id
  if spec_id is not None and options_id is not None and spec_id != options_id:
    msg = (
      f"conflicting FAMPNN checkpoint ids: spec.checkpoint_id={spec_id!r} but "
      f"spec.fampnn.checkpoint_id={options_id!r}; set one of them, or make them equal"
    )
    raise ValueError(msg)
  return str(spec_id or options_id or _DEFAULT_CHECKPOINT_ID)


def _inputs(spec: Any) -> list[Any]:  # noqa: ANN401
  raw = getattr(spec, "inputs", ())
  if isinstance(raw, (str, Path)):
    return [raw]
  return list(raw)


def _prepare_structure(item: Any) -> FAMPNNPrepared:  # noqa: ANN401
  if isinstance(item, FAMPNNPrepared):
    return item
  if not isinstance(item, (str, Path)):
    raise PottsInputError("fampnn_requires_pdb_or_prepared")
  try:
    st = read_fampnn_pdb(item)
  except ValueError as exc:
    raise PottsInputError(str(exc)) from exc
  num_l = st.aatype.shape[0]
  aatype_fampnn = st.aatype.astype(np.int32)
  return FAMPNNPrepared(
    atom37_xyz=st.atom37_xyz.astype(np.float32),
    atom37_mask=st.atom37_mask.astype(np.float32),
    mask=np.ones((num_l,), dtype=np.float32),
    residue_idx=st.residue_index.astype(np.int32),
    chain_idx=st.chain_index.astype(np.int32),
    aatype_aminx=np.asarray(FAMPNN_TO_AMINX_ARR)[aatype_fampnn].astype(np.int32),
    aatype_fampnn=aatype_fampnn,
    missing_atom_mask=st.missing_atom_mask.astype(np.float32),
    chain_id_mapping=dict(st.chain_id_mapping),
  )


class _FAMPNNStages:
  """Execution stage callable for FAMPNN scoring and MAR sampling."""

  def __init__(self, model: FAMPNNModel, spec: Any, purpose: str) -> None:  # noqa: ANN401
    """Initialize ``_FAMPNNStages``.

    Parameters
    ----------
    model : FAMPNNModel
      Loaded FAMPNN model.
    spec : Any
      Run or score specification.
    purpose : str
      Execution purpose string.

    Returns
    -------
    None
      Initializes instance state.
    """
    self._model = model
    self._spec = spec
    self._purpose = purpose
    self._options = _options(spec)

  def _run_score(
    self,
    prepared: tuple[FAMPNNPrepared, ...],
  ) -> Mapping[str, jax.Array]:
    logit_rows: list[jax.Array] = []
    nll_rows: list[jax.Array] = []
    for item in prepared:
      num_l = item.atom37_xyz.shape[0]
      seq_mask = jnp.zeros((num_l,), dtype=jnp.float32)
      out = self._model(
        jnp.asarray(item.atom37_xyz, dtype=jnp.float32),
        jnp.asarray(item.atom37_mask, dtype=jnp.float32),
        jnp.asarray(item.aatype_fampnn, dtype=jnp.int32),
        seq_mask,
        jnp.asarray(item.mask, dtype=jnp.float32),
        jnp.asarray(item.residue_idx, dtype=jnp.int32),
        jnp.asarray(item.chain_idx, dtype=jnp.int32),
      )
      log_probs = jax.nn.log_softmax(out.logits_aminx, axis=-1)
      wt_amx = jnp.asarray(item.aatype_aminx, dtype=jnp.int32)
      picked = jnp.take_along_axis(log_probs, wt_amx[:, None], axis=-1)[:, 0]
      logit_rows.append(out.logits_aminx)
      nll_rows.append(-picked)
    if self._purpose == "score:logits":
      return {"logits": jnp.stack(logit_rows, axis=0)}
    return {"nll": jnp.stack(nll_rows, axis=0)}

  def __call__(
    self,
    batch: FamilyBatch,
    *,
    chunk_start: int,
    chunk_count: int,
    scalar_dropout: bool,
    dropout_key: jax.Array,
  ) -> Mapping[str, jax.Array]:
    """Execute one FAMPNN batch stage.

    Parameters
    ----------
    batch : FamilyBatch
      Prepared structure batch.
    chunk_start : int
      Chunk start index.
    chunk_count : int
      Number of items in chunk.
    scalar_dropout : bool
      Whether scalar dropout is enabled.
    dropout_key : jax.Array
      PRNG key for sampling.

    Returns
    -------
    Mapping[str, jax.Array]
      Output arrays keyed by sink schema name.
    """
    del chunk_start, chunk_count, scalar_dropout
    prepared = cast("tuple[FAMPNNPrepared, ...]", batch.arrays["prepared"])
    if self._purpose in ("score:logits", "score:nll"):
      return self._run_score(prepared)

    seed = int(getattr(self._spec, "random_seed", 0) or 0)
    num_samples = int(getattr(self._spec, "num_samples", 1) or 1)
    temps = _temperatures(getattr(self._spec, "temperature", None))
    opt = self._options

    seq_batches: list[jax.Array] = []
    logit_batches: list[jax.Array] = []
    xyz_batches: list[jax.Array] = []
    psce_batches: list[jax.Array] = []

    for idx, item in enumerate(prepared):
      inputs = _sample_inputs(item, opt)
      t_seqs: list[jax.Array] = []
      t_logits: list[jax.Array] = []
      t_xyz: list[jax.Array] = []
      t_psce: list[jax.Array] = []
      for t_idx, temp in enumerate(temps):
        s_keys = jax.random.split(
          jax.random.fold_in(dropout_key, seed + idx * 100 + t_idx),
          num_samples,
        )
        outs = [_sample_one(self._model, inputs, opt, temp, s_key) for s_key in s_keys]
        t_seqs.append(jnp.stack([o[0] for o in outs], axis=0))
        t_logits.append(jnp.stack([o[1] for o in outs], axis=0))
        t_xyz.append(jnp.stack([o[2] for o in outs], axis=0))
        t_psce.append(jnp.stack([o[3] for o in outs], axis=0))
      seq_batches.append(jnp.stack(t_seqs, axis=1))
      logit_batches.append(jnp.stack(t_logits, axis=1))
      xyz_batches.append(jnp.stack(t_xyz, axis=1))
      psce_batches.append(jnp.stack(t_psce, axis=1))

    return {
      "sequences": jnp.stack(seq_batches, axis=0),
      "logits": jnp.stack(logit_batches, axis=0),
      "packed_atom37_xyz": jnp.stack(xyz_batches, axis=0),
      "psce": jnp.stack(psce_batches, axis=0),
    }


_DEFAULT_TEMPERATURE = 0.1  # seq_design.yaml:20


def _temperatures(raw: Any) -> list[float]:  # noqa: ANN401
  """Sampling temperatures of the spec; upstream's ``temperature: 0.1`` when the spec leaves it unset."""
  if isinstance(raw, (int, float)) and not isinstance(raw, bool):
    return [float(raw)]
  if isinstance(raw, (list, tuple)) and raw:
    return [_DEFAULT_TEMPERATURE if t is None else float(t) for t in raw]
  return [_DEFAULT_TEMPERATURE]


def _sample_inputs(item: FAMPNNPrepared, opt: FAMPNNOptions) -> dict[str, Any]:
  """Device arrays and override masks for one structure (``get_override_masks``, ``mode="seq_design"``)."""
  num_l = item.atom37_xyz.shape[0]
  missing = item.missing_atom_mask
  if missing is None:
    missing = np.zeros_like(item.atom37_mask)
  aov, sov = override_masks(
    opt.fixed_pos_seq,
    opt.fixed_pos_scn,
    item.chain_id_mapping or {},
    item.residue_idx,
    item.chain_idx,
  )
  if bool(np.any(np.asarray(sov) > np.asarray(aov))):
    msg = "Sidechain cannot be fixed at any positions where sequence is not fixed (fixed_pos_scn must lie inside fixed_pos_seq)"
    raise ValueError(msg)
  xyz = jnp.asarray(item.atom37_xyz, dtype=jnp.float32)
  return {
    "x": xyz,
    "aatype": jnp.asarray(item.aatype_fampnn, dtype=jnp.int32),
    "valid": jnp.asarray(item.mask, dtype=jnp.float32),
    "missing": jnp.asarray(missing, dtype=jnp.float32),
    "residue_idx": jnp.asarray(item.residue_idx, dtype=jnp.int32),
    "chain_idx": jnp.asarray(item.chain_idx, dtype=jnp.int32),
    "aov": jnp.asarray(aov, dtype=jnp.float32),
    "sov": jnp.asarray(sov, dtype=jnp.float32),
    "num_l": num_l,
  }


@eqx.filter_jit
def _sample_core(
  model: FAMPNNModel,
  arrays: dict[str, jax.Array],
  noise: Any,  # noqa: ANN401
  timesteps: jax.Array,
  timesteps_scn: jax.Array,
  cfg: tuple,
) -> Any:  # noqa: ANN401
  temperature, seq_only, repack_last, psce_threshold, step_scale, s_churn, s_noise, s_t_min, s_t_max = cfg
  return mar_sample_upstream(
    model,
    arrays["x"],
    arrays["aatype"],
    arrays["valid"],
    arrays["missing"],
    arrays["residue_idx"],
    arrays["chain_idx"],
    timesteps,
    noise,
    temperature=temperature,
    seq_only=seq_only,
    repack_last=repack_last,
    psce_threshold=psce_threshold,
    aatype_override_mask=arrays["aov"],
    scn_override_mask=arrays["sov"],
    timesteps_scn=timesteps_scn,
    step_scale=step_scale,
    s_churn=s_churn,
    s_noise=s_noise,
    s_t_min=s_t_min,
    s_t_max=s_t_max,
    conf_min_bin=model.conf_min_bin,
    conf_max_bin=model.conf_max_bin,
  )


def _sample_one(
  model: FAMPNNModel,
  inputs: dict[str, Any],
  opt: FAMPNNOptions,
  temperature: float,
  key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """One designed sample: ``(sequence, log seq_probs, atom37 xyz, psce)`` in Aminx token order / Atom37 layout."""
  num_l = int(inputs["num_l"])
  noise = draw_sampler_noise(
    key,
    num_l,
    opt.num_seq_steps,
    opt.num_scn_steps,
    repack_last=opt.repack_last,
    s_churn=opt.s_churn,
  )
  ts = timestep_schedule(opt.seq_timestep_mode, opt.num_seq_steps, opt.seq_t_start, opt.seq_t_end)
  ts_scn = timestep_schedule(opt.scn_timestep_mode, opt.num_scn_steps, opt.scn_t_start, opt.scn_t_end)
  cfg = (
    float(temperature),
    bool(opt.seq_only),
    bool(opt.repack_last),
    None if opt.psce_threshold is None else float(opt.psce_threshold),
    float(opt.step_scale),
    float(opt.s_churn),
    float(opt.s_noise),
    float(opt.s_t_min),
    float(opt.s_t_max),
  )
  arrays = {k: v for k, v in inputs.items() if k != "num_l"}
  res = _sample_core(model, arrays, noise, ts, ts_scn, cfg)
  seq_aminx = FAMPNN_TO_AMINX_ARR[jnp.clip(res.aatype_final, 0, 20)]
  # logits: log of the unscaled distribution each residue was drawn from (aux["seq_probs"], sd_model.py:252)
  probs = res.seq_probs[:, jnp.asarray(AMINX_TO_FAMPNN_ARR)]
  logits_aminx = jnp.log(jnp.maximum(probs, 1e-30))
  psce37 = jnp.zeros((num_l, 37), dtype=res.psce.dtype).at[:, NON_BB_IDXS_ARR].set(res.psce)
  return seq_aminx, logits_aminx, res.x_final, psce37


def _load_fampnn_eqx(path: Path, checkpoint_id: str) -> FAMPNNModel:
  is_cath = "cath" in checkpoint_id or "cath" in path.name
  has_conf = not is_cath
  meta: dict[str, Any] = {}
  base_path = Path(str(path).removesuffix(".zst"))
  manifest_path = base_path.with_suffix(".manifest.json")
  if manifest_path.is_file():
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    has_conf = bool(meta.get("has_confidence", has_conf))
  skeleton = FAMPNNModel(
    has_confidence=has_conf,
    num_encoder_layers=int(meta.get("num_encoder_layers", 3 if is_cath else 5)),
    num_decoder_layers=int(meta.get("num_decoder_layers", 3 if is_cath else 5)),
    use_sidechain_features=bool(meta.get("use_sidechain_features", True)),
    num_gvp_layers=int(meta.get("num_gvp_layers", 3 if is_cath else 5)),
    gvp_w_out_mode=str(meta.get("gvp_w_out_mode", "linear" if is_cath else "gvp")),
    gvp_rotate_out=bool(meta.get("gvp_rotate_out", is_cath)),
    has_w_out=bool(meta.get("has_w_out", True)),
    scn_depth=int(meta.get("scn_depth", 4)),
    has_f_embedder=bool(meta.get("has_f_embedder", True)),
    conf_num_bins=int(meta.get("conf_num_bins", 50)),
    conf_has_upstream_heads=bool(meta.get("conf_has_upstream_heads", True)),
    scn_sigma_data=float(meta.get("scn_sigma_data", 0.6399999856948853)),
    edm_s_min=float(meta.get("edm_s_min", 0.01)),
    edm_s_max=float(meta.get("edm_s_max", 80.0)),
    edm_rho=float(meta.get("edm_rho", 7.0)),
    conf_min_bin=float(meta.get("conf_min_bin", 0.0)),
    conf_max_bin=float(meta.get("conf_max_bin", 4.0)),
    key=jax.random.PRNGKey(0),
  )
  if path.suffix == ".zst":
    import io  # noqa: PLC0415

    import zstandard as zstd  # noqa: PLC0415

    raw_bytes = zstd.ZstdDecompressor().decompress(path.read_bytes())
    return cast("FAMPNNModel", eqx.tree_deserialise_leaves(io.BytesIO(raw_bytes), skeleton))
  return cast("FAMPNNModel", eqx.tree_deserialise_leaves(path, skeleton))


def _load_fampnn_torch(path: Path) -> FAMPNNModel:
  raw = load_restricted_torch_checkpoint(path)
  return convert_fampnn_state_dict(raw)


class FAMPNNDriver:
  """FamilyDriver implementation for the FAMPNN model family."""

  name = "fampnn"
  options_type = FAMPNNOptions
  mpnn_fallback_purposes: frozenset[str] = frozenset()

  def handles(self, spec: Any, purpose: str) -> bool:  # noqa: ANN401
    """Return ``True`` for ``sample``, ``score:nll``, and ``score:logits``.

    Parameters
    ----------
    spec : Any
      Run or score specification.
    purpose : str
      Execution purpose string.

    Returns
    -------
    bool
      Whether ``purpose`` is handled natively by ``FAMPNNDriver``.
    """
    del spec
    return purpose in _HANDLED

  def load(self, spec: Any) -> FAMPNNModel:  # noqa: ANN401
    """Load a converted ``FAMPNNModel`` checkpoint.

    Parameters
    ----------
    spec : Any
      Run or score specification.

    Returns
    -------
    FAMPNNModel
      Loaded ``FAMPNNModel``.
    """
    _reject_unsupported_constraints(spec)
    ckpt_id = _checkpoint_id(spec)
    local = getattr(spec, "model_local_path", None)
    if local is not None:
      path = Path(str(local))
      if path.suffix.lower() in (".pt", ".ckpt", ".pth"):
        return _load_fampnn_torch(path)
      return _load_fampnn_eqx(path, ckpt_id)
    artifact = _resolve_driver_checkpoint(spec, "fampnn", checkpoint_id=ckpt_id)
    if artifact is None:
      msg = (
        "FAMPNNDriver requires model_local_path, a checkpoint registry entry with sha256, "
        "or a checkpoint_id in the pinned family table (aminx.io.weights.family_checkpoints)"
      )
      raise ValueError(msg)
    return _load_fampnn_eqx(Path(artifact), ckpt_id)

  def mpnn_core(self, model: eqx.Module) -> Any:  # noqa: ANN401
    """FAMPNN uses a 5+5 GVP+MPNN full-atom architecture and has no stock Aminx core.

    Parameters
    ----------
    model : eqx.Module
      Loaded FAMPNN model.

    Returns
    -------
    Any
      Never returns; always raises ``ValueError``.

    Raises
    ------
    ValueError
      Always raised if a non-handled fallback purpose requests ``mpnn_core``.
    """
    del model
    msg = "fampnn does not support stock ProteinMPNN fallback purposes"
    raise ValueError(msg)

  def batches(self, spec: Any) -> Iterator[FamilyBatch]:  # noqa: ANN401
    """Return an iterator of prepared single-structure batches.

    Parameters
    ----------
    spec : Any
      Run or score specification.

    Returns
    -------
    Iterator[FamilyBatch]
      Iterator of prepared single-structure batches.
    """
    pending: list[tuple[int, str]] = []
    for index, item in enumerate(_inputs(spec)):
      try:
        prepared = _prepare_structure(item)
      except PottsInputError as exc:
        pending.append((index, str(exc)))
        continue
      yield FamilyBatch(
        input_indices=(index,),
        arrays={"prepared": (prepared,)},
        skipped=tuple(pending),
        lengths=(int(prepared.atom37_xyz.shape[0]),),
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
    """Plan tiling axes for FAMPNN execution.

    Parameters
    ----------
    spec : Any
      Run or score specification.
    purpose : str
      Execution purpose string.
    batch : FamilyBatch
      Current structure batch.

    Returns
    -------
    list[AxisSpec]
      Planned axis specifications.
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
    ]

  def stages(self, spec: Any, purpose: str, model: eqx.Module) -> FamilyStages:  # noqa: ANN401
    """Return the execution stage callable for ``purpose``.

    Parameters
    ----------
    spec : Any
      Run or score specification.
    purpose : str
      Execution purpose string.
    model : eqx.Module
      Loaded FAMPNN model.

    Returns
    -------
    FamilyStages
      Stage callable.

    Raises
    ------
    ValueError
      If ``purpose`` is ``score:nll`` or ``score:logits`` and ``sequences_to_score`` is set. Those
      purposes score the wild-type sequence only.
    """
    candidates = tuple(getattr(spec, "sequences_to_score", ()) or ())
    if purpose in ("score:nll", "score:logits") and candidates:
      msg = (
        f"fampnn {purpose} scores the wild-type sequence only and does not apply "
        "sequences_to_score; unset sequences_to_score for fampnn"
      )
      raise ValueError(msg)
    return _FAMPNNStages(cast("FAMPNNModel", model), spec, purpose)

  def result_schema(self, spec: Any, purpose: str) -> Mapping[str, SinkArraySpec]:  # noqa: ANN401
    """Return sink array schema for ``purpose``.

    Parameters
    ----------
    spec : Any
      Run or score specification.
    purpose : str
      Execution purpose string.

    Returns
    -------
    Mapping[str, SinkArraySpec]
      Output array specifications.
    """
    del spec
    if purpose == "score:logits":
      return {
        "logits": SinkArraySpec(dims=("L", "A"), dtype="float32", attrs={}),
      }
    if purpose == "score:nll":
      return {
        "nll": SinkArraySpec(dims=("L",), dtype="float32", attrs={}),
      }
    if purpose == "sample":
      return {
        "sequences": SinkArraySpec(dims=("N_samples", "D_temp", "L"), dtype="int32", attrs={}),
        "logits": SinkArraySpec(dims=("N_samples", "D_temp", "L", "A"), dtype="float32", attrs={}),
        "packed_atom37_xyz": SinkArraySpec(
          dims=("N_samples", "D_temp", "L", "Atom37", "xyz"),
          dtype="float32",
          attrs={},
        ),
        "psce": SinkArraySpec(dims=("N_samples", "D_temp", "L", "Atom37"), dtype="float32", attrs={}),
      }
    msg = f"fampnn does not support purpose {purpose!r}"
    raise ValueError(msg)


FAMILY_DRIVERS.register("fampnn")(FAMPNNDriver())
