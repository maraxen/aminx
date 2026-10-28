"""Caliby family driver for ``score:energy``, ``score:nll``, ``score:logits``, and ``sample``.

Supports all 8 upstream Caliby checkpoints:
- 5 Potts checkpoints (``caliby``, ``soluble_caliby``, ``soluble_caliby_v1``,
  ``caliby_distill``, ``soluble_caliby_distill``)
- 3 sidechain diffusion packer checkpoints (``caliby_packer_000``,
  ``caliby_packer_010``, ``caliby_packer_030``)
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from xtrax.tiling import AxisSpec

from aminx.families.caliby.convert import (
  CALIBY_PACKER_CHECKPOINTS,
  convert_caliby_packer_state_dict,
  convert_caliby_potts_state_dict,
)
from aminx.families.caliby.model import CalibyModel
from aminx.families.caliby.potts import (
  AMINX21_ALPHABET,
  AMINX21_TO_CALIBY21_IDXS,
  CALIBY21_TO_AMINX21_IDXS,
  aggregate_ensemble_potts_params,
  compute_caliby_potts_energy,
  generate_gaussian_ensemble,
  slice_potts_params,
)
from aminx.families.caliby.sidechain_diffusion import CalibySidechainMLP, edm_sidechain_sample
from aminx.families.potts_mpnn.featurize import PottsInputError, parse_pdb_upstream
from aminx.host.bucketing import bucket_ladder
from aminx.host.family_driver import FAMILY_DRIVERS, FamilyBatch, FamilyStages, SinkArraySpec
from aminx.host.prep import _resolve_driver_checkpoint
from aminx.run.options import CalibyOptions

_HANDLED: frozenset[str] = frozenset({
  "sample",
  "score:energy",
  "score:nll",
  "score:logits",
})

_AMINX_TO_CALIBY_NP = np.asarray(AMINX21_TO_CALIBY21_IDXS, dtype=np.int32)
_CALIBY_TO_AMINX_JAX = jnp.asarray(CALIBY21_TO_AMINX21_IDXS, dtype=jnp.int32)


class CalibyPrepared(NamedTuple):
  """Prepared single-structure inputs for Caliby scoring or sampling.

  Attributes
  ----------
  atom23_xyz : np.ndarray
    Atom23 coordinates of shape ``(L, 23, 3)``.
  atom23_mask : np.ndarray
    Atom23 validity mask of shape ``(L, 23)``.
  token_mask : np.ndarray
    Token validity mask of shape ``(L,)``.
  residue_idx : np.ndarray
    Residue sequence numbers of shape ``(L,)``.
  token_idx : np.ndarray
    Monotonic token indices of shape ``(L,)``.
  asym_id : np.ndarray
    Chain IDs of shape ``(L,)``.
  entity_id : np.ndarray
    Entity IDs of shape ``(L,)``.
  sym_id : np.ndarray
    Symmetry IDs of shape ``(L,)``.
  aatype_aminx : np.ndarray
    Wild-type sequence encoded in Aminx 21-token alphabet ``(L,)``.
  aatype_caliby : np.ndarray
    Wild-type sequence encoded in Caliby AF3 21-token alphabet ``(L,)``.
  candidate_sequences_aminx : np.ndarray
    Candidate sequences of shape ``(N_cand, L)`` in Aminx alphabet.
  """

  atom23_xyz: np.ndarray
  atom23_mask: np.ndarray
  token_mask: np.ndarray
  residue_idx: np.ndarray
  token_idx: np.ndarray
  asym_id: np.ndarray
  entity_id: np.ndarray
  sym_id: np.ndarray
  aatype_aminx: np.ndarray
  aatype_caliby: np.ndarray
  candidate_sequences_aminx: np.ndarray


def _options(spec: Any) -> CalibyOptions:  # noqa: ANN401
  opt = getattr(spec, "caliby", None)
  if opt is None:
    return CalibyOptions()
  return cast("CalibyOptions", opt)


def _is_packer_checkpoint(spec: Any) -> bool:  # noqa: ANN401
  """Return whether ``spec`` selects a Caliby sidechain packer checkpoint.

  Checkpoint id resolution matches ``CalibyDriver.load``: ``spec.checkpoint_id``
  when set, otherwise ``CalibyOptions.checkpoint_id``. A path whose name contains
  ``packer`` is also a packer. The default options id ``caliby`` stays Potts.
  """
  options = _options(spec)
  ckpt_id = str(getattr(spec, "checkpoint_id", None) or options.checkpoint_id or "caliby_distill")
  local = getattr(spec, "model_local_path", None)
  path_name = Path(str(local)).name if local is not None else ""
  return ckpt_id in CALIBY_PACKER_CHECKPOINTS or "packer" in path_name


def _inputs(spec: Any) -> list[Any]:  # noqa: ANN401
  raw = getattr(spec, "inputs", ())
  if isinstance(raw, (str, Path)):
    return [raw]
  return list(raw)


def _encode_aminx_seq(seq_str: str) -> np.ndarray:
  return np.asarray(
    [AMINX21_ALPHABET.index(ch) if ch in AMINX21_ALPHABET else 20 for ch in seq_str],
    dtype=np.int32,
  )


def _prepare_structure(item: Any, spec: Any) -> CalibyPrepared:  # noqa: ANN401
  if isinstance(item, CalibyPrepared):
    return item
  if not isinstance(item, (str, Path)):
    raise PottsInputError("caliby_requires_pdb_or_prepared")
  path = Path(item)
  parsed_list = parse_pdb_upstream(path, skip_gaps=False)
  parsed = parsed_list[0]

  chain_letters = sorted(
    key.removeprefix("seq_chain_")
    for key in parsed
    if key.startswith("seq_chain_")
  )
  coords_list: list[np.ndarray] = []
  seq_chars: list[str] = []
  res_idx_list: list[int] = []
  asym_list: list[int] = []

  for c_idx, letter in enumerate(chain_letters):
    seq_c = str(parsed[f"seq_chain_{letter}"])
    coords_c = parsed[f"coords_chain_{letter}"]
    n_c = len(seq_c)
    bb4 = np.stack(
      [
        np.asarray(coords_c[f"N_chain_{letter}"], dtype=np.float32),
        np.asarray(coords_c[f"CA_chain_{letter}"], dtype=np.float32),
        np.asarray(coords_c[f"C_chain_{letter}"], dtype=np.float32),
        np.asarray(coords_c[f"O_chain_{letter}"], dtype=np.float32),
      ],
      axis=1,
    )
    coords_list.append(bb4)
    seq_chars.extend(list(seq_c))
    res_idx_list.extend(range(1, n_c + 1))
    asym_list.extend([c_idx] * n_c)

  bb4_all = np.concatenate(coords_list, axis=0)
  num_l = bb4_all.shape[0]
  atom23_xyz = np.zeros((num_l, 23, 3), dtype=np.float32)
  atom23_xyz[:, :4, :] = np.nan_to_num(bb4_all, nan=0.0)
  atom23_mask = np.zeros((num_l, 23), dtype=np.float32)
  bb_valid = np.all(np.isfinite(bb4_all), axis=-1).astype(np.float32)
  atom23_mask[:, :4] = bb_valid
  token_mask = np.all(bb_valid > 0, axis=-1).astype(np.float32)

  wt_seq_str = "".join(seq_chars)
  aatype_aminx = _encode_aminx_seq(wt_seq_str)
  aatype_caliby = _AMINX_TO_CALIBY_NP[aatype_aminx]

  raw_cands = getattr(spec, "sequences_to_score", None)
  if raw_cands:
    cand_rows = [_encode_aminx_seq(str(s)) for s in raw_cands]
    candidates = np.stack(cand_rows, axis=0)
  else:
    candidates = aatype_aminx[None, :]

  return CalibyPrepared(
    atom23_xyz=atom23_xyz,
    atom23_mask=atom23_mask,
    token_mask=token_mask,
    residue_idx=np.asarray(res_idx_list, dtype=np.int32),
    token_idx=np.arange(num_l, dtype=np.int32),
    asym_id=np.asarray(asym_list, dtype=np.int32),
    entity_id=np.asarray(asym_list, dtype=np.int32),
    sym_id=np.zeros((num_l,), dtype=np.int32),
    aatype_aminx=aatype_aminx,
    aatype_caliby=aatype_caliby,
    candidate_sequences_aminx=candidates,
  )


def _potts_21_in_aminx_order(
  model: CalibyModel,
  prep: CalibyPrepared,
  options: CalibyOptions,
  seed: int,
) -> tuple[jax.Array, jax.Array]:
  """Compute 21-state Potts parameters ``(h_21, J_21)`` in Aminx canonical alphabet order."""
  xyz = jnp.asarray(prep.atom23_xyz, dtype=jnp.float32)
  mask23 = jnp.asarray(prep.atom23_mask, dtype=jnp.float32)
  tok_mask = jnp.asarray(prep.token_mask, dtype=jnp.float32)
  res_idx = jnp.asarray(prep.residue_idx, dtype=jnp.int32)
  tok_idx = jnp.asarray(prep.token_idx, dtype=jnp.int32)
  asym_id = jnp.asarray(prep.asym_id, dtype=jnp.int32)
  ent_id = jnp.asarray(prep.entity_id, dtype=jnp.int32)
  sym_id = jnp.asarray(prep.sym_id, dtype=jnp.int32)

  if options.ensemble_conformers > 1 and options.ensemble_noise_std > 0.0:
    ens_xyz = generate_gaussian_ensemble(
      xyz,
      mask23,
      num_conformers=options.ensemble_conformers,
      noise_std=options.ensemble_noise_std,
      key=jax.random.PRNGKey(seed),
    )
    h_list: list[jax.Array] = []
    j_list: list[jax.Array] = []
    for c_idx in range(ens_xyz.shape[0]):
      out_c = model(
        ens_xyz[c_idx],
        mask23,
        tok_mask,
        res_idx,
        tok_idx,
        asym_id,
        ent_id,
        sym_id,
      )
      h_list.append(out_c.h_32)
      j_list.append(out_c.j_32)
    h_32, j_32 = aggregate_ensemble_potts_params(jnp.stack(h_list, axis=0), jnp.stack(j_list, axis=0))
  else:
    out = model(xyz, mask23, tok_mask, res_idx, tok_idx, asym_id, ent_id, sym_id)
    h_32, j_32 = out.h_32, out.j_32

  return slice_potts_params(h_32, j_32, to_aminx_alphabet=True)


class _CalibyStages:
  """Execution stage callable for Caliby scoring and sampling."""

  def __init__(self, model: eqx.Module, spec: Any, purpose: str) -> None:  # noqa: ANN401
    """Initialize ``_CalibyStages``.

    Parameters
    ----------
    model : eqx.Module
      Loaded Caliby model.
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

  def _run_packer(
    self,
    prepared: tuple[CalibyPrepared, ...],
    dropout_key: jax.Array,
    seed: int,
  ) -> Mapping[str, jax.Array]:
    packer = cast("CalibySidechainMLP", self._model)
    packed_rows: list[jax.Array] = []
    seq_rows: list[jax.Array] = []
    for idx, item in enumerate(prepared):
      step_key = jax.random.fold_in(dropout_key, seed + idx)
      packed = edm_sidechain_sample(
        packer,
        jnp.asarray(item.atom23_xyz, dtype=jnp.float32),
        jnp.asarray(item.atom23_mask, dtype=jnp.float32),
        jnp.asarray(item.token_mask, dtype=jnp.float32),
        jnp.asarray(item.residue_idx, dtype=jnp.int32),
        jnp.asarray(item.token_idx, dtype=jnp.int32),
        jnp.asarray(item.asym_id, dtype=jnp.int32),
        jnp.asarray(item.entity_id, dtype=jnp.int32),
        jnp.asarray(item.sym_id, dtype=jnp.int32),
        jnp.asarray(item.aatype_caliby, dtype=jnp.int32),
        num_steps=self._options.packer_num_steps,
        sigma_min=self._options.packer_sigma_min,
        sigma_max=self._options.packer_sigma_max,
        rho=self._options.packer_rho,
        s_churn=self._options.packer_s_churn,
        step_scale=self._options.packer_step_scale,
        key=step_key,
      )
      packed_rows.append(packed.astype(jnp.float32))
      seq_rows.append(jnp.asarray(item.aatype_aminx, dtype=jnp.int32)[None, None, :])
    return {
      "sequences": jnp.stack(seq_rows, axis=0),
      "packed_atom23_xyz": jnp.stack(packed_rows, axis=0),
    }

  def _run_potts_sample(
    self,
    potts_model: CalibyModel,
    prepared: tuple[CalibyPrepared, ...],
    dropout_key: jax.Array,
    seed: int,
  ) -> Mapping[str, jax.Array]:
    num_samples = int(getattr(self._spec, "num_samples", 1) or 1)
    temps_raw = getattr(self._spec, "temperature", 0.01)
    temps = [float(temps_raw)] if isinstance(temps_raw, (int, float)) else [float(t) for t in temps_raw]
    seq_batches: list[jax.Array] = []
    energy_batches: list[jax.Array] = []
    for idx, item in enumerate(prepared):
      h_21, j_21 = _potts_21_in_aminx_order(potts_model, item, self._options, seed + idx)
      tok_mask = jnp.asarray(item.token_mask, dtype=jnp.float32)
      num_l = h_21.shape[0]
      j_off = j_21 * (1.0 - jnp.eye(num_l, dtype=jnp.float32))[:, :, None, None]
      gap_pen = jax.nn.one_hot(jnp.full((num_l,), 20, dtype=jnp.int32), 21, dtype=jnp.float32) * self._options.omit_gap_penalty
      h_design = h_21 + gap_pen

      temp_seqs: list[jax.Array] = []
      temp_energies: list[jax.Array] = []
      for t_idx, temp in enumerate(temps):
        t_eff = max(temp, 1e-4)
        sample_keys = jax.random.split(
          jax.random.fold_in(dropout_key, seed + idx * 100 + t_idx),
          num_samples,
        )
        s_list: list[jax.Array] = []
        e_list: list[jax.Array] = []
        for s_key in sample_keys:
          k_init, k_sweep = jax.random.split(s_key, 2)
          cur_seq = jax.random.categorical(k_init, -h_design / t_eff, axis=-1)
          cur_oh = jax.nn.one_hot(cur_seq, 21, dtype=jnp.float32) * tok_mask[:, None]
          local_field = h_design + jnp.einsum("ijab,jb->ia", j_off, cur_oh)
          sampled_seq = jax.random.categorical(k_sweep, -local_field / t_eff, axis=-1).astype(jnp.int32)
          e_val = compute_caliby_potts_energy(h_21, j_21, sampled_seq, tok_mask)
          s_list.append(sampled_seq)
          e_list.append(e_val)
        temp_seqs.append(jnp.stack(s_list, axis=0))
        temp_energies.append(jnp.stack(e_list, axis=0))
      seq_batches.append(jnp.stack(temp_seqs, axis=1))
      energy_batches.append(jnp.stack(temp_energies, axis=1))
    return {
      "sequences": jnp.stack(seq_batches, axis=0),
      "energy": jnp.stack(energy_batches, axis=0),
    }

  def __call__(
    self,
    batch: FamilyBatch,
    *,
    chunk_start: int,
    chunk_count: int,
    scalar_dropout: bool,
    dropout_key: jax.Array,
  ) -> Mapping[str, jax.Array]:
    """Execute one Caliby batch stage.

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
      PRNG key for sampling/dropout.

    Returns
    -------
    Mapping[str, jax.Array]
      Output arrays keyed by sink schema name.
    """
    del chunk_start, chunk_count, scalar_dropout
    prepared = cast("tuple[CalibyPrepared, ...]", batch.arrays["prepared"])
    seed = int(getattr(self._spec, "random_seed", 0) or 0)

    if isinstance(self._model, CalibySidechainMLP):
      if self._purpose != "sample":
        msg = f"caliby packer supports only purpose 'sample', got {self._purpose!r}"
        raise ValueError(msg)
      return self._run_packer(prepared, dropout_key, seed)

    potts_model = cast("CalibyModel", self._model)
    if self._purpose == "score:energy":
      energy_rows: list[jax.Array] = []
      cand_ids: list[jax.Array] = []
      for item in prepared:
        h_21, j_21 = _potts_21_in_aminx_order(potts_model, item, self._options, seed)
        cands = jnp.asarray(item.candidate_sequences_aminx, dtype=jnp.int32)
        tok_mask = jnp.asarray(item.token_mask, dtype=jnp.float32)
        energies = compute_caliby_potts_energy(h_21, j_21, cands, tok_mask)
        energy_rows.append(energies.astype(jnp.float32))
        cand_ids.append(jnp.arange(cands.shape[0], dtype=jnp.int32))
      return {
        "energy": jnp.stack(energy_rows, axis=0),
        "candidate_ids": jnp.stack(cand_ids, axis=0),
      }

    if self._purpose in ("score:logits", "score:nll"):
      logit_rows: list[jax.Array] = []
      nll_rows: list[jax.Array] = []
      for item in prepared:
        h_21, j_21 = _potts_21_in_aminx_order(potts_model, item, self._options, seed)
        wt_oh = jax.nn.one_hot(jnp.asarray(item.aatype_aminx, dtype=jnp.int32), 21, dtype=jnp.float32)
        num_l = h_21.shape[0]
        off_diag = (1.0 - jnp.eye(num_l, dtype=jnp.float32))[:, :, None, None]
        cond_field = h_21 + jnp.einsum("ijab,jb->ia", j_21 * off_diag, wt_oh)
        pseudo_logits = -cond_field
        log_probs = jax.nn.log_softmax(pseudo_logits, axis=-1)
        picked = jnp.take_along_axis(
          log_probs,
          jnp.asarray(item.aatype_aminx, dtype=jnp.int32)[:, None],
          axis=-1,
        )[:, 0]
        logit_rows.append(pseudo_logits)
        nll_rows.append(-picked)
      if self._purpose == "score:logits":
        return {"logits": jnp.stack(logit_rows, axis=0)}
      return {"nll": jnp.stack(nll_rows, axis=0)}

    return self._run_potts_sample(potts_model, prepared, dropout_key, seed)


def _load_caliby_eqx(path: Path, checkpoint_id: str) -> eqx.Module:
  base_path = Path(str(path).removesuffix(".zst"))
  manifest_path = base_path.with_suffix(".manifest.json")
  model_version = (
    1 if ("v1" in checkpoint_id or "distill" in checkpoint_id or "packer" in checkpoint_id) else 0
  )
  kind = "packer" if checkpoint_id in CALIBY_PACKER_CHECKPOINTS or "packer" in path.name else "potts"
  meta: dict[str, Any] = {}
  if manifest_path.is_file():
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    kind = str(meta.get("kind", kind))
    model_version = int(meta.get("model_version", model_version))
  if kind == "packer":
    skeleton: eqx.Module = CalibySidechainMLP(
      in_channels=int(meta.get("in_channels", 230)),
      out_channels=int(meta.get("out_channels", 99)),
      depth=int(meta.get("depth", 4)),
      num_encoder_layers=int(meta.get("num_encoder_layers", 10)),
      num_decoder_layers=int(meta.get("num_decoder_layers", 10)),
      model_version=model_version,
      update_decoder_edges=bool(meta.get("update_decoder_edges", True)),
      has_w_out=bool(meta.get("has_w_out", True)),
      sigma_data=float(meta.get("sigma_data", 1.0)),
      key=jax.random.PRNGKey(0),
    )
  else:
    skeleton = CalibyModel(
      num_encoder_layers=int(meta.get("num_encoder_layers", 3)),
      num_decoder_layers=int(meta.get("num_decoder_layers", 3)),
      model_version=model_version,
      parameterization=str(meta.get("parameterization", "factor")),
      use_sequence=bool(meta.get("use_sequence", True)),
      update_decoder_edges=bool(meta.get("update_decoder_edges", True)),
      has_w_out=bool(meta.get("has_w_out", False)),
      has_bg=bool(meta.get("has_bg", False)),
      bg_dim=int(meta.get("bg_dim", 1)),
      key=jax.random.PRNGKey(0),
    )
  if path.suffix == ".zst":
    import io  # noqa: PLC0415

    import zstandard as zstd  # noqa: PLC0415

    raw_bytes = zstd.ZstdDecompressor().decompress(path.read_bytes())
    return eqx.tree_deserialise_leaves(io.BytesIO(raw_bytes), skeleton)
  return eqx.tree_deserialise_leaves(path, skeleton)


class _DummyCfg:
  """Inert stand-in for omegaconf/hydra config objects inside Caliby checkpoints."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
    """Initialize placeholder configuration container.

    Parameters
    ----------
    args : Any
      Positional arguments ignored.
    kwargs : Any
      Keyword arguments ignored.

    Returns
    -------
    None
      Initializes dummy instance.
    """
    del args, kwargs

  def __setstate__(self, state: Any) -> None:  # noqa: ANN401
    """Restore pickled state dictionary into instance attributes.

    Parameters
    ----------
    state : Any
      Pickled state object.

    Returns
    -------
    None
      Updates ``self.__dict__`` when ``state`` is a ``dict``.
    """
    if isinstance(state, dict):
      self.__dict__.update(state)


_TORCH_STORAGE_TYPES: tuple[str, ...] = (
  "BoolStorage",
  "ByteStorage",
  "CharStorage",
  "ShortStorage",
  "IntStorage",
  "LongStorage",
  "HalfStorage",
  "BFloat16Storage",
  "FloatStorage",
  "DoubleStorage",
)

# Globals a Lightning/Caliby ``.ckpt`` pickle may reference. ``find_class`` receives the raw
# stream names, so the Python-2 spellings (``__builtin__``) are listed as written, alongside
# the Python-3 spellings. Data-rebuild helpers only.
_CALIBY_PICKLE_ALLOWED_GLOBALS: frozenset[tuple[str, str]] = frozenset(
  {
    ("__builtin__", "dict"),
    ("__builtin__", "list"),
    ("__builtin__", "long"),
    ("builtins", "dict"),
    ("builtins", "int"),
    ("builtins", "list"),
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("typing", "Any"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch._utils", "_rebuild_parameter_with_state"),
  }
  | {("torch", name) for name in _TORCH_STORAGE_TYPES},
)

# omegaconf/hydra config objects are replaced by the inert ``_DummyCfg`` below.
_CALIBY_PICKLE_STUB_PREFIXES: tuple[str, ...] = ("omegaconf", "hydra")


class _RestrictedCalibyUnpickler(pickle.Unpickler):
  """Unpickler that resolves only tensor-rebuild data globals and inert config stubs.

  Any other global (for example ``os.system``) is refused, so loading a third-party
  ``.ckpt`` cannot execute code through a pickle ``__reduce__`` payload.
  """

  def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401
    """Resolve an allow-listed global, substituting ``_DummyCfg`` for config stubs.

    Parameters
    ----------
    module : str
      Module path string from the pickle stream.
    name : str
      Attribute name from the pickle stream.

    Returns
    -------
    Any
      Resolved allow-listed object or ``_DummyCfg``.

    Raises
    ------
    pickle.UnpicklingError
      If ``module.name`` is neither allow-listed nor an omegaconf/hydra stub.
    """
    if module.startswith(_CALIBY_PICKLE_STUB_PREFIXES):
      return _DummyCfg
    if (module, name) in _CALIBY_PICKLE_ALLOWED_GLOBALS:
      return super().find_class(module, name)
    msg = f"Refusing to unpickle global {module}.{name} from a Caliby checkpoint"
    raise pickle.UnpicklingError(msg)


def _load_caliby_torch(path: Path) -> eqx.Module:
  import torch  # noqa: PLC0415

  pmod = type(
    "pickle_mod",
    (),
    {
      "Unpickler": _RestrictedCalibyUnpickler,
      "load": lambda f, **_kw: _RestrictedCalibyUnpickler(f).load(),
    },
  )
  raw = torch.load(path, map_location="cpu", weights_only=False, pickle_module=pmod)
  if "packer" in path.name:
    return convert_caliby_packer_state_dict(raw)
  return convert_caliby_potts_state_dict(raw)


class CalibyDriver:
  """FamilyDriver implementation for the Caliby model family."""

  name = "caliby"
  options_type = CalibyOptions
  mpnn_fallback_purposes: frozenset[str] = frozenset()

  def handles(self, spec: Any, purpose: str) -> bool:  # noqa: ANN401
    """Return whether ``purpose`` is handled for this checkpoint.

    A packer checkpoint handles only ``sample``. A Potts checkpoint handles
    ``sample``, ``score:energy``, ``score:nll``, and ``score:logits``.

    Parameters
    ----------
    spec : Any
      Run or score specification.
    purpose : str
      Execution purpose string.

    Returns
    -------
    bool
      Whether ``purpose`` is handled natively by ``CalibyDriver``.
    """
    if _is_packer_checkpoint(spec):
      return purpose == "sample"
    return purpose in _HANDLED

  def load(self, spec: Any) -> eqx.Module:  # noqa: ANN401
    """Load a converted ``CalibyModel`` or ``CalibySidechainMLP`` checkpoint.

    Parameters
    ----------
    spec : Any
      Run or score specification.

    Returns
    -------
    eqx.Module
      Loaded ``CalibyModel`` or ``CalibySidechainMLP``.
    """
    options = _options(spec)
    ckpt_id = str(getattr(spec, "checkpoint_id", None) or options.checkpoint_id or "caliby_distill")
    local = getattr(spec, "model_local_path", None)
    if local is not None:
      path = Path(str(local))
      if path.suffix.lower() in (".ckpt", ".pt", ".pth"):
        return _load_caliby_torch(path)
      return _load_caliby_eqx(path, ckpt_id)
    artifact = _resolve_driver_checkpoint(spec, "caliby")
    if artifact is None:
      msg = (
        "CalibyDriver requires model_local_path, a checkpoint registry entry with sha256, "
        "or a checkpoint_id in the pinned family table (aminx.io.weights.family_checkpoints)"
      )
      raise ValueError(msg)
    return _load_caliby_eqx(Path(artifact), ckpt_id)

  def mpnn_core(self, model: eqx.Module) -> Any:  # noqa: ANN401
    """Caliby uses its own 32-token AF3 vocabulary and does not expose a stock 21-token Aminx core.

    Parameters
    ----------
    model : eqx.Module
      Loaded Caliby model.

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
    msg = "caliby does not support stock ProteinMPNN fallback purposes"
    raise ValueError(msg)

  def batches(self, spec: Any) -> Iterator[FamilyBatch]:  # noqa: ANN401
    """Yield one ``FamilyBatch`` per input structure.

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
        prepared = _prepare_structure(item, spec)
      except PottsInputError as exc:
        pending.append((index, str(exc)))
        continue
      yield FamilyBatch(
        input_indices=(index,),
        arrays={"prepared": (prepared,)},
        skipped=tuple(pending),
        lengths=(int(prepared.atom23_xyz.shape[0]),),
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
    """Plan tiling axes for Caliby execution.

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
      Loaded Caliby model.

    Returns
    -------
    FamilyStages
      Stage callable.
    """
    return _CalibyStages(model, spec, purpose)

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
    if _is_packer_checkpoint(spec):
      if purpose != "sample":
        msg = f"caliby packer does not support purpose {purpose!r}"
        raise ValueError(msg)
      return {
        "sequences": SinkArraySpec(dims=("N_samples", "D_temp", "L"), dtype="int32", attrs={}),
        "packed_atom23_xyz": SinkArraySpec(dims=("L", "A", "3"), dtype="float32", attrs={}),
      }
    if purpose == "score:energy":
      return {
        "energy": SinkArraySpec(dims=("N_cand",), dtype="float32", attrs={}),
        "candidate_ids": SinkArraySpec(dims=("N_cand",), dtype="int32", attrs={}),
      }
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
        "energy": SinkArraySpec(dims=("N_samples", "D_temp"), dtype="float32", attrs={}),
      }
    msg = f"caliby does not support purpose {purpose!r}"
    raise ValueError(msg)


FAMILY_DRIVERS.register("caliby")(CalibyDriver())
