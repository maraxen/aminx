"""FAMPNN family driver for ``sample``, ``score:nll``, and ``score:logits``.

Implements Masked Autoregressive (MAR) cosine-schedule joint sequence + sidechain
generation (``mar_sample``) and teacher-forced / masked scoring across all 3
released FAMPNN checkpoints (``fampnn_0_0``, ``fampnn_0_3``, ``fampnn_0_3_cath``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from xtrax.tiling import AxisSpec

from aminx.families.fampnn.constants import (
  AMINX_ALPHABET,
  AMINX_TO_FAMPNN_PERM,
  FAMPNN_TO_AMINX_ARR,
  RESTYPE_ATOM37_MASK,
)
from aminx.families.fampnn.convert import convert_fampnn_state_dict
from aminx.families.fampnn.model import FAMPNNModel
from aminx.families.fampnn.sidechain_diffusion import edm_sample
from aminx.families.potts_mpnn.featurize import PottsInputError, parse_pdb_upstream
from aminx.host.bucketing import bucket_ladder
from aminx.host.family_driver import FAMILY_DRIVERS, FamilyBatch, FamilyStages, SinkArraySpec
from aminx.host.prep import _resolve_driver_checkpoint
from aminx.run.options import FAMPNNOptions

_HANDLED: frozenset[str] = frozenset({
  "sample",
  "score:nll",
  "score:logits",
})

_AMINX_TO_FAMPNN_NP = np.asarray(AMINX_TO_FAMPNN_PERM, dtype=np.int32)


class FAMPNNPrepared(NamedTuple):
  """Prepared single-structure inputs for FAMPNN scoring and sampling.

  Attributes
  ----------
  atom37_xyz : np.ndarray
    Atom37 coordinates of shape ``(L, 37, 3)`` (with backbone slots ``0, 1, 2, 4`` populated).
  atom37_mask : np.ndarray
    Atom37 presence mask of shape ``(L, 37)``.
  mask : np.ndarray
    Valid residue mask of shape ``(L,)``.
  residue_idx : np.ndarray
    Residue sequence numbers of shape ``(L,)``.
  chain_idx : np.ndarray
    Chain IDs of shape ``(L,)``.
  aatype_aminx : np.ndarray
    Wild-type sequence in Aminx 21-token order ``(L,)``.
  aatype_fampnn : np.ndarray
    Wild-type sequence in FAMPNN 21-token order ``(L,)``.
  """

  atom37_xyz: np.ndarray
  atom37_mask: np.ndarray
  mask: np.ndarray
  residue_idx: np.ndarray
  chain_idx: np.ndarray
  aatype_aminx: np.ndarray
  aatype_fampnn: np.ndarray


def mar_sample(
  model: FAMPNNModel,
  atom37_xyz: Float[Array, "L 37 3"],
  atom37_mask: Float[Array, "L 37"],
  aatype_init: Int[Array, " L"],
  seq_mask_init: Float[Array, " L"],
  mask: Float[Array, " L"],
  residue_idx: Int[Array, " L"],
  chain_idx: Int[Array, " L"],
  *,
  num_seq_steps: int = 8,
  temperature: float = 0.1,
  num_scn_steps: int = 18,
  sigma_min: float = 0.002,
  sigma_max: float = 80.0,
  rho: float = 7.0,
  s_churn: float = 0.0,
  step_scale: float = 1.5,
  pack_each_step: bool = True,
  key: jax.Array | None = None,
) -> tuple[Int[Array, " L"], Float[Array, "L 21"], Float[Array, "L 37 3"], Float[Array, "L 37"]]:
  """Run Masked Autoregressive (MAR) cosine-schedule joint sequence + sidechain sampling.

  Uses ``jax.lax.fori_loop`` over the ``num_seq_steps`` MAR unmasking iterations
  (allowlisted in ``tests/lint/l_drv_allowlist.toml`` with ``reason = "sequential_dependency"``).

  Parameters
  ----------
  model : FAMPNNModel
    Loaded ``FAMPNNModel`` instance.
  atom37_xyz : Float[Array, "L 37 3"]
    Initial Atom37 coordinates ``(L, 37, 3)``.
  atom37_mask : Float[Array, "L 37"]
    Initial Atom37 validity mask ``(L, 37)``.
  aatype_init : Int[Array, " L"]
    Initial residue types in FAMPNN token order ``0..20``.
  seq_mask_init : Float[Array, " L"]
    Initial sequence visibility mask ``(L,)`` (``1.0`` fixed, ``0.0`` to design).
  mask : Float[Array, " L"]
    Valid residue mask ``(L,)``.
  residue_idx : Int[Array, " L"]
    Residue sequence numbers ``(L,)``.
  chain_idx : Int[Array, " L"]
    Chain IDs ``(L,)``.
  num_seq_steps : int, default=8
    Number of MAR unmasking steps.
  temperature : float, default=0.1
    Categorical sequence sampling temperature.
  num_scn_steps : int, default=18
    Number of EDM sidechain diffusion steps per unmasking step.
  sigma_min : float, default=0.002
    Minimum EDM noise level.
  sigma_max : float, default=80.0
    Maximum EDM noise level.
  rho : float, default=7.0
    Karras schedule curvature exponent.
  s_churn : float, default=0.0
    EDM churn parameter.
  step_scale : float, default=1.5
    EDM drift step scale.
  pack_each_step : bool, default=True
    Whether to run EDM sidechain packing at every MAR step or only at the final step.
  key : jax.Array | None, optional
    PRNG key for random unmasking order, categorical sampling, and EDM diffusion.

  Returns
  -------
  tuple[Int[Array, " L"], Float[Array, "L 21"], Float[Array, "L 37 3"], Float[Array, "L 37"]]
    Tuple ``(seq_aminx, logits_aminx, packed_atom37_xyz, psce_atom37)``. ``logits_aminx`` are the
    logits of a forward pass on the returned sequence, so their argmax agrees with ``seq_aminx``.
  """
  if key is None:
    key = jax.random.PRNGKey(0)
  dtype = atom37_xyz.dtype
  num_l = atom37_xyz.shape[0]
  t_eff = max(temperature, 1e-4)

  k_order, k_loop, k_final_scn = jax.random.split(key, 3)
  # Rank designable positions by random priority so cosine schedule unmasks them deterministically
  designable = (mask > 0) & (seq_mask_init <= 0)
  num_designable = jnp.sum(designable.astype(jnp.int32))
  rand_scores = jax.random.uniform(k_order, (num_l,), dtype=dtype)
  # Fixed positions get priority -1.0; padded positions get +2.0; designable get [0, 1)
  priority = jnp.where(seq_mask_init > 0, -1.0, jnp.where(mask > 0, rand_scores, 2.0))
  ranks = jnp.argsort(jnp.argsort(priority))
  num_fixed = jnp.sum((seq_mask_init > 0).astype(jnp.int32))

  steps_eff = max(num_seq_steps, 1)
  init_logits = jnp.zeros((num_l, 21), dtype=dtype)
  init_state = (aatype_init, seq_mask_init.astype(dtype), atom37_xyz, atom37_mask, init_logits)

  def _mar_step(
    step_i: int,
    carry: tuple[
      Int[Array, " L"],
      Float[Array, " L"],
      Float[Array, "L 37 3"],
      Float[Array, "L 37"],
      Float[Array, "L 21"],
    ],
  ) -> tuple[
    Int[Array, " L"],
    Float[Array, " L"],
    Float[Array, "L 37 3"],
    Float[Array, "L 37"],
    Float[Array, "L 21"],
  ]:
    cur_aa, cur_seq_mask, cur_xyz, cur_atom_mask, _ = carry
    step_key = jax.random.fold_in(k_loop, step_i)
    k_cat, k_scn = jax.random.split(step_key, 2)

    fwd_out = model(
      cur_xyz, cur_atom_mask, cur_aa, cur_seq_mask, mask, residue_idx, chain_idx,
    )
    logits_20 = fwd_out.logits_fampnn[:, :20] / t_eff
    sampled_aa = jax.random.categorical(k_cat, logits_20, axis=-1).astype(jnp.int32)

    # Cosine unmasking schedule: fraction remaining masked after step_i+1 is cos(pi/2 * (step_i+1)/T)
    frac_unmasked = 1.0 - jnp.cos(0.5 * math.pi * (step_i + 1) / steps_eff)
    target_unmasked = num_fixed + jnp.ceil(frac_unmasked * num_designable).astype(jnp.int32)
    newly_unmasked = (ranks < target_unmasked) & (cur_seq_mask <= 0) & (mask > 0)

    next_aa = jnp.where(newly_unmasked, sampled_aa, cur_aa)
    next_seq_mask = jnp.where(newly_unmasked, jnp.ones_like(cur_seq_mask), cur_seq_mask)

    if pack_each_step:
      next_xyz = edm_sample(
        model.scn_diffusion,
        cur_xyz,
        cur_atom_mask,
        fwd_out.h_v,
        next_aa,
        num_steps=num_scn_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        s_churn=s_churn,
        step_scale=step_scale,
        key=k_scn,
      )
      std_m37 = RESTYPE_ATOM37_MASK[jnp.clip(next_aa, 0, 20)].astype(dtype)
      next_atom_mask = jnp.where(newly_unmasked[:, None], std_m37, cur_atom_mask)
    else:
      next_xyz = cur_xyz
      next_atom_mask = cur_atom_mask

    return (next_aa, next_seq_mask, next_xyz, next_atom_mask, fwd_out.logits_aminx)

  final_aa, final_seq_mask, final_xyz, final_atom_mask, final_logits_aminx = jax.lax.fori_loop(
    0, steps_eff, _mar_step, init_state,
  )

  if not pack_each_step:
    fwd_final = model(
      final_xyz, final_atom_mask, final_aa, final_seq_mask, mask, residue_idx, chain_idx,
    )
    final_xyz = edm_sample(
      model.scn_diffusion,
      final_xyz,
      final_atom_mask,
      fwd_final.h_v,
      final_aa,
      num_steps=num_scn_steps,
      sigma_min=sigma_min,
      sigma_max=sigma_max,
      rho=rho,
      s_churn=s_churn,
      step_scale=step_scale,
      key=k_final_scn,
    )
    final_atom_mask = RESTYPE_ATOM37_MASK[jnp.clip(final_aa, 0, 20)].astype(dtype)

  # The loop's last forward pass ran before the final step unmasked its positions, so its logits
  # describe a sequence that was never returned. One forward on the final state fixes that, and
  # the confidence head reads the same pass.
  fwd_last = model(
    final_xyz, final_atom_mask, final_aa, final_seq_mask, mask, residue_idx, chain_idx,
  )
  final_logits_aminx = fwd_last.logits_aminx
  if model.scn_confidence is not None:
    psce_37 = model.scn_confidence(
      final_xyz,
      final_atom_mask,
      fwd_last.h_v,
      final_aa,
      mask,
      residue_idx,
      chain_idx,
      h_esv=fwd_last.h_e,
      e_idx=fwd_last.e_idx,
    )
  else:
    psce_37 = jnp.full((num_l, 37), jnp.nan, dtype=dtype)

  seq_aminx = FAMPNN_TO_AMINX_ARR[jnp.clip(final_aa, 0, 20)]
  return seq_aminx, final_logits_aminx, final_xyz, psce_37


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


def _encode_aminx_seq(seq_str: str) -> np.ndarray:
  return np.asarray(
    [AMINX_ALPHABET.index(ch) if ch in AMINX_ALPHABET else 20 for ch in seq_str],
    dtype=np.int32,
  )


def _prepare_structure(item: Any) -> FAMPNNPrepared:  # noqa: ANN401
  if isinstance(item, FAMPNNPrepared):
    return item
  if not isinstance(item, (str, Path)):
    raise PottsInputError("fampnn_requires_pdb_or_prepared")
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
  chain_idx_list: list[int] = []

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
    chain_idx_list.extend([c_idx] * n_c)

  bb4_all = np.concatenate(coords_list, axis=0)
  num_l = bb4_all.shape[0]
  atom37_xyz = np.zeros((num_l, 37, 3), dtype=np.float32)
  # Atom37 slots: 0=N, 1=CA, 2=C, 4=O
  bb_clean = np.nan_to_num(bb4_all, nan=0.0)
  atom37_xyz[:, 0, :] = bb_clean[:, 0, :]
  atom37_xyz[:, 1, :] = bb_clean[:, 1, :]
  atom37_xyz[:, 2, :] = bb_clean[:, 2, :]
  atom37_xyz[:, 4, :] = bb_clean[:, 3, :]

  bb_valid = np.all(np.isfinite(bb4_all), axis=-1).astype(np.float32)
  atom37_mask = np.zeros((num_l, 37), dtype=np.float32)
  atom37_mask[:, 0] = bb_valid[:, 0]
  atom37_mask[:, 1] = bb_valid[:, 1]
  atom37_mask[:, 2] = bb_valid[:, 2]
  atom37_mask[:, 4] = bb_valid[:, 3]
  mask = np.all(bb_valid > 0, axis=-1).astype(np.float32)

  wt_seq = "".join(seq_chars)
  aatype_aminx = _encode_aminx_seq(wt_seq)
  aatype_fampnn = _AMINX_TO_FAMPNN_NP[aatype_aminx]

  return FAMPNNPrepared(
    atom37_xyz=atom37_xyz,
    atom37_mask=atom37_mask,
    mask=mask,
    residue_idx=np.asarray(res_idx_list, dtype=np.int32),
    chain_idx=np.asarray(chain_idx_list, dtype=np.int32),
    aatype_aminx=aatype_aminx,
    aatype_fampnn=aatype_fampnn,
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
    temps_raw = getattr(self._spec, "temperature", 0.1)
    temps = [float(temps_raw)] if isinstance(temps_raw, (int, float)) else [float(t) for t in temps_raw]

    seq_batches: list[jax.Array] = []
    logit_batches: list[jax.Array] = []
    xyz_batches: list[jax.Array] = []
    psce_batches: list[jax.Array] = []

    for idx, item in enumerate(prepared):
      num_l = item.atom37_xyz.shape[0]
      seq_mask_init = jnp.zeros((num_l,), dtype=jnp.float32)
      t_seqs: list[jax.Array] = []
      t_logits: list[jax.Array] = []
      t_xyz: list[jax.Array] = []
      t_psce: list[jax.Array] = []
      for t_idx, temp in enumerate(temps):
        s_keys = jax.random.split(
          jax.random.fold_in(dropout_key, seed + idx * 100 + t_idx),
          num_samples,
        )
        s_list: list[jax.Array] = []
        l_list: list[jax.Array] = []
        x_list: list[jax.Array] = []
        p_list: list[jax.Array] = []
        for s_key in s_keys:
          seq_amx, logits_amx, packed_xyz, psce = mar_sample(
            self._model,
            jnp.asarray(item.atom37_xyz, dtype=jnp.float32),
            jnp.asarray(item.atom37_mask, dtype=jnp.float32),
            jnp.asarray(item.aatype_fampnn, dtype=jnp.int32),
            seq_mask_init,
            jnp.asarray(item.mask, dtype=jnp.float32),
            jnp.asarray(item.residue_idx, dtype=jnp.int32),
            jnp.asarray(item.chain_idx, dtype=jnp.int32),
            num_seq_steps=self._options.num_seq_steps,
            temperature=temp,
            num_scn_steps=self._options.num_scn_steps,
            sigma_min=self._options.sigma_min,
            sigma_max=self._options.sigma_max,
            rho=self._options.rho,
            s_churn=self._options.s_churn,
            step_scale=self._options.step_scale,
            pack_each_step=self._options.pack_each_step,
            key=s_key,
          )
          s_list.append(seq_amx)
          l_list.append(logits_amx)
          x_list.append(packed_xyz)
          p_list.append(psce)
        t_seqs.append(jnp.stack(s_list, axis=0))
        t_logits.append(jnp.stack(l_list, axis=0))
        t_xyz.append(jnp.stack(x_list, axis=0))
        t_psce.append(jnp.stack(p_list, axis=0))
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
    key=jax.random.PRNGKey(0),
  )
  if path.suffix == ".zst":
    import io  # noqa: PLC0415

    import zstandard as zstd  # noqa: PLC0415

    raw_bytes = zstd.ZstdDecompressor().decompress(path.read_bytes())
    return cast("FAMPNNModel", eqx.tree_deserialise_leaves(io.BytesIO(raw_bytes), skeleton))
  return cast("FAMPNNModel", eqx.tree_deserialise_leaves(path, skeleton))


def _load_fampnn_torch(path: Path) -> FAMPNNModel:
  import torch  # noqa: PLC0415

  raw = torch.load(path, map_location="cpu", weights_only=False)
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
    """
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
