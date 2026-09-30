"""LASErMPNN family driver for scoring and proofreading.

Registration runs when this module is imported. Scoring reuses
``score_structure``; the alphabet permutation lives here, at the driver
boundary, because a silent off-by-one in that permutation still emits
plausible sequences.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from xtrax.tiling import AxisSpec

from aminx.families.laser_mpnn.featurize import (
  LASER_ALPHABET,
  LaserFeatures,
  LaserInputError,
  featurize,
)
from aminx.host.family_driver import ALPHABET, FAMILY_DRIVERS, FamilyBatch, SinkArraySpec
from aminx.host.prep import _resolve_local_checkpoint_from_registry
from aminx.model.laser.decoder import LaserDecoder, score_structure
from aminx.model.laser.encoders import LaserEncoder
from aminx.model.laser.graphs import GraphStructure
from aminx.model.laser.joint_decode import LaserJointDecode
from aminx.model.laser.proofread import (
  conditional_focus_probs,
  focus_rows,
  reduce_proofread,
  unconditional_logits,
)
from aminx.run.options import LaserOptions

_HANDLED = frozenset(
  {
    "score:nll",
    "score:logits",
    "score:proofread_unconditional",
    "score:proofread_conditional",
  },
)
_GRAPH_K = 48
_LIG_LIG_K = 5
_LIG_CUTOFF = 20.0
_ENCODER_PREFIXES = (
  "ligand_encoder.",
  "ligand_encoder_output_gvp.",
  "backbone_frame_vec_input_layer.",
  "protein_encoder_layers.",
  "prot_prot_rbf_encoding.",
  "lig_prot_rbf_encoding.",
  "prot_prot_edge_input_layer.",
  "lig_prot_edge_input_layer.",
)
_DECODER_PREFIXES = (
  "protein_decoder_layers.",
  "sequence_label_embedding.",
  "sequence_output_layer.",
  "chi_prediction_layers.",
  "chi_vector_update_layers.",
  "chi_vector_layer_norms.",
)


def _shared_alphabet() -> tuple[tuple[int, ...], tuple[int, ...]]:
  """Canonical-from-LASEr and LASEr-from-canonical index tables.

  The two alphabets use the same 21 letters in different orders (E/Q are the
  visible swap versus the AlphaFold order). Asserting a bijection at import
  is what stops a shifted table from scoring the wrong residue as if it were
  the right one.
  """
  if len(ALPHABET) != 21 or len(LASER_ALPHABET) != 21 or set(ALPHABET) != set(LASER_ALPHABET):
    msg = "LASEr and canonical alphabets do not share the same 21 letters"
    raise RuntimeError(msg)
  laser_of_canonical = tuple(LASER_ALPHABET.index(letter) for letter in ALPHABET)
  if sorted(laser_of_canonical) != list(range(21)):
    msg = "canonical-to-LASEr alphabet map is not a bijection"
    raise RuntimeError(msg)
  canonical_of_laser = [0] * 21
  for canonical, laser in enumerate(laser_of_canonical):
    canonical_of_laser[laser] = canonical
  back = tuple(canonical_of_laser[index] for index in laser_of_canonical)
  if back != tuple(range(21)):
    msg = "LASEr alphabet map does not round-trip"
    raise RuntimeError(msg)
  return laser_of_canonical, tuple(canonical_of_laser)


_LASER_OF_CANONICAL_INDEX, _CANONICAL_OF_LASER_INDEX = _shared_alphabet()
# Axis ``i`` of a canonical logit vector is ``LASER_ALPHABET`` index
# ``_LASER_OF_CANONICAL[i]``. The inverse maps a LASEr sequence index onto
# ``ALPHABET``.
LASER_OF_CANONICAL = jnp.asarray(_LASER_OF_CANONICAL_INDEX, dtype=jnp.int32)
CANONICAL_OF_LASER = jnp.asarray(_CANONICAL_OF_LASER_INDEX, dtype=jnp.int32)


def canonical_logits(laser_logits: Float[Array, "... 21"]) -> Float[Array, "... 21"]:
  """Reorder LASEr sequence logits onto ``ALPHABET``."""
  return jnp.take(laser_logits, LASER_OF_CANONICAL, axis=-1)


def sequence_nll(
  logits: Float[Array, "l 21"],
  canonical_index: Int[Array, " l"],
) -> Float[Array, " l"]:
  """Per-position ``-log_softmax`` gathered at the given sequence.

  This is the sign-flip of ``dump_laser_oracles._score_arrays`` ``seq_log_prob``
  once logits are already in the canonical alphabet.
  """
  log_prob = jax.nn.log_softmax(logits, axis=-1)
  picked = jnp.take_along_axis(log_prob, canonical_index[:, None], axis=-1)
  return -picked[:, 0]


def _x64_enabled() -> bool:
  """Runtime x64 flag. The ty stub for ``jax.config`` does not list it."""
  return bool(getattr(jax.config, "jax_enable_x64", False))


def decoding_order(
  chain_mask: np.ndarray,
  contact: np.ndarray,
  *,
  seed: int,
) -> np.ndarray:
  """Fixed residues, then designable non-contact, then designable contact.

  Tier 2 is empty at inference (``extra_atom_contact_mask`` is zeros). The
  uniform stream is one draw per row from ``random_seed``, so a parent and a
  fresh subprocess that share the seed and the masks share the permutation.
  """
  length = int(np.asarray(chain_mask).shape[0])
  dtype = jnp.float64 if _x64_enabled() else jnp.float32
  uniforms = jax.random.uniform(jax.random.PRNGKey(seed), (length,), dtype=dtype)
  fixed = jnp.asarray(chain_mask, dtype=jnp.bool_)
  touched = jnp.asarray(contact, dtype=jnp.bool_)
  tier = jnp.where(
    fixed,
    jnp.asarray(0, dtype=dtype),
    jnp.where(touched, jnp.asarray(2, dtype=dtype), jnp.asarray(1, dtype=dtype)),
  )
  return np.asarray(jnp.argsort(uniforms + tier), dtype=np.int32)


class LASErMPNN(eqx.Module):
  """Encoder, decoder, and the ligand period/group tables the encoder reads."""

  encoder: LaserEncoder
  decoder: LaserDecoder
  period_index: Int[Array, " 118"]
  group_index: Int[Array, " 118"]
  pr_pr_knn_graph_k: Int[Array, ""]
  lig_pr_knn_graph_k: Int[Array, ""]
  lig_lig_knn_graph_k: Int[Array, ""]
  lig_pr_distance_cutoff: Float[Array, ""]

  def __init__(
    self,
    *,
    key: jax.Array,
    period_index: np.ndarray | None = None,
    group_index: np.ndarray | None = None,
    pr_pr_knn_graph_k: int = _GRAPH_K,
    lig_pr_knn_graph_k: int = _GRAPH_K,
    lig_lig_knn_graph_k: int = _LIG_LIG_K,
    lig_pr_distance_cutoff: float = _LIG_CUTOFF,
  ) -> None:
    keys = jax.random.split(key, 2)
    self.encoder = LaserEncoder(key=keys[0])
    self.decoder = LaserDecoder(key=keys[1])
    self.period_index = (
      jnp.zeros((118,), dtype=jnp.int32)
      if period_index is None
      else jnp.asarray(period_index, dtype=jnp.int32)
    )
    self.group_index = (
      jnp.zeros((118,), dtype=jnp.int32)
      if group_index is None
      else jnp.asarray(group_index, dtype=jnp.int32)
    )
    self.pr_pr_knn_graph_k = jnp.asarray(pr_pr_knn_graph_k, dtype=jnp.int32)
    self.lig_pr_knn_graph_k = jnp.asarray(lig_pr_knn_graph_k, dtype=jnp.int32)
    self.lig_lig_knn_graph_k = jnp.asarray(lig_lig_knn_graph_k, dtype=jnp.int32)
    cutoff_dtype = jnp.float64 if _x64_enabled() else jnp.float32
    self.lig_pr_distance_cutoff = jnp.asarray(lig_pr_distance_cutoff, dtype=cutoff_dtype)


class _Prepared(NamedTuple):
  """One structure and the candidate sequences to teacher-force."""

  features: LaserFeatures
  laser_indices: np.ndarray
  canonical_indices: np.ndarray
  order: np.ndarray


def _graph(model: LASErMPNN) -> GraphStructure:
  return GraphStructure(
    pr_pr_knn_graph_k=int(model.pr_pr_knn_graph_k),
    lig_pr_knn_graph_k=int(model.lig_pr_knn_graph_k),
    lig_lig_knn_graph_k=int(model.lig_lig_knn_graph_k),
    lig_pr_distance_cutoff=float(model.lig_pr_distance_cutoff),
  )


def _options(spec: Any) -> LaserOptions:  # noqa: ANN401
  options = getattr(spec, "laser", None)
  if options is None:
    return LaserOptions()
  return cast("LaserOptions", options)


def budget_mask_for(
  resnums: np.ndarray,
  ss_code: tuple[str, ...] | list[str],
  exposed: np.ndarray,
  *,
  selection: str | None,
  constrain_to_exposed_non_ss: bool,
) -> np.ndarray:
  """Rows the ALA/GLY budget may floor.

  An empty selection leaves the mask empty, matching upstream's default of
  zeros. ``constrain_ala_gly_to_exposed_non_ss`` further requires an exposed
  loop (not helix or sheet).
  """
  mask = np.zeros((int(np.asarray(resnums).shape[0]),), dtype=bool)
  if selection:
    wanted = {int(part) for part in selection.split(",") if part}
    mask = np.isin(np.asarray(resnums), list(wanted))
  if constrain_to_exposed_non_ss:
    non_ss = np.asarray([code not in {"H", "E"} for code in ss_code])
    mask = mask & non_ss & np.asarray(exposed, dtype=bool)
  return mask


def _strict(spec: Any) -> bool:  # noqa: ANN401
  """``strict_load`` applies only when the caller passed ``LaserOptions``.

  Parity scripts load a checkpoint with no options object. Defaulting that
  path to strict would reject the f64 ``D_mu`` skip and the score wave.
  """
  options = getattr(spec, "laser", None)
  if options is None:
    return False
  return bool(getattr(options, "strict_load", True))


def _inputs(spec: Any) -> list[Any]:  # noqa: ANN401
  raw = spec.inputs
  if isinstance(raw, (str, Path)):
    return [raw]
  return list(raw)


def _working_dtype() -> np.dtype[np.floating]:
  if _x64_enabled():
    return np.dtype(np.float64)
  return np.dtype(np.float32)


def _letters(sequence: str) -> np.ndarray:
  try:
    laser = [LASER_ALPHABET.index(letter) for letter in sequence]
  except ValueError as exc:
    msg = f"sequence contains a residue outside {LASER_ALPHABET}"
    raise ValueError(msg) from exc
  return np.asarray(laser, dtype=np.int32)


def _canonical_index(laser_indices: np.ndarray) -> np.ndarray:
  table = np.asarray(CANONICAL_OF_LASER)
  return table[laser_indices]


def _chi_for_candidate(features: LaserFeatures, laser_row: np.ndarray) -> np.ndarray:
  """Keep native chi only on residues whose letter did not change.

  A mismatched candidate must not be scored with the native rotamer (that is
  a different sequence's side chain). Matching letters keep the featurizer's
  chi, including its NaN slots.
  """
  native = np.asarray(features.sequence_indices)
  chi = np.asarray(features.chi_angles, dtype=_working_dtype())
  same = laser_row == native
  return np.where(same[:, None], chi, np.nan)


def _score_prepared(model: LASErMPNN, item: _Prepared, purpose: str) -> jax.Array:
  structure = _graph(model)
  period = np.asarray(model.period_index, dtype=np.int64)
  group = np.asarray(model.group_index, dtype=np.int64)
  rows: list[jax.Array] = []
  for row, canon in zip(item.laser_indices, item.canonical_indices, strict=True):
    scored = score_structure(
      model.encoder,
      model.decoder,
      item.features.backbone_coords,
      item.features.ligand_coords,
      item.features.ligand_atomic_numbers,
      item.features.ligand_subbatch_indices,
      period,
      group,
      structure,
      item.order,
      np.asarray(row, dtype=np.int32),
      _chi_for_candidate(item.features, row),
    )
    logits = canonical_logits(jnp.asarray(scored.sequence_logits))
    if purpose == "score:logits":
      rows.append(logits.astype(jnp.float32))
    else:
      rows.append(sequence_nll(logits, jnp.asarray(canon, dtype=jnp.int32)).astype(jnp.float32))
  return jnp.stack(rows)


def _proofread_batch(
  model: LASErMPNN,
  prepared: tuple[_Prepared, ...],
  options: LaserOptions,
  *,
  purpose: str,
  scalar_dropout: bool,
  seed: int,
) -> Mapping[str, jax.Array]:
  """Unconditional logits, or the dropout ensemble, on the focus rows."""
  means: list[np.ndarray] = []
  stds: list[np.ndarray] = []
  plus: list[np.ndarray] = []
  ids: list[np.ndarray] = []
  structure = _graph(model)
  period = np.asarray(model.period_index)
  group = np.asarray(model.group_index)
  joint = LaserJointDecode(key=jax.random.PRNGKey(seed))
  for item in prepared:
    features = item.features
    rows = focus_rows(
      np.asarray(features.first_shell_ligand_contact_mask),
      np.asarray(features.resnum_indices),
      options.selection_string,
    )
    if purpose == "score:proofread_unconditional":
      logits = unconditional_logits(
        model.encoder,
        model.decoder,
        features.backbone_coords,
        features.ligand_coords,
        features.ligand_atomic_numbers,
        features.ligand_subbatch_indices,
        period,
        group,
        structure,
        np.asarray(features.sequence_indices),
        np.asarray(features.chi_angles),
      )
      chosen = canonical_logits(jnp.asarray(logits)[rows])
      means.append(np.asarray(jax.nn.softmax(chosen, axis=-1), dtype=np.float32))
      ids.append(np.asarray(features.row_to_resindex)[rows].astype(np.int32))
      continue
    length = int(features.sequence_indices.shape[0])
    n_orders = int(options.n_decoding_orders)
    n_dropouts = int(options.n_dropouts)
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(length).astype(np.int32) for _ in range(n_orders)]
    focus_probs: list[np.ndarray] = []
    for focus in rows:
      probs = conditional_focus_probs(
        model.encoder,
        model.decoder,
        joint,
        features.backbone_coords,
        features.ligand_coords,
        features.ligand_atomic_numbers,
        features.ligand_subbatch_indices,
        period,
        group,
        structure,
        np.asarray(features.sequence_indices),
        np.asarray(features.chi_angles),
        int(focus),
        orders,
        [[None] * n_orders for _ in range(n_dropouts)],
        scalar=bool(options.proofread_dropout) and scalar_dropout,
        vector=False,
        repack_all=bool(options.repack_all or options.repack_only),
      )
      focus_probs.append(probs)
    stacked_probs = np.stack(focus_probs, axis=2)
    mean, std, both = reduce_proofread(jnp.asarray(stacked_probs))
    # Alphabet order is canonical. mean_plus_std stays a sum, not a softmax.
    means.append(np.asarray(canonical_logits(mean), dtype=np.float32))
    stds.append(np.asarray(canonical_logits(std), dtype=np.float32))
    plus.append(np.asarray(canonical_logits(both), dtype=np.float32))
    ids.append(np.asarray(features.row_to_resindex)[rows].astype(np.int32))
  if purpose == "score:proofread_unconditional":
    return {
      "proofread_mean": jnp.stack([jnp.asarray(row) for row in means]),
      "residue_ids": jnp.stack([jnp.asarray(row) for row in ids]),
    }
  return {
    "proofread_mean": jnp.stack([jnp.asarray(row) for row in means]),
    "proofread_std": jnp.stack([jnp.asarray(row) for row in stds]),
    "proofread_mean_plus_std": jnp.stack([jnp.asarray(row) for row in plus]),
    "residue_ids": jnp.stack([jnp.asarray(row) for row in ids]),
  }


class _ScoreStages:
  """Teacher-forced score for one batch. Not an ``eqx.Module`` and not jitted.

  The decoder's own ``filter_jit`` owns the traced body. Looping candidates
  here stays on the host, where each sequence is a concrete index array.
  """

  def __init__(self, model: LASErMPNN, purpose: str, options: LaserOptions, seed: int) -> None:
    self._model = model
    self._purpose = purpose
    self._options = options
    self._seed = seed

  def __call__(
    self,
    batch: FamilyBatch,
    *,
    chunk_start: int,
    chunk_count: int,
    scalar_dropout: bool,
    dropout_key: jax.Array,
  ) -> Mapping[str, jax.Array]:
    del chunk_start, chunk_count, dropout_key
    prepared = cast("tuple[_Prepared, ...]", batch.arrays["prepared"])
    if self._purpose.startswith("score:proofread_"):
      return _proofread_batch(
        self._model,
        prepared,
        self._options,
        purpose=self._purpose,
        scalar_dropout=scalar_dropout,
        seed=self._seed,
      )
    stacked = jnp.stack([_score_prepared(self._model, item, self._purpose) for item in prepared])
    key = "logits" if self._purpose == "score:logits" else "nll"
    payload: dict[str, jax.Array] = {key: stacked}
    if self._options.output_fasta or self._options.output_fasta_only:
      payload["fasta"] = jnp.stack(
        [jnp.asarray(item.canonical_indices[0]) for item in prepared],
      )
    if self._options.output_fasta_only:
      return {"fasta": payload["fasta"]}
    return payload


def _prepare(path: Path, spec: Any, options: LaserOptions) -> _Prepared:  # noqa: ANN401
  features = featurize(
    path,
    fix_from_bfactor=options.fix_from_bfactor,
    use_water=options.use_water,
    noncanonical_aa_ligand=options.noncanonical_aa_ligand,
    ignore_ligand=options.ignore_ligand,
    dtype=_working_dtype(),
  )
  length = int(features.sequence_indices.shape[0])
  rows = [_letters(sequence) for sequence in _spec_sequences(spec)]
  for index, row in enumerate(rows):
    if int(row.shape[0]) != length:
      msg = f"sequences_to_score[{index}] length {row.shape[0]} != structure length {length}"
      raise ValueError(msg)
  laser = np.stack(rows)
  return _Prepared(
    features,
    laser,
    _canonical_index(laser),
    decoding_order(
      np.asarray(features.chain_mask),
      np.asarray(features.extra_atom_contact_mask),
      seed=int(spec.random_seed),
    ),
  )


def _spec_sequences(spec: Any) -> tuple[str, ...]:  # noqa: ANN401
  raw = getattr(spec, "sequences_to_score", ())
  return tuple(str(item) for item in raw)


def _set_path(tree: Any, parts: list[str], value: jax.Array) -> Any:  # noqa: ANN401
  head, *rest = parts
  if head.isdigit():
    index = int(head)
    seq = list(tree)
    seq[index] = _set_path(seq[index], rest, value) if rest else value
    return tuple(seq)
  child = _set_path(getattr(tree, head), rest, value) if rest else value
  return eqx.tree_at(lambda obj, name=head: getattr(obj, name), tree, child)


def _assign_state(
  module: Any,  # noqa: ANN401
  state: dict[str, Any],
  prefixes: tuple[str, ...],
  dtype: np.dtype[np.floating],
) -> Any:  # noqa: ANN401
  for key, value in state.items():
    text = str(key)
    if not text.startswith(prefixes):
      continue
    # D_mu is a derived linspace. The f64 laser_score wave leaves it alone
    # so a cast does not fight the encoder's own reconstruction.
    if dtype == np.dtype(np.float64) and text.endswith("D_mu"):
      continue
    array = np.asarray(value.detach().cpu().numpy())
    if np.issubdtype(array.dtype, np.floating):
      array = array.astype(dtype, copy=False)
    module = _set_path(module, text.split("."), jnp.asarray(array))
  return module


def _torch_blob(path: Path) -> dict[str, Any]:
  import torch  # noqa: PLC0415

  blob = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(blob, dict):
    msg = f"{path} is not a checkpoint dict"
    raise TypeError(msg)
  return blob


def _graph_from_blob(blob: dict[str, Any]) -> GraphStructure:
  params = blob["params"]
  if not isinstance(params, dict):
    msg = "checkpoint params is not a dict"
    raise TypeError(msg)
  model_params = params["model_params"]
  if not isinstance(model_params, dict):
    msg = "checkpoint model_params is not a dict"
    raise TypeError(msg)
  raw = model_params["graph_structure"]
  if not isinstance(raw, dict):
    msg = "checkpoint graph_structure is not a dict"
    raise TypeError(msg)
  return GraphStructure(
    pr_pr_knn_graph_k=int(raw["pr_pr_knn_graph_k"]),
    lig_pr_knn_graph_k=int(raw["lig_pr_knn_graph_k"]),
    lig_lig_knn_graph_k=int(raw["lig_lig_knn_graph_k"]),
    lig_pr_distance_cutoff=float(raw["lig_pr_distance_cutoff"]),
  )


def _assert_strict_keys(
  module: Any,  # noqa: ANN401
  state: dict[str, Any],
  prefixes: tuple[str, ...],
  dtype: np.dtype[np.floating],
) -> None:
  """Reject a checkpoint key the module cannot store.

  ``D_mu`` is skipped on f64 because the encoder rebuilds that linspace. Other
  prefix keys must name a real field, which is what ``strict_load`` asks for.
  """
  for key in state:
    text = str(key)
    if not text.startswith(prefixes):
      continue
    if dtype == np.dtype(np.float64) and text.endswith("D_mu"):
      continue
    obj: Any = module
    for part in text.split("."):
      if part.isdigit():
        obj = obj[int(part)]
        continue
      if not hasattr(obj, part):
        msg = f"strict_load: checkpoint key {text} is not a model field"
        raise ValueError(msg)
      obj = getattr(obj, part)


def _load_torch(path: Path, *, strict: bool = False) -> LASErMPNN:
  """Build an Equinox LASErMPNN from a torch checkpoint. Torch stays lazy."""
  blob = _torch_blob(path)
  state = blob["model_state_dict"]
  if not isinstance(state, dict):
    msg = "model_state_dict is not a dict"
    raise TypeError(msg)
  graph = _graph_from_blob(blob)
  dtype = _working_dtype()
  period = np.asarray(state["ligand_featurizer.atomic_number_idx_to_period_idx"])
  group = np.asarray(state["ligand_featurizer.atomic_number_idx_to_group_idx"])
  model = LASErMPNN(
    key=jax.random.PRNGKey(0),
    period_index=period.astype(np.int32, copy=False),
    group_index=group.astype(np.int32, copy=False),
    pr_pr_knn_graph_k=graph.pr_pr_knn_graph_k,
    lig_pr_knn_graph_k=graph.lig_pr_knn_graph_k,
    lig_lig_knn_graph_k=graph.lig_lig_knn_graph_k,
    lig_pr_distance_cutoff=graph.lig_pr_distance_cutoff,
  )
  if strict:
    _assert_strict_keys(model.encoder, state, _ENCODER_PREFIXES, dtype)
    _assert_strict_keys(model.decoder, state, _DECODER_PREFIXES, dtype)
  encoder = _assign_state(model.encoder, state, _ENCODER_PREFIXES, dtype)
  decoder = _assign_state(model.decoder, state, _DECODER_PREFIXES, dtype)
  model = eqx.tree_at(lambda module: module.encoder, model, encoder)
  return eqx.tree_at(lambda module: module.decoder, model, decoder)


def _load_eqx(path: Path) -> LASErMPNN:
  skeleton = LASErMPNN(key=jax.random.PRNGKey(0))
  loaded = eqx.tree_deserialise_leaves(path, skeleton)
  return cast("LASErMPNN", loaded)


class LaserDriver:
  """Score LASErMPNN through the family-driver seam."""

  name = "lasermpnn"
  options_type = LaserOptions
  mpnn_fallback_purposes: frozenset[str] = frozenset()

  def handles(self, spec: Any, purpose: str) -> bool:  # noqa: ANN401
    """True for scoring and both proofread purposes."""
    del spec
    return purpose in _HANDLED

  def load(self, spec: Any) -> LASErMPNN:  # noqa: ANN401
    """Load a converted LASErMPNN. Does not enter inference mode.

    ``model_local_path`` wins. A ``.pt`` path is converted on the fly
    (torch is imported only there). Otherwise the registry artifact is
    loaded, and ``lasermpnn`` entries must carry ``sha256``.
    """
    local = getattr(spec, "model_local_path", None)
    if local is not None:
      path = Path(str(local))
      if path.suffix.lower() == ".pt":
        return _load_torch(path, strict=_strict(spec))
      return _load_eqx(path)
    artifact = _resolve_local_checkpoint_from_registry(spec)
    if artifact is None:
      msg = "LaserDriver requires model_local_path or a checkpoint registry entry with sha256"
      raise ValueError(msg)
    path = Path(artifact)
    if path.suffix.lower() == ".pt":
      return _load_torch(path, strict=_strict(spec))
    return _load_eqx(path)

  def mpnn_core(self, model: eqx.Module) -> None:
    """LASErMPNN has no embedded stock MPNN, so there is no fallback core."""
    del model

  def batches(self, spec: Any) -> Iterator[FamilyBatch]:  # noqa: ANN401
    """One structure per batch, featurized by the B0 host port."""
    options = _options(spec)
    pending: list[tuple[int, str]] = []
    for index, item in enumerate(_inputs(spec)):
      if not isinstance(item, (str, Path)):
        pending.append((index, "lasermpnn_requires_pdb"))
        continue
      try:
        prepared = _prepare(Path(item), spec, options)
      except (LaserInputError, ValueError) as exc:
        pending.append((index, str(exc)))
        continue
      length = int(prepared.features.sequence_indices.shape[0])
      yield FamilyBatch(
        input_indices=(index,),
        arrays={"prepared": (prepared,)},
        skipped=tuple(pending),
        lengths=(length,),
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
    """One structure axis. Candidates stay inside the stage."""
    del spec, purpose
    return [
      AxisSpec(
        name="structures",
        cardinality=max(len(batch.input_indices), 1),
        default_batch_size=1,
        heterogeneous=True,
      ),
    ]

  def stages(self, spec: Any, purpose: str, model: eqx.Module) -> _ScoreStages:  # noqa: ANN401
    """Per-batch teacher-forced score, or a proofread reduction."""
    return _ScoreStages(cast("LASErMPNN", model), purpose, _options(spec), int(spec.random_seed))

  def result_schema(self, spec: Any, purpose: str) -> Mapping[str, SinkArraySpec]:  # noqa: ANN401
    """Per-position arrays. ``L_total`` is sliced by the family runner."""
    options = _options(spec)
    fasta = {
      "fasta": SinkArraySpec(dims=("N_cand", "L_total"), dtype="int32", attrs={}),
    }
    if purpose == "score:logits":
      body = {
        "logits": SinkArraySpec(dims=("N_cand", "L_total", "alphabet"), dtype="float32", attrs={}),
      }
    elif purpose == "score:nll":
      body = {
        "nll": SinkArraySpec(dims=("N_cand", "L_total"), dtype="float32", attrs={}),
      }
    elif purpose == "score:proofread_unconditional":
      body = {
        "proofread_mean": SinkArraySpec(
          dims=("N_cand", "R", "alphabet"),
          dtype="float32",
          attrs={},
        ),
        "residue_ids": SinkArraySpec(dims=("N_cand", "R"), dtype="int32", attrs={}),
      }
    elif purpose == "score:proofread_conditional":
      body = {
        "proofread_mean": SinkArraySpec(
          dims=("N_cand", "R", "alphabet"),
          dtype="float32",
          attrs={},
        ),
        "proofread_std": SinkArraySpec(dims=("N_cand", "R", "alphabet"), dtype="float32", attrs={}),
        "proofread_mean_plus_std": SinkArraySpec(
          dims=("N_cand", "R", "alphabet"),
          dtype="float32",
          attrs={},
        ),
        "residue_ids": SinkArraySpec(dims=("N_cand", "R"), dtype="int32", attrs={}),
      }
    else:
      msg = f"lasermpnn does not support {purpose} in v1"
      raise ValueError(msg)
    if options.output_fasta_only:
      return fasta
    if options.output_fasta:
      return {**body, **fasta}
    return body


FAMILY_DRIVERS.register("lasermpnn")(LaserDriver())
