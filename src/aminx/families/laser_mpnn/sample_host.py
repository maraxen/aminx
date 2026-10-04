"""LASEr ``sample`` body: joint decode, tied decode, and rotamer coordinates.

The scan is ``decode_order`` / ``tied_decode``. This module only builds the
order, the uniform stream, and the result-schema arrays around those calls.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from xtrax.tiling import AxisSpec, BatchPlanner

from aminx.families.laser_mpnn.driver import (
  CANONICAL_OF_LASER,
  apply_backbone_noise,
  backbone_noise_level,
  budget_mask_for,
  decoding_order,
  load_joint_decode,
)
from aminx.families.laser_mpnn.featurize import (
  LaserFeatures,
  LaserInputError,
  build_rotamers,
  featurize,
  residue_identifiers,
)
from aminx.host.family_driver import FamilyBatch, SinkArraySpec
from aminx.host.omit_aa_bias import compile_omit_aa_bias, omit_aa_is_active
from aminx.io.laser_pdb import LigandAtoms, write_laser_pdb
from aminx.model.laser.graphs import GraphStructure
from aminx.model.laser.joint_decode import chi_position_mask, decode_order
from aminx.model.laser.tied import tied_decode
from aminx.run.options import LaserOptions

_DRAWS_PER_STEP = 5


class _Sampled(NamedTuple):
  """One structure, the mask the order rule sees, and the optional tied partner."""

  features: LaserFeatures
  chain_mask: np.ndarray
  tied: LaserFeatures | None


def _x64_enabled() -> bool:
  """Runtime x64 flag. The ty stub for ``jax.config`` does not list it."""
  return bool(getattr(jax.config, "jax_enable_x64", False))


def _options(spec: Any) -> LaserOptions:  # noqa: ANN401
  options = getattr(spec, "laser", None)
  if options is None:
    return LaserOptions()
  return cast("LaserOptions", options)


def _inputs(spec: Any) -> list[Any]:  # noqa: ANN401
  raw = spec.inputs
  if isinstance(raw, (str, Path)):
    return [raw]
  return list(raw)


def _working_dtype() -> np.dtype[np.floating]:
  if _x64_enabled():
    return np.dtype(np.float64)
  return np.dtype(np.float32)


def _graph(model: Any) -> GraphStructure:  # noqa: ANN401
  return GraphStructure(
    pr_pr_knn_graph_k=int(model.pr_pr_knn_graph_k),
    lig_pr_knn_graph_k=int(model.lig_pr_knn_graph_k),
    lig_lig_knn_graph_k=int(model.lig_lig_knn_graph_k),
    lig_pr_distance_cutoff=float(model.lig_pr_distance_cutoff),
  )


def _temperatures(spec: Any) -> tuple[float | None, ...]:  # noqa: ANN401
  temps = tuple(spec.run_spec.sampling.temperature)
  if not temps:
    return (None,)
  return tuple(None if value is None else float(value) for value in temps)


def _featurize_one(
  path: Path,
  spec: Any,  # noqa: ANN401
  options: LaserOptions,
  *,
  noise_fold: int,
) -> LaserFeatures:
  """B0 features plus LASEr backbone noise. Fold 0 matches the score-path key."""
  features = featurize(
    path,
    fix_from_bfactor=options.fix_from_bfactor,
    use_water=options.use_water,
    noncanonical_aa_ligand=options.noncanonical_aa_ligand,
    ignore_ligand=options.ignore_ligand,
    dtype=_working_dtype(),
  )
  level = backbone_noise_level(spec)
  if level > 0.0:
    base = jax.random.fold_in(jax.random.PRNGKey(int(spec.random_seed)), 0x6262)
    key = base if noise_fold == 0 else jax.random.fold_in(base, noise_fold)
    features = dataclasses.replace(
      features,
      backbone_coords=apply_backbone_noise(features.backbone_coords, level, key=key),
    )
  return features


def _chain_mask(features: LaserFeatures, spec: Any, options: LaserOptions) -> np.ndarray:  # noqa: ANN401
  """LASEr ``chain_mask``: 1 is fixed. ``repack_only`` fixes every row."""
  mask = np.array(features.chain_mask, dtype=bool, copy=True)
  fixed = getattr(spec, "fixed_positions", None)
  if fixed is not None:
    for index in np.asarray(fixed, dtype=np.int32).reshape(-1):
      slot = int(index)
      if 0 <= slot < mask.shape[0]:
        mask[slot] = True
  if options.repack_only:
    mask[:] = True
  return mask


def _tied_partner(
  options: LaserOptions,
  spec: Any,  # noqa: ANN401
  length: int,
) -> LaserFeatures | None:
  if not options.tied_second_input:
    return None
  try:
    partner = _featurize_one(Path(options.tied_second_input), spec, options, noise_fold=1)
  except LaserInputError as exc:
    msg = f"tied_second_input: {exc}"
    raise ValueError(msg) from exc
  partner_length = int(partner.sequence_indices.shape[0])
  if partner_length != length:
    msg = f"tied structures must have the same residue count ({length} != {partner_length})"
    raise ValueError(msg)
  return partner


def sample_batches(spec: Any) -> Iterator[FamilyBatch]:  # noqa: ANN401
  """One structure per batch. Tied partners are featurized here, decoded later."""
  options = _options(spec)
  pending: list[tuple[int, str]] = []
  for index, item in enumerate(_inputs(spec)):
    if not isinstance(item, (str, Path)):
      pending.append((index, "lasermpnn_requires_pdb"))
      continue
    try:
      features = _featurize_one(Path(item), spec, options, noise_fold=0)
    except LaserInputError as exc:
      pending.append((index, str(exc)))
      continue
    length = int(features.sequence_indices.shape[0])
    sampled = _Sampled(
      features,
      _chain_mask(features, spec, options),
      _tied_partner(options, spec, length),
    )
    yield FamilyBatch(
      input_indices=(index,),
      arrays={"sampled": (sampled,)},
      skipped=tuple(pending),
      lengths=(length,),
    )
    pending = []
  if pending:
    yield FamilyBatch(
      input_indices=(),
      arrays={"sampled": ()},
      skipped=tuple(pending),
      lengths=(),
    )


def sample_axes(spec: Any, batch: FamilyBatch) -> list[AxisSpec]:  # noqa: ANN401
  """``samples`` and ``temperatures``, planned through ``BatchPlanner``."""
  n_samples = max(int(getattr(spec, "num_samples", 1) or 1), 1)
  n_temp = max(len(_temperatures(spec)), 1)
  specs = [
    AxisSpec(
      name="structures",
      cardinality=max(len(batch.input_indices), 1),
      default_batch_size=1,
      heterogeneous=True,
    ),
    AxisSpec(name="samples", cardinality=n_samples, default_batch_size=n_samples),
    AxisSpec(name="temperatures", cardinality=n_temp, default_batch_size=n_temp),
  ]
  BatchPlanner().plan(specs)
  return specs


def _include_coords(options: LaserOptions) -> bool:
  """Coordinates stand in for the PDB sink. FASTA-only runs omit them."""
  return not options.output_fasta_only


def _include_tied_logits(spec: Any, options: LaserOptions) -> bool:  # noqa: ANN401
  return bool(options.tied_second_input) and bool(getattr(spec, "return_logits", False))


def sample_schema(spec: Any) -> dict[str, SinkArraySpec]:  # noqa: ANN401
  """§5.6 sample channels. ``sidechain_coords`` is omitted for FASTA-only runs."""
  options = _options(spec)
  fasta = {
    "fasta": SinkArraySpec(dims=("N", "L_total"), dtype="int32", attrs={}),
  }
  if options.output_fasta_only:
    return fasta
  sequence = SinkArraySpec(dims=("N", "L_total"), dtype="int32", attrs={})
  log_prob = SinkArraySpec(dims=("N", "L_total"), dtype="float32", attrs={})
  chi = SinkArraySpec(dims=("N", "L_total", "4"), dtype="float32", attrs={})
  mask = SinkArraySpec(dims=("N", "L_total", "4"), dtype="bool", attrs={})
  schema: dict[str, SinkArraySpec] = {
    "sequence": sequence,
    "seq_log_prob": log_prob,
    "chi_deg": chi,
    "chi_mask": mask,
  }
  if _include_coords(options):
    schema["sidechain_coords"] = SinkArraySpec(
      dims=("N", "L_total", "atom", "xyz"),
      dtype="float32",
      attrs={},
    )
  if options.tied_second_input:
    schema["seq_log_prob_2"] = log_prob
    schema["chi_deg_2"] = chi
    if _include_coords(options):
      schema["sidechain_coords_2"] = schema["sidechain_coords"]
    if _include_tied_logits(spec, options):
      schema["seq_logits_2"] = SinkArraySpec(
        dims=("N", "L_total", "alphabet"),
        dtype="float32",
        attrs={},
      )
      schema["chi_logits_2"] = SinkArraySpec(
        dims=("N", "L_total", "4", "chi_bin"),
        dtype="float32",
        attrs={},
      )
  if options.output_fasta:
    schema.update(fasta)
  return schema


def _logit_bias(
  spec: Any,  # noqa: ANN401
  chain_mask: np.ndarray,
) -> np.ndarray | None:
  """Canonical omit/bias, then LASEr column order. ``None`` when nothing is set."""
  length = int(chain_mask.shape[0])
  omit = tuple(getattr(spec, "omit_aa", ()) or ())
  per_position = getattr(spec, "omit_aa_per_position", None)
  bias = getattr(spec, "bias", None)
  if bias is None and not omit_aa_is_active(omit, per_position):
    return None
  compiled = compile_omit_aa_bias(
    bias,
    fixed_mask_row=chain_mask.astype(np.float32),
    omit_aa=omit,
    omit_aa_per_position=per_position,
    seq_len=length,
  )
  canonical = np.asarray(compiled, dtype=_working_dtype())
  table = np.asarray(CANONICAL_OF_LASER)
  return canonical[:, table]


def _budget(features: LaserFeatures, options: LaserOptions) -> np.ndarray:
  return budget_mask_for(
    np.asarray(features.resnum_indices),
    features.ss_code,
    np.asarray(features.exposed_mask),
    selection=options.budget_residue_selection,
    constrain_to_exposed_non_ss=bool(options.constrain_ala_gly_to_exposed_non_ss),
  )


def _charged(features: LaserFeatures, options: LaserOptions) -> np.ndarray:
  length = int(features.sequence_indices.shape[0])
  if not options.disable_charged_fs:
    return np.zeros((length,), dtype=bool)
  return np.asarray(features.first_shell_ligand_contact_mask, dtype=bool)


def _draw_dtype() -> jnp.dtype:
  if _x64_enabled():
    return jnp.float64
  return jnp.float32


def _order_and_uniforms(
  spec: Any,  # noqa: ANN401
  chain_mask: np.ndarray,
  contact: np.ndarray,
  key: jax.Array,
  *,
  tied: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
  """Per-sample order from ``decoding_order``, plus the uniform stream decode consumes."""
  injected_order = getattr(spec, "injected_decoding_order", None)
  injected_uniforms = getattr(spec, "injected_uniforms", None)
  order_key, draw_key, chi_key = jax.random.split(key, 3)
  if injected_order is None:
    seed = int(jax.random.randint(order_key, (), 0, np.int32(2**31 - 1)))
    order = decoding_order(chain_mask, contact, seed=seed)
  else:
    order = np.asarray(injected_order, dtype=np.int32)
  length = int(chain_mask.shape[0])
  if injected_uniforms is not None:
    uniforms = np.asarray(injected_uniforms, dtype=np.float64)
    return order, uniforms, None
  draw_dtype = _draw_dtype()
  if tied:
    seq_u = np.asarray(jax.random.uniform(draw_key, (length,), dtype=draw_dtype), dtype=np.float64)
    chi_u = np.asarray(
      jax.random.uniform(chi_key, (length, 4, 2), dtype=draw_dtype),
      dtype=np.float64,
    )
    return order, seq_u, chi_u
  uniforms = np.asarray(
    jax.random.uniform(draw_key, (length * _DRAWS_PER_STEP,), dtype=draw_dtype),
    dtype=np.float64,
  )
  return order, uniforms, None


def _canonical_sequence(laser_index: np.ndarray) -> np.ndarray:
  table = np.asarray(CANONICAL_OF_LASER)
  return table[np.asarray(laser_index, dtype=np.int32)].astype(np.int32, copy=False)


def _token_prob(logits: np.ndarray, laser_index: np.ndarray) -> np.ndarray:
  """``softmax(stored)[sampled]``: the post-min-p probability of the drawn token."""
  probs = jax.nn.softmax(jnp.asarray(logits), axis=-1)
  index = jnp.asarray(laser_index, dtype=jnp.int32)
  picked = jnp.take_along_axis(probs, index[:, None], axis=-1)[:, 0]
  return np.asarray(picked, dtype=np.float32)


def _coords(
  model: Any,  # noqa: ANN401
  backbone: np.ndarray,
  chi: np.ndarray,
  sequence: np.ndarray,
) -> np.ndarray:
  built = build_rotamers(
    backbone,
    np.asarray(chi),
    np.asarray(sequence),
    _working_dtype(),
    add_nonrotatable_hydrogens=bool(model.build_hydrogens),
  )
  return np.asarray(built, dtype=np.float32)


class SampleStages:
  """Per-chunk LASEr sample. One structure is decoded once per sample and temperature."""

  def __init__(self, model: Any, spec: Any) -> None:  # noqa: ANN401
    self._model = model
    self._spec = spec
    self._options = _options(spec)
    self._joint = load_joint_decode(spec)

  def __call__(
    self,
    batch: FamilyBatch,
    *,
    chunk_start: int,
    chunk_count: int,
    scalar_dropout: bool,
    dropout_key: jax.Array,
  ) -> Mapping[str, jax.Array]:
    del scalar_dropout
    rows = [
      self._one(item, chunk_start, chunk_count, dropout_key) for item in batch.arrays["sampled"]
    ]
    keys = rows[0].keys()
    return {key: jnp.stack([row[key] for row in rows]) for key in keys}

  def _one(
    self,
    item: _Sampled,
    chunk_start: int,
    chunk_count: int,
    dropout_key: jax.Array,
  ) -> dict[str, np.ndarray]:
    temperatures = _temperatures(self._spec)
    n_temp = len(temperatures)
    sequence: list[np.ndarray] = []
    seq_log_prob: list[np.ndarray] = []
    chi_deg: list[np.ndarray] = []
    chi_mask: list[np.ndarray] = []
    coords: list[np.ndarray] = []
    seq_log_prob_2: list[np.ndarray] = []
    chi_deg_2: list[np.ndarray] = []
    coords_2: list[np.ndarray] = []
    seq_logits_2: list[np.ndarray] = []
    chi_logits_2: list[np.ndarray] = []
    for offset in range(chunk_count):
      for temp_index, temperature in enumerate(temperatures):
        flat = (chunk_start + offset) * n_temp + temp_index
        key = jax.random.fold_in(dropout_key, flat)
        decoded = self._decode(item, temperature, key)
        sequence.append(decoded["sequence"])
        seq_log_prob.append(decoded["seq_log_prob"])
        chi_deg.append(decoded["chi_deg"])
        chi_mask.append(decoded["chi_mask"])
        if "sidechain_coords" in decoded:
          coords.append(decoded["sidechain_coords"])
        if "seq_log_prob_2" in decoded:
          seq_log_prob_2.append(decoded["seq_log_prob_2"])
          chi_deg_2.append(decoded["chi_deg_2"])
        if "sidechain_coords_2" in decoded:
          coords_2.append(decoded["sidechain_coords_2"])
        if "seq_logits_2" in decoded:
          seq_logits_2.append(decoded["seq_logits_2"])
          chi_logits_2.append(decoded["chi_logits_2"])
    produced: dict[str, np.ndarray] = {
      "sequence": np.stack(sequence),
      "seq_log_prob": np.stack(seq_log_prob),
      "chi_deg": np.stack(chi_deg),
      "chi_mask": np.stack(chi_mask),
    }
    if coords:
      produced["sidechain_coords"] = np.stack(coords)
    if seq_log_prob_2:
      produced["seq_log_prob_2"] = np.stack(seq_log_prob_2)
      produced["chi_deg_2"] = np.stack(chi_deg_2)
    if coords_2:
      produced["sidechain_coords_2"] = np.stack(coords_2)
    if seq_logits_2:
      produced["seq_logits_2"] = np.stack(seq_logits_2)
      produced["chi_logits_2"] = np.stack(chi_logits_2)
    if self._options.output_fasta or self._options.output_fasta_only:
      produced["fasta"] = produced["sequence"]
    if self._options.output_fasta_only:
      return {"fasta": produced["fasta"]}
    return produced

  def _decode(
    self,
    item: _Sampled,
    temperature: float | None,
    key: jax.Array,
  ) -> dict[str, np.ndarray]:
    if item.tied is not None:
      return self._decode_tied(item, temperature, key)
    return self._decode_one(item, temperature, key)

  def _decode_one(
    self,
    item: _Sampled,
    temperature: float | None,
    key: jax.Array,
  ) -> dict[str, np.ndarray]:
    features = item.features
    options = self._options
    model = self._model
    order, uniforms, _chi_u = _order_and_uniforms(
      self._spec,
      item.chain_mask,
      np.asarray(features.extra_atom_contact_mask),
      key,
      tied=False,
    )
    decoded = decode_order(
      model.encoder,
      model.decoder,
      self._joint,
      features.backbone_coords,
      features.ligand_coords,
      features.ligand_atomic_numbers,
      features.ligand_subbatch_indices,
      np.asarray(model.period_index),
      np.asarray(model.group_index),
      _graph(model),
      np.asarray(features.sequence_indices),
      np.asarray(features.chi_angles),
      item.chain_mask,
      np.asarray(features.first_shell_ligand_contact_mask),
      order,
      uniforms,
      sequence_temperature=temperature,
      chi_temperature=options.chi_temp,
      fs_sequence_temp=options.fs_sequence_temp,
      seq_min_p=float(options.seq_min_p),
      chi_min_p=float(options.chi_min_p),
      ala_budget=int(options.ala_budget),
      gly_budget=int(options.gly_budget),
      disabled_residues=tuple(options.disabled_residues),
      ignore_chain_mask_zeros=bool(options.ignore_chain_mask_zeros),
      repack_all=bool(options.repack_all),
      repack_only=bool(options.repack_only),
      budget_mask=_budget(features, options),
      charged_mask=_charged(features, options),
      bias=_logit_bias(self._spec, item.chain_mask),
    )
    laser_index = np.asarray(decoded.sequence)
    produced: dict[str, np.ndarray] = {
      "sequence": _canonical_sequence(laser_index),
      "seq_log_prob": _token_prob(np.asarray(decoded.sequence_logits), laser_index),
      "chi_deg": np.asarray(decoded.chi_degrees, dtype=np.float32),
      "chi_mask": np.asarray(chi_position_mask(laser_index)),
    }
    if _include_coords(options):
      produced["sidechain_coords"] = _coords(
        model,
        features.backbone_coords,
        np.asarray(decoded.chi_degrees),
        laser_index,
      )
    return produced

  def _decode_tied(
    self,
    item: _Sampled,
    temperature: float | None,
    key: jax.Array,
  ) -> dict[str, np.ndarray]:
    features = item.features
    partner = item.tied
    if partner is None:
      msg = "tied sample is missing the second structure"
      raise ValueError(msg)
    options = self._options
    model = self._model
    order, seq_u, chi_u = _order_and_uniforms(
      self._spec,
      item.chain_mask,
      np.asarray(features.extra_atom_contact_mask),
      key,
      tied=True,
    )
    if chi_u is None:
      chi_u = np.zeros((int(order.shape[0]), 4, 2), dtype=np.float64)
    # ``tied_decode`` refuses seq_min_p / chi_min_p / fs temp / charged / ignore-zeros.
    decoded = tied_decode(
      model.encoder,
      model.decoder,
      self._joint,
      features.backbone_coords,
      partner.backbone_coords,
      features.ligand_coords,
      partner.ligand_coords,
      features.ligand_atomic_numbers,
      partner.ligand_atomic_numbers,
      features.ligand_subbatch_indices,
      partner.ligand_subbatch_indices,
      np.asarray(model.period_index),
      np.asarray(model.group_index),
      _graph(model),
      np.asarray(features.sequence_indices),
      np.asarray(features.chi_angles),
      np.asarray(partner.chi_angles),
      item.chain_mask,
      order,
      seq_u,
      chi_u,
      sequence_temperature=temperature,
      chi_temperature=options.chi_temp,
      lambda_=float(options.tied_interpolation_lambda),
      repack_all=bool(options.repack_all or options.repack_only),
      disabled_residues=tuple(options.disabled_residues),
      seq_min_p=float(options.seq_min_p),
      chi_min_p=float(options.chi_min_p),
      fs_sequence_temp=options.fs_sequence_temp,
      disable_charged_fs=bool(options.disable_charged_fs),
      ignore_chain_mask_zeros=bool(options.ignore_chain_mask_zeros),
      ala_budget=int(options.ala_budget),
      gly_budget=int(options.gly_budget),
      budget_mask=_budget(features, options),
      bias=_logit_bias(self._spec, item.chain_mask),
    )
    laser_index = np.asarray(decoded.sequence)
    produced: dict[str, np.ndarray] = {
      "sequence": _canonical_sequence(laser_index),
      "seq_log_prob": _token_prob(np.asarray(decoded.sequence_logits_1), laser_index),
      "chi_deg": np.asarray(decoded.chi_degrees_1, dtype=np.float32),
      "chi_mask": np.asarray(chi_position_mask(laser_index)),
      "seq_log_prob_2": _token_prob(np.asarray(decoded.sequence_logits_2), laser_index),
      "chi_deg_2": np.asarray(decoded.chi_degrees_2, dtype=np.float32),
    }
    if _include_coords(options):
      produced["sidechain_coords"] = _coords(
        model,
        features.backbone_coords,
        np.asarray(decoded.chi_degrees_1),
        laser_index,
      )
      produced["sidechain_coords_2"] = _coords(
        model,
        partner.backbone_coords,
        np.asarray(decoded.chi_degrees_2),
        laser_index,
      )
    if _include_tied_logits(self._spec, options):
      produced["seq_logits_2"] = np.asarray(decoded.sequence_logits_2, dtype=np.float32)
      produced["chi_logits_2"] = np.asarray(decoded.chi_logits_2, dtype=np.float32)
    return produced


def format_sample_pdb(
  features: LaserFeatures,
  sequence_indices: np.ndarray,
  sidechain_coords: np.ndarray,
  *,
  bfactors: np.ndarray | None = None,
  ligand: LigandAtoms | None = None,
) -> str:
  """PDB text for one sample. The sample purpose's array outputs stay as they are.

  ``sequence_indices`` are LASEr alphabet indices, the same vector
  ``build_rotamers`` used for ``sidechain_coords``. ``bfactors`` defaults to 0;
  pass the sampled-token probability to fill the B-factor column the way
  upstream does. ``ligand`` is appended as HETATM when the caller has the
  input ligand atoms.
  """
  chains: list[str] = []
  numbers: list[int] = []
  icodes: list[str] = []
  for chain, number, icode in residue_identifiers(features):
    chains.append(chain)
    numbers.append(number)
    icodes.append(icode)
  return write_laser_pdb(
    sidechain_coords,
    sequence_indices,
    chains,
    numbers,
    icodes,
    bfactors=bfactors,
    ligand=ligand,
  )
