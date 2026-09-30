"""Dump LASErMPNN parity oracles (f64 and f32) from the pinned upstream checkout.

Run in the torch CPU oracle environment. This script imports upstream LASErMPNN,
torch, and numpy only (never aminx)::

    python3 scripts/parity/dump_laser_oracles.py \\
        --laser-root <path> --fixtures-dir <dir> --out <dir> \\
        [--checkpoints nothing_heldout ...] [--precisions f64 f32]

``--selftest`` checks the inverse-CDF edges, that every shim restores the
original callable, and (when the default checkpoint and ``4jnj-1_prot.pdb`` are
present) that an unshimmed encoder forward is bit-identical across two seeded
repeats. It does not write dumps.

Writes ``<out>/<wave>/oracle_<prec>.npz`` for each §7.2 LASEr wave
(``laser_layers``, ``laser_encoder``, ``laser_score``, ``laser_decode_step``,
``laser_rotamers``) and ``<out>/oracle_manifest.toml`` (upstream commit via
``VENDOR_PIN.toml`` when the checkout has no ``.git``, torch version, shim
sha256, per-checkpoint weights sha256).

Fixtures are the 20 PDBs in ``tests/fixtures/laser/fixtures_manifest.toml``
plus upstream ``example_pdbs/4jnj-1_prot.pdb``. ``laser_score`` also stores the
unconditional proofread logits (one forward, ``return_unconditional_probabilities``).
``--proofread-conditional`` adds one focus residue on ``4jnj-1_prot`` with
injected scalar-dropout masks (vector dropout stays in eval).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import logging
import subprocess
import sys
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from oracle_shims.laser import (
  SHIM_SITES,
  assert_inverse_cdf_edges,
  assert_vector_dropout_eval,
  LITERAL_FACTORIES,
  f64_runtime,
  float32_literals_follow_active_dtype,
  ideal_coord_modules,
  ideal_coords_follow_backbone_dtype,
  injected_decoding_order_uniforms,
  injected_scalar_dropout,
  injected_uniform_draws,
  shim_sha256,
)

logger = logging.getLogger("dump_laser_oracles")

SEED = 0
SEQ_TEMPERATURE = 0.3
CHI_TEMPERATURE = 0.3
N_DECODE_DRAWS = 8
LAYER_MODULES: frozenset[str] = frozenset(
  {
    "backbone_frame_vec_input_layer",
    "backbone_frame_vec_norm",
    "ligand_encoder.input_gvp",
    "ligand_encoder.gat_layers.0",
    "protein_encoder_layers.0.hetgat",
  },
)
WAVES: tuple[str, ...] = (
  "laser_layers",
  "laser_encoder",
  "laser_score",
  "laser_decode_step",
  "laser_rotamers",
)
CHECKPOINTS: dict[str, str] = {
  "nothing_heldout": "model_weights/laser_weights_0p1A_nothing_heldout.pt",
  "noise_ligandmpnn_split": "model_weights/laser_weights_0p1A_noise_ligandmpnn_split.pt",
  "soluble_65000": "model_weights/soluble_weights_no_heldout_drop_clusters_optstep_65000.pt",
}
SHIMMED_WAVES = frozenset({"laser_score", "laser_decode_step"})


@dataclass
class Fixture:
  """One PDB the dump featurizes."""

  name: str
  path: Path


@dataclass
class Buckets:
  """Per-wave numpy payloads for one precision."""

  precision: str
  waves: dict[str, dict[str, np.ndarray]] = field(
    default_factory=lambda: {wave: {} for wave in WAVES},
  )


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _np(value: torch.Tensor) -> np.ndarray:
  return value.detach().cpu().numpy()


def _toml_str(value: str) -> str:
  return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toml_list(values: Iterable[str]) -> str:
  return "[" + ", ".join(_toml_str(item) for item in values) + "]"


def _package_dir(laser_root: Path) -> Path:
  root = laser_root.resolve()
  if (root / "utils" / "model.py").is_file():
    return root
  nested = root / "LASErMPNN"
  if (nested / "utils" / "model.py").is_file():
    return nested
  msg = f"{laser_root} does not contain LASErMPNN utils/model.py"
  raise SystemExit(msg)


def _load_pin(package: Path) -> tuple[str, dict[str, str]]:
  pin_path = package / "VENDOR_PIN.toml"
  document = tomllib.loads(pin_path.read_text())
  commit = str(document["commit"])
  weights = document["weights_sha256"]
  if not isinstance(weights, dict) or not commit:
    msg = f"could not parse commit and weights_sha256 from {pin_path}"
    raise SystemExit(msg)
  return commit, {str(key): str(val) for key, val in weights.items()}


def _upstream_commit(package: Path, pin_commit: str) -> str:
  try:
    head = subprocess.check_output(  # noqa: S603
      ["git", "-C", str(package), "rev-parse", "HEAD"],  # noqa: S607
      text=True,
      stderr=subprocess.DEVNULL,
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    logger.info("no git HEAD under %s; manifest commit is the VENDOR_PIN", package)
    return pin_commit
  if head != pin_commit:
    msg = f"LASErMPNN HEAD {head} != VENDOR_PIN commit {pin_commit}"
    raise SystemExit(msg)
  return pin_commit


def _collect_fixtures(package: Path, fixtures_dir: Path, repo: Path) -> list[Fixture]:
  manifest_path = repo / "tests" / "fixtures" / "laser" / "fixtures_manifest.toml"
  ordered = tomllib.loads(manifest_path.read_text())["ordered_ids"]
  fixtures: list[Fixture] = []
  for stem in ordered:
    path = fixtures_dir / f"{stem}.pdb"
    if not path.is_file():
      msg = f"missing LASEr fixture {path}"
      raise SystemExit(msg)
    fixtures.append(Fixture(name=str(stem), path=path))
  example = package / "example_pdbs" / "4jnj-1_prot.pdb"
  if not example.is_file():
    msg = f"missing upstream example pdb {example}"
    raise SystemExit(msg)
  fixtures.append(Fixture(name="4jnj-1_prot", path=example))
  return fixtures


def _dtype_of(precision: str) -> torch.dtype:
  if precision == "f64":
    return torch.float64
  if precision == "f32":
    return torch.float32
  msg = f"unknown precision {precision}"
  raise ValueError(msg)


def _cast_floats(value: object, dtype: torch.dtype) -> None:
  if isinstance(value, torch.Tensor):
    return
  if isinstance(value, dict):
    for item in value.values():
      _cast_floats(item, dtype)
    return
  named = getattr(value, "__dict__", None)
  if not isinstance(named, dict):
    return
  for key, item in named.items():
    if isinstance(item, torch.Tensor) and item.is_floating_point():
      named[key] = item.to(dtype=dtype)
    elif item is not None and not isinstance(item, torch.Tensor):
      _cast_floats(item, dtype)


def _put(
  buckets: Buckets,
  waves: tuple[str, ...],
  key: str,
  value: torch.Tensor | np.ndarray,
) -> None:
  array = value if isinstance(value, np.ndarray) else _np(value)
  for wave in waves:
    if key in buckets.waves[wave]:
      msg = f"duplicate oracle key {key} in {wave}"
      raise RuntimeError(msg)
    buckets.waves[wave][key] = array


def _load_model(
  checkpoint: Path,
  precision: str,
) -> tuple[torch.nn.Module, dict[str, object], list[str], list[str]]:
  from LASErMPNN.run_inference import load_model_from_parameter_dict  # noqa: PLC0415

  torch.manual_seed(SEED)
  model, params = load_model_from_parameter_dict(str(checkpoint), "cpu", strict=False)
  missing, unexpected = model.load_state_dict(  # type: ignore[attr-defined]
    torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"],
    strict=False,
  )
  if precision == "f64":
    model.double()
  model.eval()
  assert_vector_dropout_eval(model)
  return model, params, list(missing), list(unexpected)


def _featurize(path: Path, *, dtype: torch.dtype) -> object:
  from LASErMPNN.run_inference import ProteinComplexData, get_protein_hierview  # noqa: PLC0415

  view = get_protein_hierview(str(path))
  data = ProteinComplexData(view, str(path), verbose=False)
  batch = data.output_batch_data(fix_beta=False)
  _cast_floats(batch, dtype)
  return batch


def _graphs(
  model: torch.nn.Module,
  params: dict[str, object],
  batch: object,
  dtype: torch.dtype,
) -> None:
  model_params = params["model_params"]
  if not isinstance(model_params, dict):
    msg = "checkpoint params['model_params'] is not a dict"
    raise TypeError(msg)
  graph = model_params["graph_structure"]
  batch.construct_graphs(  # type: ignore[attr-defined]
    model.rotamer_builder,  # type: ignore[attr-defined]
    model.ligand_featurizer,  # type: ignore[attr-defined]
    **graph,
    protein_training_noise=0.0,
    ligand_training_noise=0.0,
    subgraph_only_dropout_rate=0.0,
    num_adjacent_residues_to_drop=0,
    build_hydrogens=bool(model_params["build_hydrogens"]),
  )
  _cast_floats(batch, dtype)


def _leaves(prefix: str, value: object, out: dict[str, torch.Tensor]) -> None:
  if isinstance(value, torch.Tensor):
    out[prefix] = value.detach()
    return
  scalars = getattr(value, "scalars", None)
  vectors = getattr(value, "vectors", None)
  if isinstance(scalars, torch.Tensor) and isinstance(vectors, torch.Tensor):
    out[prefix + "__scalars"] = scalars.detach()
    out[prefix + "__vectors"] = vectors.detach()
    return
  if isinstance(value, tuple | list):
    for index, item in enumerate(value):
      _leaves(f"{prefix}__{index}", item, out)


def _layer_hooks(model: torch.nn.Module) -> tuple[dict[str, dict[str, torch.Tensor]], list[object]]:
  captured: dict[str, dict[str, torch.Tensor]] = {}
  handles: list[object] = []

  def _make(name: str):  # noqa: ANN202
    def hook(_module: torch.nn.Module, _inputs: object, output: object) -> None:
      if name in captured:
        return
      leaves: dict[str, torch.Tensor] = {}
      _leaves("out", output, leaves)
      captured[name] = leaves

    return hook

  for path, module in model.named_modules():
    if path in LAYER_MODULES:
      handles.append(module.register_forward_hook(_make(path)))
  return captured, handles


def _score_arrays(
  model: torch.nn.Module,
  batch: object,
  order: torch.Tensor,
) -> dict[str, torch.Tensor]:
  sequence = batch.sequence_indices  # type: ignore[attr-defined]
  chi = batch.chi_angles  # type: ignore[attr-defined]
  logits, chi_logits, enc_s, enc_e, lig_s, lig_e, order_lp = model.get_logits_for_score(  # type: ignore[attr-defined]
    batch,
    order,
    sequence,
    chi,
  )
  seq_log = torch.log_softmax(logits, dim=-1)
  gathered = seq_log.gather(-1, sequence.long().unsqueeze(-1)).squeeze(-1)
  basis = model.rotamer_builder.compute_binned_degree_basis_function(chi).nan_to_num()  # type: ignore[attr-defined]
  bins = basis.argmax(dim=-1)
  chi_log = torch.log_softmax(chi_logits, dim=-1)
  picked = chi_log.gather(-1, bins.unsqueeze(-1)).squeeze(-1)
  chi_log_prob = torch.where(chi.isnan(), torch.full_like(picked, torch.nan), picked)
  return {
    "sequence_logits": logits,
    "chi_logits": chi_logits,
    "seq_log_prob": gathered,
    "chi_log_prob": chi_log_prob,
    "chi_bin": bins,
    "encoder_scalars": enc_s,
    "encoder_edges": enc_e,
    "ligand_scalars": lig_s,
    "lig_prot_edges": lig_e,
    "decoding_order_log_probs": order_lp,
  }


def _enable_scalar_dropout(model: torch.nn.Module) -> list[torch.nn.Dropout]:
  touched: list[torch.nn.Dropout] = []
  for module in model.modules():
    if isinstance(module, torch.nn.Dropout) and module.p > 0:
      module.train()
      touched.append(module)
  assert_vector_dropout_eval(model)
  return touched


def _dump_fixture(  # noqa: PLR0915
  buckets: Buckets,
  model: torch.nn.Module,
  params: dict[str, object],
  rbf_cls: object,
  *,
  checkpoint: str,
  fixture: Fixture,
  precision: str,
  rng: np.random.Generator,
  proofread_conditional: bool,
) -> None:
  dtype = _dtype_of(precision)
  prefix = f"{checkpoint}__{fixture.name}__"
  runtime = f64_runtime(rbf_cls) if precision == "f64" else contextlib.nullcontext()
  with runtime, torch.no_grad():
    batch = _featurize(fixture.path, dtype=dtype)
    order_u = rng.random(int(batch.sequence_indices.shape[0]) + 8)  # type: ignore[attr-defined]
    with injected_decoding_order_uniforms(order_u):
      _graphs(model, params, batch, dtype)
      batch.generate_decoding_order(stack_tensors=False)  # type: ignore[attr-defined]
    order = batch.decoding_order.detach().clone()  # type: ignore[attr-defined]
    captured, handles = _layer_hooks(model)
    try:
      lig, prot, pr_e, lig_e = model.apply_encoding_layers(batch)  # type: ignore[attr-defined]
    finally:
      for handle in handles:
        handle.remove()  # type: ignore[attr-defined]
    for name, leaves in captured.items():
      safe = name.replace(".", "_")
      for leaf, tensor in leaves.items():
        _put(buckets, ("laser_layers",), prefix + f"layer__{safe}__{leaf}", tensor)
    _put(buckets, ("laser_encoder",), prefix + "prot_scalars", prot.scalars)
    _put(buckets, ("laser_encoder",), prefix + "prot_vectors", prot.vectors)
    _put(buckets, ("laser_encoder",), prefix + "lig_scalars", lig.scalars)
    _put(buckets, ("laser_encoder",), prefix + "lig_vectors", lig.vectors)
    _put(buckets, ("laser_encoder",), prefix + "pr_pr_eattr", pr_e)
    _put(buckets, ("laser_encoder",), prefix + "lig_pr_eattr", lig_e)
    _put(buckets, ("laser_encoder", "laser_score"), prefix + "pr_pr_idx", batch.pr_pr_edge_index)  # type: ignore[attr-defined]
    _put(buckets, ("laser_encoder",), prefix + "lig_pr_idx", batch.lig_pr_edge_index)  # type: ignore[attr-defined]
    _put(
      buckets,
      ("laser_encoder", "laser_score"),
      prefix + "sequence_indices",
      batch.sequence_indices,
    )  # type: ignore[attr-defined]
    _put(buckets, ("laser_encoder", "laser_rotamers"), prefix + "chi_angles", batch.chi_angles)  # type: ignore[attr-defined]
    _put(buckets, ("laser_encoder",), prefix + "chain_mask", batch.chain_mask)  # type: ignore[attr-defined]
    _put(buckets, ("laser_score",), prefix + "decoding_order", order)
    _put(
      buckets,
      ("laser_score",),
      prefix + "order_uniforms",
      np.asarray(order_u, dtype=np.float64),
    )
    scored = _score_arrays(model, batch, order)
    for key, tensor in scored.items():
      _put(buckets, ("laser_score",), prefix + key, tensor)
    uncond, *_rest = model.forward(batch, return_unconditional_probabilities=True)  # type: ignore[attr-defined]
    _put(buckets, ("laser_score",), prefix + "proofread_uncond_logits", uncond)
    _put(
      buckets,
      ("laser_score",),
      prefix + "first_shell",
      batch.first_shell_ligand_contact_mask,  # type: ignore[attr-defined]
    )
    model_params = params["model_params"]
    if not isinstance(model_params, dict):
      msg = "checkpoint params['model_params'] is not a dict"
      raise TypeError(msg)
    coords = model.rotamer_builder.build_rotamers(  # type: ignore[attr-defined]
      batch.backbone_coords,  # type: ignore[attr-defined]
      batch.chi_angles,  # type: ignore[attr-defined]
      batch.sequence_indices,  # type: ignore[attr-defined]
      add_nonrotatable_hydrogens=bool(model_params["build_hydrogens"]),
    )
    _put(buckets, ("laser_rotamers",), prefix + "sidechain_coords", coords)
    _put(buckets, ("laser_rotamers",), prefix + "backbone_coords", batch.backbone_coords)  # type: ignore[attr-defined]
    step = int(order[0].item()) if order.numel() else 0
    # upstream utils/model.py:770 gathers decoding_order_sort_indices by node id, so the
    # order must span every residue: a 1x1 tensor raises IndexError once any edge points
    # at a node beyond it. One AR step is selected by chain_mask instead (model.py:791 -
    # 1 takes sequence/chi from the input, 0 samples it), and `step` is already the first
    # entry of the generated order, so the captured step decodes with no decoded context.
    batch.decoding_order = order.reshape(1, -1).to(dtype)  # type: ignore[attr-defined]
    # With ignore_chain_mask_zeros=True upstream samples ONLY the True rows of chain_mask
    # (utils/model.py:736) and leaves the rest at the not-decoded sentinel, so exactly one
    # residue draws. Under the default (False) the mask does not filter the loop at all:
    # every residue still calls Categorical.sample and consumes injected uniforms, which is
    # what exhausted the stream. Upstream's inline comment at model.py:754 states the
    # opposite polarity to its own docstring; the docstring matches the code.
    batch.chain_mask = torch.zeros_like(batch.chain_mask, dtype=torch.bool)  # type: ignore[attr-defined]
    batch.chain_mask[step] = True  # type: ignore[attr-defined]
    draws = rng.random((1, N_DECODE_DRAWS))
    with injected_uniform_draws(draws) as cursor:
      sampled = model.sample(  # type: ignore[attr-defined]
        batch,
        sequence_sample_temperature=SEQ_TEMPERATURE,
        chi_angle_sample_temperature=CHI_TEMPERATURE,
        disabled_residues=["X"],
        disable_pbar=True,
        ignore_chain_mask_zeros=True,
      )
    # Under ignore_chain_mask_zeros upstream overwrites every unsampled residue with X
    # (utils/model.py:937), and disabled_residues=["X"] stops the sampled one from being X.
    # So "exactly one AR step" means: X everywhere except `step`, and not X at `step`.
    # Without this, an inverted chain_mask would decode the whole chain and still write a
    # plausible-looking oracle.
    from LASErMPNN.utils.constants import aa_short_to_idx  # noqa: PLC0415

    x_index = aa_short_to_idx["X"]
    indices = sampled.sampled_sequence_indices
    others = (torch.arange(indices.shape[0], device=indices.device) != step)
    if not bool((indices[others] == x_index).all()):
      sampled_elsewhere = (indices[others] != x_index).sum().item()
      msg = (
        f"laser_decode_step expected residue {step} alone to be sampled; "
        f"{sampled_elsewhere} other residues are not X"
      )
      raise RuntimeError(msg)
    if int(indices[step]) == x_index:
      msg = f"laser_decode_step residue {step} came back as X despite disabled_residues"
      raise RuntimeError(msg)
    _put(buckets, ("laser_decode_step",), prefix + "step_index", np.asarray(step))
    _put(buckets, ("laser_decode_step",), prefix + "uniforms", draws.astype(np.float64))
    _put(
      buckets,
      ("laser_decode_step",),
      prefix + "uniforms_consumed",
      np.asarray(cursor.consumed_steps),
    )
    _put(buckets, ("laser_decode_step",), prefix + "sequence_logits", sampled.sequence_logits[step])
    _put(
      buckets,
      ("laser_decode_step",),
      prefix + "sequence_index",
      sampled.sampled_sequence_indices[step],
    )
    _put(buckets, ("laser_decode_step",), prefix + "chi_logits", sampled.chi_logits[step])
    _put(buckets, ("laser_decode_step",), prefix + "chi_degrees", sampled.sampled_chi_degrees[step])
    if proofread_conditional and fixture.name == "4jnj-1_prot":
      _proofread_one_focus(buckets, model, batch, prefix=prefix, rng=rng, dtype=dtype)
  logger.info("dumped %s %s %s", checkpoint, precision, fixture.name)


def _proofread_one_focus(
  buckets: Buckets,
  model: torch.nn.Module,
  batch: object,
  *,
  prefix: str,
  rng: np.random.Generator,
  dtype: torch.dtype,
) -> None:
  """Two decoding orders, one dropout rep, one focus residue. Vector dropout stays off."""
  shell = batch.first_shell_ligand_contact_mask  # type: ignore[attr-defined]
  focus = int(shell.nonzero()[0].item()) if bool(shell.any()) else 0
  probs: list[torch.Tensor] = []
  mask_blobs: list[np.ndarray] = []
  for copy in range(2):
    order = torch.arange(batch.sequence_indices.shape[0], dtype=dtype)  # type: ignore[attr-defined]
    if copy == 1:
      order = torch.flip(order, dims=(0,))
    batch.decoding_order = order.view(1, -1)  # type: ignore[attr-defined]
    batch.chain_mask = torch.ones_like(batch.chain_mask)  # type: ignore[attr-defined]
    batch.chain_mask[focus] = False  # type: ignore[attr-defined]
    draws = rng.random((1, int(order.shape[0]) * 5 + 4))
    touched = _enable_scalar_dropout(model)
    try:
      with injected_scalar_dropout(model, rng=rng) as dropout, injected_uniform_draws(draws):
        sampled = model.sample(  # type: ignore[attr-defined]
          batch,
          sequence_sample_temperature=1.0,
          chi_angle_sample_temperature=1.0,
          disabled_residues=["X"],
          disable_pbar=True,
          repack_all=True,
        )
        for path, rows in dropout.recorded.items():
          for index, row in enumerate(rows):
            mask_blobs.append(np.asarray(row))
            _put(
              buckets,
              ("laser_score",),
              prefix + f"proofread_mask__{copy}__{path.replace('.', '_')}__{index}",
              row,
            )
    finally:
      for module in touched:
        module.eval()
    probs.append(torch.softmax(sampled.sequence_logits[focus], dim=-1))
  stacked = torch.stack(probs, dim=0)
  _put(buckets, ("laser_score",), prefix + "proofread_focus", np.asarray(focus))
  _put(buckets, ("laser_score",), prefix + "proofread_conditional_p", stacked)
  _put(
    buckets,
    ("laser_score",),
    prefix + "proofread_n_masks",
    np.asarray(len(mask_blobs)),
  )


def _save_npz(path: Path, payload: dict[str, np.ndarray]) -> str:
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(path, **payload)
  return _sha256_file(path)


def _write_manifest(
  path: Path,
  *,
  commit: str,
  checkpoints: list[dict[str, object]],
  dumps: list[dict[str, object]],
) -> None:
  lines = [
    f"upstream_commit = {_toml_str(commit)}",
    f"torch_version = {_toml_str(torch.__version__)}",
    f"seed = {SEED}",
    f"shim_sha256 = {_toml_str(shim_sha256())}",
    f"shim_sites = {_toml_list(SHIM_SITES)}",
    "shimmed = true",
    "",
  ]
  for row in checkpoints:
    lines.append("[[checkpoint]]")
    lines.append(f"id = {_toml_str(str(row['id']))}")
    lines.append(f"path = {_toml_str(str(row['path']))}")
    lines.append(f"weights_sha256 = {_toml_str(str(row['weights_sha256']))}")
    lines.append(f"missing_keys = {_toml_list(row['missing_keys'])}")  # type: ignore[arg-type]
    lines.append(f"unexpected_keys = {_toml_list(row['unexpected_keys'])}")  # type: ignore[arg-type]
    lines.append("")
  for row in dumps:
    lines.append("[[dump]]")
    lines.append(f"wave = {_toml_str(str(row['wave']))}")
    lines.append(f"precision = {_toml_str(str(row['precision']))}")
    lines.append(f"path = {_toml_str(str(row['path']))}")
    lines.append(f"sha256 = {_toml_str(str(row['sha256']))}")
    lines.append(f"shimmed = {'true' if row['shimmed'] else 'false'}")
    lines.append("")
  path.write_text("\n".join(lines))


def _check_literal_factory_shim() -> None:
  """The tensor factories are restored, and an unlisted site keeps float32."""
  before = {name: getattr(torch, name) for name in LITERAL_FACTORIES}
  with float32_literals_follow_active_dtype():
    # This call site is not in _FLOAT32_LITERAL_SITES, so the literal must survive.
    if torch.zeros(1, dtype=torch.float32).dtype is not torch.float32:
      msg = "float32 literal shim fired at an unlisted call site"
      raise RuntimeError(msg)
  for name, original in before.items():
    if getattr(torch, name) is not original:
      msg = f"torch.{name} was not restored"
      raise RuntimeError(msg)


def _check_ideal_coords_shim() -> None:
  """Positive and negative control for ``ideal_coords_follow_backbone_dtype``.

  Negative: with no upstream module imported the shim must raise rather than
  silently patch nothing. Positive: once imported, the constant is float64
  inside the block, and the original object is restored on exit.
  """
  modules = ideal_coord_modules()
  if not modules:
    try:
      with ideal_coords_follow_backbone_dtype():
        pass
    except RuntimeError:
      logger.info("ideal-coords shim raises when upstream is not imported (negative control)")
      return
    msg = "ideal_coords_follow_backbone_dtype silently patched nothing"
    raise RuntimeError(msg)
  before = [getattr(module, "ideal_prot_aa_coords") for module in modules]  # noqa: B009
  with ideal_coords_follow_backbone_dtype():
    for module in modules:
      if getattr(module, "ideal_prot_aa_coords").dtype is not torch.float64:  # noqa: B009
        msg = f"ideal_prot_aa_coords in {module} was not widened to float64 inside the shim"
        raise RuntimeError(msg)
  for module, original in zip(modules, before, strict=True):
    if getattr(module, "ideal_prot_aa_coords") is not original:  # noqa: B009
      msg = f"ideal_prot_aa_coords in {module} was not restored"
      raise RuntimeError(msg)
  logger.info("ideal-coords shim widened and restored %d module(s)", len(modules))


def _selftest(package: Path, fixtures_dir: Path) -> None:  # noqa: PLR0915
  """Shim edges, restoration, and an unshimmed encoder repeat when 4jnj is present."""
  assert_inverse_cdf_edges()
  before_sample = torch.distributions.Categorical.sample
  before_rand = torch.rand
  before_drop = torch.nn.Dropout.forward
  before_float = torch.Tensor.float
  before_default = torch.get_default_dtype()
  with injected_uniform_draws(np.full((1, 4), 0.5, dtype=np.float64)):
    pass
  with injected_decoding_order_uniforms(np.zeros(4, dtype=np.float64)):
    pass
  tiny = torch.nn.Linear(2, 2)
  with injected_scalar_dropout(tiny, {"weight": []}):
    pass
  if torch.distributions.Categorical.sample is not before_sample:
    msg = "Categorical.sample was not restored"
    raise RuntimeError(msg)
  if torch.rand is not before_rand:
    msg = "torch.rand was not restored"
    raise RuntimeError(msg)
  if torch.nn.Dropout.forward is not before_drop:
    msg = "nn.Dropout.forward was not restored"
    raise RuntimeError(msg)
  if torch.Tensor.float is not before_float:
    msg = "Tensor.float was not restored"
    raise RuntimeError(msg)
  if torch.get_default_dtype() != before_default:
    msg = "default dtype was not restored"
    raise RuntimeError(msg)
  if len(shim_sha256()) != 64:
    msg = "shim_sha256 is not a sha256 hex digest"
    raise RuntimeError(msg)
  _check_ideal_coords_shim()
  _check_literal_factory_shim()
  example = package / "example_pdbs" / "4jnj-1_prot.pdb"
  weights = package / CHECKPOINTS["nothing_heldout"]
  if not example.is_file() or not weights.is_file():
    logger.info("selftest skipped the encoder repeat (missing 4jnj or weights)")
    return
  pin_commit, pin_weights = _load_pin(package)
  _upstream_commit(package, pin_commit)
  rel = CHECKPOINTS["nothing_heldout"]
  filename = Path(rel).name
  digest = _sha256_file(weights)
  pinned = pin_weights.get(filename)
  if pinned is not None and digest != pinned:
    msg = f"sha256 mismatch for {filename}"
    raise SystemExit(msg)
  try:
    model, params, _missing, _unexpected = _load_model(weights, "f32")
  except ImportError as exc:
    logger.info("selftest passed (shim); encoder repeat skipped (%s)", exc)
    return

  def once() -> torch.Tensor:
    torch.manual_seed(SEED)
    batch = _featurize(example, dtype=torch.float32)
    _graphs(model, params, batch, torch.float32)
    _lig, prot, _pr, _lp = model.apply_encoding_layers(batch)  # type: ignore[attr-defined]
    return prot.scalars

  first = once()
  second = once()
  if not torch.equal(first, second):
    msg = "unshimmed encoder is not bit-identical across seeded repeats"
    raise RuntimeError(msg)
  if fixtures_dir.is_dir():
    logger.info("selftest saw fixtures dir %s", fixtures_dir)
  logger.info("selftest passed")


def _run_dumps(
  args: argparse.Namespace,
  package: Path,
  commit: str,
  pin_weights: dict[str, str],
) -> None:
  if args.out is None:
    msg = "--out is required unless --selftest is set"
    raise SystemExit(msg)
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  repo = Path(__file__).resolve().parents[2]
  fixtures = _collect_fixtures(package, Path(args.fixtures_dir), repo)
  from LASErMPNN.utils.model import RBF_Encoding  # noqa: PLC0415

  checkpoint_rows: list[dict[str, object]] = []
  dump_rows: list[dict[str, object]] = []
  for precision in args.precisions:
    buckets = Buckets(precision=precision)
    rng = np.random.default_rng(SEED)
    for checkpoint_id in args.checkpoints:
      if checkpoint_id not in CHECKPOINTS:
        msg = f"unknown checkpoint id {checkpoint_id}"
        raise SystemExit(msg)
      rel = CHECKPOINTS[checkpoint_id]
      path = package / rel
      digest = _sha256_file(path)
      pinned = pin_weights[Path(rel).name]
      if digest != pinned:
        msg = f"sha256 mismatch for {rel}: file {digest} pin {pinned}"
        raise SystemExit(msg)
      model, params, missing, unexpected = _load_model(path, precision)
      checkpoint_rows.append(
        {
          "id": f"{checkpoint_id}_{precision}",
          "path": rel,
          "weights_sha256": digest,
          "missing_keys": missing,
          "unexpected_keys": unexpected,
        },
      )
      for fixture in fixtures:
        _dump_fixture(
          buckets,
          model,
          params,
          RBF_Encoding,
          checkpoint=checkpoint_id,
          fixture=fixture,
          precision=precision,
          rng=rng,
          proofread_conditional=bool(args.proofread_conditional),
        )
    names = np.asarray([fixture.name for fixture in fixtures])
    ids = np.asarray(list(args.checkpoints))
    for wave in WAVES:
      payload = buckets.waves[wave]
      payload["precision"] = np.asarray(precision)
      payload["seed"] = np.asarray(SEED)
      payload["fixture_names"] = names
      payload["checkpoint_ids"] = ids
      dest = out / wave / f"oracle_{precision}.npz"
      digest = _save_npz(dest, payload)
      dump_rows.append(
        {
          "wave": wave,
          "precision": precision,
          "path": str(dest.relative_to(out)),
          "sha256": digest,
          "shimmed": wave in SHIMMED_WAVES,
        },
      )
      logger.info("wrote %s (%s)", dest, digest)
  _write_manifest(
    out / "oracle_manifest.toml",
    commit=commit,
    checkpoints=checkpoint_rows,
    dumps=dump_rows,
  )


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--laser-root", type=Path, required=True)
  parser.add_argument("--fixtures-dir", type=Path)
  parser.add_argument("--out", type=Path)
  parser.add_argument("--checkpoints", nargs="*", default=list(CHECKPOINTS))
  parser.add_argument("--precisions", nargs="+", default=["f64", "f32"], choices=["f64", "f32"])
  parser.add_argument("--selftest", action="store_true")
  parser.add_argument("--proofread-conditional", action="store_true")
  args = parser.parse_args()
  package = _package_dir(args.laser_root)
  # The pinned checkout directory is named LASErMPNN, so its parent is the import root.
  sys.path.insert(0, str(package.parent))
  if package.name != "LASErMPNN":
    msg = f"{package} must be a directory named LASErMPNN so `import LASErMPNN` resolves"
    raise SystemExit(msg)
  pin_commit, pin_weights = _load_pin(package)
  commit = _upstream_commit(package, pin_commit)
  if args.selftest:
    fixtures = args.fixtures_dir
    _selftest(package, Path(fixtures) if fixtures is not None else package)
    return
  if args.fixtures_dir is None:
    msg = "--fixtures-dir is required unless --selftest is set"
    raise SystemExit(msg)
  _run_dumps(args, package, commit, pin_weights)


if __name__ == "__main__":
  main()
