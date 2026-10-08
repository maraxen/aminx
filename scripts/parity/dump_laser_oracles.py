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

``--only order`` writes ``<out>/laser_order/order_f64.npz`` and appends one new
``[[dump]]`` sha entry. It does not regenerate any existing npz or rewrite any
existing sha line. ``torch.rand`` inside ``_masked_sort_for_decoding_order`` is
the shim in ``oracle_shims/laser.py``. Orders are integers, so f64 is enough.

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
# Multi-threaded float reductions in the pairwise distances flip near-tied neighbours between
# processes, so pr_pr_idx / lig_pr_idx (saved below, and compared exactly by aminx's tests) are not
# reproducible at default threads. Measured on ProtonPotts: 9/12 processes flipped at 10 threads,
# 0/12 at 1. aminx debt #2584; spec 261007_protonpottsmpnn-support §32.
TORCH_NUM_THREADS = 1
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
  """Detach to a numpy array that OWNS its memory.

  ``.cpu()`` is a no-op for a CPU tensor and ``.numpy()`` shares storage with it,
  so without the copy ``_put`` would bank a view. The npz is only written after
  every wave has run on the same batch and model, so any later in-place op on
  that storage would retroactively edit an already-"recorded" oracle value.
  """
  return value.detach().cpu().numpy().copy()


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


def _leaves(
  prefix: str,
  value: object,
  out: dict[str, torch.Tensor],
  *,
  clone: bool = False,
) -> None:
  def _keep(tensor: torch.Tensor) -> torch.Tensor:
    detached = tensor.detach()
    return detached.clone() if clone else detached

  if isinstance(value, torch.Tensor):
    out[prefix] = _keep(value)
    return
  scalars = getattr(value, "scalars", None)
  vectors = getattr(value, "vectors", None)
  if isinstance(scalars, torch.Tensor) and isinstance(vectors, torch.Tensor):
    out[prefix + "__scalars"] = _keep(scalars)
    out[prefix + "__vectors"] = _keep(vectors)
    return
  if isinstance(value, tuple | list):
    for index, item in enumerate(value):
      _leaves(f"{prefix}__{index}", item, out, clone=clone)


def _layer_hooks(model: torch.nn.Module) -> tuple[dict[str, dict[str, torch.Tensor]], list[object]]:
  """Capture each layer's first call: inputs BEFORE it runs, outputs after.

  The split matters. ``HomoGATv2.forward`` and ``HeteroGATv2.forward`` assign
  ``nodes.scalars = self.final_atten_aggr(...)`` onto the ``EquivariantData``
  they were *passed*, so a post-forward hook reading ``inputs`` sees the mutated
  object and records a mid-layer intermediate under ``in__``. Replaying a layer
  from that is not a parity test of anything. Measured before this split: the
  ligand GAT's ``in__0__scalars`` differed from the preceding GVP's output by
  6.46, and ``EquivariantLayerNorm(GVP(in__))`` reproduced ``out__`` to 5.8e-15
  -- i.e. ``in__`` was the post-attention intermediate. Inputs are cloned so a
  later genuine in-place tensor op cannot rewrite what we already captured.
  """
  captured: dict[str, dict[str, torch.Tensor]] = {}
  seen_in: set[str] = set()
  seen_out: set[str] = set()
  handles: list[object] = []

  def _make_pre(name: str):  # noqa: ANN202
    def pre_hook(_module: torch.nn.Module, inputs: object) -> None:
      if name in seen_in:
        return
      seen_in.add(name)
      _leaves("in", inputs, captured.setdefault(name, {}), clone=True)

    return pre_hook

  def _make_post(name: str):  # noqa: ANN202
    def hook(_module: torch.nn.Module, _inputs: object, output: object) -> None:
      if name in seen_out:
        return
      seen_out.add(name)
      _leaves("out", output, captured.setdefault(name, {}))

    return hook

  for path, module in model.named_modules():
    if path in LAYER_MODULES:
      handles.append(module.register_forward_pre_hook(_make_pre(path)))
      handles.append(module.register_forward_hook(_make_post(path)))
  return captured, handles


def _check_layer_hook_inputs() -> None:
  """Negative control: the capture must survive a layer that mutates its input.

  A module whose forward overwrites an attribute of its argument is exactly the
  upstream GAT shape. Asserting only that ``in__`` exists would pass on the old
  post-forward hook too, so this asserts the captured value is the ORIGINAL --
  the check fails on the implementation it is meant to reject.
  """

  class _Holder:
    def __init__(self, scalars: torch.Tensor, vectors: torch.Tensor) -> None:
      self.scalars = scalars
      self.vectors = vectors

  class _Mutator(torch.nn.Module):
    def forward(self, held: _Holder) -> _Holder:  # noqa: D102
      held.scalars = held.scalars + 100.0  # upstream's `nodes.scalars = ...`
      return held

  original = torch.zeros(2, 3)
  held = _Holder(original.clone(), torch.zeros(2, 1, 3))
  model = _Mutator()
  global LAYER_MODULES  # noqa: PLW0603
  previous = LAYER_MODULES
  LAYER_MODULES = frozenset({""})
  try:
    captured, handles = _layer_hooks(model)
    try:
      model(held)
    finally:
      for handle in handles:
        handle.remove()  # type: ignore[attr-defined]
  finally:
    LAYER_MODULES = previous
  leaves = captured.get("", {})
  got = leaves.get("in__0__scalars")
  if got is None:
    msg = "_layer_hooks captured no in__0__scalars for the mutating control"
    raise SystemExit(msg)
  if not torch.equal(got, original):
    msg = (
      "_layer_hooks recorded the MUTATED input: "
      f"max|captured - original| = {(got - original).abs().max().item()}. "
      "Inputs must be captured in a forward PRE-hook."
    )
    raise SystemExit(msg)
  logger.info("selftest: layer-hook inputs are pre-mutation")


def _check_put_owns_memory() -> None:
  """Negative control: a banked oracle value must survive in-place mutation.

  Every wave runs on the same batch and model before the npz is written, so a
  value stored as a view into torch storage can be edited after the fact. This
  mutates the source tensor after ``_put`` and asserts the banked array is
  unchanged. Asserting only that the key exists would pass on the aliasing
  implementation too.
  """
  buckets = Buckets(precision="f64")
  tensor = torch.zeros(4)
  _put(buckets, ("laser_encoder",), "control__owns_memory", tensor)
  banked = buckets.waves["laser_encoder"]["control__owns_memory"]
  before = banked.copy()
  tensor.add_(7.0)  # exactly what a later wave's in-place op would do
  if not np.array_equal(banked, before):
    msg = (
      "_put banked a VIEW into torch storage: an in-place mutation after capture "
      f"changed the recorded value by {np.abs(banked - before).max()}. _np must copy."
    )
    raise SystemExit(msg)
  logger.info("selftest: _put banks memory it owns")


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
          identities = dropout.edges.get(path, [])
          for index, row in enumerate(rows):
            mask_blobs.append(np.asarray(row))
            _put(
              buckets,
              ("laser_score",),
              prefix + f"proofread_mask__{copy}__{path.replace('.', '_')}__{index}",
              row,
            )
            identity = identities[index] if index < len(identities) else None
            if identity is not None:
              _put(
                buckets,
                ("laser_score",),
                prefix + f"proofread_edge__{copy}__{path.replace('.', '_')}__{index}",
                np.asarray(identity, dtype=np.int64),
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
    f"torch_num_threads = {torch.get_num_threads()}",
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
  _check_layer_hook_inputs()
  _check_put_owns_memory()
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


def _append_dump_row(path: Path, row: dict[str, object]) -> None:
  """Append one ``[[dump]]`` table. An existing path keeps its recorded sha."""
  rel = str(row["path"])
  digest = str(row["sha256"])
  text = path.read_text(encoding="utf-8") if path.is_file() else ""
  needle = f"path = {_toml_str(rel)}"
  if needle in text:
    if f"sha256 = {_toml_str(digest)}" not in text:
      msg = f"refusing to modify existing manifest sha for {rel}"
      raise SystemExit(msg)
    return
  block = "\n".join(
    [
      "[[dump]]",
      f"wave = {_toml_str(str(row['wave']))}",
      f"precision = {_toml_str(str(row['precision']))}",
      f"path = {_toml_str(rel)}",
      f"sha256 = {_toml_str(digest)}",
      f"shimmed = {'true' if row['shimmed'] else 'false'}",
      "",
    ],
  )
  prefix = "" if text.endswith("\n") or text == "" else "\n"
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as handle:
    handle.write(prefix + block)


def _tier_stream(chain_mask: np.ndarray, contact: np.ndarray) -> np.ndarray:
  """Tier-0, tier-1, tier-2 uniforms in ascending row index. Values sit in (0, 1).

  Fixed rows sit near 0.9 and designable rows near 0.05, so dropping the tier
  offset reverses them. Contact rows sit near 0.4, between those two.
  """
  fixed = np.asarray(chain_mask, dtype=bool)
  designable = ~fixed
  touched = np.asarray(contact, dtype=bool) & designable
  plain = designable & ~touched
  parts: list[np.ndarray] = []
  for mask, base in ((fixed, 0.90), (plain, 0.05), (touched, 0.40)):
    count = int(mask.sum())
    if count == 0:
      continue
    parts.append(base + np.linspace(0.0, 0.04, count, dtype=np.float64))
  if not parts:
    return np.zeros(0, dtype=np.float64)
  return np.concatenate(parts)


def _scatter_stream(stream: np.ndarray, chain_mask: np.ndarray, contact: np.ndarray) -> np.ndarray:
  """Map the concatenated tier stream back onto rows (ascending index within a tier)."""
  fixed = np.asarray(chain_mask, dtype=bool)
  designable = ~fixed
  touched = np.asarray(contact, dtype=bool) & designable
  plain = designable & ~touched
  out = np.zeros(fixed.shape[0], dtype=np.float64)
  offset = 0
  for mask in (fixed, plain, touched):
    index = np.flatnonzero(mask)
    count = int(index.size)
    out[index] = stream[offset : offset + count]
    offset += count
  if offset != int(stream.shape[0]):
    msg = f"tier stream length {stream.shape[0]} != consumed {offset}"
    raise RuntimeError(msg)
  return out


def _residue_mask(value: np.ndarray, *, name: str) -> np.ndarray:
  """Collapse a leading or trailing batch of 1 so masks are ``(L,)``."""
  array = np.squeeze(np.asarray(value))
  if array.ndim != 1:
    msg = f"{name} has shape {tuple(np.asarray(value).shape)}, expected a residue vector"
    raise RuntimeError(msg)
  return np.asarray(array, dtype=bool)


def _assign_mask(batch: object, name: str, value: np.ndarray) -> None:
  current = getattr(batch, name)
  flat = np.asarray(value, dtype=bool).reshape(-1)
  if flat.size != int(current.numel()):
    msg = f"{name} has {int(current.numel())} entries, got {flat.size}"
    raise RuntimeError(msg)
  tensor = torch.as_tensor(flat, device=current.device)
  tensor = tensor.to(dtype=current.dtype).reshape(tuple(current.shape))
  setattr(batch, name, tensor)


def _decoding_order_from_batch(
  batch: object,
  chain_mask: np.ndarray,
  contact: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  stream = _tier_stream(chain_mask, contact)
  _assign_mask(batch, "chain_mask", chain_mask)
  _assign_mask(batch, "extra_atom_contact_mask", contact)
  with injected_decoding_order_uniforms(stream) as cursor:
    generate = getattr(batch, "generate_decoding_order")
    generate(stack_tensors=False)
  if cursor.offset != int(stream.shape[0]):
    msg = f"decoding-order shim consumed {cursor.offset} of {stream.shape[0]} uniforms"
    raise RuntimeError(msg)
  order_tensor = getattr(batch, "decoding_order")
  order = order_tensor.detach().cpu().to(dtype=torch.int64).reshape(-1).numpy()
  row_u = _scatter_stream(stream, chain_mask, contact)
  tier = np.where(chain_mask.astype(bool), 0.0, np.where(contact.astype(bool), 2.0, 1.0))
  reconstructed = np.argsort(row_u + tier, kind="stable")
  if not np.array_equal(order, reconstructed):
    msg = "upstream decoding order disagrees with argsort(u + tier) on the injected stream"
    raise RuntimeError(msg)
  if np.array_equal(order, np.argsort(row_u, kind="stable")):
    msg = "dropping the tier offset leaves the order unchanged; the control would not fire"
    raise RuntimeError(msg)
  return order, stream


def _mixed_fixed(chain_mask: np.ndarray) -> np.ndarray:
  """Keep a featurizer mask that already mixes tiers. Otherwise fix a prefix."""
  mask = np.asarray(chain_mask, dtype=bool).copy()
  if bool(mask.any()) and bool((~mask).any()):
    return mask
  count = max(1, int(mask.shape[0]) // 5)
  mask[:] = False
  mask[:count] = True
  return mask


def _run_order_dumps(
  args: argparse.Namespace,
  package: Path,
  commit: str,
  pin_weights: dict[str, str],
) -> None:
  """Inference order (tier 2 empty) and a hand-set 3-tier order. New files only."""
  if args.out is None:
    msg = "--out is required"
    raise SystemExit(msg)
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  rel = CHECKPOINTS["nothing_heldout"]
  weights = package / rel
  digest = _sha256_file(weights)
  pinned = pin_weights[Path(rel).name]
  if digest != pinned:
    msg = f"sha256 mismatch for {rel}: file {digest} pin {pinned}"
    raise SystemExit(msg)
  example = package / "example_pdbs" / "4jnj-1_prot.pdb"
  if not example.is_file():
    msg = f"missing upstream example pdb {example}"
    raise SystemExit(msg)
  pdb_sha = _sha256_file(example)
  batch = _featurize(example, dtype=torch.float64)
  base_chain = _residue_mask(getattr(batch, "chain_mask").detach().cpu().numpy(), name="chain_mask")
  base_contact = _residue_mask(
    getattr(batch, "extra_atom_contact_mask").detach().cpu().numpy(),
    name="extra_atom_contact_mask",
  )
  if bool(base_contact.any()):
    msg = "inference featurizer produced a non-empty extra_atom_contact_mask"
    raise RuntimeError(msg)
  length = int(base_chain.shape[0])
  if length < 6:
    msg = f"inference fixture length {length} is too short for three tiers"
    raise RuntimeError(msg)
  infer_chain = _mixed_fixed(np.asarray(base_chain, dtype=bool))
  infer_contact = np.zeros(length, dtype=bool)
  with torch.no_grad():
    infer_order, infer_stream = _decoding_order_from_batch(batch, infer_chain, infer_contact)
    hand_chain = np.zeros(length, dtype=bool)
    hand_chain[:2] = True
    hand_contact = np.zeros(length, dtype=bool)
    hand_contact[2:4] = True
    hand_order, hand_stream = _decoding_order_from_batch(batch, hand_chain, hand_contact)
  payload = {
    "precision": np.asarray("f64"),
    "upstream_commit": np.asarray(commit),
    "checkpoint_id": np.asarray("nothing_heldout"),
    "weights_sha256": np.asarray(digest),
    "fixture_name": np.asarray("4jnj-1_prot"),
    "pdb_sha256": np.asarray(pdb_sha),
    "infer_chain_mask": infer_chain,
    "infer_contact": infer_contact,
    "infer_uniforms": infer_stream,
    "infer_order": infer_order,
    "hand_chain_mask": hand_chain,
    "hand_contact": hand_contact,
    "hand_uniforms": hand_stream,
    "hand_order": hand_order,
  }
  dest = out / "laser_order" / "order_f64.npz"
  file_digest = _save_npz(dest, payload)
  relative = "laser_order/order_f64.npz"
  sidecar = dest.with_suffix(".npz.sha256")
  if sidecar.is_file() and sidecar.read_text(encoding="utf-8").strip() != file_digest:
    msg = f"refusing to modify existing sha entry {sidecar}"
    raise SystemExit(msg)
  if not sidecar.is_file():
    sidecar.write_text(file_digest + "\n", encoding="utf-8")
  _append_dump_row(
    out / "oracle_manifest.toml",
    {
      "wave": "laser_order",
      "precision": "f64",
      "path": relative,
      "sha256": file_digest,
      "shimmed": True,
    },
  )
  logger.info("wrote %s (%s)", dest, file_digest)


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
  parser.add_argument(
    "--only",
    choices=("order",),
    help=(
      "Write laser_order/order_f64.npz and a new sha entry, and do not regenerate existing dumps."
    ),
  )
  args = parser.parse_args()
  torch.set_num_threads(TORCH_NUM_THREADS)
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
  if args.only == "order":
    _run_order_dumps(args, package, commit, pin_weights)
    return
  if args.fixtures_dir is None:
    msg = "--fixtures-dir is required unless --selftest or --only order is set"
    raise SystemExit(msg)
  _run_dumps(args, package, commit, pin_weights)


if __name__ == "__main__":
  main()
