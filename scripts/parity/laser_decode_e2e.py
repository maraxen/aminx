"""Graded LASEr joint-decode parity against upstream ``model.sample`` (do not run from the fixer).

Clean arm: the same 21 fixtures as ``laser_score_parity``, an injected decoding
order, and argmax for sequence and χ. Pass requires exact sequence agreement,
χ-bin agreement == 1.0, and max circular |delta chi_deg| <= 1e-6 degrees (f64)
on chi_mask. Anything else is fail. There is no inconclusive band.

Negative controls, each of which must land in fail:
  reversed decoding order -> agreement < 1.0
  aminx χ bin + 1 mod num_chi_bins -> fail
  χ offset := 0 -> fail
  tied χ2 := χ1 -> fail
  tied λ mixed on logits instead of probabilities -> fail

The tied fixture is one complex copied with a coordinate shift, an injected
order, and injected uniforms. χ streams are (sample, step, χ, structure).

The clean arm and each mutant arm are separate subprocesses. Results go to
``$BTH_RESULTS_PATH``.
"""

from __future__ import annotations

import os
import pathlib

# x64 before any JAX import. Parent and children share this so f64 parity and
# the injected order see the same dtype on both sides.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

MUTANTS = (
  "reversed_order",
  "chi_bin_plus_one",
  "offset_zero",
  "chi2_equals_chi1",
  "lambda_on_logits",
)
SEED = 42
PASS_DEG = 1e-6
CHECKPOINT_NAME = "laser_weights_0p1A_nothing_heldout.pt"
CHECKPOINT_SHA256 = "304fe02a4807c310bdd9d68c988ae87619da3cf2025d5c223fb31030aa411173"
EXAMPLE_SHA256 = "0a933729c4915cfee14ff34408a57e1d65fd98845df189ba79ad4f4d8470ae7e"
EXAMPLE_NAME = "4jnj-1_prot.pdb"


def _repo() -> Path:
  return Path(__file__).resolve().parents[2]


def _default_laser_root() -> Path:
  return Path(os.environ.get("AMINX_LASER_ROOT", "~/repos/LASErMPNN")).expanduser()


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _parse(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--laser-root", type=Path, default=_default_laser_root())
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--oracle-python", default=os.environ.get("LASER_ORACLE_PYTHON"))
  parser.add_argument("--mutants", default=",".join(MUTANTS))
  parser.add_argument("--controls-out", type=Path, default=None)
  parser.add_argument("--arm", default=None)
  parser.add_argument("--payload-out", type=Path, default=None)
  parser.add_argument("--job", type=Path, default=None)
  parser.add_argument("--upstream", type=Path, default=None)
  parser.add_argument("--work-dir", type=Path, default=None)
  parser.add_argument("--arm-timeout", type=float, default=86400.0)
  parser.add_argument("--oracle-worker", action="store_true")
  return parser.parse_args(argv)


def _checkpoint(args: argparse.Namespace) -> Path:
  if args.checkpoint is not None:
    return args.checkpoint
  return args.laser_root / "model_weights" / CHECKPOINT_NAME


def _manifest() -> dict[str, Any]:
  path = _repo() / "tests" / "fixtures" / "laser" / "fixtures_manifest.toml"
  return tomllib.loads(path.read_text(encoding="utf-8"))


def _fixture_paths(root: Path) -> list[Path]:
  manifest = _manifest()
  folder = _repo() / "tests" / "fixtures" / "laser"
  paths = [root / "example_pdbs" / EXAMPLE_NAME]
  paths.extend(folder / f"{stem}.pdb" for stem in manifest["ordered_ids"])
  return paths


def _check_inputs(root: Path, checkpoint: Path) -> None:
  example = root / "example_pdbs" / EXAMPLE_NAME
  got = _sha256(example)
  if got != EXAMPLE_SHA256:
    msg = f"{example} sha256 {got} != {EXAMPLE_SHA256}"
    raise SystemExit(msg)
  manifest = _manifest()
  folder = _repo() / "tests" / "fixtures" / "laser"
  for name, digest in manifest["files"].items():
    path = folder / name
    got = _sha256(path)
    if got != digest:
      msg = f"{path} sha256 {got} != {digest}"
      raise SystemExit(msg)
  got = _sha256(checkpoint)
  if got != CHECKPOINT_SHA256:
    msg = f"{checkpoint} sha256 {got} != {CHECKPOINT_SHA256}"
    raise SystemExit(msg)


def _results_path() -> Path:
  env = os.environ.get("BTH_RESULTS_PATH")
  if not env:
    msg = "laser_decode_e2e requires $BTH_RESULTS_PATH"
    raise SystemExit(msg)
  return Path(env)


def _band(exact_seq: bool, chi_agree: float, max_delta: float) -> str:
  # Pre-registered at spec §7.3. There is no inconclusive band.
  if exact_seq and chi_agree == 1.0 and max_delta <= PASS_DEG:
    return "pass"
  return "fail"


def _install_mutant(arm: str | None, joint: Any, decoder: Any) -> tuple[Any, Any]:  # noqa: ANN401
  if arm in (None, "clean", "reversed_order", "chi2_equals_chi1", "lambda_on_logits"):
    return joint, decoder
  import equinox as eqx
  import jax
  import jax.numpy as jnp

  from aminx.model.laser.layers import DenseMLP

  if arm == "offset_zero":
    # A zero offset head makes chi_deg the bin centre. Upstream offsets are not
    # zero, so the circular gap leaves the 1e-6 degree band.
    layers = []
    for mlp in joint.chi_offset_prediction_layers:
      updated: list[Any] = []
      for layer in mlp.layers:
        if not isinstance(layer, eqx.nn.Linear):
          updated.append(layer)
          continue
        linear = eqx.tree_at(lambda item: item.weight, layer, jnp.zeros_like(layer.weight))
        if linear.bias is not None:
          linear = eqx.tree_at(lambda item: item.bias, linear, jnp.zeros_like(linear.bias))
        updated.append(linear)
      layers.append(eqx.tree_at(lambda item: item.layers, mlp, tuple(updated)))
    return eqx.tree_at(lambda item: item.chi_offset_prediction_layers, joint, tuple(layers)), decoder
  if arm == "chi_bin_plus_one":

    class _ShiftedChi(eqx.Module):
      """Roll χ logits so the argmax bin is +1 mod num_chi_bins."""

      inner: DenseMLP

      def __call__(self, features: jax.Array) -> jax.Array:
        return jnp.roll(self.inner(features), 1, axis=-1)

    shifted = tuple(_ShiftedChi(inner=layer) for layer in decoder.chi_prediction_layers)
    decoder = eqx.tree_at(lambda item: item.chi_prediction_layers, decoder, shifted)
    return joint, decoder
  msg = f"unknown mutant {arm}"
  raise SystemExit(msg)


def _work_dir(args: argparse.Namespace) -> Path:
  work = args.work_dir or Path(os.environ.get("TMPDIR", "/tmp")) / "laser_decode_e2e"
  work.mkdir(parents=True, exist_ok=True)
  return work


def _logger() -> logging.Logger:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
  logger = logging.getLogger("laser_decode_e2e")
  logger.setLevel(logging.INFO)
  return logger


def _build_job(root: Path) -> dict[str, Any]:
  import numpy as np

  from aminx.families.laser_mpnn.driver import _working_dtype, decoding_order
  from aminx.families.laser_mpnn.featurize import featurize

  structures: dict[str, Any] = {}
  for path in _fixture_paths(root):
    features = featurize(path, dtype=_working_dtype())
    order = decoding_order(
      np.asarray(features.chain_mask),
      np.asarray(features.extra_atom_contact_mask),
      seed=SEED,
    )
    structures[path.stem] = {
      "pdb": str(path),
      "order": [int(index) for index in order],
    }
  return {"seed": SEED, "structures": structures}


def _load_pair(checkpoint: Path) -> tuple[Any, Any, Any]:  # noqa: ANN401
  from types import SimpleNamespace

  import jax

  from aminx.families.laser_mpnn.driver import (
    LaserDriver,
    _assign_state,
    _torch_blob,
    _working_dtype,
  )
  from aminx.model.laser.joint_decode import LaserJointDecode

  model = LaserDriver().load(SimpleNamespace(model_local_path=str(checkpoint)))
  state = _torch_blob(checkpoint)["model_state_dict"]
  joint = _assign_state(
    LaserJointDecode(key=jax.random.PRNGKey(0)),
    state,
    ("chi_offset_prediction_layers.",),
    _working_dtype(),
  )
  return model, joint, state


def _shift_pdb(source: Path, dest: Path, dx: float = 5.0) -> None:
  """Copy a PDB with every atom moved on x.

  The two tied structures must not be the same coordinates. Equal encodings
  make a logit-space mix draw the same token as a probability-space mix, and
  the λ control would not fail.
  """
  lines: list[str] = []
  for line in source.read_text(encoding="utf-8").splitlines(keepends=True):
    if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
      shifted = float(line[30:38]) + dx
      line = f"{line[:30]}{shifted:8.3f}{line[38:]}"
    lines.append(line)
  dest.write_text("".join(lines), encoding="utf-8")


def _tied_payload(root: Path, work: Path) -> dict[str, Any]:
  import numpy as np

  from aminx.families.laser_mpnn.driver import _working_dtype, decoding_order
  from aminx.families.laser_mpnn.featurize import featurize

  source = _fixture_paths(root)[0]
  shifted = work / f"{source.stem}_shift.pdb"
  _shift_pdb(source, shifted)
  features = featurize(source, dtype=_working_dtype())
  order = decoding_order(
    np.asarray(features.chain_mask),
    np.asarray(features.extra_atom_contact_mask),
    seed=SEED,
  )
  length = int(order.shape[0])
  rng = np.random.default_rng(SEED)
  seq_u = rng.random(length)
  chi_u = rng.random((length, 4, 2))
  return {
    "pdb1": str(source),
    "pdb2": str(shifted),
    "order": [int(index) for index in order],
    "seq_uniforms": seq_u.tolist(),
    "chi_uniforms": chi_u.tolist(),
    "lambda": 0.5,
  }


def _tied_stream(payload: dict[str, Any]) -> Any:
  """One shim stream: sequence, then χ0 structure 1, χ0 structure 2, ..."""
  import numpy as np

  seq = np.asarray(payload["seq_uniforms"], dtype=np.float64)
  chi = np.asarray(payload["chi_uniforms"], dtype=np.float64)
  length = seq.shape[0]
  stream = np.zeros((1, length, 9), dtype=np.float64)
  stream[0, :, 0] = seq
  for index in range(4):
    stream[0, :, 1 + 2 * index] = chi[:, index, 0]
    stream[0, :, 2 + 2 * index] = chi[:, index, 1]
  return stream


def _decode_tied(payload: dict[str, Any], checkpoint: Path, arm: str | None) -> dict[str, Any]:
  import numpy as np

  from aminx.families.laser_mpnn.driver import _working_dtype
  from aminx.families.laser_mpnn.featurize import featurize
  from aminx.model.laser.graphs import GraphStructure
  from aminx.model.laser.tied import tied_decode

  model, joint, _state = _load_pair(checkpoint)
  joint, decoder = _install_mutant(arm, joint, model.decoder)
  dtype = _working_dtype()
  left = featurize(
    payload["pdb1"],
    dtype=dtype,
    lig_pr_knn_k=int(model.lig_pr_knn_graph_k),
    lig_pr_distance_cutoff=float(model.lig_pr_distance_cutoff),
  )
  right = featurize(
    payload["pdb2"],
    dtype=dtype,
    lig_pr_knn_k=int(model.lig_pr_knn_graph_k),
    lig_pr_distance_cutoff=float(model.lig_pr_distance_cutoff),
  )
  order = [int(index) for index in payload["order"]]
  if arm == "reversed_order":
    order = list(reversed(order))
  structure = GraphStructure(
    pr_pr_knn_graph_k=int(model.pr_pr_knn_graph_k),
    lig_pr_knn_graph_k=int(model.lig_pr_knn_graph_k),
    lig_lig_knn_graph_k=int(model.lig_lig_knn_graph_k),
    lig_pr_distance_cutoff=float(model.lig_pr_distance_cutoff),
  )
  decoded = tied_decode(
    model.encoder,
    decoder,
    joint,
    left.backbone_coords,
    right.backbone_coords,
    left.ligand_coords,
    right.ligand_coords,
    left.ligand_atomic_numbers,
    right.ligand_atomic_numbers,
    left.ligand_subbatch_indices,
    right.ligand_subbatch_indices,
    np.asarray(model.period_index),
    np.asarray(model.group_index),
    structure,
    np.asarray(left.sequence_indices),
    np.asarray(left.chi_angles),
    np.asarray(right.chi_angles),
    np.asarray(left.chain_mask),
    np.asarray(order, dtype=np.int32),
    np.asarray(payload["seq_uniforms"], dtype=np.float64),
    np.asarray(payload["chi_uniforms"], dtype=np.float64),
    sequence_temperature=1.0,
    chi_temperature=1.0,
    lambda_=float(payload["lambda"]),
    lambda_on_logits=arm == "lambda_on_logits",
    disabled_residues=("X",),
  )
  degrees2 = np.asarray(decoded.chi_degrees_2)
  bins2 = np.asarray(decoded.chi_bins_2)
  logits2 = np.asarray(decoded.chi_logits_2)
  if arm == "chi2_equals_chi1":
    # Copying structure 1's χ onto structure 2 drops the second draw.
    degrees2 = np.asarray(decoded.chi_degrees_1)
    bins2 = np.asarray(decoded.chi_bins_1)
    logits2 = np.asarray(decoded.chi_logits_1)
  return {
    "sequence": [int(index) for index in decoded.sequence],
    "chi_degrees_1": np.asarray(decoded.chi_degrees_1).tolist(),
    "chi_degrees_2": degrees2.tolist(),
    "chi_bins_1": np.asarray(decoded.chi_bins_1).tolist(),
    "chi_bins_2": bins2.tolist(),
    "chi_logits_2": logits2.tolist(),
    "sequence_probs_1": _softmax_rows(decoded.sequence_logits_1).tolist(),
    "sequence_probs_2": _softmax_rows(decoded.sequence_logits_2).tolist(),
  }


def _softmax_rows(logits: Any) -> Any:  # noqa: ANN401
  import numpy as np

  values = np.asarray(logits, dtype=np.float64)
  shifted = values - np.max(values, axis=-1, keepdims=True)
  exp = np.exp(shifted)
  return exp / np.sum(exp, axis=-1, keepdims=True)


def _tied_metrics(aminx: dict[str, Any], upstream: dict[str, Any]) -> dict[str, float | bool]:
  import numpy as np

  from aminx.model.laser.joint_decode import chi_position_mask, circular_abs_delta_deg

  seq_a = np.asarray(aminx["sequence"], dtype=np.int64)
  seq_u = np.asarray(upstream["sequence"], dtype=np.int64)
  exact = bool(seq_a.shape == seq_u.shape and np.array_equal(seq_a, seq_u))
  agreement = float(np.mean(seq_a == seq_u)) if seq_a.shape == seq_u.shape else 0.0
  mask = chi_position_mask(seq_u) if seq_a.shape == seq_u.shape else np.zeros((0, 4), dtype=bool)
  bin_hits: list[float] = []
  deltas: list[float] = []
  for side in (1, 2):
    bins_a = np.asarray(aminx[f"chi_bins_{side}"])
    bins_u = np.asarray(upstream[f"chi_bins_{side}"])
    deg_a = np.asarray(aminx[f"chi_degrees_{side}"])
    deg_u = np.asarray(upstream[f"chi_degrees_{side}"])
    if bins_a.shape != bins_u.shape or not np.any(mask):
      bin_hits.append(0.0 if bins_a.shape != bins_u.shape else 1.0)
      deltas.append(0.0 if not np.any(mask) else float("inf"))
      continue
    bin_hits.append(float(np.mean(bins_a[mask] == bins_u[mask])))
    delta = circular_abs_delta_deg(deg_a, deg_u)
    delta = np.where(np.isfinite(delta), delta, np.inf)
    deltas.append(float(np.max(np.asarray(delta)[mask])))
  return {
    "exact_seq": exact,
    "sequence_agreement": agreement,
    "chi_bin_agreement": float(min(bin_hits) if bin_hits else 0.0),
    "max_circular_delta": float(max(deltas) if deltas else float("inf")),
  }


def _oracle_tied(model: Any, model_params: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:  # noqa: ANN401
  import torch

  from LASErMPNN.run_inference import ProteinComplexData, get_protein_hierview
  from oracle_shims.laser import injected_uniform_draws

  def _batch(path: str) -> Any:  # noqa: ANN401
    view = get_protein_hierview(path)
    data = ProteinComplexData(view, path, verbose=False)
    batch = data.output_batch_data(fix_beta=False)
    _cast_floats(batch, torch.float64)
    batch.construct_graphs(
      model.rotamer_builder,
      model.ligand_featurizer,
      **model_params["graph_structure"],
      protein_training_noise=0.0,
      ligand_training_noise=0.0,
      subgraph_only_dropout_rate=0.0,
      num_adjacent_residues_to_drop=0,
      build_hydrogens=bool(model_params["build_hydrogens"]),
    )
    _cast_floats(batch, torch.float64)
    order = torch.tensor(payload["order"], dtype=torch.float64).unsqueeze(0)
    batch.decoding_order = order
    return batch

  captured: list[int] = []
  stream = _tied_stream(payload)
  with injected_uniform_draws(stream):
    shimmed = torch.distributions.Categorical.sample

    def _collect(self: Any, sample_shape: Any = None) -> Any:  # noqa: ANN401
      out = shimmed(self, torch.Size() if sample_shape is None else sample_shape)
      captured.append(int(out.reshape(-1)[0].detach().cpu()))
      return out

    torch.distributions.Categorical.sample = _collect  # ty: ignore[invalid-assignment]
    first, second = model.tied_sample(
      _batch(payload["pdb1"]),
      _batch(payload["pdb2"]),
      lambda_=float(payload["lambda"]),
      sequence_sample_temperature=1.0,
      chi_angle_sample_temperature=1.0,
      disabled_residues=["X"],
      disable_pbar=True,
      repack_all=False,
    )
  length = len(payload["order"])
  bins1 = [[0, 0, 0, 0] for _ in range(length)]
  bins2 = [[0, 0, 0, 0] for _ in range(length)]
  # Per step the shim consumes sequence, then χ index × structure.
  cursor = 0
  for step in range(length):
    cursor += 1
    for chi in range(4):
      bins1[step][chi] = captured[cursor]
      cursor += 1
      bins2[step][chi] = captured[cursor]
      cursor += 1
  # Bins are in decoding-step order. Scatter them back onto residue rows.
  order = [int(index) for index in payload["order"]]
  residue_bins1 = [[0, 0, 0, 0] for _ in range(length)]
  residue_bins2 = [[0, 0, 0, 0] for _ in range(length)]
  for step, residue in enumerate(order):
    residue_bins1[residue] = bins1[step]
    residue_bins2[residue] = bins2[step]
  return {
    "sequence": [int(index) for index in first.sampled_sequence_indices.detach().cpu()],
    "chi_degrees_1": first.sampled_chi_degrees.detach().cpu().tolist(),
    "chi_degrees_2": second.sampled_chi_degrees.detach().cpu().tolist(),
    "chi_bins_1": residue_bins1,
    "chi_bins_2": residue_bins2,
    "sequence_probs_1": torch.softmax(first.sequence_logits, dim=-1).detach().cpu().tolist(),
    "sequence_probs_2": torch.softmax(second.sequence_logits, dim=-1).detach().cpu().tolist(),
  }


def _decode_one(payload: dict[str, Any], checkpoint: Path, arm: str | None) -> dict[str, Any]:
  import numpy as np

  from aminx.families.laser_mpnn.driver import _working_dtype
  from aminx.families.laser_mpnn.featurize import featurize
  from aminx.model.laser.graphs import GraphStructure
  from aminx.model.laser.joint_decode import decode_order

  model, joint, _state = _load_pair(checkpoint)
  joint, decoder = _install_mutant(arm, joint, model.decoder)
  features = featurize(
    payload["pdb"],
    dtype=_working_dtype(),
    lig_pr_knn_k=int(model.lig_pr_knn_graph_k),
    lig_pr_distance_cutoff=float(model.lig_pr_distance_cutoff),
  )
  order = [int(index) for index in payload["order"]]
  if arm == "reversed_order":
    order = list(reversed(order))
  structure = GraphStructure(
    pr_pr_knn_graph_k=int(model.pr_pr_knn_graph_k),
    lig_pr_knn_graph_k=int(model.lig_pr_knn_graph_k),
    lig_lig_knn_graph_k=int(model.lig_lig_knn_graph_k),
    lig_pr_distance_cutoff=float(model.lig_pr_distance_cutoff),
  )
  decoded = decode_order(
    model.encoder,
    decoder,
    joint,
    features.backbone_coords,
    features.ligand_coords,
    features.ligand_atomic_numbers,
    features.ligand_subbatch_indices,
    np.asarray(model.period_index),
    np.asarray(model.group_index),
    structure,
    np.asarray(features.sequence_indices),
    np.asarray(features.chi_angles),
    np.asarray(features.chain_mask),
    np.asarray(features.first_shell_ligand_contact_mask),
    np.asarray(order, dtype=np.int32),
    np.zeros((1,), dtype=np.float64),
    sequence_temperature=None,
    chi_temperature=None,
    disabled_residues=("X",),
  )
  return {
    "sequence": [int(index) for index in decoded.sequence],
    "chi_degrees": np.asarray(decoded.chi_degrees).tolist(),
    "chi_logits": np.asarray(decoded.chi_logits).tolist(),
    "sequence_probs": _softmax_rows(decoded.sequence_logits).tolist(),
  }


def _metrics(aminx: dict[str, Any], upstream: dict[str, Any]) -> dict[str, float | bool]:
  import numpy as np

  from aminx.model.laser.joint_decode import chi_position_mask, circular_abs_delta_deg

  seq_a = np.asarray(aminx["sequence"], dtype=np.int64)
  seq_u = np.asarray(upstream["sequence"], dtype=np.int64)
  if seq_a.shape != seq_u.shape:
    return {
      "exact_seq": False,
      "sequence_agreement": 0.0,
      "chi_bin_agreement": 0.0,
      "max_circular_delta": float("inf"),
    }
  exact = bool(np.array_equal(seq_a, seq_u))
  agreement = float(np.mean(seq_a == seq_u))
  mask = chi_position_mask(seq_u)
  bins_a = np.argmax(np.asarray(aminx["chi_logits"]), axis=-1)
  bins_u = np.argmax(np.asarray(upstream["chi_logits"]), axis=-1)
  if not np.any(mask):
    chi_agree = 1.0
    max_delta = 0.0
  else:
    chi_agree = float(np.mean(bins_a[mask] == bins_u[mask]))
    delta = circular_abs_delta_deg(np.asarray(aminx["chi_degrees"]), np.asarray(upstream["chi_degrees"]))
    delta = np.where(np.isfinite(delta), delta, np.inf)
    max_delta = float(np.max(np.asarray(delta)[mask]))
  return {
    "exact_seq": exact,
    "sequence_agreement": agreement,
    "chi_bin_agreement": chi_agree,
    "max_circular_delta": max_delta,
  }


def _run_arm(args: argparse.Namespace) -> dict[str, Any]:
  import numpy as np

  job = json.loads(args.job.read_text(encoding="utf-8"))
  upstream = json.loads(args.upstream.read_text(encoding="utf-8"))
  checkpoint = _checkpoint(args)
  rows: list[dict[str, float | bool]] = []
  cells: dict[str, Any] = {"structures": {}, "tied": None}
  for name, payload in job["structures"].items():
    decoded = _decode_one(payload, checkpoint, args.arm)
    rows.append(_metrics(decoded, upstream[name]))
    cells["structures"][name] = {
      "orders": [
        {
          "order": 0,
          "dropout": 0,
          "aminx": decoded.get("sequence_probs"),
          "upstream": upstream[name].get("sequence_probs"),
        },
      ],
    }
  if "tied" in job and "tied" in upstream:
    tied = _decode_tied(job["tied"], checkpoint, args.arm)
    rows.append(_tied_metrics(tied, upstream["tied"]))
    cells["tied"] = {
      "orders": [
        {
          "order": 0,
          "dropout": 0,
          "aminx": {
            "sequence_probs_1": tied.get("sequence_probs_1"),
            "sequence_probs_2": tied.get("sequence_probs_2"),
            "sequence": tied.get("sequence"),
          },
          "upstream": {
            "sequence_probs_1": upstream["tied"].get("sequence_probs_1"),
            "sequence_probs_2": upstream["tied"].get("sequence_probs_2"),
            "sequence": upstream["tied"].get("sequence"),
          },
        },
      ],
    }
  if args.work_dir is not None:
    path = Path(args.work_dir) / f"cells_{args.arm}.json"
    path.write_text(json.dumps(cells), encoding="utf-8")
  exact = all(bool(row["exact_seq"]) for row in rows)
  seq_agree = float(np.min([row["sequence_agreement"] for row in rows])) if rows else 0.0
  chi_agree = float(np.min([row["chi_bin_agreement"] for row in rows])) if rows else 0.0
  max_delta = float(np.max([row["max_circular_delta"] for row in rows])) if rows else float("inf")
  # Agreement across the whole set is 1.0 only when every fixture is exact.
  if exact:
    seq_agree = 1.0
  if exact and all(row["chi_bin_agreement"] == 1.0 for row in rows):
    chi_agree = 1.0
  return {
    "band": _band(exact, chi_agree, max_delta),
    "sequence_agreement": seq_agree,
    "chi_bin_agreement": chi_agree,
    "max_circular_delta": max_delta,
  }


def _oracle_python(args: argparse.Namespace) -> str:
  if args.oracle_python:
    return str(args.oracle_python)
  try:
    import torch  # noqa: F401
  except ImportError:
    msg = "set LASER_ORACLE_PYTHON to the torch oracle interpreter"
    raise SystemExit(msg) from None
  return sys.executable


def _launch(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
  # A fresh interpreter is load-bearing. A jit/filter_jit cache warmed by an
  # earlier arm would replay the unmutated trace and the harness would report
  # a kill it never demonstrated. That includes trace-time mutants such as
  # rolling the χ head or zeroing the offset.
  return subprocess.run(
    command,
    check=False,
    capture_output=True,
    text=True,
    timeout=timeout,
  )


def _parent(args: argparse.Namespace, logger: logging.Logger) -> dict[str, Any]:
  checkpoint = _checkpoint(args)
  _check_inputs(args.laser_root, checkpoint)
  if float(args.arm_timeout) <= 0:
    msg = "--arm-timeout must be positive"
    raise SystemExit(msg)
  mutants = [item for item in args.mutants.split(",") if item]
  unknown = [item for item in mutants if item not in MUTANTS]
  if unknown:
    msg = f"unknown mutant {unknown[0]}"
    raise SystemExit(msg)
  work = _work_dir(args)
  job = _build_job(args.laser_root)
  job["tied"] = _tied_payload(args.laser_root, work)
  job_path = work / "job.json"
  upstream_path = work / "upstream.json"
  job_path.write_text(json.dumps(job), encoding="utf-8")
  script = str(Path(__file__).resolve())
  oracle = _launch(
    [
      _oracle_python(args),
      script,
      "--oracle-worker",
      "--laser-root",
      str(args.laser_root),
      "--checkpoint",
      str(checkpoint),
      "--job",
      str(job_path),
      "--upstream",
      str(upstream_path),
      "--work-dir",
      str(work),
    ],
    float(args.arm_timeout),
  )
  if oracle.returncode != 0:
    logger.error("oracle worker failed: %s", oracle.stderr[-500:])
    raise SystemExit(oracle.stderr[-500:])
  measured: dict[str, dict[str, Any]] = {}
  for arm in ("clean", *mutants):
    payload_out = work / f"payload_{arm}.json"
    payload_out.unlink(missing_ok=True)
    try:
      launched = _launch(
        [
          sys.executable,
          script,
          "--arm",
          arm,
          "--job",
          str(job_path),
          "--upstream",
          str(upstream_path),
          "--checkpoint",
          str(checkpoint),
          "--laser-root",
          str(args.laser_root),
          "--work-dir",
          str(work),
          "--payload-out",
          str(payload_out),
        ],
        float(args.arm_timeout),
      )
    except subprocess.TimeoutExpired as exc:
      logger.error("arm %s timed out", arm)
      measured[arm] = {"band": "error", "detail": str(exc)}
      continue
    if launched.returncode != 0:
      measured[arm] = {"band": "error", "detail": launched.stderr[-500:]}
      continue
    try:
      measured[arm] = json.loads(payload_out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
      measured[arm] = {"band": "error", "detail": launched.stderr[-500:] or "empty payload"}
  clean = str(measured["clean"]["band"])
  statuses = {
    mutant: "failed"
    if measured[mutant]["band"] == "fail"
    else "passed"
    if measured[mutant]["band"] == "pass"
    else "error"
    for mutant in mutants
  }
  if args.controls_out is not None:
    args.controls_out.parent.mkdir(parents=True, exist_ok=True)
    args.controls_out.write_text(
      json.dumps(
        {
          "clean": clean,
          "mutants": statuses,
          "weights": {str(checkpoint): _sha256(checkpoint)},
        },
        indent=2,
      )
      + "\n",
      encoding="utf-8",
    )
  n_failed = sum(status == "failed" for status in statuses.values())
  clean_row = measured["clean"]
  return {
    "clean": clean,
    "n_listed": len(mutants),
    "n_failed": n_failed,
    "sequence_agreement": float(clean_row.get("sequence_agreement", float("nan"))),
    "chi_bin_agreement": float(clean_row.get("chi_bin_agreement", float("nan"))),
    "max_circular_delta": float(clean_row.get("max_circular_delta", float("nan"))),
  }


def _package_parent(root: Path) -> Path:
  if (root / "utils" / "model.py").is_file():
    return root.parent
  nested = root / "LASErMPNN"
  if (nested / "utils" / "model.py").is_file():
    return root
  msg = f"{root} does not contain LASErMPNN utils/model.py"
  raise SystemExit(msg)


def _cast_floats(value: object, dtype: Any) -> None:  # noqa: ANN401
  import torch

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


def _oracle_worker(args: argparse.Namespace) -> None:
  import torch

  sys.path.insert(0, str(_package_parent(args.laser_root)))
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
  from LASErMPNN.run_inference import (  # noqa: PLC0415
    ProteinComplexData,
    get_protein_hierview,
    load_model_from_parameter_dict,
  )
  from LASErMPNN.utils.model import RBF_Encoding  # noqa: PLC0415
  from oracle_shims.laser import f64_runtime  # noqa: PLC0415

  checkpoint = _checkpoint(args)
  torch.manual_seed(0)
  model, params = load_model_from_parameter_dict(str(checkpoint), "cpu", strict=False)
  state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
  model.load_state_dict(state, strict=False)
  model.double()
  model.eval()
  job = json.loads(args.job.read_text(encoding="utf-8"))
  decoded: dict[str, dict[str, list[float]]] = {}
  model_params = params["model_params"]
  with f64_runtime(RBF_Encoding), torch.no_grad():
    for name, payload in job["structures"].items():
      view = get_protein_hierview(payload["pdb"])
      data = ProteinComplexData(view, payload["pdb"], verbose=False)
      batch = data.output_batch_data(fix_beta=False)
      _cast_floats(batch, torch.float64)
      batch.construct_graphs(  # type: ignore[attr-defined]
        model.rotamer_builder,
        model.ligand_featurizer,
        **model_params["graph_structure"],
        protein_training_noise=0.0,
        ligand_training_noise=0.0,
        subgraph_only_dropout_rate=0.0,
        num_adjacent_residues_to_drop=0,
        build_hydrogens=bool(model_params["build_hydrogens"]),
      )
      _cast_floats(batch, torch.float64)
      order = torch.tensor(payload["order"], dtype=torch.float64).unsqueeze(0)
      batch.decoding_order = order
      sampled = model.sample(
        batch,
        sequence_sample_temperature=None,
        chi_angle_sample_temperature=None,
        disabled_residues=["X"],
        disable_pbar=True,
        seq_min_p=0.0,
        chi_min_p=0.0,
        ignore_chain_mask_zeros=False,
        repack_all=False,
      )
      decoded[name] = {
        "sequence": [int(index) for index in sampled.sampled_sequence_indices.detach().cpu()],
        "chi_degrees": sampled.sampled_chi_degrees.detach().cpu().tolist(),
        "chi_logits": sampled.chi_logits.detach().cpu().tolist(),
        "sequence_probs": torch.softmax(sampled.sequence_logits, dim=-1).detach().cpu().tolist(),
      }
    if "tied" in job:
      decoded["tied"] = _oracle_tied(model, model_params, job["tied"])
  args.upstream.write_text(json.dumps(decoded), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
  args = _parse(sys.argv[1:] if argv is None else argv)
  if args.oracle_worker:
    _oracle_worker(args)
    return
  if args.arm:
    payload = _run_arm(args)
    text = json.dumps(payload)
    if args.payload_out is not None:
      args.payload_out.write_text(text, encoding="utf-8")
    else:
      sys.stdout.write(text)
    return
  # Missing results path is fatal before any arm runs. A pass-looking payload
  # written only to --payload-out still records outcome=unknown.
  path = _results_path()
  logger = _logger()
  results = _parent(args, logger)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
