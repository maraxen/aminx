# ruff: noqa: S101
"""Batched LASEr decode matches one ``decode_order`` call per sample."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("prody", reason="needs the laser extra")

from aminx.families.laser_mpnn.driver import (  # noqa: E402
  LASErMPNN,
  LaserDriver,
  _graph,
  _working_dtype,
  budget_mask_for,
  load_joint_decode,
)
from aminx.families.laser_mpnn.featurize import featurize  # noqa: E402
from aminx.families.laser_mpnn.sample_host import SampleStages, sample_batches  # noqa: E402
from aminx.host.family_driver import FAMILY_DRIVERS  # noqa: E402
from aminx.host.runner import sample  # noqa: E402
from aminx.model.laser.joint_decode import (  # noqa: E402
  LaserJointDecode,
  decode_order,
  decode_order_batched,
)
from aminx.run.options import LaserOptions  # noqa: E402
from tests.families.laser_mpnn.test_driver import _write_pdb  # noqa: E402
from tests.families.laser_mpnn.test_sample import _spec  # noqa: E402

_RTOL = 1e-5
_ATOL = 1e-5


@pytest.fixture
def registered() -> Iterator[LaserDriver]:
  driver = LaserDriver()
  FAMILY_DRIVERS.register("lasermpnn")(driver)
  yield driver
  FAMILY_DRIVERS.discard("lasermpnn")


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
  path = tmp_path / "tiny.eqx"
  eqx.tree_serialise_leaves(path, LASErMPNN(key=jax.random.PRNGKey(0)))
  return path


def _token_logprob(logits: np.ndarray, sequence: np.ndarray) -> np.ndarray:
  """Per-sample log-prob of the drawn letter. ``sequence`` is laser indices."""
  log_probs = jax.nn.log_softmax(jnp.asarray(logits), axis=-1)
  index = jnp.asarray(sequence, dtype=jnp.int32)
  picked = jnp.take_along_axis(log_probs, index[..., None], axis=-1)
  return np.asarray(picked[..., 0])


def _assert_close(got: np.ndarray, ref: np.ndarray) -> None:
  np.testing.assert_allclose(got, ref, rtol=_RTOL, atol=_ATOL, equal_nan=True)


def _parity(
  model: LASErMPNN,
  joint: LaserJointDecode,
  pdb: Path,
  *,
  n_samples: int,
  temperatures: Sequence[float | None],
  seed: int,
) -> None:
  features = featurize(pdb, dtype=_working_dtype())
  length = int(features.sequence_indices.shape[0])
  batch = n_samples * len(temperatures)
  rng = np.random.default_rng(seed)
  orders = np.stack(
    [rng.permutation(length).astype(np.int32) for _ in range(batch)],
  )
  uniforms = rng.random((batch, length * 5))
  temps = [temperature for _ in range(n_samples) for temperature in temperatures]
  chain_mask = np.zeros((length,), dtype=bool)
  first_shell = np.asarray(features.first_shell_ligand_contact_mask)
  budget = budget_mask_for(
    np.asarray(features.resnum_indices),
    features.ss_code,
    np.asarray(features.exposed_mask),
    selection=None,
    constrain_to_exposed_non_ss=False,
  )
  charged = np.zeros((length,), dtype=bool)
  shared = {
    "sequence_indices": np.asarray(features.sequence_indices),
    "chi_angles": np.asarray(features.chi_angles),
    "chain_mask": chain_mask,
    "first_shell": first_shell,
    "budget_mask": budget,
    "charged_mask": charged,
  }
  refs = [
    decode_order(
      model.encoder,
      model.decoder,
      joint,
      features.backbone_coords,
      features.ligand_coords,
      features.ligand_atomic_numbers,
      features.ligand_subbatch_indices,
      np.asarray(model.period_index),
      np.asarray(model.group_index),
      _graph(model),
      shared["sequence_indices"],
      shared["chi_angles"],
      shared["chain_mask"],
      shared["first_shell"],
      orders[index],
      uniforms[index],
      sequence_temperature=temps[index],
      disabled_residues=("X",),
      budget_mask=budget,
      charged_mask=charged,
    )
    for index in range(batch)
  ]
  batched = decode_order_batched(
    model.encoder,
    model.decoder,
    joint,
    features.backbone_coords,
    features.ligand_coords,
    features.ligand_atomic_numbers,
    features.ligand_subbatch_indices,
    np.asarray(model.period_index),
    np.asarray(model.group_index),
    _graph(model),
    shared["sequence_indices"],
    shared["chi_angles"],
    shared["chain_mask"],
    shared["first_shell"],
    orders,
    uniforms,
    sequence_temperature=tuple(temps),
    disabled_residues=("X",),
    budget_mask=budget,
    charged_mask=charged,
  )
  ref_sequence = np.stack([np.asarray(row.sequence) for row in refs])
  ref_chi = np.stack([np.asarray(row.chi_degrees) for row in refs])
  ref_logits = np.stack([np.asarray(row.sequence_logits) for row in refs])
  np.testing.assert_array_equal(np.asarray(batched.sequence), ref_sequence)
  _assert_close(np.asarray(batched.chi_degrees), ref_chi)
  _assert_close(np.asarray(batched.sequence_logits), ref_logits)
  _assert_close(
    _token_logprob(np.asarray(batched.sequence_logits), np.asarray(batched.sequence)),
    _token_logprob(ref_logits, ref_sequence),
  )


def test_batched_decode_matches_decode_order(tmp_path: Path, model_path: Path) -> None:
  """4 samples x 2 temperatures, plus a None/float mix, match per-sample decode."""
  pdb = tmp_path / "complex.pdb"
  _write_pdb(pdb)
  model = eqx.tree_deserialise_leaves(model_path, LASErMPNN(key=jax.random.PRNGKey(0)))
  joint = LaserJointDecode(key=jax.random.PRNGKey(11))
  _parity(model, joint, pdb, n_samples=4, temperatures=(0.2, 0.7), seed=11)
  _parity(model, joint, pdb, n_samples=2, temperatures=(None, 0.5), seed=12)


def test_sample_host_matches_per_sample_reference(
  registered: LaserDriver,
  tmp_path: Path,
  model_path: Path,
) -> None:
  """The batched host repeats the per-sample sequences for one seed."""
  del registered
  pdb = tmp_path / "complex.pdb"
  _write_pdb(pdb)
  n_samples = 4
  temperatures = (0.2, 0.7)
  spec = _spec(
    pdb,
    model_path,
    num_samples=n_samples,
    temperature=temperatures,
    random_seed=7,
    samples_chunk_size=2,
    laser=LaserOptions(disabled_residues=("X",)),
  )
  got = np.asarray(sample(spec)["structures"]["0"]["arrays"]["sequence"])
  model = LaserDriver().load(spec)
  stages = SampleStages(model, spec)
  item = next(sample_batches(spec)).arrays["sampled"][0]
  dropout_key = jax.random.PRNGKey(int(spec.random_seed))
  rows = []
  for offset in range(n_samples):
    for temp_index, temperature in enumerate(temperatures):
      key = jax.random.fold_in(dropout_key, offset * len(temperatures) + temp_index)
      rows.append(stages._decode(item, temperature, key)["sequence"])  # noqa: SLF001
  np.testing.assert_array_equal(got, np.stack(rows))


def test_perturbed_uniforms_change_one_sample(tmp_path: Path, model_path: Path) -> None:
  """Changing one sample's uniforms changes that sample's sequence."""
  pdb = tmp_path / "complex.pdb"
  _write_pdb(pdb)
  model = eqx.tree_deserialise_leaves(model_path, LASErMPNN(key=jax.random.PRNGKey(0)))
  joint = LaserJointDecode(key=jax.random.PRNGKey(3))
  features = featurize(pdb, dtype=_working_dtype())
  length = int(features.sequence_indices.shape[0])
  batch = 4
  rng = np.random.default_rng(3)
  orders = np.stack([rng.permutation(length).astype(np.int32) for _ in range(batch)])
  uniforms = rng.random((batch, length * 5))
  perturbed = np.array(uniforms, copy=True)
  perturbed[0] = 0.999
  kwargs = {
    "sequence_temperature": (50.0, 50.0, 1.0, 1.0),
    "disabled_residues": ("X",),
    "budget_mask": None,
    "charged_mask": None,
  }
  reference = decode_order_batched(
    model.encoder,
    model.decoder,
    joint,
    features.backbone_coords,
    features.ligand_coords,
    features.ligand_atomic_numbers,
    features.ligand_subbatch_indices,
    np.asarray(model.period_index),
    np.asarray(model.group_index),
    _graph(model),
    np.asarray(features.sequence_indices),
    np.asarray(features.chi_angles),
    np.zeros((length,), dtype=bool),
    np.asarray(features.first_shell_ligand_contact_mask),
    orders,
    uniforms,
    **kwargs,
  )
  changed = decode_order_batched(
    model.encoder,
    model.decoder,
    joint,
    features.backbone_coords,
    features.ligand_coords,
    features.ligand_atomic_numbers,
    features.ligand_subbatch_indices,
    np.asarray(model.period_index),
    np.asarray(model.group_index),
    _graph(model),
    np.asarray(features.sequence_indices),
    np.asarray(features.chi_angles),
    np.zeros((length,), dtype=bool),
    np.asarray(features.first_shell_ligand_contact_mask),
    orders,
    perturbed,
    **kwargs,
  )
  assert not np.array_equal(np.asarray(changed.sequence[0]), np.asarray(reference.sequence[0]))
  np.testing.assert_array_equal(np.asarray(changed.sequence[1:]), np.asarray(reference.sequence[1:]))
