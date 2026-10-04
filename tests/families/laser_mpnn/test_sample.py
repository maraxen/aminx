# ruff: noqa: S101
"""LaserDriver ``sample``: schema, constraints, determinism, and the graded decode."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("prody", reason="needs the laser extra")

from aminx.families.laser_mpnn.driver import (  # noqa: E402
  ALPHABET,
  LASER_ALPHABET,
  LASER_OF_CANONICAL,
  LASErMPNN,
  LaserDriver,
  _graph,
  _working_dtype,
  budget_mask_for,
)
from aminx.families.laser_mpnn.featurize import (  # noqa: E402
  MAX_ATOMS,
  MAX_PROTONATED_ATOMS,
  featurize,
)
from aminx.host.family_driver import FAMILY_DRIVERS  # noqa: E402
from aminx.host.runner import sample  # noqa: E402
from aminx.model.laser.joint_decode import LaserJointDecode, decode_order  # noqa: E402
from aminx.run.options import LaserOptions  # noqa: E402
from aminx.run.specs import SamplingSpecification  # noqa: E402
from tests.families.laser_mpnn.test_driver import _write_pdb  # noqa: E402

_BASE_KEYS = ("sequence", "seq_log_prob", "chi_deg", "chi_mask", "sidechain_coords")


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


def _spec(pdb: Path, model_path: Path, **kwargs: object) -> SamplingSpecification:
  laser = kwargs.pop("laser", LaserOptions())
  return SamplingSpecification(
    inputs=str(pdb),
    model_family="lasermpnn",
    checkpoint_id="lasermpnn_test",
    model_local_path=model_path,
    laser=laser,  # type: ignore[arg-type]
    **kwargs,  # type: ignore[arg-type]
  )


def test_sample_handles_and_schema(registered: LaserDriver) -> None:
  """(a) ``sample`` is handled and the schema matches §5.6."""
  del registered
  driver = LaserDriver()
  assert driver.handles(None, "sample")
  schema = driver.result_schema(SamplingSpecification(inputs="a.pdb", model_family="lasermpnn"), "sample")
  assert set(schema) == set(_BASE_KEYS)
  assert schema["sequence"].dims == ("N", "L_total")
  assert schema["sequence"].dtype == "int32"
  assert schema["seq_log_prob"].dims == ("N", "L_total")
  assert schema["seq_log_prob"].dtype == "float32"
  assert schema["chi_deg"].dims == ("N", "L_total", "4")
  assert schema["chi_deg"].dtype == "float32"
  assert schema["chi_mask"].dims == ("N", "L_total", "4")
  assert schema["chi_mask"].dtype == "bool"
  assert schema["sidechain_coords"].dims == ("N", "L_total", "atom", "xyz")
  tied = driver.result_schema(
    SamplingSpecification(
      inputs="a.pdb",
      model_family="lasermpnn",
      laser=LaserOptions(tied_second_input="b.pdb"),
      return_logits=False,
    ),
    "sample",
  )
  assert set(tied) == {
    *_BASE_KEYS,
    "seq_log_prob_2",
    "chi_deg_2",
    "sidechain_coords_2",
  }
  assert tied["chi_deg_2"].dims == ("N", "L_total", "4")
  assert "sequence" in tied and "chi_mask" in tied
  with_logits = driver.result_schema(
    SamplingSpecification(
      inputs="a.pdb",
      model_family="lasermpnn",
      laser=LaserOptions(tied_second_input="b.pdb"),
      return_logits=True,
    ),
    "sample",
  )
  assert "seq_logits_2" in with_logits
  assert with_logits["seq_logits_2"].dims == ("N", "L_total", "alphabet")
  assert with_logits["chi_logits_2"].dims == ("N", "L_total", "4", "chi_bin")
  fasta_only = driver.result_schema(
    SamplingSpecification(
      inputs="a.pdb",
      model_family="lasermpnn",
      laser=LaserOptions(output_fasta_only=True),
    ),
    "sample",
  )
  assert set(fasta_only) == {"fasta"}
  score = driver.result_schema(
    SamplingSpecification(inputs="a.pdb", model_family="lasermpnn"),
    "score:nll",
  )
  assert set(score) == {"nll"}


def test_sample_shapes_tokens_and_fixed_positions(
  registered: LaserDriver,
  tmp_path: Path,
  model_path: Path,
) -> None:
  """(b) Declared shapes, disabled letters stay out, fixed rows keep the native token."""
  del registered
  pdb = tmp_path / "complex.pdb"
  native = _write_pdb(pdb)
  disabled = tuple(letter for letter in LASER_ALPHABET if letter != "G")
  result = sample(
    _spec(
      pdb,
      model_path,
      num_samples=1,
      temperature=None,
      random_seed=3,
      fixed_positions=[0],
      laser=LaserOptions(disabled_residues=disabled),
    ),
  )
  arrays = result["structures"]["0"]["arrays"]
  sequence = np.asarray(arrays["sequence"])
  assert sequence.shape == (1, len(native))
  assert np.asarray(arrays["seq_log_prob"]).shape == (1, len(native))
  assert np.asarray(arrays["chi_deg"]).shape == (1, len(native), 4)
  assert np.asarray(arrays["chi_mask"]).shape == (1, len(native), 4)
  assert sequence[0, 0] == ALPHABET.index(native[0])
  assert sequence[0, 1] == ALPHABET.index("G")
  designed = sequence[0, 1:]
  forbidden = {ALPHABET.index(letter) for letter in disabled}
  assert forbidden.isdisjoint(set(int(token) for token in designed))


def test_sample_is_deterministic(registered: LaserDriver, tmp_path: Path, model_path: Path) -> None:
  """(c) The same seed repeats; a different seed does not."""
  del registered
  pdb = tmp_path / "complex.pdb"
  _write_pdb(pdb)
  options = LaserOptions(chi_temp=1.0, disabled_residues=("X",))

  def _run(seed: int) -> np.ndarray:
    result = sample(
      _spec(
        pdb,
        model_path,
        num_samples=2,
        temperature=0.8,
        random_seed=seed,
        laser=options,
      ),
    )
    arrays = result["structures"]["0"]["arrays"]
    return np.concatenate(
      [
        np.asarray(arrays["sequence"], dtype=np.float64).reshape(-1),
        np.asarray(arrays["seq_log_prob"], dtype=np.float64).reshape(-1),
        np.asarray(arrays["chi_deg"], dtype=np.float64).reshape(-1),
      ],
    )

  def _same(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.allclose(left, right, rtol=0, atol=0, equal_nan=True))

  first = _run(5)
  assert first.shape[0] > 0
  assert _same(first, _run(5))
  assert not _same(first, _run(9))


def test_temperature_axis_emits_one_row_per_temperature(
  registered: LaserDriver,
  tmp_path: Path,
  model_path: Path,
) -> None:
  """(d) Two temperatures and two samples produce four rows."""
  del registered
  pdb = tmp_path / "complex.pdb"
  _write_pdb(pdb)
  result = sample(
    _spec(
      pdb,
      model_path,
      num_samples=2,
      temperature=(0.2, 0.7),
      random_seed=3,
      laser=LaserOptions(disabled_residues=("X",)),
    ),
  )
  sequence = np.asarray(result["structures"]["0"]["arrays"]["sequence"])
  assert sequence.shape[0] == 4
  assert np.asarray(result["structures"]["0"]["arrays"]["chi_deg"]).shape[0] == 4


def test_sample_matches_decode_order_bit_for_bit(
  registered: LaserDriver,
  tmp_path: Path,
  model_path: Path,
) -> None:
  """(e) Injected order and uniforms: the driver is ``decode_order``, bit for bit."""
  del registered
  pdb = tmp_path / "complex.pdb"
  _write_pdb(pdb)
  features = featurize(pdb, dtype=_working_dtype())
  length = int(features.sequence_indices.shape[0])
  order = np.arange(length, dtype=np.int32)[::-1].copy()
  uniforms = np.zeros((1,), dtype=np.float64)
  seed = 11
  options = LaserOptions(disabled_residues=("X",))
  spec = _spec(
    pdb,
    model_path,
    num_samples=1,
    temperature=None,
    random_seed=seed,
    laser=options,
  )
  spec.injected_decoding_order = order
  spec.injected_uniforms = uniforms
  result = sample(spec)
  model = eqx.tree_deserialise_leaves(model_path, LASErMPNN(key=jax.random.PRNGKey(0)))
  direct = decode_order(
    model.encoder,
    model.decoder,
    LaserJointDecode(key=jax.random.PRNGKey(seed)),
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
    order,
    uniforms,
    sequence_temperature=None,
    chi_temperature=None,
    fs_sequence_temp=None,
    seq_min_p=0.0,
    chi_min_p=0.0,
    ala_budget=4,
    gly_budget=0,
    disabled_residues=("X",),
    ignore_chain_mask_zeros=False,
    repack_all=False,
    repack_only=False,
    budget_mask=budget_mask_for(
      np.asarray(features.resnum_indices),
      features.ss_code,
      np.asarray(features.exposed_mask),
      selection=None,
      constrain_to_exposed_non_ss=False,
    ),
    charged_mask=np.zeros((length,), dtype=bool),
    bias=None,
  )
  arrays = result["structures"]["0"]["arrays"]
  got = np.asarray(arrays["sequence"][0])
  laser = np.asarray(LASER_OF_CANONICAL)[got]
  np.testing.assert_array_equal(laser, np.asarray(direct.sequence, dtype=np.int32))
  got_chi = np.asarray(arrays["chi_deg"][0])
  direct_chi = np.asarray(direct.chi_degrees, dtype=np.float32)
  np.testing.assert_allclose(got_chi, direct_chi, rtol=0, atol=0, equal_nan=True)
  direct_prob = jax.nn.softmax(jnp.asarray(direct.sequence_logits), axis=-1)
  picked = np.asarray(direct_prob)[np.arange(length), np.asarray(direct.sequence)]
  np.testing.assert_array_equal(
    np.asarray(arrays["seq_log_prob"][0]),
    np.asarray(picked, dtype=np.float32),
  )


def _tied_options(partner: Path, knob: str) -> LaserOptions:
  text = str(partner)
  if knob == "seq_min_p":
    return LaserOptions(tied_second_input=text, seq_min_p=0.05)
  if knob == "chi_min_p":
    return LaserOptions(tied_second_input=text, chi_min_p=0.05)
  if knob == "fs_sequence_temp":
    return LaserOptions(tied_second_input=text, fs_sequence_temp=0.2)
  if knob == "disable_charged_fs":
    return LaserOptions(tied_second_input=text, disable_charged_fs=True)
  if knob == "ignore_chain_mask_zeros":
    return LaserOptions(tied_second_input=text, ignore_chain_mask_zeros=True)
  msg = f"unknown tied knob {knob}"
  raise ValueError(msg)


@pytest.mark.parametrize(
  "knob",
  [
    "seq_min_p",
    "chi_min_p",
    "fs_sequence_temp",
    "disable_charged_fs",
    "ignore_chain_mask_zeros",
  ],
)
def test_tied_refuses_knobs_tied_decode_rejects(
  registered: LaserDriver,
  tmp_path: Path,
  model_path: Path,
  knob: str,
) -> None:
  """(f) Knobs ``tied_decode`` refuses surface as ``ValueError``."""
  del registered
  pdb = tmp_path / "complex.pdb"
  _write_pdb(pdb)
  partner = tmp_path / "partner.pdb"
  partner.write_text(pdb.read_text(encoding="utf-8"), encoding="utf-8")
  options = _tied_options(partner, knob)
  with pytest.raises(ValueError, match=knob):
    sample(
      _spec(
        pdb,
        model_path,
        num_samples=1,
        temperature=None,
        random_seed=3,
        laser=options,
      ),
    )


def test_build_hydrogens_widens_sidechain_coords(
  registered: LaserDriver,
  tmp_path: Path,
) -> None:
  """(g) ``build_hydrogens`` selects the protonated atom axis."""
  del registered
  pdb = tmp_path / "complex.pdb"
  _write_pdb(pdb)
  spec = SamplingSpecification(
    inputs=str(pdb),
    model_family="lasermpnn",
    checkpoint_id="lasermpnn_test",
    num_samples=1,
    temperature=None,
    random_seed=3,
    laser=LaserOptions(disabled_residues=("X",)),
  )
  driver = LaserDriver()
  batch = next(item for item in driver.batches(spec) if item.input_indices)
  protonated = LASErMPNN(key=jax.random.PRNGKey(0), build_hydrogens=True)
  heavy = LASErMPNN(key=jax.random.PRNGKey(0), build_hydrogens=False)
  key = jax.random.PRNGKey(3)
  wide = driver.stages(spec, "sample", protonated)(
    batch,
    chunk_start=0,
    chunk_count=1,
    scalar_dropout=False,
    dropout_key=key,
  )
  narrow = driver.stages(spec, "sample", heavy)(
    batch,
    chunk_start=0,
    chunk_count=1,
    scalar_dropout=False,
    dropout_key=key,
  )
  assert np.asarray(wide["sidechain_coords"]).shape[-2] == MAX_PROTONATED_ATOMS + 1
  assert np.asarray(narrow["sidechain_coords"]).shape[-2] == MAX_ATOMS
