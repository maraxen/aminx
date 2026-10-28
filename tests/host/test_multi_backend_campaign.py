"""Unit tests for multi-backend campaign planning, ThermoMPNNDriver, and weight hoisting."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from aminx.families.thermo_mpnn import ThermoMPNNDriver
from aminx.host.campaign import (
  BackendArmSpec,
  _checkpoint_identity,
  build_manifest_row,
  normalize_backend_arm,
  plan_campaign_manifest,
)
from aminx.host.family_driver import DRIVER_BACKED_FAMILY_MODULES, FAMILY_DRIVERS
from aminx.host.plan import HoistedTopologyExecutor, TopologyKey, extract_topology_key
from aminx.io import weights as weights_mod
from aminx.io.weights import LEGACY_ALIAS_MAP, get_topology_for_checkpoint, load_model
from aminx.model import Aminx
from aminx.run.specs import SamplingSpecification


def test_thermompnn_driver_registration_and_capabilities() -> None:
  """Verify ThermoMPNNDriver is registered and rejects unsupported fallback purposes."""
  assert "thermompnn" in DRIVER_BACKED_FAMILY_MODULES
  drv = FAMILY_DRIVERS.get("thermompnn")
  assert isinstance(drv, ThermoMPNNDriver)
  spec = SamplingSpecification(inputs=["dummy.pdb"], model_family="thermompnn")
  assert drv.handles(spec, "score:ddg")
  assert not drv.handles(spec, "sample")
  with pytest.raises(NotImplementedError):
    drv.sample(None, None)


def test_normalize_backend_arm_aliases_and_drivers() -> None:
  """Verify normalize_backend_arm resolves all 6 vanilla families and driver families."""
  for alias, expected_ckpt in [
    ("original", "proteinmpnn_v_48_020"),
    ("soluble", "solublempnn_v_48_020"),
    ("abmpnn", "abmpnn_v_48_020"),
    ("hypermpnn", "hypermpnn_v_48_020"),
    ("cyclicmpnn", "cyclicmpnn_v_48_010"),
    ("antibmpnn", "antibmpnn_v_48_000"),
    ("ligand", "ligandmpnn_v_32_010_25"),
    ("pottsmpnn", "pottsmpnn_vanilla_20"),
    ("lasermpnn", "lasermpnn_nothing_heldout"),
    ("thermompnn", "thermo_mpnn_v1_default"),
  ]:
    arm = normalize_backend_arm(alias)
    assert isinstance(arm, BackendArmSpec)
    assert arm.arm_id == alias
    assert arm.checkpoint_id == expected_ckpt


def test_single_backend_backward_compatible_row_hash(tmp_path: Path) -> None:
  """Verify backends=None produces byte-identical row hashes to legacy single-backend planning."""
  spec = SamplingSpecification(
    inputs=["dummy.pdb"],
    checkpoint_id="proteinmpnn_v_48_020",
    ligand_conditioning=False,
    sidechain_conditioning=False,
  )
  direct_row = build_manifest_row(
    spec,
    campaign_id="camp_compat",
    job_index=0,
    chunk_id=0,
    sample_start=0,
    sample_count=4,
    fixed_policy="none",
    state_weight_profile="equal",
    planner_version="planner_v1",
    dataset_fingerprint="fp",
    environment_image="img",
    git_sha="deadbeef",
    config_hash="cfg",
    job_id="camp_compat-job-0",
    backend_arm=None,
  )
  rows = plan_campaign_manifest(
    base_spec=spec,
    campaign_id="camp_compat",
    designs_per_library_type=4,
    samples_chunk_size=4,
    output_root=tmp_path,
    backends=None,
    git_sha="deadbeef",
    config_hash="cfg",
  )
  assert len(rows) == 4  # 2x2 ligand x sidechain grid
  assert rows[0]["manifest_row_hash"] == direct_row["manifest_row_hash"]
  assert "backend_arm_id" not in rows[0]


def test_multi_backend_campaign_planning_and_checkpoint_identity(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verify backends=[...] expands rows across arms with distinct hashes and resolves family checkpoints."""
  spec = SamplingSpecification(inputs=["dummy.pdb"], checkpoint_id="proteinmpnn_v_48_020")
  rows = plan_campaign_manifest(
    base_spec=spec,
    campaign_id="camp_multi",
    designs_per_library_type=4,
    samples_chunk_size=4,
    output_root=tmp_path,
    backends=["original", "soluble", "antibmpnn", "thermompnn"],
    git_sha="deadbeef",
  )
  assert len(rows) == 4 * 4  # 4 arms x 4 grid variants
  hashes = {r["manifest_row_hash"] for r in rows}
  assert len(hashes) == len(rows)
  arm_ids = {r["backend_arm_id"] for r in rows}
  assert arm_ids == {"original", "soluble", "antibmpnn", "thermompnn"}

  # Verify _checkpoint_identity resolves both packaged and family_weights.json checkpoints
  dummy_wt = tmp_path / "proteinmpnn_v_48_020.eqx.zst"
  dummy_wt.write_bytes(b"dummy")
  monkeypatch.setenv("AMINX_WEIGHTS_DIR", str(tmp_path))
  id_orig = _checkpoint_identity({"checkpoint_id": "proteinmpnn_v_48_020"})
  assert id_orig["checkpoint_id"] == "proteinmpnn_v_48_020"
  monkeypatch.delenv("AMINX_WEIGHTS_DIR")
  id_thermo = _checkpoint_identity({"checkpoint_id": "thermo_mpnn_v1_default"})
  assert id_thermo["route"] == "family_checkpoint_id"
  assert id_thermo["family"] == "thermompnn"

  with pytest.raises(ValueError, match="empty sequence"):
    plan_campaign_manifest(
      base_spec=spec,
      campaign_id="camp_empty",
      designs_per_library_type=4,
      samples_chunk_size=4,
      output_root=tmp_path,
      backends=[],
    )


def test_six_vanilla_mpnn_families_compile_once_via_hoisted_executor(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Prove eqx.partition/eqx.combine hoisting compiles once across all 6 vanilla MPNN weight families."""
  families = ("original", "soluble", "abmpnn", "hypermpnn", "cyclicmpnn", "antibmpnn")
  versions = {
    "original": "v_48_020",
    "soluble": "v_48_020",
    "abmpnn": "v_48_020",
    "hypermpnn": "v_48_020",
    "cyclicmpnn": "v_48_010",
    "antibmpnn": "v_48_000",
  }

  def _mock_load_weights(
    checkpoint_id: str = "proteinmpnn_v_48_020",
    local_path: str | Path | None = None,
    skeleton: Aminx | None = None,
  ) -> Aminx:
    del checkpoint_id, local_path
    assert skeleton is not None
    return skeleton

  monkeypatch.setattr(weights_mod, "load_weights", _mock_load_weights)

  models = [
    load_model(
      model_weights=fam,
      model_version=versions[fam],
    )
    for fam in families
  ]
  for fam in families:
    assert (fam, versions[fam]) in LEGACY_ALIAS_MAP
  keys = [extract_topology_key(m) for m in models]
  assert len(set(keys)) == 1
  shared_key: TopologyKey = keys[0]

  executor = HoistedTopologyExecutor()
  x_dummy = jnp.ones((8, 128), dtype=jnp.float32)

  def _step(model: object, x: jax.Array) -> jax.Array:
    w_out = getattr(model, "w_out")
    return x @ w_out.weight.T + w_out.bias

  outputs = [executor.run(m, _step, x_dummy, topology_key=shared_key) for m in models]
  assert len(outputs) == 6
  assert executor.compile_counts[shared_key] == 1
  assert executor.total_compiles() == 1
