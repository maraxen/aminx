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


def test_thermompnn_driver_dispatch(tmp_path: Path) -> None:
  """Verify ThermoMPNNDriver implements full FamilyDriver interface and scores single/ensemble/double ΔΔG."""
  assert "thermompnn" in DRIVER_BACKED_FAMILY_MODULES
  drv = FAMILY_DRIVERS.get("thermompnn")
  assert isinstance(drv, ThermoMPNNDriver)
  spec = SamplingSpecification(inputs=["dummy.pdb"], model_family="thermompnn")
  assert drv.handles(spec, "score:ddg")
  assert not drv.handles(spec, "sample")
  with pytest.raises(NotImplementedError):
    drv.sample(None, None)

  ubq_pdb = Path(__file__).resolve().parents[1] / "data" / "1ubq.pdb"
  prep = drv.prepare_structure(ubq_pdb)
  assert prep["coords"].shape == (76, 4, 3)
  assert prep["length"] == 76

  batch = drv.collate_batch([prep])
  assert batch.lengths == (76,)
  axes = drv.axes(spec, "score:ddg", batch)
  assert len(axes) == 2
  schema = drv.result_schema(spec, "score:ddg")
  assert "ddg" in schema

  from aminx.families.thermo_mpnn.model import ThermoMPNN, ThermoMPNNDouble

  key = jax.random.PRNGKey(0)
  m_single = ThermoMPNN(key=key)
  assert drv.mpnn_core(m_single) is m_single.mpnn
  stage_fn = drv.stages(spec, "score:ddg", m_single)
  out_single = stage_fn(
    batch, chunk_start=0, chunk_count=1, scalar_dropout=False, dropout_key=key,
  )
  assert out_single["ddg"].shape == (76, 21)

  out_dir = drv.write_outputs(tmp_path / "thermo_out.zarr", out_single, structure_id="1ubq")
  assert out_dir.exists()

  m_double = ThermoMPNNDouble(key=key)
  out_double = drv.score(
    m_double,
    jnp.asarray(prep["coords"][:8]),
    jnp.asarray(prep["mask"][:8]),
    jnp.asarray(prep["residue_index"][:8]),
    jnp.asarray(prep["chain_index"][:8]),
    jnp.asarray(prep["sequence"][:8]),
    double_mutant=True,
  )
  assert out_double["ddg"].shape == (8, 21)
  assert out_double["ddg_double"].shape == (7, 21, 21)
  assert out_double["pair_indices"].shape == (7, 2)


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


def test_multi_backend_manifest(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verify backends=[...] expands rows across arms with distinct hashes and resolves family checkpoints."""
  ubq_pdb = Path(__file__).resolve().parents[1] / "data" / "1ubq.pdb"
  spec = SamplingSpecification(inputs=[str(ubq_pdb)], checkpoint_id="proteinmpnn_v_48_020")
  rows = plan_campaign_manifest(
    base_spec=spec,
    campaign_id="camp_multi",
    designs_per_library_type=4,
    samples_chunk_size=4,
    output_root=tmp_path,
    backends=["original", "soluble", "antibmpnn", "thermompnn"],
    git_sha="deadbeef",
  )
  # 3 vanilla MPNN arms x 4 grid variants + 1 driver-backed thermompnn arm x 1 variant = 13 rows
  assert len(rows) == 3 * 4 + 1
  thermo_rows = [r for r in rows if r["backend_arm_id"] == "thermompnn"]
  assert len(thermo_rows) == 1
  assert thermo_rows[0]["sampling_spec"]["ligand_conditioning"] is False
  assert thermo_rows[0]["sampling_spec"]["sidechain_conditioning"] is False
  hashes = {r["manifest_row_hash"] for r in rows}
  assert len(hashes) == len(rows)
  arm_ids = {r["backend_arm_id"] for r in rows}
  assert arm_ids == {"original", "soluble", "antibmpnn", "thermompnn"}

  # Verify execute_manifest dispatches thermompnn row via score() and writes to UnifiedCampaignSink
  import aminx.families.thermo_mpnn.convert as thermo_convert_mod
  from aminx.host.campaign import execute_manifest
  from aminx.io.campaign_sink import CampaignSinkInspector

  monkeypatch.setattr(
    thermo_convert_mod,
    "load_weights",
    lambda checkpoint_id="thermo_mpnn_v1_default", local_path=None, skeleton=None: skeleton,
  )
  sink_dir = tmp_path / "campaign_sink.zarr"
  executor = HoistedTopologyExecutor()
  exec_results = execute_manifest(
    thermo_rows,
    sink_root=sink_dir,
    hoisted_executor=executor,
  )
  assert exec_results["successful_rows"] == 1
  insp = CampaignSinkInspector(sink_dir)
  assert "thermompnn" in insp.summary().arms

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


def test_topology_key_weight_hoisting(
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
  ckpt_topo = get_topology_for_checkpoint("proteinmpnn_v_48_020")
  assert shared_key.k_neighbors == ckpt_topo["k_neighbors"]
  assert shared_key.num_positional_embeddings == ckpt_topo["num_positional_embeddings"]
  assert shared_key.physics_feature_dim == ckpt_topo["physics_feature_dim"]
  assert ckpt_topo["model_type"] == "protein"

  # Verify distinct static fields produce distinct TopologyKeys
  m_phys = Aminx(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    k_neighbors=48,
    physics_feature_dim=5,
    key=jax.random.PRNGKey(9),
  )
  assert extract_topology_key(m_phys) != shared_key

  executor = HoistedTopologyExecutor()
  x_dummy = jnp.ones((8, 128), dtype=jnp.float32)

  def _step(model: object, x: jax.Array) -> jax.Array:
    w_out = getattr(model, "w_out")
    return x @ w_out.weight.T + w_out.bias

  outputs = [executor.run(m, _step, x_dummy, topology_key=shared_key) for m in models]
  assert len(outputs) == 6
  assert executor.compile_counts[shared_key] == 1
  assert executor.total_compiles() == 1
