"""Shape classes compile once per process (debt #3510).

tev_design's design screen runs ~1,500 ``score_conditionals`` units. On 0.5.4:
- every distinct LigandMPNN ligand atom count M recompiled the encode and score kernels (one
  compile per substrate P1' residue in tev's smoke);
- every distinct sequences-per-unit N did as well.

What this file pins:

* the ladders, and that the atom-axis padding is end-padding with mask 0;
* padded ligand atoms carry no weight: padded == unpadded within SCORE_BATCH_COMPOSITION_ATOL for
  logits, NLL and the ligand-proximal loss, below and above ``atom_context_num`` (25), even when
  the padded atoms have nonzero coordinates and real atom types;
* N padding is dropped from the store, the original N is recorded, and real rows are unchanged
  within the same bound;
* ``execute_manifest`` runs rows grouped by shape class but reports in manifest order, and done
  markers and resume are unaffected;
* regression: after the first unit of a class, units in that class compile zero times. A positive
  control with the ligand ladder disabled shows that the counter sees the recompiles the ladder
  removes.

task_id: 261119_aminx-3510-compile-shape-classes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import aminx.host.campaign as campaign
from aminx.host import _sampling_helper
from aminx.host.bucketing import (
  ligand_atom_rung,
  pad_ligand_atom_axis,
  sequence_count_rung,
)
from aminx.host.campaign import execute_manifest, row_compile_key, write_manifest
from aminx.host.campaign_score import (
  ScoreConditionalsUnit,
  plan_score_conditionals_manifest,
  read_score_conditionals_store,
)
from aminx.host.runner import score
from aminx.run.specs import ScoringSpecification
from aminx.scoring.score import SCORE_BATCH_COMPOSITION_ATOL

_STRUCTURE = "tests/data/1ubq.pdb"
_CHECKPOINT = "proteinmpnn_v_48_020"
_LIGAND_CHECKPOINT = "ligandmpnn_v_32_020_25"
_UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
_L = len(_UBIQUITIN)
_MAX_LENGTH = 96
_PEPTIDE = slice(_L - 10, _L)


def _designable() -> np.ndarray:
  design = np.ones(_L, np.int8)
  design[_PEPTIDE] = 0
  return design


def _masks(k: int, seed: int = 0) -> np.ndarray:
  """Random-order causal masks over the designable block; every row sees the peptide."""
  design = _designable().astype(bool)
  rng = np.random.default_rng(seed)
  out = np.zeros((k, _L, _L), np.int8)
  idx = np.flatnonzero(design)
  for j in range(k):
    order = np.full(_L, -1)
    order[idx] = rng.permutation(idx.size)
    see = (order[None, :] < order[:, None]) & design[:, None] & design[None, :]
    see |= ~design[None, :]
    see |= (~design[:, None]) & design[None, :]
    np.fill_diagonal(see, False)
    out[j] = see.astype(np.int8)
  return out


def _mutants(n: int) -> list[str]:
  letters = "ACDEFGHIKLMNPQRSTVWY"
  return [_UBIQUITIN[:5] + letters[i % 20] + _UBIQUITIN[6:] for i in range(n)]


def _unit(tmp_path: Path, **overrides: Any) -> ScoreConditionalsUnit:  # noqa: ANN401
  masks_path = overrides.pop("ar_masks_path", None)
  if masks_path is None:
    masks_path = str(tmp_path / "masks.npy")
    np.save(masks_path, _masks(2))
  fields: dict[str, Any] = {
    "label": "centre",
    "structure": _STRUCTURE,
    "checkpoint_id": _CHECKPOINT,
    "sequences": _mutants(2),
    "ar_masks_path": masks_path,
    "designable_mask": _designable().tolist(),
    "logits_memory_budget_mb": 64,
    "max_length": _MAX_LENGTH,
  }
  fields.update(overrides)
  return ScoreConditionalsUnit(**fields)


def _plan(root: Path, *units: ScoreConditionalsUnit) -> list[dict[str, Any]]:
  return plan_score_conditionals_manifest(
    list(units), campaign_id="sc", output_root=root / "out", git_sha="0" * 40, git_provenance_source="explicit",
  )


def _manifest(root: Path, rows: list[dict[str, Any]]) -> Path:
  path = root / "manifest.json"
  write_manifest(path, rows, metadata={"campaign_id": "sc"})
  return path


def _ligand_npz(path: Path, n_atoms: int, *, n_masked: int = 0, seed: int = 0) -> str:
  """A ligand context of ``n_atoms`` real atoms, then ``n_masked`` mask-0 atoms with real-looking data."""
  from aminx.host._sampling_helper import _canonical_structure_ids_for_spec  # noqa: PLC0415

  (structure_id,) = _canonical_structure_ids_for_spec(
    ScoringSpecification(inputs=[_STRUCTURE], sequences_to_score=[_UBIQUITIN], max_length=_MAX_LENGTH),
  )
  rng = np.random.default_rng(seed)
  total = n_atoms + n_masked
  y = rng.normal(scale=6.0, size=(_MAX_LENGTH, total, 3)).astype(np.float32)
  y_t = np.full((_MAX_LENGTH, total), 6, dtype=np.int32)
  y_m = np.zeros((_MAX_LENGTH, total), dtype=np.float32)
  y_m[:, :n_atoms] = 1.0
  np.savez(path, **{f"{structure_id}::Y": y, f"{structure_id}::Y_t": y_t, f"{structure_id}::Y_m": y_m})
  return str(path)


# --- ladders and padding (no model) ----------------------------------------------------------------


def test_ladders() -> None:
  assert [ligand_atom_rung(m) for m in (0, 1, 8, 16, 17, 25, 33, 1024, 1025)] == [
    16, 16, 16, 16, 32, 32, 64, 1024, 2048,
  ]
  assert [sequence_count_rung(n) for n in (1, 2, 3, 5, 17, 25, 600)] == [1, 2, 4, 8, 24, 32, 1024]
  with pytest.raises(ValueError, match="non-negative"):
    ligand_atom_rung(-1)


def test_ligand_padding_is_end_padding_with_mask_zero() -> None:
  rng = np.random.default_rng(0)
  y = rng.normal(size=(2, 5, 3, 3)).astype(np.float32)
  y_t = np.ones((2, 5, 3), np.int32)
  y_m = np.ones((2, 5, 3), np.float32)
  py, pt, pm = pad_ligand_atom_axis(y, y_t, y_m, 16)
  assert py.shape == (2, 5, 16, 3)
  assert pt.shape == pm.shape == (2, 5, 16)
  np.testing.assert_array_equal(py[:, :, :3], y)
  np.testing.assert_array_equal(pt[:, :, :3], y_t)
  np.testing.assert_array_equal(pm[:, :, :3], y_m)
  assert not pm[:, :, 3:].any()
  assert not py[:, :, 3:].any()


# --- ordering, report and resume -------------------------------------------------------------------


def test_row_compile_key_never_raises(tmp_path: Path) -> None:
  rows = _plan(tmp_path, _unit(tmp_path))
  broken = json.loads(json.dumps(rows[0]))
  broken["unit_spec"]["ar_masks_path"] = str(tmp_path / "missing.npy")
  assert row_compile_key(broken) is None
  key = row_compile_key(rows[0])
  assert key is not None
  assert key[4:] == (2, 2, 0)  # K, N bucket, no ligand


def test_execute_manifest_groups_by_class_and_reports_in_manifest_order(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  units = [
    _unit(tmp_path, label="a1", sequences=_mutants(1)),
    _unit(tmp_path, label="b3", sequences=_mutants(3)),
    _unit(tmp_path, label="a1b", sequences=[_mutants(2)[1]]),
    _unit(tmp_path, label="b4", sequences=_mutants(4)),
  ]
  rows = _plan(tmp_path, *units)
  order: list[str] = []

  def fake_run(_manifest: object, *, row_hash: str, **_kwargs: object) -> dict[str, Any]:
    order.append(row_hash)
    return {"status": "completed", "compile_telemetry": {"backend_compiles": 5 if len(order) in (1, 3) else 0}}

  monkeypatch.setattr(campaign, "run_manifest_row", fake_run)
  report = execute_manifest(rows)
  hashes = [r["manifest_row_hash"] for r in rows]
  assert order == [hashes[0], hashes[2], hashes[1], hashes[3]]
  assert [r["manifest_row_hash"] for r in report["row_results"]] == hashes
  summary = {entry["computed_rows"]: entry for entry in report["shape_classes"]}
  assert len(report["shape_classes"]) == 2
  assert all(e["first_unit_compiles"] == 5 and e["later_unit_compiles"] == 0 for e in summary.values())

  order.clear()
  execute_manifest(rows, group_by_shape_class=False)
  assert order == hashes


def test_resume_is_unaffected_by_grouping(tmp_path: Path) -> None:
  units = [
    _unit(tmp_path, label="n1", sequences=_mutants(1)),
    _unit(tmp_path, label="n3", sequences=_mutants(3)),
    _unit(tmp_path, label="n1b", sequences=[_mutants(2)[1]]),
  ]
  rows = _plan(tmp_path, *units)
  manifest = _manifest(tmp_path, rows)
  partial = execute_manifest(manifest, row_hashes=[rows[1]["manifest_row_hash"]])
  assert partial["successful_rows"] == 1
  marker_path = Path(rows[1]["output_h5_path"] + ".done.json")
  before = json.loads(marker_path.read_text())

  full = execute_manifest(manifest)
  decisions = {r["manifest_row_hash"]: (r["reuse"] or {}).get("decision") for r in full["row_results"]}
  assert decisions[rows[1]["manifest_row_hash"]] == "reused"
  assert full["recomputed_rows"] == 0
  assert full["successful_rows"] == len(rows)
  after = json.loads(marker_path.read_text())
  assert after["content_digest_sha256"] == before["content_digest_sha256"]

  telemetry = before["compile_telemetry"]
  assert telemetry["schema_version"] == "campaign_compile_telemetry_v1"
  assert isinstance(telemetry["backend_compiles"], int)
  assert telemetry["shape_class"] == json.dumps([str(p) for p in row_compile_key(rows[1])])
  assert "enabled" in telemetry["jax_compilation_cache"]
  # Telemetry never reaches the store (it would make identical runs digest differently).
  attrs = read_score_conditionals_store(rows[1]["output_h5_path"])["attrs"]
  assert "compile_telemetry" not in attrs
  assert attrs["n_sequences"] == 3
  assert attrs["n_sequences_bucket"] == 4


# --- numerics ---------------------------------------------------------------------------------------


def test_sequence_padding_leaves_real_rows_unchanged(tmp_path: Path) -> None:
  sequences = _mutants(3)  # bucket 4: one padded copy
  rows = _plan(tmp_path, _unit(tmp_path, sequences=sequences))
  execute_manifest(_manifest(tmp_path, rows))
  store = read_score_conditionals_store(rows[0]["output_h5_path"])
  assert store["logits"].shape == (2, 3, _L, 21)
  assert store["nll"].shape == (2, 3)
  assert store["attrs"]["sequences"] == sequences
  direct = score(
    ScoringSpecification(
      inputs=[_STRUCTURE], checkpoint_id=_CHECKPOINT, sequences_to_score=sequences, ar_masks=_masks(2),
      designable_mask=_designable(), return_logits=True, backbone_noise=0.0, max_length=_MAX_LENGTH,
    ),
  )
  np.testing.assert_allclose(
    store["logits"], np.asarray(direct["logits"])[0][..., :_L, :], atol=SCORE_BATCH_COMPOSITION_ATOL, rtol=0,
  )
  np.testing.assert_allclose(store["nll"], np.asarray(direct["scores"])[0], atol=SCORE_BATCH_COMPOSITION_ATOL, rtol=0)


def _ligand_score(lig: str, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
  return score(
    ScoringSpecification(
      inputs=[_STRUCTURE], checkpoint_id=_LIGAND_CHECKPOINT, sequences_to_score=[_UBIQUITIN],
      ligand_context_path=lig, return_logits=True, backbone_noise=0.0, max_length=_MAX_LENGTH, **kwargs,
    ),
  )


def _assert_same_score(a: dict[str, Any], b: dict[str, Any], *, loss_xy: bool = False) -> None:
  np.testing.assert_allclose(np.asarray(a["logits"]), np.asarray(b["logits"]), atol=SCORE_BATCH_COMPOSITION_ATOL, rtol=0)
  np.testing.assert_allclose(np.asarray(a["scores"]), np.asarray(b["scores"]), atol=SCORE_BATCH_COMPOSITION_ATOL, rtol=0)
  if loss_xy:
    np.testing.assert_array_equal(np.asarray(a["mask_xy"]), np.asarray(b["mask_xy"]))
    np.testing.assert_allclose(np.asarray(a["loss_xy"]), np.asarray(b["loss_xy"]), atol=SCORE_BATCH_COMPOSITION_ATOL, rtol=0)


@pytest.mark.slow
@pytest.mark.requires_weights
@pytest.mark.parametrize("n_atoms", [8, 27])  # below and above atom_context_num=25
def test_ligand_atom_padding_carries_no_weight(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_atoms: int,
) -> None:
  lig = _ligand_npz(tmp_path / "lig.npz", n_atoms)
  padded = _ligand_score(lig, return_loss_xy=True)
  # Same residue trimming, ligand ladder off: the unpadded reference.
  monkeypatch.setattr(_sampling_helper, "ligand_atom_rung", lambda m: m)
  unpadded = _ligand_score(lig, return_loss_xy=True)
  _assert_same_score(padded, unpadded, loss_xy=True)
  # Planted: mask-0 atoms with REAL coordinates and atom types, not zeros, still carry no weight.
  planted = _ligand_score(_ligand_npz(tmp_path / "planted.npz", n_atoms, n_masked=9), return_loss_xy=True)
  _assert_same_score(planted, unpadded, loss_xy=True)
  # Instrument control: a real (mask-1) atom change does move the logits.
  moved = _ligand_score(_ligand_npz(tmp_path / "moved.npz", n_atoms, seed=1))
  assert np.abs(np.asarray(moved["logits"]) - np.asarray(unpadded["logits"])).max() > 1e-3


# --- regression: one compile per shape class, zero within it -----------------------------------------


def _class_rows(root: Path) -> list[dict[str, Any]]:
  root.mkdir(parents=True, exist_ok=True)
  masks_path = str(root / "masks.npy")
  np.save(masks_path, _masks(2, seed=3))
  lig20 = _ligand_npz(root / "lig20.npz", 20)
  lig27 = _ligand_npz(root / "lig27.npz", 27, seed=1)
  lig40 = _ligand_npz(root / "lig40.npz", 40, seed=2)

  def unit(label: str, lig: str, n: int) -> ScoreConditionalsUnit:
    return _unit(
      root, label=label, checkpoint_id=_LIGAND_CHECKPOINT, ar_masks_path=masks_path, ligand_context_path=lig,
      sequences=_mutants(n),
    )

  # Class A = (ligand rung 32, N bucket 4): rows 0, 2, 3. Class B = (ligand rung 64): row 1.
  return _plan(root, unit("a20n3", lig20, 3), unit("b40n4", lig40, 4), unit("a27n4", lig27, 4), unit("a20n4", lig20, 4))


@pytest.mark.slow
@pytest.mark.requires_weights
def test_one_compile_per_shape_class(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  rows = _class_rows(tmp_path / "bucketed")
  keys = [row_compile_key(r) for r in rows]
  assert keys[0] == keys[2] == keys[3] != keys[1]
  report = execute_manifest(_manifest(tmp_path / "bucketed", rows))
  assert report["successful_rows"] == 4
  hashes = [r["manifest_row_hash"] for r in rows]
  executed = [r["manifest_row_hash"] for r in sorted(report["row_results"], key=lambda r: r["execution_position"])]
  assert executed == [hashes[0], hashes[2], hashes[3], hashes[1]]
  by_rows = {e["computed_rows"]: e for e in report["shape_classes"]}
  assert by_rows[3]["later_unit_compiles"] == 0, report["shape_classes"]
  for index in (2, 3):
    marker = json.loads(Path(rows[index]["output_h5_path"] + ".done.json").read_text())
    assert marker["compile_telemetry"]["backend_compiles"] == 0
    assert marker["compile_telemetry"]["first_unit_of_class_in_process"] is False

  # Positive control: the same rows with the ligand ladder off recompile inside "class A" (M=20 vs
  # M=27 are different shapes now), so the zero above is the ladder's doing, not a blind counter.
  monkeypatch.setattr(_sampling_helper, "ligand_atom_rung", lambda m: m)
  control = _class_rows(tmp_path / "control")
  control_report = execute_manifest(_manifest(tmp_path / "control", control))
  marker = json.loads(Path(control[2]["output_h5_path"] + ".done.json").read_text())
  assert marker["compile_telemetry"]["backend_compiles"] > 0, control_report["shape_classes"]
