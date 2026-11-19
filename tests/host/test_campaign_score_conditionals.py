"""``task="score_conditionals"`` campaign rows and batched ``ar_masks`` scoring (#3486, #3485).

What this file pins:

* existing row kinds hash EXACTLY as on origin/main 86a41a03 (golden values computed there), so
  live campaigns resume instead of mass-quarantining;
* the planner refuses what has produced silently wrong screens: token ids instead of letters,
  sequences of different lengths, an all-zero mask, a designable position with an empty mask row,
  a logits payload over budget, a ligand on a checkpoint with no ligand channel;
* the unit input hash covers the BYTES of the mask stack and the ligand context;
* ``ar_masks`` (K, L, L) in one call equals K single-mask calls (batched == looped) within the
  batch-composition bound measured for #3487, and is bitwise invariant to the seed;
* a row runs end to end through ``run_manifest_row``, resumes as ``already_done``, passes the
  gates, and is refused by the packed runner;
* knob differential for the new path: identical input -> exactly 0; a changed context ("P1'")
  token -> a nonzero change at the positions that see it; a changed seed -> exactly 0.

task_id: 261118_aminx-campaign-teacher-forced-scoring
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.host.campaign import (
  _collect_manifest_gate_state,
  compute_unit_input_hash,
  plan_campaign_manifest,
  run_manifest_row,
  write_manifest,
)
from aminx.host.campaign_score import (
  ScoreConditionalsUnit,
  plan_score_conditionals_manifest,
  read_score_conditionals_store,
)
from aminx.host.runner import score
from aminx.run.specs import SamplingSpecification, ScoringSpecification
from aminx.utils.autoregression import ar_mask_from_decoding_order

_STRUCTURE = "tests/data/1ubq.pdb"
_CHECKPOINT = "proteinmpnn_v_48_020"
_UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
_L = len(_UBIQUITIN)
_MAX_LENGTH = 96
# The last ten residues play the peptide: fixed context every row sees, never designed.
_PEPTIDE = slice(_L - 10, _L)
_P1_PRIME = _L - 3
# #3487 (run 7e50cce9, CPU, default precision): alone-vs-batched |delta logit| <= 1.9e-6 and
# content-invariant bitwise. Batched-vs-looped here is the same effect; 5x margin.
_BATCH_ATOL = 1e-5
_MOVED = 1e-3

# Golden hashes for EXISTING row kinds, computed on origin/main 86a41a03 with the planner calls in
# _golden_rows(). A change here re-keys every live campaign's resume.
_GOLDEN = {
  "sampling": (
    "927736fd2bbbff7d8e0b7084b0b8260ccadc2ead571db99117c49e31af0b1adf",
    "a9ea7c530fdcd71c2d3a6747dcc05050d87240b4700824df49ec0fc3ef844067",
  ),
  "thermompnn": (
    "56de24863539070943ac73a492b394bc267774af99938b1f98cc3aad30ca3c57",
    "0ee80ba114a502a7fd35cc63524c0356b092e4441db14f00fce3ad5b10712a2c",
  ),
}


def _golden_rows() -> dict[str, dict[str, Any]]:
  base = SamplingSpecification(
    inputs=["golden/ref.pdb"], checkpoint_id=_CHECKPOINT, random_seed=3, temperature=0.1,
  )
  common = {"campaign_id": "golden", "output_root": "/golden/out", "git_sha": "0" * 40, "git_provenance_source": "explicit"}
  sampling = plan_campaign_manifest(base_spec=base, designs_per_library_type=2, samples_chunk_size=2, **common)[0]
  thermo = plan_campaign_manifest(
    base_spec=base, designs_per_library_type=1, samples_chunk_size=1, backends=["thermompnn"], **common,
  )[0]
  return {"sampling": sampling, "thermompnn": thermo}


@pytest.mark.parametrize("kind", ["sampling", "thermompnn"])
def test_existing_row_kinds_hash_exactly_as_on_main(kind: str) -> None:
  row = _golden_rows()[kind]
  assert "task" not in row
  assert row["manifest_row_hash"] == _GOLDEN[kind][0]
  assert compute_unit_input_hash(row["sampling_spec"])[0] == _GOLDEN[kind][1]


# --- masks and units ------------------------------------------------------------------------------


def _designable() -> np.ndarray:
  design = np.ones(_L, np.int8)
  design[_PEPTIDE] = 0
  return design


def _masks(k: int, seed: int = 0) -> np.ndarray:
  """K masks in the shape of tev_design's rule, on 1ubq with its last ten residues as the peptide.

  Every row sees every peptide column; a designable row also sees the designable positions earlier
  in that mask's random master order; a peptide row also sees the whole designable block.
  """
  design = _designable().astype(bool)
  rng = np.random.default_rng(seed)
  out = np.zeros((k, _L, _L), np.int8)
  idx = np.flatnonzero(design)
  for j in range(k):
    order = np.full(_L, -1)
    order[idx] = rng.permutation(idx.size)
    rank = np.where(design, order, -1)
    see = (rank[None, :] < rank[:, None]) & design[:, None] & design[None, :]
    see |= ~design[None, :]  # every row sees the peptide columns
    see |= (~design[:, None]) & design[None, :]  # peptide rows see the designable block
    np.fill_diagonal(see, False)
    out[j] = see.astype(np.int8)
  return out


def _write_masks(path: Path, masks: np.ndarray) -> str:
  np.save(path, masks)
  return str(path)


def _with_token(seq: str, pos: int, token: str) -> str:
  return seq[:pos] + token + seq[pos + 1 :]


def _unit(tmp_path: Path, **overrides: Any) -> ScoreConditionalsUnit:
  masks_path = overrides.pop("ar_masks_path", None) or _write_masks(tmp_path / "masks.npy", _masks(3))
  fields: dict[str, Any] = {
    "label": "centre",
    "structure": _STRUCTURE,
    "checkpoint_id": _CHECKPOINT,
    "sequences": [_UBIQUITIN, _with_token(_UBIQUITIN, 5, "A")],
    "ar_masks_path": masks_path,
    "designable_mask": _designable().tolist(),
    "logits_memory_budget_mb": 64,
    "max_length": _MAX_LENGTH,
  }
  fields.update(overrides)
  return ScoreConditionalsUnit(**fields)


def _plan(tmp_path: Path, *units: ScoreConditionalsUnit) -> list[dict[str, Any]]:
  return plan_score_conditionals_manifest(
    list(units), campaign_id="sc", output_root=tmp_path / "out", git_sha="0" * 40, git_provenance_source="explicit",
  )


# --- planner refusals -----------------------------------------------------------------------------


def test_token_ids_are_refused(tmp_path: Path) -> None:
  with pytest.raises(TypeError, match="letter strings"):
    _plan(tmp_path, _unit(tmp_path, sequences=[[0, 1, 2]]))


def test_non_mpnn_letters_and_mixed_lengths_are_refused(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match="non-MPNN letters"):
    _plan(tmp_path, _unit(tmp_path, sequences=[_UBIQUITIN.replace("M", "B", 1)]))
  with pytest.raises(ValueError, match="one length"):
    _plan(tmp_path, _unit(tmp_path, sequences=[_UBIQUITIN, _UBIQUITIN[:-1]]))


def test_zeroed_mask_is_refused(tmp_path: Path) -> None:
  masks = _masks(2)
  masks[1] = 0
  with pytest.raises(ValueError, match="all zero"):
    _plan(tmp_path, _unit(tmp_path, ar_masks_path=_write_masks(tmp_path / "z.npy", masks)))


def test_empty_row_at_a_designable_position_is_refused(tmp_path: Path) -> None:
  masks = _masks(2)
  masks[0, 4, :] = 0  # position 4 is designable
  with pytest.raises(ValueError, match="designable position 4"):
    _plan(tmp_path, _unit(tmp_path, ar_masks_path=_write_masks(tmp_path / "e.npy", masks)))


def test_self_visible_mask_is_refused(tmp_path: Path) -> None:
  masks = _masks(1)
  masks[0, 3, 3] = 1
  with pytest.raises(ValueError, match="zero diagonal"):
    _plan(tmp_path, _unit(tmp_path, ar_masks_path=_write_masks(tmp_path / "d.npy", masks)))


def test_logits_budget_is_enforced_at_plan_time(tmp_path: Path) -> None:
  # 3 masks x 2 seqs x 76 x 21 x 4 B = 38 KiB: 1 MiB passes, a budget below it must fail.
  _plan(tmp_path, _unit(tmp_path, logits_memory_budget_mb=1))
  big = _write_masks(tmp_path / "big.npy", np.repeat(_masks(1), 400, axis=0))
  with pytest.raises(ValueError, match="logits_memory_budget_mb=1"):
    _plan(tmp_path, _unit(tmp_path, ar_masks_path=big, logits_memory_budget_mb=1))


def test_ligand_on_a_protein_checkpoint_is_refused(tmp_path: Path) -> None:
  lig = tmp_path / "lig.npz"
  np.savez(lig, x=np.zeros(1))
  with pytest.raises(ValueError, match="no ligand channel"):
    _plan(tmp_path, _unit(tmp_path, ligand_context_path=str(lig)))


def test_p1_prime_change_is_a_different_row_sharing_the_masks(tmp_path: Path) -> None:
  a = _unit(tmp_path)
  b = _unit(tmp_path, label="P1p_A", ar_masks_path=a.ar_masks_path,
            sequences=[_with_token(s, _P1_PRIME, "A") for s in a.sequences])
  ra, rb = _plan(tmp_path, a, b)
  assert ra["manifest_row_hash"] != rb["manifest_row_hash"]
  assert ra["unit_spec"]["ar_masks_sha256"] == rb["unit_spec"]["ar_masks_sha256"]
  assert ra["task"] == "score_conditionals"
  assert "sampling_spec" not in ra


def test_input_hash_covers_mask_and_ligand_bytes(tmp_path: Path) -> None:
  (row,) = _plan(tmp_path, _unit(tmp_path))
  payload = row["unit_spec"]
  before, components = compute_unit_input_hash(payload)
  assert "task_files" in components
  np.save(payload["ar_masks_path"], _masks(3, seed=99))  # same path, new bytes
  assert compute_unit_input_hash(payload)[0] != before

  lig = tmp_path / "lig.npz"
  np.savez(lig, x=np.zeros(3))
  with_lig = dict(payload, ligand_context_path=str(lig))
  h1 = compute_unit_input_hash(with_lig)[0]
  np.savez(lig, x=np.ones(3))
  assert compute_unit_input_hash(with_lig)[0] != h1


# --- runner: batched ar_masks ---------------------------------------------------------------------


def _spec(**kwargs: Any) -> ScoringSpecification:
  return ScoringSpecification(
    inputs=[_STRUCTURE], checkpoint_id=_CHECKPOINT, max_length=_MAX_LENGTH, return_logits=True,
    backbone_noise=0.0, **kwargs,
  )


def test_ar_mask_and_ar_masks_are_exclusive() -> None:
  m = _masks(2)
  with pytest.raises(ValueError, match="not both"):
    _spec(sequences_to_score=[_UBIQUITIN], ar_mask=m[0], ar_masks=m)


@pytest.mark.parametrize("checkpoint", [_CHECKPOINT, "solublempnn_v_48_020"])
def test_batched_masks_equal_looped_masks(checkpoint: str) -> None:
  """(K, L, L) in one call == one call per mask (#3485), within the #3487 bound."""
  masks = _masks(3)
  seqs = [_UBIQUITIN, _with_token(_UBIQUITIN, 5, "A")]

  def _ck(**kw: Any) -> ScoringSpecification:
    return ScoringSpecification(
      inputs=[_STRUCTURE], checkpoint_id=checkpoint, max_length=_MAX_LENGTH, return_logits=True,
      backbone_noise=0.0, **kw,
    )

  batched = score(_ck(sequences_to_score=seqs, ar_masks=masks))
  logits = np.asarray(batched["logits"])
  assert logits.shape[:3] == (1, 3, 2)
  assert np.asarray(batched["scores"]).shape == (1, 3, 2)
  for k in range(3):
    looped = np.asarray(score(_ck(sequences_to_score=seqs, ar_mask=masks[k]))["logits"])
    np.testing.assert_allclose(logits[0, k, :, :_L], looped[0, :, :_L], rtol=0, atol=_BATCH_ATOL)


def test_batched_masks_are_seed_invariant() -> None:
  masks = _masks(2)
  a = np.asarray(score(_spec(sequences_to_score=[_UBIQUITIN], ar_masks=masks, random_seed=0))["logits"])
  b = np.asarray(score(_spec(sequences_to_score=[_UBIQUITIN], ar_masks=masks, random_seed=12345))["logits"])
  np.testing.assert_array_equal(a, b)


def test_causal_masks_are_strictly_causal_in_the_batched_path() -> None:
  """A token change at order rank r moves no logit at ranks <= r (real weights, batched path)."""
  order = np.random.default_rng(3).permutation(_L).astype(np.int32)
  causal = np.asarray(ar_mask_from_decoding_order(jnp.asarray(order)))[None].repeat(2, 0)
  pos = int(order[_L // 2])
  seqs = [_UBIQUITIN, _with_token(_UBIQUITIN, pos, "W" if _UBIQUITIN[pos] != "W" else "A")]
  out = np.asarray(score(_spec(sequences_to_score=seqs, ar_masks=causal))["logits"])[0, 0]
  rank = np.empty(_L, int)
  rank[order] = np.arange(_L)
  earlier = rank <= rank[pos]
  assert np.abs(out[0, :_L][earlier] - out[1, :_L][earlier]).max() <= _BATCH_ATOL
  assert np.abs(out[0, :_L][~earlier] - out[1, :_L][~earlier]).max() > _MOVED


# --- campaign end to end --------------------------------------------------------------------------


def _run(tmp_path: Path, rows: list[dict[str, Any]], index: int = 0) -> dict[str, Any]:
  manifest = tmp_path / "manifest.json"
  write_manifest(manifest, rows, metadata={"campaign_id": "sc"})
  return run_manifest_row(manifest, row_index=index)


def test_row_runs_resumes_and_passes_the_gates(tmp_path: Path) -> None:
  rows = _plan(tmp_path, _unit(tmp_path))
  first = _run(tmp_path, rows)
  assert first["status"] == "completed"
  store = read_score_conditionals_store(first["output_h5_path"])
  assert store["logits"].shape == (3, 2, _L, 21)
  assert store["nll"].shape == (3, 2)
  assert store["attrs"]["schema_version"] == "score_conditionals_v1"
  assert store["attrs"]["sequences"] == rows[0]["unit_spec"]["sequences"]

  # The store is exactly what runner.score returns for the same unit.
  direct = score(_spec(sequences_to_score=list(rows[0]["unit_spec"]["sequences"]), ar_masks=_masks(3),
                       designable_mask=_designable()))
  np.testing.assert_array_equal(store["logits"], np.asarray(direct["logits"])[0][..., :_L, :])

  again = _run(tmp_path, rows)
  assert again["status"] == "already_done"
  assert again["reuse"]["reason"] == "stamp_verified"

  state = _collect_manifest_gate_state(tmp_path / "manifest.json")
  assert state.completed_rows == 1
  assert not state.metadata_issues
  assert not state.runtime_issues


def test_execute_manifest_runs_task_rows(tmp_path: Path) -> None:
  """The entry point the docs give consumers: execute_manifest over a mixed-condition manifest."""
  from aminx.host.campaign import execute_manifest  # noqa: PLC0415

  masks_path = _write_masks(tmp_path / "shared.npy", _masks(2))
  rows = _plan(
    tmp_path,
    _unit(tmp_path, ar_masks_path=masks_path),
    _unit(tmp_path, label="P1p_W", ar_masks_path=masks_path,
          sequences=[_with_token(s, _P1_PRIME, "W") for s in (_UBIQUITIN, _with_token(_UBIQUITIN, 5, "A"))]),
  )
  manifest = tmp_path / "manifest.json"
  write_manifest(manifest, rows, metadata={"campaign_id": "sc"})
  summary = execute_manifest(manifest)
  assert summary["selected_rows"] == 2
  assert summary["successful_rows"] == 2
  assert summary["failed_rows"] == 0
  for row in rows:
    store = read_score_conditionals_store(row["output_h5_path"])
    assert store["logits"].shape == (2, 2, _L, 21)
  again = execute_manifest(manifest)
  assert again["reused_rows"] == 2, again
  assert again["recomputed_rows"] == 0


def test_packed_runner_refuses_task_rows(tmp_path: Path) -> None:
  from aminx.host.pack_shim import run_packed_manifest_row  # noqa: PLC0415

  rows = _plan(tmp_path, _unit(tmp_path))
  with pytest.raises(NotImplementedError, match="does not support task='score_conditionals'"):
    run_packed_manifest_row(rows, row_index=0)


# --- knob differential for the new path ------------------------------------------------------------


@pytest.mark.knob_differential
def test_knob_differential_score_conditionals(tmp_path: Path) -> None:
  """Identical input -> exactly 0; changed P1' context token -> nonzero; changed seed -> exactly 0."""
  masks_path = _write_masks(tmp_path / "shared.npy", _masks(2))
  centre = _unit(tmp_path, ar_masks_path=masks_path)
  repeat = _unit(tmp_path, ar_masks_path=masks_path)
  p1p = _unit(tmp_path, label="P1p_W", ar_masks_path=masks_path,
              sequences=[_with_token(s, _P1_PRIME, "W") for s in centre.sequences])
  reseed = _unit(tmp_path, label="seed7", ar_masks_path=masks_path, random_seed=7)

  def _logits(unit: ScoreConditionalsUnit, root: str) -> np.ndarray:
    rows = plan_score_conditionals_manifest(
      [unit], campaign_id=root, output_root=tmp_path / root, git_sha="0" * 40, git_provenance_source="explicit",
    )
    return read_score_conditionals_store(_run(tmp_path / root, rows)["output_h5_path"])["logits"]

  (tmp_path / "a").mkdir()
  (tmp_path / "b").mkdir()
  (tmp_path / "c").mkdir()
  (tmp_path / "d").mkdir()
  base = _logits(centre, "a")
  np.testing.assert_array_equal(base, _logits(repeat, "b"))
  np.testing.assert_array_equal(base, _logits(reseed, "d"))
  moved = np.abs(base - _logits(p1p, "c"))
  designable = _designable().astype(bool)
  assert moved[..., designable, :].max() > _MOVED, "the P1' token did not reach the designable rows"


# --- LigandMPNN path ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.requires_weights
def test_ligand_context_reaches_the_row(tmp_path: Path) -> None:
  from aminx.host._sampling_helper import _canonical_structure_ids_for_spec  # noqa: PLC0415

  (structure_id,) = _canonical_structure_ids_for_spec(_spec(sequences_to_score=[_UBIQUITIN]))
  rng = np.random.default_rng(0)
  lig = tmp_path / "lig.npz"
  np.savez(
    lig,
    **{
      f"{structure_id}::Y": rng.normal(scale=6.0, size=(_MAX_LENGTH, 8, 3)).astype(np.float32),
      f"{structure_id}::Y_t": np.full((_MAX_LENGTH, 8), 6, dtype=np.int32),
      f"{structure_id}::Y_m": np.ones((_MAX_LENGTH, 8), dtype=np.float32),
    },
  )
  masks_path = _write_masks(tmp_path / "m.npy", _masks(2))
  on = _unit(tmp_path, checkpoint_id="ligandmpnn_v_32_020_25", ar_masks_path=masks_path, ligand_context_path=str(lig))
  off = _unit(tmp_path, label="off", checkpoint_id="ligandmpnn_v_32_020_25", ar_masks_path=masks_path)
  rows = _plan(tmp_path, on, off)
  a = read_score_conditionals_store(_run(tmp_path, rows, 0)["output_h5_path"])
  b = read_score_conditionals_store(_run(tmp_path, rows, 1)["output_h5_path"])
  assert a["attrs"]["ligand_context_sha256"] is not None
  assert np.abs(a["logits"] - b["logits"]).max() > _MOVED


# --- chunked pairs: one cached program, no per-call recompile -------------------------------------


def test_chunked_pairs_reuse_the_executable_and_match_vmapped(monkeypatch: pytest.MonkeyPatch) -> None:
  """Forced into tiles by a tiny budget: zero compiles on the warm call; same logits as vmapped.

  The pair map used to be a lax.map around a per-call closure, which recompiled on every call
  (2.7-3.1 s per call at K=64, C=7, L=512 on GPU) once the planner chose to chunk.
  """
  import jax  # noqa: PLC0415

  from aminx.tiling.memory_budget import MEMORY_BUDGET_ENV  # noqa: PLC0415

  compiles = [0]

  def _on(event: str, _d: float, **_k: object) -> None:
    if event == "/jax/core/compile/backend_compile_duration":
      compiles[0] += 1

  jax.monitoring.register_event_duration_secs_listener(_on)
  seqs = [_UBIQUITIN, _with_token(_UBIQUITIN, 5, "A")]
  vmapped = np.asarray(score(_spec(sequences_to_score=seqs, ar_masks=_masks(3, seed=5)))["logits"])

  monkeypatch.setenv(MEMORY_BUDGET_ENV, "1")  # one pair per tile
  chunked = np.asarray(score(_spec(sequences_to_score=seqs, ar_masks=_masks(3, seed=5)))["logits"])
  np.testing.assert_allclose(chunked[..., :_L, :], vmapped[..., :_L, :], rtol=0, atol=_BATCH_ATOL)
  before = compiles[0]
  score(_spec(sequences_to_score=seqs, ar_masks=_masks(3, seed=6)))
  assert compiles[0] == before, f"warm chunked call compiled {compiles[0] - before} executables"


# --- audit 261118 remediations ---------------------------------------------------------------------


def test_masks_must_cover_exactly_the_structure_residues() -> None:
  """A 70-residue sequence and mask on a 76-residue structure is refused, not padded with X."""
  short = _UBIQUITIN[:70]
  masks = np.stack([np.tril(np.ones((70, 70), np.int8), -1) + np.triu(np.ones((70, 70), np.int8), 1)] * 2)
  with pytest.raises(ValueError, match="exactly L residues after chain selection"):
    score(_spec(sequences_to_score=[short], ar_masks=masks))


def test_row_and_payload_task_must_agree(tmp_path: Path) -> None:
  from aminx.host.campaign import validate_manifest_rows  # noqa: PLC0415

  (row,) = _plan(tmp_path, _unit(tmp_path))
  row["unit_spec"] = dict(row["unit_spec"], task="something_else")
  with pytest.raises(ValueError, match="must agree"):
    validate_manifest_rows([row])


def test_task_files_are_hashed_in_full_not_by_size_and_mtime(tmp_path: Path) -> None:
  """A same-size, same-mtime replacement of a >64 KiB mask file still changes the input hash."""
  import os  # noqa: PLC0415

  big = np.repeat(_masks(1), 20, axis=0)  # 20 x 76 x 76 int8 = ~115 KiB
  path = _write_masks(tmp_path / "big.npy", big)
  (row,) = _plan(tmp_path, _unit(tmp_path, ar_masks_path=path))
  before = compute_unit_input_hash(row["unit_spec"])[0]
  stat = os.stat(path)
  flipped = big.copy()
  flipped[0, 5, :] = big[1, 5, :] if not np.array_equal(big[0, 5], big[1, 5]) else 1 - big[0, 5]
  flipped[0, 5, 5] = 0
  np.save(path, flipped)
  os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
  assert os.stat(path).st_size == stat.st_size
  assert compute_unit_input_hash(row["unit_spec"])[0] != before


# --- code review 261118 remediations ---------------------------------------------------------------


class _StubFamilyDriver:
  """A driver that claims every scoring purpose; it must never run for ar_masks."""

  name = "stub_family"
  mpnn_fallback_purposes: frozenset[str] = frozenset()

  def handles(self, spec: object, purpose: str) -> bool:
    del spec, purpose
    return True


def test_family_driver_refuses_ar_masks_in_score(monkeypatch: pytest.MonkeyPatch) -> None:
  import aminx.host.runner as runner  # noqa: PLC0415

  def _must_not_run(*_a: object, **_k: object) -> None:
    pytest.fail("the family driver ran although ar_masks was set")

  monkeypatch.setattr(runner, "_family_driver_for", lambda spec: _StubFamilyDriver())
  monkeypatch.setattr(runner, "run_family_driver", _must_not_run)
  with pytest.raises(NotImplementedError, match="does not support ar_masks"):
    score(_spec(sequences_to_score=[_UBIQUITIN], ar_masks=_masks(2)))


@pytest.mark.parametrize("checkpoint", ["lasermpnn_v1", "ligandmpnn_sc_v_32_002_16", "caliby"])
def test_unsupported_checkpoints_are_refused_at_plan_time(tmp_path: Path, checkpoint: str) -> None:
  with pytest.raises(ValueError, match="not supported"):
    _plan(tmp_path, _unit(tmp_path, checkpoint_id=checkpoint))


def test_unit_result_carries_logits_for_the_sink(tmp_path: Path) -> None:
  rows = _plan(tmp_path, _unit(tmp_path))
  result = _run(tmp_path, rows)
  assert np.asarray(result["logits"]).shape == (3, 2, _L, 21)
