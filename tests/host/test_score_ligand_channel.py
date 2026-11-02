"""``runner.score`` must receive the ligand a LigandMPNN design was conditioned on (#167).

``ScoringSpecification`` had no ligand field, so a spec built from a sampling row silently lost
``ligand_conditioning`` / ``ligand_context_path`` and a LigandMPNN checkpoint was scored WITHOUT
its ligand -- a different, worse-conditioned NLL, with nothing to say so. Building the spec by
filtering a sampling row's keys against the scoring spec's fields made the drop invisible.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from aminx.host.runner import score
from aminx.run.specs import SamplingSpecification, ScoringSpecification
from aminx.scoring.ligand_cutoff import DEFAULT_LIGAND_CUTOFF_ANGSTROMS

_STRUCTURE = "tests/data/1ubq.pdb"
_LIGAND_CHECKPOINT = "ligandmpnn_v_32_020_25"
_PROTEIN_CHECKPOINT = "proteinmpnn_v_48_020"
_MAX_LENGTH = 128
_UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def _spec(checkpoint_id: str = _LIGAND_CHECKPOINT, **kwargs) -> ScoringSpecification:
  return ScoringSpecification(
    inputs=[_STRUCTURE],
    checkpoint_id=checkpoint_id,
    sequences_to_score=[_UBIQUITIN],
    max_length=_MAX_LENGTH,
    **kwargs,
  )


def test_scoring_spec_carries_the_sampling_ligand_fields() -> None:
  """Field-membership filtering of a sampling row must not drop the ligand."""
  scoring = {f.name for f in dataclasses.fields(ScoringSpecification)}
  for name in ("ligand_conditioning", "ligand_context_path"):
    assert name in scoring
    assert name in {f.name for f in dataclasses.fields(SamplingSpecification)}


def test_ligand_context_path_string_is_coerced_to_path() -> None:
  spec = _spec(ligand_context_path="/tmp/ligand.npz")
  assert isinstance(spec.ligand_context_path, Path)


def test_side_chain_conditioning_is_refused_not_dropped() -> None:
  """score_sequence has no atom_37 parameter; a silent drop is the failure being closed."""
  with pytest.raises(NotImplementedError, match="sidechain_conditioning"):
    score(_spec(sidechain_conditioning=True))


@pytest.mark.parametrize("path_kwarg", [{"ligand_conditioning": True}, {"ligand_context_path": "/x.npz"}])
def test_ligand_on_the_averaged_path_is_refused(path_kwarg: dict) -> None:
  with pytest.raises(NotImplementedError, match="averaged-feature"):
    score(_spec(average_node_features=True, **path_kwarg))


def test_loss_xy_is_refused_on_the_fused_multistate_path() -> None:
  """The fused multi-state path returns before the loss loop; it must refuse, not drop loss_xy."""
  spec = _spec(
    return_loss_xy=True,
    state_position_map=np.zeros((1, len(_UBIQUITIN)), dtype=np.int32),
  )
  with pytest.raises(NotImplementedError, match="return_loss_xy"):
    score(spec)


@pytest.mark.parametrize("bad_cutoff", [0.0, -2.0, float("nan")])
def test_loss_xy_refuses_a_non_positive_cutoff(bad_cutoff: float) -> None:
  with pytest.raises(ValueError, match="positive distance"):
    score(_spec(return_loss_xy=True, ligand_cutoff_for_score=bad_cutoff))


def test_ligand_cutoff_default_is_the_shared_constant() -> None:
  """One literal, not a getattr fallback in the runner."""
  assert ScoringSpecification(inputs=[_STRUCTURE], sequences_to_score=[_UBIQUITIN]).ligand_cutoff_for_score == (
    DEFAULT_LIGAND_CUTOFF_ANGSTROMS
  )


class _StubFamilyDriver:
  """A driver that claims every scoring purpose. It must never run for return_loss_xy."""

  name = "stub_family"
  mpnn_fallback_purposes: frozenset[str] = frozenset()

  def handles(self, spec: object, purpose: str) -> bool:
    del spec, purpose
    return True


def test_loss_xy_is_refused_when_a_family_driver_handles_the_request(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Driver paths return before the loss loop; a silent drop of return_loss_xy is refused."""
  import aminx.host.runner as runner

  def _driver_must_not_run(*args: object, **kwargs: object) -> None:
    pytest.fail("the family driver ran although return_loss_xy=True")

  monkeypatch.setattr(runner, "_family_driver_for", lambda spec: _StubFamilyDriver())
  monkeypatch.setattr(runner, "run_family_driver", _driver_must_not_run)
  with pytest.raises(NotImplementedError, match="stub_family"):
    score(_spec(return_loss_xy=True))


def test_ligand_on_a_protein_checkpoint_is_refused() -> None:
  """A protein checkpoint has no ligand channel; accepting the request would drop it."""
  with pytest.raises(ValueError, match="no ligand channel"):
    score(_spec(_PROTEIN_CHECKPOINT, ligand_conditioning=True))


# --- end-to-end, real checkpoint ---------------------------------------------------------------


def _write_synthetic_ligand(path: Path, structure_id: str, *, n_atoms: int = 8) -> None:
  rng = np.random.default_rng(0)
  np.savez(
    path,
    **{
      f"{structure_id}::Y": rng.normal(scale=6.0, size=(_MAX_LENGTH, n_atoms, 3)).astype(np.float32),
      f"{structure_id}::Y_t": np.full((_MAX_LENGTH, n_atoms), 6, dtype=np.int32),
      f"{structure_id}::Y_m": np.ones((_MAX_LENGTH, n_atoms), dtype=np.float32),
    },
  )


@pytest.mark.slow
@pytest.mark.requires_weights
def test_ligand_changes_the_score_and_missing_tensors_raise(tmp_path: Path) -> None:
  """Differential: same structure, sequence and checkpoint; only the ligand differs."""
  from aminx.host._sampling_helper import _canonical_structure_ids_for_spec

  free = _spec()
  (structure_id,) = _canonical_structure_ids_for_spec(free)
  ligand_file = tmp_path / "ligand.npz"
  _write_synthetic_ligand(ligand_file, structure_id)

  ligand_free = float(np.asarray(score(free)["scores"]).reshape(-1)[0])
  with_ligand = float(
    np.asarray(
      score(_spec(ligand_conditioning=True, ligand_context_path=ligand_file))["scores"],
    ).reshape(-1)[0],
  )
  assert np.isfinite(ligand_free)
  assert np.isfinite(with_ligand)
  assert ligand_free != with_ligand, "the ligand did not reach the score -- still dropped"

  # ligand_conditioning=True with nothing to condition on is an error, never a silent fallback.
  with pytest.raises(ValueError, match="requires ligand context tensors"):
    score(_spec(ligand_conditioning=True))


@pytest.mark.slow
@pytest.mark.requires_weights
def test_loss_xy_shapes_with_two_noise_levels(tmp_path: Path) -> None:
  """With D=2 noise levels: loss_xy and ligand_confidence are (S, C, D); mask_xy is (S, L).

  Not executed locally: this needs the LigandMPNN checkpoint and runs on titanix.
  """
  from aminx.host._sampling_helper import _canonical_structure_ids_for_spec

  base = _spec(return_loss_xy=True, backbone_noise=(0.0, 0.1))
  (structure_id,) = _canonical_structure_ids_for_spec(base)
  ligand_file = tmp_path / "ligand.npz"
  _write_synthetic_ligand(ligand_file, structure_id)

  spec = _spec(
    return_loss_xy=True,
    backbone_noise=(0.0, 0.1),
    ligand_conditioning=True,
    ligand_context_path=ligand_file,
  )
  out = score(spec)
  n_struct, n_cand, n_noise = 1, 1, 2

  assert np.asarray(out["scores"]).shape == (n_struct, n_cand, n_noise)
  assert np.asarray(out["loss_xy"]).shape == (n_struct, n_cand, n_noise)
  assert np.asarray(out["ligand_confidence"]).shape == (n_struct, n_cand, n_noise)
  # mask_xy depends only on the structure; its residue axis is padded, so check rank and S only.
  assert np.asarray(out["mask_xy"]).ndim == 2
  assert np.asarray(out["mask_xy"]).shape[0] == n_struct
  np.testing.assert_allclose(
    np.asarray(out["ligand_confidence"]),
    np.exp(-np.asarray(out["loss_xy"])),
    rtol=1e-5,
  )
