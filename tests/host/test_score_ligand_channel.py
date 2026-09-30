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
