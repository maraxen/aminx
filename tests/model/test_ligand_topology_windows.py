"""LigandMPNN has three distinct top-k windows; each must reach the model it configures.

* ``k_neighbors``      -- residue kNN graph (checkpoint-parameterised: 30/32/48).
* ``atom_context_num`` -- nearest ligand atoms per residue (checkpoint-parameterised).
* ``SIDE_CHAIN_CONTEXT_NEIGHBORS`` -- hardcoded neighbour sub-slice for side-chain context.

None of them changes a stored tensor shape, so a wrong value deserialises silently. This is the
failure mode #161 realised (``atom_context_num`` parsed correctly, then never delivered) and
#152 asked to guard.
"""

from __future__ import annotations

import jax
import pytest

from aminx.io import weights as weights_mod
from aminx.io.weights import get_topology_for_checkpoint, load_model
from aminx.model import ligand_features
from aminx.model.ligand_features import SIDE_CHAIN_CONTEXT_NEIGHBORS
from aminx.model.ligand_mpnn import PrxteinLigandMPNN

# (checkpoint id, k_neighbors, atom_context_num) as the trained checkpoints declare them.
LIGAND_CHECKPOINTS = [
  ("ligandmpnn_v_32_005_25", 32, 25),
  ("ligandmpnn_v_32_010_25", 32, 25),
  ("ligandmpnn_v_32_020_25", 32, 25),
  ("ligandmpnn_v_32_030_25", 32, 25),
]


@pytest.mark.parametrize(("checkpoint_id", "k_neighbors", "atom_context_num"), LIGAND_CHECKPOINTS)
def test_topology_declares_the_checkpoint_windows(
  checkpoint_id: str, k_neighbors: int, atom_context_num: int
) -> None:
  topo = get_topology_for_checkpoint(checkpoint_id)
  assert topo["model_type"] == "ligand"
  assert topo["k_neighbors"] == k_neighbors
  assert topo["atom_context_num"] == atom_context_num


@pytest.mark.parametrize(("checkpoint_id", "k_neighbors", "atom_context_num"), LIGAND_CHECKPOINTS)
def test_load_model_delivers_every_window_to_the_ligand_model(
  monkeypatch: pytest.MonkeyPatch,
  checkpoint_id: str,
  k_neighbors: int,
  atom_context_num: int,
) -> None:
  """The value ``get_topology_for_checkpoint`` returns is the value the model holds (#161).

  ``load_weights`` is stubbed to hand back the skeleton, so this checks the construction site
  alone and needs no weight file.
  """
  monkeypatch.setattr(weights_mod, "load_weights", lambda *, skeleton, **_: skeleton)
  model = load_model(checkpoint_id)
  assert isinstance(model, PrxteinLigandMPNN)
  assert model.features.k_neighbors == k_neighbors
  assert model.features.atom_context_num == atom_context_num


def test_load_model_delivers_packer_atom_context(monkeypatch: pytest.MonkeyPatch) -> None:
  """Control: the packer branch always delivered it, and must keep doing so."""
  monkeypatch.setattr(weights_mod, "load_weights", lambda *, skeleton, **_: skeleton)
  model = load_model("ligandmpnn_sc_v_32_002_16")
  assert model.features.atom_context_num == 16


def test_skeleton_constructor_accepts_and_stores_atom_context_num() -> None:
  model = PrxteinLigandMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=32,
    atom_context_num=25,
    key=jax.random.PRNGKey(0),
  )
  assert model.features.atom_context_num == 25


def test_the_three_windows_are_distinct_for_every_shipped_ligand_checkpoint() -> None:
  """The side-chain sub-slice is its own constant, not derived from either checkpoint window."""
  assert SIDE_CHAIN_CONTEXT_NEIGHBORS == 16
  for checkpoint_id, k_neighbors, atom_context_num in LIGAND_CHECKPOINTS:
    assert len({SIDE_CHAIN_CONTEXT_NEIGHBORS, k_neighbors, atom_context_num}) == 3, checkpoint_id


def test_side_chain_slice_uses_the_named_constant() -> None:
  """A future edit to the literal must show up as a diff on the constant, not a bare ``16``."""
  import inspect

  source = inspect.getsource(ligand_features.ProteinFeaturesLigand.__call__)
  assert "E_idx[:, :SIDE_CHAIN_CONTEXT_NEIGHBORS]" in source
  assert "E_idx[:, :16]" not in source


@pytest.mark.requires_weights
def test_real_ligand_checkpoint_loads_with_25_atoms() -> None:
  model = load_model("ligandmpnn_v_32_020_25")
  assert model.features.atom_context_num == 25
