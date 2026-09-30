"""Bias slots must match the reference layer-for-layer (#162, #163, membrane ``node_embedding``).

equinox's ``Linear`` defaults to ``use_bias=True`` while several reference layers are
``bias=False`` (and one reference layer has a trained bias the port omitted). The mismatch is
silent in BOTH directions: a checkpoint bias with no slot is dropped, and a skeleton bias with no
checkpoint value keeps its random init and ships as if trained. Weight deserialisation is
positional, so neither shows up as an error.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.model.encoder import PhysicsEncoder
from aminx.model.ligand_mpnn import PrxteinLigandMPNN

_CONVERT = Path(__file__).resolve().parents[2] / "scripts" / "convert_weights.py"


def _ligand_skeleton() -> PrxteinLigandMPNN:
  return PrxteinLigandMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=32,
    key=jax.random.PRNGKey(0),
  )


def test_ligand_positional_embedding_keeps_its_trained_bias() -> None:
  """#162: reference ``features.embeddings.linear`` has a trained (16,) bias."""
  bias = _ligand_skeleton().features.embeddings.w_pos.bias
  assert bias is not None
  assert bias.shape == (16,)


def test_ligand_v_c_has_no_bias() -> None:
  """#163: reference ``V_C`` is ``bias=False``; a slot here holds untrained init noise."""
  assert _ligand_skeleton().v_c.bias is None


def test_physics_projection_has_no_bias() -> None:
  """Reference membrane ``node_embedding`` is ``bias=False``."""
  encoder = PhysicsEncoder(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_layers=1,
    physics_feature_dim=3,
    key=jax.random.PRNGKey(0),
  )
  assert encoder.physics_projection.bias is None


# --- the converter refuses a bias mismatch in either direction -------------------------------


@pytest.fixture(scope="module")
def convert_weights():
  spec = importlib.util.spec_from_file_location("convert_weights_under_test", _CONVERT)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _linear(*, use_bias: bool) -> eqx.nn.Linear:
  return eqx.nn.Linear(4, 3, use_bias=use_bias, key=jax.random.PRNGKey(1))


def test_converter_accepts_matching_bias(convert_weights) -> None:
  w, b = np.ones((3, 4), np.float32), np.arange(3, dtype=np.float32)
  out = convert_weights.convert_linear_layer(w, b, _linear(use_bias=True))
  np.testing.assert_array_equal(out.bias, b)
  out = convert_weights.convert_linear_layer(w, None, _linear(use_bias=False))
  assert out.bias is None


def test_converter_refuses_to_drop_a_trained_bias(convert_weights) -> None:
  """Positive control for #162: the checkpoint has a bias, the skeleton has no slot."""
  w, b = np.ones((3, 4), np.float32), np.ones(3, np.float32)
  with pytest.raises(ValueError, match="no slot for"):
    convert_weights.convert_linear_layer(w, b, _linear(use_bias=False))


def test_converter_refuses_to_ship_random_init_as_trained(convert_weights) -> None:
  """Positive control for #163: the skeleton has a bias, the checkpoint has none.

  The old behaviour kept the skeleton's random init; assert it is no longer reachable.
  """
  w = np.ones((3, 4), np.float32)
  with pytest.raises(ValueError, match="random init"):
    convert_weights.convert_linear_layer(w, None, _linear(use_bias=True))


# --- the SHIPPED weights carry the right slots ------------------------------------------------

_SHIPPED_LIGAND = [
  "ligandmpnn_v_32_005_25",
  "ligandmpnn_v_32_010_25",
  "ligandmpnn_v_32_020_25",
  "ligandmpnn_v_32_030_25",
]


@pytest.mark.requires_weights
@pytest.mark.parametrize("checkpoint_id", _SHIPPED_LIGAND)
def test_shipped_ligand_weights_carry_trained_pos_bias_and_no_v_c_bias(checkpoint_id: str) -> None:
  from aminx.io.weights import load_model

  model = load_model(checkpoint_id)
  assert model.v_c.bias is None
  bias = model.features.embeddings.w_pos.bias
  assert bias is not None
  # Equinox init is Uniform(+-1/sqrt(128)) ~ 0.088 with E|x| ~ 0.044; a trained bias is not
  # confined to that. This only guards against an all-zero / never-filled slot, not against
  # equality with the reference (the .pt-backed test below does that when the file exists).
  assert float(jnp.max(jnp.abs(bias))) > 0.0


_REFERENCE_ROOT = Path(os.environ.get("REFERENCE_PATH", "~/repos/LigandMPNN")).expanduser()
_REFERENCE_PT = _REFERENCE_ROOT / "model_params" / "ligandmpnn_v_32_020_25.pt"


@pytest.mark.requires_weights
@pytest.mark.skipif(not _REFERENCE_PT.is_file(), reason="reference .pt not present")
def test_shipped_pos_bias_equals_reference_checkpoint_bias() -> None:
  torch = pytest.importorskip("torch")
  from aminx.io.weights import load_model

  state = torch.load(_REFERENCE_PT, map_location="cpu", weights_only=False)["model_state_dict"]
  expected = state["features.embeddings.linear.bias"].numpy()
  got = np.asarray(load_model("ligandmpnn_v_32_020_25").features.embeddings.w_pos.bias)
  np.testing.assert_array_equal(got, expected)
