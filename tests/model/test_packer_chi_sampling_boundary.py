"""Boundary tests for packer chi sampling (Clause 3: packer_chi_sampling).

The upstream reference reconstructs side-chain atoms with a 3D NeRF engine. aminx stops at the
von Mises mixture parameters that ``Packer`` predicts. These tests exercise the real
``Packer.__call__`` output, not arithmetic restated in the test.

The waiver itself, debt ``debt-packer-chi-sampling-engine`` (#3213), is tracked in Praxia. The
tests cannot verify that registration, so they do not claim to.
"""

from __future__ import annotations

import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import aminx.model.packer as packer_module
from aminx.model.packer import Packer
from aminx.types.bundles import PackerBundle, PackerResult
from aminx.types.configs import InferenceConfig

_NUM_MIX = 3
_NUM_CONTEXT_ATOMS = 16
_SEQ_LEN = 12
_CONFIG = InferenceConfig(inference=True)


def _small_packer() -> Packer:
  """Packer at the dimensions the packer parity test uses, with fewer layers."""
  return Packer(
    edge_features=128,
    node_features=128,
    num_positional_embeddings=16,
    num_rbf=16,
    top_k=30,
    atom_context_num=_NUM_CONTEXT_ATOMS,
    hidden_dim=128,
    num_encoder_layers=1,
    num_decoder_layers=1,
    dropout=0.0,
    num_mix=_NUM_MIX,
    key=jax.random.PRNGKey(0),
  )


def _synthetic_bundle(seq_len: int = _SEQ_LEN) -> PackerBundle:
  rng = np.random.default_rng(42)
  return PackerBundle(
    sequence=jnp.asarray(rng.integers(0, 21, (seq_len,)), dtype=jnp.int32),
    backbone_coords=jnp.asarray(rng.standard_normal((seq_len, 14, 3)), dtype=jnp.float32),
    backbone_mask=jnp.ones((seq_len, 14), dtype=jnp.float32),
    ligand_coords=jnp.asarray(
      rng.standard_normal((seq_len, _NUM_CONTEXT_ATOMS, 3)), dtype=jnp.float32
    ),
    ligand_mask=jnp.ones((seq_len, _NUM_CONTEXT_ATOMS), dtype=jnp.float32),
    ligand_atom_types=jnp.asarray(
      rng.integers(0, 119, (seq_len, _NUM_CONTEXT_ATOMS)), dtype=jnp.int32
    ),
    mask=jnp.ones((seq_len,), dtype=jnp.float32),
    residue_index=jnp.arange(seq_len, dtype=jnp.int32),
    chain_labels=jnp.zeros((seq_len,), dtype=jnp.int32),
  )


def test_packer_output_shapes_from_the_real_call() -> None:
  """Packer.__call__ returns (L, 4, num_mix) for mean, concentration and mix_logits."""
  result = _small_packer()(jax.random.PRNGKey(1), _synthetic_bundle(), _CONFIG)

  assert isinstance(result, PackerResult)
  assert result.mean.shape == (_SEQ_LEN, 4, _NUM_MIX)
  assert result.concentration.shape == (_SEQ_LEN, 4, _NUM_MIX)
  assert result.mix_logits.shape == (_SEQ_LEN, 4, _NUM_MIX)


def test_concentration_floor_holds_for_the_real_packer_at_default_init() -> None:
  result = _small_packer()(jax.random.PRNGKey(2), _synthetic_bundle(), _CONFIG)
  assert np.all(np.asarray(result.concentration) >= 0.1)


def test_concentration_floor_is_the_softplus_offset_when_raw_logit_is_very_negative() -> None:
  """Drive the raw concentration channel to -50 through the real head: output must be 0.1.

  The torsion head is zeroed so its output is exactly its bias. Layout of the (4 * num_mix * 3)
  bias is (chi, mix, [mean, concentration, mix_logit]), matching the reshape in Packer.decode.
  """
  packer = _small_packer()
  bias = np.zeros((4, _NUM_MIX, 3), dtype=np.float32)
  bias[..., 0] = 0.75  # mean channel, checked below
  bias[..., 1] = -50.0  # raw concentration logit: softplus(-50) is ~0
  bias[..., 2] = 0.2  # mix logit channel
  packer = eqx.tree_at(
    lambda p: (p.w_torsions.weight, p.w_torsions.bias),
    packer,
    (
      jnp.zeros_like(packer.w_torsions.weight),
      jnp.asarray(bias.reshape(-1)),
    ),
  )

  result = packer(jax.random.PRNGKey(3), _synthetic_bundle(), _CONFIG)

  assert np.asarray(result.concentration) == pytest.approx(0.1, abs=1e-6)
  assert np.asarray(result.mean) == pytest.approx(0.75, abs=1e-6)
  assert np.asarray(result.mix_logits) == pytest.approx(0.2, abs=1e-6)


def test_waiver_premise_no_nerf_engine_in_the_packer_module() -> None:
  """The waiver assumes aminx has no NeRF reconstruction. If that changes, the waiver must be revisited."""
  source = inspect.getsource(packer_module).lower()
  assert "nerf" not in source
