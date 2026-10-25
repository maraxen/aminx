"""run_potts must aggregate multiple backbone states as a product of experts, exactly.

The decisive test is the identity-backbone invariant that poe.py itself documents: for S copies of the
SAME backbone, the joint energy must equal S times the single-backbone energy. Because
sampling.log_energy is linear in the product w (*) J, and each state carries its own contact graph,
summing w and J independently would be wrong; these tests pin the aggregation that is actually correct.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

from aminx.potts.model import PottsModel
from aminx.potts.runner import PottsResult, run_potts
from aminx.potts.sampling import log_energy
from aminx.potts.spec import PottsRunSpec
from aminx.types.bundles import GeometryBundle

N_RES = 4


def _geometry(key: PRNGKeyArray, n_states: int, n_residues: int = N_RES) -> GeometryBundle:
  """A GeometryBundle whose states are IDENTICAL, so the PoE invariant applies exactly."""
  one = jax.random.normal(key, (n_residues, 4, 3), dtype=jnp.float32) * 3.0
  coords = jnp.broadcast_to(one, (n_states, n_residues, 4, 3))
  return GeometryBundle(
    coords=jnp.asarray(coords),
    mask=jnp.ones((n_states, n_residues), dtype=jnp.float32),
    residue_index=jnp.arange(n_residues, dtype=jnp.int32)[None, :].repeat(n_states, axis=0),
    chain_index=jnp.zeros((n_states, n_residues), dtype=jnp.int32),
    n_states=n_states,
    n_canonical=20,
    n_flat=n_residues,
  )


def _weights(tmpdir: str, key: PRNGKeyArray) -> str:
  model = PottsModel(
    hidden_dim=128,
    num_aa=21,
    k_neighbors=2,
    edge_features_dim=128,
    trw_iters=15,
    key=key,
    training=False,
  )
  path = Path(tmpdir) / "model.eqx"
  eqx.tree_serialise_leaves(str(path), model)
  return str(path)


def _spec(weights_path: str, n_backbones: int) -> PottsRunSpec:
  return PottsRunSpec(
    n_backbones=n_backbones,
    weights_path=weights_path,
    caliby_path=None,
    k_neighbors=2,
    training=False,
  )


@pytest.mark.potts
@pytest.mark.slow
class TestRunPottsMultistatePoe:
  """Multi-state input must be fused as a product of experts, not silently reduced to state 0."""

  def test_identity_backbone_energy_is_s_times_single(self) -> None:
    """S copies of one backbone must give exactly S times the single-backbone energy.

    This is the invariant poe.py documents, applied end to end through run_potts. It is the test that
    would fail if (h, J, W) were aggregated by summing W and J independently.
    """
    key = jax.random.key(0)
    with tempfile.TemporaryDirectory() as tmp:
      wp = _weights(tmp, key)
      single = run_potts(_spec(wp, 1), _geometry(key, 1), key)
      triple = run_potts(_spec(wp, 3), _geometry(key, 3), key)

      seq = jnp.zeros((N_RES,), dtype=jnp.int32)
      mask = jnp.ones((N_RES,), dtype=jnp.float32)
      e1 = float(log_energy(seq, single.h, single.J, single.W, mask))
      e3 = float(log_energy(seq, triple.h, triple.J, triple.W, mask))
      assert e3 == pytest.approx(3.0 * e1, rel=1e-4, abs=1e-4), f"{e3} != 3 * {e1}"

  def test_identity_backbone_h_and_w_scale(self) -> None:
    """h and W add across identical states; J, being a weighted mean, does not change."""
    key = jax.random.key(1)
    with tempfile.TemporaryDirectory() as tmp:
      wp = _weights(tmp, key)
      single = run_potts(_spec(wp, 1), _geometry(key, 1), key)
      triple = run_potts(_spec(wp, 3), _geometry(key, 3), key)
      assert jnp.allclose(triple.h, 3.0 * single.h, rtol=1e-4, atol=1e-4)
      assert jnp.allclose(triple.W, 3.0 * single.W, rtol=1e-4, atol=1e-4)
      assert jnp.allclose(triple.J, single.J, rtol=1e-4, atol=1e-4)

  def test_reports_the_ensemble_it_computed(self) -> None:
    """n_backbones must describe what was actually computed, and shapes stay per-residue."""
    key = jax.random.key(2)
    with tempfile.TemporaryDirectory() as tmp:
      wp = _weights(tmp, key)
      res = run_potts(_spec(wp, 3), _geometry(key, 3), key)
      assert isinstance(res, PottsResult)
      assert res.n_backbones == 3
      assert res.h.shape == (N_RES, 21)
      assert res.J.shape == (N_RES, N_RES, 21, 21)
      assert res.W.shape == (N_RES, N_RES)
      assert res.marginals.shape == (N_RES, 21)
      assert res.calibrated_marginals.shape == (N_RES, 21)

  def test_marginals_are_normalised(self) -> None:
    """The PoE marginals are a softmax over summed log-marginals and must be a distribution."""
    key = jax.random.key(3)
    with tempfile.TemporaryDirectory() as tmp:
      wp = _weights(tmp, key)
      res = run_potts(_spec(wp, 2), _geometry(key, 2), key)
      assert jnp.allclose(res.marginals.sum(axis=-1), 1.0, atol=1e-5)
      assert bool(jnp.all(res.marginals >= 0.0))

  def test_spec_disagreeing_with_geometry_is_refused(self) -> None:
    """spec.n_backbones is copied into the result, so it must match the states supplied."""
    key = jax.random.key(4)
    with tempfile.TemporaryDirectory() as tmp:
      wp = _weights(tmp, key)
      with pytest.raises(ValueError, match="n_backbones"):
        run_potts(_spec(wp, 1), _geometry(key, 3), key)
      with pytest.raises(ValueError, match="n_backbones"):
        run_potts(_spec(wp, 4), _geometry(key, 2), key)

  def test_single_state_still_works(self) -> None:
    """Regression: the single-state path is unchanged."""
    key = jax.random.key(5)
    with tempfile.TemporaryDirectory() as tmp:
      wp = _weights(tmp, key)
      res = run_potts(_spec(wp, 1), _geometry(key, 1), key)
      assert res.n_backbones == 1
      assert res.h.shape == (N_RES, 21)
      assert res.W is not None
