"""run_potts must refuse spec/geometry state-count mismatches instead of misreporting n_backbones.

These tests need no weights: the state-count guard runs in step 0 before any model is loaded. When
``spec.n_backbones`` agrees with the number of states supplied in ``geometry`` (single-state or
multi-state PoE), execution proceeds past the guard to weight loading.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from aminx.potts.runner import run_potts
from aminx.potts.spec import PottsRunSpec
from aminx.types.bundles import GeometryBundle


def _geometry(n_states: int, n_residues: int = 5) -> GeometryBundle:
  """A GeometryBundle with a leading state axis of ``n_states``."""
  return GeometryBundle(
    coords=jnp.zeros((n_states, n_residues, 4, 3), dtype=jnp.float32),
    mask=jnp.ones((n_states, n_residues), dtype=jnp.float32),
    residue_index=jnp.arange(n_residues, dtype=jnp.int32)[None, :].repeat(n_states, axis=0),
    chain_index=jnp.zeros((n_states, n_residues), dtype=jnp.int32),
    n_states=n_states,
    n_canonical=20,
    n_flat=n_residues,
  )


def _spec(n_backbones: int = 1) -> PottsRunSpec:
  """A spec naming a weights path that does not exist, so reaching the loader is observable."""
  return PottsRunSpec(
    weights_path="/nonexistent/weights.eqx",
    caliby_path=None,
    k_neighbors=2,
    n_backbones=n_backbones,
  )


@pytest.mark.potts
class TestRunPottsRefusesMultistate:
  """run_potts requires spec.n_backbones to match the supplied geometry state count."""

  def test_multi_state_geometry_is_refused(self) -> None:
    """Multi-state geometry with default single-backbone spec raises ValueError instead of using state 0."""
    with pytest.raises(ValueError, match="backbone state"):
      run_potts(_spec(), _geometry(n_states=3), jax.random.key(0))

  def test_n_backbones_above_one_is_refused(self) -> None:
    """spec.n_backbones is copied verbatim into the result, so a mismatch with geometry is refused."""
    with pytest.raises(ValueError, match="n_backbones"):
      run_potts(_spec(n_backbones=4), _geometry(n_states=1), jax.random.key(0))

  def test_refusal_precedes_weight_loading(self) -> None:
    """The guard fires before weights are touched: the spec names a path that does not exist, and the
    error is the ValueError refusal rather than a file error."""
    with pytest.raises(ValueError, match="n_backbones"):
      run_potts(_spec(), _geometry(n_states=2), jax.random.key(0))

  def test_single_state_is_not_refused(self) -> None:
    """Matching spec.n_backbones and geometry n_states is NOT refused: it proceeds past the guard and
    fails on the missing weights file instead."""
    with pytest.raises(FileNotFoundError):
      run_potts(_spec(n_backbones=1), _geometry(n_states=1), jax.random.key(0))
    with pytest.raises(FileNotFoundError):
      run_potts(_spec(n_backbones=2), _geometry(n_states=2), jax.random.key(0))
