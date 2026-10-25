"""run_potts must refuse multi-state input instead of silently inferring from state 0.

These tests need no weights: the guards run before any model is loaded, which is the point of placing
them in step 0. A regression would otherwise be invisible -- the old behaviour returned a well-formed
PottsResult whose ``n_backbones`` described an ensemble it had not computed.
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
  """run_potts is single-backbone; multi-state input must fail loudly, not quietly."""

  def test_multi_state_geometry_is_refused(self) -> None:
    """A leading geometry axis of size > 1 raises instead of being reduced to state 0."""
    with pytest.raises(NotImplementedError, match="backbone states"):
      run_potts(_spec(), _geometry(n_states=3), jax.random.key(0))

  def test_n_backbones_above_one_is_refused(self) -> None:
    """spec.n_backbones is copied verbatim into the result, so a value never computed is refused."""
    with pytest.raises(NotImplementedError, match="n_backbones"):
      run_potts(_spec(n_backbones=4), _geometry(n_states=1), jax.random.key(0))

  def test_refusal_precedes_weight_loading(self) -> None:
    """The guard fires before weights are touched: the spec names a path that does not exist, and the
    error is the refusal rather than a file error."""
    with pytest.raises(NotImplementedError):
      run_potts(_spec(), _geometry(n_states=2), jax.random.key(0))

  def test_single_state_is_not_refused(self) -> None:
    """One state is NOT refused: it proceeds past the guard and fails on the missing weights file
    instead, which is how we know the guard does not over-trigger."""
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 -- asserting it is NOT the refusal
      run_potts(_spec(), _geometry(n_states=1), jax.random.key(0))
    assert not isinstance(excinfo.value, NotImplementedError)
