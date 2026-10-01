"""LigandMPNN features must read the backbone O atom by NAME, whichever layout the coordinates arrive in.

Parsed structures reach the model in the atom37 column order ``(N, CA, C, CB, O, ...)``; the compact export
wrapper and the reference-parity tests pass ``(N, CA, C, O)``.  ``ProteinFeatures`` already goes through
``compute_backbone_coordinates``, which handles both.  ``ProteinFeaturesLigand`` indexed columns 0-3 directly,
so for the parsed layout it read CB where it needed O -- every distance, angle and edge feature built on O was
wrong, for every LigandMPNN run through the public run API.  The model-level parity tests could not see it
because they feed the compact layout.  Found by the end-to-end run-API parity experiment (aminx #2326): the
reference disagreed by ~5 nats on real input and matched to 6e-2 once only the layout was changed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.io import weights as weights_mod
from aminx.io.weights import load_model

CHECKPOINT = "ligandmpnn_v_32_020_25"
L, M = 24, 25


@pytest.fixture(scope="module")
def model():
  """A randomly initialised ligand model (weight loading stubbed): features are weight-independent in structure."""
  mp = pytest.MonkeyPatch()
  mp.setattr(weights_mod, "load_weights", lambda *, skeleton, **_: skeleton)
  try:
    return load_model(CHECKPOINT)
  finally:
    mp.undo()


def _backbone(seed: int = 0) -> np.ndarray:
  """A plausible random backbone in atom37 layout with distinct CB and O columns (L, 37, 3)."""
  rng = np.random.default_rng(seed)
  ca = np.cumsum(rng.normal(0, 1.5, (L, 3)), axis=0).astype(np.float32)
  atom37 = np.zeros((L, 37, 3), np.float32)
  atom37[:, 0] = ca + rng.normal(0, 0.6, (L, 3))  # N
  atom37[:, 1] = ca  # CA
  atom37[:, 2] = ca + rng.normal(0, 0.6, (L, 3))  # C
  atom37[:, 3] = ca + rng.normal(2.0, 0.6, (L, 3))  # CB  (atom37 column 3)
  atom37[:, 4] = ca + rng.normal(-2.0, 0.6, (L, 3))  # O   (atom37 column 4)
  return atom37


def _call(model, coords: np.ndarray):
  rng = np.random.default_rng(1)
  y = jnp.asarray(rng.normal(0, 4.0, (L, M, 3)).astype(np.float32))
  y_t = jnp.asarray(rng.integers(1, 100, (L, M)).astype(np.int32))
  y_m = jnp.asarray((rng.random((L, M)) > 0.3).astype(np.float32))
  return model.features(
    jax.random.PRNGKey(0),
    jnp.asarray(coords),
    jnp.ones(L),
    jnp.arange(L),
    jnp.zeros(L, jnp.int32),
    y,
    y_t,
    y_m,
  )


def _max_diff(a, b) -> float:
  return max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(a, b, strict=True))


def test_atom37_and_compact_layouts_give_identical_ligand_features(model) -> None:
  atom37 = _backbone()
  compact = atom37[:, [0, 1, 2, 4]]  # (N, CA, C, O)
  assert _max_diff(_call(model, atom37), _call(model, compact)) < 1e-5


def test_the_test_is_sensitive_to_which_column_is_read_as_oxygen(model) -> None:
  """Negative control: handing the CB column to the O slot MUST change the features."""
  atom37 = _backbone()
  compact_correct = atom37[:, [0, 1, 2, 4]]
  compact_cb_as_o = atom37[:, [0, 1, 2, 3]]
  assert _max_diff(_call(model, compact_correct), _call(model, compact_cb_as_o)) > 1e-2
