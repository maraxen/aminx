"""Regression for #111 through the real vmap-traced bundle path (debts #118, #122).

#111 fixed two bugs in every sampling dispatch closure of ``host/kernel_dispatch.py``:
a TracerArrayConversionError from indexing NumPy per-structure arrays with a traced
structure index, and a KeyError from looking up lowercase ``y/y_t/y_m`` in the ligand
context (whose keys are ``Y/Y_t/Y_m``). The earlier tests mocked the ligand context away.
These run the shared ``_structure_bundle_kwargs`` helper (now the only copy of that
indexing) under ``jax.vmap`` with real, non-None ligand tensors, all the way through
``build_inference_bundle``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from aminx.host.kernel_dispatch import _LIGAND_BUNDLE_KEYS, _structure_bundle_kwargs
from aminx.inference.bundle_builder import build_inference_bundle

B, L, A = 3, 5, 4


def _inputs():
  rng = np.random.default_rng(0)
  # Deliberately NumPy, as the batched ensemble delivers them: a traced index into a
  # NumPy array is exactly the #111 failure.
  per_structure = {
    "coords": rng.normal(size=(B, L, 4, 3)).astype(np.float32),
    "mask": np.ones((B, L), dtype=np.float32),
    "residue_index": np.tile(np.arange(L, dtype=np.int32), (B, 1)),
    "chain_index": np.zeros((B, L), dtype=np.int32),
    "fixed_mask": np.zeros((B, L), dtype=np.float32),
    "fixed_tokens": np.zeros((B, L), dtype=np.int32),
    "tie_group_map": None,
    "state_position_map": None,
    "structure_mapping": None,
  }
  ligand_context = {
    "Y": rng.normal(size=(B, L, A, 3)).astype(np.float32),
    "Y_t": rng.integers(1, 10, size=(B, L, A)).astype(np.int32),
    "Y_m": np.ones((B, L, A), dtype=np.float32),
    "atom_37": None,
    "atom_37_mask": None,
    "chain_mask": None,
  }
  shared = {"bias": None, "state_weights": None, "mode": "sample_ar", "inference": True}
  return per_structure, ligand_context, shared


def test_ligand_keys_match_prepare_ligand_context() -> None:
  assert dict(_LIGAND_BUNDLE_KEYS)["ligand_coords"] == "Y"
  assert dict(_LIGAND_BUNDLE_KEYS)["ligand_atom_types"] == "Y_t"
  assert dict(_LIGAND_BUNDLE_KEYS)["ligand_mask"] == "Y_m"


def test_traced_structure_index_selects_each_structures_ligand() -> None:
  per_structure, ligand_context, shared = _inputs()

  def one(structure_idx):
    kwargs = _structure_bundle_kwargs(
      structure_idx, per_structure=per_structure, ligand_context=ligand_context, shared=shared,
    )
    bundle, _ = build_inference_bundle(**kwargs, backbone_noise=0.0, temperature=1.0)
    return bundle.ligand.ligand_coords, bundle.ligand.ligand_atom_types, bundle.geometry.coords

  lig_xyz, lig_types, coords = jax.vmap(one)(jnp.arange(B))

  # (B, S=1, L, A, ...) -- each vmapped structure got ITS OWN ligand rows.
  np.testing.assert_array_equal(np.asarray(lig_xyz)[:, 0], ligand_context["Y"])
  np.testing.assert_array_equal(np.asarray(lig_types)[:, 0], ligand_context["Y_t"])
  np.testing.assert_array_equal(np.asarray(coords)[:, 0], per_structure["coords"])
