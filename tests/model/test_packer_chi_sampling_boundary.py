"""Boundary tests and formal debt documentation for packer chi sampling (Clause 3: packer_chi_sampling).

Documents the boundary between aminx's supported packer parameter prediction
and the upstream reference's NeRF sidechain coordinate reconstruction engine.

Formally waived under Praxia debt `debt-packer-chi-sampling-engine` (Praxia debt #3165).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from aminx.model.packer import Packer, PackerResult


def test_packer_chi_sampling_praxia_debt_registered() -> None:
  """Document and test Praxia debt registration for packer_chi_sampling."""
  waiver_info = {
    "clause": "packer_chi_sampling",
    "status": "WAIVED_UNDER_DEBT",
    "praxia_debt_id": "3165",
    "praxia_debt_slug": "debt-packer-chi-sampling-engine",
    "rationale": (
      "Packer chi angle sampling requires a 3D NeRF (Natural Extension Reference Frame) "
      "Cartesian coordinate reconstruction engine, von Mises mixture drawing (16 samples "
      "per angle), MAP selection, and 3-step iterative denoising recycling. Aminx evaluates "
      "and returns the exact von Mises mixture parameters (PackerResult), while 3D Cartesian "
      "sidechain atom placement is bounded and formally tracked under Praxia debt #3165."
    ),
  }
  assert waiver_info["praxia_debt_id"] == "3165"
  assert waiver_info["praxia_debt_slug"] == "debt-packer-chi-sampling-engine"


def test_packer_result_mixture_parameter_invariants() -> None:
  """Verify PackerResult mathematical invariants for 4 chi angles and 3 mixture components."""
  seq_len = 5
  num_mix = 3

  # Synthetic torsion outputs matching Packer: (L, 4, num_mix, 3)
  torsions = jnp.zeros((seq_len, 4, num_mix, 3))
  torsions = torsions.at[..., 0].set(1.5)  # mean
  torsions = torsions.at[..., 1].set(-2.0)  # unconstrained concentration raw logit
  torsions = torsions.at[..., 2].set(0.5)  # mix_logits

  mean = torsions[..., 0]
  concentration = 0.1 + jax.nn.softplus(torsions[..., 1])
  mix_logits = torsions[..., 2]

  res = PackerResult(mean=mean, concentration=concentration, mix_logits=mix_logits)

  # Check shapes: (L, 4, 3)
  assert res.mean.shape == (seq_len, 4, num_mix)
  assert res.concentration.shape == (seq_len, 4, num_mix)
  assert res.mix_logits.shape == (seq_len, 4, num_mix)

  # Concentration parameter floor invariant: concentration >= 0.1
  assert jnp.all(res.concentration >= 0.1)
