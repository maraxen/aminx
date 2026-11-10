# ruff: noqa: S101
"""Unit tests for AntiBMPNN knob semantics (spec 261030 Phase 4).

Covers all 6 knobs:
- checkpoint_id (antibmpnn_model__model_name)
- cdr_scheme (antibmpnn_cdr__scheme)
- heavy_chain_id (antibmpnn_cdr__heavy_chain)
- light_chain_id (antibmpnn_cdr__light_chain)
- cdr_bias_weight (antibmpnn_cdr__bias_weight)
- compute_descriptors (__aminx_only__antibmpnn_descriptors)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.model.cdr_bias import (
  ALPHABET_21,
  CDRPhysicochemicalDescriptors,
  CDRRegionMasks,
  build_cdr_region_mask,
  compute_cdr_physicochemical_descriptors,
  make_cdr_frequency_logit_transform,
)
from aminx.run.options import AntiBMPNNOptions


def _antibody_fixture() -> dict[str, jnp.ndarray]:
  """Construct a representative antibody Fv residue sequence and numbering fixture."""
  # Heavy chain residues 1..115, Light chain residues 1..107
  h_res = np.arange(1, 116, dtype=np.int32)
  l_res = np.arange(1, 108, dtype=np.int32)
  res_idx = jnp.asarray(np.concatenate([h_res, l_res]))

  h_chains = np.zeros_like(h_res)
  l_chains = np.ones_like(l_res)
  chain_idx = jnp.asarray(np.concatenate([h_chains, l_chains]))
  mask = jnp.ones_like(res_idx, dtype=jnp.float32)

  return {
    "res_idx": res_idx,
    "chain_idx": chain_idx,
    "mask": mask,
  }


def test_knob_semantics_antibmpnn_checkpoint_id() -> None:
  """AntiBMPNN checkpoint_id resolution and configuration."""
  opt_default = AntiBMPNNOptions()
  assert opt_default.checkpoint_id is None

  opt_custom = AntiBMPNNOptions(checkpoint_id="AntiBMPNN/antibmpnn_v_48_010.pt")
  assert opt_custom.checkpoint_id == "AntiBMPNN/antibmpnn_v_48_010.pt"


def test_knob_semantics_cdr_scheme() -> None:
  """cdr_scheme controls Chothia vs IMGT loop boundary definitions."""
  fx = _antibody_fixture()

  mask_chothia = build_cdr_region_mask(
    fx["res_idx"],
    fx["chain_idx"],
    heavy_chain_id=0,
    light_chain_id=1,
    scheme="chothia",
    mask=fx["mask"],
  )

  mask_imgt = build_cdr_region_mask(
    fx["res_idx"],
    fx["chain_idx"],
    heavy_chain_id=0,
    light_chain_id=1,
    scheme="imgt",
    mask=fx["mask"],
  )

  # CDR loop boundaries differ between Chothia (e.g. CDR-H1: 26-32) and IMGT (CDR-H1: 27-38)
  assert not jnp.array_equal(mask_chothia.h1, mask_imgt.h1), "Chothia and IMGT CDR-H1 masks must differ"
  assert not jnp.array_equal(mask_chothia.l1, mask_imgt.l1), "Chothia and IMGT CDR-L1 masks must differ"
  assert not jnp.array_equal(mask_chothia.any_cdr, mask_imgt.any_cdr)


def test_knob_semantics_heavy_chain_id() -> None:
  """heavy_chain_id maps CDR-H1, H2, H3 loop masks to the specified chain."""
  fx = _antibody_fixture()

  # Normal: heavy = chain 0, light = chain 1
  mask_std = build_cdr_region_mask(
    fx["res_idx"],
    fx["chain_idx"],
    heavy_chain_id=0,
    light_chain_id=1,
    scheme="chothia",
    mask=fx["mask"],
  )

  # Swapped: heavy = chain 1, light = chain 0
  mask_swap = build_cdr_region_mask(
    fx["res_idx"],
    fx["chain_idx"],
    heavy_chain_id=1,
    light_chain_id=0,
    scheme="chothia",
    mask=fx["mask"],
  )

  # Heavy CDR residues in mask_std should belong to chain 0; in mask_swap they belong to chain 1
  assert jnp.all(fx["chain_idx"][mask_std.h1 > 0] == 0)
  assert jnp.all(fx["chain_idx"][mask_swap.h1 > 0] == 1)
  assert not jnp.array_equal(mask_std.h1, mask_swap.h1)


def test_knob_semantics_light_chain_id() -> None:
  """light_chain_id maps CDR-L1, L2, L3 loop masks to the specified chain."""
  fx = _antibody_fixture()

  mask_std = build_cdr_region_mask(
    fx["res_idx"],
    fx["chain_idx"],
    heavy_chain_id=0,
    light_chain_id=1,
    scheme="chothia",
    mask=fx["mask"],
  )

  mask_swap = build_cdr_region_mask(
    fx["res_idx"],
    fx["chain_idx"],
    heavy_chain_id=1,
    light_chain_id=0,
    scheme="chothia",
    mask=fx["mask"],
  )

  # Light CDR residues in mask_std should belong to chain 1; in mask_swap they belong to chain 0
  assert jnp.all(fx["chain_idx"][mask_std.l1 > 0] == 1)
  assert jnp.all(fx["chain_idx"][mask_swap.l1 > 0] == 0)
  assert not jnp.array_equal(mask_std.l1, mask_swap.l1)


def test_knob_semantics_cdr_bias_weight() -> None:
  """cdr_bias_weight scales the log-odds logit bias applied across CDR loops."""
  fx = _antibody_fixture()
  num_l = fx["res_idx"].shape[0]

  cdr_masks = build_cdr_region_mask(
    fx["res_idx"],
    fx["chain_idx"],
    heavy_chain_id=0,
    light_chain_id=1,
    scheme="chothia",
    mask=fx["mask"],
  )

  # Target frequencies favoring aromatic and charged residues
  target_freq = jnp.linspace(0.01, 0.09, 20)
  raw_logits = jnp.zeros((num_l, 21), dtype=jnp.float32)

  transform_full = make_cdr_frequency_logit_transform(
    cdr_mask=cdr_masks,
    target_frequencies=target_freq,
    weight=1.0,
  )
  logits_full = transform_full(raw_logits)

  transform_half = make_cdr_frequency_logit_transform(
    cdr_mask=cdr_masks,
    target_frequencies=target_freq,
    weight=0.5,
  )
  logits_half = transform_half(raw_logits)

  transform_zero = make_cdr_frequency_logit_transform(
    cdr_mask=cdr_masks,
    target_frequencies=target_freq,
    weight=0.0,
  )
  logits_zero = transform_zero(raw_logits)

  # At weight 0.0, logits remain untouched
  assert jnp.allclose(logits_zero, raw_logits, atol=1e-6)

  # At weight 0.5, bias is exactly half the full bias
  bias_full = logits_full - raw_logits
  bias_half = logits_half - raw_logits
  assert jnp.allclose(bias_half, 0.5 * bias_full, atol=1e-5)
  assert jnp.max(jnp.abs(bias_full)) > 0.1, "bias should be non-trivial"


def test_knob_semantics_compute_descriptors() -> None:
  """compute_descriptors calculates Barley PD1/PD2 and Lehninger charge."""
  # Poly-Trp sequence (high volume, low polarity, neutral charge)
  seq_trp = jnp.array([ALPHABET_21.index("W")] * 10, dtype=jnp.int32)
  desc_trp = compute_cdr_physicochemical_descriptors(seq_trp)

  # Poly-Asp sequence (low volume, high polarity, negatively charged at pH 7.4)
  seq_asp = jnp.array([ALPHABET_21.index("D")] * 10, dtype=jnp.int32)
  desc_asp = compute_cdr_physicochemical_descriptors(seq_asp)

  # Poly-Lys sequence (positively charged at pH 7.4)
  seq_lys = jnp.array([ALPHABET_21.index("K")] * 10, dtype=jnp.int32)
  desc_lys = compute_cdr_physicochemical_descriptors(seq_lys)

  # Volume: Trp > Asp
  assert float(desc_trp.pd1_volume) > float(desc_asp.pd1_volume)

  # Polarity / Hydrophilicity in Barley PC2: Asp (+1.25) > Trp (-1.27)
  assert float(desc_asp.pd2_hydro) > float(desc_trp.pd2_hydro)

  # Net charge: Lys (+10) > Trp (0) > Asp (-10)
  assert float(desc_lys.pd3_charge) > 5.0
  assert abs(float(desc_trp.pd3_charge)) < 0.5
  assert float(desc_asp.pd3_charge) < -5.0
