# ruff: noqa: S101
"""FrustraMPNN knob semantics tests for live options.

Tests runtime semantic behavior of all 5 live FrustraMPNNOptions knobs:
1. `checkpoint_id`: Selection and resolution of model weights.
2. `chains`: Chain filtering on multi-chain structures.
3. `positions`: Position subsetting vs full sequence evaluation.
4. `subtract_mut`: Absolute frustration scores vs wild-type subtracted deltas.
5. `dtype`: Floating-point precision (float32 default vs float64) and invalid dtype refusal.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.families.frustra_mpnn.driver import FrustraMPNNDriver
from aminx.families.frustra_mpnn.model import FrustraMPNN
from aminx.run.options import FrustraMPNNOptions

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "fixtures"
_CANONICAL_CKPT = "megascale_default_la-true_esm-False_split1"
_1UBQ_PDB = Path(__file__).resolve().parents[1] / "data" / "1ubq.pdb"


def _make_toy_two_chain_pdb(path: Path) -> None:
  """Write a minimal two-chain PDB (Chain A: 5 res, Chain B: 4 res)."""
  lines = []
  serial = 1
  chains = {"A": "ACDEF", "B": "GHIK"}
  for chain_id, seq in chains.items():
    for i, _aa in enumerate(seq, start=1):
      for j, atom in enumerate(("N", "CA", "C", "O")):
        lines.append(
          f"ATOM  {serial:5d}  {atom:<3s} ALA {chain_id}{i:4d}    "
          f"{i*3.8 + j:8.3f}{ord(chain_id)*10.0:8.3f}{0.0:8.3f}  1.00 20.00           {atom[0]}"
        )
        serial += 1
  lines.append("END")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_knob_semantics_checkpoint_id() -> None:
  """checkpoint_id resolves option > spec.checkpoint_id > default and validates weights."""
  driver = FrustraMPNNDriver()

  # 1. Default resolution
  opts_default = FrustraMPNNOptions()
  assert opts_default.checkpoint_id == _CANONICAL_CKPT

  spec_default = SimpleNamespace(
    checkpoint_id=None,
    options=opts_default,
    model_local_path=_FIXTURE_DIR / f"{_CANONICAL_CKPT}.ckpt",
  )
  model = driver.load(spec_default)
  assert isinstance(model, FrustraMPNN)
  assert model.light_attention.feature_weight.shape == (512, 512, 9)
  assert model.both_out.layers[2].weight.shape == (21, 32)

  # 2. Option precedence over spec
  spec_override = SimpleNamespace(
    checkpoint_id="other_checkpoint",
    options=FrustraMPNNOptions(checkpoint_id=_CANONICAL_CKPT),
    model_local_path=_FIXTURE_DIR / f"{_CANONICAL_CKPT}.ckpt",
  )
  model_ov = driver.load(spec_override)
  assert isinstance(model_ov, FrustraMPNN)

  # 3. Non-existent checkpoint raises loud error
  spec_bad = SimpleNamespace(
    checkpoint_id="nonexistent_frustra_model",
    options=FrustraMPNNOptions(checkpoint_id="nonexistent_frustra_model"),
    model_local_path=None,
  )
  with pytest.raises((FileNotFoundError, KeyError)):
    driver.load(spec_bad)


def test_knob_semantics_chains(tmp_path: Path) -> None:
  """chains knob filters predictions to designated chain(s) on multi-chain complexes."""
  driver = FrustraMPNNDriver()
  pdb_path = tmp_path / "complex.pdb"
  _make_toy_two_chain_pdb(pdb_path)

  spec_all = SimpleNamespace(
    inputs=str(pdb_path),
    checkpoint_id=_CANONICAL_CKPT,
    model_local_path=_FIXTURE_DIR / f"{_CANONICAL_CKPT}.ckpt",
    options=FrustraMPNNOptions(chains=None),
  )
  prep_all = driver.prepare_structure(pdb_path, chains=None)
  assert prep_all["coords"].shape[0] == 9  # 5 from A + 4 from B

  prep_a = driver.prepare_structure(pdb_path, chains="A")
  assert prep_a["coords"].shape[0] == 5
  assert set(prep_a["chain_index"]) == {0}

  prep_b = driver.prepare_structure(pdb_path, chains="B")
  assert prep_b["coords"].shape[0] == 4

  prep_both = driver.prepare_structure(pdb_path, chains=("A", "B"))
  assert prep_both["coords"].shape[0] == 9


def test_knob_semantics_positions() -> None:
  """positions knob subsets scored sites without changing values at evaluated positions."""
  driver = FrustraMPNNDriver()
  model = driver.load_model(_CANONICAL_CKPT, local_path=_FIXTURE_DIR / f"{_CANONICAL_CKPT}.ckpt")
  prep = driver.prepare_structure(_1UBQ_PDB)

  # Full evaluation (all 76 positions)
  spec_full = SimpleNamespace(
    inputs=str(_1UBQ_PDB),
    checkpoint_id=_CANONICAL_CKPT,
    options=FrustraMPNNOptions(positions=None),
  )
  stages_full = driver.stages(spec_full, "score:frustration", model)
  batch_full = driver.collate_batch((prep,), input_indices=(0,))
  out_full = stages_full(
    batch_full,
    chunk_start=0,
    chunk_count=1,
    scalar_dropout=False,
    dropout_key=jax.random.PRNGKey(0),
  )
  scores_full = np.asarray(out_full["frustration_scores"])
  assert scores_full.shape == (76, 21)

  # Subsetting positions (positions 0, 5, 10)
  target_pos = (0, 5, 10)
  spec_sub = SimpleNamespace(
    inputs=str(_1UBQ_PDB),
    checkpoint_id=_CANONICAL_CKPT,
    options=FrustraMPNNOptions(positions=target_pos),
  )
  stages_sub = driver.stages(spec_sub, "score:frustration", model)
  out_sub = stages_sub(
    batch_full,
    chunk_start=0,
    chunk_count=1,
    scalar_dropout=False,
    dropout_key=jax.random.PRNGKey(0),
  )
  scores_sub = np.asarray(out_sub["frustration_scores"])

  # Scores at target positions must be identical to full evaluation
  for pos in target_pos:
    np.testing.assert_allclose(scores_sub[pos], scores_full[pos], atol=1e-6)

  # Non-target positions must be NaN in subsetting mode
  for pos in range(76):
    if pos not in target_pos:
      assert np.isnan(scores_sub[pos]).all()


def test_knob_semantics_subtract_mut() -> None:
  """subtract_mut=False yields raw indices; subtract_mut=True zeroes wild-type residues."""
  driver = FrustraMPNNDriver()
  prep = driver.prepare_structure(_1UBQ_PDB)
  seq = np.asarray(prep["sequence"])
  L = len(seq)

  # Raw scoring (subtract_mut = False, default)
  model_raw = driver.load_model(
    _CANONICAL_CKPT,
    local_path=_FIXTURE_DIR / f"{_CANONICAL_CKPT}.ckpt",
    subtract_mut=False,
  )
  spec_raw = SimpleNamespace(checkpoint_id=_CANONICAL_CKPT, options=FrustraMPNNOptions(subtract_mut=False))
  stages_raw = driver.stages(spec_raw, "score:frustration", model_raw)
  batch = driver.collate_batch((prep,), input_indices=(0,))
  res_raw = np.asarray(
    stages_raw(
      batch,
      chunk_start=0,
      chunk_count=1,
      scalar_dropout=False,
      dropout_key=jax.random.PRNGKey(0),
    )["frustration_scores"],
  )

  # Wild-type scores in raw mode must NOT be all zeros
  wt_scores_raw = res_raw[np.arange(L), seq]
  assert not np.allclose(wt_scores_raw, 0.0, atol=1e-3), "Raw wild-type scores must be non-zero"

  # Delta scoring (subtract_mut = True)
  model_sub = driver.load_model(
    _CANONICAL_CKPT,
    local_path=_FIXTURE_DIR / f"{_CANONICAL_CKPT}.ckpt",
    subtract_mut=True,
  )
  spec_sub = SimpleNamespace(checkpoint_id=_CANONICAL_CKPT, options=FrustraMPNNOptions(subtract_mut=True))
  stages_sub = driver.stages(spec_sub, "score:frustration", model_sub)
  res_sub = np.asarray(
    stages_sub(
      batch,
      chunk_start=0,
      chunk_count=1,
      scalar_dropout=False,
      dropout_key=jax.random.PRNGKey(0),
    )["frustration_scores"],
  )

  # Wild-type scores in subtract_mut mode MUST be identically 0.0
  wt_scores_sub = res_sub[np.arange(L), seq]
  np.testing.assert_allclose(wt_scores_sub, 0.0, atol=1e-6)

  # For mutant residues, res_sub[i, a] == res_raw[i, a] - res_raw[i, seq[i]]
  expected_sub = res_raw - wt_scores_raw[:, None]
  np.testing.assert_allclose(res_sub, expected_sub, atol=1e-5)


def test_knob_semantics_dtype() -> None:
  """dtype selects float32 or float64 compute precision and rejects invalid dtypes."""
  driver = FrustraMPNNDriver()
  prep = driver.prepare_structure(_1UBQ_PDB)

  # 1. float32 execution
  model_f32 = driver.load_model(
    _CANONICAL_CKPT,
    local_path=_FIXTURE_DIR / f"{_CANONICAL_CKPT}.ckpt",
    dtype="float32",
  )
  spec_f32 = SimpleNamespace(checkpoint_id=_CANONICAL_CKPT, options=FrustraMPNNOptions(dtype="float32"))
  stages_f32 = driver.stages(spec_f32, "score:frustration", model_f32)
  batch_f32 = driver.collate_batch((prep,), input_indices=(0,))
  out_f32 = stages_f32(
    batch_f32,
    chunk_start=0,
    chunk_count=1,
    scalar_dropout=False,
    dropout_key=jax.random.PRNGKey(0),
  )
  scores_f32 = out_f32["frustration_scores"]
  assert scores_f32.dtype == jnp.float32

  # 2. float64 execution
  prev_x64 = bool(getattr(jax.config, "jax_enable_x64", False))
  jax.config.update("jax_enable_x64", True)
  try:
    model_f64 = driver.load_model(
      _CANONICAL_CKPT,
      local_path=_FIXTURE_DIR / f"{_CANONICAL_CKPT}.ckpt",
      dtype="float64",
    )
    spec_f64 = SimpleNamespace(checkpoint_id=_CANONICAL_CKPT, options=FrustraMPNNOptions(dtype="float64"))
    stages_f64 = driver.stages(spec_f64, "score:frustration", model_f64)
    batch_f64 = driver.collate_batch((prep,), input_indices=(0,))
    out_f64 = stages_f64(
      batch_f64,
      chunk_start=0,
      chunk_count=1,
      scalar_dropout=False,
      dropout_key=jax.random.PRNGKey(0),
    )
    scores_f64 = out_f64["frustration_scores"]
    assert scores_f64.dtype == jnp.float64

    # 3. Precision band: f64 vs f32 diff is finite and within floating-point tolerance
    diff = float(np.max(np.abs(np.asarray(scores_f64, dtype=np.float64) - np.asarray(scores_f32, dtype=np.float64))))
    assert diff < 1e-4
    assert diff > 0.0, "float64 should exhibit slight numerical divergence from float32"
  finally:
    jax.config.update("jax_enable_x64", prev_x64)

  # 4. Invalid dtype refusal
  with pytest.raises(ValueError, match="Unknown FrustraMPNN dtype"):
    FrustraMPNNOptions(dtype="float16")  # type: ignore[arg-type]
  with pytest.raises(ValueError, match="Unknown FrustraMPNN dtype"):
    FrustraMPNNOptions(dtype="bfloat16")  # type: ignore[arg-type]
