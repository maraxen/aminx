"""P07 autoregressive-sample export: exactness, live controls, ONNX, census.

CPU, L=32, pinned ``proteinmpnn_v_48_020`` weights (same loader as the P04 export tests).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest

# Skip loudly, with the fix, where the ONNX toolchain is absent. CI installs it (--extra onnx) so these RUN there
# (aminx debt #2392); without this guard a missing package is a collection ERROR that reds the whole job.
for _mod in ('onnx', 'onnxruntime', 'jax2onnx'):
  pytest.importorskip(_mod, reason="needs the ONNX toolchain: uv sync --extra onnx (aminx debt #2392)")

import jax2onnx
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from aminx.export.rng_audit import find_rng_primitives
from aminx.export.wrappers import (
  PINNED_CHECKPOINT_ID,
  make_p07_sample,
  p07_bundle,
  zero_dropout,
)
from aminx.inference.logits import make_stage_set  # noqa: TID251
from aminx.inference.sample_autoregressive import gumbel_noise_for_key, kernel
from aminx.io.weights import load_model
from scripts.browser_validation.export_safety_census import safety_checks
from scripts.browser_validation.onnx_audit import find_onnx_rng_ops

if TYPE_CHECKING:
  from aminx.model import Aminx

# pytest asserts; this tree excludes tests/ from ruff, and the evidence command clears that.
# ruff: noqa: S101

P07 = Callable[..., tuple[jax.Array, jax.Array]]

L = 32
_KEYS = (0, 1, 2)
_OMIT = -1.0e8


def _structure(seed: int = 0) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  rng = np.random.default_rng(seed)
  coords = jnp.asarray((rng.normal(size=(L, 4, 3)) * 10.0).astype(np.float32))
  mask = jnp.ones((L,), dtype=jnp.float32)
  residue_index = jnp.arange(L, dtype=jnp.int32)
  chain_index = jnp.zeros((L,), dtype=jnp.int32)
  return coords, mask, residue_index, chain_index


def _order(seed: int) -> jax.Array:
  perm = np.random.default_rng(seed).permutation(L).astype(np.int32)
  return jnp.asarray(perm)


def _controls(
  *,
  order_seed: int = 4,
  bias: jax.Array | None = None,
  fixed_mask: jax.Array | None = None,
  fixed_tokens: jax.Array | None = None,
  temperature: float = 1.0,
  tie_group_map: jax.Array | None = None,
) -> dict[str, jax.Array]:
  return {
    "decoding_order": _order(order_seed),
    "bias": jnp.zeros((L, 21), dtype=jnp.float32) if bias is None else bias,
    "fixed_mask": jnp.zeros((L,), dtype=jnp.float32) if fixed_mask is None else fixed_mask,
    "fixed_tokens": jnp.zeros((L,), dtype=jnp.int32) if fixed_tokens is None else fixed_tokens,
    "temperature": jnp.asarray(temperature, dtype=jnp.float32),
    "tie_group_map": jnp.arange(L, dtype=jnp.int32) if tie_group_map is None else tie_group_map,
  }


def _call(
  fn: P07,
  coords: jax.Array,
  mask: jax.Array,
  residue_index: jax.Array,
  chain_index: jax.Array,
  noise: jax.Array,
  controls: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array]:
  return fn(
    coords,
    mask,
    residue_index,
    chain_index,
    noise,
    controls["decoding_order"],
    controls["bias"],
    controls["fixed_mask"],
    controls["fixed_tokens"],
    controls["temperature"],
    controls["tie_group_map"],
  )


def _key_sample(
  native: Aminx,
  stage_set: object,
  key: jax.Array,
  coords: jax.Array,
  mask: jax.Array,
  residue_index: jax.Array,
  chain_index: jax.Array,
  controls: dict[str, jax.Array],
) -> tuple[np.ndarray, np.ndarray]:
  bundle, config = p07_bundle(
    coords,
    mask,
    residue_index,
    chain_index,
    controls["decoding_order"],
    controls["bias"],
    controls["fixed_mask"],
    controls["fixed_tokens"],
    controls["temperature"],
    controls["tie_group_map"],
  )
  result = kernel(native, key, bundle, config, stage_set)  # type: ignore[arg-type]
  log_probs = jax.nn.log_softmax(result.logits, axis=-1)
  return np.asarray(result.sequence), np.asarray(log_probs)


@pytest.fixture(scope="module")
def checkpoint_model() -> Aminx:
  return load_model(checkpoint_id=PINNED_CHECKPOINT_ID)


@pytest.fixture(scope="module")
def native_model(checkpoint_model: Aminx) -> Aminx:
  model, _ = zero_dropout(checkpoint_model)
  return model


@pytest.fixture(scope="module")
def p07(checkpoint_model: Aminx) -> P07:
  return make_p07_sample(checkpoint_model, make_stage_set())


@pytest.fixture(scope="module")
def stage_set() -> object:
  return make_stage_set()


@pytest.mark.requires_weights
def test_exactness_matches_key_sampler(p07: P07, native_model: Aminx, stage_set: object) -> None:
  """Tokens match the key path for the key's own Gumbel noise; log_probs within 1e-5."""
  coords, mask, residue_index, chain_index = _structure(0)
  controls = _controls()
  for seed in _KEYS:
    key = jax.random.PRNGKey(seed)
    noise = gumbel_noise_for_key(key, L)
    tokens, log_probs = _call(p07, coords, mask, residue_index, chain_index, noise, controls)
    ref_tokens, ref_log_probs = _key_sample(
      native_model,
      stage_set,
      key,
      coords,
      mask,
      residue_index,
      chain_index,
      controls,
    )
    assert np.array_equal(np.asarray(tokens), ref_tokens), seed
    assert np.allclose(np.asarray(log_probs), ref_log_probs, atol=1e-5), seed


@pytest.mark.requires_weights
def test_fixed_mask_forces_tokens(p07: P07) -> None:
  coords, mask, residue_index, chain_index = _structure(1)
  positions = np.array([4, 11, 27], dtype=np.int32)
  key = jax.random.PRNGKey(0)
  noise = gumbel_noise_for_key(key, L)
  base_controls = _controls()
  base_tokens, _ = _call(p07, coords, mask, residue_index, chain_index, noise, base_controls)
  fixed_tokens = np.asarray(base_tokens).copy()
  fixed_tokens[positions] = (fixed_tokens[positions] + 1) % 20
  fixed_mask = jnp.zeros((L,), dtype=jnp.float32).at[jnp.asarray(positions)].set(1.0)
  controls = _controls(
    fixed_mask=fixed_mask,
    fixed_tokens=jnp.asarray(fixed_tokens, dtype=jnp.int32),
  )
  tokens, _ = _call(p07, coords, mask, residue_index, chain_index, noise, controls)
  got = np.asarray(tokens)
  assert np.array_equal(got[positions], fixed_tokens[positions])
  assert not np.array_equal(got, np.asarray(base_tokens))


@pytest.mark.requires_weights
def test_bias_column_removes_amino_acid(p07: P07) -> None:
  coords, mask, residue_index, chain_index = _structure(2)
  key = jax.random.PRNGKey(1)
  noise = gumbel_noise_for_key(key, L)
  base_controls = _controls()
  base_tokens, _ = _call(p07, coords, mask, residue_index, chain_index, noise, base_controls)
  base = np.asarray(base_tokens)
  omitted = int(base[0])
  bias = jnp.zeros((L, 21), dtype=jnp.float32).at[:, omitted].set(_OMIT)
  controls = _controls(bias=bias)
  tokens, _ = _call(p07, coords, mask, residue_index, chain_index, noise, controls)
  got = np.asarray(tokens)
  assert not np.any(got == omitted)
  assert not np.array_equal(got, base)


@pytest.mark.requires_weights
def test_temperature_changes_tokens(p07: P07) -> None:
  coords, mask, residue_index, chain_index = _structure(3)
  changed = False
  for seed in range(8):
    noise = gumbel_noise_for_key(jax.random.PRNGKey(seed), L)
    hot, _ = _call(p07, coords, mask, residue_index, chain_index, noise, _controls(temperature=1.0))
    cold, _ = _call(
      p07,
      coords,
      mask,
      residue_index,
      chain_index,
      noise,
      _controls(temperature=0.1),
    )
    if not np.array_equal(np.asarray(hot), np.asarray(cold)):
      changed = True
      break
  assert changed


@pytest.mark.requires_weights
def test_tie_group_map_shares_tokens(p07: P07) -> None:
  coords, mask, residue_index, chain_index = _structure(4)
  i, j = 3, 17
  forced = False
  for seed in range(8):
    noise = gumbel_noise_for_key(jax.random.PRNGKey(seed), L)
    untied, _ = _call(p07, coords, mask, residue_index, chain_index, noise, _controls())
    if int(np.asarray(untied)[i]) == int(np.asarray(untied)[j]):
      continue
    tie = jnp.arange(L, dtype=jnp.int32).at[j].set(i)
    tied, _ = _call(
      p07,
      coords,
      mask,
      residue_index,
      chain_index,
      noise,
      _controls(tie_group_map=tie),
    )
    got = np.asarray(tied)
    assert got[i] == got[j]
    assert not np.array_equal(got, np.asarray(untied))
    forced = True
    break
  assert forced


@pytest.mark.requires_weights
def test_decoding_order_changes_tokens(p07: P07) -> None:
  coords, mask, residue_index, chain_index = _structure(5)
  order_a = _order(10)
  order_b = _order(11)
  assert not np.array_equal(np.asarray(order_a), np.asarray(order_b))
  changed = False
  for seed in _KEYS:
    noise = gumbel_noise_for_key(jax.random.PRNGKey(seed), L)
    tokens_a, _ = _call(
      p07,
      coords,
      mask,
      residue_index,
      chain_index,
      noise,
      _controls(order_seed=10),
    )
    tokens_b, _ = _call(
      p07,
      coords,
      mask,
      residue_index,
      chain_index,
      noise,
      _controls(order_seed=11),
    )
    if not np.array_equal(np.asarray(tokens_a), np.asarray(tokens_b)):
      changed = True
      break
  assert changed


def _numpy_inputs(
  coords: jax.Array,
  mask: jax.Array,
  residue_index: jax.Array,
  chain_index: jax.Array,
  noise: jax.Array,
  controls: dict[str, jax.Array],
) -> tuple[np.ndarray, ...]:
  arrays = (
    coords,
    mask,
    residue_index,
    chain_index,
    noise,
    controls["decoding_order"],
    controls["bias"],
    controls["fixed_mask"],
    controls["fixed_tokens"],
    controls["temperature"],
    controls["tie_group_map"],
  )
  return tuple(np.asarray(a) for a in arrays)


@pytest.mark.requires_weights
def test_onnx_export_ort_bitwise_and_rng_free(p07: P07, tmp_path: Path) -> None:
  """jax2onnx lowers the scan; ORT tokens match JAX; the graph has no RNG op."""
  coords, mask, residue_index, chain_index = _structure(6)
  base = _controls()
  noises = [gumbel_noise_for_key(jax.random.PRNGKey(seed), L) for seed in _KEYS]
  # jax2onnx patches jnp.stack for the conversion and the patch stays installed, so every
  # eager JAX call (and the jaxpr walk) has to happen before to_onnx.
  jaxpr = jax.make_jaxpr(p07)(
    *_numpy_inputs(
      coords,
      mask,
      residue_index,
      chain_index,
      noises[0],
      base,
    ),
  )
  assert find_rng_primitives(jaxpr) == []
  jax_draws = []
  for noise in noises:
    tokens, log_probs = _call(p07, coords, mask, residue_index, chain_index, noise, base)
    jax_draws.append(
      (
        _numpy_inputs(coords, mask, residue_index, chain_index, noise, base),
        np.asarray(tokens),
        np.asarray(log_probs),
      ),
    )
  key = jax.random.PRNGKey(0)
  noise = gumbel_noise_for_key(key, L)
  positions = np.array([4, 11, 27], dtype=np.int32)
  fixed_tokens = jnp.arange(L, dtype=jnp.int32) % 20
  fixed_mask = jnp.zeros((L,), dtype=jnp.float32).at[jnp.asarray(positions)].set(1.0)
  bias = jnp.zeros((L, 21), dtype=jnp.float32).at[:, 3].set(_OMIT)
  tie = jnp.arange(L, dtype=jnp.int32).at[17].set(3)
  variants = [
    ("fixed", _controls(fixed_mask=fixed_mask, fixed_tokens=fixed_tokens)),
    ("bias", _controls(bias=bias)),
    ("temperature", _controls(temperature=0.1)),
    ("tie", _controls(tie_group_map=tie)),
    ("order", _controls(order_seed=11)),
  ]
  jax_controls = []
  for name, controls in variants:
    tokens, log_probs = _call(p07, coords, mask, residue_index, chain_index, noise, controls)
    jax_controls.append(
      (
        name,
        _numpy_inputs(coords, mask, residue_index, chain_index, noise, controls),
        np.asarray(tokens),
        np.asarray(log_probs),
      ),
    )

  specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in jax_draws[0][0]]
  out = tmp_path / "p07_L32.onnx"
  # Same call as scripts/browser_validation/layer_b_build.py:_convert_to_onnx.
  jax2onnx.to_onnx(
    p07,
    specs,
    model_name="p07_sample_L32",
    output_path=str(out),
    return_mode="file",
  )
  model = onnx.load(str(out))
  assert find_onnx_rng_ops(model) == []

  session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
  input_names = [item.name for item in session.get_inputs()]

  def _ort(inputs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    feeds = dict(zip(input_names, inputs, strict=True))
    tokens, log_probs = session.run(None, feeds)
    return np.asarray(tokens), np.asarray(log_probs)

  for inputs, jax_tokens, jax_lp in jax_draws:
    ort_tokens, ort_lp = _ort(inputs)
    assert np.array_equal(ort_tokens, jax_tokens)
    assert np.allclose(ort_lp, jax_lp, atol=1e-4, rtol=0.0)
  for name, inputs, jax_tokens, jax_lp in jax_controls:
    ort_tokens, ort_lp = _ort(inputs)
    assert np.array_equal(ort_tokens, jax_tokens), name
    assert np.allclose(ort_lp, jax_lp, atol=1e-4, rtol=0.0), name


@pytest.mark.requires_weights
def test_export_safety_census_no_sort_stability(p07: P07) -> None:
  """``export_safety_census.safety_checks`` on the P07 wrapper; no sort-stability."""
  coords, mask, residue_index, chain_index = _structure(7)
  controls = _controls()
  noise = gumbel_noise_for_key(jax.random.PRNGKey(0), L)
  inputs = _numpy_inputs(coords, mask, residue_index, chain_index, noise, controls)

  def fn() -> tuple[jax.Array, jax.Array]:
    return p07(*inputs)

  findings = safety_checks(fn)
  for entry in findings:
    rules = [b["rule"] for b in entry["blockers"]]
    assert "sort-stability" not in rules, entry
