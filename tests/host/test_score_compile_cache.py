"""``runner.score`` must not recompile on a second identical-shape call (debt #3484).

On 0.5.3 every ``score()`` call rebuilt ``make_score_split_fns``'s ``jax.jit`` closures over a
freshly loaded model, and re-lowered the candidate activation estimate, so a warm call cost three
backend compiles (run b25cbd4e: 2.5 s warm vs 5.8 s cold at L=76 on CPU). tev_design calls score()
once per decoding order, so that was paid K times per screen.

What this file pins:

* the compile counter sees compiles at all (instrument control: a never-seen jitted function
  must register exactly one compile), so a zero below is not vacuous;
* a second ``score()`` call with identical shapes and a DIFFERENT ar_mask triggers zero backend
  compiles, and returns the same numbers as the first call's path would (the mask changed, so
  the logits differ; the default-mask calls are compared bitwise);
* two independently loaded models of one architecture share the split kernels' executables.

task_id: 261118_aminx-campaign-teacher-forced-scoring
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.host.runner import score
from aminx.run.specs import ScoringSpecification
from aminx.utils.autoregression import ar_mask_from_decoding_order

_BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"
_STRUCTURE = "tests/data/1ubq.pdb"
_CHECKPOINT = "proteinmpnn_v_48_020"
_UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
# A max_length no other test in this process uses, so call 1 is genuinely cold for this file.
_MAX_LENGTH = 96

_COMPILES = [0]


def _on_event(event: str, _duration: float, **_kwargs: object) -> None:
  if event == _BACKEND_COMPILE_EVENT:
    _COMPILES[0] += 1


jax.monitoring.register_event_duration_secs_listener(_on_event)


def _causal_mask(seed: int) -> np.ndarray:
  order = np.random.default_rng(seed).permutation(len(_UBIQUITIN)).astype(np.int32)
  return np.asarray(ar_mask_from_decoding_order(jnp.asarray(order)))


def _score(ar_mask: np.ndarray | None) -> dict[str, Any]:
  spec = ScoringSpecification(
    inputs=[_STRUCTURE],
    checkpoint_id=_CHECKPOINT,
    sequences_to_score=[_UBIQUITIN],
    max_length=_MAX_LENGTH,
    return_logits=True,
    ar_mask=ar_mask,
    backbone_noise=0.0,
  )
  out = score(spec)
  jax.block_until_ready(out["logits"])
  return out


def _compiles_during(fn: Callable[[], Any]) -> tuple[int, Any]:
  before = _COMPILES[0]
  out = fn()
  return _COMPILES[0] - before, out


def test_counter_sees_a_compile() -> None:
  """Instrument control: a brand-new jitted function compiles on first use, then never again."""

  @jax.jit
  def _fresh(x: jax.Array) -> jax.Array:
    return jnp.sin(x) * 3.0 + 0.123456

  x = jax.block_until_ready(jnp.ones(7))  # built outside the window: creating it compiles too
  first, _ = _compiles_during(lambda: jax.block_until_ready(_fresh(x)))
  second, _ = _compiles_during(lambda: jax.block_until_ready(_fresh(x)))
  assert first >= 1
  assert second == 0


def test_second_identical_shape_score_call_does_not_recompile() -> None:
  _score(_causal_mask(0))  # warm every kernel at this shape
  n_compiles, out = _compiles_during(lambda: _score(_causal_mask(1)))
  assert n_compiles == 0, f"warm score() call compiled {n_compiles} executables (#3484)"
  assert out["logits"].shape[-1] == 21


def test_warm_default_mask_call_is_bitwise_stable() -> None:
  """Same spec twice: zero compiles, identical bits (the cache did not change the program)."""
  a = _score(None)
  n_compiles, b = _compiles_during(lambda: _score(None))
  assert n_compiles == 0
  np.testing.assert_array_equal(np.asarray(a["logits"]), np.asarray(b["logits"]))
  np.testing.assert_array_equal(np.asarray(a["scores"]), np.asarray(b["scores"]))


def test_two_loaded_models_share_split_kernels() -> None:
  from aminx.io.weights import load_model  # noqa: PLC0415
  from aminx.scoring.score import make_score_split_fns  # noqa: PLC0415

  coords = jnp.zeros((1, 40, 37, 3), jnp.float32) + jnp.arange(40, dtype=jnp.float32)[None, :, None, None]
  mask = jnp.ones((1, 40), jnp.float32)
  residue_index = jnp.arange(40, dtype=jnp.int32)[None]
  chain_index = jnp.zeros((1, 40), jnp.int32)

  def _encode(model: object) -> object:
    return jax.block_until_ready(
      make_score_split_fns(model).encode_structure(coords, mask, residue_index, chain_index),  # type: ignore[arg-type]
    )

  _encode(load_model(checkpoint_id=_CHECKPOINT))
  n_compiles, _ = _compiles_during(lambda: _encode(load_model(checkpoint_id=_CHECKPOINT)))
  assert n_compiles == 0


if __name__ == "__main__":  # pragma: no cover
  pytest.main([__file__, "-v"])
