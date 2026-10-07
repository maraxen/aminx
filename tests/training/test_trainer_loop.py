"""Weight-free checks for training-loop correctness (debt #2535).

Covers the shared weight-decay mask, gradient-accumulation equivalence, and the
resume state builder. No model weights.
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from xtrax.training.types import ResumableState

from aminx.training.specs import TrainingSpecification
from aminx.training.trainer import (
  _accumulate_value_and_grad,
  _assemble_resumable_state,
  create_optimizer,
)


class _DecayToy(eqx.Module):
  """One 2-D weight and one 1-D bias, both inexact."""

  weight: jax.Array
  bias: jax.Array


def _spec(
  tmp_path: Path,
  *,
  warmup_steps: int,
  weight_decay_mask: str,
) -> TrainingSpecification:
  return TrainingSpecification(
    inputs="toy",
    checkpoint_dir=tmp_path / f"ck-{warmup_steps}-{weight_decay_mask}",
    learning_rate=1.0,
    weight_decay=0.5,
    warmup_steps=warmup_steps,
    total_steps=32,
    weight_decay_mask=weight_decay_mask,  # type: ignore[arg-type]
    num_epochs=1,
    batch_size=2,
    accum_steps=1,
    gradient_clip=None,
  )


def _one_update(optimizer: optax.GradientTransformation, model: _DecayToy, *, schedule_count: int | None) -> _DecayToy:
  """One AdamW step with a zero gradient.

  Warmup cosine starts at 0, so a step-0 update cannot show decay. When
  ``schedule_count`` is set, the schedule counter is moved there first and this
  is still a single update (at the peak, ``count == warmup_steps``).
  """
  params = eqx.filter(model, eqx.is_inexact_array)
  opt_state = optimizer.init(params)
  if schedule_count is not None:
    schedule_state = opt_state[-1]
    opt_state = (*opt_state[:-1], schedule_state._replace(count=jnp.int32(schedule_count)))
  grads = jax.tree.map(jnp.zeros_like, params)
  updates, _opt_state = optimizer.update(grads, opt_state, params)
  return eqx.apply_updates(model, updates)


@pytest.mark.parametrize("warmup_steps", [0, 4])
@pytest.mark.parametrize("weight_decay_mask", ["no_bias", "all"])
def test_weight_decay_mask_is_shared_by_both_optimizer_paths(
  tmp_path: Path,
  warmup_steps: int,
  weight_decay_mask: str,
) -> None:
  spec = _spec(tmp_path, warmup_steps=warmup_steps, weight_decay_mask=weight_decay_mask)
  model = _DecayToy(weight=jnp.ones((2, 2)), bias=jnp.ones((2,)))
  schedule_count = warmup_steps if warmup_steps > 0 else None
  updated = _one_update(create_optimizer(spec), model, schedule_count=schedule_count)

  if weight_decay_mask == "no_bias":
    assert jnp.allclose(updated.bias, model.bias)
    assert not jnp.allclose(updated.weight, model.weight)
  else:
    assert not jnp.allclose(updated.bias, model.bias)
    assert not jnp.allclose(updated.weight, model.weight)


def test_weight_decay_mask_default_and_validation(tmp_path: Path) -> None:
  spec = TrainingSpecification(inputs="toy", checkpoint_dir=tmp_path / "default")
  assert spec.weight_decay_mask == "no_bias"
  with pytest.raises(ValueError, match="weight_decay_mask"):
    TrainingSpecification(
      inputs="toy",
      checkpoint_dir=tmp_path / "bad",
      weight_decay_mask="biases",  # type: ignore[arg-type]
    )


class _Affine(eqx.Module):
  weight: jax.Array
  bias: jax.Array

  def __call__(self, x: jax.Array) -> jax.Array:
    return x @ self.weight + self.bias


def _micro_loss(model: _Affine, micro: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
  x, y = micro
  pred = jax.vmap(model)(x)
  return jnp.mean((pred - y) ** 2), pred


def test_accumulate_matches_full_batch() -> None:
  key = jax.random.PRNGKey(0)
  key, k_w, k_b, k_x, k_y = jax.random.split(key, 5)
  model = _Affine(weight=jax.random.normal(k_w, (3, 2)), bias=jax.random.normal(k_b, (2,)))
  x = jax.random.normal(k_x, (4, 3))
  y = jax.random.normal(k_y, (4, 2))

  def full_loss(m: _Affine) -> tuple[jax.Array, jax.Array]:
    return _micro_loss(m, (x, y))

  (loss, logits), grads = eqx.filter_value_and_grad(full_loss, has_aux=True)(model)
  loss_a, logits_a, grads_a = _accumulate_value_and_grad(
    model,
    _micro_loss,
    (x.reshape(2, 2, 3), y.reshape(2, 2, 2)),
    batch_size=4,
    accum_steps=2,
  )

  assert logits_a.shape == logits.shape == (4, 2)
  assert jnp.allclose(logits_a, logits)
  assert jnp.allclose(loss_a, loss)
  for leaf, leaf_a in zip(jax.tree.leaves(grads), jax.tree.leaves(grads_a), strict=True):
    assert jnp.allclose(leaf, leaf_a)


def test_accumulate_rejects_non_divisible_batch() -> None:
  model = _Affine(weight=jnp.ones((3, 2)), bias=jnp.ones((2,)))
  with pytest.raises(ValueError, match="divisible"):
    _accumulate_value_and_grad(
      model,
      _micro_loss,
      (jnp.ones((3, 3)), jnp.ones((3, 2))),
      batch_size=3,
      accum_steps=2,
    )


def test_resume_state_keeps_restored_key_and_extras() -> None:
  model = eqx.nn.Linear(2, 2, key=jax.random.PRNGKey(0))
  restored = ResumableState(
    step=jnp.int32(6),
    key=jax.random.PRNGKey(99),
    model=model,
    opt_state=None,
    extras={"epoch": jnp.int32(2), "marker": jnp.int32(11)},
  )
  state = _assemble_resumable_state(
    model=model,
    opt_state=None,
    random_seed=0,
    restored=restored,
  )
  assert int(state.step) == 6
  assert jnp.array_equal(state.key, restored.key)
  assert int(state.extras["epoch"]) == 2
  assert int(state.extras["marker"]) == 11


def test_fresh_state_uses_spec_seed() -> None:
  model = eqx.nn.Linear(2, 2, key=jax.random.PRNGKey(0))
  state = _assemble_resumable_state(
    model=model,
    opt_state=None,
    random_seed=7,
    restored=None,
  )
  assert int(state.step) == 0
  assert jnp.array_equal(state.key, jax.random.PRNGKey(7))
  assert int(state.extras["epoch"]) == 0
