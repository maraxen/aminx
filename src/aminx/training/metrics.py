"""Training and evaluation metrics.

TODO(io_callback integration): ``to_dict`` uses ``device_get`` for small scalars (OK for logs).
Revisit only if emitting large telemetry from traced steps (see aminx/TODO_io_callback.txt).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class TrainingMetrics(eqx.Module):
  """Container for training metrics."""

  loss: jax.Array
  accuracy: jax.Array
  perplexity: jax.Array
  learning_rate: jax.Array | float | None = None
  grad_norm: jax.Array | None = None

  def to_dict(self) -> dict[str, float | None]:
    """Convert metrics to a dictionary of Python floats."""
    metrics_dict: dict[str, float | None] = {
      "loss": float(jax.device_get(self.loss)),
      "accuracy": float(jax.device_get(self.accuracy)),
      "perplexity": float(jax.device_get(self.perplexity)),
    }
    if self.learning_rate is not None:
      metrics_dict["learning_rate"] = float(jax.device_get(self.learning_rate))
    if self.grad_norm is not None:
      metrics_dict["grad_norm"] = float(jax.device_get(self.grad_norm))
    return metrics_dict


class EvaluationMetrics(eqx.Module):
  """Container for evaluation metrics."""

  val_loss: jax.Array
  val_accuracy: jax.Array
  val_perplexity: jax.Array

  def to_dict(self) -> dict[str, float]:
    """Convert metrics to a dictionary of Python floats."""
    return {
      "val_loss": float(jax.device_get(self.val_loss)),
      "val_accuracy": float(jax.device_get(self.val_accuracy)),
      "val_perplexity": float(jax.device_get(self.val_perplexity)),
    }


def compute_grad_norm(grads: dict) -> jax.Array:
  """Compute global gradient norm across all parameters.

  Args:
      grads: PyTree of gradients

  Returns:
      Global gradient norm

  """
  leaves = jax.tree_util.tree_leaves(grads)
  return jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in leaves))
