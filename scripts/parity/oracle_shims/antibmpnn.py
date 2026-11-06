"""Oracle shims for AntiBMPNN upstream sampling and noise injection.

Provides context managers to intercept non-deterministic PRNG calls in upstream
zeysun/AntiBMPNN and underlying ProteinMPNN sampling routines (torch.randn for decoding
order permutations, torch.multinomial and Categorical.sample for residue selection),
replacing them with externally injected streams without editing upstream code.

Enforces single-threaded execution (torch.set_num_threads(1)) to eliminate multi-threaded
reduction non-determinism.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

SHIM_SITES: tuple[str, ...] = (
  "Running_AntiBMPNN_run.py:35",
  "Running_AntiBMPNN_run.py:163",
  "Running_AntiBMPNN_run.py:295",
  "ProteinMPNN/protein_mpnn_utils.py:sample",
)

_ORIGINAL_MULTINOMIAL = torch.multinomial
_ORIGINAL_CATEGORICAL_SAMPLE = torch.distributions.Categorical.sample
_ORIGINAL_RANDN = torch.randn


def shim_sha256() -> str:
  """SHA-256 of this module file, recorded in oracle_manifest.toml."""
  return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def enforce_single_threaded() -> None:
  """Set PyTorch to single-threaded CPU execution to prevent float reduction non-determinism."""
  torch.set_num_threads(1)


def inverse_cdf_index(probs: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
  """Inverse-CDF indices for probs (..., V) and uniform matching leading dims.

  Both probs and uniform are promoted to float64. The result is int64.
  """
  probabilities = probs.detach().to(dtype=torch.float64)
  unit = uniform.detach().to(dtype=torch.float64, device=probabilities.device)
  cumulative = torch.cumsum(probabilities, dim=-1)
  target = unit * cumulative[..., -1]
  index = torch.searchsorted(cumulative, target.unsqueeze(-1), right=True).squeeze(-1)
  vocab = probabilities.shape[-1]
  positions = torch.arange(vocab, device=probabilities.device, dtype=torch.long)
  positive = probabilities > 0
  last_positive = torch.where(positive, positions, torch.zeros_like(positions)).amax(dim=-1)
  clamped = torch.minimum(index.to(dtype=torch.long), last_positive)
  return torch.clamp(clamped, min=0, max=vocab - 1)


class AntiBMPNNUniformCursor:
  """Cursor for stepping through injected uniform arrays for token draws."""

  def __init__(self, uniforms: torch.Tensor) -> None:
    self.uniforms = uniforms
    self.step = 0

  def next_uniform(self, expected_shape: tuple[int, ...] | torch.Size) -> torch.Tensor:
    if self.step >= self.uniforms.shape[0]:
      msg = f"Injected AntiBMPNN uniform stream exhausted at step {self.step} (total {self.uniforms.shape[0]})"
      raise RuntimeError(msg)
    val = self.uniforms[self.step]
    self.step += 1
    if val.shape != expected_shape:
      val = val.reshape(expected_shape)
    return val


_ACTIVE_UNIFORMS: AntiBMPNNUniformCursor | None = None
_ACTIVE_ORDER_RANDN: torch.Tensor | None = None


def _shim_multinomial(
  probs: torch.Tensor,
  num_samples: int,
  replacement: bool = False,
  *args: Any,
  **kwargs: Any,
) -> torch.Tensor:
  del replacement, args, kwargs
  cursor = _ACTIVE_UNIFORMS
  if cursor is None:
    return _ORIGINAL_MULTINOMIAL(probs, num_samples)
  if num_samples != 1:
    msg = "AntiBMPNN inverse-CDF shim implements num_samples=1 only"
    raise RuntimeError(msg)
  if probs.ndim == 1:
    unit = cursor.next_uniform(())
    idx = inverse_cdf_index(probs, unit)
    return idx.reshape(1)
  n_rows = int(probs[..., 0].numel())
  flat = probs.reshape(n_rows, probs.shape[-1])
  unit = cursor.next_uniform((n_rows,))
  idx = inverse_cdf_index(flat, unit)
  return idx.reshape(*probs.shape[:-1], 1)


def _shim_categorical_sample(
  self: torch.distributions.Categorical,
  sample_shape: torch.Size | None = None,
) -> torch.Tensor:
  cursor = _ACTIVE_UNIFORMS
  shape = torch.Size() if sample_shape is None else sample_shape
  if cursor is None:
    return _ORIGINAL_CATEGORICAL_SAMPLE(self, shape)
  probs = self.probs
  if probs.ndim == 1:
    unit = cursor.next_uniform(())
    return inverse_cdf_index(probs, unit)
  n_rows = int(probs[..., 0].numel())
  flat = probs.reshape(n_rows, probs.shape[-1])
  unit = cursor.next_uniform((n_rows,))
  idx = inverse_cdf_index(flat, unit)
  return idx.reshape(probs.shape[:-1])


def _shim_randn(*size: Any, **kwargs: Any) -> torch.Tensor:
  global _ACTIVE_ORDER_RANDN  # noqa: PLW0603
  if _ACTIVE_ORDER_RANDN is not None:
    res = _ACTIVE_ORDER_RANDN
    _ACTIVE_ORDER_RANDN = None
    target_shape = torch.Size(size[0] if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)) else size)
    if res.shape != target_shape:
      res = res.reshape(target_shape)
    dtype = kwargs.get("dtype", torch.float32)
    device = kwargs.get("device", torch.device("cpu"))
    return res.to(dtype=dtype, device=device)
  return _ORIGINAL_RANDN(*size, **kwargs)


@contextmanager
def injected_antibmpnn_noise(
  *,
  order_noise: np.ndarray | torch.Tensor | None = None,
  uniforms: np.ndarray | torch.Tensor | None = None,
) -> Iterator[dict[str, Any]]:
  """Context manager intercepting AntiBMPNN PRNG calls with externally injected arrays.

  Restores all original PyTorch random functions upon exit.
  """
  global _ACTIVE_UNIFORMS, _ACTIVE_ORDER_RANDN  # noqa: PLW0603
  enforce_single_threaded()

  prev_multinomial = torch.multinomial
  prev_categorical = torch.distributions.Categorical.sample
  prev_randn = torch.randn

  cursor_dict: dict[str, Any] = {}

  if uniforms is not None:
    u_tensor = torch.as_tensor(np.asarray(uniforms, dtype=np.float64), dtype=torch.float64)
    _ACTIVE_UNIFORMS = AntiBMPNNUniformCursor(u_tensor)
    cursor_dict["uniforms"] = _ACTIVE_UNIFORMS
    torch.multinomial = _shim_multinomial  # type: ignore[assignment]
    torch.distributions.Categorical.sample = _shim_categorical_sample  # type: ignore[method-assign]

  if order_noise is not None:
    _ACTIVE_ORDER_RANDN = torch.as_tensor(np.asarray(order_noise, dtype=np.float32), dtype=torch.float32)
    cursor_dict["order"] = _ACTIVE_ORDER_RANDN
    torch.randn = _shim_randn  # type: ignore[assignment]

  try:
    yield cursor_dict
  finally:
    torch.multinomial = prev_multinomial  # type: ignore[assignment]
    torch.distributions.Categorical.sample = prev_categorical  # type: ignore[method-assign]
    torch.randn = prev_randn  # type: ignore[assignment]
    _ACTIVE_UNIFORMS = None
    _ACTIVE_ORDER_RANDN = None
