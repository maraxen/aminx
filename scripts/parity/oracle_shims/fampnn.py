"""Oracle shims for FAMPNN upstream sampling and noise injection.

Provides context managers to intercept non-deterministic PRNG calls in upstream
richardshuai/fampnn (Categorical.sample, torch.multinomial, torch.randn, torch.rand)
and replace them with externally injected streams without editing upstream code.

Enforces single-threaded execution (torch.set_num_threads(1)) to eliminate multi-threaded
float reduction non-determinism.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

SHIM_SITES: tuple[str, ...] = (
  "fampnn/sampling/seq_design.py:72",
  "fampnn/interpolants/mar_interpolant.py:12",
  "fampnn/interpolants/mar_interpolant.py:162",
  "fampnn/interpolants/edm_interpolant.py:55",
  "fampnn/interpolants/edm_interpolant.py:248",
)

_ORIGINAL_MULTINOMIAL = torch.multinomial
_ORIGINAL_CATEGORICAL_SAMPLE = torch.distributions.Categorical.sample
_ORIGINAL_RAND = torch.rand
_ORIGINAL_RANDN = torch.randn
_ORIGINAL_RANDN_LIKE = torch.randn_like


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


class UniformStreamCursor:
  """Cursor for stepping through injected uniform arrays."""

  def __init__(self, uniforms: torch.Tensor) -> None:
    self.uniforms = uniforms
    self.step = 0

  def next_uniform(self, expected_shape: tuple[int, ...] | torch.Size) -> torch.Tensor:
    if self.step >= self.uniforms.shape[0]:
      msg = f"Injected uniform stream exhausted at step {self.step} (total {self.uniforms.shape[0]})"
      raise RuntimeError(msg)
    val = self.uniforms[self.step]
    self.step += 1
    if val.shape != expected_shape:
      val = val.reshape(expected_shape)
    return val


class GaussianStreamCursor:
  """Cursor for stepping through injected Gaussian noise arrays."""

  def __init__(self, noise: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> None:
    if isinstance(noise, (list, tuple)):
      self.noise_list = list(noise)
    else:
      self.noise_list = [noise]
    self.index = 0

  def next_noise(self, shape: tuple[int, ...] | torch.Size, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if self.index >= len(self.noise_list):
      msg = f"Injected gaussian stream exhausted at call {self.index} (total {len(self.noise_list)})"
      raise RuntimeError(msg)
    arr = self.noise_list[self.index]
    self.index += 1
    if arr.shape != shape:
      arr = arr.reshape(shape)
    return arr.to(dtype=dtype, device=device)


_ACTIVE_UNIFORMS: UniformStreamCursor | None = None
_ACTIVE_UNMASK_RAND: torch.Tensor | None = None
_ACTIVE_GAUSSIAN: GaussianStreamCursor | None = None


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
    msg = "FAMPNN inverse-CDF shim implements num_samples=1 only"
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


def _shim_rand(*size: Any, **kwargs: Any) -> torch.Tensor:
  global _ACTIVE_UNMASK_RAND  # noqa: PLW0603
  if _ACTIVE_UNMASK_RAND is not None:
    # Consume active unmask rand
    res = _ACTIVE_UNMASK_RAND
    _ACTIVE_UNMASK_RAND = None
    target_shape = torch.Size(size[0] if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)) else size)
    if res.shape != target_shape:
      res = res.reshape(target_shape)
    dtype = kwargs.get("dtype", torch.float32)
    device = kwargs.get("device", torch.device("cpu"))
    return res.to(dtype=dtype, device=device)
  return _ORIGINAL_RAND(*size, **kwargs)


def _shim_randn(*size: Any, **kwargs: Any) -> torch.Tensor:
  cursor = _ACTIVE_GAUSSIAN
  if cursor is not None:
    target_shape = torch.Size(size[0] if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)) else size)
    dtype = kwargs.get("dtype", torch.float32)
    device = kwargs.get("device", torch.device("cpu"))
    return cursor.next_noise(target_shape, dtype=dtype, device=device)
  return _ORIGINAL_RANDN(*size, **kwargs)


def _shim_randn_like(input_tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
  cursor = _ACTIVE_GAUSSIAN
  if cursor is not None:
    dtype = kwargs.get("dtype", input_tensor.dtype)
    device = kwargs.get("device", input_tensor.device)
    return cursor.next_noise(input_tensor.shape, dtype=dtype, device=device)
  return _ORIGINAL_RANDN_LIKE(input_tensor, **kwargs)


@contextmanager
def injected_fampnn_noise(
  *,
  uniforms: np.ndarray | torch.Tensor | None = None,
  unmask_priorities: np.ndarray | torch.Tensor | None = None,
  sidechain_noise: np.ndarray | torch.Tensor | list[torch.Tensor] | None = None,
) -> Iterator[dict[str, Any]]:
  """Context manager intercepting FAMPNN PRNG calls with externally injected arrays.

  Restores all original PyTorch functions upon exit.
  """
  global _ACTIVE_UNIFORMS, _ACTIVE_UNMASK_RAND, _ACTIVE_GAUSSIAN  # noqa: PLW0603
  enforce_single_threaded()

  prev_multinomial = torch.multinomial
  prev_categorical = torch.distributions.Categorical.sample
  prev_rand = torch.rand
  prev_randn = torch.randn
  prev_randn_like = torch.randn_like

  cursor_dict: dict[str, Any] = {}

  if uniforms is not None:
    u_tensor = torch.as_tensor(np.asarray(uniforms, dtype=np.float64), dtype=torch.float64)
    _ACTIVE_UNIFORMS = UniformStreamCursor(u_tensor)
    cursor_dict["uniforms"] = _ACTIVE_UNIFORMS
    torch.multinomial = _shim_multinomial  # type: ignore[assignment]
    torch.distributions.Categorical.sample = _shim_categorical_sample  # type: ignore[method-assign]

  if unmask_priorities is not None:
    _ACTIVE_UNMASK_RAND = torch.as_tensor(np.asarray(unmask_priorities, dtype=np.float32), dtype=torch.float32)
    cursor_dict["unmask"] = _ACTIVE_UNMASK_RAND
    torch.rand = _shim_rand  # type: ignore[assignment]

  if sidechain_noise is not None:
    if isinstance(sidechain_noise, (list, tuple)):
      noise_tensors = [torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32) for x in sidechain_noise]
      _ACTIVE_GAUSSIAN = GaussianStreamCursor(noise_tensors)
    else:
      noise_tensor = torch.as_tensor(np.asarray(sidechain_noise, dtype=np.float32), dtype=torch.float32)
      _ACTIVE_GAUSSIAN = GaussianStreamCursor(noise_tensor)
    cursor_dict["gaussian"] = _ACTIVE_GAUSSIAN
    torch.randn = _shim_randn  # type: ignore[assignment]
    torch.randn_like = _shim_randn_like  # type: ignore[assignment]

  try:
    yield cursor_dict
  finally:
    torch.multinomial = prev_multinomial  # type: ignore[assignment]
    torch.distributions.Categorical.sample = prev_categorical  # type: ignore[method-assign]
    torch.rand = prev_rand  # type: ignore[assignment]
    torch.randn = prev_randn  # type: ignore[assignment]
    torch.randn_like = prev_randn_like  # type: ignore[assignment]
    _ACTIVE_UNIFORMS = None
    _ACTIVE_UNMASK_RAND = None
    _ACTIVE_GAUSSIAN = None
