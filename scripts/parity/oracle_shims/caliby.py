"""Oracle shims for Caliby upstream sampling and noise injection.

Provides context managers to intercept non-deterministic PRNG calls in upstream
ProteinDesignLab/caliby (torch.randn, torch.randn_like, torch.rand, np.random.rand, np.random.random)
and replace them with externally injected streams without editing upstream code.

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
  "caliby/potts.py:292",
  "caliby/potts.py:620",
  "caliby/model/seq_denoiser/scn_diffusion_mlp.py:160",
  "caliby/interpolants/edm_interpolant.py:10",
  "caliby/interpolants/edm_interpolant.py:217",
)

_ORIGINAL_RANDN = torch.randn
_ORIGINAL_RANDN_LIKE = torch.randn_like
_ORIGINAL_RAND = torch.rand
_ORIGINAL_NP_RANDOM_RAND = np.random.rand
_ORIGINAL_NP_RANDOM_RANDOM = np.random.random


def shim_sha256() -> str:
  """SHA-256 of this module file, recorded in oracle_manifest.toml."""
  return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def enforce_single_threaded() -> None:
  """Set PyTorch to single-threaded CPU execution to prevent float reduction non-determinism."""
  torch.set_num_threads(1)


class CalibyGaussianCursor:
  """Cursor for stepping through injected Gaussian noise arrays for Caliby EDM packer."""

  def __init__(self, noise: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> None:
    if isinstance(noise, (list, tuple)):
      self.noise_list = list(noise)
    else:
      self.noise_list = [noise]
    self.index = 0

  def next_noise(self, shape: tuple[int, ...] | torch.Size, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if self.index >= len(self.noise_list):
      msg = f"Injected caliby gaussian stream exhausted at call {self.index} (total {len(self.noise_list)})"
      raise RuntimeError(msg)
    arr = self.noise_list[self.index]
    self.index += 1
    if arr.shape != shape:
      arr = arr.reshape(shape)
    return arr.to(dtype=dtype, device=device)


class CalibyUniformCursor:
  """Cursor for stepping through injected uniform arrays for Caliby MCMC acceptance."""

  def __init__(self, uniforms: torch.Tensor) -> None:
    self.uniforms = uniforms
    self.index = 0

  def next_scalar(self) -> float:
    if self.index >= self.uniforms.numel():
      msg = f"Injected caliby uniform stream exhausted at call {self.index}"
      raise RuntimeError(msg)
    val = float(self.uniforms.reshape(-1)[self.index].item())
    self.index += 1
    return val

  def next_array(self, shape: tuple[int, ...] | torch.Size) -> torch.Tensor:
    total = int(torch.prod(torch.tensor(shape)).item()) if shape else 1
    if self.index + total > self.uniforms.numel():
      msg = f"Injected caliby uniform stream exhausted: need {total}, remaining {self.uniforms.numel() - self.index}"
      raise RuntimeError(msg)
    flat = self.uniforms.reshape(-1)[self.index : self.index + total]
    self.index += total
    return flat.reshape(shape)


_ACTIVE_GAUSSIAN: CalibyGaussianCursor | None = None
_ACTIVE_UNIFORMS: CalibyUniformCursor | None = None


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


def _shim_rand(*size: Any, **kwargs: Any) -> torch.Tensor:
  cursor = _ACTIVE_UNIFORMS
  if cursor is not None:
    target_shape = torch.Size(size[0] if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)) else size)
    dtype = kwargs.get("dtype", torch.float32)
    device = kwargs.get("device", torch.device("cpu"))
    return cursor.next_array(target_shape).to(dtype=dtype, device=device)
  return _ORIGINAL_RAND(*size, **kwargs)


def _shim_np_rand(*shape: int) -> np.ndarray | float:
  cursor = _ACTIVE_UNIFORMS
  if cursor is not None:
    if not shape:
      return cursor.next_scalar()
    target_shape = tuple(shape)
    tensor = cursor.next_array(target_shape)
    return tensor.cpu().numpy()
  return _ORIGINAL_NP_RANDOM_RAND(*shape)


def _shim_np_random(size: Any = None) -> np.ndarray | float:
  cursor = _ACTIVE_UNIFORMS
  if cursor is not None:
    if size is None:
      return cursor.next_scalar()
    shape = tuple(size) if isinstance(size, (tuple, list)) else (int(size),)
    tensor = cursor.next_array(shape)
    return tensor.cpu().numpy()
  return _ORIGINAL_NP_RANDOM_RANDOM(size)


@contextmanager
def injected_caliby_noise(
  *,
  gaussian_noise: np.ndarray | torch.Tensor | list[torch.Tensor] | None = None,
  uniforms: np.ndarray | torch.Tensor | None = None,
) -> Iterator[dict[str, Any]]:
  """Context manager intercepting Caliby PRNG calls with externally injected arrays.

  Restores all original PyTorch and NumPy random functions upon exit.
  """
  global _ACTIVE_GAUSSIAN, _ACTIVE_UNIFORMS  # noqa: PLW0603
  enforce_single_threaded()

  prev_randn = torch.randn
  prev_randn_like = torch.randn_like
  prev_rand = torch.rand
  prev_np_rand = np.random.rand
  prev_np_random = np.random.random

  cursor_dict: dict[str, Any] = {}

  if gaussian_noise is not None:
    if isinstance(gaussian_noise, (list, tuple)):
      noise_tensors = [torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32) for x in gaussian_noise]
      _ACTIVE_GAUSSIAN = CalibyGaussianCursor(noise_tensors)
    else:
      noise_tensor = torch.as_tensor(np.asarray(gaussian_noise, dtype=np.float32), dtype=torch.float32)
      _ACTIVE_GAUSSIAN = CalibyGaussianCursor(noise_tensor)
    cursor_dict["gaussian"] = _ACTIVE_GAUSSIAN
    torch.randn = _shim_randn  # type: ignore[assignment]
    torch.randn_like = _shim_randn_like  # type: ignore[assignment]

  if uniforms is not None:
    u_tensor = torch.as_tensor(np.asarray(uniforms, dtype=np.float64), dtype=torch.float64)
    _ACTIVE_UNIFORMS = CalibyUniformCursor(u_tensor)
    cursor_dict["uniforms"] = _ACTIVE_UNIFORMS
    torch.rand = _shim_rand  # type: ignore[assignment]
    np.random.rand = _shim_np_rand  # type: ignore[assignment]
    np.random.random = _shim_np_random  # type: ignore[assignment]

  try:
    yield cursor_dict
  finally:
    torch.randn = prev_randn  # type: ignore[assignment]
    torch.randn_like = prev_randn_like  # type: ignore[assignment]
    torch.rand = prev_rand  # type: ignore[assignment]
    np.random.rand = prev_np_rand  # type: ignore[assignment]
    np.random.random = prev_np_random  # type: ignore[assignment]
    _ACTIVE_GAUSSIAN = None
    _ACTIVE_UNIFORMS = None
