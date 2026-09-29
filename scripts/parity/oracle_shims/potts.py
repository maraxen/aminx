"""Inverse-CDF draw shim and tied-order injection for upstream PottsMPNN.

Shim 1 replaces ``torch.multinomial`` (the call ``Categorical.sample`` makes
internally) at every Potts draw site in spec §7.1 and the T0.2 transcription:

- ``potts_mpnn_utils.py:1405`` ``PottsMPNN.sample``
- ``potts_mpnn_utils.py:1480`` ``PottsMPNN.decoder``
- ``potts_mpnn_utils.py:1590`` ``PottsMPNN.tied_sample``
- ``potts_mpnn_utils.py:1680`` ``PottsMPNN.tied_decoder``
- ``run_utils.py:174`` ``optimize_sequence`` potts / potts_converge
- ``run_utils.py:263`` ``optimize_sequence`` nodes
- ``run_utils.py:421`` ``tied_optimize_sequence`` potts / potts_converge
- ``run_utils.py:512`` ``tied_optimize_sequence`` nodes

The rule is ``c = cumsum(p.double())``; ``i = searchsorted(c, u * c[-1], right=True)``;
``i = min(i, last index with p > 0)``. ``u`` is an injected float64 stream indexed
``(sample, step[, chi])``. One ``torch.multinomial`` call consumes one step
(and, when the stream has a chi axis, one chi, wrapping onto the next step).

Shim 2 does not patch the untied decoder: pass ``decoding_order=``. Tied
``tied_decoder`` / ``tied_sample`` have no order argument; ``injected_tied_randn``
replaces the ``randn`` they argsort and records the flattened ``decoding_order``
they return.

Shim 3 (``weight_dtype_positional_encodings``, f64 dumps only) replaces the
``d_onehot.float()`` cast at ``potts_mpnn_utils.py:940`` with a cast to the linear
weight dtype so a ``model.double()`` forward runs in float64 end to end.

With no context manager entered, ``torch.multinomial`` and the tied methods
are the originals.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

SHIM_SITES: tuple[str, ...] = (
  "potts_mpnn_utils.py:1405",
  "potts_mpnn_utils.py:1480",
  "potts_mpnn_utils.py:1590",
  "potts_mpnn_utils.py:1680",
  "run_utils.py:174",
  "run_utils.py:263",
  "run_utils.py:421",
  "run_utils.py:512",
  "potts_mpnn_utils.py:940",
)

_ORIGINAL_MULTINOMIAL = torch.multinomial


class DrawCursor:
  """Step / chi cursor for one ``injected_uniform_draws`` block."""

  def __init__(self, uniforms: torch.Tensor) -> None:
    self.uniforms = uniforms
    self.step = 0
    self.chi = 0

  @property
  def consumed_steps(self) -> int:
    return self.step


_ACTIVE: DrawCursor | None = None


def shim_sha256() -> str:
  """SHA-256 of this module file, recorded in ``oracle_manifest.toml``."""
  return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def inverse_cdf_index(probs: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
  """Inverse-CDF indices for ``probs`` (..., V) and ``uniform`` matching the leading dims.

  ``uniform`` is promoted to float64. The result is int64.
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


def _take_uniform(cursor: DrawCursor, n_rows: int) -> torch.Tensor:
  uniforms = cursor.uniforms
  if cursor.step >= uniforms.shape[1]:
    msg = f"injected uniform stream exhausted at step {cursor.step} (length {uniforms.shape[1]})"
    raise RuntimeError(msg)
  if uniforms.ndim == 2:
    column = uniforms[:, cursor.step]
    cursor.step += 1
  elif uniforms.ndim == 3:
    column = uniforms[:, cursor.step, cursor.chi]
    cursor.chi += 1
    if cursor.chi >= uniforms.shape[2]:
      cursor.chi = 0
      cursor.step += 1
  else:
    msg = "uniforms must have shape (sample, step) or (sample, step, chi)"
    raise RuntimeError(msg)
  if column.numel() == 1 and n_rows != 1:
    column = column.expand(n_rows)
  if column.numel() != n_rows:
    msg = f"uniform sample axis {column.numel()} does not match multinomial rows {n_rows}"
    raise RuntimeError(msg)
  return column.reshape(n_rows)


def _shim_multinomial(
  probs: torch.Tensor,
  num_samples: int,
  replacement: bool = False,  # noqa: FBT001, FBT002
  *args: object,
  **kwargs: object,
) -> torch.Tensor:
  del replacement, args, kwargs
  cursor = _ACTIVE
  if cursor is None:
    msg = "potts multinomial shim is installed without an active uniform stream"
    raise RuntimeError(msg)
  if num_samples != 1:
    msg = "potts inverse-CDF shim implements num_samples=1 only"
    raise RuntimeError(msg)
  if probs.ndim == 1:
    unit = _take_uniform(cursor, 1)
    index = inverse_cdf_index(probs, unit.reshape(()))
    return index.reshape(1)
  n_rows = int(probs[..., 0].numel())
  flat = probs.reshape(n_rows, probs.shape[-1])
  unit = _take_uniform(cursor, n_rows)
  index = inverse_cdf_index(flat, unit)
  return index.reshape(*probs.shape[:-1], 1)


@contextmanager
def injected_uniform_draws(uniforms: np.ndarray | torch.Tensor) -> Iterator[DrawCursor]:
  """Patch ``torch.multinomial`` for the block; restore it on exit.

  ``uniforms`` is float64 with shape ``(sample, step)`` or ``(sample, step, chi)``.
  """
  global _ACTIVE  # noqa: PLW0603
  tensor = torch.as_tensor(np.asarray(uniforms, dtype=np.float64), dtype=torch.float64)
  if tensor.ndim not in (2, 3):
    msg = "uniforms must have shape (sample, step) or (sample, step, chi)"
    raise ValueError(msg)
  if _ACTIVE is not None:
    msg = "injected_uniform_draws cannot nest"
    raise RuntimeError(msg)
  cursor = DrawCursor(tensor)
  previous = torch.multinomial
  _ACTIVE = cursor
  torch.multinomial = _shim_multinomial  # type: ignore[assignment]
  try:
    yield cursor
  finally:
    torch.multinomial = previous  # type: ignore[assignment]
    _ACTIVE = None


def untied_decoding_order(order: torch.Tensor) -> torch.Tensor:
  """Return ``order`` for ``decoder(..., decoding_order=)``. Untied order is not patched."""
  return order


@contextmanager
def injected_tied_randn(
  model_cls: object,
  randn: torch.Tensor,
) -> Iterator[list[torch.Tensor]]:
  """Replace ``randn`` inside ``tied_decoder`` and ``tied_sample``; record flattened orders.

  The yielded list receives each returned ``decoding_order`` (already flattened by upstream).
  """
  recorded: list[torch.Tensor] = []
  original_decoder = model_cls.tied_decoder
  original_sample = model_cls.tied_sample

  def tied_decoder(
    self: object,
    h_v: torch.Tensor,
    e_idx: torch.Tensor,
    h_e: torch.Tensor,
    randn_in: torch.Tensor,
    *args: object,
    **kwargs: object,
  ) -> object:
    use = randn.to(device=randn_in.device, dtype=randn_in.dtype)
    out = original_decoder(self, h_v, e_idx, h_e, use, *args, **kwargs)
    recorded.append(out[0]["decoding_order"].detach())
    return out

  def tied_sample(
    self: object,
    x: torch.Tensor,
    randn_in: torch.Tensor,
    *args: object,
    **kwargs: object,
  ) -> object:
    use = randn.to(device=randn_in.device, dtype=randn_in.dtype)
    out = original_sample(self, x, use, *args, **kwargs)
    recorded.append(out[0]["decoding_order"].detach())
    return out

  model_cls.tied_decoder = tied_decoder
  model_cls.tied_sample = tied_sample
  try:
    yield recorded
  finally:
    model_cls.tied_decoder = original_decoder
    model_cls.tied_sample = original_sample


@contextmanager
def weight_dtype_positional_encodings(encodings_cls: object) -> Iterator[None]:
  """Cast the relative-offset one-hot to the linear weight dtype in ``PositionalEncodings``.

  Upstream ``potts_mpnn_utils.py:940`` hard-casts with ``d_onehot.float()``, which breaks a
  ``model.double()`` forward. Under float32 weights the patched forward is the original.
  """
  original = encodings_cls.forward

  def forward(self: torch.nn.Module, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    max_rel = self.max_relative_feature
    d = torch.clip(offset + max_rel, 0, 2 * max_rel) * mask + (1 - mask) * (2 * max_rel + 1)
    d_onehot = torch.nn.functional.one_hot(d, 2 * max_rel + 1 + 1)
    return self.linear(d_onehot.to(self.linear.weight.dtype))

  encodings_cls.forward = forward
  try:
    yield
  finally:
    encodings_cls.forward = original
