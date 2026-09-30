"""Inverse-CDF, decoding-order, dropout, and f64 dtype shims for upstream LASErMPNN.

Vendored sources are never edited. With no context manager entered, ``Categorical.sample``,
``torch.rand``, ``nn.Dropout.forward``, and ``Tensor.float`` are the originals.

Shim 1 replaces ``Categorical.sample`` (it calls ``torch.multinomial``) at the §7.1 draw
sites. Entropy-decoder sites ``utils/model.py:1065`` and ``:1104`` are not in
``SHIM_SITES`` and keep the original sampler. The rule is
``c = cumsum(p.double())``; ``i = searchsorted(c, u * c[-1], right=True)``;
``i = min(i, last index with p > 0)``. ``u`` is an injected float64 stream indexed
``(sample, step[, chi])``. One ``Categorical.sample`` call consumes one step (and,
when the stream has a chi axis, one chi, wrapping onto the next step).

Shim 2 patches ``torch.rand`` only at ``utils/pdb_dataset.py:1641`` inside
``_masked_sort_for_decoding_order``. Other ``torch.rand`` uses are untouched. The
injected stream is the concatenation of tier-0, tier-1, and tier-2 uniforms, each
in ascending row index, matching the three ``torch.rand`` calls in that function.

Shim 3 replaces ``nn.Dropout.forward`` with ``x * mask / (1 - p)`` while the module
is in train mode and ``p > 0``. Masks are keyed by ``(module path, call index)``.
``_VDropout`` is never patched; dumps assert every ``_VDropout.training is False``.

Shim 4 (f64 dumps) redirects hardcoded ``.float()`` calls listed in ``SHIM_SITES``
to the active dtype, sets the torch default dtype to float64 for the block (so
``torch.empty`` / ``torch.zeros`` / ``torch.linspace`` created during the forward
are not float32), rebuilds ``RBF_Encoding`` centres (``utils/model.py:1296``,
linspace with no dtype) in the distance dtype, and widens the loaded ideal-residue
coordinates (``utils/pdb_dataset.py:885`` moves them with ``.to(device)`` and no
dtype). It also redirects the ``dtype=torch.float`` literals listed in
``_FLOAT32_LITERAL_SITES``: ``run_inference.py`` pins float32 by literal rather
than by ``.float()`` at several featurization sites, which ``_shim_float``
cannot see. Under float32 the dtype context is not entered: ``.float()``, the
tensor factories, and the default dtype are all the originals.

Inspected and left unpatched, because they feed integer counts rather than a
float64 matmul: ``utils/model.py:527``, ``:528``, ``:782``, ``:786``
(``(index == aa).float()`` into ``scatter``). ``utils/model.py:1304`` adds Python
``1e-4`` onto the RBF tensor, so it follows the tensor dtype without a cast.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

SHIM_SITES: tuple[str, ...] = (
  "utils/model.py:627",
  "utils/model.py:676",
  "utils/model.py:677",
  "utils/model.py:859",
  "utils/model.py:899",
  "utils/pdb_dataset.py:1641",
  "nn.Dropout.forward",
  "utils/model.py:314",
  "utils/model.py:385",
  "utils/model.py:679",
  "utils/model.py:682",
  "utils/model.py:901",
  "utils/model.py:1106",
  "utils/model.py:1188",
  "utils/model.py:1296",
  "utils/pdb_dataset.py:525",
  "utils/pdb_dataset.py:598",
  "utils/pdb_dataset.py:603",
  "utils/pdb_dataset.py:928",
  "utils/pdb_dataset.py:944",
  "utils/pdb_dataset.py:1022",
  "utils/pdb_dataset.py:885",
  "run_inference.py:86",
  "run_inference.py:198",
  "run_inference.py:268",
  "run_inference.py:318",
  "utils/ligand_featurization.py:59",
  "utils/ligand_featurization.py:78",
  "utils/ligand_featurization.py:87",
  "torch.set_default_dtype",
)

# (filename suffix, lineno) for Tensor.float redirects. Draw sites are separate.
_FLOAT_SITES: frozenset[tuple[str, int]] = frozenset(
  {
    ("model.py", 314),
    ("model.py", 385),
    ("model.py", 679),
    ("model.py", 682),
    ("model.py", 901),
    ("model.py", 1106),
    ("model.py", 1188),
    ("pdb_dataset.py", 525),
    ("pdb_dataset.py", 598),
    ("pdb_dataset.py", 603),
    ("pdb_dataset.py", 928),
    ("pdb_dataset.py", 944),
    ("pdb_dataset.py", 1022),
    ("ligand_featurization.py", 59),
    ("ligand_featurization.py", 78),
    ("ligand_featurization.py", 87),
    ("run_inference.py", 86),
  },
)

# (filename suffix, lineno) where upstream pins float32 with a ``dtype=torch.float``
# literal instead of a ``.float()`` call, so the Tensor.float redirect cannot see it.
# Each is on the featurization path reached by ``ProteinComplexData.output_batch_data``.
_FLOAT32_LITERAL_SITES: frozenset[tuple[str, int]] = frozenset(
  {
    ("run_inference.py", 198),
    ("run_inference.py", 268),
    ("run_inference.py", 318),
  },
)
LITERAL_FACTORIES: tuple[str, ...] = ("full", "zeros", "ones", "empty", "tensor")
_DRAW_LINES: frozenset[int] = frozenset({627, 676, 677, 859, 899})
_ORDER_LINE = 1641

_ORIGINAL_SAMPLE = torch.distributions.Categorical.sample
_ORIGINAL_RAND = torch.rand
_ORIGINAL_DROPOUT = torch.nn.Dropout.forward
_ORIGINAL_FLOAT = torch.Tensor.float


class DrawCursor:
  """Step / chi cursor for one ``injected_uniform_draws`` block."""

  def __init__(self, uniforms: torch.Tensor) -> None:
    self.uniforms = uniforms
    self.step = 0
    self.chi = 0

  @property
  def consumed_steps(self) -> int:
    return self.step


class _OrderCursor:
  def __init__(self, uniforms: torch.Tensor) -> None:
    self.uniforms = uniforms
    self.offset = 0


class _DropoutCursor:
  def __init__(
    self,
    masks: dict[str, list[torch.Tensor]],
    rng: np.random.Generator | None,
  ) -> None:
    self.masks = masks
    self.rng = rng
    self.calls: dict[str, int] = {}
    self.recorded: dict[str, list[np.ndarray]] = {}


_DRAWS: DrawCursor | None = None
_ORDER: _OrderCursor | None = None
_DROPOUT: _DropoutCursor | None = None
_ACTIVE_DTYPE: torch.dtype | None = None


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
    msg = f"uniform sample axis {column.numel()} does not match categorical rows {n_rows}"
    raise RuntimeError(msg)
  return column.reshape(n_rows)


def _caller() -> inspect.FrameInfo:
  return inspect.stack()[2]


def _site(frame: inspect.FrameInfo) -> tuple[str, int]:
  return (Path(frame.filename).name, frame.lineno)


def _shim_sample(
  self: torch.distributions.Categorical,
  sample_shape: torch.Size | None = None,
) -> torch.Tensor:
  cursor = _DRAWS
  frame = _caller()
  shape = torch.Size() if sample_shape is None else sample_shape
  if cursor is None or Path(frame.filename).name != "model.py" or frame.lineno not in _DRAW_LINES:
    return _ORIGINAL_SAMPLE(self, shape)
  if shape not in (torch.Size(), torch.Size([])):
    msg = "LASEr inverse-CDF shim implements a scalar sample_shape only"
    raise RuntimeError(msg)
  probs = self.probs
  if probs.ndim == 1:
    unit = _take_uniform(cursor, 1)
    return inverse_cdf_index(probs, unit.reshape(()))
  n_rows = int(probs[..., 0].numel())
  flat = probs.reshape(n_rows, probs.shape[-1])
  unit = _take_uniform(cursor, n_rows)
  return inverse_cdf_index(flat, unit).reshape(*probs.shape[:-1])


def _shim_rand(*args: object, **kwargs: object) -> torch.Tensor:
  cursor = _ORDER
  frame = inspect.stack()[1]
  if cursor is None or _site(frame) != ("pdb_dataset.py", _ORDER_LINE):
    return _ORIGINAL_RAND(*args, **kwargs)
  size = args[0]
  if not isinstance(size, tuple) or len(size) != 1:
    msg = "decoding-order shim expects torch.rand((n,), ...)"
    raise RuntimeError(msg)
  n = int(size[0])  # type: ignore[arg-type]
  end = cursor.offset + n
  if end > cursor.uniforms.numel():
    msg = f"decoding-order uniform stream exhausted at {cursor.offset} (need {n} more)"
    raise RuntimeError(msg)
  drawn = cursor.uniforms[cursor.offset : end]
  cursor.offset = end
  device = kwargs.get("device")
  if device is not None:
    drawn = drawn.to(device=device)  # type: ignore[arg-type]
  return drawn


def _shim_dropout(self: torch.nn.Dropout, x: torch.Tensor) -> torch.Tensor:
  cursor = _DROPOUT
  if cursor is None or (not self.training) or self.p == 0:
    return _ORIGINAL_DROPOUT(self, x)
  path = getattr(self, "_laser_oracle_path", "")
  if not path:
    msg = "dropout shim saw an nn.Dropout without a module path"
    raise RuntimeError(msg)
  queued = cursor.masks.get(path, [])
  call = cursor.calls.get(path, 0)
  if call < len(queued):
    mask = queued[call].to(device=x.device, dtype=x.dtype)
  elif cursor.rng is not None:
    keep = cursor.rng.random(tuple(x.shape)) >= float(self.p)
    mask = torch.as_tensor(keep, device=x.device, dtype=x.dtype)
    cursor.recorded.setdefault(path, []).append(keep.astype(np.float64))
  else:
    msg = f"no injected dropout mask for {path} call {call}"
    raise RuntimeError(msg)
  cursor.calls[path] = call + 1
  return x * mask / (1.0 - self.p)


def _shim_float(self: torch.Tensor) -> torch.Tensor:
  dtype = _ACTIVE_DTYPE
  if dtype is None:
    return _ORIGINAL_FLOAT(self)
  frame = inspect.stack()[1]
  if _site(frame) not in _FLOAT_SITES:
    return _ORIGINAL_FLOAT(self)
  return self.to(dtype=dtype)


@contextmanager
def injected_uniform_draws(uniforms: np.ndarray | torch.Tensor) -> Iterator[DrawCursor]:
  """Patch in-scope ``Categorical.sample`` calls for the block; restore them on exit.

  ``uniforms`` is float64 with shape ``(sample, step)`` or ``(sample, step, chi)``.
  """
  global _DRAWS  # noqa: PLW0603
  tensor = torch.as_tensor(np.asarray(uniforms, dtype=np.float64), dtype=torch.float64)
  if tensor.ndim not in (2, 3):
    msg = "uniforms must have shape (sample, step) or (sample, step, chi)"
    raise ValueError(msg)
  if _DRAWS is not None:
    msg = "injected_uniform_draws cannot nest"
    raise RuntimeError(msg)
  cursor = DrawCursor(tensor)
  previous = torch.distributions.Categorical.sample
  _DRAWS = cursor
  torch.distributions.Categorical.sample = _shim_sample  # type: ignore[method-assign]
  try:
    yield cursor
  finally:
    torch.distributions.Categorical.sample = previous  # type: ignore[method-assign]
    _DRAWS = None


@contextmanager
def injected_decoding_order_uniforms(uniforms: np.ndarray | torch.Tensor) -> Iterator[_OrderCursor]:
  """Replace ``torch.rand`` at ``pdb_dataset.py:1641`` with ``uniforms`` (1d, float64)."""
  global _ORDER  # noqa: PLW0603
  tensor = torch.as_tensor(np.asarray(uniforms, dtype=np.float64).reshape(-1), dtype=torch.float64)
  if _ORDER is not None:
    msg = "injected_decoding_order_uniforms cannot nest"
    raise RuntimeError(msg)
  cursor = _OrderCursor(tensor)
  previous = torch.rand
  _ORDER = cursor
  torch.rand = _shim_rand  # type: ignore[method-assign]
  try:
    yield cursor
  finally:
    torch.rand = previous  # type: ignore[method-assign]
    _ORDER = None


@contextmanager
def injected_scalar_dropout(
  model: torch.nn.Module,
  masks: dict[str, list[np.ndarray] | list[torch.Tensor]] | None = None,
  *,
  rng: np.random.Generator | None = None,
) -> Iterator[_DropoutCursor]:
  """Patch ``nn.Dropout.forward`` and stamp each module with its ``named_modules`` path.

  Only train-mode modules with ``p > 0`` consume a mask. Pre-built ``masks`` win.
  If a call has no pre-built mask and ``rng`` is set, a keep-mask is drawn from
  ``rng.random(shape) >= p`` and appended to ``cursor.recorded``. ``_VDropout``
  is not patched.
  """
  global _DROPOUT  # noqa: PLW0603
  if _DROPOUT is not None:
    msg = "injected_scalar_dropout cannot nest"
    raise RuntimeError(msg)
  stamped: list[torch.nn.Dropout] = []
  packed: dict[str, list[torch.Tensor]] = {}
  supplied = {} if masks is None else masks
  for path, module in model.named_modules():
    if isinstance(module, torch.nn.Dropout):
      module._laser_oracle_path = path  # type: ignore[attr-defined]  # noqa: SLF001
      stamped.append(module)
      rows = supplied.get(path, [])
      packed[path] = [torch.as_tensor(np.asarray(row), dtype=torch.float64) for row in rows]
  cursor = _DropoutCursor(packed, rng)
  previous = torch.nn.Dropout.forward
  _DROPOUT = cursor
  torch.nn.Dropout.forward = _shim_dropout  # type: ignore[method-assign]
  try:
    yield cursor
  finally:
    torch.nn.Dropout.forward = previous  # type: ignore[method-assign]
    _DROPOUT = None
    for module in stamped:
      if hasattr(module, "_laser_oracle_path"):
        delattr(module, "_laser_oracle_path")


def assert_vector_dropout_eval(model: torch.nn.Module) -> None:
  """Raise if any ``_VDropout`` submodule is in train mode."""
  for path, module in model.named_modules():
    if module.__class__.__name__ == "_VDropout" and module.training:
      msg = f"_VDropout {path} is in train mode; vector dropout must stay off"
      raise RuntimeError(msg)


@contextmanager
def rbf_centres_follow_distance_dtype(rbf_cls: object) -> Iterator[None]:
  """Rebuild ``RBF_Encoding`` centres in ``distances.dtype``.

  Upstream ``utils/model.py:1296`` registers ``torch.linspace`` with no dtype, so a
  later ``model.double()`` promotes float32-rounded centres. Under float32 distances
  the rebuilt centres match the buffer.
  """
  original = rbf_cls.forward  # type: ignore[attr-defined]

  def forward(self: torch.nn.Module, distances: torch.Tensor) -> torch.Tensor:
    centres = torch.linspace(
      self.bin_min,
      self.bin_max,
      self.num_bins,
      device=distances.device,
      dtype=distances.dtype,
    ).view(1, -1)
    expanded = torch.unsqueeze(distances, -1)
    sigma = (self.bin_max - self.bin_min) / self.num_bins
    return torch.exp(-(((expanded - centres) / sigma) ** 2)) + 1e-4

  rbf_cls.forward = forward  # type: ignore[attr-defined]
  try:
    yield
  finally:
    rbf_cls.forward = original  # type: ignore[attr-defined]


# Matched as a suffix: upstream imports as LASErMPNN.utils.pdb_dataset when the pinned
# checkout's parent is the import root, but a bare `utils.pdb_dataset` is also valid.
_IDEAL_COORD_MODULES = ("utils.pdb_dataset", "utils.pdb_dataset_ligandmpnn")
_IDEAL_COORD_ATTR = "ideal_prot_aa_coords"


def ideal_coord_modules() -> list[object]:
  """Imported upstream modules holding the ideal-coordinate constant."""
  found: list[object] = []
  for name, module in list(sys.modules.items()):
    if not any(name == suffix or name.endswith(f".{suffix}") for suffix in _IDEAL_COORD_MODULES):
      continue
    if isinstance(getattr(module, _IDEAL_COORD_ATTR, None), torch.Tensor):
      found.append(module)
  return found


def _literal_factory_shim(name: str) -> object:
  """Wrap ``torch.<name>`` so a float32 dtype literal at a listed site follows the block dtype."""
  original = getattr(torch, name)

  def shim(*args: object, **kwargs: object) -> torch.Tensor:
    dtype = _ACTIVE_DTYPE
    # Cheap guards first: these factories are called constantly, and resolving the
    # caller frame on every call would dominate the dump.
    if dtype is None or kwargs.get("dtype") is not torch.float32:
      return original(*args, **kwargs)
    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    if caller is None:
      return original(*args, **kwargs)
    site = (Path(caller.f_code.co_filename).name, caller.f_lineno)
    if site not in _FLOAT32_LITERAL_SITES:
      return original(*args, **kwargs)
    return original(*args, **{**kwargs, "dtype": dtype})

  return shim


@contextmanager
def float32_literals_follow_active_dtype() -> Iterator[None]:
  """Redirect ``dtype=torch.float`` literals at ``_FLOAT32_LITERAL_SITES`` to the block dtype.

  ``run_inference.py`` builds several featurization tensors with an explicit
  ``dtype=torch.float`` rather than ``.float()``, so ``_shim_float`` never sees
  them and an f64 forward dies with ``Index put requires the source and
  destination dtypes match`` (``run_inference.py:301``). Only the listed sites
  are redirected, and only when the call itself passes ``dtype=torch.float32``.
  """
  originals = {name: getattr(torch, name) for name in LITERAL_FACTORIES}
  for name in LITERAL_FACTORIES:
    setattr(torch, name, _literal_factory_shim(name))
  try:
    yield
  finally:
    for name, original in originals.items():
      setattr(torch, name, original)


@contextmanager
def ideal_coords_follow_backbone_dtype() -> Iterator[None]:
  """Widen the loaded ideal-residue coordinates to float64 for an f64 block.

  ``utils/pdb_dataset.py:885`` builds the ideal alanine N/CA/C frame with
  ``.to(bb_coords.device)`` and no dtype, so under f64 backbone coordinates
  ``compute_alignment_matrices`` (``utils/build_rotamers.py:516``) raises
  ``expected scalar type Float but found Double``.

  This rebinds the module-level constant rather than reimplementing
  ``idealize_backbone_coords``, so upstream's arithmetic stays verbatim.
  ``ideal_prot_aa_coords`` is data loaded from ``files/ideal_aa_coords_prot.pt``,
  not a computed grid, so float32 -> float64 is an exact widening: the f64 dump
  sees the same values in wider arithmetic. (Contrast the RBF centres, which are
  recomputed by ``rbf_centres_follow_distance_dtype`` and do change.)
  """
  patched: list[tuple[object, torch.Tensor]] = []
  for module in ideal_coord_modules():
    original = getattr(module, _IDEAL_COORD_ATTR)
    patched.append((module, original))
    setattr(module, _IDEAL_COORD_ATTR, original.double())
  if not patched:
    msg = (
      f"ideal_coords_follow_backbone_dtype patched nothing: none of {_IDEAL_COORD_MODULES} "
      f"is imported with {_IDEAL_COORD_ATTR}. Import upstream LASErMPNN before entering the f64 block."
    )
    raise RuntimeError(msg)
  try:
    yield
  finally:
    for module, original in patched:
      setattr(module, _IDEAL_COORD_ATTR, original)


@contextmanager
def f64_runtime(rbf_cls: object) -> Iterator[None]:
  """Run one f64 dump block: default dtype, ``.float()`` redirects, RBF centres, ideal coords."""
  global _ACTIVE_DTYPE  # noqa: PLW0603
  if _ACTIVE_DTYPE is not None:
    msg = "f64_runtime cannot nest"
    raise RuntimeError(msg)
  previous_default = torch.get_default_dtype()
  previous_float = torch.Tensor.float
  _ACTIVE_DTYPE = torch.float64
  torch.set_default_dtype(torch.float64)
  torch.Tensor.float = _shim_float  # type: ignore[method-assign]
  try:
    with (
      rbf_centres_follow_distance_dtype(rbf_cls),
      ideal_coords_follow_backbone_dtype(),
      float32_literals_follow_active_dtype(),
    ):
      yield
  finally:
    torch.Tensor.float = previous_float  # type: ignore[method-assign]
    torch.set_default_dtype(previous_default)
    _ACTIVE_DTYPE = None


def assert_inverse_cdf_edges() -> None:
  """Zero-probability tails are never drawn; ``u`` near 1 returns the last positive index."""
  probs64 = torch.tensor([0.0, 0.25, 0.75, 0.0], dtype=torch.float64)
  almost64 = torch.tensor(1.0 - (2.0**-53), dtype=torch.float64)
  last64 = int(inverse_cdf_index(probs64, almost64))
  if last64 != 2:
    msg = f"u=1-2**-53 returned {last64}, expected last positive index 2"
    raise RuntimeError(msg)
  probs32 = probs64.to(dtype=torch.float32)
  almost32 = torch.tensor(1.0 - (2.0**-24), dtype=torch.float32)
  last32 = int(inverse_cdf_index(probs32, almost32))
  if last32 != 2:
    msg = f"u=1-2**-24 returned {last32}, expected last positive index 2"
    raise RuntimeError(msg)
  for unit in (0.0, 0.1, 0.5, 0.9, float(almost64)):
    drawn = int(inverse_cdf_index(probs64, torch.tensor(unit, dtype=torch.float64)))
    if drawn == 3 or probs64[drawn] == 0:
      msg = f"inverse-CDF drew a zero-probability index {drawn} at u={unit}"
      raise RuntimeError(msg)
