"""Restricted unpickling for torch checkpoints that upstream model repositories ship.

A ``.pt`` or ``.ckpt`` file is a pickle. ``torch.load`` with the stock unpickler resolves any
global the pickle names, so a third-party checkpoint can run code through a ``__reduce__``
payload. Loading goes through an allow-list instead: tensor-rebuild helpers, plain containers,
and inert stand-ins for the omegaconf/hydra config objects that Lightning checkpoints carry.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

_TORCH_STORAGE_TYPES: tuple[str, ...] = (
  "BoolStorage",
  "ByteStorage",
  "CharStorage",
  "ShortStorage",
  "IntStorage",
  "LongStorage",
  "HalfStorage",
  "BFloat16Storage",
  "FloatStorage",
  "DoubleStorage",
)

# Globals a Lightning/upstream torch checkpoint pickle may reference. ``find_class`` receives the
# raw stream names, so the Python-2 spellings (``__builtin__``) are listed as written, alongside
# the Python-3 spellings. Data-rebuild helpers only.
TORCH_CHECKPOINT_ALLOWED_GLOBALS: frozenset[tuple[str, str]] = frozenset(
  {
    ("__builtin__", "dict"),
    ("__builtin__", "list"),
    ("__builtin__", "long"),
    ("builtins", "dict"),
    ("builtins", "int"),
    ("builtins", "list"),
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("typing", "Any"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch._utils", "_rebuild_parameter_with_state"),
  }
  | {("torch", name) for name in _TORCH_STORAGE_TYPES},
)

# omegaconf/hydra config objects are replaced by the inert ``_DummyCfg`` below.
TORCH_CHECKPOINT_STUB_PREFIXES: tuple[str, ...] = ("omegaconf", "hydra")


class _DummyCfg:
  """Inert stand-in for omegaconf/hydra config objects inside torch checkpoints."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
    """Initialize placeholder configuration container.

    Parameters
    ----------
    args : Any
      Positional arguments ignored.
    kwargs : Any
      Keyword arguments ignored.

    Returns
    -------
    None
      Initializes dummy instance.
    """
    del args, kwargs

  def __setstate__(self, state: Any) -> None:  # noqa: ANN401
    """Restore pickled state dictionary into instance attributes.

    Parameters
    ----------
    state : Any
      Pickled state object.

    Returns
    -------
    None
      Updates ``self.__dict__`` when ``state`` is a ``dict``.
    """
    if isinstance(state, dict):
      self.__dict__.update(state)


class RestrictedTorchUnpickler(pickle.Unpickler):
  """Unpickler that resolves only tensor-rebuild data globals and inert config stubs.

  Any other global (for example ``os.system``) is refused, so loading a third-party checkpoint
  cannot execute code through a pickle ``__reduce__`` payload.
  """

  def find_class(self, module: str, name: str) -> Any:  # noqa: ANN401
    """Resolve an allow-listed global, substituting ``_DummyCfg`` for config stubs.

    Parameters
    ----------
    module : str
      Module path string from the pickle stream.
    name : str
      Attribute name from the pickle stream.

    Returns
    -------
    Any
      Resolved allow-listed object or ``_DummyCfg``.

    Raises
    ------
    pickle.UnpicklingError
      If ``module.name`` is neither allow-listed nor an omegaconf/hydra stub.
    """
    if module.startswith(TORCH_CHECKPOINT_STUB_PREFIXES):
      return _DummyCfg
    if (module, name) in TORCH_CHECKPOINT_ALLOWED_GLOBALS:
      return super().find_class(module, name)
    msg = f"Refusing to unpickle global {module}.{name} from a torch checkpoint"
    raise pickle.UnpicklingError(msg)


def load_restricted_torch_checkpoint(path: Path) -> Any:  # noqa: ANN401
  """Load a torch ``.pt`` or ``.ckpt`` checkpoint through :class:`RestrictedTorchUnpickler`.

  Parameters
  ----------
  path : Path
    Checkpoint file.

  Returns
  -------
  Any
    The deserialized object (usually a state dict, or a Lightning dict holding one).

  Raises
  ------
  pickle.UnpicklingError
    If the pickle names a global outside the allow-list.
  """
  import torch  # noqa: PLC0415

  pmod = type(
    "pickle_mod",
    (),
    {
      "Unpickler": RestrictedTorchUnpickler,
      "load": lambda f, **_kw: RestrictedTorchUnpickler(f).load(),
    },
  )
  return torch.load(path, map_location="cpu", weights_only=False, pickle_module=pmod)
