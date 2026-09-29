"""Host-side sampling orchestration (Phase 5 prep).

:data:`~aminx.registry.SAMPLERS` holds **sampler factories** (``model, …`` →
:class:`~aminx.protocols.SamplerFn`). The default
``\"make_sample_sequences\"`` entry registers when :mod:`aminx.sampling.sample`
is imported (see module tail there).

Unknown registry keys raise :class:`KeyError` with known keys listed in the message
from :meth:`SamplingDriver.build_sampler_fn`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from aminx.host._base_driver import BaseDriver
from aminx.registry import SAMPLERS

if TYPE_CHECKING:
  from aminx.run.specs import SamplingSpecification
  from aminx.types.protocols import SamplerFn


class SamplingDriver(BaseDriver["SamplingSpecification"]):
  """Resolve a registered sampler factory from a :class:`~aminx.run.specs.SamplingSpecification`."""

  def __init__(
    self,
    spec: SamplingSpecification,
    *,
    sampler_factory_key: str = "make_sample_sequences",
  ) -> None:
    super().__init__(spec)
    self.sampler_factory_key = sampler_factory_key

  def sampler_factory_keys(self) -> tuple[str, ...]:
    """Registered sampler factory keys after built-in modules self-register (imports ``sampling.sample``)."""
    importlib.import_module("aminx.sampling.sample")
    return tuple(SAMPLERS.keys())

  def build_sampler_fn(self, model: object, **factory_kwargs: object) -> SamplerFn:
    """Return ``SAMPLERS[self.sampler_factory_key](model, **factory_kwargs)``.

    Loads :mod:`aminx.sampling.sample` first so built-in factories are registered.
    """
    importlib.import_module("aminx.sampling.sample")
    try:
      factory = SAMPLERS.get(self.sampler_factory_key)
    except KeyError as err:
      keys = ", ".join(SAMPLERS.keys())
      msg = f"Unknown sampler factory key {self.sampler_factory_key!r}; known keys: {keys}"
      raise KeyError(msg) from err
    return factory(model, **factory_kwargs)
