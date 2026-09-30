"""StageBundleAdapter wraps StageSet with StageBundle interface.

Presents StageSet's encoder_sink and decoder_sink tuples through StageBundle's
single-callable property interface, chaining multiple sinks without collapsing
side effects.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx


class StageBundleAdapter(eqx.Module):
  """Adapts StageSet to present StageBundle interface.

  Maps StageSet's tuple-based sinks (encoder_sink, decoder_sink) to
  StageBundle's single-callable properties by chaining sinks in order.

  Attributes
  ----------
  _stage_set : StageSet
      The wrapped StageSet instance.

  Notes
  -----
  Unlike StageBundle which enforces Optional[Callable] at class definition,
  StageBundleAdapter dynamically chains tuples of callables into single
  properties. Properties return None if the tuple is empty, matching StageBundle
  semantics for inactive stages.

  Topology queries (active_stages, has_stage) inspect wrapped properties,
  listing only non-None callables at adapter level.
  """

  _stage_set: Any  # StageSet (avoid TYPE_CHECKING-only import in runtime annotation)

  def _chain(self, sinks: tuple[Callable[..., Any], ...]) -> Callable[..., Any] | None:
    """Chain tuple of sinks into single callable, or return None if empty.

    Parameters
    ----------
    sinks : tuple[Callable[..., Any], ...]
        Tuple of callables to chain. Order is preserved.

    Returns
    -------
    Callable[..., Any] | None
        Single callable that invokes all sinks in order, or None if tuple empty.
    """
    if not sinks:
      return None

    def chained(
      *args: Any,
      **kwargs: Any,
    ) -> None:
      """Invoke all sinks in order with same arguments."""
      for fn in sinks:
        fn(*args, **kwargs)

    return chained

  @property
  def encoder_sink(self) -> Callable[..., Any] | None:
    """Chained encoder_sink tuple, or None if empty.

    Returns
    -------
    Callable[..., Any] | None
        Single callable invoking all encoder sinks in order, or None.
    """
    return self._chain(self._stage_set.encoder_sink)

  @property
  def decoder_sink(self) -> Callable[..., Any] | None:
    """Chained decoder_sink tuple, or None if empty.

    Returns
    -------
    Callable[..., Any] | None
        Single callable invoking all decoder sinks in order, or None.
    """
    return self._chain(self._stage_set.decoder_sink)

  @property
  def encode_fn(self) -> Callable[..., Any] | None:
    """encode_fn property stub (not defined on StageSet).

    Returns None for now; can be extended if StageSet adds encode_fn field.

    Returns
    -------
    Callable[..., Any] | None
        Always None (not implemented on StageSet).
    """
    return None

  @property
  def decode_fn(self) -> Callable[..., Any] | None:
    """decode_fn property stub (not defined on StageSet).

    Returns None for now; can be extended if StageSet adds decode_fn field.

    Returns
    -------
    Callable[..., Any] | None
        Always None (not implemented on StageSet).
    """
    return None

  def active_stages(self) -> list[str]:
    """Return names of non-None callable properties.

    Topology introspection: lists which sinks are actually active.

    Returns
    -------
    list[str]
        Sorted list of property names with non-None callables.
    """
    stages = []
    if self.encoder_sink is not None:
      stages.append("encoder_sink")
    if self.decoder_sink is not None:
      stages.append("decoder_sink")
    if self.encode_fn is not None:
      stages.append("encode_fn")
    if self.decode_fn is not None:
      stages.append("decode_fn")
    return sorted(stages)

  def has_stage(self, name: str) -> bool:
    """Check if named property is non-None callable.

    Parameters
    ----------
    name : str
        Property name to check (e.g., 'encoder_sink', 'decoder_sink').

    Returns
    -------
    bool
        True if property exists and is non-None callable; False otherwise.
    """
    prop = getattr(self, name, None)
    return prop is not None and callable(prop)


__all__ = ["StageBundleAdapter"]
