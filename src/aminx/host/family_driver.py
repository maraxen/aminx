"""FamilyDriver seam: protocol, batch metadata, and the driver registry.

No driver is registered here. PottsMPNN and LASErMPNN drivers land in later
tasks; an empty registry leaves every existing MPNN runner path unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import jax
from equinox import Module as EqxModule

from aminx.registry import Registry

if TYPE_CHECKING:
  from xtrax.tiling import AxisSpec

  from aminx.types.protocols import ModelProtocol

# Canonical 21-letter alphabet stamped on driver sink roots (X is the gap token).
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

DRIVER_MODEL_FAMILIES: frozenset[str] = frozenset({"pottsmpnn", "lasermpnn"})


class FamilyBatch(NamedTuple):
  """One host batch of structures for a family driver.

  ``lengths`` is the per-structure ``L_total`` (unpadded residue count), aligned
  with ``input_indices``. Device arrays in ``arrays`` may be padded; the runner
  slices each structure back to ``lengths`` before staging.
  """

  input_indices: tuple[int, ...]
  arrays: Any
  skipped: tuple[tuple[int, str], ...]
  lengths: tuple[int, ...]


class SinkArraySpec(NamedTuple):
  """One named array in ``FamilyDriver.result_schema``."""

  dims: tuple[str, ...]
  dtype: str
  attrs: Mapping[str, str | int | float]


class FamilyStages(Protocol):
  """Jitted per-chunk body for one driver purpose.

  ``scalar_dropout`` is static. Vector-channel dropout stays off.
  ``dropout_key`` is ``fold_in(key, 0)`` for unconditional proofreading; for
  conditional proofreading the stage applies
  ``fold_in(fold_in(fold_in(key, focus_idx), dropout_idx), order_idx)``.
  """

  def __call__(
    self,
    batch: FamilyBatch,
    *,
    chunk_start: int,
    chunk_count: int,
    scalar_dropout: bool,
    dropout_key: jax.Array,
  ) -> Mapping[str, jax.Array]:
    """Return schema-keyed arrays for this batch and sample chunk."""
    ...


class FamilyDriver(Protocol):
  """Per-family owner of load, batching, axes, stages, and sink schema.

  ``load`` must not apply ``eqx.nn.inference_mode``. Fallback purposes call
  ``mpnn_core`` on that result and the MPNN prep path applies inference mode.
  """

  name: str
  options_type: type[Any]
  mpnn_fallback_purposes: frozenset[str]

  def handles(self, spec: Any, purpose: str) -> bool:  # noqa: ANN401
    """True when this driver owns ``purpose`` for ``spec``."""
    ...

  def load(self, spec: Any) -> EqxModule:  # noqa: ANN401
    """Load the family module. Does not enter inference mode."""
    ...

  def mpnn_core(self, model: EqxModule) -> ModelProtocol | None:
    """Embedded stock MPNN, or None when this family has no fallback core."""
    ...

  def batches(self, spec: Any) -> Iterator[FamilyBatch]:  # noqa: ANN401
    """Yield fixed-shape batches. ``input_indices`` strictly increase."""
    ...

  def axes(self, spec: Any, purpose: str, batch: FamilyBatch) -> list[AxisSpec]:  # noqa: ANN401
    """Named xtrax axes for this batch. Drivers do not vmap these themselves."""
    ...

  def stages(self, spec: Any, purpose: str, model: EqxModule) -> FamilyStages:  # noqa: ANN401
    """Per-purpose stage callable."""
    ...

  def result_schema(self, spec: Any, purpose: str) -> Mapping[str, SinkArraySpec]:  # noqa: ANN401
    """Arrays one chunk must return. Per-position dims are named ``L_total``."""
    ...


class FamilyDriverRegistry:
  """``Registry`` wrapper whose ``get`` is None for an unregistered family.

  ``Registry.get`` raises ``KeyError``. Dispatch needs a missing family to mean
  "stay on the MPNN path", so this wrapper is the object ``FAMILY_DRIVERS`` is.
  """

  def __init__(self, name: str) -> None:
    self._registry: Registry[FamilyDriver] = Registry(name)

  def register(self, key: str) -> Callable[[FamilyDriver], FamilyDriver]:
    """Register a driver under ``key`` (the ``model_family`` string)."""
    return self._registry.register(key)

  def get(self, key: str | None) -> FamilyDriver | None:
    """Return the driver for ``key``, or None when nothing is registered."""
    if not isinstance(key, str):
      return None
    try:
      return self._registry.get(key)
    except KeyError:
      return None

  def discard(self, key: str) -> None:
    """Drop one registration. Tests use this; production code does not."""
    self._registry._items.pop(key, None)  # noqa: SLF001 — Registry has no public unregister


FAMILY_DRIVERS = FamilyDriverRegistry("family_drivers")


def refuse_driver_family(spec: object, *, surface: str) -> None:
  """Raise when a driver family enters an MPNN-only lineage surface.

  Chunked grid, campaign, streaming, and multistate-PoE semantics are not
  implemented by family drivers. ProteinMPNN and LigandMPNN are unchanged.
  """
  family = getattr(spec, "model_family", None)
  if family in DRIVER_MODEL_FAMILIES:
    msg = (
      f"{surface} does not support model_family={family!r} in v1 "
      "(driver-family chunked-lineage semantics are not implemented)"
    )
    raise ValueError(msg)
