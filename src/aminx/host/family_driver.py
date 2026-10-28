"""FamilyDriver seam: protocol, batch metadata, and the driver registry.

No driver is registered here. PottsMPNN and LASErMPNN drivers land in later
tasks; an empty registry leaves every existing MPNN runner path unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, get_args

import jax
from equinox import Module as EqxModule

from aminx.registry import Registry

if TYPE_CHECKING:
  from xtrax.tiling import AxisSpec

  from aminx.types.protocols import ModelProtocol

# Canonical 21-letter alphabet stamped on driver sink roots (X is the gap token).
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

# ONE source of truth for which ``model_family`` names exist. Validation (RunSpecification,
# runner, prep, CLI) and dispatch all read these tables, so a name cannot be valid in one
# place and unknown in another.
# The declared family type. RunSpecification annotates model_family with this alias, so the
# type checker, the runtime guard and the stock list below all read the same Literal.
ModelFamily = Literal[
  "proteinmpnn",
  "ligandmpnn",
  "membrane",
  "pottsmpnn",
  "protonpottsmpnn",
  "lasermpnn",
  "caliby",
  "fampnn",
  "thermompnn",
]
# Driver-backed families need a registered FamilyDriver, imported on first use from the
# module named here. A missing driver is an error, never a fallback to the stock path.
DRIVER_BACKED_FAMILY_MODULES: dict[str, str] = {
  "pottsmpnn": "aminx.families.potts_mpnn",
  "protonpottsmpnn": "aminx.families.protonpotts_mpnn",
  "lasermpnn": "aminx.families.laser_mpnn",
  "caliby": "aminx.families.caliby",
  "fampnn": "aminx.families.fampnn",
  "thermompnn": "aminx.families.thermo_mpnn",
}
# Every declared family that is not driver-backed runs on the stock MPNN path.
STOCK_MODEL_FAMILIES: tuple[str, ...] = tuple(
  name for name in get_args(ModelFamily) if name not in DRIVER_BACKED_FAMILY_MODULES
)
# Optional-dependency gate per driver-backed family: (importable module, pip extra).
FAMILY_EXTRA_REQUIREMENTS: dict[str, tuple[str, str]] = {
  "lasermpnn": ("prody", "laser"),
}
MODEL_FAMILIES: tuple[str, ...] = (*STOCK_MODEL_FAMILIES, *DRIVER_BACKED_FAMILY_MODULES)
DRIVER_MODEL_FAMILIES: frozenset[str] = frozenset(DRIVER_BACKED_FAMILY_MODULES)


def validate_model_family(value: object) -> str:
  """Return ``value`` when it names a known family; raise ValueError listing the valid names."""
  if isinstance(value, str) and value in MODEL_FAMILIES:
    return value
  valid = ", ".join(repr(name) for name in MODEL_FAMILIES)
  msg = f"unknown model_family {value!r}; valid model families are: {valid}"
  raise ValueError(msg)


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
