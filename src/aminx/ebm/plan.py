"""xtrax tiling composition wiring for the ProteinEBM energy/score path (backlog node **E4**).

This module is the "encode-once decision" half of E4's composition wiring:
given a cardinality for one of the axes ``ProteinEBMModel.energy``/``.score``
get dispatched over (decoys, mutants, unfolded-ensemble members, multistate
states), build the same ``xtrax.tiling.plan.AxisSpec`` ->
``xtrax.tiling.plan.BatchPlanner`` -> ``AxisDecision`` pipeline
``aminx.host.plan``/``aminx.tiling.axes`` use for the inverse-folding sampling
path, and dispatch an arbitrary per-element ``body`` over that axis via the
selected strategy. ``aminx.ebm.dispatch`` builds the three disjoint,
callable entry points (decoy ranking / ΔΔG / conformational biasing) on top
of this module's ``plan_axis``/``dispatch_axis``; the fusion primitives below
are the composition seam E7's difference-Fuse needs.

**Why a parallel registry, not a reuse of ``aminx.tiling.axes``/
``aminx.host.plan``.** Design spec §3.2/§10 (BLOCKER-2) and the EPIC DAG's E4
row: ``ProteinEBMModel.energy()``/``.score()`` take coords/aatype/``t``/mask
-- no neighbor graph, no sequence one-hot, no ``InferenceBundle``/
``EncoderOutput`` shape. Forcing this onto ``aminx.host.plan``'s
``InferencePlan``/``StageSet`` machinery would mean either widening those
logit-typed slots to ``Any`` (losing their contract value) or bolting on a
special case the existing tests/callers never anticipated. The *pattern* that
machinery uses -- ``AxisSpec`` + ``BatchPlanner`` + a strategy-name dispatch
over ``Vmap``/``SafeMap``/``Scan`` (mirroring
``host/kernel_dispatch.py::_dispatch_axis``) -- is genuinely generic and is
exactly what this module reuses; the *instances* (axis registry, dispatch
entry points) are new and disjoint, keeping the EBM path isolated from the
live, tested inverse-folding dispatch it must not risk (per the DAG's E0-E3.6/
E8/E3.5 precedent and this node's explicit hard requirement).

**Scope.** E4 is the wiring contract only -- wiring three sibling application
functions onto ``ProteinEBMModel`` via a single/small number of tiled axes.
Noise-time sweeps, per-mutant WT-relative corrections with real unfolded-state
sampling, and the accuracy-gated decoy/ΔΔG/biasing logic are E5/E6/E7's job,
not this node's.

**Why ``EnergyFusionFn`` is a local Protocol, not an import of
``aminx.types.stages.FuseFn``.** ``pyproject.toml``'s
``[tool.ruff.lint.flake8-tidy-imports.banned-api]`` bans importing
``aminx.types.stages`` repo-wide (ADR 260605_potts-parallel-not-stageset:
"PottsModel is parallel, not a stageset consumer"; the one exemption,
``potts/designer.py``, is a deliberate, reviewed boundary crossing). ProteinEBM
is the same story -- a new parallel model family, not a stageset consumer --
so ``aminx.ebm`` earns no exemption either, and shouldn't need one:
``EnergyFusionFn`` below reimplements the *shape* of the Tier-1 generic
``FuseFn[PerItem, Combined]`` protocol (same ``(per_item, bias=None) ->
Combined`` signature) as a local, zero-dependency definition, which is both
lint-clean and keeps ``aminx.ebm`` fully import-isolated from
``aminx.types.stages`` -- strictly stronger than the design spec's "don't
reuse ``DecodingFusionFn``" requirement.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from xtrax.tiling import AxisDecision, AxisSpec, BatchPlanner, SafeMap, Scan, Vmap
from xtrax.transforms import safe_scan

from aminx.ebm.contracts import Energy
from aminx.utils.safe_map import safe_map

if TYPE_CHECKING:
  from collections.abc import Callable


class EBMAxisNames:
  """Named access to the ``aminx.ebm`` dispatch axes.

  Mirrors ``aminx.host.plan.AxisNames``'s named-string-constant pattern, but
  is a **separate registry** -- see the module docstring's isolation
  rationale. Renaming/extending one never risks the other.
  """

  N_DECOYS = "n_decoys"
  N_MUTANTS = "n_mutants"
  N_ENSEMBLE = "n_ensemble"
  N_STATES = "n_states"


# Module-level AxisSpec templates. ``cardinality`` is a placeholder (always
# overridden per call site via dataclasses.replace in plan_axis) -- mirrors
# aminx.tiling.axes's registry pattern (module-level AxisSpec constants,
# replaced at call time with the real, possibly-traced-shape-derived count).
_N_DECOYS = AxisSpec(name=EBMAxisNames.N_DECOYS, cardinality=1, default_batch_size=8)
_N_MUTANTS = AxisSpec(name=EBMAxisNames.N_MUTANTS, cardinality=1, default_batch_size=8)
_N_ENSEMBLE = AxisSpec(name=EBMAxisNames.N_ENSEMBLE, cardinality=1, default_batch_size=8)
_N_STATES = AxisSpec(name=EBMAxisNames.N_STATES, cardinality=1, default_batch_size=4)

_AXIS_TEMPLATES: dict[str, AxisSpec] = {
  EBMAxisNames.N_DECOYS: _N_DECOYS,
  EBMAxisNames.N_MUTANTS: _N_MUTANTS,
  EBMAxisNames.N_ENSEMBLE: _N_ENSEMBLE,
  EBMAxisNames.N_STATES: _N_STATES,
}


def plan_axis(
  name: str,
  cardinality: int,
  *,
  default_batch_size: int | None = None,
) -> AxisDecision:
  """Build a single-axis :class:`~xtrax.tiling.AxisDecision` for one EBM axis.

  Wraps ``AxisSpec`` + ``BatchPlanner`` (the same primitives
  ``aminx.host.plan`` uses) for exactly one axis at a time -- E4's dispatch
  entry points each tile a single axis (decoys, mutants, ensemble members, or
  states), not the nested multi-axis product ``host/kernel_dispatch.py``
  builds for the sampling path.

  Args:
      name: One of ``EBMAxisNames``'s axis-name constants.
      cardinality: The real element count for this call (e.g. number of
          decoys actually being scored).
      default_batch_size: Override the axis template's default batch-size
          threshold (BatchPlanner's Vmap-vs-SafeMap cutoff). ``None`` keeps
          the template's default.

  Returns:
      The single :class:`~xtrax.tiling.AxisDecision` BatchPlanner produced
      for this axis (``.strategy`` is a ``Vmap``/``SafeMap`` instance per
      BatchPlanner's cardinality-vs-batch-size rule).

  Raises:
      KeyError: If ``name`` is not a registered EBM axis.

  """
  if name not in _AXIS_TEMPLATES:
    msg = f"aminx.ebm.plan: unknown axis {name!r}; expected one of {sorted(_AXIS_TEMPLATES)}"
    raise KeyError(msg)
  template = _AXIS_TEMPLATES[name]
  spec = dataclasses.replace(
    template,
    cardinality=cardinality,
    default_batch_size=default_batch_size
    if default_batch_size is not None
    else template.default_batch_size,
  )
  plan = BatchPlanner().plan([spec])
  return plan.decisions[0]


def dispatch_axis[T, U](
  strategy: object,
  body: Callable[[T], U],
  xs: T,
) -> U:
  """Dispatch iteration over an axis using a BatchPlanner-selected strategy.

  Analogous to ``aminx.host.kernel_dispatch._dispatch_axis``, deliberately
  reimplemented here (not imported) to keep ``aminx.ebm`` free of any
  dependency on ``aminx.host`` -- the reusable asset is the strategy-dispatch
  *pattern* (dispatch by ``isinstance`` against the sealed xtrax
  ``AxisStrategy`` union, not a logit-typed helper), per the module
  docstring's isolation rationale. Like ``_dispatch_axis``, this calls
  ``jax.vmap``/``safe_map``/``safe_scan`` directly rather than going through
  ``xtrax.tiling.make_axis_dispatch``'s iterator objects -- ``make_axis_
  dispatch`` returns a bare ``object`` in xtrax's own type stubs (the concrete
  iterator classes are resolved by a lazy import at call time), which
  ``ty`` cannot see through; calling the strategies directly keeps this
  function's own signature meaningfully typed.

  ``strategy`` is deliberately typed as plain ``object``, not the narrower
  ``Vmap | SafeMap | Scan``: ``AxisDecision.strategy``'s declared type is
  xtrax's full sealed ``AxisStrategy`` union (which also includes
  ``DedupGather``/``Bucket`` -- neither importable from the public
  ``xtrax.tiling`` surface this file is restricted to), so a narrower
  parameter type would reject a value ``plan_axis`` can legitimately hand
  back. The ``isinstance`` checks below are the real (runtime) type
  narrowing; anything outside E4's scope falls through to the ``TypeError``.

  Args:
      strategy: An xtrax ``Vmap``/``SafeMap``/``Scan`` instance, as returned
          by ``plan_axis(...).strategy``.
      body: Per-element function, ``body(x) -> y``. For ``Scan`` this is
          wrapped as a carry-passthrough transition (no real carry threading
          is needed for E4's deterministic energy/score dispatch).
      xs: Input pytree; leading axis is the dispatched axis.

  Returns:
      Stacked per-element results, same leading axis size as ``xs``.

  Raises:
      TypeError: If ``strategy`` is a ``DedupGather``/``Bucket`` (or any
          other strategy outside this node's scope) -- ``DedupGather`` is an
          E6 dedup-eligible-mutants extension and ``Bucket`` is the E4.5
          host-side residue-length gate; neither belongs to this node's
          per-element dispatch contract.

  """
  if isinstance(strategy, Vmap):
    return jax.vmap(body)(xs)
  if isinstance(strategy, SafeMap):
    return safe_map(body, xs, batch_size=strategy.batch_size)
  if isinstance(strategy, Scan):

    def _scan_body(carry: object, x: T) -> tuple[object, U]:
      return carry, body(x)

    init = strategy.init if strategy.init is not None else jnp.array(0)
    _, ys = safe_scan(_scan_body, init, xs)
    return ys
  msg = (
    f"aminx.ebm.plan.dispatch_axis: unsupported strategy {type(strategy).__name__} "
    "for E4's wiring scope (Vmap/SafeMap/Scan only)."
  )
  raise TypeError(msg)


_PerItem = TypeVar("_PerItem")
_Combined = TypeVar("_Combined")


@runtime_checkable
class _FuseProtocol(Protocol[_PerItem, _Combined]):
  """Local structural analog of ``aminx.types.stages.FuseFn[PerItem, Combined]``.

  Same Tier-1-generic shape (``(per_item, bias=None) -> Combined``); defined
  here rather than imported so ``aminx.ebm`` has zero dependency on
  ``aminx.types.stages`` (see module docstring).
  """

  def __call__(self, per_item: _PerItem, bias: _PerItem | None = None) -> _Combined: ...


EnergyVector = Float[Array, "S"]
"""A 1-D array of per-element scalar energies along a dispatched axis (``S`` = axis size).

Shared shape alias for whichever axis is currently being fused (decoys,
mutants, unfolded-ensemble members, or multistate states) -- ``S`` is a
per-call size, not a fixed global constant.
"""

EnergyFusionFn = _FuseProtocol[EnergyVector, Energy]
"""Generic energy-fusion primitive: reduce ``S`` per-element energies to one scalar.

**Not** a reuse of ``aminx.types.stages.DecodingFusionFn`` (typed to the
logit-shaped ``DecodeOutput``) -- design spec §3.3/§10 BLOCKER-2. Structural
peer of the Tier-1 generic ``FuseFn[PerItem, Combined]`` protocol
(``aminx.types.stages``, the same mechanism
``LogitTransformFn``/``ARLogitTransformFn`` use for the logit path),
specialized to scalar energies instead of logit tensors -- but reimplemented
locally (see ``_FuseProtocol`` above) rather than imported, per the module
docstring's import-isolation rationale.
"""


class DifferenceFuse(eqx.Module):
  """Difference-Fuse: ``E(state_a) - E(state_b)`` over a state axis.

  The multistate/conformational-biasing composition seam (design spec §3.3
  row 3, §5): a fixed-sequence energy gap between two conformational states.
  Concrete ``EnergyFusionFn`` instance -- stateless (no learned parameters),
  an ``eqx.Module`` only so it PyTree-flattens cleanly alongside any future
  composition that stores it as a field (mirrors how the readout heads and
  other stateless-but-composable pieces in this subpackage are modules).
  """

  index_a: int = eqx.field(static=True, default=0)
  index_b: int = eqx.field(static=True, default=1)

  def __call__(
    self,
    per_item: EnergyVector,
    bias: Energy | None = None,
  ) -> Energy:
    """``per_item[index_a] - per_item[index_b]``, plus an optional additive bias."""
    result = per_item[self.index_a] - per_item[self.index_b]
    if bias is not None:
      result = result + bias
    return result


class MeanFuse(eqx.Module):
  """Mean-Fuse: arithmetic mean over an ensemble axis.

  Needed by E6's unfolded-ensemble average (design spec §3.3 row 2): the
  Monte-Carlo unfolded-state correction is the mean energy over ``p_UF``
  samples. Concrete ``EnergyFusionFn`` instance.
  """

  def __call__(
    self,
    per_item: EnergyVector,
    bias: Energy | None = None,
  ) -> Energy:
    """Mean of ``per_item``, plus an optional additive bias."""
    result = jnp.mean(per_item)
    if bias is not None:
      result = result + bias
    return result


difference_fuse = DifferenceFuse()
mean_fuse = MeanFuse()


__all__ = [
  "DifferenceFuse",
  "EBMAxisNames",
  "EnergyFusionFn",
  "EnergyVector",
  "MeanFuse",
  "difference_fuse",
  "dispatch_axis",
  "mean_fuse",
  "plan_axis",
]
