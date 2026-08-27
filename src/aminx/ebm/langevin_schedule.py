"""Outer noise-schedule loop + net-new model-swap dispatcher for Langevin annealing (backlog node **E9**, OUTER half).

Wraps the already-committed, do-not-modify inner primitive
(:func:`aminx.ebm.langevin.run_langevin_equilibration`, a single-``t``,
single-model ``lax.while_loop`` equilibration -- see that module's own
docstring for its deliberate scope limit) in an **outer** ``xtrax``
``CarrySpec``+``Scan`` over a noise-level schedule, with a **net-new**
``jax.lax.cond``/``jax.lax.switch`` dispatcher that swaps between several
*pre-loaded* :class:`~aminx.ebm.model.ProteinEBMModel` checkpoints as ``t``
crosses per-checkpoint thresholds -- the reference's ``get_dynamics_model(t)``.

Ported/derived from ``~/repos/ProteinEBM/protein_ebm/scripts/run_dynamics.py``
(the full outer driver / ``main()``, not just the already-ported per-step
update lines) and this epic's design authority:
``.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md`` §1 Fork 7 +
risk register BLOCKER-1/MAJOR-4, and
``.praxia/docs/specs/260709_proteinebm-aminx-decomposition.md`` §3.3's
"Langevin annealing / folding simulation" row + §10 BLOCKER-1/MAJOR-4.

**Scope.** This module covers exactly one ``run_dynamics.py`` round
(``round_idx == 0``): a single monotonic descent through a ``t``-schedule
against one (round-invariant) checkpoint set, with a model-swap keyed only on
``t``. The reference's **multi-round resampling** machinery (``round_idx``
loop, host-side quantile filtering + hierarchical clustering + multinomial
importance resampling, ``resample_dynamics_model``/``resample_dynamics_checkpoint``,
forward-``diffuser.forward_marginal``-based re-noising *between rounds*) is a
**separate backlog node, E10** (the design authority's "structure prediction
pipeline" row, a ``pipeline()`` of ``Scan`` stages + a host-side ``Sink``/
``Fuse`` between rounds) and is **not implemented here** -- see the "Verified
findings from the reference" section below for exactly what was checked and
ruled out of this node's scope, not merely assumed absent.

## Resolved design tension: "pad-to-max+mask" vs. the already-committed ``while_loop``

The design authority (spec §3.3's Langevin row, MAJOR-4, and EPIC Fork 7) says
variable inner trip-count requires "pad-to-max+mask". The already-committed
``aminx.ebm.langevin.run_langevin_equilibration`` instead uses
``jax.lax.while_loop`` directly with **no** padding/masking, and its own
docstring says so explicitly ("not a compile-time-constant... while_loop does
not require it to be a compile-time constant, unlike ``lax.scan``'s
``length``"). This dispatch was told to resolve that tension by *reading the
reference*, not guessing. Having done so:

* The reference's "variable inner trip-count" is a **per-outer-t-level**
  property, computed by ``get_steps_for_time`` (``run_dynamics.py`` lines
  115-176) and applied once per level at line 731
  (``current_iters = get_steps_for_time(t, args, step_idx, round_idx)``),
  then consumed by a single Python ``for i in range(current_iters + 1):``
  (line 746) that applies **uniformly to the whole ``bsize`` batch tensor** --
  there is no per-batch-element (per-trajectory) divergence in step count
  anywhere in the file. ``get_steps_for_time`` supports genuine ramping
  (interval scheduling, linear/quadratic ramp-up/down, a step-function ramp,
  a ``last_t_steps`` override for the final level) -- confirmed not merely a
  theoretical option: the shipped README example
  (``~/repos/ProteinEBM/README.md`` lines 146-153) passes
  ``--ramp_start 0.5 --step_function_ramp --min_steps 0``, i.e. production
  runs really do vary ``n_steps`` across levels.
* Because this variability lives entirely on the **outer** (schedule-position)
  axis and the **inner** loop already uses ``while_loop`` (which accepts a
  fresh traced scalar trip count on every invocation with zero padding
  overhead -- that is precisely what "not a compile-time constant" buys you),
  looping the *outer* schedule via ``lax.scan`` with ``xs`` carrying a
  per-level ``n_steps`` value alongside ``t`` needs **no padding or masking
  at all**: each scan iteration simply calls the while_loop-based
  ``run_langevin_equilibration`` with that iteration's own scalar
  ``n_steps``, exactly as it would if called eagerly in a Python loop with a
  different literal each time.
* Conclusion: "pad-to-max+mask" was the correct mitigation for the *original*
  nested-``Scan`` mental model this design tension's risk-register entry
  describes (an inner ``lax.scan`` would need a static ``length``, forcing
  padding to accommodate per-level ``n_steps`` differences) -- it is **not**
  needed, and is **not implemented**, given the inner primitive's actual,
  already-shipped choice of ``while_loop``. This is a resolved discrepancy
  between the risk register's stated mitigation and the as-built E9-core
  primitive, not an oversight in this module: this module's ``xs`` simply
  carries ``(t, n_steps)`` pairs per level (see :func:`run_annealing_schedule`),
  with ``n_steps`` free to vary across levels with no special handling.
* What this module does **not** claim: there is no *batch-of-independent-
  trajectories* dimension inside this function (no ``vmap`` over parallel
  chains with per-chain step counts) -- that would be a separate, outer
  ``vmap``/``SafeMap`` layer around this whole function (mirroring
  ``run_dynamics.py``'s ``bsize`` batching, which is opaque here: a caller
  wanting a batch of independent annealing trajectories should ``vmap`` over
  ``run_annealing_schedule`` itself, exactly as ``tests/ebm/test_langevin.py``
  already does for the inner primitive), and JAX's own batched-``while_loop``
  masking semantics (mentioned as a hypothesis in this dispatch's brief) are
  simply not exercised by anything in this module.

## Verified findings from the reference

1. **Noise schedule is ``linspace``, not geometric.** ``run_dynamics.py``
   lines 666-667: ``reverse_steps = np.linspace(args.t_min, args.t_max,
   current_reverse_steps)[::-1]``, ``dt_rev = (args.t_max - args.t_min) /
   current_reverse_steps``. Defaults: ``t_min=0.001``, ``t_max=1.0``,
   ``reverse_steps=200`` (i.e. 200 evenly-t-spaced levels, descending from
   ``t_max`` to ``t_min``), ``dt=args.dt=0.001`` (the per-substep
   Euler-Maruyama step, held fixed across the whole run, distinct from
   ``dt_rev`` -- see finding 5 below for why ``dt_rev`` is not reproduced
   here). Per-level step count defaults to the constant ``args.steps``
   (default 100) absent any ramp flags, but see the padding-tension
   discussion above for why it is not actually constant in general.

2. **Model-swap threshold: confirmed real value, not merely plausible.**
   ``run_dynamics.py``'s ``get_dynamics_model`` (lines 262-268):
   ```python
   def get_dynamics_model(t):
       if lo_time_model is not None and t < args.lo_time_threshold:
           return lo_time_model
       if resample_dynamics_model is not None:
           return resample_dynamics_model
       return model
   ```
   i.e. ``t < lo_time_threshold -> lo_time_model``, else ``-> model`` (strict
   ``<``; the boundary ``t == threshold`` goes to the *high*-t/base model,
   not the low-time one). Default ``lo_time_threshold = -1`` (off: ``t`` is
   never ``< -1``, so no switching happens unless the flag is set). The
   README's shipped fast-folder example (lines 146-153) *does* set
   ``--lo_time_threshold 0.1``, and the weight README (lines 48-54)
   independently corroborates the same split: ``model_1_frozen_1m_md.pt`` is
   "Used for sampling the ``t>0.1`` noise levels" and
   ``model_6_expert_frozen_1m_md.pt`` ("ProteinEBM-x") is used for "sampling
   the ``t<0.1`` noise levels" -- so ``0.1`` is a real, production threshold,
   not a guess. The ``resample_dynamics_model`` branch (round-index-gated,
   only ever populated from ``round_idx >= 1``) is **out of this node's
   scope** -- it is E10's multi-round-resampling model swap, not E9's
   t-threshold swap; :func:`select_model_for_t` below implements only the
   ``lo_time_model``-vs-``model`` half.

3. **No forward-renoising step between t-levels within a round (verified
   absence, not an oversight).** Reading the full ``t``-schedule loop (lines
   720-871) for ``round_idx == 0``: the only place ``diffuser.forward_marginal``
   is called is (a) once, at the very start of a round, to jump straight to
   ``t_max`` from the raw input structure (lines 704-713 -- a
   one-time initialization this function's ``coords`` argument is assumed to
   already reflect, not a per-level step), and (b) between **resampling
   rounds** (``round_idx > 0``, lines 691-702, re-noising resampled structures
   back up to ``args.resample_noise_time``) -- confirmed E10 scope. Within a
   single round's descent through the schedule there is no renoising call at
   all; :func:`run_annealing_schedule` accordingly does not call
   :func:`aminx.ebm.diffusion.forward_marginal` anywhere, and this is a
   verified finding, not a gap.

4. **A real, additional simplification this module makes on top of E9-core's
   own documented scope limit (flagged plainly, not hidden).** Beyond the
   "no renoising between levels" finding above, the reference's per-level
   inner loop (lines 746-871) does two things ``run_langevin_equilibration``
   does **not** reproduce (per that module's own docstring, which already
   scopes itself to "lines ~830-843" only): (a) for ``i < current_iters``, a
   *second*, forward Euler-Maruyama "Ito step" immediately after every
   equilibration step, evaluated at ``t - dt`` (lines 861-863, "now run
   forward ito step"), and (b) for the single final ``i == current_iters``
   iteration of each level, a **different** update using ``dt_rev`` (the
   level-spacing step) **without** ``effective_temp_scaling`` (lines 844-845)
   -- a dedicated "transport to the next time level" step. Since this
   dispatch's job is to loop the already-committed, do-not-modify
   ``run_langevin_equilibration`` (which implements neither (a) nor (b)) and
   was explicitly told not to re-litigate that module's own scope, the outer
   scan here relies entirely on the *implicit* mechanism of simply changing
   ``t`` for the next level while carrying the same physical ``coords``
   forward -- there is no explicit transport/Ito-push step of any kind
   between levels in this module. This is a real, compounding numerical
   deviation from ``run_dynamics.py``'s exact discretization (on top of
   E9-core's own already-documented one), not merely a naming difference;
   anyone later validating trajectory-level parity against the PyTorch
   reference should know this loop does not attempt to reproduce ``dt_rev``/
   the extra Ito push, only the coarser "equilibrate ``n_steps`` at fixed
   ``t``, then move to the next scheduled ``t``" behavior.

5. **``temp_scaling_after_lo_time`` (lines 740-744) is out of scope.** The
   reference additionally varies ``effective_temp_scaling`` itself by ``t``
   (a *second*, independent ``t``-threshold switch, gated by the same
   ``lo_time_threshold`` but for temperature, not model choice). This
   dispatch's required signature takes a single, schedule-constant
   ``effective_temp_scaling: float``; the reference's ``t``-dependent
   temperature variant is not reproduced. Documented here rather than
   silently dropped.

## Model-swap dispatcher memory cost (BLOCKER-1, plainly stated, not solved)

:func:`select_model_for_t` dispatches over **pre-loaded** model instances via
``jax.lax.cond``/``jax.lax.switch`` -- both trace *every* branch, so every
model passed to it must be resident on-device simultaneously for the whole
call. For ``N`` ~85M-parameter ProteinEBM checkpoints, that is an ``N`` x 85M
device-memory residency cost with no lazy loading or per-``t``-range
segmentation; this module does not attempt to solve that (the design
authority's own noted residual/mitigation is "host-side per-t-range
segmentation", a caller-side concern, e.g. only ever holding 2 checkpoints
resident for a 2-checkpoint schedule, never N > 2 for this node's scope).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
from xtrax.tiling import CarrySpec
from xtrax.transforms import safe_scan

from aminx.ebm.diffusion import DEFAULT_COORDINATE_SCALING
from aminx.ebm.langevin import DEFAULT_EFFECTIVE_TEMP_SCALING, run_langevin_equilibration

if TYPE_CHECKING:
  from jaxtyping import Array, Float, Int, PRNGKeyArray

  from aminx.ebm.contracts import AAType, Coords, DiffusionTime, ResidueMask
  from aminx.ebm.model import ProteinEBMModel

  NoiseLevelSchedule = Float[Array, "L"]
  StepCountSchedule = Int[Array, "L"]


def _validate_model_threshold_counts(n_models: int, n_thresholds: int) -> None:
  """Shared arg-count guard for :func:`select_model_for_t`."""
  if n_models == 0:
    msg = "select_model_for_t: `models` must be non-empty."
    raise ValueError(msg)
  expected = max(n_models - 1, 0)
  if n_thresholds != expected:
    msg = (
      f"select_model_for_t: {n_models} model(s) require exactly {expected} "
      f"threshold(s) (one boundary between each adjacent pair, ascending "
      f"t-range order); got {n_thresholds}."
    )
    raise ValueError(msg)


def select_model_for_t(
  models: tuple[ProteinEBMModel, ...],
  thresholds: tuple[float, ...],
  t: DiffusionTime,
) -> ProteinEBMModel:
  """Net-new ``jax.lax.cond``/``jax.lax.switch`` model-swap dispatcher (BLOCKER-1).

  Mirrors ``run_dynamics.py``'s ``get_dynamics_model(t)`` t-threshold branch
  (module docstring, finding 2) -- **not** its ``resample_dynamics_model``
  round-index branch, which is E10 scope. ``models`` must be given in
  **ascending t-range order**: ``models[0]`` handles the lowest-``t`` range
  (``[0, thresholds[0])``), ``models[-1]`` handles the highest-``t`` range
  (``[thresholds[-1], 1]``), and each ``thresholds[i]`` is the boundary
  between ``models[i]`` and ``models[i + 1]``. Boundary convention matches
  the reference's strict ``<`` exactly: at ``t == thresholds[i]`` exactly,
  the *higher*-``t`` model (``models[i + 1]``) is selected, not the lower one
  -- e.g. for the reference's 2-model, ``threshold=0.1`` case, ``t == 0.1``
  selects the base (high-t) model, matching ``t < 0.1`` evaluating to
  ``False`` there.

  For exactly 2 models this uses ``jax.lax.cond`` directly (as the dispatch
  brief specifically calls for); for 3+ models it uses ``jax.lax.switch``
  with a bucketized index (``sum(t >= thresholds)``, which reduces to the
  same 0/1 answer as the 2-model ``cond`` case when there is exactly one
  threshold). A single model (``len(models) == 1``, ``thresholds == ()``) is
  returned unconditionally with no ``cond``/``switch`` at all -- the
  degenerate, no-swap case.

  **Both/all branches must trace to an identical output pytree structure.**
  ``jax.lax.cond``/``jax.lax.switch`` require every branch to return the same
  treedef (same static ``eqx.field(static=True)`` metadata, same array
  shapes/dtypes for every leaf) -- this holds for any two
  :class:`~aminx.ebm.model.ProteinEBMModel` instances built from the *same*
  architecture hyperparameters (``token_s``/``token_z``/depth/heads/etc.),
  i.e. two checkpoints of the same architecture, which is exactly BLOCKER-1's
  documented assumption. If it does not hold (e.g. mismatched ``token_s``
  between checkpoints), JAX itself raises a clear pytree-mismatch error at
  trace time -- this function does not add its own redundant shape check on
  top of that. See ``tests/ebm/test_langevin_schedule.py``'s
  ``TestJitVmapCompatibility`` for a same-architecture, different-random-key
  pair standing in for two real checkpoints, verifying this holds in
  practice (not just asserted in prose).

  **Memory cost, not solved here (BLOCKER-1).** See module docstring's
  "Model-swap dispatcher memory cost" section -- every model in ``models``
  must be resident on-device for this call; this function performs no
  lazy-loading or host-side segmentation.

  Args:
      models: Pre-loaded model instances, ascending t-range order (see
        above). Never loaded from disk here -- checkpoint I/O is E3.5's job,
        already done upstream of this call.
      thresholds: ``len(models) - 1`` ascending t-boundaries between adjacent
        models.
      t: The diffusion time to dispatch on.

  Returns:
      The selected model instance (by construction, has the same pytree
      structure as every other element of ``models``).

  Raises:
      ValueError: If ``len(thresholds) != len(models) - 1``.

  """
  n_models = len(models)
  _validate_model_threshold_counts(n_models, len(thresholds))

  if n_models == 1:
    return models[0]

  if n_models == 2:
    lo_model, hi_model = models
    (threshold,) = thresholds
    return jax.lax.cond(t < threshold, lambda: lo_model, lambda: hi_model)

  thresholds_arr = jnp.asarray(thresholds)
  index = jnp.sum(t >= thresholds_arr).astype(jnp.int32)
  branches = tuple((lambda m=model: m) for model in models)
  return jax.lax.switch(index, branches)


def run_annealing_schedule(
  models: tuple[ProteinEBMModel, ...],
  thresholds: tuple[float, ...],
  coords: Coords,
  aatype: AAType,
  mask: ResidueMask,
  noise_schedule: NoiseLevelSchedule,
  n_steps_per_level: int | StepCountSchedule,
  dt: float,
  key: PRNGKeyArray,
  *,
  use_metropolis: bool = False,
  effective_temp_scaling: float = DEFAULT_EFFECTIVE_TEMP_SCALING,
  coordinate_scaling: float = DEFAULT_COORDINATE_SCALING,
) -> Coords:
  """Outer ``xtrax`` ``CarrySpec``+``Scan`` over the noise-level schedule (E9, OUTER half).

  Carry = ``(coords, key)``; ``xs`` = the per-level ``(t, n_steps)`` schedule.
  Each scan step: dispatch the model for that level's ``t`` via
  :func:`select_model_for_t`, then run exactly ``n_steps`` of the
  already-committed :func:`aminx.ebm.langevin.run_langevin_equilibration` at
  that ``t``, carrying the resulting ``coords`` (and a freshly split ``key``)
  into the next level. See the module docstring's "Verified findings" for
  exactly what this does and does not reproduce from ``run_dynamics.py``
  (in particular: no forward-renoising between levels, and no ``dt_rev``
  transport/extra-Ito-push step -- both verified-absent or
  verified-out-of-scope, not oversights).

  **Why a ``CarrySpec`` is built but not routed through the full
  ``AxisSpec``+``BatchPlanner`` pipeline.** ``xtrax.tiling.plan.BatchPlanner
  .plan()``'s Phase 0 (the only place a ``CarrySpec`` is normally consumed)
  does nothing to a ``CarrySpec`` beyond unwrapping it verbatim into
  ``Scan(init=cs.init, transition=cs.transition, ordered_sinks=cs.ordered_sinks)``
  (``xtrax/tiling/plan.py`` lines 227-231) -- there is no Vmap-vs-SafeMap
  cardinality trade-off to adjudicate here (a ``CarrySpec``'d axis is *always*
  ``Scan``, unconditionally), and this function is a single dedicated scan,
  not a multi-axis dispatch registry entry like ``aminx.ebm.plan``'s
  ``plan_axis``/``dispatch_axis`` (whose own ``Scan`` branch is a
  carry-*passthrough* map, documented there as unsuitable for a real
  evolving carry -- see that module's ``dispatch_axis`` docstring). Building
  an ``AxisSpec``+``BatchPlan`` here purely to immediately unwrap it back to
  the same ``(init, transition)`` pair would be ceremony with no behavioral
  difference; this function instead builds the ``CarrySpec`` (the same
  vocabulary the design authority names, and the same object type
  ``aminx.host.plan``/``aminx.run.specs`` declare carry-bearing axes with
  elsewhere in this repo) and executes it directly via
  ``xtrax.transforms.safe_scan`` -- provably identical to what
  ``BatchPlanner.plan()`` would have produced.

  PRNG discipline: ``key`` is split exactly once per scanned level
  (``key_i, round_key = jax.random.split(key_i)``), mirroring
  ``run_langevin_equilibration``'s own per-step convention; each level's
  ``round_key`` is handed to that level's ``run_langevin_equilibration`` call
  (which itself splits it once per inner step -- see that function's own
  PRNG-discipline note).

  Args:
      models: Pre-loaded model instances, ascending t-range order (see
        :func:`select_model_for_t`).
      thresholds: ``len(models) - 1`` ascending t-boundaries.
      coords: Initial coordinates, ``(N, 3)``, already at the noise level of
        ``noise_schedule[0]`` -- the one-time jump to the schedule's starting
        ``t`` (module docstring finding 3) is the caller's responsibility,
        not this function's.
      aatype: Per-residue amino-acid type, ``(N,)``, held fixed across every
        level.
      mask: Residue validity mask, ``(N,)``, held fixed across every level.
      noise_schedule: The per-level ``t`` values, ``(L,)``, typically a
        ``linspace(t_min, t_max, L)[::-1]`` descending schedule (module
        docstring finding 1) -- but this function does not itself construct
        or validate monotonicity; it only scans over whatever is given.
      n_steps_per_level: Either a single ``int`` (broadcast to a constant
        ``n_steps`` for every level) or an explicit ``(L,)`` int array (for
        the reference's genuine per-level ramping -- module docstring's
        padding-tension discussion). Either way, no padding/masking is
        needed (see module docstring).
      dt: Euler-Maruyama step size, held fixed across every level and every
        inner step (module docstring finding 4: the reference's separate,
        per-level ``dt_rev`` transport step is not reproduced here).
      key: PRNG key for the whole schedule (see PRNG discipline above).
      use_metropolis: Forwarded to every level's
        ``run_langevin_equilibration`` call, unchanged across the whole
        schedule (static Python ``bool``, per that function's own contract).
      effective_temp_scaling: Forwarded to every level, unchanged across the
        whole schedule (module docstring finding 5: the reference's
        ``temp_scaling_after_lo_time`` t-dependent variant is not
        reproduced).
      coordinate_scaling: Forwarded to every level, unchanged across the
        whole schedule.

  Returns:
      Final coordinates after the full schedule, ``(N, 3)``.

  Raises:
      ValueError: If ``len(thresholds) != len(models) - 1``, or if
        ``n_steps_per_level`` is given as an array whose length does not
        match ``noise_schedule``.

  """
  _validate_model_threshold_counts(len(models), len(thresholds))

  noise_schedule_arr = jnp.asarray(noise_schedule)
  n_levels = noise_schedule_arr.shape[0]

  if isinstance(n_steps_per_level, int):
    n_steps_arr = jnp.full((n_levels,), n_steps_per_level, dtype=jnp.int32)
  else:
    n_steps_arr = jnp.asarray(n_steps_per_level, dtype=jnp.int32)
    if n_steps_arr.shape[0] != n_levels:
      msg = (
        f"run_annealing_schedule: n_steps_per_level has length "
        f"{n_steps_arr.shape[0]}, expected {n_levels} (== len(noise_schedule))."
      )
      raise ValueError(msg)

  def _round_transition(
    carry: tuple[Coords, PRNGKeyArray],
    x: tuple[DiffusionTime, jax.Array],
  ) -> tuple[tuple[Coords, PRNGKeyArray], None]:
    coords_i, key_i = carry
    t_i, n_steps_i = x
    key_i, round_key = jax.random.split(key_i)
    model_i = select_model_for_t(models, thresholds, t_i)
    # `run_langevin_equilibration`'s own docstring: "n_steps: ... traced or
    # static; while_loop does not require it to be a compile-time constant"
    # -- its type hint (`int`) is narrower than what it actually accepts at
    # runtime (a traced scalar `jax.Array`, exactly what a scanned `xs`
    # element is). `cast` here documents that gap for the type checker
    # without touching the untouchable `langevin.py` file.
    coords_next = run_langevin_equilibration(
      model_i,
      coords_i,
      aatype,
      t_i,
      mask,
      cast("int", n_steps_i),
      dt,
      round_key,
      use_metropolis=use_metropolis,
      effective_temp_scaling=effective_temp_scaling,
      coordinate_scaling=coordinate_scaling,
    )
    return (coords_next, key_i), None

  carry_spec = CarrySpec(
    axis_name="ebm_noise_schedule",
    init=(coords, key),
    transition=_round_transition,
  )
  xs = (noise_schedule_arr, n_steps_arr)
  (final_coords, _final_key), _ys = safe_scan(carry_spec.transition, carry_spec.init, xs)
  return final_coords


__all__ = ["run_annealing_schedule", "select_model_for_t"]
