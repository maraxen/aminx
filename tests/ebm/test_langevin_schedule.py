"""Tests for ``aminx.ebm.langevin_schedule`` (backlog node **E9**, OUTER half).

Matches the rigor established by ``tests/ebm/test_langevin.py`` for the inner
primitive: toy-model, non-vacuous convergence/dispatch checks for the fast
tests, plus a real (small) ``ProteinEBMModel`` jit/vmap compatibility check.
Scope matches the dispatch: a single round's descent through a noise-level
schedule, with a t-threshold model-swap; the multi-round resampling driver
(E10) is out of scope and not tested here.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from aminx.ebm.langevin_schedule import run_annealing_schedule, select_model_for_t
from aminx.ebm.model import ProteinEBMModel

N = 5
AATYPE = jnp.zeros((N,), dtype=jnp.int32)
MASK = jnp.ones((N,), dtype=bool)


class _ToyConstantModel(eqx.Module):
  """Toy model whose ``aux_score`` is a fixed, distinguishable constant vector.

  Distinct from ``test_langevin.py``'s ``_ToyCenteredModel``/``_ToyFlatModel``
  (which exist to test convergence/MH detailed-balance): this toy exists only
  to make it trivial to tell, after running a schedule, *which* model handled
  which round -- ``energy`` accumulates a per-model tag so a difference in
  which model ran shows up as a difference in the final coordinates, not just
  in some side-channel log.

  **Must be an ``eqx.Module`` (a real JAX pytree), not a plain class.**
  ``select_model_for_t`` dispatches over ``models`` via ``jax.lax.cond``/
  ``jax.lax.switch``, which requires every branch's return value to be a
  valid JAX pytree with a matching treedef -- an arbitrary plain Python
  object (as ``test_langevin.py``'s toy models are, since those are only
  ever *called*, never themselves selected between by ``cond``/``switch``)
  is not a valid JAX type and raises at trace time. Production models
  (``ProteinEBMModel``) are already ``eqx.Module``s, so this constraint is
  real, not a testing artifact.
  """

  tag: jax.Array

  def __init__(self, tag: float) -> None:
    self.tag = jnp.asarray(tag, dtype=jnp.float32)

  def energy(self, coords: jax.Array, aatype: jax.Array, t: jax.Array, mask: jax.Array) -> jax.Array:
    del aatype, t, mask
    return jnp.zeros(()) * jnp.sum(coords)

  def aux_score(self, coords: jax.Array, aatype: jax.Array, t: jax.Array, mask: jax.Array) -> jax.Array:
    del aatype, t, mask
    return jnp.full_like(coords, self.tag)


class TestSelectModelForTDispatcher:
  """(1) Model-swap dispatcher: distinguishable toy models, both sides of the threshold."""

  def test_two_models_pick_correct_side_below_and_above_threshold(self) -> None:
    """Note: compares by ``.tag`` value, not Python object identity.

    ``jax.lax.cond`` always traces and reconstructs its selected branch's
    pytree via flatten/unflatten -- even called eagerly outside ``jit`` --
    so the returned model is a new (but structurally/value-identical)
    instance, never the exact same Python object. Value equality is the
    correct check here, not ``is``.
    """
    lo_model = _ToyConstantModel(tag=-1.0)
    hi_model = _ToyConstantModel(tag=1.0)
    threshold = 0.1

    below = select_model_for_t((lo_model, hi_model), (threshold,), jnp.array(0.05))
    above = select_model_for_t((lo_model, hi_model), (threshold,), jnp.array(0.5))

    assert jnp.allclose(below.tag, lo_model.tag)
    assert jnp.allclose(above.tag, hi_model.tag)

  def test_two_models_boundary_goes_to_high_t_model(self) -> None:
    """Matches the reference's strict `t < threshold` exactly (module docstring finding 2):

    at t == threshold, the *high*-t model wins (t < threshold is False there).
    """
    lo_model = _ToyConstantModel(tag=-1.0)
    hi_model = _ToyConstantModel(tag=1.0)
    threshold = 0.1

    at_boundary = select_model_for_t((lo_model, hi_model), (threshold,), jnp.array(threshold))

    assert jnp.allclose(at_boundary.tag, hi_model.tag)

  def test_two_models_dispatch_is_jit_compatible_and_actually_selects(self) -> None:
    lo_model = _ToyConstantModel(tag=-2.0)
    hi_model = _ToyConstantModel(tag=2.0)
    threshold = 0.1

    def pick_and_score(t: jax.Array) -> jax.Array:
      model = select_model_for_t((lo_model, hi_model), (threshold,), t)
      return model.aux_score(jnp.zeros((N, 3)), AATYPE, t, MASK)

    jitted = jax.jit(pick_and_score)
    below_score = jitted(jnp.array(0.05))
    above_score = jitted(jnp.array(0.5))

    assert jnp.allclose(below_score, -2.0)
    assert jnp.allclose(above_score, 2.0)

  def test_three_models_uses_switch_and_picks_each_range(self) -> None:
    m0 = _ToyConstantModel(tag=0.0)
    m1 = _ToyConstantModel(tag=1.0)
    m2 = _ToyConstantModel(tag=2.0)
    thresholds = (0.1, 0.5)

    picked_low = select_model_for_t((m0, m1, m2), thresholds, jnp.array(0.05))
    picked_mid = select_model_for_t((m0, m1, m2), thresholds, jnp.array(0.3))
    picked_high = select_model_for_t((m0, m1, m2), thresholds, jnp.array(0.9))

    assert jnp.allclose(picked_low.tag, m0.tag)
    assert jnp.allclose(picked_mid.tag, m1.tag)
    assert jnp.allclose(picked_high.tag, m2.tag)

  def test_single_model_returns_unconditionally(self) -> None:
    """No `cond`/`switch` is invoked for a single model, so `is` identity does hold here."""
    only_model = _ToyConstantModel(tag=7.0)
    picked = select_model_for_t((only_model,), (), jnp.array(0.5))
    assert picked is only_model

  def test_mismatched_threshold_count_raises(self) -> None:
    lo_model = _ToyConstantModel(tag=-1.0)
    hi_model = _ToyConstantModel(tag=1.0)
    with pytest.raises(ValueError, match="threshold"):
      select_model_for_t((lo_model, hi_model), (0.1, 0.2), jnp.array(0.3))


class _ToyCenteredModel:
  """Same toy quadratic-energy convergence model as ``test_langevin.py``'s ``_ToyCenteredModel``.

  Reused here (not imported -- kept test-local per that module's own
  precedent of defining its toy fixtures locally) to verify the *outer*
  schedule actually makes non-trivial progress, not merely that it runs
  without erroring.
  """

  def __init__(self, target_centered: jax.Array, k: float = 1.0) -> None:
    self.target_centered = target_centered
    self.k = k

  def _centered_energy(self, coords: jax.Array) -> jax.Array:
    mean = jnp.mean(coords, axis=0)
    centered = coords - mean
    diff = centered - self.target_centered
    return 0.5 * self.k * jnp.sum(diff**2)

  def energy(self, coords: jax.Array, aatype: jax.Array, t: jax.Array, mask: jax.Array) -> jax.Array:
    del aatype, t, mask
    return self._centered_energy(coords)

  def aux_score(self, coords: jax.Array, aatype: jax.Array, t: jax.Array, mask: jax.Array) -> jax.Array:
    del aatype, t, mask
    return -jax.grad(self._centered_energy)(coords)


class TestRunAnnealingScheduleEndToEnd:
  """(2) Outer schedule: short synthetic multi-round schedule, non-trivial coordinate evolution."""

  def test_schedule_iterates_expected_number_of_rounds_and_moves_coords(self) -> None:
    target_centered = jnp.zeros((N, 3))
    model = _ToyCenteredModel(target_centered, k=20.0)
    models = (model,)
    thresholds: tuple[float, ...] = ()

    # Low-t regime (mirrors test_langevin.py's tuned convergence test, which
    # uses t=0.01): at higher t, diffusion_coef(t) is much larger (more
    # per-step noise relative to the score-following drift), so this
    # schedule -- deliberately low ``t`` throughout, as a real annealing
    # schedule's tail would be -- is where 3 rounds x 100 steps reliably
    # converges without needing an enormous temp_scale to fight the noise
    # floor at higher t.
    noise_schedule = jnp.array([0.05, 0.03, 0.01])
    n_steps_per_level = 100
    dt = 2e-4
    temp_scale = 3.0

    key = jax.random.PRNGKey(0)
    k_start, k_run = jax.random.split(key)
    coords0 = jax.random.normal(k_start, (N, 3)) * 0.5

    def dist_to_target(coords: jax.Array) -> jax.Array:
      return jnp.linalg.norm((coords - jnp.mean(coords, axis=0)) - target_centered)

    d0 = float(dist_to_target(coords0))
    e0 = float(model.energy(coords0, AATYPE, noise_schedule[0], MASK))

    final_coords = run_annealing_schedule(
      models,
      thresholds,
      coords0,
      AATYPE,
      MASK,
      noise_schedule,
      n_steps_per_level,
      dt,
      k_run,
      effective_temp_scaling=temp_scale,
    )

    d_final = float(dist_to_target(final_coords))
    e_final = float(model.energy(final_coords, AATYPE, noise_schedule[-1], MASK))

    print(f"E0={e0:.4f} Ef={e_final:.4f} d0={d0:.4f} df={d_final:.4f}")

    assert final_coords.shape == coords0.shape
    assert jnp.all(jnp.isfinite(final_coords))
    # Non-vacuous: 3 rounds x 50 steps of real Langevin dynamics measurably
    # reduces distance-to-target -- not a no-op / identity scan.
    assert d_final < 0.7 * d0
    assert e_final < 0.7 * e0
    # Not literally unchanged from the initial coordinates.
    assert not jnp.allclose(final_coords, coords0, atol=1e-3)

  def test_schedule_actually_uses_both_models_across_the_threshold(self) -> None:
    """A schedule straddling the threshold must run the low-t model for the low-t rounds.

    Constructed so the final coordinates are only reachable if both models
    were genuinely dispatched (not e.g. the high-t model silently used for
    every round): the driving model's score is a fixed, non-uniform,
    **zero-mean-over-residues** pattern (not a spatially-uniform constant --
    see the note below on why a uniform score is the wrong toy here), scaled
    by a distinguishable per-model ``tag``, and the noise schedule has one
    round on each side of the threshold.

    Note on why the score must be non-uniform: ``langevin_step`` always
    re-centers (subtracts the per-step mean over residues) after every
    update. A spatially *uniform* score (e.g. ``_ToyConstantModel`` from
    ``TestSelectModelForTDispatcher`` above) pushes every residue by the same
    amount, which is then exactly canceled by that re-centering -- so it
    would leave **no** detectable trace in the final coordinates regardless
    of which model ran. A zero-mean-over-residues *pattern* (this test's
    ``_ToyPatternModel``) survives re-centering (which only removes the mean,
    not relative/structured differences), so its correct dispatch is
    actually observable.
    """

    class _ToyPatternModel(eqx.Module):
      """Toy model whose score is ``pattern * tag`` -- see class-level note above."""

      pattern: jax.Array
      tag: jax.Array

      def __init__(self, pattern: jax.Array, tag: float) -> None:
        self.pattern = pattern
        self.tag = jnp.asarray(tag, dtype=jnp.float32)

      def energy(self, coords: jax.Array, aatype: jax.Array, t: jax.Array, mask: jax.Array) -> jax.Array:
        del aatype, t, mask
        return jnp.zeros(()) * jnp.sum(coords)

      def aux_score(self, coords: jax.Array, aatype: jax.Array, t: jax.Array, mask: jax.Array) -> jax.Array:
        del coords, aatype, t, mask
        return self.pattern * self.tag

    # Zero-mean-over-residues by construction (jnp.linspace(-1, 1, N) for
    # N=5 is exactly [-1, -0.5, 0, 0.5, 1], mean 0), broadcast over xyz.
    pattern = jnp.linspace(-1.0, 1.0, N)[:, None] * jnp.ones((1, 3))
    assert abs(float(jnp.mean(pattern))) < 1e-6

    hi_model = _ToyPatternModel(pattern, tag=0.0)  # inert (zero score) for t >= threshold
    lo_model = _ToyPatternModel(pattern, tag=5.0)  # strong, distinguishable push for t < threshold
    threshold = 0.15

    noise_schedule = jnp.array([0.5, 0.05])  # round 0: hi_model (inert); round 1: lo_model (driven)
    n_steps_per_level = 3
    dt = 1e-3
    temp_scale = 20.0  # amplifies the drift term only, not the noise term (see langevin_step)

    key = jax.random.PRNGKey(1)
    coords0 = jnp.zeros((N, 3))

    final_coords = run_annealing_schedule(
      (lo_model, hi_model),
      (threshold,),
      coords0,
      AATYPE,
      MASK,
      noise_schedule,
      n_steps_per_level,
      dt,
      key,
      effective_temp_scaling=temp_scale,
    )

    # Project the final coordinates onto `pattern`: if the dispatcher always
    # picked hi_model (inert, tag=0), this projection reflects diffusive
    # noise only (empirically, ~5-10 in magnitude for this seed/config); if
    # it correctly switches to lo_model for round 1, the projection is
    # dominated by the strong, structured drift (empirically ~30).
    projection = float(jnp.sum(final_coords * pattern))
    print(f"pattern_projection={projection:.4f} (expect >> 10 if lo_model ran)")
    assert projection > 15.0

  def test_variable_n_steps_per_level_is_accepted_and_runs(self) -> None:
    """Per-level n_steps array (the reference's genuine ramping case) -- no padding needed."""
    target_centered = jnp.zeros((N, 3))
    model = _ToyCenteredModel(target_centered, k=5.0)

    noise_schedule = jnp.array([0.3, 0.2, 0.1])
    n_steps_per_level = jnp.array([10, 30, 5], dtype=jnp.int32)  # genuinely level-dependent
    dt = 1e-3
    key = jax.random.PRNGKey(2)
    coords0 = jax.random.normal(jax.random.PRNGKey(3), (N, 3)) * 0.3

    final_coords = run_annealing_schedule(
      (model,),
      (),
      coords0,
      AATYPE,
      MASK,
      noise_schedule,
      n_steps_per_level,
      dt,
      key,
    )

    assert final_coords.shape == coords0.shape
    assert jnp.all(jnp.isfinite(final_coords))

  def test_mismatched_n_steps_array_length_raises(self) -> None:
    model = _ToyCenteredModel(jnp.zeros((N, 3)))
    noise_schedule = jnp.array([0.3, 0.2, 0.1])
    bad_n_steps = jnp.array([10, 30], dtype=jnp.int32)  # length 2, schedule length 3

    with pytest.raises(ValueError, match="n_steps_per_level"):
      run_annealing_schedule(
        (model,),
        (),
        jnp.zeros((N, 3)),
        AATYPE,
        MASK,
        noise_schedule,
        bad_n_steps,
        1e-3,
        jax.random.PRNGKey(4),
      )


TOKEN_S = 16
TOKEN_Z = 8


def _make_small_model(key: jax.Array) -> ProteinEBMModel:
  return ProteinEBMModel(
    token_s=TOKEN_S,
    token_z=TOKEN_Z,
    dim_fourier=12,
    conditioning_transition_layers=1,
    transformer_depth=2,
    transformer_heads=2,
    key=key,
  )


class TestJitVmapCompatibility:
  """(3) jit/vmap compatibility against a real small ProteinEBMModel pair.

  Two small models built from the *same* architecture hyperparameters but
  different random init keys stand in for two independently-trained
  checkpoints of the same architecture (BLOCKER-1's documented assumption
  for ``select_model_for_t``/``run_annealing_schedule``'s model-swap dispatch
  -- same pytree structure, different leaf values).
  """

  def test_select_model_for_t_is_jit_compatible_with_real_models(self) -> None:
    model_lo = _make_small_model(jax.random.PRNGKey(30))
    model_hi = _make_small_model(jax.random.PRNGKey(31))
    threshold = 0.1

    def pick(t: jax.Array) -> ProteinEBMModel:
      return select_model_for_t((model_lo, model_hi), (threshold,), t)

    jitted = eqx.filter_jit(pick)
    picked_lo = jitted(jnp.array(0.05))
    picked_hi = jitted(jnp.array(0.5))

    # Structural sanity: both selections are usable ProteinEBMModel instances
    # with the expected static config (same architecture on both sides of
    # the swap, per BLOCKER-1's assumption).
    assert picked_lo.token_s == TOKEN_S
    assert picked_hi.token_s == TOKEN_S

    coords = jax.random.normal(jax.random.PRNGKey(32), (N, 3)) * 0.1
    t = jnp.array(0.2)
    e_lo = picked_lo.energy(coords, AATYPE, t, MASK)
    e_hi = picked_hi.energy(coords, AATYPE, t, MASK)
    assert jnp.isfinite(e_lo)
    assert jnp.isfinite(e_hi)

  def test_run_annealing_schedule_is_jit_compatible_with_real_models(self) -> None:
    model_lo = _make_small_model(jax.random.PRNGKey(33))
    model_hi = _make_small_model(jax.random.PRNGKey(34))
    threshold = 0.1

    coords = jax.random.normal(jax.random.PRNGKey(35), (N, 3)) * 0.1
    noise_schedule = jnp.array([0.5, 0.2, 0.05])
    n_steps_per_level = 2
    dt = 1e-3
    key = jax.random.PRNGKey(36)

    def run(c: jax.Array, kk: jax.Array) -> jax.Array:
      return run_annealing_schedule(
        (model_lo, model_hi),
        (threshold,),
        c,
        AATYPE,
        MASK,
        noise_schedule,
        n_steps_per_level,
        dt,
        kk,
      )

    eager = run(coords, key)
    jitted = eqx.filter_jit(run)(coords, key)

    assert jnp.allclose(eager, jitted, atol=1e-5)
    assert jnp.all(jnp.isfinite(jitted))

  def test_run_annealing_schedule_is_vmap_compatible_over_keys(self) -> None:
    model_lo = _make_small_model(jax.random.PRNGKey(37))
    model_hi = _make_small_model(jax.random.PRNGKey(38))
    threshold = 0.1

    coords = jax.random.normal(jax.random.PRNGKey(39), (N, 3)) * 0.1
    noise_schedule = jnp.array([0.5, 0.2, 0.05])
    n_steps_per_level = 2
    dt = 1e-3
    keys = jax.random.split(jax.random.PRNGKey(40), 3)

    vmapped = eqx.filter_vmap(
      lambda kk: run_annealing_schedule(
        (model_lo, model_hi),
        (threshold,),
        coords,
        AATYPE,
        MASK,
        noise_schedule,
        n_steps_per_level,
        dt,
        kk,
      ),
    )(keys)

    assert vmapped.shape == (3, N, 3)
    assert jnp.all(jnp.isfinite(vmapped))


def test_empty_noise_schedule_raises_value_error() -> None:
  """An empty noise_schedule raises ValueError rather than silently returning initial coords."""
  model = _make_small_model(jax.random.PRNGKey(50))
  coords = jnp.ones((N, 3), dtype=jnp.float32)
  with pytest.raises(ValueError, match="noise_schedule must be non-empty"):
    run_annealing_schedule(
      (model,),
      (),
      coords,
      AATYPE,
      MASK,
      jnp.zeros((0,), dtype=jnp.float32),
      2,
      0.01,
      jax.random.PRNGKey(51),
    )
