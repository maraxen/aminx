"""Host-side pH design of one structure: merged Potts table to a ranked list of designs.

Orchestration port of what upstream ``PottsMPNNPHEngine`` does between ``run_ph_redesign`` and its
optimisers, for ``backend="potts"`` only. Upstream: ``ProtonPottsMPNN/foundry/models/mpnn/src/mpnn/
inference_engines/potts_mpnn_ph.py``.

- ``run_ph_redesign`` 1030-1090: one unit of work per (criteria, placement seed); results deduplicated and
  sorted. This module is ONE unit (one structure, one native seed).
- ``_run_design_task`` 666-677 and ``_run_placement_one_seed`` 1268-1290: the placement plan is built on the
  NATIVE sequence, then each plan is designed ``samples_per_site`` times. A plan with no designable positions
  is skipped (``if not plan.designable: continue``).
- ``_score_design`` 1440-1496: ``final_potts_energy`` is the whole-sequence Potts energy; ``selective_energy`` is
  the sum over pins of ``e_P - mean_d e_D`` on the FINAL sequence, with each pin held at its protonated token.
  A centre-free design has no pins, so its selective energy is 0.0.

Deliberate deviation: upstream ``PHDesignSet.deduped`` (~line 612) keeps one design per ``design_id``, and
``design_id`` omits the temperature and other sweep knobs, so a sweep silently drops designs. This module
does NOT dedup. It returns every design, sorted by ``(final_potts_energy, original order)`` with a stable sort.

Randomness: upstream seeds torch per design (``torch.manual_seed`` in ``_run_placement_one_seed``). This module
draws the inverse-CDF uniforms from a JAX key (one stream per sample, ``fold_in(key, sample)``), so sampled
designs do NOT reproduce upstream's RNG stream. Replay parity needs explicit ``uniforms`` (and ``cdf_order``), as
in ``scripts/protonpotts/protonpotts_ph_block_parity.py``.

Repetitive-window residue numbers: binder positions carry their residue id. Every other position carries the
sentinel ``-10**6 - 100 * position``, because upstream counts repetitive neighbours only among binder-chain
residues (``res_id_to_pos``, potts_mpnn_ph.py 1660).

Plain Python and numpy around the jitted optimisers; no ``jax.vmap``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from aminx.families.potts_mpnn.etab import potts_energy
from aminx.families.protonpotts_mpnn.ph_descent import block_descent, rep_class_mask
from aminx.families.protonpotts_mpnn.ph_plan import (
  Pin,
  Plan,
  block_table,
  plan_centre_free,
  plan_from_center_types,
  plan_from_explicit_centers,
  valid_token_mask,
)
from aminx.families.protonpotts_mpnn.ph_potentials import candidate_energies, candidate_energies_at

if TYPE_CHECKING:
  from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig
  from aminx.families.protonpotts_mpnn.ph_descent import DescentResult
  from aminx.families.protonpotts_mpnn.ph_greedy import GreedyResult

_SENTINEL = 10**6


class PHDesign(NamedTuple):
  sequence: np.ndarray  # (L,) int32 aminx tokens, the FULL sequence (target positions native)
  method: str  # "block_descent" | "greedy_energy_block"
  sample: int  # index within the plan (0..samples_per_site-1)
  pins: tuple[Pin, ...]  # () for centre-free
  designable: tuple[int, ...]
  label: str  # Plan.label
  final_potts_energy: float  # whole-sequence energy via potts_energy
  selective_energy: float  # sum over pins of (e_P - mean_d e_D); 0.0 when there are no pins
  selective_energies: tuple[float, ...]
  n_draws: int


def _plan_for(
  table: jnp.ndarray,
  e_idx: np.ndarray,
  native: np.ndarray,
  binder_mask: np.ndarray,
  res_id: np.ndarray,
  config: PHDesignConfig,
) -> Plan | None:
  """Placement plan for this structure, or None when the upstream planner returns no plan.

  Explicit centres take precedence in the upstream ``enumerate_placement_plans`` order (1333-1337) but
  ``PHDesignConfig`` allows both to be set, so the two are refused together here instead of guessing.
  """
  dep_map = config.dep_map_dict()
  if config.method == "greedy_energy_block":
    if config.center_types or config.explicit_centers:
      raise NotImplementedError(
        "greedy_energy_block with pinned centres is not ported (ph_greedy takes no pins); "
        "use centre_types=() and explicit_centers=() for the centre-free plan",
      )
    if config.infill_scope != "chain":
      raise ValueError("centre-free greedy_energy_block needs infill_scope='chain'")
    return plan_centre_free(binder_mask)
  if config.explicit_centers and config.center_types:
    raise ValueError("set either explicit_centers or center_types, not both")
  if config.explicit_centers:
    return plan_from_explicit_centers(
      e_idx,
      binder_mask,
      res_id,
      config.explicit_centers,
      dep_map,
      infill_scope=config.infill_scope,
      neighbour_k=config.neighbour_k,
      max_mutations=config.max_mutations,
    )
  if not config.center_types:
    raise ValueError("block_descent needs center_types or explicit_centers (center_count >= 1)")
  # Upstream ranks by the NATIVE sequence's conditional energies (scan_potts), per the module docstring.
  field = np.asarray(
    candidate_energies(table, jnp.asarray(e_idx), jnp.asarray(native, dtype=jnp.int32)),
  )
  return plan_from_center_types(
    field,
    e_idx,
    binder_mask,
    res_id,
    config.center_types,
    dep_map,
    infill_scope=config.infill_scope,
    neighbour_k=config.neighbour_k,
    max_mutations=config.max_mutations,
  )


def force_pins(seq: np.ndarray, pins: Sequence[Pin]) -> np.ndarray:
  """Copy of ``seq`` with every pinned centre at its protonated token.

  Upstream ``_selective_energy_of`` scores a sequence that already carries all centres at their
  protonated token, in the presence of the others. Designs already satisfy this (the optimiser pins
  them); selectivity scoring of a given sequence must apply it first.
  """
  out = np.array(seq, dtype=np.int32, copy=True)
  for pin in pins:
    out[pin.position] = pin.prot_idx
  return out


def selectivity_gaps(
  table: jnp.ndarray,
  e_idx: jnp.ndarray,
  seq: np.ndarray,
  pins: Sequence[Pin],
) -> tuple[float, tuple[float, ...]]:
  """(summed, per-pin) selectivity gaps of ``seq``, which must already carry the centres forced.

  Per pin ``c``: ``e_c(protonated) - mean_d e_c(d)``, with ``e_c`` the conditional energies at ``c``
  from ``candidate_energies_at`` on the merged table. ``()`` and ``0.0`` when there are no pins.
  """
  if not pins:
    return 0.0, ()
  seq_j = jnp.asarray(seq, dtype=jnp.int32)
  positions = jnp.asarray([p.position for p in pins], dtype=jnp.int32)
  rows = np.asarray(candidate_energies_at(table, e_idx, seq_j, positions))
  gaps = tuple(
    float(rows[i, p.prot_idx] - np.mean([rows[i, d] for d in p.dep_idxs]))
    for i, p in enumerate(pins)
  )
  return float(sum(gaps)), gaps


def _score(
  table: jnp.ndarray,
  e_idx: jnp.ndarray,
  seq: np.ndarray,
  pins: Sequence[Pin],
) -> tuple[float, tuple[float, ...]]:
  """(final Potts energy, per-pin selective gaps) of a full sequence. Port of upstream ``_score_design``."""
  seq_j = jnp.asarray(seq, dtype=jnp.int32)
  valid = jnp.ones(seq.shape[0], dtype=bool)
  final = float(potts_energy(table, e_idx, valid, seq_j))
  _total, gaps = selectivity_gaps(table, e_idx, seq, pins)
  return final, gaps


def _sample_uniforms(
  sample: int,
  *,
  need: int,
  dtype: jnp.dtype,
  key: jax.Array | None,
  uniforms: Sequence[np.ndarray] | None,
) -> jnp.ndarray:
  """Uniforms for one sample, padded to ``need`` entries.

  Replay (``uniforms`` given) pads a shorter recorded array with zeros and uses longer ones verbatim, as
  ``protonpotts_ph_block_parity.run_call`` does. Otherwise draws ``need`` uniforms from ``fold_in(key, sample)``.
  """
  if uniforms is not None:
    recorded = np.asarray(uniforms[sample], dtype=dtype).reshape(-1)
    padded = np.zeros(max(need, recorded.shape[0]), dtype=dtype)
    padded[: recorded.shape[0]] = recorded
    return jnp.asarray(padded)
  return jax.random.uniform(jax.random.fold_in(key, sample), (need,), dtype=dtype)


def _draw_length(config: PHDesignConfig, n_designable: int) -> int:
  """Uniforms the optimiser requires: ``block_max_rounds * N`` for block descent, ``cv_max * N`` for greedy."""
  rounds = config.block_max_rounds if config.method == "block_descent" else config.cv_max
  return max(1, rounds * n_designable)


def _run_optimiser(
  plan: Plan,
  table: jnp.ndarray,
  e_idx: jnp.ndarray,
  e_idx_np: np.ndarray,
  native: np.ndarray,
  residue: np.ndarray,
  config: PHDesignConfig,
  draws_u: jnp.ndarray,
  cdf_order: jnp.ndarray | None,
) -> DescentResult | GreedyResult:
  """One optimiser run for a plan: pin arrays and block table as in ``protonpotts_ph_block_parity.run_call``."""
  valid = jnp.asarray(valid_token_mask(config.forbidden_tokens))
  rep = jnp.asarray(rep_class_mask(config.repetitive_window_parents))
  residue_j = jnp.asarray(residue, dtype=jnp.int32)
  native_j = jnp.asarray(native, dtype=jnp.int32)
  designable_j = jnp.asarray(plan.designable, dtype=jnp.int32)
  if config.method == "greedy_energy_block":
    # Lazy: the greedy optimiser is a sibling module; only this path needs it.
    from aminx.families.protonpotts_mpnn.ph_greedy import greedy_energy_block  # noqa: PLC0415

    return greedy_energy_block(
      table,
      e_idx,
      native_j,
      designable_j,
      valid,
      rep,
      residue_j,
      config=config,
      uniforms=draws_u,
      cdf_order=cdf_order,
    )
  blocks, block_valid = block_table(e_idx_np, plan.designable, config.block_size)
  depth = max((len(p.dep_idxs) for p in plan.pins), default=1)
  pin_dep = np.zeros((len(plan.pins), depth), dtype=np.int32)
  pin_dep_valid = np.zeros((len(plan.pins), depth), dtype=bool)
  for i, pin in enumerate(plan.pins):
    pin_dep[i, : len(pin.dep_idxs)] = pin.dep_idxs
    pin_dep_valid[i, : len(pin.dep_idxs)] = True
  return block_descent(
    table,
    e_idx,
    native_j,
    designable_j,
    jnp.asarray(blocks),
    jnp.asarray(block_valid),
    jnp.asarray([p.position for p in plan.pins], dtype=jnp.int32),
    jnp.asarray([p.prot_idx for p in plan.pins], dtype=jnp.int32),
    jnp.asarray(pin_dep),
    jnp.asarray(pin_dep_valid),
    valid,
    rep,
    residue_j,
    config=config,
    uniforms=draws_u,
    cdf_order=cdf_order,
  )


def design_structure(
  table: jnp.ndarray,
  e_idx: jnp.ndarray,
  native: jnp.ndarray,
  binder_mask: jnp.ndarray,
  res_id: jnp.ndarray,
  config: PHDesignConfig,
  *,
  key: jax.Array | None = None,
  uniforms: Sequence[np.ndarray] | None = None,
  cdf_order: jnp.ndarray | None = None,
) -> list[PHDesign]:
  """All pH designs of one structure, sorted by final Potts energy (ascending, stable).

  ``table (L, K, V, V)``, ``e_idx (L, K)``: merged Potts table and neighbour index, slot 0 = self.
  ``native (L,)``: the structure's own aminx tokens (the placement scan and the design start).
  ``binder_mask (L,) bool``: designable positions of ``config.binder_chain``. Raises ``ValueError`` if empty.
  ``res_id (L,)``: residue numbers.

  Plans: ``explicit_centers`` -> ``plan_from_explicit_centers``; ``center_types`` ->
  ``plan_from_center_types`` on ``candidate_energies(native)``; ``greedy_energy_block`` -> ``plan_centre_free``.
  A plan function that returns None (upstream ``[]``, e.g. more centre types than binder positions) gives ``[]``.

  Sampling: at ``temperature > 0`` each sample needs uniforms. Either ``key`` (a JAX PRNG key, folded with the
  sample index) or ``uniforms`` (one array per sample, replay) must be given. ``uniforms`` takes precedence.
  At ``temperature <= 0`` no uniforms are used.

  Output: every design is returned, including duplicates across sweep knobs. Upstream ``PHDesignSet.deduped``
  would drop designs whose ``design_id`` (which omits temperature and other sweep knobs) repeats. This
  function does not reproduce that; the order is ``(final_potts_energy, original order)``.
  """
  binder_mask = np.asarray(binder_mask, dtype=bool)
  if not binder_mask.any():
    raise ValueError("binder_mask has no True entries: no designable binder position")
  if config.seed_source != "native":
    raise NotImplementedError(
      f"seed_source={config.seed_source!r} is not ported; design starts from native",
    )

  sampling = config.temperature > 0
  if sampling and key is None and uniforms is None:
    raise ValueError("temperature > 0 needs a jax PRNG key or explicit uniforms")
  if sampling and uniforms is not None and len(uniforms) != config.samples_per_site:
    raise ValueError(
      f"uniforms has {len(uniforms)} entries, expected samples_per_site={config.samples_per_site}",
    )

  table_j = jnp.asarray(table)
  e_idx_j = jnp.asarray(e_idx, dtype=jnp.int32)
  e_idx_np = np.asarray(e_idx, dtype=np.int32)
  native_np = np.asarray(native, dtype=np.int32)
  res_id_np = np.asarray(res_id)
  length = native_np.shape[0]
  dtype = table_j.dtype

  plan = _plan_for(table_j, e_idx_np, native_np, binder_mask, res_id_np, config)
  if plan is None or not plan.designable:
    return []

  need = _draw_length(config, len(plan.designable))
  residue = np.where(binder_mask, res_id_np, -_SENTINEL - 100 * np.arange(length))
  designs: list[PHDesign] = []
  for sample in range(config.samples_per_site):
    if sampling:
      draws_u = _sample_uniforms(sample, need=need, dtype=dtype, key=key, uniforms=uniforms)
    else:
      draws_u = jnp.zeros((1,), dtype=dtype)
    result = _run_optimiser(
      plan,
      table_j,
      e_idx_j,
      e_idx_np,
      native_np,
      residue,
      config,
      draws_u,
      cdf_order,
    )

    seq = np.asarray(result.seq, dtype=np.int32)
    final_energy, gaps = _score(table_j, e_idx_j, seq, plan.pins)
    designs.append(
      PHDesign(
        sequence=seq,
        method=config.method,
        sample=sample,
        pins=tuple(plan.pins),
        designable=tuple(plan.designable),
        label=plan.label,
        final_potts_energy=final_energy,
        selective_energy=float(sum(gaps)),
        selective_energies=gaps,
        n_draws=int(result.n_draws),
      ),
    )

  # Stable sort: ties keep plan-then-sample order.
  designs.sort(key=lambda d: d.final_potts_energy)
  return designs
