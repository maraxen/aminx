"""Decoding-schedule selection for wave-color (chromatic) scheduling research (W0.3).

Wires a schedule selector across the five arms required by the variance battery
(W1a.2): `random_ar`, `fixed_n_to_c`, `frozen_random_sigma`, `chromatic`,
`improper_coloring`. Coloring adjacency is built from the model's edge set
(per-state CA k-NN — same metric as `aminx.utils.wave_parallel`, extended with
ligand-mediated contacts) so it can be checked against the actual model k-NN via
`assert_model_adj_subset` (confound C_knn_alignment, B3).

See `mpnn_ext/.praxia/docs/preregistration/260630_wave-color-scheduling-fundamental-tests.md`
section 3 (W0.3) for the spec this implements.

Multi-group (G > 1) decode: SUPPORTED. The `chromatic` and `improper_coloring`
arms built here produce multi-group waves — the whole point of within-color
Jacobi parallelism — and `AutoregressiveDecode.__call__` decodes every active
group slot of a wave from one shared forward pass (`step_fn` vmaps over the `G`
axis, masked by `group_valid`). Fixed in 85d8c480 (2026-06-30); regression-tested
by `tests/inference/decode/test_autoregressive.py::
test_chromatic_schedule_full_coverage_and_parallelism` (which asserts G > 1 is
actually exercised, not merely available) and
`::test_coloring_arm_via_per_sample_builder_decodes_every_position`.

This docstring previously claimed the opposite — that only group slot 0 was
decoded, so positions in slots 1..G-1 were "silently never sampled". That text
outlived the fix by six weeks and was taken at face value on 2026-08-11, which
got a blocker filed against a bug that no longer existed. Both tests above assert
the real behavior with a verified negative control, so the claim cannot be
re-derived from prose alone. If you change this paragraph, change those tests.

Real remaining constraint (NOT a decode bug): the coloring arms are HOST-SIDE
ONLY. `WaveScheduleBundle.from_colors` and the coloring adjacency builder use
`.tolist()` and Python loops, so requesting `schedule="chromatic"` from inside a
`jax.jit`/`vmap` trace raises `ConcretizationTypeError`. Today's traced drivers
(`host/kernel_dispatch.py`, `sampling/sample.py`) therefore cannot select them
via `schedule=`. The supported route is to build the schedule out of trace with
`build_wave_schedule_per_sample` and hand it to `build_inference_bundle(wave=...)`,
which bypasses `schedule=` entirely. Wiring a driver to do that is a driver task,
not a kernel one.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray

from aminx.types.bundles import WaveScheduleBundle

DecodingSchedule = Literal[
  "random_ar",
  "fixed_n_to_c",
  "frozen_random_sigma",
  "chromatic",
  "improper_coloring",
]

_COLORING_SCHEDULES = ("chromatic", "improper_coloring")
_ORDER_SCHEDULES = ("random_ar", "fixed_n_to_c", "frozen_random_sigma")

ScheduleKeyPolicy = Literal["unused", "run_level", "per_sample"]

SCHEDULE_KEY_POLICY: dict[DecodingSchedule, ScheduleKeyPolicy] = {
  # Identity order (arange(L)). The key is never consulted.
  "fixed_n_to_c": "unused",
  # Seed protocol M6: a FRESH key per sample. This arm's whole purpose is to expose
  # permutation noise, which requires the permutation to actually vary sample to sample.
  "random_ar": "per_sample",
  # Seed protocol M6: the SAME key reused across samples, so sigma is held fixed and
  # schedule_variance converges to the categorical floor instead of the permutation-noise
  # floor. Reusing one key is the definition of this arm, not an accident.
  "frozen_random_sigma": "run_level",
  # Plain chromatic coloring is fully DETERMINISTIC given the adjacency -- greedy
  # DSATUR over a fixed ordering -- and never reads the key at all.
  "chromatic": "unused",
  "improper_coloring": "run_level",
}
"""Which key discipline each schedule arm requires.

This table exists because the distinction used to live ONLY in
:func:`select_decoding_order`'s docstring, as an instruction to "caller discipline". No
caller implemented it: ``bundle_builder.build_inference_bundle`` was the sole caller and
silently defaulted to ``jax.random.PRNGKey(0)``, so every randomized arm was built from one
constant key. The measurable consequence was that ``random_ar`` and
``frozen_random_sigma`` produced byte-identical schedules -- collapsing the two arms whose
*difference* is the entire measurement -- and a variance battery comparing them would have
reported a null caused by plumbing rather than by the model.

A policy that a caller must remember is a policy that will be forgotten. Encoding it as
data lets :func:`build_wave_schedule_per_sample` apply it mechanically and lets
``build_inference_bundle`` reject a missing key instead of inventing one.
"""


def schedule_key_policy(mode: DecodingSchedule) -> ScheduleKeyPolicy:
  """Return the key discipline required by a schedule arm.

  Args:
    mode: One of the five schedule arms.

  Returns:
    ``"unused"``, ``"run_level"``, or ``"per_sample"``.

  Raises:
    ValueError: If ``mode`` is not a known schedule arm.
  """
  if mode not in SCHEDULE_KEY_POLICY:
    msg = f"Unknown schedule mode: {mode!r}. Known: {sorted(SCHEDULE_KEY_POLICY)}"
    raise ValueError(msg)
  return SCHEDULE_KEY_POLICY[mode]


def schedule_consumes_key(mode: DecodingSchedule) -> bool:
  """Whether a schedule arm reads its ``key`` at all.

  Args:
    mode: One of the five schedule arms.

  Returns:
    False for both ``fixed_n_to_c`` and ``chromatic``, True for all other arms.

  Raises:
    ValueError: If ``mode`` is not a known schedule arm.
  """
  return schedule_key_policy(mode) != "unused"


def build_wave_schedule_per_sample(
  mode: DecodingSchedule,
  *,
  key: PRNGKeyArray,
  num_samples: int,
  tie_group_map: Int[Array, " L"],
  coords: Float[Array, "L 4 3"] | None = None,
  mask: Float[Array, " L"] | None = None,
  k_neighbors: int = 48,
  structure_mapping: Int[Array, " L"] | None = None,
  ligand_coords: Float[Array, "A 3"] | None = None,
  ligand_mask: Float[Array, " A"] | None = None,
) -> WaveScheduleBundle:
  """Build a sample-axis-stacked :class:`WaveScheduleBundle`, honoring the key policy.

  Applies :data:`SCHEDULE_KEY_POLICY` mechanically: ``per_sample`` arms get one fresh key
  per sample via ``jax.random.split``, while ``run_level`` and ``unused`` arms are built
  once and broadcast. The result always carries a leading axis of length ``num_samples``,
  so a caller can ``vmap`` the decode over ``(key, wave_schedule)`` with ``in_axes=(0, 0)``
  regardless of which arm is in play -- the arm's semantics live in this function rather
  than in the caller's ``in_axes``.

  Args:
    mode: One of the five schedule arms.
    key: Run-level PRNG key. Split per sample for ``per_sample`` arms; used as-is
      otherwise.
    num_samples: Length of the leading sample axis to produce. Must be positive.
    tie_group_map: (L,) tie group id per position.
    coords: (L, 4, 3) backbone coordinates. Required by the coloring arms.
    mask: (L,) residue mask. Required by the coloring arms.
    k_neighbors: k for the coloring adjacency k-NN.
    structure_mapping: Optional per-position structure index for multistate inputs.
    ligand_coords: Optional (A, 3) ligand atom coordinates.
    ligand_mask: Optional (A,) ligand atom mask.

  Returns:
    A ``WaveScheduleBundle`` whose four fields each carry a leading ``num_samples`` axis.

  Raises:
    ValueError: If ``mode`` is unknown or ``num_samples`` is not positive.

  Notes:
    **This is a HOST-SIDE function and cannot be called under ``jax.jit``/``vmap``.**
    ``WaveScheduleBundle.from_tie_groups`` calls ``decoding_order.tolist()`` and derives
    ``num_waves`` from the data, so tracing it raises ``ConcretizationTypeError``. Build
    the stacked bundle here, outside any trace, then pass it in as a traced argument.

    Stacking is well defined because a permutation only reorders the tie groups, it does
    not change *which* groups exist -- so ``num_waves`` and every field shape are identical
    across samples for a fixed ``tie_group_map``. Verified in
    ``tests/inference/test_schedule_key_policy.py``.
  """
  policy = schedule_key_policy(mode)
  if num_samples <= 0:
    msg = f"num_samples must be positive, got {num_samples}"
    raise ValueError(msg)

  build_kwargs = {
    "tie_group_map": tie_group_map,
    "coords": coords,
    "mask": mask,
    "k_neighbors": k_neighbors,
    "structure_mapping": structure_mapping,
    "ligand_coords": ligand_coords,
    "ligand_mask": ligand_mask,
  }

  if policy == "per_sample":
    sample_keys = jax.random.split(key, num_samples)
    bundles = [build_wave_schedule(mode, key=k, **build_kwargs) for k in sample_keys]
  else:
    # One schedule, repeated. Built once so the (host-side, Python-loop-heavy) coloring
    # construction is not paid num_samples times for an identical result.
    shared = build_wave_schedule(mode, key=key, **build_kwargs)
    bundles = [shared] * num_samples

  return jax.tree.map(lambda *leaves: jnp.stack(leaves), *bundles)


def build_position_adjacency(
  coords: Float[Array, "L 4 3"],
  mask: Float[Array, " L"],
  *,
  k_neighbors: int,
  structure_mapping: Int[Array, " L"] | None = None,
  ligand_coords: Float[Array, "A 3"] | None = None,
  ligand_mask: Float[Array, " A"] | None = None,
  ligand_contact_radius: float = 6.0,
) -> list[set[int]]:
  """Build the coloring adjacency from the model's edge set (confound B3).

  Protein-protein edges use CA k-NN — the same metric `aminx.utils.wave_parallel`
  uses for its existing wave-coloring utility — masked by `mask` and (for
  multistate union graphs, W3.1) `structure_mapping`. Ligand-mediated edges
  connect any two positions that both have a CA within `ligand_contact_radius`
  of a (masked-valid) ligand atom: a conservative superset of the model's
  learned ligand-conditioning edges. This adjacency is an approximation of the
  model's true edge set — `assert_model_adj_subset` and W1b.2 (edge-set-
  containment deliberate-failure test) calibrate/validate it against the
  model's actual `neighbor_indices`.

  Returns:
    adjacency: length-L list of neighbor-index sets (undirected, symmetric).

  """
  seq_len = coords.shape[0]
  ca = coords[:, 1, :]  # (L, 3) — CA, per wave_parallel.py convention
  distances = jnp.linalg.norm(ca[:, None, :] - ca[None, :], axis=-1)
  distances_masked = jnp.where(
    (mask[:, None] * mask[None, :]).astype(jnp.bool_),
    distances,
    jnp.inf,
  )
  distances_masked = distances_masked.at[jnp.arange(seq_len), jnp.arange(seq_len)].set(jnp.inf)
  if structure_mapping is not None:
    same_structure = structure_mapping[:, None] == structure_mapping[None, :]
    distances_masked = jnp.where(same_structure, distances_masked, jnp.inf)

  k = min(k_neighbors, seq_len - 1) if seq_len > 1 else 0
  adjacency: list[set[int]] = [set() for _ in range(seq_len)]
  if k > 0:
    _, knn_idx = jax.lax.top_k(-distances_masked, k)
    knn_idx_np = np.asarray(knn_idx)
    finite = np.asarray(jnp.take_along_axis(distances_masked, knn_idx, axis=-1) < jnp.inf)
    for i in range(seq_len):
      for slot in range(k):
        if not finite[i, slot]:
          continue
        j = int(knn_idx_np[i, slot])
        adjacency[i].add(j)
        adjacency[j].add(i)

  if ligand_coords is not None and ligand_mask is not None and ligand_coords.shape[0] > 0:
    lig_dists = jnp.linalg.norm(ca[:, None, :] - ligand_coords[None, :, :], axis=-1)
    lig_dists = jnp.where(ligand_mask[None, :].astype(jnp.bool_), lig_dists, jnp.inf)
    in_contact = jnp.any(lig_dists < ligand_contact_radius, axis=-1) & mask.astype(jnp.bool_)
    contact_positions = np.flatnonzero(np.asarray(in_contact))
    for i in contact_positions:
      for j in contact_positions:
        if i != j:
          adjacency[int(i)].add(int(j))

  return adjacency


def assert_model_adj_subset(
  model_neighbor_indices: Int[Array, "L K"],
  coloring_adjacency: list[set[int]],
  mask: Float[Array, " L"] | None = None,
) -> None:
  """Assert `model_adj ⊆ coloring_adj` (confound C_knn_alignment, B3).

  Raises AssertionError naming the first model edge with no corresponding entry
  in `coloring_adjacency`.
  """
  neighbor_np = np.asarray(model_neighbor_indices)
  mask_np = np.asarray(mask) if mask is not None else None
  for i in range(neighbor_np.shape[0]):
    if mask_np is not None and mask_np[i] <= 0.5:
      continue
    for neighbor in neighbor_np[i].reshape(-1):
      j = int(neighbor)
      if j == i:
        continue
      if mask_np is not None and mask_np[j] <= 0.5:
        continue
      if j not in coloring_adjacency[i]:
        msg = (
          f"model_adj is not a subset of coloring_adj: model edge ({i}, {j}) "
          "has no corresponding coloring-adjacency entry (C_knn_alignment, B3)."
        )
        raise AssertionError(msg)


def color_positions(
  adjacency: list[set[int]],
  *,
  improper: bool = False,
  key: PRNGKeyArray | None = None,
) -> np.ndarray:
  """Greedy DSATUR-style graph coloring of `adjacency` (chromatic / improper-coloring arms).

  `improper=True` produces the W1a.1/B1 control arm: edges are relabeled through
  a random permutation of node identities before coloring, so the resulting
  color classes have the same degree-sequence "shape" as a proper coloring but
  are not actually independent sets of the *real* adjacency — deliberately
  violating the conditional-independence property chromatic scheduling relies
  on (information-ablation probe).
  """
  n = len(adjacency)
  coloring_adjacency = adjacency
  if improper:
    if key is None:
      msg = "improper=True requires a PRNG `key` to permute the adjacency."
      raise ValueError(msg)
    perm = np.asarray(jax.random.permutation(key, n))
    coloring_adjacency = [{int(perm[j]) for j in adjacency[int(perm[i])]} for i in range(n)]

  colors = np.full(n, -1, dtype=np.int32)
  order = sorted(range(n), key=lambda x: -len(coloring_adjacency[x]))
  for i in order:
    neighbor_colors = {colors[j] for j in coloring_adjacency[i] if colors[j] >= 0}
    c = 0
    while c in neighbor_colors:
      c += 1
    colors[i] = c
  return colors


def select_decoding_order(
  key: PRNGKeyArray | None,
  seq_len: int,
  mode: Literal["random_ar", "fixed_n_to_c", "frozen_random_sigma"],
) -> Int[Array, " L"]:
  """Decoding order for the three non-coloring schedule arms.

  `random_ar` and `frozen_random_sigma` both draw a random permutation; the
  distinction is caller discipline (seed protocol M6): `random_ar` must be
  called with a fresh `key` per sample, `frozen_random_sigma` with the *same*
  `key` reused across samples (sigma fixed) so its `schedule_variance`
  converges to the categorical floor rather than the permutation-noise floor.
  """
  if mode == "fixed_n_to_c":
    return jnp.arange(seq_len, dtype=jnp.int32)
  if key is None:
    msg = f"schedule mode {mode!r} requires a PRNG `key`."
    raise ValueError(msg)
  return jax.random.permutation(key, seq_len).astype(jnp.int32)


def build_wave_schedule(
  mode: DecodingSchedule,
  *,
  tie_group_map: Int[Array, " L"],
  key: PRNGKeyArray | None = None,
  coords: Float[Array, "L 4 3"] | None = None,
  mask: Float[Array, " L"] | None = None,
  k_neighbors: int = 48,
  structure_mapping: Int[Array, " L"] | None = None,
  ligand_coords: Float[Array, "A 3"] | None = None,
  ligand_mask: Float[Array, " A"] | None = None,
) -> WaveScheduleBundle:
  """Build a `WaveScheduleBundle` for one of the five schedule arms (W0.3).

  `chromatic` and `improper_coloring` require `coords`/`mask` (and optionally
  `structure_mapping`/`ligand_coords`/`ligand_mask`) to build the model-edge-set
  coloring adjacency (B3); the other three arms only need `tie_group_map`. See
  module docstring for the known decode-kernel gap affecting the coloring arms.
  """
  seq_len = tie_group_map.shape[-1]
  if mode in _ORDER_SCHEDULES:
    if schedule_consumes_key(mode) and key is None:
      msg = f"schedule mode {mode!r} requires a PRNG `key`."
      raise ValueError(msg)
    decoding_order = select_decoding_order(key, seq_len, mode)
    return WaveScheduleBundle.from_tie_groups(tie_group_map, decoding_order)

  if mode not in _COLORING_SCHEDULES:
    msg = f"Unknown schedule mode: {mode!r}"
    raise ValueError(msg)

  if coords is None or mask is None:
    msg = f"schedule mode {mode!r} requires `coords` and `mask` to build the coloring adjacency."
    raise ValueError(msg)

  adjacency = build_position_adjacency(
    coords,
    mask,
    k_neighbors=k_neighbors,
    structure_mapping=structure_mapping,
    ligand_coords=ligand_coords,
    ligand_mask=ligand_mask,
  )
  group_ids_sorted = sorted({int(g) for g in tie_group_map.tolist()})
  representative_position = {
    g: int(jnp.argmax((tie_group_map == g).astype(jnp.int32))) for g in group_ids_sorted
  }
  group_adjacency: list[set[int]] = []
  for g in group_ids_sorted:
    pos = representative_position[g]
    neighbor_positions = adjacency[pos]
    neighbor_groups = {int(tie_group_map[p]) for p in neighbor_positions}
    neighbor_groups.discard(g)
    group_adjacency.append({group_ids_sorted.index(ng) for ng in neighbor_groups})

  group_colors = color_positions(group_adjacency, improper=(mode == "improper_coloring"), key=key)
  return WaveScheduleBundle.from_colors(jnp.asarray(group_colors), tie_group_map)


__all__ = [
  "SCHEDULE_KEY_POLICY",
  "DecodingSchedule",
  "ScheduleKeyPolicy",
  "assert_model_adj_subset",
  "build_position_adjacency",
  "build_wave_schedule",
  "build_wave_schedule_per_sample",
  "color_positions",
  "schedule_consumes_key",
  "schedule_key_policy",
  "select_decoding_order",
]
