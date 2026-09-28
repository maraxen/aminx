"""Autoregression utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from collections.abc import Sequence

  from aminx.run.specs import RunSpecification
  from aminx.types.arrays import AutoRegressiveMask, DecodingOrder
  from aminx.types.bundles import WaveScheduleBundle
  from aminx.utils.data_structures import Protein


def get_decoding_step_map(
  tie_group_map: jnp.ndarray,
  group_decoding_order: jnp.ndarray,
  num_groups: int | None = None,
) -> jnp.ndarray:
  """Map each residue to its decoding step index based on group order."""
  # Use N as a safe upper bound for num_groups if not provided
  N = tie_group_map.shape[0]
  M = group_decoding_order.shape[0]

  group_to_step = jnp.zeros(N, dtype=jnp.int32).at[group_decoding_order].set(jnp.arange(M))
  return group_to_step[tie_group_map]


def make_autoregressive_mask(decoding_step_map: jnp.ndarray) -> jnp.ndarray:
  """Create an (N, N) AR mask for group-based decoding."""
  steps_i = decoding_step_map[:, None]
  steps_j = decoding_step_map[None, :]
  return steps_i >= steps_j


def resolve_tie_groups(
  spec: RunSpecification,
  combined_protein: Protein,
  structure_mappings: Sequence[dict] | None = None,
  num_structures: int | None = None,
) -> jnp.ndarray:
  """Resolve tie groups for tied_positions modes.

  Args:
      spec: RunSpecification with tied_positions.
      combined_protein: Protein dataclass with batch_dim=1 (1, seq_len, ...).
      structure_mappings: Optional, for 'auto' mode.
      num_structures: Optional, number of structures concatenated (for 'direct' mode).

  Returns:
      tie_group_map: jnp.ndarray of shape (n,) with group ids.

  """
  chain_ids = combined_protein.chain_index[0]
  residue_indices = combined_protein.residue_index[0]
  n = chain_ids.shape[0]

  tie_group_map = jnp.arange(n, dtype=jnp.int32)
  tied_positions = spec.tied_positions

  if tied_positions is None:
    return tie_group_map

  if tied_positions == "direct":
    if num_structures is None:
      if combined_protein.mapping is not None:
        structure_indices = combined_protein.mapping[0]  # Remove batch dim
        num_inputs = int(jnp.max(structure_indices)) + 1
      else:
        msg = (
          "Cannot determine number of structures for 'direct' mode. "
          "The concatenated protein should have a 'mapping' field with structure indices, "
          "or pass num_structures explicitly."
        )
        raise ValueError(msg)
    else:
      num_inputs = num_structures

    ll = n // num_inputs
    if n % ll != 0:
      msg = (
        f"Inputs must be same length for 'direct' mode. "
        f"Total length {n} is not divisible by {num_inputs} structures."
      )
      raise ValueError(msg)
    k = n // ll
    return jnp.tile(jnp.arange(ll, dtype=jnp.int32), k)

  if tied_positions == "auto":
    if structure_mappings is None:
      msg = "structure_mappings required for 'auto' mode."
      raise ValueError(msg)
    for seq_pos, struct_pos_list in enumerate(structure_mappings):
      if len(struct_pos_list) > 1:
        group_id = n + seq_pos
        tie_group_map = tie_group_map.at[jnp.array(struct_pos_list)].set(group_id)
    _, tie_group_map = jnp.unique(tie_group_map, return_inverse=True)
    return tie_group_map

  def _collect_group_indices(
    groups: Sequence[tuple[int, int]],
    chain_id_arr: jnp.ndarray,
    residue_idx_arr: jnp.ndarray,
  ) -> list[tuple[int, list[int]]]:
    group_map = defaultdict(list)
    for group_idx, group in enumerate(groups):
      if isinstance(group[0], (list | tuple)):
        for tup in group:
          group_map[group_idx].append(tup)
      else:
        group_map[group_idx].append(group)
    group_indices = []
    for group_idx, tuples in group_map.items():
      indices = []
      for chain_idx, res_idx in tuples:
        mask = (chain_id_arr == chain_idx) & (residue_idx_arr == res_idx)
        idx = jnp.where(mask)[0]
        if idx.size > 0:
          indices.append(idx[0])
      group_indices.append((group_idx, indices))
    return group_indices

  group_indices = _collect_group_indices(
    tied_positions,
    chain_ids,
    residue_indices,
  )
  for group_idx, indices in group_indices:
    if indices:
      group_id = n + group_idx
      tie_group_map = tie_group_map.at[jnp.array(indices)].set(group_id)
  _, tie_group_map = jnp.unique(tie_group_map, return_inverse=True)
  return tie_group_map


@jax.jit
def generate_ar_mask(
  decoding_order: DecodingOrder,
  chain_idx: jnp.ndarray | None = None,
  tie_group_map: jnp.ndarray | None = None,
  num_groups: int | None = None,
) -> AutoRegressiveMask:
  """Get the self-excluding autoregressive mask for the given decoding order.

  **Argument convention -- read before passing a deliberate order.** The untied branch
  compares ``decoding_order`` VALUES as a RANK array (``decoding_order[i]`` = the decode
  step of position ``i``), whereas everything in ``utils.decoding_order`` returns an ORDER
  array (``order[t]`` = the position decoded at step ``t``); the tied branch indexes it as an
  ORDER array. For a uniformly random permutation the two readings are identically
  distributed, so the mix-up is invisible until someone supplies a structured order (blocks,
  a rotation, a counterfactual schedule), which silently yields the mask of a different
  order. ColabDesign applies ``order.argsort()`` first for this reason. To turn an ORDER
  array into a mask use :func:`ar_mask_from_decoding_order` (#166).

  **The diagonal is 0: no position ever reads its own slot.** This matches both
  independent reference implementations, verified against primary source:

  * stock LigandMPNN (``dauparas/LigandMPNN``, ``model_utils.py:227-231`` and four further
    call sites) builds ``1 - torch.triu(torch.ones(L, L))``. ``torch.triu`` defaults to
    ``diagonal=0``, so this is strictly lower-triangular, and conjugating it by a
    permutation matrix preserves the zero diagonal for any decoding order.
  * ColabDesign (``colabdesign/mpnn/utils.py:19-26``) builds ``jnp.tri(L, k=-1)`` and
    permutes it by ``order.argsort()``.

  Until 2026-08-27 the untied branch here was ``(row_indices >= col_indices)`` --
  non-strict -- so the diagonal was 1 and every position read its own slot. The rationale
  given was that a not-yet-drawn position's slot holds an inert placeholder. That premise
  did not hold: ``inference/decode/autoregressive.py`` initialized undrawn slots to token
  index 0, and ``MPNN_ALPHABET`` is ``"ACDEFGHIKLMNPQRSTVWYX"``, so index 0 is ALANINE
  (the unknown token X is index 20). ``model/decoder.py:124`` embeds it as
  ``one_hot_sequence @ w_s_weight`` -- a real, nonzero row of a pretrained matrix. Since
  the decoder gathers this mask into ``attention_mask`` to gate the *sequence* edge
  features (``model/decoder.py:144-146``) and a residue is always among its own KNN
  neighbours (self-distance 0 is the global minimum), each position was being fed the
  assertion "I am alanine" about itself at the exact step it decided its own identity.
  Measured downstream: mean JSD 0.0296 over designable positions, with an
  alanine-specific probability shift of -0.0049 against a mean of 0.0016 across the other
  twenty tokens.

  The undrawn sentinel is now ``-1``, whose one-hot is the zero vector -- matching the
  reference's ``h_S = torch.zeros_like(h_V)``. Self-exclusion alone already makes undrawn
  slots unreachable (a causal mask hides them from every other position), so the sentinel
  is defence in depth, and it makes "not yet drawn" distinguishable from "alanine" in
  stored sequences.

  Tie semantics are deliberately UNCHANGED: same-tie-group positions remain mutually
  visible. Whether that matches the reference's tied decoding is a separate, open
  question -- it is not the diagonal, and it is not settled here.

  For full context minus self (non-causal; correct for scoring a known sequence, wrong for
  an autoregressive decode) use :func:`full_context_ar_mask`.
  """
  N = decoding_order.shape[0]

  if tie_group_map is None:
    row_indices = decoding_order[:, None]
    col_indices = decoding_order[None, :]
    ar_mask = (row_indices >= col_indices).astype(jnp.int32)
  else:
    # Use N as the static size for range-based ops
    # group_mask: (N, N)
    group_mask = tie_group_map[decoding_order][None, :] == jnp.arange(N)[:, None]

    # Identify which groups are actually present
    group_present = jnp.any(group_mask, axis=1)

    # group_first_occurrence: (N,)
    group_first_occurrence = jnp.argmax(group_mask, axis=1)

    # Sort groups by their first occurrence in the decoding order
    # We only care about present groups
    #
    # IREE does not honour JAX's stable-sort tie order (xtrax export safety rule
    # `sort-stability`), and `jnp.argsort` defaults to `stable=True`. Every absent
    # group shares the SAME sentinel key (N + 1) here, so ties are the common
    # case, not an edge case. Fold the group index into the sort key instead of
    # relying on the default's tie order -- same fix as `model.features.top_k`
    # (PR #156) and `utils.decoding_order.random_decoding_order`: sort
    # lexicographically on (key, index) with `num_keys=2`, a strict total order
    # (no two indices are equal), so every correct sort returns this permutation
    # whether or not it is stable.
    group_sort_key = jnp.where(group_present, group_first_occurrence, N + 1)
    group_index = jax.lax.broadcasted_iota(jnp.int32, group_sort_key.shape, 0)
    _, group_decoding_order = jax.lax.sort(
      (group_sort_key, group_index),
      dimension=0,
      is_stable=False,
      num_keys=2,
    )

    # If num_groups is provided, we can use it to mask the decoding steps
    # but for now, we just use the full order found.
    # The decoding_step_map will only be indexed by tie_group_map.

    decoding_step_map = get_decoding_step_map(tie_group_map, group_decoding_order)
    ar_mask = make_autoregressive_mask(decoding_step_map).astype(jnp.int32)

  if chain_idx is not None:
    same_chain = (chain_idx[:, None] == chain_idx[None, :]).astype(jnp.int32)
    ar_mask = ar_mask * same_chain

  # Self-exclusion, applied once for BOTH branches. For the untied branch this is exactly
  # equivalent to making the comparison strict -- `decoding_order` is a permutation, so
  # `decoding_order[i] == decoding_order[j]` only when i == j -- and it leaves the tied
  # branch's same-group mutual visibility untouched, which is the deliberate scope
  # boundary described in the docstring.
  return ar_mask * (1 - jnp.eye(N, dtype=ar_mask.dtype))


def _position_wave_index(wave: WaveScheduleBundle, tie_group_map: jnp.ndarray) -> jnp.ndarray:
  """(L,) index of the wave each position decodes in -- the one rule every consumer shares.

  Both :func:`generate_wave_ar_mask` (what a position may see) and
  :func:`decoding_order_from_wave` (the order that is reported) are functions of this, so
  they cannot disagree about when a position is decoded.

  A position's wave is the first wave its tie group appears in. Positions whose group never
  appears (a *partial* schedule -- e.g. a caller that only schedules the positions it
  actually needs) get the sentinel ``num_waves``. The sentinel must be LARGER than any real
  wave index, not smaller: the mask compares via ``>``/``==``, so a too-small sentinel (e.g.
  -1) would make every omitted position look "earlier than everything" and leak its (never
  decoded, all-zero) value as false context. A too-large sentinel makes omitted positions
  "infinitely late" -- invisible to every real position and to each other.
  """
  seq_len = tie_group_map.shape[0]
  num_waves, max_groups_per_wave = wave.group_ids.shape
  never_scheduled_sentinel = num_waves
  wave_index_grid = jnp.broadcast_to(
    jnp.arange(num_waves, dtype=jnp.int32)[:, None],
    (num_waves, max_groups_per_wave),
  )
  flat_group_ids = jnp.where(wave.group_valid, wave.group_ids, 0).reshape(-1)
  flat_wave_index = jnp.where(
    wave.group_valid,
    wave_index_grid,
    never_scheduled_sentinel,
  ).reshape(-1)
  # Scatter-min: each group id ends up mapped to the (single, real) wave index it was
  # assigned to; invalid (padding) entries carry the sentinel and never win over a real
  # (smaller) wave index.
  group_wave_index = (
    jnp.full((seq_len,), never_scheduled_sentinel, dtype=jnp.int32)
    .at[flat_group_ids]
    .min(flat_wave_index)
  )
  return group_wave_index[tie_group_map]


@jax.jit
def generate_wave_ar_mask(
  wave: WaveScheduleBundle,
  tie_group_map: jnp.ndarray,
) -> AutoRegressiveMask:
  """Get the autoregressive mask for a `WaveScheduleBundle` (wave-color scheduling, W0.3+).

  Generalizes `generate_ar_mask` from a flat total-order `decoding_order` to a
  `WaveScheduleBundle` that may pack multiple (conditionally independent) tie
  groups into the same wave (chromatic/improper-coloring arms, `G > 1`).
  Position `i` is visible to position `j` iff `j`'s wave is strictly earlier
  than `i`'s, OR `i` and `j` are in the same tie group (same wave, same group —
  tied positions always see each other, matching `generate_ar_mask`). Positions
  in the *same* wave but *different* groups do NOT see each other — this is the
  Jacobi-within-a-color independence the chromatic schedule relies on.

  For a `G == 1` schedule (`WaveScheduleBundle.from_tie_groups`/`.empty`, i.e.
  every existing non-coloring arm), this reduces exactly to `generate_ar_mask`'s
  semantics: each wave has a unique rank, so "same wave" implies "same group".

  Parameters
  ----------
  wave : WaveScheduleBundle
      Wave/group schedule, shape (W waves, G groups-per-wave, P positions).
  tie_group_map : Int[Array, "L"]
      Tie group id per position (state-0 convention, matching the rest of the
      AR decode kernel).

  Returns
  -------
  AutoRegressiveMask
      Shape (L, L); 1 where position i can see position j's decoded sequence.

  """
  seq_len = tie_group_map.shape[0]
  position_wave_index = _position_wave_index(wave, tie_group_map)

  same_wave = position_wave_index[:, None] == position_wave_index[None, :]
  earlier_wave = position_wave_index[:, None] > position_wave_index[None, :]
  same_group = tie_group_map[:, None] == tie_group_map[None, :]

  mask = (earlier_wave | (same_wave & same_group)).astype(jnp.float32)
  # Self-exclusion, matching :func:`generate_ar_mask` and both reference implementations.
  # `same_wave & same_group` is trivially true on the diagonal, so without this every
  # position read its own slot -- see generate_ar_mask's docstring for why that is wrong
  # and what it measured. Only the diagonal is removed: same-wave/same-group mutual
  # visibility (and the same-wave/different-group invisibility that gives this schedule
  # its Jacobi independence) are both preserved exactly.
  return mask * (1 - jnp.eye(seq_len, dtype=mask.dtype))


def ar_mask_from_decoding_order(
  decoding_order: DecodingOrder,
  tie_group_map: jnp.ndarray | None = None,
) -> AutoRegressiveMask:
  """Causal, self-excluding mask for an ORDER array (``decoding_order[t]`` = position at step t).

  This is the mask to pair with what ``utils.decoding_order`` functions return. It is built
  through the same schedule the sampler decodes with (``WaveScheduleBundle.from_decoding_order``
  then :func:`generate_wave_ar_mask`), so a mask built here and a wave built from the same
  order cannot disagree.

  Do not pass an order to :func:`generate_ar_mask` directly: its untied branch compares
  ``decoding_order`` values as a RANK array, while its tied branch indexes by it as an ORDER
  array. For a uniformly random permutation the mix-up is invisible in distribution, which
  is why it survived; for any deliberate order (a custom ``decoding_order_fn``, a
  counterfactual schedule) it silently yields a mask for a different order (debt #1982).

  Args:
    decoding_order: (L,) ORDER array.
    tie_group_map: Optional (L,) tie group id per position. Tied positions are mutually
      visible and decode at their earliest member's step, as in :func:`generate_ar_mask`.

  Returns:
    (L, L) float32 mask; ``mask[i, j] == 1`` iff position ``i`` sees position ``j``.
  """
  from aminx.types.bundles import WaveScheduleBundle  # noqa: PLC0415 -- type-only at module level

  seq_len = decoding_order.shape[0]
  groups = jnp.arange(seq_len, dtype=jnp.int32) if tie_group_map is None else tie_group_map
  wave = WaveScheduleBundle.from_decoding_order(jnp.asarray(decoding_order), tie_group_map)
  return generate_wave_ar_mask(wave, groups)


def decoding_order_from_wave(
  wave: WaveScheduleBundle,
  tie_group_map: jnp.ndarray,
) -> jax.Array:
  """The ORDER array a `WaveScheduleBundle` actually decodes (jit/vmap-safe).

  Positions sorted by the wave their tie group first appears in; positions that share a
  wave (a tie group, or conditionally independent groups of a chromatic wave) are ordered
  by position, and positions the schedule never reaches come last. For a wave built by
  ``WaveScheduleBundle.from_decoding_order(order, tie_group_map)`` this recovers ``order``
  exactly when untied, and ``random_decoding_order``'s (step, position) convention when
  tied.

  Args:
    wave: The schedule.
    tie_group_map: (L,) tie group id per position (state-0 convention).

  Returns:
    (L,) int32 ORDER array.
  """
  seq_len = tie_group_map.shape[0]
  position_wave_index = _position_wave_index(wave, tie_group_map)
  positions = jnp.arange(seq_len, dtype=jnp.int32)
  # Two explicit keys, (wave, position): a strict total order, so the result does not depend
  # on sort stability (IREE's sort is not stable; see utils/decoding_order.py).
  _, order = jax.lax.sort(
    (position_wave_index, positions),
    dimension=0,
    is_stable=False,
    num_keys=2,
  )
  return jnp.asarray(order, dtype=jnp.int32)


def full_context_ar_mask(seq_len: int) -> jnp.ndarray:
  """Every position sees every other position's sequence, but not its own: ``1 - I``.

  ``ar_mask[i, j] == 1`` means position ``i`` SEES position ``j``'s sequence. The decoder
  gathers it into ``attention_mask`` and uses it as ``mask_bw`` to gate the *sequence* edge
  features, while ``1 - attention_mask`` gates the structure-only path
  (``model/decoder.py:144-147``). **An all-zero ``ar_mask`` therefore admits no sequence
  information at all** and reduces any "conditional" quantity to a function of structure
  alone.

  This lives here, public and next to the other mask builders, because that mistake has now
  been made at four separate sites, each with a comment or docstring asserting the opposite:

  - ``forward_jac`` -- made the categorical Jacobian identically zero for its whole
    existence, fixed in 40f7edfc. Its comment read "fully conditional / no autoregressive
    masking, every position sees every other", which is precisely backwards.
  - the runner's reverse-mode path -- same construct, fixed in 40f7edfc. Not catastrophic
    there (the score reads ``one_hot`` directly, so context loss shifts the gradient by
    ~0.16% rather than annihilating it) but wrong for a mutation-effect estimate.
  - ``sampling.conditional_logits``'s split ``decode_fn`` default, and therefore the
    runner's batched ``conditional_logits`` feature, which relied on it (#4222).
  - the runner's ``decoded_node_features`` feature (#4204).

  - ``scoring/score.py`` -- a FIFTH site, and a different variant: not an all-zero mask but
    a self-*inclusive* one, ``generate_ar_mask(decoding_order)``, whose diagonal is 1. That
    leaked each residue's own identity into its own prediction (+0.036243 nats on 1LVB
    chain A, paired over 8 seeds, t = 41.3) and made the score depend on the PRNG key at
    ``backbone_noise=0``, since a fresh order meant a fresh mask.

  Every one of those failures was silent: right shape, right dtype, no error, no warning.
  Note the two failure modes are opposites -- too little context and too much -- so a test
  that only checks "the mask is not all zeros" catches one and misses the other.

  ``1 - I`` matches the "full context minus self" default that
  ``inference/bundle_builder.py:227-228`` uses for ``mode="score_conditional"``, so a
  teacher-forced conditional built with this mask agrees with the full
  ``make_conditional_logits_fn`` path. Self-exclusion is wanted independently: a position's
  dependence on its own token is self-dependence, not a coupling.

  An all-zero mask is still a legitimate thing to ask for -- it is what "unconditional"
  means -- but it must be passed **explicitly**, never arrived at by omission.

  Args:
    seq_len: Number of positions.

  Returns:
    ``(seq_len, seq_len)`` float32 mask, ones off the diagonal and zeros on it.
  """
  return jnp.ones((seq_len, seq_len), dtype=jnp.float32) - jnp.eye(seq_len, dtype=jnp.float32)
