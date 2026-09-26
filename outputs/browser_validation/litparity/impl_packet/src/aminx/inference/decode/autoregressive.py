"""AutoregressiveDecode mode class (Task 9).

Autoregressive sampling kernel that iterates through a wave schedule (sequence of
tied position groups), carrying the evolving sequence through the waves via a
ScanIterator. The wave-axis carry is reified as a CarryShape metadata struct
(Risk D-10 mitigation).

After the wave scan completes, a post-hoc scatter scan maps per-wave logits back
to per-position logits (Risk D-11 mitigation) — this stays outside the iterator.

Key invariants:
- The wave_iterator field is always xtrax.tiling.JaxScanIterator (structural
  invariant; no user-facing W-axis strategy knob per Risk D-3).
- The state_iterator is injected at factory time, controlling S-axis parallelism.
- CarryShape is metadata-only; the actual init-array is materialized inside
  __call__ from the metadata.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.lax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from xtrax.tiling import CarryShape, MapIterator, ScanIterator

from aminx.inference.decode._kernel import (
  _decode_one_step,
  _project_logits,
  _realign_states_to_reference,
)
from aminx.inference.sample_autoregressive import SampleResult
from aminx.types.bundles import EncoderOutput, InferenceBundle, WaveScheduleBundle
from aminx.types.configs import InferenceConfig
from aminx.types.stages import StageSet

# Type alias for decoding order function
DecodingOrderFn = Callable[[WaveScheduleBundle], Any]

#: Sentinel token for a position that has not been drawn yet.
#:
#: Deliberately NOT 0. ``MPNN_ALPHABET`` is ``"ACDEFGHIKLMNPQRSTVWYX"``, so index 0 is
#: alanine and index 20 is the unknown token X -- neither expresses "no draw yet".
#: ``jax.nn.one_hot(-1, 21)`` is the all-zero vector, so a -1 slot contributes no sequence
#: signal to the decoder, exactly as the reference does with ``h_S = torch.zeros_like(h_V)``.
#: Using a distinct negative value also makes "undrawn" recoverable from a stored
#: mid-decode sequence, which index 0 could not express.
#:
#: Any position still holding this value after a completed decode was never scheduled.
UNDRAWN_TOKEN = -1


def _fuse_one_group(
  logits: jnp.ndarray,
  mask_group: jnp.ndarray,
  tie_group_fuse: Callable | None,
) -> jnp.ndarray:
  """Average per-position `logits` (L, 21) over one group's boolean mask (L,).

  Returns zeros (not -inf/nan) for an all-False mask (inactive/padded group
  slot), so callers can safely combine results across the G axis (e.g. via
  einsum) without NaN propagation from 0 * (-inf).
  """
  if tie_group_fuse is not None:
    fused = tie_group_fuse(logits, mask_group).reshape((21,))
    return jnp.where(jnp.any(mask_group), fused, 0.0)

  masked = jnp.where(mask_group[:, None], logits, -jnp.inf)
  n_tied = jnp.sum(mask_group)
  avg = jax.scipy.special.logsumexp(masked, axis=0) - jnp.log(jnp.maximum(n_tied, 1))
  return jnp.where(n_tied > 0, avg, 0.0)


class AutoregressiveDecode(eqx.Module):
  """Autoregressive decode mode: carry-bearing wave iteration with post-hoc scatter.

  Combines a MapIterator (for state-axis stateless iteration) and a ScanIterator
  (for wave-axis carry-bearing iteration). The wave carry is the evolving sequence,
  reified as a CarryShape metadata struct to defer materialization to __call__.

  After the main wave scan, a post-hoc scatter scan (outside the iterator) maps
  per-wave logits back to per-position logits, preserving the two-scan structure
  from driver.py:decode_ar (Risk D-11).

  Parameters
  ----------
  model : Any
      Model instance with decoder and w_out attributes.
  decoding_order_fn : Callable
      Function (wave_schedule) -> decoding_order. Determines the order in which
      positions are decoded within the wave schedule.
  state_iterator : MapIterator
      Iterator for the S axis (VmapIterator, SafeMapIterator, etc.).
      Injected at factory time; determines parallelism strategy.
  wave_iterator : ScanIterator
      Iterator for the W axis (always JaxScanIterator; field exists for
      type-symmetry only; structural invariant per Risk D-3).
  wave_carry : CarryShape
      Metadata for the wave-axis carry (name="sequence", shape=(L,),
      dtype=jnp.int32). The actual init-array is materialized inside __call__.

  Attributes
  ----------
  model : Any
      The MPNN model. A dynamic field: its weight arrays are traced JAX
      leaves so filter_jit partitions them as runtime inputs (not hashed
      static constants).
  decoding_order_fn : Callable = eqx.field(static=True)
      Decoding order function is static.
  state_iterator : MapIterator
      State axis iterator (injected at factory time).
  wave_iterator : ScanIterator
      Wave axis iterator (JaxScanIterator). Used when use_while_loop=False.
  wave_carry : CarryShape = eqx.field(static=True)
      Metadata for carry shape (name, shape, dtype; no value).
  use_while_loop : bool = eqx.field(static=True)
      When True, wave axis uses ``jax.lax.while_loop`` instead of
      ``jax.lax.scan``.  Lowers to a single XLA WhileOp — faster to compile
      than a Scan op, especially for large n_waves.
      **Not reverse-mode differentiable.** Set via
      ``make_decode_fn(..., autoregressive_config=AutoregressiveConfig(inference_only=True))``
      (see ``aminx.inference.decode.factory.make_decode_fn`` and
      ``aminx.inference.decode.mode.AutoregressiveConfig``); never set True in
      training code paths.

  Notes
  -----
  This class implements Pattern 5 (injection): both state_iterator and
  wave_iterator are injected dependencies. Different combinations allow
  flexible composition of S-axis and W-axis iteration strategies without
  code duplication.

  The post-hoc scatter scan is performed inside __call__, not inside the
  wave_iterator. This preserves the two-scan structure from driver.py:decode_ar:
  1. Wave scan (inside iterator): carries sequence, outputs per-wave logits.
  2. Scatter scan (post-hoc, outside iterator): maps per-wave to per-position logits.

  When use_while_loop=True the wave scan uses lax.while_loop; carry bundles
  (step, sequence, logits_stack) so no external output stacking is needed.
  wave_iterator is retained in the pytree but unused on this path.

  """

  model: Any
  decoding_order_fn: DecodingOrderFn = eqx.field(static=True)
  state_iterator: MapIterator
  wave_iterator: ScanIterator
  wave_carry: CarryShape = eqx.field(static=True)
  use_while_loop: bool = eqx.field(static=True, default=False)

  def __call__(
    self,
    key: PRNGKeyArray,
    enc: EncoderOutput,
    bundle: InferenceBundle,
    config: InferenceConfig,
    stage_set: StageSet,
  ) -> SampleResult:
    """Autoregressive decode: carry sequence through wave scan, then scatter logits.

    Parameters
    ----------
    key : PRNGKeyArray
        PRNG key for sampling randomness.
    enc : EncoderOutput
        Encoder output. Shape: node (S, L, H_n), edge (S, L, K, H_e).
    bundle : InferenceBundle
        Inference bundle with conditioning (ar_mask, tie_group_map, fixed_mask,
        fixed_tokens, temperature, bias) and wave schedule.
    config : InferenceConfig
        Inference configuration.
    stage_set : StageSet
        Pipeline stages with ar_logit_transform, decode_step,
        logit_transform, tie_group_fuse.

    Returns
    -------
    SampleResult
        Result with sequence (L,) and logits (L, 21).

    Notes
    -----
    Workflow:
    1. Materialize init sequence from wave_carry metadata.
    2. Compute decoding order via decoding_order_fn.
    3. Build scan_body that:
       a. For each state, call _decode_one_step via state_iterator.
       b. Project to logits.
       c. Fuse per-position logits via ar_logit_transform.
       d. Average across tied positions via tie_group_fuse.
       e. Sample from averaged logits.
       f. Update sequence for sampled positions.
       g. Return (new_sequence, per_wave_logits).
    4. Call wave_iterator(scan_body, init, jnp.arange(n_waves)).
    5. Post-hoc scatter: map (n_waves, V) logits to (L, V) via second scan.
    6. Return SampleResult(final_sequence, logits).

    """
    L = enc.node_features.shape[1]
    S = enc.node_features.shape[0]
    cond = bundle.conditioning
    wave = bundle.wave
    n_waves, max_groups_per_wave = wave.group_ids.shape

    # 1. Materialize init sequence from metadata with actual length L
    # UNDRAWN SENTINEL. Not 0: `MPNN_ALPHABET` is "ACDEFGHIKLMNPQRSTVWYX", so index 0 is
    # ALANINE (the unknown token X is index 20), and `model/decoder.py:124` embeds it as
    # `one_hot_sequence @ w_s_weight` -- a real, nonzero row of a pretrained matrix. A
    # zero-initialized slot therefore asserted "alanine here", not "nothing here".
    #
    # `jax.nn.one_hot(-1, 21)` is the all-zero vector, so -1 reproduces the reference's
    # construction exactly (`h_S = torch.zeros_like(h_V)`, LigandMPNN model_utils.py:252):
    # an undrawn position contributes no sequence signal at all.
    #
    # This is defence in depth rather than the primary fix. A self-excluding causal
    # ar_mask already makes undrawn slots unreachable -- a position cannot see itself, and
    # every position that decodes later is masked out for everyone earlier. The sentinel
    # additionally makes "not yet drawn" distinguishable from "alanine" in any sequence
    # that is stored or inspected mid-decode, which index 0 could not express.
    # UNDRAWN_TOKEN is negative, so the carry dtype must be signed. `CarryShape.dtype` is
    # typed `Any` and enforces nothing; every construction site passes jnp.int32 today
    # (`decode/factory.py`), so this asserts an invariant that currently holds by caller
    # convention rather than by type. Under an unsigned dtype -1 would wrap to a large
    # positive value -- `jax.nn.one_hot` would still yield the zero vector (it zeroes ANY
    # out-of-range index), so the embedding stays correct, but every `seq < 0` check
    # downstream would silently stop detecting undecided positions.
    if not jnp.issubdtype(self.wave_carry.dtype, jnp.signedinteger):
      msg = (
        f"wave_carry dtype {self.wave_carry.dtype} is not a signed integer type, so "
        f"UNDRAWN_TOKEN ({UNDRAWN_TOKEN}) cannot be represented. Undecided positions would "
        "wrap to a large positive value and stop being detectable by a negativity check."
      )
      raise TypeError(msg)
    init_sequence = jnp.full((L,), UNDRAWN_TOKEN, dtype=self.wave_carry.dtype)

    # Initialize with fixed positions
    init_sequence = jnp.where(
      cond.fixed_mask > 0.5,
      cond.fixed_tokens,
      init_sequence,
    ).astype(jnp.int32)

    # 1b. Precompute, once, the first (wave, slot) occurrence of every REAL tie
    # group (looked up via cond.tie_group_map, not wave.group_ids -- those only
    # coincide for from_tie_groups/from_colors-built schedules; WaveScheduleBundle.
    # empty() assigns one group per position regardless of ties, so a tie group
    # with >1 member position appears at >1 (wave, slot) under empty() and must
    # only be sampled once). For from_tie_groups/from_colors schedules every
    # real group already appears exactly once, so this is a no-op there.
    pos0_grid = wave.group_positions[:, :, 0]  # (W, G)
    real_group_id_grid = cond.tie_group_map[0, pos0_grid]  # (W, G)
    wave_index_grid = jnp.broadcast_to(jnp.arange(n_waves, dtype=jnp.int32)[:, None], (n_waves, max_groups_per_wave))
    slot_grid = jnp.broadcast_to(jnp.arange(max_groups_per_wave, dtype=jnp.int32)[None, :], (n_waves, max_groups_per_wave))
    combined_rank_grid = wave_index_grid * max_groups_per_wave + slot_grid
    no_occurrence_sentinel = n_waves * max_groups_per_wave
    flat_group_id = jnp.where(wave.group_valid, real_group_id_grid, 0).reshape(-1)
    flat_rank = jnp.where(wave.group_valid, combined_rank_grid, no_occurrence_sentinel).reshape(-1)
    group_first_rank = jnp.full((L,), no_occurrence_sentinel, dtype=jnp.int32).at[flat_group_id].min(flat_rank)

    # 2. Build step function that processes one wave at a time. A wave may pack
    # multiple (conditionally independent, for a proper coloring) tie groups
    # into its G group slots -- all groups in an active wave are decoded from
    # the SAME shared forward pass (computed once below) and sampled/updated
    # in parallel (Jacobi-within-a-color), not sequentially.
    def step_fn(sequence: jnp.ndarray, wave_idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
      """Process one wave: decode, fuse, sample, update sequence for all its groups.

      Parameters
      ----------
      sequence : array (L,)
          Current sequence (evolving carry).
      wave_idx : scalar int
          Index into the wave schedule.

      Returns
      -------
      (new_sequence, step_logits) : (array (L,), array (G, 21))
          Updated sequence and per-group-slot logits for this wave (zeros for
          inactive/padded group slots).

      """
      # Group identities for every slot in this wave (garbage/0 where inactive).
      pos0 = wave.group_positions[wave_idx, :, 0]  # (G,)
      group_id = cond.tie_group_map[0, pos0]  # (G,)
      is_active = wave.group_valid[wave_idx].astype(jnp.bool_)  # (G,)
      this_rank = wave_idx * max_groups_per_wave + jnp.arange(max_groups_per_wave, dtype=jnp.int32)  # (G,)
      is_first_occurrence = is_active & (group_first_rank[group_id] == this_rank)  # (G,)
      mask_group = (cond.tie_group_map[0][None, :] == group_id[:, None]) & is_first_occurrence[:, None]  # (G, L)
      wave_has_active_group = jnp.any(is_first_occurrence)

      def do_sample(seq: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Decode (once, shared across groups), fuse/sample/update per group."""
        # One-hot encode sequence for all S states
        seq_oh = jax.nn.one_hot(seq, 21)  # (L, 21)
        seq_oh_stack = jnp.broadcast_to(
          seq_oh[None, ...],
          (S, L, 21),
        )  # (S, L, 21)

        # Per-state inputs
        per_state_inputs = (
          enc.node_features,
          enc.edge_features,
          enc.neighbor_indices,
          enc.mask,
          cond.ar_mask,
          seq_oh_stack,
        )

        # Per-state decode closure
        def per_state_fn(inputs):
          node_features, edge_features, neighbor_indices, mask, ar_mask, seq_oh = inputs
          return _decode_one_step(
            model=self.model,
            node_features=node_features,
            edge_features=edge_features,
            neighbor_indices=neighbor_indices,
            mask=mask,
            ar_mask=ar_mask,
            sequence_oh=seq_oh,
            key=key,
            inference=config.inference,
            decode_step=stage_set.decode_step,
          )

        # Apply state_iterator to iterate over S axis
        decoded = self.state_iterator(per_state_fn, per_state_inputs, in_axes=0)

        # Project to logits: (S, L, H) -> (S, L, 21)
        logits = _project_logits(self.model, decoded)

        # Realign states to the shared reference frame before cross-state fusion
        # (identity state_position_map is a no-op -- pre-fix behavior). See
        # ConditioningBundle.state_position_map / _kernel._realign_states_to_reference.
        logits = _realign_states_to_reference(logits, cond.state_position_map)

        # Fuse per-position logits across states with and without bias
        # For stored logits: bias-free.
        # F004 (task_id `260826_aminx-invariant-audit`): bias is consumed JOINTLY
        # with the logits here, so its length must come from the SAME geometry
        # source as the logits -- the post-realign reference frame, which can be
        # narrower than ``cond.bias``'s padded-chain length whenever
        # state_position_map is narrower than seq_len. Keying these defaults to
        # ``cond.bias`` verbatim crashed with a vmap size mismatch whenever the
        # two disagreed (bundle_builder keyed the synthesized default to padded
        # geometry while this site consumed realigned geometry). When the
        # lengths already agree (max_length == chain length) nothing below moves
        # a byte: both branches reduce to the pre-fix arrays exactly.
        cond_bias = cond.bias
        if cond_bias.shape[-2] != logits.shape[-2]:
          # Gather bias into the reference frame through the map's canonical row
          # (state 0 defines the reference numbering); gap positions (-1) get
          # zero bias -- the additive neutral element for a logit bias.
          spm_row = cond.state_position_map[0]
          is_valid = spm_row != -1
          safe_indices = jnp.where(is_valid, spm_row, 0)
          gathered = cond_bias[safe_indices]
          cond_bias = jnp.where(is_valid[:, None], gathered, 0.0)
        zeros_bias = jnp.zeros_like(cond_bias)
        if stage_set.ar_logit_transform is not None:
          stored_logits = jax.vmap(
            stage_set.ar_logit_transform,
            in_axes=(1, 0),
            out_axes=0,
          )(logits, zeros_bias)  # (L, 21)
          sampling_logits = jax.vmap(
            stage_set.ar_logit_transform,
            in_axes=(1, 0),
            out_axes=0,
          )(logits, cond_bias)  # (L, 21)
        else:
          stored_logits = stage_set.logit_transform(
            logits,
            bias=zeros_bias,
          )
          sampling_logits = stage_set.logit_transform(
            logits,
            bias=cond_bias,
          )

        # Per-group logit averaging (tied positions within each group slot).
        avg_stored = jax.vmap(
          lambda mg: _fuse_one_group(stored_logits, mg, stage_set.tie_group_fuse),
        )(mask_group)  # (G, 21)
        avg_sampling = jax.vmap(
          lambda mg: _fuse_one_group(sampling_logits, mg, stage_set.tie_group_fuse),
        )(mask_group)  # (G, 21)

        # Sample each group from its own bias-applied logits, independently.
        subkeys = jax.vmap(lambda gid: jax.random.fold_in(key, gid))(group_id)  # (G, 2)
        sampled = jax.vmap(
          lambda k, sampling_logits_g: jax.random.categorical(k, sampling_logits_g / cond.temperature),
        )(subkeys, avg_sampling)  # (G,)

        # Update all positions in each group (fixed positions override the sample).
        fixed_mask_bool = cond.fixed_mask.astype(jnp.bool_)
        is_group_fixed = jnp.any(fixed_mask_bool[None, :] & mask_group, axis=1)  # (G,)
        group_fixed_token = jnp.max(
          jnp.where(fixed_mask_bool[None, :] & mask_group, cond.fixed_tokens[None, :], 0),
          axis=1,
        )  # (G,)
        final_token = jnp.where(is_group_fixed, group_fixed_token, sampled).astype(jnp.int32)  # (G,)

        # Groups within a wave are disjoint (by construction of a valid
        # coloring), so summing each group's masked contribution is safe.
        token_grid = jnp.where(mask_group, final_token[:, None], 0)  # (G, L)
        any_group_covers_position = jnp.any(mask_group, axis=0)  # (L,)
        new_seq = jnp.where(any_group_covers_position, jnp.sum(token_grid, axis=0), seq)

        return new_seq, avg_stored

      def no_sample(seq: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Skip update: no active group slots in this (padded) wave."""
        return seq, jnp.zeros((max_groups_per_wave, 21))

      return jax.lax.cond(wave_has_active_group, do_sample, no_sample, sequence)

    # 3. Run wave iteration
    if self.use_while_loop:
      # lax.while_loop: not reverse-mode differentiable.
      # Lowers to a single XLA WhileOp — significantly faster to compile
      # than lax.scan for large n_waves. Inference-only path.
      def _while_body(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
      ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        step, seq, logits_stack = carry
        new_seq, step_logits = step_fn(seq, step)
        return (step + 1, new_seq, logits_stack.at[step].set(step_logits))

      _, final_seq, logits_stack = jax.lax.while_loop(
        lambda c: c[0] < n_waves,
        _while_body,
        (jnp.int32(0), init_sequence, jnp.zeros((n_waves, max_groups_per_wave, 21))),
      )
    else:
      final_seq, logits_stack = self.wave_iterator(
        step_fn,
        init_sequence,
        jnp.arange(n_waves),
      )

    # 4. Post-hoc scatter: map per-wave (per-group-slot) logits to per-position logits
    def scatter_logits(
      logits_final: jnp.ndarray,
      wave_idx: jnp.ndarray,
    ) -> tuple[jnp.ndarray, None]:
      """Scatter one wave's per-group logits to their positions."""
      is_active = wave.group_valid[wave_idx].astype(jnp.bool_)  # (G,)
      pos0 = wave.group_positions[wave_idx, :, 0]  # (G,)
      group_id = cond.tie_group_map[0, pos0]  # (G,)
      this_rank = wave_idx * max_groups_per_wave + jnp.arange(max_groups_per_wave, dtype=jnp.int32)  # (G,)
      is_first_occurrence = is_active & (group_first_rank[group_id] == this_rank)  # (G,)
      # (L, G): position p selects group slot g iff p is in g's group AND this
      # wave is that group's first (only the first occurrence's wave actually
      # computed real logits; step_fn's no_sample wrote zeros elsewhere).
      # Groups within a wave are disjoint, so at most one slot is True per row.
      position_selects_group = (cond.tie_group_map[0][:, None] == group_id[None, :]) & is_first_occurrence[None, :]
      step_logits = logits_stack[wave_idx]  # (G, 21)
      selected_logits = jnp.einsum("lg,gv->lv", position_selects_group.astype(jnp.float32), step_logits)
      any_valid = jnp.any(position_selects_group, axis=1)  # (L,)
      new_logits_final = jnp.where(any_valid[:, None], selected_logits, logits_final)
      return new_logits_final, None

    logits_init = jnp.zeros((L, 21))
    logits_final, _ = jax.lax.scan(
      scatter_logits,
      logits_init,
      jnp.arange(n_waves),
      unroll=1,
    )

    # 5. Return SampleResult
    return SampleResult(
      sequence=final_seq,
      logits=logits_final,
    )


__all__ = ["AutoregressiveDecode"]
