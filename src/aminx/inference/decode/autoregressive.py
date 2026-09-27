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
from typing import Any, TypeVar

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

#: Wave-loop carry: the sequence (full recompute) or (sequence, layer cache) (incremental).
CarryT = TypeVar("CarryT")


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


def _fuse_and_sample(
  logits: jnp.ndarray,
  cond_bias: jnp.ndarray,
  mask_group: jnp.ndarray,
  fixed_mask: jnp.ndarray,
  fixed_tokens: jnp.ndarray,
  group_id: jnp.ndarray,
  key: PRNGKeyArray,
  stage_set: StageSet,
  temperature: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Fuse states, average tied positions, sample one token per group slot.

  Shared by the full-recompute and incremental waves so both run the same ops. The
  position axis N is either all L positions (full recompute) or the wave's position slab
  (incremental); nothing here depends on which.

  Parameters
  ----------
  logits : (S, N, 21)
      Per-state logits, already realigned to the reference frame.
  cond_bias : (N, 21)
      Additive logit bias for the same N positions.
  mask_group : (G, N)
      Position membership of each group slot's first occurrence (all-False = inactive).
  fixed_mask, fixed_tokens : (N,)
      Fixed-position override for the same N positions.
  group_id : (G,)
      Tie-group id per slot (seeds each slot's sampling key).

  Returns
  -------
  (final_token, avg_stored) : ((G,), (G, 21))
      Token per group slot (fixed tokens override samples) and bias-free fused logits.

  """
  zeros_bias = jnp.zeros_like(cond_bias)
  if stage_set.ar_logit_transform is not None:
    stored_logits = jax.vmap(
      stage_set.ar_logit_transform,
      in_axes=(1, 0),
      out_axes=0,
    )(logits, zeros_bias)  # (N, 21)
    sampling_logits = jax.vmap(
      stage_set.ar_logit_transform,
      in_axes=(1, 0),
      out_axes=0,
    )(logits, cond_bias)  # (N, 21)
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
    lambda k, sampling_logits_g: jax.random.categorical(k, sampling_logits_g / temperature),
  )(subkeys, avg_sampling)  # (G,)

  # Fixed positions override the sample.
  fixed_mask_bool = fixed_mask.astype(jnp.bool_)
  is_group_fixed = jnp.any(fixed_mask_bool[None, :] & mask_group, axis=1)  # (G,)
  group_fixed_token = jnp.max(
    jnp.where(fixed_mask_bool[None, :] & mask_group, fixed_tokens[None, :], 0),
    axis=1,
  )  # (G,)
  final_token = jnp.where(is_group_fixed, group_fixed_token, sampled).astype(jnp.int32)  # (G,)
  return final_token, avg_stored


def _decode_positions(
  layers: tuple[Any, ...],
  w_s: jnp.ndarray,
  node_features: jnp.ndarray,
  edge_features: jnp.ndarray,
  neighbor_indices: jnp.ndarray,
  mask: jnp.ndarray,
  ar_neighbors: jnp.ndarray,
  cache: jnp.ndarray,
  sequence: jnp.ndarray,
  positions: jnp.ndarray,
  position_valid: jnp.ndarray,
  key: PRNGKeyArray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Conditional decoder over a slab of B positions, reading everyone else from `cache`.

  Reproduces ``Decoder.call_conditional`` row-for-row for the rows in ``positions``. At
  layer l a row reads its K neighbours' layer-l features: from this pass if the neighbour
  is in the slab (same-wave positions update together, as in full recompute), otherwise
  from ``cache[l - 1]`` (layer 0 is the encoder output). The edge context is the same
  ``mask_bw * [e, emb(seq_j), h_j] + mask_fw * [e, 0, h_enc_j]`` as
  ``pack_conditional_decoder_static_edges`` + ``conditional_decoder_layer_edge_features``.

  Parameters
  ----------
  node_features : (L, H) encoder node features.  edge_features : (L, K, E).
  neighbor_indices : (L, K).  mask : (L,).
  ar_neighbors : (L, K) ar_mask gathered at each row's neighbours.
  cache : (max(n_layers - 1, 1), L, H) layer-1..n-1 features of already-decoded positions.
  sequence : (L,) current tokens (UNDRAWN_TOKEN = -1 embeds to zero).
  positions : (B,) slab rows (padding rows point at 0).  position_valid : (B,).

  Returns
  -------
  (h_final, layer_outputs) : ((B, H), (n_layers - 1, B, H))
      Final decoder features for the slab, and its layer-1..n-1 features for the cache.

  """
  nbr = neighbor_indices[positions]  # (B, K)
  e = edge_features[positions]  # (B, K, E)
  row_mask = mask[positions]  # (B,)
  ar_rows = ar_neighbors[positions]  # (B, K)
  mask_bw = row_mask[:, None] * ar_rows
  mask_fw = row_mask[:, None] * (1 - ar_rows)
  emb_nbr = jax.nn.one_hot(sequence[nbr], 21) @ w_s  # (B, K, H)
  h_enc_nbr = node_features[nbr]  # (B, K, H)
  fw_context = mask_fw[..., None] * jnp.concatenate(
    [e, jnp.zeros_like(h_enc_nbr), h_enc_nbr],
    axis=-1,
  )
  # (B, K, B): neighbour k of row b is slab row c (padding rows never match).
  in_slab = (nbr[:, :, None] == positions[None, None, :]) & position_valid[None, None, :]
  in_slab_any = jnp.any(in_slab, axis=-1, keepdims=True)

  keys = jax.random.split(key, len(layers))
  h = node_features[positions]  # (B, H)
  outputs = []
  for i, layer in enumerate(layers):
    if i == 0:
      h_nbr = h_enc_nbr
    else:
      from_pass = jnp.einsum("bkc,ch->bkh", in_slab.astype(h.dtype), h)
      h_nbr = jnp.where(in_slab_any, from_pass, cache[i - 1][nbr])
    context = mask_bw[..., None] * jnp.concatenate([e, emb_nbr, h_nbr], axis=-1) + fw_context
    h = layer(h, context, row_mask, inference=True, key=keys[i])
    if i < len(layers) - 1:
      outputs.append(h)
  # A 1-layer decoder has no cached layers; return one inert row so the cache never has a
  # zero-size axis (SafeMap/lax.map batching reshapes every input and divides by its size).
  layer_outputs = jnp.stack(outputs) if outputs else jnp.zeros((1, *h.shape), h.dtype)
  return h, layer_outputs


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
  incremental: str = eqx.field(static=True, default="auto")
  max_positions_per_wave: int | None = eqx.field(static=True, default=None)

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
    wave_index_grid = jnp.broadcast_to(
      jnp.arange(n_waves, dtype=jnp.int32)[:, None],
      (n_waves, max_groups_per_wave),
    )
    slot_grid = jnp.broadcast_to(
      jnp.arange(max_groups_per_wave, dtype=jnp.int32)[None, :],
      (n_waves, max_groups_per_wave),
    )
    combined_rank_grid = wave_index_grid * max_groups_per_wave + slot_grid
    no_occurrence_sentinel = n_waves * max_groups_per_wave
    flat_group_id = jnp.where(wave.group_valid, real_group_id_grid, 0).reshape(-1)
    flat_rank = jnp.where(wave.group_valid, combined_rank_grid, no_occurrence_sentinel).reshape(-1)
    group_first_rank = (
      jnp.full((L,), no_occurrence_sentinel, dtype=jnp.int32).at[flat_group_id].min(flat_rank)
    )

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
      this_rank = wave_idx * max_groups_per_wave + jnp.arange(
        max_groups_per_wave,
        dtype=jnp.int32,
      )  # (G,)
      is_first_occurrence = is_active & (group_first_rank[group_id] == this_rank)  # (G,)
      mask_group = (cond.tie_group_map[0][None, :] == group_id[:, None]) & is_first_occurrence[
        :,
        None,
      ]  # (G, L)
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
        final_token, avg_stored = _fuse_and_sample(
          logits,
          cond_bias,
          mask_group,
          cond.fixed_mask,
          cond.fixed_tokens,
          group_id,
          key,
          stage_set,
          cond.temperature,
        )

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

    # 3. Run wave iteration: full recompute, or the incremental cache when it is exact.
    def iterate(
      step: Callable[[CarryT, jnp.ndarray], tuple[CarryT, jnp.ndarray]],
      init_carry: CarryT,
    ) -> tuple[CarryT, jnp.ndarray]:
      if self.use_while_loop:
        # lax.while_loop: not reverse-mode differentiable.
        # Lowers to a single XLA WhileOp — significantly faster to compile
        # than lax.scan for large n_waves. Inference-only path.
        def _while_body(
          carry: tuple[jnp.ndarray, Any, jnp.ndarray],
        ) -> tuple[jnp.ndarray, Any, jnp.ndarray]:
          step_idx, inner, logits_stack = carry
          new_inner, step_logits = step(inner, step_idx)
          return (step_idx + 1, new_inner, logits_stack.at[step_idx].set(step_logits))

        _, final_carry, logits_stack = jax.lax.while_loop(
          lambda c: c[0] < n_waves,
          _while_body,
          (jnp.int32(0), init_carry, jnp.zeros((n_waves, max_groups_per_wave, 21))),
        )
        return final_carry, logits_stack
      return self.wave_iterator(step, init_carry, jnp.arange(n_waves))

    def run_full() -> tuple[jnp.ndarray, jnp.ndarray]:
      return iterate(step_fn, init_sequence)

    # Incremental decoding is exact when (a) every position's already-decoded neighbours
    # were decoded in an earlier wave or this one (the ar_mask agrees with the wave
    # schedule), (b) states share the reference frame (identity state_position_map), and
    # (c) no wave covers more positions than the slab. Static conditions pick the path at
    # trace time; the data-dependent ones are checked on device (see `can_increment`).
    static_ok = (
      self.incremental != "off"
      and stage_set.decode_step is None
      and config.inference
      and cond.bias.shape[-2] == L
      and cond.state_position_map.shape == (S, L)
    )
    if not static_ok:
      final_seq, logits_stack = run_full()
    else:
      layers = self.model.decoder.layers
      n_cache = max(len(layers) - 1, 1)  # >= 1: see _decode_positions
      hidden = enc.node_features.shape[-1]
      slab = max_groups_per_wave * wave.group_positions.shape[2]
      if self.max_positions_per_wave is not None:
        slab = max(slab, self.max_positions_per_wave)
      slab = min(slab, L)
      w_s = self.model.w_s_embed.weight

      # Wave in which each position is sampled (its tie group's first occurrence).
      pos_rank = group_first_rank[cond.tie_group_map[0]]  # (L,)
      decode_wave = jnp.where(
        pos_rank < no_occurrence_sentinel,
        pos_rank // max_groups_per_wave,
        n_waves,
      ).astype(jnp.int32)
      order_pos = jnp.argsort(decode_wave, stable=True)
      wave_start = jnp.searchsorted(
        decode_wave[order_pos],
        jnp.arange(n_waves + 1, dtype=jnp.int32),
        side="left",
      ).astype(jnp.int32)

      # Which (row, neighbour) pairs read the neighbour's decoder features.
      nbr_all = enc.neighbor_indices  # (S, L, K)
      ar_neighbors = jnp.take_along_axis(cond.ar_mask, nbr_all, axis=2)  # (S, L, K)
      valid_row = enc.mask > 0.5
      valid_nbr = (
        jnp.take_along_axis(enc.mask, nbr_all.reshape(S, -1), axis=1).reshape(nbr_all.shape) > 0.5
      )
      reads = valid_row[..., None] & valid_nbr & (ar_neighbors > 0.5)
      wave_row = decode_wave[None, :, None]
      wave_nbr = decode_wave[nbr_all]
      # Same-wave visibility (tie groups): the wave's cache rows must be recomputed after
      # its tokens are drawn, because later readers see them with those tokens.
      same_wave = reads & (wave_nbr == wave_row) & (nbr_all != jnp.arange(L)[None, :, None])
      needs_refresh = (
        jnp.zeros((n_waves + 1,), jnp.int32)
        .at[decode_wave]
        .max(jnp.any(same_wave, axis=(0, 2)).astype(jnp.int32))[:n_waves]
        > 0
      )

      per_state_static = (
        enc.node_features,
        enc.edge_features,
        enc.neighbor_indices,
        enc.mask,
        ar_neighbors,
      )

      def decode_slab(
        seq: jnp.ndarray,
        cache: jnp.ndarray,
        positions: jnp.ndarray,
        pvalid: jnp.ndarray,
      ) -> tuple[jnp.ndarray, jnp.ndarray]:
        def per_state_fn(inputs: tuple[jnp.ndarray, ...]) -> tuple[jnp.ndarray, jnp.ndarray]:
          node_features, edge_features, neighbor_indices, mask, ar_nb, cache_s = inputs
          return _decode_positions(
            layers,
            w_s,
            node_features,
            edge_features,
            neighbor_indices,
            mask,
            ar_nb,
            cache_s,
            seq,
            positions,
            pvalid,
            key,
          )

        return self.state_iterator(per_state_fn, (*per_state_static, cache), in_axes=0)

      def inc_step(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        wave_idx: jnp.ndarray,
      ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        pos0 = wave.group_positions[wave_idx, :, 0]
        group_id = cond.tie_group_map[0, pos0]
        is_active = wave.group_valid[wave_idx].astype(jnp.bool_)
        this_rank = wave_idx * max_groups_per_wave + jnp.arange(
          max_groups_per_wave,
          dtype=jnp.int32,
        )
        is_first_occurrence = is_active & (group_first_rank[group_id] == this_rank)
        base = wave_start[wave_idx]
        slot_idx = jnp.arange(slab, dtype=jnp.int32)
        pvalid = slot_idx < (wave_start[wave_idx + 1] - base)
        positions = jnp.where(pvalid, order_pos[jnp.clip(base + slot_idx, 0, L - 1)], 0)
        write_idx = jnp.where(pvalid, positions, L)  # L = dropped by mode="drop"
        mask_group = (
          (cond.tie_group_map[0][positions][None, :] == group_id[:, None])
          & is_first_occurrence[:, None]
          & pvalid[None, :]
        )  # (G, B)

        def do_sample(
          c: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
          seq, cache = c
          h_final, layer_out = decode_slab(seq, cache, positions, pvalid)
          logits = _project_logits(self.model, h_final)  # (S, B, 21)
          final_token, avg_stored = _fuse_and_sample(
            logits,
            cond.bias[positions],
            mask_group,
            cond.fixed_mask[positions],
            cond.fixed_tokens[positions],
            group_id,
            key,
            stage_set,
            cond.temperature,
          )
          token_grid = jnp.where(mask_group, final_token[:, None], 0)  # (G, B)
          covered = jnp.any(mask_group, axis=0)
          new_seq = seq.at[write_idx].set(
            jnp.where(covered, jnp.sum(token_grid, axis=0), seq[positions]),
            mode="drop",
          )
          layer_out = jax.lax.cond(
            needs_refresh[wave_idx],
            lambda: decode_slab(new_seq, cache, positions, pvalid)[1],
            lambda: layer_out,
          )
          new_cache = cache.at[:, :, write_idx].set(layer_out, mode="drop")
          return (new_seq, new_cache), avg_stored

        def no_sample(
          c: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
          return c, jnp.zeros((max_groups_per_wave, 21))

        return jax.lax.cond(jnp.any(is_first_occurrence), do_sample, no_sample, carry)

      def run_incremental() -> tuple[jnp.ndarray, jnp.ndarray]:
        init_cache = jnp.zeros((S, n_cache, L, hidden), enc.node_features.dtype)
        (seq, _), logits_stack = iterate(inc_step, (init_sequence, init_cache))
        return seq, logits_stack

      if self.incremental == "force":
        final_seq, logits_stack = run_incremental()
      else:
        consistent = ~jnp.any(reads & (wave_nbr > wave_row))
        identity_frame = jnp.all(
          cond.state_position_map == jnp.arange(L, dtype=cond.state_position_map.dtype)[None, :],
        )
        fits_slab = jnp.all((wave_start[1:] - wave_start[:-1]) <= slab)
        can_increment = consistent & identity_frame & fits_slab
        # The predicate depends on the bundle only, so under vmap over keys it stays a real
        # branch (only one path runs).
        final_seq, logits_stack = jax.lax.cond(can_increment, run_incremental, run_full)

    # 4. Per-position logits: each position takes its group's slot at the group's first
    # occurrence (O(L) gather; replaces a per-wave (L, G) scatter scan that was O(L^2)).
    pos_first_rank = group_first_rank[cond.tie_group_map[0]]  # (L,)
    scheduled = pos_first_rank < no_occurrence_sentinel
    safe_rank = jnp.where(scheduled, pos_first_rank, 0)
    logits_final = jnp.where(
      scheduled[:, None],
      logits_stack[safe_rank // max_groups_per_wave, safe_rank % max_groups_per_wave],
      0.0,
    )

    # 5. Return SampleResult
    return SampleResult(
      sequence=final_seq,
      logits=logits_final,
    )


__all__ = ["AutoregressiveDecode"]
