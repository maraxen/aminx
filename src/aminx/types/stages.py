"""Stage protocols and type aliases for logic injection.

Tier 1 Protocols (generic, reusable):
- TransformFn[In, Out] — stateless transformation
- RollingFn[Carry, In, Out] — scan-body with init_carry
- FuseFn[PerItem, Combined] — reduce-across-axis

Tier 2 Aliases (MPNN-specific):
- FeaturizeFn, EncoderStepFn, EncoderStateFn, ProteinEncodeFn, LigandEncodeFn
- ConditionalDecodeFn, UnconditionalDecodeFn
- LogitTransformFn (as FuseFn), ARLogitTransformFn (as FuseFn)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

import equinox as eqx
import jax
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

if TYPE_CHECKING:
  from aminx.inference.decode.protocols import DecoderSinkFn
  from aminx.types.boundaries import AxisBoundary
  from aminx.types.bundles import DecodeOutput, EncoderOutput

# Type variables for generic protocols
In = TypeVar("In")
Out = TypeVar("Out")
Carry = TypeVar("Carry")
PerItem = TypeVar("PerItem")
Combined = TypeVar("Combined")


@runtime_checkable
class TransformFn(Protocol[In, Out]):
  """Stateless transformation: In → Out.

  Generic function protocol for any host- or JAX-traceable transformation.
  Examples: featurization, encoding, decoding.
  """

  def __call__(self, input: In) -> Out: ...


@runtime_checkable
class RollingFn(Protocol[Carry, In, Out]):
  """Carry-based scan body: (carry, input) → (carry, output).

  Used for state-accumulating transformations (e.g. encoder state threading).
  Carry structure must be fixed at JAX trace time.
  """

  def init_carry(self) -> Carry: ...

  def __call__(
    self,
    carry: Carry,
    state_idx: Int[Array, ""],
    input: In,
  ) -> tuple[Carry, Out]: ...


@runtime_checkable
class FuseFn(Protocol[PerItem, Combined]):
  """Reduce-across-axis fusion: per_item → combined.

  Used for combining per-item (e.g. per-state) results into a single combined result.
  Examples: logit stacking and reduction, state aggregation.
  """

  def __call__(self, per_item: PerItem, bias: PerItem | None = None) -> Combined: ...


# Tier 2 Aliases: MPNN-Specific Type Specializations
# In practice these take specific Pytree bundles, but are typed flexibly for now.

FeaturizeFn = TransformFn[Any, Any]
EncoderStepFn = TransformFn[Any, Any]
EncoderStateFn = RollingFn[Any, Any, Any]
ProteinEncodeFn = TransformFn[Any, Any]
LigandEncodeFn = TransformFn[Any, Any]

ConditionalDecodeFn = TransformFn[Any, Any]
UnconditionalDecodeFn = TransformFn[Any, Any]

LogitTransformFn = FuseFn[Float[Array, "S L V"], Float[Array, "L V"]]
# ARLogitTransformFn: concrete signature (S, V) + (V,) -> (V,) — bias always passed, never None
# Callers provide jnp.zeros_like(bias_shape) for no-op bias
ARLogitTransformFn = FuseFn[Float[Array, "S V"], Float[Array, "V"]]


@runtime_checkable
class BatchLogitFn(Protocol):
  """Protocol for fusing stacked state logits into a single canonical set.

  Equivalent to the generic FuseFn specialized for logit tensors.
  Accepts optional bias to be applied after fusion.
  """

  def __call__(
    self,
    per_state: Float[Array, "S ... V"],
    bias: Float[Array, "... V"] | None = None,
  ) -> Float[Array, "... V"]:
    """Fuse per-state logits and optional bias into canonical logits.

    Parameters
    ----------
    per_state : Float[Array, "S ... V"]
        Per-state logits. S = number of states, V = vocabulary size (21 amino acids).
    bias : Float[Array, "... V"] | None
        Optional per-position logit bias to add after fusion. If None, no bias applied.

    Returns
    -------
    Float[Array, "... V"]
        Fused logits with state dimension reduced.

    """
    ...


@runtime_checkable
class TieGroupFuseFn(Protocol):
  """Protocol for fusing tied-position logits into a single canonical set.

  Signature: (logits: (L, V), mask: (L,)) -> (V,)
  where L is the sequence length (positions in group selected by mask)
  and V is the vocabulary size (usually 21).
  """

  def __call__(
    self,
    logits: Float[Array, "L V"],
    mask: Bool[Array, "L"],
  ) -> Float[Array, "V"]:
    """Fuse tied-position logits into a single canonical set.

    Parameters
    ----------
    logits : Float[Array, "L V"]
        Per-position logits for all positions in the group. L = sequence length.
    mask : Bool[Array, "L"]
        Boolean mask selecting tied positions (True = in group).

    Returns
    -------
    Float[Array, "V"]
        Fused logits for the tied group.

    """
    ...


@runtime_checkable
class SampleStepFn(Protocol):
  """Protocol for sampling tokens from logits using stochastic strategies.

  Signature: (logits: (... V), key: PRNGKeyArray) -> (... ,)
  where V is the vocabulary size (usually 21 amino acids) and "..." represents
  zero or more leading dimensions for batch/group processing.
  """

  def __call__(
    self,
    logits: Float[Array, "... V"],
    key: PRNGKeyArray,
  ) -> Int[Array, "..."]:
    """Sample tokens from logits using a stochastic strategy.

    Parameters
    ----------
    logits : Float[Array, "... V"]
        Per-position or per-group logits for sampling. V = vocabulary size
        (21 amino acids). Leading dimensions "..." may be empty (single position)
        or represent batch/group dimensions.
    key : PRNGKeyArray
        JAX random key for sampling.

    Returns
    -------
    Int[Array, "..."]
        Sampled tokens with leading dimensions matching input logits.
        Typical implementations use jax.random.categorical, Gumbel-Softmax,
        or Straight-Through Estimator (STE) strategies.

    Examples
    --------
    Categorical sampling (single position):
        logits: (V,) → tokens: ()

    Gumbel-Softmax (differentiable, single position):
        logits: (V,) → tokens: ()

    Straight-Through Estimator / STE (single position):
        logits: (V,) → tokens: ()

    Group sampling (multiple positions):
        logits: (L, V) → tokens: (L,) where L = group size

    """
    ...


@runtime_checkable
class EncoderSinkFn(Protocol):
  """Optional side-effect hook called once per noise-level encoding.

  When encoding_fusion is set, fires D times per structure (once per noise
  level), before fusion. Implementations MUST use io_callback with ordered=False.
  """

  def __call__(
    self,
    enc: EncoderOutput,
    batch_idx: jax.Array,
    structure_idx: jax.Array,
    noise_idx: jax.Array,
  ) -> None: ...


@runtime_checkable
class EncodingFusionFn(Protocol):
  """Fuse D noise-level encoded outputs into K outputs for decoding.

  Called after encoding at D noise levels, before decoding. Receives a stacked
  EncoderOutput with a leading D axis and returns an EncoderOutput with a
  leading K axis — K is arbitrary (K=1 for averaging, K=D for identity,
  K<D for cluster representatives, etc.).
  """

  def __call__(self, stacked: EncoderOutput) -> EncoderOutput: ...


@runtime_checkable
class DecodingFusionFn(Protocol):
  """Fuse K decoded outputs into a single final DecodeOutput.

  Called after decoding at K noise/sample levels, before final output.
  Receives a stacked DecodeOutput with K outputs (sequences and logits)
  and returns a DecodeOutput with fused results — enabling logit ensembling,
  best-of-K selection, or other post-decode aggregation strategies.
  """

  def __call__(self, stacked: DecodeOutput) -> DecodeOutput: ...


class ConditionalDecodeStep(eqx.Module):
  """Wraps decoder.call_conditional for use as a stage_set.decode_step.

  Encapsulates the conditional decoding operation in a composable module
  for clean wiring into inference kernels. Holds the sequence embedding
  lookup table (w_s_embed) for one-hot → embedding conversion.

  Parameters
  ----------
  decoder : Any
      Model's decoder module (e.g., model.decoder).
  w_s_embed : Any
      Sequence embedding lookup weights (e.g., model.w_s_embed.weight).

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  decoder: Any  # model.decoder
  w_s_embed: Any  # model.w_s_embed.weight

  def __call__(
    self,
    node_f: Any,
    edge_f: Any,
    nei: Any,
    mask: Any,
    ar_mask: Any,
    seq_oh: Any,
    *,
    key: Any,
    inference: Any,
  ) -> Any:
    """Decode conditional logits from features and sequence context.

    Parameters
    ----------
    node_f : Any
        Node (residue) features from encoder. Shape ``(L, D)`` where
        L = sequence length, D = embedding dimension.
    edge_f : Any
        Edge features between neighbors. Shape ``(L, K, D)`` where
        K = num neighbors.
    nei : Any
        Neighbor indices for gather operations. Shape ``(L, K)``.
    mask : Any
        Per-residue validity mask. Shape ``(L,)``.
    ar_mask : Any
        Autoregressive attention mask (1.0 = visible, 0.0 = masked).
        Shape ``(L, L)``.
    seq_oh : Any
        One-hot encoded input sequence (previous predictions + fixed).
        Shape ``(L, V)`` where V = vocab size (21).
    key : Any
        PRNG key for any stochastic operations.
    inference : Any
        Inference configuration object.

    Returns
    -------
    Any
        Logits array of shape ``(L, V)`` — per-position per-amino-acid scores.

    """
    return self.decoder.call_conditional(
      node_f,
      edge_f,
      nei,
      mask,
      ar_mask,
      seq_oh,
      self.w_s_embed,
      key=key,
      inference=inference,
    )


class UnconditionalDecodeStep(eqx.Module):
  """Wraps decoder.__call__ for unconditional scoring (no sequence context).

  Encapsulates the unconditional decoding operation in a composable module
  for clean wiring into inference kernels. No sequence embedding required.

  Parameters
  ----------
  decoder : Any
      Model's decoder module (e.g., model.decoder).

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  decoder: Any  # model.decoder

  def __call__(
    self,
    node_f: Any,
    edge_f: Any,
    nei: Any,
    mask: Any,
    *,
    key: Any,
    inference: Any,
  ) -> Any:
    """Decode unconditional logits from backbone features alone.

    Parameters
    ----------
    node_f : Any
        Node (residue) features from encoder. Shape ``(L, D)`` where
        L = sequence length, D = embedding dimension.
    edge_f : Any
        Edge features between neighbors. Shape ``(L, K, D)`` where
        K = num neighbors.
    nei : Any
        Neighbor indices for gather operations. Shape ``(L, K)``.
    mask : Any
        Per-residue validity mask. Shape ``(L,)``.
    key : Any
        PRNG key for any stochastic operations.
    inference : Any
        Inference configuration object.

    Returns
    -------
    Any
        Logits array of shape ``(L, V)`` — per-position per-amino-acid scores.

    """
    return self.decoder(node_f, edge_f, nei, mask, key=key, inference=inference)


class StageSet(eqx.Module):
  """Composable typed bag of inference pipeline stages for modular decode paths.

  eqx.Module makes this a JAX PyTree — weight arrays inside FuseFn
  implementations (e.g., ARLogitFuse, TieGroupProductOfExperts) are traced
  through JIT correctly. StageSet is the primary extensibility point:
  callers wire in encode_fn, driver, and stage_set, then invoke the pipeline
  with different stage_set instances for sampling vs. scoring vs. ARv2 modes.

  **Topology Inference Rules** (kernel reads these to select code path):
  - If ``sample_step is not None`` → autoregressive (AR) sampling mode
  - If ``isinstance(decode_step, UnconditionalDecodeStep)`` → unconditional scoring
  - Otherwise → conditional scoring (decode_step is ConditionalDecodeStep)
  - If ``encoding_fusion is not None`` → noise axis maps over encode only;
    fusion reduces D → K; decode maps over K outputs
  - If ``encoder_sink`` is non-empty → iterates tuple, fires io_callback per sink
    per noise-level encoding, before fusion

  Parameters
  ----------
  logit_transform : BatchLogitFn | None
      Multi-state → single-state logit fusion. Signature:
      ``(S, L, V) + (L, V) → (L, V)``
      Reduces per-state logits to unified logits using state_weights and
      fusion strategy (e.g., arithmetic/geometric mean, product of experts).
      None = identity (single-state passthrough, no fusion).
  ar_logit_transform : BatchLogitFn | None
      Per-position bias injection during autoregressive decode.
      Signature: ``(S, V) + (V,) → (V,)``
      Takes per-state logits + position-specific bias, outputs final logits
      for sampling. Bias is always passed (never None); callers provide
      jnp.zeros_like for no-op.
      None = identity.
  decode_step : ConditionalDecodeStep | UnconditionalDecodeStep | None
      Decoder forward pass. ConditionalDecodeStep requires ar_mask and seq_oh
      (sequence one-hot); UnconditionalDecodeStep ignores them.
      None = topology not yet wired (internal invariant).
  sample_step : SampleStepFn | None
      Sampling strategy: None = scoring-only mode (no sampling);
      categorical/Gumbel-Softmax/STE = sample from logits.
      Signature: (logits: (... V), key: PRNGKeyArray) → (... ,)
      Examples: eqx.Module subclasses with __call__(logits, key) → tokens.
  tie_group_fuse : TieGroupFuseFn | None
      Reduce over tied positions. Signature:
      ``(logits: (L, V), mask: (L,)) → (V,)``
      Implements logit fusion across positions sharing the same tie_group_id
      (e.g., TieGroupProductOfExperts for log-softmax sum across ties).
      None = no tied-position fusion (each position treated independently).
  encoder_sink : tuple[EncoderSinkFn, ...]
      Zero or more side-effect hooks called once per noise-level encoding.
      Each sink fires D times per structure (once per noise level), before fusion.
      Sink implementations MUST use io_callback with ordered=False.
      Empty tuple (default) = no sinks.
  axis_boundaries : dict[str, AxisBoundary]
      Per-axis pipeline boundary configuration mapping axis names to AxisBoundary
      instances. Each AxisBoundary bundles optional fuse, tap, and sink operations
      for one named axis. All fields are static — no JAX arrays. Empty dict (default)
      = no axis-level boundaries.

  Notes
  -----
  Plain callables (lambdas, functools.partial without parameters) and
  stateless functions can be used in slots; eqx.filter_jit marks non-array
  content static. Weight-bearing implementations must be eqx.Module subclasses
  for JAX tracing.

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  logit_transform: BatchLogitFn | None = None
  ar_logit_transform: BatchLogitFn | None = None
  decode_step: ConditionalDecodeStep | UnconditionalDecodeStep | None = None
  sample_step: SampleStepFn | None = None  # None = scoring mode; categorical/gumbel/ste = sampling
  tie_group_fuse: TieGroupFuseFn | None = None
  encoder_sink: tuple[EncoderSinkFn, ...] = ()
  decoder_sink: tuple[DecoderSinkFn, ...] = eqx.field(static=True, default_factory=tuple)
  encoding_fusion: EncodingFusionFn | None = None
  decoding_fusion: DecodingFusionFn | None = None
  axis_boundaries: dict[str, AxisBoundary] = eqx.field(static=True, default_factory=dict)


__all__ = [
  "ARLogitTransformFn",
  "BatchLogitFn",
  "ConditionalDecodeFn",
  "ConditionalDecodeStep",
  "DecodingFusionFn",
  "EncoderSinkFn",
  "EncoderStateFn",
  "EncoderStepFn",
  "EncodingFusionFn",
  "FeaturizeFn",
  "FuseFn",
  "LigandEncodeFn",
  "LogitTransformFn",
  "ProteinEncodeFn",
  "RollingFn",
  "StageSet",
  "TieGroupFuseFn",
  "TransformFn",
  "UnconditionalDecodeFn",
  "UnconditionalDecodeStep",
]
