"""Composable logit combination strategies as BatchLogitFn implementations.

These modules operate on stacked tensors of shape (S, ..., V) where S is the
number of states and V is the vocabulary size (usually 21). They reduce
the S dimension to produce a single logit set.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from aminx.registry import Registry
from aminx.types.stages import BatchLogitFn

# Registry for logit combination strategies
LOGIT_STRATEGIES: Registry[type[BatchLogitFn]] = Registry[type[BatchLogitFn]]("logit_strategies")


@LOGIT_STRATEGIES.register("arithmetic_mean")
class ArithmeticMeanLogits(eqx.Module):
  """Weighted arithmetic mean in log-space across states.

  Implements: log(Σ(wᵢ · exp(Lᵢ)) / Σwᵢ) via numerically stable logsumexp.
  Registered strategy name: ``"arithmetic_mean"``.

  Parameters
  ----------
  weights : Float[Array, "S"]
      Per-state weights. S = number of states. Traced JAX leaf.

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  weights: Float[Array, S]

  def __call__(
    self,
    per_state: Float[Array, "S ... V"],
    bias: Float[Array, "... V"] | None = None,
  ) -> Float[Array, "... V"]:
    """Fuse per-state logits via weighted arithmetic mean in log-space.

    Parameters
    ----------
    per_state : Float[Array, "S ... V"]
        Per-state logits. S = number of states, V = vocabulary size.
    bias : Float[Array, "... V"] | None
        Optional per-position logit bias to add after fusion.

    Returns
    -------
    Float[Array, "... V"]
        Fused logits, shape ``(..., V)``.

    """
    # Broadcast weights from shape (1,) to (S,) if needed, or validate match.
    S = per_state.shape[0]
    w_len = self.weights.shape[0]
    if w_len == 1:
      # Broadcast single weight to all states
      weights = jnp.broadcast_to(self.weights, (S,))
    elif w_len == S:
      # Weights already match state count
      weights = self.weights
    else:
      # Misconfigured: weights length is neither 1 nor S
      raise ValueError(
        f"state_weights length {w_len} incompatible with {S} states; "
        f"expected length 1 or {S}",
      )

    # log_sum_exp(L_i + log(w_i)) - log(sum(w_i))
    # Safety for zero weights: log(1e-9) if weight is 0
    log_w = jnp.log(jnp.where(weights > 0, weights, 1e-9))

    # Add log_w to logits (broadcasting over ..., V)
    # per_state is (S, ..., V), log_w is (S,)
    # We need to reshape log_w to (S, 1, ..., 1)
    dims_to_add = per_state.ndim - 1
    log_w_reshaped = log_w.reshape((S,) + (1,) * dims_to_add)

    weighted_logits = per_state + log_w_reshaped

    # Numerically stable logsumexp over S
    max_logits = jnp.max(weighted_logits, axis=0)
    shifted = weighted_logits - max_logits[None, ...]
    sum_exp = jnp.sum(jnp.exp(shifted), axis=0)

    log_sum_w = jnp.log(jnp.sum(weights))

    result = max_logits + jnp.log(sum_exp) - log_sum_w

    if bias is not None:
      result = result + bias

    return result


@LOGIT_STRATEGIES.register("geometric_mean")
class GeometricMeanLogits(eqx.Module):
  """Weighted geometric mean across states.

  Implements: Σ(wᵢ · Lᵢ) / (T · Σwᵢ) where T is temperature.
  Registered strategy name: ``"geometric_mean"``.

  Parameters
  ----------
  weights : Float[Array, "S"]
      Per-state weights. S = number of states. Traced JAX leaf.
  temperature : float
      Temperature scaling factor. Static (not a JAX array). Default: 1.0.

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  weights: Float[Array, S]
  temperature: float = eqx.field(static=True, default=1.0)

  def __call__(
    self,
    per_state: Float[Array, "S ... V"],
    bias: Float[Array, "... V"] | None = None,
  ) -> Float[Array, "... V"]:
    """Fuse per-state logits via weighted geometric mean.

    Parameters
    ----------
    per_state : Float[Array, "S ... V"]
        Per-state logits. S = number of states, V = vocabulary size.
    bias : Float[Array, "... V"] | None
        Optional per-position logit bias to add after fusion.

    Returns
    -------
    Float[Array, "... V"]
        Fused logits, shape ``(..., V)``.

    """
    # Broadcast weights from shape (1,) to (S,) if needed, or validate match.
    S = per_state.shape[0]
    w_len = self.weights.shape[0]
    if w_len == 1:
      # Broadcast single weight to all states
      weights = jnp.broadcast_to(self.weights, (S,))
    elif w_len == S:
      # Weights already match state count
      weights = self.weights
    else:
      # Misconfigured: weights length is neither 1 nor S
      raise ValueError(
        f"state_weights length {w_len} incompatible with {S} states; "
        f"expected length 1 or {S}",
      )

    dims_to_add = per_state.ndim - 1
    w_reshaped = weights.reshape((S,) + (1,) * dims_to_add)

    weighted_logits = per_state * w_reshaped
    sum_weighted = jnp.sum(weighted_logits, axis=0)
    sum_w = jnp.sum(weights)

    result = sum_weighted / (self.temperature * sum_w)

    if bias is not None:
      result = result + bias

    return result


@LOGIT_STRATEGIES.register("product")
class ProductOfProbabilities(eqx.Module):
  """Scaled weighted sum of logits across states.

  Implements: ``κ · Σ(wᵢ · Lᵢ)`` where ``κ`` is ``sharpness``.
  Registered strategy name: ``"product"``.

  The two parameters are deliberately separate because they mean different
  things, and conflating them is what makes a "product" silently stop being one:

  * ``weights`` are **mixing proportions** — which states matter, and how much
    relative to each other.
  * ``sharpness`` is the **concentration** of the fused distribution — the thing
    that makes this a *product* rather than an *average*.

  .. warning::

     With ``Σw = 1`` and ``sharpness = 1`` this is **not** a product of experts.
     It is a weighted geometric mean (a *logarithmic opinion pool*),
     ``Π pᵢ^{wᵢ}``. The decisive check: for S identical experts a product must
     give ``p^S`` (sharper), whereas ``Σw = 1`` returns ``p`` unchanged. The
     ``Σw = 1`` constraint is precisely what converts the product into an
     average, so a caller that normalises its weight vector and leaves
     ``sharpness`` at 1 gets averaging semantics while the strategy is still
     named ``"product"``.

  Recipes:

  ===========================  ===============  =========================================
  ``weights``                  ``sharpness``    Operator
  ===========================  ===============  =========================================
  all ones (default)           ``1.0``          plain product of experts, ``Σ Lᵢ``
  ``Σw = 1``                   ``None``         weighted PoE, exponent mass matched to S
  ``Σw = 1``                   ``1.0``          logarithmic opinion pool (**not** a product)
  any                          ``κ``            logits scaled by ``κ``
  ===========================  ===============  =========================================

  ``sharpness`` is the reciprocal of a sampling temperature: scaling logits by
  ``κ`` is identical to sampling at ``T/κ``. Set it deliberately — if it varies
  with the number of active states, effective temperature varies with it, and
  any comparison across weight vectors confounds mixing with concentration.

  Parameters
  ----------
  weights : Float[Array, "S"]
      Per-state mixing proportions. S = number of states. Traced JAX leaf.
  sharpness : float | None
      Logit scale ``κ``. Static (not a JAX array). Default ``1.0``, which
      preserves the unscaled weighted sum. ``None`` means "use S", i.e. scale so
      the total exponent mass matches a plain product of S experts
      (``Σ aᵢ = S`` for ``aᵢ = S·wᵢ``); with uniform ``Σw = 1`` weights this
      reproduces ``Σ Lᵢ`` exactly.

  References
  ----------
  .. [Hinton2002] Hinton, G. E. "Training products of experts by minimizing
     contrastive divergence." *Neural Computation* 14(8):1771-1800 (2002).
     https://doi.org/10.1162/089976602760128018

  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  weights: Float[Array, S]
  sharpness: float | None = eqx.field(static=True, default=1.0)

  def __call__(
    self,
    per_state: Float[Array, "S ... V"],
    bias: Float[Array, "... V"] | None = None,
  ) -> Float[Array, "... V"]:
    """Fuse per-state logits via scaled weighted sum.

    Parameters
    ----------
    per_state : Float[Array, "S ... V"]
        Per-state logits. S = number of states, V = vocabulary size.
    bias : Float[Array, "... V"] | None
        Optional per-position logit bias to add after fusion.

    Returns
    -------
    Float[Array, "... V"]
        Fused logits, shape ``(..., V)``. Bias is added AFTER the sharpness
        scaling, so it stays on the fused-logit scale and is not amplified.

    """
    # Broadcast weights from shape (1,) to (S,) if needed, or validate match.
    S = per_state.shape[0]
    w_len = self.weights.shape[0]
    if w_len == 1:
      # Broadcast single weight to all states
      weights = jnp.broadcast_to(self.weights, (S,))
    elif w_len == S:
      # Weights already match state count
      weights = self.weights
    else:
      # Misconfigured: weights length is neither 1 nor S
      raise ValueError(
        f"state_weights length {w_len} incompatible with {S} states; "
        f"expected length 1 or {S}",
      )

    dims_to_add = per_state.ndim - 1
    w_reshaped = weights.reshape((S,) + (1,) * dims_to_add)

    weighted_logits = per_state * w_reshaped
    result = jnp.sum(weighted_logits, axis=0)

    # `None` means "match a plain product of S experts": with a_i = S*w_i and
    # sum(w) = 1 the exponent mass sums to S, so uniform weights recover sum(L_i).
    kappa = float(S) if self.sharpness is None else self.sharpness
    if kappa != 1.0:
      result = result * kappa

    if bias is not None:
      result = result + bias

    return result


class ARLogitFuse(eqx.Module):
  """Default per-step logit fuser for autoregressive decode.

  Reduces per-state logits to a single canonical set via arithmetic mean,
  then applies bias for sampling/scoring. Called once per position during
  autoregressive decode.

  Notes
  -----
  Bias is always a concrete array (use ``jnp.zeros(V)`` for no-op).
  This is used in ``sample_autoregressive.kernel`` as the per-position fuser
  when ``stage_set.ar_logit_transform`` is set.

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  def __call__(self, logits: Float[Array, "S V"], bias: Float[Array, V]) -> Float[Array, V]:
    """Fuse per-state logits and add bias.

    Parameters
    ----------
    logits : Float[Array, "S V"]
        Per-state logits. S = number of states, V = vocabulary size (21).
    bias : Float[Array, "V"]
        Per-position logit bias to add after fusion.

    Returns
    -------
    Float[Array, "V"]
        Fused logits with bias applied.

    """
    return jnp.mean(logits, axis=0) + bias


# ---------------------------------------------------------------------------
# Tie-group fuse strategies
# ---------------------------------------------------------------------------

TIE_GROUP_STRATEGIES: Registry = Registry("tie_group_strategies")


@TIE_GROUP_STRATEGIES.register("logsumexp_mean")
class TieGroupLogsumexpMean(eqx.Module):
  """Logsumexp mean across tied positions (legacy behavior).

  Computes: ``logsumexp(logits[group], axis=0) - log(n_tied)``.
  Registered strategy name: ``"logsumexp_mean"``.

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187

  """

  def __call__(
    self,
    logits: Float[Array, "L V"],
    mask: Bool[Array, L],
  ) -> Float[Array, V]:
    """Fuse tied-position logits via logsumexp mean.

    Parameters
    ----------
    logits : Float[Array, "L V"]
        Per-position logits for all positions in the group.
    mask : Bool[Array, "L"]
        Boolean mask selecting tied positions.

    Returns
    -------
    Float[Array, "V"]
        Fused logits for the tied group.

    """
    group = jnp.where(mask[:, None], logits, -jnp.inf)
    n = jnp.sum(mask)
    return jax.scipy.special.logsumexp(group, axis=0) - jnp.log(jnp.maximum(n, 1))


@TIE_GROUP_STRATEGIES.register("product_of_experts")
class TieGroupProductOfExperts(eqx.Module):
  """Product-of-experts across tied positions (matches LigandMPNN reference).

  Computes: ``sum(log_softmax(logits[group]), axis=0)``.
  Registered strategy name: ``"product_of_experts"``.
  This matches the PyTorch reference's ``_combine_reference_tied_log_probs``.

  References
  ----------
  .. [LigandMPNN] Dauparas, J., et al. "Atomic context-conditioned protein
     sequence design using LigandMPNN." *Nature Methods* 22(4):717-723 (2025).
     https://doi.org/10.1038/s41592-025-02626-1

  .. [LigandMPNN-code] Dauparas, J. LigandMPNN source code (commit 3870631).
     https://github.com/dauparas/LigandMPNN

  """

  def __call__(
    self,
    logits: Float[Array, "L V"],
    mask: Bool[Array, L],
  ) -> Float[Array, V]:
    """Fuse tied-position logits via product-of-experts.

    Parameters
    ----------
    logits : Float[Array, "L V"]
        Per-position logits for all positions in the group.
    mask : Bool[Array, "L"]
        Boolean mask selecting tied positions.

    Returns
    -------
    Float[Array, "V"]
        Fused logits for the tied group.

    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.sum(jnp.where(mask[:, None], log_probs, 0.0), axis=0)


def make_stage_set(
  strategy: str = "arithmetic_mean",
  strategy_temperature: float = 1.0,
  state_weights: jax.Array | None = None,
  decoder_sink: tuple = (),
  sharpness: float | None = 1.0,
) -> StageSet:
  """Construct a StageSet with strategy-resolved logit_transform.

  Parameters
  ----------
  strategy : str
      Key into LOGIT_STRATEGIES (``"arithmetic_mean"``, ``"geometric_mean"``, ``"product"``).
  strategy_temperature : float
      Temperature forwarded to strategies that accept it (e.g. GeometricMeanLogits).
  state_weights : jax.Array or None
      Per-state mixing proportions. When None, defaults to uniform weights of
      shape (1,).
  decoder_sink : tuple, optional
      Zero or more decoder-side effect hooks. Empty tuple (default) = no sinks.
  sharpness : float | None
      Logit scale ``κ``, forwarded to strategies that accept it (currently
      ``ProductOfProbabilities``). Default ``1.0`` preserves existing behaviour.
      Pass ``None`` for "match a plain product of S experts". This is separate
      from ``state_weights`` on purpose: weights say which states matter,
      sharpness says how concentrated the fused distribution is. Normalised
      weights (``Σw = 1``) with ``sharpness=1`` give a logarithmic opinion pool,
      NOT a product — see :class:`ProductOfProbabilities`.

  Returns
  -------
  StageSet
      Stage set with logit_transform, ar_logit_transform, and tie_group_fuse wired.
      ``ar_logit_transform`` is the SAME instance as ``logit_transform``, so the
      autoregressive path honours ``strategy``, ``state_weights`` and
      ``strategy_temperature`` identically to the non-AR path.

  """
  from aminx.types.stages import StageSet

  strategy_cls = LOGIT_STRATEGIES.get(strategy)
  if strategy_cls is None:
    msg = f"Logit strategy '{strategy}' not found in LOGIT_STRATEGIES"
    raise ValueError(msg)

  weights = (
    jnp.asarray(state_weights, dtype=jnp.float32)
    if state_weights is not None
    else jnp.ones(1, dtype=jnp.float32)
  )

  # Forward only the optional knobs this strategy actually declares. Introspecting
  # the dataclass fields (eqx.Module IS a dataclass) rather than catching TypeError
  # keeps this correct now that more than one optional field exists -- a bare
  # try/except cannot tell "this class has no `sharpness`" from "constructing it
  # raised TypeError for an unrelated reason", and would silently fall back to
  # dropping BOTH knobs.
  field_names = {f.name for f in dataclasses.fields(strategy_cls)}
  optional_kwargs: dict[str, float | None] = {}
  if "temperature" in field_names:
    optional_kwargs["temperature"] = strategy_temperature
  if "sharpness" in field_names:
    optional_kwargs["sharpness"] = sharpness

  logit_transform = strategy_cls(weights, **optional_kwargs)

  return StageSet(
    logit_transform=logit_transform,
    # AR decode fuses with the SAME configured instance, not a hardcoded mean.
    #
    # This slot used to be `ARLogitFuse()` -- constructed with no strategy and no
    # weights, so its body (`jnp.mean(logits, axis=0) + bias`) silently overrode
    # BOTH `strategy` and `state_weights` on the autoregressive path. Callers
    # asking for `multi_state_strategy="product"` with non-uniform
    # `state_weights` got an unweighted arithmetic mean and no error, while the
    # manifest recorded the requested strategy. Weighted product-of-experts
    # fusion therefore did not exist on the AR path at all, and a zero weight
    # did NOT drop its state (under a plain mean it contributes like any other).
    # `strategy_temperature` was lost the same way.
    #
    # Reusing the one instance -- rather than building a second strategy-aware AR
    # fuser -- is deliberate: it makes AR and non-AR fusion identical BY
    # CONSTRUCTION, so the two can never drift apart again. The strategy classes
    # are typed `Float[Array, "S ... V"]` with an optional trailing `bias`, which
    # is exactly the `(S, V) + (V,) -> (V,)` contract the AR kernel calls with,
    # so no adapter is needed.
    #
    # `ARLogitFuse` is kept and still exported: it remains the correct explicit
    # choice for an intentionally unweighted mean. It is simply no longer
    # substituted for whatever the caller actually configured.
    ar_logit_transform=logit_transform,
    decode_step=None,
    sample_step=None,
    tie_group_fuse=TieGroupProductOfExperts(),
    decoder_sink=decoder_sink,
  )


__all__ = [
  "LOGIT_STRATEGIES",
  "TIE_GROUP_STRATEGIES",
  "ARLogitFuse",
  "ArithmeticMeanLogits",
  "GeometricMeanLogits",
  "ProductOfProbabilities",
  "TieGroupLogsumexpMean",
  "TieGroupProductOfExperts",
  "make_stage_set",
]
