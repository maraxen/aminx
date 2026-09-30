"""Logit and score aggregation utilities for sampling output post-processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from jaxtyping import Array


def compute_pseudo_perplexity(
  sampled_logits: Array,
  sampled_sequences: Array,
  mask: Array | None = None,
) -> Array:
  """Compute pseudo-perplexity (negative log likelihood per residue).

  Pseudo-perplexity measures sequence likelihood by computing the average
  negative log-likelihood across masked residues. It provides a score
  for evaluating sampled sequence quality.

  Args:
    sampled_logits: Logits array with shape [batch, samples, noise, temp, seq_len, 21]
    sampled_sequences: One-hot encoded or integer sequences [batch, samples, noise, temp, seq_len]
    mask: Optional mask with shape [batch, seq_len]. If None, all residues are used.

  Returns:
    pseudo_perplexity: Array with shape [batch, samples, noise, temp] containing
      the exponentiated average negative log-likelihood per sequence.

  """
  one_hot_sequences = jax.nn.one_hot(sampled_sequences, num_classes=21)
  log_probs = jax.nn.log_softmax(sampled_logits, axis=-1)

  # Sum over sequence length (last 2 dims: one-hot @ log_softmax)
  # nll shape: [batch, samples, noise, temp]
  nll = -jnp.sum(one_hot_sequences * log_probs, axis=(-1, -2))

  # If mask not provided, assume all residues are valid
  if mask is None:
    # Assume the batch dimension is valid
    mask = jnp.ones((one_hot_sequences.shape[0], one_hot_sequences.shape[-2]), dtype=jnp.float32)

  # mask is [batch, seq_len]; sum over seq_len gives [batch]
  mask_sum = jnp.sum(mask, axis=-1)  # [batch]

  # Normalize by masked sum and broadcast to [batch, 1, 1, 1] for division
  pseudo_perplexity = jnp.exp(nll / mask_sum[:, None, None, None])

  return pseudo_perplexity


def pad_to_max(
  arr: jax.Array,
  target_len: int,
  axis: int = -1,
  pad_value: int = 0,
) -> jax.Array:
  """Pad the specified dimension of a JAX array to target_len.

  This utility is used to align logits and sequence arrays to the maximum
  sequence length before concatenation.

  Args:
    arr: JAX array to pad.
    target_len: Target length for the specified axis.
    axis: Axis along which to pad (default: -1, the last axis).
    pad_value: Value to use for padding (default: 0).

  Returns:
    Padded JAX array with shape [..., target_len, ...] along the specified axis.

  """
  diff = target_len - arr.shape[axis]
  if diff == 0:
    return arr
  padding_config = [(0, 0)] * arr.ndim
  # Handle negative axis
  axis = axis % arr.ndim
  padding_config[axis] = (0, diff)
  return jnp.pad(arr, padding_config, constant_values=pad_value)


def aggregate_logits(
  all_logits: list[jax.Array],
  max_len: int | None = None,
) -> jax.Array:
  """Aggregate logits from multiple batches by padding and concatenating.

  Args:
    all_logits: List of logit arrays, each with shape [batch, samples, noise, temp, seq_len, 21].
    max_len: Maximum sequence length. If None, computed from all_logits.

  Returns:
    Concatenated logits array with shape [total_batch, samples, noise, temp, seq_len, 21].

  """
  if not all_logits:
    return jnp.array([], dtype=jnp.float32)

  if max_len is None:
    max_len = max(arr.shape[-2] for arr in all_logits)

  # Pad logits along seq_len dimension (axis=-2)
  all_logits_padded = [pad_to_max(logits, max_len, axis=-2, pad_value=0) for logits in all_logits]

  return jnp.concatenate(all_logits_padded, axis=0)


def aggregate_pseudo_perplexities(
  all_pseudo_perplexities: list[jax.Array],
) -> jax.Array:
  """Aggregate pseudo-perplexity scores from multiple batches.

  Args:
    all_pseudo_perplexities: List of pseudo-perplexity arrays.

  Returns:
    Concatenated pseudo-perplexity array.

  """
  if not all_pseudo_perplexities:
    return jnp.array([], dtype=jnp.float32)

  return jnp.concatenate(all_pseudo_perplexities, axis=0)


class LogitFingerprint(NamedTuple):
  """Per-position EDA summary of a sampled-logits array, reduced over the sample axis.

  See `mpnn_ext/.praxia/docs/preregistration/260630_wave-color-scheduling-fundamental-tests.md`
  section 4 for the spec this implements (W0.1). All fields drop the sample axis from the
  input; remaining leading axes (batch, noise, temp, ...) are preserved.

  Attributes:
    mean_prob: Mean softmax probability per position, per amino acid. (..., L, 21)
    entropy_mean: Mean per-sample Shannon entropy (bits) per position. (..., L)
    entropy_std: Std of per-sample Shannon entropy (bits) per position. (..., L)
    top_k: Indices of the top-k amino acids by `mean_prob`. (..., L, k)
    top_k_prob: Probabilities of the top-k amino acids. (..., L, k)
    argmax: Most probable amino acid index per position. (..., L)
    confidence_mean: Mean per-sample max-probability per position. (..., L)
    confidence_std: Std of per-sample max-probability per position. (..., L)
    jsd_to_reference: Base-2 JSD of `mean_prob` against a reference marginal, when provided
      (zeros otherwise — the schedule-vs-schedule comparison happens downstream at W1a.3). (..., L)
    schedule_variance: Mean pairwise base-2 JSD between per-position marginals computed over
      `bootstrap_b` bootstrap sample-subsets (estimator m2 — pairwise, not to-grand-mean). (..., L)

  """

  mean_prob: Array
  entropy_mean: Array
  entropy_std: Array
  top_k: Array
  top_k_prob: Array
  argmax: Array
  confidence_mean: Array
  confidence_std: Array
  jsd_to_reference: Array
  schedule_variance: Array


def _jsd_base2(p: Array, q: Array) -> Array:
  """Base-2 Jensen-Shannon divergence between two categorical distributions (last axis)."""
  m = 0.5 * (p + q)

  def _kl_base2(a: Array, b: Array) -> Array:
    return jnp.sum(
      jax.scipy.special.xlogy(a, a) - jax.scipy.special.xlogy(a, b),
      axis=-1,
    ) / jnp.log(2.0)

  return 0.5 * _kl_base2(p, m) + 0.5 * _kl_base2(q, m)


def compute_logit_fingerprint(
  sampled_logits: Array,
  key: Array,
  *,
  sample_axis: int = 1,
  reference_mean_prob: Array | None = None,
  top_k: int = 3,
  bootstrap_b: int = 20,
  bootstrap_subset_size: int | None = None,
) -> LogitFingerprint:
  """Compute the `LogitFingerprint` EDA return (W0.1; prereg section 4).

  Args:
    sampled_logits: Logits, canonically `[batch, samples, noise, temp, L, 21]`; any
      array whose `sample_axis` indexes repeated draws (e.g. independent AR samples)
      and whose last axis is the 21 amino-acid classes.
    key: PRNG key used to draw the `bootstrap_b` bootstrap sample-subsets for
      `schedule_variance`.
    sample_axis: Axis over which samples are reduced. Default 1 (canonical shape order).
    reference_mean_prob: Optional reference marginal with the same shape as `mean_prob`
      (i.e. `sampled_logits` with `sample_axis` removed), used for `jsd_to_reference`.
      When None, `jsd_to_reference` is zeros (no reference comparison at this call site).
    top_k: Number of top amino acids to report per position.
    bootstrap_b: Number of bootstrap sample-subsets B for the `schedule_variance`
      estimator (m2). Pinned by the W0.1-GATE synthetic-invariant tests.
    bootstrap_subset_size: Size of each bootstrap subset. Defaults to
      `samples // 2` (must be >= 2 for a meaningful pairwise JSD).

  Returns:
    LogitFingerprint with all fields reduced over `sample_axis`.

  """
  probs = jax.nn.softmax(sampled_logits, axis=-1)
  n_samples = probs.shape[sample_axis]
  probs_moved = jnp.moveaxis(probs, sample_axis, 0)  # (N, ..., L, 21)

  mean_prob = jnp.mean(probs_moved, axis=0)

  entropy_per_sample = jnp.sum(jax.scipy.special.entr(probs_moved), axis=-1) / jnp.log(2.0)
  entropy_mean = jnp.mean(entropy_per_sample, axis=0)
  entropy_std = jnp.std(entropy_per_sample, axis=0)

  top_k_prob, top_k_idx = jax.lax.top_k(mean_prob, top_k)
  argmax = jnp.argmax(mean_prob, axis=-1)

  confidence_per_sample = jnp.max(probs_moved, axis=-1)
  confidence_mean = jnp.mean(confidence_per_sample, axis=0)
  confidence_std = jnp.std(confidence_per_sample, axis=0)

  if reference_mean_prob is None:
    jsd_to_reference = jnp.zeros_like(entropy_mean)
  else:
    jsd_to_reference = _jsd_base2(mean_prob, reference_mean_prob)

  subset_size = (
    bootstrap_subset_size if bootstrap_subset_size is not None else max(n_samples // 2, 2)
  )
  subset_keys = jax.random.split(key, bootstrap_b)

  def _subset_mean(subset_key: Array) -> Array:
    idx = jax.random.randint(subset_key, (subset_size,), 0, n_samples)
    return jnp.mean(jnp.take(probs_moved, idx, axis=0), axis=0)

  marginals = jax.vmap(_subset_mean)(subset_keys)  # (B, ..., L, 21)
  pair_i, pair_j = jnp.triu_indices(bootstrap_b, k=1)
  pairwise_jsd = jax.vmap(_jsd_base2)(marginals[pair_i], marginals[pair_j])  # (n_pairs, ..., L)
  schedule_variance = jnp.mean(pairwise_jsd, axis=0)

  return LogitFingerprint(
    mean_prob=mean_prob,
    entropy_mean=entropy_mean,
    entropy_std=entropy_std,
    top_k=top_k_idx,
    top_k_prob=top_k_prob,
    argmax=argmax,
    confidence_mean=confidence_mean,
    confidence_std=confidence_std,
    jsd_to_reference=jsd_to_reference,
    schedule_variance=schedule_variance,
  )
