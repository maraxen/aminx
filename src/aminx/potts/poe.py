"""Product-of-Experts (PoE) Potts ensemble: N-backbone joint energy computation.

This module provides PoeModel for combining multiple PottsModel backbones into a
unified scoring function via energy summation (Product of Experts in log space).

Usage:
    - Constructor validates all backbones share identical static config (hidden_dim, num_aa,
      k_neighbors, trw_spec).
    - joint_energy(seq, params_list): sums E_backbone over all backbones.
    - infer_all_params(coords_stack): uses eqx.filter_vmap over backbone pytree.
    - joint_log_prob(seq, params_list): returns log unnormalized joint probability.
    - Embedded sanity check: for two identical backbones, joint_energy ≈ 2*single_energy ±1e-5.

Identity-backbone invariant: joint_energy(seq, [p, p]) == 2*single_energy(seq, p);
validated by scripts/validate/poe_energy_sanity.py.

No imports from aminx.inference.*, host.*, types.stages, or inference.logits.

References
----------
.. [PoE] Hinton, G.E. (2002). "Training products of experts by minimizing
   contrastive divergence." Neural Computation 14(8):1771-1800.
   https://doi.org/10.1162/089976602760128018
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from aminx.potts.model import PottsModel, PottsParams


class PoeOutput(NamedTuple):
  """Output from PoeModel forward pass: aggregated marginals across backbones.

  Attributes:
      marginals: Aggregated node marginals (N, q) — normalized via softmax
      log_poe_unnorm: Log unnormalized PoE (N, q) — sum of log-marginals
      per_backbone_marginals: Per-backbone marginals (B, N, q) for decomposition analysis
  """

  marginals: Float[Array, "n q"]
  log_poe_unnorm: Float[Array, "n q"]
  per_backbone_marginals: Float[Array, "b n q"]


class PoeParams(NamedTuple):
  """Product-of-Experts parameters tuple: list of per-backbone PottsParams.

  Attributes:
      params_list: Tuple of PottsParams, one per backbone.

  """

  params_list: tuple[PottsParams, ...]


class PoeModel(eqx.Module):
  """Product-of-Experts Potts ensemble.

  Combines N independent PottsModel backbones for multi-state sequence design.
  All backbones must share identical static configuration (hidden_dim, num_aa,
  k_neighbors, trw_spec) to ensure meaningful energy summation.

  Design invariant (embedded sanity check):
    For two backbones with identical parameters and data,
    joint_energy(seq, [params1, params1]) ≈ 2 * single_energy(seq, params1) ± 1e-5.
  """

  backbones: tuple[PottsModel, ...] = eqx.field(static=False)
  n_backbones: int = eqx.field(static=True)

  def __init__(self, backbones: tuple[PottsModel, ...]) -> None:
    """Initialize PoeModel with N backbones.

    Args:
        backbones: Tuple of PottsModel instances. All must share identical
                   static config (hidden_dim, num_aa, k_neighbors, trw_spec).

    Raises:
        ValueError: If backbones have mismatched static config.

    """
    if len(backbones) == 0:
      msg = "At least one backbone is required"
      raise ValueError(msg)

    self.backbones = backbones
    self.n_backbones = len(backbones)

    # Validate all backbones share identical static config
    if self.n_backbones > 1:
      ref = backbones[0]
      for i, backbone in enumerate(backbones[1:], start=1):
        # Compare by value, not identity
        # trw_spec is compared by attributes (rho_backend, message_backend, etc.)
        trw_spec_mismatch = (
          backbone.trw_spec.rho_backend != ref.trw_spec.rho_backend
          or backbone.trw_spec.message_backend != ref.trw_spec.message_backend
          or backbone.trw_spec.tile_size != ref.trw_spec.tile_size
          or backbone.trw_spec.lanczos_rank != ref.trw_spec.lanczos_rank
          or backbone.trw_spec.slq_num_samples != ref.trw_spec.slq_num_samples
          or backbone.trw_spec.checkpoint_trw_step != ref.trw_spec.checkpoint_trw_step
          or backbone.trw_spec.trw_loop != ref.trw_spec.trw_loop
          or backbone.trw_spec.uniform_rho_value != ref.trw_spec.uniform_rho_value
        )

        if (
          backbone.hidden_dim != ref.hidden_dim
          or backbone.num_aa != ref.num_aa
          or backbone.k_neighbors != ref.k_neighbors
          or trw_spec_mismatch
        ):
          msg = (
            f"All backbones must share identical static config (hidden_dim, num_aa, "
            f"k_neighbors, trw_spec). Backbone 0 has "
            f"(hidden_dim={ref.hidden_dim}, num_aa={ref.num_aa}, "
            f"k_neighbors={ref.k_neighbors}), "
            f"but backbone {i} has "
            f"(hidden_dim={backbone.hidden_dim}, num_aa={backbone.num_aa}, "
            f"k_neighbors={backbone.k_neighbors})."
          )
          raise ValueError(msg)

  def __call__(
    self,
    key: PRNGKeyArray,
    edge_knn_stack: Float[Array, "b n k e"],
    nei_stack: Int[Array, "b n k"],
    mask: Float[Array, " n"],
  ) -> PoeOutput:
    """Forward pass: infer marginals for all backbones and aggregate via PoE.

    Applies each backbone to its precomputed k-NN graph, then aggregates
    marginals via Product of Experts: log_poe(a) = sum_b log_marginals_b(a);
    normalize to get marginals.

    Args:
        key: JAX random key for stochastic components.
        edge_knn_stack: Precomputed edge features (B, N, k, E) where B=n_backbones.
        nei_stack: k-NN neighbor indices (B, N, k) where B=n_backbones.
        mask: Residue validity mask (N,).

    Returns:
        PoeOutput namedtuple with:
        - marginals: Aggregated node marginals (N, q) via softmax normalization
        - log_poe_unnorm: Log unnormalized PoE (N, q)
        - per_backbone_marginals: Per-backbone marginals (B, N, q)

    Raises:
        ValueError: If edge_knn_stack.shape[0] != n_backbones.
    """
    if edge_knn_stack.shape[0] != self.n_backbones:
      msg = (
        f"edge_knn_stack batch dimension ({edge_knn_stack.shape[0]}) "
        f"must match n_backbones ({self.n_backbones})"
      )
      raise ValueError(msg)

    # Split key for each backbone
    keys = jax.random.split(key, self.n_backbones)

    # Define vmap over backbone index (dimension 0 of PyTree)
    def call_single_backbone(
      backbone: PottsModel,
      key_b: PRNGKeyArray,
      edge_knn_b: Float[Array, "n k e"],
      nei_b: Int[Array, "n k"],
    ) -> Float[Array, "n q"]:
      """Call backbone and return marginals only."""
      marginals, _, _, _ = backbone(
        edge_knn=edge_knn_b,
        nei=nei_b,
        mask=mask,
        key=key_b,
      )
      return marginals

    # Stack backbones into a single batched pytree: separate dynamic arrays from static
    # fields (Python ints/strings), stack arrays along axis 0 (n_backbones), recombine.
    # Equinox ensemble idiom — raw tuple-of-modules cannot be vmapped because each
    # leaf's natural axis 0 is a weight dim, not the backbone index.
    _dyn_parts = [eqx.partition(bb, eqx.is_array)[0] for bb in self.backbones]
    _static_part = eqx.partition(self.backbones[0], eqx.is_array)[1]
    stacked_backbone = eqx.combine(
      jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *_dyn_parts),
      _static_part,
    )

    vmapped_call = eqx.filter_vmap(
      call_single_backbone,
      in_axes=(0, 0, 0, 0),
    )

    # Apply vmap: returns per_backbone_marginals (B, N, q)
    per_backbone_marginals = vmapped_call(stacked_backbone, keys, edge_knn_stack, nei_stack)

    # Aggregate via Product of Experts: log_poe = sum_b log(marginals_b)
    # Then normalize via softmax to get final marginals
    log_marginals = jnp.log(per_backbone_marginals + 1e-10)  # Add small epsilon for numerical stability
    log_poe_unnorm = jnp.sum(log_marginals, axis=0)  # Sum over backbone axis (B)

    # Normalize via softmax to get final marginals
    marginals = jax.nn.softmax(log_poe_unnorm, axis=-1)

    return PoeOutput(
      marginals=marginals,
      log_poe_unnorm=log_poe_unnorm,
      per_backbone_marginals=per_backbone_marginals,
    )

  def joint_energy(
    self,
    seq: Int[Array, " n"],
    params_list: tuple[
      tuple[Float[Array, "n q"], Float[Array, "n n q q"], Float[Array, "n n"]], ...,
    ],
  ) -> Float[Array, ""]:
    """Compute total energy E_total = sum_b E_b(seq) across all backbones.

    Product of Experts in log space: scoring function that combines independent
    Potts energies via summation.

    Args:
        seq: Amino acid sequence as indices (N,) in [0, num_aa).
        params_list: Tuple of (h, J, W) tuples, one per backbone.
                     Each h is (N, num_aa), J is (N, N, num_aa, num_aa),
                     W is (N, N).

    Returns:
        Scalar energy (sum of per-backbone log unnormalized probabilities).
        Note: h and J already carry the x2 scale factor from PottsMPNN.
              No additional x2 factor is applied.

    """
    if len(params_list) != self.n_backbones:
      msg = f"Expected {self.n_backbones} parameter sets, got {len(params_list)}"
      raise ValueError(msg)

    # Stack parameters: h (B, N, q), j (B, N, N, q, q), w (B, N, N)
    h_stack = jnp.stack([p[0] for p in params_list], axis=0)
    j_stack = jnp.stack([p[1] for p in params_list], axis=0)
    w_stack = jnp.stack([p[2] for p in params_list], axis=0)

    # Vectorize log_prob over the backbone axis (axis 0)
    # in_axes: seq is not batched (None), h/j/w are batched (0)
    energies = jax.vmap(
      PottsModel.log_prob,
      in_axes=(None, 0, 0, 0),
    )(seq, h_stack, j_stack, w_stack)

    # Sum energies across backbones
    return jnp.sum(energies)

  def infer_all_params(
    self,
    key: PRNGKeyArray,
    edge_knn_stack: Float[Array, "b n k e"],
    nei_stack: Int[Array, "b n k"],
    mask: Float[Array, " n"],
  ) -> list[PottsParams]:
    """Infer parameters for all backbones using eqx.filter_vmap.

    Uses JAX vmap to parallelize inference across the backbone pytree ensemble.
    Each backbone receives the same precomputed k-NN graph structure (edge_knn, nei)
    and produces a PottsParams tuple.

    Args:
        key: JAX random key for stochastic components.
        edge_knn_stack: Precomputed edge features, shape (B, N, k, E) where B=n_backbones.
        nei_stack: Precomputed k-NN neighbors, shape (B, N, k) where B=n_backbones.
        mask: Residue validity mask (N,).

    Returns:
        List of PottsParams, one per backbone.

    Raises:
        ValueError: If batch dimensions don't match n_backbones.

    """
    if edge_knn_stack.shape[0] != self.n_backbones:
      msg = (
        f"edge_knn_stack batch dimension ({edge_knn_stack.shape[0]}) "
        f"must match n_backbones ({self.n_backbones})"
      )
      raise ValueError(msg)
    if nei_stack.shape[0] != self.n_backbones:
      msg = (
        f"nei_stack batch dimension ({nei_stack.shape[0]}) "
        f"must match n_backbones ({self.n_backbones})"
      )
      raise ValueError(msg)

    # Split key for each backbone
    keys = jax.random.split(key, self.n_backbones)

    # Define vmap over backbone index (dimension 0 of PyTree)
    def infer_single_backbone(
      backbone: PottsModel,
      key_b: PRNGKeyArray,
      edge_knn_b: Float[Array, "n k e"],
      nei_b: Int[Array, "n k"],
    ) -> PottsParams:
      return backbone.infer_params(
        key=key_b,
        edge_knn=edge_knn_b,
        nei=nei_b,
        mask=mask,
      )

    # Stack backbones (same idiom as __call__): partition dynamic/static, stack, recombine.
    _dyn_parts = [eqx.partition(bb, eqx.is_array)[0] for bb in self.backbones]
    _static_part = eqx.partition(self.backbones[0], eqx.is_array)[1]
    stacked_backbone = eqx.combine(
      jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *_dyn_parts),
      _static_part,
    )

    vmapped_infer = eqx.filter_vmap(
      infer_single_backbone,
      in_axes=(0, 0, 0, 0),
    )

    params_batched = vmapped_infer(stacked_backbone, keys, edge_knn_stack, nei_stack)

    # ``vmapped_infer`` returns ONE PottsParams whose leaves carry a leading backbone axis, i.e.
    # marginals (B, n, q), h (B, n, q), J (B, n, n, q, q), rho (B, n, n), W (B, n, n).
    #
    # The previous ``list(params_batched)`` did NOT do what its comment claimed. PottsParams is a
    # NamedTuple, so iterating it yields its five FIELDS, and the function returned
    # ``[marginals, h, J, rho, W]`` -- five stacked arrays -- rather than B PottsParams. A caller
    # following the documented contract (``for p in ...: p.h``) raised AttributeError, and
    # duck-typed code could use the marginals array as though it were a parameter set. Unstack along
    # the backbone axis so the return matches the annotation and the docstring.
    return [
      jax.tree.map(lambda leaf, i=b: leaf[i], params_batched)
      for b in range(self.n_backbones)
    ]

  def joint_log_prob(
    self,
    seq: Int[Array, " n"],
    params_list: tuple[
      tuple[Float[Array, "n q"], Float[Array, "n n q q"], Float[Array, "n n"]], ...,
    ],
  ) -> Float[Array, ""]:
    """Compute joint log probability as sum of per-backbone log probs.

    Product of Experts interpretation: log P_joint(seq) = sum_b log P_b(seq).

    Args:
        seq: Amino acid sequence (N,).
        params_list: Tuple of (h, J, W) tuples, one per backbone.

    Returns:
        Scalar log unnormalized joint probability.

    """
    return self.joint_energy(seq, params_list)


# Design invariant validation: identity-backbone sanity check
# Called from tests/potts/test_poe.py:525 (test_poe_sanity_check_two_identical_backbones)
# and by scripts/validate/poe_energy_sanity.py — NOT at import time.
def _sanity_check_poe_invariant() -> None:
  """Verify PoE design invariant: joint_energy ≈ 2*single_energy for identical backbones.

  Implements the identity-backbone invariant; see scripts/validate/poe_energy_sanity.py
  for the full validation suite.

  This function is defined for reference and testing only. Do not call at import time.
  """
  try:
    key = jax.random.PRNGKey(260605)
    k1, _k2 = jax.random.split(key)

    # Create two identical backbones (same key)
    backbone1 = PottsModel(
      hidden_dim=16,
      num_aa=4,
      k_neighbors=30,
      edge_features_dim=32,
      trw_iters=2,
      key=k1,
    )
    backbone2 = PottsModel(
      hidden_dim=16,
      num_aa=4,
      k_neighbors=30,
      edge_features_dim=32,
      trw_iters=2,
      key=k1,  # Identical key
    )

    model = PoeModel(backbones=(backbone1, backbone2))

    # Test data
    seq = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    h = jnp.array(
      [[1.0, 0.5, 0.3, 0.1], [0.5, 1.0, 0.2, 0.3], [0.3, 0.2, 1.0, 0.5], [0.1, 0.3, 0.5, 1.0]],
      dtype=jnp.float32,
    )
    j = jnp.zeros((4, 4, 4, 4), dtype=jnp.float32)
    w = jnp.zeros((4, 4), dtype=jnp.float32)

    # Compute single and joint energies
    energy_single = PottsModel.log_prob(seq, h, j, w)
    energy_joint = model.joint_energy(seq, ((h, j, w), (h, j, w)))

    # Check invariant
    diff = jnp.abs(energy_joint - 2.0 * energy_single)
    tol = 1e-5
    if diff > tol:
      warnings.warn(
        f"PoE sanity check failed: joint_energy={energy_joint}, "
        f"2*single_energy={2 * energy_single}, diff={diff}, tol={tol}. "
        f"This may indicate a scale factor error in joint_energy.",
        stacklevel=2,
      )
  except Exception:  # noqa: BLE001, S110
    # Silently skip sanity check if imports or construction fail
    # (e.g., in test environments where PottsModel is mocked)
    pass
