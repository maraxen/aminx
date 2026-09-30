"""Potts model with differentiable TRW inference on k-NN graphs from ProteinFeatures.

Architecture: PottsModel is a parallel model family (NOT a StageSet consumer).
See ADR 260605_potts-parallel-not-stageset for design rationale.
This module is the TRW structure model (aminx.potts); PottsMPNN is aminx.families.potts_mpnn (ADR 260929_pottsmpnn-lasermpnn-family-drivers).

h and J carry a factor-of-2 from the directed-slot PottsMPNN convention
(counting each pairwise edge twice in the MRF). This factor is preserved
to maintain numerical consistency with weight recapture from mistypotts.
See etab_to_dense_h_j_w for reference.

Alphabet: Potts uses the canonical MPNN alphabet (q=21, identity mapping).
See ADR 260605_potts-alphabet-alignment for alignment details.

References
----------
.. [PottsMPNN] Birnbaum and Keating. "Beyond native sequence recovery: Improved
   modeling of the sequence-energy landscape of protein structures." bioRxiv, 2026.
   https://doi.org/10.64898/2026.01.14.699067
.. [TRW] Wainwright, M.J., Jaakkola, T.S., and Willsky, A.S. (2003).
   "Tree-reweighted belief propagation algorithms and approximate ML estimation
   by pseudo-moment matching." AISTATS 2003.
   https://doi.org/10.7551/mitpress/3968.003.0085
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from aminx.potts._trw import DifferentiableTRW
from aminx.potts._trw_spec import PottsTRWRunSpec

# Canonical Potts amino acid alphabet (20 standard amino acids + unknown).
# Standard ordering: A C D E F G H I K L M N P Q R S T V W Y X
# q=21 total (X represents unknown/gap indicator, aligning with MPNN alphabet)
POTTS_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
if len(POTTS_ALPHABET) != 21:
  raise ValueError(f"POTTS_ALPHABET must have length 21, got {len(POTTS_ALPHABET)}")

# Permutation array mapping Potts indices to MPNN indices.
# Since both use the same alphabet, this is the identity permutation.
# dtype=jnp.int32 is required for cross-platform type stability (see ADR 260605_potts-alphabet-alignment:76-78)
POTTS_TO_MPNN_ALPHABET_MAP = jnp.array(
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
  dtype=jnp.int32,
)


class PottsParams(NamedTuple):
  """Inferred parameters from PottsModel forward pass.

  Attributes:
      marginals: Node marginals (N, num_aa)
      h: Node unary potentials (N, num_aa) with x2 scale factor
      J: Pairwise potentials (N, N, num_aa, num_aa) with x2 scale factor
      rho: Tree-reweighting parameters (N, N)
      W: Graph adjacency matrix (N, N)
  """

  marginals: Float[Array, "n num_aa"]
  h: Float[Array, "n num_aa"]
  J: Float[Array, "n n num_aa num_aa"]
  rho: Float[Array, "n n"]
  W: Float[Array, "n n"]


class PottsModel(eqx.Module):
  """Potts MPNN with TRW inference head on k-NN geometric graphs.

  This model:
  - Builds k-NN edges from structure coordinates using ProteinFeatures
  - Projects edge features and node context → unary (h) and pairwise (J) potentials
  - Runs differentiable tree-reweighted message passing (TRW) for node marginals
  - Supports multiple rho backends and loop strategies via PottsTRWRunSpec

  All static fields (hidden_dim, num_aa, k_neighbors) must be eqx.field(static=True)
  to enable JIT compilation and PyTree filtering.

  k_neighbors is sourced from model metadata (checkpoint) and is never exposed
  as a constructor argument to users.
  """

  hidden_dim: int = eqx.field(static=True)
  num_aa: int = eqx.field(static=True)
  k_neighbors: int = eqx.field(static=True)

  # Learned parameters
  node_lin: eqx.nn.Linear
  project_h: eqx.nn.Linear
  project_j: eqx.nn.Linear

  # TRW inference engine (static config)
  trw: DifferentiableTRW = eqx.field(static=True)
  trw_spec: PottsTRWRunSpec = eqx.field(static=True)

  # Optional auxiliary features dimension
  aux_node_dim: int = eqx.field(static=True)

  # Optional suppression of pairwise couplings
  suppress_pairwise: bool = eqx.field(static=True)

  # Training mode flag for safety guards
  training: bool = eqx.field(static=True)

  def __init__(
    self,
    hidden_dim: int,
    num_aa: int,
    k_neighbors: int,
    edge_features_dim: int,
    trw_iters: int,
    *,
    key: PRNGKeyArray,
    trw_damping: float = 0.5,
    trw_rho_floor: float = 1e-6,
    aux_node_dim: int = 0,
    suppress_pairwise: bool = False,
    trw_spec: PottsTRWRunSpec | None = None,
    training: bool = False,
  ) -> None:
    """Initialize PottsModel.

    Args:
        hidden_dim: Dimension of hidden node representations
        num_aa: Number of amino acid types (q in TRW; typically 21)
        k_neighbors: Number of k-NN neighbors (read-only from metadata)
        edge_features_dim: Dimension of edge features from ProteinFeatures
        trw_iters: Number of TRW message-passing iterations
        key: JAX random key for parameter initialization
        trw_damping: Damping factor for TRW message updates [0, 1]
        trw_rho_floor: Minimum threshold for tree-reweighting parameters
        aux_node_dim: Optional auxiliary node feature dimension
        suppress_pairwise: If True, zero out all pairwise couplings
        trw_spec: PottsTRWRunSpec with backend and solver config
                 (default: PottsTRWRunSpec.default_dense())
        training: If True, model is in training mode (static field for safety guards)

    Raises:
        ValueError: If training=True and trw_spec.trw_loop='fori' (OOM risk).
    """
    _, k1, k2, k3 = jax.random.split(key, 4)

    self.hidden_dim = int(hidden_dim)
    self.num_aa = int(num_aa)
    self.k_neighbors = int(k_neighbors)
    self.aux_node_dim = int(aux_node_dim)
    self.suppress_pairwise = bool(suppress_pairwise)
    self.training = bool(training)

    # Projection layers
    node_in_dim = int(edge_features_dim) + self.aux_node_dim
    self.node_lin = eqx.nn.Linear(node_in_dim, self.hidden_dim, key=k1)
    self.project_h = eqx.nn.Linear(self.hidden_dim, self.num_aa, key=k2)

    # Pairwise projection: concatenate [h_i, h_j, edge_features]
    pair_in_dim = 2 * self.hidden_dim + int(edge_features_dim)
    self.project_j = eqx.nn.Linear(pair_in_dim, self.num_aa * self.num_aa, key=k3)

    # TRW engine
    self.trw_spec = PottsTRWRunSpec.default_dense() if trw_spec is None else trw_spec

    # Guard: prevent fori loop in training mode (OOM risk in reverse-mode autodiff)
    if self.training and self.trw_spec.trw_loop == "fori":
      msg = (
        "trw_loop=fori is unsafe for training: materialises all TRW intermediate states "
        "under reverse-mode autodiff causing OOM. Use trw_loop=scan with checkpoint_trw_step=True."
      )
      raise ValueError(msg)

    self.trw = DifferentiableTRW(
      q=self.num_aa,
      trw_iters=int(trw_iters),
      damping=float(trw_damping),
      rho_floor=float(trw_rho_floor),
      spec=self.trw_spec,
    )

  @staticmethod
  def build_adjacency_from_neighbors(
    nei: Int[Array, "n k"],
    mask: Float[Array, " n"],
  ) -> Float[Array, "n n"]:
    """Build k-NN adjacency matrix W from precomputed neighbor indices.

    Takes k-NN neighbor indices (computed externally via ProteinFeatures or similar)
    and builds a symmetric adjacency matrix with masking.

    Args:
        nei: Neighbor indices (N, k) as output from graph construction algorithm
        mask: Residue mask (N,) in [0, 1]; masked positions excluded from edges

    Returns:
        Adjacency matrix W (N, N) with symmetric edges and zero diagonal.
    """
    nei = jnp.asarray(nei, dtype=jnp.int32)
    mask_arr = jnp.asarray(mask, dtype=jnp.float32)

    n = nei.shape[0]
    row = jnp.arange(n, dtype=jnp.int32)[:, None]
    col = nei.astype(jnp.int32)
    valid = (mask_arr[row] > 0) & (mask_arr[col] > 0) & (col >= 0) & (col < n)
    w_vals = jnp.where(valid, 1.0, 0.0)
    w = jnp.zeros((n, n), dtype=jnp.float32)
    w = w.at[row, col].set(w_vals)
    w = jnp.maximum(w, w.T)
    return w.at[jnp.arange(n), jnp.arange(n)].set(0.0)

  def __call__(
    self,
    edge_knn: Float[Array, "n k e"],
    nei: Int[Array, "n k"],
    mask: Float[Array, " n"],
    key: PRNGKeyArray | None = None,
    aux_node_features: Array | None = None,
  ) -> tuple[Array, Array, Array, Array]:
    """Forward pass: extract Potts h/J from precomputed graph, run TRW.

    Args:
        edge_knn: Precomputed edge features (N, k, E) from k-NN graph construction
        nei: k-NN neighbor indices (N, k) from k-NN graph construction
        mask: Valid residue indicator (N,) in [0, 1]
        key: JAX random key (unused, for API compatibility)
        aux_node_features: Optional auxiliary features (N, aux_node_dim)

    Returns:
        tuple of four arrays:
        - marginals: Node marginals (N, num_aa)
        - h: Unary potentials (N, num_aa) with x2 scale factor
        - J: Pairwise potentials (N, N, num_aa, num_aa) with x2 scale factor
        - rho: Tree-reweighting parameters (N, N)

    Note:
        No batch axis. Caller applies eqx.filter_vmap for batching.
        Caller is responsible for computing edge_knn and nei via ProteinFeatures
        or similar k-NN graph construction method in the integration layer.
    """
    # Ensure inputs are JAX arrays with correct dtype
    edge_knn = jnp.asarray(edge_knn, dtype=jnp.float32)
    nei = jnp.asarray(nei, dtype=jnp.int32)
    mask_arr = jnp.asarray(mask, dtype=jnp.float32)

    n = edge_knn.shape[0]
    row = jnp.arange(n, dtype=jnp.int32)[:, None]
    col = nei.astype(jnp.int32)
    valid = (mask_arr[row] > 0) & (mask_arr[col] > 0) & (col >= 0) & (col < n)

    # Build adjacency matrix W from k-NN edges (reuse nei from ProteinFeatures)
    w_vals = jnp.where(valid, 1.0, 0.0)
    w = jnp.zeros((n, n), dtype=edge_knn.dtype)
    w = w.at[row, col].set(w_vals)
    w = jnp.maximum(w, w.T)
    w = w.at[jnp.arange(n), jnp.arange(n)].set(0.0)

    # Dense edge features: (N, N, E)
    e_dim = edge_knn.shape[-1]
    edge_dense = jnp.zeros((n, n, e_dim), dtype=edge_knn.dtype)
    edge_dense = edge_dense.at[row, col].set(jnp.where(valid[..., None], edge_knn, 0.0))
    edge_dense = 0.5 * (edge_dense + jnp.swapaxes(edge_dense, 0, 1))

    # Mean-pool edge features to node level
    pooled = jnp.mean(edge_knn, axis=1)

    # Node projection: [pooled_edges, aux_features] -> hidden
    if self.aux_node_dim > 0:
      if aux_node_features is None:
        msg = "aux_node_features required when aux_node_dim > 0"
        raise ValueError(msg)
      aux = jnp.asarray(aux_node_features, dtype=pooled.dtype)
      node_in = jnp.concatenate([pooled, aux], axis=-1)
    else:
      node_in = pooled

    # Apply node linear projection with ReLU
    node_h = jax.nn.relu(jax.vmap(self.node_lin)(node_in))

    # Unary potentials
    h = jax.vmap(self.project_h)(node_h)

    # Pairwise potentials: concatenate [h_i, h_j, edge_features] -> J
    hi = jnp.broadcast_to(node_h[:, None, :], (n, n, self.hidden_dim))
    hj = jnp.broadcast_to(node_h[None, :, :], (n, n, self.hidden_dim))
    pair = jnp.concatenate([hi, hj, edge_dense], axis=-1)
    j_raw = jax.vmap(jax.vmap(self.project_j))(pair).reshape(n, n, self.num_aa, self.num_aa)

    # Symmetrize J
    j_sym = 0.5 * (j_raw + jnp.transpose(j_raw, (1, 0, 3, 2)))
    j_sym = jnp.where(w[..., None, None] > 0, j_sym, 0.0)

    # Suppress pairwise couplings if requested
    if self.suppress_pairwise:
      j_sym = jnp.zeros_like(j_sym)

    # Run TRW inference
    marginals, rho = self.trw(w, j_sym, h)

    return marginals, h, j_sym, rho

  def infer_params(
    self,
    edge_knn: Float[Array, "n k e"],
    nei: Int[Array, "n k"],
    mask: Float[Array, " n"],
    key: PRNGKeyArray | None = None,
    aux_node_features: Array | None = None,
  ) -> PottsParams:
    """Infer Potts parameters and return as namedtuple.

    Runs TRW inference on precomputed k-NN graph and computes adjacency matrix W.

    Args:
        edge_knn: Precomputed edge features (N, k, E) from k-NN graph construction
        nei: k-NN neighbor indices (N, k) from k-NN graph construction
        mask: Residue mask (N,)
        key: JAX random key (unused, for API compatibility)
        aux_node_features: Optional auxiliary node features (N, aux_node_dim)

    Returns:
        PottsParams namedtuple with marginals, h, J, rho, W.
    """
    marginals, h, j_sym, rho = self(
      edge_knn,
      nei,
      mask,
      key=key,
      aux_node_features=aux_node_features,
    )

    # Build W adjacency using static helper
    w = self.build_adjacency_from_neighbors(nei, mask)

    return PottsParams(
      marginals=marginals,
      h=h,
      J=j_sym,
      rho=rho,
      W=w,
    )

  @staticmethod
  def log_prob(
    seq: Int[Array, " n"],
    h: Float[Array, "n q"],
    j: Float[Array, "n n q q"],
    w: Float[Array, "n n"],
    mask: Float[Array, " n"] | None = None,
  ) -> Array:
    """Pure function: log-probability of sequence under Potts energy.

    Computes:
        log P(seq) = h_seq + 0.5 * sum_ij W_ij * J_seq_seq - partition_normalization

    This is a pure JAX function with no model state dependency.
    It can be used as a scoring function in sampling algorithms.

    Args:
        seq: Sequence as amino acid indices (N,) in [0, q)
        h: Unary potentials (N, q) with x2 scale factor from PottsMPNN
        j: Pairwise potentials (N, N, q, q) with x2 scale factor
        w: Graph adjacency (N, N) binary or weighted edge indicator
        mask: Optional residue mask (N,). Masked positions (mask=0) contribute 0 to
              both unary and pairwise terms. Unmasked positions (mask>0) contribute
              normally. Allows scoring partial sequences (e.g., chain breaks).

    Returns:
        Scalar log-probability (unnormalized).
    """
    n = seq.shape[0]
    q = h.shape[-1]

    # Unary contribution with per-residue masking
    h_seq = jnp.take_along_axis(h, seq[..., None], axis=-1).squeeze(-1)
    if mask is not None:
      mask_arr = jnp.asarray(mask, dtype=h_seq.dtype)
      h_seq = h_seq * mask_arr
    unary_energy = jnp.sum(h_seq)

    # Pairwise contribution with per-position masking and x2 scale factor
    seq_i = seq[..., None, None]  # (n, 1, 1)
    seq_j = seq[None, ..., None]  # (1, n, 1)
    j_seq = jnp.take_along_axis(
      j.reshape(n, n, q * q),
      (seq_i * q + seq_j).reshape(n, n, 1),
      axis=-1,
    ).reshape(n, n)

    if mask is not None:
      mask_arr = jnp.asarray(mask, dtype=j_seq.dtype)
      mask_pair = jnp.outer(mask_arr, mask_arr)
      j_seq = j_seq * mask_pair

    pairwise_energy = 0.5 * jnp.sum(w * j_seq)

    # Total energy (log unnormalized probability)
    return unary_energy + pairwise_energy
