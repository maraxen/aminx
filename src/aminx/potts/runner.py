"""Potts model runner: orchestrate weight loading, calibration, and inference.

This module provides the high-level run_potts function that:
1. Loads PottsModel weights from spec.weights_path
2. Loads CalibrationModule (IdentityCalibration if caliby_path=None)
3. Extracts coordinates and masks from GeometryBundle
4. Infers Potts parameters via model.infer_params
5. Applies post-hoc calibration to marginals
6. Optionally uses PoeModel for multi-backbone ensembles
7. Returns PottsResult with all outputs

Architecture: PottsModel is PARALLEL, NOT a StageSet consumer.
See ADR 260605_potts-parallel-not-stageset for design rationale.

Method names: infer_params/sample/score (NOT encode/decode).
PottsResult is a dataclass, not an eqx.Module.

No imports from aminx.inference.decode, aminx.host.plan, aminx.types.stages,
or aminx.inference.logits.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import zstandard as zstd
from jaxtyping import Array, PRNGKeyArray

from aminx.model import ProteinFeatures
from aminx.potts.calibration import load_calibration
from aminx.potts.model import PottsModel, PottsParams
from aminx.potts.poe import PoeModel
from aminx.potts.sampling import gibbs_sweep, parallel_tempering
from aminx.potts.spec import PottsRunSpec  # noqa: TC001
from aminx.types.bundles import GeometryBundle  # noqa: TC001


@dataclass(frozen=False)
class PottsResult:
  """Result dataclass (not eqx.Module) containing Potts inference outputs.

  Attributes:
      marginals: Node marginals (N, 21) after optional calibration
      h: Unary potentials (N, 21) with x2 scale factor from PottsMPNN
      J: Pairwise potentials (N, N, 21, 21) with x2 scale factor
      rho: Tree-reweighting parameters (N, N)
      calibrated_marginals: Calibrated marginals (N, 21) after CalibrationModule
      n_backbones: Number of backbones in ensemble
      samples: Sampled sequences (n_samples, N) with dtype int32. None if n_samples=0.
      W: Graph adjacency matrix (N, N) built from the k-NN edges. run_potts always
          populates it; it is optional only so that existing direct constructions of
          this dataclass keep working. Needed alongside h and J to score a sequence
          with sampling.log_energy, which takes w as its contact weights.
  """

  marginals: Array
  h: Array
  J: Array
  rho: Array
  calibrated_marginals: Array
  n_backbones: int
  samples: Array | None = None
  W: Array | None = None


def _load_potts_model(weights_path: str, spec: PottsRunSpec, key: PRNGKeyArray) -> PottsModel:
  """Load PottsModel from checkpoint via eqx.tree_deserialise_leaves.

  Args:
      weights_path: Path to checkpoint file (bare .eqx or .eqx.zst).
      spec: PottsRunSpec with model hyperparameters.
      key: JAX random key for skeleton initialization.

  Returns:
      Loaded PottsModel with weights deserialized.

  Raises:
      FileNotFoundError: If checkpoint does not exist.
      ValueError: If k_neighbors is invalid or model construction fails.
  """
  # Validate k_neighbors before building skeleton
  if spec.k_neighbors <= 0:
    msg = (
      f"PottsRunSpec.k_neighbors={spec.k_neighbors} is invalid. "
      "Supply k_neighbors from checkpoint metadata (logged by pottsmpnn_to_eqx.py --dry-run)."
    )
    raise ValueError(msg)

  weights_path_obj = Path(weights_path)
  if not weights_path_obj.exists():
    msg = f"Weights checkpoint not found: {weights_path}"
    raise FileNotFoundError(msg)

  # Construct skeleton with hyperparameters from spec
  skeleton = PottsModel(
    hidden_dim=128,  # Standard Potts backbone dimension
    num_aa=21,  # Standard amino acid alphabet size
    k_neighbors=spec.k_neighbors,
    edge_features_dim=128,  # Standard edge feature dimension
    trw_iters=15,  # Standard TRW iterations
    key=key,
    trw_spec=spec.trw_spec,
    training=spec.training,
  )

  # Load weights via zstd decompression if needed
  if weights_path.endswith(".zst"):
    with Path(weights_path).open("rb") as f:
      dctx = zstd.ZstdDecompressor()
      decompressed = dctx.decompress(f.read())
      stream = io.BytesIO(decompressed)  # ty: ignore[unresolved-attribute]
      loaded = eqx.tree_deserialise_leaves(stream, skeleton)
  else:
    loaded = eqx.tree_deserialise_leaves(weights_path, skeleton)

  return loaded


def run_potts(
  spec: PottsRunSpec,
  geometry: GeometryBundle,
  key: PRNGKeyArray,
  config: dict | None = None,
) -> PottsResult:
  """Execute Potts inference: load model, extract geometry, infer params, calibrate.

  High-level orchestration function that:
  1. Loads PottsModel weights from spec.weights_path
  2. Loads CalibrationModule (IdentityCalibration if caliby_path=None)
  3. Extracts (coords, mask, residue_index, chain_index) from GeometryBundle
  4. Runs model.infer_params to compute Potts parameters
  5. Applies calibration to marginals
  6. For more than one backbone state, aggregates via PoeModel (product of experts)
  7. Returns PottsResult with marginals, h, J, rho, calibrated_marginals

  Args:
      spec: PottsRunSpec with weights_path, caliby_path, k_neighbors, training flag.
      geometry: GeometryBundle with coords, mask, residue_index, chain_index.
      key: JAX random key for stochastic components.
      config: Optional configuration dict (reserved for future extensions).

  Returns:
      PottsResult with all Potts outputs and calibrated marginals.

  Raises:
      FileNotFoundError: If weights_path or caliby_path (if not None) does not exist.
      ValueError: If geometry extraction or model inference fails, or if ``spec.n_backbones``
          disagrees with the number of states actually supplied in ``geometry``.
  """
  if config is None:
    config = {}

  # Step 0: refuse multi-state input BEFORE loading anything.
  #
  # This function infers Potts parameters from ONE backbone. Previously a leading geometry axis of
  # size S > 1 was silently reduced to state 0 while the returned PottsResult still carried
  # ``n_backbones=spec.n_backbones``, so a caller asking for an ensemble got single-backbone
  # parameters in an object that claimed otherwise, with nothing in the result to reveal it.
  # Multi-state fusion (PoeModel) is not implemented here; infer per state and fuse explicitly.
  _coords = geometry.coords
  if getattr(_coords, "ndim", 0) == 4 and int(_coords.shape[0]) != 1:
    msg = (
      f"run_potts received {int(_coords.shape[0])} backbone states, but multi-state Potts inference "
      f"is not implemented here. Earlier versions silently used state 0 and still reported "
      f"n_backbones={int(_coords.shape[0])}, so the result misrepresented what was computed. Pass a "
      f"single state, or call run_potts per state and fuse the returned parameters explicitly."
    )
    raise NotImplementedError(msg)
  if spec.n_backbones != 1:
    msg = (
      f"spec.n_backbones={spec.n_backbones}, but run_potts performs single-backbone inference and "
      f"would return a PottsResult reporting an ensemble size it never computed. Set n_backbones=1."
    )
    raise NotImplementedError(msg)

  # Step 1: Load PottsModel
  model = _load_potts_model(spec.weights_path, spec, key)

  # Step 2: Load CalibrationModule (identity if caliby_path is None)
  calibration = load_calibration(spec.caliby_path)

  # Step 3: Extract geometry arrays using GeometryBundle accessors
  # GeometryBundle has fields: coords, mask, residue_index, chain_index
  coords = geometry.coords  # (S, L, 4, 3) if S > 1, else (L, 4, 3) depending on implementation
  mask = geometry.mask  # (S, L) or (L)
  residue_index = geometry.residue_index  # (S, L) or (L)
  chain_index = geometry.chain_index  # (S, L) or (L)

  # Handle batched geometry. A leading axis of size S means S backbone STATES.
  #
  # Earlier versions reduced this to ``coords[0]`` while still returning ``n_backbones=spec.n_backbones``,
  # so a caller asking for an ensemble received single-backbone parameters inside a result that claimed
  # otherwise, with nothing in the result to reveal it. All S states are now used (step 6).
  n_states = 1
  if coords.ndim == 4:  # (S, L, 4, 3)
    n_states = int(coords.shape[0])
    if n_states == 1:
      coords = coords[0]  # (L, 4, 3)
      mask = mask[0]  # (L,)
      residue_index = residue_index[0]  # (L,)
      chain_index = chain_index[0]  # (L,)

  # ``spec.n_backbones`` is copied verbatim into the result, so it must describe what is actually computed.
  if spec.n_backbones != n_states:
    msg = (
      f"spec.n_backbones={spec.n_backbones} but geometry supplies {n_states} backbone state(s); the "
      f"returned PottsResult would report an ensemble size it did not compute. Make them agree."
    )
    raise ValueError(msg)

  # Ensure correct dtypes
  coords = jnp.asarray(coords, dtype=jnp.float32)
  mask = jnp.asarray(mask, dtype=jnp.float32)
  residue_index = jnp.asarray(residue_index, dtype=jnp.int32)
  chain_index = jnp.asarray(chain_index, dtype=jnp.int32)

  # Handle 37-atom format (if input is 4, pad to 37)
  if coords.shape[-2] == 4:
    # Pad from 4 atoms (N, CA, C, O) to 37 (standard pdb_atom37 format). Written over the TRAILING
    # two axes so it holds for a single state (L, 4, 3) and for a multi-state stack (S, L, 4, 3)
    # alike; the earlier form indexed coords.shape[:2] and silently assumed the single-state rank.
    coords_37 = jnp.zeros((*coords.shape[:-2], 37, 3), dtype=jnp.float32)
    coords = coords_37.at[..., :4, :].set(coords)

  # Step 3.5: Compute k-NN graph features via ProteinFeatures.
  # Each STATE gets its own graph: the k-nearest-neighbour set moves with the backbone, so the
  # neighbour indices cannot be computed once and reused across states.
  features = ProteinFeatures(
    node_features=128,
    edge_features=128,
    k_neighbors=model.k_neighbors,
    key=key,
  )

  def _graph_of(coords_s: Array, mask_s: Array, res_s: Array, chain_s: Array) -> tuple[Array, Array]:
    edge_knn_s, nei_s, _, _ = features(
      key,
      coords_s,
      mask_s,
      res_s,
      chain_s,
      backbone_noise=None,
    )
    return jnp.asarray(edge_knn_s, dtype=jnp.float32), jnp.asarray(nei_s, dtype=jnp.int32)

  if n_states == 1:
    edge_knn, nei = _graph_of(coords, mask, residue_index, chain_index)

    # Step 4: Infer Potts parameters via model.infer_params
    params = model.infer_params(
      key=key,
      edge_knn=edge_knn,
      nei=nei,
      mask=mask,
    )
    poe_marginals = params.marginals
  else:
    # Step 4 (multi-state): one graph per state, then PoeModel over S copies of this model.
    mask_0 = mask[0]
    graphs = [
      _graph_of(coords[s], mask[s], residue_index[s], chain_index[s]) for s in range(n_states)
    ]
    edge_knn_stack = jnp.stack([g[0] for g in graphs], axis=0)
    nei_stack = jnp.stack([g[1] for g in graphs], axis=0)

    poe = PoeModel(tuple(model for _ in range(n_states)))
    params_list = poe.infer_all_params(
      key=key,
      edge_knn_stack=edge_knn_stack,
      nei_stack=nei_stack,
      mask=mask_0,
    )
    # Product of experts over the marginals, derived from the parameters just inferred rather than
    # from a second forward pass: PoeModel.__call__ would re-run every backbone to recompute exactly
    # these marginals. Identical arithmetic to PoeOutput.marginals, one inference instead of two.
    log_poe = sum(jnp.log(p.marginals + 1e-10) for p in params_list)
    poe_marginals = jax.nn.softmax(log_poe, axis=-1)

    # Step 5 (multi-state): aggregate the Potts parameters so that the returned (h, J, W) reproduce
    # the EXACT product-of-experts energy under sampling.log_energy, whose pairwise term is
    #     0.5 * sum_ik w[i,k] * J[i,k,a,b]
    # and is therefore linear in the product w (*) J. Summing w and J independently would be WRONG,
    # because each state has its own contact graph. Writing E_poe = sum_s E_s:
    #     h_poe = sum_s h_s
    #     W_poe = sum_s W_s                      (retains interpretable contact structure)
    #     J_poe = (sum_s W_s (*) J_s) / W_poe    (zero where no state has that edge)
    # so that W_poe (*) J_poe == sum_s (W_s (*) J_s) identically.
    h_poe = sum(p.h for p in params_list)
    w_sum = sum(p.W for p in params_list)
    wj_sum = sum(p.W[:, :, None, None] * p.J for p in params_list)
    safe_w = jnp.where(w_sum > 0, w_sum, 1.0)
    j_poe = jnp.where(w_sum[:, :, None, None] > 0, wj_sum / safe_w[:, :, None, None], 0.0)
    rho_poe = sum(p.rho for p in params_list) / float(n_states)  # descriptive mean; TRW parameter

    params = PottsParams(
      marginals=poe_marginals,
      h=h_poe,
      J=j_poe,
      rho=rho_poe,
      W=w_sum,
    )
    mask = mask_0

  # Step 5: Apply calibration to marginals
  calibrated_marginals = calibration(poe_marginals)

  # Step 6: multi-state aggregation is done above; the parameters below are the PoE ensemble's.

  # Step 7: Sampling (if n_samples > 0)
  samples: Array | None = None
  if spec.n_samples > 0:
    # Prepare mask and weight matrix for sampling
    mask_sample = jnp.asarray(mask, dtype=jnp.float32)
    w = params.W if hasattr(params, "W") else jnp.ones_like(params.rho)

    # Initialize sequence at argmax of marginals
    seq_init = jnp.argmax(params.marginals, axis=1).astype(jnp.int32)

    if spec.sampler_type == "gibbs":
      # Gibbs sampler: run multiple independent chains and collect samples
      all_samples = []
      for _ in range(spec.n_chains):
        key_chain, key = jax.random.split(key)
        seq_current = seq_init

        for _ in range(spec.n_samples):
          key_sweep, key_chain = jax.random.split(key_chain)
          seq_current, _ = gibbs_sweep(
            key_sweep,
            seq_current,
            h=params.h,
            j=params.J,
            w=w,
            mask=mask_sample,
          )
          all_samples.append(seq_current)

      # Stack all samples: shape (n_chains * n_samples, N)
      samples = jnp.stack(all_samples, axis=0).astype(jnp.int32)

    elif spec.sampler_type == "pt":
      # Parallel tempering: single batch of samples from cold replica
      seqs_final, _ = parallel_tempering(
        key,
        seq_init,
        h=params.h,
        j=params.J,
        w=w,
        mask=mask_sample,
        n_replicas=spec.n_chains,
        n_sweeps=spec.n_samples,
      )
      # Return samples from cold replica (last one)
      samples = seqs_final[-1:, :].astype(jnp.int32)

  # Step 8: Return PottsResult
  return PottsResult(
    marginals=params.marginals,
    h=params.h,
    J=params.J,
    rho=params.rho,
    calibrated_marginals=calibrated_marginals,
    n_backbones=spec.n_backbones,
    samples=samples,
    W=params.W,
  )
