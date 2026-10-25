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
from aminx.potts.model import PottsModel
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
  6. Refuses multi-state input: single-backbone inference only (see Raises)
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
      ValueError: If geometry extraction or model inference fails.
      NotImplementedError: If more than one backbone state is supplied, via a leading geometry axis
          of size > 1 or via ``spec.n_backbones != 1``. Single-backbone inference only. Raised
          before any weights are loaded.
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

  # Squeeze the single-state leading axis if present. S > 1 was already refused in step 0, so this
  # never discards a state the caller supplied.
  if coords.ndim == 4:  # (1, L, 4, 3)
    coords = coords[0]  # (L, 4, 3)
    mask = mask[0]  # (L,)
    residue_index = residue_index[0]  # (L,)
    chain_index = chain_index[0]  # (L,)

  # Ensure correct dtypes
  coords = jnp.asarray(coords, dtype=jnp.float32)
  mask = jnp.asarray(mask, dtype=jnp.float32)
  residue_index = jnp.asarray(residue_index, dtype=jnp.int32)
  chain_index = jnp.asarray(chain_index, dtype=jnp.int32)

  # Handle 37-atom format (if input is 4, pad to 37)
  if coords.shape[-2] == 4:
    # Pad from 4 atoms (N, CA, C, O) to 37 (standard pdb_atom37 format)
    n, _ = coords.shape[:2]
    coords_37 = jnp.zeros((n, 37, 3), dtype=jnp.float32)
    coords_37 = coords_37.at[:, :4, :].set(coords)
    coords = coords_37

  # Step 3.5: Compute k-NN graph features via ProteinFeatures
  # This integration layer responsibility: compute graph and pass to pure potts model
  features = ProteinFeatures(
    node_features=128,
    edge_features=128,
    k_neighbors=model.k_neighbors,
    key=key,
  )

  edge_knn, nei, _, _ = features(
    key,
    coords,
    mask,
    residue_index,
    chain_index,
    backbone_noise=None,
  )

  # Cast to JAX arrays to ensure proper types for infer_params
  edge_knn = jnp.asarray(edge_knn, dtype=jnp.float32)
  nei = jnp.asarray(nei, dtype=jnp.int32)

  # Step 4: Infer Potts parameters via model.infer_params
  params = model.infer_params(
    key=key,
    edge_knn=edge_knn,
    nei=nei,
    mask=mask,
  )

  # Step 5: Apply calibration to marginals
  calibrated_marginals = calibration(params.marginals)

  # Step 6: single backbone only. Multi-state input is refused in step 0 rather than silently
  # truncated, so these parameters always correspond to the one state that was actually given.

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
