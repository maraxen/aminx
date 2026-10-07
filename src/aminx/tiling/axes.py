"""Canonical AxisSpec registry for all sampling/batching dispatch axes.

ALL_AXES ordering (innermost first, outermost last) is the demotion-priority
contract for xtrax's joint-budget BatchPlanner (EPIC #1541 T-PLANNER):
candidates are demoted from Vmap to SafeMap strictly in the order specs are
given. This list's order IS that contract -- there is no separate axis_index
field (xtrax.tiling.AxisSpec doesn't have one); reordering this list changes
demotion priority.

  0: n_residues       — residue dimension within a computation
  1: n_ligand_atoms   — per-residue atom count
  2: n_states         — multistate stack (heterogeneous)
  3: n_structures     — batch of proteins (heterogeneous)
  4: n_samples        — sample sweep
  5: n_temperatures   — temperature sweep
  6: n_noises         — backbone noise sweep
  7: n_jacobian_pairs — residue-pair products (deferred)
  8: n_combine        — multistate combine (deferred)
  9: n_apc_pairs      — all-pair contact scoring (deferred)
  10: n_replicates    — backbone-noise-keyed replicate draws (encode axis, inspect/score)
  11: n_candidates    — externally provided candidate sequences (decode axis, inspect/score)
"""

from xtrax.tiling import AxisSpec

# Residue/position dimension within a single structure. Fixed per batch by runner length bucketing (xtrax BUCKET_LADDER rung).
N_RESIDUES = AxisSpec(
  name="n_residues",
  cardinality=1200,
  default_batch_size=0,
  tile_granularity=128,
  heterogeneous=False,
)

# Ligand atom dimension (ligand_mpnn.py:437 triple-vmap). Fixed per structure.
N_LIGAND_ATOMS = AxisSpec(
  name="n_ligand_atoms",
  cardinality=64,
  default_batch_size=0,
  tile_granularity=1,
  heterogeneous=False,
)

# Multistate stack axis (ProteinBundle.n_states). Shapes vary across states.
N_STATES = AxisSpec(
  name="n_states",
  cardinality=64,
  default_batch_size=1,  # tile_granularity: iterate one element at a time
  tile_granularity=1,
  heterogeneous=True,
)

# Batch of protein structures (SamplingSpecification.batch_size). Lengths vary before runner length bucketing.
N_STRUCTURES = AxisSpec(
  name="n_structures",
  cardinality=32,
  default_batch_size=1,  # tile_granularity: iterate one element at a time
  tile_granularity=1,
  heterogeneous=True,
  dedup_eligible=True,  # repeated backbone structures can be deduplicated
)

# Sequence sample sweep (SamplingSpecification.samples_batch_size, samples_chunk_size).
N_SAMPLES = AxisSpec(
  name="n_samples",
  cardinality=128,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
)

# Temperature sweep axis (SamplingSpecification.temperature_batch_size).
N_TEMPERATURES = AxisSpec(
  name="n_temperatures",
  cardinality=8,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
)

# Backbone noise sweep axis (SamplingSpecification.noise_batch_size).
N_NOISES = AxisSpec(
  name="n_noises",
  cardinality=8,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
)

# Residue-pair axis for Jacobian computation (SamplingSpecification.jacobian_batch_size). DEFERRED.
N_JACOBIAN_PAIRS = AxisSpec(
  name="n_jacobian_pairs",
  cardinality=10000,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
)

# Multistate combine step (SamplingSpecification.combine_batch_size). DEFERRED.
N_COMBINE = AxisSpec(
  name="n_combine",
  cardinality=64,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
)

# All-pair contact scoring (SamplingSpecification.apc_batch_size, apc_residue_batch_size). DEFERRED.
N_APC_PAIRS = AxisSpec(
  name="n_apc_pairs",
  cardinality=10000,
  default_batch_size=1,
  tile_granularity=1,
  heterogeneous=False,
)

# Replicate draws at fixed backbone-noise magnitude, distinct PRNG keys
# (make_batched_conditional_logits_split_fn encode axis). Distinct from n_noises,
# which sweeps the noise magnitude itself.
N_REPLICATES = AxisSpec(
  name="n_replicates",
  cardinality=256,
  default_batch_size=0,
  tile_granularity=1,
  heterogeneous=False,
)

# Externally provided candidate sequences for teacher-forced scoring/inspection
# (make_batched_conditional_logits_split_fn decode axis). Keyless/deterministic at
# the decode step (inference_mode, dropout off).
N_CANDIDATES = AxisSpec(
  name="n_candidates",
  cardinality=128,
  default_batch_size=0,
  tile_granularity=1,
  heterogeneous=False,
)

ALL_AXES: list[AxisSpec] = [
  N_RESIDUES,
  N_LIGAND_ATOMS,
  N_STATES,
  N_STRUCTURES,
  N_SAMPLES,
  N_TEMPERATURES,
  N_NOISES,
  N_JACOBIAN_PAIRS,
  N_COMBINE,
  N_APC_PAIRS,
  N_REPLICATES,
  N_CANDIDATES,
]
