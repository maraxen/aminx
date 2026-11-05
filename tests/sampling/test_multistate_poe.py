"""Tests for aminx.sampling.multistate_poe.

Regression coverage for the 2026-07-13 finding (tev_design task
260709_multistate-fusion-strategy-comparison, Phase 3): no campaign-reachable path ever built a
genuinely stacked num_states>1 bundle for autoregressive sampling. See
.praxia/docs/decisions/260713_no-real-multistate-sampling-path-exists.md.
"""

from __future__ import annotations

from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.host.prep import prep_protein_stream_and_model
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.inference.logits import make_stage_set
from aminx.model import Aminx
from aminx.run.batch_mapping import MappedBy
from aminx.run.specs import SamplingSpecification
from aminx.sampling.multistate_poe import sample_multistate_poe_bead, sample_states_fused

REPO_TESTS_DATA = __file__.rsplit("/tests/", 1)[0] + "/tests/data"


def _build_synthetic_bundle(num_states: int, num_residues: int, seed: int):
  """Mirrors tests/inference/decode/test_autoregressive.py's _build_synthetic_fixture."""
  rng = np.random.default_rng(seed)
  jax_key = jax.random.PRNGKey(seed)

  model = Aminx(
    node_features=64,
    edge_features=64,
    hidden_features=64,
    num_encoder_layers=2,
    num_decoder_layers=2,
    k_neighbors=5,
    dropout_rate=0.0,
    key=jax_key,
  )
  model = eqx.tree_inference(model, value=True)

  coordinates = jnp.array(rng.normal(size=(num_residues, 4, 3)).astype(np.float32))
  mask = jnp.ones((num_residues,), dtype=jnp.float32)
  residue_index = jnp.arange(num_residues, dtype=jnp.int32)
  chain_index = jnp.zeros((num_residues,), dtype=jnp.int32)

  coordinates_stack = jnp.stack([coordinates] * num_states, axis=0)
  mask_stack = jnp.stack([mask] * num_states, axis=0)
  residue_index_stack = jnp.stack([residue_index] * num_states, axis=0)
  chain_index_stack = jnp.stack([chain_index] * num_states, axis=0)

  state_weights = jnp.ones(num_states) / num_states
  bundle, config = build_inference_bundle(
    coords=coordinates_stack,
    mask=mask_stack,
    residue_index=residue_index_stack,
    chain_index=chain_index_stack,
    state_weights=state_weights,
    sequence=None,
    mode="sample_ar",
  )
  return model, bundle, config


class TestSampleStatesFused:
  """Coverage for the low-level sample_states_fused primitive (already-built bundle)."""

  def test_shapes_and_dtypes(self):
    model, bundle, config = _build_synthetic_bundle(num_states=1, num_residues=10, seed=1)
    stage_set = make_stage_set(strategy="product", state_weights=bundle.conditioning.state_weights)
    n_samples = 4

    sequences, logits = sample_states_fused(
      model, bundle, config, stage_set, jax.random.PRNGKey(0), n_samples,
    )

    L = 10
    assert sequences.shape == (n_samples, L)
    assert logits.shape == (n_samples, L, 21)
    assert sequences.dtype == jnp.int32
    assert logits.dtype == jnp.float32

  def test_different_sample_keys_give_different_sequences(self):
    """n_samples>1 must actually be independent draws, not the same sample repeated."""
    model, bundle, config = _build_synthetic_bundle(num_states=1, num_residues=16, seed=2)
    stage_set = make_stage_set(strategy="product", state_weights=bundle.conditioning.state_weights)

    sequences, _ = sample_states_fused(model, bundle, config, stage_set, jax.random.PRNGKey(7), 8)

    # Not every one of the 8 samples should be byte-identical -- if they were, sample_keys
    # aren't actually varying anything (a real bug: vmap over a constant, or a key not
    # threaded through).
    unique_rows = {tuple(row.tolist()) for row in sequences}
    assert len(unique_rows) > 1, "all 8 samples were identical -- sample keys aren't varying"

  def test_genuine_multistate_fusion_changes_output(self):
    """The actual claim this module exists for: a real num_states>1 bundle with a
    non-identity state_position_map produces DIFFERENT fused logits than the identity map --
    proving states are genuinely combined, not independently decoded and discarded.
    Mirrors test_autoregressive.py's test_ar_decode_state_position_map_changes_fused_logits,
    but through THIS module's public sample_states_fused, not a hand-rolled AutoregressiveDecode
    construction -- this is the actual call path sample_multistate_poe_bead uses in production.
    """
    num_states, num_residues = 2, 8
    model, bundle, config = _build_synthetic_bundle(
      num_states=num_states, num_residues=num_residues, seed=11,
    )
    stage_set = make_stage_set(strategy="product", state_weights=bundle.conditioning.state_weights)
    key = jax.random.PRNGKey(0)

    _, baseline_logits = sample_states_fused(model, bundle, config, stage_set, key, 1)

    permuted_row = jnp.roll(jnp.arange(num_residues), shift=1)
    custom_map = jnp.stack([jnp.arange(num_residues), permuted_row])
    permuted_bundle = eqx.tree_at(
      lambda b: b.conditioning.state_position_map,
      bundle,
      custom_map,
    )
    _, permuted_logits = sample_states_fused(model, permuted_bundle, config, stage_set, key, 1)

    assert not jnp.allclose(permuted_logits, baseline_logits), (
      "a non-identity state_position_map must change the fused logits -- if it doesn't, "
      "states aren't actually being combined"
    )

  def test_single_state_is_still_a_no_op_regression_guard(self):
    """num_states=1 (the pre-existing, correct behavior for every campaign row before this
    module) must be unaffected: identity state_position_map on a single state changes nothing.
    """
    model, bundle, config = _build_synthetic_bundle(num_states=1, num_residues=8, seed=12)
    stage_set = make_stage_set(strategy="product", state_weights=bundle.conditioning.state_weights)
    key = jax.random.PRNGKey(0)

    _, logits_a = sample_states_fused(model, bundle, config, stage_set, key, 1)
    _, logits_b = sample_states_fused(model, bundle, config, stage_set, key, 1)

    assert jnp.array_equal(logits_a, logits_b), "identical inputs must give identical output"

  def test_multistate_state_axis_resolved_via_batchplanner_not_hardcoded_vmap(self):
    """Regression test for aminx debt #942 (found investigating tev_design's necklace PoE
    campaign): sample_autoregressive.kernel() previously hardcoded strategy=Vmap() for the
    state axis unconditionally -- invisible to any budget accounting, fine at num_states=1,
    but for a genuinely fused multi-state bundle it batches every decoder layer's per-state
    MLPs simultaneously. At production sample counts this produced a single fused GEMM XLA's
    autotuner could not find a valid kernel config for: sample_count=128 crashed after an
    809s compile ("Autotuning failed for HLO: f32[128,12582912]{1,0} fusion(...)"),
    sample_count=512 failed differently ("9 out of 89 instructions"). This synthetic fixture
    is far too small to reproduce the crash itself (that needs a real production-scale model
    and shape) -- this test instead verifies the fix's actual wiring: sample_states_fused must
    resolve the state axis through BatchPlanner against the canonical N_STATES AxisSpec
    (default_batch_size=1) and pass a real state_strategy through to the kernel, rather than
    leaving kernel()'s Vmap default in effect.

    Uses a (num_states, num_residues, n_samples) combination not used by any other test in
    this file/class, so eqx.filter_jit is guaranteed to retrace (not serve a cached trace from
    a different test) and the mock genuinely observes the call.
    """
    from aminx.sampling import multistate_poe as poe_mod
    from aminx.tiling.strategy import SafeMap

    num_states, num_residues, n_samples = 4, 6, 2
    model, bundle, config = _build_synthetic_bundle(
      num_states=num_states, num_residues=num_residues, seed=31,
    )
    stage_set = make_stage_set(strategy="product", state_weights=bundle.conditioning.state_weights)

    real_kernel = poe_mod._sample_autoregressive_kernel
    with patch.object(poe_mod, "_sample_autoregressive_kernel", wraps=real_kernel) as mock_kernel:
      sample_states_fused(model, bundle, config, stage_set, jax.random.PRNGKey(0), n_samples)

    # jax.vmap traces _one_sample's Python body once to build the batched computation
    # graph (it does not literally call the Python function n_samples times) -- assert
    # at least one traced call, not exactly n_samples.
    assert mock_kernel.call_count >= 1
    for call in mock_kernel.call_args_list:
      state_strategy = call.kwargs["state_strategy"]
      assert isinstance(state_strategy, SafeMap), (
        f"num_states={num_states} > N_STATES.default_batch_size=1 must resolve to SafeMap "
        f"(the axis's own canonical template default), got {type(state_strategy).__name__} "
        "-- state axis is not going through BatchPlanner"
      )
      assert state_strategy.tile == 1

  def test_single_state_still_passes_an_explicit_resolved_strategy(self):
    """num_states=1 <= N_STATES.default_batch_size=1 -> Vmap is the correct (and harmless,
    since cardinality=1 makes Vmap/SafeMap equivalent) BatchPlanner decision; must still be
    an explicitly resolved strategy passed through, not silently relying on kernel()'s own
    Vmap default (which would mask a regression if N_STATES' default_batch_size ever changes).
    """
    from aminx.sampling import multistate_poe as poe_mod
    from aminx.tiling.strategy import Vmap

    model, bundle, config = _build_synthetic_bundle(num_states=1, num_residues=9, seed=32)
    stage_set = make_stage_set(strategy="product", state_weights=bundle.conditioning.state_weights)

    real_kernel = poe_mod._sample_autoregressive_kernel
    with patch.object(poe_mod, "_sample_autoregressive_kernel", wraps=real_kernel) as mock_kernel:
      sample_states_fused(model, bundle, config, stage_set, jax.random.PRNGKey(0), 1)

    state_strategy = mock_kernel.call_args.kwargs["state_strategy"]
    assert isinstance(state_strategy, Vmap)


class TestSampleMultistatePoeBead:
  """Coverage for the high-level orchestration function (real structure loading)."""

  def test_rejects_single_input(self):
    spec = SamplingSpecification(inputs=[f"{REPO_TESTS_DATA}/1ubq.pdb"], batch_size=1)
    with pytest.raises(ValueError, match=">=2 states"):
      sample_multistate_poe_bead(spec, jax.random.PRNGKey(0), n_samples=1)

  def test_rejects_batch_size_mismatch(self):
    spec = SamplingSpecification(
      inputs=[f"{REPO_TESTS_DATA}/1ubq.pdb", f"{REPO_TESTS_DATA}/1mbn.pdb"],
      batch_size=1,
    )
    with pytest.raises(ValueError, match="spec.batch_size"):
      sample_multistate_poe_bead(spec, jax.random.PRNGKey(0), n_samples=1)

  def test_real_structures_two_states_end_to_end(self):
    """Real end-to-end path: real PDB parsing (create_protein_dataset, unmocked) for 2 real,
    genuinely different small structures, stacked into one num_states=2 bundle, sampled with
    product-strategy fusion. Only the model WEIGHTS are mocked (a synthetic Aminx, not a real
    trained checkpoint) -- structure loading/padding/batching/bundle-building/fusion/sampling
    are all real, unmocked code paths.
    """
    inputs = [f"{REPO_TESTS_DATA}/1ubq.pdb", f"{REPO_TESTS_DATA}/1mbn.pdb"]
    spec = SamplingSpecification(
      inputs=inputs,
      batch_size=len(inputs),
      multi_state_strategy="product",
      return_logits=True,
    )

    synthetic_model = Aminx(
      node_features=32,
      edge_features=32,
      hidden_features=32,
      num_encoder_layers=1,
      num_decoder_layers=1,
      k_neighbors=8,
      dropout_rate=0.0,
      key=jax.random.PRNGKey(3),
    )
    synthetic_model = eqx.tree_inference(synthetic_model, value=True)

    with patch("aminx.host.prep.load_model", return_value=synthetic_model):
      sequences, logits = sample_multistate_poe_bead(
        spec, jax.random.PRNGKey(0), n_samples=2,
      )

    assert sequences.ndim == 2
    assert sequences.shape[0] == 2
    assert logits.shape[0] == 2
    assert logits.shape[-1] == 21
    assert sequences.dtype == jnp.int32
    assert bool(jnp.all(jnp.isfinite(logits)))

  def test_decoding_order_seed_reaches_the_fused_sampler(self):
    """The bead hands the spec's ``decoding_order_seed`` to ``sample_states_fused`` (None when unset).

    Without this wiring a seed set on a campaign row would be silently dropped on the fused path
    while the single-state path honoured it, which is exactly the unpaired comparison the knob exists
    to remove. The sampler is spied; structure loading and bundle building are real.
    """
    inputs = [f"{REPO_TESTS_DATA}/1ubq.pdb", f"{REPO_TESTS_DATA}/1mbn.pdb"]
    synthetic_model = eqx.tree_inference(
      Aminx(
        node_features=32,
        edge_features=32,
        hidden_features=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=8,
        dropout_rate=0.0,
        key=jax.random.PRNGKey(3),
      ),
      value=True,
    )
    seen: dict[str, object] = {}

    def _spy(*args, **kwargs):
      seen["seed"] = kwargs.get("decoding_order_seed", "ABSENT")
      return jnp.zeros((1, 4), jnp.int32), jnp.zeros((1, 4, 21), jnp.float32)

    for order_seed, expected in ((None, None), (4, 4), (0, 0)):
      spec = SamplingSpecification(
        inputs=inputs,
        batch_size=len(inputs),
        multi_state_strategy="product",
        decoding_order_seed=order_seed,
      )
      seen.clear()
      with (
        patch("aminx.host.prep.load_model", return_value=synthetic_model),
        patch("aminx.sampling.multistate_poe.sample_states_fused", _spy),
      ):
        sample_multistate_poe_bead(spec, jax.random.PRNGKey(0), n_samples=1)
      assert seen["seed"] == expected

  def test_fixed_position_context_reaches_the_fused_sampler(self):
    """The bead hands the spec's ``fixed_position_context`` to ``sample_states_fused``.

    Dropping it would honour the mode on single-state rows and silently ignore it on fused rows.
    """
    inputs = [f"{REPO_TESTS_DATA}/1ubq.pdb", f"{REPO_TESTS_DATA}/1mbn.pdb"]
    synthetic_model = eqx.tree_inference(
      Aminx(
        node_features=32,
        edge_features=32,
        hidden_features=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=8,
        dropout_rate=0.0,
        key=jax.random.PRNGKey(3),
      ),
      value=True,
    )
    seen: dict[str, object] = {}

    def _spy(*args, **kwargs):
      seen["mode"] = kwargs.get("fixed_position_context", "ABSENT")
      return jnp.zeros((1, 4), jnp.int32), jnp.zeros((1, 4, 21), jnp.float32)

    for mode in ("random_order", "canonical_order", "mutual_context"):
      spec = SamplingSpecification(
        inputs=inputs,
        batch_size=len(inputs),
        multi_state_strategy="product",
        fixed_position_context=mode,
      )
      seen.clear()
      with (
        patch("aminx.host.prep.load_model", return_value=synthetic_model),
        patch("aminx.sampling.multistate_poe.sample_states_fused", _spy),
      ):
        sample_multistate_poe_bead(spec, jax.random.PRNGKey(0), n_samples=1)
      assert seen["mode"] == mode

  def test_real_state_position_map_through_full_pipeline_changes_output(self):
    """The core claim this module exists for, proven through the FULL real pipeline --
    not just a hand-built synthetic bundle (see TestSampleStatesFused's version of this
    assertion). Computes a genuine, non-identity state_position_map via
    aminx.utils.align.build_state_position_map on two real, genuinely different-length
    structures (1ubq=76 residues, 1mbn=153), threads it through
    prep_protein_stream_and_model -> _prepare_fixed_controls -> _prepare_ligand_context ->
    build_inference_bundle exactly as sample_multistate_poe_bead does in production, and
    confirms the resulting fused logits differ from the identity-map (default) case under
    the SAME PRNG key. Added 2026-07-13 per independent PR audit finding F1: the original
    test suite proved fusion-is-real only on a synthetic bundle, and proved the real
    pipeline doesn't crash only with a trivial identity map -- never both together.
    """
    from aminx.utils.align import build_state_position_map

    inputs = [f"{REPO_TESTS_DATA}/1ubq.pdb", f"{REPO_TESTS_DATA}/1mbn.pdb"]
    base_spec = SamplingSpecification(
      inputs=inputs,
      batch_size=len(inputs),
      multi_state_strategy="product",
      return_logits=True,
    )

    synthetic_model = Aminx(
      node_features=32,
      edge_features=32,
      hidden_features=32,
      num_encoder_layers=1,
      num_decoder_layers=1,
      k_neighbors=8,
      dropout_rate=0.0,
      key=jax.random.PRNGKey(3),
    )
    synthetic_model = eqx.tree_inference(synthetic_model, value=True)

    with patch("aminx.host.prep.load_model", return_value=synthetic_model):
      protein_iterator, _ = prep_protein_stream_and_model(base_spec)
      batched_ensemble = list(protein_iterator)[0]

    seq_len = batched_ensemble.coordinates.shape[1]
    n_states = batched_ensemble.coordinates.shape[0]
    native_lens = [int(m.sum()) for m in batched_ensemble.mask]
    assert native_lens[0] != native_lens[1], (
      "fixture structures must have genuinely different native lengths for this test "
      "to exercise a real, non-identity alignment"
    )

    # aminx.utils.align expects native sequences -1-padded to a common max NATIVE length
    # (not the campaign's full padded seq_len) -- extract each state's true native prefix
    # via its own mask, matching tev_design's compute_state_position_map exactly.
    max_native_len = max(native_lens)
    native_seqs = jnp.full((n_states, max_native_len), -1, dtype=jnp.int32)
    for s in range(n_states):
      native_seqs = native_seqs.at[s, : native_lens[s]].set(
        batched_ensemble.aatype[s, : native_lens[s]].astype(jnp.int32),
      )

    real_map_native = build_state_position_map(native_seqs, reference_state_index=0)
    assert not jnp.array_equal(real_map_native[1], jnp.arange(max_native_len)), (
      "expected a genuinely non-identity alignment for state 1 given different native "
      "lengths -- if this is identity, the alignment call itself isn't exercising real "
      "divergence and this test proves nothing"
    )

    # Pad the alignment result (S, max_native_len) out to the campaign's full seq_len
    # (S, 512) with -1, matching build_necklace_p2_manifest.py's own
    # compute_state_position_map padding convention.
    real_map = jnp.full((n_states, seq_len), -1, dtype=jnp.int32)
    real_map = real_map.at[:, :max_native_len].set(real_map_native)

    key = jax.random.PRNGKey(0)

    def _sample_with_map(state_position_map):
      spec = SamplingSpecification(
        inputs=inputs,
        batch_size=len(inputs),
        multi_state_strategy="product",
        return_logits=True,
        state_position_map=state_position_map,
      )
      with patch("aminx.host.prep.load_model", return_value=synthetic_model):
        return sample_multistate_poe_bead(spec, key, n_samples=1)

    _, identity_logits = _sample_with_map(None)
    _, real_map_logits = _sample_with_map(real_map)

    assert not jnp.allclose(identity_logits, real_map_logits), (
      "a real, non-identity state_position_map computed from actual structural alignment "
      "must change fused logits through the FULL real pipeline -- if it doesn't, states "
      "aren't genuinely being combined despite build_inference_bundle/sample_states_fused "
      "each individually appearing to work"
    )
    assert bool(jnp.all(jnp.isfinite(real_map_logits)))

  def test_ligandmpnn_sidechain_conditioning_end_to_end(self):
    """The real necklace campaign's actual production configuration
    (model_family="ligandmpnn", sidechain_conditioning=True, e.g. the
    ligand_mpnn_v32_020_25_sc_only model) -- entirely untested by the rest of this suite
    (which only exercises the default proteinmpnn/no-ligand path) until this test, added
    2026-07-13 per independent PR audit finding F4. No real ligand molecule is needed --
    sidechain_conditioning alone routes through _prepare_ligand_context's atom_37 path
    using batched_ensemble.coordinates/atom_mask, both already populated by real PDB
    parsing regardless of ligand presence.

    Uses PrxteinLigandMPNN (ligand_mpnn_use_side_chain_context=True), not the base Aminx
    class -- a first attempt with a plain Aminx model failed with
    `TypeError: Aminx.__call__() got an unexpected keyword argument 'atom_37'`, matching
    the model-construction pattern already established in
    tests/inference/test_side_chain_context.py's own _small_ligand_model helper.
    """
    from aminx.model.ligand_mpnn import PrxteinLigandMPNN

    inputs = [f"{REPO_TESTS_DATA}/1ubq.pdb", f"{REPO_TESTS_DATA}/1mbn.pdb"]
    spec = SamplingSpecification(
      inputs=inputs,
      batch_size=len(inputs),
      multi_state_strategy="product",
      model_family="ligandmpnn",
      sidechain_conditioning=True,
      return_logits=True,
    )

    synthetic_model = PrxteinLigandMPNN(
      node_features=32,
      edge_features=32,
      hidden_features=32,
      num_encoder_layers=2,
      num_decoder_layers=2,
      k_neighbors=6,
      num_context_layers=2,
      dropout_rate=0.0,
      ligand_mpnn_use_side_chain_context=True,
      key=jax.random.PRNGKey(4),
    )
    synthetic_model = eqx.tree_inference(synthetic_model, value=True)

    with patch("aminx.host.prep.load_model", return_value=synthetic_model):
      sequences, logits = sample_multistate_poe_bead(
        spec, jax.random.PRNGKey(0), n_samples=2,
      )

    assert sequences.shape[0] == 2
    assert logits.shape[0] == 2
    assert logits.shape[-1] == 21
    assert bool(jnp.all(jnp.isfinite(logits)))

  def test_mapped_by_fixed_mask_resolves_via_canonical_structure_ids(self):
    """MappedBy is a feature of fixed_mask/fixed_tokens, not just of the independent-structures
    sample() path -- this module's combined-bead batch resolves it too, keyed by canonical
    structure id (same as kernel_dispatch.py), since one state == one spec.inputs entry here.
    A uniform (identical per key) MappedBy must resolve and run cleanly through the real
    pipeline, same as an equivalent plain array would.
    """
    max_length = 200
    inputs = [f"{REPO_TESTS_DATA}/1ubq.pdb", f"{REPO_TESTS_DATA}/1mbn.pdb"]
    uniform_mask = np.zeros((max_length,), dtype=np.float32)
    uniform_mask[0] = 1.0
    uniform_tokens = np.zeros((max_length,), dtype=np.int32)
    spec = SamplingSpecification(
      inputs=inputs,
      batch_size=len(inputs),
      max_length=max_length,
      multi_state_strategy="product",
      return_logits=True,
      fixed_mask=MappedBy(by="path", mapping={"1ubq": uniform_mask, "1mbn": uniform_mask}),
      fixed_tokens=MappedBy(by="path", mapping={"1ubq": uniform_tokens, "1mbn": uniform_tokens}),
    )

    synthetic_model = Aminx(
      node_features=32,
      edge_features=32,
      hidden_features=32,
      num_encoder_layers=1,
      num_decoder_layers=1,
      k_neighbors=8,
      dropout_rate=0.0,
      key=jax.random.PRNGKey(3),
    )
    synthetic_model = eqx.tree_inference(synthetic_model, value=True)

    with patch("aminx.host.prep.load_model", return_value=synthetic_model):
      sequences, logits = sample_multistate_poe_bead(
        spec, jax.random.PRNGKey(0), n_samples=1,
      )

    assert sequences.shape[0] == 1
    assert bool(jnp.all(jnp.isfinite(logits)))

  def test_mapped_by_with_per_state_different_values_raises_the_existing_uniformity_error(self):
    """A MappedBy resolving to genuinely DIFFERENT masks per state must still hit this
    module's pre-existing "fixed positions must be design-level, identical across states"
    guard -- proving MappedBy resolves through to real per-row values (not silently
    collapsed/ignored), while confirming this module's own uniformity restriction still
    applies on top of it.
    """
    max_length = 200
    inputs = [f"{REPO_TESTS_DATA}/1ubq.pdb", f"{REPO_TESTS_DATA}/1mbn.pdb"]
    mask_a = np.zeros((max_length,), dtype=np.float32)
    mask_a[0] = 1.0
    mask_b = np.zeros((max_length,), dtype=np.float32)
    mask_b[1] = 1.0  # deliberately different position than mask_a
    tokens = np.zeros((max_length,), dtype=np.int32)
    spec = SamplingSpecification(
      inputs=inputs,
      batch_size=len(inputs),
      max_length=max_length,
      multi_state_strategy="product",
      fixed_mask=MappedBy(by="path", mapping={"1ubq": mask_a, "1mbn": mask_b}),
      fixed_tokens=MappedBy(by="path", mapping={"1ubq": tokens, "1mbn": tokens}),
    )

    synthetic_model = Aminx(
      node_features=32,
      edge_features=32,
      hidden_features=32,
      num_encoder_layers=1,
      num_decoder_layers=1,
      k_neighbors=8,
      dropout_rate=0.0,
      key=jax.random.PRNGKey(3),
    )
    synthetic_model = eqx.tree_inference(synthetic_model, value=True)

    with (
      patch("aminx.host.prep.load_model", return_value=synthetic_model),
      pytest.raises(ValueError, match="design-level"),
    ):
      sample_multistate_poe_bead(spec, jax.random.PRNGKey(0), n_samples=1)
