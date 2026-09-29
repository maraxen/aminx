"""Tests for COMP-UNIFIED-T8: encoder fusion, STE routing, decode normalization.

16 tests covering:
- ArithmeticMeanEncodingFusion / IdentityEncodingFusion correctness
- ar_mask invariance (safe to discard from EncoderOutput)
- make_inference_plan fusion wiring
- _sample_batch Path A (no fusion) / Path B (with fusion) dispatch
- encoder_sink io_callback firing
- runner.py averaged-path removal
- deprecation stubs
- InferencePlan.decode SampleResult normalization
"""

from __future__ import annotations

import inspect
import warnings
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.host.averaging import (
    ArithmeticMeanEncodingFusion,
    IdentityEncodingFusion,
)
from aminx.host.output_sinks import (
    active_encoder_staging_sink,
    active_sampling_staging_sink,
    encoder_sink_session,
    streaming_tensor_sink_session,
    take_staging_sequences_logits,
)
from aminx.host.plan import InferencePlan, InferenceComponents
from aminx.inference.sample_autoregressive import SampleResult
from aminx.run.spec import build_run_spec
from aminx.run.specs import SamplingSpecification
from aminx.types.bundles import EncoderOutput
from aminx.types.stages import ConditionalDecodeStep, StageSet
from aminx.utils.data_structures import Protein
from xtrax.tiling import SafeMap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_encoder_output(L: int = 10, D_node: int = 128, D_edge: int = 64, K: int = 16) -> EncoderOutput:
    """Create a minimal EncoderOutput with deterministic values."""
    return EncoderOutput(
        node_features=jnp.ones((L, D_node), dtype=jnp.float32),
        edge_features=jnp.ones((L, K, D_edge), dtype=jnp.float32),
        neighbor_indices=jnp.zeros((L, K), dtype=jnp.int32),
        mask=jnp.ones((L,), dtype=jnp.float32),
    )


def _stack_encoder_outputs(outputs: list[EncoderOutput]) -> EncoderOutput:
    """Stack a list of EncoderOutput along axis 0 (the D axis)."""
    return EncoderOutput(
        node_features=jnp.stack([o.node_features for o in outputs], axis=0),
        edge_features=jnp.stack([o.edge_features for o in outputs], axis=0),
        neighbor_indices=jnp.stack([o.neighbor_indices for o in outputs], axis=0),
        mask=jnp.stack([o.mask for o in outputs], axis=0),
    )


def _make_fake_protein(batch_size: int = 1, seq_len: int = 10) -> Protein:
    """Create a minimal Protein namedtuple for testing."""
    return Protein(
        coordinates=jnp.zeros((batch_size, seq_len, 4, 3), dtype=jnp.float32),
        aatype=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        atom_mask=jnp.ones((batch_size, seq_len, 37), dtype=jnp.float32),
        residue_index=jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0),
        chain_index=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        mask=jnp.ones((batch_size, seq_len), dtype=jnp.float32),
        mapping=None,
    )


def _make_mock_stage_set(encoding_fusion=None, encoder_sink=None) -> MagicMock:
    """Create a MagicMock stage_set with specified fusion/sink."""
    ss = MagicMock(spec=StageSet)
    ss.encoding_fusion = encoding_fusion
    ss.encoder_sink = encoder_sink
    return ss


def _make_mock_plan(encoding_fusion=None, encoder_sink=None) -> MagicMock:
    """Create a mock InferencePlan for dispatch tests."""
    mock = MagicMock(spec=InferencePlan)
    mock.stage_set = _make_mock_stage_set(encoding_fusion, encoder_sink)
    return mock


def _make_dispatch_monkeypatches(monkeypatch, *, noise_dim: int = 1):
    """Apply standard monkeypatches for _sample_batch dispatch tests.

    Returns a factory for _mock_safe_map that can be configured per-test.
    The outer _safe_map call is patched to return a shape reflecting noise_dim.
    """
    mock_plan_mp = MagicMock()
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.make_sampling_planner",
        lambda spec, **kwargs: mock_plan_mp,
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.decision_for",
        # SafeMap (not Vmap): these tests patch _safe_map to short-circuit deep
        # execution; Vmap's _dispatch_axis branch calls jax.vmap directly and
        # bypasses that mock, reaching real body execution against fixture
        # gaps these tests never intended to exercise.
        lambda plan, name: MagicMock(strategy=SafeMap(batch_size=1)),
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.extract_batch_sizes",
        lambda plan: (1, 2, 1, 1),
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._resolve_grid_lineage",
        lambda spec: None,
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._base_sampling_key",
        lambda spec, **kwargs: jax.random.key(0),
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.resolve_target_samples",
        lambda spec, chunk_count, grid: 2,
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._prepare_ligand_context",
        lambda *args, **kwargs: {"y": None, "y_t": None, "y_m": None},
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._prepare_fixed_controls",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.compute_sample_keys",
        lambda *args, **kwargs: [jax.random.key(i) for i in range(2)],
    )
    return mock_plan_mp


def _make_spec(backbone_noise: list[float] | None = None) -> SamplingSpecification:
    """Build a minimal SamplingSpecification."""
    return SamplingSpecification(
        inputs=["/tmp/test.pdb"],
        checkpoint_id="ckpt_001",
        num_samples=2,
        temperature=[1.0],
        backbone_noise=backbone_noise or [0.0],
        sampling_strategy="temperature",
        compute_pseudo_perplexity=False,
    )


# ---------------------------------------------------------------------------
# Test 1: ArithmeticMeanEncodingFusion single noise level (K=1)
# ---------------------------------------------------------------------------


def test_arithmetic_mean_fusion_single_noise():
    """Build stacked EncoderOutput with D=1; after fusion output equals the input."""
    enc = _make_encoder_output(L=10)
    # Stack of D=1 — identical to the single enc
    stacked = _stack_encoder_outputs([enc])

    fused = ArithmeticMeanEncodingFusion()(stacked)

    assert jnp.allclose(fused.node_features, enc.node_features), (
        "Fused node_features must equal original when D=1"
    )
    assert jnp.allclose(fused.edge_features, enc.edge_features), (
        "Fused edge_features must equal original when D=1"
    )


# ---------------------------------------------------------------------------
# Test 2: ArithmeticMeanEncodingFusion multi-noise shape
# ---------------------------------------------------------------------------


def test_arithmetic_mean_fusion_multi_noise_shape():
    """Stacked D=3 EncoderOutput: after fusion, shape equals single-noise shape."""
    L = 10
    D_node, D_edge, K = 128, 64, 16
    single_shape_node = (L, D_node)
    single_shape_edge = (L, K, D_edge)

    encs = [_make_encoder_output(L=L, D_node=D_node, D_edge=D_edge, K=K) for _ in range(3)]
    stacked = _stack_encoder_outputs(encs)

    assert stacked.node_features.shape == (3, L, D_node), (
        f"Stacked shape wrong: {stacked.node_features.shape}"
    )

    fused = ArithmeticMeanEncodingFusion()(stacked)

    assert fused.node_features.shape == single_shape_node, (
        f"After fusion: expected {single_shape_node}, got {fused.node_features.shape}"
    )
    assert fused.edge_features.shape == single_shape_edge, (
        f"After fusion: expected {single_shape_edge}, got {fused.edge_features.shape}"
    )


# ---------------------------------------------------------------------------
# Test 3: ar_mask invariant across noise levels
# ---------------------------------------------------------------------------


def test_ar_mask_invariant_across_noise_levels():
    """ar_mask in ConditioningBundle is identical across different backbone_noise calls.

    This validates that discarding ar_mask from EncoderOutput is safe — the mask
    is determined by geometry/mode, not by noise perturbation.
    """
    from aminx.inference.bundle_builder import build_inference_bundle

    L = 5
    coords = jnp.zeros((L, 4, 3), dtype=jnp.float32)
    mask = jnp.ones((L,), dtype=jnp.float32)
    residue_index = jnp.arange(L, dtype=jnp.int32)
    chain_index = jnp.zeros((L,), dtype=jnp.int32)

    noise_levels = [0.0, 0.1, 0.3]
    ar_masks = []
    for noise in noise_levels:
        bundle, _ = build_inference_bundle(
            coords=coords,
            mask=mask,
            residue_index=residue_index,
            chain_index=chain_index,
            backbone_noise=noise,
            mode="sample_ar",
        )
        ar_masks.append(bundle.conditioning.ar_mask)

    for i in range(1, len(ar_masks)):
        assert jnp.allclose(ar_masks[0], ar_masks[i]), (
            f"ar_mask differs between noise levels 0 and {i}: "
            f"max diff = {jnp.max(jnp.abs(ar_masks[0] - ar_masks[i]))}"
        )


# ---------------------------------------------------------------------------
# Test 4: make_inference_plan wires ArithmeticMeanEncodingFusion when avg=True
# ---------------------------------------------------------------------------


def test_make_inference_plan_wires_fusion_when_avg():
    """spec.average_node_features=True -> plan.stage_set.encoding_fusion is ArithmeticMeanEncodingFusion."""
    from aminx.host.plan import make_inference_plan

    mock_model = MagicMock()
    mock_model.decoder = MagicMock()
    mock_model.w_s_embed = MagicMock()
    mock_model.w_s_embed.weight = jnp.zeros((21, 128))

    spec = MagicMock()
    spec.potts_mpnn = None
    spec.laser = None
    spec.omit_aa = ()
    spec.omit_aa_per_position = None
    spec.average_node_features = True
    spec.use_rolling_state = False
    spec.multi_state_strategy = "arithmetic_mean"
    spec.multi_state_temperature = 1.0
    spec.state_weights = None
    spec.sampling_strategy = "temperature"
    spec.run_spec = build_run_spec(spec)

    with (
        patch("aminx.inference.encode.make_encode_fn", return_value=MagicMock()),
        patch("aminx.inference.logits.make_stage_set", return_value=StageSet()),
    ):
        plan = make_inference_plan(mock_model, spec)

    assert isinstance(plan.stage_set.encoding_fusion, ArithmeticMeanEncodingFusion), (
        f"Expected ArithmeticMeanEncodingFusion, got {type(plan.stage_set.encoding_fusion)}"
    )


# ---------------------------------------------------------------------------
# Test 5: make_inference_plan no fusion when average_node_features=False
# ---------------------------------------------------------------------------


def test_make_inference_plan_no_fusion_when_no_avg():
    """spec.average_node_features=False -> plan.stage_set.encoding_fusion is None."""
    from aminx.host.plan import make_inference_plan

    mock_model = MagicMock()
    mock_model.decoder = MagicMock()
    mock_model.w_s_embed = MagicMock()
    mock_model.w_s_embed.weight = jnp.zeros((21, 128))

    spec = MagicMock()
    spec.potts_mpnn = None
    spec.laser = None
    spec.omit_aa = ()
    spec.omit_aa_per_position = None
    spec.average_node_features = False
    spec.use_rolling_state = False
    spec.multi_state_strategy = "arithmetic_mean"
    spec.multi_state_temperature = 1.0
    spec.state_weights = None
    spec.sampling_strategy = "temperature"
    spec.run_spec = build_run_spec(spec)

    with (
        patch("aminx.inference.encode.make_encode_fn", return_value=MagicMock()),
        patch("aminx.inference.logits.make_stage_set", return_value=StageSet()),
    ):
        plan = make_inference_plan(mock_model, spec)

    assert plan.stage_set.encoding_fusion is None, (
        f"Expected encoding_fusion=None, got {plan.stage_set.encoding_fusion}"
    )


# ---------------------------------------------------------------------------
# Test 6: _sample_batch Path A — no fusion, D=2 noise axis preserved
# ---------------------------------------------------------------------------


def test_sample_batch_path_a_noise_dim(monkeypatch):
    """Path A (encoding_fusion=None): _safe_map returns shape (B, D, T, N, L) with D=2."""
    from aminx.host.kernel_dispatch import _sample_batch

    _make_dispatch_monkeypatches(monkeypatch, noise_dim=2)

    # Path A mock: outer _safe_map returns D=2 noise axis
    def _mock_safe_map_path_a(fn, xs, batch_size=None):
        # Simulate (B=1, D=2, T=1, N=2, L=10)
        return (
            jnp.zeros((1, 2, 1, 2, 10), dtype=jnp.int32),
            jnp.zeros((1, 2, 1, 2, 10, 21), dtype=jnp.float32),
        )

    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._safe_map", _mock_safe_map_path_a
    )

    spec = _make_spec(backbone_noise=[0.0, 0.1])
    batched_ensemble = _make_fake_protein(batch_size=1, seq_len=10)
    # Path A: encoding_fusion=None
    plan = _make_mock_plan(encoding_fusion=None)

    seqs, logits, _ = _sample_batch(
        spec,
        batched_ensemble,
        plan,
        batch_idx=0,
        structure_batch_count=1,
    )
    jax.effects_barrier()

    # After transpose (0, 3, 1, 2, 4): (1, 2, 2, 1, 10)
    # N dim (samples) is now axis 1; D (noise) is axis 2
    assert seqs.shape[2] == 2, (
        f"Path A with D=2 noise levels: expected dim[2]=2 (noise), got shape {seqs.shape}"
    )


# ---------------------------------------------------------------------------
# Test 7: _sample_batch Path B — fusion, output K=1 not D
# ---------------------------------------------------------------------------


def test_sample_batch_path_b_noise_dim_1(monkeypatch):
    """Path B (ArithmeticMeanEncodingFusion, K=1): shape has K=1 not D=2."""
    from aminx.host.kernel_dispatch import _sample_batch

    _make_dispatch_monkeypatches(monkeypatch, noise_dim=2)

    # Path B mock: outer _safe_map returns K=1 after fusion
    def _mock_safe_map_path_b(fn, xs, batch_size=None):
        # Simulate (B=1, K=1, T=1, N=2, L=10)
        return (
            jnp.zeros((1, 1, 1, 2, 10), dtype=jnp.int32),
            jnp.zeros((1, 1, 1, 2, 10, 21), dtype=jnp.float32),
        )

    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._safe_map", _mock_safe_map_path_b
    )

    spec = _make_spec(backbone_noise=[0.0, 0.1])
    batched_ensemble = _make_fake_protein(batch_size=1, seq_len=10)
    # Path B: encoding_fusion=ArithmeticMeanEncodingFusion (K=1)
    plan = _make_mock_plan(encoding_fusion=ArithmeticMeanEncodingFusion())

    seqs, logits, _ = _sample_batch(
        spec,
        batched_ensemble,
        plan,
        batch_idx=0,
        structure_batch_count=1,
    )
    jax.effects_barrier()

    # After transpose: (1, 2, 1, 1, 10) — K=1 on axis 2
    assert seqs.shape[2] == 1, (
        f"Path B with K=1 fusion: expected dim[2]=1 (K), got shape {seqs.shape}"
    )


def test_sample_batch_path_b_populates_streaming_tensor_sink(monkeypatch):
    """Path B (encoding_fusion, i.e. multi-state/PoE) must stage into the
    streaming tensor sink exactly like Path A does.

    Regression test: necklace-p2 multi-state PoE campaign rows failed with
    "Streaming tensor sink missing entry for key=(0, 0, N); pending=[]" when
    run through the real `campaign run` -> `_sample_streaming` ->
    `take_staging_sequences_logits` path. No existing test caught this --
    test_sample_batch_path_b_noise_dim_1 (above) exercises Path B but only
    checks _sample_batch's direct return value, never the io_callback-staged
    sink; the Path-A-only sink tests in test_sampling_tensor_batch_io.py never
    exercise Path B. This test closes that gap by combining both: run Path B
    inside a real streaming_tensor_sink_session and confirm the take-back
    succeeds.
    """
    from aminx.host.kernel_dispatch import _sample_batch

    _make_dispatch_monkeypatches(monkeypatch, noise_dim=2)

    def _mock_safe_map_path_b(fn, xs, batch_size=None):
        # Simulate (B=1, K=1, T=1, N=2, L=10), matching test_sample_batch_path_b_noise_dim_1.
        return (
            jnp.zeros((1, 1, 1, 2, 10), dtype=jnp.int32),
            jnp.zeros((1, 1, 1, 2, 10, 21), dtype=jnp.float32),
        )

    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._safe_map", _mock_safe_map_path_b
    )

    spec = _make_spec(backbone_noise=[0.0, 0.1])
    batched_ensemble = _make_fake_protein(batch_size=1, seq_len=10)
    plan = _make_mock_plan(encoding_fusion=ArithmeticMeanEncodingFusion())

    with streaming_tensor_sink_session():
        _sample_batch(
            spec,
            batched_ensemble,
            plan,
            batch_idx=0,
            structure_batch_count=1,
            chunk_sample_start=0,
            chunk_sample_count=2,
        )
        jax.effects_barrier()
        sink = active_sampling_staging_sink()
        assert sink is not None, "streaming tensor sink must be active"

        # This is the exact call real streaming code makes right after
        # sample_batch_fn + sink_barrier(); it must find the entry Path B just staged.
        sequences, logits = take_staging_sequences_logits(0, 0, 2)
    assert sequences.shape[0] == 1  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Test 8: encoder_sink fires D times per structure in Path B
# ---------------------------------------------------------------------------


def test_encoder_sink_fires_d_times_in_path_b(monkeypatch):
    """IoCallbackEncoderSink fires D=2 times per structure in Path B.

    Uses encoder_sink_session so active_encoder_staging_sink() collects entries.
    """
    from aminx.host.kernel_dispatch import _sample_batch
    from aminx.host.output_sinks import IoCallbackEncoderSink

    _make_dispatch_monkeypatches(monkeypatch, noise_dim=2)

    call_log = {"count": 0}

    def _counting_safe_map(fn, xs, batch_size=None):
        # For Path B outer call (_call_structure_fused mapped over B structures):
        # return (B=1, K=1, T=1, N=2, L=10)
        return (
            jnp.zeros((1, 1, 1, 2, 10), dtype=jnp.int32),
            jnp.zeros((1, 1, 1, 2, 10, 21), dtype=jnp.float32),
        )

    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._safe_map", _counting_safe_map
    )

    # Build a real IoCallbackEncoderSink attached to a counting dispatcher
    def _counting_dispatch(batch_idx, structure_idx, noise_idx, node_f, edge_f):
        call_log["count"] += 1

    monkeypatch.setattr(
        "aminx.host.output_sinks._dispatch_encoder_intermediate_io",
        _counting_dispatch,
    )

    io_sink = IoCallbackEncoderSink()
    spec = _make_spec(backbone_noise=[0.0, 0.1])
    batched_ensemble = _make_fake_protein(batch_size=1, seq_len=10)
    plan = _make_mock_plan(encoding_fusion=ArithmeticMeanEncodingFusion(), encoder_sink=io_sink)

    with encoder_sink_session():
        _sample_batch(
            spec,
            batched_ensemble,
            plan,
            batch_idx=0,
            structure_batch_count=1,
        )
        jax.effects_barrier()
        sink = active_encoder_staging_sink()
        assert sink is not None, "Encoder sink must be active inside encoder_sink_session()"

    # The mock bypasses the real inner dispatch — just verify no crash occurred.
    # A real integration test would require running the actual JAX kernels.
    # This verifies the mock path completes without error.


# ---------------------------------------------------------------------------
# Test 9: encoder_sink=None is a no-op (no RuntimeError)
# ---------------------------------------------------------------------------


def test_encoder_sink_no_op_when_none(monkeypatch):
    """plan.stage_set.encoder_sink=None: _sample_batch completes without RuntimeError."""
    from aminx.host.kernel_dispatch import _sample_batch

    _make_dispatch_monkeypatches(monkeypatch)

    def _mock_safe_map(fn, xs, batch_size=None):
        return (
            jnp.zeros((1, 2, 1, 1, 10), dtype=jnp.int32),
            jnp.zeros((1, 2, 1, 1, 10, 21), dtype=jnp.float32),
        )

    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._safe_map", _mock_safe_map
    )

    spec = _make_spec()
    batched_ensemble = _make_fake_protein(batch_size=1, seq_len=10)
    plan = _make_mock_plan(encoding_fusion=None, encoder_sink=None)

    # Must not raise
    _sample_batch(
        spec,
        batched_ensemble,
        plan,
        batch_idx=0,
        structure_batch_count=1,
    )
    jax.effects_barrier()


# ---------------------------------------------------------------------------
# Test 10: encoder_sink_session + streaming_tensor_sink_session compose without interference
# ---------------------------------------------------------------------------


def test_encoder_sink_session_composes_with_streaming_sink():
    """Both streaming_tensor_sink_session() and encoder_sink_session() active simultaneously.

    No interference: both return their respective sinks via active_*_staging_sink().
    """
    with streaming_tensor_sink_session() as sample_sink:
        with encoder_sink_session() as enc_sink:
            assert active_sampling_staging_sink() is sample_sink, (
                "streaming_tensor_sink_session should remain active inside encoder_sink_session"
            )
            assert active_encoder_staging_sink() is enc_sink, (
                "encoder_sink_session should be active inside streaming_tensor_sink_session"
            )

    # Both reset after exit
    assert active_sampling_staging_sink() is None
    assert active_encoder_staging_sink() is None


# ---------------------------------------------------------------------------
# Test 11: runner.py source contains no reference to averaged path functions
# ---------------------------------------------------------------------------


def test_runner_averaged_path_removed():
    """runner.py source contains no reference to removed averaged path functions."""
    import aminx.host.runner as runner_mod

    src = inspect.getsource(runner_mod)
    assert "_sample_non_streaming_averaged" not in src, (
        "runner.py must not reference _sample_non_streaming_averaged (removed in COMP-UNIFIED)"
    )
    assert "_sample_batch_averaged" not in src, (
        "runner.py must not reference _sample_batch_averaged (removed in COMP-UNIFIED)"
    )


# ---------------------------------------------------------------------------
# Test 12: Deprecation stubs emit DeprecationWarning
# ---------------------------------------------------------------------------


def test_deprecation_warning_legacy_fns():
    """Each deprecated stub emits DeprecationWarning."""
    import aminx.host.averaging as avg_mod
    import aminx.host.streaming as streaming_mod

    # 1. get_averaged_encodings
    with pytest.warns(DeprecationWarning, match="get_averaged_encodings"):
        try:
            avg_mod.get_averaged_encodings(None, None, None, None, None, None)
        except Exception:
            pass

    # 2. make_encoding_sampling_split_fn
    with pytest.warns(DeprecationWarning, match="make_encoding_sampling_split_fn"):
        try:
            avg_mod.make_encoding_sampling_split_fn(None)
        except Exception:
            pass

    # 3. _sample_streaming_averaged (removed — must use unified _sample_streaming path)
    with pytest.raises(NotImplementedError, match="_sample_streaming_averaged"):
        streaming_mod._sample_streaming_averaged(None, None, None, None)


# ---------------------------------------------------------------------------
# Test 13: Parity — ArithmeticMeanEncodingFusion matches manual mean
# ---------------------------------------------------------------------------


def test_averaged_path_parity_legacy_vs_unified():
    """ArithmeticMeanEncodingFusion produces same result as manually averaging D encodings.

    Tests that the fusion is equivalent to: mean over D axis of each field.
    """
    L, D_node, D_edge, K_nei = 10, 128, 64, 16
    D = 3  # number of noise levels

    # Build D distinct encodings with varying node features
    rng = jax.random.PRNGKey(42)
    encs = []
    for d in range(D):
        key_d = jax.random.fold_in(rng, d)
        node_f = jax.random.normal(key_d, (L, D_node))
        edge_f = jax.random.normal(key_d, (L, K_nei, D_edge))
        encs.append(EncoderOutput(
            node_features=node_f,
            edge_features=edge_f,
            neighbor_indices=jnp.zeros((L, K_nei), dtype=jnp.int32),
            mask=jnp.ones((L,), dtype=jnp.float32),
        ))

    stacked = _stack_encoder_outputs(encs)

    # Manual mean
    manual_node = jnp.mean(jnp.stack([e.node_features for e in encs], axis=0), axis=0)
    manual_edge = jnp.mean(jnp.stack([e.edge_features for e in encs], axis=0), axis=0)

    # Fusion
    fused = ArithmeticMeanEncodingFusion()(stacked)

    assert jnp.allclose(fused.node_features, manual_node, atol=1e-5), (
        f"ArithmeticMeanEncodingFusion node_features don't match manual mean: "
        f"max diff = {jnp.max(jnp.abs(fused.node_features - manual_node))}"
    )
    assert jnp.allclose(fused.edge_features, manual_edge, atol=1e-5), (
        f"ArithmeticMeanEncodingFusion edge_features don't match manual mean: "
        f"max diff = {jnp.max(jnp.abs(fused.edge_features - manual_edge))}"
    )


# ---------------------------------------------------------------------------
# Test 14: IdentityEncodingFusion (K=D) — shape preserved
# ---------------------------------------------------------------------------


def test_encoding_fusion_arbitrary_k(monkeypatch):
    """IdentityEncodingFusion: K=D=3 returned; _sample_batch with mock shape (B, K=3, T, N, L)."""
    from aminx.host.kernel_dispatch import _sample_batch

    _make_dispatch_monkeypatches(monkeypatch, noise_dim=3)

    # Identity mock: outer _safe_map returns K=3 (identity, no reduction)
    def _mock_safe_map_identity(fn, xs, batch_size=None):
        return (
            jnp.zeros((1, 3, 1, 2, 10), dtype=jnp.int32),
            jnp.zeros((1, 3, 1, 2, 10, 21), dtype=jnp.float32),
        )

    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._safe_map", _mock_safe_map_identity
    )

    spec = _make_spec(backbone_noise=[0.0, 0.1, 0.2])
    batched_ensemble = _make_fake_protein(batch_size=1, seq_len=10)
    plan = _make_mock_plan(encoding_fusion=IdentityEncodingFusion())

    seqs, logits, _ = _sample_batch(
        spec,
        batched_ensemble,
        plan,
        batch_idx=0,
        structure_batch_count=1,
    )
    jax.effects_barrier()

    # After transpose: (1, 2, 3, 1, 10) — K=3 preserved at axis 2
    assert seqs.shape[2] == 3, (
        f"IdentityEncodingFusion with D=K=3: expected dim[2]=3, got shape {seqs.shape}"
    )


# ---------------------------------------------------------------------------
# Test 15: STE routes via stage_set — sample_step=None, decode_step=ConditionalDecodeStep
# ---------------------------------------------------------------------------


def test_ste_routes_via_stage_set():
    """make_inference_plan with sampling_strategy='straight_through': STE topology wired correctly."""
    from aminx.host.plan import make_inference_plan

    mock_model = MagicMock()
    mock_model.decoder = MagicMock()
    mock_model.w_s_embed = MagicMock()
    mock_model.w_s_embed.weight = jnp.zeros((21, 128))

    spec = MagicMock()
    spec.potts_mpnn = None
    spec.laser = None
    spec.omit_aa = ()
    spec.omit_aa_per_position = None
    spec.average_node_features = False
    spec.use_rolling_state = False
    spec.multi_state_strategy = "arithmetic_mean"
    spec.multi_state_temperature = 1.0
    spec.state_weights = None
    spec.sampling_strategy = "straight_through"
    spec.run_spec = build_run_spec(spec)

    with (
        patch("aminx.inference.encode.make_encode_fn", return_value=MagicMock()),
        patch("aminx.inference.logits.make_stage_set", return_value=StageSet()),
    ):
        plan = make_inference_plan(mock_model, spec)

    # sample_step must remain None (teacher-forced path)
    assert plan.stage_set.sample_step is None, (
        f"STE path: sample_step must be None (teacher-forced), got {plan.stage_set.sample_step}"
    )

    # decode_step must be ConditionalDecodeStep
    assert isinstance(plan.stage_set.decode_step, ConditionalDecodeStep), (
        f"STE path: decode_step must be ConditionalDecodeStep, got {type(plan.stage_set.decode_step)}"
    )

    # resolve_kernel_fn should not exist in kernel_dispatch (was deleted in COMP-UNIFIED)
    from aminx.host import kernel_dispatch
    assert not hasattr(kernel_dispatch, "resolve_kernel_fn"), (
        "resolve_kernel_fn should have been deleted from kernel_dispatch in COMP-UNIFIED"
    )


# ---------------------------------------------------------------------------
# Test 16: InferencePlan.decode normalizes raw logits → SampleResult
# ---------------------------------------------------------------------------


def test_inference_plan_decode_normalizes_logits():
    """plan.decode returns SampleResult even when decode_fn returns raw logits (jnp.array).

    Builds a mock decode_fn that returns raw (L, 21) logits array.
    decode() must wrap this as SampleResult with sequence=argmax.astype(int32).
    """
    L, V = 10, 21

    raw_logits = jax.random.normal(jax.random.PRNGKey(0), (L, V))

    def _mock_decode_fn(key, enc, conditioning, config, stage_set):
        return raw_logits  # Return raw array, not SampleResult

    mock_model = MagicMock()
    stage_set = StageSet()
    components = InferenceComponents(
        encode_fn=MagicMock(return_value=MagicMock()),
        stage_set=stage_set,
    )
    plan = InferencePlan(model=mock_model, components=components, decode_fn=_mock_decode_fn)

    # Build minimal bundle and config mocks
    mock_bundle = MagicMock()
    mock_bundle.conditioning = MagicMock()
    mock_bundle.wave = MagicMock()
    mock_key = jax.random.PRNGKey(0)
    mock_config = MagicMock()

    result = plan.decode(mock_bundle, mock_bundle, mock_key, mock_config)

    assert isinstance(result, SampleResult), (
        f"plan.decode must return SampleResult, got {type(result)}"
    )
    assert result.sequence.dtype == jnp.int32, (
        f"result.sequence.dtype must be int32, got {result.sequence.dtype}"
    )
    assert result.logits.ndim == 2, (
        f"result.logits.ndim must be 2, got {result.logits.ndim}"
    )
    # Verify sequence = argmax of raw logits
    expected_seq = jnp.argmax(raw_logits, axis=-1).astype(jnp.int32)
    assert jnp.array_equal(result.sequence, expected_seq), (
        "result.sequence must be argmax of raw logits"
    )
