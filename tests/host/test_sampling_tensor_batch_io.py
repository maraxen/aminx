"""Tests for COMP-NEW: tensor io_callback emission and sink integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.host.kernel_dispatch import _sample_batch
from aminx.host.output_sinks import (
    active_sampling_staging_sink,
    streaming_tensor_sink_session,
    take_staging_sequences_logits,
)
from aminx.inference.sample_autoregressive import SampleResult
from aminx.run.specs import SamplingSpecification
from aminx.utils.data_structures import Protein
from xtrax.tiling import ChunkedMap


def _patch_decision_for(monkeypatch) -> None:
    """Stub aminx.host.kernel_dispatch.decision_for to always resolve to SafeMap.

    decision_for is a free function (not a BatchPlan method) since EPIC #1541
    T-PLANNER.2 -- a bare MagicMock() standing in for make_sampling_planner's
    return value no longer auto-succeeds the way plan.decision_for(name) used
    to. SafeMap (not Vmap) specifically: these tests patch _safe_map itself to
    short-circuit deep execution (see _mock_safe_map below) -- Vmap's
    _dispatch_axis branch calls jax.vmap directly and bypasses that mock
    entirely, reaching real body execution and failing on fixture gaps
    (e.g. fixed_mask_for_vmap=None) these tests never intended to exercise.
    """
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.decision_for",
        lambda plan, name: MagicMock(strategy=ChunkedMap(batch_size=1)),
    )


def _make_fake_protein(
    batch_size: int = 1, seq_len: int = 10
) -> Protein:
    """Create a minimal Protein namedtuple for testing.

    Args:
        batch_size: Number of structures in batch
        seq_len: Sequence length (residues)

    Returns:
        Protein namedtuple with correct shapes for _sample_batch
    """
    return Protein(
        coordinates=jnp.zeros((batch_size, seq_len, 4, 3), dtype=jnp.float32),
        aatype=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        atom_mask=jnp.ones((batch_size, seq_len, 37), dtype=jnp.float32),
        residue_index=jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0),
        chain_index=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        mask=jnp.ones((batch_size, seq_len), dtype=jnp.float32),
        mapping=None,
    )


def _make_stub_kernel(seq_len: int = 10, vocab: int = 21):
    """Create a stub kernel that returns fixed-shape SampleResult.

    Args:
        seq_len: Sequence length for generated sequences
        vocab: Vocabulary size (default 21 for amino acids)

    Returns:
        Callable stub kernel compatible with _sample_batch
    """
    def _stub(prng_key, bundle, config, stage_set):
        seq = jnp.zeros((seq_len,), dtype=jnp.int32)
        logits = jnp.zeros((seq_len, vocab), dtype=jnp.float32)
        return SampleResult(sequence=seq, logits=logits)

    return _stub


def _make_mock_plan(stage_set: "MagicMock | None" = None) -> "MagicMock":
    """Create a mock InferencePlan for tests that bypasses real model/encoding.

    Sets encoding_fusion=None and encoder_sink=None to select Path A dispatch.
    """
    from aminx.host.plan import InferencePlan

    mock = MagicMock(spec=InferencePlan)
    if stage_set is not None:
        mock.stage_set = stage_set
    mock.stage_set.encoding_fusion = None
    mock.stage_set.encoder_sink = None
    return mock


# ---------------------------------------------------------------------------
# Test 1: Direct io_callback staging
# ---------------------------------------------------------------------------


def test_dispatch_tensor_io_callback_stages_to_active_sink():
    """_dispatch_sampling_tensor_batch_io stages sequences/logits to active sink."""
    from aminx.host._sampling_helper import (
        _dispatch_sampling_tensor_batch_io,
    )

    batch_idx = 0
    batch_count = 1
    chunk_start = 0
    chunk_count = 4
    seq_len = 10
    num_samples = 2

    # Create numpy arrays matching expected shape
    sequences_np = np.zeros((1, num_samples, 1, 1, seq_len), dtype=np.int32)
    logits_np = np.zeros((1, num_samples, 1, 1, seq_len, 21), dtype=np.float32)

    with streaming_tensor_sink_session():
        # Verify sink is active
        sink = active_sampling_staging_sink()
        assert sink is not None, "Sink must be active in streaming_tensor_sink_session()"

        # Call the io_callback handler directly
        _dispatch_sampling_tensor_batch_io(
            batch_idx,
            batch_count,
            chunk_start,
            chunk_count,
            sequences_np,
            logits_np,
        )

        # Drain and verify shapes
        seqs, logits = take_staging_sequences_logits(batch_idx, chunk_start, chunk_count)
        assert seqs is not None, "Sequences should be staged"
        assert logits is not None, "Logits should be staged"
        assert seqs.shape[0] == 1, f"Expected batch size 1, got {seqs.shape[0]}"
        assert logits.shape[0] == 1, f"Expected batch size 1, got {logits.shape[0]}"


def test_dispatch_tensor_io_callback_visible_from_other_thread():
    """The active sink must be visible from a thread other than the one that
    entered streaming_tensor_sink_session().

    Regression test: jax.experimental.io_callback(ordered=False) dispatches
    unordered callbacks via a background thread pool, and contextvars.ContextVar
    values set on the calling thread are NOT visible on that pool thread. This
    silently no-op'd _dispatch_sampling_tensor_batch_io's write whenever JAX
    happened to schedule the callback off-thread (confirmed empirically: it did
    for multi-structure batches, e.g. a PoE bead's multi-input fusion, and
    didn't for single-structure batches, which is why this went uncaught --
    every existing test in this file calls the dispatch function on the same
    thread that entered the session). active_sampling_staging_sink() now falls
    back to a plain module global (see output_sinks._active_streaming_tensor_sink_io)
    for exactly this case, mirroring the pre-existing _jacobian_sink_ctx pattern.
    """
    import threading

    from aminx.host._sampling_helper import _dispatch_sampling_tensor_batch_io

    batch_idx, batch_count, chunk_start, chunk_count = 0, 1, 0, 4
    seq_len, num_samples = 10, 2
    sequences_np = np.zeros((1, num_samples, 1, 1, seq_len), dtype=np.int32)
    logits_np = np.zeros((1, num_samples, 1, 1, seq_len, 21), dtype=np.float32)

    results: dict[str, object] = {}

    def _call_from_other_thread() -> None:
        try:
            results["sink_seen"] = active_sampling_staging_sink() is not None
            _dispatch_sampling_tensor_batch_io(
                batch_idx, batch_count, chunk_start, chunk_count, sequences_np, logits_np,
            )
        except Exception as exc:  # noqa: BLE001
            results["error"] = exc

    with streaming_tensor_sink_session():
        thread = threading.Thread(target=_call_from_other_thread)
        thread.start()
        thread.join(timeout=5)

        assert "error" not in results, f"Dispatch from other thread raised: {results.get('error')}"
        assert results.get("sink_seen") is True, (
            "active_sampling_staging_sink() returned None on a different thread "
            "than the one that entered streaming_tensor_sink_session() -- the "
            "fallback global is not working."
        )

        # The real bug: this must find the entry the other thread just staged.
        seqs, logits = take_staging_sequences_logits(batch_idx, chunk_start, chunk_count)
        assert seqs.shape[0] == 1
        assert logits.shape[0] == 1


# ---------------------------------------------------------------------------
# Test 2: io_callback emission can be controlled by emit_structure_batch_io flag
# ---------------------------------------------------------------------------


def test_emit_structure_batch_io_gate_in_sample_batch(monkeypatch):
    """_sample_batch emits scalar marker io_callback only when emit_structure_batch_io=True."""
    # Count _noop_sampling_structure_batch_io invocations
    call_log = {"structure": 0}

    def _counting_noop(batch_idx, batch_count):
        call_log["structure"] += 1

    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._noop_sampling_structure_batch_io",
        _counting_noop,
    )

    # Create a mock BatchPlan
    mock_plan = MagicMock()
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.make_sampling_planner",
        lambda spec, **kwargs: mock_plan,
    )
    _patch_decision_for(monkeypatch)

    # Patch extract_batch_sizes to return dummy batch sizes
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.extract_batch_sizes",
        lambda plan: (1, 2, 1, 1),
    )

    # Patch grid and key resolution
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

    # Patch _safe_map to bypass all the nested vmap layers and return dummy output arrays
    def _mock_safe_map(fn, xs, batch_size=None):
        # Return dummy arrays matching expected output shapes
        # The shape convention is: (batch, samples, noise, temp, seq_len) and
        # (batch, samples, noise, temp, seq_len, vocab)
        return (
            jnp.zeros((1, 2, 1, 1, 10), dtype=jnp.int32),
            jnp.zeros((1, 2, 1, 1, 10, 21), dtype=jnp.float32),
        )

    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._safe_map", _mock_safe_map
    )

    # Patch helper functions to avoid ligand/control prep failures
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

    spec = SamplingSpecification(
        inputs=["/tmp/test.pdb"],
        checkpoint_id="ckpt_001",
        num_samples=2,
        temperature=[1.0],
        backbone_noise=[0.0],
        sampling_strategy="temperature",
        compute_pseudo_perplexity=False,
    )
    batched_ensemble = _make_fake_protein(batch_size=1, seq_len=10)
    stage_set = MagicMock(name="stage_set")
    inference_plan = _make_mock_plan(stage_set)

    # emit_structure_batch_io=False — scalar marker should NOT fire
    call_log["structure"] = 0
    _sample_batch(
        spec,
        batched_ensemble,
        inference_plan,
        batch_idx=0,
        structure_batch_count=1,
        emit_structure_batch_io=False,
    )
    jax.effects_barrier()
    assert call_log["structure"] == 0, (
        f"emit_structure_batch_io=False should not emit scalar marker, got {call_log['structure']}"
    )

    # emit_structure_batch_io=True — scalar marker SHOULD fire once
    call_log["structure"] = 0
    _sample_batch(
        spec,
        batched_ensemble,
        inference_plan,
        batch_idx=0,
        structure_batch_count=1,
        emit_structure_batch_io=True,
    )
    jax.effects_barrier()
    assert call_log["structure"] == 1, (
        f"emit_structure_batch_io=True should emit scalar marker once, got {call_log['structure']}"
    )


# ---------------------------------------------------------------------------
# Test 3: _sample_batch graceful no-op when no sink active
# ---------------------------------------------------------------------------


def test_sample_batch_does_not_raise_without_active_sink(monkeypatch):
    """_sample_batch does not raise when no streaming_tensor_sink_session() is active.

    _dispatch_sampling_tensor_batch_io checks active_sampling_staging_sink() is None
    and returns early — this validates the graceful no-op path.
    """

    # Create a mock BatchPlan
    mock_plan = MagicMock()
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.make_sampling_planner",
        lambda spec, **kwargs: mock_plan,
    )
    _patch_decision_for(monkeypatch)

    # Patch extract_batch_sizes to return dummy batch sizes
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.extract_batch_sizes",
        lambda plan: (1, 2, 1, 1),
    )

    # Patch grid and key resolution
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

    # Patch _safe_map to bypass all the nested vmap layers
    def _mock_safe_map(fn, xs, batch_size=None):
        # Return dummy arrays matching expected output shapes
        return (
            jnp.zeros((1, 2, 1, 1, 10), dtype=jnp.int32),
            jnp.zeros((1, 2, 1, 1, 10, 21), dtype=jnp.float32),
        )

    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._safe_map", _mock_safe_map
    )

    # Patch helper functions to avoid ligand/control prep failures
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

    spec = SamplingSpecification(
        inputs=["/tmp/test.pdb"],
        checkpoint_id="ckpt_001",
        num_samples=2,
        temperature=[1.0],
        backbone_noise=[0.0],
        sampling_strategy="temperature",
        compute_pseudo_perplexity=False,
    )
    batched_ensemble = _make_fake_protein(batch_size=1, seq_len=10)
    stage_set = MagicMock(name="stage_set")
    inference_plan = _make_mock_plan(stage_set)

    # No streaming_tensor_sink_session() context — should not raise
    _sample_batch(
        spec,
        batched_ensemble,
        inference_plan,
        batch_idx=0,
        structure_batch_count=1,
    )
    jax.effects_barrier()
    # If we get here without RuntimeError, the no-op path is working


# ---------------------------------------------------------------------------
# Test 4: _sample_batch stages when sink active
# ---------------------------------------------------------------------------


def test_sample_batch_stages_when_sink_active(monkeypatch):
    """_sample_batch stages sequences/logits to sink when streaming_tensor_sink_session() is active."""

    # Create a mock BatchPlan
    mock_plan = MagicMock()
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.make_sampling_planner",
        lambda spec, **kwargs: mock_plan,
    )
    _patch_decision_for(monkeypatch)

    # Patch extract_batch_sizes to return dummy batch sizes
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.extract_batch_sizes",
        lambda plan: (1, 2, 1, 1),
    )

    # Patch grid and key resolution
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

    # Patch _safe_map to bypass all the nested vmap layers
    def _mock_safe_map(fn, xs, batch_size=None):
        # Return dummy arrays matching expected output shapes
        return (
            jnp.zeros((1, 2, 1, 1, 10), dtype=jnp.int32),
            jnp.zeros((1, 2, 1, 1, 10, 21), dtype=jnp.float32),
        )

    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._safe_map", _mock_safe_map
    )

    # Patch helper functions to avoid ligand/control prep failures
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

    spec = SamplingSpecification(
        inputs=["/tmp/test.pdb"],
        checkpoint_id="ckpt_001",
        num_samples=2,
        temperature=[1.0],
        backbone_noise=[0.0],
        sampling_strategy="temperature",
        compute_pseudo_perplexity=False,
    )
    batched_ensemble = _make_fake_protein(batch_size=1, seq_len=10)
    stage_set = MagicMock(name="stage_set")
    inference_plan = _make_mock_plan(stage_set)

    with streaming_tensor_sink_session():
        _sample_batch(
            spec,
            batched_ensemble,
            inference_plan,
            batch_idx=0,
            structure_batch_count=1,
        )
        jax.effects_barrier()
        seqs, logits = take_staging_sequences_logits(0, 0, spec.num_samples)

    assert seqs is not None
    assert logits is not None


# ---------------------------------------------------------------------------
# Test 5: io_callback is generated by _sample_batch (integration test)
# ---------------------------------------------------------------------------


def test_sample_batch_io_callback_sources_exist():
    """Verify _sample_batch module has io_callback sources for both tensor and structure markers."""
    # This is a simple smoke test verifying the code was added
    from aminx.host import kernel_dispatch

    # Verify the imports exist
    assert hasattr(
        kernel_dispatch, "_dispatch_sampling_tensor_batch_io"
    ), "kernel_dispatch must import _dispatch_sampling_tensor_batch_io"
    assert hasattr(
        kernel_dispatch, "_noop_sampling_structure_batch_io"
    ), "kernel_dispatch must import _noop_sampling_structure_batch_io"

    # Verify _sample_batch is callable
    assert callable(kernel_dispatch._sample_batch)


# ---------------------------------------------------------------------------
# Test 6: emit_structure_batch_io parameter exists and defaults to True
# ---------------------------------------------------------------------------


def test_sample_batch_emit_structure_batch_io_parameter():
    """Verify _sample_batch accepts emit_structure_batch_io parameter with default True."""
    import inspect

    from aminx.host.kernel_dispatch import _sample_batch

    sig = inspect.signature(_sample_batch)
    assert (
        "emit_structure_batch_io" in sig.parameters
    ), "_sample_batch must accept emit_structure_batch_io parameter"

    param = sig.parameters["emit_structure_batch_io"]
    assert (
        param.default is True
    ), "emit_structure_batch_io must default to True"
