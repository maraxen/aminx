"""Debt #572 follow-up: state_position_map must thread from SamplingSpecification
through _sample_batch's build_inference_bundle calls, mirroring tie_group_map/state_weights.

These tests spy on build_inference_bundle rather than re-verifying realignment math
(already covered by tests/inference/decode/test_kernel.py and test_conditional.py /
test_autoregressive.py) — the wiring gap this closes is purely "does the value reach
the bundle builder with the right shape," not "does fusion combine it correctly."
"""

from __future__ import annotations

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp

from aminx.host.plan import InferencePlan
from aminx.run.specs import SamplingSpecification
from aminx.tiling.strategy import Vmap
from aminx.types.stages import StageSet
from aminx.utils.data_structures import Protein

_STATE_POSITION_MAP = jnp.array(
    [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, -1],
    ],
    dtype=jnp.int32,
)


def _make_fake_protein(batch_size: int = 1, seq_len: int = 5) -> Protein:
    return Protein(
        coordinates=jnp.zeros((batch_size, seq_len, 4, 3), dtype=jnp.float32),
        aatype=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        atom_mask=jnp.ones((batch_size, seq_len, 37), dtype=jnp.float32),
        residue_index=jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0),
        chain_index=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        mask=jnp.ones((batch_size, seq_len), dtype=jnp.float32),
        mapping=None,
    )


def _make_dispatch_monkeypatches(
    monkeypatch, *, batch_plan: MagicMock, target_num_samples: int = 1, batch_size: int = 1, seq_len: int = 5
) -> None:
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.make_sampling_planner",
        lambda spec, **kwargs: batch_plan,
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.extract_batch_sizes",
        lambda plan: (1, 1, 1, 1),
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
        lambda spec, chunk_count, grid: target_num_samples,
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._prepare_ligand_context",
        lambda *args, **kwargs: {
            "Y": None,
            "Y_t": None,
            "Y_m": None,
            "atom_37": None,
            "atom_37_mask": None,
            "chain_mask": None,
        },
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch._prepare_fixed_controls",
        lambda *args, **kwargs: (
            jnp.zeros((batch_size, seq_len), dtype=jnp.float32),
            jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        ),
    )
    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.compute_sample_keys",
        lambda *args, **kwargs: jnp.stack([jax.random.key(0)]),
    )


def _make_decision_for_spy() -> MagicMock:
    def _decision_for(plan, axis_name: str) -> MagicMock:
        decision = MagicMock()
        decision.strategy = Vmap()
        return decision

    return MagicMock(side_effect=_decision_for)


def _make_plan_mock() -> MagicMock:
    plan = MagicMock(spec=InferencePlan)
    stage_set = MagicMock(spec=StageSet)
    stage_set.encoding_fusion = None
    stage_set.encoder_sink = ()
    plan.stage_set = stage_set
    plan.encode = MagicMock(return_value=MagicMock())
    plan.decode = MagicMock(
        return_value=MagicMock(
            sequence=jnp.zeros(5, dtype=jnp.int32),
            logits=jnp.zeros((5, 21), dtype=jnp.float32),
        )
    )
    return plan


def _run_sample_batch_capturing_bundle_kwargs(
    monkeypatch, spec, *, batch_size: int = 1, order_calls: list | None = None,
) -> list[dict]:
    from aminx.host.kernel_dispatch import _sample_batch

    # The bundles here are MagicMocks, so the per-sample random decoding order cannot run
    # for real; record that it was requested instead.
    def _spy_with_decoding_order(bundle, key, decoding_order_fn=None, num_groups=None):
        if order_calls is not None:
            order_calls.append({"bundle": bundle, "key": key, "fn": decoding_order_fn})
        return bundle

    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.with_decoding_order", _spy_with_decoding_order,
    )

    batch_plan = MagicMock()
    _make_dispatch_monkeypatches(monkeypatch, batch_plan=batch_plan, batch_size=batch_size)
    monkeypatch.setattr("aminx.host.kernel_dispatch.decision_for", _make_decision_for_spy())

    captured_calls: list[dict] = []

    def _spy_build_inference_bundle(**kwargs):
        captured_calls.append(kwargs)
        return MagicMock(), MagicMock()

    monkeypatch.setattr(
        "aminx.host.kernel_dispatch.build_inference_bundle",
        _spy_build_inference_bundle,
    )

    def _spy_dispatch_axis(strategy, body, xs, *, batch_size_fallback: int = 0):
        # Real Python-level loop over xs (ignores the declared strategy entirely) so
        # the real kernel body -- and thus the build_inference_bundle spy -- actually
        # runs, instead of jax.vmap-tracing through MagicMock encode/decode results.
        length = xs.shape[0] if hasattr(xs, "shape") else len(xs)
        results = [body(xs[i]) for i in range(length)]
        return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *results)

    monkeypatch.setattr("aminx.host.kernel_dispatch._dispatch_axis", _spy_dispatch_axis)

    plan = _make_plan_mock()

    _sample_batch(
        spec,
        _make_fake_protein(batch_size=batch_size),
        plan,
        batch_idx=0,
        structure_batch_count=1,
        emit_structure_batch_io=False,
    )
    jax.effects_barrier()

    return captured_calls


def test_state_position_map_reaches_build_inference_bundle(monkeypatch):
    """A SamplingSpecification with state_position_map set must thread the exact
    (S, L) array through to build_inference_bundle's state_position_map kwarg."""
    spec = SamplingSpecification(
        inputs=["/tmp/test.pdb"],
        checkpoint_id="ckpt_001",
        state_position_map=_STATE_POSITION_MAP,
    )

    calls = _run_sample_batch_capturing_bundle_kwargs(monkeypatch, spec)

    assert calls, "build_inference_bundle was never called"
    for call in calls:
        assert call["state_position_map"] is not None
        assert jnp.array_equal(call["state_position_map"], _STATE_POSITION_MAP)


def test_state_position_map_none_passes_none_through(monkeypatch):
    """Default (no state_position_map) must pass None -- bundle_builder resolves
    the identity default itself, same as today's tie_group_map behavior."""
    spec = SamplingSpecification(
        inputs=["/tmp/test.pdb"],
        checkpoint_id="ckpt_001",
    )

    calls = _run_sample_batch_capturing_bundle_kwargs(monkeypatch, spec)

    assert calls, "build_inference_bundle was never called"
    for call in calls:
        assert call["state_position_map"] is None


def test_state_position_map_broadcasts_across_batch(monkeypatch):
    """A single (S, L) state_position_map shared across a multi-structure batch
    must reach every structure's build_inference_bundle call unchanged."""
    spec = SamplingSpecification(
        inputs=["/tmp/a.pdb", "/tmp/b.pdb"],
        checkpoint_id="ckpt_001",
        state_position_map=_STATE_POSITION_MAP,
    )

    calls = _run_sample_batch_capturing_bundle_kwargs(monkeypatch, spec, batch_size=2)

    assert len(calls) >= 2
    for call in calls:
        assert jnp.array_equal(call["state_position_map"], _STATE_POSITION_MAP)


def test_every_sample_decodes_with_its_own_random_order(monkeypatch):
    """runner.sample's default is a random decoding order per sample, not N->C.

    The order key is DERIVED from the sample key (decoding_order_key), so the decode's own
    key stream is untouched, and no decoding_order_fn is set by default (the shared random
    design order is used).
    """
    from aminx.inference.bundle_builder import decoding_order_key

    spec = SamplingSpecification(inputs=["/tmp/test.pdb"], checkpoint_id="ckpt_001")
    order_calls: list = []
    _run_sample_batch_capturing_bundle_kwargs(monkeypatch, spec, order_calls=order_calls)

    assert order_calls, "no sample was given a decoding order -- decoding would be N->C"
    sample_key = jax.random.key(0)  # compute_sample_keys is stubbed to this single key
    for call in order_calls:
        assert call["fn"] is None
        assert jnp.array_equal(
            jax.random.key_data(call["key"]),
            jax.random.key_data(decoding_order_key(sample_key)),
        )
