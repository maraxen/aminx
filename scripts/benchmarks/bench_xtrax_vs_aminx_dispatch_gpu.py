#!/usr/bin/env python3
"""Cluster GPU throughput bench: legacy dispatch vs T2.4's xtrax adapter.

EPIC #1541 / T2.GATE (#1556, `.praxia/docs/specs/260611_aminx-xtrax-refactor.md`).
The synthetic CPU microbenchmark (bench_xtrax_vs_aminx_tiling.py) confirmed
decision-parity and recompile-parity are 100% reproducible, but found its own
throughput measurement is NOT reproducible at microsecond scale (1.03x-1.79x
across identical repeated runs -- pure CPU timing noise, not signal). This
script is the actual T2.GATE throughput requirement: real GPU hardware,
production-representative shapes, enough repeats for a statistically stable
estimate.

Production-representative shape: L=208 (TEV protease, 1LVB.cif chain A --
the real structure already used and validated for this project's W1a.1
confirmatory campaign), synthetic random geometry (this benchmark measures
dispatch/tiling overhead, not model biological correctness, so synthetic
coordinates at a real production SIZE are sufficient -- no need to parse a
real CIF file). Batch-size (num_states) sweep spans a realistic range; the
necklace campaign's own (states x models x replicates x candidates) grid
does not yet fix concrete R/C values (still symbolic in the prereg), so this
sweeps a representative range rather than one pinned number.

Runs an actual ConditionalDecode call (same pattern as
tests/tiling/test_t2_gate_bitforbit_golden.py's bit-for-bit check, but timed
for throughput instead of checked for exact equality) through legacy
make_axis_dispatch vs T2.4's make_axis_dispatch_via_xtrax, on the identical
aminx-native strategy decision.

Per the spec's own premortem, this must be a STANDING gate re-run on the
production shape distribution, not a one-shot at flip time -- register every
real run via `bth run`, not just the first one.

Usage:
    # L1 (local dry-run): verify imports/paths, no compute
    uv run python scripts/benchmarks/bench_xtrax_vs_aminx_dispatch_gpu.py --dry-run

    # L2 (local smoke test, <60s, CPU is fine): one small case, few reps
    uv run python scripts/benchmarks/bench_xtrax_vs_aminx_dispatch_gpu.py --smoke \
        --out /tmp/smoke.json

    # L3 / full run (needs GPU): submit via myxcel preempt-gpu preset
    uv run python scripts/benchmarks/bench_xtrax_vs_aminx_dispatch_gpu.py \
        --n-warmup 10 --n-timed 50 \
        --out outputs/results/xtrax_vs_aminx_dispatch_gpu.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

SEQ_LEN = 208  # TEV protease, 1LVB.cif chain A -- the real structure this
# project's W1a.1 confirmatory campaign already used and validated.

# num_states values spanning Vmap and SafeMap regimes at production scale.
# The necklace campaign's (states x models x replicates x candidates) grid
# does not yet fix concrete R/C values (prereg leaves them symbolic) -- this
# sweeps a representative range rather than one pinned number.
NUM_STATES_CASES: list[int] = [1, 8, 32, 128]

SMOKE_NUM_STATES_CASES: list[int] = [4]


def _build_fixture(num_states: int, seq_len: int, seed: int = 42):
    """Synthetic geometry at a production-representative size.

    Not real TEV protease geometry (random coordinates) -- this benchmark
    measures dispatch/tiling overhead, not model biological correctness, so
    a real CIF fixture isn't needed. seq_len=208 alone is what makes this
    "production-representative", matching this project's actual TEV
    protease structure size (1LVB.cif chain A).
    """
    from aminx.inference.bundle_builder import build_inference_bundle
    from aminx.model import Aminx

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

    coordinates = jnp.array(rng.normal(size=(seq_len, 4, 3)).astype(np.float32))
    mask = jnp.ones((seq_len,), dtype=jnp.float32)
    residue_index = jnp.arange(seq_len, dtype=jnp.int32)
    chain_index = jnp.zeros((seq_len,), dtype=jnp.int32)

    coordinates_stack = jnp.stack([coordinates] * num_states, axis=0)
    mask_stack = jnp.stack([mask] * num_states, axis=0)
    residue_index_stack = jnp.stack([residue_index] * num_states, axis=0)
    chain_index_stack = jnp.stack([chain_index] * num_states, axis=0)

    sequence_tokens = jnp.array(rng.integers(0, 20, size=(seq_len,), dtype=np.int32))
    sequence_oh = jax.nn.one_hot(sequence_tokens, 21)

    state_weights = jnp.ones(num_states) / num_states
    bundle, config = build_inference_bundle(
        coords=coordinates_stack,
        mask=mask_stack,
        residue_index=residue_index_stack,
        chain_index=chain_index_stack,
        sequence=sequence_oh,
        state_weights=state_weights,
        mode="score_conditional",
    )
    return model, bundle, config


def run_case(num_states: int, seq_len: int, n_warmup: int, n_timed: int) -> dict:
    import time

    from aminx.host.plan import _plan_with_joint_budget
    from aminx.inference.decode.conditional import ConditionalDecode
    from aminx.inference.encode import make_encode_fn
    from aminx.inference.logits import make_stage_set
    from aminx.tiling.dispatch import make_axis_dispatch
    from xtrax.tiling import AxisSpec
    from xtrax.tiling import make_axis_dispatch as xtrax_make_axis_dispatch

    model, bundle, config = _build_fixture(num_states, seq_len)

    k_enc, k_dec = jax.random.split(jax.random.PRNGKey(0))
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)
    stage_set = make_stage_set(
        strategy="arithmetic_mean", state_weights=bundle.conditioning.state_weights,
    )

    # Let the real planner (EPIC #1541: xtrax.tiling.BatchPlanner via
    # host.plan._plan_with_joint_budget) choose Vmap vs SafeMap for this
    # num_states, same as production would -- not hand-picking the strategy.
    default_batch_size = 32
    spec = AxisSpec(
        name="state",
        cardinality=num_states,
        default_batch_size=default_batch_size,
        tile_granularity=default_batch_size,
        heterogeneous=True,  # state is aminx's canonical heterogeneous axis
    )
    elements_per_row = seq_len * 4 * 3  # coords shape per state, rough proxy
    estimate_fn = lambda decisions: (  # noqa: E731
        decisions[0].spec.cardinality
        if type(decisions[0].strategy).__name__ == "Vmap"
        else decisions[0].batch_size
    ) * elements_per_row
    plan = _plan_with_joint_budget(
        [spec],
        budget_bytes=default_batch_size * elements_per_row,
        estimate_fn=estimate_fn,
    )
    xtrax_decision = plan.decisions[0]
    strategy_name = type(xtrax_decision.strategy).__name__
    decision_strategy = xtrax_decision.strategy

    def _make_counter():
        counter = {"n": 0}

        def cond_decode_with_counter(iterator):
            counter["n"] += 1
            return ConditionalDecode(model=model, state_iterator=iterator)

        return counter, cond_decode_with_counter

    def _time_path(dispatch_fn):
        iterator = dispatch_fn(decision_strategy, axis="state")
        counter, make_decode = _make_counter()

        @eqx.filter_jit
        def run(key):
            counter["n"] += 1
            decode = ConditionalDecode(model=model, state_iterator=iterator)
            return decode(key=key, enc=enc, bundle=bundle, config=config, stage_set=stage_set)

        for _ in range(n_warmup):
            out = run(k_dec)
            jax.block_until_ready(out)

        times: list[float] = []
        for _ in range(n_timed):
            start = time.perf_counter()
            out = run(k_dec)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - start)

        # Recompiles = trace count minus warmup+timed calls that hit cache;
        # counter increments once per Python-level trace under filter_jit.
        return times, counter["n"]

    legacy_times, legacy_recompiles = _time_path(make_axis_dispatch)
    adapter_times, adapter_recompiles = _time_path(
        lambda s, *, axis="state": xtrax_make_axis_dispatch(
            s, axis=axis, heterogeneous_axes={"state"},
        )
    )

    legacy_mean = float(np.mean(legacy_times))
    legacy_stderr = float(np.std(legacy_times, ddof=1) / np.sqrt(len(legacy_times)))
    adapter_mean = float(np.mean(adapter_times))
    adapter_stderr = float(np.std(adapter_times, ddof=1) / np.sqrt(len(adapter_times)))
    throughput_ratio = adapter_mean / legacy_mean if legacy_mean > 0 else float("nan")

    return {
        "num_states": num_states,
        "seq_len": seq_len,
        "strategy": strategy_name,
        "legacy_recompiles": legacy_recompiles,
        "adapter_recompiles": adapter_recompiles,
        "recompile_parity": legacy_recompiles == adapter_recompiles,
        "legacy_mean_s": legacy_mean,
        "legacy_stderr_s": legacy_stderr,
        "adapter_mean_s": adapter_mean,
        "adapter_stderr_s": adapter_stderr,
        "adapter_vs_legacy_throughput_ratio": throughput_ratio,
        "n_warmup": n_warmup,
        "n_timed": n_timed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-timed", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="L1 gate: verify imports only")
    parser.add_argument(
        "--smoke", action="store_true", help="L2 gate: one small case, few reps, <60s",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Re-emit an already-computed result JSON (e.g. computed on cluster, "
        "pulled locally) for bathos tracking, without recomputing.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)

    if args.replay is not None:
        result = json.loads(args.replay.read_text())
        print(json.dumps(result, indent=2))
        if args.out is not None and args.out != args.replay:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2))
        sys.exit(0)

    if args.dry_run:
        from aminx.inference.decode.conditional import ConditionalDecode  # noqa: F401
        from aminx.tiling.dispatch import make_axis_dispatch  # noqa: F401
        from aminx.tiling.strategy import ChunkedMap, Vmap  # noqa: F401

        print(json.dumps({"dry_run": "ok", "device": str(jax.devices())}))
        sys.exit(0)

    if args.smoke:
        cases_spec = SMOKE_NUM_STATES_CASES
        n_warmup, n_timed = 2, 3
    else:
        cases_spec = NUM_STATES_CASES
        n_warmup, n_timed = args.n_warmup, args.n_timed

    logger.info(f"Device: {jax.devices()}")

    cases = []
    for num_states in cases_spec:
        logger.info(f"Running num_states={num_states} seq_len={args.seq_len}")
        cases.append(run_case(num_states, args.seq_len, n_warmup, n_timed))

    all_recompile_parity = all(c["recompile_parity"] for c in cases)
    max_throughput_ratio = max(c["adapter_vs_legacy_throughput_ratio"] for c in cases)

    result = {
        "cases": cases,
        "all_recompile_parity": all_recompile_parity,
        "max_adapter_vs_legacy_throughput_ratio": max_throughput_ratio,
        "device": str(jax.devices()),
    }

    print(json.dumps(result, indent=2))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        logger.info(f"Wrote result JSON to {args.out}")

    sys.exit(0 if all_recompile_parity else 1)


if __name__ == "__main__":
    main()
