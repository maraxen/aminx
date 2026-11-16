#!/usr/bin/env python3
"""Decision-parity + recompile/throughput micro-benchmark: aminx.tiling vs xtrax.tiling.

aminx.tiling is a deliberate parallel reimplementation of xtrax.tiling (see
backlog EPIC #1541, "aminx->xtrax refactor", T2-T5). This script feeds direct
evidence into that epic's T2.GATE (#1556) DoD: "upgraded xtrax tiling
reproduces aminx bit-for-bit + identical recompile-count + throughput<=X%".

Two distinct comparisons, both on equivalent toy AxisSpecs/strategies:

1. Planner-level decision parity: does xtrax's own BatchPlanner.plan() choose
   the same strategy TYPE as aminx's, for the same (cardinality,
   default_batch_size) pair? (independent-libraries comparison; feeds
   confidence for a later P3 BatchPlanner migration, not this script's main
   claim)

2. Dispatch-level legacy-vs-adapter parity (THE load-bearing comparison for
   T2.GATE / T2.4): given the SAME aminx-native strategy decision, does
   aminx.tiling.dispatch.make_axis_dispatch_via_xtrax (T2.4's migration-ready
   adapter, added 2026-07-02) produce identical recompile-count and
   acceptable throughput overhead vs the legacy
   aminx.tiling.dispatch.make_axis_dispatch -- i.e., does swapping ONLY the
   dispatch call (the actual change T3's flip will make) preserve behavior?

Covers Vmap, SafeMap (both via the cardinality-driven AXIS_CASES sweep) and
Scan (a separate case -- Scan is never planner-selected by cardinality, so it
needs a directly-constructed strategy and a carry-bearing toy transition to
exercise jax.lax.scan's specific machinery, not just the Vmap/SafeMap map
primitives).

This is NOT a benchmark of aminx's production inference path -- it is a
synthetic, toy-function test of the tiling *primitive* in isolation. The
still-missing pieces for T2.GATE, deliberately out of scope here: (a) a
model-level bit-for-bit golden fixture (see
tests/tiling/test_t2_gate_bitforbit_golden.py), (b) a cluster GPU throughput
bench on production-representative shapes (needs real cluster submission,
scoped separately).

Usage:
    uv run python scripts/benchmarks/bench_xtrax_vs_aminx_tiling.py \
        --n-warmup 3 --n-timed 10 --out outputs/results/xtrax_vs_aminx_tiling.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax.numpy as jnp
from jax import random

logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

DATA_WIDTH = 8  # per-row feature width for the synthetic (cardinality, DATA_WIDTH) array

# (cardinality, default_batch_size) pairs spanning Vmap and SafeMap regimes.
# SafeMap cases use cardinality evenly divisible by batch_size: both aminx's
# safe_map and xtrax's safe_map raise ValueError on non-divisible chunking
# rather than silently padding (verified empirically — not a documented
# behavior difference, both libraries agree here).
AXIS_CASES: list[tuple[int, int]] = [
    (16, 32),   # cardinality <= batch_size -> Vmap
    (32, 32),   # boundary case -> Vmap
    (96, 32),   # cardinality > batch_size, evenly divisible -> SafeMap
    (256, 64),  # larger chunk count, evenly divisible -> SafeMap
]

SCAN_CARDINALITY = 32  # length of the toy carry-scan sequence


def _make_compile_counter() -> tuple[dict[str, int], object]:
    """Build a toy fn that increments a counter once per JAX trace."""
    counter = {"n": 0}

    def toy_fn(x):
        counter["n"] += 1  # executes once per Python-level trace under jit
        return x * 2.0 + 1.0

    return counter, toy_fn


def _make_scan_compile_counter() -> tuple[dict[str, int], object]:
    """Build a toy carry-bearing (carry, x) -> (carry, y) transition."""
    counter = {"n": 0}

    def toy_transition(carry, x):
        counter["n"] += 1  # executes once per Python-level trace under jit
        new_carry = carry + x
        return new_carry, new_carry * 2.0

    return counter, toy_transition


def _plan_aminx(cardinality: int, default_batch_size: int, data_width: int = DATA_WIDTH):
    """Aminx-native strategy decision for the given (cardinality, default_batch_size).

    EPIC #1541 T-PLANNER.GATE (2026-07-06) retired aminx.tiling.planner's
    BatchPlanner/AxisSpec -- the planner itself now delegates to
    xtrax.tiling.BatchPlanner (see host/plan.py's _plan_with_joint_budget),
    gated by tests/host/test_t_planner_gate_parity.py, which is the real,
    rigorous version of the comparison this function's docstring originally
    described ("feeds confidence for a later P3 BatchPlanner migration").
    That comparison is done; there is only one planning algorithm now.

    This function's remaining job -- producing an aminx-native AxisStrategy
    for _dispatch_aminx_legacy/_dispatch_via_adapter below, which both need
    one by contract -- reduces to the same deterministic cardinality-vs-
    batch_size rule AXIS_CASES documents (both libraries have always agreed
    on this rule for the non-heterogeneous, non-budget case), constructed
    directly rather than through any planner call.
    """
    del data_width  # no longer used; kept for call-site compatibility
    from aminx.tiling.strategy import ChunkedMap as AminxChunkedMap
    from aminx.tiling.strategy import Vmap as AminxVmap

    if cardinality <= default_batch_size:
        strategy = AminxVmap()
    else:
        strategy = AminxChunkedMap(batch_size=default_batch_size)

    return SimpleNamespace(strategy=strategy)


def _plan_xtrax(cardinality: int, default_batch_size: int):
    from xtrax.tiling.plan import AxisSpec as XtraxAxisSpec
    from xtrax.tiling.plan import BatchPlanner as XtraxBatchPlanner

    spec = XtraxAxisSpec(
        name="batch",
        cardinality=cardinality,
        default_batch_size=default_batch_size,
    )
    planner = XtraxBatchPlanner()
    plan = planner.plan([spec])
    return plan.decisions[0]


def _dispatch_aminx_legacy(strategy, axis: str = "batch"):
    """The path production call sites use today."""
    from aminx.tiling.dispatch import make_axis_dispatch

    return make_axis_dispatch(strategy, axis=axis)


def _dispatch_via_adapter(strategy, axis: str = "batch"):
    """Direct xtrax.tiling.make_axis_dispatch comparison path."""
    from xtrax.tiling import make_axis_dispatch as xtrax_make_axis_dispatch

    return xtrax_make_axis_dispatch(strategy, axis=axis, heterogeneous_axes={"state"})


def _time_iterator(
    iterator,
    toy_fn,
    data,
    n_warmup: int,
    n_timed: int,
) -> list[float]:
    @eqx.filter_jit
    def run(xs):
        return iterator(toy_fn, xs)

    for _ in range(n_warmup):
        out = run(data)
        out.block_until_ready()

    times: list[float] = []
    for _ in range(n_timed):
        start = time.perf_counter()
        out = run(data)
        out.block_until_ready()
        times.append(time.perf_counter() - start)
    return times


def _time_scan_iterator(
    iterator,
    transition,
    init,
    xs,
    n_warmup: int,
    n_timed: int,
) -> list[float]:
    @eqx.filter_jit
    def run(xs):
        return iterator(transition, init, xs)

    for _ in range(n_warmup):
        final_carry, ys = run(xs)
        final_carry.block_until_ready()

    times: list[float] = []
    for _ in range(n_timed):
        start = time.perf_counter()
        final_carry, ys = run(xs)
        final_carry.block_until_ready()
        times.append(time.perf_counter() - start)
    return times


def run_case(cardinality: int, default_batch_size: int, n_warmup: int, n_timed: int) -> dict:
    key = random.PRNGKey(0)
    data = random.normal(key, (cardinality, 8))

    aminx_decision = _plan_aminx(cardinality, default_batch_size)
    xtrax_decision = _plan_xtrax(cardinality, default_batch_size)

    aminx_strategy_name = type(aminx_decision.strategy).__name__
    xtrax_strategy_name = type(xtrax_decision.strategy).__name__
    decision_parity = aminx_strategy_name == xtrax_strategy_name

    legacy_iterator = _dispatch_aminx_legacy(aminx_decision.strategy)
    adapter_iterator = _dispatch_via_adapter(aminx_decision.strategy)

    legacy_counter, legacy_fn = _make_compile_counter()
    legacy_times = _time_iterator(legacy_iterator, legacy_fn, data, n_warmup, n_timed)
    legacy_recompiles = legacy_counter["n"]

    adapter_counter, adapter_fn = _make_compile_counter()
    adapter_times = _time_iterator(adapter_iterator, adapter_fn, data, n_warmup, n_timed)
    adapter_recompiles = adapter_counter["n"]

    legacy_median = float(jnp.median(jnp.array(legacy_times)))
    adapter_median = float(jnp.median(jnp.array(adapter_times)))
    throughput_ratio = adapter_median / legacy_median if legacy_median > 0 else float("nan")

    return {
        "strategy_kind": aminx_strategy_name,
        "cardinality": cardinality,
        "default_batch_size": default_batch_size,
        "aminx_strategy": aminx_strategy_name,
        "xtrax_strategy": xtrax_strategy_name,
        "decision_parity": decision_parity,
        "legacy_recompiles": legacy_recompiles,
        "adapter_recompiles": adapter_recompiles,
        "recompile_parity": legacy_recompiles == adapter_recompiles,
        "legacy_median_s": legacy_median,
        "adapter_median_s": adapter_median,
        "adapter_vs_legacy_throughput_ratio": throughput_ratio,
    }


def run_scan_case(n_warmup: int, n_timed: int) -> dict:
    """Scan is never planner-selected by cardinality -- construct directly."""
    from aminx.tiling.strategy import Scan as AminxScan

    key = random.PRNGKey(1)
    xs = random.normal(key, (SCAN_CARDINALITY,))
    init = jnp.array(0.0)

    strategy = AminxScan(init=init, transition=lambda c, x: (c + x, (c + x) * 2.0))

    legacy_iterator = _dispatch_aminx_legacy(strategy, axis="wave")
    adapter_iterator = _dispatch_via_adapter(strategy, axis="wave")

    legacy_counter, legacy_transition = _make_scan_compile_counter()
    legacy_times = _time_scan_iterator(
        legacy_iterator, legacy_transition, init, xs, n_warmup, n_timed,
    )
    legacy_recompiles = legacy_counter["n"]

    adapter_counter, adapter_transition = _make_scan_compile_counter()
    adapter_times = _time_scan_iterator(
        adapter_iterator, adapter_transition, init, xs, n_warmup, n_timed,
    )
    adapter_recompiles = adapter_counter["n"]

    legacy_median = float(jnp.median(jnp.array(legacy_times)))
    adapter_median = float(jnp.median(jnp.array(adapter_times)))
    throughput_ratio = adapter_median / legacy_median if legacy_median > 0 else float("nan")

    return {
        "strategy_kind": "Scan",
        "cardinality": SCAN_CARDINALITY,
        "default_batch_size": None,
        "aminx_strategy": "Scan",
        "xtrax_strategy": "Scan",
        "decision_parity": True,  # N/A: Scan is directly constructed, not planner-selected
        "legacy_recompiles": legacy_recompiles,
        "adapter_recompiles": adapter_recompiles,
        "recompile_parity": legacy_recompiles == adapter_recompiles,
        "legacy_median_s": legacy_median,
        "adapter_median_s": adapter_median,
        "adapter_vs_legacy_throughput_ratio": throughput_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-timed", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None, help="Path to write result JSON.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)

    cases = []
    for cardinality, default_batch_size in AXIS_CASES:
        logger.info(f"Running case cardinality={cardinality} batch_size={default_batch_size}")
        cases.append(run_case(cardinality, default_batch_size, args.n_warmup, args.n_timed))

    logger.info("Running Scan case")
    cases.append(run_scan_case(args.n_warmup, args.n_timed))

    all_decision_parity = all(c["decision_parity"] for c in cases)
    all_recompile_parity = all(c["recompile_parity"] for c in cases)
    max_throughput_ratio = max(c["adapter_vs_legacy_throughput_ratio"] for c in cases)

    result = {
        "cases": cases,
        "all_decision_parity": all_decision_parity,
        "all_recompile_parity": all_recompile_parity,
        "max_adapter_vs_legacy_throughput_ratio": max_throughput_ratio,
    }

    print(json.dumps(result, indent=2))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        logger.info(f"Wrote result JSON to {args.out}")

    if not all_decision_parity or not all_recompile_parity:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
