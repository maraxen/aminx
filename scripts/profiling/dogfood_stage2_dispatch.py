#!/usr/bin/env python3
"""Stage-2 GPU ProbeRecords from aminx's real ConditionalDecode via xtrax.

First external dogfood of xtrax.profiling (xtrax PR #99, squash 04f7557).
Mirrors scripts/benchmarks/bench_xtrax_vs_aminx_dispatch_gpu.py's
production-representative shape (L=208 TEV protease size, num_states sweep,
real planner choosing Vmap-vs-SafeMap) but benchmarks ONLY the xtrax
dispatch leg, wrapped in named scopes, under jax.profiler.trace + compiled
HLO -- emitting one stage-2 ProbeRecord per num_states case through
xtrax.profiling.emitters.emit_probe_record.

Two named scopes per record ("ebm_conditional_decode" outer, "ebm_state_axis"
around the dispatched iteration call) so TERM_RANKING's >=2-attributed-scopes
requirement is exercisable on real GPU evidence.

Stage-2 records require a GPU (platform='gpu' + auto-captured device_kind);
pass --stage 1 for a CPU smoke run (records then fail closed for ranking by
design).

Usage:
    # L1: imports only
    uv run python scripts/profiling/dogfood_stage2_dispatch.py --dry-run
    # L2: CPU smoke, one case, few reps (stage forced to 1)
    uv run python scripts/profiling/dogfood_stage2_dispatch.py --smoke \
        --out-dir /tmp/dogfood-smoke
    # L3: GPU full run (L40S, mit_preemptable):
    XTRAX_GIT_SHA=$(git rev-parse HEAD) \
        uv run python scripts/profiling/dogfood_stage2_dispatch.py \
        --n-warmup 10 --n-timed 50 --out-dir outputs/profiling/stage2
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from xtrax.profiling.claims import ClaimClass, permitted_claims
from xtrax.profiling.emitters import (
    ATTRIBUTION_NAMED_SCOPE,
    attribution_from_scopes,
    emit_probe_record,
)
from xtrax.profiling.trace import (
    parse_dispatch_counts,
    parse_scopes,
    scope_map_from_hlo_text,
)

ROOT = Path(__file__).resolve().parents[2]

# Scope-label registry (D8: vocabulary lives in drivers, never in the library
# package). Two labels so records carry >=2 attributed scopes.
LABEL_DECODE = "ebm_conditional_decode"
LABEL_STATE_AXIS = "ebm_state_axis"
KNOWN_LABELS = frozenset({LABEL_DECODE, LABEL_STATE_AXIS})

SEQ_LEN = 208  # TEV protease, 1LVB.cif chain A -- same production shape as
# scripts/benchmarks/bench_xtrax_vs_aminx_dispatch_gpu.py.
NUM_STATES_CASES: list[int] = [1, 8, 32]
SMOKE_NUM_STATES_CASES: list[int] = [4]


def _build_fixture(num_states: int, seq_len: int, seed: int = 42):
    """Synthetic geometry at production size; identical construction to
    bench_xtrax_vs_aminx_dispatch_gpu._build_fixture (cite-not-drift: changes
    there should be mirrored here)."""
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


def _load_trace_events(trace_dir: Path) -> list[dict]:
    trace_files = sorted(trace_dir.rglob("*.trace.json.gz"))
    if not trace_files:
        raise SystemExit(f"no *.trace.json.gz written under {trace_dir}")
    with gzip.open(trace_files[-1], "rt") as fh:
        data = json.load(fh)
    events = data.get("traceEvents")
    if not isinstance(events, list):
        raise SystemExit(f"{trace_files[-1]} has no traceEvents list")
    print(f"loaded {len(events)} events from {trace_files[-1].name}")
    return events


def run_case(
    num_states: int,
    seq_len: int,
    n_warmup: int,
    n_timed: int,
    out_dir: Path,
    stage: int,
) -> Path:
    from aminx.host.plan import _plan_with_joint_budget
    from aminx.inference.decode.conditional import ConditionalDecode
    from aminx.inference.encode import make_encode_fn
    from aminx.inference.logits import make_stage_set
    from aminx.tiling.dispatch import make_axis_dispatch_via_xtrax
    from aminx.tiling.strategy import SafeMap as AminxSafeMap
    from aminx.tiling.strategy import Vmap as AminxVmap
    from xtrax.tiling import AxisSpec

    model, bundle, config = _build_fixture(num_states, seq_len)

    k_enc, k_dec = jax.random.split(jax.random.PRNGKey(0))
    encode_fn = make_encode_fn(model, use_rolling_state=False)
    enc = encode_fn(bundle, k_enc, config)
    stage_set = make_stage_set(
        strategy="arithmetic_mean", state_weights=bundle.conditioning.state_weights,
    )

    default_batch_size = 32
    spec = AxisSpec(
        name="state",
        cardinality=num_states,
        default_batch_size=default_batch_size,
        tile_granularity=default_batch_size,
        heterogeneous=True,
    )
    elements_per_row = seq_len * 4 * 3
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
    decision = plan.decisions[0]
    strategy_name = type(decision.strategy).__name__
    decision_strategy = (
        AminxVmap() if strategy_name == "Vmap" else AminxSafeMap(tile=decision.batch_size)
    )

    iterator = make_axis_dispatch_via_xtrax(decision_strategy, axis="state")

    def _decode(key):
        # Both scopes live INSIDE the jitted region: the compiled HLO text
        # must carry their op_name paths for two-input attribution.
        with jax.named_scope(LABEL_DECODE):
            decode = ConditionalDecode(model=model, state_iterator=iterator)
            with jax.named_scope(LABEL_STATE_AXIS):
                return decode(key=key, enc=enc, bundle=bundle, config=config, stage_set=stage_set)

    run = eqx.filter_jit(_decode)

    for _ in range(max(n_warmup, 1)):
        run(k_dec).block_until_ready()

    case_dir = out_dir / f"states{num_states}"
    trace_dir = case_dir / "_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
        for _ in range(n_timed):
            run(k_dec).block_until_ready()
    total_step_seconds = time.perf_counter() - start

    # HLO text MUST come from the COMPILED executable: Lowered.as_text()
    # omits op_name metadata entirely, which silently blanks scope
    # attribution (found on the first L40S dogfood run -- every scope came
    # back ABSENT from an otherwise healthy trace).
    compiled_exec = eqx.filter_jit(_decode).lower(k_dec).compile()
    hlo_src = next(
        v for v in vars(compiled_exec).values() if type(v).__name__ == "Compiled"
    )
    hlo_text = hlo_src.as_text()
    hlo_path = case_dir / "hlo_as_text.txt"
    hlo_path.write_text(hlo_text)
    if not any(label in hlo_text for label in KNOWN_LABELS):
        raise SystemExit(
            "scope labels absent from compiled HLO text -- attribution "
            "would be empty; refusing to emit a degraded record"
        )

    events = _load_trace_events(trace_dir)
    scope_map = scope_map_from_hlo_text(hlo_text, KNOWN_LABELS)
    measured = parse_scopes(events, scope_map)
    scopes = {label: measured.get(label) for label in sorted(KNOWN_LABELS)}
    counts = parse_dispatch_counts(events)

    n_atoms = num_states * seq_len * 4  # heavy atoms in flight across states
    metrics: dict[str, float | int | str] = {
        "total_step_seconds": total_step_seconds,
        **counts,
    }
    path = case_dir / f"stage{stage}_ebm_cond_decode_states{num_states}.json"
    record = emit_probe_record(
        path=path,
        probe_id=f"stage{stage}_ebm_cond_decode_states{num_states}",
        stage=stage,
        n_atoms=n_atoms,
        platform="gpu" if jax.devices()[0].platform == "gpu" else "cpu",
        metrics=metrics,
        scopes=scopes,
        attribution_method={
            **attribution_from_scopes(scopes, method=ATTRIBUTION_NAMED_SCOPE),
        },
        config={
            "kernel": "aminx.ConditionalDecode",
            "dispatch": "make_axis_dispatch_via_xtrax",
            "strategy": strategy_name,
            "num_states": str(num_states),
            "seq_len": str(seq_len),
            "n_warmup": str(n_warmup),
            "n_timed": str(n_timed),
            "axis_note": "n_atoms == states x residues x 4 backbone atoms",
        },
    )

    permitted = sorted(permitted_claims(record), key=lambda c: c.name)
    print(f"wrote {path}")
    print(f"  strategy={strategy_name} n_atoms={n_atoms} wall={total_step_seconds:.4f}s")
    for label, value in scopes.items():
        if value is None:
            print(f"  {label}: ABSENT from trace")
        else:
            seconds, n_occ = value
            print(f"  {label}: {seconds:.6f}s over {n_occ} occ")
    print(f"  dispatch: {counts}")
    print(f"  permitted claims: {[c.name for c in permitted]}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "profiling" / "stage2")
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-timed", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument(
        "--num-states", type=int, nargs="+", default=None,
        help="Explicit sweep; overrides defaults/smoke.",
    )
    parser.add_argument("--stage", type=int, default=2, choices=(1, 2))
    parser.add_argument("--dry-run", action="store_true", help="L1 gate: imports only")
    parser.add_argument(
        "--smoke", action="store_true",
        help="L2 gate: CPU, one small case, few reps (forces --stage 1)",
    )
    args = parser.parse_args(argv)

    devices = jax.devices()
    platform = devices[0].platform if devices else "cpu"
    if args.dry_run:
        print(f"[dogfood] dry-run OK: platform={platform} cases={NUM_STATES_CASES}")
        return 0
    if args.smoke:
        args.stage = 1
        args.num_states = SMOKE_NUM_STATES_CASES
        args.n_warmup = 2
        args.n_timed = 5
    if args.stage >= 2:
        assert platform == "gpu", (
            f"stage=2 requires a GPU device, got {devices!r} -- "
            "use --stage 1 for a CPU smoke run"
        )
    cases = args.num_states or NUM_STATES_CASES

    written: list[Path] = []
    for num_states in cases:
        written.append(
            run_case(
                num_states=num_states,
                seq_len=args.seq_len,
                n_warmup=args.n_warmup,
                n_timed=args.n_timed,
                out_dir=args.out_dir,
                stage=args.stage,
            )
        )
    print(f"[dogfood] {len(written)} record(s) written under {args.out_dir}")

    # Fail-closed demo on the fresh set: TERM_RANKING needs >=2 such GPU
    # records with matching provenance -- report the live verdict either way.
    from xtrax.profiling.claims import assert_claim_supported
    from xtrax.profiling.record import ProbeRecord

    loaded = [ProbeRecord.read(p) for p in written]
    try:
        assert_claim_supported(loaded, ClaimClass.TERM_RANKING)
        print("[dogfood] TERM_RANKING SUPPORTED over this record set")
    except Exception as exc:  # noqa: BLE001 -- demo prints the gate verdict
        print(f"[dogfood] claim gate verdict: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
