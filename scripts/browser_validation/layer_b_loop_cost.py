"""Layer (b) loop-body structural cost report (T7, P07 diagnostic; xtrax #1983 tie-in).

Synthetic instrument checks on toy `lax.while_loop`s and the AR kernel (per
`bench_ar_incremental.py:42-80`), reporting `body_operand_elements` scaling
alpha = log2(m(2L) / m(L)) per control, via the real HLO parser in `hlo_loop_cost.py`
(never a placeholder/estimate -- every measurement comes from `compiled.as_text()`).

Canonical alpha is the (L=256 -> L=512) doubling pair (less relative constant-overhead
contamination than 128->256); the 128->256 pair is also reported, non-gated, for context.

Sidecar outcomes (order matters -- first match wins): `ctrl_blind` (any toy alpha out of
range or unavailable, or ar_off alpha < 0.8, or ar_force alpha unavailable), then
`loop_body_scales` (residual finding: ar_force alpha > 0.35, offending instructions
listed), then `pass`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(_SCRIPT_DIR))

import bv_stats_guard as bvsg  # noqa: E402
import layer_a_common as lac  # noqa: E402
from hlo_loop_cost import HLOLoopAnalyzer, primary_while_loop  # noqa: E402

from aminx.inference.bundle_builder import build_inference_bundle  # noqa: E402
from aminx.inference.decode.factory import make_decode_fn  # noqa: E402
from aminx.inference.decode.mode import AutoregressiveConfig, AutoregressiveMode  # noqa: E402
from aminx.inference.encode import make_encode_fn  # noqa: E402
from aminx.inference.logits import make_stage_set  # noqa: E402
from aminx.model.mpnn import Aminx  # noqa: E402
from aminx.tiling.strategy import Vmap  # noqa: E402
from aminx.types.bundles import WaveScheduleBundle  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

L_VALS = (128, 256, 512)
AR_MODES = ("off", "force")
N_ITERATIONS = 8
STEADY_REPEATS = 10


def _compiled_hlo_text(jitted_fn: Any, *args: Any) -> str:
  """Lower + compile `jitted_fn(*args)` and return the COMPILED (optimized) HLO text.

  For a plain `jax.jit`-wrapped callable, `.lower(*args).compile()` returns a
  `jax.stages.Compiled` with `.as_text()` directly. For an `eqx.filter_jit`-wrapped
  callable, `.compile()` returns equinox's own `Compiled` wrapper, whose `.compiled`
  attribute is the underlying `jax.stages.Compiled` (equinox's wrapper does not itself
  expose `.as_text()` -- confirmed by reading `equinox/_compile_utils.py`).
  """
  compiled = jitted_fn.lower(*args).compile()
  inner = getattr(compiled, "compiled", compiled)
  return inner.as_text()


def _safe_log2_ratio(a: int | None, b: int | None) -> float | None:
  if a is None or b is None:
    return None
  return bvsg.guarded_log2_ratio(a, b)


def _alpha_pairs(measurements: dict[int, int | None]) -> dict[str, float | None]:
  """{'alpha_<l>_<2l>', ..., 'alpha'} for every consecutive-doubling pair present.

  'alpha' is the CANONICAL value: the pair at the LARGEST L (least relative constant-
  overhead contamination). Generic over however many L values are measured, so this
  also works for a tiny-L smoke measurement (e.g. L in {16, 32}), not just the
  pre-registered {128, 256, 512}.
  """
  ls = sorted(measurements.keys())
  pairs: dict[str, float | None] = {}
  canonical: float | None = None
  for l1, l2 in zip(ls, ls[1:], strict=False):
    if l2 == 2 * l1:
      alpha = _safe_log2_ratio(measurements.get(l2), measurements.get(l1))
      pairs[f"alpha_{l1}_{l2}"] = alpha
      canonical = alpha
  pairs["alpha"] = canonical
  return pairs


def _measure_while_cost(
  jitted_run: Any, args_by_l: dict[int, tuple[Any, ...]], known_scope_labels: frozenset[str]
) -> tuple[dict[int, int | None], dict[int, list[dict[str, Any]]], dict[int, bool]]:
  """Compile `jitted_run` at each L in `args_by_l`, return (measurements, top10s, nested_while)."""
  measurements: dict[int, int | None] = {}
  top10s: dict[int, list[dict[str, Any]]] = {}
  nested: dict[int, bool] = {}
  for l_val, args in args_by_l.items():
    hlo_text = _compiled_hlo_text(jitted_run, *args)
    loops = HLOLoopAnalyzer(hlo_text, known_scope_labels=known_scope_labels).analyze()
    primary = primary_while_loop(loops)
    measurements[l_val] = primary.body_operand_elements if primary else None
    top10s[l_val] = primary.top_10_instructions if primary else []
    nested[l_val] = primary.has_nested_while if primary else False
  return measurements, top10s, nested


# ---------------------------------------------------------------------------------------
# Synthetic toy controls
# ---------------------------------------------------------------------------------------


def measure_linear_control(l_vals: tuple[int, ...] = L_VALS) -> dict[str, Any]:
  """linear: carried (L x 16) f32, body `x * 2.0 + 1.0`. Expected alpha ~= 1 exactly."""

  def cond_fn(carry: tuple[jax.Array, jax.Array]) -> jax.Array:
    i, _x = carry
    return i < N_ITERATIONS

  def body(carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
    i, x = carry
    with jax.named_scope("linear_body"):
      return i + 1, x * 2.0 + 1.0

  def run(x0: jax.Array) -> tuple[jax.Array, jax.Array]:
    i0 = jnp.zeros((), dtype=jnp.int32)
    return lax.while_loop(cond_fn, body, (i0, x0))

  jitted = jax.jit(run)
  args_by_l = {length: (jnp.zeros((length, 16), dtype=jnp.float32),) for length in l_vals}
  measurements, top10s, nested = _measure_while_cost(jitted, args_by_l, frozenset({"linear_body"}))
  alphas = _alpha_pairs(measurements)
  lo, hi = 0.9, 1.1
  alpha = alphas["alpha"]
  available = alpha is not None
  in_range = available and lo <= alpha <= hi
  return {
    "measurements": measurements,
    "top_10_at_max_l": top10s.get(l_vals[-1], []),
    "has_nested_while": nested.get(l_vals[-1], False),
    **alphas,
    "alpha_available": available,
    "expected_range": [lo, hi],
    "in_range": in_range,
  }


def measure_reduce_control(l_vals: tuple[int, ...] = L_VALS) -> dict[str, Any]:
  """reduce-to-scalar (r1 M14): carried (L,) f32 plus scalar acc, body `acc + sum(x)`.

  `x` is also updated each iteration (`x + 1.0`) so XLA cannot hoist the (loop-
  invariant, otherwise) reduction out of the loop entirely -- confirmed empirically:
  an unchanged `x` gets the whole `jnp.sum(x)` computed ONCE before the while and the
  body becomes O(1), which would blind alpha to ~0 for reasons unrelated to the metric
  itself. Weight is (L + 1) + 1 asymptotically, so alpha -> 1; required alpha in
  [0.9, 1.1]. An output-element metric would give alpha ~= 0 here -- exactly the
  blindness this control catches.
  """

  def cond_fn(carry: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
    i, _x, _acc = carry
    return i < N_ITERATIONS

  def body(carry: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
    i, x, acc = carry
    with jax.named_scope("reduce_body"):
      return i + 1, x + 1.0, acc + jnp.sum(x)

  def run(x0: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    i0 = jnp.zeros((), dtype=jnp.int32)
    acc0 = jnp.zeros((), dtype=jnp.float32)
    return lax.while_loop(cond_fn, body, (i0, x0, acc0))

  jitted = jax.jit(run)
  args_by_l = {length: (jnp.zeros((length,), dtype=jnp.float32),) for length in l_vals}
  measurements, top10s, nested = _measure_while_cost(jitted, args_by_l, frozenset({"reduce_body"}))
  alphas = _alpha_pairs(measurements)
  lo, hi = 0.9, 1.1
  alpha = alphas["alpha"]
  available = alpha is not None
  in_range = available and lo <= alpha <= hi
  return {
    "measurements": measurements,
    "top_10_at_max_l": top10s.get(l_vals[-1], []),
    "has_nested_while": nested.get(l_vals[-1], False),
    **alphas,
    "alpha_available": available,
    "expected_range": [lo, hi],
    "in_range": in_range,
  }


def measure_constant_control(l_vals: tuple[int, ...] = L_VALS) -> dict[str, Any]:
  """constant: body (48x16)@(16x16) independent of L. Expected alpha in [-0.1, 0.1].

  `a`/`b` are perturbed each iteration (`a + 0.001`) so the matmul is not loop-invariant
  and cannot be hoisted or constant-folded entirely out of the body (confirmed
  empirically -- literal/closed-over, unperturbed operands get folded away, leaving a
  body with near-zero weight for the wrong reason). A same-shape, L-independent decoy
  carry keeps the harness signature uniform across L (an L-sized input is still passed,
  simply unused by the cost-relevant part of the body).
  """

  def cond_fn(carry: tuple[jax.Array, ...]) -> jax.Array:
    i, _a, _b, _acc, _decoy = carry
    return i < N_ITERATIONS

  def body(carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    i, a, b, acc, decoy = carry
    with jax.named_scope("constant_body"):
      a2 = a + 0.001
      b2 = b + 0.001
      prod = a2 @ b2
      return i + 1, a2, b2, acc + jnp.sum(prod), decoy

  def run(a0: jax.Array, b0: jax.Array, decoy0: jax.Array) -> tuple[jax.Array, ...]:
    i0 = jnp.zeros((), dtype=jnp.int32)
    acc0 = jnp.zeros((), dtype=jnp.float32)
    return lax.while_loop(cond_fn, body, (i0, a0, b0, acc0, decoy0))

  jitted = jax.jit(run)
  a0 = jnp.ones((48, 16), dtype=jnp.float32)
  b0 = jnp.ones((16, 16), dtype=jnp.float32)
  args_by_l = {length: (a0, b0, jnp.zeros((length,), dtype=jnp.float32)) for length in l_vals}
  measurements, top10s, nested = _measure_while_cost(
    jitted, args_by_l, frozenset({"constant_body"})
  )
  alphas = _alpha_pairs(measurements)
  lo, hi = -0.1, 0.1
  alpha = alphas["alpha"]
  available = alpha is not None
  in_range = available and lo <= alpha <= hi
  return {
    "measurements": measurements,
    "top_10_at_max_l": top10s.get(l_vals[-1], []),
    "has_nested_while": nested.get(l_vals[-1], False),
    **alphas,
    "alpha_available": available,
    "expected_range": [lo, hi],
    "in_range": in_range,
  }


# ---------------------------------------------------------------------------------------
# AR kernel (real production-shaped model, per bench_ar_incremental.py:42-80)
# ---------------------------------------------------------------------------------------


def _build_model() -> Aminx:
  """Production-shaped, random-weight Aminx: 128-dim, 3+3 layers, k=48 (bench_ar_incremental)."""
  return eqx.tree_inference(
    Aminx(
      node_features=128,
      edge_features=128,
      hidden_features=128,
      num_encoder_layers=3,
      num_decoder_layers=3,
      k_neighbors=48,
      key=jax.random.PRNGKey(0),
    ),
    value=True,
  )


def _structure(length: int, seed: int) -> tuple[jax.Array, ...]:
  """Same synthetic-structure generator as bench_ar_incremental.py:39-42."""
  key = jax.random.PRNGKey(seed)
  ca = jnp.cumsum(jax.random.normal(key, (length, 1, 3)) * 2.2, axis=0)
  coords = ca + jax.random.normal(jax.random.fold_in(key, 1), (1, 4, 3)) * 0.8
  return coords, jnp.ones((length,)), jnp.arange(length), jnp.zeros((length,), dtype=jnp.int32)


def _build_decode_inputs(model: Aminx, length: int) -> tuple[Any, Any, Any, Any]:
  """(enc, bundle, config, stage_set) at `length`, wave = from_tie_groups(arange(L), arange(L))."""
  coords, mask, residue_index, chain_index = _structure(length, seed=length)
  wave = WaveScheduleBundle.from_tie_groups(jnp.arange(length), jnp.arange(length))
  bundle, config = build_inference_bundle(
    coords,
    mask,
    residue_index,
    chain_index,
    mode="sample_autoregressive",
    temperature=1.0,
    wave=wave,
  )
  enc = make_encode_fn(model, use_rolling_state=False)(bundle, jax.random.PRNGKey(0), config)
  stage_set = make_stage_set()
  return enc, bundle, config, stage_set


def _make_decode(model: Aminx, mode: str) -> Any:
  raw_decode = make_decode_fn(
    model=model,
    mode=AutoregressiveMode(),
    strategy=Vmap(),
    autoregressive_config=AutoregressiveConfig(inference_only=True, incremental=mode),
  )

  def scoped_decode(key: jax.Array, enc: Any, bundle: Any, config: Any, stage_set: Any) -> Any:
    with jax.named_scope(f"ar_decode_{mode}"):
      return raw_decode(key, enc, bundle, config, stage_set)

  return eqx.filter_jit(scoped_decode)


def measure_ar_kernel(
  l_vals: tuple[int, ...] = L_VALS, repeats: int = STEADY_REPEATS
) -> dict[str, dict[str, Any]]:
  """AR kernel cost + timing, per mode in {off, force}, as in bench_ar_incremental.py:64.

  Reports `body_operand_elements` (via the real HLO parser) at each L, the canonical
  256->512 alpha, the 128->256 alpha for context, the top-10 body instructions at the
  largest L, and `repeats` steady (post-compile) decode timings for context.
  """
  model = _build_model()
  inputs_by_l = {length: _build_decode_inputs(model, length) for length in l_vals}
  results: dict[str, dict[str, Any]] = {}
  tokens_by_mode_l: dict[str, dict[int, np.ndarray]] = {}

  for mode in AR_MODES:
    decode = _make_decode(model, mode)
    measurements: dict[int, int | None] = {}
    top10s: dict[int, list[dict[str, Any]]] = {}
    nested: dict[int, bool] = {}
    steady_seconds: dict[int, list[float]] = {}
    compile_seconds: dict[int, float] = {}
    tokens_by_l: dict[int, np.ndarray] = {}

    for l_val in l_vals:
      enc, bundle, config, stage_set = inputs_by_l[l_val]
      key0 = jax.random.PRNGKey(7)

      hlo_text = _compiled_hlo_text(decode, key0, enc, bundle, config, stage_set)
      analyzer = HLOLoopAnalyzer(hlo_text, known_scope_labels=frozenset({f"ar_decode_{mode}"}))
      loops = analyzer.analyze()
      primary = primary_while_loop(loops)
      measurements[l_val] = primary.body_operand_elements if primary else None
      top10s[l_val] = primary.top_10_instructions if primary else []
      nested[l_val] = primary.has_nested_while if primary else False

      t0 = time.perf_counter()
      out = jax.block_until_ready(decode(key0, enc, bundle, config, stage_set))
      compile_seconds[l_val] = time.perf_counter() - t0
      tokens_by_l[l_val] = np.asarray(out.sequence)

      times: list[float] = []
      for r in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(decode(jax.random.PRNGKey(7 + r + 1), enc, bundle, config, stage_set))
        times.append(time.perf_counter() - t0)
      steady_seconds[l_val] = times
      logger.info(
        "AR mode=%s L=%d body_operand_elements=%s steady_median=%.4fs",
        mode,
        l_val,
        measurements[l_val],
        float(np.median(times)) if times else float("nan"),
      )

    tokens_by_mode_l[mode] = tokens_by_l
    alphas = _alpha_pairs(measurements)
    alpha = alphas["alpha"]
    available = alpha is not None
    results[mode] = {
      "measurements": measurements,
      "top_10_at_max_l": top10s.get(l_vals[-1], []),
      "has_nested_while": nested.get(l_vals[-1], False),
      **alphas,
      "alpha_available": available,
      "compile_seconds": compile_seconds,
      "steady_seconds": steady_seconds,
      "steady_median_seconds": {
        length: (float(np.median(t)) if t else None) for length, t in steady_seconds.items()
      },
    }

  if "off" in tokens_by_mode_l and "force" in tokens_by_mode_l:
    results["tokens_equal_off_force"] = {
      str(length): bool(
        np.array_equal(tokens_by_mode_l["off"][length], tokens_by_mode_l["force"][length])
      )
      for length in l_vals
    }
  return results


# ---------------------------------------------------------------------------------------
# Sidecar assembly
# ---------------------------------------------------------------------------------------

_FOLLOWUP_1983 = (
  "## xtrax #1983: Loop-body cost instrumentation (upstream `hlo_loop_cost`)\n\n"
  "`scripts/browser_validation/hlo_loop_cost.py` (T7, 260926_browser-export-loop) implements "
  "a general HLO while-loop body-cost parser: `body_operand_elements` = Sigma over the body's "
  "instructions of {fusion: operand+output at the call site (never descended into), unfused "
  "reduce/reduce-window/dot/convolution/gather/sort: operand+output, "
  "dynamic-update-slice/scatter: update operand (+scatter indices), dynamic-slice: output, "
  "other: output-only}, with `call`/`conditional` descent (max over branches) and "
  "nested-`while` flagging. Verified against 3 synthetic ground-truth controls (linear, "
  "reduce-to-scalar, constant) plus a handwritten-HLO unit test with 3 red-checks "
  "(`tests/parity/test_hlo_loop_cost.py`). Upstream work: promote the parser out of "
  "scripts/browser_validation into `xtrax.profiling` proper (a STRUCTURAL claim class per "
  "`xtrax.profiling.claims`), replacing the `#1983` gap noted in this sprint's V13 anchor "
  "('No loop-body cost primitive exists')."
)


def _followup_1981(offending: list[dict[str, Any]]) -> str:
  names = ", ".join(e.get("name", "?") for e in offending[:5])
  return (
    "## aminx #1981: AR decoder loop-body scaling (loop_body_scales finding)\n\n"
    "layer_b_loop_cost.py (T7) measured the AR decode kernel's while-loop "
    f"body_operand_elements scaling alpha (L=256->512) above the 0.35 bar under "
    '`incremental="force"`, meaning the per-wave loop body carries data-flow weight that '
    "grows faster than expected for a supposedly O(1)-per-wave incremental cache. "
    f"Top offending body instructions (by weight): {names or '(see result JSON top_10_at_max_l)'}. "
    "Needs investigation of what in the incremental decode path (KV-style cache update, "
    "position-map gather, or fixed-mask broadcast) scales with L instead of the wave size."
  )


def _assess(
  controls: dict[str, dict[str, Any]], ar_kernel: dict[str, dict[str, Any]]
) -> tuple[str, str, str]:
  """Return (outcome, outcome_reason, followup_backlog_text).

  Declaration order: ctrl_blind, loop_body_scales, pass -- matches the sidecar's own
  [outcomes] order.
  """
  reasons: list[str] = []
  for name, ctrl in controls.items():
    if not ctrl["alpha_available"]:
      reasons.append(f"{name}: alpha unavailable (null)")
    elif not ctrl["in_range"]:
      lo, hi = ctrl["expected_range"]
      reasons.append(f"{name}: alpha {ctrl['alpha']:.4f} outside [{lo}, {hi}]")

  ar_off = ar_kernel["off"]
  if not ar_off["alpha_available"]:
    reasons.append("ar_off: alpha unavailable (null)")
  elif ar_off["alpha"] < 0.8:
    reasons.append(f"ar_off: alpha {ar_off['alpha']:.4f} < 0.8")

  ar_force = ar_kernel["force"]
  if not ar_force["alpha_available"]:
    reasons.append("ar_force: alpha unavailable (null)")

  if reasons:
    return "ctrl_blind", "; ".join(reasons), _followup_1981(ar_force.get("top_10_at_max_l", []))

  if ar_force["alpha"] > 0.35:
    reason = f"ar_force: alpha {ar_force['alpha']:.4f} > 0.35 (loop-body scaling detected)"
    return "loop_body_scales", reason, _followup_1981(ar_force.get("top_10_at_max_l", []))

  pass_reason = "all controls in range; ar_off alpha >= 0.8; ar_force alpha <= 0.35"
  return "pass", pass_reason, _FOLLOWUP_1983


def main(args: argparse.Namespace) -> int:
  provenance = lac.provenance()

  linear = measure_linear_control()
  reduce_ctrl = measure_reduce_control()
  constant = measure_constant_control()
  controls = {"linear": linear, "reduce": reduce_ctrl, "constant": constant}

  ar_kernel = measure_ar_kernel()

  outcome, outcome_reason, followup_backlog_text = _assess(controls, ar_kernel)

  result: dict[str, Any] = {
    "outcome": outcome,
    "outcome_reason": outcome_reason,
    "alpha_linear": linear["alpha"],
    "alpha_linear_available": linear["alpha_available"],
    "alpha_reduce": reduce_ctrl["alpha"],
    "alpha_reduce_available": reduce_ctrl["alpha_available"],
    "alpha_constant": constant["alpha"],
    "alpha_constant_available": constant["alpha_available"],
    "alpha_ar_off": ar_kernel["off"]["alpha"],
    "alpha_ar_off_available": ar_kernel["off"]["alpha_available"],
    "alpha_ar_force": ar_kernel["force"]["alpha"],
    "alpha_ar_force_available": ar_kernel["force"]["alpha_available"],
    "controls": controls,
    "ar_kernel": ar_kernel,
    "followup_backlog_text": followup_backlog_text,
    "provenance": provenance,
  }

  lac.emit(result, args.out)
  return 0


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Loop-body cost report (T7)")
  parser.add_argument("--out", type=str, required=True, help="Output JSON file")
  cli_args = parser.parse_args()
  sys.exit(main(cli_args))
