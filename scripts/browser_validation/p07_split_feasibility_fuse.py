"""G0b: does the fuse-and-sample step convert to a standalone ONNX graph (Graph F)?

The split design originally moved this function into hand-written JavaScript. Reading it
showed that is unnecessary and undesirable: ``_fuse_and_sample``
(``inference/decode/autoregressive.py:99-189``) is pure arithmetic with no scan and no
cond -- the ``if gumbel_noise is None`` branch resolves at trace time when explicit noise
is supplied, which is how P07 runs it. If it exports, the product-of-experts fusion, the
tied-group averaging, the Gumbel argmax and the fixed-position override all stay inside a
graph already covered by the JAX and reference comparisons, instead of becoming new
hand-written numerics that need validating from scratch.

That matters because the route research
(xtrax ``.praxia/docs/research/260914_browser-inference-routes-jaxjs-jax2onnx.md`` S5)
names hand-rolled multi-key sort / tie handling as "the single most likely way this route
produces a green check over wrong output", and notes it is invisible to float parity.

This is gate **G0b**, a binary feasibility probe, sibling to G0 (Graph E + Graph D). It
does NOT re-open G0, whose `pass` stands under its own sidecar. Same discipline: loose
1e-3 agreement bound, exact token equality, and two negative controls that must fire.

Decision record: `.praxia/docs/daily/260929_overnight-decisions.md` D1/D2.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
  from collections.abc import Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("p07_split_feasibility_fuse")

# Same rationale as G0: loose on purpose, this is feasibility not parity. G1 sets the
# tight bound. Tokens, however, must match EXACTLY -- a tie-order or argmax divergence is
# precisely what a float bound cannot see.
DEFAULT_AGREE_BOUND = 1e-3
CTRL_PERTURB_MAGNITUDE = 1.0

_CODE_PATHS = ("src", "scripts", "pyproject.toml", "uv.lock")


def _git_state(repo: Path) -> tuple[str, bool, str]:
  def _run(*args: str) -> str:
    return subprocess.run(  # noqa: S603
      ["git", *args],  # noqa: S607
      cwd=repo,
      capture_output=True,
      text=True,
      check=True,
      timeout=60,
    ).stdout.strip()

  sha = _run("rev-parse", "HEAD")
  dirty = _run("status", "--porcelain", "--untracked-files=all", "--", *_CODE_PATHS)
  return sha, not dirty, dirty


def _synthetic_fuse_inputs(
  length: int,
  n_groups: int,
  *,
  seed: int = 0,
) -> dict[str, np.ndarray]:
  """Synthetic inputs exercising every branch of _fuse_and_sample.

  Deliberately includes: a multi-member tie group (so product-of-experts fusion is not
  trivially one position), an inactive/padded group slot (all-False mask, the branch that
  must return zeros rather than -inf/NaN), and a fixed position inside an active group (so
  the override path fires).
  """
  rng = np.random.default_rng(seed)
  logits = rng.normal(size=(1, length, 21)).astype(np.float32)
  cond_bias = rng.normal(scale=0.3, size=(length, 21)).astype(np.float32)

  mask_group = np.zeros((n_groups, length), dtype=bool)
  mask_group[0, [3, 17, 42]] = True  # multi-member tie group -> real PoE fusion
  mask_group[1, [8]] = True  # singleton group
  mask_group[2, [11, 12]] = True  # another tie group, contains a fixed position
  # group 3 left all-False: inactive/padded slot, must fuse to zeros not NaN

  fixed_mask = np.zeros(length, dtype=np.float32)
  fixed_mask[11] = 1.0  # fixed position inside group 2 -> override path
  fixed_tokens = np.zeros(length, dtype=np.int32)
  fixed_tokens[11] = 7

  return {
    "logits": logits,
    "cond_bias": cond_bias,
    "mask_group": mask_group,
    "fixed_mask": fixed_mask,
    "fixed_tokens": fixed_tokens,
    "group_id": np.arange(n_groups, dtype=np.int32),
    "temperature": np.float32(0.1),
    "gumbel_noise": rng.gumbel(size=(length, 21)).astype(np.float32),
  }


def _build_graph_f() -> Callable:
  """Graph F: _fuse_and_sample with the stage set P07 actually uses, closed over."""
  from aminx.inference.decode.autoregressive import _fuse_and_sample  # noqa: PLC0415
  from aminx.inference.logits import make_stage_set  # noqa: PLC0415, TID251

  stage_set = make_stage_set()

  def graph_f(
    logits: Any,
    cond_bias: Any,
    mask_group: Any,
    fixed_mask: Any,
    fixed_tokens: Any,
    group_id: Any,
    temperature: Any,
    gumbel_noise: Any,
  ) -> tuple[Any, Any]:
    return _fuse_and_sample(
      logits=logits,
      cond_bias=cond_bias,
      mask_group=mask_group,
      fixed_mask=fixed_mask,
      fixed_tokens=fixed_tokens,
      group_id=group_id,
      key=None,
      stage_set=stage_set,
      temperature=temperature,
      gumbel_noise=gumbel_noise,
    )

  return graph_f


_ARG_ORDER = (
  "logits",
  "cond_bias",
  "mask_group",
  "fixed_mask",
  "fixed_tokens",
  "group_id",
  "temperature",
  "gumbel_noise",
)


def _convert(fn: Callable, arrays: list[np.ndarray], name: str, out_path: Path) -> None:
  import jax  # noqa: PLC0415
  import jax2onnx  # noqa: PLC0415

  from scripts.browser_validation.p07_knobs_gate import embed_external_data  # noqa: PLC0415

  specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in arrays]
  out_path.parent.mkdir(parents=True, exist_ok=True)
  jax2onnx.to_onnx(fn, specs, model_name=name, output_path=str(out_path), return_mode="file")
  embed_external_data(out_path)


def _ort_run(onnx_path: Path, arrays: list[np.ndarray]) -> list[np.ndarray]:
  import onnxruntime as ort  # noqa: PLC0415

  session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  names = [i.name for i in session.get_inputs()]
  if len(names) != len(arrays):
    msg = f"{onnx_path.name}: graph takes {len(names)} inputs, got {len(arrays)}"
    raise ValueError(msg)
  feeds = {n: np.asarray(a) for n, a in zip(names, arrays, strict=True)}
  return list(session.run(None, feeds))


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
  return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--length", type=int, default=128)
  parser.add_argument("--n-groups", type=int, default=4)
  parser.add_argument("--work-dir", type=Path, default=None)
  parser.add_argument("--agree-bound", type=float, default=DEFAULT_AGREE_BOUND)
  args = parser.parse_args(argv)

  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
  )

  repo = Path(__file__).resolve().parents[2]
  git_hash, git_clean, git_dirty_paths = _git_state(repo)
  work_dir = args.work_dir or (args.out.parent / "split_fuse")
  work_dir.mkdir(parents=True, exist_ok=True)

  result: dict[str, Any] = {
    "length": args.length,
    "n_groups": args.n_groups,
    "agree_bound": args.agree_bound,
    "git_hash": git_hash,
    "git_clean": git_clean,
    "git_dirty_paths": git_dirty_paths,
    "jax_enable_x64": False,
    "f_converted": False,
    "f_ran": False,
    "f_finite": False,
    "f_tokens_exact": False,
    "f_logits_max_abs_diff": float("nan"),
    "inactive_group_is_zero": False,
    "fixed_override_applied": False,
    "controls_total": 2,
    "controls_detected": 0,
    "ctrl_nan_detected": False,
    "ctrl_perturb_detected": False,
    "failure": "",
  }

  try:
    import jax  # noqa: PLC0415

    result["jax_enable_x64"] = bool(jax.config.jax_enable_x64)
    if jax.config.jax_enable_x64:
      msg = "jax_enable_x64 is True; the exported graph would be int64/float64. Refusing."
      raise RuntimeError(msg)

    inputs = _synthetic_fuse_inputs(args.length, args.n_groups)
    ordered = [inputs[k] for k in _ARG_ORDER]
    graph_f = _build_graph_f()

    # JAX reference FIRST -- jax2onnx patches jnp primitives at import with ops that have
    # no CPU MLIR lowering, so evaluating after conversion poisons the reference arm.
    if "jax2onnx" in sys.modules:
      msg = "jax2onnx already imported; the JAX reference arm would use patched primitives."
      raise RuntimeError(msg)
    tok_jax, avg_jax = jax.jit(graph_f)(*ordered)
    tok_jax = np.asarray(tok_jax)
    avg_jax = np.asarray(avg_jax)
    logger.info("JAX reference arm complete; tokens=%s", tok_jax.tolist())

    # Semantic checks on the JAX side: these assert the synthetic inputs actually
    # exercised the branches this graph exists to carry, so a pass cannot come from a
    # fixture that never reached them.
    result["inactive_group_is_zero"] = bool(np.all(avg_jax[3] == 0.0))
    result["fixed_override_applied"] = bool(tok_jax[2] == 7)

    f_path = work_dir / f"p07_fuse_L{args.length}_G{args.n_groups}.onnx"
    logger.info("converting Graph F (fuse-and-sample) -> %s", f_path)
    _convert(graph_f, ordered, f"p07_fuse_L{args.length}", f_path)
    result["f_converted"] = True
    result["f_onnx_bytes"] = f_path.stat().st_size

    tok_onnx, avg_onnx = _ort_run(f_path, ordered)
    result["f_ran"] = True
    result["f_finite"] = bool(np.all(np.isfinite(avg_onnx)))
    result["f_tokens_exact"] = bool(np.array_equal(np.asarray(tok_onnx), tok_jax))
    result["f_logits_max_abs_diff"] = _max_abs_diff(avg_jax, avg_onnx)
    logger.info(
      "Graph F: %d bytes, tokens_exact=%s, logits_max_abs_diff=%.3e",
      result["f_onnx_bytes"],
      result["f_tokens_exact"],
      result["f_logits_max_abs_diff"],
    )

    # Controls: both checks above are pass-by-default shapes, so plant failures.
    poisoned = np.array(avg_onnx, copy=True)
    poisoned.flat[0] = np.nan
    result["ctrl_nan_detected"] = not bool(np.all(np.isfinite(poisoned)))

    offset = np.array(avg_onnx, copy=True)
    offset.flat[0] += CTRL_PERTURB_MAGNITUDE
    result["ctrl_perturb_detected"] = _max_abs_diff(avg_jax, offset) > args.agree_bound

    result["controls_detected"] = int(result["ctrl_nan_detected"]) + int(
      result["ctrl_perturb_detected"]
    )
    logger.info("controls: %d/2", result["controls_detected"])

  except Exception as exc:  # noqa: BLE001 - a conversion refusal IS the finding
    result["failure"] = f"{type(exc).__name__}: {exc}"
    logger.exception("G0b probe failed")

  bound = args.agree_bound
  result["within_bound"] = bool(
    result["f_ran"] and result["f_logits_max_abs_diff"] <= bound and result["f_tokens_exact"]
  )
  result["outcome"] = (
    "incomplete"
    if result["jax_enable_x64"]
    or not (result["f_converted"] and result["f_ran"])
    or not result["git_clean"]
    else "fixture_blind"
    if not (result["inactive_group_is_zero"] and result["fixed_override_applied"])
    else "ctrl_blind"
    if result["controls_detected"] < result["controls_total"]
    else "pass"
    if result["within_bound"] and result["f_finite"]
    else "fail"
  )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  logger.info("outcome=%s -> %s", result["outcome"], args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
