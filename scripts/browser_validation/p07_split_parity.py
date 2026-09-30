"""G1: does the four-graph SPLIT reproduce the monolithic P07 export, and JAX?

The knobs gate (run fd80f81d) validated the MONOLITH end to end: tokens bitwise identical
to JAX and to ORT-Web on 288/288 cells, and 4.48e-05 nats against reference ProteinMPNN
teacher-forced. That makes the monolith a trustworthy local reference. This gate asks
whether the split -- encoder, wave schedule, decoder step, fuse-and-sample, with the loop
driven outside the graph -- computes the same thing.

Design: every cell feeds BOTH paths the *identical* eleven P07 input arrays, built by the
same JavaScript RunSpec builder on the same real fixtures the knobs gate used
(`js_build`/`runspec_for`/`_parse_geometry` are imported from it, not reimplemented). The
only variable is the computation path, so a disagreement cannot be blamed on differing
inputs.

Three arms per cell:
  jax       aminx JAX make_p07_sample            (the upstream reference)
  monolith  the single-graph .onnx via ORT-CPU   (validated by fd80f81d)
  split     E -> W -> [D -> F] x n_waves -> gather, composed here via ORT-CPU

The Python loop here mirrors `browser/aminx-sampler/split_loop.mjs` and
`inference/decode/autoregressive.py:469-594,811-819`. Running the composition in Python
FIRST is deliberate: it separates "is the graph decomposition correct" from "is the
JavaScript driver correct". A later gate runs the actual .mjs under Node against the same
fixtures; if this gate passes and that one fails, the fault is in the JS, not the cut.

Tokens must match EXACTLY. Log-probs carry the same 2e-4 bound the knobs gate uses, and
that bound is inherited deliberately rather than invented: it is the measured JAX-vs-ORT
envelope for this model at these buckets, and the split adds graph boundaries, not new
arithmetic.
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

logger = logging.getLogger("p07_split_parity")

N_TOKENS = 21
UNDRAWN_TOKEN = -1
DEFAULT_LOGPROB_BOUND = 2e-4
CTRL_PERTURB = 1.0
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


def _session(path: Path) -> Any:
  import onnxruntime as ort  # noqa: PLC0415

  return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _run_session(session: Any, arrays: list[np.ndarray]) -> list[np.ndarray]:
  names = [i.name for i in session.get_inputs()]
  if len(names) != len(arrays):
    msg = f"graph expects {len(names)} inputs, got {len(arrays)}"
    raise ValueError(msg)
  return list(session.run(None, dict(zip(names, arrays, strict=True))))


def _log_softmax(rows: np.ndarray) -> np.ndarray:
  shifted = rows - rows.max(axis=-1, keepdims=True)
  return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def split_decode(
  sessions: dict[str, Any],
  arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
  """Compose the four graphs. Mirrors split_loop.mjs and autoregressive.py:469-594,811-819."""
  length = int(arrays["mask"].shape[0])

  node_features, edge_features, neighbor_indices = _run_session(
    sessions["encoder"],
    [arrays["coords"], arrays["mask"], arrays["residue_index"], arrays["chain_index"]],
  )
  (
    group_ids,
    group_positions,
    group_valid,
    _position_valid,
    ar_mask,
    group_first_rank,
    pos_first_rank,
  ) = _run_session(sessions["wave"], [arrays["decoding_order"], arrays["tie_group_map"]])

  n_waves, g_per_wave = group_ids.shape
  sentinel = n_waves * g_per_wave
  tie_group_map = arrays["tie_group_map"]

  # init_sequence: UNDRAWN everywhere, fixed positions preloaded (autoregressive.py:430-437)
  sequence = np.full(length, UNDRAWN_TOKEN, dtype=np.int32)
  fixed_on = arrays["fixed_mask"] > 0.5
  sequence[fixed_on] = arrays["fixed_tokens"][fixed_on]

  logits_stack = np.zeros((n_waves, g_per_wave, N_TOKENS), dtype=np.float32)

  for w in range(n_waves):
    pos0 = group_positions[w, :, 0]
    group_id = tie_group_map[pos0].astype(np.int32)
    is_active = np.asarray(group_valid[w]).astype(bool)
    this_rank = (w * g_per_wave + np.arange(g_per_wave)).astype(np.int32)
    is_first = is_active & (group_first_rank[group_id] == this_rank)
    if not is_first.any():
      continue
    mask_group = (tie_group_map[None, :] == group_id[:, None]) & is_first[:, None]

    seq_oh = np.zeros((length, N_TOKENS), dtype=np.float32)
    drawn = sequence >= 0
    seq_oh[np.arange(length)[drawn], sequence[drawn]] = 1.0

    (logits,) = _run_session(
      sessions["decoder"],
      [
        node_features,
        edge_features,
        neighbor_indices,
        arrays["mask"],
        np.asarray(ar_mask, np.float32),
        seq_oh,
      ],
    )
    final_token, avg_stored = _run_session(
      sessions["fuse"],
      [
        np.asarray(logits, np.float32)[None, ...],
        arrays["bias"],
        mask_group,
        arrays["fixed_mask"],
        arrays["fixed_tokens"],
        group_id,
        arrays["temperature"],
        arrays["gumbel_noise"],
      ],
    )
    token_grid = np.where(mask_group, np.asarray(final_token)[:, None], 0)
    covers = mask_group.any(axis=0)
    sequence = np.where(covers, token_grid.sum(axis=0), sequence).astype(np.int32)
    logits_stack[w] = np.asarray(avg_stored, np.float32)

  rank = pos_first_rank
  scheduled = rank < sentinel
  safe = np.where(scheduled, rank, 0)
  gathered = logits_stack[safe // g_per_wave, safe % g_per_wave]
  gathered = np.where(scheduled[:, None], gathered, 0.0)
  return sequence.astype(np.int32), _log_softmax(gathered.astype(np.float32))


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--release-dir", type=Path, required=True, help="split .onnx + MANIFEST")
  parser.add_argument("--monolith", type=Path, required=True, help="p07_L<bucket>.onnx")
  parser.add_argument("--bucket", type=int, default=128)
  parser.add_argument("--logprob-bound", type=float, default=DEFAULT_LOGPROB_BOUND)
  parser.add_argument("--max-cells", type=int, default=0, help="0 = all")
  parser.add_argument("--node-bin", type=str, default=None)
  args = parser.parse_args(argv)

  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
  )

  git_hash, git_clean, git_dirty_paths = _git_state(_REPO_ROOT)
  result: dict[str, Any] = {
    "bucket": args.bucket,
    "logprob_bound": args.logprob_bound,
    "git_hash": git_hash,
    "git_clean": git_clean,
    "git_dirty_paths": git_dirty_paths,
    "jax_enable_x64": False,
    "cells_total": 0,
    "cells_tokens_exact_vs_monolith": 0,
    "cells_tokens_exact_vs_jax": 0,
    "max_logprob_abs_diff_vs_monolith": 0.0,
    "max_logprob_abs_diff_vs_jax": 0.0,
    "worst_cell": "",
    "n_waves": -1,
    "decoder_calls_total": 0,
    "controls_total": 1,
    "controls_detected": 0,
    "ctrl_perturb_detected": False,
    "failure": "",
  }

  try:
    import jax  # noqa: PLC0415

    result["jax_enable_x64"] = bool(jax.config.jax_enable_x64)
    if jax.config.jax_enable_x64:
      msg = "jax_enable_x64 is True; refusing."
      raise RuntimeError(msg)

    from scripts.browser_validation.p07_knobs_gate import (  # noqa: PLC0415
      FULL_KNOBS,
      FULL_SEEDS,
      P07_KEYS,
      _call_p07,
      _load_manifest,
      _node_bin,
      _pad_geometry,
      _parse_geometry,
      _select_fixtures,
      _structure_payload,
      _with_chain_split,
      js_build,
      runspec_for,
    )
    from aminx.export.wrappers import make_p07_sample  # noqa: PLC0415
    from aminx.inference.logits import make_stage_set  # noqa: PLC0415, TID251
    from aminx.io.weights import load_model  # noqa: PLC0415

    node = _node_bin(args.node_bin)
    fixture_rows = _select_fixtures(_load_manifest(), [args.bucket])
    import tempfile  # noqa: PLC0415

    work = Path(tempfile.mkdtemp(prefix="p07_split_parity_"))
    geometries = [_parse_geometry(row, work / "canonical") for row in fixture_rows]
    result["fixture_ids"] = [str(r["name"]) for r in fixture_rows]

    p07 = make_p07_sample(load_model(checkpoint_id="proteinmpnn_v_48_020"), make_stage_set())

    sessions = {
      key: _session(args.release_dir / f"p07_{key}_L{args.bucket}.onnx")
      for key in ("encoder", "wave", "decoder", "fuse")
    }
    mono = _session(args.monolith)
    mono_names = [i.name for i in mono.get_inputs()]

    cells = 0
    for geom in geometries:
      if int(geom["n_real"]) > args.bucket:
        continue
      padded = _pad_geometry(geom, args.bucket)
      chain_split = _pad_geometry(_with_chain_split(geom), args.bucket)
      for knob in FULL_KNOBS:
        host = chain_split if knob in ("chains_to_design", "all_combined") else padded
        for seed in FULL_SEEDS:
          if args.max_cells and cells >= args.max_cells:
            break
          spec = runspec_for(knob, seed, host)
          arrays = js_build(_structure_payload(host), spec, node)

          tok_jax, lp_jax = _call_p07(p07, arrays)
          tok_jax = np.asarray(tok_jax)
          lp_jax = np.asarray(lp_jax)

          mono_out = mono.run(
            None,
            {n: np.asarray(arrays[k]) for n, k in zip(mono_names, P07_KEYS, strict=True)},
          )
          tok_mono, lp_mono = np.asarray(mono_out[0]), np.asarray(mono_out[1])

          tok_split, lp_split = split_decode(sessions, arrays)

          cells += 1
          if np.array_equal(tok_split, tok_mono):
            result["cells_tokens_exact_vs_monolith"] += 1
          if np.array_equal(tok_split, tok_jax):
            result["cells_tokens_exact_vs_jax"] += 1
          d_mono = float(np.max(np.abs(lp_split.astype(np.float64) - lp_mono.astype(np.float64))))
          d_jax = float(np.max(np.abs(lp_split.astype(np.float64) - lp_jax.astype(np.float64))))
          if d_mono > result["max_logprob_abs_diff_vs_monolith"]:
            result["max_logprob_abs_diff_vs_monolith"] = d_mono
            result["worst_cell"] = f"{geom['name']}_{knob}_s{seed}"
          result["max_logprob_abs_diff_vs_jax"] = max(
            result["max_logprob_abs_diff_vs_jax"], d_jax
          )

          if cells == 1:
            # Control: a planted offset on the split log-probs must exceed the bound.
            poisoned = lp_split.astype(np.float64).copy()
            poisoned.flat[0] += CTRL_PERTURB
            result["ctrl_perturb_detected"] = bool(
              np.max(np.abs(poisoned - lp_mono.astype(np.float64))) > args.logprob_bound
            )
            gi = np.asarray(
              _run_session(sessions["wave"], [arrays["decoding_order"], arrays["tie_group_map"]])[0]
            )
            result["n_waves"] = int(gi.shape[0])
          result["decoder_calls_total"] += result["n_waves"]
          logger.info(
            "cell %d %s_%s_s%d: tok_mono=%s tok_jax=%s d_mono=%.3e",
            cells,
            geom["name"],
            knob,
            seed,
            np.array_equal(tok_split, tok_mono),
            np.array_equal(tok_split, tok_jax),
            d_mono,
          )
        if args.max_cells and cells >= args.max_cells:
          break
      if args.max_cells and cells >= args.max_cells:
        break

    result["cells_total"] = cells
    result["controls_detected"] = int(result["ctrl_perturb_detected"])

  except Exception as exc:  # noqa: BLE001 - a failure IS the finding
    result["failure"] = f"{type(exc).__name__}: {exc}"
    logger.exception("G1 split parity failed")

  total = result["cells_total"]
  result["outcome"] = (
    "incomplete"
    if result["jax_enable_x64"] or total == 0 or not result["git_clean"] or result["failure"]
    else "ctrl_blind"
    if result["controls_detected"] < result["controls_total"]
    else "pass"
    if result["cells_tokens_exact_vs_monolith"] == total
    and result["cells_tokens_exact_vs_jax"] == total
    and result["max_logprob_abs_diff_vs_monolith"] <= args.logprob_bound
    and result["max_logprob_abs_diff_vs_jax"] <= args.logprob_bound
    else "fail"
  )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  logger.info("outcome=%s -> %s", result["outcome"], args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
