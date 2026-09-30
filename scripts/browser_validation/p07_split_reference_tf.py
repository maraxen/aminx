"""G3a-direct: the SPLIT graphs vs reference ProteinMPNN, teacher-forced.

Everything so far anchors the split to the reference by COMPOSITION: the knobs gate's B1
measured reference PyTorch against aminx JAX (4.482e-05 nats), and G1 measured aminx JAX
against the split (tokens exact, 1.526e-05). Chaining two measured links is legitimate but
strictly weaker than measuring the thing you actually ship against the thing the field
trusts, and a composed claim is only as tight as its weaker link. This gate removes the
intermediary.

Method, mirroring B1 (`p07_knobs_gate.py:1633` `_tf_max`) so the two are comparable:
the PyTorch reference samples a sequence and returns it with its per-position log-probs
and the decoding order it used; the split is then TEACHER-FORCED on that same sequence and
order and its log-probs compared. Teacher-forcing needs exactly one conditional decoder
pass -- not the autoregressive loop -- because the whole sequence is known up front, so
this gate is cheap despite touching the reference implementation.

Scope, stated because it is narrower than the parity gates: untied lanes only. Tied lanes
require the reference-side tie fusion (`_combine_reference_tied_log_probs`) that B1 applies
for P09-s, and folding that in would mean reproducing a second piece of reference
machinery here; the tied path is already covered indirectly through B1 + G1. What this
gate adds is a DIRECT link for the untied case, which is the one the composed claim leans
on hardest.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))
_BV = _REPO_ROOT / "scripts" / "browser_validation"
if str(_BV) not in sys.path:
  sys.path.insert(0, str(_BV))

logger = logging.getLogger("p07_split_reference_tf")

N_TOKENS = 21
DEFAULT_BOUND = 1e-4  # same bound B1 uses for reference parity; inherited, not invented
CTRL_PERTURB = 1.0
_CODE_PATHS = ("src", "scripts", "browser", "pyproject.toml", "uv.lock")
UNTIED_LANES = ("P08@1.0", "P07@1.0")


def _git_state(repo: Path) -> tuple[str, bool]:
  def _run(*args: str) -> str:
    return subprocess.run(  # noqa: S603
      ["git", *args],  # noqa: S607
      cwd=repo, capture_output=True, text=True, check=True, timeout=60,
    ).stdout.strip()

  return _run("rev-parse", "HEAD"), not _run(
    "status", "--porcelain", "--untracked-files=all", "--", *_CODE_PATHS
  )


def _log_softmax(rows: np.ndarray) -> np.ndarray:
  shifted = rows - rows.max(axis=-1, keepdims=True)
  return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--release-dir", type=Path, required=True)
  parser.add_argument("--bucket", type=int, default=128)
  parser.add_argument("--bound", type=float, default=DEFAULT_BOUND)
  args = parser.parse_args(argv)

  logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
  )

  git_hash, git_clean = _git_state(_REPO_ROOT)
  result: dict[str, Any] = {
    "bucket": args.bucket,
    "bound": args.bound,
    "git_hash": git_hash,
    "git_clean": git_clean,
    "cases_total": 0,
    "cases_within_bound": 0,
    "max_abs_nats_vs_reference": -1.0,
    "worst_case": "",
    "lanes": list(UNTIED_LANES),
    "per_case": [],
    "controls_total": 1,
    "controls_detected": 0,
    "ctrl_perturb_detected": False,
    "failure": "",
  }

  try:
    import onnxruntime as ort  # noqa: PLC0415

    import layer_a_common as lac  # noqa: PLC0415
    import layer_a_sampling as las  # noqa: PLC0415
    from p07_knobs_gate import (  # noqa: PLC0415
      _lane_batch,
      _load_manifest,
      _parse_geometry,
      _select_fixtures,
    )

    import tempfile  # noqa: PLC0415

    work = Path(tempfile.mkdtemp(prefix="p07_split_ref_tf_"))
    geometries = [
      _parse_geometry(row, work / "canonical")
      for row in _select_fixtures(_load_manifest(), [args.bucket])
    ]
    _jax_model, pt_model, torch, _utils = lac.load_full_model("eqx")

    enc_sess = ort.InferenceSession(
      str(args.release_dir / f"p07_encoder_L{args.bucket}.onnx"),
      providers=["CPUExecutionProvider"],
    )
    dec_sess = ort.InferenceSession(
      str(args.release_dir / f"p07_decoder_L{args.bucket}.onnx"),
      providers=["CPUExecutionProvider"],
    )

    def run_sess(sess: Any, arrays: list[np.ndarray]) -> list[np.ndarray]:
      names = [i.name for i in sess.get_inputs()]
      return list(sess.run(None, dict(zip(names, arrays, strict=True))))

    for geom in geometries:
      if int(geom["n_real"]) > args.bucket:
        continue
      for lane in UNTIED_LANES:
        batch = _lane_batch(geom, lane)
        if getattr(batch, "groups", None):
          logger.info("skipping %s/%s: tied lane, out of scope", geom["name"], lane)
          continue
        seed_i = las._seed_for(batch.fixture_name + batch.lane)  # noqa: SLF001
        seq, ref_lp, order, _randn = las.reference_sample_one(
          pt_model, torch, batch, seed_i, temperature=1.0
        )
        seq = np.asarray(seq)
        ref_lp = np.asarray(ref_lp)
        ar_mask = np.asarray(lac.ar_mask_from_order(np.asarray(order)), dtype=np.float32)

        length = int(np.asarray(batch.mask).shape[0])
        pad = args.bucket - length
        def _pad1(a: np.ndarray, fill: float = 0.0) -> np.ndarray:
          return np.pad(a, (0, pad), constant_values=fill) if pad else a

        coords = np.asarray(batch.x4, dtype=np.float32)
        if pad:
          coords = np.pad(coords, ((0, pad), (0, 0), (0, 0)))
        mask = _pad1(np.asarray(batch.mask, dtype=np.float32)).astype(np.float32)
        residue_index = _pad1(
          np.asarray(batch.residue_index, dtype=np.int32),
          fill=float(np.asarray(batch.residue_index)[-1]),
        ).astype(np.int32)
        chain_index = _pad1(
          np.asarray(batch.chain_index, dtype=np.int32),
          fill=float(np.asarray(batch.chain_index)[-1]),
        ).astype(np.int32)
        ar_full = np.zeros((args.bucket, args.bucket), np.float32)
        ar_full[:length, :length] = ar_mask
        seq_oh = np.zeros((args.bucket, N_TOKENS), np.float32)
        seq_oh[np.arange(length), np.asarray(seq, dtype=np.int64)[:length]] = 1.0

        node_f, edge_f, nbr = run_sess(enc_sess, [coords, mask, residue_index, chain_index])
        (logits,) = run_sess(dec_sess, [node_f, edge_f, nbr, mask, ar_full, seq_oh])
        split_lp = _log_softmax(np.asarray(logits, np.float64))[:length]

        d = float(np.max(np.abs(split_lp - ref_lp.astype(np.float64)[:length])))
        name = f"{geom['name']}_{lane}"
        result["per_case"].append({"case": name, "max_abs_nats": d})
        result["cases_total"] += 1
        if d <= args.bound:
          result["cases_within_bound"] += 1
        if d > result["max_abs_nats_vs_reference"]:
          result["max_abs_nats_vs_reference"] = d
          result["worst_case"] = name
        if result["cases_total"] == 1:
          poisoned = split_lp.copy()
          poisoned.flat[0] += CTRL_PERTURB
          result["ctrl_perturb_detected"] = bool(
            np.max(np.abs(poisoned - ref_lp.astype(np.float64)[:length])) > args.bound
          )
        logger.info("%s: split vs reference max_abs %.4e nats", name, d)

    result["controls_detected"] = int(result["ctrl_perturb_detected"])

  except Exception as exc:  # noqa: BLE001 - a failure IS the finding
    result["failure"] = f"{type(exc).__name__}: {exc}"
    logger.exception("G3a-direct failed")

  total = result["cases_total"]
  result["outcome"] = (
    "incomplete"
    if total < 2 or not result["git_clean"] or result["failure"]
    else "ctrl_blind"
    if result["controls_detected"] < result["controls_total"]
    else "pass"
    if result["cases_within_bound"] == total
    else "fail"
  )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  logger.info("outcome=%s -> %s", result["outcome"], args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
