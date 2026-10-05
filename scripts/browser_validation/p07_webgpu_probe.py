"""G2-probe: is the WebGPU execution provider reachable for our graphs, and how far?

Spec ODQ-B3 scopes WebGPU to a capability probe: no numbers, no parity claims. This gate
honours that literally. It records HOW FAR the stack gets — browser exposes WebGPU,
adapter obtainable, ORT accepts the provider, encoder loads, encoder produces finite
output — and nothing else.

**Read the grading carefully: "WebGPU is unavailable" is a PASS.** The question this gate
answers is "did we successfully determine the capability", not "is WebGPU available". A
run that cleanly establishes WebGPU cannot be reached here has answered its question and
is as useful as one that establishes it can — arguably more so, since it tells an
integrator not to plan around it. Only a probe that fails to determine anything (browser
crash, page never settles) is `incomplete`.

Staging matters and is why each step is recorded separately: WebGPU may be absent, present
without a usable adapter, or adapter-capable while ORT still refuses the graph. Those are
different answers for someone deciding whether the GPU path is worth pursuing, and one
boolean would collapse them.

The encoder is the graph probed on purpose. It carries the k-NN sort, which jax2onnx
lowers to TopK with the int64 indices ONNX mandates and the WebGPU EP is documented not to
support (xtrax research 260914 S4). If any graph exposes the dtype gap, it is that one.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))
_BV = _REPO_ROOT / "scripts" / "browser_validation"
if str(_BV) not in sys.path:
  sys.path.insert(0, str(_BV))

logger = logging.getLogger("p07_webgpu_probe")

_CODE_PATHS = ("src", "scripts", "browser", "pyproject.toml", "uv.lock")
_SAMPLER = _REPO_ROOT / "browser" / "aminx-sampler"


def _git_state(repo: Path) -> tuple[str, bool]:
  def _run(*args: str) -> str:
    return subprocess.run(  # noqa: S603
      ["git", *args],  # noqa: S607
      cwd=repo, capture_output=True, text=True, check=True, timeout=60,
    ).stdout.strip()

  return _run("rev-parse", "HEAD"), not _run(
    "status", "--porcelain", "--untracked-files=all", "--", *_CODE_PATHS
  )


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--release-dir", type=Path, required=True)
  parser.add_argument("--ort-dir", type=Path, required=True)
  parser.add_argument("--bucket", type=int, default=128)
  parser.add_argument("--node-bin", type=str, default=None)
  args = parser.parse_args(argv)

  logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
  )

  git_hash, git_clean = _git_state(_REPO_ROOT)
  result: dict[str, Any] = {
    "bucket": args.bucket,
    "git_hash": git_hash,
    "git_clean": git_clean,
    "probe_ran": False,
    "navigator_gpu_present": False,
    "adapter_obtained": False,
    "adapter_info": None,
    "ort_webgpu_session_created": False,
    "ort_webgpu_error": "",
    "encoder_ran": False,
    "encoder_outputs_finite": None,
    "chromium_version": "",
    "furthest_stage": "none",
    "failure": "",
  }

  try:
    from p07_knobs_gate import _node_bin  # noqa: PLC0415

    node = _node_bin(args.node_bin)
    work = Path(tempfile.mkdtemp(prefix="p07_webgpu_probe_"))
    site = work / "site"
    (site / "models").mkdir(parents=True, exist_ok=True)
    (site / "ort").mkdir(parents=True, exist_ok=True)

    shutil.copy(_SAMPLER / "webgpu_probe_index.html", site / "index.html")
    shutil.copy(_SAMPLER / "webgpu_probe_page.mjs", site / "webgpu_probe_page.mjs")
    name = f"p07_encoder_L{args.bucket}.onnx"
    shutil.copy(args.release_dir / name, site / "models" / name)
    dist = args.ort_dir / "node_modules" / "onnxruntime-web" / "dist"
    for f in dist.iterdir():
      if f.is_file():
        shutil.copy(f, site / "ort" / f.name)

    probe_out = work / "probe.json"
    cmd = [
      node, str(_REPO_ROOT / "browser" / "layer_c" / "webgpu_probe.mjs"),
      "--site", str(site),
      "--out", str(probe_out),
      "--bucket", str(args.bucket),
    ]
    logger.info("running: %s", " ".join(cmd))
    # Capture rather than inherit: when the driver dies at import time (a missing
    # package, say) an inherited stderr did NOT reach this gate's log, and the failure
    # read as a bare "wrote no output" with the actual cause invisible. Folding the tail
    # into the message makes the next such failure diagnosable from the record alone.
    proc = subprocess.run(cmd, check=False, timeout=900, capture_output=True, text=True)  # noqa: S603
    if proc.stderr:
      logger.info("probe driver stderr tail:\n%s", proc.stderr[-2000:])
    if not probe_out.is_file():
      msg = f"probe driver wrote no output (rc={proc.returncode}): {proc.stderr[-600:]}"
      raise RuntimeError(msg)
    p = json.loads(probe_out.read_text())

    result["probe_ran"] = bool(p.get("probe_ok"))
    result["chromium_version"] = str(p.get("chromium_version") or "")
    for k in (
      "navigator_gpu_present", "adapter_obtained", "adapter_info",
      "ort_webgpu_session_created", "encoder_ran", "encoder_outputs_finite",
    ):
      if k in p:
        result[k] = p[k]
    result["ort_webgpu_error"] = str(p.get("ort_webgpu_error") or "")
    if p.get("probe_error"):
      result["failure"] = str(p["probe_error"])[:800]

    stages = [
      ("encoder_finite", bool(result["encoder_outputs_finite"])),
      ("encoder_ran", bool(result["encoder_ran"])),
      ("session_created", bool(result["ort_webgpu_session_created"])),
      ("adapter", bool(result["adapter_obtained"])),
      ("navigator_gpu", bool(result["navigator_gpu_present"])),
    ]
    result["furthest_stage"] = next((n for n, ok in stages if ok), "none")
    logger.info(
      "furthest stage reached: %s | navigator.gpu=%s adapter=%s session=%s",
      result["furthest_stage"], result["navigator_gpu_present"],
      result["adapter_obtained"], result["ort_webgpu_session_created"],
    )
    if result["ort_webgpu_error"]:
      logger.info("ORT webgpu error: %s", result["ort_webgpu_error"][:300])

  except Exception as exc:  # noqa: BLE001 - a failure IS the finding
    result["failure"] = f"{type(exc).__name__}: {exc}"
    logger.exception("WebGPU probe failed")

  result["outcome"] = (
    "incomplete"
    if not result["probe_ran"] or not result["git_clean"] or result["failure"]
    else "pass"
  )

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  logger.info("outcome=%s -> %s", result["outcome"], args.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
