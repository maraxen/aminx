"""Does widening ``samples_chunk_size`` reduce LASEr per-sample sampling cost?

The LASEr distributional confirm is priced at ~223 h, ~88% of it sampling, on a
CPU-only jaxlib. ``laser_sample_dist_confirm.py`` sets ``samples_chunk_size=8``,
and ``host/plan.resolve_chunk_size`` makes that value win outright -- without it
the chunk defaults to the full sample count, which the planner was separately
probed to accept at ``Vmap`` width 1000 for these structures. So the confirm
performs ~125 sequential dispatches of width 8 at n=1000 where the memory budget
permits one. Whether that costs anything has never been measured.

This run measures it. Five configurations on ONE structure at ONE temperature,
each timed around ``aminx.host.runner.sample`` only:

    C1  n=32  width=8    baseline -- exactly what the confirm does
    C2  n=32  width=32   the lever -- one full-width dispatch
    C3  n=16  width=8    linearity control -- must cost ~half of C1
    C4  n=8   width=1    positive control -- width 1 must be the slowest per sample
    C5  n=8   width=8    C4's pair at equal n

C4/C5 are the instrument check that matters: if width 1 is NOT measurably worse
per sample than width 8, then the width knob does not do what this script claims
to measure, and every other number here is uninterpretable. That is a real
possible result, not a formality -- the planner probe already falsified one
confident story about this axis today.

Per-unit resume and per-unit timeouts, because an interruption must cost one
configuration rather than the whole run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("laser_chunk_width")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "parity"))

# The structure is pinned by name AND digest via the confirm's own STRUCTURES map,
# so this script cannot silently measure a different backbone than the confirm.
_STRUCTURE_RELATIVE = "tests/fixtures/laser/103m_1.pdb"
_TEMPERATURE = 0.3
_MIN_P = 0.0

# (label, n, width). Fixed before the run; no configuration is chosen later.
_CONFIGS: tuple[tuple[str, int, int], ...] = (
  ("C1_n32_w8", 32, 8),
  ("C2_n32_w32", 32, 32),
  ("C3_n16_w8", 16, 8),
  ("C4_n8_w1", 8, 1),
  ("C5_n8_w8", 8, 8),
)
_PASSES = 2  # pass 0 pays this shape's compile; pass 1 is the warm measurement.

# Per-unit wall-clock ceiling. Generous against the ~41 s/sample prior so that a
# timeout means "something is wrong", not "the prior was optimistic".
_UNIT_TIMEOUT_S = 7200


@dataclass(frozen=True)
class Unit:
  label: str
  n: int
  width: int
  pass_idx: int

  @property
  def uid(self) -> str:
    return f"{self.label}_p{self.pass_idx}"


def _sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _cache_key(unit: Unit, *, script_sha: str, checkpoint_sha: str, pdb_sha: str) -> str:
  payload = "|".join(
    (
      script_sha,
      checkpoint_sha,
      pdb_sha,
      unit.label,
      str(unit.n),
      str(unit.width),
      str(unit.pass_idx),
    )
  )
  return _sha256_bytes(payload.encode())


def _xla_cache_entries(cache_dir: Path) -> int:
  if not cache_dir.is_dir():
    return 0
  return sum(1 for _ in cache_dir.rglob("*") if _.is_file())


def _build_spec(pdb: Path, checkpoint: Path, *, n: int, width: int) -> Any:
  """Mirror the confirm's SamplingSpecification, varying only samples_chunk_size.

  The field set is asserted against the confirm's own builder so that a drift in
  the confirm fails loudly here rather than letting this script time a different
  computation under the same name.
  """
  import laser_sample_dist_confirm as confirm  # type: ignore[import-not-found]

  from aminx.run.specs import SamplingSpecification

  options = confirm._laser_options(  # noqa: SLF001 - deliberate reuse, see docstring
    {"sample_temperature": _TEMPERATURE, "min_p": _MIN_P}
  )
  return SamplingSpecification(
    inputs=str(pdb),
    model_family="lasermpnn",
    model_local_path=checkpoint,
    num_samples=n,
    samples_chunk_size=width,
    return_logits=False,
    random_seed=0,
    temperature=_TEMPERATURE,
    backbone_noise=0.0,
    laser=options,
  )


def _run_one(unit: Unit, pdb: Path, checkpoint: Path, cache_dir: Path) -> dict[str, Any]:
  """Time exactly one runner_sample call. Runs inside the worker subprocess."""
  import jax

  from aminx.host.runner import sample as runner_sample

  jax.config.update("jax_enable_x64", False)

  spec = _build_spec(pdb, checkpoint, n=unit.n, width=unit.width)
  entries_before = _xla_cache_entries(cache_dir)
  # titanix is a shared host. A timing measurement taken while another session
  # saturates the cores is not comparable to one taken on an idle host, and load
  # that DRIFTS between two compared configurations can manufacture a ratio.
  # Record it so the grader can refuse rather than average over it.
  load_before = os.getloadavg()[0]
  started = time.perf_counter()
  result = runner_sample(spec)
  elapsed = time.perf_counter() - started
  load_after = os.getloadavg()[0]
  entries_after = _xla_cache_entries(cache_dir)

  arrays = result["structures"]["0"]["arrays"]
  drawn = len(arrays["sequence"])
  if drawn != unit.n:
    msg = f"{unit.uid}: drew {drawn} sequences, expected {unit.n}"
    raise SystemExit(msg)

  return {
    "uid": unit.uid,
    "label": unit.label,
    "n": unit.n,
    "width": unit.width,
    "pass_idx": unit.pass_idx,
    "elapsed_s": elapsed,
    "seconds_per_sample": elapsed / unit.n,
    "xla_cache_entries_before": entries_before,
    "xla_cache_entries_after": entries_after,
    "compiled_during_unit": entries_after > entries_before,
    "n_drawn": drawn,
    "loadavg_1m_before": load_before,
    "loadavg_1m_after": load_after,
    "n_cpus": os.cpu_count(),
  }


def _worker(args: argparse.Namespace) -> None:
  # Fail on the real cause, not six frames later inside Equinox. An unresolved
  # path arrives here as the literal string "None", which Path() accepts happily
  # and eqx then opens as "None.eqx".
  for flag, value in (("--pdb", args.pdb), ("--checkpoint", args.checkpoint)):
    if value is None or str(value) == "None" or not Path(value).is_file():
      msg = f"worker got an unusable {flag}: {value!r}"
      raise SystemExit(msg)
  unit = Unit(
    label=args.worker_label,
    n=int(args.worker_n),
    width=int(args.worker_width),
    pass_idx=int(args.worker_pass),
  )
  body = _run_one(unit, Path(args.pdb), Path(args.checkpoint), Path(args.cache_dir))
  out = Path(args.worker_out)
  out.write_text(json.dumps(body, indent=2, sort_keys=True))
  # Stamp SECOND, so a crash between the two leaves an unstamped body that is
  # recomputed rather than trusted.
  out.with_suffix(".done").write_text(body["uid"])


def _dispatch(
  unit: Unit,
  *,
  pdb: Path,
  checkpoint: Path,
  body_path: Path,
  cache_dir: Path,
) -> tuple[dict[str, Any] | None, str]:
  """Launch one unit.

  ``pdb`` and ``checkpoint`` are the RESOLVED paths, never ``args.pdb`` /
  ``args.checkpoint`` -- those are None whenever the caller relied on the
  default resolution, and ``str(None)`` reaches the worker as the literal
  "None", which Equinox then opens as "None.eqx". Measured 2026-10-06.
  """
  cmd = [
    sys.executable,
    str(Path(__file__).resolve()),
    "--worker",
    "--worker-label",
    unit.label,
    "--worker-n",
    str(unit.n),
    "--worker-width",
    str(unit.width),
    "--worker-pass",
    str(unit.pass_idx),
    "--worker-out",
    str(body_path),
    "--pdb",
    str(pdb),
    "--checkpoint",
    str(checkpoint),
    "--cache-dir",
    str(cache_dir),
  ]
  env = {
    **os.environ,
    "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
  }
  try:
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
      cmd, env=env, timeout=_UNIT_TIMEOUT_S, capture_output=True, text=True, check=False
    )
  except subprocess.TimeoutExpired:
    return None, f"timeout after {_UNIT_TIMEOUT_S}s"
  if completed.returncode != 0:
    tail = (completed.stderr or "").strip().splitlines()[-6:]
    return None, f"exit {completed.returncode}: {' / '.join(tail)}"
  if not body_path.with_suffix(".done").exists():
    return None, "worker exited 0 without writing a completion stamp"
  return json.loads(body_path.read_text()), ""


def _grade(measured: dict[str, dict[str, Any]]) -> dict[str, Any]:
  """Classify the width effect, after two instrument checks that can refuse."""

  def sps(label: str) -> float | None:
    row = measured.get(label)
    return None if row is None else float(row["seconds_per_sample"])

  c1, c2, c3, c4, c5 = (sps(label) for label, _, _ in _CONFIGS)
  complete = all(v is not None for v in (c1, c2, c3, c4, c5))

  # Positive control: width 1 must be measurably worse per sample than width 8 at
  # equal n. If it is not, the knob is inert and nothing else here means anything.
  width_control_ratio = None if (c4 is None or c5 is None) else c4 / c5
  width_control_fires = width_control_ratio is not None and width_control_ratio >= 1.20

  # Linearity control: halving n at fixed width must roughly halve the cost.
  # Band inherited from the measured scaling exponent of 0.975 (near-linear).
  linearity_ratio = None if (c1 is None or c3 is None) else c3 / c1
  linearity_holds = linearity_ratio is not None and 0.80 <= linearity_ratio <= 1.25

  # Load-stability gate. titanix is shared; a ratio measured across configurations
  # that ran under materially different CPU contention is not a measurement of
  # chunk width. Disclosed amendment, added 2026-10-06 BEFORE any graded number
  # existed, after observing load 17.47 on 20 cores from another session's job.
  loads = [
    max(float(row["loadavg_1m_before"]), float(row["loadavg_1m_after"]))
    for row in measured.values()
    if "loadavg_1m_before" in row
  ]
  load_spread = (max(loads) / min(loads)) if loads and min(loads) > 0 else None
  load_stable = load_spread is not None and load_spread <= 1.50
  load_observed = max(loads) if loads else None

  instrument_ok = bool(complete and width_control_fires and linearity_holds and load_stable)

  widen_ratio = None if (c1 is None or c2 is None) else c2 / c1
  if not instrument_ok or widen_ratio is None:
    verdict = "instrument_unverified"
  elif widen_ratio <= 0.70:
    verdict = "dispatch_bound"
  elif widen_ratio >= 0.90:
    verdict = "compute_bound"
  else:
    verdict = "inconclusive"

  return {
    "verdict": verdict,
    "instrument_ok": instrument_ok,
    "configs_complete": complete,
    "width_control_ratio": width_control_ratio,
    "width_control_fires": width_control_fires,
    "linearity_ratio": linearity_ratio,
    "linearity_holds": linearity_holds,
    "load_spread": load_spread,
    "load_stable": load_stable,
    "load_observed_max": load_observed,
    "widen_ratio": widen_ratio,
    "seconds_per_sample": {label: sps(label) for label, _, _ in _CONFIGS},
  }


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--laser-root", type=Path, default=None)
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--pdb", type=Path, default=None)
  parser.add_argument("--work-dir", type=Path, required=False)
  parser.add_argument("--out", type=Path, default=None)
  parser.add_argument("--cache-dir", type=Path, default=None)
  parser.add_argument(
    "--smoke",
    action="store_true",
    help="Run only C5 pass 0 and report its timing. Sizes the full run; grades nothing.",
  )
  parser.add_argument("--worker", action="store_true")
  parser.add_argument("--worker-label", default="")
  parser.add_argument("--worker-n", type=int, default=0)
  parser.add_argument("--worker-width", type=int, default=0)
  parser.add_argument("--worker-pass", type=int, default=0)
  parser.add_argument("--worker-out", type=Path, default=None)
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  args = _parse(argv)

  if args.worker:
    _worker(args)
    return 0

  import laser_sample_dist_pilot as pilot  # type: ignore[import-not-found]

  laser_root = Path(args.laser_root) if args.laser_root else pilot._default_laser_root()  # noqa: SLF001
  checkpoint = Path(args.checkpoint) if args.checkpoint else pilot._checkpoint(  # noqa: SLF001
    argparse.Namespace(checkpoint=None, laser_root=laser_root)
  )
  pilot._require_checkpoint(checkpoint)  # noqa: SLF001
  # The confirm resolves "tests/"-prefixed structures against the AMINX repo and
  # everything else against laser_root (laser_sample_dist_confirm.py:355-359).
  # Mirror that rather than guessing, and check the digest the confirm pins so
  # this script cannot time a different backbone under the same label.
  import laser_sample_dist_confirm as confirm  # type: ignore[import-not-found]

  if args.pdb:
    pdb = Path(args.pdb)
  elif _STRUCTURE_RELATIVE.startswith("tests/"):
    pdb = pilot._repo() / _STRUCTURE_RELATIVE  # noqa: SLF001
  else:
    pdb = laser_root / _STRUCTURE_RELATIVE
  if not pdb.is_file():
    msg = f"structure not found: {pdb}"
    raise SystemExit(msg)
  expected_digest = confirm.STRUCTURES.get(_STRUCTURE_RELATIVE)
  if expected_digest is not None and not args.pdb:
    pinned = expected_digest[0] if isinstance(expected_digest, tuple) else expected_digest
    got = _sha256_file(pdb)
    if got != pinned:
      msg = f"{pdb} sha256 {got} != confirm's pin {pinned}"
      raise SystemExit(msg)

  work_dir = Path(args.work_dir) if args.work_dir else Path("./chunk_width_work")
  work_dir.mkdir(parents=True, exist_ok=True)
  cache_dir = Path(args.cache_dir) if args.cache_dir else work_dir / "xla_cache"
  cache_dir.mkdir(parents=True, exist_ok=True)

  script_sha = _sha256_file(Path(__file__).resolve())
  checkpoint_sha = _sha256_file(checkpoint)
  pdb_sha = _sha256_file(pdb)

  units = (
    [Unit("C5_n8_w8", 8, 8, 0)]
    if args.smoke
    else [
      Unit(label, n, width, pass_idx)
      for label, n, width in _CONFIGS
      for pass_idx in range(_PASSES)
    ]
  )

  rows: list[dict[str, Any]] = []
  failures: list[dict[str, str]] = []
  n_reused = 0
  n_computed = 0

  for unit in units:
    key = _cache_key(unit, script_sha=script_sha, checkpoint_sha=checkpoint_sha, pdb_sha=pdb_sha)
    body_path = work_dir / f"{key}.json"
    if body_path.with_suffix(".done").exists():
      rows.append(json.loads(body_path.read_text()))
      n_reused += 1
      logger.info("reused %s", unit.uid)
      continue
    logger.info("running %s (n=%d width=%d)", unit.uid, unit.n, unit.width)
    body, error = _dispatch(
      unit, pdb=pdb, checkpoint=checkpoint, body_path=body_path, cache_dir=cache_dir
    )
    if body is None:
      logger.error("%s FAILED: %s", unit.uid, error)
      failures.append({"uid": unit.uid, "error": error})
      continue
    rows.append(body)
    n_computed += 1
    logger.info(
      "%s elapsed %.1f s (%.2f s/sample, compiled=%s)",
      unit.uid,
      body["elapsed_s"],
      body["seconds_per_sample"],
      body["compiled_during_unit"],
    )

  # Only the WARM pass is graded; pass 0 exists to pay this shape's compile.
  measured = {row["label"]: row for row in rows if row["pass_idx"] == _PASSES - 1}
  graded = _grade(measured) if not args.smoke else {"verdict": "smoke", "instrument_ok": False}

  results: dict[str, Any] = {
    **graded,
    "smoke": bool(args.smoke),
    "rows": rows,
    "failures": failures,
    "n_reused": n_reused,
    "n_computed": n_computed,
    "script_sha256": script_sha,
    "checkpoint_sha256": checkpoint_sha,
    "pdb": str(pdb),
    "pdb_sha256": pdb_sha,
    "structure_relative": _STRUCTURE_RELATIVE,
    "temperature": _TEMPERATURE,
    "min_p": _MIN_P,
    "unit_timeout_s": _UNIT_TIMEOUT_S,
    "work_dir": str(work_dir),
  }

  payload = json.dumps(results, indent=2, sort_keys=True)
  destinations = []
  bth_results = os.environ.get("BTH_RESULTS_PATH")
  if bth_results:
    destinations.append(Path(bth_results))
  if args.out:
    destinations.append(Path(args.out))
  for dest in destinations:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(payload)
  if not destinations:
    print(payload)

  logger.info("verdict=%s reused=%d computed=%d", results["verdict"], n_reused, n_computed)
  return 1 if failures and not args.smoke else 0


if __name__ == "__main__":
  raise SystemExit(main())
