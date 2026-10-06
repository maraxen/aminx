"""Does widening ``samples_chunk_size`` reduce LASEr per-sample sampling cost?

The LASEr distributional confirm is priced at ~223 h, ~88% of it sampling, on a
CPU-only jaxlib. ``laser_sample_dist_confirm.py:557`` sets
``samples_chunk_size=8``, and ``host/plan.resolve_chunk_size`` (plan.py:450-454)
makes that value win outright -- without it the chunk defaults to the full sample
count, which the planner was separately probed to accept at ``Vmap`` width 1000
for these structures. So the confirm performs ~125 sequential dispatches of width
8 at n=1000 where the memory budget permits one. Whether that costs anything has
never been measured.

PAIRED DESIGN, and the reason for it. titanix is shared. Measured 2026-10-06, the
1-minute load average moved from 17.47 to 9.55 on 20 cores within minutes, driven
by another session's 125-thread job. A 1.8x swing in available CPU swamps the ~30%
effect this measurement is trying to resolve, so a design that compares two
configurations run hours apart cannot answer the question at all.

Every comparison is therefore measured as an ADJACENT PAIR: the two configurations
run back to back, and the ratio is formed within the pair. A monotone drift in
contention cancels, because both members see nearly the same machine. Pairs are
repeated with the member order REVERSED on alternate repeats, so even a drift
inside a single pair cannot bias the ratio in a fixed direction.

    lever      n=32 w=8  vs  n=32 w=32   x2   -- the question
    control    n=8  w=8  vs  n=8  w=1    x1   -- width 1 must be worse per sample
    linearity  n=32 w=8  vs  n=16 w=8    x1   -- halving n must halve the cost

The control is the one that matters: if the narrowest possible width is not
measurably worse than width 8, the knob does not reach the computation this
script claims to time, and every other number here is uninterpretable. That is a
live possibility -- a confident story about this same axis (that the planner
silently demotes the sample axis to width 1 on a CPU 4 GiB budget) was probed and
falsified on 2026-10-06 before this script was written.

Per-unit resume and per-unit timeouts, because an interruption must cost one
configuration rather than the whole run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("laser_chunk_width")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "parity"))

# Pinned by name AND digest via the confirm's own STRUCTURES map, so this script
# cannot silently time a different backbone than the confirm.
_STRUCTURE_RELATIVE = "tests/fixtures/laser/103m_1.pdb"
_TEMPERATURE = 0.3
_MIN_P = 0.0

# The fixture's resolved span, counted from its own CA records, and the default
# the confirm inherits because it never sets max_length. host/runner.py:127-131
# states the cost model: "Autoregressive decode cost grows roughly with the
# square of the padded length", so the default pays about (512/160)^2 ~ 10x.
_SPAN = 154
_PAD_DEFAULT = 512
_PAD_FITTED = 160  # ceil(154/32)*32, exactly what runner.py's own warning suggests

# (label, n, width, max_length)
_Config = tuple[str, int, int, int]

_BASE: _Config = ("base_n32_w8_p512", 32, 8, _PAD_DEFAULT)
_WIDE: _Config = ("wide_n32_w32_p512", 32, 32, _PAD_DEFAULT)
_CTRL_W8: _Config = ("ctrl_n8_w8_p512", 8, 8, _PAD_DEFAULT)
_CTRL_W1: _Config = ("ctrl_n8_w1_p512", 8, 1, _PAD_DEFAULT)
_HALF_N: _Config = ("half_n16_w8_p512", 16, 8, _PAD_DEFAULT)
_PAD_FIT: _Config = ("fit_n8_w8_p160", 8, 8, _PAD_FITTED)
_PAD_REF: _Config = ("ref_n8_w8_p512", 8, 8, _PAD_DEFAULT)

# (pair name, member A, member B, repeats). Fixed before the run.
# "padding" leads because it is the larger hypothesis by an order of magnitude.
_PAIRS: tuple[tuple[str, _Config, _Config, int], ...] = (
  ("padding", _PAD_REF, _PAD_FIT, 2),
  ("lever", _BASE, _WIDE, 2),
  ("control", _CTRL_W8, _CTRL_W1, 1),
  ("linearity", _BASE, _HALF_N, 1),
)

# Per-unit wall-clock ceiling. Generous against the measured 49.8 s/sample (which
# included a cold compile, under load 17) so a timeout means something is wrong.
_UNIT_TIMEOUT_S = 7200

# Max/min of the 1-minute load averages recorded WITHIN one pair instance.
_LOAD_SPREAD_MAX = 1.50


@dataclass(frozen=True)
class Unit:
  label: str
  n: int
  width: int
  max_length: int
  pair: str
  repeat: int
  warmup: bool

  @property
  def uid(self) -> str:
    kind = "warm" if self.warmup else f"{self.pair}r{self.repeat}"
    return f"{self.label}__{kind}"


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
      str(unit.max_length),
      unit.uid,
    )
  )
  return _sha256_bytes(payload.encode())


def _xla_cache_entries(cache_dir: Path) -> int:
  if not cache_dir.is_dir():
    return 0
  return sum(1 for entry in cache_dir.rglob("*") if entry.is_file())


def _build_spec(pdb: Path, checkpoint: Path, *, n: int, width: int, max_length: int) -> Any:
  """Mirror the confirm's SamplingSpecification, varying chunk width and max_length.

  The confirm sets NEITHER max_length (so it inherits the 512 default) nor any
  padding hint, which is why the padding pair exists.
  """
  import laser_sample_dist_confirm as confirm  # type: ignore[import-not-found]

  from aminx.run.specs import SamplingSpecification

  options = confirm._laser_options(  # noqa: SLF001 - deliberate reuse, see module docstring
    {"sample_temperature": _TEMPERATURE, "min_p": _MIN_P}
  )
  return SamplingSpecification(
    inputs=str(pdb),
    model_family="lasermpnn",
    model_local_path=checkpoint,
    num_samples=n,
    samples_chunk_size=width,
    max_length=max_length,
    return_logits=False,
    random_seed=0,
    temperature=_TEMPERATURE,
    backbone_noise=0.0,
    laser=options,
  )


def _run_one(unit: Unit, pdb: Path, checkpoint: Path, cache_dir: Path) -> dict[str, Any]:
  """Time exactly one runner_sample call. Runs inside the worker subprocess."""
  import warnings

  import jax

  from aminx.host.runner import sample as runner_sample

  jax.config.update("jax_enable_x64", False)

  # aminx warns when a batch is padded to far more than its longest chain's span
  # (host/runner.py:_PaddingCheck). Capturing it here rather than letting the
  # subprocess swallow it is the whole reason the padding pair exists: the first
  # run of this script discarded that warning on every unit.
  caught: list[str] = []

  spec = _build_spec(pdb, checkpoint, n=unit.n, width=unit.width, max_length=unit.max_length)
  entries_before = _xla_cache_entries(cache_dir)
  load_before = os.getloadavg()[0]
  started = time.perf_counter()
  with warnings.catch_warnings(record=True) as record:
    warnings.simplefilter("always")
    result = runner_sample(spec)
    caught.extend(str(w.message) for w in record)
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
    "max_length": unit.max_length,
    "pair": unit.pair,
    "repeat": unit.repeat,
    "warmup": unit.warmup,
    "elapsed_s": elapsed,
    "seconds_per_sample": elapsed / unit.n,
    "xla_cache_entries_before": entries_before,
    "xla_cache_entries_after": entries_after,
    "compiled_during_unit": entries_after > entries_before,
    "n_drawn": drawn,
    "loadavg_1m_before": load_before,
    "loadavg_1m_after": load_after,
    "n_cpus": os.cpu_count(),
    "warnings": caught,
    "padding_warning_seen": any("padded to max_length" in w for w in caught),
  }


def _worker(args: argparse.Namespace) -> None:
  # Fail on the real cause, not six frames later inside Equinox. An unresolved
  # path arrives here as the literal string "None", which Path() accepts happily
  # and eqx then opens as "None.eqx". Measured 2026-10-06.
  for flag, value in (("--pdb", args.pdb), ("--checkpoint", args.checkpoint)):
    if value is None or str(value) == "None" or not Path(value).is_file():
      msg = f"worker got an unusable {flag}: {value!r}"
      raise SystemExit(msg)
  unit = Unit(
    label=args.worker_label,
    n=int(args.worker_n),
    width=int(args.worker_width),
    max_length=int(args.worker_max_length),
    pair=args.worker_pair,
    repeat=int(args.worker_repeat),
    warmup=bool(args.worker_warmup),
  )
  body = _run_one(unit, Path(args.pdb), Path(args.checkpoint), Path(args.cache_dir))
  out = Path(args.worker_out)
  out.write_text(json.dumps(body, indent=2, sort_keys=True))
  # Stamp SECOND, so a crash between the two leaves an unstamped body that is
  # recomputed rather than trusted.
  out.with_suffix(".done").write_text(body["uid"])


def _dispatch(
  unit: Unit, *, pdb: Path, checkpoint: Path, body_path: Path, cache_dir: Path
) -> tuple[dict[str, Any] | None, str]:
  """Launch one unit with the RESOLVED paths, never the raw argparse values."""
  cmd = [
    sys.executable,
    str(Path(__file__).resolve()),
    "--worker",
    "--worker-label", unit.label,
    "--worker-n", str(unit.n),
    "--worker-width", str(unit.width),
    "--worker-max-length", str(unit.max_length),
    "--worker-pair", unit.pair,
    "--worker-repeat", str(unit.repeat),
    "--worker-out", str(body_path),
    "--pdb", str(pdb),
    "--checkpoint", str(checkpoint),
    "--cache-dir", str(cache_dir),
  ]  # fmt: skip
  if unit.warmup:
    cmd.append("--worker-warmup")
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


def _plan_units() -> list[Unit]:
  """Warm-ups for each distinct shape first, then each pair's members adjacently.

  Member order reverses on odd repeats so a drift inside one pair cannot bias the
  ratio in a fixed direction.
  """
  units: list[Unit] = []
  # One warm-up per distinct WIDTH, at n == width (a single chunk). The compiled
  # shape is set by the chunk width, not by n: a run of n samples at width w is
  # ceil(n/w) dispatches of the SAME shape, and every (n, width) pair here
  # divides evenly, so no remainder chunk introduces a second shape. Warming per
  # width instead of per configuration costs 41 sample-draws rather than 96.
  seen_shapes: set[tuple[int, int]] = set()
  for _, member_a, member_b, _ in _PAIRS:
    for label, _n, width, max_length in (member_a, member_b):
      if (width, max_length) in seen_shapes:
        continue
      seen_shapes.add((width, max_length))
      units.append(
        Unit(label, width, width, max_length, pair="warmup", repeat=0, warmup=True)
      )
  for pair_name, member_a, member_b, repeats in _PAIRS:
    for repeat in range(repeats):
      ordered = (member_a, member_b) if repeat % 2 == 0 else (member_b, member_a)
      for label, n, width, max_length in ordered:
        units.append(
          Unit(label, n, width, max_length, pair=pair_name, repeat=repeat, warmup=False)
        )
  return units


def _pair_instances(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
  grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
  for row in rows:
    if row.get("warmup"):
      continue
    grouped.setdefault((row["pair"], int(row["repeat"])), []).append(row)
  return grouped


def _pair_ratio(
  instances: dict[tuple[str, int], list[dict[str, Any]]],
  pair: str,
  numerator_label: str,
  denominator_label: str,
) -> tuple[float | None, list[float], bool]:
  """Median within-pair ratio, plus whether every instance's load was stable."""
  ratios: list[float] = []
  stable = True
  found = False
  for (pair_name, _), members in sorted(instances.items()):
    if pair_name != pair or len(members) != 2:
      continue
    by_label = {row["label"]: row for row in members}
    num, den = by_label.get(numerator_label), by_label.get(denominator_label)
    if num is None or den is None:
      continue
    found = True
    loads = [
      float(row[key])
      for row in members
      for key in ("loadavg_1m_before", "loadavg_1m_after")
      if key in row
    ]
    if not loads or min(loads) <= 0 or max(loads) / min(loads) > _LOAD_SPREAD_MAX:
      stable = False
    ratios.append(float(num["seconds_per_sample"]) / float(den["seconds_per_sample"]))
  if not ratios:
    return None, [], False
  return statistics.median(ratios), ratios, (stable and found)


def _grade(rows: list[dict[str, Any]]) -> dict[str, Any]:
  instances = _pair_instances(rows)

  padding_ratio, padding_all, padding_stable = _pair_ratio(
    instances, "padding", _PAD_FIT[0], _PAD_REF[0]
  )
  lever_ratio, lever_all, lever_stable = _pair_ratio(
    instances, "lever", _WIDE[0], _BASE[0]
  )
  control_ratio, control_all, control_stable = _pair_ratio(
    instances, "control", _CTRL_W1[0], _CTRL_W8[0]
  )
  linearity_ratio, linearity_all, linearity_stable = _pair_ratio(
    instances, "linearity", _HALF_N[0], _BASE[0]
  )

  control_fires = control_ratio is not None and control_ratio >= 1.20
  linearity_holds = linearity_ratio is not None and 0.80 <= linearity_ratio <= 1.25
  load_stable = bool(lever_stable and control_stable and linearity_stable and padding_stable)
  instrument_ok = bool(
    lever_ratio is not None
    and padding_ratio is not None
    and control_fires
    and linearity_holds
    and load_stable
  )

  # The padding verdict is reported alongside the chunk-width verdict rather than
  # folded into it: they are independent levers and a reader needs both.
  if padding_ratio is None:
    padding_verdict = "unmeasured"
  elif padding_ratio <= 0.50:
    padding_verdict = "padding_dominates"
  elif padding_ratio >= 0.90:
    padding_verdict = "padding_irrelevant"
  else:
    padding_verdict = "padding_partial"

  if not instrument_ok:
    verdict = "instrument_unverified"
  elif lever_ratio <= 0.70:
    verdict = "dispatch_bound"
  elif lever_ratio >= 0.90:
    verdict = "compute_bound"
  else:
    verdict = "inconclusive"

  expected_sq = (_PAD_DEFAULT / _PAD_FITTED) ** 2
  return {
    "verdict": verdict,
    "padding_verdict": padding_verdict,
    "padding_ratio": padding_ratio,
    "padding_ratio_all": padding_all,
    "padding_speedup": (1.0 / padding_ratio) if padding_ratio else None,
    "padding_speedup_predicted_by_square_law": expected_sq,
    "span_residues": _SPAN,
    "pad_default": _PAD_DEFAULT,
    "pad_fitted": _PAD_FITTED,
    "instrument_ok": instrument_ok,
    "widen_ratio": lever_ratio,
    "widen_ratio_all": lever_all,
    "width_control_ratio": control_ratio,
    "width_control_ratio_all": control_all,
    "width_control_fires": control_fires,
    "linearity_ratio": linearity_ratio,
    "linearity_ratio_all": linearity_all,
    "linearity_holds": linearity_holds,
    "load_stable": load_stable,
    "load_stable_by_pair": {
      "padding": padding_stable,
      "lever": lever_stable,
      "control": control_stable,
      "linearity": linearity_stable,
    },
    "load_spread_max_allowed": _LOAD_SPREAD_MAX,
    "seconds_per_sample": {
      row["uid"]: row["seconds_per_sample"] for row in rows if not row.get("warmup")
    },
    "padding_warning_seen_any": any(r.get("padding_warning_seen") for r in rows),
  }


def _synthetic_row(
  label: str, n: int, width: int, pair: str, repeat: int, sps: float, load: float
) -> dict[str, Any]:
  return {
    "uid": f"{label}__{pair}r{repeat}",
    "label": label,
    "n": n,
    "width": width,
    "max_length": _PAD_DEFAULT,
    "pair": pair,
    "repeat": repeat,
    "warmup": False,
    "seconds_per_sample": sps,
    "loadavg_1m_before": load,
    "loadavg_1m_after": load,
  }


def _synthetic_rows(
  *,
  widen_sps: float,
  control_ratio: float = 2.0,
  linearity: float = 1.0,
  load: float = 10.0,
  padding_sps: float = 1.0,
) -> list[dict[str, Any]]:
  """A full synthetic result set with controllable lever and padding ratios."""
  rows: list[dict[str, Any]] = []
  for repeat in (0, 1):
    rows.append(_synthetic_row(_PAD_REF[0], 8, 8, "padding", repeat, 10.0, load))
    rows.append(_synthetic_row(_PAD_FIT[0], 8, 8, "padding", repeat, padding_sps, load))
    rows.append(_synthetic_row(_BASE[0], 32, 8, "lever", repeat, 10.0, load))
    rows.append(_synthetic_row(_WIDE[0], 32, 32, "lever", repeat, widen_sps, load))
  rows.append(_synthetic_row(_CTRL_W8[0], 8, 8, "control", 0, 10.0, load))
  rows.append(_synthetic_row(_CTRL_W1[0], 8, 1, "control", 0, 10.0 * control_ratio, load))
  rows.append(_synthetic_row(_BASE[0], 32, 8, "linearity", 0, 10.0, load))
  rows.append(_synthetic_row(_HALF_N[0], 16, 8, "linearity", 0, 10.0 * linearity, load))
  return rows


def _self_test() -> tuple[bool, list[str]]:
  """Grade synthetic ground truth, including controls that MUST refuse.

  A grader that can only return one answer is not a grader. Three of these six
  checks require a refusal, so passing them shows the instrument can say no.
  """
  failures: list[str] = []

  def check(name: str, rows: list[dict[str, Any]], expected: str) -> None:
    got = _grade(rows)["verdict"]
    if got != expected:
      failures.append(f"{name}: expected {expected}, got {got}")

  # Positive: the three substantive bands, at values either side of 0.70/0.90.
  check("lever_strong", _synthetic_rows(widen_sps=5.0), "dispatch_bound")
  check("lever_absent", _synthetic_rows(widen_sps=9.5), "compute_bound")
  check("lever_middle", _synthetic_rows(widen_sps=8.0), "inconclusive")

  # The padding verdict is independent of the chunk-width verdict.
  for name, padding_sps, expected in (
    ("padding_dominates", 1.0, "padding_dominates"),
    ("padding_irrelevant", 9.8, "padding_irrelevant"),
    ("padding_partial", 7.0, "padding_partial"),
  ):
    graded = _grade(_synthetic_rows(widen_sps=5.0, padding_sps=padding_sps))
    if graded["padding_verdict"] != expected:
      failures.append(f"{name}: expected {expected}, got {graded['padding_verdict']}")

  # Negative: each control alone must force a refusal despite a strong lever.
  check(
    "control_inert",
    _synthetic_rows(widen_sps=5.0, control_ratio=1.0),
    "instrument_unverified",
  )
  check(
    "linearity_broken",
    _synthetic_rows(widen_sps=5.0, linearity=2.0),
    "instrument_unverified",
  )
  drifted = _synthetic_rows(widen_sps=5.0)
  drifted[-1]["loadavg_1m_after"] = 30.0  # inside the linearity pair
  check("load_drift", drifted, "instrument_unverified")

  return (not failures), failures


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--laser-root", type=Path, default=None)
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--pdb", type=Path, default=None)
  parser.add_argument("--work-dir", type=Path, default=None)
  parser.add_argument("--out", type=Path, default=None)
  parser.add_argument("--cache-dir", type=Path, default=None)
  parser.add_argument(
    "--smoke",
    action="store_true",
    help="Run one cheap unit and report its timing. Sizes the run; grades nothing.",
  )
  parser.add_argument(
    "--self-test-only",
    action="store_true",
    help="Run the grader's synthetic checks and exit. Touches no model.",
  )
  parser.add_argument("--worker", action="store_true")
  parser.add_argument("--worker-label", default="")
  parser.add_argument("--worker-n", type=int, default=0)
  parser.add_argument("--worker-width", type=int, default=0)
  parser.add_argument("--worker-max-length", type=int, default=_PAD_DEFAULT)
  parser.add_argument("--worker-pair", default="")
  parser.add_argument("--worker-repeat", type=int, default=0)
  parser.add_argument("--worker-warmup", action="store_true")
  parser.add_argument("--worker-out", type=Path, default=None)
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  args = _parse(argv)

  if args.worker:
    _worker(args)
    return 0

  # The instrument is checked BEFORE the model is touched, so a broken grader
  # costs seconds rather than three hours.
  self_test_passed, self_test_failed = _self_test()
  for failure in self_test_failed:
    logger.error("self-test: %s", failure)
  logger.info(
    "self-test %s (9 checks, 3 of them refusals)", "PASSED" if self_test_passed else "FAILED"
  )
  if args.self_test_only:
    return 0 if self_test_passed else 1
  if not self_test_passed:
    msg = "grader self-test failed; refusing to measure"
    raise SystemExit(msg)

  import laser_sample_dist_confirm as confirm  # type: ignore[import-not-found]
  import laser_sample_dist_pilot as pilot  # type: ignore[import-not-found]

  laser_root = Path(args.laser_root) if args.laser_root else pilot._default_laser_root()  # noqa: SLF001
  checkpoint = (
    Path(args.checkpoint)
    if args.checkpoint
    else pilot._checkpoint(argparse.Namespace(checkpoint=None, laser_root=laser_root))  # noqa: SLF001
  )
  pilot._require_checkpoint(checkpoint)  # noqa: SLF001

  # The confirm resolves "tests/"-prefixed structures against the AMINX repo and
  # everything else against laser_root (laser_sample_dist_confirm.py:355-359).
  if args.pdb:
    pdb = Path(args.pdb)
  elif _STRUCTURE_RELATIVE.startswith("tests/"):
    pdb = pilot._repo() / _STRUCTURE_RELATIVE  # noqa: SLF001
  else:
    pdb = laser_root / _STRUCTURE_RELATIVE
  if not pdb.is_file():
    msg = f"structure not found: {pdb}"
    raise SystemExit(msg)
  pinned = confirm.STRUCTURES.get(_STRUCTURE_RELATIVE)
  if pinned is not None and not args.pdb:
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
    [Unit(*_CTRL_W8, pair="smoke", repeat=0, warmup=True)] if args.smoke else _plan_units()
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
      "%s elapsed %.1f s (%.2f s/sample, compiled=%s, load %.2f->%.2f)",
      unit.uid,
      body["elapsed_s"],
      body["seconds_per_sample"],
      body["compiled_during_unit"],
      body["loadavg_1m_before"],
      body["loadavg_1m_after"],
    )
    if body.get("padding_warning_seen"):
      logger.info("  ^ aminx emitted its padding warning for this unit")

  graded = (
    {"verdict": "smoke", "instrument_ok": False} if args.smoke else _grade(rows)
  )

  results: dict[str, Any] = {
    **graded,
    "smoke": bool(args.smoke),
    "self_test_passed": self_test_passed,
    "self_test_failed": self_test_failed,
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
