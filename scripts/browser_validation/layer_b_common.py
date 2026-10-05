"""Layer (b) shared plumbing: artifact manifest, provenance helpers (T2, D-G).

- **`manifest_append`** -- the ONE writer of `<artifacts_dir>/<artifact_subdir>/
  artifact_manifest.json` (D-G "One writer"). Append-only: an existing row's sha256 is
  never rewritten, and a conflicting duplicate key (same path, different sha256) raises.
  Writes atomically (tmp file + `os.replace`).
- **`load_manifest_verified`** -- loads a manifest and verifies every referenced
  artifact's live sha256 against the recorded one. Raises on any mismatch or missing
  file; callers translate that into an exit-3 integrity refusal (common context "Result-
  emission rule" #2).
- **`artifact_path`** -- resolves a manifest row's on-disk path under its
  `artifact_subdir`.
- **`bathos_overlay_provenance`** -- the `$BATHOSW` overlay's code identity as seen by
  THIS interpreter (`importlib.util.find_spec("bathos").origin`), for a script's own
  `versions` result field (common context "bth provenance").
- **`assert_checkpoint_pinned`** -- checks a checkpoint's live sha256
  (`aminx.io.weights.weight_provenance`) against `reference_pins.json` (V10 fixtures
  note: "record its file sha256 and check it against `$BV/reference_pins.json`").
- **`resolve_artifact_subdir`** (T3a) -- D-G's manifest rows do NOT carry a per-row
  `artifact_subdir` (T2's actual `layer_b_build.py` never wrote one); a consumer running
  at a LATER commit than the build (T3a's own pre-registration commit, for one) cannot
  recover the build's `git rev-parse --short=12 HEAD` default from its own checkout. This
  discovers it instead, by sha256-verifying the tracked manifest against each existing
  immediate subdirectory of `artifacts_base` and returning the (first, sorted) one that
  verifies cleanly -- "verify by record", never by assumed naming.
- **`converted_by_path`** (T3a) -- per-path availability (T2's BLOCKED table): a path is
  "available" iff the tracked manifest has a `role="clean"` row for every bucket in
  `buckets`.
- **`near_tie_rows`** (T3a) -- the layer-b/c near-tie rule (pre-registered bars,
  "Near-tie rule"): row `i` is a near-tie row iff any adjacent gap among its first `k+1`
  ascending-sorted float64 CA-CA distances is `<= epsilon`.
- **`compare_neighbor_indices`** (T3a) -- id-aligned EXACT neighbour-index comparison
  with the near-tie exclusion, returning both the raw and near-tie-excluded mismatch
  counts plus `n_tie_swapped_slots`.
- **`size_control_delta`** (T3a) -- geometric-bisection sizing of a perturbation
  magnitude so `metric_fn(delta) / bar` lands in `target` (spec: "geometric bisection
  over delta in [1e-7, 1e-1], <= 20 steps").
- **`classify_headroom`** (T3a) -- the headroom-rule three-way state for one
  `(measurement, bar)` pair.
- **`ep_provider_histogram`** (T3a) -- parses an ONNX Runtime chrome-trace profile
  (from a session run with `enable_profiling=True`) into `{provider: n_node_events}`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent


def artifact_sha256(path: Path) -> str:
  """sha256 of `path`'s bytes, hex-encoded."""
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_append(rows: list[dict[str, Any]], manifest_path: Path) -> None:
  """Append `rows` (each a dict with at least `"path"` and `"sha256"`) to `manifest_path`.

  Append-only (D-G "One writer"): a row whose `path` key already exists with the SAME
  sha256 is a no-op; a row whose `path` key already exists with a DIFFERENT sha256
  raises `ValueError` (a conflicting duplicate key). Writes atomically: a tmp file in
  the same directory, then `os.replace`.

  Args:
    rows: Manifest rows to add, each keyed by its own `"path"` (relative to the
      manifest's `artifact_subdir`).
    manifest_path: The manifest file to read-modify-write.

  Raises:
    ValueError: If a row's `path` already exists in the manifest with a different
      `sha256`.
  """
  manifest_path = Path(manifest_path)
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  if manifest_path.is_file():
    with manifest_path.open() as fh:
      manifest = json.load(fh)
  else:
    manifest = {"artifacts": {}}
  artifacts = manifest.setdefault("artifacts", {})

  for row in rows:
    key = row["path"]
    existing = artifacts.get(key)
    if existing is not None:
      if existing.get("sha256") != row.get("sha256"):
        msg = (
          f"manifest_append: conflicting duplicate key {key!r} -- existing sha256 "
          f"{existing.get('sha256')!r} != new {row.get('sha256')!r}. An existing row's "
          f"sha256 is never rewritten (D-G 'One writer')."
        )
        raise ValueError(msg)
      continue  # identical row already present; append-only, nothing to do.
    artifacts[key] = row

  tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp{os.getpid()}")
  with tmp_path.open("w") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
  os.replace(tmp_path, manifest_path)


def artifact_path(manifest: dict[str, Any], key: str, artifacts_dir: Path) -> Path:
  """Resolve manifest row `key`'s on-disk path under `artifacts_dir/<artifact_subdir>/`."""
  row = manifest["artifacts"][key]
  subdir = row.get("artifact_subdir", "")
  return Path(artifacts_dir) / subdir / key if subdir else Path(artifacts_dir) / key


def load_manifest_verified(manifest_path: Path, artifacts_dir: Path) -> dict[str, Any]:
  """Load `manifest_path` and verify every row's recorded sha256 against live bytes.

  Args:
    manifest_path: The (typically git-tracked) manifest JSON to load.
    artifacts_dir: Base artifact directory (`$BV_ARTIFACTS`, D-G) each row's
      `artifact_subdir` is relative to.

  Returns:
    The parsed manifest dict.

  Raises:
    FileNotFoundError: If a referenced artifact is missing on disk.
    ValueError: If a referenced artifact's live sha256 disagrees with the manifest.
  """
  manifest_path = Path(manifest_path)
  with manifest_path.open() as fh:
    manifest = json.load(fh)
  for key in manifest.get("artifacts", {}):
    path = artifact_path(manifest, key, artifacts_dir)
    if not path.is_file():
      msg = f"load_manifest_verified: manifest row {key!r} not found on disk at {path}"
      raise FileNotFoundError(msg)
    actual = artifact_sha256(path)
    expected = manifest["artifacts"][key]["sha256"]
    if actual != expected:
      msg = (
        f"load_manifest_verified: manifest row {key!r} sha256 mismatch -- recorded "
        f"{expected!r}, actual {actual!r} at {path}"
      )
      raise ValueError(msg)
  return manifest


def bathos_overlay_provenance() -> dict[str, Any]:
  """The `$BATHOSW` overlay's code identity as seen by this interpreter.

  Mirrors `local_run.sh`'s own `bth_provenance.json` (the installed `bth` TOOL's code
  identity), but for the `--with=bathos@...` overlay a script imports directly. Returns
  `{"origin": <path or None>, "<name>.py_sha256": <hex>, ...}` for whichever of
  `sidecar.py`/`mcp.py`/`cli.py` exist next to the resolved module (not every bathos
  layout ships all three, e.g. no bare `cli.py` under the cyclopts-CLI layout).
  """
  spec = importlib.util.find_spec("bathos")
  origin = spec.origin if spec is not None else None
  info: dict[str, Any] = {"origin": origin}
  if origin:
    pkg_dir = Path(origin).parent
    for name in ("sidecar.py", "mcp.py", "cli.py"):
      candidate = pkg_dir / name
      if candidate.is_file():
        info[f"{name}_sha256"] = artifact_sha256(candidate)
  return info


def assert_checkpoint_pinned(checkpoint_id: str) -> str:
  """Return the live sha256 of `checkpoint_id`'s weight file, asserting it matches
  `reference_pins.json`'s `weights_sha256["eqx_zst/<checkpoint_id>"]` (V10-adjacent
  fixtures pinning, used by `layer_b_build.py` for the checkpoint it exports from).

  Raises:
    ValueError: If the pin is missing from `reference_pins.json`, or if the live
      sha256 disagrees with it. Callers translate this into an exit-3 integrity refusal.
  """
  from aminx.io.weights import weight_provenance  # noqa: PLC0415 (heavy import, call-site only)

  pins_path = _SCRIPT_DIR / "reference_pins.json"
  with pins_path.open() as fh:
    pins = json.load(fh)
  key = f"eqx_zst/{checkpoint_id}"
  expected = pins.get("weights_sha256", {}).get(key)
  actual = weight_provenance(checkpoint_id).sha256
  if not expected:
    msg = f"assert_checkpoint_pinned: no pinned sha256 recorded for {key!r} in {pins_path}"
    raise ValueError(msg)
  if actual != expected:
    msg = (
      f"assert_checkpoint_pinned: live sha256 {actual!r} for {checkpoint_id!r} "
      f"disagrees with reference_pins.json's {expected!r}"
    )
    raise ValueError(msg)
  return actual


# ----------------------------------------------------------------------------------
# T3a additions
# ----------------------------------------------------------------------------------


def resolve_artifact_subdir(artifacts_base: Path, manifest_path: Path) -> str:
  """Discover which immediate subdir of `artifacts_base` matches `manifest_path` (D-G).

  Tries `load_manifest_verified(manifest_path, artifacts_base / candidate)` for every
  existing immediate subdirectory, sorted, and returns the first that verifies (every
  row's live sha256 matches). See module docstring: the manifest carries no
  `artifact_subdir` field per row, so this is discovered by verification, not assumed
  from `git rev-parse --short=12 HEAD` (which, at a commit AFTER the build's own, would
  be wrong).

  Raises:
    FileNotFoundError: If `artifacts_base` does not exist, or no candidate verifies.
  """
  artifacts_base = Path(artifacts_base)
  if not artifacts_base.is_dir():
    msg = f"resolve_artifact_subdir: {artifacts_base} does not exist"
    raise FileNotFoundError(msg)
  candidates = sorted(p.name for p in artifacts_base.iterdir() if p.is_dir())
  for candidate in candidates:
    try:
      load_manifest_verified(manifest_path, artifacts_base / candidate)
    except (FileNotFoundError, ValueError):
      continue
    return candidate
  msg = (
    f"resolve_artifact_subdir: no immediate subdir of {artifacts_base} sha256-verifies "
    f"against {manifest_path} (tried {candidates!r})"
  )
  raise FileNotFoundError(msg)


def converted_by_path(
  manifest: dict[str, Any], paths: tuple[str, ...], buckets: tuple[int, ...]
) -> dict[str, bool]:
  """Per-path availability from the tracked manifest (T2 BLOCKED table).

  A path is available iff `manifest["artifacts"]` has a `role="clean"` row named
  `f"{path}_L{bucket}.onnx"` for EVERY bucket in `buckets`.
  """
  artifacts = manifest.get("artifacts", {})
  result: dict[str, bool] = {}
  for path in paths:
    ok = True
    for bucket in buckets:
      row = artifacts.get(f"{path}_L{bucket}.onnx")
      if row is None or row.get("role") != "clean":
        ok = False
        break
    result[path] = ok
  return result


def near_tie_rows(ca_real: np.ndarray, k: int, epsilon: float) -> np.ndarray:
  """Near-tie rule (pre-registered bars, "Near-tie rule"): `(n_real,)` bool, True = near-tie.

  For real row `i`, sort its CA-CA distances to every OTHER real row ascending
  (float64, host-side, on the same float32 `ca_real` every backend receives): `d(1) <=
  ... <= d(n_real - 1)`. Row `i` is near-tie iff any adjacent gap among the first `k+1`
  of those values is `<= epsilon`. This is deliberately NOT `aminx.parity.compare
  .no_tie_mask` (that rule only checks the boundary gap `d(k+1) - d(k)`; this one checks
  every adjacent gap among the first `k+1`, per this spec's own near-tie rule text).

  Args:
    ca_real: `(n_real, 3)` CA coordinates of REAL rows only (already mask-filtered).
    k: Neighbour count (`k+1` distances are inspected per row).
    epsilon: Gap threshold in Angstrom (1e-4, "epsilon_tie").
  """
  ca64 = np.asarray(ca_real, dtype=np.float64)
  n_real = ca64.shape[0]
  diffs = ca64[:, None, :] - ca64[None, :, :]
  distances = np.sqrt(np.sum(diffs * diffs, axis=-1))
  np.fill_diagonal(distances, np.inf)
  sorted_distances = np.sort(distances, axis=-1)
  n_take = min(k + 1, n_real - 1)
  head = sorted_distances[:, :n_take]
  gaps = np.diff(head, axis=-1)
  return np.any(gaps <= epsilon, axis=-1)


def compare_neighbor_indices(
  idx_a: np.ndarray,
  idx_b: np.ndarray,
  real_mask: np.ndarray,
  near_tie: np.ndarray,
) -> dict[str, Any]:
  """Id-aligned EXACT neighbour-index comparison with the near-tie exclusion.

  Args:
    idx_a: `(n_real, k)` neighbour indices, one side.
    idx_b: `(n_real, k)` neighbour indices, the other side.
    real_mask: `(n_real,)` bool, which rows to evaluate (already restricted to real rows
      by the caller's slicing; this additionally allows a caller to exclude e.g. rows
      with too few real neighbours).
    near_tie: `(n_real,)` bool from `near_tie_rows`.

  Returns:
    `{n_mismatch_total, n_mismatch_non_near_tie, n_near_tie_mismatch,
    n_tie_swapped_slots, all_mismatches_near_tie}`. `n_tie_swapped_slots` sums, over
    every MISMATCHING row, the size of the symmetric difference of the two id sets (a
    non-near-tie row with any symmetric difference is itself a `n_mismatch_non_near_tie`
    count, never silently absorbed).
  """
  n_mismatch_total = 0
  n_near_tie_mismatch = 0
  n_mismatch_non_near_tie = 0
  n_tie_swapped_slots = 0
  for i in range(idx_a.shape[0]):
    if not real_mask[i]:
      continue
    set_a = set(idx_a[i].tolist())
    set_b = set(idx_b[i].tolist())
    if set_a == set_b:
      continue
    n_mismatch_total += 1
    symmetric_diff = set_a ^ set_b
    if near_tie[i]:
      n_near_tie_mismatch += 1
      n_tie_swapped_slots += len(symmetric_diff)
    else:
      n_mismatch_non_near_tie += 1
  return {
    "n_mismatch_total": n_mismatch_total,
    "n_mismatch_non_near_tie": n_mismatch_non_near_tie,
    "n_near_tie_mismatch": n_near_tie_mismatch,
    "n_tie_swapped_slots": n_tie_swapped_slots,
    "all_mismatches_near_tie": n_mismatch_total > 0 and n_mismatch_non_near_tie == 0,
  }


def size_control_delta(
  metric_fn: Any,
  bar: float,
  *,
  lo: float = 1e-7,
  hi: float = 1e-1,
  target: tuple[float, float] = (2.0, 10.0),
  max_steps: int = 20,
) -> tuple[float | None, list[dict[str, float]]]:
  """Geometric-bisection search for a `delta` with `metric_fn(delta) / bar` in `target`.

  Assumes `metric_fn` is non-decreasing in `delta` (true near-linearly for a small
  additive weight perturbation before any saturation). Each step evaluates the
  geometric midpoint of the current `[lo, hi]` bracket; if its ratio-to-bar is below
  `target[0]` the bracket's lower bound rises to the midpoint, if above `target[1]` the
  upper bound falls to it, else the search has landed and returns immediately.

  Returns:
    `(delta, tried)` -- `delta` is `None` if no step in `max_steps` landed in `target`
    (the caller's `ctrl_unsized` case); `tried` records every `{delta, ratio_to_bar}`
    step, sizing-search evidence for the commit body.
  """
  tried: list[dict[str, float]] = []
  for _ in range(max_steps):
    mid = math.sqrt(lo * hi)
    ratio = metric_fn(mid) / bar
    tried.append({"delta": mid, "ratio_to_bar": ratio})
    if target[0] <= ratio <= target[1]:
      return mid, tried
    if ratio < target[0]:
      lo = mid
    else:
      hi = mid
  return None, tried


def classify_headroom(measurement: float, bar: float) -> str:
  """Headroom rule three-way state for one `(measurement, bar)` pair.

  `measurement <= bar/2` -> `"advanced"`; `bar/2 < measurement <= bar` ->
  `"low_headroom"`; `measurement > bar` -> `"not_advanced"`.
  """
  if measurement <= bar / 2:
    return "advanced"
  if measurement <= bar:
    return "low_headroom"
  return "not_advanced"


def ep_provider_histogram(profile_path: Path) -> dict[str, int]:
  """Parse an ONNX Runtime chrome-trace profile into `{provider: n_node_events}`.

  `profile_path` is the path `InferenceSession.end_profiling()` returns. Counts every
  event carrying an `args.provider` (node-execution events); non-node events (session
  init/other) are skipped.
  """
  with Path(profile_path).open() as fh:
    events = json.load(fh)
  histogram: dict[str, int] = {}
  for event in events:
    provider = (event.get("args") or {}).get("provider")
    if provider:
      histogram[provider] = histogram.get(provider, 0) + 1
  return histogram
