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
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

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
