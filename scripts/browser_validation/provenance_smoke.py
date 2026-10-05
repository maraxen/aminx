"""Node-provenance smoke: is this machine trustworthy for a layer-(a) bathos run?

Phase 1, T4 step 3. Runs on the local machine (as part of the T4 fixer gate) and,
unmodified, on titanix inside the O6' transient systemd user unit (orchestrator,
T4 step 5). Every value is MEASURED, never assumed:

- ``git_source``/``git_hash``/``git_dirty`` come from ``bathos.git.capture_git_state``,
  the same live-git channel every other browser-validation harness's provenance
  depends on.
- ``bathos_commit`` is the imported bathos distribution's own ``direct_url.json``
  ``vcs_info.commit_id`` -- what actually resolved for this interpreter, not what a
  ``--with`` string asked for.
- ``locked_versions_match`` re-derives the pinned jax/jaxlib/numpy/torch/equinox
  versions straight out of ``uv.lock`` and compares them against what is actually
  importable, so a ``--with`` overlay that silently shadowed the project's own lock
  is caught here instead of downstream.
- ``systemd_scope_ok`` proves the O6' memory-cap mechanism (a transient
  ``systemd-run --user --scope``) actually works on this node; it is recorded but,
  per the pre-registered outcome schema, does not gate ``pass``/``fail``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

BATHOS_COMMIT_PIN = "84be544ecb45734f46e43d22f351a54d6edd6ae5"
LOCKED_PACKAGES = ("jax", "jaxlib", "numpy", "torch", "equinox")


def _find_uv_lock(start: Path) -> Path:
  """Walk upward from ``start`` for the nearest ``uv.lock`` (mirrors the CLUSTER.md
  __file__-anchoring convention: never ``Path.cwd()``, never a bare relative string)."""
  for candidate in (start, *start.parents):
    lock = candidate / "uv.lock"
    if lock.is_file():
      return lock
  msg = f"no uv.lock found walking up from {start}"
  raise FileNotFoundError(msg)


def _locked_versions(lock_path: Path) -> dict[str, str]:
  with lock_path.open("rb") as fh:
    data = tomllib.load(fh)
  versions: dict[str, str] = {}
  for package in data.get("package", []):
    name = package.get("name")
    version = package.get("version")
    if name in LOCKED_PACKAGES and version is not None:
      versions[name] = version
  return versions


def _installed_versions() -> dict[str, str]:
  import importlib.metadata as im

  installed: dict[str, str] = {}
  for name in LOCKED_PACKAGES:
    try:
      installed[name] = im.version(name)
    except im.PackageNotFoundError:
      installed[name] = ""
  return installed


def _check_locked_versions_match() -> bool:
  lock_path = _find_uv_lock(Path(__file__).resolve())
  locked = _locked_versions(lock_path)
  installed = _installed_versions()
  return all(installed.get(name) == version for name, version in locked.items())


def _bathos_commit() -> str:
  import importlib.metadata as im

  try:
    raw = im.distribution("bathos").read_text("direct_url.json")
  except im.PackageNotFoundError:
    return ""
  if not raw:
    return ""
  data = json.loads(raw)
  return data.get("vcs_info", {}).get("commit_id", "")


def _cisternal_version() -> str:
  import importlib.metadata as im

  try:
    return im.version("cisternal")
  except im.PackageNotFoundError:
    return ""


def _torch_ok() -> bool:
  try:
    import torch

    return bool((torch.ones(2) + torch.ones(2)).sum().item() == 4.0)
  except Exception:  # noqa: BLE001
    return False


def _jax_devices_and_ok() -> tuple[bool, bool]:
  """Return ``(jax_ok, cpu_only)``. ``cpu_only`` is only meaningful when ``jax_ok``."""
  try:
    import jax
    import jax.numpy as jnp

    result = jnp.asarray(1.0) + jnp.asarray(1.0)
    jax_ok = bool(result == 2.0)
    cpu_only = all(device.platform == "cpu" for device in jax.devices())
    return jax_ok, cpu_only
  except Exception:  # noqa: BLE001
    return False, False


def _systemd_scope_ok() -> bool:
  try:
    result = subprocess.run(
      ["systemd-run", "--user", "--scope", "-p", "MemoryMax=1G", "true"],
      capture_output=True,
      text=True,
      timeout=30,
    )
    return result.returncode == 0
  except Exception:  # noqa: BLE001
    return False


def run_smoke() -> dict[str, Any]:
  from bathos.git import capture_git_state

  git_state = capture_git_state(Path.cwd())
  torch_ok = _torch_ok()
  jax_ok, cpu_only = _jax_devices_and_ok()
  locked_versions_match = _check_locked_versions_match()

  return {
    "git_source": git_state.provenance_source,
    "git_hash": git_state.hash,
    "git_dirty": git_state.dirty,
    "bathos_commit": _bathos_commit(),
    "cisternal_version": _cisternal_version(),
    "torch_ok": torch_ok,
    "jax_ok": jax_ok,
    "cpu_only": cpu_only,
    "locked_versions_match": locked_versions_match,
    "systemd_scope_ok": _systemd_scope_ok(),
  }


def _write_result(out_path: Path, result: dict[str, Any]) -> None:
  """Write result to --out and, if set, to $BTH_RESULTS_PATH (the reliable outcome-eval path)."""
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w") as f:
    json.dump(result, f, indent=2, default=str)
  results_path = os.environ.get("BTH_RESULTS_PATH")
  if results_path:
    with Path(results_path).open("w") as f:
      json.dump(result, f, indent=2, default=str)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", required=True, type=Path, help="Path to write the result JSON.")
  args = parser.parse_args(argv)

  result = run_smoke()
  _write_result(args.out, result)
  return 0


if __name__ == "__main__":
  sys.exit(main())
