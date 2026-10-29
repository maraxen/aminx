"""Utilities for locating and importing the LigandMPNN reference repository."""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

log = logging.getLogger(__name__)

# LigandMPNN reference commit pinned for reproducibility.
_REFERENCE_REPO_URL = "https://github.com/dauparas/LigandMPNN.git"
_REFERENCE_COMMIT = "26ec57ac976ade5379920dbd43c7f97a91cf82de"

# Module-level cache: memoize the acquired path so we clone at most once per process.
_cached_reference_path: Path | None = None

PARITY_REPOS_ENV = "AMINX_PARITY_REPOS"


def project_root() -> Path:
  """Return repository root directory."""
  return Path(__file__).resolve().parents[2]


def parity_repo_roots() -> tuple[Path, ...]:
  """Return candidate root directories for external parity repositories.

  Reads ``$AMINX_PARITY_REPOS`` (``os.pathsep``-separated list of directories)
  first, then falls back to ``~/repos``.
  """
  roots: list[Path] = []
  raw = os.environ.get(PARITY_REPOS_ENV, "")
  if raw.strip():
    for entry in raw.split(os.pathsep):
      stripped = entry.strip()
      if stripped:
        roots.append(Path(stripped).expanduser())
  default_root = Path.home() / "repos"
  if default_root not in roots:
    roots.append(default_root)
  return tuple(roots)


def find_parity_repo_file(rel_path: str | Path) -> Path | None:
  """Locate ``rel_path`` under :func:`parity_repo_roots`, or return ``None``."""
  for root in parity_repo_roots():
    candidate = root / rel_path
    if candidate.is_file():
      return candidate
  return None


def _acquire_ephemeral_reference() -> Path:
  """Clone LigandMPNN at the pinned commit into a temp directory.

  The directory is registered for cleanup via ``atexit`` so it is removed
  when the test process exits.  Returns the path to the clone root.

  Raises ``SystemExit`` (via ``pytest.skip``) on any failure so callers can
  remain unconditional.
  """
  # Check that git is available before attempting a clone.
  try:
    subprocess.run(["git", "--version"], capture_output=True, check=True)
  except (FileNotFoundError, subprocess.CalledProcessError):
    pytest.skip(
      "git not found — cannot acquire LigandMPNN reference repo ephemerally.",
      allow_module_level=True,
    )

  tmpdir = tempfile.mkdtemp(prefix="ligandmpnn_ref_")
  atexit.register(shutil.rmtree, tmpdir, True)

  try:
    log.info("Cloning LigandMPNN reference repo to %s …", tmpdir)
    # Attempt shallow clone pinned to the specific commit.
    clone_result = subprocess.run(
      ["git", "clone", "--depth", "1", _REFERENCE_REPO_URL, tmpdir],
      capture_output=True,
    )
    if clone_result.returncode != 0:
      # Shallow clone of an arbitrary commit requires fetch; fall back to full clone.
      subprocess.run(["git", "clone", _REFERENCE_REPO_URL, tmpdir], check=True, capture_output=True)

    subprocess.run(
      ["git", "-C", tmpdir, "fetch", "--depth", "1", "origin", _REFERENCE_COMMIT],
      capture_output=True,
    )
    subprocess.run(
      ["git", "-C", tmpdir, "checkout", _REFERENCE_COMMIT],
      check=True,
      capture_output=True,
    )
  except subprocess.CalledProcessError as exc:
    pytest.skip(
      f"Failed to acquire LigandMPNN reference repo at commit {_REFERENCE_COMMIT}: {exc}",
      allow_module_level=True,
    )

  return Path(tmpdir)


def require_reference_path(*, allow_ephemeral: bool = False) -> Path:
  """Return LigandMPNN reference repo path or skip if unavailable.

  Resolution order:
  1. If ``$REFERENCE_PATH`` is set and is a directory, use it as-is
     (local-cache override; avoids any network access).
  2. If ``allow_ephemeral=True``: acquire ephemerally by ``git clone`` at the
     pinned commit into a temporary directory cleaned up on process exit.
     The result is memoized so the clone runs at most once per process.
  3. On any failure (git missing, network error, etc.) call
     ``pytest.skip(..., allow_module_level=True)`` — never hard-fail.

  Parameters
  ----------
  allow_ephemeral:
      When True, fall back to ephemeral git clone if ``$REFERENCE_PATH`` is
      not set.  Default False preserves the original behaviour (skip when no
      env var is set) for tests that call this function directly and rely on
      the reference checkout having pre-downloaded weights.
  """
  global _cached_reference_path

  env_value = os.environ.get("REFERENCE_PATH")
  if env_value:
    candidate = Path(env_value)
    if candidate.is_dir():
      return candidate
    pytest.skip(
      f"REFERENCE_PATH={env_value!r} is set but is not a directory.",
      allow_module_level=True,
    )

  if not allow_ephemeral:
    pytest.skip(
      "LigandMPNN reference repo not found. Set $REFERENCE_PATH to a local checkout.",
      allow_module_level=True,
    )

  # Memoized ephemeral path.
  if _cached_reference_path is not None and _cached_reference_path.is_dir():
    return _cached_reference_path

  _cached_reference_path = _acquire_ephemeral_reference()
  return _cached_reference_path


_WEIGHTS_BASE_URL = "https://files.ipd.uw.edu/pub/ligandmpnn/"


def ensure_reference_weights(reference_root: Path, filenames: list[str]) -> None:
  """Ensure that each weight file exists under ``<reference_root>/model_params/``.

  For any ``.pt`` file that is absent, download it from the IPD UW server.
  On download failure, call ``pytest.skip`` so the test is skipped rather than
  failed.

  When ``$REFERENCE_PATH`` points to a local checkout that already contains
  the files this is a no-op.
  """
  model_params_dir = reference_root / "model_params"
  model_params_dir.mkdir(parents=True, exist_ok=True)

  for filename in filenames:
    dest = model_params_dir / filename
    if dest.exists():
      continue
    url = _WEIGHTS_BASE_URL + filename
    log.info("Downloading %s from %s …", filename, url)
    try:
      urllib.request.urlretrieve(url, dest)
    except Exception as exc:  # noqa: BLE001
      pytest.skip(
        f"Failed to download reference weight {filename!r} from {url}: {exc}",
        allow_module_level=True,
      )


def prepend_reference_to_syspath() -> Path:
  """Add reference repo path to sys.path and return it.

  Ephemeral acquisition is enabled so this function can be used by
  ``require_heavy_parity_prereqs`` even when ``$REFERENCE_PATH`` is absent.
  """
  reference_path = require_reference_path(allow_ephemeral=True)
  ref_str = str(reference_path)
  if ref_str not in sys.path:
    sys.path.insert(0, ref_str)
  return reference_path


def require_heavy_parity_prereqs(
  *,
  python_modules: list[str] | None = None,
  reference_rel_paths: list[str] | None = None,
  converted_rel_paths: list[str] | None = None,
) -> tuple[Path, Path]:
  """Validate heavy parity prerequisites and return ``(reference_root, project_root)``.

  For any entry in ``reference_rel_paths`` that looks like
  ``model_params/*.pt``, the weight file is downloaded automatically via
  :func:`ensure_reference_weights` when it is absent from the resolved
  reference directory.
  """
  if find_spec("torch") is None:
    pytest.skip("Heavy parity requires torch in the active environment.", allow_module_level=True)
  for module_name in python_modules or []:
    if find_spec(module_name) is None:
      pytest.skip(
        f"Heavy parity requires optional dependency '{module_name}'.",
        allow_module_level=True,
      )

  reference_root = prepend_reference_to_syspath()
  repo_root = project_root()

  # Download any missing model_params/*.pt weights.
  weight_filenames = [
    Path(rel).name
    for rel in (reference_rel_paths or [])
    if rel.startswith("model_params/") and rel.endswith(".pt")
  ]
  if weight_filenames:
    ensure_reference_weights(reference_root, weight_filenames)

  for rel_path in reference_rel_paths or []:
    path = reference_root / rel_path
    if not path.exists():
      pytest.skip(
        f"Heavy parity missing reference asset: {path}",
        allow_module_level=True,
      )

  for rel_path in converted_rel_paths or []:
    path = repo_root / rel_path
    if not path.exists():
      pytest.skip(
        f"Heavy parity missing converted asset: {path}",
        allow_module_level=True,
      )

  return reference_root, repo_root


def _ensure_reference_numpy_aliases() -> None:
  """Patch NumPy aliases expected by older reference dependencies."""
  aliases: dict[str, type[object]] = {
    "bool": bool,
    "int": int,
    "object": object,
  }
  for name, target in aliases.items():
    if name not in np.__dict__:
      setattr(np, name, target)


def load_reference_function(module_filename: str, function_name: str) -> Any:  # noqa: ANN401
  """Lift a single function out of the reference SOURCE without importing the module.

  ``import data_utils`` pulls in ``prody``, which is not a dependency of this repo, so the
  reference's own aggregators are unreachable through a normal import even when the checkout
  is present with weights. Extracting the function by AST keeps them reachable.

  The point is that the reference formula is never RETYPED here. A transcribed copy of
  ``get_score`` would be a second definition that can silently stop matching the thing it is
  supposed to be a reference for -- which is the exact failure a parity test exists to catch.
  Extracted this way it tracks upstream edits automatically, and a rename or deletion
  upstream surfaces as a skip rather than as a stale copy that still passes.

  Only builtins plus ``torch`` are provided as globals, so a function with other
  module-level dependencies fails loudly here rather than silently picking up a substitute.

  Args:
    module_filename: File within the reference checkout, e.g. ``"data_utils.py"``.
    function_name: Top-level function to extract, e.g. ``"get_score"``.

  Returns:
    The compiled reference function.
  """
  import ast  # noqa: PLC0415

  torch = pytest.importorskip("torch")
  source_path = require_reference_path() / module_filename
  if not source_path.is_file():
    pytest.skip(f"Reference file not found: {source_path}", allow_module_level=True)

  tree = ast.parse(source_path.read_text())
  for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == function_name:
      namespace: dict[str, Any] = {"torch": torch}
      exec(  # noqa: S102
        compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"),
        namespace,
      )
      return namespace[function_name]

  pytest.skip(
    f"Reference function '{function_name}' not found in {source_path} — "
    "it may have been renamed upstream",
    allow_module_level=True,
  )
  raise AssertionError("unreachable")


def import_reference_module(module_name: str) -> ModuleType:
  """Import a module from the reference checkout or skip if it cannot import."""
  prepend_reference_to_syspath()
  _ensure_reference_numpy_aliases()
  try:
    return import_module(module_name)
  except (ImportError, ModuleNotFoundError) as error:
    pytest.skip(
      f"Unable to import reference module '{module_name}': {error}",
      allow_module_level=True,
    )
