"""Git provenance for one aminx run.

``capture`` records the commit of the installed aminx tree (the directory
that contains this package), not the process working directory. Each call
checks again so a long-lived process does not keep a stale commit.

Tiers, in order:

1. ``cisternal.provenance.capture_git_state`` when cisternal imports.
2. A builtin ``git`` subprocess (``provenance_source`` ``builtin``).
3. Unknown (``sha``, ``dirty``, and ``branch`` are ``None``).

A sha is returned only when it is 40 lowercase hex characters. Failures are
logged at ``DEBUG`` and never raised.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_SHA = re.compile(r"^[0-9a-f]{40}$")
_GIT_TIMEOUT_S = 10


def capture(root: Path | None = None) -> dict[str, object]:
  """Capture git provenance for the aminx source tree.

  Parameters
  ----------
  root : Path or None, optional
    Directory to inspect. Defaults to the directory of the installed
    ``aminx`` package (``Path(aminx.__file__).parent``), computed on every
    call. When that default lies under ``site-packages`` or
    ``dist-packages`` it is a built install, not a checkout, and the result
    is ``provenance_source`` ``"not-a-checkout"`` with no sha.

  Returns
  -------
  dict of str to object
    Keys ``sha`` (40-hex str or None), ``dirty`` (bool or None), ``branch``
    (str or None), and ``provenance_source`` (str).
  """
  resolved = _package_root() if root is None else Path(root)
  if root is None and _INSTALLED_DIRS.intersection(resolved.parts):
    # A built install (wheel, uvx) is not a checkout. Any repository around it
    # belongs to the enclosing project, so its commit would be misattributed.
    logger.debug("provenance skipped for installed package at %s", resolved)
    return {
      "sha": None,
      "dirty": None,
      "branch": None,
      "provenance_source": "not-a-checkout",
    }
  state = _from_cisternal(resolved)
  if state is not None:
    return state
  state = _from_git(resolved)
  if state is not None:
    return state
  logger.debug("provenance unknown for %s", resolved)
  return {
    "sha": None,
    "dirty": None,
    "branch": None,
    "provenance_source": "unknown",
  }


_INSTALLED_DIRS = frozenset({"site-packages", "dist-packages"})


def _package_root() -> Path:
  """Return the directory containing the installed ``aminx`` package."""
  import aminx  # noqa: PLC0415

  return Path(aminx.__file__).parent


def _from_cisternal(root: Path) -> dict[str, object] | None:
  """Return a cisternal capture, or ``None`` when that tier cannot run."""
  try:
    from cisternal.provenance import capture_git_state  # noqa: PLC0415
  except ImportError as exc:
    logger.debug("cisternal provenance import failed: %s", exc)
    return None
  try:
    state = capture_git_state(cwd=root)
  except Exception as exc:  # noqa: BLE001 -- tier must not raise
    logger.debug("cisternal provenance capture failed: %s", exc)
    return None
  raw_hash = getattr(state, "hash", None)
  sha = raw_hash if isinstance(raw_hash, str) and _SHA.fullmatch(raw_hash) else None
  dirty = getattr(state, "dirty", None)
  branch = getattr(state, "branch", None)
  source = getattr(state, "provenance_source", None)
  if not isinstance(dirty, bool):
    dirty = None
  if not isinstance(branch, str):
    branch = None
  if not isinstance(source, str):
    source = "unknown"
  if sha is None and source == "none":
    # cisternal found nothing to describe (not a repo); let the next tier decide.
    logger.debug("cisternal provenance found no git state for %s", root)
    return None
  return {
    "sha": sha,
    "dirty": dirty,
    "branch": branch,
    "provenance_source": "cisternal:" + source,
  }


def _from_git(root: Path) -> dict[str, object] | None:
  """Return builtin git state, or ``None`` when git cannot describe ``root``."""
  head = _run_git(root, "rev-parse", "HEAD")
  status = _run_git(root, "status", "--porcelain")
  branch_proc = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
  if head is None or status is None or branch_proc is None:
    logger.debug("builtin git provenance failed for %s", root)
    return None
  if head.returncode != 0 or status.returncode != 0 or branch_proc.returncode != 0:
    logger.debug(
      "builtin git provenance failed for %s (head=%s status=%s branch=%s)",
      root,
      head.returncode,
      status.returncode,
      branch_proc.returncode,
    )
    return None
  sha = head.stdout.strip()
  if _SHA.fullmatch(sha) is None:
    logger.debug("builtin git HEAD %r is not a 40-hex sha", sha)
    return None
  branch = branch_proc.stdout.strip()
  return {
    "sha": sha,
    "dirty": bool(status.stdout.strip()),
    "branch": branch or None,
    "provenance_source": "builtin",
  }


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
  """Run ``git -C root`` or return ``None`` on timeout or a missing binary."""
  try:
    return subprocess.run(  # noqa: S603
      ["git", "-C", str(root), *args],  # noqa: S607
      timeout=_GIT_TIMEOUT_S,
      check=False,
      capture_output=True,
      text=True,
    )
  except (OSError, subprocess.SubprocessError):
    return None


__all__ = ["capture"]
