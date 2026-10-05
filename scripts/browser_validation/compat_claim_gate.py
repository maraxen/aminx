"""Compatibility-claim gate (T8 step 5, AC-24 / AC-CG).

A sentence that names a browser or route family and a support/compat verb is allowed
only when it also names an evidence file
`outputs/browser_validation/layer_c/evidence/<route>__<browser>.json` that exists and
has `cross_origin_isolated == true`.

Exit 1 when any scanned file contains a disallowed sentence, else 0.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_BROWSER_RE = re.compile(
  r"\b(firefox|webkit|safari|edge|webgpu|chrom(e|ium)|ios|android)\b",
  re.IGNORECASE,
)
_CLAIM_RE = re.compile(
  r"\b(support(s|ed)?|compatib\w*|works?|runs?|validated|passes)\b",
  re.IGNORECASE,
)
_EVIDENCE_RE = re.compile(
  r"outputs/browser_validation/layer_c/evidence/"
  r"([A-Za-z0-9][A-Za-z0-9._-]*)__([A-Za-z0-9][A-Za-z0-9._-]*)\.json",
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
  chunks = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
  return chunks or ([text.strip()] if text.strip() else [])


def _evidence_exists(path: Path) -> bool:
  """Evidence-existence check: the named file must be on disk."""
  return path.is_file()


def _named_evidence_ok(path: Path) -> bool:
  if not _evidence_exists(path):
    return False
  payload = json.loads(path.read_text(encoding="utf-8"))
  return payload.get("cross_origin_isolated") is True


def sentence_violation(sentence: str, root: Path) -> str | None:
  """Return a short reason when `sentence` is a disallowed compatibility claim."""
  if _BROWSER_RE.search(sentence) is None or _CLAIM_RE.search(sentence) is None:
    return None
  named = _EVIDENCE_RE.findall(sentence)
  if not named:
    return "browser/compat sentence names no evidence file"
  for route, browser in named:
    path = (
      root / "outputs" / "browser_validation" / "layer_c" / "evidence" / f"{route}__{browser}.json"
    )
    if not _named_evidence_ok(path):
      return f"evidence missing or not cross-origin isolated: {path}"
  return None


def violations_in_text(text: str, root: Path) -> list[str]:
  found: list[str] = []
  for sentence in _sentences(text):
    reason = sentence_violation(sentence, root)
    if reason is not None:
      found.append(f"{reason}: {sentence}")
  return found


def violations_in_files(paths: list[Path], root: Path) -> list[str]:
  found: list[str] = []
  for path in paths:
    if not path.is_file():
      found.append(f"path is not a file: {path}")
      continue
    text = path.read_text(encoding="utf-8")
    for item in violations_in_text(text, root):
      found.append(f"{path}: {item}")
  return found


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--paths", nargs="+", type=Path, required=True)
  parser.add_argument(
    "--root",
    type=Path,
    default=None,
    help="Directory evidence paths are resolved against (default: cwd).",
  )
  args = parser.parse_args(argv)
  root = (args.root or Path.cwd()).resolve()
  found = violations_in_files(args.paths, root)
  if found:
    for item in found:
      print(item, file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
