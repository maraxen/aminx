"""Stable identity for a parity oracle's weight file.

``branch_controls.json`` used to key its ``weights`` map by the absolute path of
the checkpoint, e.g. ``/home/solab/repos/LASErMPNN/model_weights/....pt``. Step
1c of the redsox gate then looks that key up in a ``checkpoint_registry.json``
with an EXACT string match (tests/knob_gate/_coverage.py, ``registry_sha256``).

An absolute path cannot be the key. It encodes one machine's ``$HOME``, so a
registry listing it is correct on exactly one checkout and wrong everywhere
else -- and committing it bakes a single user's home directory into a shared
gate. Keying on something stable is what makes the registry portable.

The key is the path from the vendored upstream repository downwards:

    /home/solab/repos/LASErMPNN/model_weights/laser_weights_0p1A.pt
    ->          LASErMPNN/model_weights/laser_weights_0p1A.pt

That is independent of where the upstream was cloned, unambiguous across the
two families (a bare basename would not be -- two upstreams can ship the same
file name), and still readable in a diff.
"""

from __future__ import annotations

from pathlib import Path

# Vendored upstream repository directory names, as cloned under ~/repos. The key
# is anchored at whichever of these appears in the path; anything else is keyed
# by basename, which is the honest fallback for a file we do not place.
_UPSTREAM_ROOTS: tuple[str, ...] = ("LASErMPNN", "PottsMPNN", "ProtonPottsMPNN")


def stable_artifact_key(path: str | Path) -> str:
  """Registry key for ``path``: ``<upstream repo>/<path within it>``.

  Falls back to the bare file name when the path is not inside a known vendored
  upstream, so an aminx-side checkpoint still gets a machine-independent key.
  """
  resolved = Path(path).resolve()
  parts = resolved.parts
  for marker in _UPSTREAM_ROOTS:
    if marker in parts:
      # rindex: a path may legitimately contain the marker twice (a nested
      # checkout, a copy under a scratch dir); the LAST one is the real root.
      index = len(parts) - 1 - parts[::-1].index(marker)
      return "/".join(parts[index:])
  return resolved.name
