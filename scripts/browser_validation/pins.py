"""Reference-checkpoint pinning for the browser-validation harnesses (Phase 1, T4, AC-4).

Two things live here:

- :func:`assert_reference_pinned` -- every harness script that loads a reference
  ``.pt`` or an aminx-packaged ``.eqx.zst`` checkpoint calls this before trusting the
  bytes. A silently regenerated or corrupted checkpoint must fail the run closed
  (``SystemExit(3)``), not skew a parity number.
- ``python pins.py --out <path>`` -- (re)generates ``reference_pins.json`` from the
  live clones: LigandMPNN HEAD, the shallow ProteinMPNN clone HEAD, the ``3870631``
  resolution in both, and the sha256 of every reference ``.pt`` and every
  ``.eqx.zst`` aminx resolves for the in-scope checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

#: Live HEAD of $REFERENCE_PATH (dauparas/LigandMPNN), asserted by the T4 gate.
LIGANDMPNN_PIN = "26ec57ac976ade5379920dbd43c7f97a91cf82de"

#: Bathos commit pinned for both machines (T4 common context).
BATHOS_COMMIT = "84be544ecb45734f46e43d22f351a54d6edd6ae5"

#: Revision the spec asks every harness to try to resolve, in both clones.
CANDIDATE_REV = "3870631"

#: aminx checkpoint ids whose packaged ``.eqx.zst`` bytes this spec pins (AC-4):
#: proteinmpnn_v_48_{002,020}, solublempnn_v_48_020, both membrane checkpoints,
#: ligandmpnn_v_32_010_25 and ligandmpnn_sc_v_32_002_16.
AMINX_CHECKPOINTS = (
  "proteinmpnn_v_48_002",
  "proteinmpnn_v_48_020",
  "solublempnn_v_48_020",
  "global_label_membrane_mpnn_v_48_020",
  "per_residue_label_membrane_mpnn_v_48_020",
  "ligandmpnn_v_32_010_25",
  "ligandmpnn_sc_v_32_002_16",
)

DEFAULT_REFERENCE_PATH = "/home/marielle/projects/aminx/reference_ligandmpnn_clone"


def sha256_file(path: Path) -> str:
  """Return the sha256 hex digest of a file's bytes."""
  digest = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def assert_reference_pinned(path: str | Path, expected_sha: str) -> None:
  """Exit the process with ``SystemExit(3)`` unless ``path`` hashes to ``expected_sha``.

  Every harness that loads a reference checkpoint calls this immediately after
  resolving the file path and before feeding the bytes into a model -- the failure
  must be visible at the load site, not surface later as an unexplained numeric
  mismatch downstream.
  """
  candidate = Path(path)
  if not candidate.is_file():
    print(f"reference pin check: {candidate} does not exist", file=sys.stderr)
    raise SystemExit(3)
  actual = sha256_file(candidate)
  if actual != expected_sha:
    print(
      f"reference pin check FAILED for {candidate}: expected sha256={expected_sha}, got {actual}",
      file=sys.stderr,
    )
    raise SystemExit(3)


def _git_head(repo: Path) -> str:
  result = subprocess.run(
    ["git", "-C", str(repo), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
  )
  return result.stdout.strip()


def _resolve_candidate_rev(repo: Path, rev: str) -> str:
  """Return the resolved commit sha for ``rev`` in ``repo``, or the literal ``"not found"``.

  ``git rev-parse --verify`` fails (non-zero exit) both when the object genuinely
  does not exist and when a shallow clone's history does not reach it -- either way
  this repository cannot vouch for it, so both cases record the same sentinel.
  """
  result = subprocess.run(
    ["git", "-C", str(repo), "rev-parse", "--verify", f"{rev}^{{commit}}"],
    capture_output=True,
    text=True,
  )
  if result.returncode != 0:
    return "not found"
  return result.stdout.strip()


def _pt_shas(directory: Path, prefix: str) -> dict[str, str]:
  return {f"{prefix}/{pt.name}": sha256_file(pt) for pt in sorted(directory.glob("*.pt"))}


def _eqx_zst_shas() -> dict[str, str]:
  from aminx.io.weights import weight_provenance

  return {
    f"eqx_zst/{checkpoint_id}": weight_provenance(checkpoint_id).sha256
    for checkpoint_id in AMINX_CHECKPOINTS
  }


def build_pins(reference_path: Path, proteinmpnn_path: Path) -> dict[str, object]:
  """Assemble the reference_pins.json payload for the fixer's Phase-1 steps (1-4).

  Deliberately omits ``node_provenance`` and any titanix-MEASURED field -- those are
  written by the orchestrator after the provenance smoke actually runs there (T4
  step 5). ``titanix`` here carries only the static, pre-agreed configuration.
  """
  weights_sha256: dict[str, str] = {}
  weights_sha256.update(_pt_shas(reference_path / "model_params", "ligandmpnn_pt"))
  weights_sha256.update(_pt_shas(proteinmpnn_path / "vanilla_model_weights", "proteinmpnn_pt"))
  weights_sha256.update(_eqx_zst_shas())

  return {
    "ligandmpnn_commit": _git_head(reference_path),
    "proteinmpnn_commit": _git_head(proteinmpnn_path),
    "r3870631": {
      "ligandmpnn": _resolve_candidate_rev(reference_path, CANDIDATE_REV),
      "proteinmpnn": _resolve_candidate_rev(proteinmpnn_path, CANDIDATE_REV),
    },
    "weights_sha256": weights_sha256,
    "titanix": {
      "host": "titanix",
      "uv_path": "/home/solab/.local/bin/uv",
      "bth_path": "/home/solab/.local/bin/bth",
      "bathos_commit": BATHOS_COMMIT,
      "catalog_dir": "/home/solab/bv/catalog-aminx/.bth/catalog",
      "reference_clone_paths": {
        "ligandmpnn": "/home/solab/bv/ref/LigandMPNN",
        "proteinmpnn": "/home/solab/bv/ref/ProteinMPNN",
      },
      "memory_cap": "systemd-unit",
    },
  }


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--reference-path", default=DEFAULT_REFERENCE_PATH)
  parser.add_argument(
    "--proteinmpnn-path",
    default=str(Path(__file__).resolve().parents[2] / ".cache" / "reference" / "ProteinMPNN"),
  )
  parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "reference_pins.json"))
  args = parser.parse_args(argv)

  reference_path = Path(args.reference_path)
  proteinmpnn_path = Path(args.proteinmpnn_path)
  actual_ligandmpnn_head = _git_head(reference_path)
  if actual_ligandmpnn_head != LIGANDMPNN_PIN:
    print(
      f"live LigandMPNN HEAD {actual_ligandmpnn_head!r} at {reference_path} differs from the "
      f"pinned {LIGANDMPNN_PIN!r}; refusing to write a pins file that does not describe the "
      f"clone it was generated from",
      file=sys.stderr,
    )
    raise SystemExit(3)

  pins = build_pins(reference_path, proteinmpnn_path)
  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n")
  print(f"wrote {out_path}")


if __name__ == "__main__":
  main()
