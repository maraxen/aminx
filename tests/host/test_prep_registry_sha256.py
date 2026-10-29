"""Registry entries for driver-backed families must pin their artifact with a sha256.

Covers the families the registry resolver used to leave unpinned (caliby, fampnn) and the
error message, which names the checkpoint id the resolver actually looked up.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aminx.host.prep import _resolve_local_checkpoint_from_registry


def _write_registry(tmp_path: Path, entries: list[dict[str, object]]) -> Path:
  artifact = tmp_path / "weights.eqx"
  artifact.write_bytes(b"weights")
  reg_path = tmp_path / "registry.json"
  reg_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
  return reg_path


def _spec(reg_path: Path, model_family: str, checkpoint_id: str | None) -> SimpleNamespace:
  return SimpleNamespace(
    checkpoint_registry_path=reg_path,
    checkpoint_id=checkpoint_id,
    model_family=model_family,
  )


@pytest.mark.parametrize(
  ("model_family", "checkpoint_id"),
  [
    ("caliby", "caliby"),
    ("caliby", "caliby_distill"),
    ("caliby", "caliby_packer_010"),
    ("caliby", "soluble_caliby_distill"),
    ("fampnn", "fampnn_0_3"),
    ("fampnn", "fampnn_0_3_cath"),
  ],
)
def test_driver_family_entry_without_sha256_is_refused(
  tmp_path: Path, model_family: str, checkpoint_id: str,
) -> None:
  reg_path = _write_registry(
    tmp_path,
    [{"model_family": model_family, "checkpoint_id": checkpoint_id, "artifact_path": "weights.eqx"}],
  )

  with pytest.raises(ValueError, match="must include sha256"):
    _resolve_local_checkpoint_from_registry(_spec(reg_path, model_family, checkpoint_id))


def test_refusal_names_the_checkpoint_id_that_was_looked_up(tmp_path: Path) -> None:
  """The spec leaves checkpoint_id unset; the message must name the id passed to the resolver."""
  reg_path = _write_registry(
    tmp_path,
    [{"model_family": "fampnn", "checkpoint_id": "fampnn_0_0", "artifact_path": "weights.eqx"}],
  )

  with pytest.raises(ValueError, match="checkpoint_id='fampnn_0_0' must include sha256"):
    _resolve_local_checkpoint_from_registry(_spec(reg_path, "fampnn", None), "fampnn_0_0")


def test_driver_family_entry_with_matching_sha256_resolves(tmp_path: Path) -> None:
  reg_path = _write_registry(
    tmp_path,
    [
      {
        "model_family": "caliby",
        "checkpoint_id": "caliby_distill",
        "artifact_path": "weights.eqx",
        "sha256": hashlib.sha256(b"weights").hexdigest(),
      },
    ],
  )

  resolved = _resolve_local_checkpoint_from_registry(_spec(reg_path, "caliby", "caliby_distill"))

  assert resolved == str((tmp_path / "weights.eqx").resolve())


def test_non_driver_family_entry_without_sha256_is_still_accepted(tmp_path: Path) -> None:
  reg_path = _write_registry(
    tmp_path,
    [{"model_family": "proteinmpnn", "checkpoint_id": "proteinmpnn_v_48_020", "artifact_path": "weights.eqx"}],
  )

  resolved = _resolve_local_checkpoint_from_registry(_spec(reg_path, "proteinmpnn", "proteinmpnn_v_48_020"))

  assert resolved == str((tmp_path / "weights.eqx").resolve())
