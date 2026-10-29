"""Family checkpoints (PottsMPNN, ProtonPottsMPNN, LASErMPNN) resolve by id, pinned and verified.

The drivers used to need ``model_local_path`` or a local registry. These tests hold the properties of
the hub route that replaced that for the published checkpoints: the lookup honours the same order as
the MPNN weights, an explicit weights directory fails closed, a changed byte in the artifact or its
manifest is refused (with a negative control), and nothing here touches the network.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import pytest

from aminx.host.prep import _resolve_driver_checkpoint
from aminx.io import weights as weights_mod
from aminx.io.weights import (
  HF_REVISION,
  REVISION_ENV,
  WEIGHTS_DIR_ENV,
  FamilyCheckpoint,
  _resolve_weight_path,
  family_checkpoints,
  resolve_family_checkpoint,
)

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = b"pretend equinox leaves"
MANIFEST = json.dumps({"checkpoint_id": "pottsmpnn_unit", "build_hydrogens": True}).encode()


def _sha(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _entry(family: str = "pottsmpnn") -> FamilyCheckpoint:
  return FamilyCheckpoint(
    checkpoint_id="pottsmpnn_unit",
    family=family,
    filename="pottsmpnn_unit.eqx",
    sha256=_sha(ARTIFACT),
    manifest_filename="pottsmpnn_unit.json",
    manifest_sha256=_sha(MANIFEST),
  )


@pytest.fixture
def table(monkeypatch: pytest.MonkeyPatch) -> FamilyCheckpoint:
  entry = _entry()
  monkeypatch.setattr(weights_mod, "family_checkpoints", lambda: {entry.checkpoint_id: entry})
  monkeypatch.delenv(WEIGHTS_DIR_ENV, raising=False)
  monkeypatch.delenv(REVISION_ENV, raising=False)
  return entry


def _write(directory: Path, artifact: bytes = ARTIFACT, manifest: bytes = MANIFEST) -> None:
  directory.mkdir(parents=True, exist_ok=True)
  (directory / "pottsmpnn_unit.eqx").write_bytes(artifact)
  (directory / "pottsmpnn_unit.json").write_bytes(manifest)


def test_explicit_dir_resolves_and_verifies(
  table: FamilyCheckpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  _write(tmp_path)
  monkeypatch.setenv(WEIGHTS_DIR_ENV, str(tmp_path))

  record = resolve_family_checkpoint(table.checkpoint_id, family="pottsmpnn")

  assert record.source == "explicit_dir"
  assert Path(record.path) == tmp_path / "pottsmpnn_unit.eqx"
  assert record.sha256 == _sha(ARTIFACT)
  assert record.hub_repo_id is None


def test_family_names_are_not_given_the_mpnn_suffix(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  """``.eqx`` filenames must stay exact; the MPNN route would turn ``x.eqx`` into ``x.eqx.eqx.zst``."""
  assert weights_mod.normalise_checkpoint_filename("x.eqx") == "x.eqx.eqx.zst"
  (tmp_path / "x.eqx").write_bytes(b"exact")
  monkeypatch.setenv(WEIGHTS_DIR_ENV, str(tmp_path))

  source, path = _resolve_weight_path("x.eqx", normalise=False)

  assert (source, Path(path)) == ("explicit_dir", tmp_path / "x.eqx")


def test_explicit_dir_fails_closed(
  table: FamilyCheckpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A file missing from an authoritative dir raises; it must not fall through to the Hub."""
  monkeypatch.setenv(WEIGHTS_DIR_ENV, str(tmp_path))
  called: list[str] = []
  monkeypatch.setattr(weights_mod, "hf_hub_download", lambda **kw: called.append(kw["filename"]))

  with pytest.raises(FileNotFoundError, match="authoritative"):
    resolve_family_checkpoint(table.checkpoint_id)

  assert called == []


def test_corrupted_artifact_is_refused(
  table: FamilyCheckpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Negative control: one flipped byte must fail; the clean file in the sibling test must pass."""
  flipped = bytearray(ARTIFACT)
  flipped[3] ^= 0xFF
  _write(tmp_path, artifact=bytes(flipped))
  monkeypatch.setenv(WEIGHTS_DIR_ENV, str(tmp_path))

  with pytest.raises(ValueError, match="pins"):
    resolve_family_checkpoint(table.checkpoint_id)


def test_corrupted_manifest_is_refused(
  table: FamilyCheckpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The manifest carries static fields the artifact lacks, so it is pinned as well."""
  _write(tmp_path, manifest=MANIFEST.replace(b"true", b"false"))
  monkeypatch.setenv(WEIGHTS_DIR_ENV, str(tmp_path))

  with pytest.raises(ValueError, match="manifest"):
    resolve_family_checkpoint(table.checkpoint_id)


def test_unknown_id_and_family_mismatch_raise(
  table: FamilyCheckpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  _write(tmp_path)
  monkeypatch.setenv(WEIGHTS_DIR_ENV, str(tmp_path))

  with pytest.raises(ValueError, match="unknown family checkpoint"):
    resolve_family_checkpoint("pottsmpnn_nope")
  with pytest.raises(ValueError, match="belongs to family"):
    resolve_family_checkpoint(table.checkpoint_id, family="lasermpnn")


def test_hub_route_is_pinned_and_describes_the_cache_path(
  table: FamilyCheckpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  """With no directory set, both files come from the Hub at ``HF_REVISION`` (download stubbed)."""
  snapshot = tmp_path / "hub" / "snapshots" / HF_REVISION
  _write(snapshot)
  requests: list[tuple[str, str]] = []

  def fake_download(**kw: str) -> str:
    assert kw["repo_id"] == weights_mod.HF_REPO_ID
    requests.append((kw["filename"], kw["revision"]))
    return str(snapshot / kw["filename"])

  monkeypatch.setattr(weights_mod, "hf_hub_download", fake_download)

  record = resolve_family_checkpoint(table.checkpoint_id)

  assert record.source == "hub"
  assert record.hub_revision == HF_REVISION
  assert record.hub_repo_id == weights_mod.HF_REPO_ID
  assert requests == [
    ("pottsmpnn_unit.eqx", HF_REVISION),
    ("pottsmpnn_unit.json", HF_REVISION),
  ]


def test_driver_resolver_prefers_registry_then_table(
  table: FamilyCheckpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  _write(tmp_path)
  monkeypatch.setenv(WEIGHTS_DIR_ENV, str(tmp_path))
  # prep imported the names at import time, so patch its references too
  from aminx.host import prep  # noqa: PLC0415

  monkeypatch.setattr(prep, "family_checkpoints", lambda: {table.checkpoint_id: table})
  hit = SimpleNamespace(checkpoint_registry_path=None, checkpoint_id=table.checkpoint_id)
  miss = SimpleNamespace(checkpoint_registry_path=None, checkpoint_id="proteinmpnn_v_48_020")

  assert _resolve_driver_checkpoint(hit, "pottsmpnn") == str(tmp_path / "pottsmpnn_unit.eqx")
  assert _resolve_driver_checkpoint(miss, "pottsmpnn") is None


def test_laser_artifact_keeps_build_hydrogens(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
  """``build_hydrogens`` is static, so ``.eqx`` drops it; the manifest must restore it.

  Every upstream LASErMPNN checkpoint sets it True. Before this guard ``_load_eqx`` rebuilt a default
  skeleton and silently loaded such an artifact with it False.
  """
  from aminx.families.laser_mpnn.driver import LASErMPNN, _load_eqx  # noqa: PLC0415

  model = LASErMPNN(key=jax.random.PRNGKey(0), build_hydrogens=True)
  artifact = tmp_path / "lasermpnn_unit.eqx"
  eqx.tree_serialise_leaves(artifact, model)

  with caplog.at_level(logging.WARNING):
    without_manifest = _load_eqx(artifact)
  assert without_manifest.build_hydrogens is False
  assert "no manifest" in caplog.text

  (tmp_path / "lasermpnn_unit.json").write_text(json.dumps({"build_hydrogens": True}), encoding="utf-8")
  assert _load_eqx(artifact).build_hydrogens is True


def test_shipped_table_is_well_formed_and_licensed() -> None:
  """Every published id is well-formed and named in the licence record and the notice file."""
  raw = json.loads(files("aminx.io").joinpath("family_weights.json").read_text(encoding="utf-8"))
  record = next(
    (REPO / ".praxia/docs/decisions").glob("*redistribution-of-pottsmpnn-lasermpnn-and-protonpottsmpnn*.md"),
  ).read_text(encoding="utf-8")
  variant_spec = next(
    (REPO / ".praxia/docs/specs").glob("*mpnn-variant-weights-scope-spec*.md"),
  ).read_text(encoding="utf-8")
  notice = (REPO / "NOTICE-weights.md").read_text(encoding="utf-8")
  prefixes: dict[str, tuple[str, ...]] = {
    "pottsmpnn": ("pottsmpnn_",),
    "lasermpnn": ("lasermpnn_",),
    "protonpottsmpnn": ("protonpottsmpnn_",),
    "caliby": ("caliby", "soluble_caliby"),
    "fampnn": ("fampnn_",),
    "thermompnn": ("thermo_mpnn_",),
  }
  for row in raw["checkpoints"]:
    cid = row["checkpoint_id"]
    assert row["family"] in prefixes
    assert cid.startswith(prefixes[row["family"]])
    assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
    if row["family"] in ("thermompnn", "caliby", "fampnn"):
      # Compressed artifacts shipped with a ``<id>.manifest.json`` (HF_REVISION 5ff5cf16 added them).
      assert row["filename"] == f"{cid}.eqx.zst"
      assert row["manifest_filename"] == f"{cid}.manifest.json"
    else:
      assert row["filename"] == f"{cid}.eqx"
      assert row["manifest_filename"] == f"{cid}.json"
    assert re.fullmatch(r"[0-9a-f]{64}", row["manifest_sha256"])
    if row["family"] == "thermompnn":
      assert cid in variant_spec, f"{cid} has no entry in the variant weights scope spec"
    else:
      assert cid in record, f"{cid} has no entry in the licence decision record"
    assert cid in notice, f"{cid} is not named in NOTICE-weights.md"
  assert set(family_checkpoints()) == {row["checkpoint_id"] for row in raw["checkpoints"]}
