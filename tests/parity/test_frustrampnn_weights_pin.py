"""Tests pinning the FrustraMPNN weights at FRUSTRAMPNN_HF_REVISION in scripts/parity/frustrampnn_parity_v2.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "parity" / "frustrampnn_parity_v2.py"
DECL_PATH = REPO_ROOT / "scripts" / "parity" / "frustrampnn_1ubq.parity.bth.toml"


def _load_vehicle_module():
  spec = importlib.util.spec_from_file_location("_frustrampnn_parity_v2_test", SCRIPT_PATH)
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  try:
    spec.loader.exec_module(mod)
  except Exception:
    sys.modules.pop(spec.name, None)
    raise
  return mod


def test_frustrampnn_script_defines_exact_hf_revision_pin() -> None:
  """Removing or drifting FRUSTRAMPNN_HF_REVISION in frustrampnn_parity_v2.py must fail this test."""
  mod = _load_vehicle_module()
  assert hasattr(mod, "FRUSTRAMPNN_HF_REVISION"), (
    "FRUSTRAMPNN_HF_REVISION missing from scripts/parity/frustrampnn_parity_v2.py"
  )
  pin = getattr(mod, "FRUSTRAMPNN_HF_REVISION")
  assert isinstance(pin, str) and len(pin) == 40
  int(pin, 16)

  decl = tomllib.loads(DECL_PATH.read_text(encoding="utf-8"))
  weights_cfg = decl["implementation"]["weights"]
  assert weights_cfg["hf_revision"] == pin
  assert weights_cfg["hf_repo"] == mod.HF_REPO_ID


def test_frustrampnn_pin_drift_rejected(tmp_path: Path) -> None:
  """Mismatched hf_revision between declaration and script pin fails loudly."""
  mod = _load_vehicle_module()
  bad_decl = tmp_path / "drift.parity.bth.toml"
  bad_decl.write_text(
    DECL_PATH.read_text(encoding="utf-8").replace(
      mod.FRUSTRAMPNN_HF_REVISION,
      "0" * 40,
    ),
    encoding="utf-8",
  )
  with pytest.raises(RuntimeError, match="weights pin drift"):
    mod._verify_weights_provenance(bad_decl, None)


@pytest.mark.requires_weights
def test_frustrampnn_hf_pin_downloads_exact_artifact_sha256(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Verify the file served at FRUSTRAMPNN_HF_REVISION hashes to declared artifact_sha256."""
  from huggingface_hub import hf_hub_download

  mod = _load_vehicle_module()
  decl = tomllib.loads(DECL_PATH.read_text(encoding="utf-8"))
  weights_cfg = decl["implementation"]["weights"]

  monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
  dl_dir = tmp_path / "dl"
  eqx_p = Path(
    hf_hub_download(
      weights_cfg["hf_repo"],
      f"{mod.CANONICAL_CKPT}.eqx.zst",
      revision=mod.FRUSTRAMPNN_HF_REVISION,
      local_dir=str(dl_dir),
    )
  )
  man_p = Path(
    hf_hub_download(
      weights_cfg["hf_repo"],
      f"{mod.CANONICAL_CKPT}.manifest.json",
      revision=mod.FRUSTRAMPNN_HF_REVISION,
      local_dir=str(dl_dir),
    )
  )
  assert hashlib.sha256(eqx_p.read_bytes()).hexdigest() == weights_cfg["artifact_sha256"]
  manifest = json.loads(man_p.read_text(encoding="utf-8"))
  assert manifest["artifact_sha256"] == weights_cfg["artifact_sha256"]
  assert manifest["source_sha256"] == weights_cfg["attested_source_checkpoint_sha256"]

  # Also verify _verify_weights_provenance's missing-local-file HF fallback end-to-end
  fallback_decl = tmp_path / "fallback.parity.bth.toml"
  fallback_decl.write_text(
    DECL_PATH.read_text(encoding="utf-8").replace(
      weights_cfg["artifact_path"],
      "data/weights/frustrampnn/__nonexistent_local_fallback__.eqx.zst",
    ),
    encoding="utf-8",
  )
  resolved = mod._verify_weights_provenance(fallback_decl, None)
  assert resolved.is_file()
  assert hashlib.sha256(resolved.read_bytes()).hexdigest() == weights_cfg["artifact_sha256"]
