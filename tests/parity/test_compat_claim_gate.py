"""Compatibility-claim gate cases (T8 step 5)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.browser_validation import compat_claim_gate

_CHROMIUM = (
  "ORT Web wasm runs in headless Chromium (evidence: "
  "outputs/browser_validation/layer_c/evidence/ort-wasm__chromium.json)."
)


def _write_sentence(tmp_path: Path, text: str) -> Path:
  path = tmp_path / "claim.md"
  path.write_text(text + "\n", encoding="utf-8")
  return path


def _evidence(tmp_path: Path) -> Path:
  path = (
    tmp_path / "outputs" / "browser_validation" / "layer_c" / "evidence" / "ort-wasm__chromium.json"
  )
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps({"cross_origin_isolated": True}), encoding="utf-8")
  return path


def test_firefox_sentence_exits_1(tmp_path: Path) -> None:
  doc = _write_sentence(tmp_path, "ORT Web runs on Firefox.")
  assert compat_claim_gate.main(["--paths", str(doc), "--root", str(tmp_path)]) == 1


def test_chromium_sentence_with_evidence_exits_0(tmp_path: Path) -> None:
  _evidence(tmp_path)
  doc = _write_sentence(tmp_path, _CHROMIUM)
  assert compat_claim_gate.main(["--paths", str(doc), "--root", str(tmp_path)]) == 0


def test_chromium_sentence_without_evidence_exits_1(tmp_path: Path) -> None:
  doc = _write_sentence(tmp_path, _CHROMIUM)
  assert compat_claim_gate.main(["--paths", str(doc), "--root", str(tmp_path)]) == 1
