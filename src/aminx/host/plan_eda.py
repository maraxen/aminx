"""EDA plan inspection, panel rendering, and manifest verification adapter for aminx."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from xtrax.eda import (
  analyze_bucket,
  explain_plan,
  export_plan_artifacts,
  extract_plan_stats,
  generate_vega_spec,
  render,
)

from aminx.io.weights import resolve_family_checkpoint

if TYPE_CHECKING:
  from xtrax.tiling import BatchPlan

__all__ = [
  "analyze_aminx_bucket",
  "explain_aminx_plan",
  "export_and_verify_plan_artifacts",
  "generate_aminx_plan_spec",
  "render_aminx_plan_panels",
  "verify_checkpoint_and_explain_plan",
  "verify_export_manifest_sha256",
]


def explain_aminx_plan(plan: BatchPlan) -> dict[str, Any]:
  """Explains an aminx BatchPlan using xtrax.eda.

  Parameters
  ----------
  plan : BatchPlan
      The planned execution topology.

  Returns
  -------
  dict[str, Any]
      Dictionary containing explain_plan summary and per-bucket decision analyses.
  """
  stats = explain_plan(plan)
  bucket_analyses: list[dict[str, Any]] = []
  for d in getattr(plan, "decisions", []):
    strategy_name = d.strategy.__class__.__name__ if hasattr(d, "strategy") else ""
    if strategy_name == "Bucket":
      bucket_analyses.append(analyze_bucket(d))

  return {
    "plan_summary": stats,
    "bucket_analyses": bucket_analyses,
  }


def analyze_aminx_bucket(bucket: Any) -> dict[str, Any]:  # noqa: ANN401
  """Analyzes a bucket decision using xtrax.eda.analyze_bucket."""
  return analyze_bucket(bucket)


def verify_checkpoint_and_explain_plan(
  plan: BatchPlan,
  *,
  checkpoint_family: str,
  checkpoint_id: str,
) -> dict[str, Any]:
  """Verifies checkpoint manifest checksum before explaining plan topology."""
  # Resolves artifact and companion manifest, validating both sha256 pins
  resolve_family_checkpoint(checkpoint_id, family=checkpoint_family)
  return explain_aminx_plan(plan)


def render_aminx_plan_panels(
  plan: BatchPlan,
  *,
  fmt: Literal["png", "svg", "html"] = "svg",
  panels: set[str] | None = None,
  path: Path | str | None = None,
) -> bytes | str | None:
  """Renders BatchPlan visual panels using xtrax.eda.render."""
  return render(plan, fmt=fmt, panels=panels, path=path)


def generate_aminx_plan_spec(
  plan: BatchPlan,
  *,
  panels: set[str] | None = None,
) -> dict[str, Any]:
  """Generates a declarative Vega-Lite v5 specification dictionary from BatchPlan."""
  stats = extract_plan_stats(plan)
  return generate_vega_spec(stats, panels=panels)


def verify_export_manifest_sha256(
  export_dir: Path | str,
  *,
  manifest_path: Path | str | None = None,
) -> dict[str, Any]:
  """Cryptographically verifies artifact SHA-256 digests against export manifest.

  Parameters
  ----------
  export_dir : Path | str
      Directory containing exported plan artifacts.
  manifest_path : Path | str | None, default=None
      Explicit path to manifest file. If None, auto-discovers `*_manifest.json`.

  Returns
  -------
  dict[str, Any]
      Parsed and verified manifest payload.

  Raises
  ------
  FileNotFoundError
      If manifest or any declared artifact is missing.
  ValueError
      If any artifact's byte count or SHA-256 digest mismatches the manifest.
  """
  export_path = Path(export_dir)
  if manifest_path is None:
    manifest_files = sorted(export_path.glob("*_manifest.json"))
    if not manifest_files:
      msg = f"No export manifest file found in {export_path}"
      raise FileNotFoundError(msg)
    m_path = manifest_files[0]
  else:
    m_path = Path(manifest_path)

  if not m_path.exists():
    msg = f"Manifest file not found: {m_path}"
    raise FileNotFoundError(msg)

  manifest = json.loads(m_path.read_text(encoding="utf-8"))
  artifacts = manifest.get("artifacts", {})
  if not artifacts:
    msg = f"Export manifest has no declared artifacts: {m_path}"
    raise ValueError(msg)

  for filename, info in artifacts.items():
    artifact_file = export_path / filename
    if not artifact_file.exists():
      msg = f"Declared export artifact missing: {artifact_file}"
      raise FileNotFoundError(msg)

    data = artifact_file.read_bytes()
    expected_bytes = info.get("bytes")
    if expected_bytes is not None and len(data) != expected_bytes:
      msg = f"Byte length mismatch for {filename}: expected {expected_bytes}, got {len(data)}"
      raise ValueError(msg)

    expected_sha = info.get("sha256")
    observed_sha = hashlib.sha256(data).hexdigest()
    if expected_sha is not None and observed_sha != expected_sha:
      msg = (
        f"SHA-256 digest mismatch for {filename}: expected {expected_sha}, observed {observed_sha}"
      )
      raise ValueError(msg)

  return manifest


def export_and_verify_plan_artifacts(
  plan: BatchPlan,
  output_dir: Path | str,
  *,
  prefix: str = "plan",
) -> dict[str, Any]:
  """Exports byte-deterministic plan artifacts and cryptographically verifies them."""
  stats = extract_plan_stats(plan)
  manifest = export_plan_artifacts(stats, output_dir, prefix=prefix)
  verify_export_manifest_sha256(output_dir)
  return manifest
