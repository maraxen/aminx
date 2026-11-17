"""Run telemetry: batch-plan inspection, panel rendering and manifest verification for aminx.

This explains how a run is batched (strategy, cardinality, buckets, memory). It says nothing about
what a campaign sampled. Statistics of sampled libraries faceted by campaign knobs are specified in
``.praxia/docs/specs/261106_campaign-library-eda-respec.md``; these plan panels become that view's
telemetry tab.
"""

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
from xtrax.tiling import Bucket

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
  bucket_analyses: list[dict[str, Any]] = [
    analyze_bucket(d)
    for d in getattr(plan, "decisions", [])
    if isinstance(getattr(d, "strategy", None), Bucket)
  ]

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
      Explicit path to the manifest file. If None, the directory must hold exactly one
      ``*_manifest.json``; several are refused rather than guessed between.

  Returns
  -------
  dict[str, Any]
      Parsed and verified manifest payload.

  Raises
  ------
  FileNotFoundError
      If manifest or any declared artifact is missing.
  ValueError
      If any artifact's byte count or SHA-256 digest mismatches the manifest, an entry has no
      ``sha256``, a filename is not a plain name inside ``export_dir``, or the manifest is
      ambiguous.
  """
  export_path = Path(export_dir)
  if manifest_path is None:
    manifest_files = sorted(export_path.glob("*_manifest.json"))
    if not manifest_files:
      msg = f"No export manifest file found in {export_path}"
      raise FileNotFoundError(msg)
    if len(manifest_files) > 1:
      names = [m.name for m in manifest_files]
      msg = f"Several export manifests in {export_path} ({names}); pass manifest_path"
      raise ValueError(msg)
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
    if Path(filename).name != filename or filename in ("", ".", ".."):
      msg = f"Manifest entry {filename!r} is not a plain file name inside {export_path}"
      raise ValueError(msg)
    expected_sha = info.get("sha256")
    if not expected_sha:
      msg = f"Manifest entry {filename!r} has no sha256 to verify"
      raise ValueError(msg)
    artifact_file = export_path / filename
    if not artifact_file.exists():
      msg = f"Declared export artifact missing: {artifact_file}"
      raise FileNotFoundError(msg)

    data = artifact_file.read_bytes()
    expected_bytes = info.get("bytes")
    if expected_bytes is not None and len(data) != expected_bytes:
      msg = f"Byte length mismatch for {filename}: expected {expected_bytes}, got {len(data)}"
      raise ValueError(msg)

    observed_sha = hashlib.sha256(data).hexdigest()
    if observed_sha != expected_sha:
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
  # Verify this export's own manifest, not whichever *_manifest.json sorts first.
  verify_export_manifest_sha256(
    output_dir,
    manifest_path=Path(output_dir) / f"{prefix}_manifest.json",
  )
  return manifest
