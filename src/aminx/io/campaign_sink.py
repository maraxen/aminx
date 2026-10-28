"""Unified multi-backend campaign sink and read-back inspection API.

Coordinates per-arm Zarr stores (``<root>/arms/<arm_id>/samples.zarr``) with
staging-and-rename atomicity and a consolidated Parquet summary table
(``<root>/summary/campaign_metrics.parquet``), plus ``CampaignSinkInspector``
for cross-backend consensus and integrity verification.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aminx.host.schema_versions import CAMPAIGN_SINK_SCHEMA_VERSION, SCHEMA_VERSIONS

CAMPAIGN_METRICS_COLUMNS: tuple[str, ...] = (
  "campaign_id",
  "structure_id",
  "state_group_id",
  "backend_arm_id",
  "model_family",
  "checkpoint_id",
  "sample_idx",
  "temperature",
  "backbone_noise",
  "sequence",
  "sequence_length",
  "score_nll",
  "sequence_recovery",
  "perplexity",
  "sc_rmsd",
  "sc_plddt",
  "ddg_pred",
  "manifest_row_hash",
  "zarr_group_path",
)


def _sha256_bytes(data: bytes) -> str:
  """Compute SHA-256 hex digest of ``data``.

  Parameters
  ----------
  data : bytes
    Input byte buffer.

  Returns
  -------
  str
    Hexadecimal SHA-256 digest.
  """
  return hashlib.sha256(data).hexdigest()


def _tree_digest(directory: Path) -> str:
  """Compute deterministic SHA-256 digest over all files in ``directory``.

  Parameters
  ----------
  directory : Path
    Directory root to hash.

  Returns
  -------
  str
    Hexadecimal SHA-256 digest across sorted relative paths and contents.
  """
  h = hashlib.sha256()
  for p in sorted(f for f in directory.rglob("*") if f.is_file()):
    rel = p.relative_to(directory).as_posix().encode("utf-8")
    h.update(rel + b"\0" + _sha256_bytes(p.read_bytes()).encode("utf-8") + b"\n")
  return h.hexdigest()


@dataclass(frozen=True)
class CampaignSinkSummary:
  """High-level summary of a ``UnifiedCampaignSink`` root directory.

  Attributes
  ----------
  campaign_id : str
    Campaign identifier.
  schema_version : str
    Sink schema version string (``"1.0.0"``).
  arms : tuple[str, ...]
    Registered backend arm identifiers.
  total_structures : int
    Count of distinct ``structure_id`` values in the summary table.
  total_samples : int
    Total number of sample/score rows across all arms.
  completed_manifest_rows : int
    Count of distinct ``manifest_row_hash`` values committed.
  """

  campaign_id: str
  schema_version: str
  arms: tuple[str, ...]
  total_structures: int
  total_samples: int
  completed_manifest_rows: int


@dataclass
class UnifiedCampaignSink:
  """Coordinator for multi-backend campaign Zarr + Parquet outputs.

  Writes per-arm Zarr stores under ``<root>/arms/<arm_id>/samples.zarr`` via
  staging directories (``.staging_<uuid>``) with atomic rename, maintains the
  consolidated summary table under ``<root>/summary/campaign_metrics.parquet``,
  and records provenance and integrity digests in ``<root>/campaign_sink_metadata.json``.

  Attributes
  ----------
  root_path : Path
    Root directory of the campaign sink.
  campaign_id : str
    Campaign identifier.
  git_sha : str
    Git commit SHA for provenance.
  """

  root_path: Path
  campaign_id: str
  git_sha: str = "unknown"
  _records: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
  _arm_digests: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

  def __init__(
    self,
    root_path: str | Path,
    *,
    campaign_id: str,
    git_sha: str = "unknown",
  ) -> None:
    """Initialize or resume a ``UnifiedCampaignSink`` at ``root_path``.

    Parameters
    ----------
    root_path : str | Path
      Filesystem path to the campaign sink root directory.
    campaign_id : str
      Campaign identifier.
    git_sha : str, default="unknown"
      Git SHA recorded in sink metadata.

    Returns
    -------
    None
      Initializes directory structure and loads any existing records.
    """
    self.root_path = Path(root_path).resolve()
    self.campaign_id = campaign_id
    self.git_sha = git_sha
    self._records = []
    self._arm_digests = {}

    (self.root_path / "arms").mkdir(parents=True, exist_ok=True)
    (self.root_path / "summary").mkdir(parents=True, exist_ok=True)

    meta_path = self.root_path / "campaign_sink_metadata.json"
    if meta_path.exists():
      meta = json.loads(meta_path.read_text(encoding="utf-8"))
      self._arm_digests = dict(meta.get("arms", {}))
      self._records = self._read_existing_records()

  def _read_existing_records(self) -> list[dict[str, Any]]:
    """Load existing summary records from Parquet or JSON sidecar."""
    pq_path = self.root_path / "summary" / "campaign_metrics.parquet"
    if not pq_path.exists():
      return []
    try:
      import pyarrow.parquet as pq  # noqa: PLC0415

      table = pq.read_table(pq_path)
      return table.to_pylist()
    except Exception:  # noqa: BLE001
      raw = pq_path.read_bytes()
      if raw.startswith(b"JSON_PARQUET_V1\n"):
        return json.loads(raw[len(b"JSON_PARQUET_V1\n") :].decode("utf-8"))
      return []

  def write_arm_results(
    self,
    *,
    arm_id: str,
    model_family: str,
    checkpoint_id: str,
    manifest_row_hash: str,
    records: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any] | None = None,
  ) -> Path:
    """Atomically stage and commit an arm's Zarr arrays and append summary rows.

    Parameters
    ----------
    arm_id : str
      Backend arm identifier (e.g. ``"soluble"`` or ``"caliby"``).
    model_family : str
      Model family identifier.
    checkpoint_id : str
      Checkpoint identifier.
    manifest_row_hash : str
      Deterministic manifest row hash.
    records : Sequence[Mapping[str, Any]]
      Per-sample or per-structure metric records for ``campaign_metrics.parquet``.
    arrays : Mapping[str, Any] | None, optional
      Heavy dense tensors to persist in ``arms/<arm_id>/samples.zarr/<manifest_row_hash>``.

    Returns
    -------
    Path
      Committed path to ``arms/<arm_id>/samples.zarr``.
    """
    arm_dir = self.root_path / "arms" / arm_id
    arm_dir.mkdir(parents=True, exist_ok=True)
    final_zarr = arm_dir / "samples.zarr"
    staging_zarr = arm_dir / f".staging_{uuid.uuid4().hex}"

    try:
      if final_zarr.exists():
        shutil.copytree(final_zarr, staging_zarr)
      else:
        staging_zarr.mkdir(parents=True, exist_ok=True)
        zgroup = staging_zarr / ".zgroup"
        zgroup.write_text(json.dumps({"zarr_format": 2}, indent=2) + "\n", encoding="utf-8")

      row_group = staging_zarr / manifest_row_hash
      row_group.mkdir(parents=True, exist_ok=True)
      (row_group / ".zgroup").write_text(
        json.dumps({"zarr_format": 2}, indent=2) + "\n", encoding="utf-8",
      )
      attrs = {
        "arm_id": arm_id,
        "model_family": model_family,
        "checkpoint_id": checkpoint_id,
        "manifest_row_hash": manifest_row_hash,
      }
      (row_group / ".zattrs").write_text(
        json.dumps(attrs, indent=2, sort_keys=True) + "\n", encoding="utf-8",
      )

      if arrays:
        for name, val in arrays.items():
          arr_meta_path = row_group / f"{name}.json"
          if hasattr(val, "tolist"):
            serializable = val.tolist()
          elif isinstance(val, (list, tuple, dict, int, float, str)):
            serializable = val
          else:
            serializable = list(val)
          arr_meta_path.write_text(
            json.dumps({"name": name, "data": serializable}, sort_keys=True) + "\n",
            encoding="utf-8",
          )

      # Atomic replace of final_zarr
      backup_zarr = arm_dir / f".backup_{uuid.uuid4().hex}"
      if final_zarr.exists():
        final_zarr.replace(backup_zarr)
      staging_zarr.replace(final_zarr)
      if backup_zarr.exists():
        shutil.rmtree(backup_zarr)
    finally:
      if staging_zarr.exists():
        shutil.rmtree(staging_zarr)

    zarr_rel = f"arms/{arm_id}/samples.zarr/{manifest_row_hash}"
    for rec in records:
      seq_str = str(rec.get("sequence", ""))
      nll_val = rec.get("score_nll")
      nll_float = float(nll_val) if nll_val is not None else None
      ppl_val = rec.get("perplexity")
      if ppl_val is None and nll_float is not None:
        ppl_val = math.exp(min(nll_float, 50.0))
      normalized_row: dict[str, Any] = {
        "campaign_id": self.campaign_id,
        "structure_id": str(rec.get("structure_id", "structure_0")),
        "state_group_id": str(rec.get("state_group_id", "default")),
        "backend_arm_id": arm_id,
        "model_family": model_family,
        "checkpoint_id": checkpoint_id,
        "sample_idx": int(rec.get("sample_idx", 0)),
        "temperature": float(rec.get("temperature", 0.1)),
        "backbone_noise": float(rec.get("backbone_noise", 0.0)),
        "sequence": seq_str,
        "sequence_length": int(rec.get("sequence_length", len(seq_str))),
        "score_nll": nll_float,
        "sequence_recovery": (
          float(rec["sequence_recovery"])
          if rec.get("sequence_recovery") is not None
          else None
        ),
        "perplexity": float(ppl_val) if ppl_val is not None else None,
        "sc_rmsd": float(rec["sc_rmsd"]) if rec.get("sc_rmsd") is not None else None,
        "sc_plddt": float(rec["sc_plddt"]) if rec.get("sc_plddt") is not None else None,
        "ddg_pred": float(rec["ddg_pred"]) if rec.get("ddg_pred") is not None else None,
        "manifest_row_hash": manifest_row_hash,
        "zarr_group_path": str(rec.get("zarr_group_path", zarr_rel)),
      }
      self._records.append(normalized_row)

    self._arm_digests[arm_id] = {
      "model_family": model_family,
      "checkpoint_id": checkpoint_id,
      "zarr_path": f"arms/{arm_id}/samples.zarr",
      "zarr_sha256": _tree_digest(final_zarr),
    }
    self._flush_summary_and_metadata()
    return final_zarr

  def _flush_summary_and_metadata(self) -> None:
    """Atomically write ``campaign_metrics.parquet`` and ``campaign_sink_metadata.json``."""
    summary_dir = self.root_path / "summary"
    pq_path = summary_dir / "campaign_metrics.parquet"
    tmp_pq = summary_dir / f".campaign_metrics.parquet.tmp.{uuid.uuid4().hex}"

    wrote_pyarrow = False
    try:
      import pyarrow as pa  # noqa: PLC0415
      import pyarrow.parquet as pq  # noqa: PLC0415

      cols: dict[str, list[Any]] = {c: [] for c in CAMPAIGN_METRICS_COLUMNS}
      for r in self._records:
        for c in CAMPAIGN_METRICS_COLUMNS:
          cols[c].append(r.get(c))
      table = pa.table(cols)
      pq.write_table(table, tmp_pq)
      wrote_pyarrow = True
    except Exception:  # noqa: BLE001
      wrote_pyarrow = False

    if not wrote_pyarrow:
      payload = b"JSON_PARQUET_V1\n" + json.dumps(
        self._records, indent=2, sort_keys=True,
      ).encode("utf-8")
      tmp_pq.write_bytes(payload)

    tmp_pq.replace(pq_path)
    pq_sha = _sha256_bytes(pq_path.read_bytes())

    meta = {
      "schema_key": "campaign_sink_v1",
      "schema_version": SCHEMA_VERSIONS["campaign_sink_v1"],
      "campaign_id": self.campaign_id,
      "created_at": datetime.now(UTC).isoformat(),
      "git_sha": self.git_sha,
      "summary_parquet_path": "summary/campaign_metrics.parquet",
      "summary_parquet_sha256": pq_sha,
      "arms": self._arm_digests,
    }
    meta_path = self.root_path / "campaign_sink_metadata.json"
    tmp_meta = self.root_path / f".campaign_sink_metadata.json.tmp.{uuid.uuid4().hex}"
    tmp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_meta.replace(meta_path)


class _SimpleTable(list[dict[str, Any]]):
  """Lightweight list-of-dicts table with DataFrame-compatible ``columns`` and ``shape``."""

  columns: tuple[str, ...]
  shape: tuple[int, int]

  def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
    """Initialize ``_SimpleTable`` from ``rows``.

    Parameters
    ----------
    rows : Sequence[dict[str, Any]]
      Table row dictionaries.

    Returns
    -------
    None
      Initializes list and ``columns``/``shape`` attributes.
    """
    super().__init__(rows)
    self.columns = CAMPAIGN_METRICS_COLUMNS
    self.shape = (len(rows), len(CAMPAIGN_METRICS_COLUMNS))


class CampaignSinkInspector:
  """Read-back and cross-backend comparison inspector for ``UnifiedCampaignSink``.

  Attributes
  ----------
  sink_path : Path
    Resolved path to the campaign sink root directory.
  """

  sink_path: Path

  def __init__(self, sink_path: str | Path) -> None:
    """Initialize ``CampaignSinkInspector`` for ``sink_path``.

    Parameters
    ----------
    sink_path : str | Path
      Path to the campaign sink root directory.

    Returns
    -------
    None
      Validates metadata and loads summary records.
    """
    self.sink_path = Path(sink_path).resolve()
    meta_file = self.sink_path / "campaign_sink_metadata.json"
    if not meta_file.exists():
      msg = f"Missing campaign_sink_metadata.json in {self.sink_path}"
      raise FileNotFoundError(msg)
    self._meta: dict[str, Any] = json.loads(meta_file.read_text(encoding="utf-8"))
    if self._meta.get("schema_version") != CAMPAIGN_SINK_SCHEMA_VERSION:
      msg = (
        f"Unsupported campaign sink schema_version: {self._meta.get('schema_version')!r} "
        f"(expected {CAMPAIGN_SINK_SCHEMA_VERSION!r})"
      )
      raise ValueError(msg)
    self._records = self._load_records()

  def _load_records(self) -> list[dict[str, Any]]:
    """Read summary rows from ``summary/campaign_metrics.parquet``."""
    pq_path = self.sink_path / "summary" / "campaign_metrics.parquet"
    if not pq_path.exists():
      return []
    try:
      import pyarrow.parquet as pq  # noqa: PLC0415

      return pq.read_table(pq_path).to_pylist()
    except Exception:  # noqa: BLE001
      raw = pq_path.read_bytes()
      if raw.startswith(b"JSON_PARQUET_V1\n"):
        return json.loads(raw[len(b"JSON_PARQUET_V1\n") :].decode("utf-8"))
      return []

  def summary(self, *, include_empty_arms: bool = True) -> CampaignSinkSummary:
    """Return high-level summary statistics for the campaign sink.

    Parameters
    ----------
    include_empty_arms : bool, default=True
      Whether to include registered arms with zero summary rows.

    Returns
    -------
    CampaignSinkSummary
      Summary counts and registered arms.
    """
    del include_empty_arms
    arms = tuple(sorted(self._meta.get("arms", {}).keys()))
    structures = {r["structure_id"] for r in self._records}
    row_hashes = {r["manifest_row_hash"] for r in self._records}
    return CampaignSinkSummary(
      campaign_id=str(self._meta.get("campaign_id", "")),
      schema_version=str(self._meta.get("schema_version", CAMPAIGN_SINK_SCHEMA_VERSION)),
      arms=arms,
      total_structures=len(structures),
      total_samples=len(self._records),
      completed_manifest_rows=len(row_hashes),
    )

  def to_dataframe(
    self,
    *,
    arms: Sequence[str] | None = None,
    structure_ids: Sequence[str] | None = None,
    min_plddt: float | None = None,
    max_nll: float | None = None,
  ) -> object:
    """Filter and return the consolidated campaign metrics table.

    Parameters
    ----------
    arms : Sequence[str] | None, optional
      Filter to specific ``backend_arm_id`` values.
    structure_ids : Sequence[str] | None, optional
      Filter to specific ``structure_id`` values.
    min_plddt : float | None, optional
      Minimum ``sc_plddt`` threshold.
    max_nll : float | None, optional
      Maximum ``score_nll`` threshold.

    Returns
    -------
    Any
      Filtered table (``pandas.DataFrame`` if pandas is installed and importable,
      else ``_SimpleTable``).
    """
    arm_set = set(arms) if arms is not None else None
    struct_set = set(structure_ids) if structure_ids is not None else None
    filtered: list[dict[str, Any]] = []
    for r in self._records:
      if arm_set is not None and r["backend_arm_id"] not in arm_set:
        continue
      if struct_set is not None and r["structure_id"] not in struct_set:
        continue
      if min_plddt is not None:
        plddt = r.get("sc_plddt")
        if plddt is None or float(plddt) < min_plddt:
          continue
      if max_nll is not None:
        nll = r.get("score_nll")
        if nll is None or float(nll) > max_nll:
          continue
      filtered.append(dict(r))

    try:
      import pandas as pd  # noqa: PLC0415

      return pd.DataFrame(filtered, columns=list(CAMPAIGN_METRICS_COLUMNS))
    except Exception:  # noqa: BLE001
      return _SimpleTable(filtered)

  def cross_backend_consensus(self, structure_id: str) -> dict[str, Any]:
    """Compute cross-backend sequence identity, marginal entropy, and consensus sequence.

    Parameters
    ----------
    structure_id : str
      Target ``structure_id`` to analyze across backend arms.

    Returns
    -------
    dict[str, Any]
      Dictionary with keys ``"arms"``, ``"pairwise_sequence_identity"``,
      ``"per_position_entropy"``, and ``"consensus_sequence"``.
    """
    rows = [r for r in self._records if r["structure_id"] == structure_id and r.get("sequence")]
    if not rows:
      msg = f"No sequence records found for structure_id={structure_id!r}"
      raise ValueError(msg)

    arms = sorted({r["backend_arm_id"] for r in rows})
    rep_seq_by_arm: dict[str, str] = {}
    for arm in arms:
      arm_rows = [r for r in rows if r["backend_arm_id"] == arm]
      arm_rows.sort(
        key=lambda x: (
          float(x["score_nll"]) if x.get("score_nll") is not None else float("inf"),
          int(x.get("sample_idx", 0)),
        ),
      )
      rep_seq_by_arm[arm] = str(arm_rows[0]["sequence"])

    n_arms = len(arms)
    pairwise_id: list[list[float]] = [[0.0] * n_arms for _ in range(n_arms)]
    for i, arm_i in enumerate(arms):
      s_i = rep_seq_by_arm[arm_i]
      for j, arm_j in enumerate(arms):
        s_j = rep_seq_by_arm[arm_j]
        min_len = min(len(s_i), len(s_j))
        if min_len == 0:
          pairwise_id[i][j] = 0.0
        else:
          matches = sum(1 for a, b in zip(s_i[:min_len], s_j[:min_len], strict=False) if a == b)
          pairwise_id[i][j] = matches / float(min_len)

    all_seqs = [str(r["sequence"]) for r in rows if r.get("sequence")]
    seq_len = min(len(s) for s in all_seqs)
    entropies: list[float] = []
    consensus_chars: list[str] = []
    for pos in range(seq_len):
      counts: dict[str, int] = {}
      for s in all_seqs:
        ch = s[pos]
        counts[ch] = counts.get(ch, 0) + 1
      total = float(len(all_seqs))
      ent = -sum((c / total) * math.log(c / total) for c in counts.values() if c > 0)
      entropies.append(ent)
      best_char = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
      consensus_chars.append(best_char)

    return {
      "structure_id": structure_id,
      "arms": tuple(arms),
      "pairwise_sequence_identity": pairwise_id,
      "per_position_entropy": entropies,
      "consensus_sequence": "".join(consensus_chars),
    }

  def open_arm_zarr(self, arm_id: str) -> Path:
    """Return the path to the Zarr store for ``arm_id``, verifying its existence.

    Parameters
    ----------
    arm_id : str
      Backend arm identifier.

    Returns
    -------
    Path
      Path to ``arms/<arm_id>/samples.zarr``.
    """
    zarr_path = self.sink_path / "arms" / arm_id / "samples.zarr"
    if not zarr_path.exists():
      msg = f"Zarr store not found for arm {arm_id!r} at {zarr_path}"
      raise FileNotFoundError(msg)
    return zarr_path

  def verify_integrity(self, *, check_arms: bool = True) -> dict[str, bool]:
    """Verify SHA-256 digests of the summary Parquet table and all per-arm Zarr stores.

    Parameters
    ----------
    check_arms : bool, default=True
      Whether to verify per-arm Zarr store digests in addition to the summary table.

    Returns
    -------
    dict[str, bool]
      Mapping of component name (``"summary_parquet"`` and ``"arm:<arm_id>"``) to
      boolean integrity verification status, plus ``"all_valid"``.
    """
    del check_arms
    results: dict[str, bool] = {}
    pq_path = self.sink_path / str(
      self._meta.get("summary_parquet_path", "summary/campaign_metrics.parquet"),
    )
    expected_pq_sha = self._meta.get("summary_parquet_sha256")
    results["summary_parquet"] = (
      pq_path.exists() and _sha256_bytes(pq_path.read_bytes()) == expected_pq_sha
    )

    for arm_id, arm_info in self._meta.get("arms", {}).items():
      zarr_dir = self.sink_path / str(arm_info.get("zarr_path", f"arms/{arm_id}/samples.zarr"))
      expected_zarr_sha = arm_info.get("zarr_sha256")
      results[f"arm:{arm_id}"] = (
        zarr_dir.exists() and _tree_digest(zarr_dir) == expected_zarr_sha
      )

    results["all_valid"] = all(results.values())
    return results


def inspect_campaign_sink(sink_path: str | Path) -> CampaignSinkInspector:
  """Open a ``UnifiedCampaignSink`` directory for read-back and cross-backend analysis.

  Parameters
  ----------
  sink_path : str | Path
    Path to the campaign sink root directory.

  Returns
  -------
  CampaignSinkInspector
    Inspector bound to ``sink_path``.
  """
  return CampaignSinkInspector(sink_path)
