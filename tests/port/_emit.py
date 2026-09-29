"""domain=port audit records in the xtrax port emit shape."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

PORT_DIM = "port"

TierStatus = Literal["PASS", "FAIL", "WARN"]
Severity = Literal["info", "minor", "major", "critical"]


@dataclass(frozen=True, slots=True)
class TierVerdict:
  status: TierStatus
  tolerance_policy: str
  error_taxonomy_class: str
  max_discrepancy: float | None = None


@dataclass(frozen=True, slots=True)
class Evidence:
  pytest_nodeid: str
  traceback_excerpt: str = ""


@dataclass(frozen=True, slots=True)
class PortEmitRecord:
  audit_id: str
  task_id: str
  domain: Literal["port"]
  track: Literal["deterministic"]
  finding_id: str
  symbol_qualname: str
  rule_id: str
  label: Literal["bug", "observation"]
  severity: Severity
  port_parity_tier: str
  oracle_id: str
  tier_verdict: TierVerdict
  evidence: Evidence
  routing: dict[str, Any]


def _stable_hash(*parts: str) -> str:
  joined = "\x1f".join(parts)
  return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def compute_port_finding_id(
  dim: str,
  symbol_qualname: str,
  rule_id: str,
  tolerance_policy: str,
) -> str:
  """Hash(dim + qualname + rule_id + tolerance_policy), matching xtrax port emit."""
  return _stable_hash(dim, symbol_qualname, rule_id, tolerance_policy)


def _label_for_status(status: TierStatus) -> Literal["bug", "observation"]:
  if status == "FAIL":
    return "bug"
  return "observation"


def _severity_for_status(status: TierStatus) -> Severity:
  if status == "FAIL":
    return "major"
  if status == "WARN":
    return "minor"
  return "info"


def _tier_verdict_to_dict(verdict: TierVerdict) -> dict[str, Any]:
  data = asdict(verdict)
  if verdict.max_discrepancy is None:
    data.pop("max_discrepancy", None)
  return data


def _evidence_to_dict(evidence: Evidence) -> dict[str, Any]:
  data = asdict(evidence)
  if not evidence.traceback_excerpt:
    data.pop("traceback_excerpt", None)
  return data


def _build_port_record(
  *,
  task_id: str,
  symbol_qualname: str,
  port_parity_tier: str,
  oracle_id: str,
  tier_verdict: TierVerdict,
  evidence: Evidence,
  audit_id: str | None = None,
) -> PortEmitRecord:
  rule_id = port_parity_tier
  finding_id = compute_port_finding_id(
    PORT_DIM,
    symbol_qualname,
    rule_id,
    tier_verdict.tolerance_policy,
  )
  return PortEmitRecord(
    audit_id=audit_id or f"{task_id}_{port_parity_tier}",
    task_id=task_id,
    domain="port",
    track="deterministic",
    finding_id=finding_id,
    symbol_qualname=symbol_qualname,
    rule_id=rule_id,
    label=_label_for_status(tier_verdict.status),
    severity=_severity_for_status(tier_verdict.status),
    port_parity_tier=port_parity_tier,
    oracle_id=oracle_id,
    tier_verdict=tier_verdict,
    evidence=evidence,
    routing={"destination": None},
  )


def port_record_to_dict(record: PortEmitRecord) -> dict[str, Any]:
  payload = asdict(record)
  payload["tier_verdict"] = _tier_verdict_to_dict(record.tier_verdict)
  payload["evidence"] = _evidence_to_dict(record.evidence)
  return payload


def emit_tier_verdict(
  *,
  task_id: str,
  symbol_qualname: str,
  port_parity_tier: str,
  oracle_id: str,
  tier_verdict: TierVerdict,
  evidence: Evidence,
  audits_path: Path,
  audit_id: str | None = None,
) -> str:
  """Append one port tier JSONL record and return its finding_id."""
  record = _build_port_record(
    task_id=task_id,
    symbol_qualname=symbol_qualname,
    port_parity_tier=port_parity_tier,
    oracle_id=oracle_id,
    tier_verdict=tier_verdict,
    evidence=evidence,
    audit_id=audit_id,
  )
  expected = compute_port_finding_id(
    PORT_DIM,
    record.symbol_qualname,
    record.rule_id,
    record.tier_verdict.tolerance_policy,
  )
  if record.finding_id != expected:
    msg = f"finding_id mismatch: {record.finding_id!r} != {expected!r}"
    raise ValueError(msg)
  audits_path.parent.mkdir(parents=True, exist_ok=True)
  line = json.dumps(port_record_to_dict(record), sort_keys=True)
  with audits_path.open("a", encoding="utf-8") as handle:
    handle.write(line)
    handle.write("\n")
  return record.finding_id
