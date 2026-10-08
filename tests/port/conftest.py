"""Graded port-parity tier contract (spec §7.2), mirrored from xtrax port/.

Active only for items under ``tests/port/``. Hooks return immediately when
``AMINX_PORT_WAVE`` is ``__nonport__``. ``AMINX_PORT_WAVE`` selects
``tests/port/targets/<wave>.toml``; otherwise ``[tool.port] target`` in
pyproject is the fallback.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import jax
import pytest

from port._emit import Evidence, TierVerdict, emit_tier_verdict

PORT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PORT_ROOT.parents[1]

TIER_MARKERS = ("tier_1", "tier_2", "tier_3", "tier_4", "tier_5")
TIER_TO_EMIT = {
  "tier_1": "parity_tier_1",
  "tier_2": "parity_tier_2",
  "tier_3": "parity_tier_3",
  "tier_4": "parity_tier_4",
  "tier_5": "parity_tier_5",
}
DEFAULT_TASK_ID = "260929_potts-laser-xtrax-compose"
TIER_TIMEOUT_SECONDS = 120
_MAX_ABS_DIFF_RE = re.compile(
  r"Max absolute difference(?: among violations)?:\s*([0-9.eE+\-]+)"
)


@dataclass(frozen=True)
class PortWaveConfig:
  port: dict[str, Any]
  capabilities: dict[str, Any]
  parity: dict[str, Any]
  manifest: dict[str, Any]
  manifest_path: Path
  blocking_tiers: tuple[str, ...]


_TIER_GATE_STASH_KEY = pytest.StashKey[dict[str, Any]]()
_WAVE_STASH_KEY = pytest.StashKey[PortWaveConfig]()


def _nonport_wave() -> bool:
  return os.environ.get("AMINX_PORT_WAVE") == "__nonport__"


def _is_port_item(item: pytest.Item) -> bool:
  try:
    Path(str(item.path)).resolve().relative_to(PORT_ROOT)
  except ValueError:
    return False
  return True


def _resolve_port_target_path() -> Path:
  wave = os.environ.get("AMINX_PORT_WAVE")
  if wave and wave != "__nonport__":
    return (PORT_ROOT / "targets" / f"{wave}.toml").resolve()
  pyproject = REPO_ROOT / "pyproject.toml"
  data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
  configured = data.get("tool", {}).get("port", {}).get("target")
  if isinstance(configured, str) and configured:
    return (REPO_ROOT / configured).resolve()
  msg = "AMINX_PORT_WAVE is unset and [tool.port] target is missing"
  raise FileNotFoundError(msg)


def load_port_target(path: Path | None = None) -> dict[str, Any]:
  target_path = path or _resolve_port_target_path()
  if not target_path.is_file():
    msg = f"port target not found: {target_path}"
    raise FileNotFoundError(msg)
  return tomllib.loads(target_path.read_text(encoding="utf-8"))


def load_manifest(port_root: Path, wave_id: str) -> tuple[dict[str, Any], Path]:
  manifest_path = port_root / "manifests" / f"{wave_id}.toml"
  if not manifest_path.is_file():
    msg = f"manifest not found for wave_id={wave_id!r}: {manifest_path}"
    raise FileNotFoundError(msg)
  return tomllib.loads(manifest_path.read_text(encoding="utf-8")), manifest_path


def canonical_manifest_bytes(manifest_text: str) -> bytes:
  """Canonical TOML bytes for manifest_hash (excludes the manifest_hash field)."""
  lines: list[str] = []
  for line in manifest_text.splitlines():
    if re.match(r"^\s*manifest_hash\s*=", line):
      continue
    lines.append(line)
  canonical = "\n".join(lines).rstrip() + "\n"
  return canonical.encode("utf-8")


def compute_manifest_hash(manifest_text: str) -> str:
  digest = hashlib.sha256(canonical_manifest_bytes(manifest_text)).hexdigest()
  return f"sha256:{digest}"


def verify_manifest_hash(manifest_path: Path) -> None:
  text = manifest_path.read_text(encoding="utf-8")
  manifest = tomllib.loads(text)
  recorded = manifest.get("manifest", {}).get("manifest_hash")
  if not isinstance(recorded, str) or not recorded:
    pytest.fail(f"{manifest_path}: missing [manifest].manifest_hash")
  expected = compute_manifest_hash(text)
  if recorded != expected:
    pytest.fail(
      f"{manifest_path}: manifest_hash mismatch (recorded {recorded!r}, expected {expected!r})"
    )


def blocking_tiers_from_config(port_config: dict[str, Any]) -> tuple[str, ...]:
  parity = port_config.get("parity", {})
  ad_critical = bool(parity.get("ad_critical", False))
  justification = parity.get("ad_critical_justification", "")
  if ad_critical:
    if not isinstance(justification, str) or not justification.strip():
      msg = "port target parity.ad_critical=true requires non-empty ad_critical_justification"
      raise ValueError(msg)
    return TIER_MARKERS
  return tuple(tier for tier in TIER_MARKERS if tier != "tier_4")


def _require_parity_keys(parity: dict[str, Any]) -> None:
  for key in ("tolerance_policy_f64", "tolerance_policy_f32"):
    if not isinstance(parity.get(key), str):
      msg = f"port target [parity] {key} is required"
      raise ValueError(msg)
  if not isinstance(parity.get("max_traces"), int):
    msg = "port target [parity] max_traces must be an int"
    raise ValueError(msg)


def build_wave_config(port_config: dict[str, Any], manifest_path: Path) -> PortWaveConfig:
  port_section = port_config.get("port", {})
  wave_id = port_section.get("wave_id")
  if not isinstance(wave_id, str) or not wave_id:
    msg = "port target [port] wave_id is required"
    raise ValueError(msg)
  parity = port_config.get("parity", {})
  _require_parity_keys(parity)
  manifest, _ = load_manifest(PORT_ROOT, wave_id)
  return PortWaveConfig(
    port=port_section,
    capabilities=port_config.get("capabilities", {}),
    parity=parity,
    manifest=manifest,
    manifest_path=manifest_path,
    blocking_tiers=blocking_tiers_from_config(port_config),
  )


def _tier_marker_for_item(item: pytest.Item) -> str | None:
  for tier in TIER_MARKERS:
    if item.get_closest_marker(tier) is not None:
      return tier
  return None


def _tier_sort_key(item: pytest.Item) -> tuple[int, str]:
  tier = _tier_marker_for_item(item)
  if tier is None:
    return (len(TIER_MARKERS), item.nodeid)
  return (TIER_MARKERS.index(tier), item.nodeid)


def _import_reference_algo(reference_subtree: str) -> ModuleType:
  ref_path = PORT_ROOT / reference_subtree / "algo.py"
  if not ref_path.is_file():
    msg = f"reference oracle missing: {ref_path}"
    raise FileNotFoundError(msg)
  spec = importlib.util.spec_from_file_location(
    f"port_reference_{reference_subtree.replace('/', '_')}",
    ref_path,
  )
  if spec is None or spec.loader is None:
    msg = f"unable to import reference oracle: {ref_path}"
    raise RuntimeError(msg)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _has_injected_uniform(module: ModuleType) -> bool:
  """True when the sealed oracle exposes an injected-uniform key."""
  value = getattr(module, "injected_uniform", None)
  if isinstance(value, str):
    return bool(value.strip())
  return value is not None


def _policy_for_tier(parity: dict[str, Any], tier: str) -> str:
  if tier == "tier_3":
    return str(parity["tolerance_policy_f32"])
  return str(parity["tolerance_policy_f64"])


def _session_wave(item: pytest.Item) -> PortWaveConfig:
  if _WAVE_STASH_KEY in item.session.stash:
    return item.session.stash[_WAVE_STASH_KEY]
  port_config = load_port_target()
  wave_id = port_config["port"]["wave_id"]
  _, manifest_path = load_manifest(PORT_ROOT, wave_id)
  wave = build_wave_config(port_config, manifest_path)
  item.session.stash[_WAVE_STASH_KEY] = wave
  return wave


def _x64_context() -> Any:
  """Scoped x64. JAX 0.10 exposes ``jax.enable_x64`` (formerly experimental)."""
  enable = getattr(jax.experimental, "enable_x64", None)
  if enable is None:
    enable = jax.enable_x64
  return enable()


@pytest.fixture(scope="session")
def port_wave() -> PortWaveConfig:
  if _nonport_wave():
    pytest.skip("__nonport__ wave does not load a port target")
  port_config = load_port_target()
  wave_id = port_config["port"]["wave_id"]
  _, manifest_path = load_manifest(PORT_ROOT, wave_id)
  verify_manifest_hash(manifest_path)
  return build_wave_config(port_config, manifest_path)


@pytest.fixture(scope="session")
def port_target(port_wave: PortWaveConfig) -> dict[str, Any]:
  return {
    "port": port_wave.port,
    "capabilities": port_wave.capabilities,
    "parity": port_wave.parity,
  }


@pytest.fixture(scope="session")
def oracle(port_wave: PortWaveConfig) -> ModuleType:
  subtree = port_wave.port.get("reference_subtree", "")
  if not isinstance(subtree, str) or not subtree:
    msg = "port target [port] reference_subtree is required"
    raise ValueError(msg)
  return _import_reference_algo(subtree)


@pytest.fixture(scope="session")
def max_traces(port_wave: PortWaveConfig) -> int:
  """Tier 5 trace budget from ``[parity] max_traces``."""
  return int(port_wave.parity["max_traces"])


@pytest.fixture(autouse=True)
def _tier2_enable_x64(request: pytest.FixtureRequest) -> Iterator[None]:
  """Run tier_2 inside a scoped ``enable_x64()`` context."""
  if _nonport_wave() or request.node.get_closest_marker("tier_2") is None:
    yield
    return
  with _x64_context():
    yield


def pytest_configure(config: pytest.Config) -> None:
  if _nonport_wave():
    return
  config.addinivalue_line(
    "markers",
    f"timeout({TIER_TIMEOUT_SECONDS}): per-tier CPU budget",
  )


def pytest_collection_modifyitems(
  session: pytest.Session,
  config: pytest.Config,
  items: list[pytest.Item],
) -> None:
  del session, config
  if _nonport_wave():
    return
  port_items = [item for item in items if _is_port_item(item)]
  if not port_items:
    return
  for item in port_items:
    if item.get_closest_marker("port_wave") is None:
      msg = f"{item.nodeid} is under tests/port but has no port_wave marker"
      raise pytest.UsageError(msg)
  active = os.environ.get("AMINX_PORT_WAVE")
  governed = [item for item in port_items if active is None or _marker_wave(item) == active]
  if not governed:
    return
  port_config = load_port_target()
  wave_id = port_config["port"]["wave_id"]
  _, manifest_path = load_manifest(PORT_ROOT, str(wave_id))
  verify_manifest_hash(manifest_path)
  blocking = set(blocking_tiers_from_config(port_config))
  for item in governed:
    tier = _tier_marker_for_item(item)
    if tier is None:
      continue
    if tier not in blocking:
      item.add_marker(pytest.mark.skip(reason=f"{tier} skipped (ad_critical=false)"))
    item.add_marker(pytest.mark.timeout(TIER_TIMEOUT_SECONDS))
  ordered = sorted(governed, key=_tier_sort_key)
  slots = [index for index, item in enumerate(items) if item in governed]
  for index, item in zip(slots, ordered, strict=True):
    items[index] = item


def _marker_wave(item: pytest.Item) -> str | None:
  mark = item.get_closest_marker("port_wave")
  if mark is None or not mark.args:
    return None
  return str(mark.args[0])


def pytest_runtest_setup(item: pytest.Item) -> None:
  if _nonport_wave() or not _is_port_item(item):
    return
  tier = _tier_marker_for_item(item)
  if tier is None:
    return
  if tier in {"tier_2", "tier_3"}:
    wave = _session_wave(item)
    if bool(wave.capabilities.get("stochastic")):
      subtree = str(wave.port.get("reference_subtree", ""))
      oracle_module = _import_reference_algo(subtree)
      if not _has_injected_uniform(oracle_module):
        pytest.fail("stochastic wave requires an injected-uniform oracle key")
  gate = item.session.stash.get(_TIER_GATE_STASH_KEY, None)
  if gate is None:
    return
  failed_at = gate.get("failed_at")
  if failed_at is None:
    return
  failed_index = TIER_MARKERS.index(str(failed_at))
  current_index = TIER_MARKERS.index(tier)
  if current_index > failed_index:
    pytest.skip(f"blocked after {failed_at} failure")


def _audits_path() -> Path:
  override = os.environ.get("AMINX_PORT_AUDITS_PATH")
  if override:
    return Path(override)
  return REPO_ROOT / ".praxia" / "audits.jsonl"


def _extract_max_discrepancy(exc: BaseException | None) -> float | None:
  if exc is None:
    return None
  match = _MAX_ABS_DIFF_RE.search(str(exc))
  if match is None:
    return None
  try:
    return float(match.group(1))
  except ValueError:
    return None


def _emit_tier_result(
  *,
  item: pytest.Item,
  port_wave: PortWaveConfig,
  tier: str,
  status: str,
  error_taxonomy_class: str,
  max_discrepancy: float | None = None,
  traceback_excerpt: str = "",
) -> None:
  tolerance_policy = _policy_for_tier(port_wave.parity, tier)
  task_id = str(port_wave.manifest.get("manifest", {}).get("task_id", DEFAULT_TASK_ID))
  symbol_qualname = str(port_wave.port.get("symbol_qualname", ""))
  oracle_id = str(port_wave.port.get("oracle_id", ""))
  verdict = TierVerdict(
    status=status,  # type: ignore[arg-type]
    tolerance_policy=tolerance_policy,
    error_taxonomy_class=error_taxonomy_class,
    max_discrepancy=max_discrepancy,
  )
  evidence = Evidence(pytest_nodeid=item.nodeid, traceback_excerpt=traceback_excerpt)
  emit_tier_verdict(
    task_id=task_id,
    symbol_qualname=symbol_qualname,
    port_parity_tier=TIER_TO_EMIT[tier],
    oracle_id=oracle_id,
    tier_verdict=verdict,
    evidence=evidence,
    audits_path=_audits_path(),
  )


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
  outcome = yield
  if _nonport_wave() or not _is_port_item(item):
    return
  report = outcome.get_result()
  tier = _tier_marker_for_item(item)
  if tier is None or call.when != "call":
    return
  if _TIER_GATE_STASH_KEY not in item.session.stash:
    item.session.stash[_TIER_GATE_STASH_KEY] = {"failed_at": None, "completed": set()}
  gate = item.session.stash[_TIER_GATE_STASH_KEY]
  port_wave = _session_wave(item)
  if report.passed:
    gate["completed"].add(tier)
    _emit_tier_result(
      item=item,
      port_wave=port_wave,
      tier=tier,
      status="PASS",
      error_taxonomy_class="none",
    )
  elif report.failed:
    if gate["failed_at"] is None:
      gate["failed_at"] = tier
    tb_excerpt = ""
    discrepancy = None
    if call.excinfo is not None:
      tb_excerpt = str(call.excinfo.value)[:500]
      discrepancy = _extract_max_discrepancy(call.excinfo.value)
    _emit_tier_result(
      item=item,
      port_wave=port_wave,
      tier=tier,
      status="FAIL",
      error_taxonomy_class="numeric_drift",
      max_discrepancy=discrepancy,
      traceback_excerpt=tb_excerpt,
    )
