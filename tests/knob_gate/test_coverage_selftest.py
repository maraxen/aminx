"""One fixture per branch-coverage verdict in spec §7.2 (i)–(vii-b).

Removing a rule flips the fixture named in the module docstring:

* AssertionError-only kill: ``vii_raise`` would return ``pass``.
* AssertionError precedence over other failures: ``vii_b`` would return
  ``instrument_invalid``.
* Ancestor check: ``v_not_ancestor`` would return ``pass``.
* Clean/mutant separation: ``iv_pass`` would return ``fail`` (the mutant
  AssertionError would be read as a clean failure).
"""

from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import knob_gate._coverage as coverage_module
from knob_gate._closure import Closure
from knob_gate._coverage import CoolRun, check_branch_coverage

FIXTURES = Path(__file__).resolve().parents[1] / "port" / "selftest_coverage"
CASES = sorted(path.name for path in FIXTURES.iterdir() if (path / "case.json").is_file())


def _slugs(manifest_path: Path) -> list[str]:
  manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
  slugs: list[str] = []
  for row in manifest.get("branch", []):
    vehicle = row.get("vehicle", {})
    if vehicle.get("kind") == "sidecar":
      slugs.append(str(vehicle["slug"]))
  return slugs


def _materialize_run(case_dir: Path, case: dict[str, object], tmp_path: Path) -> CoolRun:
  manifest = tomllib.loads((case_dir / "manifest.toml").read_text(encoding="utf-8"))
  del manifest
  slugs = _slugs(case_dir / "manifest.toml")
  raw_run = case["run"]
  if not isinstance(raw_run, dict):
    msg = f"{case_dir.name}: case run must be an object"
    raise TypeError(msg)
  git_hash = str(raw_run["git_hash"])
  dir8 = str(case["dir8"]) if case.get("dir8") else git_hash[:8]
  output_paths: list[str] = []
  if slugs:
    slug = slugs[0]
    dest = tmp_path / slug / dir8 / "branch_controls.json"
    if case.get("write_controls", True):
      dest.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy(case_dir / "controls.json", dest)
    output_paths.append(str(dest))
  argv = raw_run["argv"]
  if not isinstance(argv, list):
    msg = f"{case_dir.name}: argv must be a list"
    raise TypeError(msg)
  return CoolRun(
    status=str(raw_run["status"]),
    outcome=str(raw_run["outcome"]),
    git_dirty=bool(raw_run["git_dirty"]),
    git_hash=git_hash,
    sidecar_sha256=str(raw_run["sidecar_sha256"]),
    argv=[str(part) for part in argv],
    output_paths=output_paths,
  )


def _write_cool_fragment(catalog: Path, run_id: str, run: CoolRun) -> None:
  dest = catalog / "runs" / "aminx"
  dest.mkdir(parents=True, exist_ok=True)
  table = pa.table(
    {
      "id": pa.array([run_id]),
      "status": pa.array([run["status"]]),
      "outcome": pa.array([run["outcome"]]),
      "git_dirty": pa.array([run["git_dirty"]]),
      "git_hash": pa.array([run["git_hash"]]),
      "sidecar_sha256": pa.array([run["sidecar_sha256"]]),
      "argv": pa.array([run["argv"]], type=pa.list_(pa.string())),
      "output_paths": pa.array([run["output_paths"]], type=pa.list_(pa.string())),
    }
  )
  pq.write_table(table, dest / f"run_{run_id}.parquet")


def _grade(case_dir: Path, case: dict[str, object], run: CoolRun, *, use_catalog: bool) -> str:
  registry = case.get("registry", {})
  if not isinstance(registry, dict):
    msg = f"{case_dir.name}: registry must be an object"
    raise TypeError(msg)
  digest = str(case.get("sidecar_digest", ""))
  changed = case.get("changed", [])
  if not isinstance(changed, list):
    msg = f"{case_dir.name}: changed must be a list"
    raise TypeError(msg)
  ancestor_ok = bool(case.get("ancestor", True))
  run_id = str(case["run_id"])

  def resolve_run(requested: str) -> CoolRun | None:
    if requested != run_id:
      return None
    return run

  kwargs: dict[str, object] = {
    "changed_paths": lambda _git_hash: [str(path) for path in changed],
    "is_ancestor": lambda _git_hash: ancestor_ok,
    "registry_sha256": lambda artifact: registry.get(str(artifact)),
    "sidecar_digest": lambda _slug: digest,
  }
  if "closure_files" in case:
    raw_files = case["closure_files"]
    if not isinstance(raw_files, list):
      msg = f"{case_dir.name}: closure_files must be a list"
      raise TypeError(msg)
    closure_files = frozenset(str(path) for path in raw_files)
    kwargs["closure_for"] = lambda slug: Closure(slug, closure_files)
  if not use_catalog:
    kwargs["resolve_run"] = resolve_run
  return check_branch_coverage(
    case_dir / "manifest.toml",
    case_dir / "outcomes.jsonl",
    case_dir / "ledger.toml",
    **kwargs,  # type: ignore[arg-type]
  )


@pytest.mark.redsox_gate
@pytest.mark.parametrize("case_name", CASES)
def test_coverage_verdict(
  case_name: str,
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  case_dir = FIXTURES / case_name
  case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
  run = _materialize_run(case_dir, case, tmp_path)
  use_catalog = bool(case.get("use_default_resolve", False))
  if use_catalog:
    catalog = tmp_path / "catalog"
    monkeypatch.setattr(coverage_module, "default_catalog_dir", lambda: catalog)
    assert coverage_module.default_catalog_dir() == catalog
    run_id = str(case["run_id"])
    _write_cool_fragment(catalog, run_id, run)
    from knob_gate._coverage import default_resolve_run

    loaded = default_resolve_run(run_id)
    assert loaded is not None
    for field in (
      "status",
      "outcome",
      "git_dirty",
      "git_hash",
      "sidecar_sha256",
      "argv",
      "output_paths",
    ):
      assert loaded[field] == run[field]
    assert _grade(case_dir, case, run, use_catalog=True) == case["expected"]
    (catalog / "runs" / "aminx" / f"run_{run_id}.parquet").unlink()
    assert _grade(case_dir, case, run, use_catalog=True) == "instrument_invalid"
    return
  assert _grade(case_dir, case, run, use_catalog=False) == case["expected"]
