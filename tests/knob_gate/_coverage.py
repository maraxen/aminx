"""Branch-coverage verdict for redsox pytest vehicles and sidecar ledgers.

Implements spec §0 and §6.6 steps 1b and 1c. A row is killed only by an
``AssertionError`` on a ``when=='call'`` record; any other mutant failure is
``instrument_invalid`` unless an AssertionError kill is also present.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tomllib
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, TypedDict, cast

try:  # imported as knob_gate._coverage (pytest) or as bare _coverage (scripts/redsox)
  from ._closure import Closure, compute_closure, row_is_stale
except ImportError:
  from _closure import Closure, compute_closure, row_is_stale  # type: ignore[no-redef]

Verdict =Literal["pass", "fail", "instrument_invalid"]

_SCOPED_PREFIXES = (
  "src/aminx/",
  "scripts/parity/",
  "scripts/recapture/",
  "tests/port/",
  "aminx-oracles/",
)
_SCOPED_FILES = frozenset({"pyproject.toml", "uv.lock"})
_RUN_FIELDS = (
  "status",
  "outcome",
  "git_dirty",
  "git_hash",
  "sidecar_sha256",
  "argv",
  "output_paths",
)
_WEIGHT_PREFIXES = ("pottsmpnn_", "lasermpnn_", "protonpottsmpnn_")


class OutcomeRecord(TypedDict):
  nodeid: str
  when: str
  outcome: str
  wasxfail: bool
  mutant: str | None
  exc_type: str | None


class CoolRun(TypedDict):
  status: str
  outcome: str
  git_dirty: bool
  git_hash: str
  sidecar_sha256: str
  argv: list[str]
  output_paths: list[str]


def repo_root() -> Path:
  """Nearest ancestor of this file that contains pyproject.toml."""
  start = Path(__file__).resolve()
  for directory in start.parents:
    if (directory / "pyproject.toml").is_file():
      return directory
  return start.parents[2]


def default_catalog_dir() -> Path:
  """aminx ``.bth.toml`` ``catalog_dir``, else ``~/.bth/catalog``."""
  start = Path.cwd()
  for directory in (start, *start.parents):
    candidate = directory / ".bth.toml"
    if not candidate.is_file():
      continue
    data = tomllib.loads(candidate.read_text(encoding="utf-8"))
    project = data.get("project", {})
    if isinstance(project, dict) and "catalog_dir" in project:
      return Path(str(project["catalog_dir"])).expanduser()
    break
  return Path.home() / ".bth" / "catalog"


def default_resolve_run(run_id: str) -> CoolRun | None:
  """Read the cool-tier fragment ``<catalog>/runs/aminx/run_<id>.parquet``."""
  import pyarrow.parquet as pq

  path = default_catalog_dir() / "runs" / "aminx" / f"run_{run_id}.parquet"
  if not path.is_file():
    return None
  table = pq.read_table(path)
  if table.num_rows < 1:
    return None
  if any(name not in table.column_names for name in _RUN_FIELDS):
    return None
  values = {name: table.column(name)[0].as_py() for name in _RUN_FIELDS}
  argv = values["argv"]
  output_paths = values["output_paths"]
  if not isinstance(argv, list) or not isinstance(output_paths, list):
    return None
  return CoolRun(
    status=str(values["status"]),
    outcome=str(values["outcome"]),
    git_dirty=bool(values["git_dirty"]),
    git_hash=str(values["git_hash"]),
    sidecar_sha256=str(values["sidecar_sha256"]),
    argv=[str(part) for part in argv],
    output_paths=[str(part) for part in output_paths],
  )


def default_changed_paths(git_hash: str) -> list[str]:
  """``git diff --name-only <hash>..HEAD``."""
  completed = subprocess.run(
    ["git", "diff", "--name-only", f"{git_hash}..HEAD"],
    check=False,
    capture_output=True,
    text=True,
    cwd=repo_root(),
  )
  if completed.returncode != 0:
    msg = completed.stderr.strip() or "git diff failed"
    raise RuntimeError(msg)
  return [line for line in completed.stdout.splitlines() if line]


def default_is_ancestor(git_hash: str) -> bool:
  """``git merge-base --is-ancestor <hash> HEAD``."""
  completed = subprocess.run(
    ["git", "merge-base", "--is-ancestor", git_hash, "HEAD"],
    check=False,
    capture_output=True,
    text=True,
    cwd=repo_root(),
  )
  return completed.returncode == 0


def default_sidecar_digest(slug: str) -> str:
  """SHA-256 of ``scripts/parity/<slug>.bth.toml`` at HEAD (raw bytes)."""
  path = repo_root() / "scripts" / "parity" / f"{slug}.bth.toml"
  return hashlib.sha256(path.read_bytes()).hexdigest()


def default_registry_sha256(artifact_path: str) -> str | None:
  """Checkpoint-registry ``sha256`` for ``artifact_path``, if a registry lists it."""
  found: dict[str, str] = {}
  root = repo_root()
  for path in root.glob("**/checkpoint_registry.json"):
    if any(part in {".venv", ".git"} for part in path.parts):
      continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
      continue
    for entry in entries:
      if not isinstance(entry, dict):
        continue
      sha = entry.get("sha256")
      artifact = entry.get("artifact_path")
      if isinstance(sha, str) and isinstance(artifact, str):
        found[artifact] = sha
  return found.get(artifact_path)


def _load_outcomes(path: Path) -> list[OutcomeRecord]:
  records: list[OutcomeRecord] = []
  if not path.is_file():
    return records
  for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    raw = json.loads(line)
    if not isinstance(raw, dict):
      continue
    mutant = raw.get("mutant")
    exc_type = raw.get("exc_type")
    records.append(
      OutcomeRecord(
        nodeid=str(raw.get("nodeid", "")),
        when=str(raw.get("when", "")),
        outcome=str(raw.get("outcome", "")),
        wasxfail=bool(raw.get("wasxfail", False)),
        mutant=str(mutant) if isinstance(mutant, str) and mutant else None,
        exc_type=str(exc_type) if isinstance(exc_type, str) and exc_type else None,
      )
    )
  return records


def _clean_state(records: Sequence[OutcomeRecord], nodeids: Sequence[str]) -> str:
  """Return ``pass``, ``fail``, or ``invalid`` for mutant-is-None records."""
  failed = False
  for nodeid in nodeids:
    scoped = [row for row in records if row["mutant"] is None and row["nodeid"] == nodeid]
    if not scoped:
      return "invalid"
    real_failures = [
      row for row in scoped if row["outcome"] in {"failed", "error"} and not row["wasxfail"]
    ]
    if real_failures:
      failed = True
      continue
    if any(row["wasxfail"] for row in scoped):
      return "invalid"
    passed_call = any(row["when"] == "call" and row["outcome"] == "passed" for row in scoped)
    if not passed_call:
      return "invalid"
  if failed:
    return "fail"
  return "pass"


def _mutant_state(
  records: Sequence[OutcomeRecord],
  nodeids: Sequence[str],
  mutant_id: str,
) -> str:
  """Return ``killed``, ``invalid``, or ``missing``.

  An AssertionError on any vehicle call kills the row. That kill takes
  precedence over a non-assertion failure on another id. A passing mutant, a
  missing record, or a non-assertion failure with no AssertionError kill is
  ``invalid`` (the caller maps ``missing`` to ``instrument_invalid`` too).
  """
  nodeid_set = set(nodeids)
  scoped = [row for row in records if row["mutant"] == mutant_id and row["nodeid"] in nodeid_set]
  if not scoped:
    return "missing"
  calls = [row for row in scoped if row["when"] == "call" and not row["wasxfail"]]
  assertion_kill = any(
    row["outcome"] in {"failed", "error"} and row["exc_type"] == "AssertionError" for row in calls
  )
  if assertion_kill:
    return "killed"
  other_failure = any(row["outcome"] in {"failed", "error"} for row in calls)
  if other_failure:
    return "invalid"
  return "invalid"


def default_closure_for(slug: str) -> Closure | None:
  """The row's import closure, or ``None`` when it cannot be computed.

  ``None`` means "fall back to the global prefix test", so any failure here (no
  vehicle script, a malformed ``closure_edges.toml``, a file that does not parse)
  can only make a row stale. It never makes one fresh.
  """
  try:
    return compute_closure(repo_root(), slug)
  except (FileNotFoundError, ValueError, SyntaxError, OSError):
    return None


def _touches_scoped(paths: Sequence[str]) -> bool:
  """The GLOBAL prefix test. Rows use ``row_is_stale``; this is the fallback's shape."""
  for raw in paths:
    path = raw.replace("\\", "/").lstrip("./")
    if path in _SCOPED_FILES:
      return True
    for prefix in _SCOPED_PREFIXES:
      if path.startswith(prefix):
        return True
  return False


def _argv_mutants(argv: Sequence[str]) -> set[str] | None:
  try:
    index = list(argv).index("--mutants")
  except ValueError:
    return None
  if index + 1 >= len(argv):
    return None
  return {part for part in argv[index + 1].split(",") if part}


def _weights_required(slug: str) -> bool:
  return slug.startswith(_WEIGHT_PREFIXES)


def _validate_sidecar(
  *,
  slug: str,
  row_ids: set[str],
  ledger: Mapping[str, object],
  resolve_run: Callable[[str], CoolRun | None],
  changed_paths: Callable[[str], Sequence[str]],
  is_ancestor: Callable[[str], bool],
  registry_sha256: Callable[[str], str | None],
  sidecar_digest: Callable[[str], str],
  closure_for: Callable[[str], Closure | None],
) -> bool:
  """Return True when every step-1c assertion holds for this slug."""
  sidecar = _object_dict(ledger.get("sidecar"))
  if sidecar is None or slug not in sidecar:
    return False
  entry = _object_dict(sidecar[slug])
  if entry is None:
    return False
  run_id = entry.get("bth_run_id")
  if not isinstance(run_id, str) or not run_id:
    return False
  run = resolve_run(run_id)
  if run is None:
    return False
  if run["status"] != "completed":
    return False
  if run["outcome"] != "pass":
    return False
  if run["git_dirty"]:
    return False
  git_hash = run["git_hash"]
  if not git_hash or not is_ancestor(git_hash):
    return False
  if row_is_stale(changed_paths(git_hash), closure_for(slug)):
    return False
  try:
    expected_digest = sidecar_digest(slug)
  except OSError:
    return False
  if run["sidecar_sha256"] != expected_digest:
    return False
  listed = _argv_mutants(run["argv"])
  if listed is None or listed != row_ids:
    return False
  suffix = f"/{slug}/{git_hash[:8]}/branch_controls.json"
  matches = [path for path in run["output_paths"] if path.replace("\\", "/").endswith(suffix)]
  if not matches:
    return False
  controls_path = Path(matches[0])
  if not controls_path.is_file():
    return False
  controls = json.loads(controls_path.read_text(encoding="utf-8"))
  if not isinstance(controls, dict):
    return False
  if controls.get("clean") != "pass":
    return False
  mutants = controls.get("mutants")
  if not isinstance(mutants, dict):
    return False
  if set(mutants) != row_ids:
    return False
  if any(value != "failed" for value in mutants.values()):
    return False
  weights = controls.get("weights", {})
  if not isinstance(weights, dict):
    return False
  if _weights_required(slug) and not weights:
    return False
  for artifact, sha in weights.items():
    if not isinstance(artifact, str) or not isinstance(sha, str):
      return False
    if registry_sha256(artifact) != sha:
      return False
  return True


def _object_dict(value: object) -> dict[str, object] | None:
  """Narrow a TOML value to ``dict[str, object]`` for ty."""
  if not isinstance(value, dict):
    return None
  return cast(dict[str, object], value)


def _branch_rows(manifest: Mapping[str, object]) -> list[dict[str, object]]:
  rows = manifest.get("branch", [])
  if not isinstance(rows, list) or not rows:
    return []
  typed: list[dict[str, object]] = []
  for row in rows:
    if isinstance(row, dict):
      typed.append(cast(dict[str, object], row))
  return typed


def check_branch_coverage(
  manifest_path: str | Path,
  outcomes_path: str | Path,
  ledger_path: str | Path,
  *,
  resolve_run: Callable[[str], CoolRun | None] | None = None,
  changed_paths: Callable[[str], Sequence[str]] | None = None,
  is_ancestor: Callable[[str], bool] | None = None,
  registry_sha256: Callable[[str], str | None] | None = None,
  sidecar_digest: Callable[[str], str] | None = None,
  closure_for: Callable[[str], Closure | None] | None = None,
) -> Verdict:
  """Grade a branch manifest against outcomes and the sidecar ledger.

  ``resolve_run``, ``changed_paths``, and ``is_ancestor`` default to the
  cool-tier parquet reader, ``git diff --name-only``, and
  ``git merge-base --is-ancestor``. ``registry_sha256`` and ``sidecar_digest``
  default to the checkpoint registry and ``scripts/parity/<slug>.bth.toml``.
  """
  resolve = resolve_run or default_resolve_run
  changed = changed_paths or default_changed_paths
  ancestor = is_ancestor or default_is_ancestor
  registry = registry_sha256 or default_registry_sha256
  digest = sidecar_digest or default_sidecar_digest
  closure = closure_for or default_closure_for

  manifest_file = Path(manifest_path)
  if not manifest_file.is_file():
    return "instrument_invalid"
  manifest = tomllib.loads(manifest_file.read_text(encoding="utf-8"))
  rows = _branch_rows(manifest)
  if not rows:
    return "instrument_invalid"
  outcomes = _load_outcomes(Path(outcomes_path))

  invalid = False
  clean_fail = False
  sidecar_ids: dict[str, set[str]] = defaultdict(set)

  for row in rows:
    vehicle = _object_dict(row.get("vehicle"))
    if vehicle is None:
      invalid = True
      continue
    kind = vehicle.get("kind")
    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id:
      invalid = True
      continue
    if kind == "pytest":
      nodeids = vehicle.get("nodeids")
      if not isinstance(nodeids, list) or not all(isinstance(node, str) for node in nodeids):
        invalid = True
        continue
      typed_ids = cast(list[str], nodeids)
      clean = _clean_state(outcomes, typed_ids)
      if clean == "fail":
        clean_fail = True
      elif clean != "pass":
        invalid = True
      mutant = _mutant_state(outcomes, typed_ids, row_id)
      if mutant != "killed":
        invalid = True
    elif kind == "sidecar":
      slug = vehicle.get("slug")
      if not isinstance(slug, str) or not slug:
        invalid = True
        continue
      sidecar_ids[slug].add(row_id)
    else:
      invalid = True

  if sidecar_ids:
    ledger_file = Path(ledger_path)
    if not ledger_file.is_file():
      invalid = True
    else:
      ledger = tomllib.loads(ledger_file.read_text(encoding="utf-8"))
      for slug, row_ids in sidecar_ids.items():
        ok = _validate_sidecar(
          slug=slug,
          row_ids=row_ids,
          ledger=ledger,
          resolve_run=resolve,
          changed_paths=changed,
          is_ancestor=ancestor,
          registry_sha256=registry,
          sidecar_digest=digest,
          closure_for=closure,
        )
        if not ok:
          invalid = True

  if invalid:
    return "instrument_invalid"
  if clean_fail:
    return "fail"
  return "pass"
