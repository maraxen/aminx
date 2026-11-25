"""Triage each prereg-order violation: did the criteria a pre-commit run graded against differ?

Task 261124_prereg-violation-triage. Follows run a5275269 (prereg_order_audit_v2, outcome
prereg_order_violated: 11 sidecars committed after a run that cites them).

A late commit breaks "committed before running" as written, which git settles. It does NOT
settle whether the criteria were chosen with the numbers in view: the sidecar may have existed,
uncommitted, with the same [outcomes] the whole time. bathos records sha256 of the sidecar's raw
bytes on every run (bathos prereg.py resolve_sidecar; runner.py explicit --sidecar path), so the
criteria a run graded against are recoverable whenever those bytes exist somewhere in git:

1. as any committed version of the sidecar (``git log --all``), or
2. inside the run's own dirty-run snapshot commit (``git show <run git_hash>:<path>``),

and are otherwise UNRECOVERABLE, which is reported as such and never guessed.

The comparison is against the FIRST committed version (the one that registered the sidecar).
"Criteria" means the parsed [outcomes] and [result_schema] tables; hypothesis wording, comments
and metadata may differ without changing what the run was graded on.

Per sidecar, over its PRE-COMMIT runs only:
  criteria_edited_before_commit -- some pre-commit run graded against different criteria
  late_commit_same_criteria     -- every pre-commit run graded against the committed criteria
  unrecoverable                 -- none differ, but at least one run's criteria cannot be read
Also reported: whether a run AFTER the commit graded against the committed criteria (a clean
re-run, which is what makes an outcome citable again under the audit's own decision rule).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("prereg_violation_triage")

_CRITERIA_KEYS = ("outcomes", "result_schema")


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
  return subprocess.run(cmd, cwd=cwd, capture_output=True, check=False)  # noqa: S603


def _git_text(repo: Path, *args: str) -> tuple[int, str]:
  proc = _run(["git", *args], cwd=repo)
  return proc.returncode, proc.stdout.decode("utf-8", "replace").strip()


def _git_blob(repo: Path, rev: str, rel: str) -> bytes | None:
  proc = _run(["git", "show", f"{rev}:{rel}"], cwd=repo)
  return proc.stdout if proc.returncode == 0 else None


def _sha(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def to_utc(value: str) -> datetime:
  parsed = datetime.fromisoformat(str(value).strip().replace(" ", "T", 1))
  if parsed.tzinfo is None or parsed.utcoffset() is None:
    msg = f"naive timestamp {value!r}: refusing to assume an offset"
    raise ValueError(msg)
  return parsed.astimezone(timezone.utc)


def repo_relative(path: str) -> str | None:
  parts = Path(path).parts
  for anchor in ("scripts", "tests", "src"):
    if anchor in parts:
      return str(Path(*parts[parts.index(anchor) :]))
  return None


def add_commit(repo: Path, rel: str) -> tuple[str, datetime] | None:
  """(sha, AUTHOR date) of the commit that added rel -- the audit's own definition."""
  code, out = _git_text(repo, "log", "--follow", "--diff-filter=A", "--format=%H %aI", "--", rel)
  if code != 0 or not out:
    return None
  sha, _, when = out.splitlines()[-1].partition(" ")
  return sha, to_utc(when)


def committed_blobs(repo: Path, rel: str) -> dict[str, str]:
  """sha256(bytes) -> commit, for every version of rel on any ref."""
  code, out = _git_text(repo, "log", "--all", "--format=%H", "--", rel)
  found: dict[str, str] = {}
  if code != 0:
    return found
  for commit in out.splitlines():
    blob = _git_blob(repo, commit, rel)
    if blob is not None:
      found.setdefault(_sha(blob), commit)
  return found


def criteria(data: bytes) -> dict[str, Any] | None:
  try:
    parsed = tomllib.loads(data.decode("utf-8"))
  except (tomllib.TOMLDecodeError, UnicodeDecodeError):
    return None
  return {k: parsed.get(k) for k in _CRITERIA_KEYS}


def criteria_delta(run: dict[str, Any], first: dict[str, Any]) -> list[str]:
  """Names of outcome labels / schema fields that were added, removed or changed."""
  delta: list[str] = []
  for key in _CRITERIA_KEYS:
    a, b = run.get(key) or {}, first.get(key) or {}
    for name in sorted(set(a) | set(b)):
      if a.get(name) != b.get(name):
        if name not in a:
          kind = "added_at_commit"
        elif name not in b:
          kind = "removed_at_commit"
        else:
          kind = "changed"
        delta.append(f"{key}.{name}:{kind}")
  return delta


def classify_run(
  repo: Path, rel: str, run: dict[str, Any], blobs: dict[str, str], first: bytes,
) -> dict[str, Any]:
  sha = (run.get("sidecar_sha256") or "").strip().lower()
  content: bytes | None = None
  source = "none"
  if not sha:
    source = "no_hash_recorded"
  elif sha in blobs:
    content, source = _git_blob(repo, blobs[sha], rel), "committed_version"
  elif run.get("git_hash"):
    snap = _git_blob(repo, run["git_hash"], rel)
    if snap is not None and _sha(snap) == sha:
      content, source = snap, "run_snapshot"
  out = {"id": run["id"], "timestamp": run["timestamp"], "outcome": run.get("outcome"),
         "sidecar_sha256": sha[:12], "content_source": source, "delta": []}
  if content is None:
    out["criteria_vs_first_commit"] = "unrecoverable"
  elif content == first:
    out["criteria_vs_first_commit"] = "identical_bytes"
  else:
    a, b = criteria(content), criteria(first)
    if a is None or b is None:
      out["criteria_vs_first_commit"] = "unparseable"
    elif a == b:
      out["criteria_vs_first_commit"] = "criteria_same"
    else:
      out["criteria_vs_first_commit"] = "criteria_differ"
      out["delta"] = criteria_delta(a, b)
  return out


def triage_sidecar(repo: Path, rel: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
  added = add_commit(repo, rel)
  first = _git_blob(repo, added[0], rel) if added else None
  if added is None or first is None:
    return {"sidecar": rel, "class": "no_add_commit", "pre_commit_runs": [],
            "post_commit_runs": [], "clean_rerun_after_commit": False}
  commit, when = added
  blobs = committed_blobs(repo, rel)
  pre, post = [], []
  for run in sorted(runs, key=lambda r: to_utc(r["timestamp"])):
    row = classify_run(repo, rel, run, blobs, first)
    (pre if to_utc(run["timestamp"]) < when else post).append(row)
  states = {r["criteria_vs_first_commit"] for r in pre}
  if "criteria_differ" in states:
    cls = "criteria_edited_before_commit"
  elif not pre:
    cls = "no_pre_commit_run"
  elif states <= {"identical_bytes", "criteria_same"}:
    cls = "late_commit_same_criteria"
  else:
    cls = "unrecoverable"
  clean = [r for r in post if r["criteria_vs_first_commit"] in ("identical_bytes", "criteria_same")]
  return {
    "sidecar": rel, "class": cls, "add_commit": commit[:12], "added_at": when.isoformat(),
    "first_version_sha256": _sha(first)[:12], "n_committed_versions": len(blobs),
    "pre_commit_runs": pre, "post_commit_runs": post,
    "clean_rerun_after_commit": bool(clean),
    "clean_rerun_outcomes": [r["outcome"] for r in clean],
  }


# ---------------------------------------------------------------------------------------------
# Self-test: a throwaway repo where every class is planted, both must-fire and must-not-fire.

_SIDE = "scripts/x/probe.bth.toml"
_BASE = (
  '[experiment]\nhypothesis = "h1"\n\n[outcomes.pass]\ncondition = "x > 1"\n'
  'is_residual = false\n\n[outcomes.other]\ncondition = "NOT x > 1"\nis_residual = true\n\n'
  '[result_schema]\nx = "float"\n'
)


def _git_ok(repo: Path, env: dict[str, str], *args: str, stdin: bytes | None = None) -> str:
  proc = subprocess.run(  # noqa: S603
    ["git", *args], cwd=repo, input=stdin, capture_output=True, check=True, env=env)
  return proc.stdout.decode().strip()


def _snapshot(repo: Path, text: str, env: dict[str, str]) -> str:
  """A dangling commit holding `text` at _SIDE -- like bathos's dirty-run snapshot (no ref)."""
  blob = _git_ok(repo, env, "hash-object", "-w", "--stdin", stdin=text.encode())
  Path(env["GIT_INDEX_FILE"]).unlink(missing_ok=True)
  _git_ok(repo, env, "update-index", "--add", "--cacheinfo", f"100644,{blob},{_SIDE}")
  tree = _git_ok(repo, env, "write-tree")
  return _git_ok(repo, env, "commit-tree", tree, "-m", "snap")


def _self_test() -> dict[str, Any]:
  failed: list[str] = []
  checks = 0
  with tempfile.TemporaryDirectory() as tmp:
    repo = Path(tmp) / "r"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "GIT_AUTHOR_DATE": "2026-01-02T00:00:00+00:00",
           "GIT_COMMITTER_DATE": "2026-01-02T00:00:00+00:00"}
    env.pop("GIT_INDEX_FILE", None)
    env.pop("GIT_DIR", None)
    _git_ok(repo, env, "init", "-q")
    (repo / _SIDE).parent.mkdir(parents=True)
    (repo / _SIDE).write_text(_BASE)
    _git_ok(repo, env, "add", _SIDE)
    _git_ok(repo, env, "-c", "commit.gpgsign=false", "commit", "-qm", "add")
    snap_env = {**env, "GIT_INDEX_FILE": str(Path(tmp) / "snap.idx")}
    reworded = _BASE.replace('"h1"', '"h1, reworded after the run"')
    loosened = _BASE.replace('"x > 1"', '"x > 0"')
    snap_same = _snapshot(repo, reworded, snap_env)
    snap_diff = _snapshot(repo, loosened, snap_env)
    early, late = "2026-01-01T00:00:00+00:00", "2026-01-03T00:00:00+00:00"

    def run(i: str, ts: str, text: str | None, git_hash: str = "") -> dict[str, Any]:
      return {"id": i, "timestamp": ts, "outcome": "pass", "git_hash": git_hash,
              "sidecar_sha256": _sha(text.encode()) if text is not None else ""}

    cases = [
      # (name, runs, expected class, expected clean re-run)
      ("same_bytes_must_not_fire", [run("a", early, _BASE)], "late_commit_same_criteria", False),
      ("reworded_must_not_fire", [run("b", early, reworded, snap_same)],
       "late_commit_same_criteria", False),
      ("loosened_must_fire", [run("c", early, loosened, snap_diff)],
       "criteria_edited_before_commit", False),
      ("loosened_beats_same", [run("d", early, _BASE), run("e", early, loosened, snap_diff)],
       "criteria_edited_before_commit", False),
      ("unrecoverable_not_guessed", [run("f", early, "never stored")], "unrecoverable", False),
      ("no_hash_not_guessed", [run("g", early, None)], "unrecoverable", False),
      ("clean_rerun_detected", [run("h", early, loosened, snap_diff), run("i", late, _BASE)],
       "criteria_edited_before_commit", True),
    ]
    for name, runs, want_cls, want_clean in cases:
      got = triage_sidecar(repo, _SIDE, runs)
      checks += 1
      if got["class"] != want_cls or got.get("clean_rerun_after_commit") != want_clean:
        failed.append(f"{name}: got {got['class']}/{got.get('clean_rerun_after_commit')}")
    checks += 1
    got = triage_sidecar(repo, _SIDE, [run("c", early, loosened, snap_diff)])
    delta = got["pre_commit_runs"][0]["delta"] if got["pre_commit_runs"] else None
    if delta != ["outcomes.pass:changed"]:
      failed.append(f"delta_names_label: {delta}")
  return {"self_test_passed": not failed, "self_test_n_checks": checks,
          "self_test_failed": failed}


# ---------------------------------------------------------------------------------------------


def _row(r: list[Any]) -> dict[str, Any]:
  return {"id": r[0], "sidecar_path": r[1], "timestamp": r[2], "sidecar_sha256": r[3] or "",
          "outcome": r[4], "git_hash": r[5] or ""}


def local_rows(bth: str, project: str, rels: set[str]) -> list[dict[str, Any]]:
  sql = ("SELECT id, sidecar_path, timestamp, sidecar_sha256, outcome, git_hash FROM runs "
         f"WHERE project_slug = '{project}' AND sidecar_path IS NOT NULL")
  proc = _run([bth, "sql", sql])
  if proc.returncode != 0:
    return []
  rows = json.loads(proc.stdout).get("rows", [])
  return [_row(r) for r in rows if repo_relative(r[1]) in rels]


def _emit(out: Path, payload: dict[str, Any]) -> None:
  out.parent.mkdir(parents=True, exist_ok=True)
  text = json.dumps(payload, indent=2, sort_keys=True)
  out.write_text(text, encoding="utf-8")
  results = os.environ.get("BTH_RESULTS_PATH")
  if results:
    Path(results).write_text(text, encoding="utf-8")


def main() -> int:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--audit-result", type=Path, required=True,
                      help="result.json of the prereg_order_audit_v2 run being triaged")
  parser.add_argument("--extra-rows", type=Path, action="append", default=[],
                      help="JSON {rows:[[id, sidecar_path, timestamp, sidecar_sha256, outcome, "
                      "git_hash], ...]} exported from another machine's catalog")
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--bth", default="bth")
  parser.add_argument("--project", default="aminx")
  parser.add_argument("--self-test-only", action="store_true",
                      help="run the planted-class self-test, read no real data, write nothing")
  args = parser.parse_args()

  base = _self_test()
  _LOG.info("self-test %s (%d checks) %s", "passed" if base["self_test_passed"] else "FAILED",
            base["self_test_n_checks"], base["self_test_failed"])
  if args.self_test_only:
    return 0 if base["self_test_passed"] else 1
  empty = {"n_triaged": 0, "n_expected": 0, "n_criteria_edited": 0,
           "n_late_commit_same_criteria": 0, "n_unrecoverable": 0, "n_other": 0,
           "n_clean_rerun": 0, "n_local_rows": 0, "n_rows": 0, "sidecars": [],
           "audit_result_sha256": ""}
  if not base["self_test_passed"]:
    _emit(args.out, {**base, **empty, "refusal": "self_test_failed"})
    return 1
  if not args.audit_result.is_file():
    _emit(args.out, {**base, **empty, "refusal": "audit_result_missing"})
    return 1
  audit_bytes = args.audit_result.read_bytes()
  violations = json.loads(audit_bytes)["violations"]
  rels = {v["sidecar"] for v in violations}

  rows = local_rows(args.bth, args.project, rels)
  n_local = len(rows)
  for extra in args.extra_rows:
    rows += [_row(r) for r in json.loads(extra.read_text())["rows"]
             if repo_relative(r[1]) in rels]
  by_id = {r["id"]: r for r in rows}
  sidecars = []
  for rel in sorted(rels):
    runs = [r for r in by_id.values() if repo_relative(r["sidecar_path"]) == rel]
    sidecars.append(triage_sidecar(args.repo, rel, runs))
  counts = {c: sum(1 for s in sidecars if s["class"] == c) for c in (
    "criteria_edited_before_commit", "late_commit_same_criteria", "unrecoverable")}
  payload = {
    **base, "refusal": None if n_local else "catalog_unavailable",
    "audit_result_sha256": _sha(audit_bytes), "n_expected": len(violations),
    "n_local_rows": n_local, "n_rows": len(by_id), "n_triaged": len(sidecars),
    "n_criteria_edited": counts["criteria_edited_before_commit"],
    "n_late_commit_same_criteria": counts["late_commit_same_criteria"],
    "n_unrecoverable": counts["unrecoverable"],
    "n_other": len(sidecars) - sum(counts.values()),
    "n_clean_rerun": sum(1 for s in sidecars if s.get("clean_rerun_after_commit")),
    "sidecars": sidecars,
  }
  _emit(args.out, payload)
  for s in sidecars:
    _LOG.info("%-60s %s clean_rerun=%s", s["sidecar"], s["class"],
              s.get("clean_rerun_after_commit"))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
