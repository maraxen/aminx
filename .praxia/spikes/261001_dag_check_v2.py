"""Mechanical DAG check for the 261001 ecosystem spec set, no-rename revision.

task_id 261001_aminx-hub-ecosystem-specs. Stdlib only. Throwaway spike (it produces the plan doc's section 8
table, not a research finding).

Inputs:
  261001_dag_given_items.json   authoritative item list handed over by the workflow (no-rename revision)
  261001_dag_given_edges.txt    the previous (rename-era) authoritative list, for the retired/added diff
  .praxia/docs/specs/261001_*.md one ```toml block of [[item]] tables per spec (read for comparison only)

Checks: duplicate ids, dangling depends_on, self-dependencies, acyclicity (Kahn), forbidden retired names
(molxmpnn, mpnnx) in any item field, list-vs-spec-toml differences. Emits a deterministic topological order,
the critical path by item count and by size weight (S=1 M=2 L=3), roots with descendant counts, a mermaid
flowchart and the topo table. --controls plants one defect per check and asserts each is detected.

Usage:
  uv run --no-project python3 .praxia/spikes/261001_dag_check_v2.py --emit-dir .praxia/spikes/261001_dag_out_v2
  uv run --no-project python3 .praxia/spikes/261001_dag_check_v2.py --controls
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import re
import tomllib
from collections import Counter
from pathlib import Path

log = logging.getLogger("dag_check_v2")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
W = {"S": 1, "M": 2, "L": 3}
FORBIDDEN = re.compile(r"molxmpnn|mpnnx", re.I)
FIELDS = ("title", "repo", "size", "depends_on", "user_decision")
# S3-28 is the assembly check whose job is to forbid the retired name, so its title names it; allow-listed.
FORBIDDEN_ALLOW = {"S3-28"}


def key(iid: str) -> tuple[int, int]:
    s, n = iid[1:].split("-")
    return int(s), int(n)


def load_given(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def load_old_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln.split("|")[0] for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def load_toml() -> tuple[list[dict], dict[str, int]]:
    items, nblocks = [], {}
    for f in sorted(glob.glob(str(ROOT / ".praxia/docs/specs/261001_*.md"))):
        text = Path(f).read_text()
        blocks = re.findall(r"```toml\n(.*?)```", text, re.S)
        blocks = [b for b in blocks if "[[item]]" in b]
        nblocks[Path(f).name] = len(blocks)
        if len(blocks) != 1:
            continue
        try:
            parsed = tomllib.loads(blocks[0])
        except tomllib.TOMLDecodeError as e:
            log.warning("%s: toml parse error: %s", f, e)
            nblocks[Path(f).name] = -1
            continue
        for it in parsed.get("item", []):
            it["_spec"] = Path(f).name
            items.append(it)
    return items, nblocks


def compare(given: list[dict], toml: list[dict]) -> list[str]:
    diffs = []
    g = {i["id"]: i for i in given}
    t = {i["id"]: i for i in toml}
    for iid in sorted(set(g) - set(t), key=key):
        diffs.append(f"{iid}: in authoritative list, not in spec toml")
    for iid in sorted(set(t) - set(g), key=key):
        diffs.append(f"{iid}: in spec toml ({t[iid]['_spec']}), not in authoritative list")
    for iid in sorted(set(g) & set(t), key=key):
        for k in ("repo", "size", "user_decision", "title"):
            if g[iid][k] != t[iid].get(k):
                diffs.append(f"{iid}.{k} differs")
        if sorted(g[iid]["depends_on"]) != sorted(t[iid].get("depends_on", [])):
            diffs.append(
                f"{iid}.depends_on: list={sorted(g[iid]['depends_on'])} toml={sorted(t[iid].get('depends_on', []))}"
            )
    return diffs


def check(items: list[dict]) -> dict:
    ids = [i["id"] for i in items]
    dup = sorted({x for x in ids if ids.count(x) > 1})
    by = {i["id"]: i for i in items}
    dangling = sorted((i["id"], d) for i in items for d in i["depends_on"] if d not in by)
    self_dep = sorted(i["id"] for i in items if i["id"] in i["depends_on"])
    forbidden = sorted(
        {
            i["id"]
            for i in items
            for k in FIELDS
            if i["id"] not in FORBIDDEN_ALLOW and FORBIDDEN.search(json.dumps(i.get(k, "")))
        }
    )
    indeg = {x: 0 for x in by}
    succ: dict[str, list[str]] = {x: [] for x in by}
    for i in by.values():
        for d in i["depends_on"]:
            if d in by:
                indeg[i["id"]] += 1
                succ[d].append(i["id"])
    ready = sorted([x for x in by if indeg[x] == 0], key=key)
    order: list[str] = []
    while ready:
        x = ready.pop(0)
        order.append(x)
        for y in succ[x]:
            indeg[y] -= 1
            if indeg[y] == 0:
                ready.append(y)
        ready.sort(key=key)
    cyclic = sorted(set(by) - set(order), key=key)
    best: dict[str, tuple[int, int, list[str]]] = {}
    bw: dict[str, tuple[int, list[str]]] = {}
    for x in order:
        deps = [d for d in by[x]["depends_on"] if d in by]
        w = W[by[x]["size"]]
        if deps:
            p = max(deps, key=lambda d: (best[d][0], best[d][1], [-k for k in key(d)]))
            best[x] = (best[p][0] + 1, best[p][1] + w, best[p][2] + [x])
            q = max(deps, key=lambda d: (bw[d][0], len(bw[d][1]), [-k for k in key(d)]))
            bw[x] = (bw[q][0] + w, bw[q][1] + [x])
        else:
            best[x] = (1, w, [x])
            bw[x] = (w, [x])
    end = max(order, key=lambda x: (best[x][0], best[x][1])) if order else None
    endw = max(order, key=lambda x: (bw[x][0], len(bw[x][1]))) if order else None

    def desc(x: str) -> int:
        seen, stack = set(), [x]
        while stack:
            for y in succ[stack.pop()]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return len(seen)

    roots = sorted([x for x in by if not by[x]["depends_on"]], key=key)
    sinks = sorted((x for x in by if not succ[x]), key=key)
    longest_to = {x: {"items": best[x][0], "weight": best[x][1], "path": best[x][2]} for x in sinks if x in best}
    return {
        "sinks": sinks,
        "longest_chain_to_sink": longest_to,
        "n_items": len(items),
        "n_edges": sum(len(i["depends_on"]) for i in items),
        "duplicates": dup,
        "dangling": dangling,
        "self_deps": self_dep,
        "forbidden_name_items": forbidden,
        "cyclic_nodes": cyclic,
        "acyclic": not cyclic,
        "topo_order": order,
        "critical_path_by_count": best[end][2] if end else [],
        "critical_path_by_count_weight": best[end][1] if end else 0,
        "critical_path_by_size_weight": bw[endw][1] if endw else [],
        "critical_path_size_weight": bw[endw][0] if endw else 0,
        "roots": {r: {"descendants": desc(r), "user_decision": by[r]["user_decision"]} for r in roots},
        "per_spec": dict(sorted(Counter(i["id"].split("-")[0] for i in items).items())),
        "per_size": dict(Counter(i["size"] for i in items)),
        "per_spec_user_decision": dict(
            sorted(Counter(i["id"].split("-")[0] for i in items if i["user_decision"]).items())
        ),
        "n_user_decision": sum(i["user_decision"] for i in items),
        "repos": dict(Counter(i["repo"] for i in items)),
    }


def controls(given: list[dict], toml: list[dict]) -> dict:
    out = {}
    m = copy.deepcopy(given)
    next(i for i in m if i["id"] == "S3-01")["depends_on"] = ["S5-33"]  # S5-33 depends on S3-01
    out["planted_cycle_detected"] = not check(m)["acyclic"]
    m = copy.deepcopy(given)
    m[0]["depends_on"].append("S9-99")
    out["planted_dangling_detected"] = bool(check(m)["dangling"])
    m = copy.deepcopy(given) + [copy.deepcopy(given[5])]
    out["planted_duplicate_detected"] = bool(check(m)["duplicates"])
    m = copy.deepcopy(given)
    m[3]["depends_on"].append(m[3]["id"])
    out["planted_self_dep_detected"] = bool(check(m)["self_deps"])
    m = copy.deepcopy(given)
    m[-1]["title"] += " (molxmpnn)"  # m[-1] is S6-32, not allow-listed
    out["planted_forbidden_name_detected"] = bool(check(m)["forbidden_name_items"])
    m = copy.deepcopy(given)
    m[-1]["size"] = "L" if m[-1]["size"] != "L" else "S"
    out["planted_list_toml_drift_detected"] = bool(compare(m, toml)) and (
        len(compare(m, toml)) > len(compare(given, toml))
    )
    return out


def mermaid(items: list[dict], cp: set[str]) -> str:
    specs: dict[str, list[dict]] = {}
    for i in items:
        specs.setdefault(i["id"].split("-")[0], []).append(i)
    lines = ["flowchart LR"]
    for s in sorted(specs, key=lambda s: int(s[1:])):
        lines.append(f"  subgraph {s}")
        for i in sorted(specs[s], key=lambda i: key(i["id"])):
            label = i["id"] + (" *" if i["user_decision"] else "")
            lines.append(f'    {i["id"].replace("-", "_")}["{label}"]')
        lines.append("  end")
    for i in sorted(items, key=lambda i: key(i["id"])):
        for d in sorted(i["depends_on"], key=key):
            lines.append(f"  {d.replace('-', '_')} --> {i['id'].replace('-', '_')}")
    lines.append("  classDef ud fill:#fde68a,stroke:#b45309,color:#111")
    lines.append("  classDef cp stroke:#dc2626,stroke-width:3px")
    lines.append("  class " + ",".join(i["id"].replace("-", "_") for i in items if i["user_decision"]) + " ud")
    lines.append("  class " + ",".join(sorted(x.replace("-", "_") for x in cp)) + " cp")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-dir", type=Path, default=None)
    ap.add_argument("--controls", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    given = load_given(HERE / "261001_dag_given_items.json")
    toml, nblocks = load_toml()
    if a.controls:
        print(json.dumps(controls(given, toml), indent=1))
        return
    res = check(given)
    res["toml_blocks_per_spec"] = nblocks
    res["toml_n_items"] = len(toml)
    res["list_vs_toml_diffs"] = compare(given, toml)
    old = load_old_ids(HERE / "261001_dag_given_edges.txt")
    new = {i["id"] for i in given}
    retired = sorted(set(old) - new, key=key)
    added = sorted(new - set(old), key=key)
    res["previous_n_items"] = len(old)
    res["retired_vs_previous"] = retired
    res["retired_per_spec"] = dict(sorted(Counter(x.split("-")[0] for x in retired).items()))
    res["added_vs_previous"] = added
    old_rows = [ln.split("|") for ln in (HERE / "261001_dag_given_edges.txt").read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
    rset = set(retired)
    res["edges_into_retired_removed_per_spec"] = dict(sorted(Counter(
        r[0].split("-")[0] for r in old_rows if r[0] not in rset
        for d in r[4].split(",") if d in rset).items()))
    res["previous_n_edges"] = sum(len([d for d in r[4].split(",") if d]) for r in old_rows)
    res["relabelled_molxmpnn_to_aminx"] = len([r for r in old_rows if r[1] == "molxmpnn" and r[0] in new])
    print(json.dumps({k: v for k, v in res.items() if k != "topo_order"}, indent=1))
    if a.emit_dir:
        a.emit_dir.mkdir(parents=True, exist_ok=True)
        (a.emit_dir / "result.json").write_text(json.dumps(res, indent=1))
        (a.emit_dir / "flow.mmd").write_text(mermaid(given, set(res["critical_path_by_count"])))
        by = {i["id"]: i for i in given}
        rows = [
            "| # | id | title | repo | size | depends_on | user_decision |",
            "|---|---|---|---|---|---|---|",
        ]
        for n, x in enumerate(res["topo_order"], 1):
            i = by[x]
            deps = ", ".join(sorted(i["depends_on"], key=key)) or "-"
            rows.append(
                f"| {n} | {x} | {i['title'].replace('|', '/')} | {i['repo']} | {i['size']} | {deps} | "
                f"{'yes' if i['user_decision'] else 'no'} |"
            )
        (a.emit_dir / "topo_table.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
