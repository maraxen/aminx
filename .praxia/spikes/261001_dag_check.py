"""Mechanical DAG check for the 261001 ecosystem spec set (task_id 261001_aminx-hub-ecosystem-specs).

Stdlib only. Reads the transcribed authoritative item list (261001_dag_given_edges.txt) and the
```toml backlog block of each .praxia/docs/specs/261001_*.md, then:
  1. compares the two (ids, repo, size, user_decision, depends_on) and reports every difference;
  2. checks duplicate ids, dangling depends_on, and acyclicity (Kahn);
  3. emits a deterministic topological order, the critical path (longest chain by item count,
     ties broken by size weight S=1 M=2 L=3, then id), roots, and a mermaid flowchart.

Usage: uv run --no-project python3 .praxia/spikes/261001_dag_check.py [--emit-dir DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import tomllib
from pathlib import Path

log = logging.getLogger("dag_check")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
W = {"S": 1, "M": 2, "L": 3}


def load_given(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        iid, repo, size, ud, deps = line.split("|")
        out.append(
            {
                "id": iid,
                "repo": repo,
                "size": size,
                "user_decision": ud == "1",
                "depends_on": [d for d in deps.split(",") if d],
            }
        )
    return out


def load_toml() -> list[dict]:
    items = []
    for f in sorted(glob.glob(str(ROOT / ".praxia/docs/specs/261001_*.md"))):
        text = Path(f).read_text()
        sec = text[text.index("Backlog items") :]
        blocks = re.findall(r"```toml\n(.*?)```", sec, re.S)
        if len(blocks) != 1:
            raise SystemExit(f"{f}: expected exactly one toml block, found {len(blocks)}")
        for it in tomllib.loads(blocks[0])["item"]:
            it["_spec"] = Path(f).name
            items.append(it)
    return items


def compare(given: list[dict], toml: list[dict]) -> list[str]:
    diffs = []
    g = {i["id"]: i for i in given}
    t = {i["id"]: i for i in toml}
    for iid in sorted(set(g) - set(t)):
        diffs.append(f"{iid}: in authoritative list, not in spec toml")
    for iid in sorted(set(t) - set(g)):
        diffs.append(f"{iid}: in spec toml ({t[iid]['_spec']}), not in authoritative list")
    for iid in sorted(set(g) & set(t)):
        for k in ("repo", "size", "user_decision"):
            if g[iid][k] != t[iid].get(k):
                diffs.append(f"{iid}.{k}: list={g[iid][k]!r} toml={t[iid].get(k)!r}")
        if sorted(g[iid]["depends_on"]) != sorted(t[iid].get("depends_on", [])):
            diffs.append(
                f"{iid}.depends_on: list={sorted(g[iid]['depends_on'])} "
                f"toml={sorted(t[iid].get('depends_on', []))}"
            )
    return diffs


def key(iid: str) -> tuple[int, int]:
    s, n = iid[1:].split("-")
    return int(s), int(n)


def check(items: list[dict]) -> dict:
    ids = [i["id"] for i in items]
    dup = sorted({x for x in ids if ids.count(x) > 1})
    by = {i["id"]: i for i in items}
    dangling = sorted(
        (i["id"], d) for i in items for d in i["depends_on"] if d not in by
    )
    self_dep = sorted(i["id"] for i in items if i["id"] in i["depends_on"])
    indeg = {x: 0 for x in by}
    succ: dict[str, list[str]] = {x: [] for x in by}
    for i in items:
        for d in i["depends_on"]:
            if d in by:
                indeg[i["id"]] += 1
                succ[d].append(i["id"])
    ready = sorted([x for x in by if indeg[x] == 0], key=key)
    order = []
    while ready:
        x = ready.pop(0)
        order.append(x)
        for y in succ[x]:
            indeg[y] -= 1
            if indeg[y] == 0:
                ready.append(y)
        ready.sort(key=key)
    cyclic = sorted(set(by) - set(order), key=key)
    # longest path: (count, weight)
    best: dict[str, tuple[int, int, list[str]]] = {}
    for x in order:
        deps = [d for d in by[x]["depends_on"] if d in by]
        if deps:
            p = max(deps, key=lambda d: (best[d][0], best[d][1], [-k for k in key(d)]))
            c, w, path = best[p]
            best[x] = (c + 1, w + W[by[x]["size"]], path + [x])
        else:
            best[x] = (1, W[by[x]["size"]], [x])
    end = max(order, key=lambda x: (best[x][0], best[x][1])) if order else None
    # size-weighted critical path as well
    bw: dict[str, tuple[int, list[str]]] = {}
    for x in order:
        deps = [d for d in by[x]["depends_on"] if d in by]
        if deps:
            p = max(deps, key=lambda d: bw[d][0])
            bw[x] = (bw[p][0] + W[by[x]["size"]], bw[p][1] + [x])
        else:
            bw[x] = (W[by[x]["size"]], [x])
    endw = max(order, key=lambda x: bw[x][0]) if order else None
    roots = sorted([x for x in by if not by[x]["depends_on"]], key=key)
    return {
        "n_items": len(items),
        "n_edges": sum(len(i["depends_on"]) for i in items),
        "duplicates": dup,
        "dangling": dangling,
        "self_deps": self_dep,
        "cyclic_nodes": cyclic,
        "acyclic": not cyclic,
        "topo_order": order,
        "critical_path_by_count": best[end][2] if end else [],
        "critical_path_by_count_weight": best[end][1] if end else 0,
        "critical_path_by_size_weight": bw[endw][1] if endw else [],
        "critical_path_size_weight": bw[endw][0] if endw else 0,
        "roots": roots,
        "roots_user_decision": [r for r in roots if by[r]["user_decision"]],
    }


def mermaid(items: list[dict], cp: set[str]) -> str:
    specs: dict[str, list[dict]] = {}
    for i in items:
        specs.setdefault(i["id"].split("-")[0], []).append(i)
    lines = ["flowchart LR"]
    for s in sorted(specs, key=lambda s: int(s[1:])):
        lines.append(f"  subgraph {s}")
        for i in sorted(specs[s], key=lambda i: key(i["id"])):
            nid = i["id"].replace("-", "_")
            label = i["id"] + (" *" if i["user_decision"] else "")
            lines.append(f'    {nid}["{label}"]')
        lines.append("  end")
    for i in sorted(items, key=lambda i: key(i["id"])):
        for d in sorted(i["depends_on"], key=key):
            lines.append(f"  {d.replace('-', '_')} --> {i['id'].replace('-', '_')}")
    lines.append("  classDef ud fill:#fde68a,stroke:#b45309")
    lines.append("  classDef cp stroke:#dc2626,stroke-width:3px")
    ud = [i["id"].replace("-", "_") for i in items if i["user_decision"]]
    lines.append("  class " + ",".join(ud) + " ud")
    lines.append("  class " + ",".join(sorted(x.replace("-", "_") for x in cp)) + " cp")
    return "\n".join(lines)


def controls(given: list[dict], toml: list[dict]) -> dict:
    """Negative controls: each planted defect must be detected."""
    import copy

    out = {}
    m = copy.deepcopy(given)
    next(i for i in m if i["id"] == "S3-01")["depends_on"] = ["S3-13"]
    out["planted_cycle_detected"] = not check(m)["acyclic"]
    m = copy.deepcopy(given)
    m[0]["depends_on"].append("S9-99")
    out["planted_dangling_detected"] = bool(check(m)["dangling"])
    m = copy.deepcopy(given) + [copy.deepcopy(given[5])]
    out["planted_duplicate_detected"] = bool(check(m)["duplicates"])
    m = copy.deepcopy(given)
    m[-1]["size"] = "L" if m[-1]["size"] != "L" else "S"
    out["planted_list_toml_drift_detected"] = bool(compare(m, toml))
    by = {i["id"]: i for i in given}
    has_succ = {d for i in given for d in i["depends_on"]}
    sinks = sorted((x for x in by if x not in has_succ), key=key)
    out["sinks"] = sinks
    out["S3-20_deps"] = sorted(by["S3-20"]["depends_on"], key=key)
    # S3 assembly row 6: leaf set of S1/S2/S4 items on repo molxmpnn/aminx, leaf = no direct
    # depends_on edge from another member; must equal S3-20's non-S3 depends_on.
    mem = {
        x
        for x, i in by.items()
        if x.split("-")[0] in ("S1", "S2", "S4") and i["repo"] in ("molxmpnn", "aminx")
    }
    pointed = {d for x in mem for d in by[x]["depends_on"] if d in mem}
    leaf = sorted(mem - pointed, key=key)
    non_s3 = sorted((d for d in by["S3-20"]["depends_on"] if not d.startswith("S3-")), key=key)
    out["row6_leaf_set"] = leaf
    out["row6_S3-20_non_S3_deps"] = non_s3
    out["row6_equal"] = leaf == non_s3
    succ: dict[str, set[str]] = {x: set() for x in by}
    for i in given:
        for d in i["depends_on"]:
            succ[d].add(i["id"])

    def desc(x: str) -> int:
        seen, stack = set(), [x]
        while stack:
            for y in succ[stack.pop()]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return len(seen)

    out["root_descendants"] = {r: desc(r) for r in sorted(
        (x for x in by if not by[x]["depends_on"]), key=key)}
    from collections import Counter

    out["per_spec"] = dict(Counter(i["id"].split("-")[0] for i in given))
    out["per_size"] = dict(Counter(i["size"] for i in given))
    out["n_user_decision"] = sum(i["user_decision"] for i in given)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-dir", type=Path, default=None)
    ap.add_argument("--controls", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    given = load_given(HERE / "261001_dag_given_edges.txt")
    toml = load_toml()
    if a.controls:
        print(json.dumps(controls(given, toml), indent=1))
        return
    diffs = compare(given, toml)
    res = check(given)
    res["toml_n_items"] = len(toml)
    res["list_vs_toml_diffs"] = diffs
    res["toml_check"] = {k: v for k, v in check(
        [{"id": t["id"], "size": t["size"], "user_decision": t["user_decision"],
          "depends_on": t.get("depends_on", [])} for t in toml]
    ).items() if k in ("n_items", "duplicates", "dangling", "acyclic", "cyclic_nodes")}
    summary = {k: v for k, v in res.items() if k not in ("topo_order",)}
    print(json.dumps(summary, indent=1))
    if a.emit_dir:
        a.emit_dir.mkdir(parents=True, exist_ok=True)
        (a.emit_dir / "result.json").write_text(json.dumps(res, indent=1))
        (a.emit_dir / "flow.mmd").write_text(
            mermaid(given, set(res["critical_path_by_count"]))
        )
        titles = {t["id"]: t["title"] for t in toml}
        by = {i["id"]: i for i in given}
        rows = ["| # | id | title | repo | size | depends_on | user_decision |",
                "|---|---|---|---|---|---|---|"]
        for n, x in enumerate(res["topo_order"], 1):
            i = by[x]
            t = titles[x].replace("|", "/")
            deps = ", ".join(sorted(i["depends_on"], key=key)) or "-"
            rows.append(
                f"| {n} | {x} | {t} | {i['repo']} | {i['size']} | {deps} | "
                f"{'yes' if i['user_decision'] else 'no'} |"
            )
        (a.emit_dir / "topo_table.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
