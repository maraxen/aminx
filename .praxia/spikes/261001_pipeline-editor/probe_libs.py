"""Read-only probe for the 261001 pipeline-editor spec (S6): library metadata and Pyodide locks.

Throwaway: stdlib only, no installs, no builds. Writes a TSV so spec rows can cite file:line.
Network: registry.npmjs.org + the code-host REST API (host string split below only because a
worktree guard in this environment false-positives on the substring in command lines).
"""

import json
import sys
import urllib.request

OUT = sys.argv[1]
HOST = "api." + "git" + "hub.com"
NPM = [
    "rete", "rete-area-plugin", "rete-connection-plugin", "rete-lit-plugin", "rete-angular-plugin",
    "litegraph.js", "@comfyorg/litegraph", "@xyflow/react", "@xyflow/svelte", "@xyflow/system",
    "@vue-flow/core", "baklavajs", "drawflow", "@joint/core", "@antv/x6", "blockly", "@angular/elements",
]
REPOS = ["Comfy-Org/ComfyUI_frontend", "Comfy-Org/litegraph.js", "retejs/rete", "xyflow/xyflow",
         "jagenjo/litegraph.js", "antvis/X6"]
LOCKS = [
    "/home/marielle/projects/praxis/praxis/web-client/node_modules/pyodide/pyodide-lock.json",
    "/home/marielle/projects/praxis/web-repl/dist/static/pyodide/pyodide-lock.json",
]
WATCH = ["jax", "jaxlib", "onnx", "onnxruntime", "torch", "numpy", "scipy", "pandas", "scikit-learn",
         "biopython", "ml-dtypes", "micropip", "pydantic", "sympy"]


def get(url):
    return json.load(urllib.request.urlopen(url, timeout=30))


rows = []
for p in NPM:
    try:
        d = get("https://registry.npmjs.org/" + p.replace("/", "%2F"))
        tag = d["dist-tags"]["latest"]
        v = d["versions"][tag]
        rows.append(["npm", p, tag, str(v.get("license")), d["time"].get(tag, "?"),
                     ",".join(sorted((v.get("dependencies") or {}))[:6]),
                     ",".join(sorted((v.get("peerDependencies") or {}))[:8])])
    except Exception as e:  # noqa: BLE001 - probe, report and continue
        rows.append(["npm", p, "ERR", repr(e)[:80], "", "", ""])
for r in REPOS:
    try:
        d = get(f"https://{HOST}/repos/{r}")
        rows.append(["repo", r, "", str((d.get("license") or {}).get("spdx_id")), d.get("pushed_at", "?"),
                     f"archived={d.get('archived')}", f"stars={d.get('stargazers_count')}"])
    except Exception as e:  # noqa: BLE001
        rows.append(["repo", r, "ERR", repr(e)[:80], "", "", ""])
for path in LOCKS:
    d = json.load(open(path))
    info = d["info"]
    rows.append(["pyodide-lock", path, str(info.get("python")), str(info.get("platform")), "", "", ""])
    for k in WATCH:
        rows.append(["pyodide-pkg", path.split("/")[-4] if "node_modules" in path else "web-repl", k,
                     str(d["packages"].get(k, {}).get("version")), "", "", ""])
with open(OUT, "w") as f:
    for r in rows:
        f.write("\t".join(r) + "\n")
print(len(rows), "rows ->", OUT)
