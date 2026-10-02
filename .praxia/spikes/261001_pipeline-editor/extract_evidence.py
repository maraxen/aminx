"""Throwaway read-only extractor: copies the exact source lines the S6 spec cites into evidence.txt,
so each ledger row can cite a workspace-relative file:line. No network, no writes outside OUT.
"""

import subprocess
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
HOME = Path.home() / "projects"
WS = Path(sys.argv[2])


def lines_from_text(text, ranges):
    ls = text.splitlines()
    out = []
    for a, b in ranges:
        out.extend(ls[a - 1 : b])
    return out


def xtrax(path, ranges):
    txt = subprocess.run(
        ["git", "show", f"v0.4.0a11:{path}"], cwd=HOME / "xtrax", capture_output=True, text=True, check=True
    ).stdout
    return lines_from_text(txt, ranges)


def local(base, path, ranges):
    return lines_from_text((base / path).read_text(), ranges)


SPECS = [
    ("xtrax@v0.4.0a11 src/xtrax/composition/graph.py", lambda: xtrax("src/xtrax/composition/graph.py", [(40, 43), (58, 64), (72, 75)])),
    ("xtrax@v0.4.0a11 src/xtrax/composition/serialize.py", lambda: xtrax("src/xtrax/composition/serialize.py", [(110, 113), (139, 145), (175, 177)])),
    ("xtrax@v0.4.0a11 src/xtrax/export/targets.py", lambda: xtrax("src/xtrax/export/targets.py", [(21, 25)])),
    ("xtrax@v0.4.0a11 src/xtrax/export/onnx.py", lambda: xtrax("src/xtrax/export/onnx.py", [(204, 209)])),
    ("praxis backend/utils/plr_static_analysis/models.py (504-530,555,568)", lambda: local(HOME / "praxis", "praxis/backend/utils/plr_static_analysis/models.py", [(504, 511), (525, 530), (555, 555), (568, 568)])),
    ("praxis backend/utils/plr_static_analysis/models.py (623-644)", lambda: local(HOME / "praxis", "praxis/backend/utils/plr_static_analysis/models.py", [(623, 626), (642, 644)])),
    ("praxis backend/models/domain/schedule.py (49,63,132-133,141)", lambda: local(HOME / "praxis", "praxis/backend/models/domain/schedule.py", [(49, 49), (63, 63), (132, 133), (141, 141)])),
    ("praxis web-client/package.json (46,65)", lambda: local(HOME / "praxis", "praxis/web-client/package.json", [(46, 46), (65, 65)])),
    ("praxis web-client/package.json (10)", lambda: local(HOME / "praxis", "praxis/web-client/package.json", [(10, 10)])),
    ("praxis web-repl/jupyter-lite.json (23)", lambda: local(HOME / "praxis", "web-repl/jupyter-lite.json", [(23, 23)])),
    ("praxis backend/models/domain/protocol.py (320)", lambda: local(HOME / "praxis", "praxis/backend/models/domain/protocol.py", [(320, 320)])),
    ("praxis computation_graph_extractor.py (880)", lambda: local(HOME / "praxis", "praxis/backend/utils/plr_static_analysis/visitors/computation_graph_extractor.py", [(880, 880)])),
    ("aminx site/index.html (12-14,25-29)", lambda: local(WS, "site/index.html", [(12, 14), (25, 29)])),
    ("aminx site/app.js (1)", lambda: local(WS, "site/app.js", [(1, 1)])),
    ("aminx browser/aminx-sampler/runspec_core.mjs (1-8)", lambda: local(WS, "browser/aminx-sampler/runspec_core.mjs", [(1, 8)])),
    ("aminx .praxia/docs/specs/260929_p07-split-export.md (2)", lambda: local(WS, ".praxia/docs/specs/260929_p07-split-export.md", [(2, 2)])),
]

with OUT.open("w") as f:
    for label, fn in SPECS:
        f.write(f"### {label}\n")
        for ln in fn():
            f.write(ln + "\n")
print("wrote", OUT)
