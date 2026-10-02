"""Assemble the 261001 plan doc from plan_template.md plus the generated flow.mmd and topo_table.md (spike)."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "docs/plans/261001_ecosystem-backlog-dag.md"
t = (HERE / "plan_template.md").read_text()
rep = {
    "toolchain (OQ-29) | 136": "toolchain (OQ-30) | 136",
    "with a named reviewer (OQ-34) | 33": "with a named reviewer (OQ-35) | 33",
    "clone location (OQ-16 to OQ-18) | 22": "clone location (OQ-17 to OQ-19) | 22",
    "| 6 (between S3 items) |": "| 6 (from kept S3 items) |",
}
for a, b in rep.items():
    if a in t:
        t = t.replace(a, b)
t = t.replace("@@FLOW@@", (HERE / "flow.mmd").read_text().rstrip())
t = t.replace("@@TOPO@@", (HERE / "topo_table.md").read_text().rstrip())
OUT.write_text(t + "\n")
print(OUT, len(t.splitlines()))
