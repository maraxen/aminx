"""S14: which hub-name candidates collide with names the user's praxia orchestrator already owns? (read-only)"""
import re
from pathlib import Path

HOME = Path.home()
P = HOME / "projects/praxia"

cargo = (P / "crates/praxia-cli/Cargo.toml").read_text()
bins = re.findall(r'\[\[bin\]\]\s*\nname = "([^"]+)"', cargo)
print("praxia-cli [[bin]] names:", bins)

cfg = (P / ".git/config").read_text()
print("praxia origin url:", re.findall(r"url = (\S+)", cfg))

ws = (P / "Cargo.toml").read_text()
members = re.findall(r'"(crates/praxia-[^"]+)"', ws)
print("praxia workspace crates named praxia-*:", len(members))

catalog = HOME / ".bth/catalog/runs"
slugs = sorted(p.name for p in catalog.iterdir() if p.is_dir())
print("bathos project slugs that are praxia-like or aminx-like:", [s for s in slugs if "praxi" in s or "aminx" in s])
print("slug aminx-hub present:", "aminx-hub" in slugs)

projects = HOME / "projects"
with_dot_praxia = sorted(p.name for p in projects.iterdir() if (p / ".praxia").is_dir())
print("projects with a .praxia directory:", len(with_dot_praxia))
print("sibling project named praxis present:", (projects / "praxis").is_dir())
