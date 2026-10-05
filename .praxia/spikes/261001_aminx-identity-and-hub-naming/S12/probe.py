"""S12: how proteinsmc main and asr's vendored proteinsmc guard the MPNN package (read-only file inspection).

Reads the real files; nothing is retyped. The AST pass classifies every import of a module whose name
starts with `aminx` / `prxteinmpnn` as: module-level runtime, under `if TYPE_CHECKING`, or inside a function.
"""
import ast
import os
import re
from pathlib import Path

HOME = Path.home() / "projects"
MAIN = HOME / "proteinsmc/src/proteinsmc/scoring/mpnn.py"
VEND = HOME / "asr/vendor/proteinsmc/src/proteinsmc/scoring/mpnn.py"
GITMODULES = HOME / "asr/.gitmodules"


def guard_lines(path):
    return [ln.strip() for ln in path.read_text().splitlines() if "find_spec(" in ln]


def classify(path, prefixes):
    tree = ast.parse(path.read_text())
    out = {"module_level_runtime": [], "type_checking": [], "in_function": []}

    def is_tc(node):
        t = node.test
        return (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
            isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"
        )

    def walk(node, where):
        for child in ast.iter_child_nodes(node):
            w = where
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                w = "in_function"
            elif isinstance(child, ast.If) and is_tc(child) and where == "module_level_runtime":
                w = "type_checking"
            if isinstance(child, ast.ImportFrom) and child.module and child.module.split(".")[0] in prefixes:
                out[w].append(f"{child.module}@{child.lineno}")
            elif isinstance(child, ast.Import):
                for a in child.names:
                    if a.name.split(".")[0] in prefixes:
                        out[w].append(f"{a.name}@{child.lineno}")
            walk(child, w)

    walk(tree, "module_level_runtime")
    return out


print("proteinsmc main guard lines:", guard_lines(MAIN))
print("asr vendored guard lines:   ", guard_lines(VEND))
print("main imports (prxteinmpnn):", classify(MAIN, {"prxteinmpnn"}))
print("vendored imports (aminx):  ", classify(VEND, {"aminx"}))
subs = re.findall(r"^\s*path = (.+)$", GITMODULES.read_text(), re.M)
print("asr .gitmodules paths:", subs)
print("vendor/proteinsmc listed as submodule:", "vendor/proteinsmc" in subs)
print("asr/vendor/proteinsmc has its own .git entry:", os.path.exists(HOME / "asr/vendor/proteinsmc/.git"))
print("main and vendored mpnn.py byte-identical:", MAIN.read_bytes() == VEND.read_bytes())
