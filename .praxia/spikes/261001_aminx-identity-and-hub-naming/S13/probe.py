"""S13: does the REAL guard line from asr's vendored proteinsmc misfire against a hub package named aminx?

The guard is read verbatim from the file (not retyped), then evaluated in isolated interpreters: one with
no `aminx` anywhere (negative control), one with a toy hub package `aminx` that has no `scoring` subpackage.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path.home() / "projects/asr/vendor/proteinsmc/src/proteinsmc/scoring/mpnn.py"
text = SRC.read_text()
m = re.search(r"^AMINX_AVAILABLE = .*$", text, re.M)
assert m, "guard line not found"
guard = m.group(0)
call_site = re.search(r"^\s+from aminx\.scoring\.score import make_score_sequence$", text, re.M)

hub = Path(tempfile.mkdtemp(prefix="s13_hub_"))
(hub / "aminx").mkdir()
(hub / "aminx" / "__init__.py").write_text("# toy hub: static-site tooling, no MPNN modules\n")

child = f"""
import importlib.util, json
{guard}
out = {{"AMINX_AVAILABLE": AMINX_AVAILABLE}}
try:
    from aminx.scoring.score import make_score_sequence
    out["call_site"] = "imported"
except ImportError as e:
    out["call_site"] = type(e).__name__ + ": " + str(e)
print(json.dumps(out))
"""


def run(extra_path):
    code = "import sys; sys.path.insert(0, %r); exec(sys.stdin.read())" % (extra_path or "/nonexistent")
    r = subprocess.run([sys.executable, "-I", "-c", code], input=child, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    return r.stdout.strip() or r.stderr.strip()


print("guard_line_as_read:", guard)
print("call_site_found_in_source:", bool(call_site))
print("control_no_aminx:", run(None))
print("toy_hub_named_aminx_present:", run(str(hub)))
