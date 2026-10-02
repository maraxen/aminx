"""S15: importlib.util.find_spec semantics for dotted names (the shape of an availability guard).

Builds throwaway packages in a temp dir. `s15_present` has an __init__ that leaves a marker file and then
raises, so we can see whether find_spec on a dotted name executes the parent package.
"""
import importlib.util as u
import sys
import tempfile
from pathlib import Path

root = Path(tempfile.mkdtemp(prefix="s15_"))
pkg = root / "s15_present"
pkg.mkdir()
marker = root / "init_ran.marker"
(pkg / "__init__.py").write_text(f"open({str(marker)!r}, 'w').write('ran')\nraise RuntimeError('parent __init__ executed')\n")
(pkg / "sub.py").write_text("X = 1\n")
sys.path.insert(0, str(root))


def attempt(label, fn):
    try:
        print(f"{label} -> {fn()!r}")
    except Exception as e:  # noqa: BLE001
        print(f"{label} -> RAISES {type(e).__name__}: {e}")


print("--- root package absent")
attempt("find_spec('s15_absent.sub.mod')", lambda: u.find_spec("s15_absent.sub.mod"))
attempt("find_spec('s15_absent')", lambda: u.find_spec("s15_absent"))
print("--- root package present, __init__ has a side effect and raises")
attempt("find_spec('s15_present')", lambda: u.find_spec("s15_present") is not None)
print("parent __init__ ran after root-only find_spec:", marker.exists())
attempt("find_spec('s15_present.sub')", lambda: u.find_spec("s15_present.sub") is not None)
print("parent __init__ ran after dotted find_spec:", marker.exists())
