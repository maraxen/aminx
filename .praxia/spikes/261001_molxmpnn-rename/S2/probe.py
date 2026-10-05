"""S2: alias-finder shim vs naive __path__ aliasing, on a toy package pair (stdlib only).

Toy: real package `molx` (host/runner.py defines class Spec), shim package `oldx`.
Approach A (proposed): sys.meta_path finder whose loader hands back the REAL module object.
Approach B (naive): sys.modules['oldx2'] = molx with molx.__path__, no finder.
"""
import importlib
import importlib.abc
import importlib.machinery
import json
import pickle
import sys
import tempfile
import warnings
from pathlib import Path

root = Path(tempfile.mkdtemp(prefix="s2_"))
(root / "molx" / "host").mkdir(parents=True)
(root / "molx" / "__init__.py").write_text("VERSION = '1'\n")
(root / "molx" / "host" / "__init__.py").write_text("")
(root / "molx" / "host" / "runner.py").write_text(
    "class Spec:\n    def __init__(self, x=1):\n        self.x = x\n"
)
(root / "molx" / "ebm").mkdir()
(root / "molx" / "ebm" / "__init__.py").write_text("")
sys.path.insert(0, str(root))

_OLD, _NEW = "oldx", "molx"
_GONE = {"oldx.ebm": "aminx.ebm moved to its own project; install that instead"}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, real):
        self._real = real
        self._saved_spec = None

    def create_module(self, spec):
        self._saved_spec = self._real.__spec__
        return self._real

    def exec_module(self, module):
        # _init_module_attrs overwrote __spec__ with the alias spec; restore the real one.
        module.__spec__ = self._saved_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + "."):
            return None
        for gone, msg in _GONE.items():
            if fullname == gone or fullname.startswith(gone + "."):
                raise ModuleNotFoundError(msg, name=fullname)
        real = importlib.import_module(_NEW + fullname[len(_OLD):])
        return importlib.machinery.ModuleSpec(
            fullname, _AliasLoader(real), is_package=hasattr(real, "__path__")
        )


# ---- Approach A: build the shim package `oldx` in-memory, emit one DeprecationWarning.
shim = type(sys)("oldx")
shim.__path__ = []
shim.__getattr__ = lambda name: getattr(importlib.import_module("molx"), name)
sys.modules["oldx"] = shim
sys.meta_path.insert(0, _AliasFinder())
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    warnings.warn("oldx is deprecated; use molx", DeprecationWarning, stacklevel=2)

results = {}
import molx.host.runner as real_runner  # noqa: E402
import oldx.host.runner as alias_runner  # noqa: E402
from oldx.host.runner import Spec as AliasSpec  # noqa: E402

results["A_module_identity"] = alias_runner is real_runner
results["A_class_identity"] = AliasSpec is real_runner.Spec
results["A_isinstance_cross"] = isinstance(AliasSpec(), real_runner.Spec)
results["A_real_spec_name_intact"] = real_runner.__spec__.name == real_runner.__name__ == "molx.host.runner"
results["A_class_module_attr"] = AliasSpec.__module__
results["A_deprecation_warning_count"] = sum(issubclass(w.category, DeprecationWarning) for w in caught)
results["A_toplevel_attr_forward"] = shim.VERSION == "1"
blob = pickle.dumps(AliasSpec(7))
results["A_pickle_records_real_path"] = b"molx.host.runner" in blob and b"oldx" not in blob
results["A_pickle_loads"] = pickle.loads(blob).x == 7
try:
    importlib.import_module("oldx.ebm")
    results["A_gone_module_raises"] = False
except ModuleNotFoundError as e:
    results["A_gone_module_raises"] = "own project" in str(e)

# ---- Approach B: naive alias (replace package in sys.modules, share __path__), no finder.
sys.meta_path.pop(0)  # remove finder so B is measured on its own
import molx  # noqa: E402

sys.modules["oldx2"] = molx
molx_alias_runner = importlib.import_module("oldx2.host.runner")
results["B_module_identity"] = molx_alias_runner is real_runner
results["B_class_identity"] = molx_alias_runner.Spec is real_runner.Spec
results["B_isinstance_cross"] = isinstance(molx_alias_runner.Spec(), real_runner.Spec)

print(json.dumps(results, indent=2, sort_keys=True))
