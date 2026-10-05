"""S3: unpickling a checkpoint that embeds an old module path, before/after rename (stdlib only).

Mirrors aminx/potts/calibration.py:159 `pickle.loads(decompressed)` of a LearnedCalibration.
"""
import io
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(tempfile.mkdtemp(prefix="s3_"))
(root / "oldpkg" / "potts").mkdir(parents=True)
(root / "oldpkg" / "__init__.py").write_text("")
(root / "oldpkg" / "potts" / "__init__.py").write_text("")
(root / "oldpkg" / "potts" / "calibration.py").write_text(
    "class LearnedCalibration:\n    def __init__(self, s=0.5):\n        self.s = s\n"
)
(root / "newpkg" / "potts").mkdir(parents=True)
(root / "newpkg" / "__init__.py").write_text("")
(root / "newpkg" / "potts" / "__init__.py").write_text("")
(root / "newpkg" / "potts" / "calibration.py").write_text(
    "class LearnedCalibration:\n    def __init__(self, s=0.5):\n        self.s = s\n"
)

# 1. Produce the "legacy checkpoint" in a child process that only knows oldpkg.
ckpt = root / "legacy.pkl"
subprocess.run(
    [sys.executable, "-c",
     "import pickle,sys; sys.path.insert(0, %r);"
     "from oldpkg.potts.calibration import LearnedCalibration as C;"
     "open(%r,'wb').write(pickle.dumps(C(0.25)))" % (str(root), str(ckpt))],
    check=True,
)
blob = ckpt.read_bytes()
results = {"blob_embeds_old_path": b"oldpkg.potts.calibration" in blob}

# 2. Consumer process that only knows newpkg (post-rename, no shim installed).
sys.path.insert(0, str(root))
sys.modules.pop("oldpkg", None)  # make sure oldpkg is not already imported here
sys.path.remove(str(root))
sys.path.insert(0, str(root / "newpkg_only"))  # nothing here
# Expose only newpkg: put a dir containing just newpkg on sys.path.
only_new = Path(tempfile.mkdtemp(prefix="s3_new_"))
(only_new / "newpkg").symlink_to(root / "newpkg", target_is_directory=True)
sys.path.insert(0, str(only_new))

try:
    pickle.loads(blob)
    results["plain_loads_fails"] = False
except ModuleNotFoundError as e:
    results["plain_loads_fails"] = True
    results["plain_error"] = str(e)


class _RemapUnpickler(pickle.Unpickler):
    LEGACY, CURRENT = "oldpkg", "newpkg"

    def find_class(self, module, name):
        if module == self.LEGACY or module.startswith(self.LEGACY + "."):
            module = self.CURRENT + module[len(self.LEGACY):]
        return super().find_class(module, name)


obj = _RemapUnpickler(io.BytesIO(blob)).load()
results["remap_loads"] = obj.s == 0.25
results["remap_class_module"] = type(obj).__module__

# Negative control: the remap must not touch unrelated modules.
buf = io.BytesIO()
import collections  # noqa: E402

pickle.dump(collections.OrderedDict(a=1), buf)
results["remap_leaves_stdlib_alone"] = _RemapUnpickler(io.BytesIO(buf.getvalue())).load() == collections.OrderedDict(a=1)
print(json.dumps(results, indent=2, sort_keys=True))
