"""Throwaway read-only probe #2 for S6: PyPI wheel tags for jax/jaxlib (any emscripten/wasm/pyodide
wheel?) and the praxis PCG fixture shape. Writes to argv[1]. No installs."""

import json
import sys
import urllib.request
from pathlib import Path

out = []
for pkg in ["jax", "jaxlib", "onnxruntime"]:
    try:
        d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=30))
        latest = d["info"]["version"]
        names = sorted({u["filename"] for u in d["urls"]})
        wasm = [n for n in names if any(t in n.lower() for t in ("emscripten", "wasm", "pyodide"))]
        plats = sorted({n.split("-")[-1].replace(".whl", "") for n in names if n.endswith(".whl")})
        out.append(f"pypi\t{pkg}\tlatest={latest}\tn_files_latest={len(names)}\twasm_like_files={len(wasm)}\tplatform_tags={plats[:12]}")
        # any historical release with a wasm-like wheel?
        hist = [u["filename"] for v, us in d["releases"].items() for u in us if any(t in u["filename"].lower() for t in ("emscripten", "wasm", "pyodide"))]
        out.append(f"pypi\t{pkg}\tall_releases_wasm_like_files={len(hist)}\t{hist[:3]}")
    except Exception as e:  # noqa: BLE001
        out.append(f"pypi\t{pkg}\tERR\t{e!r}"[:200])

fx = Path.home() / "projects/praxis/plr-sema/tests/fixtures/simple_transfer_graph.json"
g = json.loads(fx.read_text())
out.append(f"fixture\t{fx.name}\tkeys={sorted(g)}\tn_operations={len(g['operations'])}\texecution_order={g['execution_order']}\thas_loops={g['has_loops']}")
Path(sys.argv[1]).write_text("\n".join(out) + "\n")
print("\n".join(out))
