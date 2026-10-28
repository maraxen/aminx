"""Assemble the deployable aminx-ProteinMPNN-in-the-browser site into dist/.

    python3 tools/build_site.py --out dist
    python3 tools/build_site.py --out dist --models-dir /path/to/exported/models
    python3 tools/build_site.py --out dist \\
        --protonpotts-scoring-dir DIR --protonpotts-ph-dir DIR --protonpotts-decoder-dir DIR
    python3 tools/build_site.py --check      # verify the allow-list without building

WHY THIS EXISTS, AND WHY IT IS NOT A BUNDLER.

site/*.mjs and site/*.js are plain ES modules, loaded by the browser through
the same relative paths they have in the checkout -- nothing to compile. The
one thing a "build" answers is WHICH FILES a public site actually contains:
not site/tests/, and (unless explicitly asked for via --models-dir) not any
ONNX weights at all.

That makes this a copy with an allow-list, and the allow-list is written out
below rather than derived from a directory walk, for the same reason
sokrypton/localfold's tools/build_site.py gives: a derived rule ("everything
under site/ except tests/") silently ships the next file somebody adds to
site/ without anyone deciding that file belongs on a public page. --check
verifies the allow-list still matches the tree in both directions: every
listed file exists, and (for site/, excluding tests/) nothing UNLISTED does.

THE SAMPLER MOVES. browser/aminx-sampler/{aminx_sampler.mjs,runspec_core.mjs}
are the browser-safe core the whole site depends on, but they live outside
site/ in the source tree (shared with the Node CLI/test harness at
browser/layer_c/). This build copies just those two files into
dist/aminx-sampler/, and rewrites the one import in site/worker.js that
points at their SOURCE-TREE location so it resolves inside dist/ instead.
Every other site/*.mjs module (pdb_parse.mjs, design_utils.mjs) deliberately
has NO import of the sampler, so nothing else needs rewriting -- see
site/pdb_parse.mjs's header.

MODEL WEIGHTS ARE OPT-IN, and only two exact filenames are ever eligible:
p07_sample_L128.onnx and p07_sample_L256.onnx. The export embeds its weights
(jax2onnx's embed_external_data), so a `.onnx.data` file beside it is stale
by construction -- it is never in the allow-list and this script will not
copy one even if --models-dir contains it (see check_stray_onnx_data below).

PROTONPOTTS IS A SECOND SUB-SITE, at dist/protonpotts/. Its page files live
under site/protonpotts/ (PROTONPOTTS_SITE_FILES). Its scoring, pH-design and
decoder modules live in browser/protonpotts-scorer/ (PROTONPOTTS_LIB_FILES),
and that tree imports them with relative paths that only resolve in the
source layout. This build copies those five modules into dist/protonpotts/lib/
and rewrites the worker's imports of them to "./lib/<name>", so that the
deployed worker resolves them. Because the rewrite is a text edit, two
things are checked rather than trusted: every specifier the rewrite targets
must appear exactly once (a missing one would leave a Worker that 404s on
start, a duplicated one would be a silent double rewrite), and no
"../../browser/" substring may survive it. Each lib module's own imports
must point at a sibling in PROTONPOTTS_LIB_FILES, so that a new cross-directory
import cannot ship broken without this script refusing.

PROTONPOTTS WEIGHTS ARE OPT-IN AND PER KIND. Each of --protonpotts-scoring-dir,
--protonpotts-ph-dir and --protonpotts-decoder-dir names an exported directory
with a MANIFEST.json. Only the files that manifest names (buckets[*].graphs[*].file)
are copied, and each one's sha256 must equal the manifest's own digest for it.
Nothing unnamed is copied (extras such as *.onnx.data are only reported). The
whole plan is validated before dist/ is removed, so a refused directory
leaves nothing half-built. Without the flags, no dist/protonpotts/models/
directory is created and the page reports the missing models at runtime.
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
SAMPLER_SRC = ROOT / "browser" / "aminx-sampler"
PROTONPOTTS_LIB_SRC = ROOT / "browser" / "protonpotts-scorer"

# WHAT THE PUBLIC SITE CONTAINS. Written out explicitly; see the module
# docstring for why this is not a directory walk.
SITE_FILES = [
    "index.html",
    "app.js",
    "site.css",
    "worker.js",
    "config.js",
    "pdb_parse.mjs",
    "design_utils.mjs",
    "README.md",
]

# The ProtonPotts sub-site's page files, relative to site/. Listed here, not
# discovered, for the same reason as SITE_FILES.
PROTONPOTTS_SITE_FILES = [
    "protonpotts/index.html",
    "protonpotts/app.mjs",
    "protonpotts/worker.mjs",
    "protonpotts/config.js",
    "protonpotts/site.css",
    "protonpotts/README.md",
]

# The ProtonPotts modules copied to dist/protonpotts/lib/. Written out as an
# allow-list too, so a directory copy does not drag in the Node harnesses
# (run_*_node.mjs), the page modules (*_page.mjs, ort/) or the HTML pages
# (*_index.html) that live beside them in browser/protonpotts-scorer/.
PROTONPOTTS_LIB_FILES = [
    "protonpotts_inputs.mjs",
    "protonpotts_scorer.mjs",
    "ph_plan.mjs",
    "ph_design.mjs",
    "decoder_sampler.mjs",
]

PROTONPOTTS_WORKER = "protonpotts/worker.mjs"
PROTONPOTTS_LIB_DIR = "lib"
PROTONPOTTS_MODEL_KINDS = ("scoring", "ph", "decoder")

# Only these two are ever a browser-export target (browser_integration.md's
# BUCKETS ladder: 128, 256). Nothing else in a --models-dir is ever copied,
# named or not -- see check_stray_onnx_data.
MODEL_FILES = [
    "p07_sample_L128.onnx",
    "p07_sample_L256.onnx",
    "p07_encoder_L128.onnx",
    "p07_encoder_L256.onnx",
    "p07_decoder_L128.onnx",
    "p07_decoder_L256.onnx",
    "p07_fuse_L128.onnx",
    "p07_fuse_L256.onnx",
    "p07_wave_L128.onnx",
    "p07_wave_L256.onnx",
]

SAMPLER_FILES = [
    "aminx_sampler.mjs",
    "runspec_core.mjs",
    "split_driver.mjs",
    "split_loop.mjs",
]

# The exact specifier site/worker.js carries in the source tree, and what it
# becomes once aminx-sampler/ is copied in beside it. Kept as one pair here
# rather than a regex, so a change to either file is a one-line diff to find.
WORKER_DEV_IMPORT = '"../browser/aminx-sampler/aminx_sampler.mjs"'
WORKER_DIST_IMPORT = '"./aminx-sampler/aminx_sampler.mjs"'

# The source-tree prefix the ProtonPotts worker imports its lib modules by,
# and the prefix they take once copied beside it. The rewrite replaces the
# full quoted specifier, one per lib file, so no regex is involved.
PROTONPOTTS_DEV_PREFIX = "../../browser/protonpotts-scorer/"
PROTONPOTTS_DIST_PREFIX = "./lib/"
PROTONPOTTS_DEV_PREFIX_LEAK = "../../browser/"


def build_commit() -> str:
    """The commit this build came from: CI's, or a plain 'unknown' locally.

    Unlike localfold's build_commit, this never shells out to git -- that
    codebase spawns a subprocess as a fallback; this one is meant to be
    "simple and obviously correct" per the spec this file was written
    against, and CI is the only caller that needs the commit at all
    (GITHUB_SHA is always set there).
    """
    import os
    return os.environ.get("GITHUB_SHA", "unknown")


def site_allow_list_problems() -> list[str]:
    """Every way SITE_FILES / PROTONPOTTS_SITE_FILES and the actual site/ tree (minus tests/) disagree."""
    problems = []
    listed_names = SITE_FILES + PROTONPOTTS_SITE_FILES
    for name in listed_names:
        if not (SITE / name).is_file():
            problems.append(f"site allow-list lists {name!r} but site/{name} does not exist")
    on_disk = {
        p.relative_to(SITE).as_posix()
        for p in SITE.rglob("*")
        if p.is_file() and "tests" not in p.relative_to(SITE).parts
    }
    listed = set(listed_names)
    for extra in sorted(on_disk - listed):
        problems.append(
            f"site/{extra} exists but is not in SITE_FILES or PROTONPOTTS_SITE_FILES -- add it"
            " deliberately or it will never ship",
        )
    return problems


def sampler_problems() -> list[str]:
    problems = []
    for name in SAMPLER_FILES:
        if not (SAMPLER_SRC / name).is_file():
            problems.append(f"SAMPLER_FILES lists {name!r} but {SAMPLER_SRC / name} does not exist")
    return problems


def worker_import_problems() -> list[str]:
    """The rewrite this build performs must actually match something.

    A silent zero-match "rewrite" is worse than no rewrite: the built
    dist/worker.js would still import the SOURCE-TREE path, which does not
    exist once deployed, and the page would 404 trying to start a Worker.
    """
    text = (SITE / "worker.js").read_text(encoding="utf-8")
    count = text.count(WORKER_DEV_IMPORT)
    if count == 0:
        return [f"site/worker.js does not contain the expected import {WORKER_DEV_IMPORT!r}"]
    if count > 1:
        return [f"site/worker.js contains {WORKER_DEV_IMPORT!r} {count} times; expected exactly 1"]
    return []


def lib_specifier(name: str) -> str:
    """The quoted source-tree specifier for one ProtonPotts lib module."""
    return f'"{PROTONPOTTS_DEV_PREFIX}{name}"'


def protonpotts_lib_specifier(name: str) -> str:
    """The quoted dist-tree specifier for one ProtonPotts lib module."""
    return f'"{PROTONPOTTS_DIST_PREFIX}{name}"'


def import_specifiers(text: str) -> list[str]:
    """Every module specifier a JS file imports, found by a plain line scan.

    Handles `... from "x"` and `... from 'x'` (including the closing line of a
    multi-line import) and bare side-effect imports `import "x"`. Comment lines
    (// and the leading * of a block comment) are skipped. This is deliberately
    not a JS parser: it is enough to check the flat, one-directory import
    graph of browser/protonpotts-scorer/, and any shape it does not understand
    shows up as a specifier that fails the closure check, not as a silent pass.
    """
    specs = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "*", "/*")):
            continue
        found = False
        for quote in ('"', "'"):
            marker = "from " + quote
            index = stripped.find(marker)
            if index != -1:
                start = index + len(marker)
                end = stripped.find(quote, start)
                if end != -1:
                    specs.append(stripped[start:end])
                found = True
                break
        if found or not stripped.startswith("import "):
            continue
        rest = stripped[len("import "):].lstrip()
        if rest[:1] in ('"', "'"):
            end = rest.find(rest[0], 1)
            if end != -1:
                specs.append(rest[1:end])
    return specs


def protonpotts_worker_import_problems(text: str) -> list[str]:
    """The lib rewrite applied to the worker's TEXT must match and must be clean.

    Pure on purpose: takes the worker source as a string so the tests can feed
    synthetic text. Reports a specifier present more than once, zero lib
    specifiers in total, and any "../../browser/" that survives the rewrite.
    A lib file the worker does not import directly is not a problem: it may be
    imported transitively, and the closure check covers that case.
    """
    problems = []
    total = 0
    for name in PROTONPOTTS_LIB_FILES:
        count = text.count(lib_specifier(name))
        total += count
        if count > 1:
            problems.append(
                f"{PROTONPOTTS_WORKER} imports {lib_specifier(name)} {count} times; expected at most 1",
            )
    if total == 0:
        problems.append(
            f"{PROTONPOTTS_WORKER} imports none of the protonpotts-scorer lib modules"
            f" (expected {PROTONPOTTS_DEV_PREFIX}<name> for one of {PROTONPOTTS_LIB_FILES})",
        )
    rewritten = rewrite_protonpotts_worker(text)
    if PROTONPOTTS_DEV_PREFIX_LEAK in rewritten:
        problems.append(
            f"{PROTONPOTTS_WORKER} still contains {PROTONPOTTS_DEV_PREFIX_LEAK!r} after the lib rewrite;"
            " a source-tree import would survive into dist/",
        )
    return problems


def rewrite_protonpotts_worker(text: str) -> str:
    """Apply the lib-specifier rewrite to the worker's text. Callers check problems first."""
    for name in PROTONPOTTS_LIB_FILES:
        text = text.replace(lib_specifier(name), protonpotts_lib_specifier(name))
    return text


def protonpotts_worker_source_problems() -> list[str]:
    """Read the real site worker and run the pure import check on it."""
    path = SITE / PROTONPOTTS_WORKER
    if not path.is_file():
        return [f"site/{PROTONPOTTS_WORKER} does not exist"]
    return protonpotts_worker_import_problems(path.read_text(encoding="utf-8"))


def lib_closure_problems(sources: dict[str, str]) -> list[str]:
    """Every import inside the lib modules must be a sibling that is also copied.

    `sources` maps a lib file name to its text. A specifier is acceptable only
    if it is exactly "./<name>" with <name> in PROTONPOTTS_LIB_FILES: that is
    what the copied lib/ directory can resolve. Anything else -- a "../layer_c/"
    reach, a bare package name, a sibling that is not on the allow-list -- would
    ship a module that fails to load in dist/, so it is refused here.
    """
    problems = []
    for name in PROTONPOTTS_LIB_FILES:
        if name not in sources:
            problems.append(f"lib source {name!r} was not provided to the closure check")
    for name in sorted(sources):
        for spec in import_specifiers(sources[name]):
            sibling = spec[2:] if spec.startswith("./") else ""
            if not sibling or "/" in sibling or sibling not in PROTONPOTTS_LIB_FILES:
                problems.append(
                    f"browser/protonpotts-scorer/{name} imports {spec!r}; lib modules may only"
                    f" import ./<name> with <name> in PROTONPOTTS_LIB_FILES",
                )
    return problems


def protonpotts_lib_source_problems() -> list[str]:
    """Check the real lib tree: every allow-listed file exists, and the closure holds."""
    problems = []
    sources = {}
    for name in PROTONPOTTS_LIB_FILES:
        path = PROTONPOTTS_LIB_SRC / name
        if not path.is_file():
            problems.append(f"PROTONPOTTS_LIB_FILES lists {name!r} but {path} does not exist")
            continue
        sources[name] = path.read_text(encoding="utf-8")
    if problems:
        return problems
    return lib_closure_problems(sources)


def protonpotts_problems() -> list[str]:
    """Every ProtonPotts sub-site problem that does not depend on model weights."""
    return protonpotts_worker_source_problems() + protonpotts_lib_source_problems()


def check() -> int:
    problems = (
        site_allow_list_problems() + sampler_problems() + worker_import_problems()
        + protonpotts_problems()
    )
    if problems:
        print("build_site.py --check found problems:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(
        "build_site.py --check: allow-list matches the tree, worker.js import rewrite will apply,"
        " ProtonPotts lib closure holds",
    )
    return 0


def check_stray_onnx_data(models_dir: Path) -> None:
    """Print (never copy) a warning for any '.onnx.data' sitting in models_dir.

    docs/browser_integration.md: "The export embeds its weights... A stray
    .onnx.data file next to it is stale. Don't ship it and don't rely on
    it." MODEL_FILES never names one, so nothing below could copy it by
    accident -- this exists purely so a developer pointing --models-dir at a
    directory with a leftover .onnx.data gets told why it did not appear in
    dist/, instead of silently wondering.
    """
    for path in sorted(models_dir.glob("*.onnx.data")):
        print(f"   ignoring {path.name}: stray external-data file, never shipped (see"
              " docs/browser_integration.md)")


def copy_models(out_dir: Path, models_dir: Path) -> dict:
    manifest_models = {}
    dest_dir = out_dir / "models"
    check_stray_onnx_data(models_dir)
    for name in MODEL_FILES:
        source = models_dir / name
        if not source.is_file():
            print(f"   {name}: not found in {models_dir}, skipping")
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest_dir / name)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest_models[name] = {"sha256": digest, "size_bytes": source.stat().st_size}
        print(f"   {name}: {source.stat().st_size / 1024 / 1024:.1f} MiB, sha256 {digest[:16]}...")
    return manifest_models


def named_protonpotts_files(manifest: dict) -> dict[str, str]:
    """Map every file a ProtonPotts MANIFEST.json names to its manifest sha256.

    The real manifest (scripts/browser_validation/protonpotts_*_export_gate.py)
    has `buckets` as a list of {"bucket", "graphs"}, and each bucket's `graphs`
    as a mapping of graph name to {"file", "sha256", ...}. Both containers are
    accepted as either a list or a dict, so a shape change in one exporter
    fails by a named problem rather than by naming nothing.
    """
    named: dict[str, str] = {}
    problems: list[str] = []
    buckets = manifest.get("buckets", [])
    bucket_list = buckets.values() if isinstance(buckets, dict) else buckets
    for bucket in bucket_list:
        graphs = bucket.get("graphs", {}) if isinstance(bucket, dict) else {}
        graph_list = graphs.values() if isinstance(graphs, dict) else graphs
        for graph in graph_list:
            if not isinstance(graph, dict) or "file" not in graph or "sha256" not in graph:
                problems.append(f"graph entry without both 'file' and 'sha256': {graph!r}"[:200])
                continue
            named[graph["file"]] = graph["sha256"]
    if problems:
        raise ValueError("; ".join(problems))
    return named


def plan_protonpotts_models(kind: str, models_dir: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    """Validate one ProtonPotts weight directory WITHOUT copying anything.

    Returns the (name, source) pairs to copy and the problems that refuse the
    directory. Every refusal the spec asks for is decided here, before dist/
    is removed: an unsafe name, a missing file, a sha256 that disagrees with
    the manifest, or an unreadable manifest.
    """
    manifest_path = models_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        return [], [f"protonpotts {kind}: {manifest_path} does not exist"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        named = named_protonpotts_files(manifest)
    except (ValueError, AttributeError, TypeError) as error:
        return [], [f"protonpotts {kind}: cannot read {manifest_path}: {error}"]

    problems = []
    plan = []
    for name, want in sorted(named.items()):
        if not name or "/" in name or "\\" in name or ".." in name:
            problems.append(f"protonpotts {kind}: manifest names {name!r}, which is not a bare filename")
            continue
        source = models_dir / name
        if not source.is_file():
            problems.append(f"protonpotts {kind}: manifest names {name!r} but {source} is missing")
            continue
        got = hashlib.sha256(source.read_bytes()).hexdigest()
        if got != want:
            problems.append(
                f"protonpotts {kind}: {name} sha256 {got[:16]}... differs from the manifest's {want[:16]}...",
            )
            continue
        plan.append((name, source))

    if not problems:
        for extra in sorted(models_dir.iterdir()):
            if extra.is_file() and extra.name != "MANIFEST.json" and extra.name not in named:
                print(f"   protonpotts {kind}: ignoring {extra.name}: not named by MANIFEST.json, never copied")
    return plan, problems


def copy_protonpotts_models(out_dir: Path, kind: str, models_dir: Path, plan: list[tuple[str, Path]]) -> None:
    """Copy an already-validated plan into dist/protonpotts/models/<kind>/."""
    dest_dir = out_dir / "protonpotts" / "models" / kind
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(models_dir / "MANIFEST.json", dest_dir / "MANIFEST.json")
    print(f"copying protonpotts {kind} models from {models_dir}:")
    for name, source in plan:
        shutil.copy2(source, dest_dir / name)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        print(f"   {name}: {source.stat().st_size / 1024 / 1024:.1f} MiB, sha256 {digest[:16]}...")


def named_potts_files(manifest: dict) -> dict[str, str]:
    """Map every file a Potts MANIFEST.json names to its manifest sha256."""
    named: dict[str, str] = {}
    problems: list[str] = []
    buckets = manifest.get("buckets", [])
    bucket_list = buckets.values() if isinstance(buckets, dict) else buckets
    for bucket in bucket_list:
        graphs = bucket.get("graphs", {}) if isinstance(bucket, dict) else {}
        graph_list = graphs.values() if isinstance(graphs, dict) else graphs
        for graph in graph_list:
            if not isinstance(graph, dict) or "file" not in graph or "sha256" not in graph:
                problems.append(f"graph entry without both 'file' and 'sha256': {graph!r}"[:200])
                continue
            named[graph["file"]] = graph["sha256"]
    if problems:
        raise ValueError("; ".join(problems))
    return named


def plan_potts_models(models_dir: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    """Validate Potts weight directory WITHOUT copying anything."""
    manifest_path = models_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        return [], [f"potts: {manifest_path} does not exist"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        named = named_potts_files(manifest)
    except (ValueError, AttributeError, TypeError) as error:
        return [], [f"potts: cannot read {manifest_path}: {error}"]

    problems = []
    plan = []
    for name, want in sorted(named.items()):
        if not name or "/" in name or "\\" in name or ".." in name:
            problems.append(f"potts: manifest names {name!r}, which is not a bare filename")
            continue
        source = models_dir / name
        if not source.is_file():
            problems.append(f"potts: manifest names {name!r} but {source} is missing")
            continue
        got = hashlib.sha256(source.read_bytes()).hexdigest()
        if got != want:
            problems.append(
                f"potts: {name} sha256 {got[:16]}... differs from the manifest's {want[:16]}...",
            )
            continue
        plan.append((name, source))

    if not problems:
        for extra in sorted(models_dir.iterdir()):
            if extra.is_file() and extra.name != "MANIFEST.json" and extra.name not in named:
                print(f"   potts: ignoring {extra.name}: not named by MANIFEST.json, never copied")
    return plan, problems


def copy_potts_models(out_dir: Path, models_dir: Path, plan: list[tuple[str, Path]]) -> None:
    """Copy an already-validated plan into dist/potts/models/."""
    dest_dir = out_dir / "potts" / "models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(models_dir / "MANIFEST.json", dest_dir / "MANIFEST.json")
    print(f"copying potts models from {models_dir}:")
    for name, source in plan:
        shutil.copy2(source, dest_dir / name)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        print(f"   {name}: {source.stat().st_size / 1024 / 1024:.1f} MiB, sha256 {digest[:16]}...")


def build(
    out_dir: Path,
    models_dir: Path | None,
    protonpotts_dirs: dict[str, Path] | None = None,
    potts_dir: Path | None = None,
) -> int:
    problems = (
        site_allow_list_problems() + sampler_problems() + worker_import_problems()
        + protonpotts_problems()
    )
    protonpotts_dirs = protonpotts_dirs or {}
    protonpotts_plans: dict[str, list[tuple[str, Path]]] = {}
    for kind, directory in protonpotts_dirs.items():
        plan, plan_problems = plan_protonpotts_models(kind, directory)
        problems += plan_problems
        protonpotts_plans[kind] = plan

    potts_plan: list[tuple[str, Path]] = []
    if potts_dir is not None:
        potts_plan, potts_problems = plan_potts_models(potts_dir)
        problems += potts_problems

    if problems:
        print("refusing to build -- the allow-list disagrees with the tree:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for name in SITE_FILES:
        shutil.copy2(SITE / name, out_dir / name)

    sampler_dest = out_dir / "aminx-sampler"
    sampler_dest.mkdir(parents=True)
    for name in SAMPLER_FILES:
        shutil.copy2(SAMPLER_SRC / name, sampler_dest / name)

    # THE ONE REWRITE THIS BUILD PERFORMS. worker_import_problems() already
    # proved WORKER_DEV_IMPORT appears exactly once in the source file, so
    # this replace() is exact -- no regex, nothing to accidentally match
    # twice.
    worker_path = out_dir / "worker.js"
    worker_text = worker_path.read_text(encoding="utf-8")
    worker_path.write_text(
        worker_text.replace(WORKER_DEV_IMPORT, WORKER_DIST_IMPORT), encoding="utf-8",
    )

    # THE PROTONPOTTS SUB-SITE. Each listed page file keeps its subdirectory,
    # so protonpotts/worker.mjs lands at dist/protonpotts/worker.mjs and its
    # sibling config.js and app.mjs resolve as they do in site/.
    for name in PROTONPOTTS_SITE_FILES:
        destination = out_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SITE / name, destination)

    lib_dest = out_dir / "protonpotts" / PROTONPOTTS_LIB_DIR
    lib_dest.mkdir(parents=True)
    for name in PROTONPOTTS_LIB_FILES:
        shutil.copy2(PROTONPOTTS_LIB_SRC / name, lib_dest / name)

    # The ProtonPotts worker's lib imports, rewritten on the copy. The
    # problem check already proved the rewrite matches and leaves no
    # source-tree prefix behind, so this is exact.
    protonpotts_worker = out_dir / PROTONPOTTS_WORKER
    protonpotts_worker.write_text(
        rewrite_protonpotts_worker(protonpotts_worker.read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    manifest_models = {}
    if models_dir is not None:
        print(f"copying models from {models_dir}:")
        manifest_models = copy_models(out_dir, models_dir)

    for kind in PROTONPOTTS_MODEL_KINDS:
        if kind in protonpotts_dirs:
            copy_protonpotts_models(out_dir, kind, protonpotts_dirs[kind], protonpotts_plans[kind])

    if potts_dir is not None:
        copy_potts_models(out_dir, potts_dir, potts_plan)

    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "models" / "MANIFEST.json").write_text(
        json.dumps({
            "commit": build_commit(),
            "built_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "models": manifest_models,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    (out_dir / ".nojekyll").write_text("", encoding="utf-8")

    total = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    count = sum(1 for p in out_dir.rglob("*") if p.is_file())
    model_note = f", {len(manifest_models)} model file(s)" if manifest_models else " (no model weights)"
    protonpotts_note = (
        f", protonpotts models: {', '.join(sorted(protonpotts_dirs))}" if protonpotts_dirs else ""
    )
    potts_note = f", potts models from {potts_dir}" if potts_dir else ""
    print(f"{out_dir}/  {count} files, {total / 1024 / 1024:.1f} MiB{model_note}{protonpotts_note}{potts_note}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="dist", help="output directory (default: dist)")
    parser.add_argument("--models-dir", default=None,
                         help="directory holding p07_sample_L128.onnx / p07_sample_L256.onnx")
    parser.add_argument("--potts-dir", default=None,
                         help="exported Potts directory (MANIFEST.json + named graphs)")
    parser.add_argument("--protonpotts-scoring-dir", default=None,
                         help="exported ProtonPotts scoring directory (MANIFEST.json + named graphs)")
    parser.add_argument("--protonpotts-ph-dir", default=None,
                         help="exported ProtonPotts pH-design directory (MANIFEST.json + named graphs)")
    parser.add_argument("--protonpotts-decoder-dir", default=None,
                         help="exported ProtonPotts decoder directory (MANIFEST.json + named graphs)")
    parser.add_argument("--check", action="store_true",
                         help="verify the allow-list against the source tree; build nothing")
    arguments = parser.parse_args()

    if arguments.check:
        raise SystemExit(check())

    models_directory = Path(arguments.models_dir).resolve() if arguments.models_dir else None
    potts_directory = Path(arguments.potts_dir).resolve() if arguments.potts_dir else None
    protonpotts_directories = {
        kind: Path(value).resolve()
        for kind, value in (
            ("scoring", arguments.protonpotts_scoring_dir),
            ("ph", arguments.protonpotts_ph_dir),
            ("decoder", arguments.protonpotts_decoder_dir),
        )
        if value
    }
    raise SystemExit(
        build(
            (ROOT / arguments.out).resolve(),
            models_directory,
            protonpotts_directories,
            potts_directory,
        )
    )
