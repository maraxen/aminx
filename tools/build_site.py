"""Assemble the deployable aminx-ProteinMPNN-in-the-browser site into dist/.

    python3 tools/build_site.py --out dist
    python3 tools/build_site.py --out dist --models-dir /path/to/exported/models
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
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
SAMPLER_SRC = ROOT / "browser" / "aminx-sampler"

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

# Only these two are ever a browser-export target (browser_integration.md's
# BUCKETS ladder: 128, 256). Nothing else in a --models-dir is ever copied,
# named or not -- see check_stray_onnx_data.
MODEL_FILES = ["p07_sample_L128.onnx", "p07_sample_L256.onnx"]

SAMPLER_FILES = ["aminx_sampler.mjs", "runspec_core.mjs"]

# The exact specifier site/worker.js carries in the source tree, and what it
# becomes once aminx-sampler/ is copied in beside it. Kept as one pair here
# rather than a regex, so a change to either file is a one-line diff to find.
WORKER_DEV_IMPORT = '"../browser/aminx-sampler/aminx_sampler.mjs"'
WORKER_DIST_IMPORT = '"./aminx-sampler/aminx_sampler.mjs"'


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
    """Every way SITE_FILES and the actual site/ tree (minus tests/) disagree."""
    problems = []
    for name in SITE_FILES:
        if not (SITE / name).is_file():
            problems.append(f"SITE_FILES lists {name!r} but site/{name} does not exist")
    on_disk = {
        p.relative_to(SITE).as_posix()
        for p in SITE.rglob("*")
        if p.is_file() and "tests" not in p.relative_to(SITE).parts
    }
    listed = set(SITE_FILES)
    for extra in sorted(on_disk - listed):
        problems.append(
            f"site/{extra} exists but is not in SITE_FILES -- add it deliberately"
            " or it will never ship",
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


def check() -> int:
    problems = site_allow_list_problems() + sampler_problems() + worker_import_problems()
    if problems:
        print("build_site.py --check found problems:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("build_site.py --check: allow-list matches the tree, worker.js import rewrite will apply")
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


def build(out_dir: Path, models_dir: Path | None) -> int:
    problems = site_allow_list_problems() + sampler_problems() + worker_import_problems()
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

    manifest_models = {}
    if models_dir is not None:
        print(f"copying models from {models_dir}:")
        manifest_models = copy_models(out_dir, models_dir)

    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "models" / "MANIFEST.json").write_text(
        json.dumps({
            "commit": build_commit(),
            "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "models": manifest_models,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    (out_dir / ".nojekyll").write_text("", encoding="utf-8")

    total = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    count = sum(1 for p in out_dir.rglob("*") if p.is_file())
    model_note = f", {len(manifest_models)} model file(s)" if manifest_models else " (no model weights)"
    print(f"{out_dir}/  {count} files, {total / 1024 / 1024:.1f} MiB{model_note}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="dist", help="output directory (default: dist)")
    parser.add_argument("--models-dir", default=None,
                         help="directory holding p07_sample_L128.onnx / p07_sample_L256.onnx")
    parser.add_argument("--check", action="store_true",
                         help="verify the allow-list against the source tree; build nothing")
    arguments = parser.parse_args()

    if arguments.check:
        raise SystemExit(check())

    models_directory = Path(arguments.models_dir).resolve() if arguments.models_dir else None
    raise SystemExit(build((ROOT / arguments.out).resolve(), models_directory))
