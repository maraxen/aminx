"""Tests for tools/build_site.py: the ProtonPotts sub-site allow-list, import rewrite, lib closure and weights.

Positive controls (a clean input must pass) and negative controls (a deliberately broken input must be refused)
sit side by side, so a check that can only pass is visible as one. The pure checks take text or dicts, and the
weight copy path is exercised on synthetic directories under tmp_path; nothing here touches the real weights.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_build_site() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_site_under_test", ROOT / "tools" / "build_site.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bs = _load_build_site()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_weights_dir(tmp_path: Path, files: dict[str, bytes], manifest_sha: dict[str, str] | None = None) -> Path:
    """A synthetic exported directory: MANIFEST.json naming every key of `files`, with correct digests by default."""
    directory = tmp_path / "exported"
    directory.mkdir()
    graphs = {}
    for name, data in files.items():
        (directory / name).write_bytes(data)
        graphs[name.removesuffix(".onnx")] = {
            "file": name,
            "sha256": (manifest_sha or {}).get(name, _sha(data)),
            "bytes": len(data),
        }
    manifest = {"family": "protonpottsmpnn_test", "buckets": [{"bucket": 128, "graphs": graphs}]}
    (directory / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


# --- weights: positive control --------------------------------------------------------------------------------


def test_matching_manifest_plans_and_copies_named_files(tmp_path: Path) -> None:
    directory = _write_weights_dir(tmp_path, {"a.onnx": b"alpha-bytes", "b.onnx": b"beta-bytes"})

    plan, problems = bs.plan_protonpotts_models("scoring", directory)

    assert problems == []
    assert sorted(name for name, _ in plan) == ["a.onnx", "b.onnx"]
    out_dir = tmp_path / "dist"
    bs.copy_protonpotts_models(out_dir, "scoring", directory, plan)
    dest = out_dir / "protonpotts" / "models" / "scoring"
    assert (dest / "a.onnx").read_bytes() == b"alpha-bytes"
    assert (dest / "b.onnx").read_bytes() == b"beta-bytes"
    assert (dest / "MANIFEST.json").is_file()


# --- weights: negative controls -------------------------------------------------------------------------------


def test_wrong_sha256_is_refused(tmp_path: Path) -> None:
    directory = _write_weights_dir(
        tmp_path, {"a.onnx": b"alpha-bytes"}, manifest_sha={"a.onnx": _sha(b"something else")},
    )

    plan, problems = bs.plan_protonpotts_models("ph", directory)

    assert plan == []
    assert any("differs from the manifest" in problem for problem in problems)


def test_missing_named_file_is_refused(tmp_path: Path) -> None:
    directory = _write_weights_dir(tmp_path, {"a.onnx": b"alpha-bytes"})
    (directory / "a.onnx").unlink()

    plan, problems = bs.plan_protonpotts_models("decoder", directory)

    assert plan == []
    assert any("is missing" in problem for problem in problems)


@pytest.mark.parametrize("bad_name", ["../x.onnx", "sub/x.onnx", "sub\\x.onnx", "a..b.onnx"])
def test_unsafe_manifest_name_is_refused(tmp_path: Path, bad_name: str) -> None:
    directory = tmp_path / "exported"
    directory.mkdir()
    manifest = {"buckets": [{"bucket": 128, "graphs": {"g": {"file": bad_name, "sha256": "0" * 64}}}]}
    (directory / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    plan, problems = bs.plan_protonpotts_models("scoring", directory)

    assert plan == []
    assert any("not a bare filename" in problem for problem in problems)


def test_unnamed_extra_file_is_not_copied(tmp_path: Path) -> None:
    directory = _write_weights_dir(tmp_path, {"a.onnx": b"alpha-bytes"})
    (directory / "a.onnx.data").write_bytes(b"stale external data")
    (directory / "stray.onnx").write_bytes(b"not in the manifest")

    plan, problems = bs.plan_protonpotts_models("scoring", directory)
    assert problems == []
    out_dir = tmp_path / "dist"
    bs.copy_protonpotts_models(out_dir, "scoring", directory, plan)

    dest = out_dir / "protonpotts" / "models" / "scoring"
    assert sorted(p.name for p in dest.iterdir()) == ["MANIFEST.json", "a.onnx"]


def test_missing_manifest_is_refused(tmp_path: Path) -> None:
    plan, problems = bs.plan_protonpotts_models("ph", tmp_path)

    assert plan == []
    assert any("MANIFEST.json" in problem for problem in problems)


# --- import rewrite on synthetic worker text ------------------------------------------------------------------


def test_worker_with_one_lib_import_per_name_is_clean() -> None:
    text = (
        'import { run } from "../../browser/protonpotts-scorer/protonpotts_scorer.mjs";\n'
        'import { PhDesigner } from "../../browser/protonpotts-scorer/ph_design.mjs";\n'
    )

    assert bs.protonpotts_worker_import_problems(text) == []
    rewritten = bs.rewrite_protonpotts_worker(text)
    assert '"./lib/protonpotts_scorer.mjs"' in rewritten
    assert '"./lib/ph_design.mjs"' in rewritten


def test_duplicated_lib_specifier_is_flagged() -> None:
    line = 'import { run } from "../../browser/protonpotts-scorer/protonpotts_scorer.mjs";\n'
    problems = bs.protonpotts_worker_import_problems(line + line)

    assert any("2 times" in problem for problem in problems)


def test_leftover_browser_prefix_is_flagged() -> None:
    text = (
        'import { run } from "../../browser/protonpotts-scorer/protonpotts_scorer.mjs";\n'
        'import { x } from "../../browser/layer_c/other.mjs";\n'
    )

    problems = bs.protonpotts_worker_import_problems(text)

    assert any("still contains" in problem for problem in problems)


def test_worker_with_no_lib_import_is_flagged() -> None:
    problems = bs.protonpotts_worker_import_problems('import { x } from "./app.mjs";\n')

    assert any("imports none of the protonpotts-scorer lib modules" in problem for problem in problems)


# --- lib closure ----------------------------------------------------------------------------------------------


def _all_lib_sources(overrides: dict[str, str] | None = None) -> dict[str, str]:
    sources = {name: "export const x = 1;\n" for name in bs.PROTONPOTTS_LIB_FILES}
    sources.update(overrides or {})
    return sources


def test_lib_closure_accepts_sibling_imports() -> None:
    sources = _all_lib_sources({
        "protonpotts_scorer.mjs": 'import { VOCABULARY } from "./protonpotts_inputs.mjs";\n',
        "ph_design.mjs": 'import {\n  X,\n} from "./ph_plan.mjs";\n',
    })

    assert bs.lib_closure_problems(sources) == []


def test_lib_closure_flags_cross_directory_import() -> None:
    sources = _all_lib_sources({
        "ph_plan.mjs": 'import { y } from "../layer_c/x.mjs";\n',
    })

    problems = bs.lib_closure_problems(sources)

    assert any("'../layer_c/x.mjs'" in problem for problem in problems)


def test_lib_closure_flags_sibling_outside_allow_list() -> None:
    sources = _all_lib_sources({
        "ph_plan.mjs": 'import { y } from "./run_ph_design_node.mjs";\n',
    })

    problems = bs.lib_closure_problems(sources)

    assert any("run_ph_design_node.mjs" in problem for problem in problems)


# --- allow-list and the real tree -----------------------------------------------------------------------------


def test_protonpotts_files_are_on_the_allow_list() -> None:
    listed = set(bs.SITE_FILES) | set(bs.PROTONPOTTS_SITE_FILES)

    for name in bs.PROTONPOTTS_SITE_FILES:
        assert name in listed
        assert name.startswith("protonpotts/")


def test_real_tree_check_passes() -> None:
    """Only passes once site/protonpotts/ and browser/protonpotts-scorer/ exist as the allow-list describes."""
    assert bs.check() == 0


# --- potts weights: positive control --------------------------------------------------------------------------


def test_potts_matching_manifest_plans_and_copies_named_files(tmp_path: Path) -> None:
    directory = _write_weights_dir(tmp_path, {"a.onnx": b"alpha-bytes", "b.onnx": b"beta-bytes"})

    plan, problems = bs.plan_potts_models(directory)

    assert problems == []
    assert sorted(name for name, _ in plan) == ["a.onnx", "b.onnx"]
    out_dir = tmp_path / "dist"
    bs.copy_potts_models(out_dir, directory, plan)
    dest = out_dir / "potts" / "models"
    assert (dest / "a.onnx").read_bytes() == b"alpha-bytes"
    assert (dest / "b.onnx").read_bytes() == b"beta-bytes"
    assert (dest / "MANIFEST.json").is_file()


# --- potts weights: negative controls -------------------------------------------------------------------------


def test_potts_wrong_sha256_is_refused(tmp_path: Path) -> None:
    directory = _write_weights_dir(
        tmp_path, {"a.onnx": b"alpha-bytes"}, manifest_sha={"a.onnx": _sha(b"something else")},
    )

    plan, problems = bs.plan_potts_models(directory)

    assert plan == []
    assert any("differs from the manifest" in problem for problem in problems)


def test_potts_missing_named_file_is_refused(tmp_path: Path) -> None:
    directory = _write_weights_dir(tmp_path, {"a.onnx": b"alpha-bytes"})
    (directory / "a.onnx").unlink()

    plan, problems = bs.plan_potts_models(directory)

    assert plan == []
    assert any("is missing" in problem for problem in problems)


def test_potts_missing_manifest_is_refused(tmp_path: Path) -> None:
    plan, problems = bs.plan_potts_models(tmp_path)

    assert plan == []
    assert any("MANIFEST.json" in problem for problem in problems)


@pytest.mark.parametrize("bad_name", ["../x.onnx", "sub/x.onnx", "sub\\x.onnx", "a..b.onnx"])
def test_potts_unsafe_manifest_name_is_refused(tmp_path: Path, bad_name: str) -> None:
    directory = tmp_path / "exported"
    directory.mkdir()
    manifest = {"buckets": [{"bucket": 128, "graphs": {"g": {"file": bad_name, "sha256": "0" * 64}}}]}
    (directory / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    plan, problems = bs.plan_potts_models(directory)

    assert plan == []
    assert any("not a bare filename" in problem for problem in problems)

