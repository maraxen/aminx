"""Self-tests, soundness invariants, and positive/negative controls for incremental CI test selection.

Compatible with both `pytest` and `python3 -m unittest` so the selector can be
verified with stdlib Python before or without a full venv sync.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SELECTOR_PATH = _REPO_ROOT / "scripts" / "ci" / "select_affected_tests.py"


def _load_selector_module():
    spec = importlib.util.spec_from_file_location("select_affected_tests", _SELECTOR_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


selector = _load_selector_module()


class TestRepositorySoundnessInvariants(unittest.TestCase):
    """Pin structural invariants over the real aminx repository."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = selector.load_edge_config()
        cls.closures = selector.compute_all_test_closures(_REPO_ROOT, cls.config)

    def test_all_discovered_test_closures_are_sound(self) -> None:
        """Every test module in `tests/` must have zero unreviewed dynamic call sites."""
        unsound = {
            test_file: closure.undeclared_dynamic
            for test_file, closure in self.closures.items()
            if not closure.sound
        }
        self.assertEqual(
            unsound,
            {},
            f"Found test closures with unreviewed/mismatched dynamic sites: {unsound}",
        )

    def test_all_declared_edges_reference_existing_paths_and_exact_dynamic_counts(self) -> None:
        """Every file, prefix, guard, allowlist entry, and `dynamic_count` in `test_closure_edges.toml` is exact."""
        for gf in self.config.global_files:
            self.assertTrue((_REPO_ROOT / gf).is_file(), f"Missing global file: {gf}")
        for gp in self.config.global_prefixes:
            self.assertTrue((_REPO_ROOT / gp.rstrip("/")).is_dir(), f"Missing global prefix: {gp}")
        for guard in self.config.always_run:
            self.assertTrue((_REPO_ROOT / guard).is_file(), f"Missing always_run guard test: {guard}")
            self.assertIn(guard, self.closures, f"Guard test not discovered as a test file: {guard}")
        for unref in self.config.unreferenced_src_allowlist:
            self.assertTrue((_REPO_ROOT / unref).is_file(), f"Missing unreferenced_src allowlist file: {unref}")

        always_run_set = set(self.config.always_run)
        for mod_path, body in self.config.modules.items():
            self.assertTrue((_REPO_ROOT / mod_path).is_file(), f"Missing [module.{mod_path!r}] target")
            self.assertNotIn(
                "dynamic_ok",
                body,
                f"[module.{mod_path!r}] uses legacy boolean dynamic_ok; use exact integer dynamic_count",
            )
            for f in body.get("files", []):
                self.assertTrue(
                    (_REPO_ROOT / f).is_file(),
                    f"Missing declared file {f!r} in [module.{mod_path!r}]",
                )
            for dp in body.get("data_prefixes", []):
                self.assertTrue(
                    (_REPO_ROOT / dp.rstrip("/")).is_dir(),
                    f"Missing declared data_prefix {dp!r} in [module.{mod_path!r}]",
                )
            if "dynamic_count" in body:
                expected_count = body["dynamic_count"]
                self.assertIsInstance(expected_count, int)
                self.assertGreater(
                    expected_count,
                    0,
                    f"Stale dynamic_count=0 in [module.{mod_path!r}]; remove key instead",
                )
                scan = selector._scan_python_file(
                    _REPO_ROOT,
                    mod_path,
                    scan_string_literals=not (mod_path.startswith("tests/") and mod_path in always_run_set),
                )
                self.assertEqual(
                    len(scan.dynamic_lines),
                    expected_count,
                    f"[module.{mod_path!r}] declares dynamic_count={expected_count}, "
                    f"but AST scan found {len(scan.dynamic_lines)} lines: {scan.dynamic_lines}",
                )

    def test_guard_selftest_does_not_pollute_src_or_subtree_closures(self) -> None:
        """String literals in `always_run` guard tests must not pollute their own closure (C1 regression guard)."""
        own_closure = self.closures["tests/ci/test_select_affected_tests.py"]
        self.assertEqual(own_closure.data_prefixes, ())
        src_in_own = [f for f in own_closure.files if f.startswith("src/")]
        self.assertEqual(src_in_own, [])

    def test_every_src_aminx_module_is_covered_by_non_guard_test_or_allowlisted(self) -> None:
        """No `src/aminx/*.py` module may have zero PR-runnable non-guard test closures unless in `[unreferenced_src].allowlist`."""
        guard_set = set(self.config.always_run)
        covered_src: set[str] = set()
        for test_file, closure in self.closures.items():
            if test_file in guard_set or selector.is_pr_excluded_test_file(_REPO_ROOT, test_file):
                continue
            for f in closure.files:
                if f.startswith("src/aminx/") and f.endswith(".py"):
                    covered_src.add(f)

        all_src = {
            p.relative_to(_REPO_ROOT).as_posix()
            for p in (_REPO_ROOT / "src" / "aminx").rglob("*.py")
        }
        uncovered = all_src - covered_src
        unexpected_uncovered = sorted(uncovered - self.config.unreferenced_src_allowlist)
        self.assertEqual(
            unexpected_uncovered,
            [],
            f"src/aminx/ modules with 0 PR-runnable non-guard static test closures and not in [unreferenced_src].allowlist: {unexpected_uncovered}",
        )

    def test_family_driver_modules_and_conftest_global_invariants(self) -> None:
        """Pin `DRIVER_BACKED_FAMILY_MODULES` sync, `tests/conftest.py` global imports, and stdlib non-shadowing."""
        import ast as _ast

        # 1. Verify `DRIVER_BACKED_FAMILY_MODULES` in `src/aminx/host/family_driver.py` matches `_FAMILY_STRING_TO_MODULE`
        fd_tree = _ast.parse(
            (_REPO_ROOT / "src" / "aminx" / "host" / "family_driver.py").read_text(encoding="utf-8")
        )
        driver_map: dict[str, str] = {}
        for node in fd_tree.body:
            if (
                isinstance(node, _ast.AnnAssign)
                and isinstance(node.target, _ast.Name)
                and node.target.id == "DRIVER_BACKED_FAMILY_MODULES"
                and isinstance(node.value, _ast.Dict)
            ):
                for k, v in zip(node.value.keys, node.value.values, strict=True):
                    if isinstance(k, _ast.Constant) and isinstance(v, _ast.Constant):
                        driver_map[str(k.value)] = str(v.value)
        self.assertEqual(driver_map, dict(selector._FAMILY_STRING_TO_MODULE))
        for seam_test in (
            "tests/host/test_family_driver_seam.py",
            "tests/host/test_model_family_validation.py",
        ):
            soft_set = set(self.config.modules[seam_test].get("soft", []))
            self.assertTrue(
                set(driver_map.values()).issubset(soft_set),
                f"{seam_test} soft edges {soft_set} missing families from {driver_map}",
            )

        # 2. Every non-`src/` repo file imported at top level by `tests/conftest.py` must be in `[global].files`
        conf_scan = selector._scan_python_file(
            _REPO_ROOT,
            "tests/conftest.py",
            scan_string_literals=False,
        )
        for mod in conf_scan.target_modules:
            resolved = selector.resolve_module_file(
                _REPO_ROOT,
                mod,
                importer_dir=_REPO_ROOT / "tests",
            )
            if resolved is not None and not resolved.startswith("src/"):
                self.assertIn(
                    resolved,
                    self.config.global_files,
                    f"tests/conftest.py imports {resolved}, which must be in [global].files",
                )

        # 3. Stdlib `io` and `types` must not shadow `tests/io/__init__.py` or `tests/types/__init__.py`
        self.assertIsNone(selector.resolve_module_file(_REPO_ROOT, "io", importer_dir=_REPO_ROOT / "tests"))
        self.assertIsNone(selector.resolve_module_file(_REPO_ROOT, "types", importer_dir=_REPO_ROOT / "tests"))
        io_init_res = selector.select_tests(
            ["tests/io/__init__.py"],
            repo=_REPO_ROOT,
            config=self.config,
            closures=self.closures,
        )
        self.assertLess(len(io_init_res.implicated_tests), 25)
        types_init_res = selector.select_tests(
            ["tests/types/__init__.py"],
            repo=_REPO_ROOT,
            config=self.config,
            closures=self.closures,
        )
        self.assertLess(len(types_init_res.implicated_tests), 15)

    def test_Positive_Controls_family_and_module_isolation(self) -> None:
        """Positive controls: isolated module edits select the tests that cover them and exclude unrelated suites."""
        # 1. Potts runner change selects Potts tests and excludes LASEr / EBM tests
        potts_res = selector.select_tests(
            ["src/aminx/potts/runner.py"],
            repo=_REPO_ROOT,
            config=self.config,
            closures=self.closures,
        )
        self.assertEqual(potts_res.mode, "incremental")
        self.assertIn("tests/potts/test_runner_smoke.py", potts_res.implicated_tests)
        self.assertNotIn("tests/families/laser_mpnn/test_driver.py", potts_res.implicated_tests)
        self.assertNotIn("tests/ebm/test_model.py", potts_res.implicated_tests)
        self.assertNotIn("tests/ci/test_select_affected_tests.py", potts_res.implicated_tests)
        self.assertLess(len(potts_res.selected_tests), 60)

        # 2. LASEr driver change selects LASEr tests + family driver seam test, and excludes Potts / EBM tests
        laser_res = selector.select_tests(
            ["src/aminx/families/laser_mpnn/driver.py"],
            repo=_REPO_ROOT,
            config=self.config,
            closures=self.closures,
        )
        self.assertEqual(laser_res.mode, "incremental")
        self.assertIn("tests/families/laser_mpnn/test_driver.py", laser_res.implicated_tests)
        self.assertIn("tests/host/test_family_driver_seam.py", laser_res.implicated_tests)
        self.assertNotIn("tests/potts/test_runner_smoke.py", laser_res.implicated_tests)
        self.assertNotIn("tests/ebm/test_model.py", laser_res.implicated_tests)
        self.assertLess(len(laser_res.selected_tests), 60)

        # 3. EBM model change selects EBM tests and excludes Potts / LASEr tests
        ebm_res = selector.select_tests(
            ["src/aminx/ebm/model.py"],
            repo=_REPO_ROOT,
            config=self.config,
            closures=self.closures,
        )
        self.assertEqual(ebm_res.mode, "incremental")
        self.assertIn("tests/ebm/test_model.py", ebm_res.implicated_tests)
        self.assertNotIn("tests/potts/test_runner_smoke.py", ebm_res.implicated_tests)
        self.assertNotIn("tests/families/laser_mpnn/test_driver.py", ebm_res.implicated_tests)
        self.assertLess(len(ebm_res.selected_tests), 45)

        # 4. Agent MCP server change selects both direct and subprocess (`python -m aminx.agent.mcp`) tests
        mcp_res = selector.select_tests(
            ["src/aminx/agent/mcp.py"],
            repo=_REPO_ROOT,
            config=self.config,
            closures=self.closures,
        )
        self.assertEqual(mcp_res.mode, "incremental")
        self.assertIn("tests/agent/test_mcp_stdio.py", mcp_res.implicated_tests)
        self.assertIn("tests/agent/test_plugin.py", mcp_res.implicated_tests)

        # 5. Inline `python -c "import aminx.sampling.conditional_logits"` test is linked (`I2` regression guard)
        cond_res = selector.select_tests(
            ["src/aminx/sampling/conditional_logits.py"],
            repo=_REPO_ROOT,
            config=self.config,
            closures=self.closures,
        )
        self.assertEqual(cond_res.mode, "incremental")
        self.assertIn("tests/sampling/test_eqx_module_hash.py", cond_res.implicated_tests)

        # 6. Port wave target TOML and reference oracle select only the corresponding port wave test
        port_res = selector.select_tests(
            ["tests/port/targets/laser_encoder.toml", "tests/port/reference/laser_encoder/algo.py"],
            repo=_REPO_ROOT,
            config=self.config,
            closures=self.closures,
        )
        self.assertEqual(port_res.mode, "incremental")
        self.assertIn("tests/port/test_laser_encoder.py", port_res.implicated_tests)
        self.assertNotIn("tests/port/test_potts_energy.py", port_res.implicated_tests)
        self.assertIn("tests/lint/test_port_tolerances_match_targets.py", port_res.selected_tests)

        # 7. Non-Python data file (`NOTICE-weights.md`, `src/aminx/io/family_weights.json`) selects its test
        notice_res = selector.select_tests(
            ["NOTICE-weights.md"],
            repo=_REPO_ROOT,
            config=self.config,
            closures=self.closures,
        )
        self.assertEqual(notice_res.mode, "incremental")
        self.assertIn("tests/io/test_family_weights.py", notice_res.implicated_tests)

        # 8. Leaf modules (`io/designs.py`, `utils/catjac.py`) are not bloated by `if TYPE_CHECKING:` or relative-import hub cycles
        designs_res = selector.select_tests(
            ["src/aminx/io/designs.py"],
            repo=_REPO_ROOT,
            config=self.config,
            closures=self.closures,
        )
        self.assertEqual(designs_res.mode, "incremental")
        self.assertIn("tests/io/test_designs.py", designs_res.implicated_tests)
        self.assertLess(len(designs_res.implicated_tests), 10)

        catjac_res = selector.select_tests(
            ["src/aminx/utils/catjac.py"],
            repo=_REPO_ROOT,
            config=self.config,
            closures=self.closures,
        )
        self.assertEqual(catjac_res.mode, "incremental")
        self.assertIn("tests/utils/test_catjac.py", catjac_res.implicated_tests)
        self.assertLess(len(catjac_res.implicated_tests), 10)

    def test_Negative_Controls_fail_open_and_shared_core(self) -> None:
        """Negative controls: global files, unreferenced src files, deleted files, and root package init."""
        # 1. Global config files trigger `full` mode
        for global_file in (
            "pyproject.toml",
            "uv.lock",
            ".github/workflows/ci.yml",
            "tests/conftest.py",
            "scripts/ci/select_affected_tests.py",
            "scripts/ci/test_closure_edges.toml",
            "src/aminx/model_params/proteinmpnn_v_48_020.eqx",
        ):
            res = selector.select_tests(
                [global_file],
                repo=_REPO_ROOT,
                config=self.config,
                closures=self.closures,
            )
            self.assertEqual(res.mode, "full", f"Expected full mode for {global_file}, got {res.mode}")
            self.assertEqual(len(res.selected_tests), len(self.closures))

        # 2. Unreferenced `src/aminx/` module triggers `full` mode (fail-open)
        for unref in self.config.unreferenced_src_allowlist:
            res = selector.select_tests(
                [unref],
                repo=_REPO_ROOT,
                config=self.config,
                closures=self.closures,
            )
            self.assertEqual(res.mode, "full", f"Expected fail-open full mode for {unref}")

        # 3. Deleted or nonexistent `src/` or `tests/` file triggers `full` mode
        del_res = selector.select_tests(
            ["src/aminx/deleted_module_does_not_exist.py"],
            repo=_REPO_ROOT,
            config=self.config,
            closures=self.closures,
        )
        self.assertEqual(del_res.mode, "full")

        # 4. Modifying `src/aminx/__init__.py` invalidates all `aminx`-importing tests (>350 tests)
        root_init_res = selector.select_tests(
            ["src/aminx/__init__.py"],
            repo=_REPO_ROOT,
            config=self.config,
            closures=self.closures,
        )
        self.assertEqual(root_init_res.mode, "incremental")
        self.assertGreater(len(root_init_res.selected_tests), 350)

        # 5. Subdirectory `conftest.py` selects all tests in its subtree and not unrelated subtrees
        ebm_conf_res = selector.select_tests(
            ["tests/ebm/conftest.py"],
            repo=_REPO_ROOT,
            config=self.config,
            closures=self.closures,
        )
        self.assertEqual(ebm_conf_res.mode, "incremental")
        for t in self.closures:
            if t.startswith("tests/ebm/"):
                self.assertIn(t, ebm_conf_res.implicated_tests)
        self.assertNotIn("tests/potts/test_runner_smoke.py", ebm_conf_res.implicated_tests)

        # 6. Docs-only diff runs `guards_only` mode (all 20 always-run guard tests)
        docs_res = selector.select_tests(
            ["README.md", ".praxia/docs/specs/261026_incremental-ci-test-selection.md"],
            repo=_REPO_ROOT,
            config=self.config,
            closures=self.closures,
        )
        self.assertEqual(docs_res.mode, "guards_only")
        self.assertEqual(set(docs_res.selected_tests), set(self.config.always_run))

    def test_ci_workflow_wiring_invariants(self) -> None:
        """Pin `.github/workflows/ci.yml` incremental selector, cache, concurrency, and timeout wiring."""
        ci_yml = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("concurrency:", ci_yml)
        self.assertIn(
            "group: ci-${{ github.workflow }}-${{ github.event_name == 'pull_request' && github.ref || github.sha }}",
            ci_yml,
        )
        self.assertIn("cancel-in-progress: ${{ github.event_name == 'pull_request' }}", ci_yml)
        self.assertIn("timeout-minutes: 45", ci_yml)
        self.assertIn('echo "JAX_COMPILATION_CACHE_DIR=$RUNNER_TEMP/jax_compilation_cache" >> "$GITHUB_ENV"', ci_yml)
        self.assertIn("key: xla-${{ runner.os }}-py3.13-${{ hashFiles('uv.lock') }}-${{ github.sha }}", ci_yml)
        self.assertIn("scripts/ci/select_affected_tests.py", ci_yml)
        self.assertIn("--write-pytest-args .pytest-targets.txt", ci_yml)


class TestSyntheticRepoClosureMechanics(unittest.TestCase):
    """Unit tests over a synthetic repository verifying target-aware closure and fail-open rules."""

    def _make_synthetic_repo(self, root: Path, *, include_undeclared_dynamic: bool = False) -> None:
        (root / "scripts" / "ci").mkdir(parents=True)
        (root / "src" / "aminx" / "pkg_a").mkdir(parents=True)
        (root / "src" / "aminx" / "pkg_b").mkdir(parents=True)
        (root / "tests" / "sub").mkdir(parents=True)

        # Package init re-exporting pkg_b (simulating src/aminx/__init__.py)
        (root / "src" / "aminx" / "__init__.py").write_text(
            "from .pkg_b import heavy\n",
            encoding="utf-8",
        )
        (root / "src" / "aminx" / "pkg_a" / "__init__.py").write_text("", encoding="utf-8")
        (root / "src" / "aminx" / "pkg_a" / "leaf.py").write_text(
            "def f():\n    return 1\n",
            encoding="utf-8",
        )
        # Subpackage pkg_b/__init__.py imports sibling.py and defines `heavy` shadowing pkg_b/heavy.py (C2)
        (root / "src" / "aminx" / "pkg_b" / "sibling.py").write_text(
            "SIB = 42\n",
            encoding="utf-8",
        )
        (root / "src" / "aminx" / "pkg_b" / "__init__.py").write_text(
            "from .sibling import SIB\n",
            encoding="utf-8",
        )
        dyn_code = (
            "import importlib\ndef g(name):\n    return importlib.import_module(name)\n"
            if include_undeclared_dynamic
            else "def g():\n    return 2\n"
        )
        (root / "src" / "aminx" / "pkg_b" / "heavy.py").write_text(dyn_code, encoding="utf-8")

        (root / "tests" / "conftest.py").write_text("", encoding="utf-8")
        (root / "tests" / "shared_helper.py").write_text("SHARED = 1\n", encoding="utf-8")
        (root / "tests" / "test_a.py").write_text(
            "from aminx.pkg_a.leaf import f\ndef test_f():\n    assert f() == 1\n",
            encoding="utf-8",
        )
        (root / "tests" / "sub" / "conftest.py").write_text("X = 1\n", encoding="utf-8")
        (root / "tests" / "sub" / "local_helper.py").write_text("LOCAL = 2\n", encoding="utf-8")
        (root / "tests" / "sub" / "test_b.py").write_text(
            "from aminx.pkg_b import heavy\n"
            "from . import local_helper\n"
            "from ..shared_helper import SHARED\n"
            "def test_g():\n    assert heavy.g() == 2\n",
            encoding="utf-8",
        )
        (root / "tests" / "test_guard.py").write_text(
            "def test_guard():\n    assert True\n",
            encoding="utf-8",
        )

        edges_toml = """
[global]
files = ["pyproject.toml", "tests/conftest.py"]
prefixes = ["src/aminx/model_params/"]

[guards]
always_run = ["tests/test_guard.py"]

[unreferenced_src]
allowlist = []
"""
        (root / "scripts" / "ci" / "test_closure_edges.toml").write_text(
            edges_toml,
            encoding="utf-8",
        )

    def test_target_aware_import_and_subpackage_init_and_relative_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_synthetic_repo(root)
            cfg = selector.load_edge_config(root / "scripts/ci/test_closure_edges.toml")
            closures = selector.compute_all_test_closures(root, cfg)

            # test_a imports `aminx.pkg_a.leaf`:
            # - `src/aminx/__init__.py` MUST be in test_a's closure.files
            # - `src/aminx/pkg_b/heavy.py` MUST NOT leak into test_a's closure.files
            self.assertIn("src/aminx/__init__.py", closures["tests/test_a.py"].files)
            self.assertIn("src/aminx/pkg_a/leaf.py", closures["tests/test_a.py"].files)
            self.assertNotIn("src/aminx/pkg_b/heavy.py", closures["tests/test_a.py"].files)

            # test_b imports `from aminx.pkg_b import heavy`, `from . import local_helper`,
            # and `from ..shared_helper import SHARED`:
            # - `src/aminx/pkg_b/sibling.py` (imported by `pkg_b/__init__.py`) MUST be in test_b's closure (C2)
            # - `tests/sub/local_helper.py` and `tests/shared_helper.py` MUST be in test_b's closure (C4)
            self.assertIn("src/aminx/pkg_b/sibling.py", closures["tests/sub/test_b.py"].files)
            self.assertIn("src/aminx/pkg_b/heavy.py", closures["tests/sub/test_b.py"].files)
            self.assertIn("tests/sub/local_helper.py", closures["tests/sub/test_b.py"].files)
            self.assertIn("tests/shared_helper.py", closures["tests/sub/test_b.py"].files)

            # Editing `src/aminx/pkg_b/sibling.py` selects `tests/sub/test_b.py`, NOT `tests/test_a.py`
            res_sib = selector.select_tests(
                ["src/aminx/pkg_b/sibling.py"],
                repo=root,
                config=cfg,
                closures=closures,
            )
            self.assertEqual(res_sib.mode, "incremental")
            self.assertEqual(res_sib.implicated_tests, ("tests/sub/test_b.py",))

            # Editing `src/aminx/__init__.py` selects BOTH `tests/test_a.py` and `tests/sub/test_b.py`
            res_init = selector.select_tests(["src/aminx/__init__.py"], repo=root, config=cfg, closures=closures)
            self.assertEqual(res_init.mode, "incremental")
            self.assertEqual(
                res_init.implicated_tests,
                ("tests/sub/test_b.py", "tests/test_a.py"),
            )

    def test_undeclared_dynamic_call_fails_open_to_full_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_synthetic_repo(root, include_undeclared_dynamic=True)
            cfg = selector.load_edge_config(root / "scripts/ci/test_closure_edges.toml")
            closures = selector.compute_all_test_closures(root, cfg)

            self.assertFalse(closures["tests/sub/test_b.py"].sound)
            res = selector.select_tests(["src/aminx/pkg_a/leaf.py"], repo=root, config=cfg, closures=closures)
            self.assertEqual(res.mode, "full")
            self.assertIn("unsound test closures", res.reason)

    def test_cross_function_variable_shadowing_does_not_silence_dynamic_call(self) -> None:
        """A local `path` binding in one function must not mask a dynamic `path` parameter in another (C3)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_synthetic_repo(root)
            (root / "src" / "aminx" / "pkg_a" / "leaf.py").write_text(
                "import importlib.util\n"
                "from pathlib import Path\n"
                "def safe_helper():\n"
                "    path = Path(__file__).with_name('__init__.py')\n"
                "    return path\n"
                "def dynamic_loader(path: Path):\n"
                "    return importlib.util.spec_from_file_location('dyn', path)\n",
                encoding="utf-8",
            )
            cfg = selector.load_edge_config(root / "scripts/ci/test_closure_edges.toml")
            closures = selector.compute_all_test_closures(root, cfg)
            self.assertFalse(closures["tests/test_a.py"].sound)
            self.assertIn("src/aminx/pkg_a/leaf.py:7", closures["tests/test_a.py"].undeclared_dynamic)

    def test_syntax_error_in_python_file_fails_open_without_crashing(self) -> None:
        """An unparseable `.py` file in a closure marks the closure unsound and fails open to `full` (C4)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_synthetic_repo(root)
            (root / "src" / "aminx" / "pkg_a" / "leaf.py").write_text(
                "def broken(:\n",
                encoding="utf-8",
            )
            cfg = selector.load_edge_config(root / "scripts/ci/test_closure_edges.toml")
            closures = selector.compute_all_test_closures(root, cfg)
            self.assertFalse(closures["tests/test_a.py"].sound)
            res = selector.select_tests(["src/aminx/pkg_a/leaf.py"], repo=root, config=cfg, closures=closures)
            self.assertEqual(res.mode, "full")

    def test_select_from_git_diff_fails_open_on_invalid_git_ref(self) -> None:
        """If `git diff` fails (e.g. missing base ref), `select_from_git_diff` fails open to `full`."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_synthetic_repo(root)
            cfg = selector.load_edge_config(root / "scripts/ci/test_closure_edges.toml")
            closures = selector.compute_all_test_closures(root, cfg)
            res = selector.select_from_git_diff(
                "nonexistent-base-ref-0000",
                "HEAD",
                repo=root,
                config=cfg,
                closures=closures,
            )
            self.assertEqual(res.mode, "full")
            self.assertIn("git diff", res.reason)

    def test_cli_main_flags_write_pytest_args_and_json(self) -> None:
        """Verify `main()` with `--paths`, `--write-pytest-args`, `--json`, and `--explain`."""
        with tempfile.TemporaryDirectory() as tmp:
            out_file = Path(tmp) / "targets.txt"
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            with patch("sys.stdout", stdout_buf), patch("sys.stderr", stderr_buf):
                rc = selector.main(
                    [
                        "--paths",
                        "src/aminx/potts/runner.py",
                        "--explain",
                        "--json",
                        "--write-pytest-args",
                        str(out_file),
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(stdout_buf.getvalue())
            self.assertEqual(payload["mode"], "incremental")
            self.assertIn("tests/potts/test_runner_smoke.py", payload["implicated_tests"])
            written = out_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(written, payload["pytest_targets"])
            self.assertIn("=== Incremental CI Test Selection ===", stderr_buf.getvalue())


if __name__ == "__main__":
    unittest.main()
