"""Self-tests, soundness invariants, and positive/negative controls for incremental CI test selection.

Compatible with both `pytest` and `python3 -m unittest` so the selector can be
verified with stdlib Python before or without a full venv sync.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SELECTOR_PATH = _REPO_ROOT / "scripts" / "ci" / "select_affected_tests.py"


def _load_selector_module():
    spec = importlib.util.spec_from_file_location("select_affected_tests", _SELECTOR_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    import sys

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
        """Every test module in `tests/` must have zero undeclared dynamic call sites."""
        unsound = {
            test_file: closure.undeclared_dynamic
            for test_file, closure in self.closures.items()
            if not closure.sound
        }
        self.assertEqual(
            unsound,
            {},
            f"Found test closures with undeclared dynamic sites: {unsound}",
        )

    def test_all_declared_edges_reference_existing_paths(self) -> None:
        """Every file, prefix, guard, allowlist entry, and module key in `test_closure_edges.toml` exists."""
        for gf in self.config.global_files:
            self.assertTrue((_REPO_ROOT / gf).is_file(), f"Missing global file: {gf}")
        for gp in self.config.global_prefixes:
            self.assertTrue((_REPO_ROOT / gp.rstrip("/")).is_dir(), f"Missing global prefix: {gp}")
        for guard in self.config.always_run:
            self.assertTrue((_REPO_ROOT / guard).is_file(), f"Missing always_run guard test: {guard}")
            self.assertIn(guard, self.closures, f"Guard test not discovered as a test file: {guard}")
        for unref in self.config.unreferenced_src_allowlist:
            self.assertTrue((_REPO_ROOT / unref).is_file(), f"Missing unreferenced_src allowlist file: {unref}")

        for mod_path, body in self.config.modules.items():
            self.assertTrue((_REPO_ROOT / mod_path).is_file(), f"Missing [module.{mod_path!r}] target")
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

    def test_every_src_aminx_module_is_covered_or_explicitly_allowlisted(self) -> None:
        """No `src/aminx/*.py` module may have zero test closures unless in `[unreferenced_src].allowlist`."""
        covered_src: set[str] = set()
        for closure in self.closures.values():
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
            f"src/aminx/ modules with 0 static test closures and not in [unreferenced_src].allowlist: {unexpected_uncovered}",
        )

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
        self.assertLess(len(laser_res.selected_tests), 50)

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

        # 5. Port wave target TOML and reference oracle select the corresponding port wave test
        port_res = selector.select_tests(
            ["tests/port/targets/laser_encoder.toml"],
            repo=_REPO_ROOT,
            config=self.config,
            closures=self.closures,
        )
        self.assertEqual(port_res.mode, "incremental")
        self.assertIn("tests/port/test_laser_encoder.py", port_res.implicated_tests)
        self.assertIn("tests/lint/test_port_tolerances_match_targets.py", port_res.selected_tests)

        # 6. Non-Python data file (`NOTICE-weights.md`, `src/aminx/io/family_weights.json`) selects its test
        notice_res = selector.select_tests(
            ["NOTICE-weights.md"],
            repo=_REPO_ROOT,
            config=self.config,
            closures=self.closures,
        )
        self.assertEqual(notice_res.mode, "incremental")
        self.assertIn("tests/io/test_family_weights.py", notice_res.implicated_tests)

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
            ["README.md", "docs/unrelated_note.md"],
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
        self.assertIn("cancel-in-progress: ${{ github.event_name == 'pull_request' }}", ci_yml)
        self.assertIn("timeout-minutes: 45", ci_yml)
        self.assertIn("JAX_COMPILATION_CACHE_DIR:", ci_yml)
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
        (root / "src" / "aminx" / "pkg_b" / "__init__.py").write_text("", encoding="utf-8")
        dyn_code = (
            "import importlib\ndef g(name):\n    return importlib.import_module(name)\n"
            if include_undeclared_dynamic
            else "def g():\n    return 2\n"
        )
        (root / "src" / "aminx" / "pkg_b" / "heavy.py").write_text(dyn_code, encoding="utf-8")

        (root / "tests" / "conftest.py").write_text("", encoding="utf-8")
        (root / "tests" / "test_a.py").write_text(
            "from aminx.pkg_a.leaf import f\ndef test_f():\n    assert f() == 1\n",
            encoding="utf-8",
        )
        (root / "tests" / "sub" / "conftest.py").write_text("X = 1\n", encoding="utf-8")
        (root / "tests" / "sub" / "test_b.py").write_text(
            "from aminx.pkg_b.heavy import g\ndef test_g():\n    assert g() == 2\n",
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

    def test_target_aware_import_does_not_leak_sibling_reexports_of_parent_init(self) -> None:
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

            # Editing `src/aminx/pkg_b/heavy.py` selects `tests/sub/test_b.py` + guard, NOT `tests/test_a.py`
            res_b = selector.select_tests(["src/aminx/pkg_b/heavy.py"], repo=root, config=cfg, closures=closures)
            self.assertEqual(res_b.mode, "incremental")
            self.assertEqual(res_b.implicated_tests, ("tests/sub/test_b.py",))
            self.assertNotIn("tests/test_a.py", res_b.selected_tests)

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


if __name__ == "__main__":
    unittest.main()
