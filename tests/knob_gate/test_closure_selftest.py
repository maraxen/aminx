"""The per-row closure rule: what it includes, and that every doubt makes a row STALE.

Two kinds of test, because the rule has two failure directions and only one is cheap
to see. A closure that is too big merely wastes a re-run. A closure that is too
small lets a stale row grade, which is the failure the gate exists to prevent, so
the synthetic tests pin the scanner's reach (lazy imports, relative imports,
package ``__init__``) and the fail-safe policy, and the real-repo tests pin the
declared structure of the eight rows, including the negative controls: a change to
shared core code MUST still invalidate every row.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from knob_gate._closure import (
  Closure,
  compute_closure,
  load_edges,
  loaded_outside_closure,
  row_is_stale,
)
from knob_gate._coverage import default_closure_for, repo_root

pytestmark = pytest.mark.redsox_gate

LASER_SLUGS = (
  "laser_proofread_parity",
  "laser_proofread_unconditional_parity",
  "laser_decode_e2e",
  "laser_score_parity",
)
POTTS_SLUGS = (
  "potts_ar_decode",
  "potts_energy_parity",
  "potts_ar_refine_exact",
  "potts_ddg_megascale",
)
PROTON_SLUGS = ("protonpotts_parity",)
TIER1_TIER2_SLUGS = (
  "tier1_variants_parity",
  "ca_only_parity",
)
ALL_SLUGS = (*LASER_SLUGS, *POTTS_SLUGS, *PROTON_SLUGS, *TIER1_TIER2_SLUGS)


def _make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
  for rel, text in files.items():
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
  return tmp_path


def _base_files() -> dict[str, str]:
  return {
    "scripts/parity/veh.py": "from aminx.core import run\n",
    "src/aminx/__init__.py": "",
    "src/aminx/core.py": "def run():\n  from aminx.lazy import helper\n  return helper\n",
    "src/aminx/lazy.py": "def helper(): ...\n",
    "src/aminx/unrelated.py": "X = 1\n",
  }


# --- scanner reach -----------------------------------------------------------------


def test_lazy_import_inside_a_function_is_in_the_closure(tmp_path: Path) -> None:
  repo = _make_repo(tmp_path, _base_files())
  closure = compute_closure(repo, "veh", edges={})
  assert "src/aminx/lazy.py" in closure.files
  assert "src/aminx/core.py" in closure.files
  assert closure.sound


def test_unrelated_module_is_outside_the_closure(tmp_path: Path) -> None:
  repo = _make_repo(tmp_path, _base_files())
  closure = compute_closure(repo, "veh", edges={})
  assert "src/aminx/unrelated.py" not in closure.files


def test_package_init_files_on_the_import_path_are_included(tmp_path: Path) -> None:
  files = _base_files() | {
    "scripts/parity/veh.py": "import aminx.fam.sub.leaf\n",
    "src/aminx/fam/__init__.py": "",
    "src/aminx/fam/sub/__init__.py": "",
    "src/aminx/fam/sub/leaf.py": "",
  }
  closure = compute_closure(_make_repo(tmp_path, files), "veh", edges={})
  assert {
    "src/aminx/__init__.py",
    "src/aminx/fam/__init__.py",
    "src/aminx/fam/sub/__init__.py",
    "src/aminx/fam/sub/leaf.py",
  } <= closure.files


def test_relative_and_from_package_submodule_imports_resolve(tmp_path: Path) -> None:
  files = _base_files() | {
    "scripts/parity/veh.py": "from aminx.fam import mod\n",
    "src/aminx/fam/__init__.py": "",
    "src/aminx/fam/mod.py": "from . import sibling\nfrom ..lazy import helper\n",
    "src/aminx/fam/sibling.py": "",
  }
  closure = compute_closure(_make_repo(tmp_path, files), "veh", edges={})
  assert "src/aminx/fam/mod.py" in closure.files  # `from pkg import submodule`
  assert "src/aminx/fam/sibling.py" in closure.files  # `from . import sibling`
  assert "src/aminx/lazy.py" in closure.files  # `from ..lazy import ...`


def test_parity_dir_sibling_helper_is_in_the_closure(tmp_path: Path) -> None:
  files = _base_files() | {
    "scripts/parity/veh.py": "from helper_mod import f\n",
    "scripts/parity/helper_mod.py": "def f(): ...\n",
  }
  closure = compute_closure(_make_repo(tmp_path, files), "veh", edges={})
  assert "scripts/parity/helper_mod.py" in closure.files


# --- dynamic sites fail safe ---------------------------------------------------------


def _dynamic_repo(tmp_path: Path) -> Path:
  files = _base_files() | {
    "src/aminx/core.py": "import importlib\n\ndef run(name):\n  importlib.import_module(name)\n",
  }
  return _make_repo(tmp_path, files)


def test_undeclared_dynamic_import_makes_the_closure_unsound(tmp_path: Path) -> None:
  closure = compute_closure(_dynamic_repo(tmp_path), "veh", edges={})
  assert not closure.sound
  assert closure.undeclared_dynamic == ("src/aminx/core.py:4",)


def test_unsound_closure_invalidates_on_any_narrowed_path(tmp_path: Path) -> None:
  closure = compute_closure(_dynamic_repo(tmp_path), "veh", edges={})
  assert row_is_stale(["src/aminx/unrelated.py"], closure)


def test_declared_dynamic_site_with_soft_edge_is_sound_and_pulls_the_target(
  tmp_path: Path,
) -> None:
  repo = _dynamic_repo(tmp_path)
  _make_repo(repo, {"src/aminx/fam/__init__.py": "", "src/aminx/fam/driver.py": ""})
  edges = {"veh": {"soft": ["aminx.fam"], "dynamic_ok": ["src/aminx/core.py"]}}
  closure = compute_closure(repo, "veh", edges=edges)
  assert closure.sound
  assert "src/aminx/fam/__init__.py" in closure.files
  assert not row_is_stale(["src/aminx/unrelated.py"], closure)


def test_literal_import_module_is_a_normal_edge_not_a_dynamic_site(tmp_path: Path) -> None:
  files = _base_files() | {
    "src/aminx/core.py": "import importlib\n\ndef run():\n  importlib.import_module('aminx.lazy')\n",
  }
  closure = compute_closure(_make_repo(tmp_path, files), "veh", edges={})
  assert closure.sound
  assert "src/aminx/lazy.py" in closure.files


def test_soft_edge_that_resolves_to_nothing_raises(tmp_path: Path) -> None:
  repo = _make_repo(tmp_path, _base_files())
  with pytest.raises(ValueError, match="resolves to no file"):
    compute_closure(repo, "veh", edges={"veh": {"soft": ["aminx.does_not_exist"]}})


def test_missing_vehicle_script_raises(tmp_path: Path) -> None:
  repo = _make_repo(tmp_path, _base_files())
  with pytest.raises(FileNotFoundError):
    compute_closure(repo, "no_such_vehicle", edges={})


# --- the staleness policy ------------------------------------------------------------


@pytest.mark.parametrize(
  ("path", "stale"),
  [
    ("src/aminx/lazy.py", True),  # in the closure
    ("scripts/parity/veh.py", True),  # the vehicle itself
    ("src/aminx/unrelated.py", False),  # a .py outside the closure
    ("src/aminx/brand_new_module.py", False),  # not imported by anything yet
    ("src/aminx/model_params/weights.safetensors", True),  # non-.py data: stays global
    ("scripts/parity/other_vehicle.py", False),
    ("scripts/parity/other_vehicle.bth.toml", False),  # pinned per row by sidecar_sha256
    ("scripts/parity/fixtures/data.json", True),  # non-.py data under a narrowed prefix
    ("pyproject.toml", True),
    ("uv.lock", True),
    ("scripts/recapture/convert.py", True),  # still global
    ("aminx-oracles/pin.toml", True),  # no row declares it: global, the safe side
    ("tests/port/test_new.py", False),  # re-run by the gate's step 2, not row evidence
    ("README.md", False),
    ("docs/guide.md", False),
    ("./src/aminx/lazy.py", True),  # leading ./ is normalised
    ("src\\aminx\\lazy.py", True),  # and so are backslashes
  ],
)
def test_row_is_stale_policy(tmp_path: Path, path: str, stale: bool) -> None:
  closure = compute_closure(_make_repo(tmp_path, _base_files()), "veh", edges={})
  assert row_is_stale([path], closure) is stale


def test_no_closure_reproduces_the_global_rule_for_narrowed_prefixes() -> None:
  assert row_is_stale(["src/aminx/anything.py"], None)
  assert row_is_stale(["scripts/parity/anything.py"], None)
  assert not row_is_stale(["docs/x.md"], None)


def test_declared_data_prefix_is_relevant_only_to_rows_that_declare_it() -> None:
  declares = Closure("a", frozenset(), data_prefixes=("scripts/parity/fixtures/a/",))
  other = Closure("b", frozenset())
  path = "scripts/parity/fixtures/a/golden.json"
  assert row_is_stale([path], declares)
  assert row_is_stale([path], other)  # undeclared non-.py data stays global: the safe side
  assert not row_is_stale(["scripts/parity/helper.py"], declares)


# --- the real eight rows -------------------------------------------------------------


def _real(slug: str) -> Closure:
  closure = default_closure_for(slug)
  assert closure is not None, f"{slug}: closure did not compute"
  return closure


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_every_ledger_row_has_a_sound_closure_containing_its_vehicle(slug: str) -> None:
  closure = _real(slug)
  assert closure.sound, closure.undeclared_dynamic
  assert f"scripts/parity/{slug}.py" in closure.files


def test_every_declared_row_exists_in_the_manifest() -> None:
  declared = set(load_edges())
  assert declared == set(ALL_SLUGS)


@pytest.mark.parametrize("slug", LASER_SLUGS)
def test_laser_rows_do_not_depend_on_potts_code(slug: str) -> None:
  closure = _real(slug)
  assert not [f for f in closure.files if "/potts" in f], "a LASEr row reaches Potts code"
  assert any("/laser_mpnn/" in f for f in closure.files)


@pytest.mark.parametrize("slug", POTTS_SLUGS)
def test_potts_rows_do_not_depend_on_laser_code(slug: str) -> None:
  closure = _real(slug)
  assert not [f for f in closure.files if "/laser" in f], "a Potts row reaches LASEr code"
  assert any("/potts_mpnn/" in f for f in closure.files)


@pytest.mark.parametrize("slug", TIER1_TIER2_SLUGS)
def test_tier1_tier2_rows_do_not_depend_on_potts_code(slug: str) -> None:
  closure = _real(slug)
  assert not [f for f in closure.files if "/potts" in f], f"a Tier 1/2 row ({slug}) reaches Potts code"
  assert not [f for f in closure.files if "/laser" in f], f"a Tier 1/2 row ({slug}) reaches LASEr code"


def test_family_change_invalidates_only_its_own_rows() -> None:
  # ProtonPottsMPNN is built on the Potts model, energy and merge code, so a Potts change reaches its row too;
  # the reverse must not hold, and neither family reaches LASEr.
  potts_change = ["src/aminx/families/potts_mpnn/decode.py"]
  laser_change = ["src/aminx/families/laser_mpnn/driver.py"]
  proton_change = ["src/aminx/families/protonpotts_mpnn/ph_descent.py"]
  assert [s for s in ALL_SLUGS if row_is_stale(potts_change, _real(s))] == [*POTTS_SLUGS, *PROTON_SLUGS]
  assert [s for s in ALL_SLUGS if row_is_stale(laser_change, _real(s))] == list(LASER_SLUGS)
  assert [s for s in ALL_SLUGS if row_is_stale(proton_change, _real(s))] == list(PROTON_SLUGS)


def test_oracle_dirs_scoping() -> None:
  laser_oracle = ["aminx-oracles/dumps/b3_laser_owned/sample.bin"]
  proton_oracle = ["aminx-oracles-protonpotts/weights.pt"]

  # LASEr rows declare b3_laser_owned; Potts and ProtonPotts do not
  assert [s for s in ALL_SLUGS if row_is_stale(laser_oracle, _real(s))] == list(LASER_SLUGS)

  # ProtonPotts rows declare aminx-oracles-protonpotts; Potts and LASEr do not
  assert [s for s in ALL_SLUGS if row_is_stale(proton_oracle, _real(s))] == list(PROTON_SLUGS)


def test_undeclared_oracle_path_is_global_declared_one_is_row_scoped() -> None:
  # A path no row declares must stale every row; a declared one only the rows that declare it.
  uncovered = "aminx-oracles/dumps/elsewhere/x.bin"
  assert [s for s in ALL_SLUGS if row_is_stale([uncovered], _real(s))] == list(ALL_SLUGS)
  declared = "aminx-oracles/dumps/b3_laser_owned/sample.bin"
  assert [s for s in ALL_SLUGS if row_is_stale([declared], _real(s))] == list(LASER_SLUGS)


def test_uv_lock_head_is_read_from_the_ref_not_the_working_tree(tmp_path: Path) -> None:
  # Pre-fix: changed_packages_in_uv_lock ignored its head argument and always read the working tree.
  from knob_gate._closure import changed_packages_in_uv_lock  # noqa: PLC0415

  def _git(*args: str) -> str:
    return subprocess.run(
      ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True,
    ).stdout.strip()

  lock_v1 = '[[package]]\nname = "jax"\nversion = "1.0"\n'
  lock_v2 = '[[package]]\nname = "jax"\nversion = "2.0"\n'
  _git("init", "-q")
  _git("config", "user.email", "t@example.invalid")
  _git("config", "user.name", "t")
  (tmp_path / "uv.lock").write_text(lock_v1, encoding="utf-8")
  _git("add", "uv.lock")
  _git("commit", "-q", "-m", "v1")
  base = _git("rev-parse", "HEAD")
  (tmp_path / "uv.lock").write_text(lock_v2, encoding="utf-8")
  _git("commit", "-q", "-am", "v2")
  # Working tree now equals HEAD (v2). Diffing base against HEAD must see the change; diffing
  # base against base itself must not, even though the working tree differs from base.
  (tmp_path / "uv.lock").write_text(lock_v1, encoding="utf-8")
  assert changed_packages_in_uv_lock(tmp_path, base, "HEAD") == {"jax"}
  assert changed_packages_in_uv_lock(tmp_path, base, base) == set()


def test_dependency_list_change_marks_the_package_changed(tmp_path: Path) -> None:
  from knob_gate._closure import changed_packages_in_uv_lock  # noqa: PLC0415

  def _git(*args: str) -> str:
    return subprocess.run(
      ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True,
    ).stdout.strip()

  before = '[[package]]\nname = "jax"\nversion = "1.0"\n'
  after = before + '[[package.dependencies]]\nname = "ml-dtypes"\n'
  _git("init", "-q")
  _git("config", "user.email", "t@example.invalid")
  _git("config", "user.name", "t")
  (tmp_path / "uv.lock").write_text(before, encoding="utf-8")
  _git("add", "uv.lock")
  _git("commit", "-q", "-m", "before")
  base = _git("rev-parse", "HEAD")
  (tmp_path / "uv.lock").write_text(after, encoding="utf-8")
  assert changed_packages_in_uv_lock(tmp_path, base) == {"jax"}


def test_declared_oracle_dirs_follow_the_edges_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  # Pre-fix the declared set was cached once against the global file, so a different or edited edges
  # file was answered from stale data. Now the cache is keyed by path and mtime.
  from knob_gate import _closure  # noqa: PLC0415

  edges = tmp_path / "closure_edges.toml"
  edges.write_text('[slug.a]\noracle_dirs = ["aminx-oracles/dumps/x/"]\n', encoding="utf-8")
  os.utime(edges, ns=(1_000_000_000, 1_000_000_000))
  monkeypatch.setattr(_closure, "_EDGES_FILE", edges)
  path = "aminx-oracles/dumps/x/a.bin"
  assert _closure.row_is_stale([path], Closure("b", frozenset())) is False

  edges.write_text("[slug.a]\noracle_dirs = []\n", encoding="utf-8")
  os.utime(edges, ns=(2_000_000_000, 2_000_000_000))
  assert _closure.row_is_stale([path], Closure("b", frozenset())) is True


@pytest.mark.parametrize(
  ("table", "body"),
  [
    ("optional-dependencies", 'cuda = [{ name = "jax-cuda" }]\n'),
    ("dev-dependencies", 'test = [{ name = "pytest" }]\n'),
  ],
)
def test_extra_and_dev_group_changes_mark_the_package_changed(
  tmp_path: Path, table: str, body: str,
) -> None:
  from knob_gate._closure import changed_packages_in_uv_lock  # noqa: PLC0415

  def _git(*args: str) -> str:
    return subprocess.run(
      ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True,
    ).stdout.strip()

  before = '[[package]]\nname = "jax"\nversion = "1.0"\n'
  after = before + f"[package.{table}]\n" + body
  _git("init", "-q")
  _git("config", "user.email", "t@example.invalid")
  _git("config", "user.name", "t")
  (tmp_path / "uv.lock").write_text(before, encoding="utf-8")
  _git("add", "uv.lock")
  _git("commit", "-q", "-m", "before")
  base = _git("rev-parse", "HEAD")
  (tmp_path / "uv.lock").write_text(after, encoding="utf-8")
  assert changed_packages_in_uv_lock(tmp_path, base) == {"jax"}


@pytest.mark.parametrize(
  ("source", "expected"),
  [
    ("import orbax.checkpoint as ocp\n", "orbax-checkpoint"),
    ("from PIL import Image\n", "pillow"),
    ("import sklearn.linear_model\n", "scikit-learn"),
    ("import cv2\n", "opencv-python"),
    ("from google.protobuf import message\n", "protobuf"),
    ("import yaml\n", "pyyaml"),
  ],
)
def test_import_names_map_to_distribution_names(tmp_path: Path, source: str, expected: str) -> None:
  from knob_gate._closure import _imported_third_party  # noqa: PLC0415

  repo = _make_repo(tmp_path, {"scripts/parity/veh.py": source})
  assert expected in _imported_third_party(repo, "scripts/parity/veh.py")


def test_uv_lock_scoping() -> None:
  potts_closure = _real("potts_ar_decode")
  # Modifying unrelated package (e.g. ruff, sphinx) does not invalidate potts_ar_decode
  assert not row_is_stale(["uv.lock"], potts_closure, uv_changed_packages={"ruff"})
  assert not row_is_stale(["uv.lock"], potts_closure, uv_changed_packages={"sphinx"})

  # Modifying imported package (e.g. jax, numpy) does invalidate potts_ar_decode
  assert row_is_stale(["uv.lock"], potts_closure, uv_changed_packages={"jax"})
  assert row_is_stale(["uv.lock"], potts_closure, uv_changed_packages={"numpy"})

  # No changed packages passed -> conservative fallback marks stale
  assert row_is_stale(["uv.lock"], potts_closure)


@pytest.mark.parametrize("slug", PROTON_SLUGS)
def test_protonpotts_row_reaches_its_family_and_not_laser(slug: str) -> None:
  closure = _real(slug)
  assert any("/protonpotts_mpnn/" in f for f in closure.files)
  assert not [f for f in closure.files if "/laser" in f], "a ProtonPotts row reaches LASEr code"
  # The wave scripts are run as subprocesses and the converter is loaded by path: neither is an import, so each must
  # be in the closure through a declared edge, or editing a wave could not stale the row.
  for needed in (
    "scripts/parity/protonpotts_ph_driver_parity.py",
    "scripts/parity/protonpotts_features_parity.py",
    "scripts/parity/convert_protonpotts_checkpoint.py",
  ):
    assert needed in closure.files, needed


@pytest.mark.parametrize("slug", [*LASER_SLUGS, *POTTS_SLUGS, *TIER1_TIER2_SLUGS])
def test_other_families_do_not_depend_on_protonpotts_code(slug: str) -> None:
  closure = _real(slug)
  assert not [f for f in closure.files if "protonpotts" in f], "a Potts/LASEr row reaches ProtonPotts code"


@pytest.mark.parametrize(
  "shared",
  [
    "src/aminx/__init__.py",  # eager package import: every vehicle runs it
    "src/aminx/host/runner.py",
    "scripts/parity/artifact_key.py",
    "pyproject.toml",
    "uv.lock",
    "scripts/recapture/pottsmpnn_model_to_eqx.py",
  ],
)
def test_negative_control_shared_change_still_invalidates_every_row(shared: str) -> None:
  """If this ever fails, the closure has stopped being a conservative narrowing."""
  not_stale = [s for s in ALL_SLUGS if not row_is_stale([shared], _real(s))]
  assert not not_stale, f"{shared} left {not_stale} fresh"


def test_the_repo_root_this_test_sees_has_the_vehicles() -> None:
  assert (repo_root() / "scripts" / "parity" / "laser_decode_e2e.py").is_file()


# --- recording what a run really loaded -----------------------------------------------

HOOK_DIR = repo_root() / "scripts" / "redsox" / "closure_hook"


def test_loaded_outside_closure_names_only_uncovered_narrowed_py_files() -> None:
  closure = Closure("v", frozenset({"src/aminx/a.py", "scripts/parity/v.py"}))
  loaded = [
    "src/aminx/a.py",
    "src/aminx/b.py",  # narrowed .py not in the closure: the defect this check exists to find
    "src/aminx/data.json",  # not .py: ignored here (non-.py is global in row_is_stale)
    "tests/port/x.py",  # outside the narrowed prefixes
    "./src/aminx/c.py",
  ]
  assert loaded_outside_closure(loaded, closure) == ["src/aminx/b.py", "src/aminx/c.py"]


def _run_hooked(tmp_path: Path, script_name: str, env_script: str) -> Path:
  repo = tmp_path / "repo"
  (repo / "src" / "zzfake").mkdir(parents=True)
  (repo / "src" / "zzfake" / "mod.py").write_text("X = 1\n", encoding="utf-8")
  (repo / "scripts" / "parity").mkdir(parents=True)
  script = repo / "scripts" / "parity" / script_name
  src = str(repo / "src")
  script.write_text(f"import sys\nsys.path.insert(0, {src!r})\nimport zzfake.mod\n", encoding="utf-8")
  out = tmp_path / "loaded.json"
  env = {
    **os.environ,
    "AMINX_CLOSURE_OUT": str(out),
    "AMINX_CLOSURE_SCRIPT": env_script,
    "AMINX_CLOSURE_REPO": str(repo),
    "PYTHONPATH": str(HOOK_DIR),
  }
  done = subprocess.run([sys.executable, str(script)], env=env, check=False, capture_output=True)
  assert done.returncode == 0, done.stderr.decode()
  return out


def test_hook_records_loaded_repo_files_for_the_named_vehicle(tmp_path: Path) -> None:
  out = _run_hooked(tmp_path, "veh.py", "veh.py")
  loaded = json.loads(out.read_text(encoding="utf-8"))["loaded"]
  assert "src/zzfake/mod.py" in loaded
  assert "scripts/parity/veh.py" in loaded
  assert all(not p.startswith("/") for p in loaded)  # repo-relative


def test_hook_is_silent_for_a_process_that_is_not_the_vehicle(tmp_path: Path) -> None:
  """An oracle interpreter spawned by a vehicle inherits the env and must not write."""
  out = _run_hooked(tmp_path, "child.py", "veh.py")
  assert not out.exists()


@pytest.mark.parametrize("slug", TIER1_TIER2_SLUGS)
def test_tier1_tier2_runtime_closure_isolation(slug: str, tmp_path: Path) -> None:
  """Regression test: runtime execution must not load uncovered files or sibling families."""
  out_loaded = tmp_path / f"loaded_{slug}.json"
  python_bin = sys.executable
  for candidate in (
    repo_root().parent.parent / ".venv" / "bin" / "python",
    repo_root() / ".venv" / "bin" / "python",
  ):
    if candidate.is_file() and os.access(candidate, os.X_OK):
      python_bin = str(candidate)
      break
  cur_pythonpath = os.environ.get("PYTHONPATH", "")
  env = {
    **os.environ,
    "AMINX_CLOSURE_OUT": str(out_loaded),
    "AMINX_CLOSURE_SCRIPT": f"{slug}.py",
    "AMINX_CLOSURE_REPO": str(repo_root()),
    "PYTHONPATH": f"{HOOK_DIR}:{cur_pythonpath}" if cur_pythonpath else str(HOOK_DIR),
    "OMP_NUM_THREADS": "4",
  }
  cmd = [
    python_bin,
    str(repo_root() / "scripts" / "parity" / f"{slug}.py"),
    "--synthetic",
    "--out",
    str(tmp_path / f"{slug}_out.json"),
  ]
  done = subprocess.run(cmd, env=env, check=False, capture_output=True, text=True)
  assert done.returncode == 0, f"Vehicle {slug} failed:\nSTDOUT:\n{done.stdout}\nSTDERR:\n{done.stderr}"
  assert out_loaded.exists(), f"Closure hook did not record loaded files for {slug}"
  data = json.loads(out_loaded.read_text(encoding="utf-8"))
  loaded = data.get("loaded", [])
  assert not [f for f in loaded if "/potts" in f], f"Vehicle {slug} dynamically loaded potts code"
  assert not [f for f in loaded if "/laser" in f], f"Vehicle {slug} dynamically loaded laser code"
  closure = _real(slug)
  uncovered = loaded_outside_closure(loaded, closure)
  assert uncovered == [], f"Vehicle {slug} loaded files outside declared closure: {uncovered}"
