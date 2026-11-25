"""Tests asserting the project dependency layout conforms to D24 and spec §6 AE-110."""

from __future__ import annotations

import pathlib
import tomllib


def load_pyproject() -> dict:
  root = pathlib.Path(__file__).resolve().parents[1]
  with (root / "pyproject.toml").open("rb") as f:
    return tomllib.load(f)


def test_pep735_dependency_groups_exact() -> None:
  """Assert expected PEP 735 dependency groups exist (test, lint, docs, benchmark, dev)."""
  data = load_pyproject()
  groups = data.get("dependency-groups", {})
  expected_groups = {"test", "lint", "docs", "benchmark", "dev"}
  assert set(groups.keys()) == expected_groups


def test_optional_dependencies_runtime_only_extras() -> None:
  """Assert optional-dependencies holds only runtime extras matching D24 / AE-110."""
  data = load_pyproject()
  extras = data.get("project", {}).get("optional-dependencies", {})
  expected_extras = {
    "eda",
    "agent",
    "laser",
    "ebm",
    "onnx",
    "foldcomp",
    "cpu",
    "cuda12",
  }
  assert set(extras.keys()) == expected_extras


def test_no_group_in_optional_dependencies() -> None:
  """Assert no PEP 735 dependency group also appears in [project.optional-dependencies]."""
  data = load_pyproject()
  groups = set(data.get("dependency-groups", {}).keys())
  extras = set(data.get("project", {}).get("optional-dependencies", {}).keys())
  overlap = groups & extras
  assert not overlap, f"Groups must not overlap with optional-dependencies extras: {overlap}"


def test_ebm_only_packages() -> None:
  """Assert EBM-only runtime dependencies (orbax-checkpoint, einops) live only in 'ebm'."""
  data = load_pyproject()
  runtime_deps = data.get("project", {}).get("dependencies", [])
  extras = data.get("project", {}).get("optional-dependencies", {})

  ebm_deps = extras.get("ebm", [])
  for pkg in ("orbax-checkpoint", "einops"):
    assert any(dep.startswith(pkg) for dep in ebm_deps), f"{pkg} must be in ebm extra"
    for dep in runtime_deps:
      assert not dep.startswith(pkg), f"{pkg} must not be in runtime dependencies: {dep}"
    for extra_name, dep_list in extras.items():
      if extra_name != "ebm":
        for dep in dep_list:
          assert not dep.startswith(pkg), f"{pkg} found in non-ebm extra '{extra_name}': {dep}"


def test_jax_pins_bound() -> None:
  """Assert every jax/jaxlib declaration stays inside >=0.11.2,<0.12."""
  data = load_pyproject()
  all_dep_entries: list[tuple[str, str]] = [
    ("runtime", dep) for dep in data.get("project", {}).get("dependencies", [])
  ]
  for extra_name, deps in data.get("project", {}).get("optional-dependencies", {}).items():
    all_dep_entries.extend((f"extra:{extra_name}", dep) for dep in deps)
  for group_name, deps in data.get("dependency-groups", {}).items():
    all_dep_entries.extend((f"group:{group_name}", dep) for dep in deps if isinstance(dep, str))

  runtime_bases = {
    dep.split(";")[0].split(">=")[0].split("==")[0].split("<")[0].strip()
    for dep in data.get("project", {}).get("dependencies", [])
  }
  assert {"jax", "jaxlib"} <= runtime_bases

  for location, dep_str in all_dep_entries:
    base_pkg = dep_str.split(";")[0].split(">=")[0].split("==")[0].split("<")[0].strip()
    if base_pkg in {"jax", "jaxlib", "jax[cpu]", "jax[cuda12]"}:
      assert ">=0.11.2" in dep_str, f"missing >=0.11.2 pin at {location}: {dep_str}"
      assert "<0.12" in dep_str, f"missing <0.12 cap at {location}: {dep_str}"


def test_cpu_cuda12_conflict_and_xtrax_pin() -> None:
  """Assert uv declares cpu vs cuda12 conflict and xtrax remains pinned at 0.4.1b9."""
  data = load_pyproject()
  conflicts = data.get("tool", {}).get("uv", {}).get("conflicts", [])
  expected_pair = [{"extra": "cpu"}, {"extra": "cuda12"}]
  assert any(
    sorted(pair, key=lambda d: d.get("extra", "")) == expected_pair for pair in conflicts
  ), f"Missing cpu vs cuda12 conflict declaration in [tool.uv].conflicts: {conflicts}"

  runtime_deps = data.get("project", {}).get("dependencies", [])
  assert "xtrax[io,pack,export]==0.4.1b9" in runtime_deps
