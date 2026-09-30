# aminx CLAUDE.md

## Commands

| Action | Command |
| :--- | :--- |
| **Type Check** | `uv run ty check` |
| **Lint** | `uv run ruff check .` |
| **Format** | `uv run ruff format .` |
| **Tests** | `uv run pytest` |
| **JAX advisory** | `uv run jaxlint check src --no-doc` (optional; not a CI gate) |

## Tech Stack

- **Language**: Python 3.12+
- **ML Framework**: JAX + Equinox
- **Package Manager**: uv
- **Type Checking**: ty (advisory — CI does not run it, and it currently reports ~1.1k diagnostics; do not assume a clean `ty check` baseline)
- **Linting**: ruff
- **Testing**: pytest

## Code Style

- Annotate new code fully and check it with `ty` (advisory, see above); format with `ruff`
- JAX: use `jax.jit`, `jax.vmap`, `jax.lax.scan` patterns
- Equinox: define models as `eqx.Module` subclasses (not dataclasses), use `eqx.filter_jit` for PyTree-aware JIT compilation
- Numerical tolerance tests with `pytest.approx`, cross-framework validation

## Decisions & Architecture

Design decisions live in `.praxia/docs/decisions/` — see `.praxia/docs/INDEX.md` for the full ADR index.

## Cluster & Experiment Tracking

- **Cluster jobs**: Submit via `myxcel` (transparent SSH wrapper to engaging cluster) — see `/using-myxcel` skill for command reference
- **Experiment tracking**: Use `bathos` for reproducible runs — all measurement logic with sidecars, pre-registered hypotheses — see `/using-bathos` skill
