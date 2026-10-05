#!/bin/bash
# S16: ruff banned-api matches the literal module string; a ban key naming a module nobody imports (a dead key)
# never fires and ruff says nothing about it.
set -u
D="$(mktemp -d)"
cd "$D"
cat > ruff_cfg.toml <<'EOF'
[lint]
select = ["TID251"]
[lint.flake8-tidy-imports.banned-api]
"deadpkg.old" = {msg = "banned"}
EOF
printf 'import deadpkg.old\n' > imports_banned_literal.py
printf 'import livepkg.old\n' > imports_moved_module.py
printf 'import os\n' > clean.py
echo "--- ruff version"; ruff --version
echo "--- imports_banned_literal.py (positive control: must be flagged)"
ruff check --config ruff_cfg.toml --no-fix --no-cache --output-format concise imports_banned_literal.py; echo "exit=$?"
echo "--- imports_moved_module.py (the module was renamed or moved: flagged only if ruff follows moves)"
ruff check --config ruff_cfg.toml --no-fix --no-cache --output-format concise imports_moved_module.py; echo "exit=$?"
echo "--- clean.py (dead key present in config, no code imports it: any warning about the unmatched ban?)"
ruff check --config ruff_cfg.toml --no-fix --no-cache --output-format concise clean.py; echo "exit=$?"
