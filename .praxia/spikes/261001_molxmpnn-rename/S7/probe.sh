#!/bin/bash
# S7: ruff banned-api matches the literal module string; a ban on oldx.bad does not follow a rename to newx.bad.
set -u
D="$(mktemp -d)"
cd "$D"
cat > pyproject.toml <<'EOF'
[tool.ruff.lint]
select = ["TID251"]
[tool.ruff.lint.flake8-tidy-imports.banned-api]
"oldx.bad" = {msg = "banned"}
EOF
cat > before_rename.py <<'EOF'
import oldx.bad
EOF
cat > after_rename.py <<'EOF'
import newx.bad
EOF
echo "--- ruff version"; ruff --version
echo "--- before_rename.py (positive control: must be flagged)"
ruff check --isolated --config pyproject.toml --no-fix --output-format concise before_rename.py; echo "exit=$?"
echo "--- after_rename.py (rename victim: flagged only if ruff follows renames)"
ruff check --isolated --config pyproject.toml --no-fix --output-format concise after_rename.py; echo "exit=$?"
