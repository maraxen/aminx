#!/bin/bash
# Env prep for the xtrax.profiling dogfood: install the locally-built xtrax
# wheel (shipped from the packaging-fix head) into the scratch venv.
#
# Runs on mit_quicktest: fast, and --no-deps + local wheel means NO network
# is required (per policy: no dependency syncs on login nodes).
# Chained: the L40S dogfood job depends on this one via afterok.

#SBATCH --job-name=prof-envprep
#SBATCH --partition=mit_quicktest
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=outputs/logs/slurm/%x_%A.out
#SBATCH --error=outputs/logs/slurm/%x_%A.err

set -euo pipefail

PROJ="${HOME}/projects/aminx"
cd "${PROJ}"
export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR=/orcd/scratch/orcd/011/maarxaru/.cache/uv
export UV_PROJECT_ENVIRONMENT=/orcd/scratch/orcd/011/maarxaru/venvs/aminx

WHL=$(ls -t vendor/xtrax-*.whl | head -1)
echo "[envprep] installing ${WHL}"
uv pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
    --no-deps --force-reinstall "${WHL}"

"${UV_PROJECT_ENVIRONMENT}/bin/python" -c "
import xtrax, xtrax.profiling, xtrax.run, xtrax.loop.promotion_gate
print('[envprep] xtrax', xtrax.__version__, '| profiling + run + loop import OK')
"
echo "[envprep] done"
