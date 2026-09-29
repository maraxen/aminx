#!/usr/bin/env bash
# One pytest session per A1 port wave. The orchestrator runs this on titanix.
# ``-o addopts=""`` keeps the parity_heavy marker from hiding the waves.
set -euo pipefail
cd "$(dirname "$0")/../.."
waves=(
  potts_head
  potts_merge_pair_d2
  potts_merge_pair_d4
  potts_energy
  pottsmpnn_full
  declayer_f64
)
for wave in "${waves[@]}"; do
  echo "=== AMINX_PORT_WAVE=${wave} ==="
  AMINX_PORT_WAVE="${wave}" uv run --frozen --extra=dev pytest -o addopts="" "tests/port/test_${wave}.py"
done
