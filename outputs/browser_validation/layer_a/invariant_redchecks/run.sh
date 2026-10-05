#!/bin/bash
cd /home/marielle/projects/aminx/.claude/worktrees/browser-validation
export REFERENCE_PATH=/home/marielle/projects/aminx/reference_ligandmpnn_clone PROTEINMPNN_PATH=$PWD/.cache/reference/ProteinMPNN JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for n in 1 2 3 4 5 6 7 8; do
  f=$(python3 -c "exec(open('$TMPDIR/red/muts.py').read().split('import sys')[0]); print(MUTS[$n][0])")
  python3 $TMPDIR/red/muts.py $n || { echo "RED$n: MUTATION FAILED"; continue; }
  git diff --stat -- "$f"
  uv run --frozen --extra dev --extra benchmark pytest tests/parity/test_mpnn_reference_invariants.py -m parity_heavy -q -p no:cacheprovider -k "invariant_${n}_" -x > $TMPDIR/red/red$n.log 2>&1
  echo "RED$n: pytest exit=$? $(grep -E '^E +(AssertionError|assert)|AssertionError:' $TMPDIR/red/red$n.log | head -1)"
  grep -E "passed|failed" $TMPDIR/red/red$n.log | tail -1
  git checkout -- "$f"; git diff --quiet -- src && echo "RED$n: src restored"
done
