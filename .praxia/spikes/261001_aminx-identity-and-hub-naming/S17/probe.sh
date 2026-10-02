#!/bin/bash
# S17: residue of the previous rename (prxteinmpnn -> aminx) in the tracked tree of this worktree.
set -u
echo "tracked files containing prxteinmpnn (case-insens), all paths: $(git grep -l -i prxteinmpnn | wc -l)"
EXCL=(':(exclude).praxia' ':(exclude)outputs' ':(exclude).claude')
echo "  outside dated history (.praxia/, outputs/, .claude/):     $(git grep -l -i prxteinmpnn -- . "${EXCL[@]}" | wc -l)"
echo "--- those paths, grouped by first two components"
git grep -l -i prxteinmpnn -- . "${EXCL[@]}" | cut -d/ -f1-2 | sort | uniq -c | sort -rn
echo "--- src/ occurrences"
git grep -n -i prxteinmpnn -- src | cut -c1-160
echo "--- scripts/check_model_boundary.sh"
echo "lines naming the dead package prxteinmpnn: $(grep -c prxteinmpnn scripts/check_model_boundary.sh)"
echo "lines naming the live package aminx:       $(grep -c aminx scripts/check_model_boundary.sh)"
echo "src/aminx/model/_inference exists: $(test -e src/aminx/model/_inference && echo yes || echo no)"
