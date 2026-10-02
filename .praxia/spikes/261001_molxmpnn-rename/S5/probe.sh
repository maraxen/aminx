#!/bin/bash
# S5: rename blast radius and stale-name residue, measured on the tracked tree of this worktree.
set -u
echo "tracked files containing word 'aminx' (case-insens):  $(git grep -l -w -i aminx | wc -l)"
echo "  of which under src/:      $(git grep -l -w -i aminx -- src | wc -l)"
echo "  of which under tests/:    $(git grep -l -w -i aminx -- tests | wc -l)"
echo "  of which under scripts/:  $(git grep -l -w -i aminx -- scripts | wc -l)"
echo "  of which under docs/:     $(git grep -l -w -i aminx -- docs | wc -l)"
echo "--- top-level breakdown (first two path components)"
git grep -l -w -i aminx | cut -d/ -f1-2 | sort | uniq -c | sort -rn | head -12
echo "--- importers of aminx.ebm outside src/aminx/ebm (expect 0)"
git grep -n -e 'aminx\.ebm' -e 'aminx import ebm' -- src ':(exclude)src/aminx/ebm' | wc -l
echo "--- residue of the PREVIOUS rename (prxteinmpnn)"
echo "tracked files containing prxteinmpnn: $(git grep -l -i prxteinmpnn | wc -l)"
echo "check_model_boundary.sh lines naming the dead package: $(grep -c prxteinmpnn scripts/check_model_boundary.sh)"
echo "check_model_boundary.sh lines naming the live package: $(grep -c aminx scripts/check_model_boundary.sh)"
