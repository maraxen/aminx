#!/bin/bash
# S9: read the consumer-side declarations that live outside this workspace (the ledger's read: evidence may not leave the workspace).
set -u
P="$HOME/projects"
echo "=== declared dependencies on aminx"
grep -n '"aminx' "$P/mpnn_ext/pyproject.toml" "$P/asr/pyproject.toml" "$P/tev_design/pyproject.toml"
echo "=== git submodule declarations"
for r in asr mpnn_ext hautespout; do
  echo "--- $r"
  grep -n -B2 'aminx' "$P/$r/.gitmodules"
done
echo "=== find_spec guards on package names"
grep -n 'find_spec' "$P/asr/vendor/proteinsmc/src/proteinsmc/scoring/mpnn.py" "$P/proteinsmc/src/proteinsmc/scoring/mpnn.py"
echo "=== proteinsmc's own note on the dead name"
sed -n 20,25p "$P/proteinsmc/pyproject.toml"
echo "=== Claude auto-memory directory keyed by checkout path"
ls -d "$HOME/.claude/projects/-home-marielle-projects-aminx" "$HOME/.claude/projects/-home-marielle-projects-aminx/memory/MEMORY.md"
