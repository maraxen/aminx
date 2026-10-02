#!/bin/bash
# task_id 260929_potts-laser-xtrax-compose
#
# THE LEDGER WAVE. Run ONLY after the last commit under a scoped path
# (src/aminx/, scripts/parity/, scripts/recapture/, tests/port/,
# aminx-oracles/, pyproject.toml, uv.lock). _touches_scoped is GLOBAL, not
# per-slug, so any later scoped commit invalidates every ledger row at once.
# tests/knob_gate/ is NOT scoped, so ledger ids may be written afterwards.
#
# Usage:  launch_wave.sh 1 | 2 | positive
#
# The --mutants string for each slug is CHECKED against that slug's rows in
# tests/knob_gate/branch_manifest.toml before anything launches, because step
# 1c compares the raw argv string to the manifest row set by exact equality
# (_coverage.py:309) and a mismatch throws the run out hours later for a
# reason that has nothing to do with what it measured.
#
# scalar_both_off is the POSITIVE control and appears in no graded string: it
# has no manifest row (every row must report "failed", and it must PASS), and
# _argv_mutants reads the raw string, so appending it voids an otherwise
# perfect run. `launch_wave.sh positive` runs it alone, with no
# --controls-out, so it cannot be mistaken for a ledger run.
set -euo pipefail

REPO=${AMINX_WAVE_REPO:-$HOME/projects/aminx-sprint-git}
cd "$REPO"

# .praxia/audits.jsonl is TRACKED and anything running pytest appends to it.
# A dirty tree makes every run ineligible, so restore before launching.
git checkout -- . 2>/dev/null || true
git fetch -q https://github.com/maraxen/aminx.git wt/260929-potts-laser-main
git checkout -q FETCH_HEAD
H=$(git rev-parse --short=8 HEAD)
DIRTY=$(git status --porcelain | wc -l)
echo "HEAD=$H  porcelain=$DIRTY"
if [ "$DIRTY" -ne 0 ]; then
  echo "REFUSING: tree is dirty, every run would record git_dirty=true" >&2
  git status --porcelain >&2
  exit 1
fi

GROUP=${1:-}
[ -n "$GROUP" ] || { echo "usage: $0 [1|2|positive]" >&2; exit 2; }

# slug -> the exact mutant string for its graded run.
declare -A MUT=(
  [laser_proofread_parity]="reduction_swap,ddof_0,scalar_off,vector_on"
  [laser_decode_e2e]="reversed_order,chi_bin_plus_one,offset_zero,chi2_equals_chi1,lambda_on_logits"
  [potts_ar_refine_exact]="ar_mask_present_chain_m_pos,wrong_partition_sign,ntoc_refine_order"
  [potts_ar_decode]="ar_mask_present_chain_m_pos"
  [potts_energy_parity]="permute_etab_out_rows"
)

# Fail before burning hours, not after. Compares every string above to the
# manifest, so an edit to either side that desynchronises them is caught here.
check_mutants () {
  python3 - "$@" <<'PY'
import sys, tomllib, collections, pathlib
pairs = sys.argv[1:]
manifest = pathlib.Path("tests/knob_gate/branch_manifest.toml")
rows = tomllib.loads(manifest.read_text())["branch"]
by = collections.defaultdict(set)
for row in rows:
    vehicle = row.get("vehicle", {})
    if vehicle.get("kind") == "sidecar":
        by[vehicle["slug"]].add(row["id"])
bad = False
for pair in pairs:
    slug, _, raw = pair.partition("=")
    listed = {p for p in raw.split(",") if p}
    expected = by.get(slug, set())
    if listed != expected:
        bad = True
        print(f"MISMATCH {slug}", file=sys.stderr)
        print(f"  --mutants {sorted(listed)}", file=sys.stderr)
        print(f"  manifest  {sorted(expected)}", file=sys.stderr)
        print(f"  argv-only {sorted(listed - expected)}", file=sys.stderr)
        print(f"  rows-only {sorted(expected - listed)}", file=sys.stderr)
    else:
        print(f"ok {slug}: {len(listed)} mutants match the manifest")
if bad:
    print("REFUSING: step 1c would discard these runs (_coverage.py:309)", file=sys.stderr)
    sys.exit(1)
PY
}

export POTTS_ORACLE_PYTHON=$HOME/projects/aminx-oracles/.venv/bin/python3
export LASER_ORACLE_PYTHON=$HOME/projects/aminx-oracles/.venv/bin/python3
export AMINX_POTTS_ROOT=$HOME/repos/PottsMPNN

launch () {
  local v=$1 m=${MUT[$1]}
  mkdir -p "$HOME/.aminx/sidecars/$v/$H"
  nohup setsid env JAX_PLATFORMS=cpu bth run --project-slug aminx \
    --output-paths "$HOME/.aminx/sidecars/$v/$H/branch_controls.json" \
    -- uv run --no-sync python3 "scripts/parity/$v.py" \
       --mutants "$m" \
       --work-dir "/tmp/${v}_$H" \
       --controls-out "$HOME/.aminx/sidecars/$v/$H/branch_controls.json" \
    > "$HOME/${v}_$H.log" 2>&1 < /dev/null &
  echo "launched $v  (--mutants $m)"
  sleep 3
}

# Two groups, not five at once: five concurrent arms put load at 32-37 on 20
# cores and slowed every one of them. Run group 2 after group 1 has exited.
case "$GROUP" in
  1)
    SLUGS=(laser_proofread_parity laser_decode_e2e) ;;
  2)
    SLUGS=(potts_ar_refine_exact potts_ar_decode potts_energy_parity) ;;
  positive)
    nohup setsid env JAX_PLATFORMS=cpu uv run --no-sync python3 \
      scripts/parity/laser_proofread_parity.py \
      --mutants scalar_both_off --work-dir "/tmp/positive_$H" \
      > "$HOME/positive_$H.log" 2>&1 < /dev/null &
    echo "launched positive control, ungraded, no --controls-out"
    echo "it must report near-zero; a non-zero result is instrument floor"
    exit 0 ;;
  *)
    echo "usage: $0 [1|2|positive]" >&2; exit 2 ;;
esac

ARGS=()
for s in "${SLUGS[@]}"; do ARGS+=("$s=${MUT[$s]}"); done
check_mutants "${ARGS[@]}"

for s in "${SLUGS[@]}"; do launch "$s"; done

sleep 30
ps -eo pid,etime,args | grep -E "[p]otts_|[l]aser_" | grep "uv run" | cut -c1-60
