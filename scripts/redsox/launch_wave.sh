#!/bin/bash
# task_id 260929_potts-laser-xtrax-compose
#
# THE LEDGER WAVE. Run ONLY after the last commit under a scoped path
# (src/aminx/, scripts/parity/, scripts/recapture/, tests/port/,
# aminx-oracles/, pyproject.toml, uv.lock). _touches_scoped is GLOBAL, not
# per-slug, so any later scoped commit invalidates every ledger row at once.
# tests/knob_gate/ is NOT scoped, so ledger ids may be written afterwards.
#
# Usage:  launch_wave.sh 1 | 2 | 3 | positive
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

# .praxia/audits.jsonl is TRACKED and the PORT TIER appends to it, so a dirty
# tree makes every later run ineligible (git_dirty=true). Restore first.
#
# Measured 261002, and I got this wrong in both directions before measuring it
# properly, so the scope is written out: running `tests/knob_gate` and
# `tests/lint` leaves the file byte-identical (same sha256, same 206 lines),
# which is what misled me into deleting this comment as false. Running the PORT
# tier appends hundreds of records -- a gate run plus two `tests/port/` runs
# added 259 lines, each a `{"domain": "port", "rule_id": "parity_tier_1", ...}`
# finding emitted by the port harness itself.
#
# So: "anything running pytest" is too strong, "pytest never touches it" is
# false, and the rule that matters here is that ANY run of tests/port/ dirties
# a tracked file. That is exactly what this script runs next to.
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
[ -n "$GROUP" ] || { echo "usage: $0 [1|2|3|positive]" >&2; exit 2; }

# slug -> the exact mutant string for its graded run.
declare -A MUT=(
  [laser_proofread_parity]="reduction_swap,ddof_0,scalar_off,vector_on"
  [laser_decode_e2e]="reversed_order,chi_bin_plus_one,offset_zero,chi2_equals_chi1,lambda_on_logits"
  [potts_ar_refine_exact]="ar_mask_present_chain_m_pos,wrong_partition_sign,ntoc_refine_order"
  [potts_ar_decode]="ar_mask_present_chain_m_pos"
  [potts_energy_parity]="permute_etab_out_rows"
  [laser_score_parity]="permute_decoder_layer"
  [potts_ddg_megascale]="skip_transpose_merge_pair"
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
    SLUGS=(potts_ar_refine_exact potts_ar_decode potts_energy_parity laser_score_parity) ;;
  3)
    # Alone: the largest vehicle in the sprint, 202804 rows over 371 per-PDB
    # resumable units per arm. It took ~25 min at e3bc540e, so the isolation is
    # about CPU contention with the other groups, not about duration.
    #
    # It is also the vehicle that proves the exports at :100-102 are load-
    # bearing rather than belt-and-braces. Launched by hand with an otherwise
    # identical `bth run`, it died in under a minute: its oracle arm imports
    # upstream run_utils, which imports seaborn at module scope, and the sprint
    # venv has none -- _oracle_python falls back to sys.executable when
    # POTTS_ORACLE_PYTHON is unset. Through this script it just works.
    SLUGS=(potts_ddg_megascale) ;;
  positive)
    # THIS ARM COULD NEVER HAVE RUN. It called the parity script directly,
    # while every graded arm goes through `bth run` (launch(), :95) -- and
    # `$BTH_RESULTS_PATH` is set by bth, not by the script. Launched as it was,
    # it died in under a second with
    #     laser_proofread_parity requires $BTH_RESULTS_PATH
    # and wrote no payload at all. Measured 261002 on b5272b07.
    #
    # So it goes through `bth run` too. It stays ungraded by what it OMITS:
    # no --controls-out, so it writes no branch_controls.json, and no
    # --output-paths into $HOME/.aminx/sidecars/<slug>/<sha>/, which is the
    # directory the ledger and step 1c read. Its work-dir is /tmp/positive_$H,
    # well clear of the graded /tmp/<slug>_$H.
    #
    # EXPECT THE RECORD'S OUTCOME TO READ `fail`, and do not "fix" it. The
    # sidecar's criteria are written for a graded run, where every mutant must
    # be killed; scalar_both_off is a no-op that must SURVIVE. An outcome of
    # `fail` here is therefore the positive control succeeding, and an outcome
    # of `pass` would mean a no-op mutation changed the result. Read the
    # payloads under the work-dir, not the outcome column.
    nohup setsid env JAX_PLATFORMS=cpu bth run --project-slug aminx \
      -- uv run --no-sync python3 \
      scripts/parity/laser_proofread_parity.py \
      --mutants scalar_both_off --work-dir "/tmp/positive_$H" \
      > "$HOME/positive_$H.log" 2>&1 < /dev/null &
    echo "launched positive control, ungraded, no --controls-out"
    echo "it must report near-zero; a non-zero result is instrument floor"
    echo "expect the bathos outcome to read 'fail' -- see the comment above"
    exit 0 ;;
  *)
    echo "usage: $0 [1|2|3|positive]" >&2; exit 2 ;;
esac

# check_mutants only inspects the slugs handed to it, so it could not see a slug
# missing from MUT or from every group. laser_score_parity was exactly that: it
# has a manifest row (permute_decoder_layer) and appeared in no group, so no
# invocation of this script would ever have run it and the manifest could never
# have been satisfied. Checked here against the manifest itself.
check_coverage () {
  python3 - "$@" <<'PY'
import pathlib
import sys
import tomllib

known = set(sys.argv[1:])
rows = tomllib.loads(
    pathlib.Path("tests/knob_gate/branch_manifest.toml").read_text()
)["branch"]
sidecar = [r for r in rows if r.get("vehicle", {}).get("kind") == "sidecar"]
slugs = {r["vehicle"]["slug"] for r in sidecar}
missing = sorted(slugs - known)
extra = sorted(known - slugs)
if missing:
    print(
        f"REFUSING: {len(missing)} manifest sidecar slug(s) are in no launch "
        f"group, so the manifest could never be satisfied: {missing}",
        file=sys.stderr,
    )
if extra:
    print(
        f"REFUSING: {len(extra)} launched slug(s) have no manifest rows: {extra}",
        file=sys.stderr,
    )
if missing or extra:
    sys.exit(1)
print(f"ok coverage: {len(slugs)} manifest slugs all reachable from a group")
PY
}

# Every slug any group can launch, so the check is about the SCRIPT rather than
# about whichever group happens to be running.
check_coverage laser_proofread_parity laser_decode_e2e \
               potts_ar_refine_exact potts_ar_decode potts_energy_parity \
               laser_score_parity potts_ddg_megascale

ARGS=()
for s in "${SLUGS[@]}"; do ARGS+=("$s=${MUT[$s]}"); done
check_mutants "${ARGS[@]}"

for s in "${SLUGS[@]}"; do launch "$s"; done

sleep 30
ps -eo pid,etime,args | grep -E "[p]otts_|[l]aser_" | grep "uv run" | cut -c1-60
