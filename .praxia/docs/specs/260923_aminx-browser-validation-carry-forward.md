---
title: aminx browser validation — binding carry-forward notes (final adversarial round)
description: 'Mechanical, fail-closed amendments from the final adversarial round that ride in the Phase-1 fixer prompts, plus upstream bathos bugs to file'
status: approved
task_id: 260923_aminx-browser-validation
date: '260924'
---
# Carry-forward notes — aminx browser-validation spec (converged, final round ACCEPT/high)

Binding amendments from the final adversarial round (rounds/r4_*.json). They are small mechanical fixes that fail
closed, so they ride in the Phase-1 fixer prompts instead of forcing a fifth round (editing spec.md after
convergence would invalidate `verify-converged`). Each fixer prompt for the named task MUST include its notes.

- **All tasks that write a `bth run` line (T4, T6, T7, T8, T11) — F-C1 (token order):** keep the uv option tokens in
  exactly the order written and put nothing between `python` and the script path. bathos `_find_script_path`
  (runner.py:57-75 at 84be544e) reads tokens in pairs after `run`; `--no-sync` resolves only because it sits in a
  skipped slot. T4 additionally asserts the provenance-smoke catalog row has non-empty `script_sha256` and
  `sidecar_sha256`.
- **T4 gate — F-C2:** replace `(.titanix.memory_cap | IN("systemd-run", "projected"))` with
  `.titanix.memory_cap == "systemd-unit"` (step 5 already records "systemd-unit").
- **titanix_launch.sh (T4) — F-C3:** one-run-at-a-time check is
  `out=$(ssh titanix systemctl --user list-units 'bv-*' --state=active,activating --no-legend --plain)` under
  `set -e` (non-zero exit aborts the launch); refuse unless `[ -z "$out" ]`.
- **titanix_launch.sh / titanix_run.sh (T4) — F-C4:** `<session>` = the unit name `bv-$s-${H:0:12}`. The poll ends
  when `.exit` exists OR `ssh titanix systemctl --user is-active bv-$s-${H:0:12}` is not active/activating
  (inactive, failed, not-found are terminal — `--collect` may GC the unit). If `.exit` is absent the ledger records
  `exit = "killed"` plus the `journalctl --user -u <session>` tail. run.sh installs
  `trap 'echo $? > …/<session>.exit' EXIT` before its first guard.
- **Differential pre-flight rules text (T7/T8 sidecars + verdict doc) — F-C5:** state that bathos does not check the
  off/on arm exit codes (runner.py:219-227), so a crashed arm can pass the pre-flight; the in-run controls
  (`controls_detected` / `posctl_detected` / `p09_fusion_ctrl_detected`) are the binding sensitivity evidence.
- **T4 .gitignore — F-C6:** add `*.bth.lock.toml` (no-op if present) and the T4 fixer gate
  `git check-ignore -q scripts/browser_validation/x.abc.bth.lock.toml` next to the outputs/ check. Validate scripts
  write only to `--out`, never into `BTH_OUTPUT_DIR` (or also ignore `outputs/*/`).
- **ODQ-13 wording — F-C7:** read the decision as "tracked wrapper in a transient systemd user unit
  (systemd-run --user, MemoryMax=64G)" — the spec row still says "under tmux"; tmux is not installed on titanix.
- **T11 (optional hardening) — F-C8:** add `pyproject.toml` to FROZEN; the verdict doc's reproduction plan says
  "check out H_v".

## Upstream bathos bugs to file (Risk R20)
1. `bth submit` shells out to nonexistent `myxcel push-project` / `pull-project` (cluster.py:78, :151).
2. PyPI `bathos==0.13.0a4` predates the bth-run exec-as-given fix (#61) under the same version string.
3. bathos tests write pytest temp catalogs into the real `~/.bth/projects.toml` (seen on titanix).
4. The differential pre-flight never checks arm exit codes (runner.py:219-227), so a crash can yield `passed`.
