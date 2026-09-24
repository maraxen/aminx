---
title: aminx browser validation — Phase-0/1 sprint ready to execute
description: 'Pickup for a new session in this worktree: approved spec + plan, gates passed, titanix/bathos facts, how to start EXECUTE'
status: draft
task_id: 260923_aminx-browser-validation
date: '260924'
---
# aminx browser validation — Phase-0/1 sprint ready to execute

## Where

- **Worktree:** `/home/marielle/projects/aminx/.claude/worktrees/browser-validation`, branch
  `feat/browser-validation-spec`, from aminx `origin/main` `a5ced9c7`. It is local only; nothing has been pushed.
- **Spec (approved):** `.praxia/docs/specs/260923_aminx-browser-validation.md`, commit `b49a2611`.
  - 27 ACs across Phases 0-4. Phase 0/1 has executable fixer tasks T1-T11.
  - It converged after 4 adversarial rounds (final ACCEPT/high).
- **Binding carry-forward notes:** `.praxia/docs/specs/260923_aminx-browser-validation-carry-forward.md`. These are
  final-round mechanical fixes. They override conflicting spec text and are already merged into the sprint TOML's
  fixer prompts.
- **Sprint plan (approved):** `.praxia/sprint_plans/260924_aminx-browser-validation.toml`, commit `985ab0c6`. It has 11
  sequential tracks. Each fixer prompt carries the common context, the verbatim steps, and its notes. Each reviewer
  prompt carries every Gate block (fixer and orchestrator) plus the task's ACs.
- **Design records (outside the repo):** `~/.praxia/design/260923_aminx-browser-validation/`, containing recon,
  research, `rounds/r{1..4}_*.json` and the `aminx_origin_main/` snapshot.
- **Adversarial ledger:** `~/.praxia/loop_metrics/260923_aminx-browser-validation/sprint_ledger.jsonl`, sprint id
  `260924_aminx-browser-validation`; the last round decision is `converged`. The metrics script is
  `~/.praxia/loop_metrics/260807_autonomous-dev-loop/adversarial_metrics.py`. It is a copy of praxia's; aminx has none
  in-repo.

## Approvals (user, 2026-09-24)

- **Gate 1 (spec_confirmed):** "I approve this spec". Filed in the aminx repo as recommended.
- **Gate 2 (sprint_approved):** "I approve this sprint plan". Execution happens from a new session started in this
  worktree.
- **Decisions recorded in the spec:**
  - ODQ-1: jax2onnx → ONNX Runtime Web, wasm EP first.
  - ODQ-2: CA-only out of scope.
  - ODQ-13: Phase-1 runs execute on **titanix**.

## Titanix and bathos facts (measured 2026-09-24)

- **Machine:** titanix, user `solab`, 20 cores, 123 GiB RAM. `/` is 92% full, about 148 GB free.
- **bathos:** installed as a uv tool pinned to git `84be544e`. Use it by absolute path: `/home/solab/.local/bin/bth`
  and `/home/solab/.local/bin/uv`. Do NOT use PyPI 0.13.0a4: it has the same version string but predates the
  bth-run exec fix.
- **Real git clones give native provenance.** Measured: `git_hash` equal to origin/main, `git_dirty=False`, and the
  lock sha were all recorded.
- **Tools on titanix:**
  - `tmux` is not installed. Detached runs use a transient `systemd-run --user --unit=… -p MemoryMax=64G --collect`
    unit; linger is enabled.
  - The LigandMPNN weight host is reachable.
- **Scratch from earlier smoke tests:** `~/bv/aminx-smoke` and `~/bv/catalog-smoke` are safe to delete.

## Next (new session in this worktree)

1. Start `claude` in this worktree, load the orchestration skill, and run
   `/loop /praxia:autonomous-praxia-dev-loop /praxia:orchestration`.
2. Both gates are already approved. Proceed with EXECUTE on the sprint TOML above, dispatching tracks T1→T11 in
   order.
3. Re-verify every fixer claim yourself, including red-on-prior and mutation checks where a task pins behaviour.
4. The orchestrator-only steps are:
   - the titanix launches (O6′ steps 1-5);
   - T10's literature-parity protocol;
   - T11's orchestrator-written invariant tests (the re-derivation lock).

   Fixers prepare packets for these steps; they do not run them.
5. Do not push `origin`. The only push is to the `titanix-bv` bare repo.

## Upstream bathos bugs to file (Risk R20)

1. `bth submit` calls the nonexistent `myxcel push-project` / `pull-project` (`cluster.py:78,151`).
2. PyPI 0.13.0a4 predates fix #61 under the same version string.
3. bathos tests write pytest temp catalogs into the real `~/.bth/projects.toml` (seen on titanix).
4. The differential pre-flight never checks the off/on arm exit codes (`runner.py:219-227`).
