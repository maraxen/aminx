---
title: 'A1 ready-to-fire: the laser_score option-3 dispatch envelope, drafted not dispatched'
description: Exact scope, anchors, measured band and acceptance criteria for the only gate-clearing item, so the decision becomes one dispatch
status: proposed
task_id: 260929_potts-laser-xtrax-compose
date: '261006'
---
# A1 ready-to-fire: the laser_score option-3 dispatch envelope

**Drafted, NOT dispatched.** Option 3 has not been chosen, and dispatching the implementation of
an undecided design choice would pre-empt the decision. This document exists so that choosing it
costs one dispatch rather than a work session.

A1 is the **only** gate-clearing item (see `plans/261005_rewave-composition-scoped-batch.md` §1):
all 63 of gate run `fb898d24`'s failures are `tests/port/test_laser_score.py` clean-arm
failures, and nothing else in step 1 is red.

## Option 3's two preconditions are now BOTH satisfied

`decisions/261003_laser-score-tier3-f32-design.md` conditions any tolerance move on two things:

1. **"Its band must come from its own pre-registered measurement."** Satisfied. Run `3af3be02`
   (`scripts/analysis/laser_score_scale_relative_floor.py` @ `0e642d44`) reads
   `completed / pass / exit 0 / git_dirty=False`, and its artifact gives `proposed_tol = 4e-4`
   over `floor_scale_relative = 3.6297e-5` — `headroom = 10.0`, `control_rejected = true`,
   `loo_all_admitted = true` (`loo_worst_ratio = 0.121`), `n_pairs = 63`. Verified by record
   261005. **Note the margin is 10x, not the 100x quoted in earlier notes.**
2. **"The assertion changes form, not just value."** Not yet done — that is the work below.

**The target file names this precondition itself.** `tests/port/targets/laser_score.toml`'s
`f32_atol_basis` currently ends *"Do not move it without a new measurement that clears the
margin."* That measurement now exists, so the file's own condition is met — but the basis text
must be rewritten to cite `3af3be02` rather than left asserting the old prohibition.

## Scope: three files, two scoped

| file | change | scope |
| :-- | :-- | :-- |
| `tests/port/test_laser_score.py` | assertion **form** → scale-relative | **scoped** |
| `tests/port/targets/laser_score.toml` | policy string + `f32_atol_basis` rewrite | **scoped** |
| `tests/lint/test_port_tolerances_match_targets.py` | parse/compare the new form | unscoped |

Because two are scoped, **this invalidates all eight ledger rows** and must ride the single
re-wave (§0 of the plan). Sequencing: **after** the `origin/main` merge, or the rows are
re-measured twice (§0a).

## Anchors the implementer needs

- `test_laser_score.py:270-282` — `_rows_over(got, ref, rtol, atol)` computes
  `limit = atol + rtol * |ref|` **element-wise**. Option 3 is
  `max|got - ref| <= tol * max|ref|`, **per field per pair** — a reduction, not an
  element-wise envelope. This is the change of *form*.
- `test_laser_score.py:305-306` — `rtol = _RTOL[precision]`, `atol = _ATOL[precision]`; the f32
  path must select the scale-relative `tol` instead.
- `test_laser_score.py:349-353` — the failure message quotes
  `atol={atol:g}+rtol={rtol:g}`; it must report the scale-relative rule and the observed
  `max|got-ref| / max|ref|`, or a failure will be unreadable.
- `test_laser_score.py:421-424` — `test_tier_3_f32`'s docstring says "match at rtol=1e-4,
  atol=1e-7" and must be corrected, as must the module docstring at `:9`.
- `targets/laser_score.toml` — `[parity] tolerance_policy_f32 = "rtol=1e-4,atol=1e-7"` →
  the scale-relative policy string. The lint test is what keeps this and the assertion from
  drifting, which is why it must learn the new form **in the same change**.

## Acceptance criteria

1. `tests/port/test_laser_score.py` tier 3 passes on all 63 pairs under the scale-relative rule.
2. **The f64 tier is untouched** and still passes `rtol=1e-8, atol=1e-11`. Per the decision doc,
   "no option here changes the f64 tier" — it remains the actual correctness check. f64 is for
   parity assertions only and is not a production dtype.
3. `tests/lint/test_port_tolerances_match_targets.py` passes, and **fails** if the TOML policy
   and the assertion disagree. Add a negative check if the existing one cannot express the new
   form — a lint that cannot fail is the defect this very sprint fixed twice.
4. An injected **1e-2 relative defect is still rejected** by the new band. This is the whole
   point of the 10x margin; without the control the band is unjustified. The band sits ~33x
   below a 1e-2 defect per the decision doc's analysis.
5. `tests/knob_gate` still 35 passed / 2 skipped (the two skips are structural — they need
   `AMINX_REDSOX_OUTCOMES_READ`, exported only by `run_gate.py`).

## Known inconsistency to leave alone

`laser_decode_step` stays on the element-wise rule (it passed it, `2a4442d1`). Converting every
LASEr f32 tier to scale-relative would be more uniform and is explicitly **larger than this
wave needs** — the decision doc says so. Do not expand scope.

## Dispatch notes

Per standing practice: route to a Cursor fixer via `praxia dispatch run --envelope <path>`,
unsandboxed and in the background; `usd_cap: null`, `max_turns: null`, a full 40-char
`worktree_base_sha`, and a clean tree. Bump `attempt` on retry. Opus validates. Include
`task_id 260929_potts-laser-xtrax-compose`. The `worktree_base_sha` is deliberately left unset
here because it must be the tree at dispatch time — after the main merge, per §0a.

**Environment:** the implementer needs a venv with **both** xtrax 0.4.0a11 and pyarrow.
`aminx-b7i-git` has both; `aminx-confirm-git` fails on the xtrax floor and lacks the `laser`
extra; `aminx-ci` has no pyarrow. And `uv`/`bth` are not on PATH over a plain ssh — prefix
`PATH=/home/solab/.local/bin:$PATH` or the wrapped command fails with exit 1 and no traceback.
