---
title: 'laser_score tier-3 f32 design: two f32 implementations cannot meet a pairwise defect margin'
description: 'Decision request: aminx and upstream make the same f32 error vs f64, so no pairwise f32 band can separate noise from a 1% defect; three options for the user'
status: proposed
task_id: 260929_potts-laser-xtrax-compose
date: '261003'
supersedes: ''
backlog_ids: ''
---
# laser_score tier-3 f32 design: two f32 implementations cannot meet a pairwise defect margin

**Status: needs the user's decision.** Nothing has been changed on the basis of
this document; `tests/port/targets/laser_score.toml` keeps its original f32 band
and the gate keeps `rc = 1` on this one wave until a choice is made.

## What is established (graded runs, read from their records)

| run | what it measured | outcome |
| :--- | :--- | :--- |
| `e47bfaab` @ `b880882e` | f32 floor of the pairwise test, element-wise band rule | `noise_reaches_defect_scale` for `laser_score` |
| `18b97f4f` @ `6a24c7a7` | each f32 implementation against an f64 truth | `aminx_no_worse_than_upstream` |

Both pre-registered before running. Truth proxy (aminx-f64) matches upstream-f64
to 6e-14 over all 63 pairs. Scale-relative maxima (error / that pair's `max|truth|`):

| field | aminx vs truth | upstream vs truth | aminx vs upstream |
| :--- | :--- | :--- | :--- |
| `ligand_scalars` | 2.88e-2 | 2.88e-2 | 1.3e-6 |
| `encoder_scalars` | 9.57e-4 | 9.57e-4 | 9.6e-6 |
| `sequence_logits` | 5.47e-3 | 5.47e-3 | 2.5e-5 |
| `chi_logits` | 3.67e-3 | 3.67e-3 | 2.8e-5 |
| `seq_log_prob` | 5.41e-3 | 5.41e-3 | 3.6e-5 |

**The port is correct.** f64 agrees to <8e-13 on every field. In f32 the two
implementations make *the same* error -- they agree with each other ~200x more
tightly than either agrees with f64. The shared error is largest at the earliest
measured field and shrinks downstream, so it is not accumulation through depth.

**Why the current test cannot pass.** It compares the two f32 implementations
element-wise with `atol + rtol*|ref|`. The worst violations sit on near-zero
elements, so the band rule makes `atol` bind and pushes it to 6e-3 absolute,
which is 0.61 of an injected 1e-2 relative defect -- inside the pre-registered
10x margin, so no element-wise band separates noise from a real 1% error.

## Options

**1. Accept tier 3 red for this wave.** Cheapest; no change. But the gate's
`rc = 1` then never clears, so the redsox gate can never report PASS. Only
sensible if the gate learns a per-wave waiver, which is itself a new mechanism.

**2. Grade aminx-f32 against the f64 reference.** Sounds most principled, and is
the WORST at catching defects: aminx's own f32 error against truth is 5e-3 on the
logits and 2.9e-2 on `ligand_scalars`, so any band that admits it is wider than
the 1e-2 defect it must reject. It would measure "is aminx f32 accurate" (it is
exactly as accurate as upstream), not "does aminx compute what upstream does".

**3. Keep the pairwise comparison, but judge it relative to each field's scale.**
`max|aminx - upstream| <= tol * max|ref|`, per field per pair -- the same reading
already applied to the decode padding test (`f97d99b6`). The pairwise difference
is tiny (<=3.6e-5 of scale); a 10x-headroom band of ~3e-4 of scale sits **~33x
below** a 1e-2 defect, clearing the pre-registered 10x margin with room to spare.
It keeps what makes the pairwise test valuable (both sides share their f32 error,
so it cancels) and drops only the element-wise treatment of near-zero entries
that made `atol` bind.

### Recommendation: option 3

With two conditions, both required before any tolerance moves:

- **Its band must come from its own pre-registered measurement**, not from the
  table above. The table is from an attribution run that was not designed to
  derive a band; reusing its numbers to set one would be choosing a criterion
  after seeing data. Concretely: a floor script for the scale-relative rule,
  sidecar committed first, with the same 10x-headroom derivation and injected
  1e-2 control as `laser_f32_tolerance_floor.py`.
- **The assertion in `tests/port/test_laser_score.py` changes form**, not just
  value (scale-relative instead of `assert_allclose`-style). The TOML policy
  string must say so, and `tests/lint/test_port_tolerances_match_targets.py`
  must learn the new form so the reported and asserted bands still cannot drift.

Note the inconsistency this introduces: `laser_decode_step` would stay on the
element-wise rule (it passed that rule, `2a4442d1`). Moving every LASEr f32 tier
to the scale-relative rule would be more uniform, but is a larger change than
this wave needs.

## Not in question

No option here changes the f64 tier, which passes at `rtol=1e-8, atol=1e-11` and
remains the actual correctness check for this port.

