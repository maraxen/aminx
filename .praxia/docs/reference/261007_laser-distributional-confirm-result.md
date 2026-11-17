---
title: 'LASEr distributional confirm: both cells pass, and the control fired on both'
description: The sharded LASErMPNN confirmatory run completed on titanix GPUs 2 and 3 and graded pass by record on min_p0@1.0 and min_p0@0.3, with the CTRL_m negative control failing on sequence in both shards
status: proposed
task_id: 260929_potts-laser-xtrax-compose
date: '261007'
---
# LASEr distributional confirm: both cells pass, and the control fired on both

Spec B5/B7. The confirmatory run that `261005_laser-confirm-run-readiness.md` cleared for
launch has finished. Both shards graded **pass** by record. This note records the numbers and,
more importantly, what the run does and does not license.

## Graded by record, not by exit code

Both shards are `status: completed`, `outcome: pass`, `exit_code: 0` against
`scripts/parity/laser_sample_dist_confirm.bth.toml`, whose `[outcomes]` table registers
`pass` / `fail` / `instrument_invalid` / `inconclusive` / `smoke`.

| shard | run id | cell | duration | `script_sha256` | `git_hash` |
| :-- | :-- | :-- | --: | :-- | :-- |
| gpu2 | `2a50b92f-fdb7-4077-a53d-bade6075ec95` | `min_p0@1.0` | 17.51 h | `b28d94b2ec5c…` | `2eb50edb…` |
| gpu3 | `35584fbc-83ae-4072-afa2-1df52d8c4d2a` | `min_p0@0.3` | 17.53 h | `b28d94b2ec5c…` | `2eb50edb…` |

Each shard computed **20 units** (5 structures × 4 arms), `n_reused: 0` — so nothing was
inherited from an earlier attempt and the resume cache cannot have masked a stale arm. The
`script_sha256` in each shard's `result.json` equals the one the catalog recorded, so the
driver was not edited mid-flight; that is the exact failure the readiness note warned would
discard the run.

**The warm index lied about both runs and still does.** `bth sql` reported
`status: running`, `outcome: null`, `exit_code: -1` for both, and `bth compact` returned
`{"ingested": 0, "skipped": 263}` rather than correcting them. The cool-tier parquets
(`~/.bth/catalog/runs/aminx/run_<id>.parquet`, rewritten 10:10 and 10:11) carry the real
record. Anyone re-checking these two runs must read the parquet directly; the warm row is
wrong, not merely late.

## The measured result

Equivalence bands are pre-registered: `delta = 0.01` on sequence TV, `delta_chi = 0.05` on
χ1 TV. Both are two-one-sided-test style bands — the arm passes when the interval sits inside
the band, so the signed interval is what matters, not the point estimate alone.

### `min_p0@1.0` (gpu2)

| arm | observable | mean Δ | 90 % lower | 95 % upper | max Δ | verdict |
| :-- | :-- | --: | --: | --: | --: | :-- |
| A (aminx) | sequence | −0.000210 | −0.001283 | +0.001047 | 0.001137 | **pass** |
| A (aminx) | χ1 | −0.001000 | −0.002716 | +0.001330 | 0.002304 | **pass** |
| CTRL_m (m = 1.1) | sequence | +0.013630 | +0.010196 | +0.012744 | 0.015377 | **fail** |
| CTRL_m (m = 1.1) | χ1 | +0.012009 | +0.008930 | +0.013290 | 0.016638 | pass |

`h_hat = 0.0017875`, n = 1000.

### `min_p0@0.3` (gpu3)

| arm | observable | mean Δ | 90 % lower | 95 % upper | max Δ | verdict |
| :-- | :-- | --: | --: | --: | --: | :-- |
| A (aminx) | sequence | +0.000920 | −0.000381 | +0.001732 | 0.001869 | **pass** |
| A (aminx) | χ1 | −0.0000072 | −0.001268 | +0.001043 | 0.001404 | **pass** |
| CTRL_m (m = 1.25) | sequence | +0.018859 | +0.014655 | +0.017146 | 0.021797 | **fail** |
| CTRL_m (m = 1.25) | χ1 | +0.012153 | +0.008939 | +0.011548 | 0.015183 | pass |

`h_hat = 0.0015606`, n = 1000.

Per-structure sequence deltas, all well inside the band:

| structure | `min_p0@1.0` | `min_p0@0.3` |
| :-- | --: | --: |
| `4jnj-1_prot` | +0.000409 | +0.001504 |
| `103m_1` | −0.001052 | +0.000643 |
| `104m_1` | +0.001137 | +0.001869 |
| `105m_1` | −0.000575 | +0.000693 |
| `106m_1` | −0.000968 | −0.000110 |

## What makes this more than a green light

A pass is only worth the control that could have refused it. `controls_all_fail: true` on both
shards, and the mechanism is visible in the numbers: the aminx arm's sequence interval sits
inside ±0.01 while `CTRL_m`'s 90 % lower bound is +0.0102 and +0.0147 — above the band, so the
test rejects it. The instrument discriminates a 10 % and a 25 % temperature perturbation from
the real port at this n. Had the control passed too, the pass would have measured nothing.

**The χ1 control is the honest caveat.** `CTRL_m` passes on χ1 in both shards, because
`delta_chi = 0.05` is five times the sequence band and the control's χ1 displacement
(≈ 0.0120–0.0122) is comfortably inside it. So the χ1 arm's pass is **not** backed by a firing
negative control at this multiplier — the sequence channel is what the control certifies. This
is a property of the pre-registered band, not a defect found after the fact, and the band is
not to be narrowed now to manufacture a firing control: that would be choosing a criterion with
the numbers in view. What it means in practice is that the χ1 result should be cited as
"within the pre-registered band" and not as "a control-verified equivalence".

## Scope, stated rather than left to be assumed

- Five structures: `4jnj-1_prot` plus `103m_1`, `104m_1`, `105m_1`, `106m_1`, each pinned by
  sha256 in both `result.json` files.
- Two cells only, both `min_p0`. No other sampler configuration was confirmed.
- One control multiplier per cell (1.1 at T = 1.0, 1.25 at T = 0.3).
- Sequence and χ1. χ2–χ4 and the joint sequence–χ1 dependence remain unmeasured, as does the
  shimmed (U1) arm under batching.

## Provenance

`pilot_run_id: e0cefaa7-e027-4efe-bf63-07d78bcfabff` on both shards — the confirm declares the
pilot it descends from. Work dirs `/home/solab/projects/laser-confirm-gpu{2,3}` on titanix hold
the 20 unit bodies and stamps per shard; they are the evidence behind the aggregate and should
not be deleted while these two runs are cited.
