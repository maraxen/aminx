---
title: 'Distributional pilot: unspecified pssm/tied fixtures and the T=0.1 lane'
description: 'Decision request before the Potts sample_dist pilot runs: spec names no pssm or tied fixture, and the T=0.1 near-margin control looks unworkable (as with ProteinMPNN 260929)'
status: proposed
task_id: 260929_potts-laser-xtrax-compose
date: '261004'
supersedes: ''
backlog_ids: ''
---
# Distributional pilot: unspecified pssm/tied fixtures and the T=0.1 lane

**Status: needs the user's decision before the Potts pilot runs.** Nothing has
run beyond a smoke test; no sidecar has been pre-registered, because the
condition set it would pre-register is what is being decided here.

Context: the spec's distributional protocol (`potts_sample_dist`,
`laser_sample_dist`, spec section 7 'Distributional protocol' / 'Runs' /
'Pilot') had no implementation. On side branch `wt/260929-sample-dist`:
the statistics layer (`d7d3e740`, 11/11 synthetic-ground-truth tests incl.
negative controls) and the Potts pilot driver (`ceb26bef`, smoke-run end to end
on titanix). The driver is ready; these two questions block a meaningful run.

## 1. Two fixtures the spec never pins

The spec's Potts conditions include:

- **`pssm`** -- "fixture PSSM, `pssm_multi=0.5`, bias flag on, mode none, T=0.3"
- **`tied`** -- "the pinned 2-member tied fixture, beta=1, T=0.3"

"Pinned" is stated, but no path or SHA-256 is given anywhere, and no such
fixture exists in the repo. Upstream's own examples do not fit: its PSSM inputs
(`inputs/PSSM_inputs/{4YOW,3HTN}.npz`) are for other structures than the pilot
pair `{2yc3,3dkm}`, and `inputs/example_tied_positions.json` is not a 2-member
group on them (empty for 2yc3; 6w25 lists five A positions plus B1). The fixer
correctly refused to substitute either.

Options:

- **A. Define them now, deterministically, and pin them.** E.g. a PSSM for 2yc3
  derived from a fixed rule (or from the upstream example format applied to
  2yc3), and a 2-member tie on two designable positions of 2yc3 chosen by a
  stated rule; both committed with SHA-256 and recorded in the pilot sidecar
  as a spec amendment. Cheapest; the choice is arbitrary but explicit.
- **B. Use upstream's structures for those two conditions** (4YOW/3HTN for
  `pssm`; 6w25's group, cut or kept, for `tied`) -- realistic inputs, but the
  conditions then run on different structures than the rest of the pilot.
- **C. Drop `pssm` and `tied` from the distributional tier**, relying on their
  exact-tier coverage (PSSMMix and tied decode are both in `potts_ar_decode` /
  `potts_ar_refine_exact`, which grade bit-exact under injected uniforms).

## 2. The T=0.1 lane

The smoke cell `plain@T=0.1` (n=50, B=50 -- too small to conclude, but
indicative) escalated: Delta_neg(m) was 0.0022 / 0.0002 / 0.0007 for
m = 1.1 / 1.25 / 1.5 -- tiny and not monotone. At T=0.1 the sampler is near
argmax, so scaling T by 1.1-1.5 barely moves the distribution, and the
near-margin negative control cannot clear delta + 2h.

**Now seen in BOTH families.** The LASEr pilot driver (`00ef0203`) smoke cell
`min_p0@T=0.1` escalated the same way: Delta_neg 0.0022 / 0.0081 / 0.0006 for
m = 1.1 / 1.25 / 1.5 (again n=50 -- indicative, not a finding). LASEr has no
fixture gap, so for LASEr this lane is the only open question.

**This is the same mechanism the user ruled on 260929** for ProteinMPNN sampling
(memory `project_sampling-drop-p07-low-temp-lane`): the P07@0.1 lane's margin
was below estimator noise, and the lane was dropped because T=0.1 stays
covered bitwise by an exact tier.

Options:

- **A. Drop T=0.1 from the distributional tier** (consistent with 260929), and
  state which exact tier covers it.
- **B. Keep it and let the full pilot (n=1000, then n=4000 on escalation)
  decide** -- the spec's own escalation path ends in `instrument_invalid` if it
  still fails, which would be a documented, pre-registered outcome rather than a
  silent drop. Costs the extra n=4000 pilot for that lane.

## Recommendation

1C or 1A, and 2B-then-A: keep the pre-registered escalation path honest by
letting the n=1000 pilot measure the T=0.1 lane once (it is cheap at n=1000),
and if it escalates as the smoke run suggests, drop it under the 260929
precedent rather than spending the n=4000 re-pilot. For the fixtures, 1C is
the least invented and leans on coverage that already passes bit-exact; 1A is
right if the distributional check of PSSMMix/tied is wanted for its own sake.

LASEr's half (`laser_sample_dist`) is separately blocked on the LaserDriver
`sample` purpose, which the spec requires and which is being built on side
branch `wt/260929-laser-rotamers-wave`.

