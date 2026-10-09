---
title: ProtonPottsMPNN V1 -- scope, the 30-token vocabulary, and how it enters the redsox gate
description: What ProtonPottsMPNN V1 ships (score:energy/ddg/selectivity and pH design on the Potts energy), what is refused and tracked as debt, and the choices that put it in the ledger as one combined vehicle.
status: accepted
task_id: 261008_protonpotts-p10
date: '261008'
supersedes: ''
backlog_ids: '5781'
---
# ProtonPottsMPNN V1 -- scope, the 30-token vocabulary, and how it enters the redsox gate

Spec: `.praxia/docs/specs/261007_protonpottsmpnn-support.md` (sections 42-51). Implementation: PR #211.

## Decisions

Taken by the user over 261007-261008 (each is quoted from the session where it was made, in the spec's section 11 and
sections 42-49):

1. **Protonation labels are consumed, not assigned** (spec 11a). aminx reads pre-assigned labels from a JSON file; the
   end-to-end labeller (HBPLUS binary plus a FLAML ensemble) is out of scope.
2. **The pH design engine is in V1** (11b), restricted to block descent and the centre-free greedy on the Potts energy.
   MCMC, two-phase and Gibbs methods are refused naming debt #2616; decoder-backed design and the 30-token decoder are
   refused naming #2617; upstream knobs that are not exposed are #2621; known gaps of the pH surface are #2622;
   ProteinSMC kernels for the Potts-head families are #2615.
3. **Each inference path declares its alphabet** (11c). The v6 vocabulary is 30 tokens and is a property of the path,
   not a module constant; result arrays carry a `vocabulary` attribute, and aminx's token order differs from upstream's.
4. **Own options class**: `ProtonPottsOptions`, not `PottsMPNNOptions`, most of whose knobs do not apply.
5. **No fallback to stock MPNN.** A purpose the driver does not run is refused.

## Decided in P10, flagged for reversal

6. **One combined ledger vehicle, `protonpotts_parity`, not seven rows.** The seven waves share sealed dumps, so a row per
   wave would multiply manifest rows (47 either way: each control is a row) without adding independence. The vehicle
   runs the waves unchanged as subprocesses and reports controls as `<wave>.<control>`. The cost: a change that stales
   one wave re-runs all seven (minutes of CPU, not hours).
7. **The wave scripts and the checkpoint converter moved from `scripts/protonpotts/` to `scripts/parity/`.** The closure
   scanner only reads `src/` and `scripts/parity/`, and the `scripts/protonpotts/` tree is unscoped, so leaving them
   there would have put the measuring code outside the row's closure. Earlier graded runs keep their records under the
   old paths; only path strings in the moved files changed.
8. **The `protonpottsmpnn_` weights arm is inert** like the two it joins (no vehicle slug carries the prefix). Adding it
   follows the spec; tightening it would be a separate decision about a pre-registered criterion.
9. **Known gap, not closed:** `scripts/convert_weights.py` (loaded by the converter) is outside every scoped prefix, so
   editing it cannot stale the row. The converted weights are still pinned by the checkpoint sha and graded against
   upstream whenever the row runs.
10. **`block_zscales` maps its blocks through xtrax, and L-DRV gets no new allowlist reason.** An earlier P10 draft added a
    `memory_bounded_map` reason for the one carry-free `lax.scan` in `block_zscales`. The user asked for it to be composed in
    xtrax instead (261009): the block axis is an `AxisSpec`, `plan_axis_strategy` chooses a vmap or tiles of it from the
    memory budget (each block holds a V^B joint), and `make_axis_dispatch_via_xtrax` builds the map, padding a ragged last
    tile and slicing the padding off before the pool. The three genuinely sequential loops keep `sequential_dependency`.
    The axis is declared in `ph_descent.py`, not `aminx/tiling/axes.py`, because the host runner imports that registry
    and an edit there would stale every row; move it there with the next wave that touches shared files.

## Consequences

- Every existing ledger row's closure includes `host/runner.py` and `run/`, which P8 edited, so all eight are stale and
  the final `launch_wave.sh --stale-only` re-wave re-runs them with the ninth row.
- `record_trajectory` is accepted and changes nothing (trajectories are not recorded); its knob test says so.
- Three families now sit in `tests/knob_gate`: 50 alias rows and 23 knob-semantics tests cover ProtonPottsMPNN, and the
  closure self-test pins that a Potts change reaches the ProtonPotts row, never the reverse, and that no family reaches LASEr.

## Decided 261009 (user): browser shape and the decoder

11. **pH design in the browser is a per-block graph driven by a JS loop**, not one sweep graph. Block descent is a host-side
    sweep over placement blocks on the Potts energy; each block's conditional energy is one loop-free graph call and JS owns
    the sweep, the convergence test and the injected uniforms. This follows the Potts export lesson (no in-graph `Loop`/`Scan`:
    control flow forces device copies on WebGPU, and the monolith was abandoned for the split in the ProteinMPNN export). Graded
    by the same ladder as scoring: export gate (ORT-CPU/ORT-Web), Node loop gate against the real driver, headless Chromium gate.
12. **The 30-token decoder is implemented now (debt #2617)**, ahead of the remaining browser work. It is the shared blocker for
    plain ProtonPotts `sample`, the decoder-backed pH methods (`autoregressive`, `mpnn_sample`, `selective_source='decoder'`, the
    whole-chain `gibbs` baseline, `backend='mpnn'`) and a browser sampler. Order: upstream oracle dump (draw-shimmed AR step, as the
    Potts and LASEr oracles did) -> port and grade the V=30 decoder -> driver `sample` path -> decoder-backed pH methods ->
    browser export. Each step is a pre-registered wave with negative controls; no tolerance is widened.
13. **Browser work is sequenced behind its blockers**, not built speculatively: per-block pH graph first (needs nothing new),
    decoder graphs after the decoder is graded.
