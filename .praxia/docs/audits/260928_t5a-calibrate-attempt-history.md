---
title: T5a layer-c calibrate — full attempt history
description: All four tracked runs of layer_c_calibrate (one pass, three prior), and the one post-registration sidecar edit
task_id: 260926_browser-export-loop
status: final
---

# T5a layer-c calibrate: attempt history

The T5a record commit (618ea76d) cites run c32f3f58 (`pass`) but not the three tracked
runs of the same stem that preceded it. All four, read from the cool-tier parquet
(`bth compact` is broken catalog-wide, see the bathos-traps memory):

| run | commit | status | outcome | exit | sidecar sha (prefix) | cause |
|---|---|---|---|---|---|---|
| 6306b3be | 3450653b | failed | error | 1 | 6482d3b7 (as pre-registered) | crash in the sized-artifact manifest rebuild: `$BV_ARTIFACTS` unwritable from the Bash sandbox |
| 833409bf | 1175d21c | completed | ctrl_unsized | 0 | ee7df559 | same write failure, now caught; landed in the residual |
| 5bfec541 | d0a9aa12 | completed | ctrl_unsized | 0 | ee7df559 | same, guard widened |
| c32f3f58 | 383333de | completed | **pass** | 0 | ee7df559 | run unsandboxed; the cited result |

## Sidecar edit after the first tracked run

Between 3450653b and 1175d21c the sidecar changed in one place only:
`[outcomes.ctrl_unsized].condition` went from `controls_sized < controls_total` to
`controls_sized < controls_total OR NOT params_written`, so a failed manifest write
lands in a declared failure state instead of none. `[outcomes.pass]` and every threshold
are byte-identical to the pre-registration (`git diff 3450653b 383333de --
scripts/browser_validation/layer_c_calibrate.bth.toml`).

The three failures were infrastructure (sandbox write denial), not measurement; none
produced deltas. The retries were not a search for a passing measurement.
