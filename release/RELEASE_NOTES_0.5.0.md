# aminx 0.5.0

Beta, still largely experimental. This release makes the default `incremental="auto"` decode use the cached path on the sampling path (about 8x faster single-state sampling, output unchanged). It adds two serializable sampling controls (`decoding_order_seed`, `fixed_position_context`), adopts `xtrax.pack` and `xtrax.eda`, and changes the `generate_ar_mask` order contract.

## What's new

- **Faster default decode** (#273, fixes #272): `incremental="auto"` now takes the cached decode whenever its preconditions hold under the per-sample decoding-order vmap. Before, the exactness predicate was batched, `lax.cond` lowered to a select, and both bodies ran, so `auto` always paid the full O(L²·k) recompute. On production rows (TITAN RTX, L=256, 20 samples per row), single-state rows are 7.99–8.29x faster and 260/260 rows are bit-identical to main before the fix (86042c0e). Non-identity `state_position_map`, tie groups larger than the slab, `mutual_context` and caller-supplied masks keep the full-recompute fallback.
- **`decoding_order_seed`** (#265): when set, every sample's decoding order comes from that seed alone, independent of the sample key and the manifest hash, so rows can share one decoding order. `None` (default) is bit-identical to before.
- **`fixed_position_context`** (#266): `"random_order"` (default, reference behaviour, bit-identical), `"canonical_order"`, or `"mutual_context"` (a measurement mode that also changes the designed positions' distribution). Honoured on the runner and fused paths, and recorded in every store's root attrs.
- **`xtrax.pack` and `xtrax.eda` adoption** (#270): packed per-row unit storage with completion stamps (`aminx.host.pack_shim`) and planner EDA artifacts (`aminx.host.plan_eda`, optional `eda` extra).
- **Stock and parity work** (#254–#260): `ligand_cutoff_for_score` wired into `score()`, ThermoMPNN adoption without the stock dependency, Tier 1/2 parity vehicles with hard-failure guards, and a sharded knob-gate ledger with `--family` flags.

## Read before upgrading

- **`generate_ar_mask` order contract** (#255): the untied branch now reads `decoding_order` as an ORDER array, not a RANK array. A caller that passed `argsort(order)` now silently gets the mask of the inverse order. No runtime guard is possible, since a rank array is also a valid permutation. Pass the order itself. The return dtype is unchanged (int32).
- `generate_ar_mask`'s `chain_idx`, `tie_group_map` and `num_groups` are keyword-only. Passing them positionally raises `TypeError`.
- `decoding_order_seed` and `fixed_position_context` are not part of `manifest_row_hash`. Rows that differ only in these need distinct output roots.
- Dependencies: xtrax `==0.4.1b7` (was `0.4.1b5`, now with the `pack` extra), proxide `>=0.2.0` (was `>=0.1.0a17`), cisternal `>=0.1.6` for the `agent` extra. A new optional `eda` extra brings the plotting stack.

## Known limitations

- The sample-axis memory planner underestimates per-sample activation memory, so wide sample rows can run out of device memory instead of being chunked (#274). Keep rows narrow (the campaign default of 20 samples per row is fine) until that is fixed.
- `WaveScheduleBundle.mask_consistent` records a relation between the wave and `conditioning.ar_mask`, but it lives on the wave. Code that rewrites the mask after `with_decoding_order` must clear it.
- The two tests in `tests/knob_gate/test_knob_superset.py` need the `redsox` package and fail without it.
