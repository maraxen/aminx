# aminx 0.5.2

Beta, still largely experimental. This patch release stops a chunked sample axis from recompiling on every row (#282), and makes `generate_ar_mask` take its decoding order by keyword so a caller passing a rank array fails loudly instead of silently getting the inverse mask (#285). It also adds FrustraMPNN support (#271) and an xtrax campaign-EDA plugin (#277).

## What's fixed

- **Chunked sample axis recompiled every row** (#282, #283): `_dispatch_axis` ran a chunked sample axis as an eager `lax.map` over a closure rebuilt per row, so jax compiled a fresh executable each time (about 5.8 s of a ~7.8 s 128-wide row on a TITAN RTX). The unified single-state path now runs it through one module-level `filter_jit` function, and `decoding_order_seed` is passed as an int32 array (`None` and seeds outside int32 stay static; the PRNG key is unchanged). Pre-registered gates, production MS-L cell family, main vs this change: no seed set (bathos 760e6bde) 26 of 26 rows bit-identical, compile slope 1.0 to 0 per added row, 5.66x faster on 8 rows of 128; a distinct seed per row (bathos 5ac827c7) 16 of 16 rows bit-identical, slope 0, 5.48x faster; peak memory at most 1.06x main in both.

## Read before upgrading

- **Breaking: `generate_ar_mask(order=...)`** (#285): the decoding order is now the keyword-only argument `order`. Positional calls and the old `decoding_order=` keyword raise `TypeError`. Since 0.5.0 the function reads its argument as an ORDER array (`order[t]` = position decoded at step `t`), not a RANK array. A caller still passing a rank array (`argsort(order)`) silently got the mask of the inverse order in 0.5.0 and 0.5.1, because an order and a rank are both permutations of `0..L-1` and cannot be told apart by value. If you call `generate_ar_mask`, check which you pass; invert a rank array with `jnp.argsort(rank)` and pass it as `order=`. A rank array passed as `order=` is still read as an order.
- **HuggingFace weights pin not moved** (deliberate): `HF_REVISION` stays at `5ff5cf16`. The Hub default branch has since gained `megascale_default_la-true_esm-False_split1.eqx.zst`, which this pin omits, so that checkpoint is not fetchable through this release. The pin-lag test exempts exactly that file (`DELIBERATELY_NOT_PINNED`). No weight that loads today changes.

## Known limitations

- Only the unified single-state path (no encoding fusion) uses the chunked jit; the fused and legacy paths still use the eager map and are untested. One GPU model, one shape (L=256, ProteinMPNN v_48).
- The two tests in `tests/knob_gate/test_knob_superset.py` need the `redsox` package and fail without it.
