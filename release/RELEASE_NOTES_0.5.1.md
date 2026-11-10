# aminx 0.5.1

Beta, still largely experimental. This patch release fixes the sample-axis memory planner so wide sample rows chunk instead of running out of device memory (#274), and narrows the autoregressive mask to `int8` (#275).

## What's fixed

- **Sample-axis memory planner** (#274, #280): the per-sample activation estimate was about 4.1 MiB at L=256 against a measured 128-145 MiB, so a full vmap was chosen at 20, 128 and 1024 samples alike and wide rows failed with `RESOURCE_EXHAUSTED`. The estimate now uses ProteinMPNN v_48's dimensions and an empirical live-tensor factor (about 163 MiB per sample at L=256), and a demoted sample axis is widened from xtrax's minimum tile of 1 to the largest tile that fits the budget (35 at a 5.63 GiB budget, 100 at 15.95 GiB). A sample axis that fits whole is restored to a full vmap. Production MS-L cell on a TITAN RTX, 0.5.0 vs this change: 128- and 256-wide rows complete at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.30` where 0.5.0 fails; all 12 compared rows are bit-identical; the 20-sample row keeps its full vmap.

## Read before upgrading

- **AR mask dtype** (#275): `generate_ar_mask`, `generate_wave_ar_mask`, `ar_mask_from_decoding_order` and `full_context_ar_mask` now return `int8` (they were int32 or float32). `build_inference_bundle` takes `ar_mask_dtype` (default `int8`); a caller-supplied mask is read as nonzero = visible. int8 matmul, add and elementwise product of two masks wrap past 127 (`sum` and `cumsum` upcast). The `p04`/`p07` export graphs keep a float32 mask.

## Known limitations

- The planner's live-tensor factor is fitted at one shape (L=256), one model family (ProteinMPNN v_48) and one GPU stack, and scales linearly in L. Other lengths and families are unmeasured; smaller families are over-estimated and so chunk earlier than needed.
- At the largest tiles the measured peak can sit a few percent above the estimate (17102 MiB measured at tile 100 against about 16.3 GiB estimated); the budget's 20% headroom absorbs it.
- Other batch axes (temperatures, noises) still demote to tile 1 when they do not fit.
- The two tests in `tests/knob_gate/test_knob_superset.py` need the `redsox` package and fail without it.
