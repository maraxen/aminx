# aminx 0.5.3

Beta, still largely experimental. This patch release makes selenomethionine (MSE) tokenise as `X`, as the reference LigandMPNN parser does, which fixes the LigandMPNN Heavy Parity gate (#288). It also moves the tiling vocabulary onto `xtrax.tiling` (#287), which removes some import paths.

## What's fixed

- **Selenomethionine typed `M`, reference gives `X`** (#288): aminx now asks proxide for `modified_residue_typing="unknown"` on every structure-parse path. The option shipped in proxide 0.2.1, so aminx now requires `proxide>=0.2.1`; with an older proxide the parse path raises instead of silently keeping `M`. The LigandMPNN Heavy Parity gate, which failed on exactly this, passes (run 43456465602 on the branch).

## Read before upgrading

- **Scores and samples change for any structure containing selenomethionine.** The token is `X` where it was `M`. Training loaders (`training/trainer.py`) are unchanged and still type MSE as `M`, so inference now sees `X` where training saw `M` (the reference parser behaves the same way at inference).
- **Breaking import paths** (#287): `aminx.tiling.SafeMap`, `aminx.tiling.iterator.SafeMapIterator`, `aminx.utils.safe_map.safe_map`, `aminx.utils.safe_scan.safe_scan` and `aminx.tiling._deprecation` are removed. Use `aminx.tiling.ChunkedMap(batch_size=...)`, `xtrax.transforms.chunked_map` and `xtrax.safe_scan`. Code that already used `ChunkedMap`, `Vmap`, `Scan` or `DedupGather` is unaffected.
- **Dependency pins**: `xtrax[io,pack,export]==0.4.1b8` (was `0.4.1b7`, also in the `eda` and `onnx` extras) and `proxide>=0.2.1` (was `>=0.2.0`).
- **HuggingFace weights pin not moved**: `HF_REVISION` is unchanged from 0.5.2.

## Known limitations

- The proxide projection entry point (`py_parsers.rs`, `from_raw` without an `OutputSpec`) is not covered by `modified_residue_typing`; aminx's runner parse path is.
- The two tests in `tests/knob_gate/test_knob_superset.py` need the `redsox` package and fail without it.
