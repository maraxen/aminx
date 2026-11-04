# ThermoMPNN driver float16 gate records: provenance

Recorded outputs of two titanix runs. The catalog records live in the titanix scratch catalog, not in this repository.

## Run f12f3209 (old fixed-bar gate), file `thermo_mpnn_driver_f16.json`

- Run id: `f12f3209-fde2-49a8-a5f2-a07c66ea0633`
- Sidecar: `scripts/parity/thermo_mpnn_driver_f16.bth.toml`, sha256 `f79839f6ca2e...`
- Script: `scripts/parity/thermo_mpnn_driver_f16.py` at commit `8028a7d5`
- Tree: clean. Platform: titanix, CPU. Date: 2026-11-02.
- Outcome: **fail**
- Structure: 1ubq only (76 residues, 96 pairs).

## Run 18c1d29c (precision-based v2 gate), file `thermo_mpnn_driver_f16_v2.json`

- Run id: `18c1d29c-afd4-435d-a8d1-ab39d7aeb2dc`
- Sidecar: `scripts/parity/thermo_mpnn_driver_f16_v2.bth.toml`, sha256 `bbad1150548000d3780835f1358d576f6475cb4e...`
- Script: `scripts/parity/thermo_mpnn_driver_f16_v2.py` at commit `6c01a45e`
- Tree: clean. Outcome: **fail**.

Structures and mapping (confirmed from the script's `FIXTURES`):

| label | file | residues | pairs (d_ens1) | status |
|---|---|---|---|---|
| ubq | 1ubq.pdb | 76 | 96 | post-hoc (the v2 criteria were written after its numbers were seen) |
| ho1 | 1SMD.pdb | 496 | 629 | held out |
| ho2 | 1mbn.pdb | 153 | 194 | held out |

## Weight handling

The weight files were marked `skip-worktree` in the scratch clone so that git reported the tree as clean. The weights are real LFS content, not placeholders.

## Tolerance versus observed difference (v2 gate)

Tolerance is `68 * 2^-10 * scale`, where `scale` is the max absolute float32 reference. Values are from the recorded fields of run 18c1d29c.

| structure | quantity | max-abs diff | tolerance | result |
|---|---|---|---|---|
| ubq | v1_ddg | 0.0480 | 0.1764 | within |
| ubq | d_ens1 ddg | 0.0303 | 0.1446 | within |
| ubq | d_ens1 ddg_double | 0.0517 | 0.2517 | within |
| ho1 (1SMD) | v1_ddg | 0.2055 | 0.2237 | within (92%) |
| ho1 (1SMD) | d_ens1 ddg | 0.3995 | 0.1499 | **exceeds** |
| ho1 (1SMD) | d_ens1 ddg_double | 0.2916 | 0.3003 | within (97%) |
| ho2 (1mbn) | v1_ddg | 0.0637 | 0.1897 | within |
| ho2 (1mbn) | d_ens1 ddg | 0.0402 | 0.1663 | within |
| ho2 (1mbn) | d_ens1 ddg_double | 0.0633 | 0.2672 | within |

Pearson r is at least 0.9997 for every quantity.

No cause is asserted for the ho1 d_ens1 ddg excess.

## Error diagnostic (exploration), run 611e5c8c-6c6e-4744-b92c-cc9199ec6723

- Sidecar `scripts/parity/thermo_mpnn_f16_error_diagnostic.bth.toml` (sha 23bfa173...) committed in 43f183b1, before the script (92c7968b). Run on titanix CPU at 92c7968b, tree clean (weights skip-worktree as above). bth outcome `observed` (exploration sidecar); the class below is read by hand from the recorded fields.
- Files: `thermo_mpnn_f16_error_diagnostic.json`, `thermo_mpnn_f16_error_diagnostic_pairs.npz` (per-pair arrays).
- Class for ho1 (1SMD, 496 residues) d_ens1 ddg, by the sidecar's definitions: **concentrated** (no non-finite values, max |f16| 2.258 so no overflow, top-1% of 629 pairs hold 64.6% of squared error; top-5% hold 78.1%).
- Worst entry: residue i=430, a=9, f32 value -0.636 (not an extreme output), |diff| 0.3995.
- Descriptive size comparison (ddg, d_ens1): 76 residues rms 0.0050 / max 0.030 / top-1% share 0.21; 153 residues rms 0.0049 / max 0.040 / 0.24; 496 residues rms 0.0123 / max 0.400 / 0.65. rms is flat from 76 to 153 and about 2.5x at 496, while max is about 13x; the large max is driven by a few entries. Three structures cannot separate size from structure-specific effects. Mechanism not established; intermediate activations were not recorded.
- Correlation of |error| with |f32 value|: Spearman 0.21 / 0.38 / 0.28 (ubq / ho2 / ho1). With pair sequence separation (ddg_double): -0.09 / -0.04 / -0.03.
