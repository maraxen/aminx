# aminx 0.5.4

Beta, still largely experimental. This patch release lets a teacher-forced, decoding-order-conditioned screen run entirely through the campaign layer: a new `score_conditionals` row kind (#3486) backed by batched decoding masks in `runner.score` (#3485). It also stops `runner.score` recompiling on every call (#3484), fixes the candidate-axis planner running an over-budget axis untiled, and documents the batch-composition bound (#3487).

## What's new

- **`task="score_conditionals"` campaign rows** (`aminx.host.campaign_score`): one structure, N fixed sequences (MPNN letters), K masks from a `.npy`, optional LigandMPNN ligand context. Output store: `logits` `(K, N, L, 21)`, `nll` `(K, N)`. Mask bytes, ligand bytes, sequences and seed enter the unit input hash, so resume reuses a unit only when they match. Same locks, done markers and quarantine as sampling rows.
- **`ScoringSpecification.ar_masks`** `(K, L, L)`: every (mask, sequence) pair against one encoding in one jitted call; `scores` `(S, K, C)`, `logits` `(S, K, C, L, 21)`. `designable_mask` refuses empty mask rows at designable positions.
- **`SCORE_BATCH_COMPOSITION_ATOL = 1e-5`** in `aminx.scoring.score`.

## What's fixed

- `runner.score` warm calls no longer recompile (3 compiles / 2.5 s to 0 / 0.24 s per warm call; runs b25cbd4e, 6f034f38).
- The candidate-axis planner sizes `ChunkedMap` tiles from the memory budget instead of returning `batch_size=0`.

## Read before upgrading

- **Existing campaign rows are unaffected**: sampling and ThermoMPNN manifest-row hashes and unit input hashes are byte-identical to 0.5.3 (golden test).
- **Batch composition**: compare one sequence's scores across different batch compositions at `SCORE_BATCH_COMPOSITION_ATOL`, not bitwise. Within one call, rows are content-invariant bitwise.
- `campaign run --pack` refuses `score_conditionals` rows; run them unpacked.
- Includes the unreleased jax floor change (`jax>=0.11.2,<0.12`) and the move of `tests`/`docs`/`benchmark`/`dev` extras to dependency groups (see CHANGELOG).
- **HuggingFace weights pin not moved**: `HF_REVISION` is unchanged from 0.5.3.
