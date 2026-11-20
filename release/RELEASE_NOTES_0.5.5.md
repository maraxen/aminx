# aminx 0.5.5

Beta, still largely experimental. This patch release stops campaign units from recompiling for every new LigandMPNN ligand atom count and every new number of sequences per unit (#3510). Each shape class now compiles once per process.

## What's new

- **Ligand atom-axis bucketing, on the score and sample paths.** LigandMPNN's ligand atom axis is padded on the host to a ladder rung (16, 32, 64 … 1024) with mask-0 atoms. Padded atoms carry no weight.
- **Bucketing of sequences per unit** in `score_conditionals`. Units are padded to a ladder rung with copies of row 0, which are dropped from the store. Padding happens only while the padded logits fit the unit's memory budget. Stores record `n_sequences` and `n_sequences_bucket`.
- **Shape-class grouping** in `execute_manifest` (on by default; `group_by_shape_class=False` turns it off). Rows that share a compiled shape run together. Results are reported in manifest order, and locks, done markers and resume are unchanged.
- **Compile telemetry** in done markers (`compile_telemetry`): shape class, backend compiles and compile seconds, whether the unit was the first of its class, and the JAX compilation-cache setting. It never enters stores or input hashes.

## Evidence

- **Padding numerics**, pre-registered in `scripts/experiments/score_padding_numerics.bth.toml` (sidecar sha256 `ef11253b…`, committed before every run). All runs came out **within_bound** against `SCORE_BATCH_COMPOSITION_ATOL` (1e-5):
  - three on a TITAN RTX GPU (`a3a4d5cc`, `35c89713`, `d0a63567`) and one on CPU (`106109a2`);
  - maximum delta 4.1e-6, positive control 0.031, repeat delta 0;
  - families: ProteinMPNN, SolubleMPNN, LigandMPNN.
- **Compiles.** In a fresh process the first unit compiles and later units of the same class compile 0 times, on both CPU and GPU.
- **No leak between rows.** Changing only the padding rows' content leaves real rows bitwise equal, and a planted sub-bound leak is caught (#3525).

## Read before upgrading

- **Resume recomputes once.** `SAMPLING_NUMERICS_EPOCH` is now 3, so units computed on 0.5.4 or earlier are recomputed when a campaign resumes on 0.5.5, not reused. Manifest-row hashes are unchanged.
- Within one unit, score values can differ from 0.5.4 by float reassociation, within `SCORE_BATCH_COMPOSITION_ATOL`. Compare across versions at that tolerance, not bitwise.
- The shape-class key is an approximation (debt #3524). An imprecise key can cost extra compiles but never changes results.
- A persistent JAX compilation cache is not wired in yet. It will come from xtrax (xtrax#278); telemetry reports the current cache setting.
- **HuggingFace weights pin not moved**: `HF_REVISION` is unchanged from 0.5.4.
