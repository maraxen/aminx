# aminx 0.4.0

Beta, still largely experimental. This release adds three model families (Caliby, FAMPNN, ThermoMPNN as a driver), AntiBMPNN and CyclicMPNN support, and multi-backend campaigns.

## What's new

- **Caliby** (`model_family="caliby"`): port of ProteinDesignLab/caliby with the atom MPNN, graph Potts head, sidechain packer and driver. Eight checkpoints are pinned. Scores are the log-softmax over all 21 states including GAP, to match upstream.
- **FAMPNN** (`model_family="fampnn"`): port of richardshuai/fampnn, the full-atom MPNN with GVP encoder and sidechain diffusion. Checkpoints `fampnn_0_0`, `fampnn_0_3`, `fampnn_0_3_cath`, loaded through a restricted unpickler.
- **ThermoMPNN driver** (`model_family="thermompnn"`, `output_kind="ddg"`): ThermoMPNN v1/v2 and ThermoMPNN-D as a registered driver family.
- **AntiBMPNN** checkpoints (`v_48_{000,010,020,030}`) and CDR frequency bias (`aminx.model.cdr_bias`, with a keyword-only `cdr_mask=`).
- **CyclicMPNN** head-to-tail offset wrapping through `cyclic` on `Aminx` and the feature classes. Non-cyclic output is bit-identical.
- **MPNN variant weights**: AbMPNN, HyperMPNN, CyclicMPNN and CA-only ProteinMPNN, with pinned Hugging Face revisions.
- **Multi-backend campaigns**: arms across model families, a hoisted-compile cache keyed by `TopologyKey`, and a unified Zarr + Parquet campaign sink.
- **Potts**: `condition_on_fixed_sites`, and real multi-state PoE inference instead of silently using state 0.
- `RunSpec` carries `caliby`, `fampnn` and `thermompnn` options and refuses them for any other family.

## Read before upgrading

- No API that existed in 0.3.0 changed incompatibly.
- xtrax is pinned to `==0.4.1b5` (was `0.4.1b2`).
- `antibmpnn` `v_48_002` is refused instead of aliased to `v_48_000`, since it is not a published checkpoint.
- The new families need their checkpoints from the Hub at the pinned revision. Each family artifact is checked against a pinned sha256.

## Known limitations

- Caliby's Potts sampler is not Gibbs.
- Residue-numbering parity with the upstream Caliby and FAMPNN featurizers is not established.
- Caliby sidechain packing is not a separate stage, and the CDR transform does not take per-step `(V,)` logits.
- Long sequences (around L=500) can run slowly or out of memory on the new family loops.
- The opt-in AR wave loop (`AutoregressiveConfig.intent_loop`) stays off by default; its benchmark showed no speed contrast.
