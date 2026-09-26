# Literature-parity packets (T10, Mode A)

Protocol: bathos-literature-parity (phase prompts from
`bathos/agent_assets/skills/bathos-literature-parity/assets/0{1..4}_*.md`), configured by
`scripts/browser_validation/parity.bth.toml` (N = 3 reconstructors, M = 3 attackers).

## Packets

| Packet | Contents | Pin |
|---|---|---|
| `reference_packet/` | LigandMPNN `run.py`, `model_utils.py`, `data_utils.py`, `sc_utils.py`, `score.py`; ProteinMPNN `protein_mpnn_utils.py`; `proteinmpnn_2022.pdf`, `ligandmpnn.pdf` | LigandMPNN `26ec57ac976ade5379920dbd43c7f97a91cf82de`, ProteinMPNN `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57` |
| `impl_packet/` | the 16 `impl_paths` of `parity.bth.toml`, copied from `src/aminx/` at the commit that adds this README | aminx commit of this file |

## Which phase sees what

| Phase | Agents | Sees |
|---|---|---|
| 1 — blind reconstruction | 3 reconstructors (lenses math / algo / protocol) | `reference_packet/` ONLY. No aminx code, docs, specs or prior summaries. Blindness is checked by the T10 gate (no `src/aminx` or `aminx.<module>` string in any `recon_*.md`). |
| 2 — reconciliation | orchestrator + 1 reconciler | the three `recon_*.md`, `reference_packet/`, `impl_packet/` and the aminx repo (for code locations outside `impl_paths`, e.g. tie groups and wave schedules) |
| 3 — adversarial refutation | 3 attackers (lenses stats / hyper / struct) | `clauses.json`, both packets, the aminx repo; may run narrow local probes |
| 4 — adjudication | orchestrator | the three `refute_*.md`, `clauses.json`; confirms defects on >= 2 votes or a runnable test |

`clauses.json` (with every `core` flag and DEVIATION reconciliation note) is committed and its sha256
written to `outputs/browser_validation/layer_a/preregistered_params.json` as
`litparity.clauses_sha256` BEFORE Phase 3 and before any parity run (O8).
