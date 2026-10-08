---
title: using-aminx skill — driver families section (Z2, pending install)
description: The PottsMPNN/LASErMPNN section for the using-aminx skill, plus the frontmatter edits, staged here because ~/.claude/skills is sandbox-write-protected
task_id: 260929_potts-laser-xtrax-compose
status: ready-to-install
created: 261002
---

# Z2: `using-aminx` skill update

`~/.claude/skills/using-aminx/SKILL.md` predates both driver families — it was
written in June 2026 and documents ProteinMPNN/LigandMPNN only, so the skill
does not surface at all for a `pottsmpnn` or `lasermpnn` question. The README
half of Z2 landed in `4cdb1a7b`; this is the skill half.

It is staged here rather than applied because `~/.claude/skills` is
write-protected in this session's sandbox. That is a deliberate boundary around
user config, so applying it is left to the user or a non-sandboxed session.

## Install

Three edits to `~/.claude/skills/using-aminx/SKILL.md`:

1. Insert the section below immediately before `## Gotchas`.
2. Replace the frontmatter `description:` line with the one below.
3. Replace the frontmatter `triggers:` line with the one below.

### `description:`

```
description: Use when running aminx — the JAX/Equinox ProteinMPNN/LigandMPNN/PottsMPNN/LASErMPNN interface — to sample sequences, score sequences, compute conditional/unconditional logits, Potts pair energies and ddG, or LASErMPNN proofreading from a protein structure, via the Python API or the `aminx` CLI and spec files.
```

### `triggers:`

```
triggers: [aminx, proteinmpnn, ligandmpnn, pottsmpnn, lasermpnn, sample, score, logits, conditional, unconditional, inverse folding, sequence design, spec, run sample, potts, etab, ddg, proofread, proofreading, rotamer, model-family]
```

### Section to insert before `## Gotchas`

## 8. Driver families — PottsMPNN and LASErMPNN

Two further families run through the `FamilyDriver` seam rather than the stock
MPNN path: `pottsmpnn` (pairwise Potts energies, ddG) and `lasermpnn` (ligand-aware
joint sequence + rotamer design, proofreading). Select with `--model-family` /
`RunSpecification.model_family`; family knobs ride on one JSON flag each.

| family | `output_kind` accepted | knobs |
| :--- | :--- | :--- |
| `pottsmpnn` | `nll`, `logits`, `energy`, `ddg` | `PottsMPNNOptions` via `--potts-options-json` |
| `lasermpnn` | `nll`, `logits`, `proofread_unconditional`, `proofread_conditional` | `LaserOptions` via `--laser-options-json` |

An `output_kind` the family does not accept is rejected at spec construction, so
a typo fails loudly instead of silently falling back to ProteinMPNN.

```bash
aminx run score --inputs complex.pdb --model-family pottsmpnn \
  --output-kind ddg \
  --potts-options-json '{"binding_energy_optimization": "both", "mutant_csv": "m.csv"}'

aminx run score --inputs complex.pdb --model-family lasermpnn \
  --output-kind proofread_conditional \
  --laser-options-json '{"n_decoding_orders": 2, "n_dropouts": 2}'
```

```python
from aminx.host.runner import score
from aminx.run.options import LaserOptions
from aminx.run.specs import ScoringSpecification

result = score(
  ScoringSpecification(
    inputs="complex.pdb",
    model_family="lasermpnn",
    model_local_path="weights/laser.eqx",
    output_kind="proofread_conditional",
    sequences_to_score=["..."],
    laser=LaserOptions(n_decoding_orders=2, n_dropouts=2),
  ),
)
mean = result["structures"]["0"]["arrays"]["proofread_mean"]  # (R, 21)
```

**Proofreading shapes.** Both proofread kinds return **one ensemble per
structure**, not one per scored candidate — proofreading reads the structure's
own sequence. Arrays are `proofread_mean (R, 21)`, `proofread_std (R, 21)`,
`proofread_mean_plus_std (R, 21)`, `residue_ids (R,)`, where `R` is the focus set
(first-shell ligand contacts, or `selection_string`). `std` and `mean_plus_std`
are conditional-only.

**Proofreading cost.** `n_decoding_orders` and `n_dropouts` both default to
`10`, and the work is their product **per focus residue** — 100 eager decodes
each, run under `jax.disable_jit` so the host can hand a distinct dropout mask to
every call. Drop both to 2 for anything exploratory. `n_decoding_orders=1` leaves
`proofread_std` all-NaN: that is the ddof=1 std of one sample and it is kept
rather than silently patched.

**Known divergence.** `proofread_unconditional` ignores `proofread_dropout`
today. Upstream keeps every `nn.Dropout` in train mode for that forward pass
(`run_proofreading.py:20-28,205`); aminx applies none, so the two differ whenever
`proofread_dropout` is true (the default). Debt #2424. The conditional path is
unaffected and is parity-verified to ~4e-15.
