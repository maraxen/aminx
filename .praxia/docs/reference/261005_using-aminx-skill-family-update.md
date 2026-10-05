---
title: 'using-aminx skill: the model-family update Z2 still owes, as an applyable patch'
description: Exact anchored edits to add PottsMPNN/LASErMPNN to the user's global using-aminx skill, including one stale line that is now wrong rather than merely incomplete
status: proposed
task_id: 260929_potts-laser-xtrax-compose
date: '261005'
---
# using-aminx skill: the model-family update Z2 still owes, as an applyable patch

**Why this is a doc and not an edit.** Z2 (spec line 1247, "CLI docs + using-aminx skill") was
delivered by halves: the docs half landed as `docs/MODEL_FAMILIES.md` in `e7134ca4`, but the
skill lives at `~/.claude/skills/using-aminx/SKILL.md` — outside this repo and outside this
session's write allowlist. Editing the user's global skill configuration unattended is not
mine to do, so the content is staged here as anchored edits instead. Applying it is a copy,
not a rewrite.

Target file: `~/.claude/skills/using-aminx/SKILL.md` (283 lines, dated June 9, zero mentions
of either family).

## 0. The one edit that fixes something WRONG, not merely missing

Everything else below is an addition. This one is a correction, and it is the reason this is
worth applying rather than deferring: the skill currently states a *false* set of valid values.

In **§7 CLI + spec workflow**, the "Key base options" paragraph reads:

> `--model-family` (`proteinmpnn`/`ligandmpnn`)

Two of the four valid values are missing. `src/aminx/run/specs.py:241` declares:

```python
model_family: Literal["proteinmpnn", "ligandmpnn", "pottsmpnn", "lasermpnn"] | None = None
```

Replace that fragment with:

> `--model-family` (`proteinmpnn`/`ligandmpnn`/`pottsmpnn`/`lasermpnn`; see §8)

## 1. Frontmatter

The skill will not trigger on a family question as written. Replace `description:` and
`triggers:` with:

```yaml
description: Use when running aminx — the JAX/Equinox ProteinMPNN/LigandMPNN/PottsMPNN/LASErMPNN interface — to sample sequences, score sequences, compute Potts energies or ddG, run LASEr proofreading, or compute conditional/unconditional logits from a protein structure, via the Python API or the `aminx` CLI and spec files.
triggers: [aminx, proteinmpnn, ligandmpnn, pottsmpnn, lasermpnn, potts, laser, model family, sample, score, logits, energy, ddg, proofread, conditional, unconditional, inverse folding, sequence design, spec, run sample]
```

## 2. New section, to insert immediately before `## 7. CLI + spec workflow`

Numbering note: this is written as §8 to avoid renumbering §7 and every cross-reference to it.
Placing it *before* §7 while numbering it 8 is deliberate — the flag belongs next to the CLI
discussion, and renumbering would invalidate anchors elsewhere.

---

### `## 8. Model families (PottsMPNN, LASErMPNN)`

Beyond the stock ProteinMPNN/LigandMPNN path, aminx hosts two further families behind a
single flag. **Everything below was read out of the code or exercised against the CLI.**

**Flag position is the trap.** `--model-family` is declared on the *group* callbacks
(`cli.py:479` `@run_app.callback`, `:1089` `@spec_app.callback`), so it must come **before**
the verb. `aminx run score --help` does not list it at all, which is exactly how a user
concludes the flag does not exist:

```bash
# CORRECT -- flag precedes the verb
aminx run --model-family pottsmpnn score --output-kind energy --inputs 1ubq.pdb
aminx spec --model-family pottsmpnn emit-sample --inputs tests/data/1ubq.pdb

# WRONG -- not a recognised option here
aminx run score --model-family pottsmpnn --inputs 1ubq.pdb
```

`--output-kind`, by contrast, *is* a subcommand flag, so the ordering is mixed. Verified
end to end: the `spec` form above emits `"model_family": "pottsmpnn"`.

**Which purposes each family actually runs.** The family flag alone does not tell you which
model produced a number — read from `_HANDLED` and `mpnn_fallback_purposes` in both drivers:

| family | handles | declares a stock-MPNN fallback for |
| :-- | :-- | :-- |
| `pottsmpnn` | `score:energy`, `score:ddg`, `sample` | `jacobian`, `inspect`, `score:nll`, `score:logits` |
| `lasermpnn` | `sample`, `score:nll`, `score:logits`, `score:proofread_unconditional`, `score:proofread_conditional` | *(none — empty fallback set)* |

So `--model-family pottsmpnn --output-kind nll` returns **stock ProteinMPNN** scores by
design. That is correct behaviour and very easy to misread, so check the table before
attributing a number to Potts.

Allowed output kinds are enforced at spec construction by `_validate_output_kind` against
`_OUTPUT_KINDS_BY_FAMILY` (`specs.py:556-564`).

**Install extra.** LASErMPNN needs the `laser` extra because `prody` is not a core dependency
(`pyproject.toml:42-45`). PottsMPNN needs no extra. Without it you get a bare `prody`
`ImportError` that does not mention the family or the extra (debt #2483).

**Potts-specific emitter.** `aminx spec emit-potts` exists alongside the generic emitters.

**Known gaps, each with its debt id.** An unrecognised `--model-family` is currently accepted
end to end and silently runs stock ProteinMPNN — confirmed through the CLI:
`aminx spec --model-family lasermpn emit-sample ...` exits 0 and emits
`"model_family": "lasermpn"` with no error and no warning (#2484). Also open: eight unread
`Options` fields (#2435), three unimplemented Potts fields (#2433), inert `tied_positions`
(#2459), and four unported upstream features (#2067-#2070).

**Numerical expectations.** Parity work for these families runs tiered (dtype/shape, then
f64, then f32). f64 is used for parity assertions and validation only — **it is not a
production dtype**, because it costs GPU performance. Do not reach for `--double` or
`jax_enable_x64` to make a production run agree with a parity number.

Full write-up: `docs/MODEL_FAMILIES.md`.

---

## 3. Two additions to `## Gotchas`

```markdown
- **`--model-family` goes BEFORE the verb.** It is declared on the `run`/`spec` group
  callbacks, so `aminx run score --model-family pottsmpnn …` fails while
  `aminx run --model-family pottsmpnn score …` works — and `aminx run score --help` does not
  list it, which makes the flag look nonexistent. `--output-kind` *is* a subcommand flag, so
  the ordering is mixed.
- **A family flag does not guarantee that family ran.** `pottsmpnn` declares a stock-MPNN
  fallback for `jacobian`, `inspect`, `score:nll` and `score:logits`, so those return
  ProteinMPNN numbers by design. `lasermpnn` has an empty fallback set. Check §8's table
  before attributing a number to a family. An *unrecognised* family is worse: it is accepted
  silently and runs stock ProteinMPNN (#2484).
```

## 4. One addition to `## References`

```markdown
- Model families: `docs/MODEL_FAMILIES.md`; drivers under `src/aminx/families/potts_mpnn/`
  and `src/aminx/families/laser_mpnn/`; dispatch at `src/aminx/host/runner.py:71-117`
```

## 5. What this patch deliberately does NOT claim

- It does not mark either family "wired end-to-end" in the skill's "four inference paths at a
  glance" table (SKILL.md:25) or its runner-wiring note. The gate still has
  one red wave (`tests/port/test_laser_score.py`, a tier-3 f32 design decision that is the
  user's), so a blanket readiness claim would be premature.
- It does not document `--double` on `potts_refine`, which does not exist yet (#2417, Tier A4
  of the re-wave).
- It adds no Python-API examples for either family. The `_HANDLED` sets above are read from
  the drivers, but I have not exercised a family through the array API in this session, and
  the skill's existing examples are all verified ones. Adding unverified snippets would
  degrade it.
