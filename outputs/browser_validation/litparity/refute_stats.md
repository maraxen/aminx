# Phase 3 refutation: statistical correctness

task_id: `260923_aminx-browser-validation`

**Lens.** Statistical correctness: run the code on synthetic/real ground truth and check
whether reported/derivable metrics match expected values.

**Assumption hunted.** I assumed a hidden defect exists in each of the four flagged
`reconcile.md` clauses and tried to prove it with a narrow, real-fixture or synthetic-ground-truth
probe per target, always pairing the measurement with a null/positive control so a "defect found"
verdict rests on a calibrated instrument, not a bare number. Default verdict on weak evidence is
DEVIATION/inconclusive, never parity — no target below needed that fallback: all four produced
clean, reproducible, large-magnitude effects.

**Environment (every probe).** `JAX_PLATFORMS=cpu`, `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
MKL_NUM_THREADS=4`, `uv run --frozen --extra dev --extra benchmark python <script>`, single
process, no pytest. Fixtures: 6MRR (L=68) and 5L33 (L=106), `$PROTEINMPNN_PATH/inputs/PDB_monomers/pdbs/`.
Scratch scripts lived under `$TMPDIR/probes/`, never in the repo; no tracked file was edited.

---

## Target 1 — `conditional_score_context_mask` (2-hop self-identity leak)

**Assumption.** aminx's default single-pass full-conditional score (`mode="score_conditional"`,
`ar_mask` left at its `build_inference_bundle` default, which resolves to `full_context_ar_mask`
= `1 - I`) lets residue *i*'s own true identity leak into its own predicted logits through a
2-hop decoder path, even though the direct self-edge was already patched.

**Method.** Full model (`proteinmpnn_v_48_020`, eqx weights), fixture 6MRR (L=68). For 15 sampled
designed residues, mutated **only** `s_i` (the one-hot token at position *i*) and re-scored with
the same 1-I mask, measuring `max|logits[i]_after - logits[i]_before|`.

**Controls.**
- Null control (identical input run twice): `max_abs = 0.0` exactly — the call path is
  deterministic (fixed `PRNGKey(0)`, dropout off at inference), so any nonzero delta below is a
  real effect of the mutation, not noise.
- Positive control: mutate a *different* residue `j != i` (legitimately visible to *i* under
  1-I) and measure the same delta at *i* — this calibrates the instrument against a known-real
  conditioning effect of the same shape and magnitude class.

**Evidence.**

| | max (nats) | mean (nats) |
|---|---|---|
| Self-leak (mutate `s_i`, read `logits[i]`) | 0.8386 | 0.2376 |
| Positive control (mutate `s_j`, read `logits[i]`) | 0.5924 | 0.1621 |
| Null control | 0.0 | — |

Full per-residue table in the probe output (`$TMPDIR/probes/probe1_self_leak.py`, run
2026-09-25). The self-leak magnitude is not a rounding artifact riding on top of a real signal —
it is **the same order of magnitude as the positive (legitimate-conditioning) control**, on 13/15
sampled residues within 2x of each other. A residue's own identity moves its own prediction by
as much as a genuinely-visible neighbour's identity does.

**Verdict: defect found, severity CORE.** `conditional_score_context_mask`, paths P05/P06 (both
core, in the validated harness's own default full-conditional scoring mode). Max effect 0.84 nats
≈ 8386x the 1e-4 nats bar.

**Recommendation.** No cheap wiring fix exists inside a single decoder pass (per `reconcile.md`
§4, "no compatible mode ... short of L external passes with per-idx masks"). Either (a) implement
the reference's true L-pass per-residue-last conditional estimand for any caller that needs
matching `p(s_i | s_-i, X)` values, or (b) prominently document that aminx's single-pass
`score_conditional` computes a *different, self-informed* quantity and is not a numerical stand-in
for the reference's `conditional_probs`/`single_aa_score(use_sequence=False)` estimand.

---

## Target 2 — `ligand_atom_context` (16 vs 25 wiring)

**Assumption.** `get_topology_for_checkpoint` parses `atom_context_num=25` for
`ligandmpnn_v_32_010_25`, but `load_model` never passes it through, so `ProteinFeaturesLigand`
keeps its class default of 16 (a static field, uncorrectable by weight loading). This should
change ligand-conditioned logits whenever a residue has >16 ligand atoms within reach.

**Method.** No real small fixture in this corpus carries a ligand (only 3HTN/6EHB do, both large);
used synthetic ground truth instead, as the lens explicitly allows. Real 5L33 backbone (L=106),
loaded shipped `ligandmpnn_v_32_010_25` (eqx). Confirmed the wired `atom_context_num` by direct
introspection, then built a synthetic ligand cloud of 25 atoms at 2..26 Å from residue 20's
virtual Cβ, broadcast identically to every residue's ligand slab. Compared the shipped model
(16) against a surgically patched copy (`atom_context_num=25`, same weights — the per-atom
Linear layers `y_nodes`/`y_edges`/`type_linear` don't depend on atom count, only which/how-many
atoms get selected does).

**Evidence.**
- `aminx_model.features.atom_context_num` introspected directly = **16** (confirms the static
  reading in `reconcile.md`/`clauses.json` by execution, not just by reading code).
- Null control (identical run twice, wired model): `max_abs = 0.0`.
- 16-vs-25 max abs delta: **3.956 nats**, mean abs delta **0.107 nats**, over **106/106**
  residues exceeding the 1e-4 bar (`$TMPDIR/probes/probe2_ligand_atom_context.py`, run
  2026-09-25).

**Verdict: defect found, severity CORE.** `ligand_atom_context`, path P11 (core). Max effect
3.96 nats ≈ 39,562x the bar. This is on synthetic ligand geometry designed to expose the count
difference (every residue given a ligand within reach) — the direction and mechanism (16 vs 25
static field, uncorrectable by weight conversion) is the same static fact `reconcile.md` already
identified from code; this run adds a magnitude bound on real-checkpoint weights.

**Recommendation.** Wire `topology["atom_context_num"]` through in `load_model` (mirrors the k=48
wiring that already works); this is mechanical, `atom_context_num` is `eqx.field(static=True)` so
no weight remapping is needed, only passing the parsed int at construction time.

---

## Target 3 — `membrane_label_encoding` (`physics_projection` random-init bias)

**Assumption.** The reference membrane node-embedding `Linear(3, H)` has no bias; aminx's shipped
`.eqx.zst` keeps a random-initialisation bias because weight conversion only overwrites a bias
when the `.pt` tensor has one. Every membrane-model node embedding should be shifted by a fixed
random vector before LayerNorm, on every residue.

**Method.** Fixture 6MRR (L=68), `per_residue_label_membrane_mpnn_v_48_020` (eqx). Varied labels
(cycled 0/1/2 across residues, not the degenerate all-zero case). Compared shipped model logits
against a copy with `encoder.physics_projection.bias` zeroed via `eqx.tree_at` leaf surgery (no
tracked file touched, no reconversion).

**Evidence.**
- `physics_projection.bias` max abs = **0.5714** (shape `(128,)`), confirming the bias is live
  and non-trivial, not accidentally zero-initialized.
- Null control (identical run twice, shipped model): `max_abs = 0.0`.
- Shipped-vs-bias-zeroed max abs delta: **5.478 nats**, mean abs delta **0.808 nats**
  (`$TMPDIR/probes/probe3_membrane_bias.py`, run 2026-09-25). Every one of the 68 residues shows
  a nonzero (>1e-3 nats) shift; the largest individual residues move by 4–5.5 nats.

**Verdict: defect found, severity CORE.** `membrane_label_encoding`, paths P13/P01 (both core).
Max effect 5.48 nats ≈ 54,779x the bar — consistent with, and now precisely quantifying, the
"1–5 nat membrane divergence" already recorded as unfixed in project memory
(`project_membrane-physics-projection-bias-bug`, 2026-09-25, encoder.py:465).

**Recommendation.** Mechanical fix already named in `reconcile.md`: either construct
`physics_projection` with `use_bias=False`, or zero the bias post-conversion for every shipped
membrane checkpoint (`per_residue_label_membrane_mpnn_v_48_020.eqx.zst` and
`global_label_membrane_mpnn_v_48_020.eqx.zst`), then re-verify against the reference on real
membrane-labeled inputs.

---

## Target 4 — `sample_log_probs_output` (fixed positions not zeroed)

**Assumption.** The reference sampler returns all-zero `log_probs` rows at fixed positions
(masked by `chain_mask`); aminx's `sample_autoregressive` kernel's `step_fn` computes
`avg_stored` (the stored, bias-free log-softmax) from the real network output for **every**
group regardless of whether that group is fixed — only the sampled *token* gets overridden by
`fixed_tokens`, never the stored logits row.

**Method.** Fixture 6MRR (L=68), full model (eqx). Fixed every third residue (`fixed_mask`,
23/68 positions) to its true token, sampled once (N→C order, single state,
`mode="sample_ar"`), read `SampleResult.logits`.

**Evidence.**
- `fixed_tokens_honored_in_sequence = True` — the returned *sequence* correctly keeps the fixed
  tokens (that mechanism is not in question).
- Fixed-position logits rows: max abs **5.736 nats**, mean abs **5.534 nats**.
- Designed-position logits rows (built-in control): max abs 5.759 nats, mean abs 5.560 nats —
  statistically indistinguishable from the fixed rows.

The reference's expected value at a fixed row is exactly 0; aminx's actual value is
indistinguishable in magnitude from a genuinely-sampled designed position. This is not a partial
leak — the fixed/designed distinction the reference makes is entirely absent from what aminx
returns as `logits`.

**Verdict: defect found, severity CORE.** `sample_log_probs_output`, paths P07/P09 (both core per
`clauses.json`). Max effect 5.74 nats ≈ 57,362x the bar. Exposure is narrower than target 3
(only matters to a caller that reads returned sampling log-probs at fixed positions — e.g. a
`get_score`/confidence recomputation over a sampled sequence with any residues fixed — untied,
fully-designed positions are unaffected, matching `reconcile.md`'s own scoping), but the
magnitude where it does apply is not "minor": a downstream confidence score computed over any
sequence with fixed residues will be silently wrong by nats per fixed position.

**Recommendation.** In `decode/autoregressive.py`'s `step_fn`, zero `avg_stored` (or the returned
logits row) at positions where `is_group_fixed` is true, matching the reference's `chain_mask`
masking; alternatively, mask fixed rows to zero in `sample_autoregressive.kernel`'s final
`SampleResult.logits` before returning.

---

## Cross-target notes

- All four null controls returned exactly `0.0` — the harness itself introduces no
  nondeterminism (fixed `PRNGKey(0)`, dropout off), so every nonzero number above is directly
  attributable to the manipulation described, not to run-to-run noise.
- Targets 1 and 2 both used a genuine positive/known-effect control (mutate a legitimately
  visible position, or introspect the wired static field directly) before trusting the headline
  number, per the BATHOS discipline of pairing every "must pass" check with something that can
  fail.
- Effects are 3–5 orders of magnitude above the 1e-4 nats bar in every case; none of these are
  edge-case-only or hyperparameter-only findings, so none get `minor`/`accepted`.

```json
{"defects": [
  {"id": "conditional_score_context_mask", "paths": ["P05", "P06"], "severity": "core", "evidence": "mutating only s_i changes logits[i] by up to 0.839 nats (mean 0.238), same order as a legitimate neighbour-perturbation control (max 0.592); bar is 1e-4 nats", "runnable": true},
  {"id": "ligand_atom_context", "paths": ["P11"], "severity": "core", "evidence": "wired atom_context_num=16 (introspected) vs checkpoint-intended 25 changes ligand-conditioned logits by up to 3.956 nats (mean 0.107) on 106/106 residues; bar is 1e-4 nats", "runnable": true},
  {"id": "membrane_label_encoding", "paths": ["P13", "P01"], "severity": "core", "evidence": "physics_projection bias max-abs 0.571 (random init survives conversion); zeroing it changes logits by up to 5.478 nats (mean 0.808); bar is 1e-4 nats", "runnable": true},
  {"id": "sample_log_probs_output", "paths": ["P07", "P09"], "severity": "core", "evidence": "fixed-position sample logits are real log-softmax rows (mean abs 5.534 nats), indistinguishable from designed positions (mean abs 5.560 nats), vs reference's exact 0; bar is 1e-4 nats", "runnable": true}
]}
```
