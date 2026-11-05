---
name: aminx-porting-mpnn
description: Port a new MPNN-family model (an upstream PyTorch repo with released weights) into aminx to the family-parity standard - real oracle, sealed dumps, pre-registered vehicle, mutants, knob rows. Covers recon, spec, spikes, oracle, vehicle, gate wiring, review and the failure modes found porting Caliby, AntiBMPNN and FAMPNN.
when_to_use: 'Triggers: port a model, new family, new MPNN, parity vehicle, sealed dump, oracle dump, reference_surfaces, alias_map row, closure_edges slug, checkpoint_registry.'
---
# aminx-porting-mpnn

Developer workflow for adding a model family under `src/aminx/families/<family>/` and proving it matches upstream. Running an existing family is `using-aminx`. The standard comes from the 261104 P4 redo: the first attempt at Caliby, AntiBMPNN and FAMPNN had a clean arm that diffed aminx against aminx, randomly initialised weights, fake registry hashes, invented knobs and debts that were never filed. Every rule below is a defect that review found.

Read first: `.praxia/docs/plans/261030_family-parity-goal-prompt.md` and `.praxia/docs/specs/261030_family-redsox-parity-coverage.md` (both live on branch `wt/261030-family-parity-scope`, not yet on main). Copy structure from a finished family: `scripts/parity/{caliby,antibmpnn,fampnn}_parity_v2.*`, `tests/knob_semantics/test_<family>_knobs.py`, the family's rows in `tests/knob_gate/`.

## 1. Recon: read the code, not the README
- Clone upstream to `~/repos/<name>` (and `/home/solab/repos/` on titanix). Pin the commit sha.
- Establish from code: architecture relative to ProteinMPNN, checkpoints and their host, the CLI/YAML/argparse surface, inputs, outputs, and every source of randomness with file:line.
- Audit any existing aminx stub. Tag each assumption verified (read at file:line or probed) or unverified, and probe the unverified ones that would change the design.
- If aminx cannot load the real weights, stop and report. Do not substitute random initialisation.

## 2. Oracle: real upstream, sealed
- Import the unmodified upstream code with the real weights; run a small real fixture (`tests/data/1ubq.pdb` or an upstream example).
- Inject recorded draws at every stochastic step, cite the file:line of each injection point, and make the oracle raise on any RNG primitive that was not injected.
- Draw noise ONCE in float32 and reuse it for every precision. A dump that drew its decoding-order randn under `set_default_dtype(float64)` gave the f64 pass a different order than f32, which invalidated every cross-precision comparison.
- Write a sealed npz/pt with a pinned sha256. The loader must fail loudly when a stem is missing from the pin file; never `expected.get(stem, own_sha)`.
- Register the real weight sha256 in `tests/knob_gate/checkpoint_registry.json`, keyed `<upstream>/<path within>` (`scripts/parity/artifact_key.py`, add the repo to `_UPSTREAM_ROOTS`). No pattern-looking hashes.

## 3. Vehicle: aminx against the dump
- The clean arm diffs aminx against the sealed upstream dump, never against aminx. A vehicle must not accept a flag and ignore it (the first attempt did `del synthetic`); the v2 vehicles take sealed dumps instead of `--synthetic`. Stages: features, encoder, decoder/logits, one sampler step.
- Drive the real entry points (`host.runner.score` / `sample`, options-to-bias wiring), not only hand-composed components. A harness that composes features, encoder and decoder itself never exercises the product.
- Bands: f64 1e-7, f32 1e-4 by default. If f32 misses, decompose before choosing a band: upstream-f32 vs upstream-f64 and aminx-f32 vs upstream-f64, per stage. A band derived from the reference's own floor is acceptable; a band set over the observed value, or widened after a fail, is not. In 261104 the AntiBMPNN f32 gap was upstream's own conditioning (its f32 logits drift 2.8e-4 from its f64), and the FAMPNN 2e-6 f64 loop band had no principled value (the sealed oracle moves 1.49e-5 with exact tables).
- If a stage cannot be compared bitwise (no injection hook), say why in the sidecar and file a debt. Do not paper over it.

## 4. Mutants (negative controls)
- Aminx-side monkeypatches that must trip the diff. A mutant trips only on a FINITE over-band diff. NaN, inf, shape mismatch and exceptions are recorded separately and the crash count must be 0 (a crashed mutant had been counted as a trip). The reference implementation is `scripts/parity/caliby_parity_v2.py` (`n_crashed`, `nonfinite`); the AntiBMPNN and FAMPNN v2 vehicles predate the rule and are being brought up to it.
- Include a mutant that reintroduces each aminx fix you make to shared model code, and check it has power: a scale mutant is invisible when the scale is near 1, and a neighbour-count mutant only works if the fixture has enough residues.
- Give every arm its own mutants (a packer or secondary arm with none has no power evidence).
- Label weight-level controls (edit loaded parameters) apart from code-level mutants.

## 5. Pre-registration
- Write `scripts/parity/<family>_parity.bth.toml` with `[outcomes]` and commit it BEFORE the graded run. Never edit `[outcomes]` after a run. If the vehicle changes, write a v2 sidecar that discloses that dev runs saw numbers first and what changed.
- Run `bth run` (not `uv run bth`; on titanix `PATH=$HOME/.local/bin:$PATH`). Verify by the run record (`bth sql`: `status`, `outcome`, `sidecar_mode`, `git_dirty`), not the exit code. A plain verification run is not citable, and provenance text must not say "bth run" unless one exists.

## 6. Knobs, rows, tests
- Re-derive the `<Family>Knobs` dataclass in `tests/knob_gate/reference_surfaces.py` from upstream CLI/YAML/argparse with file:line cites. Delete invented knobs. Verify "declared but never read" claims by `rg` over upstream.
- Every aminx options field gets an `alias_map.toml` row; add `branch_manifest.toml` rows tagged `family=`; keep `ledgers/<family>.toml` empty until a real wave.
- Each live row has a passing `tests/knob_semantics/test_<family>_knobs.py::test_knob_semantics_*` that tests upstream semantics, not "returns finite".
- Unimplemented knobs use reason `deferred:<id>`. File the debt FIRST, use its real numeric id, and add it to `tests/knob_gate/DEFERRED_IDS.txt`. No placeholders.

## 7. Gate wiring (shared files: union, never overwrite)
- `closure_edges.toml`: slug for the vehicle, `soft` edge to the family package, `dynamic_ok` only for sites it truly reaches (a family importing `potts_mpnn.features` must declare the Potts driver's converter-by-path site).
- `tests/knob_gate/_coverage.py` `_WEIGHT_PREFIXES`, `test_closure_selftest.py` slug groups, `test_knob_superset.py` options unions. A prefix missing from `_WEIGHT_PREFIXES` lets a row with an empty weights map pass step 1c.
- Merge, do not rebase. Parallel family branches conflict by appending to the same gate files; keep both sides, then parse every TOML/JSON to validate.
- Edits to shared code (`options.py`, `io/upstream_pdb.py`, `potts_mpnn/featurize.py`, `model/cdr_bias.py`) stale every redsox row and get an Opus review. A full gate on merged main is mandatory before merge.

## 8. Porting defects that cost time
Read upstream for every constant: a PDB reader that renumbers residues or drops sidechains; the virtual-CB formula and which slot it fills; layer-norm and normalise eps; sinusoid or positional tables in the wrong dtype; a scale applied to one Potts factor instead of both (J scales as scale^2); a transposed right factor; neighbour count `k`; a frame forced to f32 by a dependency (openfold `Rigid`). `score()` with `ar_mask=None` is not upstream `conditional_probs` (debt #3266).

## 9. Process
- Delegate by role: Haiku/Sonnet recon and implementation, Sonnet review per component, Opus for shared code. Each implementer needs its own isolated worktree; an agent launched from inside another worktree cannot create a sibling and will report blocked.
- Verify agent reports yourself: `git ls-remote` for the head, the run record on the host. Subagents often skip `agent_result` and repeat hand-backs.
- Heavy runs go to titanix, never local. Use `/home/solab/scratch/<task>-<role>/`, sync with `git bundle`, and never write to the shared verify checkout. That checkout has LFS pointer files where weights belong: set `AMINX_WEIGHTS_DIR` to the HF snapshot (`~/.cache/huggingface/hub/models--maraxen--aminx/snapshots/<HF_REVISION>`); a `ZstdError` there means a pointer file, not corrupt weights.
- Done means: real weights and sealed dump, a graded `bth run` with `outcome=pass` and `git_dirty=false`, all mutants tripped with `n_crashed == 0`, knob rows green on titanix, no placeholder debts, and a review with no medium-or-higher finding.
