---
title: PottsMPNN + LASErMPNN as xtrax-composed model families on the central runner
description: Port KeatingLab PottsMPNN and polizzilab LASErMPNN into aminx as xtrax-composed FamilyDrivers dispatched from aminx.host.runner, with redsox knob-superset, xtrax-tier parity, and bathos-preregistered gates
task_id: 260929_potts-laser-xtrax-compose
status: converged-r16-a10
created: 260929
amends: decisions/260605_potts-parallel-not-stageset.md (scope-narrowing, see §3)
adversarial_log: audits/260929_potts-laser-spec-adversarial-log.md
---

# PottsMPNN + LASErMPNN as xtrax-composed model families

Revision history:
- r0 (8ea7604e) → r1 (3ee01410): FamilyDriver seam; removed `plan_builder`, `post_decode` slot,
  `InterpolatedTieFuse`, LASEr `sc_coords` carry.
- r1 → r2: **all PottsMPNN sampling in the driver** (`PottsARDecode` reusing aminx `DecoderLayer`s;
  removed `mpnn_randn`, hybrid routing predicate, `emits_potts`); driver outputs via xtrax
  `ZarrStagingSink` (removed staging-sink reuse claim); runner golden gate; concrete L-DRV lint;
  `nodes`=Gibbs; padding/K convention; LASEr alphabet boundary; LASEr step order; oracle shims;
  distributional protocol with pilot; per-sidecar negative controls; junit-backed redsox harness;
  quirk + divergence tables; explicit fallback raise; family temperature default.
- r2 → r3: T0.0 sink `run_id` prerequisite (existing aminx Zarr paths TypeError at pinned xtrax);
  normative row kernel + f64-capable `DecoderLayer` accumulation; `edge_valid` as decoder
  `attention_mask`; goldens split T0.5a/T0.5b minus provenance attrs; UNSET resolution hook;
  device-side tie-group order; collision-free Zarr keys; corrected pilot rule + pinned structures +
  shim check; inverse-CDF clamp; L-DRV allowances; conftest outcome plugin; spec_json Options.
- r3 → r4: tied `rank_flat` from flattened group order (static `(L_pad,M_max)` groups); LASEr χ
  offset head + χ-angle gate; pre-refine energy ranking + `refined_sequence`; PottsMPNN host
  featurizer A0 with upstream gap rows and `present`/`pad_valid` mask table (retires r3
  `edge_valid`); per-wave redsox gate loop; tied last-member select; `nodes` fw recompute; sealed
  DecLayer fixture.
- r4 → r5: AR masks use `present` only (refine/`nodes` keep `present·chain_M_pos`) + fixed-position
  fixtures/controls; `knn_boundary_tie` detection/exclusion (stock features untouched); port
  self-test wave; order-injection padding + tied oracle `randn` injection; gapped `sequences_to_score`
  alignment.
- r5 → r6: normative source-of-truth + branch-coverage clause (§0); LASEr scalar-only proofread
  dropout + `proofread_dropout`; exact proofread reduction + binding controls; `repack_all` knob;
  `num_adjacent_residues_to_drop` removed (inert); §5.4a score conditioning; tied `_2` channels + gate;
  Potts tied masked-member tail; LASEr carry init + `ignore_chain_mask_zeros` no-op; LASEr self-edge;
  tied fs-temp divergence; `strict_load` polarity.
- r6 → r7: branch-coverage enforcement (coverage vehicles, `branch_manifest.toml`, mutant hook,
  `test_branch_coverage`, stage→vehicle table); LASEr `T_eff` fs-temp resolution; `skip_calc`
  (`optimize_pdb`/`optimize_fasta`) stage graph + schema + divergences; LASEr bias step 1b; AR/opt T
  floors; per-structure `L_total` slicing; §5.4a contact/chain_mask sources.
- r7 → r8: clean/mutant outcome separation; sidecar coverage via gate-run bathos records
  (`git_hash`, `sidecar_sha256`, `branch_controls.json` in `output_paths`); order-generation stage +
  vehicle; refine fresh-order key; `skip_calc` mirrors upstream N=1/mode-none; typed `vehicle`;
  `check_branch_coverage` + separate selftest manifest; etab K slicing.
- r8 → r9: sidecar coverage simplified — gate never runs sidecars; committed `sidecar_ledger.toml`
  validated via real bathos columns (ancestor + diff-scope freshness, argv mutant set,
  `--output-paths` controls file); gate writes outside tracked tree (`AMINX_PORT_AUDITS_PATH`,
  `/outputs/` gitignored); full-verdict coverage self-test; per-wave mutant reruns; LASEr pad rows.
- r9 → r10: Python gate driver (bathos-resolvable sidecar); cool-tier parquet resolver, no in-gate
  compact; three-label vehicle outcomes incl. inconclusive + mutant `error`; broadened freshness scope
  + weights SHA check; AssertionError-only kill; schemas to T0.3; selftest (vi)/(vii) + full (v);
  Z1 operator re-run of stale sidecars at final tree.
- r10 → r11 (**converged**: round-11 challenger found 0 BLOCKER / 0 MAJOR under the strict rubric;
  its 5 MINORs applied by the orchestrator): makereport hookwrapper for `exc_type` + kill precedence;
  outcome hooks keyed on `AMINX_PORT_WAVE`, step 2 reads `AMINX_REDSOX_OUTCOMES_READ`; per-arm
  subprocess for sidecar mutants; required artifact `sha256` in registry; freshness scope covers
  `scripts/parity/**`, `scripts/recapture/**`.
- r12 → r13 (**rebased onto origin/main 79956bd0**, user 260929: the sprint branch had been cut
  from local `dogfood/xtrax-probing-stage2`; now `wt/260929-potts-laser-main`, PR supersedes #164):
  (a) **T0.0 done on main** by `9e6c340a` ("migrate SinkSpec construction across xtrax a7's required
  run_id") — no sprint task; T0.5b no longer blocked externally. (b) **D2 superseded:** main #160
  (`79956bd0`) already fixes #2017 via `utils/decoding_order.random_design_order` (fixed-first,
  group-level uniform, explicit total order) as the default sampling order; the D2 commit is not
  carried. (c) **xtrax pin is now `xtrax[io,export]==0.4.0a10` (PyPI)**, not git `56a9f551`: every
  xtrax-derived fact in this spec (SinkSpec/ZarrStagingSink/derive_sink_spec signatures, port/
  contract, stages API) is **re-verified against 0.4.0a10 by probe T0.2b** before T0.5; prose that
  disagrees is fixed per §0. (d) main's #160 also derives the MPNN sampler's wave from the decoding
  order (`WaveScheduleBundle.from_decoding_order`), so §1 rows describing the MPNN AR path as
  "order from a fixed N→C wave schedule" are historical; U-a (all PottsMPNN sampling in the driver)
  is unaffected. (e) Source anchors (file:line) into `src/aminx/**` were taken at the dogfood base
  and may have drifted; per §0 the code at HEAD wins.
- r15 → r16 (A2 execution; items (1) and (4) **post-hoc, made after observing a failure**): (1)
  `pottsmpnn_full` f32 log-prob tolerance gains `rtol=1e-4` next to `atol=1e-4` (user-approved
  260929): one element of 1512
  (soluble_30 3dkm, row 71, value −3.132) missed pure `atol=1e-4` by 1.7e-5 (rel err 3.7e-5), while
  the same code passes f64 at atol 1e-8 on every fixture incl. gap fixtures. (2) §4.1a correction:
  non-present rows inside a present row's kNN set DO reach the encoder/decoder through
  present→non-present edges; upstream feeds the Cα–Cα RBF `D_adjust` (= row `D_max` for invalid
  pairs) and the other 24 RBFs raw zeroed-coordinate distances. The Potts path
  (`families/potts_mpnn/features.py`) reproduces this; stock MPNN features are unchanged. (3) The A1
  f64 oracles were regenerated with true-f64 RBF centres (`rbf_follows_input_dtype` shim, upstream
  `potts_mpnn_utils.py:1156`). (4) aminx `RBF_CENTERS` is materialised at import, so it is float32
  whenever x64 is enabled after import; RBFs now build their centres in the input dtype at call time
  (`utils/radial_basis.rbf_centers`; f32 byte-identical, runner goldens 18/18 exact).
- r14 → r15 (A1/A3 execution; **post-hoc, made after observing a failure**): (1) `potts_energy` f32
  tolerance: a pure `rtol=1e-5` on a total that is a sum of O(L·K) terms is ill-posed when the terms
  cancel. Observed on the A3 wave: `vanilla_30 two_chain` random seq 3, total 0.01207, abs err
  1.9e-7, rel err 1.56e-5, while the same port passes f64 at rtol=1e-10 on every cell. f32 is now the
  summation backward-error bound `|got−ref| ≤ 1e-5·Σ|terms|` with `Σ|terms| = potts_energy(|etab|,
  seq)`; f64 unchanged. (2) Refine waves omit X (`constant[20]=1`): upstream refine builds 20
  candidates but samples over 21 letters and raises IndexError when X is drawn (debt #2260); A5 must
  define X handling. (3) A1 f64 dumps shim `potts_mpnn_utils.py:940` (`d_onehot.float()` → weight
  dtype). (4) A-wave parity tests run as plain tests under `tests/families/potts_mpnn/` until each
  wave has a tier target + manifest + reference `algo.py` (harness wiring task).
- r13 → r14 (T0.2b xtrax 0.4.0a10 re-verify; evidence
  `research/260929_potts-laser-t02b-xtrax-a10-reverify.md`): every xtrax-derived fact holds on
  0.4.0a10 (SinkSpec/`derive_sink_spec`/ZarrStagingSink line anchors identical; `RunSpec.run_id`
  static; AxisSpec/BatchPlanner/AxisBoundary/Fuse/`stages._callback` unchanged for our use; `port/`
  unchanged since `35c5100`). **Changed/false and fixed:** (1) T0.0 as merged does NOT add
  `sink_ids.sink_spec_for`/`spec_run_id`; it calls `xtrax.run.derive_sink_spec` directly and
  `RunSpec.run_id` stays unset, so every sink gets a fresh unlinked id and re-running into an
  existing output dir raises xtrax's own ValueError; §2.1 "Output channel" and the T0.0 row now say
  so; (2) `DesignZarrWriter` defaults `run_id` to `new_run_id()` (not a path hash); (3) only four of
  five sites were migrated: `scripts/analysis/jacobian_profile.py:187` still TypeErrors (non-gating
  follow-up); (4) MPNN AR now honours `decoding_order_fn` via `with_decoding_order` (§1 row 164
  historical); (5) runner entry-point / streaming / model_family anchors updated.
- r11 → r12 (T0.2 corrections; prose fixed to match pinned upstream per the §0 normative clause;
  evidence `research/260929_potts-laser-t02-probe-report.md`): §5.4a order tiers (upstream inference
  never sets `extra_atom_contact_mask`, tier 2 empty); LASEr stored `seq_logits` are post-min-p (§5.3
  1b); `fs_calc_*`/`fs_no_calc_burial` inert → `no_op` rows, first-shell rule = heavy-atom contact
  (§5.5/§6.1/§6.4); LASEr `bb_noise` ≠ Potts `noise` (§6.3); LASEr `temperature==0` rule (§2.3/§5.4);
  `disabled_residues` default provenance (§6.1/§6.3); dead tied-CLI flags → `no_op` (§6.3/§6.4);
  tied-refine quirks + `ZeroDivisionError` divergence (§6.5/§6.5b); proofread `resindex`↔row
  divergence (§5.4/§6.5b); `proteinmpnn_compatible` files are ProteinMPNN-only, never Potts-loadable
  (§4.1); LASEr fixtures from upstream Zenodo dataset acquired in B1(a) (§5.5/§7.3/§8); anchor fixes
  (§1, §4.3, §5.2); §10 open items resolved.

## 0. Goal, non-goals, assumed decisions

**Goal.** Through the one central runner (same CLI, same `RunSpecification`, same
`host/runner.sample`/`score`/`jacobian`/`inspect` entry points, same xtrax tiling/sinks):

```bash
aminx run --model-family pottsmpnn --checkpoint-id pottsmpnn_vanilla_20 sample --inputs x.pdb --potts-options-json opts.json
aminx run --model-family pottsmpnn --checkpoint-id pottsmpnn_vanilla_20 score  --inputs x.pdb --output-kind ddg --potts-options-json opts.json
aminx run --model-family lasermpnn --checkpoint-id lasermpnn_0p1A_nothing_heldout sample --inputs lig.pdb --laser-options-json opts.json
```

every input knob of the pinned upstream implementations is reachable from `RunSpecification`
(or is a justified exclusion/divergence row), and numerical/behavioural parity against pinned
upstream PyTorch is demonstrated by pre-registered bathos runs over an xtrax-contract tier ladder.

**"xtrax-composed" (normative, enforced by L-DRV §2.4).** Per-structure computation is `eqx.Module`
stages; batch/sample/temperature/candidate/partition/proofread-order/dropout-seed loops are xtrax
axes (`AxisSpec` → `BatchPlanner` → `make_axis_dispatch_via_xtrax`); cross-axis reductions are
`AxisBoundary` `Fuse`s; host export is xtrax `ZarrStagingSink`. Genuinely sequential dependencies
(AR steps, tied members, refine sweeps) are `lax.scan`/`while_loop` at allowlisted sites.

**Non-goals (v1)**, each a recorded exclusion row (§6.4), never silent: training; LASEr
`entropy_decoder`; PottsMPNN `mutation_search.py`; CA-only PottsMPNN; MSA vocab-22 checkpoints;
plotting; migrating TRW `aminx.potts.PottsModel`; LASEr partial-charge entry point; portable-JSON v3;
gradient (T4) consumers.

**Assumed decisions (defaults applied; user may override — log U1–U5, U-a–U-c).**
- U1 (**user-confirmed 260929**, "own driver, but still xtrax composed"): LASEr is a FamilyDriver, xtrax-composed per §0/L-DRV; MPNN contracts (`StageSet`, `EncoderOutput`, `ModelProtocol`,
  `DecodeMode`, `make_inference_plan`, `AutoregressiveDecode`, `kernel_dispatch`) unchanged.
- U2/U-b: **bit-parity** for all numerical and sampling semantics incl. quirks (§6.5); pure I/O
  plumbing bugs are fixed and recorded as divergences (§6.5b). Family defaults follow upstream.
- U3: MSA vocab-22 deferred. U4: local harness over redsox library `check_superset` (§6.6).
- U5: LASEr proofreading + two-structure tied in v1.
- U-a: all PottsMPNN sampling via the driver.
- U-c: LASEr default temperature `None` (argmax) via a family-resolved default (§2.3).
- C3-04(b): stock MPNN decoder unchanged; PottsMPNN nll/logits fallback parity asserted only on
  unpadded runs; pre-existing padded-neighbour behaviour recorded (§6.5b).
- C3-02: sink `run_id` fix is prerequisite task T0.0 in this epic; deterministic spec-hash id.

**Semantics source of truth (normative).** The behaviour of every ported stage is defined by the
pinned upstream code (PottsMPNN @ `0cb0a58`, LASErMPNN @ `e70f2c6d`) together with the T0.2 verbatim
transcriptions committed into each stage's docstring. Spec prose that describes upstream behaviour is
informative: a summary for navigation, not a second definition. Where prose and pinned upstream
disagree, upstream wins. The disagreement is a spec bug, fixed by amending the prose in the same PR;
it is never an implementation choice and may not be resolved by following the prose. The only
deliberate departures are rows in §6.5b (divergences), §6.4 (exclusions), and aminx-defined
conditioning where upstream has no caller (e.g. §5.4a); each names its upstream anchor.
**Branch coverage (normative).** Coverage vehicles are tier waves (§7.2), bathos sidecars (§7.3),
`test_knob_semantics_*` tests and the A0/B0 gates; one rule governs all.
`tests/port/branch_manifest.toml` holds one `[[branch]] {id, stage, vehicle, fixture, mutant}` row (`vehicle` =
`{kind="pytest", nodeids=[…]}` or `{kind="sidecar", slug="<name>"}`) per
documented upstream branch of every stage in the §8 stage→vehicle table: fixed/designed,
gap/`present==0`, tied (incl. masked member, mixed fixed/designed groups), `ignore_chain_mask_zeros`,
`repack_all`/`repack_only`, ligand-absent/`ignore_ligand`, argmax vs sampled, NaN input χ,
`skip_calc` (§4.3), and every T0.2-enumerated conditional (T0.2 emits `conditional_ids.txt`).
`mutant` names a function in `tests/port/mutants/<stage_slug>.py` returning a context manager that
monkeypatches the named port function (or the static branch flag it reads) to take the wrong side.
Pytest vehicles (incl. A0/B0 gates) run clean and once per row with `AMINX_PORT_MUTANT=<id>` (root
`tests/conftest.py` enters it at `pytest_sessionstart`; outcome records carry `mutant`, null when
clean). Sidecar vehicles are never run by the gate: their owning task runs them (§6.6 step 1c) with
`--mutants <comma-separated ids>` (each mutant a negative-control arm) and records the run in the
committed `tests/knob_gate/sidecar_ledger.toml` (`[sidecar.<slug>] bth_run_id = "<id>"`).
`tests/knob_gate/test_branch_coverage.py` (§6.6 step 2) asserts: every `conditional_ids.txt` id and every
§8 stage appears in a row; every clean run passed; every mutant run failed (pytest: ≥1 vehicle id
failed under it; sidecar: the ledger record validates per §6.6 step 1c). Missing fixture, missing
mutant run, passing mutant, or missing/stale/invalid ledger record → `instrument_invalid`.
Transcription slips are therefore caught by gates, not by review. Ownership: each stage task (A0–B7)
adds its own `[[branch]]` rows, `tests/port/mutants/<stage_slug>.py` entries and, for sidecar
vehicles, its `sidecar_ledger.toml` entry, in the same PR as the stage; T0.4 owns only schemas and
harness.

## 1. Recon evidence base

| Fact | Anchor |
|---|---|
| Runner entry points | `cli.py` `run` callback (`:448-461`, `--model-family` before subcommand) → `run_sample`/`run_score`/`run_jacobian`/`run_inspect` → `host/runner.py` `sample/score/jacobian/inspect` |
| score bypasses InferencePlan unless averaging | `host/runner.py:498-506` (`make_score_fn`), `one_hot(seq,21)` `:626`, `output_h5_path` NotImplemented `:489-491`, NLL result `:661-670`; `sequences_to_score` required `run/specs.py:556-561`, `cli.py:816` |
| MPNN AR ignores `decoding_order_fn`; order from wave schedule (**historical, r14: main #160 now draws a per-sample order in `host/kernel_dispatch.py:253-268` and applies it via `with_decoding_order`, `inference/bundle_builder.py:381-413` -> `WaveScheduleBundle.from_decoding_order`, honouring `decoding_order_fn`**) | `inference/decode/autoregressive.py:191-420`; `inference/bundle_builder.py:76,187-188,237-243`; `host/kernel_dispatch.py:222-258`; `decoding_order_fn` callable `run/specs.py:260`, `utils/decoding_order.py:21-24` |
| MPNN tied fuse = plain mean | `autoregressive.py:326-331` |
| Staging sink carries only seq/logits | `host/output_sinks.py:51-153`; `types/protocols.py:102-120` |
| xtrax generic array sink | `xtrax/run/zarr_sink.py:254-313` `ZarrStagingSink.stage(key, attrs, **arrays)` + `take(key)`; provenance attrs `_CORE_PROVENANCE_FIELDS` `:34` stamped on root/groups; **at pinned xtrax 0.4.0a10 (re-verified r14; anchors identical to the former 56a9f551) `SinkSpec.run_id` is required (`run/sink.py:26`); main `9e6c340a` migrated the runner-path sites to `derive_sink_spec` (`host/streaming.py:88-91`, `host/runner.py:1336-1341`, `sampling/multistate_poe.py:689`; `io/designs.py:107-109` passes `run_id` explicitly, default `new_run_id()`); `scripts/analysis/jacobian_profile.py:187` still omits it (TypeError; non-gating follow-up)**; `derive_sink_spec` `run/sink.py:61-105` (precedence explicit `run_id` > `run_spec.run_id` > `new_run_id()`); sink construction raises `ValueError` if the root already carries a different `run_id` (`zarr_sink.py:214-222`); with `flush_every=1` every `stage` drains at once (`:296`) so `take` sees nothing; `RunSpec.run_id` is static (`run/spec.py:22`) |
| Structure ids positional | `host/_sampling_helper.py:36-61`; prep forces inference mode `host/prep.py:200` |
| `DecoderLayer` reusable per row | `model/decoder.py:281` |
| Family Literal + consumers | `run/specs.py:236`; derivation `:476-490`; `_sampling_helper.py:255`, `prep.py:47-49`, `run_spec_portable_json.py:12-15,153`, `multistate_poe.py:609`, `streaming.py:84`, `_sampling_grid_lineage.py:94,111`, `campaign.py:95`, `run/spec.py:321` |
| `load_model` id normalisation | `io/weights.py:386-392`; `VOCAB_SIZE` `:453` |
| aminx sampling knobs | `temperature` default 0.1 non-Optional (`run/specs.py:574`), `bias`, `fixed_positions`, `fixed_tokens`, `num_samples`; `fixed_mask`, `tied_positions`, `noise`, `random_seed`; no omit |
| Encoder output | `types/bundles.py:457-486` `EncoderOutput(node_features, edge_features (S,L,K,D), neighbor_indices, mask)` |
| No CA featurizer / no B-factors | `rg` 0 hits each |
| TRW Potts model (distinct) | `potts/model.py:71`; ADR 260605; `tests/lint/test_potts_import_boundary.py` |
| RS-6b lint pattern | `tests/lint/test_rs6b_flat_field_gate.py:54-80`; `.ast-grep/rules/rs6b-host-flat-field-ban.yml` |
| Upstream PottsMPNN | KeatingLab/PottsMPNN @ `0cb0a58`; class `potts_mpnn_utils.py:1225-1261` (ProteinMPNN layers + `etab_out`); forward `:1267-1290`; K=min(48,L) `:1150`; CLI decode path `decoder :1415-1488` (order `:1419-1421`, masks `:1422-1428`, masked rows `:1448`, fixed select `:1483`) and `tied_decoder :1599-1687` (body `:1636-1685`, groups `:1606-1614`, last-member leak `:1665-1684`; `tied_sample :1490-1597` is the near-identical unused copy); called from `sample_seqs.py:226,235` |
| PottsMPNN energy/refine | `run_utils.get_etab :791-835`; `etab_utils.functionalize_etab` denom 4 excl. self `:177-178,253-271`; `calc_eners :299-309`; alphabets `potts_mpnn_utils.py:69` vs `etab_utils.py:362-366`; `optimize_sequence :113-198` (binding `:135-148`, accumulate `:148,176`, cap `:116-120`), `nodes :180-270`; T floor `sample_seqs.py:38-39`; energy sort `:268-287`; refine order keying `:274-279,321-345`, `run_utils.py:110-111`; input list format `sample_seqs.py:81-94`; `optimize_fasta` `:133-141`; `strict=False` `:60` after xavier `potts_mpnn_utils.py:1263-1265` |
| Upstream LASErMPNN | polizzilab/LASErMPNN @ `e70f2c6d765416f7e29d51bfd6d4e08496438878` (MIT); `utils/model.py:107`; alphabet `utils/constants.py:34`; `tied_sample :460-627` (λ mix `:626`, T None→1e-6 `:617-618`, disabled mask `:606`); `sample :727-941` (carry `:773`, prior visibility `:815-818,916-917`, step `:833-865`, χ GVP `:931-934`); rotamers post-sampling `run_inference.py:747-750`; argparse `:773-792`; `sample_model` `:511-517` |
| redsox | `checkers/superset.py:27-60` (`alias_map: dict[str,str]` `:31,43`); CLI `target_dataclasses[0]` only `cli/main.py:120-121,161-165,205-207`; U1 presence-only `reachability.py:18-40` |
| xtrax port contract | `xtrax/port/port_target.toml`, `port/tests/conftest.py` (`:64-71` target, `:103-114` manifest, `:159-171` oracle import, `:214,232` timeout, `:277-286` emit), `test_parity_safe_map.py:53` (global x64); aminx pins wheel `pyproject.toml:26` → `port/` not importable |
| aminx parity/bathos | `parity_targeted` tests `tests/parity/test_full_model_parity.py:273`, `tests/parity/test_packer_parity.py:120`, `tests/model/test_ligandmpnn_equivalence.py:348` (none via runner); `parity_heavy` deselected `pyproject.toml:229`; `.bth.toml` slug `aminx` |

## 2. Architecture

### 2.1 FamilyDriver seam

```python
# src/aminx/host/family_driver.py
class FamilyDriver(Protocol):
  name: str
  options_type: type
  mpnn_fallback_purposes: frozenset[str]   # e.g. {"jacobian","inspect","score:nll","score:logits"}
  def handles(self, spec, purpose: str) -> bool
  def load(self, spec) -> eqx.Module                        # never applies inference_mode
  def mpnn_core(self, model) -> ModelProtocol | None        # Potts: model.mpnn ; LASEr: None
  def batches(self, spec) -> Iterator[FamilyBatch]
  def axes(self, spec, purpose, batch) -> list[AxisSpec]
  def stages(self, spec, purpose, model) -> FamilyStages
  def result_schema(self, spec, purpose) -> Mapping[str, SinkArraySpec]   # depends on purpose + output_kind
class FamilyBatch(NamedTuple):
  input_indices: tuple[int, ...]           # positions in spec.inputs; strictly increasing across the stream
  arrays: PyTree                           # fixed-shape padded device inputs
  skipped: tuple[tuple[int, str], ...]     # (input_index, reason)
class SinkArraySpec(NamedTuple):
  dims: tuple[str, ...]; dtype: str; attrs: Mapping[str, str | int | float]
FAMILY_DRIVERS: Registry[FamilyDriver]     # aminx.registry.Registry
```

**Dispatch** (first statement after spec sync in `runner.sample/score/jacobian/inspect`; purpose key
is `"score:<output_kind>"` for score):
```python
if (d := FAMILY_DRIVERS.get(spec.model_family)) is not None:
  if d.handles(spec, purpose): return run_family_driver(d, spec, purpose)
  if purpose not in d.mpnn_fallback_purposes: raise ValueError(f"{d.name} does not support {purpose} in v1")
  # fallback: prep loads d.mpnn_core(d.load(spec)); None -> ValueError; MPNN code path otherwise unchanged
```
`PottsMPNNDriver`: `handles` True for `sample` (every configuration) and `score:energy`,
`score:ddg`; `mpnn_fallback_purposes={"jacobian","inspect","score:nll","score:logits"}` run on the
embedded stock `Aminx`. `LaserDriver`: `handles` True for `sample`, `score:{nll,logits,
proofread_unconditional,proofread_conditional}`; `mpnn_fallback_purposes=∅`. Setting
`decoding_order_fn` with a driver family → `ValueError`.

**Output channel (v1: no `io_callback` in drivers).** Each jitted dispatch over one `FamilyBatch` ×
one `samples` chunk **returns** `dict[str, Array]` whose keys/dims equal `result_schema(spec,
purpose)` (mismatch → `RuntimeError`). Per-position dims are named `L_total`; `run_family_driver`
slices each structure's arrays to its own `L_total` (host, from `FamilyBatch` metadata) before
`sink.stage` or in-memory assembly; pad rows are never staged; in-memory results are per structure
(chunks concatenated within a structure only). `test_family_sink_ragged_L`: two structures of
different `L_total` in one batch → per-group shapes equal each `L_total` and values equal
single-structure runs. `run_family_driver` then: (i) if `output_h5_path` set →
on one `sink = ZarrStagingSink(derive_sink_spec(spec.run_spec, output_dir=Path(spec.run_spec.io.output_h5_path), format="zarr", flush_every=1))`
(`xtrax.run.derive_sink_spec`, exactly as main's T0.0 does at `streaming.py:88-91`; `run_spec.run_id`
is unset so the id is fresh per call and re-running into an existing dir raises xtrax's ValueError,
which drivers let propagate; same root as `streaming.py:88`, not `RunSpecification.output_dir`):
`sink.stage((f"structure_{input_index}", str(chunk_start)), **chunk)`; per structure once
`sink.stage((f"structure_{input_index}",), attrs={"structure_index": input_index, "structure_id":
structure_id})`; root attrs via `sink.stage((), attrs={…})` (as `streaming.py:173`):
`schema_version="<family>_v1"`, `model_family`, `purpose`, `output_kind`,
`alphabet="ACDEFGHIKLMNPQRSTVWYX"`, `skipped_inputs`; `finalize()` at end; else (ii) concatenates
chunks into the result dict. Memory bound = one batch × one chunk (as `host/streaming.py:62-69`).
MPNN `StreamingTensorStagingSink`/`take_staging_sequences_logits`/`_structure_ids_for_batch` unused
by drivers and unchanged.

**Ids/skips.** `structure_id = _canonical_structure_ids_for_spec(spec)[input_index]`; result
carries `skipped_inputs: list[{input_index, structure_id, reason}]`. Test: 3 inputs, middle
unparsable → ids + order asserted. Test: inputs `a/model.pdb`, `b/model.pdb` → two distinct groups
whose `structure_id` attrs are both `model`.

**Inference mode/dropout.** `run_family_driver` never calls `prep_protein_stream_and_model`. Stages
take static `scalar_dropout: bool` + explicit `dropout_key`; vector-channel dropout is never active
at inference. `scalar_dropout=True` only for LASEr `score:proofread_unconditional` and
`score:proofread_conditional` when `LaserOptions.proofread_dropout` (default True ≡ upstream `not
disable_inference_dropout`, `run_proofreading.py:205,230`), mirroring upstream `load_model`
(`:21-30`: `model.eval()` then `.train()` only on `torch.nn.Dropout`; `_VDropout` stays off,
`utils/model_generics.py:615-656`). `eqx.nn.inference_mode(model, value=False)` is forbidden. Keys:
unconditional `fold_in(key,0)`; conditional cell
`fold_in(fold_in(fold_in(key,focus_idx),dropout_idx),order_idx)`; parity tiers inject masks (§7.1).
T0.2 lists every `nn.Dropout` module path and every `self.training`/`F.dropout(training=…)` read with
its state under this rule.

**Reused as-is:** `_canonical_structure_ids_for_spec`, `resolve_target_samples`,
`make_axis_dispatch_via_xtrax`, xtrax `ZarrStagingSink` + `derive_sink_spec`, `metadata` key set of
`sample()` (T0.2/T0.2b confirm names/signatures at HEAD: `_sampling_helper.py:36`, `plan.py:382`,
`tiling/dispatch.py:161`; new on main: `WaveScheduleBundle.from_decoding_order` `types/bundles.py:357`,
`ar_mask_from_decoding_order` `utils/autoregression.py:318`, unused by drivers; any extraction refactor keeps MPNN outputs byte-identical, gated by T0.5a).

### 2.2 Family plumbing

- `run/specs.py:236` Literal adds `"pottsmpnn"`, `"lasermpnn"`; `__post_init__` derives them from
  `checkpoint_id` prefixes `pottsmpnn_`/`lasermpnn_` before the `model_type` derivation; explicit
  family never overridden.
- Each `model_family` consumer (§1; T0.2 re-grep'd, full classified list in the probe report §2):
  `_prepare_ligand_context` never reached for driver families (test; on the fallback path
  `pottsmpnn != "ligandmpnn"` already behaves as `proteinmpnn`, keep the explicit test); checkpoint
  registry gains `pottsmpnn`/`lasermpnn` sections (fallback: registry key stays `pottsmpnn`,
  `load_model` returns the embedded `Aminx`); portable JSON v2 raises `ValueError` — the serialize
  guard `run_spec_portable_json.py:153` is `== "ligandmpnn"` only and must become `!= "proteinmpnn"`,
  else a driver-family spec silently round-trips as `proteinmpnn`; multistate PoE / grid lineage /
  campaign / streaming raise `ValueError` for driver families (T0.2: the family label is harmless
  there, the chunked-lineage semantics are not implemented by drivers). Fallback purposes see
  `model_family="pottsmpnn"` but an `Aminx` model: the only consumers reached on that path
  (`_prepare_ligand_context`, `prep.py` registry lookup) treat it as `proteinmpnn`.
- Unsupported `(family, purpose, output_kind)` → `ValueError` at `__post_init__`.

### 2.3 Spec/CLI surface

- `ScoringSpecification.output_kind: Literal["nll","logits","energy","ddg","proofread_unconditional",
  "proofread_conditional"] = "nll"`; `sequences_to_score` required only for `nll`/`logits`.
- `RunSpecification.potts_mpnn: PottsMPNNOptions | None`, `laser: LaserOptions | None`
  (`None` → family defaults); `build_run_spec` threads to `run_spec.potts_mpnn`/`run_spec.laser`;
  options for a non-matching family → `ValueError`.
- `SamplingSpecification.omit_aa: Sequence[str] = ()`, `omit_aa_per_position: Mapping[int,str] |
  None = None`, threaded into `run_spec.sampling`; added to the RS-6b ast-grep sampling-field list
  with a planted-violation case. Compiled host-side to bias `-1e8` at designed positions (upstream
  constant).
- **Temperature family default (U-c).** `SamplingSpecification.temperature: Sequence[float|None] |
  float | None | Unset = UNSET` (`UNSET` = module singleton in `run/specs.py`, `__reduce__` returns
  itself). **Resolution point:** base `RunSpecification.__post_init__` calls new hook
  `self._resolve_family_defaults()` immediately after the `model_family` derivation block and before
  `self._sync_run_spec()` (currently `specs.py:519-520`); base hook no-op; `SamplingSpecification`
  overrides: `UNSET` → `(0.1,)` for proteinmpnn/ligandmpnn/membrane/pottsmpnn, `(None,)` for
  lasermpnn; scalar/`None` → 1-tuple; sequence → tuple; any `None` with non-lasermpnn →
  `ValueError`; lasermpnn element `0.0` → `None` (argmax; the batch-CLI rule: `--sequence_temp` is
  `type=float` so 0.0 is falsy, `run_batch_inference.py:245,370`; the single-input/tied CLIs take a
  *string* where `'0'` is truthy, so upstream passes `0.0` and divides by it → NaN,
  `run_inference.py:777,796`, recorded §6.5b `laser_temperature_zero`; PottsMPNN's 0 → 1e-6 floor,
  §6.5, is a different rule; tied `sample` maps both `0.0` and `None` → 1e-6, §5.4). The existing float→tuple at `specs.py:633-634` moves into the hook.
  `ScoringSpecification.temperature: float = 1.0` untouched. **Types:** `SamplingConfig.temperature:
  tuple[float|None, ...]`; new `_as_temperature_tuple` keeps `None` (`_as_float_tuple` unchanged, used
  for noise). MPNN consumers (`kernel_dispatch.py:134`, `_sampling_grid_lineage.py:98,115`,
  `campaign.py:84`, `multistate_poe.py:378,560`) read via `mpnn_temperatures(run_spec) ->
  tuple[float,...]` which raises on `None`. **replace():** post-init value is concrete →
  `dataclasses.replace` treats it as explicit; `replace(s, model_family=…, temperature=UNSET)`
  re-resolves. **JSON:** `(None,)` ↔ `[null]`; `_coerce_field_value` temperature branch accepts
  `int|float|None` elements. **CLI:** `--temperature` default unset → kwarg omitted. In a
  `temperatures` axis `None` is a static per-element branch: argmax, no min-p, unless
  `LaserOptions.fs_sequence_temp` is set (then 1e-6 on non-first-shell rows, §5.3 step 5). Tests: UNSET never
  reaches `build_run_spec`; per-family defaults; None+MPNN ValueError; both replace cases; `[null]`
  round-trip; T0.5a goldens exact.
- CLI (T0.6): `run score --output-kind`; `_RunBase` `--potts-options-json`, `--laser-options-json`
  via `_base_spec_kwargs`. **spec_json:** `_OPTIONS_FIELDS = {"potts_mpnn": PottsMPNNOptions,
  "laser": LaserOptions}`; encode = `{f.name: _to_json_value(v)}` per field; decode = dict →
  `cls(**…)` (lists → tuples where the field default is a tuple; unknown keys →
  `SpecJSONDecodeError`). The partition probe (`spec_partition.py:128-135`) cannot carry Options, so
  dedicated round-trip tests with `pottsmpnn_*`/`lasermpnn_*` ids cover spec_json and campaign.

### 2.4 L-DRV lint (`tests/lint/test_l_drv.py`, stdlib `ast`)

Scope: S_F = `src/aminx/families/**`, S_M = `src/aminx/model/{potts_mpnn,laser}/**`,
S_H = `src/aminx/host/family_*.py`.
- **R1** in S_F ∪ S_H: no call to `jax.vmap`/`vmap`/`eqx.filter_vmap`/`jax.pmap`/`shard_map` (named
  axes go through `BatchPlanner`/`make_axis_dispatch_via_xtrax`). Intra-layer neighbour/atom vmaps
  allowed in S_M only.
- **R2** in S_F ∪ S_M: no `For`/`While`/comprehension inside (a) `__call__` of any `eqx.Module`
  subclass or (b) any function decorated with, or passed as first argument to, `jax.jit`/
  `eqx.filter_jit`/`lax.scan`/`lax.while_loop`/`lax.fori_loop`/`lax.cond`; except iterables
  `self.<name>`, `enumerate(self.<name>)`, `zip(self.<name>, <any>)`, `range(<int literal ≤ 4>)`,
  `range(len(self.<name>))`.
- **R3** `lax.scan`/`while_loop`/`fori_loop` in S_F ∪ S_M only at `(path, enclosing qualname)` in
  `tests/lint/l_drv_allowlist.toml` with `reason="sequential_dependency"`; expected entries:
  `PottsARDecode` (steps, tied groups, tied members), `PottsRefine` (`potts` sweep,
  `potts_converge` while, `nodes` sweep), `LaserJointDecode` (steps). Stale entries fail.
- **R4** inside R2 bodies: no `.item()`, `.tolist()`, `np.asarray`, `jax.device_get`;
  `int()`/`float()`/`len()` only on literals, `<expr>.shape[<int literal>]`/`.ndim`/`.size`, or
  `self.<name>` declared `eqx.field(static=True)` or as a tuple of modules in the class body.
- **R5** no import of `aminx.potts` in S_F ∪ S_M ∪ S_H.
One planted-violation test per rule (temp file in scope, as `test_rs6b_flat_field_gate.py:54-80`)
+ clean-tree test + allowed-pattern fixture (`enumerate(self.layers)`, `zip(self.layers, keys)`,
`int(x.shape[0])`, `len(self.layers)`) that must pass.

## 3. ADR amendment

New `decisions/260929_pottsmpnn-lasermpnn-family-drivers.md` (`amends: 260605`):
- 260605 governs the TRW structure model (`aminx.potts.*`); unchanged, lint unchanged.
- PottsMPNN sampling and energy purposes run in a FamilyDriver whose decode stage reuses aminx
  `DecoderLayer` modules; its nll/logits scoring and jacobian/inspect run the unmodified MPNN paths
  on the embedded `Aminx` core. 260605 Option III (EncoderOutput widening) stays rejected.
- LASErMPNN is a FamilyDriver with its own encoder-output/result types; MPNN contracts not widened.
  Its own driver does **not** exempt it from composition: `LaserDriver` is xtrax-composed under the
  same §0 normative definition and L-DRV lint as PottsMPNN (eqx stage modules; samples/temperatures/
  ligand_atoms/focus_residue/decoding_order/dropout_seed as xtrax axes via `BatchPlanner`; proofread
  and tied reductions as `AxisBoundary` Fuses; export via xtrax `ZarrStagingSink`).
- New modules: `aminx.model.potts_mpnn`, `aminx.families.potts_mpnn`, `aminx.model.laser`,
  `aminx.families.laser`, `aminx.host.family_driver`, `aminx.host.family_runner`; L-DRV R5.
- `potts/model.py` docstring: one-line disambiguation pointer.

## 4. PottsMPNN

### 4.1 Model and weights

`PottsMPNN(eqx.Module)` = **composition**: `mpnn: Aminx` (stock aminx ProteinMPNN, shared
weights converted) + `potts_head: PottsHead`. `ModelProtocol` (incl. `__call__`, `stage_schema`)
satisfied by `mpnn` by construction; no subclassing, no capability field changes.

- Conversion `scripts/recapture/pottsmpnn_model_to_eqx.py` (bathos-tracked) reuses
  `scripts/convert_weights.py`'s ProteinMPNN key map; only `etab_out.*` is new. T0.2 records
  `load_state_dict(strict=False)` missing/unexpected keys per checkpoint; non-empty missing →
  blocking finding. **T0.2 result:** the 5 in-scope checkpoints have empty missing/unexpected sets;
  the six `proteinmpnn_compatible_model_weights/*` files have missing `{etab_out.weight,
  etab_out.bias}` (they are ProteinMPNN-only; loaded as Potts they would run Xavier-random `etab_out`,
  `potts_mpnn_utils.py:1263-1265`) and must never be registered under the `pottsmpnn` family (B1). Registered with **required** `sha256` = SHA-256 of the converted artifact prep
  loads (registry `sha256` is optional today, `host/prep.py:60`; required for `pottsmpnn_*`/
  `lasermpnn_*`), plus separate `source_sha256` (upstream torch file) + upstream commit.
- In scope: `vanilla/pottsmpnn_{20,30}`, `soluble/sol_pottsmpnn_{20,30}`, `ft/potts_ft` (T0.2:
  same constructor config as the others, 120/120 tensors differ from `vanilla_20`, no `ft`-specific
  overrides); `proteinmpnn_compatible_model_weights/` → `duplicate` exclusion (T0.2: all 118 shared
  tensors bit-equal to the matching vanilla/soluble file, but no `etab_out.*`, so not loadable as
  Potts; a registry test asserts none is registered under `pottsmpnn` and that a converter fed one
  raises on the missing `etab_out`).
- `mpnn_ext/external/aminx/weights/pottsmpnn/*.eqx.zst` not reused (TRW-model artefacts).
- Self-edge invariant test: `neighbor_indices[...,0]==arange(L)` on present rows.

### 4.1a Host featurizer (A0)

PottsMPNN driver purposes do not use proxide (`prep.py:102-122`: no upstream gap rows,
`residue_idx`, or chain order). `aminx.families.potts_mpnn.featurize` (host numpy, called from
`batches()`) ports:
- `parse_pdb_upstream(path, chains, skip_gaps)` = `parse_PDB`/`parse_PDB_biounits`
  (`potts_mpnn_utils.py:73-205`): ATOM + HETATM MSE→MET; first occurrence per atom per
  (resnum, icode) wins (altloc, multi-MODEL); rows `min_resn..max_resn`, one per icode, sorted;
  unknown names → `-`; missing resnum → gap row (`-`, N/CA/C/O NaN) unless `skip_gaps`
  (`:143-156`); absent chains dropped; non-`.pdb` inputs → `skipped_inputs` reason
  `pottsmpnn_requires_pdb`.
- `tied_featurize_port` (`:293-512`): chain order = sorted designed chains + sorted fixed chains;
  `-`→`X`; `S_true` (model alphabet), `chain_M`, `chain_M_pos` (per-chain 1-based p → chain-local write
  `:415`, global row = `global_idx_start[chain]+p−1` by concatenation), `chain_encoding`, `residue_idx = 100(c−1)+row`
  (`:373,408`), omit/pssm/`bias_by_res` arrays, tied groups + `tied_beta` (`:445-460`);
  `present = isfinite(sum X over atoms,xyz)`, then `X[isnan]=0` (`:510-512`).
Output length `L_total` (gap rows included), padded to `L_pad` with
`pad_valid = arange(L_pad) < L_total`; pad rows X=0, S=X, present=0.

| Site | Mask |
|---|---|
| kNN (`mpnn.features(mask=present)`) | invalid pairs → +inf; `lax.top_k` lower-index-first tie-break ⇒ gap rows before pad rows; `K_static=min(48,L_pad)`; first `min(48,L_total)` slots of real rows = upstream set (`:1143-1150`) on non-tie rows (upstream invalid pairs get row D_max, `:1148`, never selected when #present > K). Structures with #present ≤ K_eff < L_total are `knn_boundary_tie` (host-detected): upstream topk breaks the D_max tie arbitrarily, so no fill reproduces it; aminx keeps +inf; stock `ProteinFeatures` untouched |
| encoder `mask_V`/`mask_attend` | `present` (`:1789-1792`) |
| `PottsHead` | `present[i]·pad_valid[E_idx[i,k]]`, then slot-0×eye (`:1795-1797`) |
| `merge_pair` | reverse edge exists ∧ `pad_valid` both ends; no valid reverse → undivided (`etab_utils.py:176-183`); `exclude_self` ≡ slot `k==0` |
| `potts_energy`, `positional_potts_energy` | `pad_valid` both ends (gap rows contribute 0 via row mask + zero `-`/`X` slots, as upstream) |
| row kernel `attention_mask` (AR + `nodes`) | `nbr_valid[i,k] = pad_valid[E_idx[i,k]]` |
| AR `mask_bw/mask_fw` (decoder, tied_decoder), `S_true` branch | `present` only (raw mask, `:1426-1428`, `:1620-1622`, `:1448`); `chain_M_pos` enters AR only via `cm` (order key + fixed/sampled select, `:1419`/`:1603`) |
| `nodes`/refine `mask_bw/mask_fw`, refine skip | `present·chain_M_pos` (`sample_seqs.py:346,355`) |

**Index space.** All driver per-position inputs (`fixed_positions`, `fixed_tokens`,
`tied_positions`, `bias` rows, `omit_aa_per_position`) are 0-based over A0 rows; `mutant_csv` `pos`
is 0-based into the gap-filled chain (`run_utils.py:737`). Fallback purposes keep proxide; their
fixtures are gap-free.

**Gate (A0).** Field-level equality vs upstream `parse_PDB`+`tied_featurize` on: 2 example_pdbs;
a numbering-gap chain with `L_total ≤ 48`; a gapped chain with `L_total > 48` and #present < 48; an
insertion-code chain; a partial-backbone residue; each with `skip_gaps ∈ {False, True}`. `E_idx`
compared as per-row sets. Gap rows (all-tie): record §6.5b `gap_row_knn_tiebreak` (only
observable: `potts_etab` on edges incident to gap rows; excluded, listed by index).
`knn_boundary_tie` structures (#present ≤ K_eff < L_total): A0 compares only the deterministic
subset `{j : d_ij < D_max_i} ⊆ E_idx[i]` on present rows; §6.5b `knn_boundary_tie` scope = all
outputs of that structure; runtime `logger.warning`; excluded from exact waves and sidecars. Exact
tiers use the gapped `L_total ≤ 48` fixture.
`test_knob_semantics_skip_gaps` on the gapped fixture.

### 4.2 Potts head, etab conventions, padding

- `PottsHead(edge_features (L,K,H), E_idx, present, pad_valid) -> etab_raw (L,K,20,20)`: linear →
  `present[i]·pad_valid[E_idx[i,k]]` mask → slot-0 × eye(20) (upstream masks rows only,
  `potts_mpnn_utils.py:1284,1795`).
- `merge_pair(etab, E_idx, pad_valid, *, denom, exclude_self)`: reverse-edge gather; an entry merges
  iff its reverse edge exists and both ends are `pad_valid`, else stays undivided
  (`etab_utils.py:176-183`); `exclude_self` ≡ slot `k==0`; ports `merge_duplicate_pairE` for both
  call sites.
- `etab_forward = merge_pair(etab_raw, denom=2, exclude_self=False)` — consumed by `PottsRefine`,
  emitted by `potts_etab` sink (`etab_convention="forward_denom2"`).
- `etab_energy = merge_pair(etab_forward, denom=4, exclude_self=True)` — consumed by
  `potts_energy`/ddG/`PottsSampleEnergy`.
- `potts_energy(etab_energy, E_idx, pad_valid, seq_etab_alphabet)` ports `calc_eners` (no ×0.5).
  Sequences are encoded in the etab_utils alphabet (`-`=20, `X`=21, both zero-padded slots —
  replicated, §6.5); explicit model↔etab alphabet conversion with round-trip test.
- **Padding/K convention.** Parity convention = upstream **unpadded over A0 rows**
  (K=min(48,L_total), gap rows included as real nodes). Masks per the §4.1a table (`present` vs
  `pad_valid`; the r3 `edge_valid` rule is retired — upstream applies no node mask on edges at
  etab/merge/energy/decoder). Tests: `test_energy_invariant_to_padding` (`L_pad∈{L_total,128,512}`,
  f64 exact on `pad_valid` edges); `test_valid_neighbour_set_matches_upstream`
  (`L_total∈{30,48,49}`); oracle fixtures include an L=30 chain and a binding partition with
  `L_p<48`. (d) In the PottsARDecode/`nodes` row kernel, `nbr_valid[t:t+1]` is passed as
  `DecoderLayer(attention_mask=…)` (`decoder.py:333-336`) — masking **messages** from pad rows;
  zeroing context alone is insufficient since `message_mlp([h_i,0]) ≠ 0`. Test
  `test_decode_invariant_to_padding`: `L_total∈{30,49}`, `L_pad∈{L_total,128}`, f64, injected
  order/uniforms; tokens exact, `pad_valid` rows of `h_V_stack` rtol 1e-12. **Fallback (stock MPNN)
  unchanged:** `score:nll`/`score:logits` fallback parity asserted only on unpadded, gap-free
  single-input runs (`max_length=L_total`); padded `L_total<48<L_pad` divergence recorded in §6.5b.

### 4.3 Purposes and stages

| purpose / output_kind | Path | Stages / axes |
|---|---|---|
| `sample` (any mode/PSSM/tied/bias_by_res) | driver | `MPNNEncode → PottsARDecode → PottsSampleEnergy → [PottsRefine iff mode≠none] → sinks`; host ranking after last chunk; axes `samples`, `temperatures` |
| `sample` with `optimize_pdb`/`optimize_fasta` (upstream `skip_calc`, `sample_seqs.py:128-157`; if mode==none the run is ordinary `sample` with N forced to 1, upstream `:40` (`:41` only normalises `"none"`→`""`), `:130-157`) | driver | `MPNNEncode → PottsRefine` on loaded sequences; no PottsARDecode/PottsSampleEnergy/ranking; `num_samples` forced to 1 with a `logger.warning` if >1 requested (upstream `:40` forces silently; logging only, not a behaviour divergence); `knob_semantics_optimize_pdb` covers mode none → AR output N=1 and `num_samples=4` → N=1 + warning; axis `samples` = loaded sequences. `optimize_fasta`: entries whose key `startswith(<pdb><suffix>)` (`:319`, prefix quirk kept), file order, `:` stripped (`:322`); `optimize_pdb`: native per-chain sequences in A0 chain order (§6.5b). Length ≠ L_total → `ValueError`; encoded by `seq_to_ints` (`:347`). Refine order: `upstream_refine_order` with empty stored orders (`:325-343` keying incl. reuse when suffix empty; §6.5b). PDB (iff `write_pdb`) = `refined_sequence` of the last loaded key (`:368-373`). Tests `knob_semantics_optimize_pdb`, `_optimize_fasta` |
| `score:energy` | driver | `MPNNEncode → PottsHead → etab_energy → potts_energy`; axis `candidates` = `sequences_to_score` if non-empty, else the `score:ddg` resolution (mutant_fasta / mutant_csv / DMS), always plus WT; absolute energies; partitions never evaluated (upstream `ddG=False`). Each `sequences_to_score` entry must have length L_total in A0 row order (sorted designed chains then sorted fixed chains, `tied_featurize :327`) with gap rows as `-`/`X`, else `ValueError` naming expected length + chain order; test `test_score_energy_gapped_alignment` |
| `score:ddg` | driver | as energy; candidates = mutant_fasta / mutant_csv else single-mutant DMS (respecting `exclude_chains`); `ddg=E(mut)−E(wt)`; binding: axis `partition` (§4.4); `mean_norm` per PDB after ddG |
| `score:nll`, `score:logits`, `jacobian`, `inspect` | MPNN fallback on `model.mpnn` | unchanged |

- **`PottsARDecode(eqx.Module)`**: fields `layers: tuple[DecoderLayer,...]`, `w_s_embed`, `w_out`
  (references to `mpnn`'s submodules). Ports upstream `decoder` (`:1415-1488`) and `tied_decoder`
  (`:1599-`) as incremental per-position decode: `lax.scan` over order, carry `(S (L,), h_S (L,H),
  h_V_stack (n_dec+1,L,H))`; step t gathers row t, applies each layer on the `(1,K,·)` slice with
  `mask_bw/mask_fw` from the order exactly as `:1422-1428`, scatters row t; O(n_dec·K)/step.
  **Row kernel (normative).** Once per structure, outside the scan: `rank_flat` (Tied, below;
  equals `argsort(order)` when every group is a singleton); `mask_attend[i,k] =
  rank_flat[E_idx[i,k]] < rank_flat[i]` (= upstream `order_mask_backward` gathered by `E_idx`,
  `:1423-1426`/`:1616-1619`, O(L·K)); with `m = present` (AR; upstream raw mask, `:1426-1428`):
  `mask_bw = m[:,None]·mask_attend`; `mask_fw = m[:,None]·(1−mask_attend)`; `h_EXV_fw = mask_fw[...,None]·concatenate_neighbor_nodes(
  h_V, concatenate_neighbor_nodes(zeros_like(h_V), h_E, E_idx), E_idx)`. Step t:
  `h_ES_t = concatenate_neighbor_nodes(h_S, h_E[t:t+1], E_idx[t:t+1])`; per layer l:
  `ctx = mask_bw[t:t+1,:,None]·concatenate_neighbor_nodes(h_V_stack[l], h_ES_t, E_idx[t:t+1]) +
  h_EXV_fw[t:t+1]`; `h_V_stack[l+1] = h_V_stack[l+1].at[t].set(layer(h_V_stack[l][t:t+1], ctx,
  present[t:t+1], attention_mask=nbr_valid[t:t+1], inference=True)[0])` (mirrors `:1455-1462`; for
  `present[t]==0` the write equals upstream's skipped loop because `DecoderLayer` returns `mask·h_V`
  (`decoder.py:352-355`; upstream `:912-914`), valid for finite gap-row inputs (A0 gate asserts
  `isfinite`); test `h_V_stack[1:, present==0] == 0`; aminx
  `concatenate_neighbor_nodes` = upstream `cat_neighbors_nodes` [edge, neighbour] order).
  **Precision prerequisite (T0.5):** `DecoderLayer.__call__` accumulates in
  `jnp.promote_types(message.dtype, jnp.float32)` instead of hardcoded float32
  (`decoder.py:338-341`) — byte-identical for f32/bf16/f16 (T0.5a goldens); test `DecoderLayer`
  against sealed f64 fixture `declayer_f64.npz` (weights, `h_V`, `h_E`, `mask_V`, torch `DecLayer`
  output) produced by `dump_potts_oracles.py` (A1; SHA-256 in `oracle_manifest.toml`), rtol 1e-12,
  no torch in the aminx env; T0.2 greps `src/aminx/model/**` for other
  hardcoded float32 on encoder/decoder paths.
  Order `argsort((chain_mask·chain_M_pos·mask + 1e-4)·|randn|)` (`:1419-1421`),
  `randn = jax.random.normal(key_order,(L,))`; injected `decoding_order` has length L_total;
  device order = `concatenate([injected, arange(L_total, L_pad)])`. Untied oracle: passed to
  `decoder(decoding_order=)`. Tied oracle: inject `randn`, then pass the oracle's returned flattened
  `decoding_order` to aminx (regroup-invariant; test asserts `rank_flat` equality). Masked rows
  take `S_true` (`:1448`); fixed rows `S_t·cm + S_true·(1−cm)` (`:1483`). Per-step probabilities via
  PSSMMix (below). **Refine fresh-order key** (`skip_calc` or stored order missing):
  `argsort((chain_mask + 1e-4)·|randn|)` with no `chain_M_pos` and no `present`
  (`sample_seqs.py:338-339`); under `fix_decoding_order` the seed is per `:332-337`.
  **Tied (also covers untied: all singletons, `M_max=1`).** Host builds `tie_groups: int32 (L_pad,
  M_max)`, −1-padded (shape depends only on static buckets `L_pad` and `M_max` = next power of 2 ≥
  largest group, min 1): rows = `tied_positions` groups in listing order (members in listing order),
  then one singleton `[i]` for each row `i < L_total` not in any group, ascending, **including
  `present==0` rows**, then all-−1 rows. Host asserts groups disjoint with union `range(L_total)`;
  overlapping groups → `ValueError` (§6.5b). On device:
  ```python
  key   = jnp.where(pad_valid, (chain_mask*cmp*present + 1e-4)*jnp.abs(randn), jnp.inf)  # pad rows last
  order = decoding_order if injected else jnp.argsort(key)
  rank  = jnp.argsort(order)
  BIG   = jnp.iinfo(jnp.int32).max
  vm    = tie_groups >= 0
  safe  = jnp.where(vm, tie_groups, 0)
  group_key   = jnp.where(vm, rank[safe], BIG).min(axis=1)             # (L_pad,)
  group_order = jnp.argsort(group_key, stable=True)                    # empty rows last
  size   = vm.sum(axis=1).astype(jnp.int32)
  s_sorted = size[group_order]
  offset = jnp.zeros_like(size).at[group_order].set(jnp.cumsum(s_sorted) - s_sorted)
  m_idx  = jnp.cumsum(vm, axis=1) - 1                                  # valid-member index, listing order
  rank_flat = (jnp.full((L_pad,), BIG, jnp.int32)
               .at[jnp.where(vm, tie_groups, L_pad)].set(offset[:, None] + m_idx, mode="drop"))
  rank_flat = jnp.where(pad_valid, rank_flat, L_pad + jnp.arange(L_pad))
  ```
  `rank_flat` = inverse permutation of upstream flattened `new_decoding_order` (`:1606-1614`) and
  drives `mask_attend`/`mask_bw`/`mask_fw`/`h_EXV_fw`. Outer `lax.scan` over `g_step ∈ range(L_pad)`,
  `g = group_order[g_step]`, `lax.cond(size[g]==0, skip, body)`; inner scan over `m ∈ range(M_max)`
  skipping `tie_groups[g,m] < 0`; member m sees rows written by earlier members; group `h_S` = 0
  until sampled; masked singletons and groups with any `present==0` member take the `S_true` branch
  (§6.5): members run in listing order until the first `present==0` member m*; m* and all later
  members are not layer-updated and add no logits (`stopped` carry flag, `lax.cond`); all members get
  `S_true[m*]` (`:1641-1647`); `knob_semantics_tied_masked_member` fixture `[present, masked,
  present]`, negative control updating members after m* → fail; otherwise logits `Σ_m tied_beta[t_m]·(W_out(h_{t_m})/T)/size[g]`; `bias_by_res`/pssm/omit **and the
  fixed/sampled select** are read at `t_last = tie_groups[g, size[g]−1]`:
  `tok = cm[t_last]·draw + (1−cm[t_last])·S_true[t_last]` (`cm = chain_mask·chain_M_pos·present`),
  written to all members (§6.5). Tests `test_tied_rank_flat_matches_upstream`: `rank_flat[:L_total]
  == argsort(oracle decoding_order)` and `mask_attend == gather(oracle order_mask_backward, E_idx)`
  for 3 seeds, fixture with a group whose listing order ≠ raw-rank order and a non-member whose raw
  rank lies between two members; negative control raw `rank` must fail; overlap `ValueError` test.
- **PSSMMix** (inside PottsARDecode step, upstream order; T0.2 transcribes `:1391-1404`/decoder
  equivalents verbatim into the docstring): `p = softmax(logits/T − omit·1e8 + bias/T +
  bias_by_res/T)`; if `(pssm_bias_flag and coef.numel()>0) or pssm_bias.numel()>0`:
  `p = (1−coef·multi)·p + coef·multi·pssm_bias`; if `pssm_log_odds_flag`: `p=p·(mask+0.001)`,
  renormalise; `omit_AA_mask` renormalise; draw (§7.1 sampling rule).
- **PottsRefine** (bit-parity port): `potts` (one sweep), `potts_converge` (`lax.while_loop`, stop
  test literally `(ener_delta != 0) & (iters < max_iters)`, `max_iters=1000`), `nodes`. Temperature
  floor per upstream (`sample_seqs.py:38-39`) then draw (never argmin). Refine order per upstream
  keying quirk (§6.5). Binding: at interface positions `predicted_E=(E_c−E_c[cur])−(E_p−E_p[cur])`
  (`:148`); `ener_delta += predicted_E[chosen]` (`:176`); under `both` non-interface positions add
  absolute energies; `only` skips non-interface (`:135-136`); "≈1000 sweeps" holds only for
  `binding_energy_optimization="none"`. **`nodes`** = one sequential Gibbs sweep: init `S=seq`,
  `h_S=W_s(S)` everywhere, `h_V_stack[0]=h_V`, `h_V_stack[1..]=0`, `mask_bw=1` except self slot
  (`:188-190`); per visited position (skip `mask==0`/`chain_mask==0`): update row t layer by layer
  reading `h_V_stack[l]` (unvisited rows 0 for l≥1, `:183,221-241`), `W_out/T_opt` → PSSMMix → draw
  → write `h_S[t]`, `S[t]` (`:263-268`); reuses the row kernel with `mask_attend := (k≠0)`, i.e.
  `mask_bw = m[:,None]·(k≠0)`, `mask_fw = m[:,None]·(k==0)`, `m = present·chain_M_pos`
  (`sample_seqs.py:346`, `run_utils.py:187-191`); `h_EXV_fw` is **recomputed** from this `mask_fw`
  (self slot only; never the AR-order `h_EXV_fw`); `h_S` at t not masked (`h_S_masked` is a plain
  clone, `:203`); `attention_mask = nbr_valid`. `test_knob_semantics_nodes_gibbs` negative control:
  AR-order `h_EXV_fw` must fail.
  Tied refine (`tied_optimize_sequence`, `run_utils.py:273-519`, incl. `tied_epistasis`) is
  transcribed in the probe report §4.4; its quirks are bit-parity obligations listed in §6.5, its one
  crash is §6.5b. Tests
  `test_knob_semantics_nodes_gibbs`, `test_knob_semantics_binding_converge_stop`.
- **PottsSampleEnergy + host ranking.** Device: `sample_energy[n] = potts_energy(etab_energy,
  S_AR[n])` on the **pre-refine** AR sequence (`sample_seqs.py:259`); every sample refined
  independently (`:321-366`). After a structure's last `samples` chunk, `run_family_driver` computes
  `sample_rank = argsort(all sample_energy, stable=True)` over all N (= Python `sorted`),
  `best = sample_rank[0]`. PDB (iff `write_pdb`): `sequence[best]` if mode none, else
  `refined_sequence[best]` (not re-ranked, `:368-373`). Sampled FASTA = `sequence` in rank order named
  `<pdb><suffix>_<sidx>` (no `_sidx` when N=1) (`:268-272,392-396`); optimized FASTA =
  `refined_sequence` in the same order (`:378-382`). `skip_calc` path: see its row. Test `knob_semantics_rank_before_refine`: seed where AR-energy argmin ≠
  refined-energy argmin; PDB follows AR energy and matches oracle.

### 4.4 Binding partitions

Each partition's chains re-featurised by A0 (own L, kNN, encode, head; `get_etab :816-833`). Axis
`partition` (ragged L → `AxisSpec(bucket_boundaries=…)`; §4.2 padding convention makes energies
bucket-invariant); `AxisBoundary` Fuse on scalar energies `E_bind = E_complex − Σ_p E_p`; partitions
scored only when `partition_flag` (`energy_prediction.py:57-75`).

### 4.5 Result schema (`result_schema(spec, purpose)`)

`sample`: `sequence (N,L) i32` (AR, pre-refine), `sample_energy (N,) f32` (of `sequence`),
`refined_sequence (N,L) i32` iff mode≠none, `sample_rank (N,) i32` (structure-level:
`attrs={"level":"structure"}`, staged once via `sink.stage((f"structure_{i}",), sample_rank=…)`,
excluded from the per-chunk key check), opt-in `potts_etab
(L,K,20,20)`, `potts_E_idx`, `refine_energy_trace`. `score:energy`: `energy (N_cand,) f32`,
`candidate_ids`. `score:ddg`: `ddg (N_mut,) f32`, `mutant_ids`, `ddg_expt (N_mut,) f32|nan`.
`emit_dense_hJ` converts host-side only. Under `skip_calc`: `refined_sequence (N_loaded,L_total) i32`
only (+ opt-in `potts_etab`, `potts_E_idx`, `refine_energy_trace`). Per-position dims are `L_total`. The
`potts_etab`/`potts_E_idx` neighbour dim is sliced to `K_eff = min(48, L_total)` (device uses
`K_static = min(48, L_pad)`, §4.1a); `test_family_sink_ragged_L` also asserts
`potts_etab.shape[1] == min(48, L_total)` for an `L_total < 48` structure.

## 5. LASErMPNN

### 5.1 Types

```python
class LaserEncoderOutput(eqx.Module):
  prot_scalars: Float[Array,"L Hs"];  prot_vectors: Float[Array,"L V 3"]
  lig_scalars:  Float[Array,"A Hl"];  lig_vectors:  Float[Array,"A Vl 3"]
  pr_pr_eattr:  Float[Array,"L K E"]; pr_pr_idx: Int[Array,"L K"]
  lig_pr_eattr: Float[Array,"L Kl E"]; lig_pr_idx: Int[Array,"L Kl"]
  prot_mask, lig_mask, pr_pr_mask, lig_pr_mask
class LaserSampleResult(eqx.Module):
  sequence: Int[Array,"L"]; seq_logits: Float[Array,"L 21"]
  chi_deg: Float[Array,"L 4"]; chi_logits: Float[Array,"L 4 Nbins"]; chi_mask: Bool[Array,"L 4"]
```
Dims from checkpoint `model_params` (T0.2).

### 5.1a Alphabet boundary

`aminx.families.laser.alphabet.LASER_ALPHABET = "ARNDCEQGHILKMFPSTWYVX"` (`utils/constants.py:34`;
T0.2 verifies exact string) with `to_laser`/`to_mpnn` int32 permutations. Converted exactly once:
inbound in `batches()` (`sequences_to_score`, `fixed_tokens`, `bias`/`omit_aa*` columns,
`disabled_residues`, budget/charged letters by letter); outbound in result assembly (`sequence`,
`seq_logits`, `seq_log_prob`, `proofread_mean/std` columns). Sinks/results in aminx order; stages in
LASEr order. Oracle outputs mapped to aminx order before comparison. Tests
`test_laser_alphabet_roundtrip`, `test_knob_semantics_laser_bias_single_aa`.

### 5.2 Model port

| Upstream | aminx | Notes |
|---|---|---|
| `LigandFeaturizer` (`utils/ligand_featurization.py:7`; the 6 `ligand_featurizer.*` buffers), `LigandEncoderModule` (`model.py:1146`, init `:1150`; `:1312` is the unused `SpiceDatasetPretrainingModule`) | `LaserLigandEncoder` | weights from main checkpoint (T0.2 confirmed keys present) |
| `HomoGATv2`, `HeteroGATv2`, `GVP`, `DenseGVP`, `EquivariantLayerNorm` | `aminx.model.laser.layers` | edge-list → dense padded neighbour axis + mask; `scatter_softmax` → masked softmax returning 0 on empty K |
| encoder/decoder layers | `LaserEncoder`, `LaserDecoderLayer` | |
| χ heads | `LaserChiHead` | bins from checkpoint `chi_angle_rbf_bin_width` |
| `chi_offset_prediction_layers[k]` (`model.py:902`) | `LaserChiOffsetHead` | input `cat[chi_logits_k, one_hot(bin_k, Nbins)]`, `chi_logits_k` post-`minp_warp` when χT set, raw under argmax (`:897`) |
| `RotamerBuilder` | `aminx.model.laser.rotamers` — post-decode only (PDB writer / `sidechain_coords` sink) | ideal geometry `files/*.pt` → hashed `.npz` |

Graphs host-side, fixed shapes (protein kNN on Cα `k_pp` including self, `knn_graph(loop=True)`
`utils/pdb_dataset.py:450,518` — the self-edge is always masked since rank<rank is false,
`model.py:274,804`; test every row contains itself, negative control loop=False fails B0 parity;
ligand kNN `k_ll` loop=True per-ligand `:519`; ligand→protein `compute_ligand_protein_knn_graph`
`:520,797`, and ligand→protein kNN
`k_lp` + cutoff mask; from checkpoint `graph_structure`); ligand atoms padded to `ligand_atoms`
buckets. T0.2 enumerates every `radius_graph`/`scatter_*` site with its dense replacement.

### 5.3 Joint sequence+χ AR decode (`LaserJointDecode`)

`lax.scan` over decoding order (length L, masked); carry `(seq, seq_emb, chi_enc (L,4,R),
node_stack_s (n_dec+1,L,Hs), node_stack_v (n_dec+1,L,V,3), ala_count, gly_count)`. Step t
(`utils/model.py:833-865`):
1. decoder layers → `sequence_output_layer`;
1b. `logits += bias_row[t]`, `bias_row` = aminx `bias` (L,21) + −1e8 at `omit_aa`/`omit_aa_per_position`
   letters on designed rows (§2.3), in LASEr order (§5.1a). Untempered, before steps 2–5: divided by
   `T_eff`, seen by min-p and argmax, as in aminx MPNN (`inference/decode/autoregressive.py:320-323,336`).
   Steps 2–4 assign, so masked entries are unaffected. Stored `seq_logits` = the step-5
   logits **after** `minp_warp_logits` (post-min-p, untempered; `-inf` at removed tokens when
   `seq_min_p>0`; when `T_eff` is None no warp, so the step-4 logits) recomputed with `bias_row=0`:
   upstream reassigns `curr_out_logits = minp_warp_logits(...)` (`utils/model.py:854`, only when a
   temperature is set) and stores the reassigned tensor (`:881`), so with zero bias this is exactly
   upstream's stored tensor; `seq_log_prob = softmax(stored)[sampled]` is therefore renormalised over
   the kept tokens (`run_inference.py:744`). It is not the pre-min-p tensor.
   `test_knob_semantics_laser_stored_logits_minp` (`seq_min_p=0.05`, T=0.3: stored logits equal oracle
   `sequence_logits` incl. the `-inf` pattern, and `seq_log_prob`; negative control storing pre-warp
   logits fails; `seq_min_p=0` and `T_eff` None cases equal oracle). `test_knob_semantics_laser_bias_minp`: bias
   moves a residue across the min-p threshold (seq_min_p=0.05, T=0.3, injected uniforms) vs analytic;
   negative controls bias-after-min-p and bias-after-`/T` fail.
2. disabled residues := `finfo.min` on rows `~chain_mask` (LASEr `chain_mask=1` = fixed), or on rows
   `chain_mask` under `ignore_chain_mask_zeros` (`:836`);
3. ALA/GLY over budget := `finfo.min` (`:840-841`);
4. `disable_charged_fs` K/R/D/E := `-inf` (`:843-848`);
5. `T_eff = T` if `fs_sequence_temp` is None, else `where(first_shell_ligand_contact_mask,
   fs_sequence_temp, 1e-6 if T is None else T)` (per-residue, never None; `run_inference.py:541-544`).
   If `T_eff` is None: argmax; else `minp_warp_logits(logits, seq_min_p)` on **untempered** logits →
   `/T_eff` → softmax → draw;
6. fixed/sampled select (`:861-865`);
7. χ1..χ4 sequential (`:889-934`): `chi_logits_k = head_k(cat[s, seq_emb[t], chi_prev])`; argmax if
   χT None else `minp_warp(·, chi_min_p)` → `/chi_temp` → draw; `offset_k =
   LaserChiOffsetHead_k(cat[chi_logits_k, one_hot(bin_k)])`; `angle_k = remainder(bin_deg[bin_k] +
   offset_k + 180, 360) − 180` (`jnp.remainder` = torch floor-mod); unless `ignore_chain_mask_zeros`
   or `repack_all`: `angle_k = input_chi` where `chain_mask & ~isnan(input_chi)` (`:907-914`);
   `enc_k = RBF(angle_k)` appended to `chi_prev` for **every** k; `chi_enc[t,k]`, `chi_deg[t,k]`,
   `chi_logits[t,k]` written only where `chi_mask(aa_{X→G})[k]` (`:922-925`) — this `chi_enc` is what
   later residues see (`:816`); χ-GVP update after k ∈ {0,1,2} (`:931-934`).
**Carry init:** `seq=21` all rows; `seq_emb=embed(21)` all rows incl. fixed; `chi_enc=NaN` (read as
0, `:816`), `chi_deg=NaN`, `chi_logits=0`, `seq_logits=0`; `node_stack[0]`=encoder, `[1..]=0`;
masked-edge features built once from the init `seq_emb` (`:764-766`); fixed labels enter at their
own step (`:863,868`). Under `ignore_chain_mask_zeros` a `chain_mask==0` step is a full no-op
(`:793-796`), yet the row stays a visible predecessor via the order-only mask (`:804`) with
`embed(21)`, zero χ and `node_stack[l]`; final token `X`, logits 0.
Prior residues visible only via seq embedding + binned-χ RBF. Per-step O(n_dec·K) + 4 χ heads; no
rotamer build in-loop; jaxpr loop-body guard asserts no O(L²) op.

#### 5.3.1 Edge semantics (one test each)
1. Budget counts every decoded residue incl. fixed (`:782-788`); region from
   `budget_residue_selection` (ProDy host) or exposed-non-SS heuristic (B0).
2. Disabled `finfo.min` vs charged `-inf` (step 2/4).
3. min-p only when `T_eff` set; `T_eff` None (T None, no fs temp) → argmax.
4. `sample`: input χ kept only where fixed and non-NaN (`:907-914`). Tied: input χ `nan_to_num`ed
   (`:491-492`) and kept wherever fixed unless `repack_all` (`:690-692`) — a NaN fixed χ becomes 0°;
   `test_knob_semantics_tied_chi_nan_fixed`.
5. `ignore_chain_mask_zeros` writes `X` at unsampled positions (`:936-937`).
6. `fs_sequence_temp` set → always sampled, min-p active; first-shell rows use `fs_sequence_temp`,
   others use T (1e-6 if T None). `test_knob_semantics_fs_sequence_temp` over T∈{None,0.3} ×
   seq_min_p∈{0,0.05} vs oracle, injected uniforms; negative control "1e-6 on non-first-shell rows
   regardless of T" fails at T=0.3.
7. `ignore_ligand` → empty lig-prot K; masked softmax 0; equals upstream empty-ligand output.
8. `repack_only` ⇒ `chain_mask=1`; effective `repack_all = repack_all ∨ repack_only`
   (`run_inference.py:729-740`; `run_batch_inference.py:203,217`); `repack_all` alone keeps
   chain_mask and only disables input-χ retention (`:907`); `test_knob_semantics_repack_all`.
9. Deterministic E2E: injected order + argmax seq/χ exact vs upstream.
10. `ignore_chain_mask_zeros` inverts the disabled-row set.
11. E2E (injected order, argmax): (a) a fixed row ordered after a neighbouring designed row; negative
    control true-label init → fail; (b) under `ignore_chain_mask_zeros`; negative control skipped
    rows invisible → fail.

### 5.4 Purposes

| purpose | Stages / axes |
|---|---|
| `sample` | encode → `LaserJointDecode` → RotamerBuilder (post) → sinks; axes `samples`, `temperatures` |
| `sample` tied (`tied_second_input`) | encode both (equal L else `ValueError`), shared order, per-step `λ·P1+(1−λ)·P2` (§6.5); bias/omit per §5.3 step 1b added to each structure's logits before its softmax: `λ·softmax((l₁+b)/T)+(1−λ)·softmax((l₂+b)/T)`; tied semantics: disabled mask on `~chain_mask` only (`:606`), no min-p, no charged mask, T None→1e-6 sampled (`:617-618`; aminx also maps `0.0`→1e-6, §2.3, §6.5b); setting `seq_min_p`/`chi_min_p`/`fs_sequence_temp`/`disable_charged_fs`/`ignore_chain_mask_zeros` with tied → `ValueError` (none is a `tied_sample` parameter, `utils/model.py:460-465`; list confirmed by T0.2). The tied CLI's `--disable_charged_fs`, `--disabled_residues`, `--ebd` are parsed but dead (§6.3) |
| `score:nll`, `score:logits` | teacher-forced seq+χ log-probs (`get_logits_for_score :261` port; conditioning per §5.4a) — primary parity surface |
| `score:proofread_unconditional` | one forward, `return_unconditional_probabilities`, softmax; residues = first shell or `selection_string` |
| `score:proofread_conditional` | axes `focus_residue` × `dropout_seed` (`n_dropouts`) × `decoding_order` (`n_decoding_orders`). Focus set = the unconditional pass's set: `first_shell_ligand_contact_mask` (`run_proofreading.py:55`), or if `selection_string` non-empty, resindices of `(same residue as (sel)) and name CA` (`:69-79`, None → ValueError); `residue_ids` ordering = upstream `ylabels`. **`resindex` ↔ row (T0.2):** upstream *assumes* ProDy `resindex` (counts every residue of the parsed `AtomGroup`: ligand, water, non-amino, dropped) equals the batch row (`run_proofreading.py:120`, `:69-79`); true on `4jnj-1_prot.pdb`, false when any non-row residue precedes the focus residue in file order. B0 emits `row_to_resindex`; focus rows/selection resindices are mapped through it (§6.5b `proofread_resindex_identity`). Per (focus r, rep d): one batch of `n_decoding_orders` copies, only r designable (others `chain_mask=1`), fresh orders per rep (§5.4a rule), per-copy scalar-dropout masks; `sample` at T=1, χT=1, `disabled_residues=('X',)`, `seq_min_p=0` (identity, `model.py:61-62`), `repack_all` per Options; `p[d,o]=softmax(sequence_logits[r])` on stored post-disabled-mask untempered logits (`:838,881`). Fuse: `m_d=mean_o p`, `s_d=std_o(p, ddof=1)`; `proofread_mean=mean_d m_d`, `proofread_std=mean_d s_d`, `proofread_mean_plus_std=proofread_mean+proofread_std`; no renormalisation; `n_decoding_orders=1` → std NaN (kept); sampling `bias`/`omit_aa*` not applied (scoring purpose) |
| `jacobian`, `inspect` | `ValueError` (no fallback) |

**§5.4a `score:nll|logits` conditioning (normative aminx defaults; upstream `get_logits_for_score`
has no caller in the pinned repo).** (i) Order: one per structure, shared by all candidates, by the
upstream sampler rule `_masked_sort_for_decoding_order` (`utils/pdb_dataset.py:1618-1650`):
`order = argsort(u + tier)`, tier 0 = `chain_mask` (fixed), 1 = designable∧¬contact, 2 =
designable∧contact, contact = `batch.extra_atom_contact_mask` (not `first_shell_ligand_contact_mask`), named in the B0 gate
field list. **T0.2 (B3): upstream inference never sets it**: the featurizer writes `'extra_atom_contact_mask':
zeros_bool` (`run_inference.py:350,371`; tied `run_inference_tied.py:404,425`; batch collate
`run_batch_inference.py:129` only concatenates it), and nothing in `construct_graphs` reassigns it. The
non-trivial definition `pdb_dataset.py:1233-1239` (heavy atoms of `all_extra_coords_data`) belongs to
the training-dataset loader. Hence at inference tier 2 is **empty** and the order is fixed first, then
all designable, random within tier; that is what `sample`, proofread and (aminx default) `score:nll|logits`
use. The 3-tier rule is kept verbatim for parity of `_masked_sort_for_decoding_order`, and its tier-2
branch is exercised only by injecting `extra_atom_contact_mask` on the oracle batch by hand (the
upstream inference featurizer cannot produce it); any non-empty aminx-defined contact mask for
`score:nll|logits` is out of v1 and would have to be named separately from the `sample` rule. `chain_mask` = the B0 mask
from `fixed_positions`/`fixed_mask`/`fix_from_bfactor`, as for `sample`; `u = jax.random.uniform(key_order,(L,))` from `random_seed`; the same rule (with the empty tier 2 above)
drives `sample` and proofread orders; injectable `decoding_order` (length L) in parity tiers; emitted under
`return_decoding_orders`. The injectable uniform stream follows the upstream layout: concatenation of
tier-0, tier-1, tier-2 uniforms, each in ascending row index (`utils/pdb_dataset.py:1618-1650`); aminx
maps it back to rows before `argsort(u + tier)`. Padding rows (aminx shape-bucket rows beyond the
structure's residue count; upstream `curr_batch_mask`, `pdb_dataset.py:1625-1633`, selects only real
rows) belong to no tier: sort key `+inf`, they fill `order[n_real:]`, are excluded from the injected
stream (length `n_real`) and are never decoded; parity compares `order[:n_real]`. Gap rows
(`present==0`) are real rows and keep their tier. (ii) χ: `chi[i,k] = input_chi[i,k]` iff `aa_to_chi_angle_mask[cand_i
(X→G)][k]` ∧ input non-NaN, else NaN. (iii) Outputs: `seq_log_prob (N,L) =
log_softmax(sequence_logits)[cand]` (no disabled mask, bias/omit, temperature or min-p); `chi_log_prob (N,L,4) =
log_softmax(chi_logits_k)[bin(chi)]`, NaN where χ is NaN (T0.2 transcribes `bin`); `scores (N,) =
−mean_{prot_mask} seq_log_prob` (nats); `chi_nll (N,) = −nanmean chi_log_prob`; opt-in `seq_logits`,
`chi_logits`. Tests: `test_knob_semantics_laser_score_order` (seeds differ; injected oracle order
matches); `test_laser_score_chi_candidate_mismatch` (native A→K, native K→G); negative control:
native χ for a mismatched candidate → fail.

### 5.5 Host featurizer (B0)

B-factors (`fix_from_bfactor` requires {0,1}), side-chain coords + χ (NaN → mask), ligand
elements/H/water (`use_water`)/ncAA-as-ligand, φ/ψ, SS/exposure, first shell. **First shell (T0.2):**
the consumed `first_shell_ligand_contact_mask` is the *post-`construct_graphs`* mask
(`pdb_dataset.py:590` unconditionally overwrites the `output_batch_data` mask, `run_inference.py:321-328`,
before every consumer reads it), i.e. the heavy-atom contact rule `compute_first_shell_node_idces`
(`:756-786`, called `:528,555`): residue in the ligand-protein edge set with a heavy atom `<5.0 Å` of a
ligand heavy atom (`+0.3 Å` for Gly/X, CA only); ligand-free → all-False (`:449-454`). B0 does **not**
implement the alpha-hull burial or the 10 Å CA test for this mask (`fs_calc_*` knobs are inert, §6.4).
Proxide
(`prep.py:102-122`) lacks B-factors → B0 extends the proxide call or parses via ProDy in `batches()`.
B0 also emits `row_to_resindex` (§5.4). **Fixtures (T0.2, B2):** the pinned repo ships only
`example_pdbs/4jnj-1_prot.pdb`; `databases/` holds `README.md` + `dataset_split_info.zip` (split JSON,
no structures). Gate: field-level equality vs upstream `BatchData` on 5 fixtures
(`example_pdbs/4jnj-1_prot.pdb` + 4 complexes from the upstream Zenodo PDB dataset, entries 1–4 of the
`laser_score_parity` list, §7.3), acquired and pinned (URL + SHA-256 of the archive and of each
extracted file) in task step **B1(a)** (§8). `construct_graphs(num_adjacent_residues_to_drop=·)` is inert
at inference (`sampled_pseudoligands is None`; only the pseudoligand path uses it,
`utils/pdb_dataset.py:443,481-488,512`; hardcoded 6/0 in `run_proofreading.py:50,133` /
`run_inference_tied.py:586,598`, not an argparse dest); B0 gate asserts upstream graphs with 0 and 6
are identical on the 5 fixtures; T0.2 confirmed statically (probe report §6.9).

### 5.6 Result schema

`sample`: `sequence`, `seq_log_prob`, `chi_deg (N,L,4)`, `chi_mask`, opt-in `sidechain_coords`;
PDB (hydrogens per checkpoint `build_hydrogens`); FASTA (`output_fasta`, `output_fasta_only`).
`score:nll/logits`: `seq_log_prob`, `chi_log_prob`, `scores`, `chi_nll`, opt-in logits.
`proofread_*`: `proofread_mean (R,21)`, `proofread_std (R,21)`, `proofread_mean_plus_std (R,21)`
(conditional only), `residue_ids`. `sample` tied: base channels are structure 1 (upstream-returned,
`run_inference_tied.py:612-624`); always adds `seq_log_prob_2`, `chi_deg_2 (N,L,4)`, opt-in
`seq_logits_2`/`chi_logits_2`/`sidechain_coords_2`; `sequence`/`chi_mask` shared
(`utils/model.py:647,650`); PDB/FASTA from structure 1 only.

## 6. Knob surface and redsox coverage

### 6.1 Options (flat dataclasses)

`PottsMPNNOptions`: `optimization_mode: Literal["none","potts","potts_converge","nodes"]="potts"`,
`optimization_temperature=0.0`, `binding_energy_optimization: Literal["none","both","only"]="none"`,
`binding_energy_json`, `binding_energy_cutoff=8.0`, `mean_norm=False`,
`filter_nan=False` (no-op quirk), `mutant_fasta`, `mutant_csv`, `exclude_chains`, `pssm_json`,
`pssm_threshold`, `pssm_multi`, `pssm_log_odds_flag`, `pssm_bias_flag`, `bias_by_res_json`,
~~`tied_beta`~~ **(DELETED 261003, not plumbed: upstream never takes it as config — `potts_mpnn_utils.py:444` inits `np.ones(L_max)`, `:456` fills it from the tied_positions JSON weights — and `featurize._tied_groups` already reproduces that derivation; a scalar option would invent semantics upstream lacks)**, `tied_epistasis`, `skip_gaps`, `optimize_pdb`, `optimize_fasta`, `write_pdb=True`,
`emit_etab=False`, `emit_dense_hJ=False`, `chain_design_mask_json`. Path-valued fields are
`str | None`. (Upstream `ddG` maps to `output_kind`, §6.3.)

`LaserOptions`: `fs_sequence_temp`, `chi_temp`, `seq_min_p`, `chi_min_p`,
`disabled_residues=("X","C")` (the batch-CLI default, `run_batch_inference.py:380`; single-input
`run_inference.py:514`/`sample()` `utils/model.py:731` default `['X']`, see §6.3),
`disable_charged_fs`, `repack_only`, `fix_from_bfactor`,
`ignore_ligand`, `use_water`, `noncanonical_aa_ligand`, `ala_budget=4`, `gly_budget=0`,
`constrain_ala_gly_to_exposed_non_ss`, `budget_residue_selection`, `ignore_chain_mask_zeros`,
`tied_second_input`, `tied_interpolation_lambda=0.0`, `selection_string`, `n_decoding_orders=10`,
`n_dropouts=10`, `proofread_dropout=True`, `repack_all=False`, `strict_load=True`, `output_fasta`,
`output_fasta_only`. (Defaults transcribed by the extractor; generated alias table is authoritative.)
`fs_calc_ca_distance`, `fs_calc_burial_hull_alpha_value`, `fs_no_calc_burial` are **not** Options
fields: T0.2 showed they are inert for every consumed output (§5.5), so they are `no_op` exclusion
rows (§6.4).

### 6.2 Reference surfaces (generated)

`scripts/redsox/extract_upstream_knobs.py` (bathos-tracked, pure AST): every upstream `*.py` with
argparse (glob) → dests; `sample_model`/entry-point parameters; PottsMPNN `cfg.inference.X`/
`cfg.model.X` reads and `'X' in cfg.inference` tests ∪ example-YAML keys. Emits
`tests/knob_gate/reference_surfaces.py`: one frozen dataclass per entry point, fields namespaced
`<entrypoint>__<dest>`, header with upstream SHA; regeneration diff test. Plus a manual dataclass
`pottsmpnn_input_list` (`pdb`, `designed_chains`, `fixed_chains`; line format
`pdb|designed:chains|fixed:chains`, `sample_seqs.py:81-94`), marked `# MANUAL`, skipped by the diff.

### 6.3 Alias table

`tests/knob_gate/alias_map.toml`, one row per reference field: `{ref, targets:[..], equivalence ∈
{identical, semantic, exclusion, divergence}, reason?, parity_test_ids:[..], note}`. Namespacing ⇒
no implicit name matches. Known non-identical rows: `*__chain_dict_json` and
`pottsmpnn_input_list__{designed,fixed}_chains` → `[fixed_mask, chain_design_mask_json]` semantic;
`fix_decoding_order`+`decoding_order_offset` → `[random_seed]` semantic (seeded-order test; order
source is the driver, not `decoding_order_fn`); Potts `noise` (eval `augment_eps`, all backbone
atoms, applied whenever `augment_eps>0` incl. eval, `potts_mpnn_utils.py:1170-1171`) → `[noise]`
semantic (knob test: iid Gaussian N/CA/C/O per atom); LASEr `bb_noise` → `[noise]` semantic with a
**different** rule (`pdb_dataset.py:412-415`, only if `>0`: coordinates rounded to 2 decimals, then
one `randn((N,1,3))` translation per residue shared by all 5 backbone atoms, i.e. a rigid
per-residue shift; tied CLI noises each structure independently, `run_inference_tied.py:576-599`):
the LASEr driver has its own noise stage and `test_knob_semantics_laser_noise` (per-residue rigid
translation + rounding vs oracle with injected `randn`; negative control iid-per-atom fails);
`run_inference*__fs_calc_ca_distance`, `__fs_calc_burial_hull_alpha_value`, `__fs_no_calc_burial`
(and `run_batch_inference*`) → `exclusion`/`no_op` (§6.4; T0.2: overwritten by `construct_graphs`,
`pdb_dataset.py:590`); `run_inference_tied__disable_charged_fs`, `__disabled_residues`,
`__entropy_decoder` → `exclusion`/`no_op` (parsed, never consumed: `run_inference_tied.py:620` comments
out the `disable_charged_residue_mask=` argument, `run_inference(...)` has no `disabled_residues`
parameter, `use_edo` unused in the tied path; note the spec's `ValueError` for `tied_second_input` +
`disable_charged_fs` set via the *Options* is deliberate and stricter than upstream, which silently
accepts the flag); `run_inference__disabled_residues`: no such flag (fixed `['X']`,
`run_inference.py:514`; batch/tied CLIs default `'X,C'`), so the parity call for the single-input
entry point passes `disabled_residues=("X",)` and the alias row records this;
`*__repack_all` → `[repack_all]` identical; `*__repack_only`,
`run_batch_inference*__repack_only_input_sequence` → `[repack_only]` semantic;
`run_proofreading__disable_inference_dropout` → `[proofread_dropout]` semantic (inverted);
`run_batch_inference*__ignore_key_mismatch` (store_false, passed as `strict=`,
`run_batch_inference.py:250,379`) and `run_inference*__strict_load` (`--ignore_statedict_mismatch`,
store_false, `run_inference.py:783`) → `[strict_load]` semantic (dest True ≡ `strict_load=True`; note
records name/polarity inversion);
`laser_*__sequence_temp` → `[temperature]` semantic (family default None, §2.3; `0`/`0.0` → argmax
per §2.3, and `run_inference*__sequence_temp` is a *string* flag, `'0'` truthy, `''` → None);
`designs_per_batch`/`max_tokens`/`inputs_processed_simultaneously` → `[batch_size]` semantic
(output-invariance test); `model_weights`/`check_path` → `[checkpoint_id, model_local_path]`;
`device` → exclusion `device`; `verbose`/`disable_pbar`/`silent` → `io_only`; `model.*`,
`graph_structure.*`, `build_hydrogens` → `checkpoint_derived`; `filter` → `[filter_nan]` (no-op);
`optimize_fasta` → `[optimize_fasta]` divergence (§6.5b); `*__ddG` → `[output_kind]` semantic
(True ≡ `ddg`, False ≡ `energy`).

### 6.4 Exclusions

`reason ∈ {io_only, device, visualization, training_only, checkpoint_derived, no_op, internal,
duplicate, deferred:<debt#>}`. Deferred (v1), filed as aminx tech debt 260929: `entropy_decoder` →
`deferred:2069`; `mutation_search.*` → `deferred:2070`; CA-only → `deferred:2068`; MSA vocab-22 →
`deferred:2067`; `run_predict_partial_charges.*` → `deferred:<filed in T0.4>`; `run_inference_ligandmpnn.py`,
`run_batch_inference_ligandmpnn.py` → `duplicate` (T0.2 confirmed: they import
`LASErMPNN.utils.model_ligandmpnn.LigandMPNN`; aminx already implements LigandMPNN).
`proteinmpnn_compatible_model_weights/*` → `duplicate` (§4.1). **`no_op` rows (T0.2, each with an
inverse-differential `knob_semantics_*` test: set == unset byte-identical in aminx and oracle):**
`run_inference*__fs_calc_ca_distance`, `__fs_calc_burial_hull_alpha_value`, `__fs_no_calc_burial`
(+ `run_batch_inference*`; inert, §5.5); `run_inference_tied__disable_charged_fs`,
`__disabled_residues`, `__entropy_decoder` (dead tied-CLI flags, §6.3). These replace the earlier
live-knob treatment; none of them is an Options field. `DEFERRED_IDS` is a checked-in list refreshed by T0.4.

### 6.5 Upstream quirks preserved (bit-parity)

| Quirk | Anchor | Handling / test |
|---|---|---|
| PSSM mixing fires whenever `pssm_bias` non-empty (precedence) | `potts_mpnn_utils.py:1392` | PSSMMix; `knob_semantics_pssm_precedence` |
| `potts_converge` accumulates absolute energy (non-binding) → ~1000 sweeps; binding sums ΔE | `run_utils.py:116-120,148,176` | PottsRefine; `knob_semantics_potts_converge`, `..._binding_converge_stop` |
| AR `temperature==0` and `optimization_temperature==0` → 1e-6, then draw | `sample_seqs.py:38-39` | host floor per `temperatures` axis element and on `optimization_temperature` before dispatch; `knob_semantics_t0_floor` tests AR T=0 and T_opt=0 separately vs oracle |
| `filter` no-op | `etab_utils.py:312` | `no_op` inverse differential |
| Refine order lookup key mismatch (`str(i)` vs `"_i"`) → N→C for `num_samples>1`; chain-suffix miss → fresh randn order | `sample_seqs.py:274-279,321-345`; `run_utils.py:110-111` | host `upstream_refine_order(...)`; `knob_semantics_refine_order` |
| Ranked by pre-refine AR energy; PDB = refined seq of that best | `sample_seqs.py:257-287,368-373` | host ranking; `knob_semantics_rank_before_refine` |
| Tied bias/PSSM/omit **and fixed/sampled select** read at last listed member; designed-last group overwrites fixed members with the draw, fixed-last writes `S_true[t_last]` to all | `potts_mpnn_utils.py:1665-1684` (`tied_sample :1575-1594`) | PottsARDecode tied; `knob_semantics_tied_last_member` covers fixed-last and designed-last mixed groups |
| `X`/`-` score as zero-padded slots 21/20 | `etab_utils.py:362-366`; `sample_seqs.py:211` | potts_energy; `knob_semantics_x_gap_energy` |
| LASEr tied λ weights structure 1; default 0.0 = structure 2 only (help text says opposite) | `utils/model.py:626` vs `run_inference_tied.py:875` | code semantics, default 0.0; `knob_semantics_tied_lambda` |
| LASEr fs temp: T None → 1e-6 on non-first-shell rows (sampled, min-p active); set T kept | `run_inference.py:541-544` | §5.3 step 5; §5.3.1-6 |
| Tied group with any `mask==0` member: all members take that member's `S_true`, no draw; rows already written by earlier members stay | `potts_mpnn_utils.py:1641-1647` (and `:1551-1557`) | PottsARDecode tied `lax.cond` at member scan; `knob_semantics_tied_masked_member` |
| Tied refine (`tied_optimize_sequence`, `run_utils.py:273-519`), non-`nodes` modes: a group is skipped when **any** member has `mask==0` **or** `chain_mask==0` (fixed), and every member is then overwritten with the *first such member's* current residue (`seq[pos_inner]=seq[pos]`) | `run_utils.py:331-340` | PottsRefine tied; `knob_semantics_tied_refine_frozen_group` (group with one fixed member; negative control designed-only skip fails) |
| `tied_epistasis`: mutants set all members jointly (`:349-355`), but the positional energy, binding lookup (`inter_mask`, partition) and the `t` used for `bias_by_res`/PSSM/omit are read at the **leaked loop variable `pos`** = last member (`:361,:404`) | `run_utils.py:349-361,404` | PottsRefine tied; `knob_semantics_tied_epistasis_leaked_pos` (negative control: first member fails) |
| Tied binding branch subtracts the unbound energy **without** the current-identity reference subtraction used untied (probabilities identical by softmax shift-invariance; the accumulated `ener_delta` is not), so tied `potts_converge` with binding runs to the 1000-sweep cap like the non-binding case | `run_utils.py:371,398` vs `:148` | PottsRefine tied; `knob_semantics_tied_binding_no_reference` (ener_delta trace vs oracle) |
| Tied `nodes` has **no per-member `chain_mask` skip** (only `mask==0`, `:473`); `chain_mask` enters via the last-member hard pick (`:513`) and the caller's `mask*chain_M_pos`; a masked member copies the current `S[:,t]`, not `S_true` (`:474`) | `run_utils.py:466-516` | PottsRefine tied `nodes`; `knob_semantics_tied_nodes_member_skip` |

### 6.5b Divergences (fixed upstream I/O bugs)

| Upstream bug | Anchor | aminx | Test |
|---|---|---|---|
| `optimize_fasta` asserts its own path, then reads `out_dir/out_name.fasta` | `sample_seqs.py:133-137` | reads `optimize_fasta` | `test_divergence_optimize_fasta_path` (oracle fixture uses path == `filename` so both agree) |
| `knn_boundary_tie`: #present ≤ K_eff < L_total → upstream torch.topk breaks the row-D_max tie arbitrarily | `potts_mpnn_utils.py:1144-1150` | aminx +inf fill, lower-index-first; runtime warning; scope = all outputs of that structure; excluded from exact waves/sidecars | A0 deterministic-subset test |
| `optimize_fasta` reads refine orders from implicit prior-run file `out_dir/out_name_decoding_order.json` (no config key) | `sample_seqs.py:122,178-180,325-330` | never read; fresh-order branch (`:331-343`), = upstream when the file is absent | `test_divergence_refine_order_json` (oracle out_dir without file) |
| `optimize_pdb` concatenates chains in listing/alphabet order, misaligned with featurization order | `sample_seqs.py:146-155`; `potts_mpnn_utils.py:268-290`, `:327` | per-chain sequences concatenated in A0 order | `test_divergence_optimize_pdb_chain_order` (fixed-first listing); parity fixture where orders agree |
| LASEr tied with `fs_sequence_temp` → NameError (undefined `batch`) | `utils/model.py:485`; `run_inference_tied.py:607-616` | `ValueError` naming the flag | `test_divergence_tied_fs_temp` |
| Overlapping tied groups double-decode shared positions | `potts_mpnn_utils.py:1606-1614` | `ValueError` | `test_tied_overlap_raises` |
| Tied refine non-epistasis path divides by `num_pos` (`predicted_E /= num_pos`, float `0.0`); when every member is non-interface under `binding_energy_optimization="only"`, `num_pos==0` → upstream `ZeroDivisionError` crash | `run_utils.py:401` | `ValueError` naming the group (all members non-interface under `only`) | `test_divergence_tied_only_zero_pos` (oracle fixture asserts the upstream raise; aminx raises `ValueError`) |
| LASEr `temperature==0`: batch CLI (`--sequence_temp` `type=float`) maps 0.0 → None (argmax); single-input/tied CLIs (`--temp` string, `'0'` truthy) pass `0.0` → `softmax(logits/0.0)` NaN | `run_batch_inference.py:245,370`; `run_inference.py:777,796`; `run_inference_tied.py:880` | `sample`: `0.0` → argmax (batch rule, §2.3); tied: `0.0` → 1e-6 (same as None, `utils/model.py:617-618`); the NaN path is never reproduced | `test_divergence_laser_temperature_zero` |
| Proofread `resindex` used as batch row index (assumes ProDy `resindex` == row) | `run_proofreading.py:69-79,120` | explicit `row_to_resindex` map from B0; identical to upstream whenever identity holds (all parity fixtures, incl. `4jnj-1_prot.pdb`); when it fails upstream fixes/selects the wrong residue or raises `IndexError`, aminx warns and maps correctly | `test_divergence_proofread_resindex_identity` (fixture with a non-row residue before the focus residue; parity fixture where identity holds) |
| (pre-existing aminx, not upstream) stock MPNN decoder passes invalid-neighbour messages when `L_total<48<L_pad` | `decoder.py:144-147`; `features.py:165-184` | unchanged; PottsMPNN fallback parity only unpadded | backlog id to be filed by the orchestrator (draft in probe report §8; not filed by T0.2) |

### 6.6 Gate

redsox added as a dev dependency pinned to a SHA. Harness `tests/knob_gate/test_knob_superset.py`:

```python
ROWS = tomllib.load(open(ALIAS, "rb"))["row"]
REFS = [c for c in vars(reference_surfaces).values() if is_dataclass(c)]
REF_F = {f.name for c in REFS for f in fields(c)}
TGT = (RunSpecification, SamplingSpecification, ScoringSpecification, PottsMPNNOptions, LaserOptions)
TGT_F = {f.name for c in TGT for f in fields(c)}
LIVE = [r for r in ROWS if r["equivalence"] != "exclusion"]

def test_rows_bijective():
    refs = [r["ref"] for r in ROWS]; assert len(refs) == len(set(refs)) and set(refs) == REF_F
def test_superset():
    alias = {r["ref"]: r["targets"][0] for r in LIVE}
    filt = [make_dataclass(c.__name__, [(f.name, object) for f in fields(c) if f.name in alias]) for c in REFS]
    res = check_superset(TGT, filt, alias); assert res["verdict"] == "SUPERSET-HYPOTHESIS-PASS", res
    for r in LIVE: assert r["targets"] and set(r["targets"]) <= TGT_F, r
def test_exclusions():
    for r in ROWS:
        if r["equivalence"] == "exclusion":
            assert not r.get("targets") and (r["reason"] in REASONS or r["reason"].startswith("deferred:"))
            if r["reason"].startswith("deferred:"): assert r["reason"][9:] in DEFERRED_IDS
def test_parity_ids_passed():
    passed = passed_nodeids(os.environ["AMINX_REDSOX_OUTCOMES_READ"])   # call passed, no failed phase, not xfail
    for r in LIVE: assert r["parity_test_ids"] and set(r["parity_test_ids"]) <= passed, r
    sem = {t for r in LIVE for t in r["targets"]
           if any(i.split("::")[-1].startswith("test_knob_semantics_") and i in passed for i in r["parity_test_ids"])}
    assert NEW_FIELDS <= sem   # all fields of both Options + omit_aa, omit_aa_per_position, output_kind
```
Gate script `scripts/redsox/run_gate.py`, run on titanix as `bth run --project-slug aminx -- uv run
--no-sync python3 scripts/redsox/run_gate.py` (sidecar `scripts/redsox/run_gate.bth.toml`; a `bash`
argv would leave bathos unable to resolve the sidecar, `runner.py:57-63`). `REPO` = first ancestor of
`Path(__file__)` containing `pyproject.toml`. Steps 0–2 are `subprocess` calls with the argv shown;
the script writes `{rc, step2_passed, n_ids, n_mutant_runs}` to `$BTH_RESULTS_PATH` and exits 0
whenever it graded the tree (FAIL included), non-zero only on harness crash. `[outcomes]` disjoint:
`pass = "rc = 0 AND step2_passed = true AND n_ids > 0"`,
`fail = "NOT (rc = 0 AND step2_passed = true AND n_ids > 0)"`:
0. `uv run --no-sync python3 scripts/redsox/gate_ids.py --out $OUT` → per-wave `ids_<W>.txt` /
   `files_<W>.txt` (union of alias `parity_test_ids` + every `tests/knob_semantics`, `tests/port`,
   `tests/golden` test id, partitioned by wave; `__nonport__` holds all ids outside `tests/port/**` so
   `tests/port/conftest.py` is never loaded in that wave; the port self-test is its own wave
   `port_selftest` with `tests/port/targets/port_selftest.toml` + manifest).
1. `rc=0; for W in __nonport__ $(ls tests/port/targets/*.toml | xargs -n1 basename -s .toml | sort); do AMINX_PORT_WAVE=$W AMINX_REDSOX_SELECT=$OUT/ids_$W.txt AMINX_REDSOX_OUTCOMES=$OUT/outcomes.jsonl uv run --no-sync pytest -o addopts="" $(cat $OUT/files_$W.txt) || rc=1; done`
   (no `-m`). Every module under `tests/port/` (incl. selftest, `port_wave("port_selftest")`) declares
   `pytestmark = pytest.mark.port_wave("<wave>")` (missing marker = collection error); its ids go only
   to that wave's list. Hooks in `tests/conftest.py`, active **only when `AMINX_PORT_WAVE` is set**
   (step 2 runs with it unset, so its hooks are off and it reads outcomes via the read-only
   `AMINX_REDSOX_OUTCOMES_READ`): deselect (never skip) items whose `port_wave` ≠ `AMINX_PORT_WAVE`
   or whose nodeid ∉ ids file; `pytest.UsageError` if a listed id is not collected; a
   `pytest_runtest_makereport` hookwrapper copies `call.excinfo.typename` onto
   `report.user_properties`; `pytest_runtest_logreport` appends
   `{nodeid, wave, when, outcome, wasxfail, mutant, exc_type}` (`mutant` null on clean runs;
   `exc_type` read from `user_properties`, else null). `passed_nodeids`
   considers only records with `mutant is None`: an id passes iff its call phase passed in such a
   record with `wave == declared wave`, with no failed phase and no `wasxfail` among its `mutant is
   None` records; `skipped` never counts. `test_branch_coverage` groups records by `(mutant, nodeid)`:
   clean verdict uses `mutant is None`; row R's verdict uses `mutant == R.id` restricted to R's
   vehicle ids. Gate
   passes iff `rc==0` and step 2 passes. `tests/port/conftest.py` hooks return early if
   `AMINX_PORT_WAVE == "__nonport__"` (defensive).
1b. For each `branch_manifest` row with a pytest vehicle: group the row's `vehicle.nodeids` by
   declared wave (`__nonport__` for A0/B0 gates) and run one invocation per (row, wave), ids file =
   that group, `AMINX_PORT_WAVE` = that wave, `AMINX_PORT_MUTANT=<id>`, appended to `outcomes.jsonl`;
   the row's verdict pools every group (≥1 vehicle id whose `when=='call'` record under the mutant failed with
   `exc_type=='AssertionError'` ⇒ killed; only if no such kill exists and some non-Assertion failure
   occurred ⇒ `instrument_invalid`); exit status not
   folded into `rc` (judged by step 2).
1c. Sidecar vehicles (validated here, never run here). For each sidecar slug S with
   `branch_manifest` rows, the owning task (or the gate operator, when the record is stale) runs, from
   a committed tree with `git status --porcelain` empty, on titanix:
   `bth run --project-slug aminx --output-paths $SC/S/<git-sha8>/branch_controls.json -- uv run --no-sync python3 scripts/parity/S.py --mutants <S's row ids, sorted, comma-joined> --controls-out $SC/S/<git-sha8>/branch_controls.json`
   with `SC=${AMINX_SIDECAR_OUT:-$HOME/.aminx/sidecars}` (absolute, outside every worktree, never
   `/tmp`); then commit `sidecar_ledger.toml` with the new id. The script writes
   `branch_controls.json` = `{clean: "pass"|"inconclusive"|"fail", mutants: {id:
   "failed"|"passed"|"error"}, weights: {path: sha256}}` — `clean` is the §7.3 band label of the
   unmutated measurement; a mutant is `"failed"` only if its measurement completed and landed in S's
   §7.3 fail band (for sidecars without an inconclusive band: "otherwise"); an exception gives
   `"error"`; `weights` lists every checkpoint loaded; the clean arm and **each mutant arm run in a
   fresh subprocess** (a jit/filter_jit cache traced by an earlier arm would otherwise hide a
   trace-time mutant; harness unit test shows a trace-time mutant changes the output) — and
   `{clean, n_listed, n_failed}` to
   `$BTH_RESULTS_PATH`. S's `.bth.toml` `[outcomes]` are disjoint, with K = `n_listed > 0 AND n_failed =
   n_listed`: `pass = "clean = 'pass' AND K"`, `inconclusive = "clean = 'inconclusive' AND K"`,
   `fail = "NOT (clean IN ('pass','inconclusive') AND K)"` (bathos `outcome` = name of first matching
   label, `sidecar.py:696-712`); the ledger accepts only `outcome=='pass'`.
   The default `resolve_run(id)` reads the cool-tier fragment `<catalog>/runs/aminx/run_<id>.parquet`
   (full id; `<catalog>` = aminx `.bth.toml` `catalog_dir`, else `~/.bth/catalog`; `catalog.py:38-47`,
   columns `schema.py:27-54`) with pyarrow (added to the aminx dev group in T0.4); it never reads the
   warm tier and the gate never runs `bth compact` (an in-run compact freezes in-flight rows,
   `compact.py:957-962`). It asserts: `status=='completed'` and `outcome=='pass'`; `git_dirty` false;
   `git merge-base --is-ancestor <git_hash> HEAD`; `git diff --name-only <git_hash>..HEAD` touches
   nothing under `src/aminx/**`, `scripts/parity/**`, `scripts/recapture/**`, `tests/port/**`,
   `aminx-oracles/**`, `pyproject.toml`, `uv.lock` (the ledger itself is out of scope); every
   `branch_controls.json` `weights` SHA-256 equals the checkpoint registry's `sha256` (the converted
   artifact prep loads; required on every `pottsmpnn_*`/`lasermpnn_*` entry, §4.1) at HEAD;
   `sidecar_sha256 == sha256(scripts/parity/S.bth.toml)` at HEAD (raw bytes, `sidecar.py:434`); the
   `--mutants` value in `argv`, as a set, equals S's manifest row ids; an `output_paths` entry ends in
   `/S/<git_hash[:8]>/branch_controls.json`, the file exists, its `mutants` key set equals S's row ids,
   every value `"failed"`, and `clean=="pass"`. Any failed assertion or missing record →
   `instrument_invalid`. The gate host must hold the catalog (titanix).
2. `AMINX_REDSOX_OUTCOMES_READ=$OUT/outcomes.jsonl uv run --no-sync pytest -o addopts="" tests/knob_gate -q`
   (`AMINX_PORT_WAVE` unset ⇒ outcome hooks off)
   (incl. `test_branch_coverage`). `OUT=${AMINX_GATE_OUT:-$REPO/outputs/gate/<utc-ts>}` (absolute;
   `/outputs/` is gitignored, T0.4); `run_gate.py` sets, in every child env, `AMINX_PORT_AUDITS_PATH=$OUT/port_audits.jsonl`.
redsox U1 reachability runs only as a smoke check (presence-only). T0.4 runs the gate on current
aminx first (expected FAIL = implementation checklist). Follow-up filed: redsox tech debt #2071
(redsox workspace) — CLI multi-target + recursion + per-class aliases + exclusions + behavioural U1.

## 7. Parity

### 7.1 Oracles and shims

- Upstreams vendored read-only at pinned SHAs: `/home/marielle/repos/PottsMPNN` @ `0cb0a58`,
  `/home/marielle/repos/LASErMPNN` @ `e70f2c6d`.
- Oracle env: separate project `aminx-oracles/` (torch, torch_scatter, torch_cluster, prody) on
  titanix; never in aminx's env.
- `scripts/parity/dump_{potts,laser}_oracles.py`: `torch.manual_seed(0)` before model construction;
  record missing/unexpected keys; each fixture in f64 and f32; sealed `.npz` (inputs, per-layer
  intermediates, outputs) + `oracle_manifest.toml` (upstream SHA, weights SHA-256, dump SHA-256,
  precision, `shimmed`, `shim_sha256`, `shim_sites`). Oracle generation is a bathos run.
- **Shims** (`aminx-oracles/shims/`, monkeypatched; vendored sources never edited):
  1. Draws (`torch.multinomial`/`Categorical.sample` at `run_utils.py:174,263`,
     `potts_mpnn_utils.py:1480,1590`, LASEr `model.py:627,859`; χ sites `model.py:899` (`sample`) and
     `:676`,`:677` (`tied_sample`, structure 1 then 2 per χ index); entropy-decoder sites `:1065,1104`
     excluded) →
     `c = cumsum(p.double(), -1)`; `i = searchsorted(c, u·c[-1], right=True)`;
     `i = min(i, last index with p>0)`, with an injected f64 uniform stream indexed by
     (sample, step[, χ]). aminx implements the identical rule in **all** driver draws, in f64 under
     x64 (parity tiers) and in `p.dtype` otherwise (injected-uniform parity is structural, not a
     test-only path). Tests: zero-probability tail token never drawn; `u = 1−2⁻⁵³` (f64) and
     `1−2⁻²⁴` (f32) return the last positive-probability index.
  2. Order: PottsMPNN native `decoder(decoding_order=)` / `optimize_sequence(decoding_order=)`
     (no patch); LASEr order draw patched at `utils/pdb_dataset.py:1641` (`rand_urns = torch.rand(...) + mask_idx`
     in `_masked_sort_for_decoding_order`; module-local shim, other `torch.rand` uses untouched).
  3. Dropout (`_VDropout` never patched; the dump asserts every `_VDropout.training is False`):
     LASEr `nn.Dropout.forward` → `x*mask/(1-p)` with injected masks keyed by (module path,
     call idx); aminx proofread stage accepts the same masks.
- Exact-token tiers are **f64 only**; f32 reports match rate and permits a mismatch only where
  `min_k|cdf_k − u| < 1e-6` (logged), else fail.

### 7.2 Tier contract (derived from xtrax `port/` @ xtrax `35c5100`; extensions listed)

Per wave `tests/port/targets/<wave>.toml`: `[port] wave_id, symbol_qualname, oracle_id
("ref:tests/port/reference/<wave>:v1:sha256:<hash>"), reference_subtree`; `[capabilities]
stochastic, dynamic_shape`; `[parity] ad_critical=false, ad_critical_justification="",
tolerance_policy_f64, tolerance_policy_f32, max_traces`; `[access] reference_write_identities,
fixer_read_only_on_reference=true`. `tests/port/manifests/<wave>.toml` with `manifest_hash`
verified by conftest. Sealed `tests/port/reference/<wave>/algo.py` (`# REFERENCE: DO NOT MODIFY`)
loads the sealed `.npz` by hash. Markers `tier_1..tier_5` in aminx `pyproject.toml`; conftest
enforces T1→T2→T3→(T4 iff ad_critical)→T5; `domain=port` records to `.praxia/audits.jsonl` in xtrax
record shape. **Extensions vs xtrax:** (1) two tolerance keys (xtrax has one, `port_target.toml:17`);
the audit record carries the policy of the tier that ran; (2) `AMINX_PORT_WAVE` overrides `[tool.port]
target` (xtrax reads pyproject only, `conftest.py:64-71`); CI iterates waves; (3) conftest reads
`stochastic`: a stochastic wave without an injected-uniform oracle key is a T2/T3 error; (4) T2 runs
in a scoped `jax.experimental.enable_x64()` fixture; (5) `pytest-timeout` added to dev deps
(`conftest.py:214,232`); (6) `AMINX_PORT_AUDITS_PATH` overrides the `domain=port` audit path (xtrax
hardcodes `REPO_ROOT/.praxia/audits.jsonl`, `conftest.py:277`); `run_gate.py` sets it to
`$OUT/port_audits.jsonl` so the gate leaves the tracked tree clean. T5 uses `max_traces`.
**Self-test first (T0.3):** sign-flipped kernel must fail T2; retrace-per-call kernel must fail T5;
`tests/knob_gate/test_branch_coverage.py` delegates to `check_branch_coverage(manifest_path,
outcomes_path, ledger_path, *, resolve_run, changed_paths, is_ancestor) ->
Literal["pass","fail","instrument_invalid"]` in `tests/knob_gate/_coverage.py` (callables default to
the cool-tier parquet reader (§6.6 step 1c), `git diff --name-only <h>..HEAD`, `git merge-base --is-ancestor <h> HEAD`). The self-test
injects fakes over synthetic fixtures in `tests/port/selftest_coverage/`, one per branch, asserting:
(i) planted row `selftest_noop` whose mutant run passes → `instrument_invalid`; (ii) row with no
mutant record → `instrument_invalid`; (iii) clean vehicle id failed → `fail`; (iv) all valid (clean
passes, pytest mutant fails, sidecar record valid) → `pass`; (v) one sidecar fixture per broken
condition → `instrument_invalid`: `git_dirty` true; `sidecar_sha256` mismatch; `outcome != "pass"`;
argv mutant set ≠ manifest rows; a controls file with one mutant `"passed"`; missing controls file;
record commit not an ancestor of HEAD; diff touching a scoped path; `outcome == "inconclusive"`;
weights SHA ≠ registry; `status != "completed"`; `clean == "fail"`; controls `mutants` key set ≠
manifest rows (extra and missing); `output_paths` directory ≠ `git_hash[:8]`; slug with no ledger
entry — every step-1c assertion has exactly one (v) fixture; (vi) default `resolve_run` against a
temporary catalog holding only a cool-tier fragment resolves every field, missing fragment →
`instrument_invalid`; (vii) row `selftest_raise` whose mutant raises `RuntimeError` →
`instrument_invalid`; (vii-b) a row with one AssertionError-killed id and one RuntimeError id →
killed (not invalid). Follow-up: `xtrax.port` pytest plugin in the wheel.

| Wave | Symbol | stochastic | tol f64 / f32 | max_traces |
|---|---|---|---|---|
| `potts_head` | `PottsHead` | no | rtol=1e-10,atol=1e-12 / rtol=1e-5,atol=1e-6 | 1 |
| `potts_merge_pair_d2` | `merge_pair(denom=2)` | no | exact / atol=1e-7 | 1 |
| `potts_merge_pair_d4` | `merge_pair(denom=4,exclude_self)` | no | exact / atol=1e-7 | 1 |
| `potts_energy` | `potts_energy` | no | rtol=1e-10 / \|err\| ≤ 1e-5·Σ\|terms\| (r15) | 1 |
| `potts_ar_decode` | `PottsARDecode` (+tied, PSSMMix) with injected uniforms/order; fixture with `fixed_positions` (`chain_M_pos=0` rows); negative control AR `m = present·chain_M_pos` must fail | yes | exact tokens / match rate | 1 |
| `potts_refine` | `PottsRefine` (all modes) injected uniforms | yes | exact tokens / match rate | 1 |
| `pottsmpnn_full` | etab_forward + teacher-forced log-probs | no | atol 1e-8 / log-prob rtol=1e-4,atol=1e-4 (r16) | 1 per bucket |
| `laser_layers` | GATv2/GVP/LN | no | rtol=1e-9 / rtol=1e-5 | 1 |
| `laser_encoder` | encoders | no | rtol=1e-8 / rtol=1e-4 | 1 per bucket |
| `laser_score` | teacher-forced seq+χ log-probs | no | atol 1e-8 / 1e-4 nats | 1 per bucket |
| `laser_decode_step` | one AR step, injected uniforms | yes | exact / match rate | 1 |
| `laser_rotamers` | RotamerBuilder | no | atol 1e-9 Å / 1e-4 Å | 1 |

Tolerances fixed in the wave TOML before first run; loosening only via a new bathos run citing the
measured deviation.

### 7.3 Bathos sidecars (committed before running)

Invocation `bth run --project-slug aminx -- uv run --no-sync python3 scripts/parity/<name>.py …`
(titanix; L1/L2 local gates first); verification by record (the cool-tier fragment
`run_<id>.parquet`, or `bth compact` then `bth sql "SELECT id,status,outcome,exit_code,command FROM
runs WHERE id LIKE '<p>%'"` after the run has finished) + spot-check output paths.
Every sidecar declares `pass`/`inconclusive`/`fail`, a measured-path negative control that must FAIL
(else `instrument_invalid`), and a synthetic ground truth where applicable.

| Sidecar | Data (path+SHA-256 in TOML) | pass | inconclusive | fail | Negative control / ground truth |
|---|---|---|---|---|---|
| `runner_goldens` (T0.5a in-memory at pre-refactor SHA; T0.5b Zarr at T0.0 merge SHA — Zarr paths TypeError before T0.0) | matrix {sample in-memory, sample Zarr, score nll, jacobian in-memory, jacobian Zarr, inspect} × 2 MPNN ids (1 proteinmpnn, 1 ligandmpnn) × 2 fixtures, pinned; `uv.lock` SHA-256, jax/jaxlib versions, titanix GPU index recorded; capture + compare same host/GPU | arrays byte-equal + dtype + shape; attrs equal on root and every group after dropping `_CORE_PROVENANCE_FIELDS` (`zarr_sink.py:34`); `metadata["specification"]` compared only on field names present at capture SHA; other metadata keys exact | — | any diff | `random_seed=1` must differ; planted change to one non-provenance attr must fail |
| `potts_energy_parity` | `PottsMPNN/inputs/example_pdbs/*` + L=30 chain + 200 random seqs/structure | \|ΔE\| ≤ 1e-4 + 1e-5·\|E_up\| | ≤ 10× bound | > 10× | permute `etab_out` rows → fail; hand-built 3-residue etab with analytic E → 1e-12 (f64) |
| `potts_ddg_megascale` | `energy_benchmark_datasets/megascale_test_subset.csv` (all rows; n in TOML) | max \|Δddg\| ≤ 1e-4 | (1e-4,1e-3] | > 1e-3 | skip transpose in `merge_pair` → fail |
| `potts_ar_refine_exact` | example_pdbs, 50 seeds, injected uniforms/order, f64 | exact match = 1.0 | [0.99,1.0) | < 0.99 | wrong partition sign; N→C order; AR `m = present·chain_M_pos` on a `fixed_positions` fixture → fail |
| `potts_sample_dist` | distributional protocol, Potts conditions | see below | | | T×m; N→C (plain, T=1.0) |
| `laser_score_parity` | `4jnj-1_prot.pdb` + 20 complexes from the upstream Zenodo PDB dataset (list, source URL and per-file SHA-256 pinned by B1(a); the pinned repo's `databases/` holds no structures); entries 1–4 double as the B0 fixtures | max \|Δ log-prob\| ≤ 1e-4 | (1e-4,1e-3] | > 1e-3 | permute one decoder layer → fail |
| `laser_decode_e2e` | same 21, injected order, argmax | exact seq + χ-bin = 1.0 AND max circular \|Δchi_deg\| ≤ 1e-6° (f64) on chi_mask | — | otherwise | reversed order → < 1.0; aminx χ bin +1 mod Nbins → fail; offset := 0 → fail. Tied fixture (injected order + uniforms, χ streams indexed (sample,step,χ,structure); oracle calls `model.tied_sample` directly): exact sequence + χ₁/χ₂ bins, chi_deg ≤1e-6°; negative controls χ₂:=χ₁ → fail, λ on logits → fail |
| `laser_sample_dist` | distributional protocol, LASEr conditions | see below | | | T×m |
| `laser_proofread_parity` | 5 complexes, injected masks | max \|Δ mean\|, \|Δ std\| ≤ 1e-4 | (1e-4,1e-3] | > 1e-3 | reduction swap (std over reps) → fail; ddof=0 → fail; scalar dropout off → \|Δmean\|>1e-3 somewhere; vector dropout on → fail. Ground truth: n_orders=2, n_dropouts=1, injected masks: `proofread_std=\|p₁−p₂\|/√2` (f64 exact); n_orders=1: mean = single cell, std all-NaN. Plus `test_knob_semantics_proofread_dropout` |
| `knob_semantics_<knob>` (pytest, collected by gate) | per-knob fixture where the knob binds | set ≠ default on a pre-registered observable AND set case matches oracle/analytic | — | no difference or mismatch | built-in (differential); `no_op` rows invert: set == unset byte-identical in aminx and oracle |

**Distributional protocol.** Runs per (structure s, condition c, temperature T): aminx A, upstream
U1 (shimmed, iid u), U2 (unshimmed), n=1000 each, disjoint seeds. Per designed position i
`TV_i(X,Y)=½Σ_a|p̂_X(a)−p̂_Y(a)|`, `D(X,Y)=mean_i TV_i`, `Δ_s = D(A,U1) − D(U2,U1)` (equal n cancels
plug-in bias to first order). Bootstrap B=2000 resampling sequences within A, U1, U2 independently.
Per (c,T), pooled `Δ̄ = mean_s Δ_s`: **pass** = 95% one-sided upper bound < δ AND `max_s Δ_s < 2δ`;
**fail** = 90% CI lower bound > δ; else **inconclusive**. Sidecar pass iff all (c,T) pass; fail if
any fail. Default δ = 0.02 (TV units), pilot may set within [0.01, 0.05]. LASEr χ1: positions where
upstream's modal AA has χ1, restricted to samples with that AA, TV over 36×10° bins, δ_χ = 0.05.
Potts conditions: `plain` (mode none, no PSSM/ties) T∈{0.1,0.3,1.0}; `pssm` (fixture PSSM,
`pssm_multi=0.5`, bias flag on, mode none) T=0.3; `refine` (`potts`, `optimization_temperature=0.5`,
AR T=0.3); `tied` (2-member fixture, β=1) T=0.3; default T_opt=0 covered only by the exact tier.
LASEr conditions: T∈{0.1,0.3,1.0} min_p=0; T=0.3 min_p=0.05. Negative controls (must fail):
near-margin `T_A = m·T` (m from pilot, default 1.25); N→C order (Potts plain T=1.0); r1 gross
controls secondary.

**Runs.** U3 = second shimmed upstream run with fresh iid uniforms, seeds disjoint from U1 (a null
replicate of U1). **Structures (pinned path+SHA-256).** Potts pilot `inputs/example_pdbs/{2yc3,3dkm}.pdb`;
Potts confirmatory `{3gg7,4jox,6w25,swe1_ligand}.pdb`; `tied` uses the pinned 2-member tied fixture.
LASEr pilot: entries 1–2 of the `laser_score_parity` list; LASEr confirmatory: `4jnj-1_prot.pdb` +
entries 3–6. **Pilot** (`*_pilot.bth.toml`, exploratory, cites nothing), per (c,T) at n=1000: U1,
U2, U3, control `C_m` = shimmed upstream at `m·T`, m∈{1.1,1.25,1.5}. Null statistic
`Δ⁰_s = D(U3,U1) − D(U2,U1)`; `ĥ` = half-width of the two-sided 90% bootstrap CI (B=2000) of
`mean_s Δ⁰_s` (unscaled; conservative for S_conf ≥ S_pilot); `q̂_s` = max over pilot structures of
the 97.5th bootstrap percentile of `Δ⁰_s`. `δ = clip(max(2ĥ, q̂_s/2), 0.01, 0.05)`;
`Δ_neg(m) = mean_s[D(C_m,U1) − D(U2,U1)]`; choose smallest m with `Δ_neg(m) ≥ δ + 2ĥ`. Escalation:
if `max(2ĥ, q̂_s/2) > 0.05` or no m qualifies → n=4000, re-pilot once; still failing →
`instrument_invalid`. Confirmatory: the U1 bootstrap resample is shared by both terms of `Δ_s`.
**Shim check** (every (c,T)): `mean_s[D(U2,U1) − D(U3,U1)] > ĥ` → `instrument_invalid` (shim
distorts sampling). δ, m, n, ĥ committed into the confirmatory sidecar before it runs; its negative
control is aminx at `m·T`.

## 8. Fixer tasks

| ID | Task | Depends | Gate |
|---|---|---|---|
| T0.0 | **DONE on main (`9e6c340a`), r13; r14 corrected to what main actually does.** **Sink run_id fix (prerequisite).** No aminx helper (`sink_ids.py`/`sink_spec_for`/`spec_run_id` do not exist). Sites with a `RunSpec` call `xtrax.run.derive_sink_spec(spec.run_spec, output_dir=..., format="zarr", flush_every=...)` (`sink.py:61-105`) directly: `host/streaming.py:88-91`, `host/runner.py:1336-1341` (jacobian), `sampling/multistate_poe.py:689`. `run_spec.run_id` is never set (static field → retrace), so precedence falls to `new_run_id()`: a fresh, unlinked id per sink; deriving it from job/spec identity is a separate open design question. Re-run policy = xtrax's own: a store whose root `run_id` differs raises `ValueError` at sink construction (`zarr_sink.py:214-222`); no aminx-level check. Spec-less site: `DesignZarrWriter(..., run_id=None)` (`io/designs.py:74,107-109`) defaults to `new_run_id()`, accepts an explicit id to reopen. ~~**Missed:** `scripts/analysis/jacobian_profile.py:187` still builds `SinkSpec` without `run_id` (TypeError; non-gating follow-up fix via `new_run_id()`)~~ **FIXED 261002 (`0af362a1`)** via `new_run_id()`, as prescribed. The defect was total, not partial: `run_id` is the first field, has no default, and `__post_init__` rejects a non-str, so every `--zarr-out` invocation raised before writing a byte. Verified end to end — the store writes, `zarr_content_digest` resolves, and a second differing `run_id` into the same directory raises `ValueError` rather than interleaving two profiles. `zarr_run_id` is now recorded in the payload. Swept untruncated: exactly two `SinkSpec(` constructions exist repo-wide (this and `src/aminx/io/designs.py:108`), both pass `run_id`; every other site goes through `derive_sink_spec` | T0.5a | done on main for the four migrated sites (`tests/io/test_designs.py`, `tests/host/test_multistate_poe_campaign_integration.py`); driver sink follows the same `derive_sink_spec` call; goldens compare arrays + non-provenance attrs only |
| T0.1 | Vendor upstreams at pinned SHAs; `aminx-oracles/` env on titanix | — | files + manifest |
| T0.2 | Probe report (§10) | T0.1 | DONE: `research/260929_potts-laser-t02-probe-report.md`; corrections applied in r12 |
| T0.3 | `tests/port/` contract + self-test; `tests/knob_gate/_coverage.py` + `selftest_coverage/` fixtures (i)–(vii); `branch_manifest` / `sidecar_ledger` schemas (`tests/knob_gate/schemas/*.json`) | — | selftest on titanix |
| T0.4 | Extractor, reference surfaces, alias skeleton, exclusions, harness, `run_gate.py` + `run_gate.bth.toml`, pyarrow in dev group, `test_branch_coverage.py` wiring, `/outputs/` gitignore (bathos creates `outputs/<id8>` after capturing git state, `runner.py:901-903`), empty `branch_manifest.toml` + `sidecar_ledger.toml` (schemas from T0.3), mutant hook; run vs current aminx (FAIL baseline); file debt for `run_predict_partial_charges` (deferred items #2067–#2070 and redsox #2071 already filed 260929) | T0.1, T0.3 | bathos run FAIL recorded |
| D1 | **Resolve debt #2051**: `Aminx.__call__` honours `inference` (no encoder dropout at inference on freshly constructed models); runner path already forces `inference_mode` (`host/prep.py:200`) so runner outputs are unchanged; fallback purposes of PottsMPNN (`model.mpnn`) depend on this | — | unit test: two inference calls on a freshly constructed model are bit-identical; negative control dropout-on differs; runner outputs unchanged (T0.5a captured after) |
| D2 | **SUPERSEDED on main by #160 (`random_design_order`); not carried after the r13 rebase.** ~~Resolve debt #2017~~: MPNN `random_decoding_order` places fixed/non-designed positions first as the reference does (`argsort((chain_mask+1e-4)·\|randn\|)`); intentional change to MPNN sampling order when fixed positions are set | — | parity test vs reference ordering rule on a fixed-positions fixture; negative control (old order) fails; outputs without fixed positions byte-identical |
| T0.5a | Runner goldens, in-memory rows, captured **after D1+D2 land** (goldens freeze corrected behaviour; §7.3 row) | D1, D2 | `runner_goldens` sidecar (capture) |
| T0.5b | Runner goldens, Zarr rows, at T0.0 merge SHA | T0.0 | `runner_goldens` sidecar (capture) |
| T0.5 | FamilyDriver protocol, registry, runner dispatch + raise, `run_family_driver` (ZarrStagingSink channel, keys, ids/skips, inference mode), family Literal/derivation/consumer branches, `DecoderLayer` promote_types accumulation, L-DRV lint | T0.0, T0.2, T0.5a, T0.5b | unit tests; lint; goldens exact on titanix |
| T0.6 | `output_kind`, Options, `omit_aa*` (+RS-6b list), temperature family default, CLI flags, spec_json/campaign round-trip, fail-loud validation | T0.5 | unit tests; goldens exact |
| A0 | PottsMPNN host featurizer (§4.1a) | T0.1, T0.2 | A0 gate |
| A1 | PottsMPNN oracle dumps (f64+f32, shims, `declayer_f64.npz`) | T0.1, T0.2 | manifest |
| A2 | `PottsMPNN` composition + conversion + self-edge test | T0.3, T0.5, A0, A1 | wave `pottsmpnn_full`; `DecoderLayer` f64 fixture test |
| A3 | `PottsHead`, `merge_pair`, `potts_energy`, §4.1a masks, alphabet conversion | T0.3, A0, A1 | waves `potts_head`, `potts_merge_pair_d2/d4`, `potts_energy` |
| A4 | Driver `score:energy|ddg`, mutants/DMS, partitions axis, sinks | A2, A3, T0.6 | `potts_energy_parity`, `potts_ddg_megascale` |
| A5 | `PottsARDecode` (+tied, PSSMMix), `PottsRefine` (all modes, binding, order quirk), `PottsSampleEnergy` + host ranking, all Potts Options | A4 | waves `potts_ar_decode`, `potts_refine`; `potts_ar_refine_exact`; pilot → `potts_sample_dist`; knob tests |
| A6 | ADR + lint updates | A2 | lint green |
| B0 | LASEr host featurizer | T0.1, T0.2, B1(a) | field-level parity (5 fixtures) |
| B1 | (a) **Fixture acquisition:** fetch the upstream protonated-PDB dataset from Zenodo (chunks `10.5281/zenodo.17990180` + `10.5281/zenodo.17990253` per `databases/README.md`, reassembled with `cat`, md5 `c9418cb9368c8068a6053feebbff5fda` as documented upstream, ~50 GB, PDB format; note `download_protonated_pdb_training_dataset.sh` fetches a different Zenodo record, `https://zenodo.org/records/15035128/files/all_data.zip`, the shelve database, which is not a PDB source), select the 20-complex `laser_score_parity` list (ids and selection rule recorded; entries 1–4 = B0 extras), extract only those PDBs, and commit `tests/fixtures/laser/fixtures_manifest.toml` (source URL, archive SHA-256, per-file SHA-256, chosen ids); run on titanix as a bathos-tracked script (bulk download is a data-acquisition run, its record is the provenance); (b) LASEr oracle dumps | T0.1, T0.2 | (a) manifest with SHA-256s resolves and files hash-match; (b) manifest |
| B1.5 | LASEr weight conversion | B1 | 0 unmapped keys |
| B2 | layers | T0.3, B1.5 | wave `laser_layers` |
| B3 | encoders + ligand bucket axis | T0.3, B0, B2 | wave `laser_encoder` |
| B4 | LaserDriver `score:nll|logits`, alphabet boundary | T0.3, B3, T0.6 | wave `laser_score`; `laser_score_parity` |
| B5 | `LaserJointDecode` + §5.3.1 | T0.3, B4 | wave `laser_decode_step`; `laser_decode_e2e`; pilot → `laser_sample_dist` |
| B6 | RotamerBuilder (post) + PDB/FASTA sinks | T0.3, B5 | wave `laser_rotamers` |
| B7 | proofreading (both), two-structure tied, remaining Laser Options | B5 | `laser_proofread_parity`; knob tests |
| Z1 | Operator, on titanix from a clean committed worktree at the final tree: re-runs (per §6.6 step 1c) every vehicle sidecar whose ledger record fails the ancestor or diff-scope check, commits `sidecar_ledger.toml`, then runs the gate | A5, A6, B6, B7 | Gate cool-tier record: `status=='completed'`, `outcome=='pass'`, `sidecar_sha256 == sha256(run_gate.bth.toml)` (non-empty), `git_dirty` false; includes `test_branch_coverage` |
| Z2 | CLI docs + `using-aminx` skill | Z1 | — |

**Stage → vehicle (every stage in the `[[branch]]` `stage` column must be listed here; the Vehicles
column lists primary vehicles only — any vehicle allowed by §0 may back a row, e.g.
`knob_semantics_refine_order`, `_tied_last_member`, `_tied_masked_member`, `_t0_floor`,
`_pssm_precedence`, `_x_gap_energy`, `_skip_gaps`, `_laser_bias_minp`, `_laser_stored_logits_minp`,
`_fs_sequence_temp`, `_laser_score_order`, `_laser_noise`, `_tied_refine_frozen_group`,
`_tied_epistasis_leaked_pos`, `_tied_binding_no_reference`, `_tied_nodes_member_skip`, the `no_op`
inverse differentials of §6.4, `test_tied_rank_flat_matches_upstream`):**

| Stage | Vehicles |
|---|---|
| A0 / B0 featurizers | A0 gate / B0 gate |
| Order generation (Potts AR key, Potts refine fresh key, LASEr 3-tier) | `test_knob_semantics_order_generation`: Potts — oracle and aminx given the same `randn`, exact order on a fixture with `fixed_positions` + a gap row, for AR and refine keys; LASEr — `torch.rand` in `_masked_sort_for_decoding_order` shimmed with injected per-tier uniforms, aminx given the same stream, exact order on (a) an inference-featurized fixture (tier 2 empty, as upstream inference always produces, §5.4a) and (b) an oracle batch with fixed/non-contact rows and `batch.extra_atom_contact_mask` set by hand for the contact rows (the upstream inference featurizer cannot produce tier 2). Mutants: drop tier offset; drop `chain_M_pos` from AR key; swap AR↔refine keys |
| MPNNEncode (Potts) | `pottsmpnn_full` |
| PottsHead, merge_pair d2/d4, potts_energy | their waves; `potts_energy_parity` |
| PottsARDecode (+tied, PSSMMix) | `potts_ar_decode`; `potts_ar_refine_exact` |
| PottsRefine (all modes, tied, `skip_calc`) | `potts_refine`; `knob_semantics_nodes_gibbs`, `_binding_converge_stop`, `_optimize_pdb`, `_optimize_fasta` |
| PottsSampleEnergy + host ranking | `potts_ar_refine_exact` (compares `sample_energy`, `sample_rank`); `knob_semantics_rank_before_refine` |
| Partition fuse (binding ddG) | `knob_semantics_binding_partition_ddg` (2-chain fixture, oracle per-partition `get_etab`; mutant `+Σ_p`) |
| score:energy/ddg candidates | `potts_energy_parity`, `potts_ddg_megascale` |
| LASEr layers / encoders / score | `laser_layers`, `laser_encoder`, `laser_score` + `laser_score_parity` |
| LaserJointDecode step / multi-step | `laser_decode_step` / `laser_decode_e2e` + §5.3.1 knob tests |
| LASEr tied decode | `laser_decode_e2e` tied fixture; `knob_semantics_tied_lambda`, `_tied_chi_nan_fixed` |
| Proofread (both) + fuse | `laser_proofread_parity`; `knob_semantics_proofread_dropout` |
| RotamerBuilder | `laser_rotamers` |

## 9. Risks

| Risk | L | Mitigation |
|---|---|---|
| `run_family_driver`/temperature-default refactor perturbs MPNN path | M | T0.5a exact goldens |
| merge_pair / padding edge cases | H | exact d2/d4 waves; padding-invariance test; L=30 & short-partition fixtures |
| PottsARDecode row-kernel diverges from full-sequence DecoderLayer | M | wave `potts_ar_decode` f64 exact on injected path; `pottsmpnn_full` teacher-forced |
| LASEr scan compile time | M | O(n_dec·K) step; jaxpr guard; bucketed L |
| Ragged ligand / partition recompiles | M | bucket axes; T5 `max_traces` |
| Oracle env (torch_scatter/cluster) | M | `aminx-oracles/` on titanix; sealed dumps |
| Semantic mismatch behind aliases | H | namespaced refs; parity ids must PASS (junit); differential knob tests for all new fields |
| Stochastic parity false pass/fail | M | shims + f64 exact tiers; TOST-style protocol with piloted δ and near-margin controls |
| Upstream `strict=False` random weights | M | T0.2 key audit blocks |
| Oracle fixture size | L | dumps on titanix, hashes in repo |

## 10. Probe report

Resolved during authoring (260929): `EncoderOutput` carries `edge_features`/`neighbor_indices`
(`types/bundles.py:457-473`); no CA featurizer; aminx `bias`/`fixed_positions`/`fixed_tokens`/
`fixed_mask`, no omit; `temperature` is `Sequence[float] | float` default 0.1.

**Resolved by T0.2** (see `research/260929_potts-laser-t02-probe-report.md`, "report §n"; anchors are
pinned-upstream or worktree `file:line`; corrections C1–C12 and blockers B1–B3 are applied in r12 above).
The torch probes (key audits, tensor equalities, dropout enumeration, `4jnj` first-shell/`resindex`/graph
facts) were **exploratory: run inline on titanix, no bathos sidecar**, and cite nothing as a finding; they
will be re-confirmed by the tracked A1/B1 oracle-dump bathos runs (and the A0/B0 gates for the
single-fixture facts) before any number is cited outside the report.
- Runner helpers (report §1): `_canonical_structure_id(s)_for_spec` `host/_sampling_helper.py:22,36`
  (reads only `spec.inputs`, any spec type); `_structure_ids_for_batch` `:48`; `resolve_target_samples`
  `host/plan.py:382`; `make_axis_dispatch_via_xtrax` `tiling/dispatch.py:161` (docstring "not wired" is
  stale, ~15 call sites); entry points `host/runner.py:79,480,881,1220` (r14; were `60,441,820,1143`); `sample()` metadata keys
  `specification`, `skipped_inputs`, `structure_ids`, `lineage`; both aminx Zarr sites built
  `SinkSpec` without `run_id` (T0.0 premise confirmed; migrated on main, r14).
- `model_family` consumers (report §2): complete classified list; only new runner reader is the
  dispatch; portable-JSON guard `run_spec_portable_json.py:153` must become `!= "proteinmpnn"`;
  `_prepare_ligand_context` and the `prep.py` registry lookup are the only fallback-path consumers.
- PottsMPNN keys (report §3): in-scope 5 checkpoints missing/unexpected = `[]`; `proteinmpnn_compatible/*`
  miss `etab_out.*` (B1, excluded); `ft/potts_ft` = same config, 120/120 tensors differ; compat trunk
  118/118 bit-equal; T floor = `1e-6` on exact `0` only (`sample_seqs.py:38-39`).
- Verbatim transcriptions (report §4): `decoder :1415-1488`, `tied_decoder :1599-1687`,
  `optimize_sequence`/`nodes :75-271`, `tied_optimize_sequence :273-519`, refine order keying,
  `skip_calc :126-159`; tied-refine quirks now §6.5/§6.5b.
- aminx side (report §5): only hardcoded-float32 on the core is `decoder.py:338-341`; slot-0 self
  edge holds for `present` rows only (gap/pad rows tie, `features.py:143-186`).
- LASEr (report §6): `model_params` dims Hs=256, V=10, E=128, Hl=256, Vl=15, K=48, Kl=48, k_ll=5,
  cutoff 20.0, Nbins=72; the 3 LASEr checkpoints load `strict=True` and carry identical params; ligand
  encoder keys ARE in the main checkpoint (83 keys; pretrained file not needed);
  `LASER_ALPHABET` = `'ARNDCEQGHILKMFPSTWYVX'` (`utils/constants.py:34`, non-trivial E/Q permutation);
  graph sites G1–G10 enumerated with dense replacements (`radius_graph` has zero call sites); draw
  sites `model.py:859,899` / tied `:627,676,677`, order draw `pdb_dataset.py:1641`; dropout: 74
  modules (72 `nn.Dropout`, 44 with `p>0`, 1 `_VDropout`), `self.training` read once
  (`model_generics.py:631`); `tied_sample` ignored-knob list confirmed exactly; LigandMPNN scripts
  drive LigandMPNN (`duplicate` confirmed); fixtures not in repo (B2, B1(a)).
- Branch-coverage ids: `conditional_ids.txt` content is in report §7 (to be committed at
  `tests/port/conditional_ids.txt` by T0.3/T0.4).
- Not resolved by T0.2: the aminx invalid-neighbour debt entry (§6.5b last row) is drafted in report
  §8 but **not filed**; the orchestrator files it.
