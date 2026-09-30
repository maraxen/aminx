"""Declared expected behavior for every SamplingSpecification field in campaign mode.

This table is the audit's core artifact. It is three things at once:

1. **The harness spec** -- what `test_knob_differential.py` asserts per field.
2. **The triage record** -- the verdict for each field, with its evidence.
3. **The drift guard** -- a new spec field with no entry here fails CI.

Why declarations at all, rather than just diffing values through the pipeline: a raw
differential reports "value did not survive" for `multi_state_strategy` (a real bug) AND for
`random_seed` (not a bug -- rows are differentiated by a grid-lineage hash instead). The
mechanical pass produces *questions*; only declared intent turns them into verdicts. A harness
without this table cries wolf on day one, and a harness that cries wolf gets muted.

Every NOT_APPLICABLE/OVERRIDDEN reason must cite a mechanism or a line, never an assertion.
Judgment is precisely what failed the ten times this audit exists to answer for.

task_id: 260715_aminx-campaign-control-knob-audit
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
  """Base verdict. `reason` must cite evidence, not assert a conclusion."""

  reason: str


@dataclass(frozen=True)
class Bundle(Verdict):
  """Reaches the model as a `build_inference_bundle` argument. Tier B observable."""

  param: str


@dataclass(frozen=True)
class LoadModel(Verdict):
  """Reaches the model skeleton as a `load_model` kwarg. Tier B observable."""

  param: str


@dataclass(frozen=True)
class SpecOnly(Verdict):
  """Must survive the manifest round trip; affects the loop/plan, not a boundary arg.

  Tier A observable: the reconstructed spec's value must differ when the caller's does.
  """


@dataclass(frozen=True)
class Overridden(Verdict):
  """Campaign deliberately overrides the caller. Tier A asserts the override, not the input.

  Without this verdict a naive differential flags these as "not wired" -- false alarms.
  """

  value: object


@dataclass(frozen=True)
class NotApplicable(Verdict):
  """Genuinely inert in campaign mode. Asserted inert, so a future wiring change trips it."""


@dataclass(frozen=True)
class NonSerializable(Verdict):
  """Cannot cross the manifest JSON boundary (callables, live objects)."""


@dataclass(frozen=True)
class Internal(Verdict):
  """`init=False` plumbing, not a caller-facing knob."""


# ---------------------------------------------------------------------------
# Known-broken at a17. Each entry is xfail(strict=True) in the harness: it must FAIL
# today (proving the harness detects real bugs) and, once fixed, XPASS -> red build,
# forcing the marker's removal. Both directions guarded, permanently.
# ---------------------------------------------------------------------------
# CLOSED by S1a's field-driven manifest write. All twelve are gone from this set because they
# now genuinely survive the manifest -- verified by the suite going red (XPASS) until removed.
#
# Kept as a record rather than deleted: these twelve are what the audit was for, and the empty
# dict below is the claim "the manifest-drop class no longer exists". If any regresses, its Tier
# A test fails outright -- no marker to hide behind.
CLOSED_BY_S1A: tuple[str, ...] = (
  "fixed_mask",            # occurrence #7 -- the blocker; no residue was ever held fixed
  "fixed_positions",       # occurrence #7
  "fixed_tokens",          # occurrence #7
  "state_position_map",    # occurrence #8 -- found during this audit
  "multi_state_strategy",  # silently reverted to arithmetic_mean while the row hash claimed otherwise
  "state_weights",         # occurrence #10
  "bias",
  "tie_group_map",
  "structure_mapping",
  "ligand_mpnn_use_side_chain_context",  # manifest hop only; model hop still broken -- see TIER B
  "use_electrostatics",                  # ditto
  "use_vdw",                             # ditto
)

# Tier A: does the caller's value survive plan -> manifest JSON -> reconstructed spec?
# Empty after S1a. A new entry here means the manifest-drop class has returned.
KNOWN_BROKEN_TIER_A: dict[str, str] = {}

# Tier B: does it reach the MODEL? Empty after S3.
#
# Deliberately NOT emptied by S1a, and that split earned its keep. `ligand_mpnn_use_side_chain_context`,
# `use_electrostatics` and `use_vdw` were broken in TWO places -- dropped by the manifest AND
# never forwarded at prep.py -> load_model. S1a fixed only the first, at which point their Tier A
# tests went green while the knobs still did nothing: the model was built with sidechains off and
# physics_feature_dim checkpoint-name-derived, regardless of what the caller asked for. Twelve
# markers clearing did not mean twelve knobs worked. A single-tier harness would have declared
# victory here.
#
# S3 forwards all three (prep.py's model_conditioning dict). Removing their markers is what
# proved it: each went XPASS(strict) -> FAILED first, then green once removed -- the fix forces
# the marker off. That property only exists because these are declarative
# `request.node.add_marker(pytest.mark.xfail(strict=True))`; the imperative `pytest.xfail()`
# aborts the test body, so a test using it can NEVER XPASS and would have sat here lying
# indefinitely.
#
# A new entry here means a knob reaches the manifest and dies at the model boundary.
KNOWN_BROKEN_TIER_B: dict[str, str] = {}


# ---------------------------------------------------------------------------
# The table. One entry per SamplingSpecification field (88 at a17).
# ---------------------------------------------------------------------------
OBSERVATIONS: dict[str, Verdict] = {
  # --- reaches build_inference_bundle -------------------------------------------------
  "fixed_mask": Bundle(param="fixed_mask", reason="unioned with fixed_positions in _prepare_fixed_controls, _sampling_helper.py:482"),
  "fixed_positions": Bundle(param="fixed_mask", reason="unioned into fixed_mask, _sampling_helper.py:482 -- a full-length mask despite the name"),
  "fixed_tokens": Bundle(param="fixed_tokens", reason="_prepare_fixed_controls, _sampling_helper.py:484-497"),
  "bias": Bundle(param="bias", reason="threaded to the bundle at kernel_dispatch.py:227 and multistate_poe.py:332"),
  "tie_group_map": Bundle(param="tie_group_map", reason="threaded to the bundle at kernel_dispatch.py:228"),
  "state_weights": Bundle(param="state_weights", reason="threaded to the bundle at kernel_dispatch.py:229"),
  "state_position_map": Bundle(param="state_position_map", reason="broadcast + threaded at kernel_dispatch.py:230-232"),
  "structure_mapping": Bundle(param="structure_mapping", reason="threaded to the bundle at kernel_dispatch.py:251"),
  "ligand_context_path": Bundle(param="ligand_coords", reason="_prepare_ligand_context loads the npz, _sampling_helper.py:269-277"),
  # --- enumerated by the planner, NOT taken from base_spec ------------------------------
  # Verified: plan_campaign_manifest ignores base_spec's values and emits all 4 combinations
  # (campaign.py:612-613). A caller asking for ligand=True/sidechain=True gets 4 rows, one of
  # which is (False, False). This is why tev_design's _patch_row must overwrite both per arm.
  "ligand_conditioning": Overridden(
    value=None,
    reason=(
      "planner enumerates the 2x2 cross product (campaign.py:612-613); base_spec's value is "
      "ignored and each combo gets its own row. At the model boundary the flag is a tri-state "
      "gate (#114, _sampling_helper.py::_prepare_ligand_context): False ABLATES (ignores batch "
      "and ligand_context_path tensors), True requires them, None uses whatever is present; "
      "model_family still decides whether there is a ligand channel at all. Covered by "
      "test_ligand_conditioning_tristate.py."
    ),
  ),
  "sidechain_conditioning": Overridden(
    value=None,
    reason=(
      "planner enumerates the 2x2 cross product (campaign.py:612-613); base_spec's value is "
      "ignored. Gates atom_37/atom_37_mask at the bundle (_sampling_helper.py:315-366); prep.py "
      "implies use_side_chain_context from it at load time, and plan_campaign_manifest now also "
      "states ligand_mpnn_use_side_chain_context on each sidechain-on LigandMPNN row (#115) -- "
      "see ligand_mpnn_use_side_chain_context."
    ),
  ),
  "backbone_noise": Bundle(param="backbone_noise", reason="per-cell value from the noise axis loop, kernel_dispatch.py:224"),
  "temperature": Bundle(param="temperature", reason="per-cell value from the temperature axis loop, kernel_dispatch.py:252"),

  # --- reaches load_model ---------------------------------------------------------------
  "checkpoint_id": LoadModel(param="checkpoint_id", reason="forwarded at prep.py:133; also drives model_family auto-derivation, specs.py:485"),
  "model_weights": LoadModel(param="model_weights", reason="forwarded at both load_model call sites, prep.py:133,140"),
  "model_version": LoadModel(param="model_version", reason="forwarded at both load_model call sites, prep.py:133,140 (legacy alias)"),
  "model_local_path": LoadModel(param="local_path", reason="resolved at prep.py:129-132"),
  "checkpoint_registry_path": LoadModel(param="local_path", reason="resolved at prep.py:129-132"),
  "ligand_mpnn_use_side_chain_context": LoadModel(
    param="use_side_chain_context",
    reason=(
      "forwarded to load_model by prep.py (model_conditioning), and implied by "
      "sidechain_conditioning=True there; the campaign planner derives it per sidechain-on "
      "LigandMPNN row but never overrides an explicit value (campaign.py, #115)"
    ),
  ),
  "use_electrostatics": LoadModel(
    param="use_electrostatics",
    reason="SHOULD reach load_model (io/weights.py) -- forwarded to the dataset at prep.py:111 but not the model",
  ),
  "use_vdw": LoadModel(param="use_vdw", reason="same as use_electrostatics; prep.py:114 dataset only"),

  # --- must survive the manifest; affects plan/loop, not a boundary arg -----------------
  "inputs": SpecOnly(reason="drives structure parsing; also selects PoE vs single-input at campaign.py:865"),
  "model_family": SpecOnly(reason="THE ligand/sidechain gate (_sampling_helper.py:255); auto-derives from checkpoint_id, specs.py:485"),
  "chain_id": SpecOnly(reason="selects which chains are parsed; JSON-decoded at cli.py:1701"),
  "multi_state_strategy": SpecOnly(reason="selects stage_set.logit_transform via make_stage_set, plan.py:797"),
  "batch_size": SpecOnly(reason="dataset batching, prep.py:107; for PoE it becomes bundle.geometry.n_states"),

  # --- campaign deliberately overrides the caller ---------------------------------------
  "campaign_mode": Overridden(value=True, reason="hardcoded True, campaign.py:673 -- a campaign row is by definition campaign mode"),
  "return_logits": Overridden(value=False, reason="hardcoded False, campaign.py:674 -- logits are refused in campaign mode unless allow_logits_in_campaign"),
  "grid_mode": Overridden(value=True, reason="hardcoded True, campaign.py:675"),
  "samples_chunk_size": Overridden(value=None, reason="overwritten with the row's sample_count, campaign.py:672 -- chunking is the planner's job, not the caller's"),

  # --- genuinely inert in campaign mode --------------------------------------------------
  "random_seed": NotApplicable(
    reason=(
      "RETRACTED as a bug after tracing: _base_sampling_key folds a grid-lineage hash (job_id + "
      "strategy/conditioning) into the seed so rows differ despite the shared default 42 "
      "(_sampling_grid_lineage.py); PoE additionally folds sample_start (multistate_poe.py:474, a "
      "risk an independent PR audit found and closed 260714); the non-PoE path slices a "
      "deterministic stream. The raw seed not surviving is benign."
    ),
  ),
  "num_samples": NotApplicable(reason="campaign uses the row's sample_count/samples_chunk_size instead; resolve_target_samples, plan.py:323"),
  "sampling_strategy": NotApplicable(reason="grid_mode requires 'temperature' (validated specs.py:546-551), so a campaign row cannot vary it"),
  "job_id": NotApplicable(reason="set per-row by the planner, campaign.py:676"),
  "chunk_id": NotApplicable(reason="set per-row by the planner, campaign.py:677"),
  "sample_start": NotApplicable(reason="set per-row by the planner, campaign.py:678"),
  "sample_count": NotApplicable(reason="set per-row by the planner, campaign.py:679"),
  "output_h5_path": NotApplicable(reason="set per-row by the planner, campaign.py:680; the worker rewrites it to a partial path, campaign.py:854"),
  # Provenance-only labels written by a manifest-building caller (tev_design's necklace
  # campaign) into a row's nested sampling_spec, for anti-mislabel validation of the executed
  # output against manifest intent -- carried through, never read by any sampling/decode logic.
  # Same inert-but-carried nature as job_id above; asserted inert so a future wiring change trips it.
  "weight_profile": NotApplicable(reason="provenance-only label, never read by sampling/decode; carried for anti-mislabel validation of executed row vs manifest intent (specs.py:596-605 comment + field)"),
  "fixed_group": NotApplicable(reason="provenance-only label, never read by sampling/decode; carried for anti-mislabel validation of executed row vs manifest intent (specs.py:596-606 comment + field)"),
  "iterations": NotApplicable(reason="straight_through only; grid_mode forbids that strategy"),
  "learning_rate": NotApplicable(reason="straight_through only; grid_mode forbids that strategy"),
  "use_concrete": NotApplicable(reason="STE-loop knob (optimize_ste.py); unreachable under grid_mode's temperature strategy"),
  "concrete_tau_start": NotApplicable(reason="STE-loop knob; see use_concrete"),
  "concrete_tau_end": NotApplicable(reason="STE-loop knob; see use_concrete"),
  "allow_logits_in_campaign": NotApplicable(reason="return_logits is hardcoded False, so this never fires in a planned row"),
  "logits_memory_budget_mb": NotApplicable(reason="only meaningful with return_logits=True, which campaign hardcodes off"),
  "average_node_features": NotApplicable(reason="validated incompatible with grid_mode, specs.py:549-551"),
  "compute_pseudo_perplexity": NotApplicable(reason="requires return_logits=True, which campaign hardcodes off"),
  "return_decoding_orders": NotApplicable(reason="output-shape knob; not carried by the manifest and not a boundary arg"),
  "return_logit_fingerprint": NotApplicable(reason="output-shape knob; requires logits"),
  # CORRECTED 2026-07-15. These were declared NotApplicable("streaming/sink bookkeeping
  # (host/streaming.py); never touches the model") -- a FABRICATED reason. They feed the
  # PLANNER (host/plan.py:79-80,98-100: carry_names = {cs.axis_name for cs in carry_specs}).
  # The fabrication survived because test_every_reason_cites_something checked reason LENGTH
  # (>20 chars), not whether the citation resolved. An external architecture review caught it,
  # not the test built to catch exactly this. Both are now fixed.
  "carry_specs": NonSerializable(
    reason=(
      "list[CarrySpec]; CarrySpec.transition is a ScanTransition CALLABLE and .init holds JAX "
      "arrays. spec_json's _to_json_value has no nested-dataclass branch, so encoding a non-None "
      "value raises SpecJSONEncodeError rather than dropping it. Feeds the planner "
      "(host/plan.py:79-80,98-100), so a campaign row genuinely cannot carry a Scan axis -- a "
      "real limitation to surface, not bookkeeping to ignore."
    ),
  ),
  "dedup_specs": NonSerializable(
    reason=(
      "list[DedupSpec]; its unique_indices/index_map/k would serialize, but dedup_fn/gather_fn "
      "are optional callables and _to_json_value has no nested-dataclass branch -> "
      "SpecJSONEncodeError. Feeds the planner (host/plan.py:80,99), so a campaign row cannot "
      "carry a dedup axis."
    ),
  ),
  "multi_state_temperature": NotApplicable(reason="reaches stage_set.logit_transform, not a bundle arg; only affects geometric_mean (logits.py:130,175)"),
  "multi_state_sharpness": NotApplicable(reason="reaches stage_set.logit_transform, not a bundle arg; only affects product. Semantics are covered by tests/inference/test_product_sharpness.py, which asserts the invariant a reachability harness cannot express: S identical experts must fuse to p^S under a product and to p under a sum(w)=1 opinion pool"),
  "use_unified_driver": NotApplicable(reason="selects the dispatch path in kernel_dispatch, not a model-facing value"),
  "tied_positions": NotApplicable(reason="requires pass_mode='inter' (specs.py:440-445); the campaign literal carries neither"),
  "pass_mode": NotApplicable(reason="paired with tied_positions; not carried by the manifest"),

  # --- batching knobs: MUST be no-ops at the boundary (invariance, not change) ----------
  "samples_batch_size": NotApplicable(reason="Vmap-vs-SafeMap tile size only (plan.py:171-244). MUST NOT change model-facing values -- assert invariance; a diff here would be the bug."),
  "noise_batch_size": NotApplicable(reason="tile size only; see samples_batch_size"),
  "temperature_batch_size": NotApplicable(reason="tile size only; see samples_batch_size"),

  # --- host/IO knobs, no model surface ---------------------------------------------------
  "topology": NotApplicable(reason="parse-time input resolution"),
  "model": NotApplicable(reason="NMR model index chosen at parse time; not carried by the manifest literal"),
  "altloc": NotApplicable(reason="altloc chosen at parse time; not carried by the manifest literal"),
  "cache_path": NotApplicable(reason="host-side cache location; no model surface, not in the manifest literal"),
  "output_dir": NotApplicable(reason="host-side output location; campaign uses output_h5_path"),
  "max_buffer_size": NotApplicable(reason="host-side streaming buffer"),
  "overwrite_cache": NotApplicable(reason="host-side cache policy"),
  "max_length": NotApplicable(reason="parse/truncation policy"),
  "truncation_strategy": NotApplicable(reason="parse/truncation policy"),
  "host_resource_allocation_strategy": NotApplicable(reason="host resource policy"),
  "ram_budget_mb": NotApplicable(reason="host resource policy"),
  "max_workers": NotApplicable(reason="host resource policy"),
  "n_devices": NotApplicable(reason="host resource policy"),
  "use_preprocessed": NotApplicable(reason="dataset source selection"),
  "preprocessed_index_path": NotApplicable(reason="dataset source selection"),
  "split": NotApplicable(reason="dataset split selection; inference default, not in the manifest literal"),
  "noise": NotApplicable(reason="unified FeatureNoiseBundle list; the campaign carries the legacy backbone_noise scalar instead"),

  # --- deprecated back-compat ------------------------------------------------------------
  "backbone_noise_mode": NotApplicable(reason="deprecated; merged into the noise bundle at specs.py:330-352"),
  "estat_noise": NotApplicable(reason="deprecated; merged into the noise bundle, specs.py:355-380"),
  "estat_noise_mode": NotApplicable(reason="deprecated; merged into the noise bundle"),
  "vdw_noise": NotApplicable(reason="deprecated; merged into the noise bundle, specs.py:382-407"),
  "vdw_noise_mode": NotApplicable(reason="deprecated; merged into the noise bundle"),

  # --- cannot cross the JSON boundary ----------------------------------------------------
  "decoding_order_fn": NonSerializable(reason="callable; in _NON_JSON_ROOT_FIELDS (spec_json.py:99-108). Also never forwarded to _sample_batch."),
  "encoding_fusion": NonSerializable(reason="callable; _NON_JSON_ROOT_FIELDS"),
  "decoding_fusion": NonSerializable(reason="callable; _NON_JSON_ROOT_FIELDS"),
  "foldcomp_database": NonSerializable(reason="live object; _NON_JSON_ROOT_FIELDS"),
  "conformational_states": NonSerializable(reason="raises SpecJSONEncodeError when set"),

  # --- internals --------------------------------------------------------------------------
  "run_spec": Internal(reason="init=False; built in __post_init__, specs.py:319"),
  "_run_spec_synced": Internal(reason="init=False guard flag"),
}
