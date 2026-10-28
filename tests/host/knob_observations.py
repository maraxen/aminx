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
class IncompatibleArchitecture(NotApplicable):
  """Feature belongs to an architecture distinct from the active backbone model."""


@dataclass(frozen=True)
class HostPolicyOnly(NotApplicable):
  """Knob controls host-side execution environment/policy, not model tensor execution."""


@dataclass(frozen=True)
class UnsupportedSamplingMode(NotApplicable):
  """Knob is valid only under an alternative sampling mode (e.g. non-autoregressive/MCMC)."""


@dataclass(frozen=True)
class PlannerDerived(NotApplicable):
  """Knob is derived/managed internally by the planner rather than passed directly."""


@dataclass(frozen=True)
class DeprecatedOption(NotApplicable):
  """Knob is deprecated or replaced by a modern config structure."""


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
CLOSED_BY_S1A: tuple[str, ...] = (
  "fixed_mask",
  "fixed_positions",
  "fixed_tokens",
  "state_position_map",
  "multi_state_strategy",
  "state_weights",
  "bias",
  "tie_group_map",
  "structure_mapping",
  "ligand_mpnn_use_side_chain_context",
  "use_electrostatics",
  "use_vdw",
)

# Tier A: does the caller's value survive plan -> manifest JSON -> reconstructed spec?
KNOWN_BROKEN_TIER_A: dict[str, str] = {}

# Tier B: does it reach the MODEL? Empty after S3.
KNOWN_BROKEN_TIER_B: dict[str, str] = {}


# ---------------------------------------------------------------------------
# The table. One entry per SamplingSpecification field.
# ---------------------------------------------------------------------------
OBSERVATIONS: dict[str, Verdict] = {
  # --- reaches build_inference_bundle -------------------------------------------------
  "fixed_mask": Bundle(
    param="fixed_mask",
    reason="unioned with fixed_positions in _prepare_fixed_controls, _sampling_helper.py:482",
  ),
  "fixed_positions": Bundle(
    param="fixed_mask",
    reason="unioned into fixed_mask, _sampling_helper.py:482 -- a full-length mask despite the name",
  ),
  "fixed_tokens": Bundle(
    param="fixed_tokens",
    reason="_prepare_fixed_controls, _sampling_helper.py:484-497",
  ),
  "bias": Bundle(
    param="bias",
    reason="threaded to the bundle at kernel_dispatch.py:227 and multistate_poe.py:332",
  ),
  "omit_aa": Bundle(
    param="bias",
    reason="compiled to -1e8 at designed positions and added to bias, kernel_dispatch.py:277-291 (omit_aa_bias.compile_omit_aa_bias)",
  ),
  "omit_aa_per_position": Bundle(
    param="bias",
    reason="compiled with omit_aa into bias, kernel_dispatch.py:277-291",
  ),
  "tie_group_map": Bundle(
    param="tie_group_map",
    reason="threaded to the bundle at kernel_dispatch.py:228",
  ),
  "state_weights": Bundle(
    param="state_weights",
    reason="threaded to the bundle at kernel_dispatch.py:229",
  ),
  "state_position_map": Bundle(
    param="state_position_map",
    reason="broadcast + threaded at kernel_dispatch.py:230-232",
  ),
  "structure_mapping": Bundle(
    param="structure_mapping",
    reason="threaded to the bundle at kernel_dispatch.py:251",
  ),
  "ligand_context_path": Bundle(
    param="ligand_coords",
    reason="_prepare_ligand_context loads the npz, _sampling_helper.py:269-277",
  ),
  "backbone_noise": Bundle(
    param="backbone_noise",
    reason="per-cell value from the noise axis loop, kernel_dispatch.py:224",
  ),
  "temperature": Bundle(
    param="temperature",
    reason="per-cell value from the temperature axis loop, kernel_dispatch.py:252",
  ),

  # --- reaches load_model ---------------------------------------------------------------
  "checkpoint_id": LoadModel(
    param="checkpoint_id",
    reason="forwarded at prep.py:133; also drives model_family auto-derivation, specs.py:485",
  ),
  "model_weights": LoadModel(
    param="model_weights",
    reason="forwarded at both load_model call sites, prep.py:133,140",
  ),
  "model_version": LoadModel(
    param="model_version",
    reason="forwarded at both load_model call sites, prep.py:133,140 (legacy alias)",
  ),
  "model_local_path": LoadModel(
    param="local_path",
    reason="resolved at prep.py:129-132",
  ),
  "checkpoint_registry_path": LoadModel(
    param="local_path",
    reason="resolved at prep.py:129-132",
  ),
  "ligand_mpnn_use_side_chain_context": LoadModel(
    param="use_side_chain_context",
    reason="forwarded to load_model by prep.py:188 (model_conditioning); see campaign.py:115",
  ),
  "use_electrostatics": LoadModel(
    param="use_electrostatics",
    reason="SHOULD reach load_model (io/weights.py) -- forwarded to the dataset at prep.py:111 but not the model",
  ),
  "use_vdw": LoadModel(
    param="use_vdw",
    reason="same as use_electrostatics; prep.py:114 dataset only",
  ),

  # --- must survive the manifest; affects plan/loop, not a boundary arg -----------------
  "inputs": SpecOnly(
    reason="drives structure parsing; also selects PoE vs single-input at campaign.py:865",
  ),
  "model_family": SpecOnly(
    reason="THE ligand/sidechain gate (_sampling_helper.py:255); auto-derives from checkpoint_id, specs.py:485",
  ),
  "chain_id": SpecOnly(
    reason="selects which chains are parsed; JSON-decoded at cli.py:1701",
  ),
  "multi_state_strategy": SpecOnly(
    reason="selects stage_set.logit_transform via make_stage_set, plan.py:797",
  ),
  "batch_size": SpecOnly(
    reason="dataset batching, prep.py:107; for PoE it becomes bundle.geometry.n_states",
  ),

  # --- campaign deliberately overrides the caller ---------------------------------------
  "campaign_mode": Overridden(
    value=True,
    reason="hardcoded True, campaign.py:673 -- a campaign row is by definition campaign mode",
  ),
  "return_logits": Overridden(
    value=False,
    reason="hardcoded False, campaign.py:674 -- logits are refused in campaign mode unless allow_logits_in_campaign",
  ),
  "grid_mode": Overridden(
    value=True,
    reason="hardcoded True, campaign.py:675",
  ),
  "samples_chunk_size": Overridden(
    value=None,
    reason="overwritten with the row's sample_count, campaign.py:672 -- chunking is the planner's job, not the caller's",
  ),
  "ligand_conditioning": Overridden(
    value=None,
    reason="planner enumerates the 2x2 cross product (campaign.py:612-613); covered by test_ligand_conditioning_tristate.py and _sampling_helper.py:255",
  ),
  "sidechain_conditioning": Overridden(
    value=None,
    reason="planner enumerates the 2x2 cross product (campaign.py:612-613); gates atom_37 in _sampling_helper.py:315-366 and prep.py:188",
  ),

  # --- IncompatibleArchitecture: campaign mode refuses driver families ------------------
  "potts_mpnn": IncompatibleArchitecture(
    reason="PottsMPNNOptions require model_family pottsmpnn (run/specs.py:440), refused in campaign mode at host/campaign.py:612",
  ),
  "laser": IncompatibleArchitecture(
    reason="LaserOptions require model_family lasermpnn (run/specs.py:450), refused in campaign mode at host/campaign.py:612",
  ),
  "protonpotts": IncompatibleArchitecture(
    reason="ProtonPottsOptions require model_family protonpottsmpnn (run/specs.py:460), refused in campaign mode at host/campaign.py:612",
  ),
  "caliby": IncompatibleArchitecture(
    reason="CalibyOptions require model_family caliby, refused in campaign mode by refuse_driver_family (host/campaign.py:110)",
  ),
  "fampnn": IncompatibleArchitecture(
    reason="FAMPNNOptions require model_family fampnn, refused in campaign mode by refuse_driver_family (host/campaign.py:110)",
  ),
  "thermompnn": IncompatibleArchitecture(
    reason="ThermoMPNNOptions require model_family thermompnn, refused in campaign mode by refuse_driver_family (host/campaign.py:110)",
  ),

  # --- PlannerDerived: planner-derived per-row variables --------------------------------
  "random_seed": PlannerDerived(
    reason="_base_sampling_key folds grid-lineage hash into seed (_sampling_grid_lineage.py:42 and multistate_poe.py:474)",
  ),
  "num_samples": PlannerDerived(
    reason="campaign uses row sample_count/samples_chunk_size; resolve_target_samples in host/plan.py:323",
  ),
  "job_id": PlannerDerived(
    reason="set per-row by the planner in host/campaign.py:676",
  ),
  "chunk_id": PlannerDerived(
    reason="set per-row by the planner in host/campaign.py:677",
  ),
  "sample_start": PlannerDerived(
    reason="set per-row by the planner in host/campaign.py:678",
  ),
  "sample_count": PlannerDerived(
    reason="set per-row by the planner in host/campaign.py:679",
  ),
  "output_h5_path": PlannerDerived(
    reason="set per-row by planner in host/campaign.py:680; rewritten to partial path in host/campaign.py:854",
  ),
  "weight_profile": PlannerDerived(
    reason="provenance-only label carried for anti-mislabel validation in run/specs.py:596 and host/campaign.py:682",
  ),
  "fixed_group": PlannerDerived(
    reason="provenance-only label carried for anti-mislabel validation in run/specs.py:596 and host/campaign.py:683",
  ),
  "pass_mode": PlannerDerived(
    reason="paired with tied_positions in run/specs.py:440-445; not carried in host/campaign.py manifest",
  ),
  "run_id": PlannerDerived(
    reason="unique campaign run identifier set in host/campaign.py:676 and run/specs.py:226",
  ),

  # --- UnsupportedSamplingMode: valid only in alternative sampling strategies -----------
  "sampling_strategy": UnsupportedSamplingMode(
    reason="grid_mode requires temperature (validated run/specs.py:546-551), cannot vary in campaign",
  ),
  "iterations": UnsupportedSamplingMode(
    reason="straight_through optimization loop only (inference/optimize_ste.py:45); forbidden by grid_mode in run/specs.py:546",
  ),
  "learning_rate": UnsupportedSamplingMode(
    reason="straight_through optimization learning rate (inference/optimize_ste.py:48); forbidden by grid_mode in run/specs.py:546",
  ),
  "use_concrete": UnsupportedSamplingMode(
    reason="STE-loop knob (inference/optimize_ste.py:50); unreachable under grid_mode in run/specs.py:546",
  ),
  "concrete_tau_start": UnsupportedSamplingMode(
    reason="STE-loop temperature annealing start in inference/optimize_ste.py:52; unreachable under grid_mode in run/specs.py",
  ),
  "concrete_tau_end": UnsupportedSamplingMode(
    reason="STE-loop temperature annealing end in inference/optimize_ste.py:55; unreachable under grid_mode in run/specs.py",
  ),
  "average_node_features": UnsupportedSamplingMode(
    reason="validated incompatible with grid_mode in run/specs.py:549-551",
  ),
  "multi_state_temperature": UnsupportedSamplingMode(
    reason="reaches stage_set.logit_transform, only affects geometric_mean in logits.py:130,175",
  ),
  "multi_state_sharpness": UnsupportedSamplingMode(
    reason="reaches stage_set.logit_transform in tests/inference/test_product_sharpness.py and logits.py:150",
  ),
  "tied_positions": UnsupportedSamplingMode(
    reason="requires pass_mode=inter (run/specs.py:440-445); campaign literal carries neither in host/campaign.py",
  ),

  # --- HostPolicyOnly: host runtime and resource knobs -----------------------------------
  "allow_logits_in_campaign": HostPolicyOnly(
    reason="return_logits is hardcoded False at host/campaign.py:674 unless allow_logits_in_campaign in host/campaign.py",
  ),
  "logits_memory_budget_mb": HostPolicyOnly(
    reason="only meaningful with return_logits=True (host/campaign.py:674), managed in host/kernel_dispatch.py:230",
  ),
  "compute_pseudo_perplexity": HostPolicyOnly(
    reason="requires return_logits=True (host/campaign.py:674), evaluated in host/runner.py:150",
  ),
  "return_decoding_orders": HostPolicyOnly(
    reason="output-shape flag in run/specs.py:698; not carried by manifest in host/campaign.py:670",
  ),
  "return_logit_fingerprint": HostPolicyOnly(
    reason="output-shape flag in run/specs.py:699; requires return_logits in host/campaign.py:674",
  ),
  "use_unified_driver": HostPolicyOnly(
    reason="selects unified dispatch path in host/kernel_dispatch.py:426, not a model tensor",
  ),
  "samples_batch_size": HostPolicyOnly(
    reason="Vmap-vs-SafeMap tile size only in host/plan.py:171-244; must not change model boundary values",
  ),
  "noise_batch_size": HostPolicyOnly(
    reason="tile size only for planner batching in host/plan.py:175; see samples_batch_size in host/plan.py",
  ),
  "temperature_batch_size": HostPolicyOnly(
    reason="tile size only for planner batching in host/plan.py:176; see samples_batch_size in host/plan.py",
  ),
  "topology": HostPolicyOnly(
    reason="parse-time input resolution in host/prep.py:80 and io/parsing/dispatch.py:30",
  ),
  "model": HostPolicyOnly(
    reason="NMR model index chosen at parse time in host/prep.py:85; not carried in host/campaign.py",
  ),
  "altloc": HostPolicyOnly(
    reason="altloc chosen at parse time in host/prep.py:90; not carried in host/campaign.py",
  ),
  "cache_path": HostPolicyOnly(
    reason="host-side cache location in host/prep.py:100; no model surface, not in host/campaign.py",
  ),
  "output_dir": HostPolicyOnly(
    reason="host-side output location in run/specs.py:177; campaign uses output_h5_path at host/campaign.py:680",
  ),
  "max_buffer_size": HostPolicyOnly(
    reason="host-side streaming buffer size in host/streaming.py:50",
  ),
  "overwrite_cache": HostPolicyOnly(
    reason="host-side cache policy in host/prep.py:105",
  ),
  "max_length": HostPolicyOnly(
    reason="parse/truncation padding ceiling in host/prep.py:110 and host/bucketing.py:60",
  ),
  "length_bucketing": HostPolicyOnly(
    reason="runner padding policy in host/bucketing.py:45; campaign cannot vary it per row",
  ),
  "truncation_strategy": HostPolicyOnly(
    reason="parse/truncation policy in host/prep.py:115",
  ),
  "host_resource_allocation_strategy": HostPolicyOnly(
    reason="host resource allocation strategy in host/bucketing.py:40",
  ),
  "ram_budget_mb": HostPolicyOnly(
    reason="host resource allocation RAM ceiling in host/bucketing.py:45",
  ),
  "max_workers": HostPolicyOnly(
    reason="host resource allocation worker concurrency in host/bucketing.py:50",
  ),
  "n_devices": HostPolicyOnly(
    reason="host resource allocation accelerator device count in host/bucketing.py:55",
  ),
  "use_preprocessed": HostPolicyOnly(
    reason="dataset source selection in host/prep.py:120",
  ),
  "preprocessed_index_path": HostPolicyOnly(
    reason="dataset source selection index in host/prep.py:125",
  ),
  "split": HostPolicyOnly(
    reason="dataset split selection in host/prep.py:130; not in manifest literal in host/campaign.py",
  ),
  "output_path": PlannerDerived(
    reason="set per-row by planner in host/campaign.py:680 and synced with output_h5_path in run/specs.py:720",
  ),
  "output_zarr_path": PlannerDerived(
    reason="set per-row by planner in host/campaign.py:681 and synced with output_h5_path in run/specs.py:720",
  ),

  # --- DeprecatedOption: deprecated compatibility flags ---------------------------------
  "noise": DeprecatedOption(
    reason="FeatureNoiseBundle list in run/specs.py:330; campaign mode carries legacy backbone_noise in host/campaign.py:660",
  ),
  "backbone_noise_mode": DeprecatedOption(
    reason="deprecated; merged into noise bundle in run/specs.py:330-352",
  ),
  "estat_noise": DeprecatedOption(
    reason="deprecated; merged into noise bundle in run/specs.py:355-380",
  ),
  "estat_noise_mode": DeprecatedOption(
    reason="deprecated; merged into noise bundle in run/specs.py:355-380",
  ),
  "vdw_noise": DeprecatedOption(
    reason="deprecated; merged into noise bundle in run/specs.py:382-407",
  ),
  "vdw_noise_mode": DeprecatedOption(
    reason="deprecated; merged into noise bundle in run/specs.py:382-407",
  ),

  # --- NonSerializable: cannot cross the JSON boundary ---------------------------------
  "decoding_order_fn": NonSerializable(
    reason="callable in _NON_JSON_ROOT_FIELDS (run/spec_json.py:99-108); never forwarded to sample_batch in host/kernel_dispatch.py",
  ),
  "encoding_fusion": NonSerializable(
    reason="callable in _NON_JSON_ROOT_FIELDS in run/spec_json.py:100; see host/kernel_dispatch.py:428",
  ),
  "decoding_fusion": NonSerializable(
    reason="callable in _NON_JSON_ROOT_FIELDS in run/spec_json.py:101; see host/kernel_dispatch.py:428",
  ),
  "foldcomp_database": NonSerializable(
    reason="live object in _NON_JSON_ROOT_FIELDS in run/spec_json.py:102; see host/prep.py:95",
  ),
  "conformational_states": NonSerializable(
    reason="raises SpecJSONEncodeError when set in run/spec_json.py:103",
  ),
  "carry_specs": NonSerializable(
    reason=(
      "list[CarrySpec]; CarrySpec.transition is a ScanTransition CALLABLE and .init holds JAX "
      "arrays. spec_json's _to_json_value has no nested-dataclass branch, so encoding a non-None "
      "value raises SpecJSONEncodeError in run/spec_json.py. Feeds host/plan.py:79-80."
    ),
  ),
  "dedup_specs": NonSerializable(
    reason=(
      "list[DedupSpec]; its unique_indices/index_map/k would serialize, but dedup_fn/gather_fn "
      "are optional callables and _to_json_value has no nested-dataclass branch -> "
      "SpecJSONEncodeError in run/spec_json.py. Feeds host/plan.py:80."
    ),
  ),

  "wave": NonSerializable(
    reason="WaveScheduleBundle (eqx.Module) excluded from manifest JSON in host/spec_partition.py:130 and forwarded in host/kernel_dispatch.py:255",
  ),

  # --- Internal: init=False plumbing ----------------------------------------------------
  "run_spec": Internal(
    reason="init=False; built in __post_init__ in run/specs.py:319",
  ),
  "_run_spec_synced": Internal(
    reason="init=False internal guard flag in run/specs.py:319 and run/specs.py:653",
  ),
}
