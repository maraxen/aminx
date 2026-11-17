---
title: "Respec: xtrax.eda for campaign libraries (knob-faceted sink statistics)"
task_id: "261106_campaign-eda-respec"
date: "2026-11-06"
status: "draft"
owner_decisions: "xtrax engine + consumer metric plugins; spec against track5 now; keep plan EDA as the telemetry view; spec + prototype on real data (user, 2026-11-06)"
supersedes_scope_of: "xtrax debt #3143 (EDA visualization), as adopted in aminx PR #270"
---

# xtrax.eda for campaign libraries

## 1. Why this respec

`xtrax.eda` today (`src/xtrax/eda/`, redesign `cf2bd7f`) explains a `BatchPlan`: strategy, cardinality,
bucket, dedup, memory and reasoning panels. That is run telemetry. Debt #3143 asked for "EDA viz"
without naming the data, the adoption spec (aminx `261105_aminx-xtrax-adoption-spec.md` §6, DEC-05)
mapped it onto the only object it could read, and three review rounds checked the code against that
spec, never the spec against the need. aminx PR #270 shipped `plan_eda.py` on that basis.

The EDA a campaign needs is about **what was produced**: statistics of the sampled libraries and their
scores, faceted by the campaign's knobs, with replicate structure respected. Plan telemetry stays, as a
secondary view.

## 2. Evidence from the prototype (aminx `scripts/analysis/campaign_library_eda_prototype.py`)

Run on tev_design's state-distinguishability AR campaign (216 rows = 6 arms x 9 states x 4 seeds, 5
temperatures stored inside each sink), read-only. Pre-registered run v1 recorded **fail** (bathos run
`64c6845b`); its two causes are design requirements below. v2 (`6d38ebaf`, pass; first v2 run
`885f66c5` gave identical results) reads residue tables built with the campaign's own parser (`ca68e784`,
aminx 0.1.0a19 + proxide 0.1.0a11 from tev_design's lock at `1f40e01a`) and scopes the shuffle control
to the ProteinMPNN/SolubleMPNN arms. Review page built from `6d38ebaf` (six panels; exploratory).

| Finding | Evidence | Requirement it creates |
|---|---|---|
| The reader and join are faithful | v1 reproduced tev_design's 12 recorded JSD statistics with max abs diff 0.0 | (positive control pattern, §7) |
| Re-parsing the input does not reproduce the residue order the campaign sampled | proxide 0.1.0a17 orders 1LVB chains A,B,C,D (214,10,214,10); proxide 0.2.0 orders them 214,214,10,10. Recovery for 1LVB fell from 0.45 to 0.24 under the new order | R2: the sink records its residue table |
| Sink token order differs from the parser's | parser `aatype` is AF2 order (`ARNDC...`), sinks are MPNN order (`ACDEF...`); unmapped recovery 0.14 vs 0.45 mapped | R2: alphabet is declared per array |
| Padding is sampled and is not neutral | positions 448-511 of a 448-residue target: mean entropy 4.04 bits vs 2.3 on real residues | R2: real-residue mask; every statistic masks |
| Index-based fixing misplaces pins across states | `catalytic_triad` fixed indices 38/73/143 land on S39/V74/D144 (1LVM) and K45/R80/Q150 (reac2 x4) | R4: integrity panels; R5: no index alignment across targets |
| A whole arm can be degenerate | all four LigandMPNN arms: recovery 0.13-0.14 vs 0.05-0.06 against shuffled native (margin 0.07-0.08; ProteinMPNN/SolubleMPNN 0.33-0.35), composition entropy 3.3-3.4 bits vs 4.0-4.1, per-residue NLL under their own logits about twice the other arms'. Produced by aminx 0.1.0a19, which predates the LigandMPNN backbone-O fix `82c4ea9b` (#2326) | R4: degeneracy panel per arm |
| Some knobs live inside the sink | `temperature` is a list on every row and an array axis in the store; never a row-level facet | R3: two knob sources |

## 3. Requirements

**R1. Inputs are a finished campaign, read-only.** The engine reads a manifest (track5
`CampaignManifest`, or a consumer adapter for aminx v1 manifests) and each row's committed unit. It
never routes through a resume path that can quarantine or invalidate (`_resolve_existing_unit`,
`_invalidate_stale_output`), never writes into the campaign tree, and resolves stores through the
manifest's paths.

**R2. A sink carries its own meaning.** Zarr `dimension_names` are absent in aminx stores today. The
consumer declares a `SinkSchema`: for each array, its dims (e.g. `sequences: (sample, grid, temperature,
residue)`), its alphabet when tokenised, and which dims bind to row fields (`temperature` <- row
`temperature`). Each unit also records a **residue table** at write time: native tokens (declared
alphabet), chain id, author residue number, insertion code, real-residue mask, and the parser identity
(name, version) that produced it. Re-deriving it by re-parsing is allowed only as a fallback that is
checked against a recorded parser identity and refused on mismatch.

**R3. Knobs come from two places.** Row knobs: the track5 `grid.axes` names plus arm fields
(`arm_id`, `checkpoint_id`, `fixed_policy`, `state_weight_profile`) and `target_input`; for manifests
without a spec, the row fields that vary across rows minus bookkeeping fields. Sink knobs: dims bound to
row fields in the `SinkSchema`. The seed is a replicate unit, never a facet.

**R4. Integrity panels are first-class and come first.**
- Completion and reuse per unit (the telemetry view; plan panels from today's `xtrax.eda` live here).
- Fixed positions checked against the native residue **by residue identity** (chain + number), per target.
- Per arm: composition entropy, recovery against a composition-preserving shuffle of native (must sit
  at chance), and recovery against native; an arm whose recovery is within a margin of the shuffle is
  flagged degenerate.

**R5. Statistics respect structure.** Every statistic masks to real, designable residues. Cross-target
comparisons need an explicit residue mapping (by chain + number, or a declared alignment); index
alignment is refused. Uncertainty is computed at or above the independent unit (target, or build when
there are several builds); seeds are pooled draws. Small cells are labelled descriptive only, with no
interval and no "no effect" claim without a spike-in and a minimum detectable effect.

**R6. Metrics are consumer plugins over a typed library view.** xtrax owns the reader, knob join,
masking, faceting, uncertainty and rendering. A consumer (aminx) registers metrics that take a
`LibraryCell` (tokens, logits if present, residue table, mask, knob values) and return named values.
aminx's first set: recovery, derived per-residue NLL (labelled derived from stored logits, not a score
array), within-library pairwise identity, per-residue entropy, composition and its divergence from
native, and score distributions where score arrays exist (`score`, `score_conditional`,
`score_unconditional`; absent from the prototype's campaign).

**R7. Output is a provenanced review page.** Tables are long-format (one row per unit x knob cell x
metric), written by a bathos-tracked run and rendered as a maraxiom review page (`sweep_grid` over knob
pairs, `pooled_estimate` with seeds as points, `residue_track`, `sweep_grid` completion). The existing
plan panels become a telemetry tab on the same page.

## 4. Interfaces (sketch)

```python
# xtrax.eda.campaign
@dataclass(frozen=True)
class SinkArray: name: str; dims: tuple[str, ...]; alphabet: str | None = None; bind: Mapping[str, str] = {}
@dataclass(frozen=True)
class SinkSchema: arrays: tuple[SinkArray, ...]; residue_table: str  # array or attr holding R2's table
@dataclass(frozen=True)
class LibraryCell: knobs: Mapping[str, Any]; seed: int; target: str; tokens: np.ndarray; logits: np.ndarray | None; residues: ResidueTable; mask: np.ndarray
class LibraryMetric(Protocol):
    name: str
    def __call__(self, cell: LibraryCell) -> Mapping[str, float]: ...
def campaign_view(manifest, schema: SinkSchema, *, spec: CampaignSpec | None = None) -> CampaignView: ...
def library_tables(view: CampaignView, metrics: Sequence[LibraryMetric]) -> dict[str, pa.Table]: ...
```

Metric discovery reuses the `xtrax.eda.plugins` pattern (entry points, quarantine), with a new
`LibraryMetric` protocol; today's `PanelPlugin` takes a `PlanStatsDict` and cannot host these.

## 5. Assumptions

| Assumption | Status |
|---|---|
| track5 `ManifestRow` carries `target_input`, arm fields, `seed`, `parameters`, `output_locator` | verified, `origin/feat/261105-campaign-runner-track5:src/xtrax/campaign/manifest.py:126-145` |
| track5 row `parameters` = grid point, then campaign `parameters`, then arm `parameters` (arm wins) | verified, same file, `plan_campaign` body (merge order at its lines 36-39) |
| track5 leaves sink layout to the consumer (`output_locator` is `None` at plan time) | verified, same file, `plan_campaign` line 78 of its body |
| track5 is unmerged and may change | verified 2026-11-06: not on `origin/main` |
| aminx sinks hold `sequences`, `logits`, `fixed_mask`, `fixed_tokens`; score arrays only in scoring runs | verified for the prototype campaign; score arrays unverified in any finished campaign |
| aminx can write a residue table into the sink at sampling time | unverified; needs an aminx change (sink writer) |
| The LigandMPNN degeneracy is a defect of the producing build, not of the EDA | partly verified: the producing build (aminx 0.1.0a19, 2026-07-19) predates fix `82c4ea9b` (2026-10-01); causation untested (needs a re-sample with a fixed build) |

## 6. Non-goals

- Re-running or correcting any campaign. Findings about a campaign go to its owner.
- Replacing `plan_eda.py`; it becomes the telemetry tab.
- Interactive dashboards. The reviewed object is the static SVG on a review page.

## 7. Verification for the implementation

- Positive control: reproduce an independent, recorded analysis from the same stores exactly
  (the prototype's tev_design JSD reproduction is the template).
- Negative controls: shuffled-native recovery at chance; a planted misplaced pin is flagged; a planted
  degenerate arm is flagged; a residue table from the wrong parser version is refused.
- Pre-registered with bathos; tables built only from a terminal run.

## 8. Next steps

1. File this spec in xtrax (docs branch) and re-scope debt #3143 to it once the praxia DB is reachable.
2. aminx: write the residue table into sampling sinks (R2); add the `SinkSchema` declaration.
3. xtrax: implement `campaign_view` / `library_tables` against track5 once it merges; port the
   prototype's controls as tests.
4. Done: prototype v2 (`6d38ebaf`, pass) feeds the first review page. A per-residue track panel is
   not on it yet (the table has no condition column the recipe's EDA block needs).
5. Note for xtrax: the positive control shows the reader is faithful, not that the reproduced method
   is sound. tev_design's JSD averages over all 512 indices, padding included, and across states whose
   residue indices do not correspond.
