---
title: 'Spec-named tests triage: which of the 28 are missing coverage, which are only missing a name'
description: Triage of every test name the potts/laser spec cites but that does not exist, into genuinely-missing, covered-under-another-name, blocked, and false-positive
status: draft
task_id: 260929_potts-laser-xtrax-compose
date: '261003'
confidence: measured
sources: 'spec 260929_pottsmpnn-lasermpnn-xtrax-composition.md; src/aminx at 0a658ab7; titanix b7i CPU runs'
---
# Spec-named tests triage: which of the 28 are missing coverage, which are only missing a name

Task #25 was recorded as "~24 spec-named tests that do not exist". The count is
right and the framing is not: **a missing name is not the same as missing
coverage**, and writing all of them blind would have duplicated existing tests
in several cases and silently skipped the two that actually matter.

Method: extract every `test_[a-z0-9_]*` token from the spec (53 unique), diff
against every `def test_*` in `tests/` (2250 unique). 41 names did not match.
Triaging those 41 by reading the source — not the docstrings — gives four
groups.

## How the 41 break down

| Group | Count | Meaning |
|---|---|---|
| False positive | 13 | Not a test function: a FILE name, a fragment, or a pytest hook |
| Covered under another name | ~9 | The property is pinned; only the spec's label is absent |
| Genuinely missing — written | 2 | `test_energy_invariant_to_padding`, `test_tied_overlap_raises` |
| Genuinely missing — blocked | 1 | `test_decode_invariant_to_padding`, needs a scoped change |
| Not yet triaged | ~16 | Mostly `test_divergence_*` and LASEr knob rows |

## False positives — do not chase these

`test_potts_import_boundary`, `test_l_drv`, `test_rs6b_flat_field_gate` are
FILE names under `tests/lint/`, which all exist. `test_coverage`, `test_ids`,
`test_noop`, `test_raise`, `test_subset`, `test_designs`, `test_knob_semantics_`,
`test_family_sink_ragged_` are fragments or prefixes. `test_sessionstart`,
`test_runtest_logreport`, `test_runtest_makereport` are pytest hook names
quoted in prose.

A name-level diff cannot tell these apart from real gaps, which is why the
"~24 missing tests" figure should never have been actioned as a work estimate.

## Covered under another name

- **`test_laser_alphabet_roundtrip`** (spec:669) →
  `tests/families/laser_mpnn/test_driver.py:124
  ::test_alphabet_boundary_is_a_bijection`. It checks the permutation both
  ways, the letter-level correspondence, AND that the map is not the identity
  (`LASER_ALPHABET.index("E") != ALPHABET.index("E")`) — the non-vacuity check
  a round-trip test most often omits.

  The spec's harder claim, "converted exactly once", is separately pinned by
  `test_nll_matches_gathered_log_softmax` (:137), which indexes logits by
  `ALPHABET.index(letter)` and asserts the NLL matches at `rtol=0, atol=0`.
  A double conversion or a missing one would move E/Q and break that exact
  match. No new test needed.

## Genuinely missing — now written

### `test_energy_invariant_to_padding` → `tests/families/potts_mpnn/test_padding_invariance.py` (a074a409)

Writing it surfaced that **nothing in aminx ever pads**. Both `pad()` call
sites pass `L_total` (`driver.py:590`, `sample_host.py:277`), so the
`pad_valid` masking has never been exercised, and
`RunSpecification.max_length` (`run/specs.py:273`, default 512) does not reach
Potts padding at all.

The property is not trivial: `features.py:92` takes `k = min(k_neighbors,
coords.shape[0])` = `min(48, L_pad)`, NOT `min(48, L_total)`. Padding L=30 to
128 grows the neighbour set 30 → 48.

Measured f64, `L_total ∈ {30,49}` × `l_pad ∈ {L_total,64,128}`: worst
disagreement **5.33e-15** against energies of order 0.09–6.4. So the spec's
"f64 exact" is unachievable as written — K changes the summation *width* and
float addition is not associative. Band set at 1e-12.

**The control is the half that took the work.** The obvious one does not fire:
setting `pad_valid` all-True over a 128-row graph is a NO-OP, agreeing to the
same 1.28e-15 / 5.33e-15, because pad rows carry `present=0` and `mask_2d`
has already dropped them. The two masks are redundant on pad rows and only
`present` is load-bearing there. What does fire is switching `pad_valid` OFF
over the back half of the REAL rows: 1.38 (L=30) and 10.06 (L=49).

### `test_tied_overlap_raises` → `tests/families/potts_mpnn/test_tied_overlap.py` (0a658ab7)

The guard exists (`sample_host.py:223-226`) and was already unit-tested at
`test_sample.py:317`. What was missing is that **nobody can reach it**, now
filed as **debt #2459**:

- `features.tied_pos` is populated only at `featurize.py:441` from
  `tied_positions_dict`; when that is `None`, `:438-439` sets it to `()`.
- Both driver call sites (`driver.py:573`, `:774`) pass no
  `tied_positions_dict`.
- So `tied=bool(features.tied_pos)` (`sample_host.py:337`) is permanently
  `False` — the **entire tied decode path is dead**, not merely its error case.
- Independently, `RunSpecification.tied_positions`' only consumer,
  `utils/autoregression.resolve_tie_groups` (`:41`), has **zero callers** in
  `src/aminx/`.

Same shape as debt 2443 (`pssm_json`/`bias_by_res_json`, fixed in `fb9e91cd`):
the field parses, validates, and reaches nothing.

## Genuinely missing — blocked on a scoped change

### `test_decode_invariant_to_padding`

Spec §4.2(d) names the specific hazard: `nbr_valid[t:t+1]` must be passed as
`DecoderLayer(attention_mask=…)` to mask *messages* from pad rows, because
"zeroing context alone is insufficient since `message_mlp([h_i,0]) ≠ 0`".
`decode.py:226,235` does pass it, so the implementation looks right — but it is
untested and, as above, unexercised.

It cannot be written today. `prepare_sample` (`sample_host.py:277`) hardcodes
`l_pad = int(features.L_total)`, and unlike the energy case there is no
reachable seam below it — `absolute_energies` could be called directly, but
`PottsARDecode.__call__` takes 20+ assembled arrays that only `prepare_sample`
builds.

**Unblock:** give `prepare_sample` an `l_pad: int | None = None` parameter
defaulting to `features.L_total`, so behaviour is unchanged and the test can
ask for a larger pad. That touches `src/aminx/` — a `_SCOPED_PREFIXES` path —
so it must land **after** the ledger wave and its gate, or it invalidates all
seven rows again.

## Carry-over

Not yet triaged, in rough priority order: the six `test_divergence_*` rows
(§6.5b divergence register), `test_knob_semantics_laser_bias_minp` /
`_laser_stored_logits_minp` / `_laser_bias_single_aa` / `_laser_score_order`,
`test_knob_semantics_nodes_gibbs`, `test_knob_semantics_binding_converge_stop`,
`test_knob_semantics_order_generation`, `test_score_energy_gapped_alignment`,
`test_laser_score_chi_candidate_mismatch`,
`test_tied_rank_flat_matches_upstream`, and
`test_valid_neighbour_set_matches_upstream` (needs the upstream oracle, so it
belongs in `tests/port/` — itself a scoped path).
