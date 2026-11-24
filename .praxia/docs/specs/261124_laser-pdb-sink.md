---
title: 'LASEr sample PDB sink: full-atom PDBs through proxide.write_pdb (debt #3565, option C)'
description: 'Spec for an opt-in per-sample PDB sink on the LASErMPNN sample path: identity plumbing into LaserFeatures, a pure Full-protein builder, a SampleStages hook honouring output_fasta_only, tests, and the proxide>=0.3.0 pin bump'
status: draft
task_id: 261124_aminx-laser-pdb-sink
date: '261124'
---
# Specification: LASEr sample PDB sink

## Overview

Make `sample` on `lasermpnn` optionally write one full-atom PDB per sample (design) by building a Full-layout
proxide `Protein` from the sampled sequence, chi-built coordinates and the input structure's identifiers,
and calling `proxide.write_pdb`. This is option C of
`.praxia/docs/decisions/261004_laser-pdb-output-scope.md` (owner-decided 2026-11-23, debt #3565): the minimal
writer, with documented divergences from upstream LASErMPNN. The parity port (option B: backbone N-H
imputation, titratable-H cleanup) is a separate epic and out of scope.

Today the sample path returns `sidechain_coords` (L, atom, 3) in the sink (`sample_host.py:268`) but nothing
writes a PDB, and `LaserFeatures` has no chain id, residue number, insertion code, residue name or ligand atom
name (`featurize.py:455-486`), so a PDB cannot be written from the features as they stand.

## 1. Assumption Ledger

Tags: VERIFIED = read at file:line or spiked; UNVERIFIED = open. The spike-run harness
(`scripts/loop/adversarial_metrics.py`) does not exist in this workspace, so S1/S2 were run by hand
(`.praxia/spikes/261124_laser-pdb-sink/S{1,2}/spike.py`, outputs beside them; the directory is gitignored, so they stay local to this worktree) with the pre-registration
`spike.json` written first; they are not spike-run records. Paths are workspace-relative; proxide paths are in
the proxide repo at its 0.3.0 release worktree. Evidence items beginning `read: proxide ...`, `read: pip index ...`
and `spike:` are outside the workspace-relative `read:` grammar (other repo, a command, a local-only file) and are
not resolvable by a reviewer from this repo alone; A3's rotatable-H claim rests on a local spike only.

| ID | Assumption | If false | Status | Evidence |
|----|------------|----------|--------|----------|
| A1 | `build_rotamers(..., add_nonrotatable_hydrogens=True)` returns (L, 24, 3) with NaN for absent atoms; the heavy build returns (L, 14, 3) | Name table width and NaN mask change | VERIFIED | read: src/aminx/families/laser_mpnn/featurize.py:762-769; read: src/aminx/families/laser_mpnn/featurize.py:851-853; spike: S1 (shape 24) |
| A2 | On the protonated axis, slot k of a residue is named `_hydrogen_extended(letter)[k]` and no non-NaN slot lies past that list | Per-atom names would be wrong | VERIFIED | read: src/aminx/families/laser_mpnn/featurize.py:224-225; read: src/aminx/families/laser_mpnn/featurize.py:777-791; spike: S1 (empty "beyond", backbone NaN 0) |
| A3 | On the heavy axis, slot k is named `_ATOM_ORDER[letter][k]`; rotatable H slots (HG, HG1, HD1, HE2) can be filled, HH sometimes is not | A heavy-layout name table would mislabel atoms | VERIFIED | spike: S2 (no slot beyond `_ATOM_ORDER`; HG/HG1/HD1/HE2 filled); spike: S1 (HH NaN on 3 Tyr in 1mbn) |
| A4 | Every protein atom name starts with its element letter (C, H, N, O, S), so element = name[0] | Element column wrong | VERIFIED | spike: S2 (first letters {C,H,N,O,S}); read: src/aminx/families/laser_mpnn/featurize.py:57-78 |
| A5 | `featurize` parses with ProDy (`pr.parsePDB`), not proxide, so identity comes from ProDy `chain.getChid()`, `residue.getResnum()`, `residue.getIcode()` | Plumbing would read the wrong API | VERIFIED | read: src/aminx/families/laser_mpnn/featurize.py:1345-1370 |
| A6 | `LaserFeatures` has no chain id, resnum, icode, residue name or ligand atom/residue names; `resnum_indices` is `arange(L)`; `chain_indices` is the ProDy chain enumeration index (ligand-only chains consume indices) | Plumbing scope changes | VERIFIED | read: src/aminx/families/laser_mpnn/featurize.py:455-486; read: src/aminx/families/laser_mpnn/featurize.py:1362; read: src/aminx/families/laser_mpnn/featurize.py:1481; spike: S1 (no identity attrs, `resnum_indices` 0..4) |
| A7 | A standard residue whose N, CA or C is NaN is silently skipped (neither protein nor ligand), so L can be less than the input's residue count | Alignment rule in D1 is unnecessary | VERIFIED | read: src/aminx/families/laser_mpnn/featurize.py:1384-1385 (the spike inputs had no such residue; L equalled the ProDy count, 153 and 76) |
| A8 | Identity appended inside the same loop, after the `continue` guards and beside `_append_protein`, is row-aligned with `sequence_indices` by construction | Misaligned chain/resnum | VERIFIED | read: src/aminx/families/laser_mpnn/featurize.py:1372-1399 (all appends happen at :1386, after the HOH, non-amino, ncAA and NaN `continue`s); prior art: `git show 095a49e0` did exactly this |
| A9 | Ligand atoms come from `_LigandSink.push` calls for HOH (if `use_water`), non-amino residues, and ncAA residues plus synthetic methyl caps; only coords, Z and a per-residue `sub` index are kept, so names and residue ids are lost today | D2 plumbing unnecessary | VERIFIED | read: src/aminx/families/laser_mpnn/featurize.py:1228-1243; read: src/aminx/families/laser_mpnn/featurize.py:1282-1295; read: src/aminx/families/laser_mpnn/featurize.py:1372-1380 |
| A10 | `ignore_ligand` yields empty ligand arrays; ligand coords are the input's (`noised_features` replaces only `backbone_coords`) | Ligand placement/noise semantics | VERIFIED | read: src/aminx/families/laser_mpnn/featurize.py:1298-1314; read: src/aminx/families/laser_mpnn/driver.py:683-686 |
| A11 | `seq_log_prob` is named a log prob but holds `softmax(stored logits)[sampled token]`, a probability in [0, 1] | B-factor would be a log value | VERIFIED | read: src/aminx/families/laser_mpnn/sample_host.py:382-388; read: src/aminx/families/laser_mpnn/sample_host.py:614 |
| A12 | `produced["sequence"]` is canonical-alphabet; the LASEr letter index needs `LASER_OF_CANONICAL`; `sidechain_coords` was built from the LASEr indices | Residue names wrong | VERIFIED | read: src/aminx/families/laser_mpnn/sample_host.py:612-621; read: src/aminx/families/laser_mpnn/driver.py:113 |
| A13 | `output_fasta_only` makes `_one_level` return only `{"fasta"}`; no coords are built (`coords is None`) | Hook would see no coords | VERIFIED | read: src/aminx/families/laser_mpnn/sample_host.py:232-235; read: src/aminx/families/laser_mpnn/sample_host.py:540-542; read: src/aminx/families/laser_mpnn/sample_host.py:633-636 |
| A14 | `output_fasta` (not only) leaves every other array in place and adds `fasta` | PDB sink interaction | VERIFIED | read: src/aminx/families/laser_mpnn/sample_host.py:633-636; read: src/aminx/families/laser_mpnn/sample_host.py:290-291 |
| A15 | The runner calls `stages(batch, chunk_start, chunk_count, ...)` once per (structure batch, window) on the host, in numpy, before slicing to `L_total`; global sample row = `chunk_start*n_temp + local_idx`; a noise axis (D_noise) sits at axis 1 | Hook placement / file naming | VERIFIED | read: src/aminx/host/family_runner.py:296-305; read: src/aminx/host/family_runner.py:74-86; read: src/aminx/families/laser_mpnn/sample_host.py:431-445; read: src/aminx/families/laser_mpnn/sample_host.py:608-614 |
| A16 | `stages()` is built before any batch is consumed, so a constructor-time validation fails before sampling | Late failure after hours | VERIFIED | read: src/aminx/host/family_runner.py:166-168 |
| A17 | `_canonical_structure_id` is the input path stem (else `structure_<i>`) | File naming | VERIFIED | read: src/aminx/host/_sampling_helper.py:23-34 |
| A18 | `LaserOptions` is a frozen dataclass without `__post_init__`, so adding fields and a `__post_init__` is a local change | Options plumbing | VERIFIED | read: src/aminx/run/options.py:116-146 (no `__post_init__` in the class body) |
| A19 | `proxide.write_pdb(protein, path, *, record_type="source", insertion_codes="write", provenance="none")` exists in 0.3.0 and is exported at the top level | Wrong call | VERIFIED | read: proxide src/proxide/io/writing.py:813-820; read: proxide src/proxide/__init__.py:76 |
| A20 | Full path: needs flat `full_coordinates` (N,3), `atom_names`, `res_names`, `atom_res_index` (written verbatim as the PDB resnum), `elements`; rejects non-finite coordinates and non-finite B/occupancy; per-atom `hetero` (np.bool_) drives HETATM; per-atom insertion codes one char (blank " "); chain 1 char, resname <= 3, atom name <= 4, resnum in [-999, 9999], <= 99999 atoms else ValueError | Input NaN, limits, and pre-validation | VERIFIED | read: proxide src/proxide/io/writing.py:178-181; read: proxide src/proxide/io/writing.py:262-322; read: proxide src/proxide/io/writing.py:521-524; read: proxide src/proxide/io/writing.py:728-760; read: proxide src/proxide/core/containers.py:261-264 |
| A21 | A Full `Protein` is constructed directly with `coordinates=zeros((n,37,3))`, `aatype`, `residue_index`, `chain_index`, `chain_ids` plus the Full fields (the proxide acceptance-test pattern); no `__post_init__` validation beyond what the writer does | Builder signature | VERIFIED | read: proxide tests/io/test_writing_full_atom.py:165-184 (the `_full` helper, merged on proxide main); read: proxide src/proxide/core/containers.py:103-175 (plain dataclass fields, no `__post_init__` in the file) |
| A22 | PDB B-factor column is `%6.2f`: a probability prints with two decimals (0.873 -> 0.87), so the round trip is exact to 0.005 only | Test tolerance | VERIFIED | read: proxide src/proxide/io/writing.py:528-560 |
| A23 | `proxide==0.3.0` is not on PyPI yet: the index lists 0.1.0, 0.2.0, 0.2.1 only; the project venv here has 0.2.0 whose `write_pdb(protein, path)` has no `insertion_codes`/`provenance` | The pin bump cannot merge or pass CI until the release | VERIFIED | read: `pip index versions proxide` on 261124 -> 0.2.1 latest; read: aminx `.venv` `inspect.signature(proxide.io.write_pdb)` = `(protein, path)` |
| A24 | `options_from_json_value(LaserOptions, ...)` accepts new `str\|None` and `Literal` fields without a schema table | Spec-JSON round trip breaks | UNVERIFIED | deferred: not read this round; T4 adds a JSON round-trip test that settles it |
| A26 | `uv lock` can resolve proxide 0.3.0 once released: no dependency in the lock caps proxide (xtrax 0.4.1b9's lock entry lists no proxide requirement; only aminx pins it, `uv.lock:232`) | T1 blocked by a second constraint | VERIFIED | read: uv.lock:232; read: uv.lock (the `name = "xtrax"` package block has no proxide line, checked by awk 261124) |
| A27 | The sidecar path is `<given path name>.prov.toml` and records the given path's resolved name and sha, so writing to a temp name then renaming would orphan and mis-record it | Atomic-rename design breaks test 11 | VERIFIED | read: proxide src/proxide/io/writing.py:1158-1160; read: proxide src/proxide/io/writing.py:1137-1138 |
| A28 | A PDB input cannot carry a chain id > 1 char or a resnum outside [-999, 9999] (fixed columns 22 and 23-26), so only the atom-count limit is reachable from featurize's PDB-only input | Pre-validation scope | VERIFIED | read: src/aminx/families/laser_mpnn/featurize.py:1345 (PDB via `pr.parsePDB` only); PDB v3.3 column widths (the same widths proxide's `_columns` reads, proxide tests/io/test_writing_full_atom.py:140-155) |
| A25 | A `Protein` built per sample with `n` rows and a dummy `(n,37,3)` coordinates array is cheap enough (3 MB at L=300; 44 MB at the 99,999-atom cap) | Memory or time per sample | UNVERIFIED | deferred: no proxide 0.3.0 install exists in this sandbox (the proxide venvs are 0.1.0 and uninstalled), so the builder could not be run; T3 times it on one production-shape sample and records it in the PR |

## 2. Decisions

### D1. Identity plumbing (protein residues)

Add three fields to `LaserFeatures` (names reuse the archived commit 095a49e0 so the history is searchable):
`pdb_chain_ids: tuple[str, ...]`, `pdb_resnums: Int[np.ndarray, " L"]`, `pdb_icodes: tuple[str, ...]`.
Fill them in `featurize()` as three more lists appended immediately after the `_append_protein(...)` call
(`featurize.py:1386`), from `str(chain.getChid()).strip()`, `int(residue.getResnum())`,
`str(residue.getIcode()).strip()`. Because they are appended only on the path that appends `sequence`, every
row keeps its own identity whatever was skipped (A7, A8). Pass them through `_assemble` (one constructor, A6).
`resnum_indices` stays `arange(L)` (the model index; changing it would change scoring and the oracle waves).
The blank icode is stored as `""` and becomes `" "` at the writer. No change to `ARRAY_FIELDS`/`JSON_FIELDS`
(`featurize.py:417-446`; only exported, not consumed by tests or src, grep 261124).

Residue **names** are not input names: the sampled sequence replaces them. The builder derives the 3-letter
name from the LASEr letter with the inverse of `_LONG_TO_SHORT` (`featurize.py:34`), one canonical name per
letter (HIS, not HIP/HID; `X` -> `UNK`, open question Q5).

Input residues skipped for a NaN backbone are absent from the output PDB (a numbering gap); this is stated in
the module docstring.

### D2. Ligand atoms

Recoverable, with plumbing. Extend `_LigandSink.push` (`featurize.py:1236`) to take per-atom
`names`, a `resname`, `chain`, `resnum`, `icode` and a `synthetic: bool`, store them, and add to
`LaserFeatures` (concatenated like `ligand_elements`, one entry per ligand atom, empty tuples when ligand
arrays are empty): `ligand_atom_names: tuple[str,...]`, `ligand_res_names: tuple[str,...]`,
`ligand_chain_ids: tuple[str,...]`, `ligand_resnums: Int[A]`, `ligand_icodes: tuple[str,...]`,
`ligand_synthetic: Bool[A]`. Sources: ProDy `residue.getNames()`, `getResname()`, chain/resnum/icode of the
residue. `ligand_elements` already exists (`_elements`, `featurize.py:1106`).

- Written as HETATM: `hetero=True`, `b_factor=0.0`, `occupancy=1.0`, after all protein atoms (upstream appends
  the ligand last, decision doc step 4).
- `ignore_ligand=True`: the ligand arrays are empty (A10), so no ligand is written.
- `use_water=True`: HOH atoms are ligand atoms and are written as HETATM `HOH`.
- Noncanonical-residue ligands (`noncanonical_aa_ligand`): the residue's real atoms are written; the
  synthetic methyl caps (`_push_ncaa_ligand`, `featurize.py:1282-1295`) carry `synthetic=True` and are NOT
  written, since they are not input atoms (Q3).
- Ligand coordinates are the input's, not noised (A10), so a noised-backbone sample has an un-noised ligand.
  Document it.

### D3. Per-atom names, order, NaN handling (vectorised, no per-atom Python loop)

Module `src/aminx/families/laser_mpnn/pdb_sink.py` holds a pure builder
`build_full_protein(features, laser_sequence, coords, token_prob) -> proxide.core.containers.Protein`
(no I/O, no proxide write) and `write_sample_pdb(...)`.

1. Choose the table by the coords' atom width `W`: 24 -> protonated names (`_hydrogen_extended`), 14 -> heavy
   names (`_ATOM_ORDER`) (A1-A3). Add `residue_atom_names(letter, *, protonated)` to `featurize.py` (same
   signature as 095a49e0). Build once, at import, a `(21, W)` object array padded with `""`.
2. `names = table[laser_sequence]` -> `(L, W)`; `present = isfinite(coords).all(-1)`. Flatten row-major with
   `present`: this drops every NaN slot, including slots a residue never has (Ala slots 9+) and rotatable
   H slots that were not placed (HH). No NaN ever reaches proxide (A20). Assert `names[present] != ""` (a
   non-NaN slot with an empty name is a bug, S1/S2 found none).
3. Per-atom arrays by `np.repeat` of per-residue arrays with `present.sum(1)`: `res_names`, `atom_chain_ids`,
   `atom_res_index` (= `pdb_resnums`, the real PDB number), `insertion_codes` (`" "` for blank), `b_factors`
   (= `token_prob`, float32). `elements = [n[0] for n in names]` (A4); `occupancies` = 1.0; `hetero` False.
4. Append the ligand atoms (D2) with the same arrays, `hetero=True`.
5. Atom order within a residue is slot order (dataset order: heavy atoms, with rotatable H interleaved per
   `_ATOM_ORDER`, then the non-rotatable H), not IUPAC order. Residues follow input order.

`chain_index`/`chain_ids` for the dummy Atom37 fields: `chain_ids = sorted(set(atom_chain_ids))`,
`chain_index` its per-atom index (the A21 pattern).

### D4. The sink and hook

- **Where**: `SampleStages.__call__` (`sample_host.py:415-429`). After `rows` are computed, for each structure
  `i` in the batch call `self._emit_pdbs(structure_index=batch.input_indices[i], item=batch.arrays["sampled"][i],
  produced=rows[i], chunk_start=chunk_start)`. `__call__` runs once per (batch, window) on host numpy (A15), so
  there is no new runner hook, no xtrax change, and the returned arrays are untouched. The tied path
  (`_one_level_tied`) yields the same `produced` keys, so one hook covers both.
- **Per file**: for each row `r` in `produced["sequence"]` (shape `(N, L)` or `(N, D, L)` with a noise axis at
  axis 1, A15): `laser_sequence = LASER_OF_CANONICAL[sequence]` (A12), `coords = sidechain_coords[r]`,
  `token_prob = seq_log_prob[r]`; features = `item.features` (or `item.variants[d].features`, whose
  noised backbone matches that level's coords). Identity is level-independent.
- **File name**: `<pdb_output_dir>/<structure_id>_n<row:05d>.pdb`, `row = chunk_start*n_temp + local_row`, the
  same global row index as the zarr `N` axis, so a file joins to its arrays and to its temperature
  (`row % n_temp`); with a noise axis add `_d<level:02d>`. `structure_id` = `_canonical_structure_id` (A17);
  two inputs with the same stem collide, so the hook includes `s<input_index>` when stems repeat (checked once
  in `__init__` from `_inputs(spec)`).
- **Rerunnable, written in place**: call `write_pdb` on the final path. proxide builds and validates every
  row before it opens the file (writing.py:787-790), and its sidecar is named and hashed from the path it is
  given (A27), so a temp name plus `os.replace` would orphan the sidecar and record the wrong path. No
  temp-rename. A file is complete iff it ends with `END`; re-running overwrites (deterministic for a seed).
  There is no reuse of an existing file, so no resume logic changes `[outcomes]`.
- **Opt-in flag**: `LaserOptions.pdb_output_dir: str | None = None`. `None` (default) writes nothing and
  changes nothing: behaviour and outputs stay as today. A set value enables the sink. If it is relative it is
  resolved against `spec.run_spec.io.output_dir` when that is set, else the CWD; the directory is created on
  first write (Q1).
- **output_fasta_only**: the combination `pdb_output_dir` set AND `output_fasta_only=True` raises `ValueError`
  in `LaserOptions.__post_init__` (new), so it fails at construction and for spec-JSON loads, before any
  work. Rationale: FASTA-only runs never build coordinates (A13) and `_include_coords` documents "coordinates
  stand in for the PDB sink"; a silent no-op would hide a mistyped spec (Q2).
- **output_fasta**: independent. `output_fasta=True` with a PDB dir writes PDBs and also keeps the `fasta` array
  (A14).
- **Zarr sinks**: unchanged. `sidechain_coords`, `sequence`, `seq_log_prob` still go to the store when
  `output_h5_path` is set; the PDB sink works with or without a store (in-memory runs). Test: the arrays are
  bit-equal with and without the sink.
- **Pre-validation** (fail before sampling, as a skipped input, not mid-run): only the atom-count limit is
  reachable (A28). `sample_batches(spec)` has no model, so it cannot know the atom width; it checks the
  conservative bound `L * (MAX_PROTONATED_ATOMS + 1) + n_ligand_atoms > _MAX_PDB_ATOMS` (99,999, a module
  constant) and appends the input to `pending` with reason `pdb_sink: <n> atoms exceed the PDB limit`
  through the existing skipped-input path (`sample_host.py:188-191`). This may skip a structure that would
  have fit in the heavy layout; accepted (Q10). `write_sample_pdb` still lets a proxide `ValueError`
  propagate with the structure id added, as a backstop.

### D5. B-factor and occupancy

`b_factors` = the sampled-token probability, one value per residue, `np.repeat`ed over that residue's atoms
(A11). It is `softmax(stored logits)[token]` after min-p, so a value of 1.0 can occur for fixed or
single-candidate positions. Range [0, 1], printed `%6.2f` (A22). Ligand B = 0.0. For positions the user fixed
(`fixed_positions`, `repack_only`, `fix_from_bfactor`) the value is the model's own probability of the kept
token, not 1.0 (Q6). `occupancies` = 1.0 explicitly (a blank column would read back as the default anyway,
A22, but an explicit 1.00 matches what a refinement tool expects).

### D6. Provenance

`LaserOptions.pdb_provenance: Literal["none", "sidecar"] = "none"`, validated in `__post_init__`, passed
verbatim as `provenance=` to `write_pdb`. `"sidecar"` makes proxide write `<file>.pdb.prov.toml` (A19, read
at proxide writing.py:1158-1166). It is not forwarded when `pdb_output_dir` is `None`. aminx adds nothing
to the sidecar.

### D7. Tests (CPU-fast, `tests/families/laser_mpnn/test_pdb_sink.py`, plus additions to `test_featurize.py`)

Fixtures: `tests/data/fampnn_parser_stress.pdb` (icodes `A`, `B`, 2 ligand atoms; 76 residues, A7 spike) and
`test_driver._write_pdb`-style synthetic PDBs written in `tmp_path`; the tiny model of `test_sample.py` for the
end-to-end cases. Narrow runs only (one file); run on titanix per the user's rule.

Featurizer (no proxide needed):
1. **Identity alignment (must fire)**: synthetic 3-residue PDB, middle residue missing its `CA`: `L == 2`, the
   rows' `pdb_resnums == [1, 3]`. Control (must not fire): the same PDB complete -> `[1, 2, 3]`.
2. **Insertion codes**: stress file -> `set(pdb_icodes) >= {"", "A", "B"}`; `len` of all three identity fields
   equals `L`.
3. **Ligand identity**: stress file -> `len(ligand_atom_names) == ligand_coords.shape[0] == len(ligand_elements)`
   and names equal the ProDy names; `ignore_ligand=True` -> all empty; `noncanonical_aa_ligand` fixture ->
   caps have `ligand_synthetic=True`, real atoms False.

Builder (imports proxide >= 0.3.0 `Protein`; skipped with `importorskip` only while the pin is below 0.3.0 and
never in CI after T1):
4. **Names/NaN (must fire)**: for each of the 21 letters, both layouts, a fabricated full coord array: built
   `len(atom_names)` equals the count of finite slots; no `""` name; `elements == name[0]`. Control: the same
   array with one extra NaN slot injected -> exactly one atom fewer.
5. **Broadcast**: `b_factors` per atom equals the residue's probability; `hetero` is True exactly for ligand
   atoms; `insertion_codes` all one character; `atom_res_index` equals the input resnum.
6. **Round trip through proxide**: `write_pdb(protein, path)`, read the columns by position (as proxide's
   `_columns` does) and `parse_structure(path, OutputSpec(coord_format=CoordFormat.Full), use_jax=False)`:
   atom names/order, resnames read from the written columns (HIS), and from `parse_structure` with `HIP` mapped to `HIS` as proxide's own
   test does (`tests/io/test_writing_full_atom.py:263`; the parser relabels a His carrying HD1 and HE2, which every
   protonated His here does), coords within 5e-4, `hetero` flags, chain/resnum/icode columns,
   B column within 0.005 of the probability, HETATM count == ligand atom count.

End to end (`sample(spec)` with the tiny model, 2 samples x 2 temperatures):
7. **Fires**: `pdb_output_dir` set -> exactly 4 files, named by global row; B column varies across residues.
   **Must not fire**: default options -> no directory created, no file.
8. **Fasta interaction**: `output_fasta_only=True` + dir -> `ValueError` at `LaserOptions(...)`; `output_fasta=True`
   + dir -> PDBs written and `fasta` array present; neither flag -> no `fasta` array.
9. **Non-interference**: with and without the sink, `sequence`, `seq_log_prob`, `chi_deg`, `sidechain_coords` are
   bit-equal for the same seed.
10. **Noise axis**: two noise levels -> `_d00`/`_d01` files; backbone coordinates differ between them, identity
    columns do not.
11. **Provenance**: `"sidecar"` -> `<name>.pdb.prov.toml` exists; `"none"` -> absent.
12. **Pre-validation**: with `_MAX_PDB_ATOMS` monkeypatched to 100, a 5-residue PDB (5*24 > 100) lands in
    `skipped_inputs` with a `pdb_sink:` reason and no exception; control (must not fire): the same PDB with the
    constant unpatched samples normally, and with the sink disabled the patched constant is never consulted.
13. **JSON round trip** (A24): `LaserOptions(pdb_output_dir="x", pdb_provenance="sidecar")` through the spec JSON
    loader returns equal options.

### D8. proxide pin

`pyproject.toml:24` `proxide>=0.2.1` -> `proxide>=0.3.0`, then `uv lock`. The new floor is needed for
`insertion_codes`, per-atom `hetero` and `provenance` (A19, A23). **Hard dependency on proxide 0.3.0 being on
PyPI** (release PR maraxen/proxide#109 pending): T1 cannot merge, and no later task can pass CI, before the
release. Order: T1 is the only task blocked on it; T2 (featurizer plumbing) has no proxide dependency and may
land first. If the release slips, T3-T5 stay in a PR that is held, not worked around with a vendored writer
or `dangerouslyDisableSandbox` installs from a git SHA.

### Divergences from upstream LASErMPNN (documented in the module docstring and the decision doc)

- No backbone amide N-H imputation (decision doc step 2): backbone H absent.
- No titratable-H cleanup: no Cys disulfide/HG handling, no His tautomer choice (decision doc step 3).
  Rotatable H as built by `build_rotamers` (HG, HG1, HD1, HE2, HH when placed) are written as built.
- Atom order within a residue is dataset slot order.
- Ligand coordinates are the input's, un-noised; ncAA methyl caps are not written.
- B-factor precision is 2 decimals (PDB column), and Q6 for fixed positions.
- Residue names are canonical for the sampled letter; input names are not preserved. Skipped
  partial-backbone residues are absent.

## 3. Acceptance Criteria

- `sample` with default options is byte-for-byte unchanged in outputs (test 7, 9) and no PDB code runs.
- With `pdb_output_dir` set, every sampled design yields one PDB whose residues carry the input chain id,
  residue number and insertion code, whose atoms carry dataset names and elements, whose ligand atoms are
  HETATM, and whose B column is the sampled-token probability (tests 4-7).
- The PDB parses back through `proxide.parse_structure` with identical atom names, hetero flags and
  coordinates to 5e-4 (test 6).
- `output_fasta_only` plus a PDB dir fails at options construction; `output_fasta` composes (test 8).
- A structure over the PDB atom limit is skipped up front with a reason, never an end-of-run exception (test 12).
- `pyproject.toml` requires `proxide>=0.3.0` and the lock resolves.

## 4. Fixer Tasks

### T1. Pin bump (blocked on proxide 0.3.0 on PyPI)
Edit `pyproject.toml:24`, run `uv lock`. **Files**: `pyproject.toml`, `uv.lock` (modify).
**Gate**: `uv run python -c "import inspect, proxide; p = inspect.signature(proxide.write_pdb).parameters; assert 'insertion_codes' in p and 'provenance' in p"` exits 0; `uv lock --check` exits 0. **Scope**: ~5 LOC + lock.

### T2. Identity plumbing, protein and ligand (D1, D2)
Add the 3 + 6 `LaserFeatures` fields, extend `_LigandSink.push`/`_push_ncaa_ligand`, `_append_protein` callers,
`_assemble`; add `residue_atom_names`. No behaviour change to any existing field.
**Files**: `src/aminx/families/laser_mpnn/featurize.py` (modify), `tests/families/laser_mpnn/test_featurize.py`
(add tests 1-3). **Gate**: `uv run --group test pytest tests/families/laser_mpnn/test_featurize.py
tests/families/laser_mpnn/test_featurize_upstream_parity.py` (single files, on titanix) green, including
the unchanged parity file. **Scope**: ~120 LOC + ~90 test.

### T3. Pure builder and writer (D3, D5, D6)
Create `pdb_sink.py`: name tables, `build_full_protein`, `write_sample_pdb` (atomic rename, provenance
passthrough). Time one production-shape sample (L ~ 300, protonated) and record it in the PR (A25).
**Files**: `src/aminx/families/laser_mpnn/pdb_sink.py` (create), `tests/families/laser_mpnn/test_pdb_sink.py`
(create, tests 4-6). **Gate**: `uv run --group test pytest tests/families/laser_mpnn/test_pdb_sink.py -k "builder or round_trip"`.
**Scope**: ~160 LOC + ~170 test. Depends on T1, T2.

### T4. Options, validation, hook (D4, D6, D7 tests 7-13)
Add `pdb_output_dir`, `pdb_provenance`, `__post_init__` to `LaserOptions`; the `SampleStages._emit_pdbs`
hook in `__call__`; pre-validation in `sample_batches`; same-stem naming guard.
**Files**: `src/aminx/run/options.py`, `src/aminx/families/laser_mpnn/sample_host.py` (modify),
`tests/families/laser_mpnn/test_pdb_sink.py` (add tests 7-13). **Gate**: `uv run --group test pytest
tests/families/laser_mpnn/test_pdb_sink.py tests/families/laser_mpnn/test_sample.py` (two files, on titanix).
**Scope**: ~110 LOC + ~140 test. Depends on T3.

### T5. Docs and lint
Module docstring with the divergence list; set `261004_laser-pdb-output-scope.md` `status` to `accepted` with a
line naming option C and this spec (edit by hand only the frontmatter and the status line), then
`docs(action="index")` through the MCP tool; close debt #3565 with the PR link (owner step).
**Files**: `.praxia/docs/decisions/261004_laser-pdb-output-scope.md` (modify), `.praxia/docs/INDEX.md`
(regenerated). **Gate**: `uv run --group lint ruff check src tests/families/laser_mpnn` and
`uv run --group lint ruff format --check src tests/families/laser_mpnn` exit 0; `docs(action="check")` clean.
**Scope**: ~30 LOC.

## 5. Risks

| Risk | Mitigation |
|------|-----------|
| proxide 0.3.0 not released when work is ready (A23) | T2 lands independently; T1/T3-T5 held in a draft PR; no vendored writer |
| New `LaserFeatures` fields break a constructor or the oracle waves | One constructor (A6); `resnum_indices` untouched; T2 gate runs the upstream-parity file |
| Misaligned identity after skipped residues | Append beside `_append_protein` (A8) and test 1 with a skipped-residue control |
| The 99,999-atom PDB limit aborts a long run | Conservative pre-validation in `sample_batches`, skip with reason (D4, test 12); mmCIF output is a separate follow-up. Resnum and chain limits are unreachable from PDB input (A28) |
| Sidecar plus temp-rename orphans the `.prov.toml` and records the wrong path (A27) | Write in place, no temp-rename (D4); test 11 asserts the sidecar name and its recorded path |
| Sink slows sampling (a Protein per sample) | Vectorised builder, no atom loop; time it (A25); sink is opt-in, default off |
| Hook changes returned arrays | Hook only reads `rows`; bit-equality test 9 |
| Filename collisions (same stem, resumed run, multi-level) | Row-index and level suffixes, same-stem guard, atomic overwrite |
| Users read the PDBs as upstream-equivalent | Divergence list in docstring, decision doc and PR; no backbone H, no titratable cleanup |
| Rollback | Revert T4 (hook and options) to disable; T2's added fields are inert. Default is off, so no existing run changes |

## 6. Open questions for the owner

Each has a recommended answer; the spec is written to the recommendation.

**Resolved 2026-11-24.** The owner chose the recommended answer on Q1, Q4, Q6 and Q10 (via AskUserQuestion).
Q2, Q3, Q5, Q7, Q8 and Q9 take their recommended answers by default (low-stakes; the owner may override before
T3/T4 start). The orchestrator spot-checked three citations against the code: `sample_host.py:415`, `run/options.py:124`
(the spec's `options.py:124` is this file), and proxide `writing.py:662` (bytes built before `open`, so no temp-rename).
Before T4, the implementer must read upstream's B-factor rule (Q6) to fill in the divergence note.

- **Q1 Flag shape.** `pdb_output_dir` (set = on, `None` = off) vs a boolean plus a derived directory.
  Recommended: `pdb_output_dir`, one knob, no hidden default path; relative paths resolve against
  `io.output_dir`.
- **Q2 `output_fasta_only` + PDB dir.** Error at construction vs silent PDB suppression. Recommended: error.
- **Q3 ncAA methyl caps.** Omit (synthetic atoms, not in the input) vs write as HETATM. Recommended: omit.
- **Q4 Tied sampling.** v1 writes only the primary structure's PDB; the partner (`sidechain_coords_2`) is
  not written. Recommended: primary only now; the partner needs its own identity and name suffix, a follow-up.
- **Q5 Residue name for token `X`.** `UNK` vs `XAA` (proxide's test uses `XAA`). `X` is disabled by default
  (`disabled_residues`, `options.py:124`). Recommended: `UNK`.
- **Q6 B-factor at fixed positions.** The model's own probability of the kept token (recommended, uniform rule)
  vs 1.0 for fixed positions. Upstream's behaviour for fixed positions was not read.
- **Q7 Occupancy.** Explicit 1.00 (recommended) vs blank.
- **Q8 Sidecar default.** `"none"` (recommended; sidecar only when asked) vs `"sidecar"`.
- **Q9 Ligand B-factor.** 0.0 (recommended) vs the input's B-factors (needs one more plumbed field).
- **Q10 Over-limit structures.** Skip with a reason (recommended) vs fail the run vs write mmCIF instead.

## References

- Decision: `.praxia/docs/decisions/261004_laser-pdb-output-scope.md`; debt #3565.
- Prior art (identity plumbing design only, not its writer): aminx commit `095a49e0` (tag
  `archive/wt/260929-laser-rotamers-wave`), `featurize.py` `pdb_chain_ids`/`pdb_resnums`/`pdb_icodes`,
  `residue_atom_names`.
- proxide: `src/proxide/io/writing.py` (`write_pdb`), `tests/io/test_writing_full_atom.py` (the Full
  construction pattern), release PR maraxen/proxide#109.
- Spikes: `.praxia/spikes/261124_laser-pdb-sink/S1`, `S2`.
