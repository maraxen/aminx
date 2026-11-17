---
title: LASEr oracle seal is weaker than the Potts one, and the pin lives outside git
description: Three seal mechanisms guard the port oracles with materially different strength; the LASEr one fails open and its pin is not in version control
status: draft
task_id: 260929_potts-laser-xtrax-compose
date: '261006'
---
# LASEr oracle seal is weaker than the Potts one, and the pin lives outside git

Found 261006 while checking which spec gates are enforced by something rather
than merely true. Nothing here is firing today: every seal that exists is
intact, verified below. The finding is that the LASEr waves hold a materially
weaker guarantee than the Potts waves, and the difference is invisible from the
repo.

## Three mechanisms, not one

The port suite seals its oracles three different ways. They are not equally
strong, and no document said so.

| waves | pin lives | checked | if the pin is absent |
|---|---|---|---|
| Potts (`a1_potts`) | **in the repo** — `tests/port/reference/a1_potts/oracles.sha256`, cross-checked against sha256 literals in `potts_head/algo.py:17-18` | unconditionally | cannot happen; the file is in git |
| B0 featurize | `oracle.npz.sha256` beside the dump | unconditionally (`tests/families/laser_mpnn/test_featurize_upstream_parity.py:45-47`) | `read_text()` raises — fails closed |
| LASEr ×4 (`laser_layers`, `laser_encoder`, `laser_score`, `laser_decode_step`) | `oracle_manifest.toml` beside the dump, **outside version control** | only if present | returns `None`, dump loads **unverified** |

`sealed.py`'s `_manifest_sha` fails open at four separate points — no manifest
file, a `[[dump]]` value that is not a list, no row matching the relative path,
or a `sha256` that is not a string. Each returns `None`, and `load_npz` then
skips the comparison and returns the array.

## The sharper half: a re-dump re-seals itself

The fail-open path is the obvious defect, but the quieter one matters more.

For **Potts**, the pin is a committed file. A re-dump that changes the oracle
fails the suite until somebody edits `oracles.sha256` and commits it — a
reviewable act with an author and a diff.

For **LASEr**, the dump and its manifest are written together by the dump
script, both outside git. A re-dump rewrites the npz *and* the hash that
guards it, so the seal silently re-seals itself around the new bytes. The LASEr
seal cannot detect an unreviewed re-dump at all — not because it fails, but
because it is asked the wrong question.

This also means "the dumps are sealed" (task #28) is true in a weaker sense for
LASEr than a reader would assume.

## Verified intact, 261006

`~/projects/aminx-oracles/dumps/b3_laser_owned` — the dir the waves actually
use (`scripts/redsox/laser_blank_waves.py:260`; the `sealed.py` default
`dumps/laser` does not exist on titanix). All 11 recomputed, **0 drifted**.

```
upstream_commit = e70f2c6d765416f7e29d51bfd6d4e08496438878
torch_version   = 2.4.1+cpu
shim_sha256     = 14fd25839e7501996f9c7444c13c4f813c3118995f2f121a04717ad13143f627
```

| wave | prec | path | sha256 |
|---|---|---|---|
| laser_layers | f64 | `laser_layers/oracle_f64.npz` | `4ead4c380f5b2d1b9598ce6174b06f029bf41b328ec9bfa08310952ac4081c12` |
| laser_encoder | f64 | `laser_encoder/oracle_f64.npz` | `2bceaf5bb25441d0ce7a339dc7299801aa9400b1920f7be59cfd2a17ec3d34d4` |
| laser_score | f64 | `laser_score/oracle_f64.npz` | `191455e5035b262d622558476ef9fe7ea6ce5b9ce20cb4a1484518d10faf3e89` |
| laser_decode_step | f64 | `laser_decode_step/oracle_f64.npz` | `b3bd6ec00b0de1c282ec3dbb38ddab0f14c846230f3324754aefae328ae5a0f3` |
| laser_rotamers | f64 | `laser_rotamers/oracle_f64.npz` | `d8d959f85439804d802b1d871eddd946c7cbf0d15cc5a0e0d7078226bb4d41c7` |
| laser_layers | f32 | `laser_layers/oracle_f32.npz` | `123971f6460fb1ac77762e05e57aee7d1ff367f1dee7c4dca052d06d5c7dda3f` |
| laser_encoder | f32 | `laser_encoder/oracle_f32.npz` | `169d5df3318b501b0e419a869cd0dc541b7a8c4b526d498e935e02fe7ad302ed` |
| laser_score | f32 | `laser_score/oracle_f32.npz` | `e91edd47283385a11373565e4addbee52fc9d1ceac1645270452adf84eb4b15b` |
| laser_decode_step | f32 | `laser_decode_step/oracle_f32.npz` | `bee8961dd0b4e52749e1d6ff7b804022ced4e418bcd0e9b67b8cb1324f401ee1` |
| laser_rotamers | f32 | `laser_rotamers/oracle_f32.npz` | `229ce12a37dbbe9dab0998bff7b0859a420c3d07c6651ff58c94ad618ca9b6bc` |
| laser_order | f64 | `laser_order/order_f64.npz` | `e09c14ac95a116c506d529636343c5dc881b00598cbdde134b6f9891905fb592` |

**This table is the point of the document.** It is the baseline captured while
the waves are known to pass against these exact bytes. Whoever implements the
fix should pin *these* values, not whatever the dumps happen to be at that
later moment — otherwise the repair silently ratifies any drift that happened
in between.

## Survey of every dump directory

| dir | npz | manifest rows | uncovered | note |
|---|---|---|---|---|
| `a1_potts` | 16 | 16 | 0 | also pinned in-repo |
| `a1_potts_v1` | 15 | 15 | 0 | |
| `b0` | 1 | — | — | no `oracle_manifest.toml`, but **not a gap**: b0 uses the `.sha256` sidecar and is checked unconditionally |
| `b1_laser` | 10 | 10 | 0 | |
| `b2_laser_prehook` | 10 | 10 | 0 | |
| `b2_laser_rerun` | 10 | 10 | 0 | |
| `b3_laser_owned` | 11 | 11 | 0 | the live one |

`b0` looked like an active hole on first pass and is not one. Reading its
consumer rather than inferring from the missing manifest is what settled it.

## The fix, and why it is not done here

Adopt the Potts double-entry pattern for the four LASEr waves: commit the
hashes above to `tests/port/reference/laser_*/`, check them unconditionally,
and raise when the manifest is missing instead of returning `None`.

That edits `tests/port/reference/*/sealed.py`, which is **scoped**
(`_SCOPED_PREFIXES`, `tests/knob_gate/_coverage.py:21-28`), so it invalidates
every ledger row and must ride the post-merge re-wave. It is recorded here
rather than applied.

## Related

- [[project_potts-laser-spec-converged]]
