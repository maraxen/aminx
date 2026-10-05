# Model families: PottsMPNN and LASErMPNN

Aminx dispatches most work through its stock ProteinMPNN/LigandMPNN path. Two
families instead route through a `FamilyDriver`, which owns loading,
featurization and the forward passes for that model: **PottsMPNN** and
**LASErMPNN**.

Select one with `--model-family`. **It is a group-level option, so it goes
before the verb, not after it** — it is declared on the `aminx run` and
`aminx spec` groups (`@run_app.callback()` / `@spec_app.callback()`), and it
therefore applies to every verb underneath: `run sample|score|jacobian|inspect`
and every `spec emit-*`. Verb-specific flags like `--output-kind` still come
after the verb.

```bash
aminx run --model-family pottsmpnn score --output-kind energy --inputs 1ubq.pdb
aminx run --model-family lasermpnn sample --inputs 1ubq.pdb
aminx spec --model-family lasermpnn emit-sample --inputs 1ubq.pdb
```

`aminx run score --model-family ...` does **not** work, and
`aminx run score --help` does not list the flag, because it belongs to the
parent group. If you are looking for it, run `aminx run --help`.

You can usually leave it unset. `RunSpecification.__post_init__` derives the
family from `checkpoint_id`: a name starting `pottsmpnn_` resolves to
`pottsmpnn`, and `lasermpnn_` to `lasermpnn`. An explicit `--model-family` is
always respected, even when it disagrees with `checkpoint_id`.

## Which purposes each family actually runs

This is the part worth reading before trusting a number. A driver declares the
purposes it handles; for anything else a family may deliberately fall back to
the stock MPNN path. **The two families differ here, and the difference is not
symmetric.**

| purpose | `pottsmpnn` | `lasermpnn` |
| :--- | :--- | :--- |
| `sample` | PottsMPNN | LASErMPNN |
| `score:energy` | PottsMPNN | — (not an allowed output kind) |
| `score:ddg` | PottsMPNN | — (not an allowed output kind) |
| `score:nll` | **stock MPNN (declared fallback)** | LASErMPNN |
| `score:logits` | **stock MPNN (declared fallback)** | LASErMPNN |
| `score:proofread_unconditional` | — | LASErMPNN |
| `score:proofread_conditional` | — | LASErMPNN |
| `jacobian` | **stock MPNN (declared fallback)** | — |
| `inspect` | **stock MPNN (declared fallback)** | — |

So `--model-family pottsmpnn` with `--output-kind nll` returns **stock
ProteinMPNN** scores, not Potts scores. That is intentional — PottsMPNN
declares `jacobian`, `inspect`, `score:nll` and `score:logits` as fallback
purposes — but it means the family flag alone does not tell you which model
produced a number. Use `--output-kind energy` or `ddg` for the Potts model
itself. LASErMPNN declares an empty fallback set, so a lasermpnn request is
never answered by a different model.

Source of truth, if this table ever drifts: `_HANDLED` and
`mpnn_fallback_purposes` in `src/aminx/families/potts_mpnn/driver.py` and
`src/aminx/families/laser_mpnn/driver.py`.

## Allowed `--output-kind` per family

Enforced at spec construction by `_validate_output_kind`
(`src/aminx/run/specs.py`), which raises `ValueError` naming the allowed set.

| family | allowed output kinds |
| :--- | :--- |
| `proteinmpnn` | `nll` |
| `ligandmpnn` | `nll` |
| `membrane` | `nll` |
| `pottsmpnn` | `nll`, `logits`, `energy`, `ddg` |
| `lasermpnn` | `nll`, `logits`, `proofread_unconditional`, `proofread_conditional` |

## Installation: LASErMPNN needs an extra

LASErMPNN's host featurizer parses structures with ProDy, as upstream does, and
ProDy is **not** a core dependency. It ships in the `laser` extra:

```bash
uv sync --extra laser     # or: pip install 'aminx[laser]'
```

Without it a lasermpnn request fails with
`ModuleNotFoundError: No module named 'prody'`, raised from
`aminx/families/laser_mpnn/featurize.py` rather than from a message naming the
extra. PottsMPNN has no such requirement and works in a default install.

## Potts-specific spec helper

`aminx spec emit-potts` emits a Potts specification directly, alongside the
generic `spec emit-sample` / `emit-score` / `emit-jacobian` / `emit-inspect`.

## Known gaps

These are tracked and deliberately listed here so a result is not read as
stronger than it is.

- **An unrecognized `--model-family` is accepted and silently runs stock
  ProteinMPNN.** The flag is a free string: a typo such as `--model-family
  lasermpn` passes the CLI, passes `RunSpecification` (whose `Literal`
  annotation is not enforced at runtime), resolves no driver, and falls through
  to the stock path with no error and no warning. Check your spelling, or
  confirm the family on the emitted spec. Tracked as debt #2484.
- **Some declared options are not read.** Eight fields across
  `PottsMPNNOptions` / `LaserOptions` are accepted and ignored (debt #2435),
  and three specced `PottsMPNNOptions` fields are unimplemented (debt #2433).
  Setting one gets default behaviour, not an error.
- **Not ported from upstream:** PottsMPNN's recursive mutation beam search
  (#2070), LASErMPNN's entropy-based decoding `--ebd` (#2069), CA-only
  PottsMPNN checkpoints (#2068), and MSA vocab-22 PottsMPNN checkpoints
  (#2067).
- **`tied_positions` is inert on the Potts path** (debt #2459): setting it is a
  silent no-op.

## Numerical expectations

Both families are validated against their upstream implementations by the
`tests/port` waves, tiered as dtype/shape, f64, then f32. The f64 tiers are the
correctness check and pass tightly. f32 is a separate question: two f32
implementations of the same model make nearly the same error against an f64
truth, so an f32 comparison between them is a tighter test than either against
truth, and the bands reflect that. Use f64 only for parity assertions and
validation — it is not a production dtype, since it costs GPU performance.
