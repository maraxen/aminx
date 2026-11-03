# Parity dashboard extractor

`extract_parity.py` writes one JSON file (`schema_version` `parity_dashboard_v1`)
that a static parity + knobs dashboard page can load.

It only copies recorded values: bathos catalog records, `scripts/parity/*.bth.toml`
sidecar text, and the knob manifests. It joins and groups them (ledger to run
record, control to result status, run history per slug) but computes no new
statistic. It does not run any test or experiment.

## Inputs

- `--catalog DIR`: bathos run records, one `run_<uuid>.parquet` per run.
- `--repo DIR`: repo root (default: nearest `pyproject.toml` above the script). Reads every
  shard in `tests/knob_gate/ledgers/` (merged, via the shared `_ledger.py` rule),
  `tests/knob_gate/branch_manifest.toml`,
  `tests/knob_gate/alias_map.toml`, and `scripts/parity/*.bth.toml`.
- `--stale-rows-json FILE`: optional JSON list of stale slugs. Without it, `stale` is null.
- `--controls-root DIR`: where each vehicle run wrote its per-control statuses,
  `<DIR>/<slug>/<git sha8>/branch_controls.json` (`~/.aminx/sidecars` on titanix, the
  path `scripts/redsox/launch_wave.sh` passes as `--output-paths`). A run's own metadata
  carries only control *counts*, so without this every control status is null. Each
  ledger entry records the file it read and its sha256 under `branch_controls`.
- `--out FILE`: output path.

## Running

Reads parquet with `pyarrow`, which is not in the aminx project environment. Use an
interpreter that has `pyarrow`, for example the bathos tool environment:

```bash
# titanix / Engaging, catalog under ~/.bth/catalog/runs/aminx
<python-with-pyarrow> scripts/dashboard/extract_parity.py \
  --catalog ~/.bth/catalog/runs/aminx \
  --controls-root ~/.aminx/sidecars \
  --out ~/parity_dashboard.json
```

Local test run against a snapshot:

```bash
<python-with-pyarrow> scripts/dashboard/extract_parity.py \
  --catalog /tmp/claude/parity_dash/catalog --controls-root /tmp/claude/parity_dash/sidecars \
  --out /tmp/claude/parity_dash/parity_dashboard.json
```

Warnings (missing runs, unparsable metadata, controls without a status) are listed
in the output's `warnings` array.

## Page

`index.html` is a static page with no build step and no script dependencies. It renders
the JSON: a summary strip, the ledger rows with their per-control status and pre-registered
pass condition, every recorded run per vehicle (the cited run outlined), the latest run of
each wave script, and knob coverage by family and relation. It fetches
`parity_dashboard.json` from its own folder; opened from disk, it offers a file picker
instead. Serve both files together, e.g. `python3 -m http.server` in a folder holding
`index.html` and the extractor's output renamed to `parity_dashboard.json`.
