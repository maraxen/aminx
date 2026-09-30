# Port waves

Each wave is one pytest session. `AMINX_PORT_WAVE` selects `targets/<wave>.toml`. `tests/conftest.py` deselects every item whose `port_wave` marker is a different wave.

A1 dumps are not in the repo. Set `AMINX_A1_ORACLE_DIR` (default `~/projects/aminx-oracles/dumps/a1_potts`). A missing directory skips the wave. A sha256 mismatch against `reference/a1_potts/oracles.sha256` fails.

```bash
bash tests/port/run_waves.sh
```

The same loop, one wave at a time:

```bash
for wave in potts_head potts_merge_pair_d2 potts_merge_pair_d4 potts_energy pottsmpnn_full declayer_f64; do
  AMINX_PORT_WAVE="$wave" uv run --frozen --extra=dev pytest -o addopts="" "tests/port/test_${wave}.py"
done
```

`pottsmpnn_full` and `declayer_f64` skip until `aminx.families.potts_mpnn.model.PottsMPNN` is importable. `potts_head` also skips when `AMINX_POTTS_ROOT` has no torch checkpoints.
