"""Export a run from index.db to cool-tier parquet fragment.

Used by verify_wave when runs are recorded in log mode (~/.bth/catalog/index.db)
so that verify_wave / knob_gate can inspect them via pyarrow.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
  import duckdb
except ImportError:
  print("duckdb not available", file=sys.stderr)
  sys.exit(1)


def main() -> None:
  if len(sys.argv) < 2:
    print("usage: export_run_parquet.py <run_id> [<catalog_dir>]", file=sys.stderr)
    sys.exit(2)
  run_id = sys.argv[1]
  catalog_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / ".bth" / "catalog"
  index_db = catalog_dir / "index.db"
  if not index_db.is_file():
    print(f"no index.db at {index_db}", file=sys.stderr)
    sys.exit(1)
  out_dir = catalog_dir / "runs" / "aminx"
  out_dir.mkdir(parents=True, exist_ok=True)
  out_file = out_dir / f"run_{run_id}.parquet"
  conn = duckdb.connect(str(index_db), read_only=True)
  # Check if run exists
  res = conn.execute("SELECT id FROM runs WHERE id = ?", [run_id]).fetchall()
  if not res:
    print(f"run {run_id} not found in {index_db}", file=sys.stderr)
    sys.exit(1)
  conn.execute(
    f"COPY (SELECT * FROM runs WHERE id = ?) TO '{out_file}' (FORMAT PARQUET)",
    [run_id],
  )
  print(f"exported {run_id} -> {out_file}")


if __name__ == "__main__":
  main()
