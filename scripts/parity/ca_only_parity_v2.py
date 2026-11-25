"""CA-Only ProteinMPNN parity vehicle, v2: edge tensors compared after canonical neighbor ordering.

v1 (``ca_only_parity.py``) compared the edge tensor slot by slot and its registered run FAILED at ~39.7 on edges that
are equal up to a swap of the last two neighbor slots (the masked residue ties the farthest valid neighbor at D_max;
torch ``topk`` and ``jax.lax.top_k`` order the tie differently). Nodes and log-probs agreed to ~1e-15. This wrapper runs
the same suite with ``--canonical-edges``; the numeric bands are unchanged, and the sidecar adds the neighbor-selection
check (set equality and no untied slot differences). v1 stays as the closure of its recorded run.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
  sys.path.insert(0, str(_DIR))

from ca_only_parity import main  # noqa: E402

if __name__ == "__main__":
  if "--canonical-edges" not in sys.argv:
    sys.argv.append("--canonical-edges")
  main()
