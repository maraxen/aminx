"""Bathos parity and negative-control driver for FrustraMPNN.

Forwarding wrapper delegating to v2 implementation (frustrampnn_parity_v2.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure directory is on path for relative import
_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
  sys.path.insert(0, str(_DIR))

from frustrampnn_parity_v2 import main

if __name__ == "__main__":
  sys.exit(main())
