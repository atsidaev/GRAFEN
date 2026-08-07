#!/usr/bin/env python3
"""CLI: closed STL surface → GRAFEN volumetric hex VTU.

Examples:
  python tools/stl_to_vtu.py tests/data/cube.stl /tmp/cube.vtu \\
      --nx 24 --ny 24 --nz 24 -k 2 -H 14 14 35

  python tools/stl_to_vtu.py tests/data/ellipsoid.stl /tmp/ell.vtu \\
      --polar --nl 24 --nb 12 --nr 6 -k 2 -H 14 14 35
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grafen.stl_convert import main

if __name__ == "__main__":
    raise SystemExit(main())
