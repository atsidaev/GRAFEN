#!/usr/bin/env python3
"""CLI: closed STL surface → GRAFEN volumetric hex VTU (AABB voxel fill).

Example:
  python tools/stl_to_vtu.py tests/data/sphere.stl /tmp/sphere.vtu \\
      --nx 24 --ny 24 --nz 24 -k 2 -H 14 14 35
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without install: add package root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grafen.stl_convert import main

if __name__ == "__main__":
    raise SystemExit(main())
