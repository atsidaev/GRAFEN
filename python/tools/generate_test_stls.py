#!/usr/bin/env python3
"""Regenerate tests/data/{cube,sphere,ellipsoid}.stl under the Earth surface (z<0).

Bodies sit fully below z=0 (survey plane), matching C++ TwoCuboids layout:
tops near z=-5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grafen.stl_io import (
    cube_surface_stl,
    ellipsoid_surface_stl,
    uv_sphere_surface_stl,
    write_stl_ascii,
)

# Shared with tests/conftest.py (keep in sync)
CUBE_BOUNDS = ((-10.0, 10.0), (-5.0, 5.0), (-15.0, -5.0))
SPHERE_R = 10.0
SPHERE_CENTER = np.array([0.0, 0.0, -15.0])
ELLIPSOID_REQ = 10.0
ELLIPSOID_RPL = 20.0
ELLIPSOID_CENTER = np.array([0.0, 0.0, -25.0])


def main() -> int:
    data = Path(__file__).resolve().parents[1] / "tests" / "data"
    data.mkdir(parents=True, exist_ok=True)
    write_stl_ascii(data / "cube.stl", cube_surface_stl(CUBE_BOUNDS), name="cube")
    write_stl_ascii(
        data / "sphere.stl",
        uv_sphere_surface_stl(SPHERE_R, n_lat=24, n_lon=48) + SPHERE_CENTER,
        name="sphere",
    )
    write_stl_ascii(
        data / "ellipsoid.stl",
        ellipsoid_surface_stl(ELLIPSOID_REQ, ELLIPSOID_RPL, n_lat=24, n_lon=48)
        + ELLIPSOID_CENTER,
        name="ellipsoid",
    )
    for name in ("cube", "sphere", "ellipsoid"):
        print(data / f"{name}.stl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
