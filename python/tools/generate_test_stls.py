#!/usr/bin/env python3
"""Regenerate tests/data/{cube,sphere,ellipsoid}.stl surfaces."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grafen.stl_io import (
    cube_surface_stl,
    ellipsoid_surface_stl,
    uv_sphere_surface_stl,
    write_stl_ascii,
)


def main() -> int:
    data = Path(__file__).resolve().parents[1] / "tests" / "data"
    data.mkdir(parents=True, exist_ok=True)
    write_stl_ascii(data / "cube.stl", cube_surface_stl(((-5, 5), (-2, 2), (-2, 2))), name="cube")
    write_stl_ascii(data / "sphere.stl", uv_sphere_surface_stl(10.0, n_lat=24, n_lon=48), name="sphere")
    write_stl_ascii(
        data / "ellipsoid.stl",
        ellipsoid_surface_stl(10.0, 20.0, n_lat=24, n_lon=48),
        name="ellipsoid",
    )
    for name in ("cube", "sphere", "ellipsoid"):
        print(data / f"{name}.stl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
