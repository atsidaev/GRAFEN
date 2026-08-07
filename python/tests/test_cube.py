"""Cube: uniform-I field vs single-cuboid analytic; demag reduces |I|."""

from __future__ import annotations

import numpy as np
import pytest

from grafen.analytic import cuboid_field_uniform
from grafen.demag import solve_magnetic
from grafen.field import field_at_points
from grafen.mesh import cube_mesh

from .conftest import (
    CUBE_BOUNDS,
    check_four_way,
    mean_magnetization,
    mesh_from_stl,
    relative_rms,
    roundtrip_vtu,
)


H_PRIME = np.array([14.0, 14.0, 35.0])
K = 0.2
BOUNDS = CUBE_BOUNDS


def test_cube_uniform_field_matches_analytic_cuboid(tmp_path):
    """No demag: multi-cell mesh with I=I0 must match one-cell cuboid field."""
    i0 = H_PRIME * K
    (x0, x1), (y0, y1), (z0, z1) = BOUNDS
    corners, dens = cube_mesh((x0, x1, 4), (y0, y1, 2), (z0, z1, 2), magnetization=i0)
    c_vtu, d_vtu, _ = roundtrip_vtu(corners, dens, K, tmp_path / "cube.vtu")
    c_stl, d_stl, _ = mesh_from_stl(
        "cube.stl",
        i0,
        K,
        tmp_path / "cube_from_stl.vtu",
        nx=4,
        ny=2,
        nz=2,
        bounds=BOUNDS,
    )

    # Survey plane z=0 above buried cube (top at z=-5)
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [-6.0, 3.0, 0.0],
        ]
    )
    h_hard = field_at_points(pts, corners, dens)
    h_vtu = field_at_points(pts, c_vtu, d_vtu)
    h_stl = field_at_points(pts, c_stl, d_stl)
    h_ana = cuboid_field_uniform(BOUNDS, i0, pts)
    check_four_way(h_hard, h_vtu, h_stl, h_ana, 1e-3, "cube field vs analytic cuboid")


@pytest.mark.demag
def test_cube_demagnetization_reduces_magnetization(demag, tmp_path):
    i0 = H_PRIME * K
    (x0, x1), (y0, y1), (z0, z1) = BOUNDS
    corners, dens0 = cube_mesh((x0, x1, 4), (y0, y1, 3), (z0, z1, 3), magnetization=i0)
    c_vtu, d_vtu, _ = roundtrip_vtu(corners, dens0, K, tmp_path / "cube_demag.vtu")
    c_stl, d_stl, _ = mesh_from_stl(
        "cube.stl",
        i0,
        K,
        tmp_path / "cube_demag_from_stl.vtu",
        nx=4,
        ny=3,
        nz=3,
        bounds=BOUNDS,
    )

    i_hard = solve_magnetic(corners, K, dens0, tol=1e-3, max_iter=15, demag=demag)
    i_vtu = solve_magnetic(c_vtu, K, d_vtu, tol=1e-3, max_iter=15, demag=demag)
    i_stl = solve_magnetic(c_stl, K, d_stl, tol=1e-3, max_iter=15, demag=demag)

    i_mean_h = mean_magnetization(i_hard)
    i_mean_v = mean_magnetization(i_vtu)
    i_mean_s = mean_magnetization(i_stl)
    assert np.linalg.norm(i_mean_h - i0) / np.linalg.norm(i0) > 1e-4
    assert np.linalg.norm(i_mean_h) < np.linalg.norm(i0)
    check_four_way(i_mean_h, i_mean_v, i_mean_s, i_mean_h, 1e-10, "cube demag I mean")

    pts = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    h0 = field_at_points(pts, corners, dens0)
    h_hard = field_at_points(pts, corners, i_hard)
    h_vtu = field_at_points(pts, c_vtu, i_vtu)
    h_stl = field_at_points(pts, c_stl, i_stl)
    assert relative_rms(h_hard, h0) > 1e-4
    check_four_way(h_hard, h_vtu, h_stl, h_hard, 1e-10, "cube demag field")
