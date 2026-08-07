"""No-demagnetization physics: mean I ≈ I0; field of uniform I0.

Uses the session ``demag`` flag (not hardcoded). With demag on these
assertions fail; with ``--no-demag`` they pass.
"""

from __future__ import annotations

import numpy as np

from grafen.analytic import cuboid_field_uniform, sphere_field_exterior
from grafen.demag import solve_magnetic
from grafen.field import field_at_points
from grafen.mesh import cube_mesh, ellipsoid_mesh, sphere_mesh

from .conftest import check_four_way, mean_magnetization, mesh_from_stl, roundtrip_vtu


H_PRIME = np.array([14.0, 14.0, 35.0])
TOL_STL_H = 0.20


def test_sphere_no_demag(demag, tmp_path):
    K, R = 2.0, 10.0
    i0 = H_PRIME * K
    corners, dens0 = sphere_mesh(R, nl=4, nb=4, nr=2, magnetization=i0)
    c_vtu, d_vtu, _ = roundtrip_vtu(corners, dens0, K, tmp_path / "sphere_nodemag.vtu")
    c_stl, d_stl, _ = mesh_from_stl(
        "sphere.stl",
        i0,
        K,
        tmp_path / "sphere_nodemag_from_stl.vtu",
        nx=12,
        ny=12,
        nz=12,
    )

    i_hard = solve_magnetic(corners, K, dens0, tol=1e-3, max_iter=15, demag=demag)
    i_vtu = solve_magnetic(c_vtu, K, d_vtu, tol=1e-3, max_iter=15, demag=demag)
    i_stl = solve_magnetic(c_stl, K, d_stl, tol=1e-3, max_iter=15, demag=demag)

    check_four_way(
        mean_magnetization(i_hard),
        mean_magnetization(i_vtu),
        mean_magnetization(i_stl),
        i0,
        1e-12,
        "sphere no-demag magnetization",
    )

    pts = np.array(
        [
            [0.0, 0.0, 25.0],
            [20.0, 0.0, 20.0],
            [0.0, 30.0, 0.0],
            [-15.0, 15.0, 15.0],
        ]
    )
    h_hard = field_at_points(pts, corners, i_hard)
    h_vtu = field_at_points(pts, c_vtu, i_vtu)
    h_stl = field_at_points(pts, c_stl, i_stl)
    h_ana = sphere_field_exterior(np.zeros(3), R, i0, pts)
    check_four_way(
        h_hard,
        h_vtu,
        h_stl,
        h_ana,
        0.08,
        "sphere no-demag field",
        tol_stl=TOL_STL_H,
    )


def test_ellipsoid_no_demag(demag, tmp_path):
    K, REQ, RPL = 2.0, 10.0, 20.0
    i0 = H_PRIME * K
    corners, dens0 = ellipsoid_mesh(REQ, RPL, nl=4, nb=4, nr=2, magnetization=i0)
    c_vtu, d_vtu, _ = roundtrip_vtu(corners, dens0, K, tmp_path / "ellipsoid_nodemag.vtu")
    c_stl, d_stl, _ = mesh_from_stl(
        "ellipsoid.stl",
        i0,
        K,
        tmp_path / "ellipsoid_nodemag_from_stl.vtu",
        nx=10,
        ny=10,
        nz=16,
    )

    i_hard = solve_magnetic(corners, K, dens0, tol=1e-3, max_iter=15, demag=demag)
    i_vtu = solve_magnetic(c_vtu, K, d_vtu, tol=1e-3, max_iter=15, demag=demag)
    i_stl = solve_magnetic(c_stl, K, d_stl, tol=1e-3, max_iter=15, demag=demag)

    check_four_way(
        mean_magnetization(i_hard),
        mean_magnetization(i_vtu),
        mean_magnetization(i_stl),
        i0,
        1e-12,
        "ellipsoid no-demag magnetization",
    )


def test_cube_no_demag(demag, tmp_path):
    K = 0.2
    bounds = ((-5.0, 5.0), (-2.0, 2.0), (-2.0, 2.0))
    i0 = H_PRIME * K
    (x0, x1), (y0, y1), (z0, z1) = bounds
    corners, dens0 = cube_mesh((x0, x1, 4), (y0, y1, 2), (z0, z1, 2), magnetization=i0)
    c_vtu, d_vtu, _ = roundtrip_vtu(corners, dens0, K, tmp_path / "cube_nodemag.vtu")
    c_stl, d_stl, _ = mesh_from_stl(
        "cube.stl",
        i0,
        K,
        tmp_path / "cube_nodemag_from_stl.vtu",
        nx=4,
        ny=2,
        nz=2,
        bounds=bounds,
    )

    i_hard = solve_magnetic(corners, K, dens0, tol=1e-3, max_iter=15, demag=demag)
    i_vtu = solve_magnetic(c_vtu, K, d_vtu, tol=1e-3, max_iter=15, demag=demag)
    i_stl = solve_magnetic(c_stl, K, d_stl, tol=1e-3, max_iter=15, demag=demag)

    check_four_way(
        mean_magnetization(i_hard),
        mean_magnetization(i_vtu),
        mean_magnetization(i_stl),
        i0,
        1e-12,
        "cube no-demag magnetization",
    )

    pts = np.array(
        [
            [0.0, 0.0, 6.0],
            [8.0, 0.0, 0.0],
            [0.0, 5.0, 5.0],
            [-6.0, 3.0, 4.0],
        ]
    )
    h_hard = field_at_points(pts, corners, i_hard)
    h_vtu = field_at_points(pts, c_vtu, i_vtu)
    h_stl = field_at_points(pts, c_stl, i_stl)
    h_ana = cuboid_field_uniform(bounds, i0, pts)
    check_four_way(h_hard, h_vtu, h_stl, h_ana, 1e-3, "cube no-demag field")
