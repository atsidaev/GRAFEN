"""STL → voxel hex VTU vs known meshes / analytics (bodies below z=0)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grafen.analytic import cuboid_field_uniform, ellipsoid_magnetization, sphere_field_exterior, sphere_magnetization
from grafen.demag import solve_magnetic
from grafen.field import field_at_points
from grafen.mesh import cube_mesh
from grafen.stl_convert import convert_stl_to_vtu, points_inside_stl, voxelize_stl
from grafen.stl_io import read_stl, stl_bbox

from .conftest import (
    CUBE_BOUNDS,
    ELLIPSOID_REQ,
    ELLIPSOID_RPL,
    SPHERE_CENTER,
    SPHERE_R,
    check_relative_rms,
    check_three_way,
    mean_magnetization,
)


DATA = Path(__file__).resolve().parent / "data"
H_PRIME = np.array([14.0, 14.0, 35.0])


@pytest.fixture(scope="module")
def stl_files():
    for name in ("cube.stl", "sphere.stl", "ellipsoid.stl"):
        path = DATA / name
        assert path.is_file(), f"missing {path}; run tools/generate_test_stls.py"
        tris = read_stl(path)
        assert tris.shape[0] > 0
        _, hi = stl_bbox(tris)
        assert hi[2] <= 0.0 + 1e-9, f"{name} must lie under z=0 (got zmax={hi[2]})"
    return DATA


def test_points_inside_cube_stl(stl_files):
    tris = read_stl(stl_files / "cube.stl")
    pts = np.array(
        [
            [0.0, 0.0, -10.0],  # cube center
            [4.0, 1.0, -10.0],
            [0.0, 0.0, 0.0],  # surface — outside
            [12.0, 0.0, -10.0],  # outside laterally
            [-9.0, -4.0, -14.0],
        ]
    )
    inside = points_inside_stl(pts, tris)
    assert inside.tolist() == [True, True, False, False, True]


def test_stl_cube_voxel_matches_cube_mesh(stl_files, demag, tmp_path):
    """Axis-aligned cube STL + voxel fill of the same AABB ≡ cube_mesh."""
    K = 0.2
    bounds = CUBE_BOUNDS
    i0 = H_PRIME * K
    (x0, x1), (y0, y1), (z0, z1) = bounds
    corners_h, dens0 = cube_mesh((x0, x1, 4), (y0, y1, 2), (z0, z1, 2), magnetization=i0)

    vtu = tmp_path / "from_stl_cube.vtu"
    corners_s, dens_s, _ = convert_stl_to_vtu(
        stl_files / "cube.stl",
        vtu,
        magnetization=i0,
        kappa=K,
        nx=4,
        ny=2,
        nz=2,
        bounds=bounds,
    )
    assert corners_s.shape == corners_h.shape
    assert np.allclose(corners_h, corners_s, atol=1e-12)
    assert np.allclose(dens0, dens_s, atol=1e-12)

    i_hard = solve_magnetic(corners_h, K, dens0, tol=1e-3, max_iter=15, demag=demag)
    i_stl = solve_magnetic(corners_s, K, dens_s, tol=1e-3, max_iter=15, demag=demag)

    pts = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 5.0, 0.0], [-6.0, 3.0, 0.0]])
    h_hard = field_at_points(pts, corners_h, i_hard)
    h_stl = field_at_points(pts, corners_s, i_stl)
    if demag:
        check_three_way(h_hard, h_stl, h_hard, 1e-10, "STL cube field")
        check_three_way(
            mean_magnetization(i_hard),
            mean_magnetization(i_stl),
            mean_magnetization(i_hard),
            1e-10,
            "STL cube magnetization",
        )
    else:
        check_three_way(
            mean_magnetization(i_hard),
            mean_magnetization(i_stl),
            i0,
            1e-12,
            "STL cube magnetization",
        )
        check_three_way(h_hard, h_stl, cuboid_field_uniform(bounds, i0, pts), 1e-3, "STL cube field")


def test_stl_sphere_voxel_vs_analytic(stl_files, demag, tmp_path):
    K, R = 2.0, SPHERE_R
    i0 = H_PRIME * K
    vtu = tmp_path / "from_stl_sphere.vtu"
    corners, dens, _ = convert_stl_to_vtu(
        stl_files / "sphere.stl",
        vtu,
        magnetization=i0,
        kappa=K,
        nx=12,
        ny=12,
        nz=12,
    )
    assert corners.shape[0] > 100
    assert corners.shape[0] < 12 * 12 * 12
    assert float(corners[:, :, 2].max()) <= 0.0 + 1e-9

    i_stl = solve_magnetic(corners, K, dens, tol=1e-3, max_iter=15, demag=demag)
    i_ref = sphere_magnetization(i0, K) if demag else i0
    check_relative_rms(
        mean_magnetization(i_stl),
        i_ref,
        0.12 if demag else 1e-12,
        "STL sphere voxel magnetization",
    )

    pts = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 30.0, 0.0], [-15.0, 15.0, 0.0]])
    h_stl = field_at_points(pts, corners, i_stl)
    h_ana = sphere_field_exterior(SPHERE_CENTER, R, i_ref, pts)
    check_relative_rms(h_stl, h_ana, 0.20, "STL sphere voxel field")


def test_stl_ellipsoid_voxel_vs_analytic(stl_files, demag, tmp_path):
    K, REQ, RPL = 2.0, ELLIPSOID_REQ, ELLIPSOID_RPL
    i0 = H_PRIME * K
    vtu = tmp_path / "from_stl_ellipsoid.vtu"
    corners, dens, _ = convert_stl_to_vtu(
        stl_files / "ellipsoid.stl",
        vtu,
        magnetization=i0,
        kappa=K,
        nx=10,
        ny=10,
        nz=16,
    )
    assert corners.shape[0] > 50
    assert float(corners[:, :, 2].max()) <= 0.0 + 1e-9

    i_stl = solve_magnetic(corners, K, dens, tol=1e-3, max_iter=15, demag=demag)
    i_ref = ellipsoid_magnetization(i0, K, REQ, RPL) if demag else i0
    check_relative_rms(
        mean_magnetization(i_stl),
        i_ref,
        0.15 if demag else 1e-12,
        "STL ellipsoid voxel magnetization",
    )


def test_voxelize_rejects_empty(stl_files):
    tris = read_stl(stl_files / "sphere.stl")
    with pytest.raises(ValueError, match="no cell centers inside"):
        voxelize_stl(
            tris,
            2,
            2,
            2,
            np.zeros(3),
            0.0,
            bounds=((100.0, 110.0), (100.0, 110.0), (100.0, 110.0)),
        )
