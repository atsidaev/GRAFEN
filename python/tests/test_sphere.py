"""One sphere: demagnetization and exterior field vs closed form."""

from __future__ import annotations

import numpy as np
import pytest

from grafen.analytic import sphere_field_exterior, sphere_magnetization
from grafen.demag import solve_magnetic
from grafen.field import field_at_points
from grafen.mesh import sphere_mesh

from .conftest import (
    check_four_way,
    check_relative_rms,
    mean_magnetization,
    mesh_from_stl,
    roundtrip_vtu,
)


H_PRIME = np.array([14.0, 14.0, 35.0])
K = 2.0
R = 10.0
# Stair-step voxel surface vs body-fitted / analytic
TOL_STL_I = 0.12
TOL_STL_H = 0.20


@pytest.fixture(scope="module")
def sphere_meshes(demag, tmp_path_factory):
    i0 = H_PRIME * K
    corners, dens0 = sphere_mesh(R, nl=4, nb=4, nr=2, magnetization=i0)
    base = tmp_path_factory.mktemp("sphere")
    c_vtu, d_vtu, _ = roundtrip_vtu(corners, dens0, K, base / "sphere.vtu")
    c_stl, d_stl, _ = mesh_from_stl(
        "sphere.stl",
        i0,
        K,
        base / "sphere_from_stl.vtu",
        nx=12,
        ny=12,
        nz=12,
    )

    i_hard = solve_magnetic(corners, K, dens0, tol=1e-3, max_iter=15, demag=demag)
    i_vtu = solve_magnetic(c_vtu, K, d_vtu, tol=1e-3, max_iter=15, demag=demag)
    i_stl = solve_magnetic(c_stl, K, d_stl, tol=1e-3, max_iter=15, demag=demag)
    return {
        "i0": i0,
        "corners": corners,
        "c_vtu": c_vtu,
        "c_stl": c_stl,
        "i_hard": i_hard,
        "i_vtu": i_vtu,
        "i_stl": i_stl,
    }


@pytest.mark.demag
def test_sphere_magnetization_matches_analytic(sphere_meshes):
    i0 = sphere_meshes["i0"]
    i_ref = sphere_magnetization(i0, K)
    i_hard = mean_magnetization(sphere_meshes["i_hard"])
    i_vtu = mean_magnetization(sphere_meshes["i_vtu"])
    i_stl = mean_magnetization(sphere_meshes["i_stl"])
    check_four_way(
        i_hard,
        i_vtu,
        i_stl,
        i_ref,
        0.05,
        "sphere magnetization",
        tol_stl=TOL_STL_I,
    )
    assert np.linalg.norm(i_hard) < np.linalg.norm(i0) * 0.95


@pytest.mark.demag
def test_sphere_exterior_field_matches_dipole(sphere_meshes):
    i_ref = sphere_magnetization(H_PRIME * K, K)
    i_hard = sphere_meshes["i_hard"]
    i_vtu = sphere_meshes["i_vtu"]
    i_stl = sphere_meshes["i_stl"]
    i_mean = mean_magnetization(i_hard)

    pts = np.array(
        [
            [0.0, 0.0, 25.0],
            [20.0, 0.0, 20.0],
            [0.0, 30.0, 0.0],
            [-15.0, 15.0, 15.0],
        ]
    )
    h_hard = field_at_points(pts, sphere_meshes["corners"], i_hard)
    h_vtu = field_at_points(pts, sphere_meshes["c_vtu"], i_vtu)
    h_stl = field_at_points(pts, sphere_meshes["c_stl"], i_stl)
    h_ana = sphere_field_exterior(np.zeros(3), R, i_ref, pts)
    check_four_way(
        h_hard,
        h_vtu,
        h_stl,
        h_ana,
        0.08,
        "sphere field vs analytic dipole",
        tol_stl=TOL_STL_H,
    )

    h_dip_mean = sphere_field_exterior(np.zeros(3), R, i_mean, pts)
    check_relative_rms(h_hard, h_dip_mean, 0.08, "sphere field vs mean-I dipole")
