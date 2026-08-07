"""Two spheres: far-field demagnetization ≈ independent spheres; field ≈ two dipoles."""

from __future__ import annotations

import numpy as np
import pytest

from grafen.analytic import sphere_field_exterior, sphere_magnetization
from grafen.demag import solve_magnetic
from grafen.field import field_at_points
from grafen.mesh import merge_meshes, sphere_mesh, translate_mesh

from .conftest import check_three_way, mean_magnetization, roundtrip_vtu


H_PRIME = np.array([14.0, 14.0, 35.0])
K = 2.0
R = 5.0
SEP = 40.0  # center-to-center; far enough that mutual demag is weak


@pytest.mark.demag
def test_two_spheres_far_apart_magnetization(demag, tmp_path):
    i0 = H_PRIME * K
    c1, d1 = sphere_mesh(R, nl=3, nb=3, nr=2, magnetization=i0)
    c2, d2 = sphere_mesh(R, nl=3, nb=3, nr=2, magnetization=i0)
    c2 = translate_mesh(c2, np.array([SEP, 0.0, 0.0]))
    corners, dens0 = merge_meshes((c1, d1), (c2, d2))
    c_vtu, d_vtu, _ = roundtrip_vtu(corners, dens0, K, tmp_path / "two_spheres.vtu")

    i_hard = solve_magnetic(corners, K, dens0, tol=1e-3, max_iter=15, demag=demag)
    i_vtu = solve_magnetic(c_vtu, K, d_vtu, tol=1e-3, max_iter=15, demag=demag)
    n1 = c1.shape[0]
    i_ref = sphere_magnetization(i0, K)

    check_three_way(
        mean_magnetization(i_hard[:n1]),
        mean_magnetization(i_vtu[:n1]),
        i_ref,
        0.12,
        "two spheres I1",
    )
    check_three_way(
        mean_magnetization(i_hard[n1:]),
        mean_magnetization(i_vtu[n1:]),
        i_ref,
        0.12,
        "two spheres I2",
    )
    check_three_way(
        mean_magnetization(i_hard[:n1]),
        mean_magnetization(i_vtu[:n1]),
        mean_magnetization(i_hard[n1:]),
        0.05,
        "two spheres I1 vs I2",
    )


@pytest.mark.demag
def test_two_spheres_exterior_field_superposition(demag, tmp_path):
    i0 = H_PRIME * K
    c1, d1 = sphere_mesh(R, nl=3, nb=3, nr=2, magnetization=i0)
    c2, d2 = sphere_mesh(R, nl=3, nb=3, nr=2, magnetization=i0)
    offset = np.array([SEP, 0.0, 0.0])
    c2 = translate_mesh(c2, offset)
    corners, dens0 = merge_meshes((c1, d1), (c2, d2))
    c_vtu, d_vtu, _ = roundtrip_vtu(corners, dens0, K, tmp_path / "two_spheres_field.vtu")

    i_hard = solve_magnetic(corners, K, dens0, tol=1e-3, max_iter=15, demag=demag)
    i_vtu = solve_magnetic(c_vtu, K, d_vtu, tol=1e-3, max_iter=15, demag=demag)
    i_ref = sphere_magnetization(i0, K)

    pts = np.array(
        [
            [SEP / 2.0, 0.0, 30.0],
            [0.0, 25.0, 0.0],
            [SEP, 0.0, 25.0],
        ]
    )
    h_hard = field_at_points(pts, corners, i_hard)
    h_vtu = field_at_points(pts, c_vtu, i_vtu)
    h_ana = sphere_field_exterior(np.zeros(3), R, i_ref, pts) + sphere_field_exterior(
        offset, R, i_ref, pts
    )
    check_three_way(h_hard, h_vtu, h_ana, 0.15, "two spheres field")
