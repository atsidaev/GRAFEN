"""Two spheres: far-field demagnetization ≈ independent spheres; field ≈ two dipoles."""

from __future__ import annotations

import numpy as np

from grafen.analytic import sphere_field_exterior, sphere_magnetization
from grafen.demag import solve_demagnetization
from grafen.field import field_at_points
from grafen.mesh import merge_meshes, sphere_mesh, translate_mesh

from .conftest import check_relative_rms, mean_magnetization


H_PRIME = np.array([14.0, 14.0, 35.0])
K = 2.0
R = 5.0
SEP = 40.0  # center-to-center; far enough that mutual demag is weak


def test_two_spheres_far_apart_magnetization():
    i0 = H_PRIME * K
    c1, d1 = sphere_mesh(R, nl=3, nb=3, nr=2, magnetization=i0)
    c2, d2 = sphere_mesh(R, nl=3, nb=3, nr=2, magnetization=i0)
    c2 = translate_mesh(c2, np.array([SEP, 0.0, 0.0]))
    corners, dens0 = merge_meshes((c1, d1), (c2, d2))

    i = solve_demagnetization(corners, K, dens0, tol=1e-3, max_iter=15)
    n1 = c1.shape[0]
    i1 = mean_magnetization(i[:n1])
    i2 = mean_magnetization(i[n1:])
    i_ref = sphere_magnetization(i0, K)

    check_relative_rms(i1, i_ref, 0.12, "two spheres I1 vs analytic")
    check_relative_rms(i2, i_ref, 0.12, "two spheres I2 vs analytic")
    # Mutual coupling should keep both spheres nearly equal
    check_relative_rms(i1, i2, 0.05, "two spheres I1 vs I2")


def test_two_spheres_exterior_field_superposition():
    i0 = H_PRIME * K
    c1, d1 = sphere_mesh(R, nl=3, nb=3, nr=2, magnetization=i0)
    c2, d2 = sphere_mesh(R, nl=3, nb=3, nr=2, magnetization=i0)
    offset = np.array([SEP, 0.0, 0.0])
    c2 = translate_mesh(c2, offset)
    corners, dens0 = merge_meshes((c1, d1), (c2, d2))

    i = solve_demagnetization(corners, K, dens0, tol=1e-3, max_iter=15)
    i_ref = sphere_magnetization(i0, K)

    pts = np.array(
        [
            [SEP / 2.0, 0.0, 30.0],
            [0.0, 25.0, 0.0],
            [SEP, 0.0, 25.0],
        ]
    )
    h_num = field_at_points(pts, corners, i)
    h_ana = sphere_field_exterior(np.zeros(3), R, i_ref, pts) + sphere_field_exterior(
        offset, R, i_ref, pts
    )
    check_relative_rms(h_num, h_ana, 0.15, "two spheres field vs analytic")
