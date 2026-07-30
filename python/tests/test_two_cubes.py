"""Two cubes: superposition without demag; far demag ≈ independent."""

from __future__ import annotations

import numpy as np

from grafen.analytic import cuboid_field_uniform
from grafen.demag import solve_demagnetization
from grafen.field import field_at_points
from grafen.mesh import cube_mesh, merge_meshes, translate_mesh

from .conftest import mean_magnetization, relative_rms


H_PRIME = np.array([14.0, 14.0, 35.0])
K = 0.2
# First cube centered near origin; second translated far along x
B1 = ((-2.0, 2.0), (-1.0, 1.0), (-1.0, 1.0))
OFFSET = np.array([20.0, 0.0, 0.0])


def test_two_cubes_uniform_field_is_superposition():
    i0 = H_PRIME * K
    (x0, x1), (y0, y1), (z0, z1) = B1
    c1, d1 = cube_mesh((x0, x1, 3), (y0, y1, 2), (z0, z1, 2), magnetization=i0)
    c2, d2 = cube_mesh((x0, x1, 3), (y0, y1, 2), (z0, z1, 2), magnetization=i0)
    c2 = translate_mesh(c2, OFFSET)
    corners, dens = merge_meshes((c1, d1), (c2, d2))

    pts = np.array(
        [
            [10.0, 0.0, 5.0],
            [0.0, 4.0, 4.0],
            [20.0, 0.0, 4.0],
        ]
    )
    h_num = field_at_points(pts, corners, dens)

    b2 = (
        (B1[0][0] + OFFSET[0], B1[0][1] + OFFSET[0]),
        (B1[1][0] + OFFSET[1], B1[1][1] + OFFSET[1]),
        (B1[2][0] + OFFSET[2], B1[2][1] + OFFSET[2]),
    )
    h_ana = cuboid_field_uniform(B1, i0, pts) + cuboid_field_uniform(b2, i0, pts)
    assert relative_rms(h_num, h_ana) < 1e-3


def test_two_cubes_far_demag_nearly_independent():
    i0 = H_PRIME * K
    (x0, x1), (y0, y1), (z0, z1) = B1
    c1, d1 = cube_mesh((x0, x1, 3), (y0, y1, 2), (z0, z1, 2), magnetization=i0)
    c2, d2 = cube_mesh((x0, x1, 3), (y0, y1, 2), (z0, z1, 2), magnetization=i0)
    c2 = translate_mesh(c2, OFFSET)

    i_single = solve_demagnetization(c1, K, d1, tol=1e-3, max_iter=15)
    corners, dens0 = merge_meshes((c1, d1), (c2, d2))
    i_pair = solve_demagnetization(corners, K, dens0, tol=1e-3, max_iter=15)

    n1 = c1.shape[0]
    assert relative_rms(mean_magnetization(i_pair[:n1]), mean_magnetization(i_single)) < 0.05
    assert relative_rms(mean_magnetization(i_pair[n1:]), mean_magnetization(i_single)) < 0.05
