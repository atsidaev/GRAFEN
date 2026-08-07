"""Two cubes: superposition without demag; far demag ≈ independent."""

from __future__ import annotations

import numpy as np
import pytest

from grafen.analytic import cuboid_field_uniform
from grafen.demag import solve_magnetic
from grafen.field import field_at_points
from grafen.mesh import cube_mesh, merge_meshes, translate_mesh

from .conftest import check_three_way, mean_magnetization, roundtrip_vtu


H_PRIME = np.array([14.0, 14.0, 35.0])
K = 0.2
# First cube centered near origin; second translated far along x
B1 = ((-2.0, 2.0), (-1.0, 1.0), (-1.0, 1.0))
OFFSET = np.array([20.0, 0.0, 0.0])


def test_two_cubes_uniform_field_is_superposition(tmp_path):
    i0 = H_PRIME * K
    (x0, x1), (y0, y1), (z0, z1) = B1
    c1, d1 = cube_mesh((x0, x1, 3), (y0, y1, 2), (z0, z1, 2), magnetization=i0)
    c2, d2 = cube_mesh((x0, x1, 3), (y0, y1, 2), (z0, z1, 2), magnetization=i0)
    c2 = translate_mesh(c2, OFFSET)
    corners, dens = merge_meshes((c1, d1), (c2, d2))
    c_vtu, d_vtu, _ = roundtrip_vtu(corners, dens, K, tmp_path / "two_cubes.vtu")

    pts = np.array(
        [
            [10.0, 0.0, 5.0],
            [0.0, 4.0, 4.0],
            [20.0, 0.0, 4.0],
        ]
    )
    h_hard = field_at_points(pts, corners, dens)
    h_vtu = field_at_points(pts, c_vtu, d_vtu)

    b2 = (
        (B1[0][0] + OFFSET[0], B1[0][1] + OFFSET[0]),
        (B1[1][0] + OFFSET[1], B1[1][1] + OFFSET[1]),
        (B1[2][0] + OFFSET[2], B1[2][1] + OFFSET[2]),
    )
    h_ana = cuboid_field_uniform(B1, i0, pts) + cuboid_field_uniform(b2, i0, pts)
    check_three_way(h_hard, h_vtu, h_ana, 1e-3, "two cubes field")


@pytest.mark.demag
def test_two_cubes_far_demag_nearly_independent(demag, tmp_path):
    i0 = H_PRIME * K
    (x0, x1), (y0, y1), (z0, z1) = B1
    c1, d1 = cube_mesh((x0, x1, 3), (y0, y1, 2), (z0, z1, 2), magnetization=i0)
    c2, d2 = cube_mesh((x0, x1, 3), (y0, y1, 2), (z0, z1, 2), magnetization=i0)
    c2 = translate_mesh(c2, OFFSET)

    i_single = solve_magnetic(c1, K, d1, tol=1e-3, max_iter=15, demag=demag)
    corners, dens0 = merge_meshes((c1, d1), (c2, d2))
    c_vtu, d_vtu, _ = roundtrip_vtu(corners, dens0, K, tmp_path / "two_cubes_demag.vtu")

    i_pair_h = solve_magnetic(corners, K, dens0, tol=1e-3, max_iter=15, demag=demag)
    i_pair_v = solve_magnetic(c_vtu, K, d_vtu, tol=1e-3, max_iter=15, demag=demag)

    n1 = c1.shape[0]
    ref = mean_magnetization(i_single)
    check_three_way(
        mean_magnetization(i_pair_h[:n1]),
        mean_magnetization(i_pair_v[:n1]),
        ref,
        0.05,
        "two cubes I1 vs single",
    )
    check_three_way(
        mean_magnetization(i_pair_h[n1:]),
        mean_magnetization(i_pair_v[n1:]),
        ref,
        0.05,
        "two cubes I2 vs single",
    )
