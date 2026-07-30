"""Cube: uniform-I field vs single-cuboid analytic; demag reduces |I|."""

from __future__ import annotations

import numpy as np

from grafen.analytic import cuboid_field_uniform
from grafen.demag import solve_demagnetization
from grafen.field import field_at_points
from grafen.mesh import cube_mesh

from .conftest import check_relative_rms, mean_magnetization, relative_rms


H_PRIME = np.array([14.0, 14.0, 35.0])
K = 0.2
BOUNDS = ((-5.0, 5.0), (-2.0, 2.0), (-2.0, 2.0))


def test_cube_uniform_field_matches_analytic_cuboid():
    """No demag: multi-cell mesh with I=I0 must match one-cell cuboid field."""
    i0 = H_PRIME * K
    (x0, x1), (y0, y1), (z0, z1) = BOUNDS
    corners, dens = cube_mesh((x0, x1, 4), (y0, y1, 2), (z0, z1, 2), magnetization=i0)

    pts = np.array(
        [
            [0.0, 0.0, 6.0],
            [8.0, 0.0, 0.0],
            [0.0, 5.0, 5.0],
            [-6.0, 3.0, 4.0],
        ]
    )
    h_num = field_at_points(pts, corners, dens)
    h_ana = cuboid_field_uniform(BOUNDS, i0, pts)
    check_relative_rms(h_num, h_ana, 1e-3, "cube field vs analytic cuboid")


def test_cube_demagnetization_reduces_magnetization():
    i0 = H_PRIME * K
    (x0, x1), (y0, y1), (z0, z1) = BOUNDS
    corners, dens0 = cube_mesh((x0, x1, 4), (y0, y1, 3), (z0, z1, 3), magnetization=i0)
    i = solve_demagnetization(corners, K, dens0, tol=1e-3, max_iter=15)

    i_mean = mean_magnetization(i)
    # Weak but non-zero demag for K=0.2 (paper ~0.36% for similar aspect)
    assert np.linalg.norm(i_mean - i0) / np.linalg.norm(i0) > 1e-4
    assert np.linalg.norm(i_mean) < np.linalg.norm(i0)

    # Field with demag differs from uniform-I0 field
    pts = np.array([[0.0, 0.0, 6.0], [8.0, 0.0, 3.0]])
    h0 = field_at_points(pts, corners, dens0)
    h = field_at_points(pts, corners, i)
    assert relative_rms(h, h0) > 1e-4
