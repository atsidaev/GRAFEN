"""One sphere: demagnetization and exterior field vs closed form."""

from __future__ import annotations

import numpy as np
import pytest

from grafen.analytic import sphere_field_exterior, sphere_magnetization
from grafen.demag import solve_demagnetization
from grafen.field import field_at_points
from grafen.mesh import sphere_mesh

from .conftest import mean_magnetization, relative_rms


H_PRIME = np.array([14.0, 14.0, 35.0])
K = 2.0
R = 10.0


@pytest.fixture(scope="module")
def sphere_solution():
    i0 = H_PRIME * K
    corners, dens0 = sphere_mesh(R, nl=4, nb=4, nr=2, magnetization=i0)
    i = solve_demagnetization(corners, K, dens0, tol=1e-3, max_iter=15)
    return corners, dens0, i, i0


def test_sphere_magnetization_matches_analytic(sphere_solution):
    _, _, i, i0 = sphere_solution
    i_ref = sphere_magnetization(i0, K)
    i_mean = mean_magnetization(i)
    # Coarse mesh → mean magnetization within a few percent of analytic
    assert relative_rms(i_mean, i_ref) < 0.05
    # Demag must reduce |I| relative to I0
    assert np.linalg.norm(i_mean) < np.linalg.norm(i0) * 0.95


def test_sphere_exterior_field_matches_dipole(sphere_solution):
    corners, _, i, _ = sphere_solution
    i_mean = mean_magnetization(i)
    i_ref = sphere_magnetization(H_PRIME * K, K)

    # Observation points well outside the sphere
    pts = np.array(
        [
            [0.0, 0.0, 25.0],
            [20.0, 0.0, 20.0],
            [0.0, 30.0, 0.0],
            [-15.0, 15.0, 15.0],
        ]
    )
    h_num = field_at_points(pts, corners, i)
    h_ana = sphere_field_exterior(np.zeros(3), R, i_ref, pts)
    # Use analytic I for reference dipole; numerical uses piecewise I
    assert relative_rms(h_num, h_ana) < 0.08

    # Consistency: dipole with numerical mean I is close to numerical field
    h_dip_mean = sphere_field_exterior(np.zeros(3), R, i_mean, pts)
    assert relative_rms(h_num, h_dip_mean) < 0.08
