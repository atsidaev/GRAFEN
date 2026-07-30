"""Prolate ellipsoid: magnetization vs closed-form demagnetizing factors."""

from __future__ import annotations

import numpy as np

from grafen.analytic import ellipsoid_magnetization
from grafen.demag import solve_demagnetization
from grafen.mesh import ellipsoid_mesh

from .conftest import check_relative_rms, mean_magnetization


H_PRIME = np.array([14.0, 14.0, 35.0])
K = 2.0
REQ = 10.0
RPL = 20.0


def test_ellipsoid_magnetization_matches_analytic():
    i0 = H_PRIME * K
    corners, dens0 = ellipsoid_mesh(REQ, RPL, nl=4, nb=4, nr=2, magnetization=i0)
    i = solve_demagnetization(corners, K, dens0, tol=1e-3, max_iter=15)

    i_ref = ellipsoid_magnetization(i0, K, REQ, RPL)
    i_mean = mean_magnetization(i)

    check_relative_rms(i_mean, i_ref, 0.10, "ellipsoid magnetization")
    # For a prolate body along z with H' having large z, I_z / I0_z > I_x / I0_x
    # (smaller demagnetizing factor along the long axis)
    assert (i_ref[2] / i0[2]) > (i_ref[0] / i0[0])
    assert (i_mean[2] / i0[2]) > (i_mean[0] / i0[0]) * 0.9
