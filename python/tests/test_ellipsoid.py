"""Prolate ellipsoid: magnetization vs closed-form demagnetizing factors."""

from __future__ import annotations

import numpy as np
import pytest

from grafen.analytic import ellipsoid_magnetization
from grafen.demag import solve_magnetic
from grafen.mesh import ellipsoid_mesh, translate_mesh

from .conftest import (
    ELLIPSOID_CENTER,
    ELLIPSOID_REQ,
    ELLIPSOID_RPL,
    check_four_way,
    mean_magnetization,
    mesh_from_stl,
    roundtrip_vtu,
)


H_PRIME = np.array([14.0, 14.0, 35.0])
K = 2.0
REQ = ELLIPSOID_REQ
RPL = ELLIPSOID_RPL
CENTER = ELLIPSOID_CENTER
TOL_STL_I = 0.15


@pytest.mark.demag
def test_ellipsoid_magnetization_matches_analytic(demag, tmp_path):
    i0 = H_PRIME * K
    corners, dens0 = ellipsoid_mesh(REQ, RPL, nl=4, nb=4, nr=2, magnetization=i0)
    corners = translate_mesh(corners, CENTER)
    c_vtu, d_vtu, _ = roundtrip_vtu(corners, dens0, K, tmp_path / "ellipsoid.vtu")
    c_stl, d_stl, _ = mesh_from_stl(
        "ellipsoid.stl",
        i0,
        K,
        tmp_path / "ellipsoid_from_stl.vtu",
        nx=10,
        ny=10,
        nz=16,
    )

    i_hard = solve_magnetic(corners, K, dens0, tol=1e-3, max_iter=15, demag=demag)
    i_vtu = solve_magnetic(c_vtu, K, d_vtu, tol=1e-3, max_iter=15, demag=demag)
    i_stl = solve_magnetic(c_stl, K, d_stl, tol=1e-3, max_iter=15, demag=demag)

    i_ref = ellipsoid_magnetization(i0, K, REQ, RPL)
    check_four_way(
        mean_magnetization(i_hard),
        mean_magnetization(i_vtu),
        mean_magnetization(i_stl),
        i_ref,
        0.10,
        "ellipsoid magnetization",
        tol_stl=TOL_STL_I,
    )
    assert (i_ref[2] / i0[2]) > (i_ref[0] / i0[0])
    assert (mean_magnetization(i_hard)[2] / i0[2]) > (mean_magnetization(i_hard)[0] / i0[0]) * 0.9
