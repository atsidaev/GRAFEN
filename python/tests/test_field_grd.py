"""Tests for VTU → Surfer GRD field tool (no demag)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grafen.analytic import cuboid_field_uniform
from grafen.field_grd import (
    component_values,
    field_grid_from_model,
    parse_axis_spec,
    vtu_field_to_grd,
)
from grafen.mesh import cube_mesh
from grafen.model_io import save_model
from grafen.pygrid import read_grd7


def test_parse_axis_step():
    lo, hi, n, step = parse_axis_spec("-15,15,1", name="x")
    assert lo == -15.0
    assert step == 1.0
    assert n == 31
    assert hi == pytest.approx(-15.0 + 1.0 * 30)


def test_parse_axis_ncol():
    lo, hi, n, step = parse_axis_spec("-15,15", n=256, name="x")
    assert n == 256
    assert step == pytest.approx(30.0 / 255.0)
    assert hi == 15.0


def test_parse_axis_rejects_both():
    with pytest.raises(ValueError, match="not both"):
        parse_axis_spec("-15,15,1", n=10, name="x")


def test_vtu_to_grd_cube_hz(tmp_path: Path):
    bounds = ((-5.0, 5.0), (-2.0, 2.0), (-2.0, 2.0))
    i0 = np.array([2.8, 2.8, 7.0])
    (x0, x1), (y0, y1), (z0, z1) = bounds
    corners, dens = cube_mesh((x0, x1, 4), (y0, y1, 2), (z0, z1, 2), magnetization=i0)
    vtu = tmp_path / "cube.vtu"
    grd = tmp_path / "hz.grd"
    save_model(vtu, corners, dens, kappa=0.2)

    grid = vtu_field_to_grd(
        vtu,
        grd,
        x_spec="-20,20,5",
        y_spec="-20,20,5",
        z=6.0,
        component="hz",
    )
    assert grd.is_file()
    assert grid.xnum == 9
    assert grid.ynum == 9

    g2 = read_grd7(str(grd))
    assert g2.xnum == grid.xnum
    assert np.allclose(g2.data, grid.data)

    # Spot-check vs analytic cuboid at a few nodes
    pts = np.array(
        [
            [0.0, 0.0, 6.0],
            [10.0, 0.0, 6.0],
            [0.0, 10.0, 6.0],
        ]
    )
    h_ana = cuboid_field_uniform(bounds, i0, pts)
    g_check = field_grid_from_model(
        corners,
        dens,
        xmin=-20,
        xmax=20,
        xnum=9,
        ymin=-20,
        ymax=20,
        ynum=9,
        z=6.0,
        component="hz",
    )
    for p, href in zip(pts, h_ana):
        ix = int(round((p[0] - g_check.xmin) / g_check.dx))
        iy = int(round((p[1] - g_check.ymin) / g_check.dy))
        assert g_check.data[iy * g_check.xnum + ix] == pytest.approx(href[2], rel=1e-3)


def test_component_bt():
    h = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
    assert np.allclose(component_values(h, "bt"), [5.0, 2.0])


def test_vtu_to_grd_from_H_and_kappa(tmp_path: Path):
    """I0 = κ H' at field time; VTU may store I=0."""
    from grafen.field_grd import magnetization_for_field

    bounds = ((-5.0, 5.0), (-2.0, 2.0), (-2.0, 2.0))
    h_prime = np.array([14.0, 14.0, 35.0])
    kappa = 0.2
    i0 = h_prime * kappa
    (x0, x1), (y0, y1), (z0, z1) = bounds
    corners, dens0 = cube_mesh((x0, x1, 4), (y0, y1, 2), (z0, z1, 2), magnetization=i0)
    # Store zero I but keep kappa — field tool rebuilds I0 from --H
    dens_zero = np.zeros_like(dens0)
    vtu = tmp_path / "cube.vtu"
    save_model(vtu, corners, dens_zero, kappa=kappa)

    dens_use = magnetization_for_field(dens_zero, np.full(len(dens0), kappa), h_prime=h_prime)
    assert np.allclose(dens_use, dens0)

    g_h = vtu_field_to_grd(
        vtu,
        tmp_path / "from_H.grd",
        x_spec="-20,20,5",
        y_spec="-20,20,5",
        z=6.0,
        h_prime=h_prime,
        kappa=kappa,
    )
    g_i = vtu_field_to_grd(
        vtu,
        tmp_path / "from_I.grd",
        x_spec="-20,20,5",
        y_spec="-20,20,5",
        z=6.0,
    )
    # stored I=0 → zero field; --H path nonzero and matches explicit-I model
    assert np.allclose(g_i.data, 0.0)
    assert np.sqrt(np.mean(g_h.data**2)) > 1e-3

    vtu_i = tmp_path / "cube_I.vtu"
    save_model(vtu_i, corners, dens0, kappa=kappa)
    g_ref = vtu_field_to_grd(
        vtu_i,
        tmp_path / "ref.grd",
        x_spec="-20,20,5",
        y_spec="-20,20,5",
        z=6.0,
    )
    assert np.allclose(g_h.data, g_ref.data, rtol=1e-12, atol=1e-12)
