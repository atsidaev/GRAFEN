"""Round-trip tests for model .npz / .vtu I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from grafen.mesh import cube_mesh, sphere_mesh
from grafen.model_io import load_model, save_model


def _assert_model_equal(a, b, tol=1e-12):
    c0, d0, k0 = a
    c1, d1, k1 = b
    assert c0.shape == c1.shape
    assert np.allclose(c0, c1, atol=tol, rtol=0)
    assert np.allclose(d0, d1, atol=tol, rtol=0)
    assert np.allclose(k0, k1, atol=tol, rtol=0)


def test_npz_roundtrip_cube(tmp_path: Path):
    i0 = np.array([2.8, 2.8, 7.0])
    corners, dens = cube_mesh((-5, 5, 3), (-2, 2, 2), (-2, 2, 2), magnetization=i0)
    kappa = np.full(corners.shape[0], 0.2)
    path = tmp_path / "cube.npz"
    save_model(path, corners, dens, kappa)
    loaded = load_model(path)
    _assert_model_equal((corners, dens, kappa), loaded)


def test_vtu_roundtrip_sphere(tmp_path: Path):
    i0 = np.array([28.0, 28.0, 70.0])
    corners, dens = sphere_mesh(10.0, nl=2, nb=2, nr=1, magnetization=i0)
    kappa = 2.0
    path = tmp_path / "sphere.vtu"
    save_model(path, corners, dens, kappa)
    c1, d1, k1 = load_model(path)
    assert c1.shape == corners.shape
    assert np.allclose(c1, corners, atol=1e-12)
    assert np.allclose(d1, dens, atol=1e-12)
    assert np.allclose(k1, np.full(corners.shape[0], 2.0), atol=1e-12)


def test_vtu_python_cpp_mapping_consistent():
    """GRAFEN<->VTK index maps used by C++ model_vtu.h must match."""
    from grafen.model_io import _GRAFEN_TO_VTK, _VTK_TO_GRAFEN

    assert list(_GRAFEN_TO_VTK) == [7, 5, 4, 6, 3, 1, 0, 2]
    assert list(_VTK_TO_GRAFEN) == [6, 5, 7, 4, 2, 1, 3, 0]
    for v in range(8):
        assert int(_VTK_TO_GRAFEN[_GRAFEN_TO_VTK[v]]) == v


def test_npz_default_kappa(tmp_path: Path):
    corners, dens = cube_mesh((-1, 1, 1), (-1, 1, 1), (-1, 1, 1), magnetization=[1, 0, 0])
    path = tmp_path / "one.npz"
    save_model(path, corners, dens)
    _, _, kappa = load_model(path)
    assert kappa.shape == (1,)
    assert kappa[0] == 0.0
