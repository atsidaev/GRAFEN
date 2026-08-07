"""Shared helpers for numerical vs analytic comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grafen.model_io import load_model, save_model


def pytest_addoption(parser):
    parser.addoption(
        "--no-demag",
        action="store_true",
        default=False,
        help="Disable demagnetization CG (I stays I0); demag-marked tests still run",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "demag: test exercises demagnetization solve")


@pytest.fixture(scope="session")
def demag(pytestconfig) -> bool:
    return not pytestconfig.getoption("--no-demag")


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.asarray(a) ** 2, axis=-1))))


def relative_rms(num: np.ndarray, ref: np.ndarray) -> float:
    """RMS(|num-ref|) / RMS(|ref|)."""
    ref = np.asarray(ref, dtype=float)
    num = np.asarray(num, dtype=float)
    err = rms(num - ref)
    denom = rms(ref)
    if denom == 0.0:
        return err
    return err / denom


def check_relative_rms(
    num: np.ndarray,
    ref: np.ndarray,
    tol: float,
    label: str = "",
) -> float:
    """Print absolute and relative RMS, then assert relative RMS < tol."""
    ref = np.asarray(ref, dtype=float)
    num = np.asarray(num, dtype=float)
    abs_err = rms(num - ref)
    denom = rms(ref)
    rel = abs_err if denom == 0.0 else abs_err / denom
    prefix = f"{label}: " if label else ""
    print(f"{prefix}RMS={abs_err:.6e}  relative_RMS={rel:.6e}  (tol={tol:g})")
    assert rel < tol, f"{prefix}relative RMS {rel:.6e} >= {tol:g}"
    return rel


def check_three_way(
    hardcoded: np.ndarray,
    vtu: np.ndarray,
    analytic: np.ndarray,
    tol_ana: float,
    label: str,
    tol_io: float = 1e-10,
) -> None:
    """Compare hardcoded calc, VTU-loaded calc, and analytic reference."""
    check_relative_rms(hardcoded, analytic, tol_ana, f"{label} hardcoded vs analytic")
    check_relative_rms(vtu, analytic, tol_ana, f"{label} vtu vs analytic")
    check_relative_rms(vtu, hardcoded, tol_io, f"{label} vtu vs hardcoded")


def check_four_way(
    hardcoded: np.ndarray,
    vtu: np.ndarray,
    stl: np.ndarray,
    analytic: np.ndarray,
    tol_ana: float,
    label: str,
    *,
    tol_io: float = 1e-10,
    tol_stl: float | None = None,
) -> None:
    """Compare hardcoded, VTU round-trip, STL-voxel mesh, and analytic."""
    if tol_stl is None:
        tol_stl = tol_ana
    check_three_way(hardcoded, vtu, analytic, tol_ana, label, tol_io=tol_io)
    check_relative_rms(stl, analytic, tol_stl, f"{label} stl vs analytic")


STL_DATA = Path(__file__).resolve().parent / "data"

# Test STL bodies fully below the survey plane z=0 (tops at z=-5).
# Keep in sync with tools/generate_test_stls.py.
CUBE_BOUNDS = ((-10.0, 10.0), (-5.0, 5.0), (-15.0, -5.0))
SPHERE_R = 10.0
SPHERE_CENTER = np.array([0.0, 0.0, -15.0])
ELLIPSOID_REQ = 10.0
ELLIPSOID_RPL = 20.0
ELLIPSOID_CENTER = np.array([0.0, 0.0, -25.0])


def mesh_from_stl(
    stl_name: str,
    magnetization: np.ndarray,
    kappa: float,
    out_vtu: Path,
    **voxel_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build hex model from ``tests/data/<stl_name>`` via AABB voxelization."""
    from grafen.stl_convert import convert_stl_to_vtu

    path = STL_DATA / stl_name
    assert path.is_file(), f"missing {path}; run tools/generate_test_stls.py"
    return convert_stl_to_vtu(
        path,
        out_vtu,
        magnetization=magnetization,
        kappa=kappa,
        **voxel_kwargs,
    )


def roundtrip_vtu(
    corners: np.ndarray,
    dens: np.ndarray,
    kappa: np.ndarray | float,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Save model to VTU and load it back."""
    save_model(path, corners, dens, kappa)
    return load_model(path)


def mean_magnetization(i: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(i), axis=0)
