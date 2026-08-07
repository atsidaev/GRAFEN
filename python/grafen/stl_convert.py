"""Convert a closed STL surface into a GRAFEN volumetric hex VTU model.

Default path: axis-aligned hex voxelization of the STL AABB, keeping cells
whose centers lie inside the closed surface (ray casting).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .mesh import cube_mesh
from .model_io import save_model
from .stl_io import read_stl, stl_bbox


def _ray_triangle_hits(
    origins: np.ndarray,
    direction: np.ndarray,
    triangles: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Count Möller–Trumbore intersections of rays with triangles.

    Parameters
    ----------
    origins : (P, 3)
    direction : (3,) unit-ish direction (same for all rays)
    triangles : (T, 3, 3)

    Returns
    -------
    hits : (P,) int — number of intersections with t > eps
    """
    origins = np.asarray(origins, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    tris = np.asarray(triangles, dtype=np.float64)
    p = origins.shape[0]
    tcount = tris.shape[0]
    if p == 0 or tcount == 0:
        return np.zeros(p, dtype=np.int64)

    v0 = tris[:, 0]  # (T, 3)
    v1 = tris[:, 1]
    v2 = tris[:, 2]
    e1 = v1 - v0
    e2 = v2 - v0

    # Process in origin chunks to bound memory (P * T)
    hits = np.zeros(p, dtype=np.int64)
    chunk = max(1, min(p, 512))
    for start in range(0, p, chunk):
        o = origins[start : start + chunk]  # (C, 3)
        c = o.shape[0]
        # Broadcast: (C, 1, 3) vs (1, T, 3)
        tvec = o[:, None, :] - v0[None, :, :]  # (C, T, 3)
        pvec = np.cross(direction, e2)  # (T, 3)
        det = np.einsum("tj,tj->t", e1, pvec)  # (T,)
        valid = np.abs(det) > eps
        inv_det = np.zeros_like(det)
        inv_det[valid] = 1.0 / det[valid]

        u = np.einsum("ctj,tj->ct", tvec, pvec) * inv_det[None, :]  # (C, T)
        qvec = np.cross(tvec, e1[None, :, :])  # (C, T, 3)
        v = np.einsum("ctj,j->ct", qvec, direction) * inv_det[None, :]
        t_param = np.einsum("ctj,tj->ct", qvec, e2) * inv_det[None, :]

        inside = (
            valid[None, :]
            & (u >= 0.0)
            & (v >= 0.0)
            & ((u + v) <= 1.0)
            & (t_param > eps)
        )
        hits[start : start + c] = inside.sum(axis=1)
    return hits


def points_inside_stl(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Odd-parity ray test: True if point is inside a closed triangle mesh."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    # Slightly skewed ray reduces hits on shared edges / coplanar faces
    direction = np.array([1.0, 1e-5, 1e-6], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    hits = _ray_triangle_hits(points, direction, triangles, eps=eps)
    return (hits % 2) == 1


def voxelize_stl(
    triangles: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    magnetization: np.ndarray,
    kappa: float,
    *,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    pad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill STL AABB with hex voxels; keep cells whose centers are inside.

    Parameters
    ----------
    triangles : (T, 3, 3) closed surface
    nx, ny, nz : subdivision of the bounding box
    magnetization : (3,) initial I for every kept cell
    kappa : scalar susceptibility for every kept cell
    bounds : optional AABB override ((x0,x1),(y0,y1),(z0,z1)); default STL bbox
    pad : expand AABB by this amount on each side (before subdividing)

    Returns
    -------
    corners : (N, 8, 3), dens : (N, 3), kappa_arr : (N,)
    """
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("nx, ny, nz must be >= 1")
    tris = np.asarray(triangles, dtype=np.float64)
    if bounds is None:
        lo, hi = stl_bbox(tris)
        bounds = (
            (float(lo[0]) - pad, float(hi[0]) + pad),
            (float(lo[1]) - pad, float(hi[1]) + pad),
            (float(lo[2]) - pad, float(hi[2]) + pad),
        )
    (x0, x1), (y0, y1), (z0, z1) = bounds
    magnetization = np.asarray(magnetization, dtype=float).reshape(3)

    corners_full, dens_full = cube_mesh(
        (x0, x1, nx),
        (y0, y1, ny),
        (z0, z1, nz),
        magnetization=magnetization,
    )
    centers = corners_full.mean(axis=1)
    inside = points_inside_stl(centers, tris)
    if not np.any(inside):
        raise ValueError(
            "voxelize_stl: no cell centers inside the STL; "
            "check that the surface is closed and watertight, or refine/pad the grid"
        )
    corners = corners_full[inside]
    dens = dens_full[inside]
    kappa_arr = np.full(corners.shape[0], float(kappa), dtype=float)
    return corners, dens, kappa_arr


def stl_to_hex_model(
    stl_path: str | Path,
    *,
    magnetization: np.ndarray,
    kappa: float,
    nx: int = 16,
    ny: int = 16,
    nz: int = 16,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    pad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read STL and build a volumetric hex model via voxelization."""
    tris = read_stl(stl_path)
    return voxelize_stl(
        tris,
        nx,
        ny,
        nz,
        magnetization,
        kappa,
        bounds=bounds,
        pad=pad,
    )


def convert_stl_to_vtu(
    stl_path: str | Path,
    vtu_path: str | Path,
    *,
    magnetization: np.ndarray,
    kappa: float,
    nx: int = 16,
    ny: int = 16,
    nz: int = 16,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    pad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corners, dens, kappa_arr = stl_to_hex_model(
        stl_path,
        magnetization=magnetization,
        kappa=kappa,
        nx=nx,
        ny=ny,
        nz=nz,
        bounds=bounds,
        pad=pad,
    )
    save_model(vtu_path, corners, dens, kappa_arr)
    return corners, dens, kappa_arr


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Convert a closed STL surface to a GRAFEN hex VTU by voxelizing "
            "the bounding box and keeping cells whose centers are inside the surface."
        ),
    )
    p.add_argument("stl", type=Path, help="Input .stl (closed / watertight)")
    p.add_argument("vtu", type=Path, help="Output .vtu")
    p.add_argument("--kappa", type=float, default=0.0)
    p.add_argument(
        "--I",
        dest="magnetization",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("Ix", "Iy", "Iz"),
    )
    p.add_argument("--nx", type=int, default=16, help="voxels along X (default 16)")
    p.add_argument("--ny", type=int, default=16, help="voxels along Y (default 16)")
    p.add_argument("--nz", type=int, default=16, help="voxels along Z (default 16)")
    p.add_argument(
        "--bounds",
        type=float,
        nargs=6,
        default=None,
        metavar=("x0", "x1", "y0", "y1", "z0", "z1"),
        help="AABB override (default: STL bbox)",
    )
    p.add_argument(
        "--pad",
        type=float,
        default=0.0,
        help="Expand STL bbox by this amount on each side before voxelizing",
    )
    args = p.parse_args(argv)

    bounds = None
    if args.bounds is not None:
        x0, x1, y0, y1, z0, z1 = args.bounds
        bounds = ((x0, x1), (y0, y1), (z0, z1))

    corners, dens, kappa = convert_stl_to_vtu(
        args.stl,
        args.vtu,
        magnetization=np.array(args.magnetization, dtype=float),
        kappa=args.kappa,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        bounds=bounds,
        pad=args.pad,
    )
    print(f"Wrote {args.vtu} ({corners.shape[0]} hex cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
